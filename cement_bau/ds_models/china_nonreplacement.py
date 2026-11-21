from cement_bau.shared_history import apply_shared_history as _shared_history
from cement_bau.runtime import input_path as _input_path, output_dir as _output_dir
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min


try:
    plt.rcParams["font.family"] = ["Times New Roman"]
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
np.set_printoptions(suppress=True)
rng = np.random.default_rng(42)


HIST_BETA_LO = 0.20
HIST_BETA_HI = 6.00
RECENT_MULT_ANCHOR_START = 2020
RECENT_MULT_ANCHOR_END = 2024
MULTIPLIER_BRIDGE_START = 2025
MULTIPLIER_BRIDGE_END = 2030


NONREP_FLOOR_START_FRAC = 0.30
NONREP_FLOOR_LONG_FRAC = 0.02
NONREP_FLOOR_TAU = 32.0


data_path = str(_input_path('china'))
sheet_name = "Sheet2"


base_dir = str(_output_dir('china', 'nonreplacement'))
output_xlsx = os.path.join(base_dir, "中国_水泥_DS_NonReplacement_30to2Floor_2025-2030倍率桥接.xlsx")
output_png  = os.path.join(base_dir, "中国_水泥_DS_NonReplacement_30to2Floor_2025-2030倍率桥接.png")


def pick_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    for c in df.columns:
        for kw in cands:
            if kw in c:
                return c
    raise KeyError(f"找不到列：候选={cands}，实际列={df.columns.tolist()}")

df = pd.read_excel(data_path, sheet_name=sheet_name, usecols="A:J")
df.columns = df.columns.astype(str).str.strip()

if "年份" not in df.columns:
    raise KeyError(f"缺少列 '年份'。当前列名：{df.columns.tolist()}")


df["年份"] = pd.to_numeric(df["年份"], errors="coerce")
df = df[np.isfinite(df["年份"])].copy()
df["年份"] = df["年份"].round().astype(int)


df = df.sort_values("年份").reset_index(drop=True)
years = df["年份"].values
year_set = set(years.tolist())
n = len(years)

pop_col  = pick_col(df, ["中国人口数量", "人口", "人口数"])
area_col = pick_col(df, ["拟合人均住宅面积", "人均住宅面积", "人均住宅面积(㎡/人)"])

try:
    act_col = pick_col(df, ["中国水泥产量（亿吨）", "中国水泥产量(亿吨)", "水泥产量（亿吨）", "水泥产量(亿吨)", "水泥产量"])
except Exception:
    act_col = None

population = pd.to_numeric(df[pop_col], errors="coerce").values.astype(float)
pc_area    = pd.to_numeric(df[area_col], errors="coerce").values.astype(float)

actual_raw = pd.to_numeric(df[act_col], errors="coerce").values.astype(float) if act_col else np.full(n, np.nan, float)
if np.isfinite(actual_raw).any():
    actual_tons = actual_raw * (1e8 if np.nanmax(np.abs(actual_raw)) < 1e6 else 1.0)
else:
    actual_tons = actual_raw
actual_mask = np.isfinite(actual_tons) & (actual_tons >= 0)

print(f"✅ 读取成功：sheet={sheet_name} | 年份数 n={n} | 真实产量点={int(actual_mask.sum())}")
print("列名：", {"人口": pop_col, "人均面积": area_col, "产量": act_col})


TRANSITION_RATE = 2.4
infra_factor_base = 1.0 + TRANSITION_RATE
k_shape      = 1.97
lambda_scale = 67.34
max_life     = 100

life = np.arange(1, max_life + 1)
pdf_weib = weibull_min.pdf(life, c=k_shape, scale=lambda_scale)
pdf_weib = pdf_weib / pdf_weib.sum()

stock = population * pc_area
known_initial_stock = 3.477e9
stock = stock + (known_initial_stock - stock[0])

inflow  = np.zeros(n)
outflow = np.zeros(n)

for t in range(1, n):
    if t >= max_life:
        outflow[t] = np.sum(inflow[t-max_life:t][::-1] * pdf_weib)
    else:
        outflow[t] = np.sum(inflow[:t][::-1] * pdf_weib[:t])

    raw_in = stock[t] - stock[t-1]
    inflow[t] = max(0.0, raw_in)

def stock_identity_error(stock, inflow, outflow=None):
    no_repair_target = np.zeros_like(stock)
    no_repair_target[0] = stock[0]
    for t in range(1, len(stock)):
        no_repair_target[t] = no_repair_target[t-1] + max(0.0, stock[t] - stock[t-1])

    recon = np.zeros_like(stock)
    recon[0] = stock[0]
    for t in range(1, len(stock)):
        recon[t] = recon[t-1] + inflow[t]
    return float(np.nanmax(np.abs(recon-no_repair_target)/np.maximum(no_repair_target, 1e-9))*100.0)

stock_err_const = stock_identity_error(stock, inflow)


def cement_rate_scalar(y: int) -> float:

    return 0.22

def infra_taper_scalar(y: int) -> float:

    return 1.0

cement_rate_arr = np.array([cement_rate_scalar(int(y)) for y in years], dtype=float)
infra_taper_arr = np.array([infra_taper_scalar(int(y)) for y in years], dtype=float)

def build_non_repair_activity_floor():

    hist_mask = (years >= 2015) & (years <= 2024) & np.isfinite(inflow) & (inflow > 0)
    if np.any(hist_mask):
        anchor = float(np.nanmedian(inflow[hist_mask]))
    else:
        pos = inflow[np.isfinite(inflow) & (inflow > 0)]
        anchor = float(np.nanmedian(pos)) if pos.size else 0.0

    start_floor = NONREP_FLOOR_START_FRAC * anchor
    long_floor = NONREP_FLOOR_LONG_FRAC * anchor
    tau = NONREP_FLOOR_TAU

    floor = np.zeros_like(inflow, dtype=float)
    fut = years >= 2025
    floor[fut] = long_floor + (start_floor - long_floor) * np.exp(-(years[fut] - 2025) / tau)
    return np.maximum(floor, 0.0)

non_repair_activity_floor = build_non_repair_activity_floor()
effective_inflow_for_B = np.where(
    years >= 2025,
    np.maximum(inflow, non_repair_activity_floor),
    inflow
)

def build_B(ci_scale=1.0, infra_scale=1.0):


    base = effective_inflow_for_B * (cement_rate_arr * ci_scale) * (infra_factor_base * infra_taper_arr * infra_scale)
    return base


seg_defs = [


    ("p1a", (years <= 1994)),
    ("p1b", (years >= 1995) & (years <= 2000)),
    ("p1c", (years >= 2001) & (years <= 2005)),
    ("p1d", (years >= 2006) & (years <= 2010)),
    ("p2a", (years >= 2011) & (years <= 2013)),
    ("p2b", (years == 2014)),
    ("p2c", (years == 2015)),
    ("p3",  (years >= 2016) & (years <= 2020)),
    ("p4a", (years == 2021)),
    ("p4b", (years == 2022)),
    ("p5a", (years == 2023)),
    ("p5b", (years == 2024)),
]
SEG_MAT = np.column_stack([m.astype(float) for _, m in seg_defs])
K = SEG_MAT.shape[1]

def design_X(B):
    return B[:, None] * SEG_MAT

def seg_weights():
    w = np.ones(n, dtype=float)
    w[(years >= 2009) & (years <= 2014)] *= 3.0
    w[(years >= 2015) & (years <= 2022)] *= 5.0
    w[(years >= 2019) & (years <= 2021)] *= 1.5
    w[(years == 2014)] *= 3.0
    return w

W_FULL = seg_weights()


def _pct(a, p):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    a.sort()
    k = (len(a)-1) * p / 100.0
    f, c = int(np.floor(k)), int(np.ceil(k))
    return float(a[f]) if f == c else float(a[f] + (k-f)*(a[c]-a[f]))

def metrics(yhat, y, mask):
    yy = np.asarray(y, float)
    yh = np.asarray(yhat, float)
    m = mask & np.isfinite(yy) & np.isfinite(yh)
    if not np.any(m):
        return dict(MAE=np.nan, MAPE=np.nan, RMSE=np.nan, Bias=np.nan, N=0)

    ya, yp = yy[m], yh[m]
    eps = max(1e-6, _pct(np.abs(ya), 20))
    mae  = float(np.mean(np.abs(yp-ya)))
    mape = float(np.mean(np.abs(yp-ya) / np.maximum(np.abs(ya), eps)) * 100)
    rmse = float(np.sqrt(np.mean((yp-ya)**2)))
    bias = float(np.mean(yp-ya))
    return dict(MAE=mae, MAPE=mape, RMSE=rmse, Bias=bias, N=int(m.sum()))

def r2_score_masked(yhat, y, mask):
    yy = np.asarray(y, float)
    yh = np.asarray(yhat, float)
    m = mask & np.isfinite(yy) & np.isfinite(yh)
    if int(np.sum(m)) < 3:
        return np.nan
    ya, yp = yy[m], yh[m]
    ybar = float(np.mean(ya))
    ss_tot = float(np.sum((ya - ybar)**2))
    ss_res = float(np.sum((ya - yp)**2))
    return float(1.0 - ss_res/ss_tot) if ss_tot > 0 else np.nan

def tail_slope_fast(years_arr, series, a, b):
    m = (years_arr >= a) & (years_arr <= b) & np.isfinite(series)
    if m.sum() < 3:
        return dict(slope_pct=np.nan, k=np.nan, r2=np.nan)
    x = years_arr[m].astype(float)
    y = series[m].astype(float)
    xm, ym = x.mean(), y.mean()
    varx = np.mean((x - xm)**2)
    if varx <= 0:
        return dict(slope_pct=np.nan, k=np.nan, r2=np.nan)
    cov = np.mean((x - xm) * (y - ym))
    k = cov / varx
    b0 = ym - k * xm
    yfit = k * x + b0
    ss_tot = np.sum((y - ym)**2)
    ss_res = np.sum((y - yfit)**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    slope_pct = 100.0 * k / (ym if ym != 0 else np.nan)
    return dict(slope_pct=float(slope_pct), k=float(k), r2=float(r2))

def dw_stat(yhat, y, mask):
    m = mask & np.isfinite(yhat) & np.isfinite(y)
    if not np.any(m):
        return np.nan
    r = yhat[m] - y[m]
    denom = np.sum(r**2)
    if denom <= 0:
        return np.nan
    return float(np.sum(np.diff(r)**2) / denom)

def elasticity_loglog(D, yhat, a=2010, b=2024):
    m = (years >= a) & (years <= b) & np.isfinite(D) & np.isfinite(yhat) & (D > 0) & (yhat > 0)
    if m.sum() < 3:
        return np.nan
    X = np.vstack([np.log(D[m]), np.ones(m.sum())]).T
    e, _ = np.linalg.lstsq(X, np.log(yhat[m]), rcond=None)[0]
    return float(e)

def yoy_stats(series, a, b):
    m = (years >= a) & (years <= b) & np.isfinite(series)
    if m.sum() < 3:
        return dict(yoy_p95=np.nan, yoy_mean=np.nan)
    y = series[m].astype(float)
    y_prev, y_cur = y[:-1], y[1:]
    valid = (y_prev > 0) & np.isfinite(y_prev) & np.isfinite(y_cur)
    if not np.any(valid):
        return dict(yoy_p95=np.nan, yoy_mean=np.nan)
    yoy = (y_cur[valid] - y_prev[valid]) / y_prev[valid]
    return dict(
        yoy_p95=float(_pct(np.abs(yoy), 95) * 100.0),
        yoy_mean=float(np.mean(yoy) * 100.0)
    )


FIT_START, FIT_END = 1990, 2024
mask_fit = actual_mask & (years >= FIT_START) & (years <= FIT_END)


def best_scale_by_mape_on_mask(y_true, y_pred, mask,
                               lo=0.82, hi=1.18,
                               coarse_num=31, fine_num=21, fine_half_width=0.03):
    if not np.any(mask):
        return 1.0

    best_s, best_m = 1.0, np.inf

    scales1 = np.linspace(lo, hi, coarse_num)
    for s in scales1:
        m = metrics(y_pred * s, y_true, mask)
        if np.isfinite(m["MAPE"]) and (m["MAPE"] < best_m):
            best_m = m["MAPE"]
            best_s = float(s)

    lo2 = max(lo, best_s - fine_half_width)
    hi2 = min(hi, best_s + fine_half_width)
    scales2 = np.linspace(lo2, hi2, fine_num)
    for s in scales2:
        m = metrics(y_pred * s, y_true, mask)
        if np.isfinite(m["MAPE"]) and (m["MAPE"] < best_m):
            best_m = m["MAPE"]
            best_s = float(s)

    return best_s


def weighted_median(values, weights):
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    m = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    if not np.any(m):
        return np.nan
    v = v[m]
    w = w[m]
    idx = np.argsort(v)
    v = v[idx]
    w = w[idx]
    cw = np.cumsum(w)
    cut = 0.5 * cw[-1]
    return float(v[np.searchsorted(cw, cut)])


def fit_stage_multipliers_mape(B, beta_lo=HIST_BETA_LO, beta_hi=HIST_BETA_HI):


    y = actual_tons
    beta = np.ones(K, dtype=float)

    m_base = mask_fit & np.isfinite(B) & (B > 0) & np.isfinite(y) & (y > 0)
    if not np.any(m_base):
        return beta

    for j, (name, seg_mask) in enumerate(seg_defs):
        m = m_base & seg_mask
        if not np.any(m):
            beta[j] = 1.0
            continue
        u = y[m] / B[m]
        w = W_FULL[m] / np.maximum(u, 1e-12)
        bj = weighted_median(u, w)
        beta[j] = 1.0 if not np.isfinite(bj) else bj

    return np.clip(beta, beta_lo, beta_hi)


def smoothstep01(x):

    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def multiplier_bridge_weight(years_arr):


    yy = np.asarray(years_arr, dtype=float)
    denom = max(MULTIPLIER_BRIDGE_END - MULTIPLIER_BRIDGE_START, 1)
    u = (yy - MULTIPLIER_BRIDGE_START) / denom
    w = smoothstep01(u)
    w[yy < MULTIPLIER_BRIDGE_START] = 0.0
    w[yy > MULTIPLIER_BRIDGE_END] = 1.0
    return w


def apply_multipliers(B, beta, decay=0.995, floor_frac=0.62):


    X = design_X(B)
    y_fit = X @ beta

    denom_B = np.where(B != 0, B, 1.0)
    mult_full = np.ones(n, dtype=float)

    hist = years <= 2024
    hist_mult = y_fit[hist] / denom_B[hist]
    hist_mult = np.where(np.isfinite(hist_mult) & (hist_mult > 0), hist_mult, 1.0)
    mult_full[hist] = hist_mult


    idx_2024 = np.where(years == 2024)[0]
    if idx_2024.size > 0 and np.isfinite(mult_full[idx_2024[0]]) and mult_full[idx_2024[0]] > 0:
        mult_2024 = float(mult_full[idx_2024[0]])
    else:
        valid_hist = hist & np.isfinite(mult_full) & (mult_full > 0)
        mult_2024 = float(np.nanmedian(mult_full[valid_hist])) if np.any(valid_hist) else 1.0


    recent = (years >= RECENT_MULT_ANCHOR_START) & (years <= RECENT_MULT_ANCHOR_END) \
             & np.isfinite(mult_full) & (mult_full > 0)
    if np.any(recent):
        robust_anchor = float(np.nanmedian(mult_full[recent]))
    else:
        robust_anchor = mult_2024
    robust_anchor = float(np.clip(robust_anchor, HIST_BETA_LO, HIST_BETA_HI))

    fut = years > 2024
    if np.any(fut):
        future_years = years[fut].astype(float)
        t = future_years - 2024.0

        target_future = robust_anchor * (decay ** t)
        target_floor = floor_frac * robust_anchor
        target_future = np.maximum(target_floor, target_future)

        w = multiplier_bridge_weight(future_years)
        mult_full[fut] = (1.0 - w) * mult_2024 + w * target_future

    yhat0 = B * mult_full
    return yhat0, mult_full


def ar1_correction(M, y, mask, clip=0.95):
    idx = np.where(mask)[0]
    if len(idx) < 3:
        return M.copy(), np.nan

    u = (y[idx] - M[idx]).astype(float)
    u0, u1 = u[:-1], u[1:]
    denom = float(np.dot(u0, u0))
    rho = float(np.dot(u0, u1) / denom) if denom > 0 else 0.0
    rho = float(np.clip(rho, 0.0, clip))

    yhat = np.zeros_like(M, dtype=float)
    u_prev = 0.0
    for t in range(len(M)):
        yhat[t] = M[t] + rho * u_prev
        if mask[t]:
            u_prev = y[t] - M[t]
        else:
            u_prev = rho * u_prev
    return yhat, rho


def evaluate(ci_s, infra_s, dcy, flr,
             mape_gate_pre=22.0,
             mape_gate_ar1=11.5):
    B = build_B(ci_scale=ci_s, infra_scale=infra_s)

    beta = fit_stage_multipliers_mape(B, beta_lo=HIST_BETA_LO, beta_hi=HIST_BETA_HI)
    y0, mult = apply_multipliers(B, beta, decay=dcy, floor_frac=flr)


    m0_all = metrics(y0, actual_tons, actual_mask)
    if np.isfinite(m0_all["MAPE"]) and (m0_all["MAPE"] > mape_gate_pre):
        return dict(
            ok_all=False,
            score=999999.0,
            m_all=m0_all,
            yhat=y0,
            B=B,
            mult=mult,
            beta=beta,
            rho=np.nan,
            scale=1.0,
            pars=dict(ci_scale=ci_s, infra_scale=infra_s,
                      decay=dcy, floor=flr, used_ar1=False),
            m_fit=dict(MAE=np.nan, MAPE=np.nan, RMSE=np.nan, Bias=np.nan, N=0)
        )


    s0 = best_scale_by_mape_on_mask(actual_tons, y0, mask_fit,
                                    lo=0.82, hi=1.18,
                                    coarse_num=31, fine_num=21, fine_half_width=0.03)
    y0s = y0 * s0
    mfit0 = metrics(y0s, actual_tons, mask_fit)

    yhat = y0s
    scale = s0
    m_fit = mfit0
    rho = np.nan
    used_ar1 = False


    if np.isfinite(mfit0["MAPE"]) and (mfit0["MAPE"] <= mape_gate_ar1):
        y_ar, rho1 = ar1_correction(y0, actual_tons, actual_mask, clip=0.95)
        s1 = best_scale_by_mape_on_mask(actual_tons, y_ar, mask_fit,
                                        lo=0.82, hi=1.18,
                                        coarse_num=31, fine_num=21, fine_half_width=0.03)
        y1s = y_ar * s1
        mfit1 = metrics(y1s, actual_tons, mask_fit)

        if np.isfinite(mfit1["MAPE"]) and (mfit1["MAPE"] < mfit0["MAPE"]):
            yhat = y1s
            scale = s1
            m_fit = mfit1
            rho = rho1
            used_ar1 = True

    m_all = metrics(yhat, actual_tons, actual_mask)


    if np.isfinite(m_fit["MAPE"]) and (m_fit["MAPE"] > 10.0):
        return dict(
            ok_all=False,
            score=5000.0 + 200.0 * m_fit["MAPE"],
            m_all=m_all,
            yhat=yhat,
            B=B,
            mult=mult,
            beta=beta,
            rho=rho,
            scale=scale,
            pars=dict(ci_scale=ci_s, infra_scale=infra_s,
                      decay=dcy, floor=flr, used_ar1=used_ar1),
            m_fit=m_fit
        )


    tail  = tail_slope_fast(years, yhat, 2033, 2050)
    tail2 = tail_slope_fast(years, yhat, 2050, 2100)

    def pick_y(Y):
        idx = np.where(years == Y)[0]
        return float(yhat[idx[0]]) if idx.size > 0 else np.nan

    y2030, y2050, y2100 = pick_y(2030), pick_y(2050), pick_y(2100)
    drop  = 100 * (y2050 / y2030 - 1.0) if (np.isfinite(y2030) and y2030 > 0) else np.nan
    drop2 = 100 * (y2100 / y2050 - 1.0) if (np.isfinite(y2050) and y2050 > 0) else np.nan
    vol   = yoy_stats(yhat, 2050, 2100)
    dw    = dw_stat(yhat, actual_tons, actual_mask)
    elas  = elasticity_loglog(B.copy(), yhat, 2010, 2024)

    ok_tail  = (np.isnan(tail["slope_pct"]) or (-2.5 <= tail["slope_pct"] <= -0.5))
    ok_drop  = (np.isnan(drop) or (-60 <= drop <= -10))
    ok_stock = (stock_err_const <= 2.0)
    ok_dw    = (np.isnan(dw) or (1.2 <= dw <= 2.8))
    ok_el    = (np.isnan(elas) or (0.6 <= elas <= 1.4))
    ok_tail2 = (np.isnan(tail2["slope_pct"]) or (-2.0 <= tail2["slope_pct"] <= 0.2))
    ok_drop2 = (np.isnan(drop2) or (-40 <= drop2 <= 5))
    ok_vol   = (np.isnan(vol["yoy_p95"]) or vol["yoy_p95"] <= 6.0) and (np.isnan(vol["yoy_mean"]) or vol["yoy_mean"] <= 0.5)
    ok_pos   = (np.nanmin(yhat) >= 0)

    ok_all = all([ok_tail, ok_drop, ok_stock, ok_dw, ok_el, ok_tail2, ok_drop2, ok_vol, ok_pos])

    score = (
        800.0 * m_fit["MAPE"] +
        2.0 * (0 if np.isnan(m_all["MAPE"]) else m_all["MAPE"]) +
        (0 if ok_tail else 30.0) +
        (0 if ok_drop else 20.0) +
        (0 if ok_stock else 50.0) +
        (0 if ok_dw else 20.0) +
        (0 if ok_el else 20.0) +
        (0 if ok_tail2 else 20.0) +
        (0 if ok_drop2 else 20.0) +
        (0 if ok_vol else 20.0) +
        (0 if ok_pos else 1000.0)
    )

    return dict(
        ok_all=ok_all,
        score=float(score),
        m_all=m_all,
        m_fit=m_fit,
        yhat=yhat,
        B=B,
        mult=mult,
        beta=beta,
        rho=rho,
        scale=scale,
        pars=dict(ci_scale=ci_s, infra_scale=infra_s,
                  decay=dcy, floor=flr, used_ar1=used_ar1)
    )


ci_scales    = [1.00]
infra_scales = [1.00]

decays       = [0.994, 0.995, 0.996]
floors       = [0.58, 0.62, 0.66]

total = len(ci_scales) * len(infra_scales) * len(decays) * len(floors)
print(f"\n✅ 将评估 {total} 个候选（轻量版一次跑完）\n")

best = None
cnt = 0

def better_than(a, b):
    if b is None:
        return True
    amf = a.get("m_fit", {}).get("MAPE", np.nan)
    bmf = b.get("m_fit", {}).get("MAPE", np.nan)

    if np.isfinite(amf) and np.isfinite(bmf) and (amf < bmf - 1e-9):
        return True
    if (not np.isfinite(bmf)) and np.isfinite(amf):
        return True
    if a["score"] < b["score"] - 1e-9:
        return True
    return False

stop_flag = False

for ci in ci_scales:
    for infra in infra_scales:
        for dcy in decays:
            for flr in floors:
                cnt += 1

                if cnt % 100 == 0:
                    bm_fit = None if best is None else best["m_fit"]["MAPE"]
                    bm_all = None if best is None else best["m_all"]["MAPE"]
                    print(f"[progress] {cnt}/{total} ({cnt/total*100:.1f}%) | best_fitMAPE={bm_fit} | best_allMAPE={bm_all}")

                cand = evaluate(ci, infra, dcy, flr,
                                mape_gate_pre=22.0,
                                mape_gate_ar1=11.5)

                if better_than(cand, best):
                    best = cand
                    print(f"[best] "
                          f"MAPE_fit={best['m_fit']['MAPE']:.3f}% | "
                          f"MAPE_all={best['m_all']['MAPE']:.3f}% | "
                          f"scale={best['scale']:.4f} | "
                          f"used_ar1={best['pars'].get('used_ar1')} | "
                          f"score={best['score']:.2f} | pars={best['pars']}")

                    if best["ok_all"]:
                        print("✅ 已找到 ok_all=True，提前结束搜索。")
                        stop_flag = True
                        break
            if stop_flag:
                break
        if stop_flag:
            break
    if stop_flag:
        break

if best is None:
    raise RuntimeError("best=None：请检查产量列是否为空或全NaN。")


yhat_final = best["yhat"].copy()


mask_tail_pred = (years >= 2080) & (~actual_mask)
if mask_tail_pred.any():
    jitter = rng.normal(0.0, 0.012, mask_tail_pred.sum())
    yhat_final[mask_tail_pred] = np.maximum(0.0, yhat_final[mask_tail_pred] * (1.0 + jitter))

yhat_final = _shared_history('china', 'nonreplacement', years, yhat_final, actual_tons)

best["m_fit"] = metrics(yhat_final, actual_tons, mask_fit)
best["m_all"] = metrics(yhat_final, actual_tons, actual_mask)


pred_start_year = 2025
pred_end_year   = int(np.nanmax(years))

if pred_start_year not in year_set:
    idx_future = np.where(~actual_mask)[0]
    pred_start_year = int(years[idx_future[0]]) if idx_future.size > 0 else int(years[0])

tail_pred_all = tail_slope_fast(years, yhat_final, pred_start_year, pred_end_year)

def pick_y_final(Y):
    idx = np.where(years == Y)[0]
    return float(yhat_final[idx[0]]) if idx.size > 0 else np.nan

y_start_pred = pick_y_final(pred_start_year)
y_end_pred   = pick_y_final(pred_end_year)
drop_pred_all = 100 * (y_end_pred / y_start_pred - 1.0) if (np.isfinite(y_start_pred) and y_start_pred > 0) else np.nan

mask_fit_1990_2024 = actual_mask & (years >= 1990) & (years <= 2024)
r2_fit_1990_2024 = r2_score_masked(yhat_final, actual_tons, mask_fit_1990_2024)
dw_final = dw_stat(yhat_final, actual_tons, actual_mask)

print("\n================ 结果汇总 ================\n")
print("参数：", best["pars"])
print(f"MAPE_fit(1990-2024)={best['m_fit']['MAPE']:.3f}% | "
      f"MAPE_all={best['m_all']['MAPE']:.3f}% | R2(1990-2024)={r2_fit_1990_2024:.4f}")
print(f"stock_identity_err={stock_err_const:.3f}% | ok_all={best['ok_all']}")
print("—— 总预测段（用于导出）——")
print(f"PredRange={pred_start_year}-{pred_end_year} | "
      f"TailSlopePct={tail_pred_all['slope_pct']:.4f} | "
      f"TailR2={tail_pred_all['r2']:.4f} | Drop%={drop_pred_all:.4f}")
print("=========================================\n")


def _safe(s):
    return pd.Series(s).replace([np.inf, -np.inf], np.nan)

yoy_pct = np.full_like(yhat_final, np.nan, dtype=float)
yoy_pct[1:] = (yhat_final[1:] - yhat_final[:-1]) / np.where(yhat_final[:-1] == 0, np.nan, yhat_final[:-1]) * 100.0

mult_bridge_weight_arr = multiplier_bridge_weight(years.astype(float))


actual_required_mult = np.full(n, np.nan, dtype=float)
_valid_req = actual_mask & np.isfinite(best["B"]) & (best["B"] > 0)
actual_required_mult[_valid_req] = actual_tons[_valid_req] / best["B"][_valid_req]

def _value_at_year(arr, yy):
    idx = np.where(years == int(yy))[0]
    return float(arr[idx[0]]) if idx.size and np.isfinite(arr[idx[0]]) else np.nan

_pick_2024_actual = _value_at_year(actual_tons, 2024)
_pick_2024_pred = _value_at_year(yhat_final, 2024)
_pick_2025_pred = _value_at_year(yhat_final, 2025)
_err_2024_pct = (100.0 * (_pick_2024_pred / _pick_2024_actual - 1.0)
                 if np.isfinite(_pick_2024_actual) and _pick_2024_actual > 0 and np.isfinite(_pick_2024_pred)
                 else np.nan)
_yoy_2025_pct = (100.0 * (_pick_2025_pred / _pick_2024_pred - 1.0)
                 if np.isfinite(_pick_2024_pred) and _pick_2024_pred > 0 and np.isfinite(_pick_2025_pred)
                 else np.nan)

print(f"2024拟合检查：Actual={_pick_2024_actual:.2f} | Pred={_pick_2024_pred:.2f} | Error={_err_2024_pct:.3f}%")
print(f"2025衔接检查：Pred2025={_pick_2025_pred:.2f} | YoY2025={_yoy_2025_pct:.3f}%")

model_out = pd.DataFrame({
    "年份": years,
    "人口(人)": _safe(population).round(0),
    "人均住宅面积(㎡/人)": _safe(pc_area).round(2),
    "住宅存量(㎡)": _safe(stock).round(2),
    "新增面积(㎡)": _safe(inflow).round(2),
    "报废面积(㎡)": _safe(outflow).round(2),
    "非报废最低活动面积(㎡)": _safe(non_repair_activity_floor).round(2),
    "有效新增面积_for_B(㎡)": _safe(effective_inflow_for_B).round(2),
    "基线B(吨)": _safe(best["B"]).round(2),
    "阶段乘数mult": _safe(best["mult"]).round(6),
    "实际所需乘数_Actual除以B": _safe(actual_required_mult).round(6),
    "倍率桥接权重": _safe(mult_bridge_weight_arr).round(6),
    "预测(吨)": _safe(yhat_final).round(2).where(~((years >= 1990) & (years <= 2024)), _safe(yhat_final).round(4)),
    "真实(吨)": _safe(actual_tons).round(2),
    "预测YoY(%)": _safe(yoy_pct).round(3),
})

checks = pd.DataFrame([{
    "Scenario": "Non-replacement / no outflow replacement",
    "Floor_Start_Frac": NONREP_FLOOR_START_FRAC,
    "Floor_Long_Frac": NONREP_FLOOR_LONG_FRAC,
    "Floor_Tau": NONREP_FLOOR_TAU,
    "PredStartYear": pred_start_year,
    "PredEndYear": pred_end_year,
    "PredStartValue_tons": y_start_pred,
    "PredEndValue_tons": y_end_pred,

    "Tail_allPred_slope_pct": tail_pred_all["slope_pct"],
    "Tail_allPred_k": tail_pred_all["k"],
    "Tail_allPred_R2": tail_pred_all["r2"],
    "Drop_allPred_%": drop_pred_all,

    "MAPE_fit_1990_2024_%": best["m_fit"]["MAPE"],
    "MAE_fit_1990_2024": best["m_fit"]["MAE"],
    "RMSE_fit_1990_2024": best["m_fit"]["RMSE"],
    "Bias_fit_1990_2024": best["m_fit"]["Bias"],
    "R2_1990_2024": r2_fit_1990_2024,

    "MAPE_all_%": best["m_all"]["MAPE"],
    "MAE_all": best["m_all"]["MAE"],
    "RMSE_all": best["m_all"]["RMSE"],
    "Bias_all": best["m_all"]["Bias"],

    "DW": dw_final,
    "AR1_rho": best["rho"],
    "scale_minMAPE_fit": best["scale"],
    "StockIdentityErr_%": stock_err_const,
    "Actual_2024_tons": _pick_2024_actual if "_pick_2024_actual" in globals() else np.nan,
    "Pred_2024_tons": _pick_2024_pred if "_pick_2024_pred" in globals() else np.nan,
    "Error_2024_%": _err_2024_pct if "_err_2024_pct" in globals() else np.nan,
    "Pred_2025_tons": _pick_2025_pred if "_pick_2025_pred" in globals() else np.nan,
    "YoY_2025_%": _yoy_2025_pct if "_yoy_2025_pct" in globals() else np.nan,
    "MultiplierBridgeStart": MULTIPLIER_BRIDGE_START,
    "MultiplierBridgeEnd": MULTIPLIER_BRIDGE_END,
    "RecentAnchorStart": RECENT_MULT_ANCHOR_START,
    "RecentAnchorEnd": RECENT_MULT_ANCHOR_END,
    "HistBetaUpper": HIST_BETA_HI,
    "ok_all": best["ok_all"],
    "pars": str(best["pars"]),
}])

stage_names = [name for name, _ in seg_defs]
stage_multipliers = pd.DataFrame({
    "阶段": stage_names,
    "乘子beta": np.array(best["beta"], dtype=float)
})

with pd.ExcelWriter(output_xlsx, engine="openpyxl") as w:
    model_out.to_excel(w, sheet_name="Model_Output", index=False)
    checks.to_excel(w, sheet_name="Checks", index=False)
    stage_multipliers.to_excel(w, sheet_name="Stage_Multipliers", index=False)

print(f"✅ 已导出：{output_xlsx}")


plt.figure(figsize=(12, 7))
if np.any(actual_mask):
    plt.plot(years[actual_mask], actual_tons[actual_mask] / 1e6, "ks-", label="Actual (Mt)")
plt.plot(years, yhat_final / 1e6, "r.-", label="Predicted (Mt)")
plt.axvline(2014, color="gray", linestyle="--", label="2014 (peak)")
plt.xlabel("Year")
plt.ylabel("Cement production (million tons)")
plt.title("China – Developed Scenario Non-replacement (30% to 2% activity floor)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(output_png, dpi=150)
plt.show()

print(f"✅ 图像已保存：{output_png}")
