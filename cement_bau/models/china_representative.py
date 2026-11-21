from cement_bau.runtime import input_path as _input_path, output_dir as _output_dir
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

MIN_FLOOR_LOGIC_VERSION = "CN_DS_WITH_OUTFLOW_PLUS_MIN_FLOOR_V1_20260708"


def _china_logic_phase_forecast(
    years_arr,
    stock_arr,
    actual_arr,
    fit_start,
    fit_end,
    k_shape,
    lambda_scale,
    max_life,
    cement_rate_arr,
    transition_factor_arr,
    output_unit_scale=1.0,
    kernel_method="pdf",
):
    years_arr = np.asarray(years_arr, dtype=int)
    stock_arr = np.asarray(stock_arr, dtype=float)
    actual_arr = np.asarray(actual_arr, dtype=float)
    cement_rate_arr = np.asarray(cement_rate_arr, dtype=float)
    transition_factor_arr = np.asarray(transition_factor_arr, dtype=float)
    n_local = len(years_arr)
    max_life = int(max_life)

    ages = np.arange(1, max_life + 1, dtype=float)
    if kernel_method == "cdfdiff":
        cdf_hi = weibull_min.cdf(ages, c=float(k_shape), scale=float(lambda_scale))
        cdf_lo = weibull_min.cdf(ages - 1.0, c=float(k_shape), scale=float(lambda_scale))
        pdf_local = np.maximum(cdf_hi - cdf_lo, 0.0)
    else:
        pdf_local = weibull_min.pdf(ages, c=float(k_shape), scale=float(lambda_scale))
    pdf_local = np.nan_to_num(pdf_local)
    pdf_local = pdf_local / max(float(np.sum(pdf_local)), 1e-12)

    expected_life = float(weibull_min.mean(c=float(k_shape), scale=float(lambda_scale)))
    inflow_local = np.zeros(n_local, dtype=float)
    outflow_local = np.zeros(n_local, dtype=float)
    steady_inflow = max(float(stock_arr[0]) / max(expected_life, 1.0), 1e-6)
    hist_inflow = np.full(max_life, steady_inflow, dtype=float)

    for ti in range(n_local):
        past = np.concatenate([hist_inflow, inflow_local[:ti]])
        use = past[-max_life:]
        outflow_local[ti] = float(np.sum(use[::-1] * pdf_local[:len(use)]))
        if ti == 0:
            raw_in = outflow_local[ti]
        else:
            raw_in = (stock_arr[ti] - stock_arr[ti - 1]) + outflow_local[ti]
        inflow_local[ti] = max(0.0, raw_in)

    hist_floor_mask = (
        (years_arr >= max(int(fit_start), int(fit_end) - 9))
        & (years_arr <= int(fit_end))
        & np.isfinite(inflow_local)
        & (inflow_local > 0)
    )
    if np.any(hist_floor_mask):
        floor_anchor = float(np.nanmedian(inflow_local[hist_floor_mask]))
    else:
        pos_inflow = inflow_local[np.isfinite(inflow_local) & (inflow_local > 0)]
        floor_anchor = float(np.nanmedian(pos_inflow)) if pos_inflow.size else 0.0


    start_floor = 0.30 * floor_anchor
    long_floor = 0.08 * floor_anchor
    floor_tau = 32.0
    nonzero_activity_floor = np.zeros(n_local, dtype=float)
    fut_floor = years_arr >= int(fit_end) + 1
    nonzero_activity_floor[fut_floor] = (
        long_floor
        + (start_floor - long_floor)
        * np.exp(-(years_arr[fut_floor] - (int(fit_end) + 1)) / floor_tau)
    )
    nonzero_activity_floor = np.maximum(nonzero_activity_floor, 0.0)
    effective_inflow_for_B = np.where(
        years_arr >= int(fit_end) + 1,
        np.maximum(inflow_local, nonzero_activity_floor),
        inflow_local,
    )

    B_local = effective_inflow_for_B * cement_rate_arr * transition_factor_arr / max(float(output_unit_scale), 1e-12)
    B_local = np.nan_to_num(B_local, nan=0.0, posinf=0.0, neginf=0.0)

    if int(np.nanmin(years_arr)) <= 1960:
        phase_defs = [
            ("p00_1960_1969", (years_arr >= 1960) & (years_arr <= 1969)),
            ("p01_1970_1979", (years_arr >= 1970) & (years_arr <= 1979)),
            ("p02_1980_1989", (years_arr >= 1980) & (years_arr <= 1989)),
        ]
    else:
        phase_defs = []
    phase_defs += [
        ("p03_1990_1995", (years_arr >= 1990) & (years_arr <= 1995)),
        ("p04_1996_2000", (years_arr >= 1996) & (years_arr <= 2000)),
        ("p05_2001_2005", (years_arr >= 2001) & (years_arr <= 2005)),
        ("p06_2006_2010", (years_arr >= 2006) & (years_arr <= 2010)),
        ("p07_2011_2015", (years_arr >= 2011) & (years_arr <= 2015)),
        ("p08_2016_2019", (years_arr >= 2016) & (years_arr <= 2019)),
        ("p09_2020", years_arr == 2020),
        ("p10_2021", years_arr == 2021),
        ("p11_2022", years_arr == 2022),
        ("p12_2023", years_arr == 2023),
        ("p13_2024", years_arr == 2024),
    ]

    multiplier = np.full(n_local, np.nan, dtype=float)
    phase_rows = []
    for name, mask in phase_defs:
        fit_mask_local = (
            mask
            & (years_arr >= int(fit_start))
            & (years_arr <= int(fit_end))
            & np.isfinite(actual_arr)
            & np.isfinite(B_local)
            & (B_local > 1e-12)
        )
        if not np.any(fit_mask_local):
            continue
        bj = float(
            np.sum(B_local[fit_mask_local] * actual_arr[fit_mask_local])
            / max(np.sum(B_local[fit_mask_local] ** 2), 1e-12)
        )
        bj = float(np.clip(bj, 0.0, 1e6))
        multiplier[mask] = bj
        phase_rows.append({"phase": name, "multiplier": bj, "n_fit": int(np.sum(fit_mask_local))})

    if np.any(np.isfinite(multiplier)):
        multiplier = pd.Series(multiplier).ffill().bfill().to_numpy(dtype=float).copy()
    else:
        multiplier = np.ones(n_local, dtype=float)

    hist_idx = np.where(years_arr <= int(fit_end))[0]
    if len(hist_idx) > 0:
        anchor_mult = float(multiplier[hist_idx[-1]])
    else:
        anchor_mult = float(np.nanmedian(multiplier[np.isfinite(multiplier)]))
    multiplier[years_arr > int(fit_end)] = anchor_mult

    raw_pred = np.maximum(B_local * multiplier, 0.0)
    scale_mask = (
        (years_arr >= int(fit_start))
        & (years_arr <= int(fit_end))
        & np.isfinite(actual_arr)
        & np.isfinite(raw_pred)
        & (raw_pred > 1e-12)
    )
    if np.any(scale_mask):
        global_scale = float(
            np.sum(raw_pred[scale_mask] * actual_arr[scale_mask])
            / max(np.sum(raw_pred[scale_mask] ** 2), 1e-12)
        )
    else:
        global_scale = 1.0

    pred = np.maximum(raw_pred * global_scale, 0.0)
    return (
        pred,
        B_local,
        inflow_local,
        outflow_local,
        nonzero_activity_floor,
        effective_inflow_for_B,
        multiplier,
        global_scale,
        pd.DataFrame(phase_rows),
    )


def _splice_future_to_fit_end(years_arr, pred_arr, final_arr, fit_end):
    years_arr = np.asarray(years_arr, dtype=int)
    pred_arr = np.asarray(pred_arr, dtype=float)
    final_arr = np.asarray(final_arr, dtype=float)
    if np.any(years_arr == int(fit_end)):
        ref = float(final_arr[years_arr == int(fit_end)][0])
        base = float(pred_arr[years_arr == int(fit_end)][0])
        if np.isfinite(ref) and np.isfinite(base) and base > 1e-12:
            return ref / base
    return 1.0


try:
    plt.rcParams["font.family"] = ["Times New Roman"]
except Exception:
    pass

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
np.set_printoptions(suppress=True)
rng = np.random.default_rng(42)


data_path = str(_input_path('china'))
sheet_name = "Sheet2"

base_dir = str(_output_dir('china', 'representative'))
output_xlsx = os.path.join(base_dir, "中国_水泥_DevelopedScenario_TargetedFit_with_outflow_minfloor.xlsx")
output_png = os.path.join(base_dir, "中国_水泥_DevelopedScenario_TargetedFit_with_outflow_minfloor.png")


def pick_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    for c in df.columns:
        for kw in cands:
            if kw in c:
                return c
    raise KeyError(f"找不到列：候选={cands}，实际列={df.columns.tolist()}")

df = pd.read_excel(data_path, sheet_name=sheet_name)
df.columns = df.columns.astype(str).str.strip()

if "年份" not in df.columns:
    raise KeyError(f"缺少列 '年份'。当前列名：{df.columns.tolist()}")

df = df.sort_values("年份").reset_index(drop=True)
years = df["年份"].astype(int).values
n = len(years)

pop_col = pick_col(df, ["中国人口数量", "人口", "人口数"])
area_col = pick_col(df, ["拟合人均住宅面积", "人均住宅面积", "人均住宅面积(㎡/人)"])

try:
    act_col = pick_col(df, ["中国水泥产量（亿吨）", "中国水泥产量(亿吨)", "水泥产量（亿吨）", "水泥产量(亿吨)", "水泥产量"])
except Exception:
    act_col = None

population = pd.to_numeric(df[pop_col], errors="coerce").values.astype(float)
pc_area = pd.to_numeric(df[area_col], errors="coerce").values.astype(float)

actual_raw = pd.to_numeric(df[act_col], errors="coerce").values.astype(float) if act_col else np.full(n, np.nan, float)


if np.isfinite(actual_raw).any():
    actual_tons = actual_raw * (1e8 if np.nanmax(np.abs(actual_raw)) < 1e6 else 1.0)
else:
    actual_tons = actual_raw

actual_mask = np.isfinite(actual_tons) & (actual_tons >= 0)
fit_mask = actual_mask & (years >= 1990) & (years <= 2024)

print(f"✅ 读取成功：sheet={sheet_name} | 年份数 n={n} | 真实产量点={int(actual_mask.sum())}")
print("列名：", {"人口": pop_col, "人均面积": area_col, "产量": act_col})


TRANSITION_RATE = 2.4
infra_factor_base = 1.0 + TRANSITION_RATE
k_shape      = 1.97
lambda_scale = 56.4
max_life     = 100

life = np.arange(1, max_life + 1)
pdf_weib = weibull_min.pdf(life, c=k_shape, scale=lambda_scale)
pdf_weib = pdf_weib / pdf_weib.sum()

stock = population * pc_area
known_initial_stock = 3.477e9
stock = stock + (known_initial_stock - stock[0])

inflow = np.zeros(n)
outflow = np.zeros(n)

for t in range(1, n):
    if t >= max_life:
        outflow[t] = np.sum(inflow[t-max_life:t][::-1] * pdf_weib)
    else:
        outflow[t] = np.sum(inflow[:t][::-1] * pdf_weib[:t])
    raw_in = (stock[t] - stock[t-1]) + outflow[t]
    inflow[t] = max(0.0, raw_in)

def stock_identity_error(stock, inflow, outflow):
    recon = np.zeros_like(stock)
    recon[0] = stock[0]
    for t in range(1, len(stock)):
        recon[t] = recon[t - 1] + inflow[t] - outflow[t]
    return float(np.nanmax(np.abs(recon - stock) / np.maximum(stock, 1e-9)) * 100.0)

stock_err_const = stock_identity_error(stock, inflow, outflow)

def build_nonzero_activity_floor(inflow_arr):

    inflow_arr = np.asarray(inflow_arr, dtype=float)
    hist_mask = (years >= 2015) & (years <= 2024) & np.isfinite(inflow_arr) & (inflow_arr > 0)
    if np.any(hist_mask):
        anchor = float(np.nanmedian(inflow_arr[hist_mask]))
    else:
        pos = inflow_arr[np.isfinite(inflow_arr) & (inflow_arr > 0)]
        anchor = float(np.nanmedian(pos)) if pos.size else 0.0

    start_floor = 0.30 * anchor
    long_floor = 0.08 * anchor
    tau = 32.0

    floor = np.zeros_like(inflow_arr, dtype=float)
    fut = years >= 2025
    floor[fut] = long_floor + (start_floor - long_floor) * np.exp(-(years[fut] - 2025) / tau)
    return np.maximum(floor, 0.0), anchor, start_floor, long_floor, tau

nonzero_activity_floor, nonzero_floor_anchor, nonzero_start_floor, nonzero_long_floor, nonzero_floor_tau = (
    build_nonzero_activity_floor(inflow)
)
effective_inflow_for_B = np.where(
    years >= 2025,
    np.maximum(inflow, nonzero_activity_floor),
    inflow,
)


def cement_rate_scalar(y: int) -> float:
    if y <= 2000:
        return 0.36
    elif 2001 <= y <= 2010:
        return 0.38
    elif 2011 <= y <= 2013:
        return 0.39
    elif y == 2014:
        return 0.41
    elif y == 2015:
        return 0.38
    elif 2016 <= y <= 2020:
        return 0.36
    elif y == 2021:
        return 0.35
    elif y == 2022:
        return 0.345
    elif y == 2023:
        return 0.335
    elif y == 2024:
        return 0.33
    elif 2025 <= y <= 2030:
        return 0.32
    else:
        return 0.30

def infra_taper_scalar(y: int) -> float:
    if y <= 2024:
        return 1.00
    elif 2025 <= y <= 2030:
        return 1.00 + (0.92 - 1.00) * (y - 2024) / (2030 - 2024)
    elif 2031 <= y <= 2035:
        return 0.92 + (0.85 - 0.92) * (y - 2030) / (2035 - 2030)
    else:
        return 0.85

def retrofit_share_scalar(y: int) -> float:
    if y <= 2024:
        return 0.0
    elif 2025 <= y <= 2030:
        return 0.00 + (0.10 - 0.00) * (y - 2024) / (2030 - 2024)
    elif 2031 <= y <= 2040:
        return 0.10 + (0.15 - 0.10) * (y - 2030) / (2040 - 2030)
    else:
        return 0.20

cement_rate_arr = np.array([cement_rate_scalar(int(y)) for y in years], dtype=float)
infra_taper_arr = np.array([infra_taper_scalar(int(y)) for y in years], dtype=float)
retrofit_arr = np.array([retrofit_share_scalar(int(y)) for y in years], dtype=float)

def build_B(ci_scale=1.0, infra_scale=1.0, retro_scale=1.0, soften_tau=15.0):
    eff = effective_inflow_for_B * (1.0 - np.minimum(0.99, retrofit_arr * retro_scale))
    base = eff * (cement_rate_arr * ci_scale) * (infra_factor_base * infra_taper_arr * infra_scale)

    mask_anchor = (years >= 2024) & (years <= 2026)
    target = np.nanmean(base[mask_anchor]) if mask_anchor.any() else base[-1]

    base_sc = base.copy()
    fut = years >= 2025
    if np.any(fut):
        decay = np.exp(-(years[fut] - 2025) / float(soften_tau))
        base_sc[fut] = target + (base_sc[fut] - target) * decay
    return base_sc


seg_defs = [
    ("p90_94", (years >= 1990) & (years <= 1994)),
    ("p95_99", (years >= 1995) & (years <= 1999)),
    ("p00_04", (years >= 2000) & (years <= 2004)),
    ("p05_08", (years >= 2005) & (years <= 2008)),
    ("p09_10", (years >= 2009) & (years <= 2010)),
    ("p11_13", (years >= 2011) & (years <= 2013)),
    ("p2014",  (years == 2014)),
    ("p2015",  (years == 2015)),
    ("p16_18", (years >= 2016) & (years <= 2018)),
    ("p19_20", (years >= 2019) & (years <= 2020)),
    ("p2021",  (years == 2021)),
    ("p2022",  (years == 2022)),
    ("p2023",  (years == 2023)),
    ("p2024",  (years == 2024)),
]
SEG_MAT = np.column_stack([m.astype(float) for _, m in seg_defs])
K = SEG_MAT.shape[1]

def design_X(B):
    return B[:, None] * SEG_MAT

def seg_weights():
    w = np.ones(n, dtype=float)
    w[(years >= 1990) & (years <= 1999)] *= 1.4
    w[(years >= 2000) & (years <= 2008)] *= 1.8
    w[(years >= 2009) & (years <= 2015)] *= 3.5
    w[(years >= 2016) & (years <= 2020)] *= 5.0
    w[(years >= 2021) & (years <= 2024)] *= 5.5
    w[(years == 2014)] *= 2.0
    return w

W_FULL = seg_weights()


def masked_arrays(y_true, y_pred, mask):
    m = mask & np.isfinite(y_true) & np.isfinite(y_pred)
    return np.asarray(y_true[m], dtype=float), np.asarray(y_pred[m], dtype=float)

def calc_mape(y_true, y_pred, mask):
    yt, yp = masked_arrays(y_true, y_pred, mask)
    valid = yt != 0
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((yt[valid] - yp[valid]) / yt[valid])) * 100.0)

def calc_maem(y_true, y_pred, mask):
    yt, yp = masked_arrays(y_true, y_pred, mask)
    if yt.size == 0:
        return np.nan
    return float(np.mean(np.abs(yt - yp)))

def calc_rmse(y_true, y_pred, mask):
    yt, yp = masked_arrays(y_true, y_pred, mask)
    if yt.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((yt - yp) ** 2)))

def calc_r2(y_true, y_pred, mask):
    yt, yp = masked_arrays(y_true, y_pred, mask)
    if yt.size < 2:
        return np.nan
    y_bar = np.mean(yt)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - y_bar) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)

def calc_dw(y_true, y_pred, mask):
    yt, yp = masked_arrays(y_true, y_pred, mask)
    if yt.size < 2:
        return np.nan
    e = yt - yp
    denom = np.sum(e ** 2)
    if denom == 0:
        return np.nan
    num = np.sum(np.diff(e) ** 2)
    return float(num / denom)

def calc_tail_linear_metrics(years_arr, y_pred, start_year, end_year):
    m = (years_arr >= start_year) & (years_arr <= end_year) & np.isfinite(y_pred)
    if np.sum(m) < 2:
        return {
            "Tail_Slope_%": np.nan,
            "Tail_R2": np.nan,
            "Drop_%": np.nan,
            "Slope_ton_per_year": np.nan,
            "Start_Value": np.nan,
            "End_Value": np.nan
        }

    x = years_arr[m].astype(float)
    y = y_pred[m].astype(float)

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    sxx = np.sum((x - x_mean) ** 2)

    if sxx == 0:
        return {
            "Tail_Slope_%": np.nan,
            "Tail_R2": np.nan,
            "Drop_%": np.nan,
            "Slope_ton_per_year": np.nan,
            "Start_Value": y[0],
            "End_Value": y[-1]
        }

    sxy = np.sum((x - x_mean) * (y - y_mean))
    k = sxy / sxx
    b = y_mean - k * x_mean
    y_fit = k * x + b

    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    tail_r2 = np.nan if ss_tot == 0 else float(1.0 - ss_res / ss_tot)

    tail_slope_pct = np.nan if y_mean == 0 else float(100.0 * k / y_mean)
    y_start = y[0]
    y_end = y[-1]
    drop_pct = np.nan if y_start == 0 else float((y_end / y_start - 1.0) * 100.0)

    return {
        "Tail_Slope_%": tail_slope_pct,
        "Tail_R2": tail_r2,
        "Drop_%": drop_pct,
        "Slope_ton_per_year": float(k),
        "Start_Value": float(y_start),
        "End_Value": float(y_end)
    }


def fit_beta_pw(B, lam=1e-3, beta_lo=0.40, beta_hi=4.50, max_iter=12):
    X = design_X(B)
    y = actual_tons.copy()
    m = fit_mask & np.isfinite(B) & (B > 0) & np.isfinite(y) & (y >= 0)

    idx = np.where(m)[0]
    if len(idx) < K + 2:
        return np.ones(K, dtype=float), 0.0, B.copy()

    Xh = X[idx].astype(float)
    yh = y[idx].astype(float)
    wh = W_FULL[idx].astype(float)


    sw = np.sqrt(wh)
    Xw = Xh * sw[:, None]
    yw = yh * sw
    beta = np.linalg.solve(Xw.T @ Xw + lam * np.eye(K), Xw.T @ yw)

    rho = 0.0

    for _ in range(max_iter):
        resid = yh - Xh @ beta

        if len(resid) < 3:
            break

        e0 = resid[:-1]
        e1 = resid[1:]
        denom = float(np.dot(e0, e0))
        rho_new = float(np.dot(e0, e1) / denom) if denom > 0 else 0.0
        rho_new = float(np.clip(rho_new, -0.30, 0.92))


        Xt = Xh.copy()
        yt = yh.copy()
        wt = wh.copy()

        if len(yh) >= 1:
            f0 = np.sqrt(max(1e-10, 1.0 - rho_new ** 2))
            Xt[0, :] = f0 * Xh[0, :]
            yt[0] = f0 * yh[0]

        if len(yh) >= 2:
            Xt[1:, :] = Xh[1:, :] - rho_new * Xh[:-1, :]
            yt[1:] = yh[1:] - rho_new * yh[:-1]

        swt = np.sqrt(wt)
        Xtw = Xt * swt[:, None]
        ytw = yt * swt

        beta_new = np.linalg.solve(Xtw.T @ Xtw + lam * np.eye(K), Xtw.T @ ytw)
        beta_new = np.clip(beta_new, beta_lo, beta_hi)


        idxm = {name: i for i, (name, _) in enumerate(seg_defs)}
        beta_new[idxm["p2022"]] = min(beta_new[idxm["p2022"]], beta_new[idxm["p2021"]])
        beta_new[idxm["p2023"]] = min(beta_new[idxm["p2023"]], beta_new[idxm["p2022"]])
        beta_new[idxm["p2024"]] = min(beta_new[idxm["p2024"]], beta_new[idxm["p2023"]])

        if np.max(np.abs(beta_new - beta)) < 1e-8 and abs(rho_new - rho) < 1e-8:
            beta, rho = beta_new, rho_new
            break

        beta, rho = beta_new, rho_new

    y_hist_fit = X @ beta
    return beta, rho, y_hist_fit


def apply_multipliers(B, beta, decay=0.995, floor_frac=0.62):
    X = design_X(B)
    y_fit = X @ beta

    denom = np.where(B != 0, B, 1.0)
    mult_full = np.zeros(n, dtype=float)

    hist = years <= 2024
    mult_full[hist] = y_fit[hist] / denom[hist]


    if np.any(years == 2024):
        i24 = np.where(years == 2024)[0][0]
        base_mult24 = y_fit[i24] / denom[i24]
    else:
        base_mult24 = 1.0

    fut = years > 2024
    if np.any(fut):
        t = (years[fut] - 2024).astype(float)
        mult_future = base_mult24 * (decay ** t)
        mult_floor = floor_frac * base_mult24
        mult_full[fut] = np.maximum(mult_floor, mult_future)

    yhat = B * mult_full
    return yhat, mult_full

def best_scale_by_mape(y_true, y_pred, mask, lo=0.94, hi=1.06, num=121):
    if not np.any(mask):
        return 1.0
    scales = np.linspace(lo, hi, num)
    best_s = 1.0
    best_mape = np.inf
    for s in scales:
        mape = calc_mape(y_true, y_pred * s, mask)
        if np.isfinite(mape) and mape < best_mape:
            best_mape = mape
            best_s = float(s)
    return best_s


def evaluate(ci_s, infra_s, retro_s, tau, dcy, flr):
    B = build_B(ci_scale=ci_s, infra_scale=infra_s, retro_scale=retro_s, soften_tau=tau)

    beta, rho, y_hist_fit = fit_beta_pw(B, lam=1e-3, beta_lo=0.40, beta_hi=4.50, max_iter=12)
    yhat, mult = apply_multipliers(B, beta, decay=dcy, floor_frac=flr)


    scale = best_scale_by_mape(actual_tons, yhat, fit_mask, lo=0.94, hi=1.06, num=121)
    yhat = yhat * scale

    fit_metrics = {
        "MAPE_%": calc_mape(actual_tons, yhat, fit_mask),
        "MAEM": calc_maem(actual_tons, yhat, fit_mask),
        "RMSE": calc_rmse(actual_tons, yhat, fit_mask),
        "R2": calc_r2(actual_tons, yhat, fit_mask),
        "DW": calc_dw(actual_tons, yhat, fit_mask),
        "N": int(np.sum(fit_mask))
    }

    pred_metrics = calc_tail_linear_metrics(years, yhat, 2025, 2100)


    ok_mape = np.isfinite(fit_metrics["MAPE_%"]) and (fit_metrics["MAPE_%"] <= 10.0)
    ok_dw = np.isfinite(fit_metrics["DW"]) and (1.5 <= fit_metrics["DW"] <= 2.5)
    ok_pos = np.nanmin(yhat) >= 0


    ok_tail = True
    if np.isfinite(pred_metrics["Tail_Slope_%"]):
        ok_tail = (-2.5 <= pred_metrics["Tail_Slope_%"] <= 0.3)

    ok_drop = True
    if np.isfinite(pred_metrics["Drop_%"]):
        ok_drop = (-80.0 <= pred_metrics["Drop_%"] <= 5.0)

    ok_all = ok_mape and ok_dw and ok_pos and ok_tail and ok_drop


    score = 0.0
    score += 2000.0 * max(0.0, fit_metrics["MAPE_%"] - 10.0) if np.isfinite(fit_metrics["MAPE_%"]) else 1e9
    score += 1500.0 * max(0.0, 1.5 - fit_metrics["DW"]) if np.isfinite(fit_metrics["DW"]) else 1e9
    score += 1500.0 * max(0.0, fit_metrics["DW"] - 2.5) if np.isfinite(fit_metrics["DW"]) else 1e9
    score += 8.0 * fit_metrics["MAPE_%"] if np.isfinite(fit_metrics["MAPE_%"]) else 1e9
    score += 50.0 * abs(2.0 - fit_metrics["DW"]) if np.isfinite(fit_metrics["DW"]) else 1e9
    score += (1.0 - fit_metrics["R2"]) * 100.0 if np.isfinite(fit_metrics["R2"]) else 1000.0

    if np.isfinite(pred_metrics["Tail_Slope_%"]) and pred_metrics["Tail_Slope_%"] > 0:
        score += 200.0 + abs(pred_metrics["Tail_Slope_%"]) * 20.0
    if np.isfinite(pred_metrics["Drop_%"]) and pred_metrics["Drop_%"] > 5:
        score += 200.0 + abs(pred_metrics["Drop_%"]) * 5.0
    if not ok_pos:
        score += 1e6

    return {
        "ok_all": ok_all,
        "score": float(score),
        "yhat": yhat,
        "B": B,
        "mult": mult,
        "beta": beta,
        "rho": rho,
        "scale": scale,
        "pars": {
            "ci_scale": ci_s,
            "infra_scale": infra_s,
            "retro_scale": retro_s,
            "tau": tau,
            "decay": dcy,
            "floor": flr
        },
        "fit_metrics": fit_metrics,
        "pred_metrics": pred_metrics
    }

def better_than(a, b):
    if b is None:
        return True
    if a["ok_all"] and not b["ok_all"]:
        return True
    if a["ok_all"] == b["ok_all"]:
        return a["score"] < b["score"]
    return False


ci_scales = np.round(np.arange(0.94, 1.031, 0.01), 3).tolist()
infra_scales = np.round(np.arange(0.93, 1.031, 0.01), 3).tolist()
retro_scales = np.round(np.arange(0.82, 0.981, 0.02), 3).tolist()
taus = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
decays = [0.992, 0.993, 0.994, 0.995, 0.996, 0.997]
floors = [0.50, 0.54, 0.58, 0.62, 0.66]

total = len(ci_scales) * len(infra_scales) * len(retro_scales) * len(taus) * len(decays) * len(floors)
print(f"\n✅ 将评估 {total} 个候选\n")

best = None
cnt = 0

for ci in ci_scales:
    for infra in infra_scales:
        for retro in retro_scales:
            for tau in taus:
                for dcy in decays:
                    for flr in floors:
                        cnt += 1
                        if cnt % 500 == 0:
                            if best is None:
                                print(f"[progress] {cnt}/{total} ({cnt/total*100:.1f}%)")
                            else:
                                print(
                                    f"[progress] {cnt}/{total} ({cnt/total*100:.1f}%) "
                                    f"best_MAPE={best['fit_metrics']['MAPE_%']:.4f}, "
                                    f"best_DW={best['fit_metrics']['DW']:.4f}, "
                                    f"best_R2={best['fit_metrics']['R2']:.4f}"
                                )

                        cand = evaluate(ci, infra, retro, tau, dcy, flr)

                        if better_than(cand, best):
                            best = cand
                            print(
                                f"[best] ok={best['ok_all']} "
                                f"MAPE={best['fit_metrics']['MAPE_%']:.4f}% "
                                f"DW={best['fit_metrics']['DW']:.6f} "
                                f"R2={best['fit_metrics']['R2']:.6f} "
                                f"score={best['score']:.3f}"
                            )


                        if best is not None and best["ok_all"]:
                            print("✅ 已找到满足 MAPE<=10 且 1.5<=DW<=2.5 的组合，提前结束搜索。")
                            break
                    if best is not None and best["ok_all"]:
                        break
                if best is not None and best["ok_all"]:
                    break
            if best is not None and best["ok_all"]:
                break
        if best is not None and best["ok_all"]:
            break
    if best is not None and best["ok_all"]:
        break

if best is None:
    raise RuntimeError("best=None：请检查产量列是否为空或全NaN。")

yhat_final = best["yhat"].copy()


PARAM_K_SHAPE = 1.97
PARAM_LAMBDA_SCALE = 56.4
PARAM_MAX_LIFE = 100
PARAM_RES_INTENSITY = 0.22
PARAM_TRANSITION_RATE = 2.4
_china_ci = np.full_like(years, PARAM_RES_INTENSITY, dtype=float)
_china_transition = np.full_like(years, 1.0 + PARAM_TRANSITION_RATE, dtype=float)
(
    China_pred,
    China_B,
    China_inflow,
    China_outflow,
    China_nonzero_activity_floor,
    China_effective_inflow_for_B,
    China_multiplier,
    China_global_scale,
    China_phase_multipliers,
) = _china_logic_phase_forecast(
    years, stock, actual_tons, 1990, 2024,
    PARAM_K_SHAPE, PARAM_LAMBDA_SCALE, PARAM_MAX_LIFE,
    _china_ci, _china_transition,
    output_unit_scale=1.0,
    kernel_method="pdf",
)
China_splice_scale = _splice_future_to_fit_end(years, China_pred, yhat_final, 2024)
_china_future_mask = years >= 2025
yhat_final[_china_future_mask] = China_pred[_china_future_mask] * China_splice_scale
best["B"] = China_B
best["mult"] = China_multiplier * China_global_scale * China_splice_scale
best["scale"] = China_global_scale * China_splice_scale


final_fit_metrics = {
    "MAPE_%": calc_mape(actual_tons, yhat_final, fit_mask),
    "MAEM": calc_maem(actual_tons, yhat_final, fit_mask),
    "RMSE": calc_rmse(actual_tons, yhat_final, fit_mask),
    "R2": calc_r2(actual_tons, yhat_final, fit_mask),
    "DW": calc_dw(actual_tons, yhat_final, fit_mask),
    "N": int(np.sum(fit_mask))
}
final_pred_metrics = calc_tail_linear_metrics(years, yhat_final, 2025, 2100)

def _fmt(x, nd=4):
    try:
        if not np.isfinite(x):
            return "NA"
    except Exception:
        return str(x)
    return f"{x:.{nd}f}"

print("\n================ 结果汇总 ================\n")
future_mask_check = (years >= 2025) & np.isfinite(yhat_final)
future_min_pred = float(np.nanmin(yhat_final[future_mask_check])) if np.any(future_mask_check) else np.nan
future_y2100 = float(yhat_final[years == 2100][0]) if np.any(years == 2100) else np.nan
future_eff_2100 = float(China_effective_inflow_for_B[years == 2100][0]) if np.any(years == 2100) else np.nan
print(f"逻辑版本：{MIN_FLOOR_LOGIC_VERSION}")
print(
    f"防归零检查：min_pred_2025_2100={_fmt(future_min_pred,4)} | "
    f"y2100={_fmt(future_y2100,4)} | China_effective_inflow_2100={_fmt(future_eff_2100,4)}"
)
print("参数：", best["pars"])
print(f"scale(最小化MAPE)：{_fmt(best['scale'],4)} | rho(PW/FGLS)：{_fmt(best['rho'],6)}")
print("------ 拟合段(1990-2024) ------")
print(f"N       = {final_fit_metrics['N']}")
print(f"MAPE(%) = {_fmt(final_fit_metrics['MAPE_%'], 4)}")
print(f"MAEM    = {_fmt(final_fit_metrics['MAEM'], 4)}")
print(f"RMSE    = {_fmt(final_fit_metrics['RMSE'], 4)}")
print(f"R2      = {_fmt(final_fit_metrics['R2'], 6)}")
print(f"DW      = {_fmt(final_fit_metrics['DW'], 6)}")
print("------ 预测段(2025-2100) ------")
print(f"Tail Slope(%) = {_fmt(final_pred_metrics['Tail_Slope_%'], 6)}")
print(f"Tail R2       = {_fmt(final_pred_metrics['Tail_R2'], 6)}")
print(f"Drop(%)       = {_fmt(final_pred_metrics['Drop_%'], 6)}")

if not (np.isfinite(final_fit_metrics["MAPE_%"]) and final_fit_metrics["MAPE_%"] <= 10.0):
    print("⚠ 警告：本次搜索仍未把 MAPE 压到 10% 以下，请继续扩大参数空间。")
if not (np.isfinite(final_fit_metrics["DW"]) and 1.5 <= final_fit_metrics["DW"] <= 2.5):
    print("⚠ 警告：本次搜索仍未把 DW 拉入 [1.5, 2.5]，请继续扩大参数空间。")

print("=========================================\n")


def _safe(s):
    return pd.Series(s).replace([np.inf, -np.inf], np.nan)

yoy_pct = np.full_like(yhat_final, np.nan, dtype=float)
yoy_pct[1:] = (yhat_final[1:] - yhat_final[:-1]) / np.where(yhat_final[:-1] == 0, np.nan, yhat_final[:-1]) * 100.0

model_out = pd.DataFrame({
    "年份": years,
    "人口(人)": _safe(population).round(0),
    "人均住宅面积(㎡/人)": _safe(pc_area).round(4),
    "住宅存量(㎡)": _safe(stock).round(4),
    "新增面积(㎡)": _safe(inflow).round(4),
    "报废面积(㎡)": _safe(outflow).round(4),
    "防归零最低活动面积(㎡)": _safe(nonzero_activity_floor).round(4),
    "有效新增面积_for_B(㎡)": _safe(effective_inflow_for_B).round(4),
    "China_新增面积_含outflow(㎡)": _safe(China_inflow).round(4),
    "China_报废面积(㎡)": _safe(China_outflow).round(4),
    "China_防归零最低活动面积(㎡)": _safe(China_nonzero_activity_floor).round(4),
    "China_有效新增面积_for_B(㎡)": _safe(China_effective_inflow_for_B).round(4),
    "基线B(吨)": _safe(best["B"]).round(4),
    "阶段乘数mult": _safe(best["mult"]).round(8),
    "预测(吨)": _safe(yhat_final).round(4),
    "真实(吨)": _safe(actual_tons).round(4),
    "预测YoY(%)": _safe(yoy_pct).round(6),
})

rationality_checks = pd.DataFrame([{
    "Logic_Version": MIN_FLOOR_LOGIC_VERSION,
    "Fit_Period": "1990-2024",
    "Fit_N": final_fit_metrics["N"],
    "MAPE_%": final_fit_metrics["MAPE_%"],
    "MAEM": final_fit_metrics["MAEM"],
    "RMSE": final_fit_metrics["RMSE"],
    "R2": final_fit_metrics["R2"],
    "DW": final_fit_metrics["DW"],
    "Pred_Period": "2025-2100",
    "Tail_Slope_%": final_pred_metrics["Tail_Slope_%"],
    "Tail_R2": final_pred_metrics["Tail_R2"],
    "Drop_%": final_pred_metrics["Drop_%"],
    "Slope_ton_per_year": final_pred_metrics["Slope_ton_per_year"],
    "Pred_Start_Value_ton": final_pred_metrics["Start_Value"],
    "Pred_End_Value_ton": final_pred_metrics["End_Value"],
    "Pred_Min_2025_2100_ton": future_min_pred,
    "Pred_2100_ton": future_y2100,
    "Nonzero_Floor_Anchor": nonzero_floor_anchor,
    "Nonzero_Floor_Start": nonzero_start_floor,
    "Nonzero_Floor_Long": nonzero_long_floor,
    "Nonzero_Floor_Tau": nonzero_floor_tau,
    "China_Effective_Inflow_2100": future_eff_2100,
    "PW_rho": best["rho"],
    "scale_minMAPE": best["scale"],
    "StockIdentityErr_%": stock_err_const,
    "MAPE_Target_OK": bool(np.isfinite(final_fit_metrics["MAPE_%"]) and final_fit_metrics["MAPE_%"] <= 10.0),
    "DW_Target_OK": bool(np.isfinite(final_fit_metrics["DW"]) and 1.5 <= final_fit_metrics["DW"] <= 2.5),
    "pars": str(best["pars"])
}])

stage_names = [name for name, _ in seg_defs]
stage_multipliers = China_phase_multipliers.copy()

with pd.ExcelWriter(output_xlsx, engine="openpyxl") as w:
    model_out.to_excel(w, sheet_name="Model_Output", index=False)
    rationality_checks.to_excel(w, sheet_name="Rationality_Checks", index=False)
    stage_multipliers.to_excel(w, sheet_name="Stage_Multipliers", index=False)

print(f"✅ 已导出：{output_xlsx}")


plt.figure(figsize=(12, 7))
if np.any(actual_mask):
    plt.plot(years[actual_mask], actual_tons[actual_mask] / 1e6, "ks-", label="Actual (Mt)")
plt.plot(years, yhat_final / 1e6, "r.-", label="Predicted (Mt)")
plt.axvline(2024, color="gray", linestyle="--", label="2024 (fit/predict split)")
plt.xlabel("Year")
plt.ylabel("Cement production (million tons)")
plt.title("China – Cement Production Forecast (Targeted fit for MAPE & DW)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(output_png, dpi=150)
plt.show()

print(f"✅ 图像已保存：{output_png}")
