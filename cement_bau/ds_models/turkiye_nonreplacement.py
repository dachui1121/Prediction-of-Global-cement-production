from cement_bau.shared_history import apply_shared_history as _shared_history
from cement_bau.runtime import input_path as _input_path, output_dir as _output_dir
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min


DATA_PATH = str(_input_path('turkiye'))
SHEET_NAME = "Sheet1"

START_YEAR = 1990
HIS_END = 2024
FORECAST_END = 2100
FIT_START, FIT_END = 1990, 2024
PRED_START, PRED_END = HIS_END + 1, FORECAST_END


k_shape = 1.97
lambda_scale = 56.4
max_life = 100


res_intensity = 0.23
TRANSITION_RATE = 2.4
infra_factor_base = 1.0 + TRANSITION_RATE


FUTURE_MULT_ANCHOR_YEAR = 2024
FUTURE_MULT_DECAY = 0.995
FUTURE_MULT_FLOOR_FRAC = 0.65


BETA_LO = 0.10
BETA_HI = 30.00


SCALE_LO = 0.50
SCALE_HI = 2.50


base_dir = str(_output_dir('turkiye', 'nonreplacement'))
OUT_XLSX = os.path.join(base_dir, "土耳其_水泥需求预测_StockDriven_中国逻辑版.xlsx")
OUT_PNG = os.path.join(base_dir, "土耳其_水泥需求预测_StockDriven_中国逻辑版.png")


plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def resolve_data_path(*args):
    return DATA_PATH


def norm_col(s):
    return str(s).strip().lower().replace(" ", "").replace("_", "")


def pick_col(df: pd.DataFrame, candidates, required=True):

    cols = df.columns.astype(str).str.strip().tolist()
    norm_map = {norm_col(c): c for c in cols}

    for c in candidates:
        key = norm_col(c)
        if key in norm_map:
            return norm_map[key]

    for col in cols:
        nc = norm_col(col)
        for c in candidates:
            if norm_col(c) in nc:
                return col

    if required:
        raise KeyError(f"找不到列：候选={candidates}，实际列={cols}")
    return None


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def percentile(a, p):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return float(np.percentile(a, p))


def metrics(yhat, y, mask):
    yy = np.asarray(y, float)
    yh = np.asarray(yhat, float)
    m = mask & np.isfinite(yy) & np.isfinite(yh)
    if not np.any(m):
        return dict(MAE=np.nan, MAPE=np.nan, RMSE=np.nan, Bias=np.nan, R2=np.nan, DW=np.nan, N=0)

    ya = yy[m]
    yp = yh[m]
    eps = max(1e-9, percentile(np.abs(ya), 20))
    err = yp - ya
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(ya), eps)) * 100.0)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))

    ss_res = float(np.sum((ya - yp) ** 2))
    ss_tot = float(np.sum((ya - np.mean(ya)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan

    den = float(np.sum(err ** 2))
    dw = float(np.sum(np.diff(err) ** 2) / den) if den > 1e-12 and len(err) >= 2 else np.nan

    return dict(MAE=mae, MAPE=mape, RMSE=rmse, Bias=bias, R2=r2, DW=dw, N=int(m.sum()))


def best_scale_by_mape(y_true, y_pred, mask, lo=SCALE_LO, hi=SCALE_HI):
    best_s = 1.0
    best_m = np.inf


    scales1 = np.linspace(lo, hi, 41)
    for s in scales1:
        m = metrics(y_pred * s, y_true, mask)["MAPE"]
        if np.isfinite(m) and m < best_m:
            best_m = m
            best_s = float(s)

    lo2 = max(lo, best_s - 0.025)
    hi2 = min(hi, best_s + 0.025)
    scales2 = np.linspace(lo2, hi2, 31)
    for s in scales2:
        m = metrics(y_pred * s, y_true, mask)["MAPE"]
        if np.isfinite(m) and m < best_m:
            best_m = m
            best_s = float(s)

    return best_s


def tail_slope(years_arr, series, a=2025, b=2100):
    years_arr = np.asarray(years_arr, int)
    y = np.asarray(series, float)
    m = (years_arr >= a) & (years_arr <= b) & np.isfinite(y)
    if np.sum(m) < 3:
        return dict(slope_ton_per_year=np.nan, slope_pct_per_year=np.nan, R2=np.nan)
    x = years_arr[m].astype(float)
    yy = y[m]
    k, b0 = np.polyfit(x, yy, 1)
    fit = k * x + b0
    ss_res = float(np.sum((yy - fit) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
    slope_pct = float(100.0 * k / max(np.mean(yy), 1e-9))
    return dict(slope_ton_per_year=float(k), slope_pct_per_year=slope_pct, R2=r2)


def pick_year_value(years_arr, series, year):
    idx = np.where(np.asarray(years_arr, int) == int(year))[0]
    return float(series[idx[0]]) if idx.size > 0 and np.isfinite(series[idx[0]]) else np.nan


def total_drop_pct(years_arr, series, start_year, end_year):
    y0 = pick_year_value(years_arr, series, start_year)
    y1 = pick_year_value(years_arr, series, end_year)
    if np.isfinite(y0) and y0 > 0 and np.isfinite(y1):
        return float(100.0 * (y1 / y0 - 1.0))
    return np.nan


def fmt(x, nd=2):
    return "NA" if not np.isfinite(x) else f"{x:.{nd}f}"


data_path = resolve_data_path(DATA_PATH)
df = pd.read_excel(data_path, sheet_name=SHEET_NAME)
df.columns = df.columns.astype(str).str.strip()

c_year = pick_col(df, ["年份", "Year", "year"])


c_urban_pop = pick_col(df, ["城镇人口", "城镇人口数", "城镇人口数量", "城市人口", "urban population", "urban_pop"], required=False)
if c_urban_pop is None:
    c_pop_total = pick_col(df, ["人口（人）", "人口", "总人口", "Population"])
    c_urban_rate = pick_col(df, ["城镇化率", "Urbanization"])
else:
    c_pop_total = None
    c_urban_rate = None


c_pc_area = pick_col(df, ["预测人均住宅面积", "拟合人均住宅面积", "城镇人均居住面积", "人均住宅面积", "pc_area", "floor_area_per_capita"])


c_actual_mt = pick_col(df, ["水泥产量（百万吨）", "水泥产量(百万吨)", "cement_production_mt", "actual_mt"], required=False)
c_actual_yt = pick_col(df, ["水泥产量（亿吨）", "水泥产量(亿吨)", "actual_100million_tons"], required=False)
c_actual_generic = pick_col(df, ["水泥产量", "实际产量", "cement production", "ActualCement"], required=False)


df[c_year] = to_num(df[c_year])
df = df[np.isfinite(df[c_year])].copy()
df[c_year] = df[c_year].round().astype(int)
df = df[(df[c_year] >= START_YEAR) & (df[c_year] <= FORECAST_END)].copy()
df = df.sort_values(c_year).reset_index(drop=True)

years = df[c_year].astype(int).values
n = len(years)
if n == 0:
    raise ValueError("筛选 1990–2100 后没有有效年份数据。")

if c_urban_pop is not None:
    population = to_num(df[c_urban_pop]).values.astype(float)
else:
    total_pop = to_num(df[c_pop_total]).values.astype(float)
    urban_rate = to_num(df[c_urban_rate]).values.astype(float)
    if np.nanmax(urban_rate) > 1.5:
        urban_rate = urban_rate / 100.0
    population = total_pop * urban_rate

pc_area = to_num(df[c_pc_area]).values.astype(float)


pop_s = pd.Series(population, index=years, dtype="float64").interpolate(limit_direction="both").bfill().ffill()
pca_s = pd.Series(pc_area, index=years, dtype="float64").interpolate(limit_direction="both").bfill().ffill()
population = pop_s.values.astype(float)
pc_area = pca_s.values.astype(float)

if not np.all(np.isfinite(population)):
    bad = years[~np.isfinite(population)].tolist()
    raise ValueError(f"城镇人口存在无法补齐的缺失年份：{bad}")
if not np.all(np.isfinite(pc_area)):
    bad = years[~np.isfinite(pc_area)].tolist()
    raise ValueError(f"人均住宅面积存在无法补齐的缺失年份：{bad}")


if np.nanmax(population) < 1e5:
    population = population * 1e6

actual_tons = np.full(n, np.nan, dtype=float)
if c_actual_mt is not None:
    actual_tons = to_num(df[c_actual_mt]).values.astype(float) * 1e6
elif c_actual_yt is not None:
    actual_tons = to_num(df[c_actual_yt]).values.astype(float) * 1e8
elif c_actual_generic is not None:
    a = to_num(df[c_actual_generic]).values.astype(float)
    max_abs = np.nanmax(np.abs(a)) if np.any(np.isfinite(a)) else np.nan
    if np.isfinite(max_abs):
        if max_abs < 10:
            actual_tons = a * 1e8
        elif max_abs < 1000:
            actual_tons = a * 1e6
        else:
            actual_tons = a

actual_mask = np.isfinite(actual_tons) & (actual_tons > 0)
last_hist_year = min(HIS_END, int(np.nanmax(years[actual_mask]))) if np.any(actual_mask) else HIS_END
fit_mask = actual_mask & (years >= FIT_START) & (years <= FIT_END)


stock = population * pc_area

life = np.arange(1, max_life + 1)
pdf_weib = weibull_min.pdf(life, c=k_shape, scale=lambda_scale)
pdf_weib = np.nan_to_num(pdf_weib, nan=0.0, posinf=0.0, neginf=0.0)
pdf_weib = pdf_weib / max(pdf_weib.sum(), 1e-12)
exp_life = float(weibull_min.mean(c=k_shape, scale=lambda_scale))

inflow = np.zeros(n, dtype=float)
outflow = np.zeros(n, dtype=float)


steady_in = max(float(stock[0]) / max(exp_life, 1.0), 1e-9)
hist_in = np.full(max_life, steady_in, dtype=float)

for t in range(n):
    past = np.concatenate([hist_in, inflow[:t]])
    use = past[-max_life:]
    outflow[t] = float(np.sum(use[::-1] * pdf_weib[:len(use)]))
    if t == 0:
        inflow[t] = max(stock[1] - stock[0], 0.0) if n > 1 else 0.0
    else:
        raw_in = stock[t] - stock[t - 1]
        inflow[t] = max(0.0, raw_in)


hist_floor_mask = (
    (years >= 2015) & (years <= 2024) &
    np.isfinite(inflow) & (inflow > 0)
)
if np.any(hist_floor_mask):
    floor_anchor = float(np.nanmedian(inflow[hist_floor_mask]))
else:
    pos_in = inflow[np.isfinite(inflow) & (inflow > 0)]
    floor_anchor = float(np.nanmedian(pos_in)) if pos_in.size else 0.0

start_floor = 0.30 * floor_anchor
long_floor = 0.10 * floor_anchor
floor_tau = 35.0
non_repair_activity_floor = (
    long_floor +
    (start_floor - long_floor) * np.exp(-np.maximum(years - 2025, 0) / floor_tau)
)
non_repair_activity_floor = np.maximum(non_repair_activity_floor, 0.0)

effective_inflow_for_B = np.maximum(inflow, non_repair_activity_floor)

B = effective_inflow_for_B * res_intensity * infra_factor_base
B = np.maximum(B, 1e-9)


def get_stage_defs(years_arr):
    y = np.asarray(years_arr, int)
    return [
        ("p01_1990_1992", (y >= 1990) & (y <= 1992)),
        ("p02_1993_1995", (y >= 1993) & (y <= 1995)),
        ("p03_1996_1998", (y >= 1996) & (y <= 1998)),
        ("p04_1999_2001", (y >= 1999) & (y <= 2001)),
        ("p05_2002_2004", (y >= 2002) & (y <= 2004)),
        ("p06_2005_2007", (y >= 2005) & (y <= 2007)),
        ("p07_2008_2010", (y >= 2008) & (y <= 2010)),
        ("p08_2011_2013", (y >= 2011) & (y <= 2013)),
        ("p09_2014_2016", (y >= 2014) & (y <= 2016)),
        ("p10_2017_2019", (y >= 2017) & (y <= 2019)),
        ("p11_2020",      (y == 2020)),
        ("p12_2021",      (y == 2021)),
        ("p13_2022",      (y == 2022)),
        ("p14_2023",      (y == 2023)),
        ("p15_2024",      (y == 2024)),
    ]


def fit_stage_multipliers(B, actual_tons, years_arr):

    mult = np.ones_like(B, dtype=float)
    rows = []
    stage_defs = get_stage_defs(years_arr)

    last_valid_beta = 1.0
    for name, mask_stage in stage_defs:
        m = mask_stage & (years_arr >= FIT_START) & (years_arr <= FIT_END) & np.isfinite(actual_tons) & (actual_tons > 0) & np.isfinite(B) & (B > 0)
        if np.sum(m) >= 1:

            w = np.ones(np.sum(m), dtype=float)
            yy = years_arr[m]
            w += 0.30 * (yy >= 2015)
            bj = float(np.sum(w * B[m] * actual_tons[m]) / max(np.sum(w * B[m] * B[m]), 1e-12))
            bj = float(np.clip(bj, BETA_LO, BETA_HI))
            last_valid_beta = bj
        else:
            bj = last_valid_beta

        mult[mask_stage] = bj
        rows.append({"stage": name, "start_year": int(np.min(years_arr[mask_stage])) if np.any(mask_stage) else np.nan,
                     "end_year": int(np.max(years_arr[mask_stage])) if np.any(mask_stage) else np.nan,
                     "multiplier": bj, "n_fit": int(np.sum(m))})


    hist_defined = np.zeros_like(B, dtype=bool)
    for _, m in stage_defs:
        hist_defined |= m
    if np.any(~hist_defined & (years_arr <= FIT_END)):
        mult[~hist_defined & (years_arr <= FIT_END)] = last_valid_beta

    return mult, pd.DataFrame(rows)


mult_stage, stage_df = fit_stage_multipliers(B, actual_tons, years)


hist_cal_mask = (
    fit_mask &
    np.isfinite(actual_tons) & (actual_tons > 0) &
    np.isfinite(B) & (B > 1e-9)
)
raw_ratio = np.full(n, np.nan, dtype=float)
raw_ratio[hist_cal_mask] = actual_tons[hist_cal_mask] / B[hist_cal_mask]
hist_years_mask = years <= FIT_END

def build_smooth_mult(window, blend, down, up):
    ratio_smooth = (
        pd.Series(raw_ratio, index=years)
        .interpolate(limit_direction="both")
        .rolling(window=window, center=True, min_periods=max(3, window // 2))
        .median()
        .bfill()
        .ffill()
        .to_numpy(dtype=float)
    )
    ratio_smooth = np.clip(ratio_smooth, BETA_LO, BETA_HI)

    mult = mult_stage.copy()
    mult[hist_years_mask] = (
        blend * ratio_smooth[hist_years_mask] +
        (1.0 - blend) * mult_stage[hist_years_mask]
    )

    hist_idx = np.where(hist_years_mask)[0]
    for j in range(1, len(hist_idx)):
        i_prev = hist_idx[j - 1]
        i_cur = hist_idx[j]
        lo = mult[i_prev] * down
        hi = mult[i_prev] * up
        mult[i_cur] = float(np.clip(mult[i_cur], lo, hi))
    return mult, ratio_smooth

best_hist = None
for win in [5, 7, 9]:
    for blend in [0.40, 0.50, 0.60]:
        for down, up in [(0.75, 1.33), (0.80, 1.25), (0.85, 1.18)]:
            cand_mult, cand_ratio = build_smooth_mult(win, blend, down, up)
            cand_pre = B * cand_mult
            cand_scale = best_scale_by_mape(actual_tons, cand_pre, fit_mask)
            cand_pred = np.maximum(cand_pre * cand_scale, 0.0)
            cand_metrics = metrics(cand_pred, actual_tons, fit_mask)

            mape = cand_metrics["MAPE"]
            dw = cand_metrics["DW"]
            dw_pen = 0.0
            if np.isfinite(dw):
                dw_pen = max(0.0, 1.5 - dw) * 40.0 + max(0.0, dw - 2.5) * 40.0
            else:
                dw_pen = 100.0
            mape_pen = max(0.0, mape - 10.0) * 20.0 if np.isfinite(mape) else 1000.0
            score = (mape if np.isfinite(mape) else 1e6) + dw_pen + mape_pen

            pack = (score, cand_mult, cand_ratio, cand_scale, cand_metrics, win, blend, down, up)
            if best_hist is None or score < best_hist[0]:
                best_hist = pack

_, mult_hist, ratio_smooth, scale_global, hist_cal_metrics, smooth_window, blend, smooth_down, smooth_up = best_hist


mult_full = mult_hist.copy()
anchor_year = FUTURE_MULT_ANCHOR_YEAR if FUTURE_MULT_ANCHOR_YEAR in set(years.tolist()) else last_hist_year
anchor_window = (
    (years >= max(FIT_START, anchor_year - 4)) &
    (years <= anchor_year) &
    np.isfinite(mult_hist) & (mult_hist > 0)
)
anchor_mult = float(np.nanmedian(mult_hist[anchor_window])) if np.any(anchor_window) else np.nan
if not np.isfinite(anchor_mult) or anchor_mult <= 0:
    hist_m = (years <= last_hist_year) & np.isfinite(mult_hist) & (mult_hist > 0)
    anchor_mult = float(np.nanmedian(mult_hist[hist_m])) if np.any(hist_m) else 1.0

floor_mult = anchor_mult * FUTURE_MULT_FLOOR_FRAC
for i, y in enumerate(years):
    if y > last_hist_year:
        dt = y - anchor_year
        cand = anchor_mult * (FUTURE_MULT_DECAY ** max(dt, 0))
        mult_full[i] = max(floor_mult, cand)


pred_raw = np.maximum(B * mult_full * scale_global, 0.0)


pred = pred_raw.copy()
if np.any(years == last_hist_year) and np.any(years == PRED_START):
    idx_hist = int(np.where(years == last_hist_year)[0][0])
    idx_start = int(np.where(years == PRED_START)[0][0])
    y_anchor = pred[idx_hist]
    if np.isfinite(actual_tons[idx_hist]) and actual_tons[idx_hist] > 0:
        y_anchor = actual_tons[idx_hist]

    recent = (years >= max(FIT_START, last_hist_year - 4)) & (years <= last_hist_year) & np.isfinite(pred) & (pred > 0)
    if np.sum(recent) >= 2:
        recent_vals = pred[recent]
        recent_yoy = recent_vals[1:] / np.maximum(recent_vals[:-1], 1e-9) - 1.0
        start_yoy = float(np.nanmedian(recent_yoy))
    else:
        start_yoy = 0.0
    start_yoy = float(np.clip(start_yoy, -0.03, 0.03))

    target_start = max(y_anchor * (1.0 + start_yoy), 1e-9)
    raw_start = max(pred_raw[idx_start], 1e-9)
    splice_scale = target_start / raw_start
    pred[years >= PRED_START] = pred_raw[years >= PRED_START] * splice_scale
    pred[idx_start] = target_start

    future_idx = np.where(years > PRED_START)[0]
    for i in future_idx:
        yy = int(years[i])
        prev = max(pred[i - 1], 1e-9)
        if yy <= 2035:
            down_lim, up_lim = 0.92, 1.05
        elif yy <= 2055:
            down_lim, up_lim = 0.93, 1.035
        else:
            down_lim, up_lim = 0.95, 1.02
        pred[i] = float(np.clip(pred[i], prev * down_lim, prev * up_lim))
else:
    splice_scale = 1.0
    start_yoy = np.nan


pred_before_ar = pred.copy()
pre_ar_fit_metrics = metrics(pred, actual_tons, fit_mask)

def apply_history_ar_residual_correction(pred_arr, actual_arr, mask_arr, max_order=4):
    base_pred = np.asarray(pred_arr, dtype=float)
    best_pack = {
        "pred": base_pred.copy(),
        "metrics": metrics(base_pred, actual_arr, mask_arr),
        "order": 0,
        "alpha": 0.0,
        "coef": np.array([], dtype=float),
        "score": np.inf,
    }

    def score_metrics(met):
        mape = met["MAPE"]
        dw = met["DW"]
        if not np.isfinite(mape) or not np.isfinite(dw):
            return 1e9
        score = mape + 12.0 * abs(dw - 2.0)
        score += 300.0 * max(0.0, mape - 10.0)
        score += 200.0 * max(0.0, 1.5 - dw)
        score += 200.0 * max(0.0, dw - 2.5)
        return float(score)

    best_pack["score"] = score_metrics(best_pack["metrics"])
    idx = np.where(mask_arr & np.isfinite(actual_arr) & np.isfinite(base_pred))[0]
    if idx.size < 8:
        return best_pack

    resid = base_pred[idx] - actual_arr[idx]
    alpha_grid = np.linspace(0.0, 1.20, 121)

    for order in range(1, int(max_order) + 1):
        if resid.size <= order + 2:
            continue

        X_rows = []
        y_rows = []
        for i in range(order, resid.size):
            X_rows.append([1.0] + [resid[i - j] for j in range(1, order + 1)])
            y_rows.append(resid[i])
        X = np.asarray(X_rows, dtype=float)
        y = np.asarray(y_rows, dtype=float)
        try:
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        corr = np.zeros_like(resid)
        for i in range(order, resid.size):
            corr[i] = coef[0] + sum(coef[j] * resid[i - j] for j in range(1, order + 1))

        for alpha in alpha_grid:
            cand = base_pred.copy()
            cand[idx] = np.maximum(0.0, base_pred[idx] - alpha * corr)
            cand_metrics = metrics(cand, actual_arr, mask_arr)
            cand_score = score_metrics(cand_metrics)


            if cand_score < best_pack["score"]:
                best_pack = {
                    "pred": cand,
                    "metrics": cand_metrics,
                    "order": order,
                    "alpha": float(alpha),
                    "coef": coef.copy(),
                    "score": cand_score,
                }

    return best_pack

ar_corr_pack = apply_history_ar_residual_correction(pred, actual_tons, fit_mask, max_order=4)
pred = ar_corr_pack["pred"]
ar_corr_order = ar_corr_pack["order"]
ar_corr_alpha = ar_corr_pack["alpha"]
ar_corr_coef = ar_corr_pack["coef"]


pred = _shared_history('turkiye', 'nonreplacement', years, pred, actual_tons)


fit_metrics = metrics(pred, actual_tons, fit_mask)
pred_tail = tail_slope(years, pred, PRED_START, PRED_END)
pred_drop = total_drop_pct(years, pred, PRED_START, PRED_END)


stock_hat = np.zeros_like(stock, dtype=float)
stock_hat[0] = stock[0]
for t in range(1, n):
    stock_hat[t] = stock_hat[t - 1] + inflow[t]
stock_identity_error = stock - stock_hat

print("\n================ 土耳其 Stock-driven 中国逻辑版 ================")
print(f"数据路径：{data_path}")
print(f"历史拟合段 {FIT_START}-{FIT_END}: N={fit_metrics['N']} | "
      f"MAE={fmt(fit_metrics['MAE'],0)} | MAPE={fmt(fit_metrics['MAPE'],2)}% | "
      f"RMSE={fmt(fit_metrics['RMSE'],0)} | R2={fmt(fit_metrics['R2'],4)} | DW={fmt(fit_metrics['DW'],3)}")
print(f"预测段 {PRED_START}-{PRED_END}: TailSlope={fmt(pred_tail['slope_pct_per_year'],3)}%/y | "
      f"TailR2={fmt(pred_tail['R2'],4)} | Drop={fmt(pred_drop,2)}%")
print(f"scale_global={scale_global:.4f} | anchor_year={anchor_year} | anchor_mult={anchor_mult:.4f} | floor_mult={floor_mult:.4f}")
print(f"smooth_window={smooth_window} | blend={blend:.2f} | smooth_bounds=({smooth_down:.2f},{smooth_up:.2f})")
print(f"splice_scale={splice_scale:.4f} | forecast_start_yoy={fmt(start_yoy * 100.0 if np.isfinite(start_yoy) else np.nan, 3)}%")
print(f"AR历史残差校正：pre_MAPE={fmt(pre_ar_fit_metrics['MAPE'],2)}% | pre_DW={fmt(pre_ar_fit_metrics['DW'],3)} | "
      f"order={ar_corr_order} | alpha={ar_corr_alpha:.2f} | "
      f"post_MAPE={fmt(fit_metrics['MAPE'],2)}% | post_DW={fmt(fit_metrics['DW'],3)}")
print("===============================================================\n")


result_df = pd.DataFrame({
    "年份": years,
    "城镇人口(人)": population,
    "人均住宅面积(㎡/人)": pc_area,
    "住宅存量_target(㎡)": stock,
    "住宅存量_hat(㎡)": stock_hat,
    "新增面积_inflow(㎡/年)": inflow,
    "报废面积_outflow(㎡/年)": outflow,
    "非报废最低活动面积(㎡/年)": non_repair_activity_floor,
    "有效新增面积_for_B(㎡/年)": effective_inflow_for_B,
    "基线B_StockDriven(吨)": B,
    "历史乘子延续_m(t)": mult_full,
    "全局scale": scale_global,
    "预测_raw未平滑(吨)": pred_raw,
    "预测_AR校正前(吨)": pred_before_ar,
    "预测_最终(吨)": pred,
    "实际产量(吨)": actual_tons,
    "误差_吨": pred - actual_tons,
    "AR校正前拟合误差_吨": pred_before_ar - actual_tons,
})

weibull_df = pd.DataFrame({
    "寿命年": life,
    "Weibull_pdf": pdf_weib,
})

params_df = pd.DataFrame([
    {"参数": "k_shape", "取值": k_shape},
    {"参数": "lambda_scale", "取值": lambda_scale},
    {"参数": "max_life", "取值": max_life},
    {"参数": "Weibull_mean_life", "取值": exp_life},
    {"参数": "res_intensity_t_per_m2", "取值": res_intensity},
    {"参数": "TRANSITION_RATE", "取值": TRANSITION_RATE},
    {"参数": "infra_factor_base=1+TRANSITION_RATE", "取值": infra_factor_base},
    {"参数": "FUTURE_MULT_ANCHOR_YEAR", "取值": FUTURE_MULT_ANCHOR_YEAR},
    {"参数": "FUTURE_MULT_DECAY", "取值": FUTURE_MULT_DECAY},
    {"参数": "FUTURE_MULT_FLOOR_FRAC", "取值": FUTURE_MULT_FLOOR_FRAC},
    {"参数": "smooth_window", "取值": smooth_window},
    {"参数": "smooth_blend", "取值": blend},
    {"参数": "smooth_down_limit", "取值": smooth_down},
    {"参数": "smooth_up_limit", "取值": smooth_up},
    {"参数": "SmoothCandidate_MAPE_%", "取值": hist_cal_metrics["MAPE"]},
    {"参数": "SmoothCandidate_DW", "取值": hist_cal_metrics["DW"]},
    {"参数": "Pre_AR_Hist_MAPE_1990_2024_%", "取值": pre_ar_fit_metrics["MAPE"]},
    {"参数": "Pre_AR_Hist_DW_1990_2024", "取值": pre_ar_fit_metrics["DW"]},
    {"参数": "AR_residual_correction_order", "取值": ar_corr_order},
    {"参数": "AR_residual_correction_alpha", "取值": ar_corr_alpha},
    {"参数": "AR_residual_correction_coef", "取值": ",".join([f"{x:.10g}" for x in ar_corr_coef]) if len(ar_corr_coef) else ""},
    {"参数": "scale_global", "取值": scale_global},
    {"参数": "forecast_splice_scale", "取值": splice_scale},
    {"参数": "forecast_start_yoy", "取值": start_yoy},
    {"参数": "anchor_year", "取值": anchor_year},
    {"参数": "anchor_mult", "取值": anchor_mult},
    {"参数": "floor_mult", "取值": floor_mult},
    {"参数": "Hist_MAE_1990_2024", "取值": fit_metrics["MAE"]},
    {"参数": "Hist_MAPE_1990_2024_%", "取值": fit_metrics["MAPE"]},
    {"参数": "Hist_RMSE_1990_2024", "取值": fit_metrics["RMSE"]},
    {"参数": "Hist_R2_1990_2024", "取值": fit_metrics["R2"]},
    {"参数": "Hist_DW_1990_2024", "取值": fit_metrics["DW"]},
    {"参数": "Pred_TailSlopePct_2025_2100", "取值": pred_tail["slope_pct_per_year"]},
    {"参数": "Pred_TailR2_2025_2100", "取值": pred_tail["R2"]},
    {"参数": "Pred_DropPct_2025_2100", "取值": pred_drop},
])

stock_identity_df = pd.DataFrame({
    "年份": years,
    "stock_target(㎡)": stock,
    "stock_hat_from_identity(㎡)": stock_hat,
    "identity_error_target_minus_hat(㎡)": stock_identity_error,
})

fit_metrics_df = pd.DataFrame([fit_metrics])
tail_df = pd.DataFrame([pred_tail | {"Drop_%": pred_drop}])

Path(OUT_XLSX).parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    result_df.to_excel(writer, index=False, sheet_name="Result_Series")
    stage_df.to_excel(writer, index=False, sheet_name="Stage_Multipliers")
    params_df.to_excel(writer, index=False, sheet_name="Params")
    fit_metrics_df.to_excel(writer, index=False, sheet_name="Fit_Metrics")
    tail_df.to_excel(writer, index=False, sheet_name="Forecast_Tail")
    stock_identity_df.to_excel(writer, index=False, sheet_name="Stock_Identity")
    weibull_df.to_excel(writer, index=False, sheet_name="Weibull")

print(f"✅ 已导出 Excel：{OUT_XLSX}")


plt.figure(figsize=(13.5, 7))
mask_actual_plot = np.isfinite(actual_tons) & (actual_tons > 0)
if np.any(mask_actual_plot):
    plt.plot(years[mask_actual_plot], actual_tons[mask_actual_plot] / 1e6, "ko-", label="Actual")

hist_plot = years <= last_hist_year
pred_plot = years > last_hist_year
plt.plot(years[hist_plot], pred[hist_plot] / 1e6, "b-", linewidth=2.0, label="Model fit")
plt.plot(years[pred_plot], pred[pred_plot] / 1e6, "b--", linewidth=2.0, label="Forecast: B × historical multiplier")
plt.axvline(last_hist_year, linestyle="--", color="gray", label=f"Forecast start {last_hist_year + 1}")

plt.xlabel("Year")
plt.ylabel("Cement production / demand (million tons)")
plt.title("Turkey cement demand forecast: stock-driven baseline × historical multiplier continuation")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(frameon=True)

info = (
    f"Fit {FIT_START}-{FIT_END}: MAPE={fmt(fit_metrics['MAPE'],2)}%, "
    f"R2={fmt(fit_metrics['R2'],4)}, DW={fmt(fit_metrics['DW'],3)}\n"
    f"Forecast {PRED_START}-{PRED_END}: TailSlope={fmt(pred_tail['slope_pct_per_year'],3)}%/y, "
    f"Drop={fmt(pred_drop,2)}%"
)
plt.annotate(info, xy=(0.02, 0.98), xycoords="axes fraction", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", alpha=0.85))
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"✅ 已导出图片：{OUT_PNG}")
plt.show()
