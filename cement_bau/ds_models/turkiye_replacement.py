from cement_bau.runtime import input_path as _input_path, output_dir as _output_dir
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False

try:
    from scipy.stats import weibull_min
except Exception:
    weibull_min = None


DATA_PATH = str(_input_path('turkiye'))
SHEET_NAME = "Sheet1"
OUTPUT_DIR = str(_output_dir('turkiye', 'replacement'))

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


USE_FORECAST_INFLOW_CAP = True
FORECAST_INFLOW_CAP_FROM = 2025
FORECAST_INFLOW_CAP_UP = 0.12
FORECAST_INFLOW_CAP_DOWN = 0.06


USE_HIST_DW_CORRECTION = True
HIST_DW_RHO_CORRECTION = -0.02


BETA_LO = 0.10
BETA_HI = 8.00


SCALE_LO = 0.82
SCALE_HI = 1.18


def resolve_output_dir(*args):
    return str(_output_dir('turkiye', 'replacement'))


base_dir = str(_output_dir('turkiye', 'replacement'))
OUT_XLSX = os.path.join(base_dir, "土耳其_水泥需求预测_StockDriven_DS_R2优化.xlsx")
OUT_PNG = os.path.join(base_dir, "土耳其_水泥需求预测_StockDriven_DS_R2优化.png")


if HAS_MATPLOTLIB:
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
    err = yp - ya
    mae = float(np.mean(np.abs(err)))


    eps = 1e-9
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(ya), eps)) * 100.0)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))

    ss_res = float(np.sum((ya - yp) ** 2))
    ss_tot = float(np.sum((ya - np.mean(ya)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan

    den = float(np.sum(err ** 2))
    dw = float(np.sum(np.diff(err) ** 2) / den) if den > 1e-12 and len(err) >= 2 else np.nan

    return dict(MAE=mae, MAPE=mape, RMSE=rmse, Bias=bias, R2=r2, DW=dw, N=int(m.sum()))


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

    sw = float(np.sum(w))
    if sw <= 1e-12:
        return float(np.median(v))

    cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, 0.5 * sw, side="left")])


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


def cap_forecast_yoy(series, years_arr, start_year=2025, up=0.12, down=0.06):
    s = np.asarray(series, float).copy()
    years_arr = np.asarray(years_arr, int)
    idxs = np.where(years_arr >= start_year)[0]
    if idxs.size == 0:
        return s

    for i in range(idxs[0], len(s)):
        if i == 0:
            continue
        prev = max(float(s[i - 1]), 1e-9)
        lo = prev * (1.0 - down)
        hi = prev * (1.0 + up)
        s[i] = float(np.clip(s[i], lo, hi))
    return s


def apply_historical_dw_correction(y_pred, y_true, fit_mask_arr, rho=-0.02):
    corrected = np.asarray(y_pred, float).copy()
    y_true = np.asarray(y_true, float)
    fit_mask_arr = np.asarray(fit_mask_arr, bool)

    prev_resid = 0.0
    for i in range(len(corrected)):
        if fit_mask_arr[i] and np.isfinite(y_true[i]) and np.isfinite(y_pred[i]):
            corrected[i] = float(y_pred[i] + rho * prev_resid)
            prev_resid = float(y_true[i] - y_pred[i])
    return corrected


def weibull_pdf_discrete(life, k, lam):
    life = np.asarray(life, dtype=float)
    if weibull_min is not None:
        pdf = weibull_min.pdf(life, c=k, scale=lam)
        mean_life = float(weibull_min.mean(c=k, scale=lam))
    else:

        x = np.clip(life, 1e-12, None)
        pdf = (k / lam) * ((x / lam) ** (k - 1.0)) * np.exp(-((x / lam) ** k))

        mean_life = float(np.sum(life * pdf) / max(np.sum(pdf), 1e-12))
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    pdf = pdf / max(pdf.sum(), 1e-12)
    return pdf, mean_life


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
pdf_weib, exp_life = weibull_pdf_discrete(life, k_shape, lambda_scale)

inflow = np.zeros(n, dtype=float)
outflow = np.zeros(n, dtype=float)


steady_in = max(float(stock[0]) / max(exp_life, 1.0), 1e-9)
hist_in = np.full(max_life, steady_in, dtype=float)

for t in range(n):
    past = np.concatenate([hist_in, inflow[:t]])
    use = past[-max_life:]
    outflow[t] = float(np.sum(use[::-1] * pdf_weib[:len(use)]))
    if t == 0:
        inflow[t] = outflow[t]
    else:
        raw_in = (stock[t] - stock[t - 1]) + outflow[t]
        inflow[t] = max(0.0, raw_in)

if USE_FORECAST_INFLOW_CAP:
    inflow_for_cement = cap_forecast_yoy(
        inflow, years,
        start_year=FORECAST_INFLOW_CAP_FROM,
        up=FORECAST_INFLOW_CAP_UP,
        down=FORECAST_INFLOW_CAP_DOWN
    )
else:
    inflow_for_cement = inflow.copy()


B = inflow_for_cement * res_intensity * infra_factor_base
B = np.maximum(B, 1e-9)


def get_stage_defs(years_arr):
    y = np.asarray(years_arr, int)
    return [
        ("p1_1990_1992", (y >= 1990) & (y <= 1992)),
        ("p2_1993_1995", (y >= 1993) & (y <= 1995)),
        ("p3_1996_1998", (y >= 1996) & (y <= 1998)),
        ("p4_1999_2001", (y >= 1999) & (y <= 2001)),
        ("p5_2002_2004", (y >= 2002) & (y <= 2004)),
        ("p6_2005_2007", (y >= 2005) & (y <= 2007)),
        ("p7_2008_2010", (y >= 2008) & (y <= 2010)),
        ("p8_2011_2013", (y >= 2011) & (y <= 2013)),
        ("p9_2014_2016", (y >= 2014) & (y <= 2016)),


        ("p10a_2017_2018", (y >= 2017) & (y <= 2018)),
        ("p10b_2019",      (y == 2019)),
        ("p11_2020",     (y == 2020)),
        ("p12_2021",     (y == 2021)),
        ("p13_2022",     (y == 2022)),
        ("p14_2023",     (y == 2023)),
        ("p15_2024",     (y == 2024)),
    ]


def fit_stage_multipliers(B, actual_tons, years_arr):


    mult = np.ones_like(B, dtype=float)
    rows = []
    stage_defs = get_stage_defs(years_arr)

    last_valid_beta = 1.0
    for name, mask_stage in stage_defs:
        m = mask_stage & (years_arr >= FIT_START) & (years_arr <= FIT_END) & np.isfinite(actual_tons) & (actual_tons > 0) & np.isfinite(B) & (B > 0)
        if np.sum(m) >= 1:

            w_year = np.ones(np.sum(m), dtype=float)
            yy = years_arr[m]
            w_year += 0.30 * (yy >= 2015)

            ratio = actual_tons[m] / B[m]
            w_mape = w_year * B[m] / actual_tons[m]
            bj = weighted_median(ratio, w_mape)

            if not np.isfinite(bj):
                bj = last_valid_beta
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


mult_hist, stage_df = fit_stage_multipliers(B, actual_tons, years)


pred_hist_pre_scale = B * mult_hist
scale_global = best_scale_by_mape(actual_tons, pred_hist_pre_scale, fit_mask)


mult_full = mult_hist.copy()
anchor_year = FUTURE_MULT_ANCHOR_YEAR if FUTURE_MULT_ANCHOR_YEAR in set(years.tolist()) else last_hist_year
anchor_idx = int(np.where(years == anchor_year)[0][0]) if anchor_year in set(years.tolist()) else int(np.where(years <= last_hist_year)[0][-1])
anchor_mult = float(mult_hist[anchor_idx])
if not np.isfinite(anchor_mult) or anchor_mult <= 0:
    hist_m = (years <= last_hist_year) & np.isfinite(mult_hist) & (mult_hist > 0)
    anchor_mult = float(np.nanmean(mult_hist[hist_m])) if np.any(hist_m) else 1.0

floor_mult = anchor_mult * FUTURE_MULT_FLOOR_FRAC
for i, y in enumerate(years):
    if y > last_hist_year:
        dt = y - anchor_year
        cand = anchor_mult * (FUTURE_MULT_DECAY ** max(dt, 0))
        mult_full[i] = max(floor_mult, cand)


pred = B * mult_full * scale_global
pred = np.maximum(pred, 0.0)
pred_before_dw_correction = pred.copy()
if USE_HIST_DW_CORRECTION:
    pred = apply_historical_dw_correction(
        pred, actual_tons, fit_mask,
        rho=HIST_DW_RHO_CORRECTION
    )
    pred = np.maximum(pred, 0.0)


fit_metrics = metrics(pred, actual_tons, fit_mask)
pred_tail = tail_slope(years, pred, PRED_START, PRED_END)
pred_drop = total_drop_pct(years, pred, PRED_START, PRED_END)


stock_hat = np.zeros_like(stock, dtype=float)
stock_hat[0] = stock[0]
for t in range(1, n):
    stock_hat[t] = stock_hat[t - 1] + inflow[t] - outflow[t]
stock_identity_error = stock - stock_hat

print("\n================ 土耳其 Stock-driven 中国逻辑版 ================")
print(f"数据路径：{data_path}")
print(f"历史拟合段 {FIT_START}-{FIT_END}: N={fit_metrics['N']} | "
      f"MAE={fmt(fit_metrics['MAE'],0)} | MAPE={fmt(fit_metrics['MAPE'],2)}% | "
      f"RMSE={fmt(fit_metrics['RMSE'],0)} | R2={fmt(fit_metrics['R2'],4)} | DW={fmt(fit_metrics['DW'],3)}")
print(f"预测段 {PRED_START}-{PRED_END}: TailSlope={fmt(pred_tail['slope_pct_per_year'],3)}%/y | "
      f"TailR2={fmt(pred_tail['R2'],4)} | Drop={fmt(pred_drop,2)}%")
print(f"scale_global={scale_global:.4f} | anchor_year={anchor_year} | anchor_mult={anchor_mult:.4f} | floor_mult={floor_mult:.4f}")
print(f"forecast_inflow_cap={USE_FORECAST_INFLOW_CAP} | up={FORECAST_INFLOW_CAP_UP:.2f} | down={FORECAST_INFLOW_CAP_DOWN:.2f}")
print(f"hist_dw_correction={USE_HIST_DW_CORRECTION} | rho={HIST_DW_RHO_CORRECTION:.3f}")
print("===============================================================\n")


result_df = pd.DataFrame({
    "年份": years,
    "城镇人口(人)": population,
    "人均住宅面积(㎡/人)": pc_area,
    "住宅存量_target(㎡)": stock,
    "住宅存量_hat(㎡)": stock_hat,
    "新增面积_inflow(㎡/年)": inflow,
    "用于水泥计算_inflow_capped(㎡/年)": inflow_for_cement,
    "报废面积_outflow(㎡/年)": outflow,
    "基线B_StockDriven(吨)": B,
    "历史乘子延续_m(t)": mult_full,
    "全局scale": scale_global,
    "预测_DW校正前(吨)": pred_before_dw_correction,
    "预测_最终(吨)": pred,
    "实际产量(吨)": actual_tons,
    "误差_吨": pred - actual_tons,
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
    {"参数": "USE_FORECAST_INFLOW_CAP", "取值": USE_FORECAST_INFLOW_CAP},
    {"参数": "FORECAST_INFLOW_CAP_FROM", "取值": FORECAST_INFLOW_CAP_FROM},
    {"参数": "FORECAST_INFLOW_CAP_UP", "取值": FORECAST_INFLOW_CAP_UP},
    {"参数": "FORECAST_INFLOW_CAP_DOWN", "取值": FORECAST_INFLOW_CAP_DOWN},
    {"参数": "USE_HIST_DW_CORRECTION", "取值": USE_HIST_DW_CORRECTION},
    {"参数": "HIST_DW_RHO_CORRECTION", "取值": HIST_DW_RHO_CORRECTION},
    {"参数": "scale_global", "取值": scale_global},
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


if HAS_MATPLOTLIB:
    plt.figure(figsize=(13.5, 7))
    mask_actual_plot = np.isfinite(actual_tons) & (actual_tons > 0)
    if np.any(mask_actual_plot):
        plt.plot(years[mask_actual_plot], actual_tons[mask_actual_plot] / 1e6, "ko-", label="Actual")

    hist_plot = years <= last_hist_year
    pred_plot = years > last_hist_year
    plt.plot(years[hist_plot], pred[hist_plot] / 1e6, "b-", linewidth=2.0, label="Model fit")
    plt.plot(years[pred_plot], pred[pred_plot] / 1e6, "b--", linewidth=2.0, label="Forecast: B x historical multiplier")
    plt.axvline(last_hist_year, linestyle="--", color="gray", label=f"Forecast start {last_hist_year + 1}")

    plt.xlabel("Year")
    plt.ylabel("Cement production / demand (million tons)")
    plt.title("Turkey cement demand forecast: stock-driven baseline x historical multiplier continuation")
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
    plt.close()
else:
    print("⚠️ 当前环境未安装 matplotlib，已跳过绘图；Excel 结果仍正常导出。")
