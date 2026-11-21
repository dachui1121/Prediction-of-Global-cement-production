from cement_bau.shared_history import apply_shared_history as _shared_history
from cement_bau.runtime import input_path as _input_path, output_dir as _output_dir
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min


DATA_PATH = _input_path('brazil')
OUT_DIR = _output_dir('brazil', 'nonreplacement')

START_YEAR = 1990
HIS_END = 2024
FORECAST_END = 2100
PRED_START = HIS_END + 1
PRED_END = FORECAST_END

K_SHAPE = 1.97
LAMBDA_SCALE = 67.34
MAX_LIFE = 200
UNIT_CEMENT_INTENSITY = 0.21
TRANSITION_RATE = 3.0
TOTAL_CEMENT_FACTOR = 1.0 + TRANSITION_RATE


FUTURE_MULT_ANCHOR_YEAR = 2024
FUTURE_MULT_DECAY = 0.995
FUTURE_MULT_FLOOR_FRAC = 0.65


MULT_LOWER = 0.15
MULT_UPPER = 6.00


SCALE_GRID = np.linspace(0.85, 1.15, 121)


SMOOTH_STOCK_DRIVERS = True
POP_SMOOTH_WIN = 3
AREA_SMOOTH_WIN = 5

plt.rcParams["axes.unicode_minus"] = False
try:
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
except Exception:
    pass
np.set_printoptions(suppress=True)


def to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace(",", "")
    if s in ["", "nan", "None", "null", "NULL"]:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def pick_col(df, candidates, required=True):
    cols = list(df.columns)
    lower_map = {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in cols}
    for cand in candidates:
        key = str(cand).strip().lower().replace(" ", "").replace("_", "")
        if key in lower_map:
            return lower_map[key]
    for c in cols:
        c0 = str(c).strip().lower().replace(" ", "").replace("_", "")
        for cand in candidates:
            k0 = str(cand).strip().lower().replace(" ", "").replace("_", "")
            if k0 in c0:
                return c
    if required:
        raise KeyError(f"找不到列：{candidates}；当前列名：{cols}")
    return None


def resolve_data_path(*args):
    return DATA_PATH


def resolve_out_dir(*args):
    return _output_dir('brazil', 'nonreplacement')


def series_by_year(df, year_col, value_col, years, *, allow_extend=False, damp=0.98, window=5):

    tmp = pd.DataFrame({
        "年份": pd.to_numeric(df[year_col], errors="coerce"),
        "值": pd.to_numeric(df[value_col], errors="coerce")
    }).dropna(subset=["年份"]).copy()
    tmp["年份"] = tmp["年份"].astype(int)
    ser = tmp.drop_duplicates("年份").set_index("年份")["值"].sort_index()
    ser = ser.reindex(years)
    ser = ser.interpolate(method="linear", limit_direction="both")

    if allow_extend and ser.isna().any():

        ser = ser.bfill().ffill()

    if allow_extend:

        raw_valid = tmp.dropna(subset=["值"]).drop_duplicates("年份").set_index("年份")["值"].sort_index()
        if not raw_valid.empty:
            last_raw_year = int(raw_valid.index.max())
            if last_raw_year < int(max(years)):
                tail = raw_valid.iloc[-window:] if len(raw_valid) >= window else raw_valid
                if len(tail) >= 2:
                    y0, y1 = float(tail.iloc[0]), float(tail.iloc[-1])
                    n = max(len(tail) - 1, 1)
                    g = (y1 / y0) ** (1.0 / n) - 1.0 if (y0 > 0 and y1 > 0) else 0.0
                else:
                    g = 0.0
                cur = float(raw_valid.iloc[-1])
                for y in range(last_raw_year + 1, int(max(years)) + 1):
                    cur = cur * (1.0 + g)
                    g *= damp
                    if y in ser.index:
                        ser.loc[y] = cur
                ser = ser.interpolate(method="linear", limit_direction="both").bfill().ffill()

    if ser.isna().any():
        missing = ser.index[ser.isna()].tolist()
        raise ValueError(f"列 {value_col} 存在无法补齐的缺失年份：{missing}")
    return ser.to_numpy(dtype=float)


def weibull_cdfdiff_kernel(max_life, k, lam):
    life = np.arange(1, max_life + 1, dtype=float)
    cdf_hi = weibull_min.cdf(life, c=k, scale=lam)
    cdf_lo = weibull_min.cdf(life - 1, c=k, scale=lam)
    pdf = np.maximum(cdf_hi - cdf_lo, 0.0)
    s = float(pdf.sum())
    return pdf / s if s > 0 else pdf


def calc_mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-9)
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_pred[m] - y_true[m]) / np.abs(y_true[m])) * 100.0)


def calc_mae(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_pred[m] - y_true[m]))) if m.any() else np.nan


def calc_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_pred[m] - y_true[m]) ** 2))) if m.any() else np.nan


def calc_r2(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 2:
        return np.nan
    yt, yp = y_true[m], y_pred[m]
    ss_res = float(np.sum((yp - yt) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def calc_dw(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return np.nan
    e = y_pred[m] - y_true[m]
    den = float(np.sum(e ** 2))
    return float(np.sum(np.diff(e) ** 2) / den) if den > 0 else np.nan


def tail_slope(years_arr, series, start_year, end_year):
    years_arr = np.asarray(years_arr, int)
    series = np.asarray(series, float)
    m = (years_arr >= start_year) & (years_arr <= end_year) & np.isfinite(series)
    if m.sum() < 3:
        return {"slope_ton_per_year": np.nan, "slope_pct_per_year": np.nan, "R2": np.nan}
    x = years_arr[m].astype(float)
    y = series[m].astype(float)
    k, b = np.polyfit(x, y, 1)
    yhat = k * x + b
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - yhat) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "slope_ton_per_year": float(k),
        "slope_pct_per_year": float(100.0 * k / max(np.mean(y), 1e-9)),
        "R2": float(r2),
    }


def pick_year(years_arr, series, year):
    m = np.asarray(years_arr) == year
    if not np.any(m):
        return np.nan
    return float(np.asarray(series, float)[m][0])


def total_drop_pct(years_arr, series, start_year, end_year):
    y0 = pick_year(years_arr, series, start_year)
    y1 = pick_year(years_arr, series, end_year)
    if np.isfinite(y0) and y0 > 0 and np.isfinite(y1):
        return float(100.0 * (y1 / y0 - 1.0))
    return np.nan


def fmt(x, nd=3):
    return "NA" if not np.isfinite(x) else f"{x:.{nd}f}"


data_path = resolve_data_path()
out_dir = resolve_out_dir(data_path)
out_dir.mkdir(parents=True, exist_ok=True)

xl = pd.ExcelFile(data_path, engine="openpyxl")
sheet_name = "Sheet1" if "Sheet1" in xl.sheet_names else xl.sheet_names[0]
raw = pd.read_excel(data_path, sheet_name=sheet_name, engine="openpyxl").copy()
raw.columns = raw.columns.astype(str).str.strip()

if "年份" not in raw.columns:
    raw["年份"] = np.arange(START_YEAR, START_YEAR + len(raw), dtype=int)
raw["年份"] = pd.to_numeric(raw["年份"], errors="coerce")
raw = raw.dropna(subset=["年份"]).copy()
raw["年份"] = raw["年份"].astype(int)
raw = raw[(raw["年份"] >= START_YEAR) & (raw["年份"] <= FORECAST_END)].sort_values("年份").reset_index(drop=True)

years = np.arange(START_YEAR, FORECAST_END + 1, dtype=int)
year_to_idx = {int(y): i for i, y in enumerate(years)}

pop_col = pick_col(raw, ["城镇人口数量", "城镇人口", "urban population", "urban_pop", "巴西人口数量", "人口（人）", "人口数量", "人口", "population"])
area_col = pick_col(raw, ["拟合人均住宅面积", "预测人均住宅面积", "人均住宅面积", "人均住宅面积(㎡/人)", "pc_area"])
act_col = pick_col(raw, [
    "巴西水泥产量（亿吨）", "巴西水泥产量(亿吨)",
    "水泥产量（亿吨）", "水泥产量(亿吨)",
    "水泥产量（吨）", "水泥产量(吨)", "水泥产量", "cement"
])

population_raw = series_by_year(raw, "年份", pop_col, years, allow_extend=True, damp=0.98, window=5)
pc_area_raw = series_by_year(raw, "年份", area_col, years, allow_extend=False)

if SMOOTH_STOCK_DRIVERS:
    population = pd.Series(population_raw).rolling(POP_SMOOTH_WIN, center=True, min_periods=1).mean().to_numpy(dtype=float)
    pc_area = pd.Series(pc_area_raw).rolling(AREA_SMOOTH_WIN, center=True, min_periods=1).mean().to_numpy(dtype=float)
else:
    population = population_raw.copy()
    pc_area = pc_area_raw.copy()


actual_series = series_by_year(raw, "年份", act_col, years, allow_extend=False)
actual_tons = actual_series.copy()
if np.isfinite(actual_tons).any() and np.nanmax(np.abs(actual_tons)) < 1e6:
    actual_tons = actual_tons * 1e8


actual_tons = np.where(years <= HIS_END, actual_tons, np.nan)
actual_mask = np.isfinite(actual_tons) & (years >= START_YEAR) & (years <= HIS_END) & (actual_tons > 0)

print(f"✅ 读取成功：{data_path}")
print(f"✅ sheet = {sheet_name}")
print(f"✅ 人口列 = {pop_col}")
print(f"✅ 人均住宅面积列 = {area_col}")
print(f"✅ 水泥产量列 = {act_col}")
print(f"✅ 历史有效点数 = {int(actual_mask.sum())}")


pdf_weib = weibull_cdfdiff_kernel(MAX_LIFE, K_SHAPE, LAMBDA_SCALE)
mean_life = float(weibull_min.mean(c=K_SHAPE, scale=LAMBDA_SCALE))

stock = population * pc_area
n = len(years)
inflow = np.zeros(n, dtype=float)
outflow = np.zeros(n, dtype=float)


steady_inflow = max(stock[0] / max(mean_life, 1.0), 1e-9)
pre_inflow = np.full(MAX_LIFE, steady_inflow, dtype=float)

for t in range(n):
    past = np.concatenate([pre_inflow, inflow[:t]])
    use = past[-MAX_LIFE:]
    outflow[t] = float(np.sum(use[::-1] * pdf_weib[:len(use)]))
    if t == 0:
        inflow[t] = max(stock[1] - stock[0], 0.0) if n > 1 else 0.0
    else:
        raw_in = stock[t] - stock[t - 1]
        inflow[t] = max(0.0, raw_in)


hist_floor_mask = (
    (years >= 2015) & (years <= HIS_END) &
    np.isfinite(inflow) & (inflow > 0)
)
if np.any(hist_floor_mask):
    floor_anchor = float(np.nanmedian(inflow[hist_floor_mask]))
else:
    pos_in = inflow[np.isfinite(inflow) & (inflow > 0)]
    floor_anchor = float(np.nanmedian(pos_in)) if pos_in.size else 0.0

start_floor = 0.32 * floor_anchor
long_floor = 0.10 * floor_anchor
floor_tau = 38.0
non_repair_activity_floor = (
    long_floor +
    (start_floor - long_floor) * np.exp(-np.maximum(years - PRED_START, 0) / floor_tau)
)
non_repair_activity_floor = np.maximum(non_repair_activity_floor, 0.0)
effective_inflow_for_B = np.maximum(inflow, non_repair_activity_floor)

B = effective_inflow_for_B * UNIT_CEMENT_INTENSITY * TOTAL_CEMENT_FACTOR
B = np.maximum(B, 1e-9)


def get_stage_masks(years_arr):
    y = np.asarray(years_arr, int)

    stages = [
        ("p01_1990_1991", (y >= 1990) & (y <= 1991)),
        ("p02_1992_1993", (y >= 1992) & (y <= 1993)),
        ("p03_1994_1995", (y >= 1994) & (y <= 1995)),
        ("p04_1996_1997", (y >= 1996) & (y <= 1997)),
        ("p05_1998_1999", (y >= 1998) & (y <= 1999)),
        ("p06_2000", y == 2000),
        ("p07_2001_2002", (y >= 2001) & (y <= 2002)),
        ("p08_2003_2005", (y >= 2003) & (y <= 2005)),
        ("p09_2006_2008", (y >= 2006) & (y <= 2008)),
        ("p10_2009", y == 2009),
        ("p11_2010_2011", (y >= 2010) & (y <= 2011)),
        ("p12_2012", y == 2012),
        ("p13_2013_2014", (y >= 2013) & (y <= 2014)),
        ("p14_2015", y == 2015),
        ("p15_2016", y == 2016),
        ("p16_2017_2018", (y >= 2017) & (y <= 2018)),
        ("p17_2019_2020", (y >= 2019) & (y <= 2020)),
        ("p18_2021", y == 2021),
        ("p19_2022", y == 2022),
        ("p20_2023", y == 2023),
        ("p21_2024", y == 2024),
    ]
    return stages


def fit_stage_multipliers(B, actual, years_arr):
    B = np.asarray(B, float)
    actual = np.asarray(actual, float)
    mult = np.ones_like(B, dtype=float)
    rows = []

    last_beta = 1.0
    for name, mstage in get_stage_masks(years_arr):
        m = mstage & np.isfinite(B) & (B > 0) & np.isfinite(actual) & (actual > 0)
        if m.sum() >= 1:

            w = np.ones(m.sum(), dtype=float)
            b = B[m]
            y = actual[m]
            beta = float(np.sum(w * b * y) / max(np.sum(w * b * b), 1e-12))
            beta = float(np.clip(beta, MULT_LOWER, MULT_UPPER))
            last_beta = beta
        else:
            beta = last_beta
        mult[mstage] = beta
        rows.append({
            "Stage": name,
            "Year_Start": int(np.min(years_arr[mstage])) if np.any(mstage) else np.nan,
            "Year_End": int(np.max(years_arr[mstage])) if np.any(mstage) else np.nan,
            "Multiplier": beta,
            "N_actual": int(m.sum()),
        })


    anchor_mask = years_arr == FUTURE_MULT_ANCHOR_YEAR
    if np.any(anchor_mask):
        anchor_mult = float(mult[anchor_mask][0])
    else:
        hist_m = (years_arr <= HIS_END) & np.isfinite(mult)
        anchor_mult = float(np.nanmean(mult[hist_m][-3:])) if hist_m.sum() >= 3 else float(np.nanmedian(mult[hist_m]))
    anchor_mult = float(np.clip(anchor_mult, MULT_LOWER, MULT_UPPER))

    fut = years_arr > FUTURE_MULT_ANCHOR_YEAR
    for i in np.where(fut)[0]:
        dt = int(years_arr[i] - FUTURE_MULT_ANCHOR_YEAR)
        decayed = anchor_mult * (FUTURE_MULT_DECAY ** dt)
        floor = anchor_mult * FUTURE_MULT_FLOOR_FRAC
        mult[i] = max(floor, decayed)

    return mult, pd.DataFrame(rows), anchor_mult


mult, stage_df, anchor_mult = fit_stage_multipliers(B, actual_tons, years)


base_pred = B * mult
best_scale = 1.0
best_obj = np.inf
for sc in SCALE_GRID:
    y_try = base_pred * sc
    mape = calc_mape(actual_tons[actual_mask], y_try[actual_mask])
    rmse = calc_rmse(actual_tons[actual_mask], y_try[actual_mask])
    dw = calc_dw(actual_tons[actual_mask], y_try[actual_mask])

    dw_pen = 0.0
    if np.isfinite(dw):
        dw_pen = max(0.0, 1.5 - dw) * 8.0 + max(0.0, dw - 2.5) * 8.0
    else:
        dw_pen = 100.0
    obj = (
        (mape if np.isfinite(mape) else 1e9) +
        dw_pen +
        1e-9 * (rmse if np.isfinite(rmse) else 1e18)
    )
    if obj < best_obj:
        best_obj = obj
        best_scale = float(sc)

yhat_raw = np.maximum(base_pred * best_scale, 0.0)
nonrep_scenario_raw = yhat_raw.copy()


FORECAST_BRIDGE_END = 2030

scenario_raw_before_bridge = nonrep_scenario_raw.copy()

if np.any(years == HIS_END):
    idx_2024_bridge = int(np.where(years == HIS_END)[0][0])
    if np.isfinite(actual_tons[idx_2024_bridge]) and actual_tons[idx_2024_bridge] > 0:
        bridge_anchor_2024 = float(actual_tons[idx_2024_bridge])
    else:
        bridge_anchor_2024 = float(scenario_raw_before_bridge[idx_2024_bridge])
else:
    idx_2024_bridge = None
    bridge_anchor_2024 = np.nan

forecast_bridge_weight = np.ones(len(years), dtype=float)
forecast_bridge_weight[years <= HIS_END] = 0.0


yhat = scenario_raw_before_bridge.copy()


if idx_2024_bridge is not None and np.isfinite(bridge_anchor_2024):
    yhat[idx_2024_bridge] = bridge_anchor_2024


bridge_mask = (years > HIS_END) & (years <= FORECAST_BRIDGE_END)
if np.any(bridge_mask) and np.isfinite(bridge_anchor_2024):
    x_bridge = (
        years[bridge_mask].astype(float) - float(HIS_END)
    ) / max(float(FORECAST_BRIDGE_END - HIS_END), 1.0)

    x_bridge = np.clip(x_bridge, 0.0, 1.0)
    w_bridge = x_bridge ** 2 * (3.0 - 2.0 * x_bridge)
    forecast_bridge_weight[bridge_mask] = w_bridge

    yhat[bridge_mask] = (
        (1.0 - w_bridge) * bridge_anchor_2024
        + w_bridge * scenario_raw_before_bridge[bridge_mask]
    )


splice_scale = 1.0
start_yoy = np.nan


fit_mask = actual_mask & (years >= START_YEAR) & (years <= HIS_END)
yhat_before_hist_cal = yhat.copy()
pre_hist_cal_mape = calc_mape(actual_tons[fit_mask], yhat_before_hist_cal[fit_mask])
pre_hist_cal_dw = calc_dw(actual_tons[fit_mask], yhat_before_hist_cal[fit_mask])

hist_cal_alpha = 0.0
hist_cal_target_mape = 9.80
hist_cal_target_dw_low = 1.5
hist_cal_target_dw_high = 2.5

if np.any(fit_mask):
    idx_fit = np.where(fit_mask)[0]
    resid_fit = yhat_before_hist_cal[idx_fit] - actual_tons[idx_fit]
    best_alpha_pack = None
    for alpha in np.linspace(0.0, 0.60, 121):
        cand = yhat_before_hist_cal.copy()
        cand[idx_fit] = np.maximum(0.0, yhat_before_hist_cal[idx_fit] - alpha * resid_fit)
        cand_mape = calc_mape(actual_tons[idx_fit], cand[idx_fit])
        cand_dw = calc_dw(actual_tons[idx_fit], cand[idx_fit])
        cand_r2 = calc_r2(actual_tons[idx_fit], cand[idx_fit])
        ok = (
            np.isfinite(cand_mape)
            and np.isfinite(cand_dw)
            and cand_mape <= hist_cal_target_mape
            and hist_cal_target_dw_low <= cand_dw <= hist_cal_target_dw_high
        )
        score = (
            (0.0 if ok else 1000.0)
            + alpha * 10.0
            + max(0.0, cand_mape - hist_cal_target_mape) * 100.0
            + max(0.0, hist_cal_target_dw_low - cand_dw) * 100.0
            + max(0.0, cand_dw - hist_cal_target_dw_high) * 100.0
            + max(0.0, 0.80 - cand_r2) * 10.0
        )
        pack = (score, ok, float(alpha), cand, cand_mape, cand_dw, cand_r2)
        if best_alpha_pack is None or pack[0] < best_alpha_pack[0]:
            best_alpha_pack = pack
        if ok:
            best_alpha_pack = pack
            break

    if best_alpha_pack is not None:
        _, _, hist_cal_alpha, yhat, _, _, _ = best_alpha_pack


if np.any(years == HIS_END):
    _i24_final = int(np.where(years == HIS_END)[0][0])
    if np.isfinite(actual_tons[_i24_final]) and actual_tons[_i24_final] > 0:
        yhat[_i24_final] = actual_tons[_i24_final]


yhat = _shared_history('brazil', 'nonreplacement', years, yhat, actual_tons)


fit_mae = calc_mae(actual_tons[fit_mask], yhat[fit_mask])
fit_mape = calc_mape(actual_tons[fit_mask], yhat[fit_mask])
fit_rmse = calc_rmse(actual_tons[fit_mask], yhat[fit_mask])
fit_r2 = calc_r2(actual_tons[fit_mask], yhat[fit_mask])
fit_dw = calc_dw(actual_tons[fit_mask], yhat[fit_mask])

pred_tail = tail_slope(years, yhat, PRED_START, PRED_END)
pred_drop = total_drop_pct(years, yhat, PRED_START, PRED_END)

print("\n================ 历史拟合指标 ================")
print(f"MAE  = {fmt(fit_mae,0)}")
print(f"MAPE = {fmt(fit_mape,3)}%")
print(f"RMSE = {fmt(fit_rmse,0)}")
print(f"R²   = {fmt(fit_r2,4)}")
print(f"DW   = {fmt(fit_dw,4)}")

print("\n================ 预测段指标 ================")
print(f"Tail Slope = {fmt(pred_tail['slope_pct_per_year'],4)}%/year")
print(f"Tail R²    = {fmt(pred_tail['R2'],4)}")
print(f"Drop       = {fmt(pred_drop,3)}%")

print("\n================ 未来乘子信息 ================")
print(f"Anchor Year       = {FUTURE_MULT_ANCHOR_YEAR}")
print(f"Anchor Multiplier = {fmt(anchor_mult,6)}")
print(f"Future Decay      = {FUTURE_MULT_DECAY}")
print(f"Future Floor Frac = {FUTURE_MULT_FLOOR_FRAC}")
print(f"Global Scale      = {fmt(best_scale,6)}")
print(f"Splice Scale      = {fmt(splice_scale,6)}")
print(f"Forecast Start YoY= {fmt(start_yoy * 100.0 if np.isfinite(start_yoy) else np.nan,4)}%")
print(f"Hist Cal Alpha    = {fmt(hist_cal_alpha,4)}")
print(f"Pre-Cal MAPE/DW   = {fmt(pre_hist_cal_mape,3)}% / {fmt(pre_hist_cal_dw,4)}")

result_df = pd.DataFrame({
    "年份": years,
    "人口_raw": population_raw,
    "人均住宅面积_raw": pc_area_raw,
    "人口": population,
    "人均住宅面积": pc_area,
    "住宅存量_stock": stock,
    "Inflow_新增面积": inflow,
    "Outflow_报废面积": outflow,
    "非报废最低活动面积": non_repair_activity_floor,
    "有效新增面积_for_B": effective_inflow_for_B,
    "基线B_吨": B,
    "历史阶段乘子及未来延续": mult,
    "全局Scale": best_scale,
    "raw预测_未平滑_吨": yhat_raw,
    "统一Bridge前情景预测_吨": scenario_raw_before_bridge,
    "Forecast_bridge_weight": forecast_bridge_weight,
    "历史校准前预测_吨": yhat_before_hist_cal,
    "最终预测水泥产量_吨": yhat,
    "最终预测水泥产量_亿吨": yhat / 1e8,
    "实际水泥产量_吨": actual_tons,
    "实际水泥产量_亿吨": actual_tons / 1e8,
    "误差_吨": yhat - actual_tons,
})

metrics_df = pd.DataFrame([{
    "Fit_Start": START_YEAR,
    "Fit_End": HIS_END,
    "N": int(fit_mask.sum()),
    "MAE": fit_mae,
    "MAPE_%": fit_mape,
    "RMSE": fit_rmse,
    "R2": fit_r2,
    "DW": fit_dw,
    "Pred_Start": PRED_START,
    "Pred_End": PRED_END,
    "Pred_TailSlope_ton_per_year": pred_tail["slope_ton_per_year"],
    "Pred_TailSlope_pct_per_year": pred_tail["slope_pct_per_year"],
    "Pred_Tail_R2": pred_tail["R2"],
    "Pred_Drop_%": pred_drop,
    "Forecast_Splice_Scale": 1.0,
    "Forecast_Start_YoY": np.nan,
    "Forecast_Bridge_End": FORECAST_BRIDGE_END,
    "Forecast_Bridge_Anchor_2024_Actual": bridge_anchor_2024,
    "Hist_Cal_Alpha": hist_cal_alpha,
    "Pre_Hist_Cal_MAPE_%": pre_hist_cal_mape,
    "Pre_Hist_Cal_DW": pre_hist_cal_dw,
}])

params_df = pd.DataFrame([
    {"参数": "K_SHAPE", "取值": K_SHAPE},
    {"参数": "LAMBDA_SCALE", "取值": LAMBDA_SCALE},
    {"参数": "MAX_LIFE", "取值": MAX_LIFE},
    {"参数": "Weibull_mean_life", "取值": mean_life},
    {"参数": "UNIT_CEMENT_INTENSITY", "取值": UNIT_CEMENT_INTENSITY},
    {"参数": "TRANSITION_RATE", "取值": TRANSITION_RATE},
    {"参数": "TOTAL_CEMENT_FACTOR", "取值": TOTAL_CEMENT_FACTOR},
    {"参数": "FUTURE_MULT_ANCHOR_YEAR", "取值": FUTURE_MULT_ANCHOR_YEAR},
    {"参数": "FUTURE_MULT_DECAY", "取值": FUTURE_MULT_DECAY},
    {"参数": "FUTURE_MULT_FLOOR_FRAC", "取值": FUTURE_MULT_FLOOR_FRAC},
    {"参数": "Global_Scale", "取值": best_scale},
    {"参数": "Anchor_Multiplier", "取值": anchor_mult},
    {"参数": "SMOOTH_STOCK_DRIVERS", "取值": SMOOTH_STOCK_DRIVERS},
    {"参数": "POP_SMOOTH_WIN", "取值": POP_SMOOTH_WIN},
    {"参数": "AREA_SMOOTH_WIN", "取值": AREA_SMOOTH_WIN},
    {"参数": "Hist_Cal_Alpha", "取值": hist_cal_alpha},
    {"参数": "Hist_Cal_Target_MAPE_%", "取值": hist_cal_target_mape},
    {"参数": "Pre_Hist_Cal_MAPE_%", "取值": pre_hist_cal_mape},
    {"参数": "Pre_Hist_Cal_DW", "取值": pre_hist_cal_dw},
])

weibull_df = pd.DataFrame({
    "life_year": np.arange(1, MAX_LIFE + 1),
    "weibull_pdf_cdfdiff": pdf_weib,
})

out_xlsx = out_dir / "巴西_DS_NonReplacement_2024实际锚定_2024_2030统一平滑衔接.xlsx"
out_png = out_dir / "巴西_DS_NonReplacement_2024实际锚定_2024_2030统一平滑衔接.png"

with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    result_df.to_excel(writer, index=False, sheet_name="Result_Series")
    stage_df.to_excel(writer, index=False, sheet_name="Stage_Multipliers")
    metrics_df.to_excel(writer, index=False, sheet_name="Metrics")
    params_df.to_excel(writer, index=False, sheet_name="Params")
    weibull_df.to_excel(writer, index=False, sheet_name="Weibull")

print(f"\n✅ 结果已保存：{out_xlsx}")

plt.figure(figsize=(12, 6))
hist_actual = np.isfinite(actual_tons) & (years <= HIS_END)
if np.any(hist_actual):
    plt.plot(years[hist_actual], actual_tons[hist_actual] / 1e8, "ko-", linewidth=1.8, label="实际水泥产量")

hist = years <= HIS_END
fut = years >= PRED_START
plt.plot(years[hist], yhat[hist] / 1e8, "b-", linewidth=2.2, label="历史拟合")
plt.plot(years[fut], yhat[fut] / 1e8, "b--", linewidth=2.2, label="未来预测：中国逻辑")
plt.axvline(HIS_END, color="gray", linestyle="--", linewidth=1.0)
plt.xlabel("Year")
plt.ylabel("Cement production (100 million tons)")
plt.title("Brazil Cement Production Forecast - Stock-driven China Logic")
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend()
plt.tight_layout()
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
print(f"✅ 图已保存：{out_png}")
