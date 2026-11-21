from cement_bau.runtime import input_path as _input_path, output_dir as _output_dir
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min


DATA_PATH = str(_input_path('united_states'))
SHEET_NAME = "Sheet1"

START_YEAR = 1989
HIS_END = 2024
FORECAST_END = 2100
FIT_START, FIT_END = 1990, 2024
PRED_START, PRED_END = HIS_END + 1, FORECAST_END


k_shape = 2.8
lambda_scale = 73.5
max_life = 120


res_intensity = 0.05
TRANSITION_RATE = 3.0
infra_factor_base = 1.0 + TRANSITION_RATE


FUTURE_MULT_ANCHOR_YEAR = 2024
FUTURE_MULT_DECAY = 0.995
FUTURE_MULT_FLOOR_FRAC = 0.65


BETA_LO = 0.15
BETA_HI = 8.00


SCALE_LO = 0.82
SCALE_HI = 1.18


base_dir = str(_output_dir('united_states', 'warmup_1989'))
OUT_XLSX = os.path.join(base_dir, "美国水泥需求预测_StockDriven_1989warmup_R2优化.xlsx")
OUT_PNG = os.path.join(base_dir, "美国水泥需求预测_StockDriven_1989warmup_R2优化.png")


plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def resolve_data_path(*args):
    return DATA_PATH


def pick_col(df: pd.DataFrame, candidates):

    cols = df.columns.astype(str).str.strip().tolist()
    for c in candidates:
        if c in cols:
            return c
    for col in cols:
        for c in candidates:
            if c in col:
                return col
    raise KeyError(f"找不到列：候选={candidates}，实际列={cols}")


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


data_path = resolve_data_path(DATA_PATH)
df = pd.read_excel(data_path, sheet_name=SHEET_NAME)
df.columns = df.columns.astype(str).str.strip()

c_year = pick_col(df, ["年份", "Year", "year"])


try:
    c_urban_pop = pick_col(df, ["城镇人口数", "城镇人口数量", "城市人口", "Urban population"])
except Exception:
    c_urban_pop = None

if c_urban_pop is None:
    c_pop_total = pick_col(df, ["人口（人）", "人口", "Population"])
    c_urban_rate = pick_col(df, ["城镇化率", "Urbanization"])
else:
    c_pop_total = None
    c_urban_rate = None


try:
    c_pc_area = pick_col(df, ["预测人均住宅面积", "拟合人均住宅面积", "城镇人均居住面积", "人均住宅面积"])
except Exception:
    raise KeyError("找不到人均住宅面积列：请检查是否存在 '预测人均住宅面积' / '拟合人均住宅面积' / '城镇人均居住面积'。")

try:
    c_actual = pick_col(df, ["水泥产量", "美国水泥产量", "美国水泥产量（亿吨）", "ActualCement"])
except Exception:
    c_actual = None


df[c_year] = to_num(df[c_year])
df = df[np.isfinite(df[c_year])].copy()
df[c_year] = df[c_year].round().astype(int)
df = df[(df[c_year] >= START_YEAR) & (df[c_year] <= FORECAST_END)].copy()
df = df.sort_values(c_year).reset_index(drop=True)

years = df[c_year].astype(int).values
year_set = set(years.tolist())
n = len(years)

if c_urban_pop is not None:
    population = to_num(df[c_urban_pop]).interpolate(limit_direction="both").bfill().ffill().values.astype(float)
else:
    total_pop = to_num(df[c_pop_total]).interpolate(limit_direction="both").bfill().ffill().values.astype(float)
    urb_rate = to_num(df[c_urban_rate]).interpolate(limit_direction="both").bfill().ffill().values.astype(float)
    if np.nanmax(urb_rate) > 1.5:
        urb_rate = urb_rate / 100.0
    population = total_pop * urb_rate

pc_area = to_num(df[c_pc_area]).interpolate(limit_direction="both").bfill().ffill().values.astype(float)

if c_actual is not None:
    actual_raw = to_num(df[c_actual]).values.astype(float)
    if np.isfinite(actual_raw).any() and np.nanmax(np.abs(actual_raw)) < 1e6:
        actual_tons = actual_raw * 1e8
    else:
        actual_tons = actual_raw
else:
    actual_tons = np.full(n, np.nan, dtype=float)

actual_mask = np.isfinite(actual_tons) & (actual_tons >= 0)
mask_fit = actual_mask & (years >= FIT_START) & (years <= FIT_END)

print(f"✅ 读取成功：{data_path}")
print("列名：", {"年份": c_year, "城镇人口": c_urban_pop, "总人口": c_pop_total, "城镇化率": c_urban_rate, "人均面积": c_pc_area, "产量": c_actual})
print(f"年份数={n} | 真实产量点={int(actual_mask.sum())}")


stock = population * pc_area

life = np.arange(1, max_life + 1)
pdf_weib = weibull_min.pdf(life, c=k_shape, scale=lambda_scale)
pdf_weib = np.nan_to_num(pdf_weib)
pdf_weib = pdf_weib / pdf_weib.sum()

inflow = np.zeros(n, dtype=float)
outflow = np.zeros(n, dtype=float)


exp_life = weibull_min.mean(c=k_shape, scale=lambda_scale)
inflow[0] = max(stock[0] / max(exp_life, 1.0), 1e-9)
outflow[0] = inflow[0]


for t in range(1, n):
    k = min(t, max_life)
    outflow[t] = float(np.sum(inflow[t-k:t][::-1] * pdf_weib[:k]))
    raw_in = (stock[t] - stock[t - 1]) + outflow[t]
    inflow[t] = max(0.0, raw_in)


B = inflow * res_intensity * infra_factor_base


seg_defs = [
    ("p1a_1990_1991", (years >= 1990) & (years <= 1991)),
    ("p1b_1992_1993", (years >= 1992) & (years <= 1993)),
    ("p1c_1994_2000", (years >= 1994) & (years <= 2000)),
    ("p2a_2001_2002", (years >= 2001) & (years <= 2002)),
    ("p2b_2003_2007", (years >= 2003) & (years <= 2007)),
    ("p3a_2008", years == 2008),
    ("p3b_2009", years == 2009),
    ("p3c_2010_2012", (years >= 2010) & (years <= 2012)),
    ("p4_2013_2019", (years >= 2013) & (years <= 2019)),
    ("p5a_2020", years == 2020),
    ("p5b_2021", years == 2021),
    ("p6_2022", years == 2022),
    ("p7_2023", years == 2023),
    ("p8_2024", years == 2024),
]


W = np.ones(n, dtype=float)
W[(years >= 2008) & (years <= 2012)] *= 2.0
W[(years >= 2020) & (years <= 2024)] *= 2.0

beta = np.ones(len(seg_defs), dtype=float)
for j, (name, seg_mask) in enumerate(seg_defs):
    m = mask_fit & seg_mask & np.isfinite(B) & (B > 0) & np.isfinite(actual_tons) & (actual_tons > 0)
    if not np.any(m):
        beta[j] = 1.0
        continue


    w = W[m]
    bj = float(np.sum(w * B[m] * actual_tons[m]) / max(np.sum(w * B[m] * B[m]), 1e-12))
    beta[j] = 1.0 if not np.isfinite(bj) else float(np.clip(bj, BETA_LO, BETA_HI))


mult = np.ones(n, dtype=float)
for j, (_, seg_mask) in enumerate(seg_defs):
    mult[seg_mask] = beta[j]


if FUTURE_MULT_ANCHOR_YEAR in year_set:
    i_anchor = np.where(years == FUTURE_MULT_ANCHOR_YEAR)[0][0]
    anchor_mult = float(mult[i_anchor])
else:
    hist_idx = np.where((years <= HIS_END) & np.isfinite(mult))[0]
    anchor_mult = float(np.mean(mult[hist_idx[-min(5, len(hist_idx)):]])) if hist_idx.size else 1.0

fut = years > HIS_END
if np.any(fut):
    t = (years[fut] - FUTURE_MULT_ANCHOR_YEAR).astype(float)
    mult_future = anchor_mult * (FUTURE_MULT_DECAY ** t)
    mult_floor = FUTURE_MULT_FLOOR_FRAC * anchor_mult
    mult[fut] = np.maximum(mult_floor, mult_future)


pred0 = B * mult


scale = best_scale_by_mape(actual_tons, pred0, mask_fit)
pred_final = pred0 * scale


fit_m = metrics(pred_final, actual_tons, mask_fit)
all_m = metrics(pred_final, actual_tons, mask_fit)
tail_m = tail_slope(years, pred_final, PRED_START, FORECAST_END)

pred_start_val = pick_year_value(years, pred_final, PRED_START)
pred_end_val = pick_year_value(years, pred_final, FORECAST_END)
if np.isfinite(pred_start_val) and pred_start_val > 0 and np.isfinite(pred_end_val):
    drop_pred = 100.0 * (pred_end_val / pred_start_val - 1.0)
else:
    drop_pred = np.nan

yoy_pct = np.full(n, np.nan, dtype=float)
yoy_pct[1:] = (pred_final[1:] - pred_final[:-1]) / np.where(pred_final[:-1] == 0, np.nan, pred_final[:-1]) * 100.0

print("\n================ 结果汇总（美国：中国逻辑版_R2优化） ================")
print(f"MAPE_fit({FIT_START}-{FIT_END})={fit_m['MAPE']:.3f}% | R2={fit_m['R2']:.4f} | DW={fit_m['DW']:.4f}")
print(f"MAPE_all={all_m['MAPE']:.3f}% | MAE_all={all_m['MAE']:.2f} | RMSE_all={all_m['RMSE']:.2f}")
print(f"Future mult anchor={FUTURE_MULT_ANCHOR_YEAR}, anchor_mult={anchor_mult:.4f}, scale={scale:.4f}")
print(f"PredRange={PRED_START}-{FORECAST_END} | TailSlope={tail_m['slope_pct_per_year']:.4f}%/year | Drop={drop_pred:.2f}%")
print("===========================================================\n")


model_out = pd.DataFrame({
    "年份": years,
    "城镇人口(人)": np.round(population, 0),
    "人均住宅面积(㎡/人)": np.round(pc_area, 3),
    "住宅存量(㎡)": np.round(stock, 2),
    "新增面积_inflow(㎡)": np.round(inflow, 2),
    "报废面积_outflow(㎡)": np.round(outflow, 2),
    "基线B_吨": np.round(B, 2),
    "历史延续乘子mult": np.round(mult, 6),
    "全局scale": scale,
    "预测_吨": np.round(pred_final, 2),
    "真实_吨": np.round(actual_tons, 2),
    "预测YoY(%)": np.round(yoy_pct, 3),
})

checks = pd.DataFrame([{
    "FitStart": FIT_START,
    "FitEnd": FIT_END,
    "MAPE_fit_%": fit_m["MAPE"],
    "MAE_fit": fit_m["MAE"],
    "RMSE_fit": fit_m["RMSE"],
    "R2_fit": fit_m["R2"],
    "DW_fit": fit_m["DW"],
    "N_fit": fit_m["N"],
    "MAPE_all_%": all_m["MAPE"],
    "MAE_all": all_m["MAE"],
    "RMSE_all": all_m["RMSE"],
    "R2_all": all_m["R2"],
    "DW_all": all_m["DW"],
    "PredStart": PRED_START,
    "PredEnd": FORECAST_END,
    "PredStartValue_tons": pred_start_val,
    "PredEndValue_tons": pred_end_val,
    "DropPred_%": drop_pred,
    "TailSlope_pct_per_year": tail_m["slope_pct_per_year"],
    "TailR2": tail_m["R2"],
    "scale": scale,
    "FutureMultAnchorYear": FUTURE_MULT_ANCHOR_YEAR,
    "FutureMultAnchor": anchor_mult,
    "FutureMultDecay": FUTURE_MULT_DECAY,
    "FutureMultFloorFrac": FUTURE_MULT_FLOOR_FRAC,
}])

stage_multipliers = pd.DataFrame({
    "阶段": [name for name, _ in seg_defs],
    "乘子beta": beta,
})

params = pd.DataFrame({
    "key": [
        "k_shape", "lambda_scale", "max_life", "res_intensity", "TRANSITION_RATE", "infra_factor_base",
        "FUTURE_MULT_ANCHOR_YEAR", "FUTURE_MULT_DECAY", "FUTURE_MULT_FLOOR_FRAC",
        "BETA_LO", "BETA_HI", "SCALE_LO", "SCALE_HI",
        "removed_GDP_proxy", "removed_NNLS_regression", "removed_lag_y_recursive", "removed_future_EMA_smoothing", "removed_target_drop_constraint", "removed_retrofit"
    ],
    "value": [
        k_shape, lambda_scale, max_life, res_intensity, TRANSITION_RATE, infra_factor_base,
        FUTURE_MULT_ANCHOR_YEAR, FUTURE_MULT_DECAY, FUTURE_MULT_FLOOR_FRAC,
        BETA_LO, BETA_HI, SCALE_LO, SCALE_HI,
        True, True, True, True, True, True
    ]
})

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    model_out.to_excel(writer, sheet_name="Model_Output", index=False)
    checks.to_excel(writer, sheet_name="Checks", index=False)
    stage_multipliers.to_excel(writer, sheet_name="Stage_Multipliers", index=False)
    params.to_excel(writer, sheet_name="Params", index=False)

print(f"✅ 已导出：{OUT_XLSX}")


plt.figure(figsize=(13, 7))
if np.any(actual_mask):
    plt.plot(years[actual_mask], actual_tons[actual_mask] / 1e6, "ks-", label="Actual (Mt)")
plt.plot(years, pred_final / 1e6, "r.-", label="Predicted (Mt) - China-style")
plt.axvline(HIS_END, linestyle="--", color="gray", label=f"{HIS_END} historical end")
plt.xlabel("Year")
plt.ylabel("Cement production / demand (million tons)")
plt.title("U.S. Cement Demand Forecast – Stock-driven with Historical Multiplier Continuation")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=160)
plt.show()

print(f"✅ 图像已保存：{OUT_PNG}")
