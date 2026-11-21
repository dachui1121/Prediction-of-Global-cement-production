from cement_bau.shared_history import apply_shared_history as _shared_history
from cement_bau.runtime import input_path as _input_path, output_dir as _output_dir
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from scipy.optimize import least_squares


DATA_PATH = str(_input_path('united_kingdom'))
SHEET_NAME = "Sheet1"
NO_OUTFLOW_LOGIC_VERSION = "UK_NO_OUTFLOW_STOCK_FLOOR_V2_20260707"


YEAR_START = 1990
HIS_END = 2024
FORECAST_END = 2100


MAPE_THRESH_HIST = 10.0
R2_THRESH_HIST   = 0.75
DROP_2030_2100_MIN = -45.0


BASE_DIR = str(_output_dir('united_kingdom', 'nonreplacement'))
OUT_XLSX = os.path.join(BASE_DIR, "UK_DS_NonReplacement_2024ActualAnchor_2024_2030SmoothBridge_1990_2100.xlsx")


plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11


def pick_col(df_, keywords, required=True):
    cols = [str(c).strip() for c in df_.columns]
    for kw in keywords:
        for c in cols:
            if c == kw or (kw in c):
                return c
    if required:
        raise KeyError(f"找不到列（关键词 {keywords}），当前列名：{cols}")
    return None

def _pct(arr, p):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    a.sort()
    k = (len(a)-1)*p/100.0
    f, c = int(np.floor(k)), int(np.ceil(k))
    return float(a[f]) if f == c else float(a[f] + (k-f)*(a[c]-a[f]))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-12)
    return float(np.mean(np.abs((y_pred[m]-y_true[m]) / y_true[m])) * 100.0) if np.any(m) else np.nan

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(m) < 3:
        return np.nan
    yt, yp = y_true[m], y_pred[m]
    ss_tot = float(np.sum((yt - np.mean(yt))**2))
    ss_res = float(np.sum((yt - yp)**2))
    return float(1.0 - ss_res/ss_tot) if ss_tot > 0 else np.nan

def durbin_watson(resid):
    r = np.asarray(resid, float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return np.nan
    dr = np.diff(r)
    den = float(np.dot(r, r))
    return float(np.dot(dr, dr)/den) if den > 0 else np.nan

def ar1_rho_ols(resid):
    r = np.asarray(resid, float)
    r = r[np.isfinite(r)]
    if r.size < 4:
        return np.nan
    e0, e1 = r[:-1], r[1:]
    den = float(np.dot(e0, e0))
    return float(np.dot(e0, e1)/den) if den > 0 else np.nan

def _fmt(x, nd=3):
    return "N/A" if (x is None or (isinstance(x,float) and not np.isfinite(x))) else f"{x:.{nd}f}"

def centered_ma(x, win=3):
    x = np.asarray(x, dtype=float)
    if win <= 1:
        return x.copy()
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode='edge')
    ker = np.ones(win) / win
    return np.convolve(xpad, ker, mode='valid')

def ema(x, alpha):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        if not np.isfinite(x[i]):
            x[i] = x[i-1]
        y[i] = alpha*x[i] + (1-alpha)*y[i-1]
    return y

def _metrics(yhat, y, years_arr, a=None, b=None):
    m = np.isfinite(y) & np.isfinite(yhat)
    if a is not None: m &= (years_arr >= a)
    if b is not None: m &= (years_arr <= b)
    if not np.any(m):
        return dict(MAE=np.nan, MAPE=np.nan, RMSE=np.nan, Bias=np.nan, R2=np.nan)

    yt, yu = yhat[m], y[m]
    eps_den = max(1e-6, _pct(np.abs(yu), 20))
    mae  = float(np.mean(np.abs(yt - yu)))
    mapev = float(np.mean(np.abs(yt - yu) / np.maximum(np.abs(yu), eps_den))) * 100
    rmse = float(np.sqrt(np.mean((yt - yu)**2)))
    bias = float(np.mean(yt - yu))

    ybar = float(np.mean(yu))
    ss_tot = float(np.sum((yu - ybar)**2))
    ss_res = float(np.sum((yu - yt)**2))
    r2v = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else np.nan
    return dict(MAE=mae, MAPE=mapev, RMSE=rmse, Bias=bias, R2=r2v)

def _tail_slope(years_arr, yhat_arr, start, end):
    m = (years_arr >= start) & (years_arr <= end) & np.isfinite(yhat_arr)
    if np.sum(m) < 3:
        return dict(slope_ton_per_year=np.nan, slope_pct_per_year=np.nan, R2=np.nan)
    x  = years_arr[m].astype(float)
    yh = yhat_arr[m].astype(float)
    k, b = np.polyfit(x, yh, 1)
    slope_pct = 100.0 * k / max(np.mean(yh), 1e-12)
    ybar = np.mean(yh)
    ss_tot = np.sum((yh - ybar)**2)
    ss_res = np.sum((yh - (k*x + b))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return dict(slope_ton_per_year=float(k), slope_pct_per_year=float(slope_pct), R2=float(r2))

def _pick_year(y_arr, val_arr, year):
    m = (y_arr == year)
    if np.any(m):
        v = float(val_arr[m][0])
        return v if np.isfinite(v) else np.nan
    return np.nan


df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, engine="openpyxl").copy()
df.columns = [str(c).strip() for c in df.columns]
df = df.sort_values(pick_col(df, ["年份", "Year"])).reset_index(drop=True)

COL_YEAR   = pick_col(df, ["年份", "Year"])
COL_Y      = pick_col(df, ["水泥产量（亿吨）", "水泥产量"])
COL_GDP    = pick_col(df, ["GDP(十亿美元)", "GDP（十亿美元）", "GDP"])
COL_POP    = pick_col(df, ["人口数量", "人口"])
COL_URB    = pick_col(df, ["城镇化率", "城市化率"])
COL_URBPOP = pick_col(df, ["城镇人口数量", "城镇人口数"])
COL_AREA   = pick_col(df, ["拟合城镇人均住宅面积", "预测城镇人均住宅面积", "拟合人均住宅面积"])

for c in [COL_YEAR, COL_Y, COL_GDP, COL_POP, COL_URB, COL_URBPOP, COL_AREA]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df[(df[COL_YEAR] >= YEAR_START) & (df[COL_YEAR] <= FORECAST_END)].copy()
df[COL_YEAR] = df[COL_YEAR].astype(int)

years = df[COL_YEAR].to_numpy(int)
y_actual = df[COL_Y].to_numpy(float)


def ridge_fit(X, y, lam, penalize_intercept=False):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    p = X.shape[1]
    I = np.eye(p)
    if not penalize_intercept:
        I[0,0] = 0.0
    return np.linalg.solve(X.T @ X + lam*I, X.T @ y)

def zscore_fit_transform(X):
    X = np.asarray(X, float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu)/sd, mu, sd

def zscore_transform(X, mu, sd):
    X = np.asarray(X, float)
    return (X - mu)/sd

gdp_usd = df[COL_GDP].to_numpy(float) * 1e9
pop_total = df[COL_POP].to_numpy(float)
gdppc = gdp_usd / np.maximum(pop_total, 1e-12)
ln_gdppc = np.log(np.maximum(gdppc, 1e-12))

gdp_gr = np.full_like(gdp_usd, np.nan, float)
gdp_gr[1:] = gdp_usd[1:] / np.maximum(gdp_usd[:-1], 1e-12) - 1.0

urban01 = np.clip(df[COL_URB].to_numpy(float)/100.0, 0.02, 0.98)
ln_urban_pop = np.log(np.maximum(df[COL_URBPOP].to_numpy(float), 1e-12))
area = df[COL_AREA].to_numpy(float)
t = (years - YEAR_START)/10.0


def dyear(yy): return (years == yy).astype(float)
d2008 = dyear(2008); d2009 = dyear(2009); d2020 = dyear(2020); d2022 = dyear(2022)

logy = np.log(np.maximum(y_actual, 1e-12))
logy_lag1 = np.r_[np.nan, logy[:-1]]
logy_lag2 = np.r_[np.nan, np.nan, logy[:-2]]

mask_arx = (years <= HIS_END) & np.isfinite(y_actual) & np.isfinite(logy_lag1) & np.isfinite(logy_lag2)

X_arx = np.column_stack([
    ln_gdppc, gdp_gr, urban01, ln_urban_pop, area, t,
    d2008, d2009, d2020, d2022,
    logy_lag1, logy_lag2
])

X_train_raw = X_arx[mask_arx]
y_train_log = logy[mask_arx]

X_train_z, mu_arx, sd_arx = zscore_fit_transform(X_train_raw)
X_train = np.column_stack([np.ones(len(X_train_z)), X_train_z])

LAM_ARX_GRID = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
best_arx = None
for lam in LAM_ARX_GRID:
    beta = ridge_fit(X_train, y_train_log, lam=lam, penalize_intercept=False)
    yhat_log = X_train @ beta
    yhat = np.exp(yhat_log)
    y_true = np.exp(y_train_log)
    m = mape(y_true, yhat)
    r2 = r2_score(y_true, yhat)
    if best_arx is None or (m < best_arx[0] - 1e-9) or (abs(m-best_arx[0])<=1e-9 and r2 > best_arx[1] + 1e-9):
        best_arx = (m, r2, lam, beta)

m_arx_in, r2_arx_in, lam_arx, beta_arx = best_arx


X_all_z = zscore_transform(X_arx, mu_arx, sd_arx)
X_all = np.column_stack([np.ones(len(X_all_z)), X_all_z])
yhat_log_all = X_all @ beta_arx
yhat_all = np.exp(yhat_log_all)

arx_fit = np.full_like(y_actual, np.nan, float)
arx_fit[mask_arx] = yhat_all[mask_arx]


actual_mask = np.isfinite(y_actual)

population_raw = df[COL_URBPOP].to_numpy(float)
pc_area_raw    = df[COL_AREA].to_numpy(float)

population = centered_ma(pd.Series(population_raw).interpolate(limit_direction="both").bfill().ffill().to_numpy(float), win=3)
pc_area    = centered_ma(pd.Series(pc_area_raw).interpolate(limit_direction="both").bfill().ffill().to_numpy(float), win=3)

n = len(years)


def cement_intensity_uk(y: int) -> float:

    return 0.12

TRANSITION_RATE = 3.0

K_SHAPE = 2.95
LAMBDA_SCALE = 70.82
MAX_LIFE = 120
k_shape = K_SHAPE
lambda_scale = LAMBDA_SCALE
max_life = MAX_LIFE

life = np.arange(1, MAX_LIFE + 1)
pdf_weibull = weibull_min.pdf(life, c=K_SHAPE, scale=LAMBDA_SCALE)
pdf_weibull = np.nan_to_num(pdf_weibull)
pdf_weibull /= np.maximum(pdf_weibull.sum(), 1e-12)
expected_life = float(weibull_min.mean(c=K_SHAPE, scale=LAMBDA_SCALE))

stock = population * pc_area
if len(stock) >= 6 and stock[0] > 0 and stock[0] < 0.6 * np.median(stock[1:6]):
    stock += (np.median(stock[1:6]) - stock[0])

inflow = np.zeros(n)
outflow = np.zeros(n)
steady_inflow = max(stock[0] / max(expected_life, 1.0), 1e-6)
hist_inflow = np.full(MAX_LIFE, steady_inflow)

for t_i in range(n):
    past = np.concatenate([hist_inflow, inflow[:t_i]])
    use = past[-MAX_LIFE:]
    outflow[t_i] = float(np.sum(use[::-1] * pdf_weibull[:len(use)]))

    raw_inflow = stock[t_i] - stock[t_i - 1] if t_i > 0 else (stock[1] - stock[0] if n > 1 else 0.0)
    inflow[t_i] = max(0.0, raw_inflow)

intensity = np.array([cement_intensity_uk(int(y)) for y in years], dtype=float)

hist_floor_mask = (
    (years >= max(YEAR_START, HIS_END - 9))
    & (years <= HIS_END)
    & np.isfinite(inflow)
    & (inflow > 0)
)
if np.any(hist_floor_mask):
    floor_anchor = float(np.nanmedian(inflow[hist_floor_mask]))
else:
    pos_inflow = inflow[np.isfinite(inflow) & (inflow > 0)]
    floor_anchor = float(np.nanmedian(pos_inflow)) if pos_inflow.size else 0.0


start_floor = 0.30 * floor_anchor
long_floor = 0.12 * floor_anchor
floor_tau = 40.0
future_dt = np.maximum(years - (HIS_END + 1), 0)
non_repair_activity_floor = long_floor + (start_floor - long_floor) * np.exp(-future_dt / floor_tau)
non_repair_activity_floor[years <= HIS_END] = 0.0
non_repair_activity_floor = np.maximum(non_repair_activity_floor, 0.0)
effective_inflow_for_B = np.where(years > HIS_END, np.maximum(inflow, non_repair_activity_floor), inflow)

baseline_B_tons = effective_inflow_for_B * intensity * (1.0 + TRANSITION_RATE)
if "亿吨" in str(COL_Y):
    CEMENT_OUTPUT_UNIT_SCALE = 1e8
elif "百万吨" in str(COL_Y) or "million" in str(COL_Y).lower() or "mt" in str(COL_Y).lower():
    CEMENT_OUTPUT_UNIT_SCALE = 1e6
else:
    CEMENT_OUTPUT_UNIT_SCALE = 1.0
baseline_B = baseline_B_tons / CEMENT_OUTPUT_UNIT_SCALE

HIST_PHASES = [
    ("p01_1990_1995", (years >= 1990) & (years <= 1995)),
    ("p02_1996_2000", (years >= 1996) & (years <= 2000)),
    ("p03_2001_2005", (years >= 2001) & (years <= 2005)),
    ("p04_2006_2010", (years >= 2006) & (years <= 2010)),
    ("p05_2011_2015", (years >= 2011) & (years <= 2015)),
    ("p06_2016_2019", (years >= 2016) & (years <= 2019)),
    ("p07_2020", years == 2020),
    ("p08_2021", years == 2021),
    ("p09_2022", years == 2022),
    ("p10_2023", years == 2023),
    ("p11_2024", years == 2024),
]

multiplier = np.full(n, np.nan, dtype=float)
phase_rows = []

for phase_name, phase_mask in HIST_PHASES:
    fit_mask_phase = (
        phase_mask
        & (years <= HIS_END)
        & actual_mask
        & np.isfinite(y_actual)
        & np.isfinite(baseline_B)
        & (baseline_B > 1e-9)
    )
    if not np.any(fit_mask_phase):
        continue

    Bp = baseline_B[fit_mask_phase]
    yp = y_actual[fit_mask_phase]
    bj = float(np.sum(Bp * yp) / max(np.sum(Bp * Bp), 1e-12))
    bj = float(np.clip(bj, 0.05, 20.0))
    multiplier[phase_mask] = bj
    phase_rows.append({
        "phase": phase_name,
        "year_start": int(np.min(years[phase_mask])),
        "year_end": int(np.max(years[phase_mask])),
        "multiplier": bj,
        "n_fit": int(np.sum(fit_mask_phase)),
    })

if np.any(np.isfinite(multiplier)):
    multiplier = pd.Series(multiplier).ffill().bfill().to_numpy(dtype=float).copy()
else:
    multiplier = np.ones(n, dtype=float)

FUTURE_MULT_ANCHOR_YEAR = HIS_END
FUTURE_MULT_DECAY = 1.0
FUTURE_MULT_FLOOR_FRAC = 1.0

anchor_mask = years == FUTURE_MULT_ANCHOR_YEAR
anchor_mult = float(multiplier[anchor_mask][0]) if np.any(anchor_mask) else float(multiplier[years <= HIS_END][-1])
mult_floor = anchor_mult * FUTURE_MULT_FLOOR_FRAC

for i, yy in enumerate(years):
    if yy > HIS_END:
        dt = yy - FUTURE_MULT_ANCHOR_YEAR
        multiplier[i] = max(mult_floor, anchor_mult * (FUTURE_MULT_DECAY ** dt))

stock_pred_raw = np.maximum(baseline_B * multiplier, 0.0)
hist_scale_mask = (
    (years <= HIS_END)
    & actual_mask
    & np.isfinite(stock_pred_raw)
    & (stock_pred_raw > 1e-9)
)
if np.any(hist_scale_mask):
    p = stock_pred_raw[hist_scale_mask]
    y = y_actual[hist_scale_mask]
    global_scale = float(np.sum(p * y) / max(np.sum(p * p), 1e-12))
else:
    global_scale = 1.0

stock_pred_all = np.maximum(stock_pred_raw * global_scale, 0.0)
best_pack = "ChinaLogic: B * historical_multiplier_extension * global_scale"
best_cal_info = {"alpha": 0.0, "beta": global_scale}


hybrid = np.full_like(stock_pred_all, np.nan, float)

mask_hist = (years <= HIS_END)
hybrid[mask_hist] = arx_fit[mask_hist]

miss = mask_hist & (~np.isfinite(hybrid))
hybrid[miss] = np.where(np.isfinite(y_actual[miss]), y_actual[miss], stock_pred_all[miss])

mask_fc = (years >= HIS_END + 1)
hybrid[mask_fc] = stock_pred_all[mask_fc]


FORECAST_BRIDGE_END = 2030

scenario_raw_before_bridge = stock_pred_all.copy()
forecast_bridge_weight = np.ones(len(years), dtype=float)
forecast_bridge_weight[years <= HIS_END] = 0.0

if np.any(years == HIS_END):
    idx_2024_bridge = int(np.where(years == HIS_END)[0][0])
    if np.isfinite(y_actual[idx_2024_bridge]) and y_actual[idx_2024_bridge] > 0:
        bridge_anchor_2024 = float(y_actual[idx_2024_bridge])
    else:
        bridge_anchor_2024 = float(hybrid[idx_2024_bridge])
else:
    idx_2024_bridge = None
    bridge_anchor_2024 = np.nan


hybrid[years > HIS_END] = scenario_raw_before_bridge[years > HIS_END]


if idx_2024_bridge is not None and np.isfinite(bridge_anchor_2024):
    hybrid[idx_2024_bridge] = bridge_anchor_2024

bridge_mask = (years > HIS_END) & (years <= FORECAST_BRIDGE_END)
if np.any(bridge_mask) and np.isfinite(bridge_anchor_2024):
    x_bridge = (
        years[bridge_mask].astype(float) - float(HIS_END)
    ) / max(float(FORECAST_BRIDGE_END - HIS_END), 1.0)
    x_bridge = np.clip(x_bridge, 0.0, 1.0)
    w_bridge = x_bridge ** 2 * (3.0 - 2.0 * x_bridge)
    forecast_bridge_weight[bridge_mask] = w_bridge

    hybrid[bridge_mask] = (
        (1.0 - w_bridge) * bridge_anchor_2024
        + w_bridge * scenario_raw_before_bridge[bridge_mask]
    )


splice_scale = 1.0
forecast_splice_scale = 1.0
forecast_start_yoy = np.nan


hybrid = _shared_history('united_kingdom', 'nonreplacement', years, hybrid, y_actual)


m_hybrid_hist = _metrics(hybrid, y_actual, years, a=YEAR_START, b=HIS_END)
m_stock_hist  = _metrics(stock_pred_all, y_actual, years, a=YEAR_START, b=HIS_END)

hist_mask = (years <= HIS_END) & np.isfinite(y_actual) & np.isfinite(hybrid)
resid_hist = y_actual[hist_mask] - hybrid[hist_mask]
DW_val = durbin_watson(resid_hist)
AR1_val = ar1_rho_ols(resid_hist)

tail = _tail_slope(years, hybrid, 2033, 2100)
y2030 = _pick_year(years, hybrid, 2030)
y2100 = _pick_year(years, hybrid, 2100)
drop_30_00 = 100.0*(y2100/y2030 - 1.0) if (np.isfinite(y2030) and y2030 > 0 and np.isfinite(y2100)) else np.nan


neg_years = int(np.sum(np.isfinite(hybrid) & (hybrid < 0)))
up_years_25_00 = int(np.sum(np.diff(hybrid[(years>=2025)&(years<=2100)&np.isfinite(hybrid)]) > 0)) if np.sum((years>=2025)&(years<=2100)&np.isfinite(hybrid)) >= 3 else np.nan

reason_df = pd.DataFrame([{
    "Logic_version": NO_OUTFLOW_LOGIC_VERSION,
    "Hist_period": f"{YEAR_START}-{HIS_END}",
    "Hybrid_MAE": m_hybrid_hist["MAE"],
    "Hybrid_MAPE_%": m_hybrid_hist["MAPE"],
    "Hybrid_RMSE": m_hybrid_hist["RMSE"],
    "Hybrid_Bias": m_hybrid_hist["Bias"],
    "Hybrid_R2": m_hybrid_hist["R2"],
    "Residual_DW": DW_val,
    "Residual_AR1_rho": AR1_val,

    "ARX_in_sample_MAPE_%": m_arx_in,
    "ARX_in_sample_R2": r2_arx_in,
    "ARX_best_lambda": lam_arx,

    "Stock_hist_MAPE_%": m_stock_hist["MAPE"],
    "Stock_hist_R2": m_stock_hist["R2"],
    "Stock_method": best_pack,
    "Stock_k_shape": K_SHAPE,
    "Stock_lambda_scale": LAMBDA_SCALE,
    "Stock_max_life": MAX_LIFE,
    "Stock_unit_intensity": 0.12,
    "Stock_transition_rate": TRANSITION_RATE,
    "Cement_output_unit_scale": CEMENT_OUTPUT_UNIT_SCALE,
    "Future_multiplier_decay": FUTURE_MULT_DECAY,
    "Future_multiplier_floor_frac": FUTURE_MULT_FLOOR_FRAC,
    "Future_anchor_multiplier": anchor_mult,
    "Stock_calib_alpha": best_cal_info["alpha"] if best_cal_info else np.nan,
    "Stock_global_scale": best_cal_info["beta"] if best_cal_info else np.nan,

    "Splice_scale_2025plus": splice_scale,
    "Forecast_splice_scale": forecast_splice_scale,
    "Forecast_start_yoy": forecast_start_yoy,
    "Non_repair_activity_floor_anchor": floor_anchor,
    "Non_repair_activity_floor_start": start_floor,
    "Non_repair_activity_floor_long": long_floor,

    "Tail_start": 2033,
    "Tail_end": 2100,
    "Tail_slope_ton_per_year": tail["slope_ton_per_year"],
    "Tail_slope_pct_per_year": tail["slope_pct_per_year"],
    "Tail_line_R2": tail["R2"],

    "y2030": y2030,
    "y2100": y2100,
    "Drop_2030_to_2100_%": drop_30_00,

    "Neg_years_count": neg_years,
    "Up_years_2025_2100": up_years_25_00,

    "Thresh_MAPE_hist_%": MAPE_THRESH_HIST,
    "Thresh_R2_hist": R2_THRESH_HIST,
    "Thresh_Drop_2030_2100_min_%": DROP_2030_2100_MIN,
}])

weibull_df = pd.DataFrame([{
    "Weibull_shape_k": k_shape,
    "Weibull_scale_lambda_years": lambda_scale,
    "max_life": max_life,
    "expected_life_years": expected_life,
}])


series_df = pd.DataFrame({
    "Year": years,
    "Actual_cement_1990_2024(亿吨)": y_actual,
    "ARX_fit_1990_2024": arx_fit,
    "StockDriven_pred_all_1990_2100": stock_pred_all,
    "Scenario_raw_before_bridge": scenario_raw_before_bridge,
    "Forecast_bridge_weight": forecast_bridge_weight,
    "Hybrid_final_1990_2100": hybrid,
    "Urban_pop": population_raw,
    "Urban_pc_area": pc_area_raw,
    "Stock": stock,
    "Inflow_raw": inflow,
    "Outflow": outflow,
    "Non_repair_activity_floor_m2": non_repair_activity_floor,
    "Effective_inflow_for_B_m2": effective_inflow_for_B,
    "Baseline_B_tons": baseline_B_tons,
    "Baseline_B": baseline_B,
    "Historical_multiplier": multiplier,
    "Global_scale": global_scale,
    "ChinaLogic_pred_all": stock_pred_all,
})

phase_df = pd.DataFrame(phase_rows)

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
    series_df.to_excel(w, index=False, sheet_name="Series")
    reason_df.to_excel(w, index=False, sheet_name="Reasonableness")
    weibull_df.to_excel(w, index=False, sheet_name="Weibull_Params")
    phase_df.to_excel(w, index=False, sheet_name="Phase_Multipliers")

print(f"\n✅ Exported: {OUT_XLSX}")


print("\n================ UK HYBRID SUMMARY ================")
future_mask_check = (years >= HIS_END + 1) & np.isfinite(hybrid)
future_min_hybrid = float(np.nanmin(hybrid[future_mask_check])) if np.any(future_mask_check) else np.nan
future_y2100 = _pick_year(years, hybrid, 2100)
future_y2100_eff_inflow = _pick_year(years, effective_inflow_for_B, 2100)
print(f"[Logic] {NO_OUTFLOW_LOGIC_VERSION}")
print(f"[No-outflow check] min Hybrid 2025-2100={_fmt(future_min_hybrid,6)} | y2100={_fmt(future_y2100,6)} | eff_inflow_2100={_fmt(future_y2100_eff_inflow,3)}")
print(f"[ARX] best_lambda={lam_arx} | in-sample MAPE={_fmt(m_arx_in)}% R2={_fmt(r2_arx_in,4)}")
print(f"[Stock] method={best_pack} | global_scale={_fmt(best_cal_info['beta'])}")
print(f"[Hybrid hist] MAPE={_fmt(m_hybrid_hist['MAPE'])}% R2={_fmt(m_hybrid_hist['R2'],4)} DW={_fmt(DW_val,4)} AR1={_fmt(AR1_val,4)}")
print(f"[Tail] 2033-2100 slope={_fmt(tail['slope_pct_per_year'])}%/y | Drop2030->2100={_fmt(drop_30_00)}%")
print(f"[Weibull] k={k_shape}  lambda={lambda_scale}  max_life={max_life}  E[life]={expected_life:.2f}")


plt.figure(figsize=(13, 7))
plt.plot(years, hybrid, linewidth=2.2, label="Hybrid final (ARX history + China-logic forecast)")
plt.plot(years, stock_pred_all, linewidth=1.6, alpha=0.75, label="China-logic stock reference")
plt.plot(years, arx_fit, linewidth=1.6, alpha=0.75, label="ARX one-step fit (hist)")
plt.scatter(years[np.isfinite(y_actual)], y_actual[np.isfinite(y_actual)], s=18, alpha=0.7, label="Actual (1990-2024)")
plt.axvline(HIS_END + 1, color="gray", linestyle="--", label="Forecast start")
plt.xlim(YEAR_START, FORECAST_END)
plt.xlabel("Year"); plt.ylabel("Cement production (same unit as input)")
plt.title("UK Cement | ARX History Fit + Stock-driven China-style Forecast")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
