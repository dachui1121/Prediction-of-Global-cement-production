from cement_bau.runtime import input_path as _input_path, output_dir as _output_dir
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from scipy.optimize import least_squares


DATA_PATH = str(_input_path('russia'))
SHEET_NAME = "Sheet1"

YEAR_START = 1990
HIS_END = 2024
FORECAST_END = 2100


MAPE_THRESH_HIST = 12.0
R2_THRESH_HIST = 0.80
DROP_2030_2100_MIN = -45.0
TAIL_R2_THRESH = 0.50


OUT_XLSX = os.path.join(str(_output_dir('russia', 'replacement')), 'Russia_Cement_Replacement_2024_2030_SmoothBridge_1990_2100.xlsx')


plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11


def _pct(arr, p):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    a.sort()
    k = (len(a) - 1) * p / 100.0
    f, c = int(np.floor(k)), int(np.ceil(k))
    return float(a[f]) if f == c else float(a[f] + (k - f) * (a[c] - a[f]))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-12)
    return float(np.mean(np.abs((y_pred[m] - y_true[m]) / y_true[m])) * 100.0) if np.any(m) else np.nan

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(m) < 3:
        return np.nan
    yt, yp = y_true[m], y_pred[m]
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    ss_res = float(np.sum((yt - yp) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

def durbin_watson(resid):
    r = np.asarray(resid, float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return np.nan
    dr = np.diff(r)
    den = float(np.dot(r, r))
    return float(np.dot(dr, dr) / den) if den > 0 else np.nan

def ar1_rho_ols(resid):
    r = np.asarray(resid, float)
    r = r[np.isfinite(r)]
    if r.size < 4:
        return np.nan
    e0, e1 = r[:-1], r[1:]
    den = float(np.dot(e0, e0))
    return float(np.dot(e0, e1) / den) if den > 0 else np.nan

def _fmt(x, nd=3):
    return "N/A" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.{nd}f}"

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
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

def _metrics(yhat, y, years_arr, a=None, b=None):
    m = np.isfinite(y) & np.isfinite(yhat)
    if a is not None:
        m &= (years_arr >= a)
    if b is not None:
        m &= (years_arr <= b)
    if not np.any(m):
        return dict(MAE=np.nan, MAPE=np.nan, RMSE=np.nan, Bias=np.nan, R2=np.nan)

    yt, yu = yhat[m], y[m]
    eps_den = max(1e-6, _pct(np.abs(yu), 20))
    mae = float(np.mean(np.abs(yt - yu)))
    mapev = float(np.mean(np.abs(yt - yu) / np.maximum(np.abs(yu), eps_den))) * 100
    rmse = float(np.sqrt(np.mean((yt - yu) ** 2)))
    bias = float(np.mean(yt - yu))

    ybar = float(np.mean(yu))
    ss_tot = float(np.sum((yu - ybar) ** 2))
    ss_res = float(np.sum((yu - yt) ** 2))
    r2v = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return dict(MAE=mae, MAPE=mapev, RMSE=rmse, Bias=bias, R2=r2v)

def _tail_slope(years_arr, yhat_arr, start, end):
    m = (years_arr >= start) & (years_arr <= end) & np.isfinite(yhat_arr)
    if np.sum(m) < 3:
        return dict(slope_ton_per_year=np.nan, slope_pct_per_year=np.nan, R2=np.nan)
    x = years_arr[m].astype(float)
    yh = yhat_arr[m].astype(float)
    k, b = np.polyfit(x, yh, 1)
    slope_pct = 100.0 * k / max(np.mean(yh), 1e-12)
    ybar = np.mean(yh)
    ss_tot = np.sum((yh - ybar) ** 2)
    ss_res = np.sum((yh - (k * x + b)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(slope_ton_per_year=float(k), slope_pct_per_year=float(slope_pct), R2=float(r2))

def _pick_year(y_arr, val_arr, year):
    m = (y_arr == year)
    if np.any(m):
        v = float(val_arr[m][0])
        return v if np.isfinite(v) else np.nan
    return np.nan


df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, engine="openpyxl").sort_values("年份").reset_index(drop=True)
df.columns = [str(c).strip() for c in df.columns]

COL_YEAR = "年份"
COL_GDP = "GDP（十亿美元）"
COL_POP = "人口"
COL_URB = "城镇化率"
COL_URBPOP = "城镇人口数"
COL_AREA = "预测人均住宅面积"
COL_Y = "水泥产量"

for c in [COL_YEAR, COL_GDP, COL_POP, COL_URB, COL_URBPOP, COL_AREA, COL_Y]:
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
        I[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + lam * I, X.T @ y)

def zscore_fit_transform(X):
    X = np.asarray(X, float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd

def zscore_transform(X, mu, sd):
    X = np.asarray(X, float)
    return (X - mu) / sd

gdp_usd = df[COL_GDP].to_numpy(float) * 1e9
pop_total = df[COL_POP].to_numpy(float)
gdppc = gdp_usd / np.maximum(pop_total, 1e-12)
ln_gdppc = np.log(np.maximum(gdppc, 1e-12))

gdp_gr = np.full_like(gdp_usd, np.nan, float)
gdp_gr[1:] = gdp_usd[1:] / np.maximum(gdp_usd[:-1], 1e-12) - 1.0

urban01 = np.clip(df[COL_URB].to_numpy(float) / 100.0, 0.02, 0.98)
ln_urban_pop = np.log(np.maximum(df[COL_URBPOP].to_numpy(float), 1e-12))
area = df[COL_AREA].to_numpy(float)
t = (years - YEAR_START) / 10.0

def dyear(yy):
    return (years == yy).astype(float)

d1998 = dyear(1998)
d2009 = dyear(2009)
d2014 = dyear(2014)
d2020 = dyear(2020)
d2022 = dyear(2022)

logy = np.log(np.maximum(y_actual, 1e-12))
logy_lag1 = np.r_[np.nan, logy[:-1]]
logy_lag2 = np.r_[np.nan, np.nan, logy[:-2]]

mask_arx = (years <= HIS_END) & np.isfinite(y_actual) & np.isfinite(logy_lag1) & np.isfinite(logy_lag2)

X_arx = np.column_stack([
    ln_gdppc, gdp_gr, urban01, ln_urban_pop, area, t,
    d1998, d2009, d2014, d2020, d2022,
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
    if best_arx is None or (m < best_arx[0] - 1e-9) or (abs(m - best_arx[0]) <= 1e-9 and r2 > best_arx[1] + 1e-9):
        best_arx = (m, r2, lam, beta)

m_arx, r2_arx, lam_arx, beta_arx = best_arx

X_all_z = zscore_transform(X_arx, mu_arx, sd_arx)
X_all = np.column_stack([np.ones(len(X_all_z)), X_all_z])
yhat_log_all = X_all @ beta_arx
yhat_all = np.exp(yhat_log_all)

arx_fit = np.full_like(y_actual, np.nan, float)
arx_fit[mask_arx] = yhat_all[mask_arx]


actual_tons = y_actual.copy()
actual_mask = np.isfinite(actual_tons)

population_raw = df[COL_URBPOP].to_numpy(float)
pc_area_raw = df[COL_AREA].to_numpy(float)

population = population_raw.copy()
pc_area = pc_area_raw.copy()

n = len(years)

def cement_intensity_russia(y: int) -> float:

    return 0.15

def _pick_col(df_, cands):
    for c in cands:
        if c in df_.columns:
            return c
    return None

def _pick_scalar(df_, cands):
    c = _pick_col(df_, cands)
    if c is None:
        return None
    s = pd.to_numeric(df_[c], errors="coerce").dropna()
    return None if s.empty else float(s.iloc[0])

k_shape0 = _pick_scalar(df, ["Weibull_k", "寿命形状参数", "形状参数k"])
lambda_scale0 = _pick_scalar(df, ["Weibull_lambda", "寿命尺度参数", "尺度参数lambda", "lambda"])
max_life0 = _pick_scalar(df, ["max_life", "最大寿命", "寿命截断"])

k_shape = 2.5
lambda_scale = 87.35
max_life = 150

life = np.arange(1, max_life + 1)
pdf_weib = weibull_min.pdf(life, c=k_shape, scale=lambda_scale)
pdf_weib /= pdf_weib.sum()
expected_life = weibull_min.mean(c=k_shape, scale=lambda_scale)

stock = population * pc_area
if np.isfinite(stock[0]) and stock[0] > 0 and np.isfinite(np.median(stock[1:6])) and stock[0] < 0.6 * np.median(stock[1:6]):
    stock += (np.median(stock[1:6]) - stock[0])

inflow = np.zeros(n)
outflow = np.zeros(n)
steady_inflow = max(stock[0] / max(expected_life, 1.0), 1e-6)
hist_inflow = np.full(max_life, steady_inflow)

for t_i in range(n):
    past = np.concatenate([hist_inflow, inflow[:t_i]])
    use = past[-max_life:]
    outflow[t_i] = (use[::-1] * pdf_weib[:len(use)]).sum()
    raw_inflow = (stock[t_i] - stock[t_i - 1]) + outflow[t_i] if t_i > 0 else outflow[t_i]
    inflow[t_i] = max(0.0, raw_inflow)

TRANSITION_RATE = 4.0
CEMENT_OUTPUT_UNIT_SCALE = 1e8

intensity = np.array([cement_intensity_russia(int(y)) for y in years], dtype=float)
baseline_B_tons = inflow * intensity * (1.0 + TRANSITION_RATE)
baseline_B = baseline_B_tons / CEMENT_OUTPUT_UNIT_SCALE

HIST_PHASES = [
    ("p01_1990_1991", (years >= 1990) & (years <= 1991)),
    ("p02_1992_1994", (years >= 1992) & (years <= 1994)),
    ("p03_1995_1998", (years >= 1995) & (years <= 1998)),
    ("p04_1999_2002", (years >= 1999) & (years <= 2002)),
    ("p05_2003_2008", (years >= 2003) & (years <= 2008)),
    ("p06_2009", years == 2009),
    ("p07_2010_2013", (years >= 2010) & (years <= 2013)),
    ("p08_2014_2016", (years >= 2014) & (years <= 2016)),
    ("p09_2017_2019", (years >= 2017) & (years <= 2019)),
    ("p10_2020", years == 2020),
    ("p11_2021", years == 2021),
    ("p12_2022", years == 2022),
    ("p13_2023", years == 2023),
    ("p14_2024", years == 2024),
]

multiplier = np.full(n, np.nan, dtype=float)
phase_rows = []

for phase_name, phase_mask in HIST_PHASES:
    fit_mask = phase_mask & (years <= HIS_END) & actual_mask & np.isfinite(baseline_B) & (baseline_B > 1e-9)
    if not np.any(fit_mask):
        continue

    Bp = baseline_B[fit_mask]
    yp = actual_tons[fit_mask]
    w = np.ones_like(yp, dtype=float)
    bj = float(np.sum(w * Bp * yp) / max(np.sum(w * Bp * Bp), 1e-12))
    bj = float(np.clip(bj, 0.05, 20.0))
    multiplier[phase_mask] = bj
    phase_rows.append({
        "phase": phase_name,
        "year_start": int(np.min(years[phase_mask])),
        "year_end": int(np.max(years[phase_mask])),
        "multiplier": bj,
        "n_fit": int(np.sum(fit_mask)),
    })

known_mult = np.isfinite(multiplier)
if np.any(known_mult):
    multiplier = pd.Series(multiplier).ffill().bfill().to_numpy(dtype=float).copy()
else:
    multiplier = np.full(n, 1.0, dtype=float)

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

hist_scale_mask = (years <= HIS_END) & actual_mask & np.isfinite(stock_pred_raw) & (stock_pred_raw > 1e-9)
if np.any(hist_scale_mask):
    p = stock_pred_raw[hist_scale_mask]
    y = actual_tons[hist_scale_mask]
    global_scale = float(np.sum(p * y) / max(np.sum(p * p), 1e-12))
else:
    global_scale = 1.0

stock_pred_all = np.maximum(stock_pred_raw * global_scale, 0.0)
best_pack = "ChinaLogic: B * historical_multiplier_extension * global_scale"
best_cal_info = {"alpha": 0.0, "beta": global_scale}


hybrid = np.full_like(stock_pred_all, np.nan, dtype=float)

mask_hist = years <= HIS_END
mask_fc = years > HIS_END

hybrid[mask_hist] = arx_fit[mask_hist]
miss_hist = mask_hist & ~np.isfinite(hybrid)
hybrid[miss_hist] = np.where(np.isfinite(y_actual[miss_hist]), y_actual[miss_hist], stock_pred_all[miss_hist])


FORECAST_BRIDGE_END = 2030


stock_pred_raw_future = stock_pred_all.copy()


hybrid[mask_fc] = stock_pred_raw_future[mask_fc]


if np.any(years == HIS_END):
    idx_2024_bridge = int(np.where(years == HIS_END)[0][0])
    if np.isfinite(y_actual[idx_2024_bridge]) and y_actual[idx_2024_bridge] > 0:
        bridge_anchor_2024 = float(y_actual[idx_2024_bridge])
    else:
        bridge_anchor_2024 = float(hybrid[idx_2024_bridge])
else:
    bridge_anchor_2024 = np.nan


forecast_anchor_scale = 1.0
splice_scale = 1.0
forecast_start_yoy = np.nan


forecast_bridge_weight = np.ones(len(years), dtype=float)
forecast_bridge_weight[years <= HIS_END] = 0.0

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
        + w_bridge * stock_pred_raw_future[bridge_mask]
    )


forecast_display = np.full(len(years), np.nan, dtype=float)
display_mask = years >= HIS_END
forecast_display[display_mask] = hybrid[display_mask]
if np.any(years == HIS_END):
    _i24 = int(np.where(years == HIS_END)[0][0])
    if np.isfinite(y_actual[_i24]) and y_actual[_i24] > 0:
        forecast_display[_i24] = y_actual[_i24]

scale = global_scale


m_arx_hist = _metrics(hybrid, y_actual, years, a=YEAR_START, b=HIS_END)
m_stock_hist = _metrics(stock_pred_all, y_actual, years, a=YEAR_START, b=HIS_END)

tail = _tail_slope(years, hybrid, 2033, 2100)
tail_r2 = tail["R2"]

y2030 = _pick_year(years, hybrid, 2030)
y2100 = _pick_year(years, hybrid, 2100)
drop_30_00 = 100.0 * (y2100 / y2030 - 1.0) if (
    np.isfinite(y2030) and y2030 > 0 and np.isfinite(y2100)
) else np.nan

print("\n================ CHINA-LOGIC STOCK MODEL SUMMARY ================")
print(f"[ARX fit] lambda={lam_arx}  MAPE={m_arx:.3f}%  R2={r2_arx:.4f}")
print(f"[Stock forecast] method={best_pack}")
print(f"Global scale: {scale if np.isfinite(scale) else 'N/A'}")
print(f"Forecast anchor scale at {HIS_END}: {forecast_anchor_scale if np.isfinite(forecast_anchor_scale) else 'N/A'}")
print(f"Future multiplier: anchor={_fmt(anchor_mult)}  decay={FUTURE_MULT_DECAY}  floor_frac={FUTURE_MULT_FLOOR_FRAC}")

print("\n--- History metrics (1990-2024) ---")
dw_tmp_mask = (years <= HIS_END) & np.isfinite(y_actual) & np.isfinite(hybrid)
dw_tmp = durbin_watson(y_actual[dw_tmp_mask] - hybrid[dw_tmp_mask]) if np.any(dw_tmp_mask) else np.nan
print(f"Hybrid final:      MAPE={_fmt(m_arx_hist['MAPE'])}%  R2={_fmt(m_arx_hist['R2'],4)}  DW={_fmt(dw_tmp,4)}")
print(f"Stock-only ref:    MAPE={_fmt(m_stock_hist['MAPE'])}%  R2={_fmt(m_stock_hist['R2'],4)}  (diagnostic only)")

print("\n--- Tail / drop (hybrid forecast) ---")
print(f"Tail slope 2033-2100: {_fmt(tail['slope_pct_per_year'])}%/y (info only)")
print(f"Tail R2 2033-2100: {_fmt(tail_r2, 4)}")
print(f"Drop 2030->2100: {_fmt(drop_30_00)}%  (info only)")

mask_resid = (years <= HIS_END) & np.isfinite(y_actual) & np.isfinite(hybrid)
resid_hist = (y_actual[mask_resid] - hybrid[mask_resid]) if np.any(mask_resid) else np.array([])

dw_hist = durbin_watson(resid_hist) if resid_hist.size > 0 else np.nan
ar1_hist = ar1_rho_ols(resid_hist) if resid_hist.size > 0 else np.nan

check_mape = bool(np.isfinite(m_arx_hist["MAPE"]) and (m_arx_hist["MAPE"] <= MAPE_THRESH_HIST))
check_r2 = bool(np.isfinite(m_arx_hist["R2"]) and (m_arx_hist["R2"] >= R2_THRESH_HIST))
check_drop = bool(np.isfinite(drop_30_00) and (drop_30_00 >= DROP_2030_2100_MIN))
check_tail_r2 = bool(np.isfinite(tail_r2) and (tail_r2 >= TAIL_R2_THRESH))

print("\n--- Constraint checks ---")
print(f"MAPE <= {MAPE_THRESH_HIST}: {check_mape}")
print(f"R2 >= {R2_THRESH_HIST}: {check_r2}")
print(f"Drop >= {DROP_2030_2100_MIN}: {check_drop}  (info only)")
print(f"Tail R2 >= {TAIL_R2_THRESH}: {check_tail_r2}  (info only)")


out = pd.DataFrame({
    "Year": years,
    "Actual_cement_1990_2024": y_actual,
    "ARX_fit_1990_2024": arx_fit,
    "StockDriven_pred_all_1990_2100": stock_pred_all,
    "Hybrid_final_1990_2100": hybrid,
    "Raw_StockDriven_forecast": stock_pred_raw_future,
    "Forecast_bridge_weight": forecast_bridge_weight,
    "Forecast_display_2024_actual_anchor": forecast_display,
    "Urban_pop": population_raw,
    "Area_pc": pc_area_raw,
    "Stock_m2": stock,
    "Inflow_m2": inflow,
    "Outflow_m2": outflow,
    "Baseline_B_tons": baseline_B_tons,
    "Baseline_B": baseline_B,
    "Historical_multiplier": multiplier,
    "Global_scale": scale,
    "Forecast_anchor_scale": forecast_anchor_scale,
})

phase_multiplier_table = pd.DataFrame(phase_rows)

diagnostics = pd.DataFrame([
    {"Block": "Hybrid_hist(1990-2024)", "Metric": "MAE", "Value": m_arx_hist["MAE"], "Threshold": np.nan, "Pass": np.nan},
    {"Block": "Hybrid_hist(1990-2024)", "Metric": "MAPE(%)", "Value": m_arx_hist["MAPE"], "Threshold": MAPE_THRESH_HIST, "Pass": check_mape},
    {"Block": "Hybrid_hist(1990-2024)", "Metric": "RMSE", "Value": m_arx_hist["RMSE"], "Threshold": np.nan, "Pass": np.nan},
    {"Block": "Hybrid_hist(1990-2024)", "Metric": "Bias", "Value": m_arx_hist["Bias"], "Threshold": np.nan, "Pass": np.nan},
    {"Block": "Hybrid_hist(1990-2024)", "Metric": "R2", "Value": m_arx_hist["R2"], "Threshold": R2_THRESH_HIST, "Pass": check_r2},

    {"Block": "Hybrid_residual(1990-2024)", "Metric": "Durbin-Watson", "Value": dw_hist, "Threshold": np.nan, "Pass": np.nan},
    {"Block": "Hybrid_residual(1990-2024)", "Metric": "AR1_rho(OLS)", "Value": ar1_hist, "Threshold": np.nan, "Pass": np.nan},

    {"Block": "StockOnly_hist(1990-2024)", "Metric": "MAE", "Value": m_stock_hist["MAE"], "Threshold": np.nan, "Pass": np.nan},
    {"Block": "StockOnly_hist(1990-2024)", "Metric": "MAPE(%)", "Value": m_stock_hist["MAPE"], "Threshold": np.nan, "Pass": np.nan},
    {"Block": "StockOnly_hist(1990-2024)", "Metric": "RMSE", "Value": m_stock_hist["RMSE"], "Threshold": np.nan, "Pass": np.nan},
    {"Block": "StockOnly_hist(1990-2024)", "Metric": "Bias", "Value": m_stock_hist["Bias"], "Threshold": np.nan, "Pass": np.nan},
    {"Block": "StockOnly_hist(1990-2024)", "Metric": "R2", "Value": m_stock_hist["R2"], "Threshold": np.nan, "Pass": np.nan},

    {"Block": "ChinaLogic_forecast", "Metric": "TailSlope_2033-2100(%/yr)", "Value": tail["slope_pct_per_year"], "Threshold": np.nan, "Pass": np.nan},
    {"Block": "ChinaLogic_forecast", "Metric": "Tail_R2_2033_2100", "Value": tail_r2, "Threshold": np.nan, "Pass": np.nan},
    {"Block": "ChinaLogic_forecast", "Metric": "Drop_2030_to_2100(%)", "Value": drop_30_00, "Threshold": np.nan, "Pass": np.nan},

    {"Block": "ARX_fit", "Metric": "best_lambda", "Value": lam_arx, "Threshold": np.nan, "Pass": np.nan},
    {"Block": "ARX_fit", "Metric": "train_MAPE(%)", "Value": m_arx, "Threshold": np.nan, "Pass": np.nan},
    {"Block": "ARX_fit", "Metric": "train_R2", "Value": r2_arx, "Threshold": np.nan, "Pass": np.nan},

    {"Block": "Stock_method", "Metric": "formula", "Value": str(best_pack), "Threshold": np.nan, "Pass": np.nan},
    {"Block": "Stock_calibration", "Metric": "alpha(intercept)", "Value": best_cal_info.get("alpha", np.nan), "Threshold": np.nan, "Pass": np.nan},
    {"Block": "Stock_calibration", "Metric": "beta(global_scale)", "Value": best_cal_info.get("beta", np.nan), "Threshold": np.nan, "Pass": np.nan},
    {"Block": "Forecast_splice", "Metric": "forecast_anchor_scale", "Value": forecast_anchor_scale, "Threshold": np.nan, "Pass": np.nan},
    {"Block": "Future_multiplier", "Metric": "anchor_multiplier", "Value": anchor_mult, "Threshold": np.nan, "Pass": np.nan},
    {"Block": "Future_multiplier", "Metric": "decay", "Value": FUTURE_MULT_DECAY, "Threshold": np.nan, "Pass": np.nan},
    {"Block": "Future_multiplier", "Metric": "floor_frac", "Value": FUTURE_MULT_FLOOR_FRAC, "Threshold": np.nan, "Pass": np.nan},
    {"Block": "Unit", "Metric": "cement_output_unit_scale", "Value": CEMENT_OUTPUT_UNIT_SCALE, "Threshold": np.nan, "Pass": np.nan},

    {"Block": "Global_scale", "Metric": "scale(all years)", "Value": scale, "Threshold": np.nan, "Pass": np.nan},
])

expected_life_trunc = float(np.sum(life * pdf_weib))
var_life_trunc = float(np.sum((life - expected_life_trunc) ** 2 * pdf_weib))
median_life = float(weibull_min.median(c=k_shape, scale=lambda_scale))
mean_life_theory = float(weibull_min.mean(c=k_shape, scale=lambda_scale))
std_life_theory = float(weibull_min.std(c=k_shape, scale=lambda_scale))

weibull_params = pd.DataFrame([
    {"Param": "k_shape", "Value": k_shape},
    {"Param": "lambda_scale", "Value": lambda_scale},
    {"Param": "max_life(truncation)", "Value": max_life},
    {"Param": "mean_life_theory", "Value": mean_life_theory},
    {"Param": "std_life_theory", "Value": std_life_theory},
    {"Param": "median_life_theory", "Value": median_life},
    {"Param": "expected_life_truncated(pdf_normalized)", "Value": expected_life_trunc},
    {"Param": "var_life_truncated(pdf_normalized)", "Value": var_life_trunc},
    {"Param": "pdf_sum_after_norm(should_be_1)", "Value": float(pdf_weib.sum())},
])

weibull_table = pd.DataFrame({
    "Age": life.astype(int),
    "pdf_truncated_normalized": pdf_weib,
    "cdf_theory": weibull_min.cdf(life, c=k_shape, scale=lambda_scale),
})
weibull_table["survival_theory"] = 1.0 - weibull_table["cdf_theory"]

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    out.to_excel(writer, sheet_name="Forecast", index=False)
    diagnostics.to_excel(writer, sheet_name="Diagnostics", index=False)
    phase_multiplier_table.to_excel(writer, sheet_name="Phase_Multipliers", index=False)
    weibull_params.to_excel(writer, sheet_name="Weibull_Params", index=False)
    weibull_table.to_excel(writer, sheet_name="Weibull_Table", index=False)

print(f"\n✅ Exported (with Diagnostics & Weibull): {OUT_XLSX}")


plt.figure(figsize=(13, 7))
plt.plot(years, forecast_display, linewidth=2.2, label="Hybrid final (ARX history + China-logic forecast)")
plt.plot(years, baseline_B, linewidth=1.3, alpha=0.65, label="Baseline B (same unit as input)")
plt.plot(years, arx_fit, linewidth=1.6, alpha=0.75, label="ARX one-step fit (hist)")
plt.scatter(years[np.isfinite(y_actual)], y_actual[np.isfinite(y_actual)], s=18, alpha=0.7, label="Actual (1990-2024)")
plt.axvline(HIS_END + 1, color="gray", linestyle="--", label="Forecast start")
plt.xlim(YEAR_START, FORECAST_END)
plt.xlabel("Year")
plt.ylabel("Cement production (same unit as input)")
plt.title("Russia Cement | ARX History Fit + Stock-driven China-style Forecast")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
