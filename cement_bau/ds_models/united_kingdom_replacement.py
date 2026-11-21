from cement_bau.runtime import input_path as _input_path, output_dir as _output_dir
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min


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

    B_local = inflow_local * cement_rate_arr * transition_factor_arr / max(float(output_unit_scale), 1e-12)
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
from scipy.optimize import least_squares


DATA_PATH = str(_input_path('united_kingdom'))
SHEET_NAME = "Sheet1"

YEAR_START = 1990
HIS_END = 2024
FORECAST_END = 2100


MAPE_THRESH_HIST = 12.0
R2_THRESH_HIST   = 0.80
DROP_2030_2100_MIN = -45.0


BASE_DIR = str(_output_dir('united_kingdom', 'replacement'))
OUT_XLSX = os.path.join(BASE_DIR, "UK_DS_Replacement_2024ActualAnchor_2024_2030SmoothBridge_1990_2100.xlsx")


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

intensity = np.array([cement_intensity_uk(int(y)) for y in years], float)


k_shape = 2.95
lambda_scale = 70.82
max_life = 120

life = np.arange(1, max_life + 1)
pdf_weib = weibull_min.pdf(life, c=k_shape, scale=lambda_scale)
pdf_weib = np.nan_to_num(pdf_weib)
pdf_weib /= pdf_weib.sum() if pdf_weib.sum() > 0 else 1.0
expected_life = float(weibull_min.mean(c=k_shape, scale=lambda_scale))

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
    outflow[t_i] = float(np.sum(use[::-1] * pdf_weib[:len(use)]))
    inflow[t_i]  = (stock[t_i] - stock[t_i-1]) + outflow[t_i] if t_i > 0 else outflow[t_i]


def retrofit_share_uk(y: int) -> float:
    if y <= 2024: return 0.03
    elif y <= 2035: return 0.03 + (0.10 - 0.03) * (y - 2024) / (2035 - 2024)
    elif y <= 2050: return 0.10 + (0.16 - 0.10) * (y - 2035) / (2050 - 2035)
    else: return 0.16 + (0.22 - 0.16) * (min(y, 2100) - 2050) / (2100 - 2050)


def infra_index_uk(y: int) -> float:
    if y <= 2007: return 1.00
    elif y <= 2012: return 1.00 + (0.92 - 1.00) * (y - 2007) / 5
    elif y <= 2019: return 0.92 + (0.95 - 0.92) * (y - 2012) / 7
    elif y <= 2024: return 0.95 + (0.97 - 0.95) * (y - 2019) / 5
    elif y <= 2050: return 0.97 + (0.93 - 0.97) * (y - 2024) / (2050 - 2024)
    else: return 0.93 + (0.88 - 0.93) * (min(y, 2100) - 2050) / (2100 - 2050)

def calibrate_pred(pred, actual, years_arr, his_end):
    m = (years_arr <= his_end) & np.isfinite(pred) & np.isfinite(actual)
    if np.sum(m) < 5:
        return pred.copy(), dict(alpha=np.nan, beta=np.nan)

    p = pred[m].astype(float)
    yv = actual[m].astype(float)
    den = np.maximum(np.abs(yv), max(1e-6, _pct(np.abs(yv), 20)))

    def res_ab(ab):
        a, b = ab
        yhat = a + b * p
        return (yhat - yv) / den

    A = np.column_stack([np.ones_like(p), p])
    ab0 = np.linalg.lstsq(A, yv, rcond=None)[0]
    ab0[1] = max(ab0[1], 0.0)

    lb = np.array([-np.inf, 0.0], float)
    ub = np.array([ np.inf, np.inf], float)
    sol = least_squares(res_ab, ab0, bounds=(lb, ub), loss="soft_l1", f_scale=1.0, max_nfev=50000)
    a_hat, b_hat = sol.x

    pred_cal = a_hat + b_hat * pred
    pred_cal = np.maximum(pred_cal, 0.0)
    return pred_cal, dict(alpha=float(a_hat), beta=float(b_hat))

def run_stock_model(lam_ridge=60.0, alpha_inflow=0.45, alpha_final=0.35, tau_soft=5.0):
    inflow_s = ema(inflow, alpha=alpha_inflow)

    effective_new_area = np.zeros(n)
    infra_proxy = np.zeros(n)
    for i, yy in enumerate(years):
        rfs = 1.0 if yy <= HIS_END else (1.0 - min(0.99, retrofit_share_uk(int(yy))))
        effective_new_area[i] = max(inflow_s[i], 0.0) * rfs
        infra_proxy[i]        = population[i] * infra_index_uk(int(yy))

    res_flow = effective_new_area * intensity

    z = (years - years.mean()) / 10.0
    Z = np.column_stack([np.ones_like(z), z])

    d0809 = ((years >= 2008) & (years <= 2009)).astype(float)
    d2020 = (years == 2020).astype(float)
    d2022 = (years == 2022).astype(float)

    X = np.column_stack([
        res_flow[:, None] * Z,
        infra_proxy[:, None] * Z,
        Z,
        d0809[:, None], (d0809 * z)[:, None],
        d2020[:, None],
        d2022[:, None],
    ])

    mask_fit = (years <= HIS_END) & np.isfinite(y_actual)
    X_fit = X[mask_fit]
    y_fit = y_actual[mask_fit]

    I = np.eye(X_fit.shape[1])
    beta = np.linalg.solve(X_fit.T @ X_fit + lam_ridge * I, X_fit.T @ y_fit)

    yhat_raw = np.maximum(X @ beta, 0.0)
    final_demand = ema(yhat_raw, alpha=alpha_final)


    mask_elas = (years >= 2010) & (years <= HIS_END) & np.isfinite(y_actual)
    if np.sum(mask_elas) >= 3:
        A = np.column_stack([res_flow[mask_elas], infra_proxy[mask_elas]])
        w_learn = np.linalg.lstsq(A, final_demand[mask_elas], rcond=None)[0]
    else:
        w_learn = np.array([0.7, 0.3], float)

    w_learn = np.maximum(w_learn, 0.0)
    w_learn = w_learn / max(w_learn.sum(), 1e-9)
    D = np.clip(w_learn[0]*res_flow + w_learn[1]*infra_proxy, 1e-9, None)

    if np.sum(mask_elas) >= 3:
        x_log = np.log(D[mask_elas])
        y_log = np.log(np.maximum(final_demand[mask_elas], 1e-12))
        e, _ = np.linalg.lstsq(np.column_stack([x_log, np.ones_like(x_log)]), y_log, rcond=None)[0]
        e = float(np.clip(e, 0.6, 1.4))
    else:
        e = 1.0

    y_2024 = float(final_demand[years == HIS_END][0]) if np.any(years == HIS_END) else float(final_demand[np.isfinite(final_demand)][-1])
    D_2024 = float(D[years == HIS_END][0]) if np.any(years == HIS_END) else float(D[np.isfinite(D)][-1])

    target_dyn = y_2024 * (D / max(D_2024, 1e-12)) ** e

    out = final_demand.copy()
    start_soft = HIS_END + 1
    for i, yy in enumerate(years):
        if yy >= start_soft:
            decay = np.exp(-(yy - start_soft) / tau_soft)
            out[i] = target_dyn[i] + (out[i] - target_dyn[i]) * decay

    return np.maximum(out, 0.0)


lam_grid = [10, 20, 40, 60, 120]
alpha_inflow_grid = [0.35, 0.45, 0.55]
alpha_final_grid  = [0.25, 0.35, 0.45]
tau_grid          = [3.0, 5.0, 8.0]

best_pack = None
best_score = np.inf
best_pred_cal = None
best_cal_info = None
best_hist = None

for lam in lam_grid:
    for ai in alpha_inflow_grid:
        for af in alpha_final_grid:
            for tau in tau_grid:
                pred = run_stock_model(lam_ridge=float(lam), alpha_inflow=float(ai), alpha_final=float(af), tau_soft=float(tau))
                pred_cal, cal_info = calibrate_pred(pred, y_actual, years, HIS_END)
                m_hist = _metrics(pred_cal, y_actual, years, a=YEAR_START, b=HIS_END)

                score = m_hist["MAPE"] if np.isfinite(m_hist["MAPE"]) else 1e9
                if best_pack is None or score < best_score - 1e-6 or (abs(score-best_score) <= 1e-6 and m_hist["R2"] > best_hist["R2"]):
                    best_pack = (lam, ai, af, tau)
                    best_score = score
                    best_pred_cal = pred_cal
                    best_cal_info = cal_info
                    best_hist = m_hist

stock_pred_all = best_pred_cal


hybrid = np.full_like(stock_pred_all, np.nan, float)

mask_hist = (years <= HIS_END)
hybrid[mask_hist] = arx_fit[mask_hist]

miss = mask_hist & (~np.isfinite(hybrid))
hybrid[miss] = np.where(np.isfinite(y_actual[miss]), y_actual[miss], stock_pred_all[miss])

mask_fc = (years >= HIS_END + 1)
hybrid[mask_fc] = stock_pred_all[mask_fc]


y_arx_2024 = _pick_year(years, hybrid, HIS_END)
y_stock_2024 = _pick_year(years, stock_pred_all, HIS_END)
if y_stock_2024 and np.isfinite(y_stock_2024) and y_stock_2024 > 0:
    splice_scale = y_arx_2024 / y_stock_2024
    if np.isfinite(splice_scale) and splice_scale > 0:
        hybrid[mask_fc] *= splice_scale


PARAM_K_SHAPE = 2.95
PARAM_LAMBDA_SCALE = 70.82
PARAM_MAX_LIFE = 120
PARAM_RES_INTENSITY = 0.12
PARAM_TRANSITION_RATE = 3.0
_china_ci = np.full_like(years, PARAM_RES_INTENSITY, dtype=float)
_china_transition = np.full_like(years, 1.0 + PARAM_TRANSITION_RATE, dtype=float)
(
    China_pred,
    China_B,
    China_inflow,
    China_outflow,
    China_multiplier,
    China_global_scale,
    China_phase_multipliers,
) = _china_logic_phase_forecast(
    years, stock, y_actual, YEAR_START, HIS_END,
    PARAM_K_SHAPE, PARAM_LAMBDA_SCALE, PARAM_MAX_LIFE,
    _china_ci, _china_transition,
    output_unit_scale=1e8,
    kernel_method="pdf",
)

China_splice_scale = 1.0
stock_pred_all = China_pred.copy()
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

China_splice_scale = 1.0


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
    "Stock_best_lam": best_pack[0] if best_pack else np.nan,
    "Stock_best_alpha_inflow": best_pack[1] if best_pack else np.nan,
    "Stock_best_alpha_final": best_pack[2] if best_pack else np.nan,
    "Stock_best_tau_soft": best_pack[3] if best_pack else np.nan,
    "Stock_calib_alpha": best_cal_info["alpha"] if best_cal_info else np.nan,
    "Stock_calib_beta": best_cal_info["beta"] if best_cal_info else np.nan,

    "Splice_scale_2025plus": splice_scale,

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
})

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
    series_df.to_excel(w, index=False, sheet_name="Series")
    reason_df.to_excel(w, index=False, sheet_name="Reasonableness")
    weibull_df.to_excel(w, index=False, sheet_name="Weibull_Params")

print(f"\n✅ Exported: {OUT_XLSX}")


print("\n================ UK HYBRID SUMMARY ================")
print(f"[ARX] best_lambda={lam_arx} | in-sample MAPE={_fmt(m_arx_in)}% R2={_fmt(r2_arx_in,4)}")
print(f"[Stock] best(lam,ai,af,tau)={best_pack} | calib(alpha,beta)=({_fmt(best_cal_info['alpha'])},{_fmt(best_cal_info['beta'])})")
print(f"[Hybrid hist] MAPE={_fmt(m_hybrid_hist['MAPE'])}% R2={_fmt(m_hybrid_hist['R2'],4)} DW={_fmt(DW_val,4)} AR1={_fmt(AR1_val,4)}")
print(f"[Tail] 2033-2100 slope={_fmt(tail['slope_pct_per_year'])}%/y | Drop2030->2100={_fmt(drop_30_00)}%")
print(f"[Weibull] k={k_shape}  lambda={lambda_scale}  max_life={max_life}  E[life]={expected_life:.2f}")


plt.figure(figsize=(13, 7))
plt.plot(years, hybrid, linewidth=2.2, label="Hybrid final (ARX fit -> Stock-driven forecast)")
plt.plot(years, stock_pred_all, linewidth=1.6, alpha=0.75, label="Stock-driven (calibrated) reference")
plt.plot(years, arx_fit, linewidth=1.6, alpha=0.75, label="ARX one-step fit (hist)")
plt.scatter(years[np.isfinite(y_actual)], y_actual[np.isfinite(y_actual)], s=18, alpha=0.7, label="Actual (1990-2024)")
plt.axvline(HIS_END + 1, color="gray", linestyle="--", label="Forecast start")
plt.xlim(YEAR_START, FORECAST_END)
plt.xlabel("Year"); plt.ylabel("Cement production (same unit as input)")
plt.title("UK Cement | Hybrid: ARX fit (1990-2024) + Stock-driven forecast (2025-2100)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
