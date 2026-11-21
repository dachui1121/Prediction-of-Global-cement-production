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


DATA_PATH = str(_input_path('russia'))
SHEET_NAME = "Sheet1"

YEAR_START = 1990
HIS_END = 2024
FORECAST_END = 2100
FORECAST_START = HIS_END + 1


MAPE_THRESH_HIST = 12.0
R2_THRESH_HIST   = 0.80
DROP_2030_2100_MIN = -45.0


OUT_XLSX = os.path.join(str(_output_dir('russia', 'replacement')), 'Russia_DS_Replacement_2024ActualAnchor_2024_2030SmoothBridge_1990_2100.xlsx')


plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11


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
        y[i] = alpha*x[i] + (1-alpha)*y[i-1]
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

def _drop_pct(years_arr, yhat_arr, start_year, end_year):
    y_start = _pick_year(years_arr, yhat_arr, start_year)
    y_end = _pick_year(years_arr, yhat_arr, end_year)
    if np.isfinite(y_start) and y_start > 0 and np.isfinite(y_end):
        return 100.0 * (y_end / y_start - 1.0)
    return np.nan


df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, engine="openpyxl").sort_values("年份").reset_index(drop=True)
df.columns = [str(c).strip() for c in df.columns]


COL_YEAR = "年份"
COL_GDP  = "GDP（十亿美元）"
COL_POP  = "人口"
COL_URB  = "城镇化率"
COL_URBPOP = "城镇人口数"
COL_AREA = "预测人均住宅面积"
COL_Y    = "水泥产量"

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
d1998 = dyear(1998); d2009 = dyear(2009); d2014 = dyear(2014); d2020 = dyear(2020); d2022 = dyear(2022)

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
    if best_arx is None or (m < best_arx[0] - 1e-9) or (abs(m-best_arx[0])<=1e-9 and r2 > best_arx[1] + 1e-9):
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
pc_area_raw    = df[COL_AREA].to_numpy(float)

population = centered_ma(population_raw, 3)
pc_area    = centered_ma(pc_area_raw, 3)

n = len(years)


def cement_intensity_russia(y: int) -> float:
    return 0.15

intensity = np.array([cement_intensity_russia(int(y)) for y in years], float)


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

k_shape0      = _pick_scalar(df, ["Weibull_k","寿命形状参数","形状参数k"])
lambda_scale0 = _pick_scalar(df, ["Weibull_lambda","寿命尺度参数","尺度参数lambda","lambda"])
max_life0     = _pick_scalar(df, ["max_life","最大寿命","寿命截断"])

k_shape      = 2.5
lambda_scale = 87.35
max_life     = 150

life = np.arange(1, max_life+1)
pdf_weib = weibull_min.pdf(life, c=k_shape, scale=lambda_scale)
pdf_weib /= pdf_weib.sum()
expected_life = weibull_min.mean(c=k_shape, scale=lambda_scale)


stock = population * pc_area
if np.isfinite(stock[0]) and stock[0] > 0 and np.isfinite(np.median(stock[1:6])) and stock[0] < 0.6*np.median(stock[1:6]):
    stock += (np.median(stock[1:6]) - stock[0])

inflow  = np.zeros(n)
outflow = np.zeros(n)
steady_inflow = max(stock[0]/max(expected_life, 1.0), 1e-6)
hist_inflow = np.full(max_life, steady_inflow)

for t_i in range(n):
    past = np.concatenate([hist_inflow, inflow[:t_i]])
    use  = past[-max_life:]
    outflow[t_i] = (use[::-1] * pdf_weib[:len(use)]).sum()
    inflow[t_i]  = (stock[t_i]-stock[t_i-1]) + outflow[t_i] if t_i > 0 else outflow[t_i]


def retrofit_share_rus(y: int) -> float:
    if y <= 2024: return 0.015
    elif y <= 2035: return 0.015 + (0.07 - 0.015) * (y - 2024)/(2035 - 2024)
    elif y <= 2050: return 0.07  + (0.12 - 0.07)  * (y - 2035)/(2050 - 2035)
    else:           return 0.12  + (0.18 - 0.12)  * (min(y,2100) - 2050)/(2100 - 2050)

def infra_index_rus(y: int) -> float:
    if y <= 2000:  return 1.05
    elif y <= 2008: return 1.05 + (1.15-1.05)*(y-2000)/8
    elif y <= 2014: return 1.15 + (1.05-1.15)*(y-2008)/6
    elif y <= 2024: return 1.05 + (0.95-1.05)*(y-2014)/10
    elif y <= 2035: return 0.95 + (0.90-0.95)*(y-2024)/11
    else:           return 0.90 + (0.82-0.90) * (min(y,2100)-2035)/(2100-2035)

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

def run_stock_model(lam_ridge=60.0, alpha_inflow=0.45, alpha_final=0.35, tau_soft=4.5):
    inflow_s = ema(inflow, alpha=alpha_inflow)

    effective_new_area = np.zeros(n)
    infra_proxy = np.zeros(n)
    for i, yy in enumerate(years):
        if yy <= HIS_END:
            rfs = 1.0
        else:
            rfs = 1.0 - min(0.99, retrofit_share_rus(int(yy)))
        effective_new_area[i] = max(inflow_s[i], 0.0) * rfs
        infra_proxy[i]        = population[i] * infra_index_rus(int(yy))

    res_flow_tons = effective_new_area * intensity


    z = (years - years.mean()) / 10.0
    Z = np.column_stack([np.ones_like(z), z])
    d0914 = ((years >= 2009) & (years <= 2014)).astype(float)
    d9502 = ((years >= 1995) & (years <= 2002)).astype(float)

    X = np.column_stack([
        res_flow_tons[:,None]*Z,
        infra_proxy[:,None]*Z,
        Z,
        d0914[:,None], (d0914*z)[:,None],
        d9502[:,None], (d9502*z)[:,None],
    ])

    if np.any(actual_mask):
        X_fit = X[actual_mask]
        y_fit = actual_tons[actual_mask]
        y_fit_years = years[actual_mask]

        w = np.ones(len(y_fit))
        w[(y_fit_years>=2009)&(y_fit_years<=2014)] = 2.6
        w[(y_fit_years>=2015)&(y_fit_years<=2023)] = 1.4
        w[(y_fit_years>=1995)&(y_fit_years<=2002)] = 1.8
        w[(y_fit_years<1995)] = 1.0

        W = np.sqrt(w)[:,None]
        I = np.eye(X_fit.shape[1])

        beta = np.linalg.solve(
            (X_fit*W).T @ (X_fit*W) + lam_ridge*I,
            (X_fit*W).T @ (y_fit*W[:,0])
        )
        beta[4:6] = np.clip(beta[4:6], 0.0, None)
    else:
        beta = np.zeros(X.shape[1])

    yhat_raw = np.maximum(X @ beta, 0.0)
    final_demand = ema(yhat_raw, alpha=alpha_final)


    mask_elas = (years>=2010)&(years<=2024)&actual_mask
    A = np.column_stack([res_flow_tons[mask_elas], infra_proxy[mask_elas]]) if np.any(mask_elas) else np.zeros((0,2))
    b_y = final_demand[mask_elas] if np.any(mask_elas) else np.zeros((0,))
    w_learn = np.linalg.lstsq(A, b_y, rcond=None)[0] if (A.shape[0]>=3 and np.linalg.matrix_rank(A)>=2) else np.array([0.5,0.5])

    w_prior = np.array([0.75,0.25])
    w_mix   = 0.5*w_learn + 0.5*w_prior
    w_mix   = w_mix / max(w_mix.sum(), 1e-9)

    D = np.clip(w_mix[0]*res_flow_tons + w_mix[1]*infra_proxy, 1e-9, None)

    if np.sum(mask_elas) >= 3:
        x_log = np.log(D[mask_elas])
        y_log = np.log(final_demand[mask_elas])
        e, _ = np.linalg.lstsq(np.column_stack([x_log, np.ones_like(x_log)]), y_log, rcond=None)[0]
        e = float(np.clip(e, 0.6, 1.4))
    else:
        e = 1.0

    y_2024 = float(final_demand[years==2024][0]) if np.any(years==2024) else float(final_demand[-1])
    D_2024 = float(D[years==2024][0]) if np.any(years==2024) else float(D[-1])
    target_dyn = y_2024 * (D / max(D_2024, 1e-9)) ** e

    final_soft = final_demand.copy()
    start_soft = 2025
    for i, yy in enumerate(years):
        if yy >= start_soft:
            decay = np.exp(-(yy - start_soft)/tau_soft)
            final_soft[i] = target_dyn[i] + (final_soft[i] - target_dyn[i]) * decay

    final_soft = np.maximum(final_soft, 0.0)
    return final_soft


lam_grid = [10, 20, 40, 60, 80, 120, 180]
alpha_inflow_grid = [0.35, 0.45, 0.55]
alpha_final_grid  = [0.25, 0.30, 0.35, 0.40]
tau_grid          = [3.0, 4.5, 6.0, 8.0]

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
                pred_cal, cal_info = calibrate_pred(pred, actual_tons, years, HIS_END)
                m_hist = _metrics(pred_cal, actual_tons, years, a=None, b=HIS_END)

                score = m_hist["MAPE"] if np.isfinite(m_hist["MAPE"]) else 1e9
                if best_pack is None or score < best_score - 1e-6 or (abs(score-best_score)<=1e-6 and m_hist["R2"] > best_hist["R2"]):
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


y_arx_2024 = _pick_year(years, hybrid, 2024)
y_stock_2024 = _pick_year(years, stock_pred_all, 2024)
if y_stock_2024 is not None and np.isfinite(y_stock_2024) and y_stock_2024 > 0:
    scale = y_arx_2024 / y_stock_2024
    if np.isfinite(scale) and scale > 0:
        hybrid[mask_fc] *= scale


PARAM_K_SHAPE = 2.5
PARAM_LAMBDA_SCALE = 87.35
PARAM_MAX_LIFE = 150
PARAM_RES_INTENSITY = 0.15
PARAM_TRANSITION_RATE = 4.0
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
    output_unit_scale=1.0,
    kernel_method="pdf",
)


FORECAST_BRIDGE_END = 2030

China_pred_raw = China_pred.copy()
stock_pred_all = China_pred_raw.copy()

if np.any(years == HIS_END):
    idx_2024_bridge = int(np.where(years == HIS_END)[0][0])
    if np.isfinite(y_actual[idx_2024_bridge]) and y_actual[idx_2024_bridge] > 0:
        bridge_anchor_2024 = float(y_actual[idx_2024_bridge])
    else:
        bridge_anchor_2024 = float(hybrid[idx_2024_bridge])
else:
    idx_2024_bridge = None
    bridge_anchor_2024 = np.nan


China_splice_scale = 1.0


hybrid[mask_fc] = China_pred_raw[mask_fc]

forecast_bridge_weight = np.ones(len(years), dtype=float)
forecast_bridge_weight[years <= HIS_END] = 0.0

bridge_mask = (years > HIS_END) & (years <= FORECAST_BRIDGE_END)
if np.any(bridge_mask) and np.isfinite(bridge_anchor_2024):
    x_bridge = (
        years[bridge_mask].astype(float) - float(HIS_END)
    ) / max(float(FORECAST_BRIDGE_END - HIS_END), 1.0)

    x_bridge = np.clip(x_bridge, 0.0, 1.0)
    w_bridge = x_bridge**2 * (3.0 - 2.0*x_bridge)
    forecast_bridge_weight[bridge_mask] = w_bridge

    hybrid[bridge_mask] = (
        (1.0 - w_bridge) * bridge_anchor_2024
        + w_bridge * China_pred_raw[bridge_mask]
    )


if idx_2024_bridge is not None and np.isfinite(bridge_anchor_2024):
    hybrid[idx_2024_bridge] = bridge_anchor_2024


forecast_display = np.full(len(years), np.nan, dtype=float)
forecast_display[years >= HIS_END] = hybrid[years >= HIS_END]


m_arx_hist = _metrics(hybrid, y_actual, years, a=YEAR_START, b=HIS_END)


m_stock_hist = _metrics(stock_pred_all, y_actual, years, a=YEAR_START, b=HIS_END)


hist_resid_mask = (years <= HIS_END) & np.isfinite(y_actual) & np.isfinite(hybrid)
hist_resid = y_actual[hist_resid_mask] - hybrid[hist_resid_mask]
dw_hist = durbin_watson(hist_resid)
ar1_hist = ar1_rho_ols(hist_resid)


tail_total = _tail_slope(years, hybrid, FORECAST_START, FORECAST_END)
drop_total = _drop_pct(years, hybrid, FORECAST_START, FORECAST_END)
y_fc_start = _pick_year(years, hybrid, FORECAST_START)
y2030 = _pick_year(years, hybrid, 2030)
y2100 = _pick_year(years, hybrid, 2100)


legacy_tail = _tail_slope(years, hybrid, 2033, 2100)
drop_30_00 = _drop_pct(years, hybrid, 2030, 2100)

print("\n================ HYBRID MODEL SUMMARY ================")
print(f"[ARX fit] lambda={lam_arx}  MAPE={m_arx:.3f}%  R2={r2_arx:.4f}")
print(f"[Stock forecast] best(lam,ai,af,tau)={best_pack}  calib(alpha,beta)=({best_cal_info['alpha']:.3f},{best_cal_info['beta']:.3f})")
print(f"Hybrid splice scale (2025+): {scale if np.isfinite(scale) else 'N/A'}")

print("\n--- History metrics (1990-2024) ---")
print(f"Hybrid (ARX fit):  MAPE={_fmt(m_arx_hist['MAPE'])}%  R2={_fmt(m_arx_hist['R2'],4)}  DW={_fmt(dw_hist,4)}  AR1={_fmt(ar1_hist,4)}")
print(f"Stock-only ref:    MAPE={_fmt(m_stock_hist['MAPE'])}%  R2={_fmt(m_stock_hist['R2'],4)}")

print("\n--- Forecast rationality metrics (2025-2100, total forecast span) ---")
print(f"Tail slope ({FORECAST_START}-{FORECAST_END}): {_fmt(tail_total['slope_pct_per_year'])}%/y")
print(f"Tail R2    ({FORECAST_START}-{FORECAST_END}): {_fmt(tail_total['R2'],4)}")
print(f"Drop       ({FORECAST_START}->{FORECAST_END}): {_fmt(drop_total)}%")

print("\n--- Legacy tail / drop (for reference) ---")
print(f"Tail slope 2033-2100: {_fmt(legacy_tail['slope_pct_per_year'])}%/y")
print(f"Tail R2    2033-2100: {_fmt(legacy_tail['R2'],4)}")
print(f"Drop 2030->2100: {_fmt(drop_30_00)}%  (>= {DROP_2030_2100_MIN} desired)")


out = pd.DataFrame({
    "Year": years,
    "Actual_cement_1990_2024": y_actual,
    "ARX_fit_1990_2024": arx_fit,
    "StockDriven_pred_all_1990_2100": stock_pred_all,
    "Raw_Replacement_StockDriven": China_pred_raw,
    "Forecast_bridge_weight": forecast_bridge_weight,
    "Hybrid_final_1990_2100": hybrid,
    "Forecast_display_2024_actual_anchor": forecast_display,
    "Urban_pop": population_raw,
    "Area_pc": pc_area_raw
})

metrics_table = pd.DataFrame([
    {"Category": "History", "Metric": "Hybrid_MAPE_1990_2024_pct", "Value": m_arx_hist["MAPE"]},
    {"Category": "History", "Metric": "Hybrid_R2_1990_2024", "Value": m_arx_hist["R2"]},
    {"Category": "History", "Metric": "Hybrid_DW_1990_2024", "Value": dw_hist},
    {"Category": "History", "Metric": "Hybrid_AR1_1990_2024", "Value": ar1_hist},
    {"Category": "History", "Metric": "Stock_MAPE_1990_2024_pct", "Value": m_stock_hist["MAPE"]},
    {"Category": "History", "Metric": "Stock_R2_1990_2024", "Value": m_stock_hist["R2"]},

    {"Category": "Forecast_Total", "Metric": f"Tail_Slope_pct_per_year_{FORECAST_START}_{FORECAST_END}", "Value": tail_total["slope_pct_per_year"]},
    {"Category": "Forecast_Total", "Metric": f"Tail_R2_{FORECAST_START}_{FORECAST_END}", "Value": tail_total["R2"]},
    {"Category": "Forecast_Total", "Metric": f"Drop_pct_{FORECAST_START}_to_{FORECAST_END}", "Value": drop_total},
    {"Category": "Forecast_Total", "Metric": f"Forecast_Start_Value_{FORECAST_START}", "Value": y_fc_start},
    {"Category": "Forecast_Total", "Metric": f"Forecast_End_Value_{FORECAST_END}", "Value": y2100},

    {"Category": "Forecast_Legacy", "Metric": "Tail_Slope_pct_per_year_2033_2100", "Value": legacy_tail["slope_pct_per_year"]},
    {"Category": "Forecast_Legacy", "Metric": "Tail_R2_2033_2100", "Value": legacy_tail["R2"]},
    {"Category": "Forecast_Legacy", "Metric": "Drop_pct_2030_to_2100", "Value": drop_30_00},
    {"Category": "Forecast_Legacy", "Metric": "Value_2030", "Value": y2030},
    {"Category": "Forecast_Legacy", "Metric": "Value_2100", "Value": y2100},

    {"Category": "Model", "Metric": "ARX_lambda", "Value": lam_arx},
    {"Category": "Model", "Metric": "Stock_best_lam", "Value": best_pack[0]},
    {"Category": "Model", "Metric": "Stock_best_alpha_inflow", "Value": best_pack[1]},
    {"Category": "Model", "Metric": "Stock_best_alpha_final", "Value": best_pack[2]},
    {"Category": "Model", "Metric": "Stock_best_tau_soft", "Value": best_pack[3]},
    {"Category": "Model", "Metric": "Calibration_alpha", "Value": best_cal_info["alpha"]},
    {"Category": "Model", "Metric": "Calibration_beta", "Value": best_cal_info["beta"]},
    {"Category": "Model", "Metric": "Hybrid_splice_scale_2025plus", "Value": scale},
])

forecast_summary = pd.DataFrame([
    {
        "Forecast_Start": FORECAST_START,
        "Forecast_End": FORECAST_END,
        "Tail_Slope_pct_per_year": tail_total["slope_pct_per_year"],
        "Tail_R2": tail_total["R2"],
        "Drop_pct": drop_total,
        f"Value_{FORECAST_START}": y_fc_start,
        f"Value_{FORECAST_END}": y2100,
    }
])

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    out.to_excel(writer, sheet_name="Forecast_Data", index=False)
    metrics_table.to_excel(writer, sheet_name="Rationality_Check", index=False)
    forecast_summary.to_excel(writer, sheet_name="Forecast_Tail_Summary", index=False)

print(f"\n✅ Exported: {OUT_XLSX}")


plt.figure(figsize=(13,7))
plt.plot(
    years,
    forecast_display,
    linewidth=2.2,
    label="Replacement forecast (2024 actual anchor -> 2030 raw Stock-driven)"
)
plt.plot(
    years,
    stock_pred_all,
    linewidth=1.6,
    alpha=0.60,
    linestyle="--",
    label="Raw replacement Stock-driven reference"
)
plt.plot(
    years,
    arx_fit,
    linewidth=1.6,
    alpha=0.75,
    label="ARX one-step fit (hist)"
)
plt.scatter(years[np.isfinite(y_actual)], y_actual[np.isfinite(y_actual)], s=18, alpha=0.7, label="Actual (1990-2024)")
plt.axvline(HIS_END, color="gray", linestyle="--", label="2024 actual anchor")
plt.xlim(YEAR_START, FORECAST_END)
plt.xlabel("Year")
plt.ylabel("Cement production (same unit as input)")
plt.title("Russia DS Replacement | 2024 actual anchor + 2024-2030 smooth bridge")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
