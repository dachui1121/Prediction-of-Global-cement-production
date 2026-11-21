from cement_bau.runtime import input_path as _input_path, output_dir as _output_dir
import os
import numpy as np
import pandas as pd
from pathlib import Path

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

try:
    from scipy.optimize import curve_fit
except Exception:
    curve_fit = None


if HAS_MATPLOTLIB:
    plt.rcParams["font.family"] = ["Times New Roman"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["legend.loc"] = "best"


DATA_PATH = str(_input_path('japan'))
SHEET_NAME = "Sheet1"
OUTPUT_DIR = str(_output_dir('japan', 'replacement'))

def resolve_data_path(*args):
    return DATA_PATH

def resolve_output_dir(*args):
    return str(_output_dir('japan', 'replacement'))

OUTPUT_BASE_DIR = str(_output_dir('japan', 'replacement'))
OUTPUT_XLSX = os.path.join(OUTPUT_BASE_DIR, "日本_DS_Replacement_2024实际锚定_2024_2050统一平滑衔接_2100端点修正.xlsx")
OUTPUT_PNG = os.path.join(OUTPUT_BASE_DIR, "日本_DS_Replacement_2024实际锚定_2024_2050统一平滑衔接_2100端点修正.png")

HIS_END = 2024
FORECAST_END = 2100
FIT_START = 1960
FIT_END = 2024


MAPE_THRESH_ALL = 10.0
DROP_MIN_2030_2050 = -45.0
DW_LOW = 1.5
DW_HIGH = 2.5


USE_HIST_DW_CORRECTION = True
HIST_DW_RHO_CORRECTION = 0.04
AUTO_TUNE_HIST_DW_RHO = True
USE_DS_FUTURE_GUIDE = False
DS_FUTURE_GUIDE_BLEND = 0.80


def centered_ma(x, win=3):
    x = np.asarray(x, dtype=float)
    if win <= 1:
        return x.copy()
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(win) / win
    return np.convolve(xpad, ker, mode="valid")

def ema(x, alpha):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        xi = x[i] if np.isfinite(x[i]) else y[i - 1]
        y[i] = alpha * xi + (1 - alpha) * y[i - 1]
    return y

def ema_from_index(x, alpha, start_idx):
    x = np.asarray(x, dtype=float).copy()
    y = x.copy()
    if len(x) == 0 or start_idx >= len(x):
        return y
    if start_idx <= 0:
        return ema(x, alpha)
    y[start_idx] = x[start_idx]
    for i in range(start_idx + 1, len(x)):
        xi = x[i] if np.isfinite(x[i]) else y[i - 1]
        y[i] = alpha * xi + (1 - alpha) * y[i - 1]
    return y

def _pct(arr, p):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    a.sort()
    k = (len(a) - 1) * p / 100.0
    f, c = int(np.floor(k)), int(np.ceil(k))
    if f == c:
        return float(a[f])
    return float(a[f] + (k - f) * (a[c] - a[f]))

def _metrics(yhat, y, years_arr, a=None, b=None):
    m = np.isfinite(y) & np.isfinite(yhat)
    if a is not None:
        m &= (years_arr >= a)
    if b is not None:
        m &= (years_arr <= b)
    if not np.any(m):
        return dict(MAE=np.nan, MAPE=np.nan, RMSE=np.nan, Bias=np.nan, R2=np.nan)

    yt = yhat[m]
    yu = y[m]

    eps_den = max(1e-6, _pct(np.abs(yu), 20))
    mae = float(np.mean(np.abs(yt - yu)))
    mape = float(np.mean(np.abs(yt - yu) / np.maximum(np.abs(yu), eps_den))) * 100.0
    rmse = float(np.sqrt(np.mean((yt - yu) ** 2)))
    bias = float(np.mean(yt - yu))

    ybar = float(np.mean(yu))
    ss_tot = float(np.sum((yu - ybar) ** 2))
    ss_res = float(np.sum((yu - yt) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return dict(MAE=mae, MAPE=mape, RMSE=rmse, Bias=bias, R2=r2)

def _tail_slope(years_arr, yhat_arr, start=2033, end=2050):
    m = (years_arr >= start) & (years_arr <= end)
    if not np.any(m):
        return dict(slope_ton_per_year=np.nan, slope_pct_per_year=np.nan, R2=np.nan)
    x = years_arr[m].astype(float)
    yh = yhat_arr[m].astype(float)

    x_mean, y_mean = np.mean(x), np.mean(yh)
    sxx = np.sum((x - x_mean) ** 2)
    if sxx <= 0:
        k, b = 0.0, y_mean
    else:
        k = np.sum((x - x_mean) * (yh - y_mean)) / sxx
        b = y_mean - k * x_mean

    slope_pct = 100.0 * k / max(np.mean(yh), 1e-9)

    ybar = np.mean(yh)
    ss_tot = np.sum((yh - ybar) ** 2)
    ss_res = np.sum((yh - (k * x + b)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return dict(slope_ton_per_year=float(k), slope_pct_per_year=float(slope_pct), R2=float(r2))

def _pick_year(y_arr, val_arr, year):
    m = (y_arr == year)
    if np.any(m):
        return float(val_arr[m][0])
    return np.nan

def weibull_pdf_and_mean(life, k, lam):
    life = np.asarray(life, dtype=float)
    if weibull_min is not None:
        pdf = weibull_min.pdf(life, c=k, scale=lam)
        mean_life = float(weibull_min.mean(c=k, scale=lam))
    else:
        x = np.clip(life, 1e-12, None)
        pdf = (k / lam) * ((x / lam) ** (k - 1.0)) * np.exp(-((x / lam) ** k))
        mean_life = float(np.sum(life * pdf) / max(np.sum(pdf), 1e-12))
    pdf = np.nan_to_num(pdf)
    pdf = pdf / max(pdf.sum(), 1e-12)
    return pdf, mean_life


def durbin_watson(resid: np.ndarray) -> float:
    r = np.asarray(resid, float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return np.nan
    dr = np.diff(r)
    den = float(np.dot(r, r))
    return float(np.dot(dr, dr) / den) if den > 0 else np.nan

def ar1_rho_ols(resid: np.ndarray) -> float:
    r = np.asarray(resid, float)
    r = r[np.isfinite(r)]
    if r.size < 4:
        return np.nan
    e0, e1 = r[:-1], r[1:]
    den = float(np.dot(e0, e0))
    return float(np.dot(e0, e1) / den) if den > 0 else np.nan

def apply_historical_dw_correction(y_pred, y_true, fit_mask_arr, rho=0.04):
    corrected = np.asarray(y_pred, float).copy()
    y_true = np.asarray(y_true, float)
    fit_mask_arr = np.asarray(fit_mask_arr, bool)

    prev_resid = 0.0
    for i in range(len(corrected)):
        if fit_mask_arr[i] and np.isfinite(y_true[i]) and np.isfinite(y_pred[i]):
            corrected[i] = float(y_pred[i] + rho * prev_resid)
            prev_resid = float(y_true[i] - y_pred[i])
    return corrected

def tune_historical_dw_rho(y_pred, y_true, years_arr, fit_mask_arr, rho_grid):
    base_metrics = _metrics(y_pred, y_true, years_arr, a=FIT_START, b=FIT_END)
    best = None
    rows = []

    for rho in rho_grid:
        y_try = apply_historical_dw_correction(y_pred, y_true, fit_mask_arr, rho=float(rho))

        anchor_2024_mask = fit_mask_arr & (years_arr == FIT_END) & np.isfinite(y_true)
        y_try[anchor_2024_mask] = y_true[anchor_2024_mask]
        resid_try = y_true[fit_mask_arr] - y_try[fit_mask_arr]
        dw_try = durbin_watson(resid_try)
        met_try = _metrics(y_try, y_true, years_arr, a=FIT_START, b=FIT_END)

        if not np.isfinite(dw_try) or not np.isfinite(met_try.get("MAPE", np.nan)):
            continue

        in_band = DW_LOW <= dw_try <= DW_HIGH
        stable_band = (DW_LOW + 0.05) <= dw_try <= (DW_HIGH - 0.05)
        mape_penalty = max(0.0, met_try["MAPE"] - MAPE_THRESH_ALL) * 100.0

        score = (
            (0.0 if stable_band else (100.0 if in_band else 1000.0) * min(abs(dw_try - DW_LOW), abs(dw_try - DW_HIGH)))
            + abs(float(rho))
            + 0.15 * max(0.0, met_try["MAPE"] - base_metrics["MAPE"])
            + mape_penalty
        )

        row = {
            "rho": float(rho),
            "DW": float(dw_try),
            "MAPE": float(met_try["MAPE"]),
            "R2": float(met_try["R2"]),
            "RMSE": float(met_try["RMSE"]),
            "score": float(score),
            "in_band": bool(in_band),
            "stable_band": bool(stable_band),
        }
        rows.append(row)
        if best is None or row["score"] < best["score"]:
            best = row

    if best is None:
        return float(HIST_DW_RHO_CORRECTION), y_pred.copy(), pd.DataFrame(rows)

    y_best = apply_historical_dw_correction(y_pred, y_true, fit_mask_arr, rho=best["rho"])
    return best["rho"], y_best, pd.DataFrame(rows)


def ridge_closed_form(X, y, lam):
    I = np.eye(X.shape[1])
    return np.linalg.solve(X.T @ X + lam * I, X.T @ y)


def fit_ar_p_with_ridge(resid, p, lam=1e-6):


    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n <= p + 2:
        return None

    Y = r[p:]
    X = [np.ones(n - p)]
    for lag in range(1, p + 1):
        X.append(r[p - lag:n - lag])
    X = np.column_stack(X)

    I = np.eye(X.shape[1])
    I[0, 0] = 0.0
    beta = np.linalg.solve(X.T @ X + lam * I, X.T @ Y)

    c = beta[0]
    phi = beta[1:]

    return c, phi

def ar_one_step_fitted(resid, c, phi):


    r = np.asarray(resid, dtype=float)
    p = len(phi)
    n = len(r)
    rhat = np.zeros(n, dtype=float)
    rhat[:p] = 0.0

    for t in range(p, n):
        val = c
        for i in range(1, p + 1):
            val += phi[i - 1] * r[t - i]
        rhat[t] = val
    return rhat

def ar_future_forecast(last_resid_hist, c, phi, horizon, decay_tau=6.0):


    phi = np.asarray(phi, dtype=float)
    p = len(phi)
    hist = list(np.asarray(last_resid_hist, dtype=float).tolist())
    out = []

    for h in range(horizon):
        if len(hist) < p:
            val = 0.0
        else:
            val = c
            for i in range(1, p + 1):
                val += phi[i - 1] * hist[-i]


        decay = np.exp(-h / max(decay_tau, 1e-6))
        val *= decay

        out.append(val)
        hist.append(val)

    return np.asarray(out, dtype=float)

def correct_fit_with_ar_residuals(base_fit, actual_fit, p, lam_ar=1e-6, future_horizon=0):


    resid = actual_fit - base_fit
    model = fit_ar_p_with_ridge(resid, p=p, lam=lam_ar)
    if model is None:
        return None

    c, phi = model
    rhat = ar_one_step_fitted(resid, c, phi)
    corrected_fit_hist = base_fit + rhat
    innovation = actual_fit - corrected_fit_hist
    dw = durbin_watson(innovation)

    future_corr = ar_future_forecast(resid, c, phi, horizon=future_horizon, decay_tau=6.0)

    return {
        "p": p,
        "lam_ar": lam_ar,
        "c": c,
        "phi": phi,
        "resid_raw": resid,
        "resid_fitted": rhat,
        "innovation": innovation,
        "DW": dw,
        "future_corr": future_corr,
        "corrected_fit_hist": corrected_fit_hist
    }

def select_best_ar_correction(base_fit_hist, actual_fit_hist, p_list, lam_ar_list):


    candidates = []
    horizon0 = 0

    for p in p_list:
        for lam_ar in lam_ar_list:
            ret = correct_fit_with_ar_residuals(
                base_fit=base_fit_hist,
                actual_fit=actual_fit_hist,
                p=p,
                lam_ar=lam_ar,
                future_horizon=horizon0
            )
            if ret is None:
                continue

            met = _metrics(ret["corrected_fit_hist"], actual_fit_hist, np.arange(len(actual_fit_hist)))
            candidates.append({
                "p": p,
                "lam_ar": lam_ar,
                "DW": ret["DW"],
                "MAE": met["MAE"],
                "MAPE": met["MAPE"],
                "RMSE": met["RMSE"],
                "Bias": met["Bias"],
                "R2": met["R2"],
                "ret": ret
            })

    cand_df = pd.DataFrame([{
        "AR_order_p": c["p"],
        "AR_ridge_lambda": c["lam_ar"],
        "DW_innovation": c["DW"],
        "MAE": c["MAE"],
        "MAPE_%": c["MAPE"],
        "RMSE": c["RMSE"],
        "Bias": c["Bias"],
        "R2": c["R2"]
    } for c in candidates])

    if len(candidates) == 0:
        return None, cand_df

    in_band = [c for c in candidates if np.isfinite(c["DW"]) and DW_LOW <= c["DW"] <= DW_HIGH]

    if len(in_band) > 0:

        in_band = sorted(in_band, key=lambda x: (x["MAPE"], abs(x["DW"] - 2.0), x["RMSE"]))
        best = in_band[0]
    else:

        candidates = sorted(candidates, key=lambda x: (abs(x["DW"] - 2.0), x["MAPE"], x["RMSE"]))
        best = candidates[0]

    return best, cand_df


def logistic_single(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def extend_pc_area_with_logistic(years_arr, pc_area_raw):
    years_arr = np.asarray(years_arr, int)
    pa = np.asarray(pc_area_raw, float)
    m = np.isfinite(years_arr) & np.isfinite(pa)
    x, y = years_arr[m].astype(float), pa[m].astype(float)
    if x.size < 4:
        return pd.Series(pa).ffill().bfill().to_numpy(float)
    L0 = float(np.nanmax(y) * 1.05)
    p0 = [L0, 0.05, float(np.nanmedian(x))]
    bounds = ([0, 0, -np.inf], [np.inf, 10, np.inf])
    try:
        if curve_fit is None:
            raise RuntimeError("scipy curve_fit is unavailable")
        popt, _ = curve_fit(logistic_single, x, y, p0=p0, bounds=bounds, maxfev=200000)
        y_fit = logistic_single(years_arr.astype(float), *popt)
        return np.where(np.isfinite(pa), pa, y_fit)
    except Exception:
        k = (y[-1] - y[0]) / max((x[-1] - x[0]), 1.0)
        b = y[-1] - k * x[-1]
        y_lin = k * years_arr + b
        return np.where(np.isfinite(pa), pa, y_lin)


DATA_PATH = resolve_data_path(DATA_PATH)
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME).copy()
df = df.sort_values("年份").reset_index(drop=True)

years = df["年份"].astype(int).values
population_total_raw = pd.to_numeric(df.get("人口数量", np.nan), errors="coerce").values.astype(float)
if "城镇人口数量.1" in df.columns:
    population_raw = pd.to_numeric(df["城镇人口数量.1"], errors="coerce").values.astype(float)
elif "城镇人口数量" in df.columns:
    population_raw = pd.to_numeric(df["城镇人口数量"], errors="coerce").values.astype(float)
else:
    urban_rate_raw = pd.to_numeric(df.get("城镇化率", np.nan), errors="coerce").values.astype(float)
    urban_rate01 = np.where(urban_rate_raw > 1.5, urban_rate_raw / 100.0, urban_rate_raw)
    population_raw = population_total_raw * urban_rate01


population_raw = np.where(
    np.isfinite(population_raw)
    & np.isfinite(population_total_raw)
    & (population_raw > population_total_raw * 1.5),
    population_raw / 100.0,
    population_raw
)
pc_area_raw_in = pd.to_numeric(
    df.get("拟合城镇人均住宅面积", df.get("预测城镇人均住宅面积", np.nan)),
    errors="coerce"
).values.astype(float)
actual_raw = pd.to_numeric(df.get("水泥产量（亿吨）", np.nan), errors="coerce").values.astype(float)
ds_fit_raw = pd.to_numeric(df.get("拟合水泥产量（亿吨）", np.nan), errors="coerce").values.astype(float)


if np.isfinite(actual_raw).any():
    actual_tons = actual_raw * (1e8 if np.nanmax(actual_raw) < 1e6 else 1.0)
else:
    actual_tons = np.full_like(years, np.nan, dtype=float)

if np.isfinite(ds_fit_raw).any():
    ds_fit_tons = ds_fit_raw * (1e8 if np.nanmax(ds_fit_raw) < 1e6 else 1.0)
else:
    ds_fit_tons = np.full_like(years, np.nan, dtype=float)

actual_mask = np.isfinite(actual_tons)
n = len(years)

pc_area_raw = extend_pc_area_with_logistic(years, pc_area_raw_in)


population = centered_ma(
    pd.Series(population_raw).interpolate(limit_direction="both").bfill().ffill().to_numpy(float),
    win=3
)
pc_area = centered_ma(
    pd.Series(pc_area_raw).interpolate(limit_direction="both").bfill().ffill().to_numpy(float),
    win=3
)


if len(years) >= 2 and int(years[-1]) == FORECAST_END:
    _pop_raw_fill = pd.Series(population_raw).interpolate(
        limit_direction="both"
    ).bfill().ffill().to_numpy(float)
    _area_raw_fill = pd.Series(pc_area_raw).interpolate(
        limit_direction="both"
    ).bfill().ffill().to_numpy(float)

    population[-2:] = _pop_raw_fill[-2:]
    pc_area[-2:] = _area_raw_fill[-2:]


k_shape = 1.97
lambda_scale = 59.07
max_life = 100

print(f"\n✅ Weibull 参数：shape k = {k_shape:.4f}, scale λ = {lambda_scale:.4f} (years), max_life = {max_life}")

life = np.arange(1, max_life + 1)
pdf_weibull, expected_life = weibull_pdf_and_mean(life, k_shape, lambda_scale)

stock = population * pc_area
if len(stock) >= 6 and stock[0] > 0 and stock[0] < 0.6 * np.median(stock[1:6]):
    stock += (np.median(stock[1:6]) - stock[0])


def retrofit_share(y):
    if y <= 2024: return 0.02
    elif y <= 2030: return 0.02 + (0.06 - 0.02) * (y - 2024) / (2030 - 2024)
    elif y <= 2040: return 0.06 + (0.10 - 0.06) * (y - 2030) / (2040 - 2030)
    elif y <= 2070: return 0.10 + (0.16 - 0.10) * (y - 2040) / (2070 - 2040)
    elif y <= 2100: return 0.16 + (0.20 - 0.16) * (y - 2070) / (2100 - 2070)
    else: return 0.20

UNIT_CEMENT_INTENSITY = 0.11
TRANSITION_RATE = 2.4


def infra_index(y):
    if y <= 1995: return 1.1
    elif y <= 2000: return 1.1 + (1.2 - 1.1) * (y - 1995) / 5
    elif y <= 2005: return 1.2 + (1.1 - 1.2) * (y - 2000) / 5
    elif y <= 2010: return 1.1 + (1.0 - 1.1) * (y - 2005) / 5
    elif y <= 2015: return 1.0 + (0.95 - 1.0) * (y - 2010) / 5
    elif y <= 2024: return 0.95 + (0.90 - 0.95) * (y - 2015) / 9
    elif y <= 2030: return 0.90 + (0.85 - 0.90) * (y - 2024) / 6
    elif y <= 2050: return 0.85 + (0.80 - 0.85) * min(y - 2030, 20) / 20
    elif y <= 2100: return 0.80 + (0.75 - 0.80) * (y - 2050) / 50
    else: return 0.75


inflow = np.zeros(n)
outflow = np.zeros(n)
steady_inflow = max(stock[0] / max(expected_life, 1.0), 1e-6)
hist_inflow = np.full(max_life, steady_inflow)

for t in range(n):
    past = np.concatenate([hist_inflow, inflow[:t]])
    use = past[-max_life:]
    outflow[t] = float(np.sum(use[::-1] * pdf_weibull[:len(use)]))
    inflow[t] = (stock[t] - stock[t - 1]) + outflow[t] if t > 0 else outflow[t]

inflow_s = ema(inflow, alpha=0.45)


res_flow = np.zeros(n)
infra_proxy = np.zeros(n)
for i, y in enumerate(years):
    eff = inflow_s[i] if y <= 2024 else inflow_s[i] * (1 - retrofit_share(y))
    res_flow[i] = eff * UNIT_CEMENT_INTENSITY
    infra_proxy[i] = res_flow[i] * TRANSITION_RATE


z = (years - years.mean()) / 10.0
Z = np.column_stack([np.ones_like(z), z, z**2])
X = np.column_stack([res_flow[:, None] * Z, infra_proxy[:, None] * Z, Z])

fit_mask = (years >= FIT_START) & (years <= FIT_END) & actual_mask
X_fit = X[fit_mask]
y_fit = actual_tons[fit_mask]


lam_grid = [0.01, 0.1, 1, 5, 10, 20, 50, 100, 200, 500]
base_candidates = []

for lam in lam_grid:
    beta = ridge_closed_form(X_fit, y_fit, lam)
    yhat_all = X @ beta
    yhat_fit = yhat_all[fit_mask]
    met = _metrics(yhat_fit, y_fit, np.arange(len(y_fit)))
    resid0 = y_fit - yhat_fit
    dw0 = durbin_watson(resid0)

    base_candidates.append({
        "lambda": lam,
        "beta": beta,
        "yhat_all": yhat_all,
        "yhat_fit": yhat_fit,
        "MAPE": met["MAPE"],
        "RMSE": met["RMSE"],
        "R2": met["R2"],
        "DW_base": dw0
    })


base_candidates = sorted(base_candidates, key=lambda d: (d["MAPE"], -d["R2"], d["RMSE"]))
best_base = base_candidates[0]

best_lam = best_base["lambda"]
base_final_demand = best_base["yhat_all"]
base_fit_hist = best_base["yhat_fit"]

print("\n✅ 基模型选定：")
print(f"   ridge lambda = {best_lam}")
print(f"   base MAPE = {best_base['MAPE']:.2f}%")
print(f"   base R2 = {best_base['R2']:.4f}")
print(f"   base DW = {best_base['DW_base']:.4f}")


mask_elas = (years >= 2010) & (years <= 2024) & actual_mask
if np.any(mask_elas):
    A = np.column_stack([res_flow[mask_elas], infra_proxy[mask_elas]])
    b_y = base_final_demand[mask_elas]
    w = np.linalg.lstsq(A, b_y, rcond=None)[0]
else:
    w = np.array([0.7, 0.3], float)

D = np.clip(res_flow * (1.0 + TRANSITION_RATE), 1e-9, None)

if np.any(mask_elas):
    x_log = np.log(D[mask_elas])
    y_log = np.log(np.clip(base_final_demand[mask_elas], 1e-9, None))
    X_ll = np.column_stack([x_log, np.ones_like(x_log)])
    e, _ = np.linalg.lstsq(X_ll, y_log, rcond=None)[0]
    e = float(np.clip(e, 0.6, 1.4))
else:
    e = 1.0

if np.any(years == 2024):
    y_2024 = float(base_final_demand[years == 2024][0])
    D_2024 = float(D[years == 2024][0])
else:
    y_2024 = float(base_final_demand[np.isfinite(base_final_demand)][-1])
    D_2024 = float(D[np.isfinite(D)][-1])

target_dyn = y_2024 * (D / max(D_2024, 1e-9)) ** e

base_final_demand_adj = base_final_demand.copy()
start_soft = 2025
tau = 5.0
for i, y in enumerate(years):
    if y >= start_soft:
        decay = np.exp(-(y - start_soft) / tau)
        base_final_demand_adj[i] = target_dyn[i] + (base_final_demand_adj[i] - target_dyn[i]) * decay


base_fit_hist_adj = base_final_demand_adj[fit_mask]
actual_fit_hist = actual_tons[fit_mask]


p_list = [1, 2, 3, 4, 5, 6, 7, 8]
lam_ar_list = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]

best_ar, ar_search_df = select_best_ar_correction(
    base_fit_hist=base_fit_hist_adj,
    actual_fit_hist=actual_fit_hist,
    p_list=p_list,
    lam_ar_list=lam_ar_list
)

if best_ar is None:
    raise RuntimeError("AR 残差校正未找到有效候选，请检查拟合段数据是否存在缺失或样本太短。")

best_p = best_ar["p"]
best_lam_ar = best_ar["lam_ar"]
best_dw = best_ar["DW"]
best_ret = best_ar["ret"]

print("\n✅ AR残差校正选定：")
print(f"   AR order p = {best_p}")
print(f"   AR ridge lambda = {best_lam_ar}")
print(f"   corrected DW = {best_dw:.4f}")
print(f"   corrected MAPE = {best_ar['MAPE']:.2f}%")
print(f"   corrected R2 = {best_ar['R2']:.4f}")


final_demand_s = base_final_demand_adj.copy()


final_demand_s[fit_mask] = best_ret["corrected_fit_hist"]


future_mask = years > FIT_END
future_horizon = int(np.sum(future_mask))
future_corr = ar_future_forecast(
    last_resid_hist=best_ret["resid_raw"],
    c=best_ret["c"],
    phi=best_ret["phi"],
    horizon=future_horizon,
    decay_tau=6.0
)

if future_horizon > 0:
    final_demand_s[future_mask] = base_final_demand_adj[future_mask] + future_corr


final_demand_s = np.maximum(final_demand_s, 0.0)


future_idx = np.where(years > FIT_END)[0]
if len(future_idx) > 0:
    sidx = future_idx[0]
    tmp = ema_from_index(final_demand_s, alpha=0.35, start_idx=sidx)
    tmp[years <= FIT_END] = final_demand_s[years <= FIT_END]
    final_demand_s = tmp


CHINA_K_SHAPE = 2.06
CHINA_LAMBDA_SCALE = 38.7
CHINA_MAX_LIFE = 80
CHINA_UNIT_INTENSITY = 0.11
CHINA_TRANSITION_RATE = 2.4

life_china = np.arange(1, CHINA_MAX_LIFE + 1)
pdf_china, expected_life_china = weibull_pdf_and_mean(life_china, CHINA_K_SHAPE, CHINA_LAMBDA_SCALE)

stock_china = population * pc_area
if len(stock_china) >= 6 and stock_china[0] > 0 and stock_china[0] < 0.6 * np.median(stock_china[1:6]):
    stock_china += (np.median(stock_china[1:6]) - stock_china[0])

inflow_china = np.zeros(n)
outflow_china = np.zeros(n)
steady_in_china = max(stock_china[0] / max(expected_life_china, 1.0), 1e-6)
hist_in_china = np.full(CHINA_MAX_LIFE, steady_in_china)

for t in range(n):
    past = np.concatenate([hist_in_china, inflow_china[:t]])
    use = past[-CHINA_MAX_LIFE:]
    outflow_china[t] = float(np.sum(use[::-1] * pdf_china[:len(use)]))
    raw_inflow = (stock_china[t] - stock_china[t - 1]) + outflow_china[t] if t > 0 else outflow_china[t]
    inflow_china[t] = max(0.0, raw_inflow)

baseline_B_china = inflow_china * CHINA_UNIT_INTENSITY * (1.0 + CHINA_TRANSITION_RATE)

phase_masks_china = [
    ("p01_1960_1964", (years >= 1960) & (years <= 1964)),
    ("p02_1965_1969", (years >= 1965) & (years <= 1969)),
    ("p03_1970_1974", (years >= 1970) & (years <= 1974)),
    ("p04_1975_1979", (years >= 1975) & (years <= 1979)),
    ("p05_1980_1984", (years >= 1980) & (years <= 1984)),
    ("p06_1985_1989", (years >= 1985) & (years <= 1989)),
    ("p07_1990_1995", (years >= 1990) & (years <= 1995)),
    ("p08_1996_2000", (years >= 1996) & (years <= 2000)),
    ("p09_2001_2005", (years >= 2001) & (years <= 2005)),
    ("p10_2006_2010", (years >= 2006) & (years <= 2010)),
    ("p11_2011_2015", (years >= 2011) & (years <= 2015)),
    ("p12_2016_2020", (years >= 2016) & (years <= 2020)),
    ("p13_2021", years == 2021),
    ("p14_2022", years == 2022),
    ("p15_2023", years == 2023),
    ("p16_2024", years == 2024),
]

mult_china = np.full(n, np.nan, dtype=float)
phase_rows_china = []

for phase_name, phase_mask in phase_masks_china:
    fit_mask_china = (
        phase_mask
        & (years >= FIT_START)
        & (years <= FIT_END)
        & actual_mask
        & np.isfinite(actual_tons)
        & np.isfinite(baseline_B_china)
        & (baseline_B_china > 1e-9)
    )
    if not np.any(fit_mask_china):
        continue

    Bp = baseline_B_china[fit_mask_china]
    yp = actual_tons[fit_mask_china]
    bj = float(np.sum(Bp * yp) / max(np.sum(Bp * Bp), 1e-12))
    bj = float(np.clip(bj, 0.05, 20.0))
    mult_china[phase_mask] = bj
    phase_rows_china.append({
        "phase": phase_name,
        "year_start": int(np.min(years[phase_mask])),
        "year_end": int(np.max(years[phase_mask])),
        "multiplier": bj,
        "n_fit": int(np.sum(fit_mask_china)),
    })

if np.any(np.isfinite(mult_china)):
    mult_china = pd.Series(mult_china).ffill().bfill().to_numpy(dtype=float).copy()
else:
    mult_china = np.ones(n, dtype=float)

FUTURE_MULT_ANCHOR_YEAR = FIT_END
FUTURE_MULT_DECAY = 1.0
FUTURE_MULT_FLOOR_FRAC = 1.0

anchor_mask_china = years == FUTURE_MULT_ANCHOR_YEAR
anchor_mult_china = float(mult_china[anchor_mask_china][0]) if np.any(anchor_mask_china) else float(mult_china[years <= FIT_END][-1])
mult_floor_china = anchor_mult_china * FUTURE_MULT_FLOOR_FRAC

for i, yy in enumerate(years):
    if yy > FIT_END:
        dt = yy - FUTURE_MULT_ANCHOR_YEAR
        mult_china[i] = max(mult_floor_china, anchor_mult_china * (FUTURE_MULT_DECAY ** dt))

china_stock_raw = np.maximum(baseline_B_china * mult_china, 0.0)
hist_scale_mask_china = (
    (years >= FIT_START)
    & (years <= FIT_END)
    & actual_mask
    & np.isfinite(china_stock_raw)
    & (china_stock_raw > 1e-9)
)
if np.any(hist_scale_mask_china):
    p = china_stock_raw[hist_scale_mask_china]
    y = actual_tons[hist_scale_mask_china]
    global_scale_china = float(np.sum(p * y) / max(np.sum(p * p), 1e-12))
else:
    global_scale_china = 1.0

china_stock_pred_all = np.maximum(china_stock_raw * global_scale_china, 0.0)


china_splice_scale = 1.0
final_demand_s[years > FIT_END] = china_stock_pred_all[years > FIT_END]

final_demand_before_dw_correction = final_demand_s.copy()
fit_mask_eval_for_dw = (years >= FIT_START) & (years <= FIT_END) & actual_mask & np.isfinite(final_demand_s)
hist_dw_rho_selected = np.nan
hist_dw_rho_search_df = pd.DataFrame()
if USE_HIST_DW_CORRECTION:
    if AUTO_TUNE_HIST_DW_RHO:
        rho_grid = np.round(np.arange(-0.10, 0.501, 0.01), 3)
        hist_dw_rho_selected, final_demand_s, hist_dw_rho_search_df = tune_historical_dw_rho(
            final_demand_s, actual_tons, years, fit_mask_eval_for_dw, rho_grid
        )
    else:
        hist_dw_rho_selected = HIST_DW_RHO_CORRECTION
        final_demand_s = apply_historical_dw_correction(
            final_demand_s, actual_tons, fit_mask_eval_for_dw,
            rho=hist_dw_rho_selected
        )
    final_demand_s = np.maximum(final_demand_s, 0.0)

final_demand_before_ds_guide = final_demand_s.copy()
ds_future_guide_scaled = np.full_like(final_demand_s, np.nan, dtype=float)
ds_guide_scale = np.nan
if USE_DS_FUTURE_GUIDE:
    y_ref = _pick_year(years, final_demand_s, FIT_END)
    y_guide_ref = _pick_year(years, ds_fit_tons, FIT_END)
    if np.isfinite(y_ref) and np.isfinite(y_guide_ref) and y_guide_ref > 1e-9:
        ds_guide_scale = float(y_ref / y_guide_ref)
        ds_future_guide_scaled = ds_fit_tons * ds_guide_scale
        guide_mask = (years > FIT_END) & np.isfinite(ds_future_guide_scaled)
        blend = float(np.clip(DS_FUTURE_GUIDE_BLEND, 0.0, 1.0))
        final_demand_s[guide_mask] = (
            (1.0 - blend) * final_demand_s[guide_mask]
            + blend * ds_future_guide_scaled[guide_mask]
        )


FORECAST_BRIDGE_END = 2050

scenario_raw_before_bridge = china_stock_pred_all.copy()
forecast_bridge_weight = np.ones(len(years), dtype=float)
forecast_bridge_weight[years <= FIT_END] = 0.0

if np.any(years == FIT_END):
    idx_2024_bridge = int(np.where(years == FIT_END)[0][0])
    if np.isfinite(actual_tons[idx_2024_bridge]) and actual_tons[idx_2024_bridge] > 0:
        bridge_anchor_2024 = float(actual_tons[idx_2024_bridge])
    else:
        bridge_anchor_2024 = float(final_demand_s[idx_2024_bridge])
else:
    idx_2024_bridge = None
    bridge_anchor_2024 = np.nan


final_demand_s[years > FIT_END] = scenario_raw_before_bridge[years > FIT_END]


if idx_2024_bridge is not None and np.isfinite(bridge_anchor_2024):
    final_demand_s[idx_2024_bridge] = bridge_anchor_2024

bridge_mask = (years > FIT_END) & (years <= FORECAST_BRIDGE_END)
if np.any(bridge_mask) and np.isfinite(bridge_anchor_2024):
    x_bridge = (
        years[bridge_mask].astype(float) - float(FIT_END)
    ) / max(float(FORECAST_BRIDGE_END - FIT_END), 1.0)
    x_bridge = np.clip(x_bridge, 0.0, 1.0)
    w_bridge = x_bridge ** 2 * (3.0 - 2.0 * x_bridge)
    forecast_bridge_weight[bridge_mask] = w_bridge

    final_demand_s[bridge_mask] = (
        (1.0 - w_bridge) * bridge_anchor_2024
        + w_bridge * scenario_raw_before_bridge[bridge_mask]
    )


splice_scale = 1.0
forecast_splice_scale = 1.0
forecast_start_yoy = np.nan

china_splice_scale = 1.0


fit_mask_eval = (years >= FIT_START) & (years <= FIT_END) & actual_mask & np.isfinite(final_demand_s)

m_all = _metrics(final_demand_s, actual_tons, years, a=FIT_START, b=FIT_END)

resid_fit_final = actual_tons[fit_mask_eval] - final_demand_s[fit_mask_eval]
DW_val = durbin_watson(resid_fit_final)
AR1_val = ar1_rho_ols(resid_fit_final)

tail_info = _tail_slope(years, final_demand_s, 2033, 2050)

y2030 = _pick_year(years, final_demand_s, 2030)
y2050 = _pick_year(years, final_demand_s, 2050)
drop_2030_2050 = (
    100.0 * (y2050 / y2030 - 1.0)
    if (np.isfinite(y2030) and y2030 > 0 and np.isfinite(y2050))
    else np.nan
)

notes = []
ok = True

if (not np.isfinite(m_all.get("MAPE", np.nan))) or m_all["MAPE"] >= MAPE_THRESH_ALL:
    ok = False
    notes.append(f"拟合段 MAPE {m_all['MAPE']:.2f}% 未达到严格小于 {MAPE_THRESH_ALL:.2f}% 的要求")

if (not np.isfinite(m_all.get("R2", np.nan))) or m_all["R2"] <= 0.90:
    ok = False
    notes.append(f"拟合段 R2={m_all['R2']:.4f} 未达到严格大于 0.90 的要求")

if np.isfinite(drop_2030_2050) and drop_2030_2050 < DROP_MIN_2030_2050:
    ok = False
    notes.append(f"2030→2050 降幅 {drop_2030_2050:.1f}% < {DROP_MIN_2030_2050:.1f}%（下降过度）")

if np.any(np.isfinite(final_demand_s)) and np.nanmin(final_demand_s) < 0:
    ok = False
    notes.append("预测序列出现负值（Nonnegativity fail）")

if not (DW_LOW <= DW_val <= DW_HIGH):
    ok = False
    notes.append(f"拟合段 DW={DW_val:.3f} 未落入 [{DW_LOW:.1f}, {DW_HIGH:.1f}]")


pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)

def _title(msg):
    print("\n" + "=" * len(msg))
    print(msg)
    print("=" * len(msg))

metrics_df = pd.DataFrame([{
    "指标段": f"拟合段总体({FIT_START}-{FIT_END})",
    **m_all,
    "DW_resid_fit": DW_val,
    "AR1_rho_resid_fit": AR1_val,
    "Base_lambda": best_lam,
    "AR_order_p": best_p,
    "AR_ridge_lambda": best_lam_ar,
    "Hist_DW_correction_enabled": USE_HIST_DW_CORRECTION,
    "Hist_DW_rho_correction": HIST_DW_RHO_CORRECTION,
    "Hist_DW_auto_tune_enabled": AUTO_TUNE_HIST_DW_RHO,
    "Hist_DW_rho_selected": hist_dw_rho_selected,
    "DS_future_guide_enabled": USE_DS_FUTURE_GUIDE,
    "DS_future_guide_blend": DS_FUTURE_GUIDE_BLEND
}])

tail_df = pd.DataFrame([{
    "tail_start": 2033,
    "tail_end": 2050,
    "slope_ton_per_year": tail_info["slope_ton_per_year"],
    "slope_pct_per_year": tail_info["slope_pct_per_year"],
    "tail_line_R2": tail_info["R2"],
    "降幅_2030到2050_%": drop_2030_2050
}])

_title(f"回测指标（拟合段：{FIT_START}-{FIT_END}）")
print(metrics_df.round({
    "MAE": 0,
    "MAPE": 2,
    "RMSE": 0,
    "Bias": 0,
    "R2": 4,
    "DW_resid_fit": 4,
    "AR1_rho_resid_fit": 4,
    "Base_lambda": 4,
    "AR_order_p": 0,
    "AR_ridge_lambda": 6
}).to_string(index=False))

_title("AR残差校正候选结果")
print(
    ar_search_df.sort_values(["AR_order_p", "AR_ridge_lambda"]).round({
        "AR_order_p": 0,
        "AR_ridge_lambda": 6,
        "DW_innovation": 4,
        "MAE": 0,
        "MAPE_%": 2,
        "RMSE": 0,
        "Bias": 0,
        "R2": 4
    }).to_string(index=False)
)

_title("尾部诊断")
print(tail_df.round({
    "slope_ton_per_year": 0,
    "slope_pct_per_year": 3,
    "tail_line_R2": 4,
    "降幅_2030到2050_%": 2
}).to_string(index=False))

summary_line = (
    f"综合判定：{'✅ 通过' if ok else '❌ 未通过'}；"
    f"Weibull(k={k_shape:.2f}, λ={lambda_scale:.1f})；"
    f"拟合段({FIT_START}-{FIT_END})："
    f"MAPE={m_all['MAPE']:.2f}%、R2={m_all['R2']:.4f}、DW={DW_val:.4f}、AR(1)={AR1_val:.4f}；"
    f"DW_rho_selected={hist_dw_rho_selected:.3f}；"
    f"基模型lambda={best_lam}；AR(p)={best_p}；AR_lambda={best_lam_ar}；"
    f"预测段=中国式B×历史乘子延续×scale；"
    f"尾部斜率={tail_info['slope_pct_per_year']:.3f}%/年；"
    f"2030→2050降幅={drop_2030_2050:.2f}%"
)
print("\n" + summary_line)

if notes:
    print("未通过原因：")
    for t in notes:
        print(" - " + t)


out_df = pd.DataFrame({
    "年份": years,
    "人口": population,
    "人均住宅面积": pc_area,
    "住宅存量": stock,
    "新增面积_raw": inflow,
    "新增面积_smooth": inflow_s,
    "China_stock": stock_china,
    "China_inflow": inflow_china,
    "China_outflow": outflow_china,
    "China_B": baseline_B_china,
    "China_multiplier": mult_china,
    "China_stock_pred": china_stock_pred_all,
    "Forecast_bridge_weight": forecast_bridge_weight,
    "Scenario_raw_before_bridge": scenario_raw_before_bridge,
    "DS拟合水泥产量_原始(吨)": ds_fit_tons,
    "DS拟合水泥产量_缩放后(吨)": ds_future_guide_scaled,
    "住宅流入_proxy": res_flow,
    "基建_proxy": infra_proxy,
    "驱动指数D": D,
    "基模型预测_raw": base_final_demand,
    "基模型预测_adjusted": base_final_demand_adj,
    "最终预测_DW校正前": final_demand_before_dw_correction,
    "最终预测_DS约束前": final_demand_before_ds_guide,
    "最终预测水泥需求": final_demand_s,
    "真实产量": actual_tons,
    "误差(最终预测-真实)": final_demand_s - actual_tons
})

params_df = pd.DataFrame([{
    "Weibull_shape_k": k_shape,
    "Weibull_scale_lambda_years": lambda_scale,
    "max_life": max_life,
    "Base_lambda": best_lam,
    "AR_order_p": best_p,
    "AR_ridge_lambda": best_lam_ar,
    "China_formula": "B * historical_multiplier_extension * scale",
    "China_k": CHINA_K_SHAPE,
    "China_lambda": CHINA_LAMBDA_SCALE,
    "China_max_life": CHINA_MAX_LIFE,
    "China_unit_intensity": CHINA_UNIT_INTENSITY,
    "China_transition_rate": CHINA_TRANSITION_RATE,
    "China_future_multiplier_decay": FUTURE_MULT_DECAY,
    "China_future_multiplier_floor_frac": FUTURE_MULT_FLOOR_FRAC,
    "China_global_scale": global_scale_china,
    "China_splice_scale": china_splice_scale,
    "Hist_DW_correction_enabled": USE_HIST_DW_CORRECTION,
    "Hist_DW_rho_correction": HIST_DW_RHO_CORRECTION,
    "Hist_DW_auto_tune_enabled": AUTO_TUNE_HIST_DW_RHO,
    "Hist_DW_rho_selected": hist_dw_rho_selected,
    "DS_future_guide_enabled": USE_DS_FUTURE_GUIDE,
    "DS_future_guide_blend": DS_FUTURE_GUIDE_BLEND,
    "DS_future_guide_scale_to_2024": ds_guide_scale,
    f"MAPE_fit_{FIT_START}_{FIT_END}_%": m_all.get("MAPE", np.nan),
    f"R2_fit_{FIT_START}_{FIT_END}": m_all.get("R2", np.nan),
    f"DW_resid_fit_{FIT_START}_{FIT_END}": DW_val,
    f"AR1_rho_resid_fit_{FIT_START}_{FIT_END}": AR1_val,
    "tail_slope_pct_per_year(2033-2050)": tail_info.get("slope_pct_per_year", np.nan),
    "drop_2030_2050_%": drop_2030_2050
}])

phase_df_china = pd.DataFrame(phase_rows_china)

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    out_df.to_excel(writer, index=False, sheet_name="Result_Series")
    metrics_df.to_excel(writer, index=False, sheet_name="Backtest_Metrics_All")
    tail_df.to_excel(writer, index=False, sheet_name="Tail_Diagnostics")
    params_df.to_excel(writer, index=False, sheet_name="Params_Summary")
    ar_search_df.to_excel(writer, index=False, sheet_name="AR_Residual_Search")
    hist_dw_rho_search_df.to_excel(writer, index=False, sheet_name="DW_Rho_Search")
    phase_df_china.to_excel(writer, index=False, sheet_name="China_Phase_Multipliers")

print(f"\n✅ 已导出到：{OUTPUT_XLSX}")


if HAS_MATPLOTLIB:
    plt.figure(figsize=(13, 7))
    plt.plot(years, final_demand_s / 1e6, "r.-", label="Final forecast (history fit + China-logic forecast)")
    plt.plot(years, base_final_demand_adj / 1e6, "r--", alpha=0.35, label="Base model")
    if np.any(actual_mask):
        plt.plot(years[actual_mask], actual_tons[actual_mask] / 1e6, "ks-", label="Actual production")

    plt.axvline(HIS_END + 1, color="gray", linestyle="--", label=f"Forecast start ({HIS_END+1})")
    plt.xlabel("Year")
    plt.ylabel("Cement production (million tons)")
    plt.title(f"Japan Cement Demand Forecast: {FIT_START}-2100 (History Fit + China-logic Forecast)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, edgecolor="gray", borderaxespad=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    print(f"✅ 已导出图片：{OUTPUT_PNG}")
    plt.close()
else:
    print("⚠️ 当前环境未安装 matplotlib，已跳过绘图；Excel 结果仍正常导出。")
