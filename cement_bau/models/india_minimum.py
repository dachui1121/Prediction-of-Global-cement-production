from cement_bau.shared_history import apply_shared_history as _shared_history
from cement_bau.runtime import input_path as _input_path, output_dir as _output_dir
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from scipy.optimize import curve_fit


PRESET = "steady"
SHEET_NAME = "Sheet1"
FORECAST_START = 2025

DATA_PATH = str(_input_path('india'))


desktop = str(_output_dir('india', 'minimum'))
OUT_XLSX = os.path.join(desktop, f"India_BAU_Hybrid_FitForecast_Final_{PRESET}_MAPE10_NoRetrofit.xlsx")
OUT_PNG = os.path.join(desktop, f"India_BAU_Hybrid_FitForecast_Final_{PRESET}_MAPE10_NoRetrofit.png")
os.makedirs(desktop, exist_ok=True)

FIT_START = 1990
FIT_END = 2024
TAIL_START = 2050
TAIL_END = 2100


CAL_FACTOR_MIN = 0.45
CAL_FACTOR_MAX = 1.80

HIST_SMOOTH_WIN1 = 3
HIST_SMOOTH_WIN2 = 3
FUTURE_CAL_TAU = 16.0


USE_RESIDUAL_BRIDGE = True
RESID_ALPHA = 0.38
RESID_BLEND = 0.36
RESID_CLIP = 0.09
RESID_DECAY_TAU = 7.0
RESID_PERSIST = 0.55
RESID_LONG_WIN = 5


USE_HISTORY_ANTIZIGZAG = True
HIST_ANTIZIGZAG_LAMBDA = 0.18

USE_AR1_DIAGNOSTIC_ONLY = True


PRESETS = {
    "aggressive": {
        "TAU_CAL": 16.0, "TAPER_SHIFT": 2,
        "CAP_START": 0.040, "MAX_DROP": 0.015, "ACCEL_CAP": 0.010,
        "BLEND": 0.20,
        "WEI_SHAPE": 1.97, "WEI_SCALE": 56.4, "MAX_LIFE": 100,
        "V_HORIZON": 4, "V_TOL_UP": 0.010,
        "V_LOOK": 5, "V_DEPTH": 0.04
    },
    "steady": {
        "TAU_CAL": 14.0, "TAPER_SHIFT": 2,
        "CAP_START": 0.045, "MAX_DROP": 0.020, "ACCEL_CAP": 0.012,
        "BLEND": 0.25,
        "WEI_SHAPE": 1.97, "WEI_SCALE": 56.4, "MAX_LIFE": 100,
        "V_HORIZON": 4, "V_TOL_UP": 0.012,
        "V_LOOK": 5, "V_DEPTH": 0.04
    },
    "gentle": {
        "TAU_CAL": 12.0, "TAPER_SHIFT": 1,
        "CAP_START": 0.050, "MAX_DROP": 0.020, "ACCEL_CAP": 0.015,
        "BLEND": 0.20,
        "WEI_SHAPE": 1.97, "WEI_SCALE": 56.4, "MAX_LIFE": 100,
        "V_HORIZON": 3, "V_TOL_UP": 0.015,
        "V_LOOK": 5, "V_DEPTH": 0.05
    }
}
P = PRESETS[PRESET]


plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10


def pick(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def to_Int64(x, round0=None):
    s = pd.Series(x, dtype="float64").replace([np.inf, -np.inf], np.nan)
    if round0 is not None:
        s = s.round(round0)
    return s.astype(pd.Int64Dtype())

def to_float(x, ndigits=None):
    s = pd.Series(x, dtype="float64").replace([np.inf, -np.inf], np.nan)
    if ndigits is not None:
        s = s.round(ndigits)
    return s

def hyper_inc(t, L, k, t0):
    t = np.asarray(t, float)
    return L - k / np.maximum(t - t0, 1e-6)

def fit_hyper(x_his, y):
    y = np.asarray(y, float)
    L0 = float(np.nanmax(y) * 1.2)
    k0 = max(1.0, (L0 - float(y[0])) * (x_his[-1] - x_his[0] + 1))
    t0 = float(x_his.min() - 50.0)
    bounds = (
        [np.nanmax(y), 1e-6, x_his.min() - 200.0],
        [np.nanmax(y) * 5, 1e12, x_his.min() - 1.0]
    )
    p, _ = curve_fit(hyper_inc, x_his, y, p0=[L0, k0, t0], bounds=bounds, maxfev=100000)
    return p

def safe_cagr(arr, yrs=6, lo=-0.02, hi=0.02, default=0.006):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return default
    n = min(yrs, arr.size)
    if arr[0] <= 0:
        return default
    g = (arr[n - 1] / arr[0]) ** (1 / (n - 1)) - 1.0
    return float(np.clip(g, lo, hi))

def moving_avg(arr, win=5):
    arr = np.asarray(arr, float)
    if arr.size == 0:
        return arr.copy()
    win = max(1, int(win))
    pad_l = win // 2
    pad_r = win - 1 - pad_l
    arr_pad = np.pad(arr, (pad_l, pad_r), mode="edge")
    ker = np.ones(win, float) / win
    return np.convolve(arr_pad, ker, mode="valid")

def build_smoothed_residual_bridge(
    years, actual_axis, base_pred, fit_start, fit_end,
    forecast_start=2025, alpha=0.38, blend=0.36, clip=0.09,
    decay_tau=7.0, persist=0.55, long_win=5
):


    years = np.asarray(years, int)
    actual_axis = np.asarray(actual_axis, float)
    base_pred = np.asarray(base_pred, float)

    logcorr = np.zeros_like(base_pred, dtype=float)

    mask = (
        (years >= fit_start) & (years <= fit_end) &
        np.isfinite(actual_axis) & np.isfinite(base_pred) &
        (actual_axis > 1e-9) & (base_pred > 1e-9)
    )

    if mask.sum() < 5:
        return logcorr

    raw = np.full_like(base_pred, np.nan, dtype=float)
    raw[mask] = np.log(actual_axis[mask]) - np.log(base_pred[mask])
    raw[mask] = np.clip(raw[mask], -clip, clip)

    hist_idx = np.where(mask)[0]

    smoothed = np.full_like(base_pred, np.nan, dtype=float)
    prev = float(raw[hist_idx[0]])
    smoothed[hist_idx[0]] = prev
    for k in range(1, len(hist_idx)):
        i = hist_idx[k]
        prev = alpha * float(raw[i]) + (1.0 - alpha) * prev
        smoothed[i] = prev

    inertial = np.full_like(base_pred, np.nan, dtype=float)
    inertial[hist_idx[0]] = smoothed[hist_idx[0]]
    for k in range(1, len(hist_idx)):
        i = hist_idx[k]
        ip = hist_idx[k - 1]
        inertial[i] = (1.0 - persist) * smoothed[i] + persist * inertial[ip]

    s = pd.Series(inertial[hist_idx], index=years[hist_idx], dtype="float64")
    s = s.rolling(window=long_win, center=True, min_periods=1).mean()
    lowfreq = np.clip(s.values, -clip, clip)

    logcorr[hist_idx] = blend * lowfreq

    if forecast_start in set(years):
        i0 = int(np.where(years == forecast_start)[0][0])
        i_anchor = int(np.where(years == fit_end)[0][0]) if fit_end in set(years) else hist_idx[-1]
        last_corr = float(logcorr[i_anchor])

        for i in range(i0, len(years)):
            dt = years[i] - forecast_start + 1
            logcorr[i] = last_corr * np.exp(-dt / decay_tau)

    return logcorr

def apply_history_antizigzag(years, y, fit_start, fit_end, lam=0.18):


    years = np.asarray(years, int)
    y = np.asarray(y, float).copy()

    idx = np.where((years >= fit_start) & (years <= fit_end) & np.isfinite(y))[0]
    if len(idx) < 3:
        return y

    y_new = y.copy()
    for k in range(1, len(idx) - 1):
        i = idx[k]
        i_prev = idx[k - 1]
        i_next = idx[k + 1]
        neigh = 0.5 * (y[i_prev] + y[i_next])
        y_new[i] = (1.0 - lam) * y[i] + lam * neigh

    return y_new


def build_smooth_history_calibration(
    years, struct_base, actual_axis, fit_start, fit_end,
    cal_min=0.68, cal_max=1.52, win_median=3, win_mean=5, future_tau=14.0
):
    years = np.asarray(years, int)
    struct_base = np.asarray(struct_base, float)
    actual_axis = np.asarray(actual_axis, float)

    cal = np.ones_like(struct_base, dtype=float)

    hist_mask = (
        (years >= fit_start) & (years <= fit_end) &
        np.isfinite(actual_axis) & np.isfinite(struct_base) &
        (actual_axis > 1e-9) & (struct_base > 1e-9)
    )

    if hist_mask.sum() < 5:
        return cal

    ratio = np.full_like(struct_base, np.nan, dtype=float)
    ratio[hist_mask] = actual_axis[hist_mask] / struct_base[hist_mask]

    rr = ratio[hist_mask]
    q05, q95 = np.nanpercentile(rr, [5, 95])
    lo = max(cal_min, q05)
    hi = min(cal_max, q95)
    ratio[hist_mask] = np.clip(ratio[hist_mask], lo, hi)

    log_ratio = np.full_like(ratio, np.nan, dtype=float)
    log_ratio[hist_mask] = np.log(ratio[hist_mask])

    s = pd.Series(log_ratio, index=years, dtype="float64")
    s = s.interpolate(method="linear", limit_direction="both")
    s = s.rolling(window=win_median, center=True, min_periods=1).median()
    s = s.rolling(window=win_mean, center=True, min_periods=1).mean()

    cal_hist = np.exp(s.values)
    cal_hist = np.clip(cal_hist, cal_min, cal_max)

    cal[years <= fit_end] = cal_hist[years <= fit_end]

    if FORECAST_START in set(years):
        i0 = int(np.where(years == FORECAST_START)[0][0])
        i_anchor = int(np.where(years == fit_end)[0][0]) if fit_end in set(years) else max(i0 - 1, 0)
        cal0 = float(cal[i_anchor])

        for i in range(i0, len(years)):
            dt = years[i] - FORECAST_START + 1
            cal[i] = 1.0 + (cal0 - 1.0) * np.exp(-dt / future_tau)

    return np.clip(cal, cal_min, cal_max)


def cement_intensity(y: int) -> float:

    return 0.214

def retrofit_share(y):


    return 0.00

CI_RETRO_SCALE = 0.45
TRANSITION_RATE = 2.4
NONRES_K0 = TRANSITION_RATE

def nonres_infra_multiplier(y: int) -> float:

    return 1.0 + TRANSITION_RATE


AUTO_PEAK_SEARCH_START = 2035
AUTO_PEAK_MIN_YEARS_FROM_START = 10
AUTO_PEAK_TRAIL_WIN = 5
AUTO_PEAK_FWD_WIN = 5
AUTO_PEAK_GROWTH_TH = 0.004
AUTO_PEAK_ALPHA_TH = 0.0015

DECLINE_RATE_START = 0.003
DECLINE_RATE_END = 0.012
POST_PEAK_BUFFER = 0.002

SATURATION_HEADWIND_2060 = 0.98
SATURATION_HEADWIND_2080 = 0.90
SATURATION_HEADWIND_2100 = 0.80

def infer_auto_peak_year(
    years, baseline, alpha_total, inflow_base=None, outflow=None,
    search_start=AUTO_PEAK_SEARCH_START,
    min_years_from_start=AUTO_PEAK_MIN_YEARS_FROM_START,
    trail_win=AUTO_PEAK_TRAIL_WIN,
    fwd_win=AUTO_PEAK_FWD_WIN,
    growth_th=AUTO_PEAK_GROWTH_TH,
    alpha_th=AUTO_PEAK_ALPHA_TH,
    eps=1e-9
):
    years = np.asarray(years, int)
    baseline = np.asarray(baseline, float)
    alpha_total = np.asarray(alpha_total, float)

    base_s = moving_avg(baseline, 5)
    alpha_s = moving_avg(alpha_total, 5)

    logb = np.log(np.maximum(base_s, eps))
    loga = np.log(np.maximum(alpha_s, eps))

    candidates = []
    for i in range(len(years)):
        yr = int(years[i])

        if yr < max(search_start, FORECAST_START + min_years_from_start):
            continue
        if i - trail_win < 0 or i + fwd_win >= len(years):
            continue

        g_fwd = float((logb[i + fwd_win] - logb[i]) / max(1, fwd_win))
        g_trail = float((logb[i] - logb[i - trail_win]) / max(1, trail_win))
        a_trail = float((loga[i] - loga[i - trail_win]) / max(1, trail_win))

        cond1 = g_fwd <= growth_th
        cond2 = a_trail <= -alpha_th
        cond3 = g_trail > max(growth_th * 0.5, 0.001)

        cond4 = True
        if inflow_base is not None and outflow is not None:
            inflow_s = moving_avg(inflow_base, 5)
            outflow_s = moving_avg(outflow, 5)
            cond4 = bool(outflow_s[i] / max(inflow_s[i], eps) >= 0.40)

        if cond1 and cond2 and cond3 and cond4:
            score = (growth_th - g_fwd) + (-a_trail) + 0.20 * max(0.0, g_trail)
            candidates.append((yr, i, score))

    if candidates:
        candidates.sort(key=lambda x: (x[0], -x[2]))
        return int(candidates[0][0])

    mask = years >= max(search_start, FORECAST_START + min_years_from_start)
    if np.any(mask):
        idx_local = np.where(mask)[0]
        return int(years[idx_local[np.argmax(base_s[mask])]])

    return int(years[np.argmax(base_s)])

def demand_saturation_headwind(y, auto_peak_year):
    y = float(y)
    if y <= auto_peak_year:
        return 1.0
    elif y <= 2060:
        return 1.0 + (SATURATION_HEADWIND_2060 - 1.0) * (y - auto_peak_year) / max(1.0, (2060 - auto_peak_year))
    elif y <= 2080:
        return SATURATION_HEADWIND_2060 + (SATURATION_HEADWIND_2080 - SATURATION_HEADWIND_2060) * (y - 2060) / (2080 - 2060)
    else:
        return SATURATION_HEADWIND_2080 + (SATURATION_HEADWIND_2100 - SATURATION_HEADWIND_2080) * (y - 2080) / (2100 - 2080)

def post_peak_decline_cap(y, peak_year):
    y = float(y)
    if y <= peak_year:
        return 0.0
    return float(
        DECLINE_RATE_START +
        (DECLINE_RATE_END - DECLINE_RATE_START) * (y - peak_year) / max(1.0, (2100 - peak_year))
    )

def enforce_prepeak_rise_postpeak_decline(y, years, peak_year, rebound_buffer=POST_PEAK_BUFFER, eps=1e-9):
    y = np.asarray(y, float).copy()
    years = np.asarray(years, int)

    if peak_year not in set(years):
        return y
    if FORECAST_START not in set(years):
        return y

    peak_idx = int(np.where(years == peak_year)[0][0])
    s = int(np.where(years == FORECAST_START)[0][0])

    for t in range(s, peak_idx + 1):
        prev = float(max(y[t - 1], eps))
        y[t] = max(y[t], prev * 1.0010)

    prepeak_max = float(np.max(y[s:peak_idx + 1]))
    y[peak_idx] = max(y[peak_idx], prepeak_max)

    running_cap = float(y[peak_idx])

    for t in range(peak_idx + 1, len(y)):
        yr = int(years[t])
        prev = float(max(y[t - 1], eps))

        sat = demand_saturation_headwind(yr, peak_year)
        sat_prev = demand_saturation_headwind(int(years[t - 1]), peak_year)
        sat_ratio = sat / max(sat_prev, eps)

        decline = post_peak_decline_cap(yr, peak_year)

        upper1 = prev * (1.0 - decline + rebound_buffer)
        upper2 = running_cap * (1.0 + rebound_buffer)
        upper3 = y[t] * sat_ratio
        lower1 = prev * (1.0 - max(0.018, 1.8 * decline))

        y[t] = min(y[t], upper1, upper2, upper3)
        y[t] = max(y[t], lower1)

        running_cap = max(running_cap, y[t])

    return y


def calc_fit_metrics(y_true, y_pred, years_arr, start_year, end_year):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    years_arr = np.asarray(years_arr, dtype=int)

    m = (
        (years_arr >= start_year) & (years_arr <= end_year) &
        np.isfinite(y_true) & np.isfinite(y_pred)
    )

    yt = y_true[m]
    yp = y_pred[m]

    if len(yt) < 2:
        return {
            "n_obs": len(yt),
            "MAPE_pct": np.nan,
            "MAE_tons": np.nan,
            "RMSE_tons": np.nan,
            "R2": np.nan,
            "DW": np.nan
        }

    err = yp - yt
    mape_mask = np.abs(yt) > 1e-9
    mape = np.mean(np.abs(err[mape_mask] / yt[mape_mask])) * 100.0 if mape_mask.any() else np.nan
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))

    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    den = np.sum(err ** 2)
    dw = np.sum(np.diff(err) ** 2) / den if den > 1e-12 and len(err) >= 2 else np.nan

    return {
        "n_obs": int(len(yt)),
        "MAPE_pct": float(mape) if np.isfinite(mape) else np.nan,
        "MAE_tons": float(mae),
        "RMSE_tons": float(rmse),
        "R2": float(r2) if np.isfinite(r2) else np.nan,
        "DW": float(dw) if np.isfinite(dw) else np.nan
    }

def calc_tail_metrics(years_arr, series, tail_start, tail_end):
    years_arr = np.asarray(years_arr, dtype=float)
    series = np.asarray(series, dtype=float)

    m = (years_arr >= tail_start) & (years_arr <= tail_end) & np.isfinite(series)
    x = years_arr[m]
    y_mt = series[m] / 1e6

    if len(x) < 2:
        return {
            "tail_n": int(len(x)),
            "Tail_Slope_pct_per_year": np.nan,
            "Tail_R2": np.nan,
            "Drop": np.nan,
            "Tail_Start_Mt": np.nan,
            "Tail_End_Mt": np.nan
        }

    k, b = np.polyfit(x, y_mt, 1)
    yhat = k * x + b

    ss_res = np.sum((y_mt - yhat) ** 2)
    ss_tot = np.sum((y_mt - np.mean(y_mt)) ** 2)
    tail_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    y0 = float(y_mt[0])
    y1 = float(y_mt[-1])

    tail_slope_pct = (k / y0) * 100.0 if abs(y0) > 1e-12 else np.nan
    drop = ((y0 - y1) / y0) if abs(y0) > 1e-12 else np.nan

    return {
        "tail_n": int(len(x)),
        "Tail_Slope_pct_per_year": float(tail_slope_pct) if np.isfinite(tail_slope_pct) else np.nan,
        "Tail_R2": float(tail_r2) if np.isfinite(tail_r2) else np.nan,
        "Drop": float(drop) if np.isfinite(drop) else np.nan,
        "Tail_Start_Mt": y0,
        "Tail_End_Mt": y1
    }

def ar1_rho_from_residuals(yhat, y, mask, clip=0.95):
    m = mask.astype(bool) & np.isfinite(yhat) & np.isfinite(y)
    idx = np.where(m)[0]
    if idx.size < 4:
        return np.nan
    keep = np.concatenate([[True], (np.diff(idx) == 1)])
    idx = idx[keep]
    if idx.size < 4:
        return np.nan
    e = (y[idx] - yhat[idx]).astype(float)
    e0, e1 = e[:-1], e[1:]
    den = float(np.dot(e0, e0))
    rho = float(np.dot(e0, e1) / den) if den > 0 else np.nan
    if np.isfinite(rho):
        rho = float(np.clip(rho, -clip, clip))
    return rho


df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME).copy()

c_year = pick(df, ["年份", "Year", "year"])
c_pop = pick(df, ["城镇人口数量", "城市人口", "人口", "Population"])
c_pca = pick(df, ["拟合人均住宅面积", "人均住宅面积", "PerCapitaFloorArea"])
c_act = pick(df, ["真实水泥产量", "水泥产量", "ActualCement", "印度水泥产量（亿吨）", "水泥产量（亿吨）"])
assert c_year and c_pop and c_pca, "缺少必要列：年份/人口/人均住宅面积"

df = df[[c_year, c_pop, c_pca] + ([c_act] if c_act else [])].dropna(subset=[c_year, c_pop, c_pca])
df = df.sort_values(c_year).reset_index(drop=True)

years_input = df[c_year].astype(int).values
pop_raw = pd.to_numeric(df[c_pop], errors="coerce").values.astype(float)
pca_raw = pd.to_numeric(df[c_pca], errors="coerce").values.astype(float)

if c_act:
    act_raw = pd.to_numeric(df[c_act], errors="coerce").values.astype(float)
    actual_tons = act_raw * (1e8 if np.nanmax(np.abs(act_raw)) < 1e6 else 1.0)
else:
    actual_tons = np.full_like(years_input, np.nan, dtype=float)

x_his = years_input.astype(float)
YEAR_MIN = int(years_input.min())
YEAR_MAX = max(int(years_input.max()), 2100)


WEI_SHAPE = P["WEI_SHAPE"]
WEI_SCALE = P["WEI_SCALE"]
MAX_LIFE = P["MAX_LIFE"]
WARMUP_YEARS = 2 * MAX_LIFE

years_ext = np.arange(YEAR_MIN - WARMUP_YEARS, YEAR_MAX + 1)
yrs_future = np.arange(YEAR_MIN, YEAR_MAX + 1)

p_pop = fit_hyper(x_his, pop_raw)
p_pca = fit_hyper(x_his, pca_raw)

pop_fit_future = hyper_inc(yrs_future, *p_pop)
pca_fit_future = hyper_inc(yrs_future, *p_pca)

amap_pop = dict(zip(years_input, pop_raw))
amap_pca = dict(zip(years_input, pca_raw))
for i, y in enumerate(yrs_future):
    if y in amap_pop:
        pop_fit_future[i] = amap_pop[y]
    if y in amap_pca:
        pca_fit_future[i] = amap_pca[y]

g_pop_back = safe_cagr(pop_raw, yrs=6, lo=-0.015, hi=0.015, default=0.007)
g_pca_back = safe_cagr(pca_raw, yrs=6, lo=-0.02, hi=0.02, default=0.004)

pop0, pca0 = pop_fit_future[0], pca_fit_future[0]
years_past = np.arange(YEAR_MIN - WARMUP_YEARS, YEAR_MIN)
pop_back = np.array([pop0 * (1 - g_pop_back) ** (YEAR_MIN - y) for y in years_past])
pca_back = np.array([pca0 * (1 - g_pca_back) ** (YEAR_MIN - y) for y in years_past])

pop_fit_ext = np.r_[pop_back, pop_fit_future]
pca_fit_ext = np.r_[pca_back, pca_fit_future]

stock_ext_baseline = pop_fit_ext * pca_fit_ext
life = np.arange(1, MAX_LIFE + 1)
pdf_weib = weibull_min.pdf(life, c=WEI_SHAPE, scale=WEI_SCALE)
pdf_weib = pdf_weib / pdf_weib.sum()

inflow_ext_base = np.zeros_like(stock_ext_baseline)
outflow_ext_base = np.zeros_like(stock_ext_baseline)
for t in range(1, len(years_ext)):

    inflow_ext_base[t] = max(stock_ext_baseline[t] - stock_ext_baseline[t-1], 0.0)

sel = years_ext >= YEAR_MIN
years = years_ext[sel].astype(int)
stock_target = stock_ext_baseline[sel]
inflow_base = inflow_ext_base[sel]
outflow_base = outflow_ext_base[sel]


r_arr = np.zeros_like(years, dtype=float)
CI_new = np.array([cement_intensity(int(y)) for y in years], float)
K_arr = np.array([nonres_infra_multiplier(int(y)) for y in years], float)

alpha_res = CI_new.copy()
struct_base = inflow_base * alpha_res * K_arr


if np.isfinite(actual_tons).any():
    amap = {int(y): v for y, v in zip(years_input, actual_tons)}
    actual_axis = np.array([amap.get(int(y), np.nan) for y in years], float)
else:
    actual_axis = np.full_like(years, np.nan, float)


mask_beta = (
    np.isfinite(actual_axis) & np.isfinite(struct_base) &
    (years >= FIT_START) & (years <= FIT_END)
)
if mask_beta.any():
    beta_ls = float(
        np.dot(struct_base[mask_beta], actual_axis[mask_beta]) /
        max(np.dot(struct_base[mask_beta], struct_base[mask_beta]), 1e-9)
    )
else:
    beta_ls = 1.0
beta_ls = float(np.clip(beta_ls, 0.50, 1.80))

base_after_beta = struct_base * beta_ls

cal_hist = build_smooth_history_calibration(
    years=years,
    struct_base=base_after_beta,
    actual_axis=actual_axis,
    fit_start=FIT_START,
    fit_end=FIT_END,
    cal_min=CAL_FACTOR_MIN,
    cal_max=CAL_FACTOR_MAX,
    win_median=HIST_SMOOTH_WIN1,
    win_mean=HIST_SMOOTH_WIN2,
    future_tau=FUTURE_CAL_TAU
)

alpha_total_fit_base = alpha_res * K_arr * beta_ls * cal_hist
total_cement_fit_base = inflow_base * alpha_total_fit_base

if USE_RESIDUAL_BRIDGE:
    resid_logcorr = build_smoothed_residual_bridge(
        years=years,
        actual_axis=actual_axis,
        base_pred=total_cement_fit_base,
        fit_start=FIT_START,
        fit_end=FIT_END,
        forecast_start=FORECAST_START,
        alpha=RESID_ALPHA,
        blend=RESID_BLEND,
        clip=RESID_CLIP,
        decay_tau=RESID_DECAY_TAU,
        persist=RESID_PERSIST,
        long_win=RESID_LONG_WIN
    )
else:
    resid_logcorr = np.zeros_like(years, dtype=float)

total_cement_fit = total_cement_fit_base * np.exp(resid_logcorr)

if USE_HISTORY_ANTIZIGZAG:
    total_cement_fit = apply_history_antizigzag(
        years=years,
        y=total_cement_fit,
        fit_start=FIT_START,
        fit_end=FIT_END,
        lam=HIST_ANTIZIGZAG_LAMBDA
    )

alpha_total_fit = total_cement_fit / np.maximum(inflow_base, 1e-9)


total_cement_baseline_for_forecast = total_cement_fit.copy()

AUTO_PEAK_YEAR = infer_auto_peak_year(
    years,
    total_cement_baseline_for_forecast,
    alpha_total_fit,
    inflow_base=inflow_base,
    outflow=None
)
print(f"[Auto peak detection] Peak year identified by model: {AUTO_PEAK_YEAR}")

total_cement_forecast_shaped = enforce_prepeak_rise_postpeak_decline(
    total_cement_baseline_for_forecast,
    years,
    peak_year=AUTO_PEAK_YEAR
)


total_cement_hybrid = total_cement_fit.copy()
fut_mask = years >= FORECAST_START
total_cement_hybrid[fut_mask] = total_cement_forecast_shaped[fut_mask]

if FIT_END in set(years) and FORECAST_START in set(years):
    i_hist = int(np.where(years == FIT_END)[0][0])
    i_fut = int(np.where(years == FORECAST_START)[0][0])

    prev_val = float(total_cement_hybrid[i_hist])
    cur_val = float(total_cement_hybrid[i_fut])

    if cur_val > 0:
        scale0 = prev_val / cur_val
        for t in range(i_fut, len(years)):
            dt = years[t] - FORECAST_START
            fade = np.exp(-dt / 8.0)
            adj = 1.0 + (scale0 - 1.0) * fade
            total_cement_hybrid[t] *= adj

tmp = enforce_prepeak_rise_postpeak_decline(
    total_cement_hybrid.copy(),
    years,
    peak_year=AUTO_PEAK_YEAR
)
total_cement_hybrid[fut_mask] = tmp[fut_mask]


total_cement_hybrid = _shared_history('india', 'minimum', years, total_cement_hybrid, actual_axis)

mask_axis = np.isfinite(actual_axis) if c_act else np.zeros_like(years, dtype=bool)
rho_ar1 = ar1_rho_from_residuals(total_cement_hybrid, actual_axis, mask_axis, clip=0.95)


fit_total = calc_fit_metrics(
    y_true=actual_axis,
    y_pred=total_cement_hybrid,
    years_arr=years,
    start_year=FIT_START,
    end_year=FIT_END
)

tail_total = calc_tail_metrics(
    years_arr=years,
    series=total_cement_hybrid,
    tail_start=TAIL_START,
    tail_end=TAIL_END
)

def _fmt(x, nd=2):
    return "NA" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.{nd}f}"

print("\n================ HYBRID 总指标输出（India） ================\n")
print(f"[PRESET] {PRESET}")
print(f"beta_ls = {_fmt(beta_ls, 4)}")
print(f"Residual AR(1) rho = {_fmt(rho_ar1, 4)}")
print(f"Auto peak year = {AUTO_PEAK_YEAR}")

print("\n【总拟合段 1990–2024】")
print(f"n_obs = {_fmt(fit_total['n_obs'],0)}")
print(f"MAPE = {_fmt(fit_total['MAPE_pct'])}%")
print(f"MAE  = {_fmt(fit_total['MAE_tons']/1e6)} Mt")
print(f"RMSE = {_fmt(fit_total['RMSE_tons']/1e6)} Mt")
print(f"R2   = {_fmt(fit_total['R2'],4)}")
print(f"DW   = {_fmt(fit_total['DW'],4)}")

print(f"\n【总预测段尾部 {TAIL_START}–{TAIL_END}】")
print(f"Tail Slope = {_fmt(tail_total['Tail_Slope_pct_per_year'],4)}%/year")
print(f"Tail R2    = {_fmt(tail_total['Tail_R2'],4)}")
print(f"Drop       = {_fmt(tail_total['Drop'] * 100,2)}%")
print("\n===========================================================\n")


params_df = pd.DataFrame({
    "key": [
        "PRESET","FORECAST_START","FIT_START","FIT_END","TAIL_START","TAIL_END",
        "CAL_FACTOR_MIN","CAL_FACTOR_MAX","HIST_SMOOTH_WIN1","HIST_SMOOTH_WIN2","FUTURE_CAL_TAU",
        "RESID_ALPHA","RESID_BLEND","RESID_CLIP","RESID_DECAY_TAU","RESID_PERSIST","RESID_LONG_WIN",
        "USE_HISTORY_ANTIZIGZAG","HIST_ANTIZIGZAG_LAMBDA",
        "TAU_CAL","TAPER_SHIFT","CAP_START","MAX_DROP","ACCEL_CAP","BLEND",
        "WEI_SHAPE","WEI_SCALE","MAX_LIFE",
        "YEAR_MIN","YEAR_MAX","beta_ls","rho_ar1","AUTO_PEAK_YEAR"
    ],
    "value": [
        PRESET, FORECAST_START, FIT_START, FIT_END, TAIL_START, TAIL_END,
        CAL_FACTOR_MIN, CAL_FACTOR_MAX, HIST_SMOOTH_WIN1, HIST_SMOOTH_WIN2, FUTURE_CAL_TAU,
        RESID_ALPHA, RESID_BLEND, RESID_CLIP, RESID_DECAY_TAU, RESID_PERSIST, RESID_LONG_WIN,
        USE_HISTORY_ANTIZIGZAG, HIST_ANTIZIGZAG_LAMBDA,
        P["TAU_CAL"], P["TAPER_SHIFT"], P["CAP_START"], P["MAX_DROP"], P["ACCEL_CAP"], P["BLEND"],
        P["WEI_SHAPE"], P["WEI_SCALE"], P["MAX_LIFE"],
        YEAR_MIN, YEAR_MAX, beta_ls, rho_ar1, AUTO_PEAK_YEAR
    ]
})

timeline_df = pd.DataFrame({
    "Year": to_Int64(years),
    "stock_target": to_float(stock_target, 2),
    "inflow_base": to_float(inflow_base, 2),
    "outflow_base": to_float(outflow_base, 2),
    "retrofit_share": to_float(r_arr, 4),
    "CI_new": to_float(CI_new, 4),
    "K_nonres": to_float(K_arr, 4),
    "cal_hist": to_float(cal_hist, 6),
    "resid_logcorr": to_float(resid_logcorr, 6),
    "alpha_total_fit": to_float(alpha_total_fit, 6),
    "cement_fit_base_before_resid_bridge": to_float(total_cement_fit_base, 2),
    "cement_fit_before_shape": to_float(total_cement_fit, 2),
    "cement_forecast_shaped": to_float(total_cement_forecast_shaped, 2),
    "cement_hybrid_final": to_float(total_cement_hybrid, 2),
    "actual_tons": to_float(actual_axis, 2)
})

fit_metrics_df = pd.DataFrame([{
    "FitStart": FIT_START,
    "FitEnd": FIT_END,
    "n_obs": fit_total["n_obs"],
    "MAPE_pct": fit_total["MAPE_pct"],
    "MAE_tons": fit_total["MAE_tons"],
    "MAE_Mt": fit_total["MAE_tons"] / 1e6 if np.isfinite(fit_total["MAE_tons"]) else np.nan,
    "RMSE_tons": fit_total["RMSE_tons"],
    "RMSE_Mt": fit_total["RMSE_tons"] / 1e6 if np.isfinite(fit_total["RMSE_tons"]) else np.nan,
    "R2": fit_total["R2"],
    "DW": fit_total["DW"]
}])

tail_metrics_df = pd.DataFrame([{
    "TailStart": TAIL_START,
    "TailEnd": TAIL_END,
    "tail_n": tail_total["tail_n"],
    "Tail_Slope_pct_per_year": tail_total["Tail_Slope_pct_per_year"],
    "Tail_R2": tail_total["Tail_R2"],
    "Drop": tail_total["Drop"],
    "Drop_pct": tail_total["Drop"] * 100 if np.isfinite(tail_total["Drop"]) else np.nan,
    "Tail_Start_Mt": tail_total["Tail_Start_Mt"],
    "Tail_End_Mt": tail_total["Tail_End_Mt"]
}])

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
    params_df.to_excel(w, sheet_name="Params", index=False)
    timeline_df.to_excel(w, sheet_name="Timeline_Hybrid", index=False)
    fit_metrics_df.to_excel(w, sheet_name="Fit_Metrics", index=False)
    tail_metrics_df.to_excel(w, sheet_name="Tail_Metrics", index=False)

print(f"✅ 已导出融合结果到：{OUT_XLSX}")


plt.figure(figsize=(12, 7))

plt.plot(years, total_cement_fit / 1e6, "C0--", linewidth=1.5, label="History fit")
plt.plot(years, total_cement_forecast_shaped / 1e6, "C1--", linewidth=1.5, label="Forecast shape")
plt.plot(years, total_cement_hybrid / 1e6, "r.-", linewidth=1.8, label="Hybrid final")

if np.isfinite(actual_axis).any():
    plt.plot(
        years[np.isfinite(actual_axis)],
        actual_axis[np.isfinite(actual_axis)] / 1e6,
        "ks-",
        label="Actual (Mt)"
    )

plt.axvspan(FORECAST_START, 2100, color="gray", alpha=0.08, label="Forecast window")
plt.axvline(AUTO_PEAK_YEAR, color="purple", linestyle="--", alpha=0.8, label=f"Auto peak year = {AUTO_PEAK_YEAR}")
plt.xlabel("Year")
plt.ylabel("Cement demand / production (million tons)")
plt.title("India – BAU stock-driven cement demand (Hybrid final, no retrofit)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"✅ 图像已保存到：{OUT_PNG}")
plt.show()
