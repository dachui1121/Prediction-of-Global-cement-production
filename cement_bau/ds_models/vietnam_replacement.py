from cement_bau.runtime import input_path as _input_path, output_dir as _output_dir
import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime

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


DATA_PATH = str(_input_path('vietnam'))

SHEET_NAME = "Sheet1"

START_YEAR_DISPLAY = 1989
END_YEAR_DISPLAY   = 2100
FORECAST_END       = 2100

SCENARIO = "central"

DEMAND_CASE = "replacement"
FORECAST_BRANCH_YEAR = 2025
FORECAST_BRIDGE_END = 2030


ACTUAL_START_YEAR = 1989
ACTUAL_END_YEAR   = 2024
ACTUAL_COL_NAME   = "水泥产量"


FIT_EVAL_START = 1990
FIT_EVAL_END   = 2024

PRED_EVAL_START = 2025
PRED_EVAL_END   = 2100


MAX_LIFE = 100
infra_grid   = np.array([2.4], dtype=float)
lambda_grid  = np.array([56.4], dtype=float)
k_grid       = [1.97]


BUFFER_RELEASE = 1.00
BUFFER_START_YEAR = 2025


USE_INFLOW_CAP  = True
INFLOW_CAP_FROM = 2024
INFLOW_CAP_UP   = 0.10
INFLOW_CAP_DOWN = 0.10


EARLY_RISE_GUARD_ENABLE = False
EARLY_RISE_START_YEAR   = 2025
EARLY_RISE_END_YEAR     = 2028
EARLY_RISE_MIN_YOY      = 0.010


MAPE_MAX = 10.0
DW_MIN, DW_MAX = 1.5, 2.5
DROP_MIN = -45.0


TAIL_FIX_ENABLE = False
TAIL_FIX_START  = 2050
TAIL_FIX_END    = 2100
TAIL_TARGET_DROP_PCT = -15.0


PEAK_KILL_ENABLE = False
PEAK_KILL_START_YEAR = 2030
PEAK_LOOK = 6
PEAK_HEIGHT = 0.05


SMOOTH_LAMBDA_GRID = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.50, 1.0, 2.0, 5.0, 20.0]
SLOPE_LAMBDA_GRID  = [0.0, 0.2, 0.5, 1.0, 2.0, 8.0, 20.0]
RHO_GRID = np.round(np.linspace(-0.95, 0.95, 191), 3)


OBJ_W_MAPE = 1.00
OBJ_W_DW   = 2.50
OBJ_W_RMSE = 0.15


FUTURE_MULT_BLEND_YEARS = 12


CALIBRATION_STRENGTH = 0.91


SAVE_PLOTS = True
FIG_DPI    = 160

if HAS_MATPLOTLIB:
    plt.rcParams["font.family"] = ["Times New Roman"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["legend.loc"] = "best"


def to_float(x):
    try:
        return float(str(x).replace(",", "").replace("，", "").strip())
    except:
        return np.nan

def pct(arr, p):
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return float(np.percentile(a, p))

def ema(x, alpha):
    x = np.asarray(x, float)
    if len(x) == 0:
        return x
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

def centered_ma(x, win=3):
    x = np.asarray(x, float)
    if win <= 1:
        return x.copy()
    pad = win // 2
    xx = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(win) / win
    return np.convolve(xx, ker, mode="valid")

def extend_cagr_forward(map_year_to_val, y_end, damp=0.98, window=5):
    ys = sorted(map_year_to_val.keys())
    last_y = ys[-1]
    last_v = float(map_year_to_val[last_y])

    use = ys[-min(window, len(ys)):]
    if len(use) >= 2 and map_year_to_val[use[0]] > 0 and map_year_to_val[use[-1]] > 0:
        n = use[-1] - use[0]
        cagr = (map_year_to_val[use[-1]] / map_year_to_val[use[0]]) ** (1 / max(n, 1)) - 1.0
    else:
        cagr = 0.0

    for y in range(last_y + 1, y_end + 1):
        k = y - last_y
        g = 1.0 + cagr * (damp ** (k - 1))
        last_v = max(last_v * g, 1e-9)
        map_year_to_val[y] = float(last_v)
    return map_year_to_val

def cap_yoy(series, years, start_year, up=0.08, down=0.06, eps=1e-12):
    s = np.asarray(series, float).copy()
    years = np.asarray(years, int)
    if start_year not in set(years):
        return s
    i0 = int(np.where(years == start_year)[0][0])
    for t in range(i0 + 1, len(years)):
        prev = max(s[t-1], eps)
        lo = prev * (1.0 - down)
        hi = prev * (1.0 + up)
        s[t] = float(np.clip(s[t], lo, hi))
    return s

def weibull_cdfdiff_kernel(max_life, k, lam):
    ages = np.arange(1, max_life + 1)
    lower = np.clip(ages - 0.5, 0.0, None)
    upper = ages + 0.5
    if weibull_min is not None:
        cdf = weibull_min.cdf
        p = cdf(upper, c=k, scale=lam) - cdf(lower, c=k, scale=lam)
        p[0] = cdf(1.0, c=k, scale=lam) - cdf(0.0, c=k, scale=lam)
    else:
        def _cdf(x):
            x = np.asarray(x, dtype=float)
            x = np.clip(x, 0.0, None)
            return 1.0 - np.exp(-((x / lam) ** k))
        p = _cdf(upper) - _cdf(lower)
        p[0] = float(_cdf(1.0) - _cdf(0.0))
    p = np.nan_to_num(p)
    s = float(p.sum())
    return p / s if s > 0 else np.ones_like(p) / len(p)

def rmse_for_grid(pred, actual, mask):
    if not np.any(mask):
        return np.inf
    d = pred[mask] - actual[mask]
    return float(np.sqrt(np.mean(d * d)))

def strict_fit_metrics(y_pred, y_true, years_arr, start_year=None, end_year=None):
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    years_arr = np.asarray(years_arr, dtype=int)

    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if start_year is not None:
        mask &= (years_arr >= start_year)
    if end_year is not None:
        mask &= (years_arr <= end_year)

    yp = y_pred[mask]
    yt = y_true[mask]

    if yp.size == 0:
        return {
            "MAE": np.nan, "MAPE": np.nan, "RMSE": np.nan,
            "Bias": np.nan, "R2": np.nan, "DW": np.nan, "N": 0
        }

    e = yt - yp
    ae = np.abs(e)
    se = e ** 2

    mae = float(np.mean(ae))
    rmse = float(np.sqrt(np.mean(se)))
    bias = float(np.mean(yp - yt))

    yt_mean = float(np.mean(yt))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt_mean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    nonzero_mask = (yt != 0)
    if np.any(nonzero_mask):
        mape = float(np.mean(np.abs((yt[nonzero_mask] - yp[nonzero_mask]) / yt[nonzero_mask])) * 100.0)
    else:
        mape = np.nan

    den_dw = float(np.sum(e ** 2))
    if yp.size >= 2 and den_dw > 0:
        dw = float(np.sum((e[1:] - e[:-1]) ** 2) / den_dw)
    else:
        dw = np.nan

    return {
        "MAE": mae, "MAPE": mape, "RMSE": rmse,
        "Bias": bias, "R2": r2, "DW": dw, "N": int(yp.size)
    }

def tail_slope(years_arr, series, a, b):
    m = (years_arr >= a) & (years_arr <= b) & np.isfinite(series)
    if np.sum(m) < 3:
        return dict(slope_ton_per_year=np.nan, slope_pct_per_year=np.nan, R2=np.nan)
    x = years_arr[m].astype(float)
    y = series[m].astype(float)
    k, b0 = np.polyfit(x, y, 1)
    slope_pct = 100.0 * k / max(np.mean(y), 1e-9)
    ybar = np.mean(y)
    ss_tot = np.sum((y - ybar) ** 2)
    ss_res = np.sum((y - (k * x + b0)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(slope_ton_per_year=float(k), slope_pct_per_year=float(slope_pct), R2=float(r2))

def pick_year_value(years_arr, series, yy):
    m = (years_arr == yy) & np.isfinite(series)
    return float(series[m][0]) if np.any(m) else np.nan

def apply_tail_fix(series, years_arr, y_start=2050, y_end=2100, target_drop_pct=-15.0):
    years_arr = np.asarray(years_arr, int)
    s = np.asarray(series, float).copy()
    if (y_start not in set(years_arr)) or (y_end not in set(years_arr)):
        return s, 1.0

    i0 = int(np.where(years_arr == y_start)[0][0])
    i1 = int(np.where(years_arr == y_end)[0][0])

    y0 = float(s[i0])
    y1 = float(s[i1])
    if not (np.isfinite(y0) and y0 > 0 and np.isfinite(y1) and y1 > 0):
        return s, 1.0

    target_y1 = y0 * (1.0 + target_drop_pct / 100.0)
    if target_y1 <= 0:
        return s, 1.0

    if y1 >= target_y1:
        return s, 1.0

    ratio = target_y1 / y1
    logR = np.log(ratio)
    for i in range(i0, i1 + 1):
        w = (years_arr[i] - y_start) / max(1, (y_end - y_start))
        s[i] = s[i] * np.exp(logR * w)

    return s, ratio

def remove_sharp_peaks_logbridge_global(y, years, start_year=2030, look=6, height=0.05, eps=1e-9):
    y = np.asarray(y, float).copy()
    years = np.asarray(years, int)
    n = len(years)

    start_idx = 0
    if start_year in set(years):
        start_idx = int(np.where(years == start_year)[0][0])

    t = max(1, start_idx)
    while t < n - 1:
        L = max(start_idx, t - look)
        R = min(n - 1, t + look)
        seg = y[L:R + 1]
        p_idx = int(np.argmax(seg)) + L
        if (p_idx > L) and (p_idx < R):
            yl, yp, yr = y[L], y[p_idx], y[R]
            if yl > 0 and yp > 0 and yr > 0:
                cond_left = (yp / yl) > (1.0 + height)
                cond_right = (yp / yr) > (1.0 + height)
                if cond_left and cond_right:
                    log_yl, log_yr = np.log(max(yl, eps)), np.log(max(yr, eps))
                    for idx in range(L, R + 1):
                        w = (idx - L) / max(1, (R - L))
                        y[idx] = float(np.exp((1 - w) * log_yl + w * log_yr))
                    t = R
                    continue
        t += 1
    return y

def total_drop_pct(years_arr, series, start_year, end_year):
    y0 = pick_year_value(years_arr, series, start_year)
    y1 = pick_year_value(years_arr, series, end_year)
    if np.isfinite(y0) and y0 > 0 and np.isfinite(y1):
        return 100.0 * (y1 / y0 - 1.0)
    return np.nan


def apply_early_rise_guard(series, years_arr, actual_series=None, start_year=2025, end_year=2028, min_yoy=0.01):


    s = np.asarray(series, float).copy()
    years_arr = np.asarray(years_arr, int)
    actual_series = None if actual_series is None else np.asarray(actual_series, float)

    for y in range(start_year, end_year + 1):
        idx = np.where(years_arr == y)[0]
        prev_idx = np.where(years_arr == y - 1)[0]
        if idx.size == 0 or prev_idx.size == 0:
            continue
        i = int(idx[0])
        j = int(prev_idx[0])
        if not (np.isfinite(s[i]) and np.isfinite(s[j]) and s[j] > 0):
            continue

        prev_anchor = float(s[j])
        if y == start_year and actual_series is not None and j < len(actual_series):
            if np.isfinite(actual_series[j]) and actual_series[j] > 0:
                prev_anchor = max(prev_anchor, float(actual_series[j]))

        floor_val = prev_anchor * (1.0 + min_yoy)
        if s[i] < floor_val:
            s[i] = floor_val
    return s


def apply_forecast_bridge(series, years_arr, actual_series, start_anchor_year=2024, bridge_end_year=2030):


    s_raw = np.asarray(series, float).copy()
    s = s_raw.copy()
    years_arr = np.asarray(years_arr, int)
    actual_series = np.asarray(actual_series, float)

    idx0 = np.where(years_arr == start_anchor_year)[0]
    if idx0.size == 0:
        return s, np.zeros_like(s, dtype=float), np.nan, s_raw

    i0 = int(idx0[0])
    if i0 < len(actual_series) and np.isfinite(actual_series[i0]) and actual_series[i0] > 0:
        anchor = float(actual_series[i0])
    else:
        anchor = float(s_raw[i0])

    w_all = np.ones_like(s, dtype=float)
    w_all[years_arr <= start_anchor_year] = 0.0

    m = (years_arr > start_anchor_year) & (years_arr <= bridge_end_year)
    if np.any(m):
        x = (years_arr[m].astype(float) - float(start_anchor_year)) / max(
            float(bridge_end_year - start_anchor_year), 1.0
        )
        x = np.clip(x, 0.0, 1.0)
        w = x * x * (3.0 - 2.0 * x)
        w_all[m] = w
        s[m] = (1.0 - w) * anchor + w * s_raw[m]

    return s, w_all, anchor, s_raw

def _fmt(x, nd=2):
    return "NA" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.{nd}f}"


def fit_smooth_log_multiplier(base, actual, years, fit_start, fit_end, lam2=100.0, lam1=0.0):
    base = np.asarray(base, float)
    actual = np.asarray(actual, float)
    years = np.asarray(years, int)

    fit_mask = (
        (years >= fit_start) & (years <= fit_end) &
        np.isfinite(base) & np.isfinite(actual) &
        (base > 0) & (actual > 0)
    )
    idx = np.where(fit_mask)[0]
    n = len(idx)
    if n == 0:
        return np.zeros(0), idx

    z = np.log(actual[idx] / base[idx])

    I = np.eye(n)


    if n >= 2:
        D1 = np.zeros((n - 1, n))
        for i in range(n - 1):
            D1[i, i] = -1.0
            D1[i, i + 1] = 1.0
    else:
        D1 = np.zeros((0, n))


    if n >= 3:
        D2 = np.zeros((n - 2, n))
        for i in range(n - 2):
            D2[i, i] = 1.0
            D2[i, i + 1] = -2.0
            D2[i, i + 2] = 1.0
    else:
        D2 = np.zeros((0, n))

    A = I.copy()
    b = z.copy()

    if D1.shape[0] > 0 and lam1 > 0:
        A = np.vstack([A, np.sqrt(lam1) * D1])
        b = np.concatenate([b, np.zeros(D1.shape[0])])

    if D2.shape[0] > 0 and lam2 > 0:
        A = np.vstack([A, np.sqrt(lam2) * D2])
        b = np.concatenate([b, np.zeros(D2.shape[0])])

    g, *_ = np.linalg.lstsq(A, b, rcond=None)


    g = np.clip(g, np.log(0.35), np.log(4.50))
    return g, idx

def build_multiplier_full_from_logfit(base, years, fit_idx, g_fit, future_blend_years=12):
    years = np.asarray(years, int)
    mult = np.ones(len(years), float)

    if len(fit_idx) == 0:
        return mult

    mult_fit = np.exp(g_fit)
    mult[fit_idx] = mult_fit

    last_fit_idx = fit_idx[-1]
    last_fit_year = years[last_fit_idx]


    recent_n = min(5, len(mult_fit))
    long_anchor = float(np.mean(mult_fit[-recent_n:]))
    short_anchor = float(mult_fit[-1])

    for i in range(last_fit_idx + 1, len(years)):
        dt = years[i] - last_fit_year
        w = min(max(dt / max(future_blend_years, 1), 0.0), 1.0)
        mult[i] = (1 - w) * short_anchor + w * long_anchor


    first_fit_idx = fit_idx[0]
    if first_fit_idx > 0:
        mult[:first_fit_idx] = mult_fit[0]

    return mult


def apply_rho_whiten(y_base, y_true, years, fit_start, fit_end, rho):
    y_base = np.asarray(y_base, float)
    y_true = np.asarray(y_true, float)
    years = np.asarray(years, int)

    out = np.zeros_like(y_base, float)
    fit_mask = (
        (years >= fit_start) & (years <= fit_end) &
        np.isfinite(y_base) & np.isfinite(y_true)
    )

    u_prev = 0.0
    for t in range(len(y_base)):
        out[t] = y_base[t] + rho * u_prev
        if fit_mask[t]:
            u_prev = y_true[t] - y_base[t]
        else:
            u_prev = rho * u_prev
    return out

def auto_tune_rho(y_base, y_true, years, fit_start, fit_end, rho_grid):
    best_score = np.inf
    best_rho = 0.0
    best_pred = y_base.copy()
    best_m = strict_fit_metrics(y_base, y_true, years, fit_start, fit_end)

    for rho in rho_grid:
        yp = apply_rho_whiten(y_base, y_true, years, fit_start, fit_end, rho)
        m = strict_fit_metrics(yp, y_true, years, fit_start, fit_end)

        if not np.isfinite(m["MAPE"]) or not np.isfinite(m["DW"]):
            continue


        score = (
            OBJ_W_MAPE * m["MAPE"] +
            OBJ_W_DW   * abs(m["DW"] - 2.0) * 10.0 +
            OBJ_W_RMSE * (m["RMSE"] / 1e6)
        )


        if m["MAPE"] > MAPE_MAX:
            score += 30.0 * (m["MAPE"] - MAPE_MAX)
        if not (DW_MIN <= m["DW"] <= DW_MAX):
            if m["DW"] < DW_MIN:
                score += 40.0 * (DW_MIN - m["DW"])
            else:
                score += 40.0 * (m["DW"] - DW_MAX)

        if score < best_score:
            best_score = score
            best_rho = float(rho)
            best_pred = yp.copy()
            best_m = m.copy()

    return best_rho, best_pred, best_m, best_score


print(f"✅ Using data file: {DATA_PATH}")
raw = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, engine="openpyxl").copy()
raw.columns = raw.columns.astype(str).str.strip()

if "年份" in raw.columns:
    raw["年份"] = raw["年份"].apply(to_float).astype(int)
else:
    raw["年份"] = np.arange(ACTUAL_START_YEAR, ACTUAL_START_YEAR + len(raw), dtype=int)

raw = raw.sort_values("年份").reset_index(drop=True)

need_cols = ["城镇人口数量", "拟合人均住宅面积", ACTUAL_COL_NAME]
for c in need_cols:
    if c not in raw.columns:
        raise KeyError(f"缺少列：{c}；当前列：{raw.columns.tolist()}")

raw["城镇人口数量"] = raw["城镇人口数量"].apply(to_float).astype(float)
raw["拟合人均住宅面积"] = raw["拟合人均住宅面积"].apply(to_float).astype(float)
actual_tons = pd.to_numeric(raw[ACTUAL_COL_NAME], errors="coerce").values.astype(float)

y0 = int(min(ACTUAL_START_YEAR, raw["年份"].min()))
years_full = np.arange(y0, FORECAST_END + 1, dtype=int)

pop_map = dict(zip(raw["年份"].values, raw["城镇人口数量"].values))
pop_map = extend_cagr_forward(pop_map, FORECAST_END, damp=0.98, window=5)
population = np.array([pop_map.get(int(y), np.nan) for y in years_full], float)


orig_map_pca = dict(zip(raw["年份"].values, raw["拟合人均住宅面积"].values))
pc_area = np.array([orig_map_pca.get(int(y), np.nan) for y in years_full], dtype=float)


missing_pca_years = years_full[~np.isfinite(pc_area)]
if missing_pca_years.size > 0:
    raise ValueError(
        "数据集中的‘拟合人均住宅面积’存在缺失年份，"
        f"请先在 Excel 中补齐这些年份：{missing_pca_years.tolist()}"
    )

pc_area = np.clip(pc_area, 0.0, 60.0)


amap = dict(zip(raw["年份"].values, actual_tons))
actual_all = np.array([amap.get(int(y), np.nan) for y in years_full], float)
actual_mask = np.isfinite(actual_all)
have_actual = bool(np.any(actual_mask))

print(f"✅ Actual column: {ACTUAL_COL_NAME} (tons). Have actual points? {have_actual}")


def seg_val(y, pts):
    if y <= pts[0][0]:
        return float(pts[0][1])
    if y >= pts[-1][0]:
        return float(pts[-1][1])
    for (ya, va), (yb, vb) in zip(pts[:-1], pts[1:]):
        if ya <= y <= yb:
            t = (y - ya) / max(yb - ya, 1e-9)
            return float(va + t * (vb - va))
    return float(pts[-1][1])

def ci_res_vn(y: int) -> float:

    return 0.2

def retrofit_share_vn(y):

    return 0.0

def infra_taper_vn(y: int) -> float:

    return 1.0

def export_cap_share_vn(y: int) -> float:

    return 0.0


def compute_series(k_shape, lambda_scale, max_life, infra_factor_base):
    pdf_w = weibull_cdfdiff_kernel(max_life, k_shape, lambda_scale)

    stock_target = population * pc_area
    stock_hat = np.zeros_like(stock_target)
    inflow = np.zeros_like(stock_target)
    outflow = np.zeros_like(stock_target)

    stock_hat[0] = stock_target[0]

    steady_in = max(stock_target[0] / max(lambda_scale, 1.0), 1e-6)
    hist_in = np.full(max_life, steady_in)

    buffer = 0.0

    for t in range(len(stock_target)):
        past = np.concatenate([hist_in, inflow[:t]])
        use = past[-max_life:]
        outflow[t] = float(np.sum(use[::-1] * pdf_w[:len(use)]))

        if t == 0:
            expansion_need = 0.0
        else:
            gap = stock_target[t] - stock_hat[t-1]
            expansion_need = max(gap, 0.0)


        repl = outflow[t]

        inflow[t] = max(expansion_need + repl, 0.0)

        if t > 0:
            stock_hat[t] = stock_hat[t-1] + inflow[t] - outflow[t]


    inflow_s = inflow.copy()

    if USE_INFLOW_CAP:
        inflow_s = cap_yoy(
            inflow_s, years_full, start_year=INFLOW_CAP_FROM,
            up=INFLOW_CAP_UP, down=INFLOW_CAP_DOWN
        )

    cement_base = np.zeros_like(stock_target)
    for i, y in enumerate(years_full):

        eff_in = inflow_s[i]
        cr     = ci_res_vn(int(y))
        infra_rate = infra_factor_base * infra_taper_vn(int(y))
        base_no_export = eff_in * cr * (1.0 + infra_rate)
        share = export_cap_share_vn(int(y))
        cement_base[i] = base_no_export * (1.0 + share)

    return {
        "stock_target": stock_target,
        "stock_hat": stock_hat,
        "inflow": inflow,
        "outflow": outflow,
        "inflow_s": inflow_s,
        "cement_base": cement_base
    }


print(f"[{SCENARIO}] Grid search ... max_life={MAX_LIFE} | buffer_release={BUFFER_RELEASE}")
best_rmse, best_combo, best_res = np.inf, None, None
t0 = datetime.now()

for k_shape in k_grid:
    for lam in lambda_grid:
        for infra in infra_grid:
            try:
                res = compute_series(k_shape, lam, max_life=MAX_LIFE, infra_factor_base=infra)
                pred = res["cement_base"]
                score = rmse_for_grid(pred, actual_all, actual_mask) if have_actual else np.nan
                if have_actual and score < best_rmse:
                    best_rmse, best_combo, best_res = score, (k_shape, lam, infra), res
            except Exception as e:
                print("Grid error:", k_shape, lam, infra, e)

t1 = datetime.now()
if best_res is None:
    best_combo = (1.97, 56.4, 2.4)
    best_res = compute_series(best_combo[0], best_combo[1], max_life=MAX_LIFE, infra_factor_base=best_combo[2])

print(f"Grid done in {t1-t0}. Best RMSE={best_rmse:.2f}, params={best_combo}")

base = best_res["cement_base"].copy()


best_hist_obj = np.inf
best_mult = np.ones_like(base)
best_logfit = None

if have_actual:
    for lam2 in SMOOTH_LAMBDA_GRID:
        for lam1 in SLOPE_LAMBDA_GRID:
            try:
                g_fit, fit_idx = fit_smooth_log_multiplier(
                    base, actual_all, years_full,
                    FIT_EVAL_START, FIT_EVAL_END,
                    lam2=lam2, lam1=lam1
                )
                mult_try = build_multiplier_full_from_logfit(
                    base, years_full, fit_idx, g_fit,
                    future_blend_years=FUTURE_MULT_BLEND_YEARS
                )
                pred_try = base * mult_try
                m_try = strict_fit_metrics(pred_try, actual_all, years_full, FIT_EVAL_START, FIT_EVAL_END)

                if not np.isfinite(m_try["MAPE"]):
                    continue


                obj = m_try["MAPE"] + 0.08 * (m_try["RMSE"] / 1e6)
                if np.isfinite(m_try["DW"]):
                    obj += 1.2 * abs(m_try["DW"] - 2.0)
                else:
                    obj += 100.0

                if obj < best_hist_obj:
                    best_hist_obj = obj
                    best_mult = mult_try.copy()
                    best_logfit = {
                        "lam2": lam2,
                        "lam1": lam1,
                        "fit_idx": fit_idx.copy(),
                        "g_fit": g_fit.copy(),
                        "metrics": m_try.copy()
                    }
            except Exception as e:
                print("smooth-fit error:", lam2, lam1, e)

mult_raw = best_mult.copy()
mult = 1.0 + CALIBRATION_STRENGTH * (mult_raw - 1.0)
final_series = base * mult


rho_hat, yhat_hist_tuned, m_fit_total_pre_tail, rho_score = auto_tune_rho(
    final_series, actual_all, years_full,
    FIT_EVAL_START, FIT_EVAL_END,
    RHO_GRID
)


pred_mask_no_rho = years_full > FIT_EVAL_END
yhat_hist_tuned[pred_mask_no_rho] = final_series[pred_mask_no_rho]


tail_fix_ratio = 1.0
if TAIL_FIX_ENABLE:
    yhat_ar1_fixed, tail_fix_ratio = apply_tail_fix(
        yhat_hist_tuned, years_full,
        y_start=TAIL_FIX_START, y_end=TAIL_FIX_END,
        target_drop_pct=TAIL_TARGET_DROP_PCT
    )
    final_series_fixed, _ = apply_tail_fix(
        final_series, years_full,
        y_start=TAIL_FIX_START, y_end=TAIL_FIX_END,
        target_drop_pct=TAIL_TARGET_DROP_PCT
    )
else:
    yhat_ar1_fixed = yhat_hist_tuned.copy()
    final_series_fixed = final_series.copy()


if PEAK_KILL_ENABLE:
    yhat_ar1_fixed = remove_sharp_peaks_logbridge_global(
        yhat_ar1_fixed, years_full,
        start_year=PEAK_KILL_START_YEAR, look=PEAK_LOOK, height=PEAK_HEIGHT
    )
    final_series_fixed = remove_sharp_peaks_logbridge_global(
        final_series_fixed, years_full,
        start_year=PEAK_KILL_START_YEAR, look=PEAK_LOOK, height=PEAK_HEIGHT
    )


yhat_ar1_fixed, forecast_bridge_weight, forecast_bridge_anchor_2024, forecast_raw_before_bridge = apply_forecast_bridge(
    yhat_ar1_fixed, years_full, actual_all,
    start_anchor_year=2024, bridge_end_year=FORECAST_BRIDGE_END
)
final_series_fixed, _, _, final_raw_before_bridge = apply_forecast_bridge(
    final_series_fixed, years_full, actual_all,
    start_anchor_year=2024, bridge_end_year=FORECAST_BRIDGE_END
)


m_fit_total = strict_fit_metrics(
    yhat_ar1_fixed, actual_all, years_full,
    FIT_EVAL_START, FIT_EVAL_END
)


pred_tail_total = tail_slope(
    years_full, yhat_ar1_fixed,
    PRED_EVAL_START, PRED_EVAL_END
)

pred_drop_total = total_drop_pct(
    years_full, yhat_ar1_fixed,
    PRED_EVAL_START, PRED_EVAL_END
)


break_2024_2025_final_pct = total_drop_pct(years_full, yhat_ar1_fixed, 2024, 2025)
break_2024_2025_base_pct = total_drop_pct(years_full, base, 2024, 2025)
break_2024_2025_inflow_pct = total_drop_pct(years_full, best_res["inflow"], 2024, 2025)
break_2024_2025_inflow_s_pct = total_drop_pct(years_full, best_res["inflow_s"], 2024, 2025)


fails = []

if have_actual and np.isfinite(m_fit_total["MAPE"]) and m_fit_total["MAPE"] > MAPE_MAX:
    fails.append(f"MAPE({FIT_EVAL_START}–{FIT_EVAL_END})={m_fit_total['MAPE']:.2f}% > {MAPE_MAX}%")

if have_actual and np.isfinite(m_fit_total["DW"]) and not (DW_MIN <= m_fit_total["DW"] <= DW_MAX):
    fails.append(f"DW({FIT_EVAL_START}–{FIT_EVAL_END})={m_fit_total['DW']:.3f} not in [{DW_MIN},{DW_MAX}]")

if np.isfinite(pred_drop_total) and (pred_drop_total < DROP_MIN):
    fails.append(f"Drop({PRED_EVAL_START}→{PRED_EVAL_END})={pred_drop_total:.1f}% < {DROP_MIN}%")

ok_all = (len(fails) == 0)

print("\n================  Reasonableness (Vietnam)  ================")
print(f"[SCENARIO] {SCENARIO} | buffer_release={BUFFER_RELEASE} | max_life={MAX_LIFE}")
print(f"[TailFix] enabled={TAIL_FIX_ENABLE} | target_drop={TAIL_TARGET_DROP_PCT}% | applied_ratio@2100={tail_fix_ratio:.3f}")
print(f"[PeakKill] enabled={PEAK_KILL_ENABLE} | look={PEAK_LOOK} | height={PEAK_HEIGHT}")

if best_logfit is not None:
    print(f"[SmoothMultiplier] lam2={best_logfit['lam2']} | lam1={best_logfit['lam1']}")

print(f"[ResidualWhiten] best_rho={rho_hat:.3f} | rho_score={rho_score:.3f}")
print(f"[CalibrationStrength] {CALIBRATION_STRENGTH:.2f}")
print(
    f"[EarlyRiseGuard] enabled={EARLY_RISE_GUARD_ENABLE} | "
    f"range={EARLY_RISE_START_YEAR}–{EARLY_RISE_END_YEAR} | "
    f"min_yoy={EARLY_RISE_MIN_YOY*100:.2f}%"
)

print(
    f"[Total Fit {FIT_EVAL_START}–{FIT_EVAL_END}] "
    f"N={m_fit_total['N']} | "
    f"MAPE={_fmt(m_fit_total['MAPE'])}% | "
    f"RMSE={_fmt(m_fit_total['RMSE'],0)} | "
    f"R2={_fmt(m_fit_total['R2'],3)} | "
    f"DW={_fmt(m_fit_total['DW'],3)} | "
    f"MAE={_fmt(m_fit_total['MAE'],0)} | "
    f"Bias={_fmt(m_fit_total['Bias'],0)}"
)

print(
    f"[Total Forecast {PRED_EVAL_START}–{PRED_EVAL_END}] "
    f"Tail Slope={_fmt(pred_tail_total['slope_pct_per_year'],2)}%/y | "
    f"Tail R2={_fmt(pred_tail_total['R2'],3)} | "
    f"Drop={_fmt(pred_drop_total,1)}%"
)

print(
    f"[Breakpoint 2024→2025] "
    f"Final={_fmt(break_2024_2025_final_pct,2)}% | "
    f"Base={_fmt(break_2024_2025_base_pct,2)}% | "
    f"Inflow(raw)={_fmt(break_2024_2025_inflow_pct,2)}% | "
    f"Inflow(capped)={_fmt(break_2024_2025_inflow_s_pct,2)}%"
)

print(f"[Rule] require MAPE < {MAPE_MAX}% and DW in [{DW_MIN}, {DW_MAX}]")
print("结论：", "✅ PASS" if ok_all else "❌ NEEDS REVIEW")
if not ok_all:
    for r in fails:
        print(" -", r)
print("============================================================\n")


diag_checks = pd.DataFrame([{
    "Scenario": SCENARIO,
    "MAX_LIFE": MAX_LIFE,
    "BUFFER_RELEASE": BUFFER_RELEASE,
    "DemandCase": DEMAND_CASE,
    "ForecastBranchYear": FORECAST_BRANCH_YEAR,
    "ForecastBridgeEnd": FORECAST_BRIDGE_END,
    "ForecastBridgeAnchor2024": forecast_bridge_anchor_2024,
    "Best_k_shape": best_combo[0],
    "Best_lambda": best_combo[1],
    "Best_infra_factor": best_combo[2],

    "FitEval_Start": FIT_EVAL_START,
    "FitEval_End": FIT_EVAL_END,
    "Hist_MAPE_Total_1990_2024": m_fit_total["MAPE"],
    "Hist_RMSE_Total_1990_2024": m_fit_total["RMSE"],
    "Hist_R2_Total_1990_2024": m_fit_total["R2"],
    "Hist_DW_Total_1990_2024": m_fit_total["DW"],
    "Hist_MAE_Total_1990_2024": m_fit_total["MAE"],
    "Hist_Bias_Total_1990_2024": m_fit_total["Bias"],
    "Hist_N_Total_1990_2024": m_fit_total["N"],

    "PredEval_Start": PRED_EVAL_START,
    "PredEval_End": PRED_EVAL_END,
    "Pred_TailSlopePct_Total_2025_2100": pred_tail_total["slope_pct_per_year"],
    "Pred_TailSlopeTonPerYear_Total_2025_2100": pred_tail_total["slope_ton_per_year"],
    "Pred_TailR2_Total_2025_2100": pred_tail_total["R2"],
    "Pred_DropPct_Total_2025_2100": pred_drop_total,
    "Pred_DropMinRule": DROP_MIN,
    "Breakpoint_2024_2025_Final_pct": break_2024_2025_final_pct,
    "Breakpoint_2024_2025_Base_pct": break_2024_2025_base_pct,
    "Breakpoint_2024_2025_InflowRaw_pct": break_2024_2025_inflow_pct,
    "Breakpoint_2024_2025_InflowCapped_pct": break_2024_2025_inflow_s_pct,

    "SmoothFit_lam2": np.nan if best_logfit is None else best_logfit["lam2"],
    "SmoothFit_lam1": np.nan if best_logfit is None else best_logfit["lam1"],
    "Calibration_Strength": CALIBRATION_STRENGTH,
    "ResidualWhiten_rho": rho_hat,

    "TailFix_enabled": TAIL_FIX_ENABLE,
    "TailFix_target_drop_pct": TAIL_TARGET_DROP_PCT,
    "TailFix_ratio_at_2100": tail_fix_ratio,
    "PeakKill_enabled": PEAK_KILL_ENABLE,
    "EarlyRiseGuard_enabled": EARLY_RISE_GUARD_ENABLE,
    "EarlyRiseGuard_start": EARLY_RISE_START_YEAR,
    "EarlyRiseGuard_end": EARLY_RISE_END_YEAR,
    "EarlyRiseGuard_min_yoy_pct": EARLY_RISE_MIN_YOY * 100.0,

    "PASS": ok_all,
    "FailReasons": "; ".join(fails) if not ok_all else ""
}])

script_dir = os.path.dirname(os.path.abspath(__file__))
desktop = str(_output_dir('vietnam', 'replacement'))
save_dir = str(_output_dir('vietnam', 'replacement'))


mask_disp = (years_full >= START_YEAR_DISPLAY) & (years_full <= END_YEAR_DISPLAY)
plot_path = os.path.join(save_dir, f"越南_DS_Replacement_2024_2030_SmoothBridge_{SCENARIO}.png")

if HAS_MATPLOTLIB:
    plt.figure(figsize=(13, 7))
    if have_actual:
        plt.plot(
            years_full[actual_mask & mask_disp],
            (actual_all[actual_mask & mask_disp] / 1e6),
            "ks-", label="Actual (Mt)"
        )

    plt.plot(years_full[mask_disp], (yhat_ar1_fixed[mask_disp] / 1e6), "ro-", label="Model final")
    plt.plot(years_full[mask_disp], (final_series_fixed[mask_disp] / 1e6), "g--", label="Calibrated no-whiten")
    plt.plot(years_full[mask_disp], (base[mask_disp] / 1e6), "b--", label="Base")
    plt.axvline(2024, linestyle="--", color="gray", label="2024 anchor")

    plt.xlabel("Year")
    plt.ylabel("Cement production (Million tons)")
    plt.title(f"Vietnam DS Replacement: shared history + 2024-2030 smooth bridge - scenario: {SCENARIO}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, edgecolor="gray")
    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(plot_path, dpi=FIG_DPI)
    plt.close()
else:
    SAVE_PLOTS = False
    print("⚠️ 当前环境未安装 matplotlib，已跳过绘图；Excel 结果仍正常导出。")


out_xlsx = os.path.join(save_dir, f"越南_DS_Replacement_2024_2030_SmoothBridge_1989_2100_{SCENARIO}.xlsx")

result_df = pd.DataFrame({
    "年份": years_full,
    "城镇人口数量": np.round(population, 0),
    "人均住宅面积": np.round(pc_area, 3),
    "stock_target": np.round(best_res["stock_target"], 3),
    "stock_hat": np.round(best_res["stock_hat"], 3),
    "inflow": np.round(best_res["inflow"], 3),
    "outflow": np.round(best_res["outflow"], 3),
    "inflow_s": np.round(best_res["inflow_s"], 3),
    "base_pred_tons": np.round(base, 0),
    "forecast_bridge_weight": np.round(forecast_bridge_weight, 6),
    "forecast_raw_before_bridge": np.round(forecast_raw_before_bridge, 0),
    "calib_multiplier": np.round(mult, 6),
    "calib_multiplier_raw": np.round(mult_raw, 6),
    "final_pred_tons_no_whiten": np.round(final_series, 0),
    "final_pred_tons_no_whiten_fixed": np.round(final_series_fixed, 0),
    "final_pred_tons_final": np.round(yhat_ar1_fixed, 0),
    "final_yoy_pct": np.round(pd.Series(yhat_ar1_fixed).pct_change().values * 100.0, 3),
    "base_yoy_pct": np.round(pd.Series(base).pct_change().values * 100.0, 3),
    "inflow_yoy_pct_raw": np.round(pd.Series(best_res["inflow"]).pct_change().values * 100.0, 3),
    "inflow_yoy_pct_capped": np.round(pd.Series(best_res["inflow_s"]).pct_change().values * 100.0, 3),
    "actual_tons": np.round(actual_all, 0)
})

with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
    result_df.to_excel(w, sheet_name="Result_Series", index=False)
    diag_checks.to_excel(w, sheet_name="Diagnostics", index=False)

print(f"✅ 已导出：{out_xlsx}")
print(f"✅ 图已保存：{plot_path}" if SAVE_PLOTS else "")
print(f"最优参数（k_shape, lambda_scale, infra_factor_base）：{best_combo}")
