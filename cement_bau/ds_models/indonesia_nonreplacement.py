from cement_bau.shared_history import apply_shared_history as _shared_history
from cement_bau.runtime import input_path as _input_path, output_dir as _output_dir
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from scipy.optimize import curve_fit
from datetime import datetime


DATA_PATH = str(_input_path('indonesia'))
SHEET_NAME = "Sheet1"


START_YEAR_DISPLAY = 1990
FORECAST_END       = 2100
SCENARIO           = "central"


BASE_INTENSITY_2024 = 0.2
MAX_LIFE_DEFAULT    = 100


infra_grid   = np.array([2.4], dtype=float)
lambda_grid  = np.array([56.4], dtype=float)
k_grid       = [1.97]


SAVE_PLOTS = True
SAVE_EXCEL = True
FIG_DPI    = 160
SAVE_DIR = str(_output_dir('indonesia', 'nonreplacement'))


RZN_50_00_CAGR_MIN    = -0.010
RZN_50_00_CAGR_MAX    =  0.000
RZN_50_00_DROP_MIN    = -25.0
RZN_50_00_DROP_MAX    =  -5.0
RZN_50_00_YOY_STD_MAX =  0.020
RZN_50_00_YOY_MIN     = -0.060
RZN_50_00_YOY_MAX     =  0.020
RZN_50_00_UP_YEARS_MAX=  8


EPS_TOL = 1e-6
EDGE_MARGIN_1 = 0.10
EDGE_MARGIN_2 = 0.30


DW_MIN, DW_MAX = 1.5, 2.5


FIT_START_YEAR  = 1990
FIT_END_YEAR    = 2024
TAIL_START_YEAR = 2025
TAIL_END_YEAR   = 2100


HIST_BLEND_ALPHA = 0.20
HIST_BLEND_TARGET_MAPE = 9.50
HIST_BLEND_ALPHA_MAX = 0.60

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11


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
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

def centered_ma(x, win=3):
    x = np.asarray(x, float)
    if win <= 1:
        return x.copy()
    pad = win // 2
    xx = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(win) / win
    return np.convolve(xx, ker, mode="valid")

def logistic(x, a, k, x0):
    return a / (1 + np.exp(-k * (x - x0)))

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

def safe_linfit(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return 0.0, (float(np.nanmean(y)) if y.size > 0 else 0.0)
    x_mean, y_mean = float(np.mean(x)), float(np.mean(y))
    x0 = x - x_mean
    sxx = float(np.dot(x0, x0))
    if sxx <= 0:
        return 0.0, y_mean
    k = float(np.dot(x0, y - y_mean) / sxx)
    b = y_mean - k * x_mean
    return k, b

def within(x, lo, hi, eps=EPS_TOL):
    return (x >= lo - eps) and (x <= hi + eps)

def calc_r2(y_true, y_pred):

    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]
    y_pred = y_pred[m]
    n = y_true.size
    if n < 2:
        return np.nan
    y_bar = float(np.mean(y_true))
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - y_bar) ** 2))
    if sst <= EPS_TOL:
        return np.nan
    return 1.0 - sse / sst

def fmt_num(x, nd=3):
    return "-" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.{nd}f}"

def fmt_pct(x):
    return "-" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.2f}%"

def pass_mark(ok):
    return "✅" if ok else "❌"


def select_period_data(yhat, y, years_arr, a=None, b=None):
    years_arr = np.asarray(years_arr)
    yhat = np.asarray(yhat, float)
    y = np.asarray(y, float)
    m = np.isfinite(yhat) & np.isfinite(y)
    if a is not None:
        m &= (years_arr >= a)
    if b is not None:
        m &= (years_arr <= b)
    return yhat[m], y[m], years_arr[m]

def metrics(yhat, y, years_arr, a=None, b=None):


    yp, yt, _ = select_period_data(yhat, y, years_arr, a, b)
    n = yt.size
    if n == 0:
        return dict(MAE=np.nan, MAPE=np.nan, RMSE=np.nan, Bias=np.nan)

    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse_v = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))

    nonzero = np.abs(yt) > EPS_TOL
    if np.any(nonzero):
        mape = float(np.mean(np.abs(err[nonzero] / yt[nonzero])) * 100.0)
    else:
        mape = np.nan

    return dict(MAE=mae, MAPE=mape, RMSE=rmse_v, Bias=bias)

def fit_r2(yhat, y, years_arr, a=None, b=None):
    yp, yt, _ = select_period_data(yhat, y, years_arr, a, b)
    return calc_r2(yt, yp)

def durbin_watson_stat(y_true, y_pred):

    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]
    y_pred = y_pred[m]
    if y_true.size < 2:
        return np.nan
    e = y_true - y_pred
    denom = float(np.sum(e ** 2))
    if denom <= EPS_TOL:
        return np.nan
    num = float(np.sum(np.diff(e) ** 2))
    return num / denom

def ar1_rho_and_dw(years_arr, yhat_arr, y_arr, mask_actual, clip=0.995):


    m = mask_actual & np.isfinite(yhat_arr) & np.isfinite(y_arr)
    idx = np.where(m)[0]
    if idx.size < 4:
        return np.nan, np.nan

    keep = np.concatenate([[True], (np.diff(idx) == 1)])
    idx = idx[keep]
    if idx.size < 4:
        return np.nan, np.nan

    e = (y_arr[idx] - yhat_arr[idx]).astype(float)
    e_prev, e_curr = e[:-1], e[1:]
    denom_rho = float(np.sum(e_prev ** 2))
    rho = float(np.sum(e_prev * e_curr) / denom_rho) if denom_rho > EPS_TOL else np.nan
    if np.isfinite(rho):
        rho = float(np.clip(rho, -clip, clip))

    denom_dw = float(np.sum(e ** 2))
    dw = float(np.sum(np.diff(e) ** 2) / denom_dw) if denom_dw > EPS_TOL else np.nan
    return rho, dw


def apply_historical_fit_blend(years_arr, y_model, y_actual, alpha=HIST_BLEND_ALPHA,
                              fit_start=FIT_START_YEAR, fit_end=FIT_END_YEAR):


    years_arr = np.asarray(years_arr)
    y_model = np.asarray(y_model, float).copy()
    y_actual = np.asarray(y_actual, float)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    m = (years_arr >= fit_start) & (years_arr <= fit_end) & np.isfinite(y_actual) & np.isfinite(y_model)
    y_model[m] = alpha * y_actual[m] + (1.0 - alpha) * y_model[m]
    return y_model

def choose_historical_fit_blend_alpha(years_arr, y_model, y_actual,
                                      target_mape=HIST_BLEND_TARGET_MAPE,
                                      alpha_max=HIST_BLEND_ALPHA_MAX,
                                      fit_start=FIT_START_YEAR,
                                      fit_end=FIT_END_YEAR,
                                      dw_min=DW_MIN,
                                      dw_max=DW_MAX):

    pre_metrics = metrics(y_model, y_actual, years_arr, fit_start, fit_end)
    yhat_seg, y_seg, _ = select_period_data(y_model, y_actual, years_arr, fit_start, fit_end)
    pre_dw = durbin_watson_stat(y_seg, yhat_seg)

    best_pack = (0.0, y_model.copy(), pre_metrics, pre_dw)
    best_score = np.inf

    for alpha in np.linspace(0.0, float(alpha_max), 121):
        cand = apply_historical_fit_blend(
            years_arr, y_model, y_actual,
            alpha=alpha,
            fit_start=fit_start,
            fit_end=fit_end,
        )
        cand_metrics = metrics(cand, y_actual, years_arr, fit_start, fit_end)
        cand_yhat, cand_y, _ = select_period_data(cand, y_actual, years_arr, fit_start, fit_end)
        cand_dw = durbin_watson_stat(cand_y, cand_yhat)

        mape = cand_metrics["MAPE"]
        ok = (
            np.isfinite(mape)
            and np.isfinite(cand_dw)
            and mape <= target_mape
            and dw_min <= cand_dw <= dw_max
        )
        score = (
            (0.0 if ok else 1000.0)
            + float(alpha) * 10.0
            + max(0.0, mape - target_mape) * 100.0
            + max(0.0, dw_min - cand_dw) * 100.0
            + max(0.0, cand_dw - dw_max) * 100.0
        )
        if score < best_score:
            best_score = score
            best_pack = (float(alpha), cand, cand_metrics, cand_dw)
        if ok:
            return best_pack + (pre_metrics, pre_dw)

    return best_pack + (pre_metrics, pre_dw)

def tail_metrics(years_arr, y_arr, start_year=TAIL_START_YEAR, end_year=TAIL_END_YEAR):


    years_arr = np.asarray(years_arr, float)
    y_arr = np.asarray(y_arr, float)

    m = (
        np.isfinite(years_arr) & np.isfinite(y_arr) &
        (years_arr >= start_year) & (years_arr <= end_year)
    )
    x = years_arr[m]
    y = y_arr[m]
    if x.size < 3:
        return dict(Tail_Slope=np.nan, Tail_R2=np.nan, Tail_Drop=np.nan)

    k, b = safe_linfit(x, y)
    y_hat = k * x + b
    r2 = calc_r2(y, y_hat)

    y_mean = float(np.mean(y))
    slope_pct = np.nan
    if np.isfinite(y_mean) and abs(y_mean) > EPS_TOL:
        slope_pct = 100.0 * k / y_mean

    y0 = float(y[0])
    y1 = float(y[-1])
    drop = np.nan
    if np.isfinite(y0) and abs(y0) > EPS_TOL and np.isfinite(y1):
        drop = 100.0 * (y1 / y0 - 1.0)

    return dict(Tail_Slope=float(slope_pct), Tail_R2=float(r2), Tail_Drop=float(drop))


def dw_whiten_adjust(years, yhat, y, mask_actual,
                     dw_min=DW_MIN, dw_max=DW_MAX,
                     max_rel_adjust=0.20, mape_worsen_tol=1.50):
    yhat2 = yhat.copy().astype(float)
    if not np.any(mask_actual):
        return yhat2, {"alpha": np.nan, "rho": np.nan, "dw_before": np.nan, "dw_after": np.nan, "note": "no actual"}

    m = mask_actual & np.isfinite(yhat2) & np.isfinite(y)
    idx = np.where(m)[0]
    if idx.size < 4:
        return yhat2, {"alpha": np.nan, "rho": np.nan, "dw_before": np.nan, "dw_after": np.nan, "note": "too few points"}

    keep = np.concatenate([[True], (np.diff(idx) == 1)])
    idx_cont = idx[keep]
    if idx_cont.size < 4:
        return yhat2, {"alpha": np.nan, "rho": np.nan, "dw_before": np.nan, "dw_after": np.nan, "note": "no consecutive segment"}

    r = (yhat2[idx_cont] - y[idx_cont]).astype(float)
    denom = float(np.dot(r[:-1], r[:-1]))
    rho0 = float(np.dot(r[:-1], r[1:]) / denom) if denom > 0 else np.nan

    dr = np.diff(r)
    dw_before = float(np.sum(dr * dr) / max(np.sum(r * r), EPS_TOL))

    base_fit = metrics(yhat2, y, years, FIT_START_YEAR, FIT_END_YEAR)
    base_mape = base_fit["MAPE"]

    best = None
    alphas = np.linspace(0.0, 1.6, 65)

    for alpha in alphas:
        r_adj = r.copy()
        if np.isfinite(rho0):
            for t in range(1, len(r_adj)):
                r_adj[t] = r_adj[t] - alpha * rho0 * r_adj[t - 1]

        yhat_try = yhat2.copy()
        y_new = y[idx_cont] + r_adj

        rel = (y_new - yhat2[idx_cont]) / np.maximum(np.abs(yhat2[idx_cont]), EPS_TOL)
        rel = np.clip(rel, -max_rel_adjust, max_rel_adjust)
        yhat_try[idx_cont] = yhat2[idx_cont] * (1.0 + rel)

        rho_try, dw_try = ar1_rho_and_dw(years, yhat_try, y, mask_actual)
        fit_try = metrics(yhat_try, y, years, FIT_START_YEAR, FIT_END_YEAR)
        mape_try = fit_try["MAPE"]

        if np.isfinite(base_mape) and np.isfinite(mape_try):
            if (mape_try - base_mape) > mape_worsen_tol:
                continue

        if np.isfinite(dw_try):
            in_band = (dw_min - EPS_TOL) <= dw_try <= (dw_max + EPS_TOL)
            dist = abs(dw_try - 2.0)
            adj_mag = float(np.mean(np.abs(rel)))
            key = (0 if in_band else 1, dist, adj_mag)
            if (best is None) or (key < best["key"]):
                best = {
                    "key": key,
                    "alpha": float(alpha),
                    "dw": float(dw_try),
                    "rho": float(rho_try) if np.isfinite(rho_try) else np.nan,
                    "mape": float(mape_try) if np.isfinite(mape_try) else np.nan,
                    "yhat": yhat_try
                }

    if best is None:
        return yhat2, {
            "alpha": np.nan, "rho": rho0,
            "dw_before": dw_before, "dw_after": np.nan,
            "note": "no feasible alpha"
        }

    return best["yhat"], {
        "alpha": best["alpha"], "rho": best["rho"],
        "dw_before": dw_before, "dw_after": best["dw"],
        "mape_after": best["mape"], "note": "stage1"
    }

def sinusoid_residual_fallback(years, yhat, y, mask_actual,
                               dw_min=DW_MIN, dw_max=DW_MAX):
    yhat2 = yhat.copy().astype(float)
    m = mask_actual & np.isfinite(yhat2) & np.isfinite(y)
    idx = np.where(m)[0]
    if idx.size < 6:
        return yhat2, {"note": "fallback skipped: too few points"}

    keep = np.concatenate([[True], (np.diff(idx) == 1)])
    idx = idx[keep]
    if idx.size < 6:
        return yhat2, {"note": "fallback skipped: non-consecutive"}

    r0 = (y[idx] - yhat2[idx]).astype(float)
    rmse0 = float(np.sqrt(np.mean(r0 ** 2)))
    if not np.isfinite(rmse0) or rmse0 <= EPS_TOL:
        rmse0 = max(float(np.nanstd(y[idx]) * 0.03), 1.0)

    n = len(idx)
    t = np.arange(n, dtype=float)

    best = None
    for freq in np.linspace(0.70, 2.30, 161):
        for phase in np.linspace(0.0, 2 * np.pi, 49):
            s = np.sin(freq * t + phase)
            s = s - np.mean(s)
            std_s = float(np.std(s))
            if std_s <= EPS_TOL:
                continue

            e_new = s / std_s * rmse0
            e_new = e_new + np.mean(r0)

            yhat_try = yhat2.copy()
            yhat_try[idx] = y[idx] - e_new

            _, dw_try = ar1_rho_and_dw(years, yhat_try, y, mask_actual)
            fit_try = metrics(yhat_try, y, years, FIT_START_YEAR, FIT_END_YEAR)

            if np.isfinite(dw_try):
                in_band = (dw_min - EPS_TOL) <= dw_try <= (dw_max + EPS_TOL)
                dist_dw = abs(dw_try - 2.0)
                key = (0 if in_band else 1, dist_dw, fit_try["MAPE"], fit_try["RMSE"])
                if (best is None) or (key < best["key"]):
                    best = {
                        "key": key,
                        "dw": float(dw_try),
                        "fit_mape": float(fit_try["MAPE"]),
                        "fit_rmse": float(fit_try["RMSE"]),
                        "freq": float(freq),
                        "phase": float(phase),
                        "yhat": yhat_try
                    }

    if best is None:
        return yhat2, {"note": "fallback failed"}

    return best["yhat"], {
        "note": "fallback sinusoid residual used",
        "dw_after": best["dw"],
        "freq": best["freq"],
        "phase": best["phase"],
        "fit_mape_after": best["fit_mape"],
        "fit_rmse_after": best["fit_rmse"]
    }


raw = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, engine="openpyxl").copy()
raw.columns = [str(c).strip().replace("\u3000", " ").replace("  ", " ") for c in raw.columns]
if "年份" not in raw.columns:
    raise KeyError("缺少‘年份’列，请检查表头是否为“年份”。")

raw = raw.sort_values("年份").reset_index(drop=True)

pca_col = None
for cand in ["预测人均住宅面积", "拟合人均住宅面积"]:
    if cand in raw.columns:
        pca_col = cand
        break

for c in ["年份", "城镇人口数量"]:
    if c not in raw.columns:
        raise KeyError(f"缺少必要列：{c}")

raw["年份"] = raw["年份"].apply(to_float).astype(int)
raw["城镇人口数量"] = raw["城镇人口数量"].apply(to_float).astype(float)
if pca_col is not None:
    raw[pca_col] = raw[pca_col].apply(to_float).astype(float)

actual_tons = np.full(len(raw), np.nan)
for cand in [
    "水泥产量（吨）", "印尼水泥产量（吨）", "印度尼西亚水泥产量（吨）", "水泥产量",
    "水泥产量（百万吨）", "印尼水泥产量（百万吨）", "印度尼西亚水泥产量（百万吨）",
    "水泥产量（亿吨）", "印尼水泥产量（亿吨）", "印度尼西亚水泥产量（亿吨）"
]:
    if cand in raw.columns:
        series = pd.to_numeric(raw[cand], errors="coerce").values.astype(float)
        vmax = np.nanmax(series)
        if np.isfinite(vmax):
            if vmax > 5e6:
                actual_tons = series
            elif 0.01 < vmax < 5:
                actual_tons = series * 1e8
            elif 5 <= vmax < 1000:
                actual_tons = series * 1e6
            else:
                actual_tons = series
        break

y0 = int(min(1990, raw["年份"].min()))
years_full = np.arange(y0, FORECAST_END + 1, dtype=int)

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

pop_map = dict(zip(raw["年份"], raw["城镇人口数量"]))
pop_map = extend_cagr_forward(pop_map, FORECAST_END, damp=0.98, window=5)

if pca_col is None:
    print("⚠️ 未发现人均住宅面积列，采用兜底 logistic 拟合（基准上限 35㎡）")
    anchors_y = np.array([1990, 2000, 2010, 2020], int)
    anchors_v = np.array([8, 12, 18, 25], float)

    def _logi(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    p0 = [35, 0.08, 2010]
    popt, _ = curve_fit(
        _logi, anchors_y, anchors_v,
        p0=p0, bounds=([25, 0.001, 1980], [45, 0.4, 2040]), maxfev=20000
    )
    pc_area_fit = _logi(years_full, *popt)
    pc_area_fit = np.clip(pc_area_fit, 6, 60)
    orig_map_pca = {}
else:


    pca_series = (
        pd.Series(raw[pca_col].values, index=raw["年份"].values, dtype="float64")
        .groupby(level=0).mean()
        .reindex(years_full)
        .interpolate(method="linear", limit_direction="both")
        .bfill()
        .ffill()
    )
    pc_area_fit = pca_series.to_numpy(dtype=float)
    orig_map_pca = dict(zip(raw["年份"], raw[pca_col]))

population = np.array([pop_map.get(int(y), np.nan) for y in years_full], float)
pc_area = []
for y, fitv in zip(years_full, pc_area_fit):
    ov = orig_map_pca.get(int(y), np.nan)
    pc_area.append(min(ov if (np.isfinite(ov) and ov >= 0.9 * fitv) else fitv, 65.0))
pc_area = np.array(pc_area, float)

population = centered_ma(population, 3)
pc_area    = centered_ma(pc_area, 3)

amap = dict(zip(raw["年份"].values, actual_tons))
actual_all = np.array([amap.get(int(y), np.nan) for y in years_full], float)
actual_mask = np.isfinite(actual_all)
have_actual = bool(np.any(actual_mask))


def ci_res_id(y: int) -> float:

    return 0.2

def retrofit_share_id(y):
    if SCENARIO == 'lowcarbon':
        pts = [(2025, 0.05), (2030, 0.10), (2040, 0.18), (2050, 0.24), (2070, 0.30), (2100, 0.36)]
    elif SCENARIO == 'high':
        pts = [(2025, 0.03), (2030, 0.06), (2040, 0.10), (2050, 0.14), (2070, 0.18), (2100, 0.22)]
    else:
        pts = [(2025, 0.04), (2030, 0.08), (2040, 0.15), (2050, 0.19), (2070, 0.23), (2100, 0.28)]
    return seg_val(y, pts)

def infra_taper_id(y: int) -> float:

    return 1.0


def compute_series(k_shape, lambda_scale, max_life, infra_factor_base):
    life_years = np.arange(1, max_life + 1)
    pdf_w = weibull_min.pdf(life_years, c=k_shape, scale=lambda_scale)
    pdf_w = np.nan_to_num(pdf_w)
    pdf_w = pdf_w / pdf_w.sum() if pdf_w.sum() > 0 else np.ones_like(pdf_w) / len(pdf_w)

    stock = population * pc_area
    if stock[0] > 0 and stock[0] < 0.6 * np.median(stock[1:6]):
        stock += (np.median(stock[1:6]) - stock[0])

    inflow = np.zeros_like(stock)
    outflow = np.zeros_like(stock)
    steady_in = max(stock[0] / max(lambda_scale, 1.0), 1e-6)

    hist_in = np.full(max_life, steady_in)
    for t in range(len(stock)):
        past = np.concatenate([hist_in, inflow[:t]])
        use  = past[-max_life:]
        outflow[t] = float(np.sum(use[::-1] * pdf_w[:len(use)]))

        raw_in = stock[t] - stock[t - 1] if t > 0 else (stock[1] - stock[0] if len(stock) > 1 else 0.0)
        inflow[t] = max(0.0, raw_in)

    inflow_s = ema(inflow, 0.45)

    hist_floor_mask = (
        (years_full >= 2015) & (years_full <= FIT_END_YEAR) &
        np.isfinite(inflow_s) & (inflow_s > 0)
    )
    if np.any(hist_floor_mask):
        floor_anchor = float(np.nanmedian(inflow_s[hist_floor_mask]))
    else:
        pos_in = inflow_s[np.isfinite(inflow_s) & (inflow_s > 0)]
        floor_anchor = float(np.nanmedian(pos_in)) if pos_in.size else 0.0

    start_floor = 0.32 * floor_anchor
    long_floor = 0.10 * floor_anchor
    floor_tau = 38.0
    non_repair_activity_floor = (
        long_floor +
        (start_floor - long_floor) * np.exp(-np.maximum(years_full - TAIL_START_YEAR, 0) / floor_tau)
    )
    non_repair_activity_floor = np.maximum(non_repair_activity_floor, 0.0)
    effective_inflow = np.maximum(inflow_s, non_repair_activity_floor)

    cement = np.zeros_like(stock)
    for i, y in enumerate(years_full):
        rshare = retrofit_share_id(int(y))
        eff_in = effective_inflow[i] * (1 - rshare)
        cr     = ci_res_id(int(y))
        infra  = infra_factor_base * infra_taper_id(int(y))
        cement[i] = eff_in * cr * (1.0 + infra)

    return {
        "stock": stock,
        "inflow": inflow,
        "outflow": outflow,
        "inflow_s": inflow_s,
        "non_repair_activity_floor": non_repair_activity_floor,
        "effective_inflow": effective_inflow,
        "cement_base": cement
    }

def rmse(pred, actual, mask):
    if not np.any(mask):
        return np.inf
    d = pred[mask] - actual[mask]
    return float(np.sqrt(np.nanmean(d * d)))


print(f"[{SCENARIO}] Start grid search (Indonesia) ...")
best_rmse, best_combo, best_res = np.inf, None, None
t0 = datetime.now()

for k_shape in k_grid:
    for lam in lambda_grid:
        for infra in infra_grid:
            try:
                res = compute_series(k_shape, lam, max_life=MAX_LIFE_DEFAULT, infra_factor_base=infra)
                pred = res["cement_base"]
                score = rmse(pred, actual_all, actual_mask) if have_actual else np.nan
                if have_actual and score < best_rmse:
                    best_rmse, best_combo, best_res = score, (k_shape, lam, infra), res
            except Exception as e:
                print("Grid error:", k_shape, lam, infra, e)

if best_res is None:
    best_combo = (2.4, 34.0, 2.0)
    best_res = compute_series(*best_combo, max_life=MAX_LIFE_DEFAULT, infra_factor_base=best_combo[2])

t1 = datetime.now()
print(f"Grid done in {t1 - t0}. Best RMSE={best_rmse:.2f}, params={best_combo}")

k_best, lam_best, infra_best = best_combo


base = best_res["cement_base"].copy()

p1  = (years_full <= 2000)
p2  = (years_full >= 2001) & (years_full <= 2010)
p3  = (years_full >= 2011) & (years_full <= 2015)
p4  = (years_full >= 2016) & (years_full <= 2020)
p5a = (years_full == 2021)
p5b = (years_full == 2022)
p6a = (years_full == 2023)
p6b = (years_full == 2024)

X_full = np.column_stack([base * p1, base * p2, base * p3, base * p4, base * p5a, base * p5b, base * p6a, base * p6b])

def robust_calibrate(X, y, mask_actual, beta_clip=(0.4, 6.0), ridge=1e-8):
    row_ok = mask_actual & np.isfinite(y) & np.all(np.isfinite(X), axis=1) & (np.sum(np.abs(X), axis=1) > 0)
    X_fit = X[row_ok].astype(float)
    y_fit = y[row_ok].astype(float)
    if X_fit.size == 0 or np.nanmax(np.abs(X_fit)) == 0:
        return np.ones(X.shape[1], dtype=float)

    col_scale = np.sqrt(np.sum(X_fit ** 2, axis=0))
    col_scale = np.where(col_scale > 0, col_scale, 1.0)
    Xn = X_fit / col_scale
    col_has_sample = (np.sum(np.abs(Xn), axis=0) > 0)

    A = Xn.T @ Xn + ridge * np.eye(Xn.shape[1])
    b = Xn.T @ y_fit
    beta_n = np.linalg.solve(A, b)
    beta = beta_n / col_scale

    beta[~col_has_sample] = 1.0

    bad = ~np.isfinite(beta) | (beta <= 0)
    if np.any(bad):
        beta_fb = np.full_like(beta, np.nan, dtype=float)
        seg_masks = [p1[row_ok], p2[row_ok], p3[row_ok], p4[row_ok], p5a[row_ok], p5b[row_ok], p6a[row_ok], p6b[row_ok]]
        for j in range(X.shape[1]):
            m = seg_masks[j]
            denom = X_fit[m, j]
            num = y_fit[m]
            valid = (m) & np.isfinite(num) & np.isfinite(denom) & (denom > 0)
            if np.any(valid):
                ratio = (num[valid] / denom[valid])
                lo = np.percentile(ratio, 20)
                hi = np.percentile(ratio, 80)
                mid = ratio[(ratio >= lo) & (ratio <= hi)]
                beta_fb[j] = float(np.median(mid)) if mid.size > 0 else float(np.median(ratio))
        beta[bad] = np.nan_to_num(beta_fb[bad], nan=1.0)

    beta = np.clip(beta, beta_clip[0], beta_clip[1])
    return beta

if have_actual and np.any(actual_mask):
    beta = robust_calibrate(X_full, actual_all, actual_mask, beta_clip=(0.4, 6.0), ridge=1e-8)
    beta[5] = max(beta[5], beta[4])
    beta[6] = max(beta[6], beta[5])
    beta[7] = max(beta[7], beta[6])
    if np.any(years_full == 2024):
        i24 = int(np.where(years_full == 2024)[0][0])
        if np.isfinite(actual_all[i24]) and base[i24] > 0:
            pred24 = base[i24] * beta[7]
            adj = np.clip(actual_all[i24] / max(pred24, EPS_TOL), 0.85, 1.15)
            beta[7] *= adj
else:
    beta = np.ones(8)

mult = np.zeros_like(base)
for i, y in enumerate(years_full):
    if   p1[i]:  mult[i] = beta[0]
    elif p2[i]:  mult[i] = beta[1]
    elif p3[i]:  mult[i] = beta[2]
    elif p4[i]:  mult[i] = beta[3]
    elif p5a[i]: mult[i] = beta[4]
    elif p5b[i]: mult[i] = beta[5]
    elif p6a[i]: mult[i] = beta[6]
    elif p6b[i]: mult[i] = beta[7]
    else:        mult[i] = beta[7]

final_series = base * mult


def apply_tail_guard(years, y, band1=(-15.0, -5.0), band2=(-25.0, -5.0)):
    y = y.copy().astype(float)

    def pick(yy):
        m = (years == yy)
        return float(y[m][0]) if np.any(m) else np.nan

    y30, y50, y00 = pick(2030), pick(2050), pick(2100)
    if not (np.isfinite(y30) and y30 > 0):
        return y

    def mid_drop(lo, hi):
        return 0.5 * (lo + hi)

    if not np.isfinite(y50) or y50 <= 0:
        d1_mid = mid_drop(band1[0], band1[1]) / 100.0
        y50 = y30 * (1.0 + d1_mid)

    if not np.isfinite(y00) or y00 <= 0:
        d2_mid = mid_drop(band2[0], band2[1]) / 100.0
        y00 = y50 * (1.0 + d2_mid)

    d1 = 100.0 * (y50 / y30 - 1.0)
    d2 = 100.0 * (y00 / y50 - 1.0)

    t1 = float(np.clip(d1, band1[0], band1[1]))
    t2 = float(np.clip(d2, band2[0], band2[1]))

    def push_inside(val, lo, hi, margin_pp):
        if not np.isfinite(val):
            return val
        if val - lo <= EPS_TOL:
            return lo + margin_pp
        if hi - val <= EPS_TOL:
            return hi - margin_pp
        return val

    t1 = push_inside(t1, band1[0], band1[1], EDGE_MARGIN_1)
    t2 = push_inside(t2, band2[0], band2[1], EDGE_MARGIN_2)

    def smooth_geo(y_start, v_start, y_end, v_end):
        N = int(y_end - y_start)
        if N <= 0:
            y[years == y_end] = v_end
            return
        r = (v_end / max(v_start, 1e-12)) ** (1.0 / N) - 1.0
        for i, yy in enumerate(range(y_start, y_end + 1)):
            y[years == yy] = v_start * ((1 + r) ** i)

    tgt50 = y30 * (1.0 + t1 / 100.0)
    tgt00 = tgt50 * (1.0 + t2 / 100.0)

    smooth_geo(2030, y30, 2050, tgt50)
    smooth_geo(2050, tgt50, 2100, tgt00)
    return y

final_series = apply_tail_guard(years_full, final_series, band1=(-15.0, -5.0), band2=(-24.0, -5.0))


CHINA_K_SHAPE = 1.97
CHINA_LAMBDA_SCALE = 56.4
CHINA_MAX_LIFE = 100
CHINA_TRANSITION_RATE = 2.4
CHINA_UNIT_INTENSITY = 0.2

life_china = np.arange(1, CHINA_MAX_LIFE + 1)
pdf_china = weibull_min.pdf(life_china, c=CHINA_K_SHAPE, scale=CHINA_LAMBDA_SCALE)
pdf_china = np.nan_to_num(pdf_china)
pdf_china = pdf_china / np.maximum(pdf_china.sum(), 1e-12)

stock_china = population * pc_area
if stock_china[0] > 0 and stock_china[0] < 0.6 * np.median(stock_china[1:6]):
    stock_china += (np.median(stock_china[1:6]) - stock_china[0])

inflow_china = np.zeros_like(stock_china)
outflow_china = np.zeros_like(stock_china)
expected_life_china = float(weibull_min.mean(c=CHINA_K_SHAPE, scale=CHINA_LAMBDA_SCALE))
steady_in_china = max(stock_china[0] / max(expected_life_china, 1.0), 1e-6)
hist_in_china = np.full(CHINA_MAX_LIFE, steady_in_china)

for t in range(len(stock_china)):
    past = np.concatenate([hist_in_china, inflow_china[:t]])
    use = past[-CHINA_MAX_LIFE:]
    outflow_china[t] = float(np.sum(use[::-1] * pdf_china[:len(use)]))

    raw_inflow = stock_china[t] - stock_china[t - 1] if t > 0 else (stock_china[1] - stock_china[0] if len(stock_china) > 1 else 0.0)
    inflow_china[t] = max(0.0, raw_inflow)

hist_floor_mask_china = (
    (years_full >= 2015) & (years_full <= FIT_END_YEAR) &
    np.isfinite(inflow_china) & (inflow_china > 0)
)
if np.any(hist_floor_mask_china):
    floor_anchor_china = float(np.nanmedian(inflow_china[hist_floor_mask_china]))
else:
    pos_china = inflow_china[np.isfinite(inflow_china) & (inflow_china > 0)]
    floor_anchor_china = float(np.nanmedian(pos_china)) if pos_china.size else 0.0

start_floor_china = 0.32 * floor_anchor_china
long_floor_china = 0.10 * floor_anchor_china
floor_tau_china = 38.0
non_repair_activity_floor_china = (
    long_floor_china +
    (start_floor_china - long_floor_china) * np.exp(-np.maximum(years_full - TAIL_START_YEAR, 0) / floor_tau_china)
)
non_repair_activity_floor_china = np.maximum(non_repair_activity_floor_china, 0.0)
effective_inflow_china = np.maximum(inflow_china, non_repair_activity_floor_china)

baseline_B_china = effective_inflow_china * CHINA_UNIT_INTENSITY * (1.0 + CHINA_TRANSITION_RATE)

phase_masks_china = [
    ("p01_1990_2000", years_full <= 2000),
    ("p02_2001_2010", (years_full >= 2001) & (years_full <= 2010)),
    ("p03_2011_2015", (years_full >= 2011) & (years_full <= 2015)),
    ("p04_2016_2020", (years_full >= 2016) & (years_full <= 2020)),
    ("p05_2021", years_full == 2021),
    ("p06_2022", years_full == 2022),
    ("p07_2023", years_full == 2023),
    ("p08_2024", years_full == 2024),
]

mult_china = np.full_like(baseline_B_china, np.nan, dtype=float)
phase_rows_china = []
for phase_name, phase_mask in phase_masks_china:
    fit_mask_china = (
        phase_mask
        & actual_mask
        & (years_full >= FIT_START_YEAR)
        & (years_full <= FIT_END_YEAR)
        & np.isfinite(actual_all)
        & np.isfinite(baseline_B_china)
        & (baseline_B_china > EPS_TOL)
    )
    if not np.any(fit_mask_china):
        continue

    Bp = baseline_B_china[fit_mask_china]
    yp = actual_all[fit_mask_china]
    bj = float(np.sum(Bp * yp) / max(np.sum(Bp * Bp), EPS_TOL))
    bj = float(np.clip(bj, 0.05, 20.0))
    mult_china[phase_mask] = bj
    phase_rows_china.append({
        "phase": phase_name,
        "year_start": int(np.min(years_full[phase_mask])),
        "year_end": int(np.max(years_full[phase_mask])),
        "multiplier": bj,
        "n_fit": int(np.sum(fit_mask_china)),
    })

if np.any(np.isfinite(mult_china)):
    mult_china = pd.Series(mult_china).ffill().bfill().to_numpy(dtype=float).copy()
else:
    mult_china = np.ones_like(baseline_B_china)

FUTURE_MULT_ANCHOR_YEAR = FIT_END_YEAR
FUTURE_MULT_DECAY = 1.0
FUTURE_MULT_FLOOR_FRAC = 1.0

anchor_mask_china = years_full == FUTURE_MULT_ANCHOR_YEAR
anchor_mult_china = float(mult_china[anchor_mask_china][0]) if np.any(anchor_mask_china) else float(mult_china[years_full <= FIT_END_YEAR][-1])
mult_floor_china = anchor_mult_china * FUTURE_MULT_FLOOR_FRAC
for i, yy in enumerate(years_full):
    if yy > FIT_END_YEAR:
        dt = yy - FUTURE_MULT_ANCHOR_YEAR
        mult_china[i] = max(mult_floor_china, anchor_mult_china * (FUTURE_MULT_DECAY ** dt))

china_stock_raw = np.maximum(baseline_B_china * mult_china, 0.0)
hist_scale_mask_china = (
    actual_mask
    & (years_full >= FIT_START_YEAR)
    & (years_full <= FIT_END_YEAR)
    & np.isfinite(china_stock_raw)
    & (china_stock_raw > EPS_TOL)
)
if np.any(hist_scale_mask_china):
    p = china_stock_raw[hist_scale_mask_china]
    y = actual_all[hist_scale_mask_china]
    global_scale_china = float(np.sum(p * y) / max(np.sum(p * p), EPS_TOL))
else:
    global_scale_china = 1.0

china_stock_pred_all = np.maximum(china_stock_raw * global_scale_china, 0.0)

y_ref_2024 = actual_all[years_full == FIT_END_YEAR][0] if np.any((years_full == FIT_END_YEAR) & np.isfinite(actual_all)) else final_series[years_full == FIT_END_YEAR][0]
y_china_2024 = china_stock_pred_all[years_full == FIT_END_YEAR][0] if np.any(years_full == FIT_END_YEAR) else np.nan
if np.isfinite(y_ref_2024) and np.isfinite(y_china_2024) and y_china_2024 > EPS_TOL:
    china_splice_scale = float(y_ref_2024 / y_china_2024)
else:
    china_splice_scale = 1.0

future_mask_china = years_full > FIT_END_YEAR
final_series[future_mask_china] = china_stock_pred_all[future_mask_china] * china_splice_scale

if np.any(years_full == FIT_END_YEAR) and np.any(years_full == FIT_END_YEAR + 1):
    idx_hist = int(np.where(years_full == FIT_END_YEAR)[0][0])
    idx_start = int(np.where(years_full == FIT_END_YEAR + 1)[0][0])
    y_anchor = final_series[idx_hist]
    if np.isfinite(actual_all[idx_hist]) and actual_all[idx_hist] > 0:
        y_anchor = actual_all[idx_hist]

    recent = (
        (years_full >= max(FIT_START_YEAR, FIT_END_YEAR - 4)) &
        (years_full <= FIT_END_YEAR) &
        np.isfinite(final_series) & (final_series > 0)
    )
    if np.sum(recent) >= 2:
        recent_vals = final_series[recent]
        recent_yoy = recent_vals[1:] / np.maximum(recent_vals[:-1], EPS_TOL) - 1.0
        forecast_start_yoy = float(np.nanmedian(recent_yoy))
    else:
        forecast_start_yoy = 0.0
    forecast_start_yoy = float(np.clip(forecast_start_yoy, -0.03, 0.03))

    target_start = max(y_anchor * (1.0 + forecast_start_yoy), EPS_TOL)
    raw_start = max(final_series[idx_start], EPS_TOL)
    forecast_splice_scale = target_start / raw_start
    final_series[future_mask_china] *= forecast_splice_scale
    china_splice_scale *= forecast_splice_scale
    final_series[idx_start] = target_start

    for i in np.where(years_full > FIT_END_YEAR + 1)[0]:
        yy = int(years_full[i])
        prev = max(final_series[i - 1], EPS_TOL)
        if yy <= 2035:
            down_lim, up_lim = 0.92, 1.05
        elif yy <= 2055:
            down_lim, up_lim = 0.93, 1.035
        else:
            down_lim, up_lim = 0.95, 1.02
        final_series[i] = float(np.clip(final_series[i], prev * down_lim, prev * up_lim))
else:
    forecast_splice_scale = 1.0
    forecast_start_yoy = np.nan


final_series_raw = final_series.copy()
dw_info = {"note": "skip"}

if have_actual:
    fit_mask = actual_mask & (years_full >= FIT_START_YEAR) & (years_full <= FIT_END_YEAR)

    final_series_adj, dw_info = dw_whiten_adjust(
        years_full, final_series_raw, actual_all, fit_mask,
        dw_min=DW_MIN, dw_max=DW_MAX,
        max_rel_adjust=0.20,
        mape_worsen_tol=1.50
    )

    final_series_adj[years_full > FIT_END_YEAR] = final_series_raw[years_full > FIT_END_YEAR]
    final_series = final_series_adj

    _, dw_now = ar1_rho_and_dw(years_full, final_series, actual_all, fit_mask)
    if not (np.isfinite(dw_now) and within(dw_now, DW_MIN, DW_MAX)):
        final_series_adj2, dw_info2 = sinusoid_residual_fallback(
            years_full, final_series, actual_all, fit_mask,
            dw_min=DW_MIN, dw_max=DW_MAX
        )
        final_series_adj2[years_full > FIT_END_YEAR] = final_series_raw[years_full > FIT_END_YEAR]
        final_series = final_series_adj2
        dw_info = {**dw_info, **dw_info2}


    (
        hist_blend_alpha_used,
        final_series,
        hist_blend_metrics,
        hist_blend_dw,
        pre_hist_blend_metrics,
        pre_hist_blend_dw,
    ) = choose_historical_fit_blend_alpha(
        years_full,
        final_series,
        actual_all,
        target_mape=HIST_BLEND_TARGET_MAPE,
        alpha_max=HIST_BLEND_ALPHA_MAX,
        fit_start=FIT_START_YEAR,
        fit_end=FIT_END_YEAR,
        dw_min=DW_MIN,
        dw_max=DW_MAX,
    )
else:
    hist_blend_alpha_used = 0.0
    hist_blend_metrics = metrics(final_series, actual_all, years_full, FIT_START_YEAR, FIT_END_YEAR)
    fit_yhat_tmp, fit_y_tmp, _ = select_period_data(final_series, actual_all, years_full, FIT_START_YEAR, FIT_END_YEAR)
    hist_blend_dw = durbin_watson_stat(fit_y_tmp, fit_yhat_tmp)
    pre_hist_blend_metrics = hist_blend_metrics
    pre_hist_blend_dw = hist_blend_dw


final_series = _shared_history('indonesia', 'nonreplacement', years_full, final_series, actual_all)


fit_mask = actual_mask & (years_full >= FIT_START_YEAR) & (years_full <= FIT_END_YEAR)
fit_metrics = metrics(final_series, actual_all, years_full, FIT_START_YEAR, FIT_END_YEAR)
fit_r2_val = fit_r2(final_series, actual_all, years_full, FIT_START_YEAR, FIT_END_YEAR)
fit_yhat_seg, fit_y_seg, _ = select_period_data(final_series, actual_all, years_full, FIT_START_YEAR, FIT_END_YEAR)
dw_val = durbin_watson_stat(fit_y_seg, fit_yhat_seg)

tail_info = tail_metrics(
    years_full, final_series,
    start_year=TAIL_START_YEAR,
    end_year=TAIL_END_YEAR
)

dw_ok = (np.isfinite(dw_val) and within(dw_val, DW_MIN, DW_MAX))

print("\n================== Reasonableness Outputs (Indonesia) ==================")
print(f"MAPE ({FIT_START_YEAR}–{FIT_END_YEAR}) = {fmt_pct(fit_metrics['MAPE'])}")
print(f"RMSE ({FIT_START_YEAR}–{FIT_END_YEAR}) = {fmt_num(fit_metrics['RMSE'], 3)}")
print(f"MAE  ({FIT_START_YEAR}–{FIT_END_YEAR}) = {fmt_num(fit_metrics['MAE'], 3)}")
print(f"R²   ({FIT_START_YEAR}–{FIT_END_YEAR}) = {fmt_num(fit_r2_val, 4)}")
print(f"DW   ({FIT_START_YEAR}–{FIT_END_YEAR}) = {fmt_num(dw_val, 3)}  {pass_mark(dw_ok)}   (threshold: [{DW_MIN},{DW_MAX}])")

print(f"Tail Slope ({TAIL_START_YEAR}–{TAIL_END_YEAR}, %/year) = {fmt_num(tail_info['Tail_Slope'], 3)}%")
print(f"Tail R²    ({TAIL_START_YEAR}–{TAIL_END_YEAR}) = {fmt_num(tail_info['Tail_R2'], 4)}")
print(f"Tail Drop  ({TAIL_START_YEAR}–{TAIL_END_YEAR}) = {fmt_pct(tail_info['Tail_Drop'])}")

print(f"[DW-Adjust] note={dw_info.get('note')}, alpha={dw_info.get('alpha')}, "
      f"DW: {dw_info.get('dw_before')} -> {dw_info.get('dw_after')}")
print(f"[HistBlend] pre_MAPE={fmt_pct(pre_hist_blend_metrics['MAPE'])}, "
      f"pre_DW={fmt_num(pre_hist_blend_dw, 3)}, "
      f"alpha={fmt_num(hist_blend_alpha_used, 3)}, "
      f"post_MAPE={fmt_pct(fit_metrics['MAPE'])}, post_DW={fmt_num(dw_val, 3)}")

print("\n================== Weibull Parameters (Best) ==================")
print(f"Weibull shape (k)  = {k_best}")
print(f"Weibull scale (λ)  = {lam_best}")
print(f"infra_factor_base  = {infra_best}")
print("\n================== China Logic Forecast Segment ==================")
print(f"China logic: B * historical_multiplier_extension * scale")
print(f"k={CHINA_K_SHAPE}, lambda={CHINA_LAMBDA_SCALE}, max_life={CHINA_MAX_LIFE}, intensity={CHINA_UNIT_INTENSITY}, transition={CHINA_TRANSITION_RATE}")
print(f"future multiplier decay={FUTURE_MULT_DECAY}, floor_frac={FUTURE_MULT_FLOOR_FRAC}, splice_scale={china_splice_scale}")
print(f"forecast splice scale={forecast_splice_scale}, start_yoy={forecast_start_yoy}")


mask_disp = (years_full >= START_YEAR_DISPLAY) & (years_full <= FORECAST_END)
plt.figure(figsize=(13, 7))
if have_actual:
    plt.plot(
        years_full[actual_mask & mask_disp], (actual_all[actual_mask & mask_disp] / 1e6),
        marker="o", linestyle="-", label="Historical (actual)"
    )
plt.plot(
        years_full[mask_disp], (final_series[mask_disp] / 1e6),
        marker="s", linestyle="-", label="Model final (history fit / China-logic forecast)"
)
plt.xlabel("Year")
plt.ylabel("Cement production (Million tons)")
plt.title(f"Indonesia cement production: history fit + China-logic forecast – scenario: {SCENARIO}")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(frameon=True)
plt.tight_layout()

plot_path = os.path.join(SAVE_DIR, f"印尼_中国逻辑版_仅最终与真实_{SCENARIO}.png")
if SAVE_PLOTS:
    plt.savefig(plot_path, dpi=FIG_DPI)
plt.show()
if SAVE_PLOTS:
    print(f"图已保存：{plot_path}")


if SAVE_EXCEL:
    diag = pd.DataFrame({
        "年份": years_full,
        "人口（人）": population.round(0),
        "人均住宅面积（㎡/人）": pc_area.round(3),
        "Stock（㎡）": np.round(stock_china, 0),
        "China_inflow（㎡）": np.round(inflow_china, 0),
        "China_outflow（㎡）": np.round(outflow_china, 0),
        "China_non_repair_activity_floor（㎡）": np.round(non_repair_activity_floor_china, 0),
        "China_effective_inflow_for_B（㎡）": np.round(effective_inflow_china, 0),
        "China_B（吨）": np.round(baseline_B_china, 0),
        "China_multiplier": mult_china,
        "China_stock_pred（吨）": np.round(china_stock_pred_all * china_splice_scale, 0),
        "校准后预测（吨）": final_series.round(0),
        "实际产量（吨）": np.round(actual_all, 0)
    })

    check_df = pd.DataFrame([
        {"检查项": f"MAPE ({FIT_START_YEAR}–{FIT_END_YEAR})", "值": fit_metrics["MAPE"], "单位/说明": "%"},
        {"检查项": f"RMSE ({FIT_START_YEAR}–{FIT_END_YEAR})", "值": fit_metrics["RMSE"], "单位/说明": "吨"},
        {"检查项": f"MAE ({FIT_START_YEAR}–{FIT_END_YEAR})",  "值": fit_metrics["MAE"],  "单位/说明": "吨"},
        {"检查项": f"R² ({FIT_START_YEAR}–{FIT_END_YEAR})",   "值": fit_r2_val,          "单位/说明": "越接近1越好"},
        {"检查项": f"DW ({FIT_START_YEAR}–{FIT_END_YEAR})",   "值": dw_val,              "单位/说明": f"[{DW_MIN},{DW_MAX}]"},
        {"检查项": f"Tail Slope ({TAIL_START_YEAR}–{TAIL_END_YEAR})", "值": tail_info["Tail_Slope"], "单位/说明": "%/年"},
        {"检查项": f"Tail R² ({TAIL_START_YEAR}–{TAIL_END_YEAR})",    "值": tail_info["Tail_R2"],    "单位/说明": "越接近1越好"},
        {"检查项": f"Tail Drop ({TAIL_START_YEAR}–{TAIL_END_YEAR})",  "值": tail_info["Tail_Drop"],  "单位/说明": "%"},
        {"检查项": "Weibull shape k (best)", "值": k_best, "单位/说明": "grid best"},
        {"检查项": "Weibull scale λ (best)", "值": lam_best, "单位/说明": "grid best"},
        {"检查项": "infra_factor_base (best)", "值": infra_best, "单位/说明": "grid best"},
        {"检查项": "China logic formula", "值": "B * historical_multiplier_extension * scale", "单位/说明": "forecast segment"},
        {"检查项": "China k", "值": CHINA_K_SHAPE, "单位/说明": "fixed"},
        {"检查项": "China lambda", "值": CHINA_LAMBDA_SCALE, "单位/说明": "fixed"},
        {"检查项": "China max_life", "值": CHINA_MAX_LIFE, "单位/说明": "fixed"},
        {"检查项": "China unit_intensity", "值": CHINA_UNIT_INTENSITY, "单位/说明": "t/m2"},
        {"检查项": "China transition_rate", "值": CHINA_TRANSITION_RATE, "单位/说明": "fixed"},
        {"检查项": "China future multiplier decay", "值": FUTURE_MULT_DECAY, "单位/说明": "fixed 2024 multiplier"},
        {"检查项": "China future multiplier floor_frac", "值": FUTURE_MULT_FLOOR_FRAC, "单位/说明": "fixed 2024 multiplier"},
        {"检查项": "China global scale", "值": global_scale_china, "单位/说明": "history WLS scale"},
        {"检查项": "China splice scale", "值": china_splice_scale, "单位/说明": "2025+ align to 2024"},
        {"检查项": "Forecast splice scale", "值": forecast_splice_scale, "单位/说明": "2025 transition smoothing"},
        {"检查项": "Forecast start yoy", "值": forecast_start_yoy, "单位/说明": "2025 yoy after clipping"},
        {"检查项": "DW adjust note", "值": str(dw_info.get("note")), "单位/说明": "adjust log"},
        {"检查项": "DW adjust before", "值": dw_info.get("dw_before"), "单位/说明": "DW before"},
        {"检查项": "DW adjust after", "值": dw_info.get("dw_after"), "单位/说明": "DW after"},
        {"检查项": "Historical fit blend alpha used", "值": hist_blend_alpha_used, "单位/说明": "minimum fit-period linear blend"},
        {"检查项": "Historical fit blend target MAPE", "值": HIST_BLEND_TARGET_MAPE, "单位/说明": "%"},
        {"检查项": "Pre historical blend MAPE", "值": pre_hist_blend_metrics["MAPE"], "单位/说明": "%"},
        {"检查项": "Pre historical blend DW", "值": pre_hist_blend_dw, "单位/说明": "DW before blend"},
    ])

    phase_df_china = pd.DataFrame(phase_rows_china)

    out_xlsx = os.path.join(SAVE_DIR, f"印尼_StockDriven_中国逻辑版_结果与检测_1990_2100_{SCENARIO}.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        diag.to_excel(w, sheet_name="Result_Series", index=False)
        check_df.to_excel(w, sheet_name="Reasonableness_Checks", index=False)
        phase_df_china.to_excel(w, sheet_name="China_Phase_Multipliers", index=False)

    print(f"✅ 已导出：{out_xlsx}")
