from cement_bau.runtime import input_path as _input_path, output_dir as _output_dir
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min


DATA_PATH = str(_input_path('iran'))
SHEET_NAME = "Sheet1"

YEAR_START   = 1990
HIS_END      = 2024
FORECAST_END = 2100

SCENARIO = "central"


FAST_MODE = True


MAPE_THRESH_HIST = 15.0
R2_THRESH_HIST   = 0.70
DROP_2050_2100_MIN = -45.0


OUT_XLSX = os.path.join(str(_output_dir('iran', 'replacement')), 'Iran_Cement_Hybrid_ARXfit_StockDrivenForecast_1990_2100.xlsx')

plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11


def pick_col(df, keywords, required=True):
    cols = [str(c).strip() for c in df.columns]
    cols_u = [c.upper() for c in cols]
    for kw in keywords:
        kw_u = kw.upper()
        for c, cu in zip(cols, cols_u):
            if (c == kw) or (kw in c) or (kw_u in cu):
                return c
    if required:
        raise KeyError(f"找不到列：{keywords}；当前列名={cols}")
    return None

def to_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0.0)
    if not np.any(m):
        return np.nan
    return float(np.mean(np.abs((y_pred[m] - y_true[m]) / y_true[m])) * 100.0)

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(m) < 2:
        return np.nan
    yt, yp = y_true[m], y_pred[m]
    sse = float(np.sum((yt - yp) ** 2))
    sst = float(np.sum((yt - np.mean(yt)) ** 2))
    return float(1.0 - sse / sst) if sst > 0 else np.nan

def linear_fit_closed_form(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 2:
        return np.nan, np.nan

    x = x[m]
    y = y[m]
    x_bar = float(np.mean(x))
    y_bar = float(np.mean(y))
    sxx = float(np.sum((x - x_bar) ** 2))
    sxy = float(np.sum((x - x_bar) * (y - y_bar)))
    if sxx <= 0:
        return np.nan, np.nan

    k = sxy / sxx
    b = y_bar - k * x_bar
    return float(k), float(b)

def tail_r2_linear(years_arr, y_arr, a, b):


    years_arr = np.asarray(years_arr, float)
    y_arr = np.asarray(y_arr, float)

    m = (years_arr >= a) & (years_arr <= b) & np.isfinite(y_arr)
    if np.sum(m) < 3:
        return np.nan

    x = years_arr[m]
    y = y_arr[m]
    k, c = linear_fit_closed_form(x, y)
    if not np.isfinite(k) or not np.isfinite(c):
        return np.nan

    y_fit = k * x + c
    sse = float(np.sum((y - y_fit) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return float(1.0 - sse / sst) if sst > 0 else np.nan

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

    yh = np.asarray(yhat[m], float)
    yu = np.asarray(y[m], float)
    err = yh - yu

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    r2v = r2_score(yu, yh)

    m_mape = np.isfinite(yu) & np.isfinite(yh) & (yu != 0.0)
    if np.any(m_mape):
        mapev = float(np.mean(np.abs((yh[m_mape] - yu[m_mape]) / yu[m_mape])) * 100.0)
    else:
        mapev = np.nan

    return dict(MAE=mae, MAPE=mapev, RMSE=rmse, Bias=bias, R2=r2v)

def _pick_year(y_arr, val_arr, year):
    m = (y_arr == year)
    if np.any(m):
        v = float(val_arr[m][0])
        return v if np.isfinite(v) else np.nan
    return np.nan

def detect_cement_col_and_unit(df):
    cand_cols = [c for c in df.columns if ("水泥" in str(c)) or ("CEMENT" in str(c).upper())]
    cand_cols = sorted(cand_cols, key=lambda x: (0 if "产量" in str(x) else 1, len(str(x))))
    if not cand_cols:
        raise KeyError("未找到水泥产量列（含 '水泥' 或 'cement'）")

    col = cand_cols[0]
    s = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    vmax = np.nanmax(s)
    name = str(col)
    if ("百万吨" in name) or ("MT" in name.upper()) or ("MILLION" in name.upper()):
        tons = s * 1e6
    elif ("亿吨" in name):
        tons = s * 1e8
    elif ("吨" in name):
        tons = s
    else:
        if np.isfinite(vmax) and (5 <= vmax < 1500):
            tons = s * 1e6
        else:
            tons = s
    return col, tons


df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, engine="openpyxl").sort_values("年份").reset_index(drop=True)
df.columns = [str(c).strip() for c in df.columns]

COL_YEAR = pick_col(df, ["年份", "Year"])

COL_GDP = pick_col(df, ["GDP_合并", "GDP总量（十亿美元）", "GDP（十亿美元）", "GDP总量", "GDP"], required=False)
if COL_GDP is None:
    gdp_candidates = [c for c in df.columns if "GDP" in str(c).upper()]
    if not gdp_candidates:
        raise KeyError("未找到GDP列")
    COL_GDP = max(gdp_candidates, key=lambda c: pd.to_numeric(df[c], errors="coerce").notna().sum())

COL_POP = pick_col(df, ["人口数量", "人口", "Population", "POP"], required=True)
COL_URB = pick_col(df, ["城镇化率", "城市化率", "Urbanization", "Urban rate"], required=True)

COL_URBPOP = pick_col(df, ["城镇人口数量", "城镇人口数", "城镇人口", "Urban pop"], required=False)

COL_AREA = pick_col(df, [
    "预测_人均住宅面积", "预测人均住宅面积", "预测_城镇人均住宅面积",
    "拟合_人均住宅面积", "拟合人均住宅面积", "城镇人均住宅面积",
    "预测_人均住宅面积_Logistic", "预测_人均住宅面积_Logistic_BAU"
], required=False)
if COL_AREA is None:
    raise KeyError("未找到人均住宅面积列（预测/拟合）。请在表里加一列预测人均住宅面积。")

COL_CEMENT, cement_tons = detect_cement_col_and_unit(df)


df["__cement_tons__"] = cement_tons

for c in [COL_YEAR, COL_GDP, COL_POP, COL_URB, COL_AREA]:
    df[c] = to_num(df[c])

df[COL_YEAR] = df[COL_YEAR].astype("Int64")
df = df[(df[COL_YEAR] >= YEAR_START) & (df[COL_YEAR] <= FORECAST_END)].copy()
df[COL_YEAR] = df[COL_YEAR].astype(int)

years = df[COL_YEAR].to_numpy(int)
n = len(years)

gdp_bn = df[COL_GDP].to_numpy(float)
pop_total = df[COL_POP].to_numpy(float)
urban01 = df[COL_URB].to_numpy(float)
urban01 = urban01/100.0 if np.nanmax(urban01) > 1.5 else urban01
urban01 = np.clip(urban01, 0.02, 0.98)

if COL_URBPOP is not None:
    df[COL_URBPOP] = to_num(df[COL_URBPOP])
    urban_pop = df[COL_URBPOP].to_numpy(float)
else:
    urban_pop = pop_total * urban01

area_pc = df[COL_AREA].to_numpy(float)

y_actual = df["__cement_tons__"].to_numpy(float)
actual_mask = np.isfinite(y_actual) & (years <= HIS_END)


def ridge_fit(X, y, lam, penalize_intercept=False):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    p = X.shape[1]
    I = np.eye(p)
    if not penalize_intercept:
        I[0, 0] = 0.0
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

gdp_usd = gdp_bn * 1e9
gdppc = gdp_usd / np.maximum(pop_total, 1e-12)
ln_gdppc = np.log(np.maximum(gdppc, 1e-12))

gdp_gr = np.full(n, np.nan, float)
gdp_gr[1:] = gdp_usd[1:] / np.maximum(gdp_usd[:-1], 1e-12) - 1.0

ln_urban_pop = np.log(np.maximum(urban_pop, 1e-12))
t = (years - YEAR_START)/10.0

logy = np.log(np.maximum(y_actual, 1e-12))
logy_lag1 = np.r_[np.nan, logy[:-1]]
logy_lag2 = np.r_[np.nan, np.nan, logy[:-2]]

mask_arx = (years <= HIS_END) & np.isfinite(y_actual) & np.isfinite(logy_lag1) & np.isfinite(logy_lag2)


if FAST_MODE:
    X_arx = np.column_stack([ln_gdppc, gdp_gr, urban01, ln_urban_pop, area_pc, t, logy_lag1, logy_lag2])
else:
    def dyear(yy):
        return (years == yy).astype(float)
    d2012 = dyear(2012)
    d2018 = dyear(2018)
    d2020 = dyear(2020)
    X_arx = np.column_stack([ln_gdppc, gdp_gr, urban01, ln_urban_pop, area_pc, t, d2012, d2018, d2020, logy_lag1, logy_lag2])

X_train_raw = X_arx[mask_arx]
y_train_log = logy[mask_arx]

X_train_z, mu_arx, sd_arx = zscore_fit_transform(X_train_raw)
X_train = np.column_stack([np.ones(len(X_train_z)), X_train_z])

LAM_ARX_GRID = [0.0, 0.01, 0.1, 1.0] if FAST_MODE else [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

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
yhat_all = np.exp(X_all @ beta_arx)

arx_fit = np.full(n, np.nan, float)
arx_fit[mask_arx] = yhat_all[mask_arx]


urban_pop_sm = centered_ma(urban_pop, 3)
area_pc_sm = centered_ma(area_pc, 3)
stock = np.maximum(urban_pop_sm, 0.0) * np.maximum(area_pc_sm, 0.0)

def ci_res_iran(y: int) -> float:

    return 0.25

intensity = np.array([ci_res_iran(int(y)) for y in years], float)

def retrofit_share_iran(y: int):
    if SCENARIO == 'lowcarbon':
        pts = [(2025,0.05),(2030,0.11),(2040,0.20),(2050,0.26),(2070,0.32),(2100,0.38)]
    elif SCENARIO == 'high':
        pts = [(2025,0.03),(2030,0.06),(2040,0.10),(2050,0.14),(2070,0.18),(2100,0.22)]
    else:
        pts = [(2025,0.04),(2030,0.09),(2040,0.16),(2050,0.20),(2070,0.24),(2100,0.30)]
    if y <= pts[0][0]:
        return pts[0][1]
    if y >= pts[-1][0]:
        return pts[-1][1]
    for (ya, va), (yb, vb) in zip(pts[:-1], pts[1:]):
        if ya <= y <= yb:
            t = (y-ya)/(yb-ya)
            return va + t*(vb-va)
    return pts[-1][1]

def infra_taper_iran(y: int) -> float:

    return 1.0

def calibrate_pred_affine(pred, actual, years_arr, his_end):
    m = (years_arr <= his_end) & np.isfinite(pred) & np.isfinite(actual)
    if np.sum(m) < 2:
        return pred.copy(), dict(alpha=np.nan, beta=np.nan)

    p = np.asarray(pred[m], float)
    yv = np.asarray(actual[m], float)

    p_bar = float(np.mean(p))
    y_bar = float(np.mean(yv))
    spp = float(np.sum((p - p_bar) ** 2))
    spy = float(np.sum((p - p_bar) * (yv - y_bar)))

    if spp <= 0:
        b_hat = 0.0
        a_hat = y_bar
    else:
        b_hat = spy / spp
        b_hat = max(0.0, float(b_hat))
        a_hat = float(y_bar - b_hat * p_bar)

    pred_cal = np.maximum(a_hat + b_hat * pred, 0.0)
    return pred_cal, dict(alpha=float(a_hat), beta=float(b_hat))

def stock_model_predict(k_shape, lambda_scale, infra_factor,
                        alpha_inflow=0.45, alpha_final=0.35, max_life=100):

    life = np.arange(1, max_life+1)
    pdf_weib = weibull_min.pdf(life, c=k_shape, scale=lambda_scale)
    pdf_weib = np.nan_to_num(pdf_weib)
    pdf_weib /= np.maximum(pdf_weib.sum(), 1e-12)

    inflow = np.zeros(n)
    outflow = np.zeros(n)

    expected_life = float(weibull_min.mean(c=k_shape, scale=lambda_scale))
    steady_in = max(stock[0] / max(expected_life, 1.0), 1e-6)
    hist_inflow = np.full(max_life, steady_in)

    for t_i in range(n):
        past = np.concatenate([hist_inflow, inflow[:t_i]])
        use  = past[-max_life:]
        outflow[t_i] = float(np.sum(use[::-1] * pdf_weib[:len(use)]))
        inflow[t_i]  = (stock[t_i]-stock[t_i-1]) + outflow[t_i] if t_i > 0 else outflow[t_i]

    inflow_s = ema(inflow, alpha=alpha_inflow)

    eff_new_area = np.zeros(n)
    infra_series = np.zeros(n)
    for i, yy in enumerate(years):
        r = retrofit_share_iran(int(yy)) if yy >= 2025 else 0.0
        eff_new_area[i] = max(inflow_s[i], 0.0) * (1.0 - min(r, 0.95))
        infra_series[i] = infra_factor * infra_taper_iran(int(yy))

    pred = eff_new_area * intensity * (1.0 + infra_series)
    pred = np.maximum(pred, 0.0)
    pred = ema(pred, alpha=alpha_final)
    return pred


if FAST_MODE:

    k_grid_s = [1.97]
    lam_grid_s = [56.4]
    infra_grid_s = [2.4]
    alpha_inflow_grid = [0.45]
    alpha_final_grid  = [0.35]
else:
    k_grid_s = [1.97]
    lam_grid_s = np.round(np.linspace(30, 55, 11), 2)
    infra_grid_s = np.array([2.4], dtype=float)
    alpha_inflow_grid = [0.35, 0.45, 0.55]
    alpha_final_grid  = [0.25, 0.30, 0.35, 0.40]

best_pack = None
best_pred = None
best_cal  = None
best_hist = None


for ksh in k_grid_s:
    for lam in lam_grid_s:
        for infra in infra_grid_s:
            for ai in alpha_inflow_grid:
                for af in alpha_final_grid:
                    pred0 = stock_model_predict(ksh, lam, infra, alpha_inflow=ai, alpha_final=af)
                    pred_cal, cal_info = calibrate_pred_affine(pred0, y_actual, years, HIS_END)
                    m_hist = _metrics(pred_cal, y_actual, years, a=YEAR_START, b=HIS_END)

                    score = m_hist["MAPE"] if np.isfinite(m_hist["MAPE"]) else 1e9
                    if (best_pack is None) or (score < best_hist["MAPE"] - 1e-6) or (abs(score - best_hist["MAPE"]) <= 1e-6 and m_hist["R2"] > best_hist["R2"] + 1e-6):
                        best_pack = (ksh, lam, infra, ai, af)
                        best_pred = pred_cal
                        best_cal  = cal_info
                        best_hist = m_hist


if FAST_MODE and best_pack is not None:
    k0, lam0, infra0, ai0, af0 = best_pack
    k_grid_ref = sorted(list(set([k0, max(1.8, k0-0.2), min(3.0, k0+0.2)])))
    lam_grid_ref = sorted(list(set([lam0-2, lam0, lam0+2])))
    infra_grid_ref = sorted(list(set([infra0-0.2, infra0, infra0+0.2])))

    for ksh in k_grid_ref:
        for lam in lam_grid_ref:
            for infra in infra_grid_ref:
                pred0 = stock_model_predict(ksh, lam, infra, alpha_inflow=ai0, alpha_final=af0)
                pred_cal, cal_info = calibrate_pred_affine(pred0, y_actual, years, HIS_END)
                m_hist = _metrics(pred_cal, y_actual, years, a=YEAR_START, b=HIS_END)
                score = m_hist["MAPE"] if np.isfinite(m_hist["MAPE"]) else 1e9
                if (score < best_hist["MAPE"] - 1e-6) or (abs(score - best_hist["MAPE"]) <= 1e-6 and m_hist["R2"] > best_hist["R2"] + 1e-6):
                    best_pack = (ksh, lam, infra, ai0, af0)
                    best_pred = pred_cal
                    best_cal  = cal_info
                    best_hist = m_hist

stock_pred_all = best_pred


hybrid = np.full(n, np.nan, float)

mask_hist = (years <= HIS_END)
hybrid[mask_hist] = arx_fit[mask_hist]

miss = mask_hist & (~np.isfinite(hybrid))
hybrid[miss] = np.where(np.isfinite(y_actual[miss]), y_actual[miss], stock_pred_all[miss])

mask_fc = (years >= HIS_END + 1)
hybrid[mask_fc] = stock_pred_all[mask_fc]

y_arx_2024 = _pick_year(years, hybrid, 2024)
y_stock_2024 = _pick_year(years, stock_pred_all, 2024)
if np.isfinite(y_arx_2024) and np.isfinite(y_stock_2024) and y_stock_2024 > 0:
    splice_scale = y_arx_2024 / y_stock_2024
    hybrid[mask_fc] *= splice_scale
else:
    splice_scale = np.nan


m_hybrid_hist = _metrics(hybrid, y_actual, years, a=YEAR_START, b=HIS_END)

resid_hist_mask = (years <= HIS_END) & np.isfinite(y_actual) & np.isfinite(hybrid)
resid_hist = y_actual[resid_hist_mask] - hybrid[resid_hist_mask]
dw = durbin_watson(resid_hist)
ar1 = ar1_rho_ols(resid_hist)


tail_start, tail_end = 2035, 2070
m_tail = (years >= tail_start) & (years <= tail_end) & np.isfinite(hybrid)

if np.sum(m_tail) >= 3:
    x_tail = years[m_tail].astype(float)
    y_tail = hybrid[m_tail].astype(float)
    k_lin, b_lin = linear_fit_closed_form(x_tail, y_tail)
    if np.isfinite(k_lin):
        slope_pct = 100.0 * k_lin / np.mean(y_tail) if np.mean(y_tail) != 0 else np.nan
    else:
        slope_pct = np.nan
    tail_r2 = tail_r2_linear(years, hybrid, tail_start, tail_end)
else:
    slope_pct = np.nan
    tail_r2 = np.nan

y2050 = _pick_year(years, hybrid, 2050)
y2100 = _pick_year(years, hybrid, 2100)
drop_50_00 = 100.0*(y2100/y2050 - 1.0) if (np.isfinite(y2050) and y2050 > 0 and np.isfinite(y2100)) else np.nan

nonneg_ok = bool(np.all(hybrid[np.isfinite(hybrid)] >= 0))
drop_ok = bool(np.isfinite(drop_50_00) and (drop_50_00 >= DROP_2050_2100_MIN))
pass_hist = (m_hybrid_hist["MAPE"] <= MAPE_THRESH_HIST) and (m_hybrid_hist["R2"] >= R2_THRESH_HIST)
ok_all = pass_hist and nonneg_ok and drop_ok

summary = (
    f"综合判定：{'✅ 通过' if ok_all else '❌ 未通过'}；"
    f"FAST_MODE={FAST_MODE}；"
    f"ARX(lambda={lam_arx})：MAPE={m_arx:.2f}%, R2={r2_arx:.3f}；"
    f"Stock(best={best_pack}, calib a={best_cal['alpha']:.2e}, b={best_cal['beta']:.3f})；"
    f"Hybrid(<=2024): MAPE={m_hybrid_hist['MAPE']:.2f}%, RMSE={m_hybrid_hist['RMSE']:.3e}, R2={m_hybrid_hist['R2']:.3f}, DW={dw:.3f}, AR1={ar1:.3f}；"
    f"Tail(2035–2070): Slope={slope_pct:.2f}%/y, Tail_R2={tail_r2:.3f}；"
    f"2050→2100={drop_50_00:.1f}%（要求 >= {DROP_2050_2100_MIN}%）"
)

print("\n================ IRAN HYBRID MODEL SUMMARY ================")
print(summary)
print(f"Splice scale (2025+ align to 2024 ARX): {splice_scale}")


out = pd.DataFrame({
    "Year": years,
    "Actual_cement(tons)": y_actual,
    "ARX_fit_hist(tons)": arx_fit,
    "StockDriven_pred_all(tons)": stock_pred_all,
    "Hybrid_final(tons)": hybrid,
    "GDP_total_bnUSD": gdp_bn,
    "Pop_total": pop_total,
    "Urban_rate(0-1)": urban01,
    "Urban_pop": urban_pop,
    "Area_pc_m2": area_pc
})


diagnostics = pd.DataFrame([
    {"Block": "Hybrid_hist(1990-2024)", "Metric": "MAE",      "Value": m_hybrid_hist["MAE"],  "Threshold": np.nan,            "Pass": np.nan},
    {"Block": "Hybrid_hist(1990-2024)", "Metric": "MAPE(%)",  "Value": m_hybrid_hist["MAPE"], "Threshold": MAPE_THRESH_HIST,   "Pass": bool(m_hybrid_hist["MAPE"] <= MAPE_THRESH_HIST) if np.isfinite(m_hybrid_hist["MAPE"]) else np.nan},
    {"Block": "Hybrid_hist(1990-2024)", "Metric": "RMSE",     "Value": m_hybrid_hist["RMSE"], "Threshold": np.nan,            "Pass": np.nan},
    {"Block": "Hybrid_hist(1990-2024)", "Metric": "Bias",     "Value": m_hybrid_hist["Bias"], "Threshold": np.nan,            "Pass": np.nan},
    {"Block": "Hybrid_hist(1990-2024)", "Metric": "R2",       "Value": m_hybrid_hist["R2"],   "Threshold": R2_THRESH_HIST,     "Pass": bool(m_hybrid_hist["R2"] >= R2_THRESH_HIST) if np.isfinite(m_hybrid_hist["R2"]) else np.nan},

    {"Block": "Residual(1990-2024)", "Metric": "Durbin-Watson", "Value": dw,  "Threshold": np.nan, "Pass": np.nan},
    {"Block": "Residual(1990-2024)", "Metric": "AR1_rho(OLS)",  "Value": ar1, "Threshold": np.nan, "Pass": np.nan},

    {"Block": "Tail", "Metric": "TailSlope_2035-2070(%/yr)", "Value": slope_pct, "Threshold": np.nan, "Pass": np.nan},
    {"Block": "Tail", "Metric": "Tail_R2_2035-2070",         "Value": tail_r2,   "Threshold": np.nan, "Pass": np.nan},

    {"Block": "Trend", "Metric": "Drop_2050_to_2100(%)", "Value": drop_50_00, "Threshold": DROP_2050_2100_MIN,
     "Pass": bool(drop_50_00 >= DROP_2050_2100_MIN) if np.isfinite(drop_50_00) else np.nan},

    {"Block": "Sanity", "Metric": "NonNegative",      "Value": 1 if nonneg_ok else 0, "Threshold": 1, "Pass": nonneg_ok},
    {"Block": "Sanity", "Metric": "HistPass(MAPE&R2)","Value": 1 if pass_hist else 0, "Threshold": 1, "Pass": pass_hist},
    {"Block": "Sanity", "Metric": "OverallPass",      "Value": 1 if ok_all else 0,    "Threshold": 1, "Pass": ok_all},

    {"Block": "ARX", "Metric": "best_lambda",   "Value": lam_arx, "Threshold": np.nan, "Pass": np.nan},
    {"Block": "ARX", "Metric": "train_MAPE(%)", "Value": m_arx,   "Threshold": np.nan, "Pass": np.nan},
    {"Block": "ARX", "Metric": "train_R2",      "Value": r2_arx,  "Threshold": np.nan, "Pass": np.nan},

    {"Block": "Stock", "Metric": "best_pack(k,lambda,infra,ai,af)", "Value": str(best_pack), "Threshold": np.nan, "Pass": np.nan},
    {"Block": "Stock", "Metric": "calib_alpha(intercept)",          "Value": best_cal.get("alpha", np.nan), "Threshold": np.nan, "Pass": np.nan},
    {"Block": "Stock", "Metric": "calib_beta(slope)",               "Value": best_cal.get("beta", np.nan),  "Threshold": np.nan, "Pass": np.nan},

    {"Block": "Splice", "Metric": "splice_scale(2025+)", "Value": splice_scale, "Threshold": np.nan, "Pass": np.nan},
    {"Block": "Meta",   "Metric": "SCENARIO",   "Value": SCENARIO,          "Threshold": np.nan, "Pass": np.nan},
    {"Block": "Meta",   "Metric": "FAST_MODE",  "Value": int(bool(FAST_MODE)), "Threshold": np.nan, "Pass": np.nan},
    {"Block": "Meta",   "Metric": "SummaryText","Value": summary,           "Threshold": np.nan, "Pass": np.nan},
])


k_best, lambda_best, infra_best, ai_best, af_best = best_pack


max_life_used = 100

life = np.arange(1, max_life_used + 1)
pdf = weibull_min.pdf(life, c=k_best, scale=lambda_best)
pdf = np.nan_to_num(pdf)
pdf_trunc = pdf / np.maximum(pdf.sum(), 1e-12)


mean_theory   = float(weibull_min.mean(c=k_best, scale=lambda_best))
std_theory    = float(weibull_min.std(c=k_best, scale=lambda_best))
median_theory = float(weibull_min.median(c=k_best, scale=lambda_best))


mean_trunc = float(np.sum(life * pdf_trunc))
var_trunc  = float(np.sum((life - mean_trunc) ** 2 * pdf_trunc))

weibull_params = pd.DataFrame([
    {"Param": "k_shape(best)", "Value": k_best},
    {"Param": "lambda_scale(best)", "Value": lambda_best},
    {"Param": "max_life(truncation_used)", "Value": max_life_used},
    {"Param": "mean_life_theory", "Value": mean_theory},
    {"Param": "std_life_theory", "Value": std_theory},
    {"Param": "median_life_theory", "Value": median_theory},
    {"Param": "mean_life_truncated(pdf_normalized)", "Value": mean_trunc},
    {"Param": "var_life_truncated(pdf_normalized)",  "Value": var_trunc},
    {"Param": "pdf_sum_after_norm(should_be_1)",     "Value": float(pdf_trunc.sum())},
    {"Param": "infra_factor(best)", "Value": infra_best},
    {"Param": "alpha_inflow(best)", "Value": ai_best},
    {"Param": "alpha_final(best)",  "Value": af_best},
])

weibull_table = pd.DataFrame({
    "Age": life.astype(int),
    "pdf_truncated_normalized": pdf_trunc,
    "cdf_theory": weibull_min.cdf(life, c=k_best, scale=lambda_best),
})
weibull_table["survival_theory"] = 1.0 - weibull_table["cdf_theory"]


with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    out.to_excel(writer, sheet_name="Forecast", index=False)
    diagnostics.to_excel(writer, sheet_name="Diagnostics", index=False)
    weibull_params.to_excel(writer, sheet_name="Weibull_Params", index=False)
    weibull_table.to_excel(writer, sheet_name="Weibull_Table", index=False)

print(f"\n✅ Exported (with Diagnostics & Weibull): {OUT_XLSX}")


mask_disp = (years >= YEAR_START) & (years <= FORECAST_END)

plt.figure(figsize=(13,7))
plt.plot(years[mask_disp], hybrid[mask_disp]/1e6, linewidth=2.2, label="Hybrid final (ARX fit -> Stock-driven forecast)")
plt.plot(years[mask_disp], stock_pred_all[mask_disp]/1e6, linewidth=1.5, alpha=0.75, label="Stock-driven (calibrated) reference")
plt.plot(years[mask_disp], arx_fit[mask_disp]/1e6, linewidth=1.2, alpha=0.75, label="ARX one-step fit (hist)")

mA = np.isfinite(y_actual) & (years <= HIS_END)
plt.scatter(years[mA], y_actual[mA]/1e6, s=18, alpha=0.7, label="Actual (<=2024)")
plt.axvline(HIS_END+1, color="gray", linestyle="--", label="Forecast start")

plt.xlim(YEAR_START, FORECAST_END)
plt.xlabel("Year")
plt.ylabel("Cement production (Million tons)")
plt.title("Iran Cement | Hybrid: ARX fit (1990-2024) + Stock-driven forecast (2025-2100)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.annotate(summary.replace("综合判定：",""), xy=(0.02,0.98), xycoords="axes fraction", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", alpha=0.85))
plt.tight_layout()
plt.show()
