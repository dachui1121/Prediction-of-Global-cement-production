from cement_bau.floor_area_runtime import floor_input as _floor_input, floor_output as _floor_output, floor_anchors as _floor_anchors
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


ANCHORS_ID = _floor_anchors('indonesia')


excel_path = str(_floor_input('indonesia', 'macro'))
sheet_name = "Sheet1"


G_LONG = 0.012
TAU    = 25.0


L_MAX = 60.0
L_LOWER_MARGIN = 2.0
USE_POP_TERM = True
POP_TERM_KIND = "ln_pop"

LOSS = "soft_l1"
F_SCALE = 1.0
REG_W = 0.01

YEAR_MIN = 1989
YEAR_MAX = 2100

def pick_col(df_, keywords, required=True):
    cols = df_.columns.astype(str).tolist()
    for kw in keywords:
        for c in cols:
            if c == kw or (kw in c):
                return c
    if required:
        raise KeyError(f"找不到列（关键词 {keywords}），当前列名：{cols}")
    return None

def to_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

def estimate_log_growth(years_arr, series_arr, win=8):
    y = np.asarray(years_arr, int)
    s = np.asarray(series_arr, float)
    m = np.isfinite(s) & (s > 0)
    y, s = y[m], s[m]
    if len(s) < 3:
        return 0.02
    end = y[-1]
    start = max(y[0], end - win)
    mm = (y >= start) & (y <= end)
    y2, s2 = y[mm], s[mm]
    if len(s2) < 3:
        return 0.02
    g = (np.log(s2[-1]) - np.log(s2[0])) / max(1, (y2[-1] - y2[0]))
    return float(np.clip(g, -0.02, 0.12))

def logistic_cov(params, ln_gdppc, U, pop_term=None, use_pop=False):
    if use_pop:
        L, k, a, b, c, d = params
        z = a + b * ln_gdppc + c * U + d * pop_term
    else:
        L, k, a, b, c = params
        z = a + b * ln_gdppc + c * U
    return L / (1.0 + np.exp(-k * z))


df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
df.columns = df.columns.astype(str).str.strip()
df = df.dropna(how="all").replace([np.inf, -np.inf], np.nan)

year_col = pick_col(df, ["年份", "Year"])
pop_col  = pick_col(df, ["人口"])
urb_col  = pick_col(df, ["城镇化率", "城市化率"])
gdp_col  = pick_col(df, ["GDP"])

df[year_col] = to_num(df[year_col])
df[pop_col]  = to_num(df[pop_col])
df[urb_col]  = to_num(df[urb_col])
df[gdp_col]  = to_num(df[gdp_col])

df = df.dropna(subset=[year_col, pop_col, urb_col]).copy()
df[year_col] = df[year_col].round().astype(int)
df = df.sort_values(year_col).drop_duplicates(subset=[year_col]).reset_index(drop=True)

years_all = df[year_col].to_numpy(dtype=int)
pop_all   = df[pop_col].to_numpy(dtype=float)
urb_all   = df[urb_col].to_numpy(dtype=float)
gdp_in_all = df[gdp_col].to_numpy(dtype=float)

urb01_all = urb_all / 100.0 if np.nanmax(urb_all) > 1.5 else urb_all.copy()

m = (years_all >= YEAR_MIN) & (years_all <= YEAR_MAX)
years = years_all[m]
pop   = pop_all[m]
urb01 = urb01_all[m]
gdp_in = gdp_in_all[m]
idx = {y: i for i, y in enumerate(years)}


mask_gdp = np.isfinite(gdp_in) & (gdp_in > 0) & np.isfinite(pop) & (pop > 0)
years_gdp = years[mask_gdp]
if len(years_gdp) < 5:
    raise ValueError("可用GDP点太少，请确认表中已有 2025–2030 GDP。")

gdp_pc_usd = np.full_like(pop, np.nan, dtype=float)
gdp_pc_usd[mask_gdp] = (gdp_in[mask_gdp] * 1e9) / pop[mask_gdp]
last_gdp_year = int(years_gdp.max())

def growth_from_2025_2030():
    if (2025 in idx) and (2030 in idx):
        i25, i30 = idx[2025], idx[2030]
        if np.isfinite(gdp_pc_usd[i25]) and np.isfinite(gdp_pc_usd[i30]) and gdp_pc_usd[i25] > 0 and gdp_pc_usd[i30] > 0:
            return float((np.log(gdp_pc_usd[i30]) - np.log(gdp_pc_usd[i25])) / 5.0)
    return None

g0 = growth_from_2025_2030()
if g0 is None:
    g0 = estimate_log_growth(years_gdp, gdp_pc_usd[mask_gdp], win=8)

i_last = idx[last_gdp_year]
for i in range(i_last + 1, len(years)):
    t = years[i]
    gt = G_LONG + (g0 - G_LONG) * np.exp(-(t - last_gdp_year) / TAU)
    gdp_pc_usd[i] = gdp_pc_usd[i-1] * np.exp(gt)

gdp_ext_billion = (gdp_pc_usd * pop) / 1e9
gdp_merged_billion = gdp_in.copy()
gdp_merged_billion[~np.isfinite(gdp_merged_billion)] = gdp_ext_billion[~np.isfinite(gdp_merged_billion)]

gdp_pc_usd_all = (gdp_merged_billion * 1e9) / np.maximum(pop, 1e-12)
ln_gdppc_all = np.log(np.maximum(gdp_pc_usd_all, 1e-12))

if USE_POP_TERM:
    if POP_TERM_KIND == "ln_pop":
        pop_term_all = np.log(np.maximum(pop, 1e-12))
    else:
        pop_term_all = np.log(np.maximum(pop * urb01, 1e-12))
else:
    pop_term_all = None


anchor_years = np.array(sorted(ANCHORS_ID.keys()), dtype=int)
anchor_area  = np.array([ANCHORS_ID[y] for y in anchor_years], dtype=float)

ln_gdppc_a = np.array([ln_gdppc_all[idx[y]] for y in anchor_years], dtype=float)
U_a        = np.array([urb01[idx[y]] for y in anchor_years], dtype=float)
P_a        = np.array([pop_term_all[idx[y]] for y in anchor_years], dtype=float)

x1, x2, x3, y = ln_gdppc_a, U_a, P_a, anchor_area

def residuals(params):
    pred = logistic_cov(params, x1, x2, x3, use_pop=True)
    reg = REG_W * np.array([params[1], params[3], params[4], params[5]])
    return np.r_[pred - y, reg]

ymax = float(np.nanmax(y))
L_min = ymax + L_LOWER_MARGIN
L0 = float(np.clip(max(35.0, ymax + 10.0), L_min + 0.5, L_MAX - 0.5))

p0 = np.array([L0, 1.0, -1.0, 0.5, 2.0, 0.0], float)
lb = np.array([L_min, 1e-3, -50.0, 0.0, 0.0, -10.0], float)
ub = np.array([L_MAX, 50.0,  50.0, 50.0, 50.0,  10.0], float)

res = least_squares(residuals, p0, bounds=(lb, ub), loss=LOSS, f_scale=F_SCALE, max_nfev=200000)
p_hat = res.x

A_pred = logistic_cov(p_hat, ln_gdppc_all, urb01, pop_term_all, use_pop=True)


anchor_col = np.full(len(years), np.nan, dtype=float)
for yk, vk in ANCHORS_ID.items():
    anchor_col[idx[yk]] = vk

out = pd.DataFrame({
    "年份": years,
    "人口": pop,
    "城镇化率(0-1)": urb01,
    "GDP_原始输入(十亿美元)": gdp_in,
    "GDP_外推(十亿美元)": gdp_ext_billion,
    "GDP_合并(十亿美元)": gdp_merged_billion,
    "人均GDP(USD/人)": gdp_pc_usd_all,
    "ln(人均GDP)": ln_gdppc_all,
    "ln(人口)": pop_term_all,
    "锚点_人均住宅面积(㎡/人)": anchor_col,
    "预测_人均住宅面积_Logistic_BAU(㎡/人)": A_pred
})

out.to_excel(str(_floor_output('indonesia') / '印尼_GDP外推_人均住宅面积_Logistic_BAU_1989-2100.xlsx'), index=False)
print("DONE")
