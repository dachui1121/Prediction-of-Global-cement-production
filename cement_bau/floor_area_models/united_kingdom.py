from cement_bau.floor_area_runtime import floor_input as _floor_input, floor_output as _floor_output, floor_anchors as _floor_anchors
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from pathlib import Path


ANCHORS_UK = _floor_anchors('united_kingdom')


EXCEL_PATH = str(_floor_input('united_kingdom', 'macro'))
SHEET_NAME = "Sheet1"


G_LONG = 0.012
TAU    = 25.0


L_MAX = 60.0
L_LOWER_MARGIN = 2.0
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

def logistic_cov(params, ln_gdppc, U, ln_pop):
    L, k, a, b, c, d = params
    z = a + b * ln_gdppc + c * U + d * ln_pop
    return L / (1.0 + np.exp(-k * z))


df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, engine="openpyxl")
df.columns = df.columns.astype(str).str.strip()
df = df.dropna(how="all").replace([np.inf, -np.inf], np.nan)

year_col = pick_col(df, ["年份", "Year"])
pop_col  = pick_col(df, ["人口数量", "人口", "人口数"])
urb_col  = pick_col(df, ["城镇化率", "城市化率"])
gdp_col  = pick_col(df, ["GDP(十亿美元)", "GDP总量", "GDP"])

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

print(f"✅ UK列识别：年份='{year_col}', 人口='{pop_col}', 城镇化率='{urb_col}', GDP='{gdp_col}'")
print(f"✅ 年份范围：{years.min()}–{years.max()}（共 {len(years)} 年）")


mask_gdp = np.isfinite(gdp_in) & (gdp_in > 0) & np.isfinite(pop) & (pop > 0)
years_gdp = years[mask_gdp]
if len(years_gdp) < 10:
    raise ValueError("可用GDP点太少。请确认 Excel 中已有历史GDP，并填入 2025–2030 GDP锚点。")


gdp_pc = np.full_like(pop, np.nan, dtype=float)
gdp_pc[mask_gdp] = gdp_in[mask_gdp] / pop[mask_gdp]

last_gdp_year = int(years_gdp.max())

def growth_from_2025_2030():
    if (2025 in idx) and (2030 in idx):
        i25, i30 = idx[2025], idx[2030]
        if np.isfinite(gdp_pc[i25]) and np.isfinite(gdp_pc[i30]) and gdp_pc[i25] > 0 and gdp_pc[i30] > 0:
            return float((np.log(gdp_pc[i30]) - np.log(gdp_pc[i25])) / (2030 - 2025))
    return None

g0 = growth_from_2025_2030()
if g0 is None:
    g0 = estimate_log_growth(years_gdp, gdp_pc[mask_gdp], win=8)


i_last = idx[last_gdp_year]
for i in range(i_last + 1, len(years)):
    t = years[i]
    gt = G_LONG + (g0 - G_LONG) * np.exp(-(t - last_gdp_year) / TAU)
    gdp_pc[i] = gdp_pc[i - 1] * np.exp(gt)


gdp_ext = gdp_pc * pop


gdp_merged = gdp_in.copy()
gdp_merged[~np.isfinite(gdp_merged)] = gdp_ext[~np.isfinite(gdp_merged)]


gdp_pc_all = gdp_merged / np.maximum(pop, 1e-12)
ln_gdppc_all = np.log(np.maximum(gdp_pc_all, 1e-12))
ln_pop_all   = np.log(np.maximum(pop, 1e-12))


anchor_years = np.array(sorted(ANCHORS_UK.keys()), dtype=int)
anchor_area  = np.array([ANCHORS_UK[y] for y in anchor_years], dtype=float)

ln_gdppc_a = np.array([ln_gdppc_all[idx[y]] if y in idx else np.nan for y in anchor_years], dtype=float)
U_a        = np.array([urb01[idx[y]]        if y in idx else np.nan for y in anchor_years], dtype=float)
ln_pop_a   = np.array([ln_pop_all[idx[y]]   if y in idx else np.nan for y in anchor_years], dtype=float)

m_fit = np.isfinite(ln_gdppc_a) & np.isfinite(U_a) & np.isfinite(ln_pop_a) & np.isfinite(anchor_area)
if m_fit.sum() < 4:
    raise ValueError(f"有效锚点不足：{m_fit.sum()} 个。请检查锚点年份是否都存在，并且 GDP/人口/城镇化率无缺失。")

x1 = ln_gdppc_a[m_fit]
x2 = U_a[m_fit]
x3 = ln_pop_a[m_fit]
y  = anchor_area[m_fit]

def residuals(params):
    pred = logistic_cov(params, x1, x2, x3)

    reg = REG_W * np.array([params[1], params[3], params[4], params[5]])
    return np.r_[pred - y, reg]

ymax = float(np.nanmax(y))
L_min = ymax + L_LOWER_MARGIN
if L_min >= L_MAX:
    raise ValueError(f"锚点最大值={ymax:.2f} 导致 L_min={L_min:.2f} >= L_MAX={L_MAX}，请调大 L_MAX 或降低 L_LOWER_MARGIN。")


L0 = float(np.clip(max(45.0, ymax + 10.0), L_min + 0.5, L_MAX - 0.5))
k0, a0, b0, c0, d0 = 1.0, -1.0, 0.8, 1.5, 0.0

p0 = np.array([L0, k0, a0, b0, c0, d0], float)
lb = np.array([L_min, 1e-3, -50.0, 0.0, 0.0, -10.0], float)
ub = np.array([L_MAX, 50.0,  50.0, 50.0, 50.0,  10.0], float)

res = least_squares(
    residuals, p0, bounds=(lb, ub),
    loss=LOSS, f_scale=F_SCALE,
    max_nfev=200000
)

L_hat, k_hat, a_hat, b_hat, c_hat, d_hat = res.x
area_pred = logistic_cov(res.x, ln_gdppc_all, urb01, ln_pop_all)


anchor_col = np.full(len(years), np.nan, dtype=float)
for yy, vv in ANCHORS_UK.items():
    if yy in idx:
        anchor_col[idx[yy]] = vv

out = pd.DataFrame({
    "年份": years,
    "人口": pop,
    "城镇化率(0-1)": urb01,
    "GDP_原始输入(十亿美元)": gdp_in,
    "GDP_外推(十亿美元)": gdp_ext,
    "GDP_合并(十亿美元)": gdp_merged,
    "人均GDP(GDPpc)": gdp_pc_all,
    "ln(人均GDP)": ln_gdppc_all,
    "ln(人口)": ln_pop_all,
    "锚点_人均住宅面积(㎡/人)": anchor_col,
    "预测_人均住宅面积_Logistic(㎡/人)": area_pred
})


sandbox_out = (_floor_output('united_kingdom') / '英国_GDP外推_人均住宅面积_Logistic_BAU_1989-2100.xlsx')
out.to_excel(sandbox_out, index=False, engine="openpyxl")


desktop_out = (_floor_output('united_kingdom') / '英国_GDP外推_人均住宅面积_Logistic_BAU_1989-2100.xlsx')
try:
    out.to_excel(desktop_out, index=False, engine="openpyxl")
except Exception:
    pass

print("\n================== DONE ==================")
print(f"GDP外推：锚点为 2025–2030 人均GDP对数增速；长期收敛到 {G_LONG*100:.2f}%/y，TAU={TAU}")
print("Logistic（不固定L但有上限）参数：")
print(f"  L_hat={L_hat:.6f} (<= {L_MAX})")
print(f"  k_hat={k_hat:.6f}")
print(f"  a_hat={a_hat:.6f}")
print(f"  b_hat={b_hat:.6f}  (lnGDPpc)")
print(f"  c_hat={c_hat:.6f}  (Urbanization)")
print(f"  d_hat={d_hat:.6f}  (ln_pop)")
print(f"拟合：loss={LOSS}, status={res.status}, cost={res.cost:.6f}, nfev={res.nfev}")
print(f"📁 输出：\n  - {sandbox_out}\n  - {desktop_out}")
print("==========================================\n")
