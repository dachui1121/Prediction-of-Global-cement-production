from cement_bau.floor_area_runtime import floor_input as _floor_input, floor_output as _floor_output, floor_anchors as _floor_anchors
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from pathlib import Path


ANCHORS_BR = _floor_anchors('brazil')

excel_path = str(_floor_input('brazil', 'macro'))
sheet_name = "Sheet1"

YEAR_MIN, YEAR_MAX = 1989, 2100
G_LONG, TAU = 0.012, 25.0


L_MAX = 80.0
L_LOWER_MARGIN = 2.0

LOSS, F_SCALE, REG_W = "soft_l1", 1.0, 0.01
POP_REF_YEAR = 1990


def to_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

def pick_col(df_, keywords, required=True):
    cols = df_.columns.astype(str).tolist()
    for kw in keywords:
        for c in cols:
            if c == kw or (kw in c):
                return c
    if required:
        raise KeyError(f"找不到列（关键词 {keywords}），当前列名：{cols}")
    return None

def estimate_log_growth(years_arr, series_arr, win=8):
    y = np.asarray(years_arr, int)
    s = np.asarray(series_arr, float)
    m = np.isfinite(s) & (s > 0)
    y, s = y[m], s[m]
    if len(s) < 3:
        return 0.02
    end = int(y[-1])
    start = max(int(y[0]), end - int(win))
    mm = (y >= start) & (y <= end)
    y2, s2 = y[mm], s[mm]
    if len(s2) < 3:
        return 0.02
    g = (np.log(s2[-1]) - np.log(s2[0])) / max(1, (y2[-1] - y2[0]))
    return float(np.clip(g, -0.02, 0.12))


def logistic_kpop(params, ln_gdppc, U, pop_term):
    L, k0, k1, a, b, c = params

    kt = np.maximum(1e-6, k0 + k1 * pop_term)
    z  = a + b * ln_gdppc + c * U
    return L / (1.0 + np.exp(-kt * z))


df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
df.columns = df.columns.astype(str).str.strip()
df = df.dropna(how="all").replace([np.inf, -np.inf], np.nan)

year_col = pick_col(df, ["年份", "Year"])
pop_col  = pick_col(df, ["巴西人口数量", "人口（人）", "人口", "人口数"])
urb_col  = pick_col(df, ["城镇化率", "城市化率"])
gdp_col  = pick_col(df, ["巴西GDP总量", "GDP总量（十亿美元）", "GDP总量", "GDP"])

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

urb01_all = urb_all/100.0 if np.nanmax(urb_all)>1.5 else urb_all.copy()

m = (years_all>=YEAR_MIN) & (years_all<=YEAR_MAX)
years = years_all[m]
pop   = pop_all[m]
urb01 = urb01_all[m]
gdp_in= gdp_in_all[m]
idx = {y:i for i,y in enumerate(years)}

print(f"✅ 列识别：{year_col=}, {pop_col=}, {urb_col=}, {gdp_col=}")
print(f"✅ 年份范围：{years.min()}–{years.max()}")


mask_gdp = np.isfinite(gdp_in) & (gdp_in>0) & np.isfinite(pop) & (pop>0)
years_gdp = years[mask_gdp]
if len(years_gdp) < 10:
    raise ValueError("GDP可用点过少，请检查 Sheet1 是否包含历史GDP及 2025–2030 GDP锚点。")

gdp_pc = np.full_like(pop, np.nan, float)
gdp_pc[mask_gdp] = gdp_in[mask_gdp] / pop[mask_gdp]

def growth_from_2025_2030():
    if 2025 in idx and 2030 in idx:
        i25, i30 = idx[2025], idx[2030]
        if np.isfinite(gdp_pc[i25]) and np.isfinite(gdp_pc[i30]) and gdp_pc[i25]>0 and gdp_pc[i30]>0:
            return float((np.log(gdp_pc[i30]) - np.log(gdp_pc[i25])) / 5.0)
    return None

g0 = growth_from_2025_2030()
if g0 is None:
    g0 = estimate_log_growth(years_gdp, gdp_pc[mask_gdp], win=8)

last_gdp_year = int(years_gdp.max())
i_last = idx[last_gdp_year]
for i in range(i_last+1, len(years)):
    t = years[i]
    gt = G_LONG + (g0 - G_LONG) * np.exp(-(t-last_gdp_year)/TAU)
    gdp_pc[i] = gdp_pc[i-1] * np.exp(gt)

gdp_ext = gdp_pc * pop
gdp_merged = gdp_in.copy()
gdp_merged[~np.isfinite(gdp_merged)] = gdp_ext[~np.isfinite(gdp_merged)]

gdp_pc_all = gdp_merged / np.maximum(pop, 1e-12)
ln_gdppc_all = np.log(np.maximum(gdp_pc_all, 1e-12))


if POP_REF_YEAR not in idx:
    pop_ref_year = int(years.min())
else:
    pop_ref_year = POP_REF_YEAR
pop_ref = float(pop[idx[pop_ref_year]])
pop_term_all = np.maximum(0.0, np.log(np.maximum(pop / max(pop_ref, 1e-12), 1e-12)))


anchor_years = np.array(sorted(ANCHORS_BR.keys()), int)
anchor_area  = np.array([ANCHORS_BR[y] for y in anchor_years], float)

ln_a = np.array([ln_gdppc_all[idx[y]] if y in idx else np.nan for y in anchor_years], float)
U_a  = np.array([urb01[idx[y]]        if y in idx else np.nan for y in anchor_years], float)
P_a  = np.array([pop_term_all[idx[y]] if y in idx else np.nan for y in anchor_years], float)

m_fit = np.isfinite(ln_a) & np.isfinite(U_a) & np.isfinite(P_a) & np.isfinite(anchor_area)
if m_fit.sum() < 4:
    raise ValueError(f"有效锚点不足：{m_fit.sum()} 个。")

x1, x2, x3, y = ln_a[m_fit], U_a[m_fit], P_a[m_fit], anchor_area[m_fit]

ymax = float(np.max(y))
L_min = ymax + L_LOWER_MARGIN
if L_min >= L_MAX:
    raise ValueError("L_MAX太小或锚点太大，请调大L_MAX或调小margin。")


p0 = np.array([
    min(L_MAX-0.5, max(L_min+0.5, ymax+10.0)),
    1.0,
    0.5,
    -1.0,
    0.6,
    2.0
], float)


lb = np.array([L_min, 1e-4, 0.0, -50.0, 0.0, 0.0], float)
ub = np.array([L_MAX, 50.0, 50.0,  50.0, 50.0, 50.0], float)

def residuals(params):
    pred = logistic_kpop(params, x1, x2, x3)

    reg = REG_W * np.array([params[1], params[2], params[4], params[5]])
    return np.r_[pred - y, reg]

res = least_squares(residuals, p0, bounds=(lb, ub), loss=LOSS, f_scale=F_SCALE, max_nfev=200000)
p_hat = res.x
L_hat, k0_hat, k1_hat, a_hat, b_hat, c_hat = p_hat


A_pred = logistic_kpop(p_hat, ln_gdppc_all, urb01, pop_term_all)


anchor_col = np.full(len(years), np.nan, float)
for yy,vv in ANCHORS_BR.items():
    if yy in idx:
        anchor_col[idx[yy]] = vv

A_yoy = np.full_like(A_pred, np.nan, float)
A_yoy[1:] = A_pred[1:] - A_pred[:-1]

out = pd.DataFrame({
    "年份": years,
    "人口(人)": pop,
    "城镇化率(0-1)": urb01,
    "GDP_原始输入(十亿美元)": gdp_in,
    "GDP_外推(十亿美元)": gdp_ext,
    "GDP_合并(十亿美元)": gdp_merged,
    "人均GDP(GDPpc)": gdp_pc_all,
    "ln(人均GDP)": ln_gdppc_all,
    f"pop_term=max(0,ln(pop/pop_ref)), ref={pop_ref_year}": pop_term_all,
    "锚点_人均住宅面积(㎡/人)": anchor_col,
    "预测_人均住宅面积(㎡/人)": np.round(A_pred, 6),
    "年增量(㎡/人/年)": np.round(A_yoy, 6),
})

out_path = (_floor_output('brazil') / '巴西_GDP外推_人均住宅面积_BAU_增强版_k受人口影响_1989-2100.xlsx')
out.to_excel(out_path, index=False, engine="openpyxl")

print("\n================== DONE ==================")
print(f"GDP锚点增速 g0={g0*100:.3f}%/y, 长期收敛={G_LONG*100:.3f}%/y, TAU={TAU}")
print("参数：")
print(f"  L={L_hat:.6f} (<= {L_MAX})")
print(f"  k0={k0_hat:.6f} (>0)")
print(f"  k1={k1_hat:.6f} (>=0, 受人口项影响增长速度)")
print(f"  a={a_hat:.6f}, b={b_hat:.6f}(>=0), c={c_hat:.6f}(>=0)")
print(f"拟合：loss={LOSS}, status={res.status}, cost={res.cost:.6f}, nfev={res.nfev}")
print(f"📁 导出：{out_path}")
print("==========================================\n")
