from cement_bau.floor_area_runtime import floor_input as _floor_input, floor_output as _floor_output, floor_anchors as _floor_anchors
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from pathlib import Path


ANCHORS_JP = _floor_anchors('japan')


excel_path = str(_floor_input('japan', 'macro'))
sheet_name = "Sheet1"


G_LONG = 0.006
TAU    = 25.0


L_MAX = 60.0
L_LOWER_MARGIN = 2.0


LOSS = "soft_l1"
F_SCALE = 1.0
REG_W = 0.02


PLATEAU_START = 2050
W_SLOPE = 0.8
W_NEG   = 2.0


YEAR_MIN = 1989
YEAR_MAX = 2100


def pick_col(df_, keywords, required=True):

    cols = df_.columns.astype(str).tolist()
    for kw in keywords:
        for c in cols:
            if c == kw:
                return c
    for kw in keywords:
        for c in cols:
            if kw in c:
                return c
    if required:
        raise KeyError(f"找不到列（关键词 {keywords}），当前列名：{cols}")
    return None

def to_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

def logistic_cov(params, ln_gdppc, U, pop_term):

    L, k, a, b, c, d = params
    z = a + b * ln_gdppc + c * U + d * pop_term
    return L / (1.0 + np.exp(-k * z))


df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
df.columns = df.columns.astype(str).str.strip()
df = df.dropna(how="all").replace([np.inf, -np.inf], np.nan)

year_col = pick_col(df, ["年份", "Year"])
pop_col  = pick_col(df, ["人口数量", "总人口", "人口"])
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

print(f"✅ 识别列：年份='{year_col}', 总人口='{pop_col}', 城镇化率='{urb_col}', GDP='{gdp_col}'")
print(f"✅ 年份范围：{years.min()}–{years.max()}（共 {len(years)} 年）")


mask_gdp = np.isfinite(gdp_in) & (gdp_in > 0) & np.isfinite(pop) & (pop > 0)
if mask_gdp.sum() < 5:
    raise ValueError("可用GDP点太少。请确认 Excel 中已填入历史GDP + 2025–2030 GDP锚点。")

gdp_pc = np.full_like(pop, np.nan, dtype=float)
gdp_pc[mask_gdp] = gdp_in[mask_gdp] / pop[mask_gdp]
last_gdp_year = int(years[mask_gdp].max())


if (2025 in idx) and (2030 in idx):
    g0 = float((np.log(gdp_pc[idx[2030]]) - np.log(gdp_pc[idx[2025]])) / (2030 - 2025))
else:
    raise ValueError("缺少 2025 或 2030，用于锚定 GDP 增速。")


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


pop_term_all = np.log(np.maximum(pop / 1e6, 1e-12))

print(f"✅ GDP外推锚点增速 g0 ≈ {g0*100:.3f}%/y；长期收敛 {G_LONG*100:.2f}%/y, TAU={TAU}")


anchor_years = np.array(sorted(ANCHORS_JP.keys()), dtype=int)
y_anchor = np.array([ANCHORS_JP[y] for y in anchor_years], dtype=float)


x1_a = np.array([ln_gdppc_all[idx[y]] if y in idx else np.nan for y in anchor_years], dtype=float)
x2_a = np.array([urb01[idx[y]]        if y in idx else np.nan for y in anchor_years], dtype=float)
x3_a = np.array([pop_term_all[idx[y]] if y in idx else np.nan for y in anchor_years], dtype=float)

m_fit = np.isfinite(x1_a) & np.isfinite(x2_a) & np.isfinite(x3_a) & np.isfinite(y_anchor)
if m_fit.sum() < 5:
    raise ValueError(f"有效锚点不足：{m_fit.sum()} 个。请检查锚点年份是否都在表中且有GDP/人口/城镇化率。")

x1_a, x2_a, x3_a, y_anchor = x1_a[m_fit], x2_a[m_fit], x3_a[m_fit], y_anchor[m_fit]


ymax = float(np.nanmax(y_anchor))
L_min = ymax + L_LOWER_MARGIN
if L_min >= L_MAX:
    raise ValueError(f"锚点最大值={ymax:.2f}，L_min={L_min:.2f} >= L_MAX={L_MAX}。")


if PLATEAU_START not in idx:
    raise ValueError(f"表中缺少 {PLATEAU_START} 年，无法施加平台软约束。请确保年份到 2100 且包含 2050。")
i_pl0 = idx[PLATEAU_START]

def residuals(p):

    pred_anchor = logistic_cov(p, x1_a, x2_a, x3_a)
    r_anchor = pred_anchor - y_anchor


    reg = REG_W * np.array([p[1], p[3], p[4], p[5]])


    A_full = logistic_cov(p, ln_gdppc_all, urb01, pop_term_all)
    dA = np.diff(A_full[i_pl0:])

    r_slope = np.sqrt(W_SLOPE) * dA
    r_neg   = np.sqrt(W_NEG) * np.minimum(0.0, dA)

    return np.r_[r_anchor, reg, r_slope, r_neg]


lb = np.array([L_min, 1e-3, -50.0, 0.0, 0.0, -20.0], float)
ub = np.array([L_MAX, 50.0,  50.0, 50.0, 50.0,  0.0], float)


L0 = float(np.clip(max(45.0, ymax + 10.0), L_min + 0.5, L_MAX - 0.5))
p0_candidates = [
    np.array([L0, 2.0, -5.0, 0.2, 3.0, -0.2], float),
    np.array([L0, 1.2, -2.0, 0.1, 2.0, -0.1], float),
    np.array([L0, 3.0, -8.0, 0.4, 3.5, -0.4], float),
]

best = None
for p0 in p0_candidates:
    r = least_squares(
        residuals, p0, bounds=(lb, ub),
        loss=LOSS, f_scale=F_SCALE,
        max_nfev=200000
    )
    if (best is None) or (r.cost < best.cost):
        best = r

p_hat = best.x
A_pred = logistic_cov(p_hat, ln_gdppc_all, urb01, pop_term_all)

L_hat, k_hat, a_hat, b_hat, c_hat, d_hat = p_hat

print("\n================== FIT DONE ==================")
print("Logistic（L<=60，无强约束，平台软约束 2050+）参数：")
print(f"  L_hat={L_hat:.6f} (<= {L_MAX})")
print(f"  k_hat={k_hat:.6f}")
print(f"  a_hat={a_hat:.6f}")
print(f"  b_hat={b_hat:.6e}  (lnGDPpc, constrained >=0)")
print(f"  c_hat={c_hat:.6f}  (Urbanization)")
print(f"  d_hat={d_hat:.6e}  (ln(pop/million), constrained <=0)")
print(f"拟合：loss={LOSS}, status={best.status}, cost={best.cost:.6f}, nfev={best.nfev}")
print("==============================================\n")


anchor_col = np.full(len(years), np.nan, dtype=float)
for yy, vv in ANCHORS_JP.items():
    if yy in idx:
        anchor_col[idx[yy]] = vv

out = pd.DataFrame({
    "年份": years,
    "总人口": pop,
    "城镇化率(0-1)": urb01,
    "GDP_原始输入(十亿美元)": gdp_in,
    "GDP_外推(十亿美元)": gdp_ext,
    "GDP_合并(十亿美元)": gdp_merged,
    "人均GDP": gdp_pc_all,
    "ln(人均GDP)": ln_gdppc_all,
    "ln(人口/百万)": pop_term_all,
    "锚点_人均住宅面积(㎡/人)": anchor_col,
    "预测_人均住宅面积_Logistic_BAU_平台版_Lmax60(㎡/人)": A_pred
})


out_path = (_floor_output('japan') / '日本_人均住宅面积_BAU_上升后平台_Lmax60_2050后趋稳_1989-2100.xlsx')
out.to_excel(out_path, index=False, engine="openpyxl")


key_years = [1993, 2008, 2018, 2023, 2030, 2050, 2075, 2100]
snap = out[out["年份"].isin(key_years)][
    ["年份", "锚点_人均住宅面积(㎡/人)", "预测_人均住宅面积_Logistic_BAU_平台版_Lmax60(㎡/人)"]
].copy()


plateau = out[out["年份"] >= PLATEAU_START].copy()
plateau["ΔA"] = plateau["预测_人均住宅面积_Logistic_BAU_平台版_Lmax60(㎡/人)"].diff()
dA_stats = plateau["ΔA"].dropna().describe()

print(f"📁 已导出：{out_path}")
print("\n关键年份：")
print(snap.to_string(index=False))
print("\n2050–2100 年增量统计（越接近0越趋稳）：")
print(dA_stats.to_string())
print("\n================== DONE ==================\n")
