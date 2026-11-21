from cement_bau.floor_area_runtime import floor_input as _floor_input, floor_output as _floor_output, floor_anchors as _floor_anchors
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from pathlib import Path


EXCEL_PATH =str(_floor_input('china', 'macro'))

SHEET_COV  = "Sheet2"
SHEET_AREA = "Sheet1"


YEAR_MIN, YEAR_MAX = 1989, 2100

G_LONG = 0.012
TAU    = 25.0


L_MAX = 60.0
L_LOWER_MARGIN = 2.0

USE_POP_TERM = True
POP_TERM_KIND = "ln_urban_pop"


LOSS = "soft_l1"
F_SCALE = 1.0
REG_W = 0.01


OUT_EXCEL = (_floor_output('china') / '中国_GDP外推_城镇人均住宅面积_Logistic_BAU_1989-2100.xlsx')
OUT_FIG_GDP  = (_floor_output('china') / 'China_GDP_1989-2100.png')
OUT_FIG_AREA = (_floor_output('china') / 'China_Urban_Area_pc_1989-2100.png')


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

def logistic_cov(params, ln_gdppc, U, pop_term=None, use_pop=True):

    if use_pop:
        L, k, a, b, c, d = params
        z = a + b * ln_gdppc + c * U + d * pop_term
    else:
        L, k, a, b, c = params
        z = a + b * ln_gdppc + c * U
    return L / (1.0 + np.exp(-k * z))


df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_COV, engine="openpyxl")
df.columns = df.columns.astype(str).str.strip()
df = df.dropna(how="all").replace([np.inf, -np.inf], np.nan)

year_col = pick_col(df, ["年份", "Year"])
pop_col  = pick_col(df, ["中国人口数量", "人口"])
urb_col  = pick_col(df, ["中国城镇化率", "城镇化率", "城市化率"])
gdp_col  = pick_col(df, ["中国GDP总量（十亿美元）", "GDP总量（十亿美元）", "GDP总量", "GDP"])
urban_pop_col = pick_col(df, ["城镇人口数量"], required=False)

df[year_col] = to_num(df[year_col])
df[pop_col]  = to_num(df[pop_col])
df[urb_col]  = to_num(df[urb_col])
df[gdp_col]  = to_num(df[gdp_col])
if urban_pop_col is not None:
    df[urban_pop_col] = to_num(df[urban_pop_col])

df = df.dropna(subset=[year_col, pop_col, urb_col]).copy()
df[year_col] = df[year_col].round().astype(int)
df = df.sort_values(year_col).drop_duplicates(subset=[year_col]).reset_index(drop=True)

years_all   = df[year_col].to_numpy(dtype=int)
pop_all     = df[pop_col].to_numpy(dtype=float)
urb_all     = df[urb_col].to_numpy(dtype=float)
gdp_in_all  = df[gdp_col].to_numpy(dtype=float)
urban_pop_all = df[urban_pop_col].to_numpy(dtype=float) if urban_pop_col is not None else None

urb01_all = urb_all / 100.0 if np.nanmax(urb_all) > 1.5 else urb_all.copy()


m = (years_all >= YEAR_MIN) & (years_all <= YEAR_MAX)
years = years_all[m]
pop   = pop_all[m]
urb01 = urb01_all[m]
gdp_in = gdp_in_all[m]
urban_pop = urban_pop_all[m] if urban_pop_all is not None else None

idx = {y: i for i, y in enumerate(years)}

print(f"✅ 识别列：年份='{year_col}', 人口='{pop_col}', 城镇化率='{urb_col}', GDP='{gdp_col}'"
      + (f", 城镇人口='{urban_pop_col}'" if urban_pop_col else ""))
print(f"✅ 年份范围：{years.min()}–{years.max()}（共 {len(years)} 年）")


mask_gdp = np.isfinite(gdp_in) & (gdp_in > 0) & np.isfinite(pop) & (pop > 0)
years_gdp = years[mask_gdp]
if len(years_gdp) < 5:
    raise ValueError("可用GDP点太少。请确认 Sheet2 中已填入历史GDP + 2025–2030 GDP锚点。")

gdp_pc = np.full_like(pop, np.nan, dtype=float)
gdp_pc[mask_gdp] = gdp_in[mask_gdp] / pop[mask_gdp]

last_gdp_year = int(years_gdp.max())
i_last = idx[last_gdp_year]

def growth_from_2025_2030():
    if (2025 in idx) and (2030 in idx):
        i25, i30 = idx[2025], idx[2030]
        if np.isfinite(gdp_pc[i25]) and np.isfinite(gdp_pc[i30]) and gdp_pc[i25] > 0 and gdp_pc[i30] > 0:
            return float((np.log(gdp_pc[i30]) - np.log(gdp_pc[i25])) / (2030 - 2025))
    return None

g0 = growth_from_2025_2030()
if g0 is None:
    g0 = estimate_log_growth(years_gdp, gdp_pc[mask_gdp], win=8)


for i in range(i_last + 1, len(years)):
    t = years[i]
    gt = G_LONG + (g0 - G_LONG) * np.exp(-(t - last_gdp_year) / TAU)
    gdp_pc[i] = gdp_pc[i-1] * np.exp(gt)

gdp_ext = gdp_pc * pop
gdp_merged = gdp_in.copy()
gdp_merged[~np.isfinite(gdp_merged)] = gdp_ext[~np.isfinite(gdp_merged)]

gdp_pc_all = gdp_merged / np.maximum(pop, 1e-12)
ln_gdppc_all = np.log(np.maximum(gdp_pc_all, 1e-12))


if USE_POP_TERM:
    if POP_TERM_KIND == "ln_pop":
        pop_term_all = np.log(np.maximum(pop, 1e-12))
    elif POP_TERM_KIND == "ln_urban_pop":
        if urban_pop is not None and np.isfinite(urban_pop).sum() > 0:
            pop_term_all = np.log(np.maximum(urban_pop, 1e-12))
        else:
            pop_term_all = np.log(np.maximum(pop * urb01, 1e-12))
    else:
        raise ValueError("POP_TERM_KIND 只能是 'ln_pop' 或 'ln_urban_pop'")
else:
    pop_term_all = None

print(f"✅ GDP 外推：last_gdp_year={last_gdp_year}；锚点增速 g0={g0*100:.3f}%/y；长期收敛={G_LONG*100:.3f}%/y；TAU={TAU}")


dfA = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_AREA, engine="openpyxl")
dfA.columns = dfA.columns.astype(str).str.strip()
ycolA = pick_col(dfA, ["年份", "Year"])
acolA = pick_col(dfA, ["城镇人均住宅面积"])

dfA[ycolA] = to_num(dfA[ycolA]).round().astype("Int64")
dfA[acolA] = to_num(dfA[acolA])
dfA = dfA.dropna(subset=[ycolA, acolA]).copy()
dfA[ycolA] = dfA[ycolA].astype(int)
dfA = dfA.sort_values(ycolA).drop_duplicates(subset=[ycolA]).reset_index(drop=True)

anchor_years = dfA[ycolA].to_numpy(dtype=int)
anchor_area  = dfA[acolA].to_numpy(dtype=float)


ln_a = np.array([ln_gdppc_all[idx[y]] if y in idx else np.nan for y in anchor_years], dtype=float)
U_a  = np.array([urb01[idx[y]]        if y in idx else np.nan for y in anchor_years], dtype=float)
if USE_POP_TERM:
    P_a = np.array([pop_term_all[idx[y]] if y in idx else np.nan for y in anchor_years], dtype=float)
else:
    P_a = None

m_fit = np.isfinite(ln_a) & np.isfinite(U_a) & np.isfinite(anchor_area)
if USE_POP_TERM:
    m_fit = m_fit & np.isfinite(P_a)

if m_fit.sum() < 6:
    raise ValueError(f"有效锚点不足：{m_fit.sum()} 个。请检查 Sheet1 的住宅面积年份是否与 Sheet2 覆盖。")

x1 = ln_a[m_fit]
x2 = U_a[m_fit]
y  = anchor_area[m_fit]
x3 = P_a[m_fit] if USE_POP_TERM else None


def residuals(params):
    pred = logistic_cov(params, x1, x2, x3, use_pop=USE_POP_TERM)

    if USE_POP_TERM:
        reg = REG_W * np.array([params[1], params[3], params[4], params[5]])
    else:
        reg = REG_W * np.array([params[1], params[3], params[4]])
    return np.r_[pred - y, reg]

ymax = float(np.nanmax(y))
L_min = ymax + L_LOWER_MARGIN
if L_min >= L_MAX:
    raise ValueError(f"锚点最大值={ymax:.2f}，导致 L_min={L_min:.2f} >= L_MAX={L_MAX}。请调参。")


L0 = float(np.clip(max(45.0, ymax + 10.0), L_min + 0.5, L_MAX - 0.5))
k0, a0, b0, c0, d0 = 1.0, -1.0, 0.5, 2.0, 0.0

if USE_POP_TERM:
    p0 = np.array([L0, k0, a0, b0, c0, d0], float)

    lb = np.array([L_min, 1e-3, -50.0, 0.0, 0.0, -10.0], float)
    ub = np.array([L_MAX, 50.0,  50.0, 50.0, 50.0,  10.0], float)
else:
    p0 = np.array([L0, k0, a0, b0, c0], float)
    lb = np.array([L_min, 1e-3, -50.0, 0.0, 0.0], float)
    ub = np.array([L_MAX, 50.0,  50.0, 50.0, 50.0], float)

res = least_squares(
    residuals, p0, bounds=(lb, ub),
    loss=LOSS, f_scale=F_SCALE,
    max_nfev=200000
)
p_hat = res.x


A_pred = logistic_cov(p_hat, ln_gdppc_all, urb01, pop_term_all, use_pop=USE_POP_TERM)

if USE_POP_TERM:
    L_hat, k_hat, a_hat, b_hat, c_hat, d_hat = p_hat
else:
    L_hat, k_hat, a_hat, b_hat, c_hat = p_hat
    d_hat = np.nan

print("\n================== FIT DONE ==================")
print("Logistic（不固定L，但 L<=60，无强约束）参数：")
print(f"  L_hat={float(L_hat):.6f} (<= {L_MAX})")
print(f"  k_hat={float(k_hat):.6f}")
print(f"  a_hat={float(a_hat):.6f}")
print(f"  b_hat={float(b_hat):.6f}  (lnGDPpc)")
print(f"  c_hat={float(c_hat):.6f}  (Urbanization)")
if USE_POP_TERM:
    print(f"  d_hat={float(d_hat):.6f}  ({POP_TERM_KIND})")
print(f"拟合：loss={LOSS}, status={res.status}, cost={res.cost:.6f}, nfev={res.nfev}")
print("==============================================\n")


anchor_col = np.full(len(years), np.nan, dtype=float)
for yy, vv in zip(anchor_years, anchor_area):
    if yy in idx:
        anchor_col[idx[yy]] = vv

out = pd.DataFrame({
    "年份": years,
    "人口": pop,
    "城镇化率(0-1)": urb01,
    "城镇人口数量": urban_pop if urban_pop is not None else pop * urb01,
    "GDP_原始输入(十亿美元)": gdp_in,
    "GDP_外推(十亿美元)": gdp_ext,
    "GDP_合并(十亿美元)": gdp_merged,
    "人均GDP(GDPpc)": gdp_pc_all,
    "ln(人均GDP)": ln_gdppc_all,
    "锚点_城镇人均住宅面积(㎡/人)": anchor_col,
    "预测_城镇人均住宅面积_Logistic_BAU(㎡/人)": A_pred
})

out.to_excel(OUT_EXCEL, index=False, engine="openpyxl")
print(f"✅ 已导出结果Excel：{OUT_EXCEL}")


plt.figure()
plt.plot(out["年份"], out["GDP_合并(十亿美元)"])
plt.xlabel("Year")
plt.ylabel("GDP (Billion USD)")
plt.title("China GDP (Merged) 1989-2100")
plt.tight_layout()
plt.savefig(OUT_FIG_GDP, dpi=300)
plt.show()
print(f"✅ 已保存图片：{OUT_FIG_GDP}")


plt.figure()
plt.plot(out["年份"], out["预测_城镇人均住宅面积_Logistic_BAU(㎡/人)"], label="Predicted")
plt.scatter(out["年份"], out["锚点_城镇人均住宅面积(㎡/人)"], s=14, label="Anchors")
plt.xlabel("Year")
plt.ylabel("Area (m²/person)")
plt.title("Urban Residential Floor Area per Capita (BAU)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_FIG_AREA, dpi=300)
plt.show()
print(f"✅ 已保存图片：{OUT_FIG_AREA}")

print("\n================== ALL DONE ==================")
print("Excel:", OUT_EXCEL)
print("Fig1 :", OUT_FIG_GDP)
print("Fig2 :", OUT_FIG_AREA)
print("==============================================")
