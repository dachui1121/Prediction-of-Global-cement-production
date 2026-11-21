from cement_bau.floor_area_runtime import floor_input as _floor_input, floor_output as _floor_output, floor_anchors as _floor_anchors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from pathlib import Path


excel_path = str(_floor_input('turkiye', 'macro'))
sheet_name = "Sheet1"


ANCHORS_TR = _floor_anchors('turkiye')

YEAR_MIN = 1989
YEAR_MAX = 2100


G_LONG = 0.012
TAU    = 25.0


L_MAX = 60.0
L_LOWER_MARGIN = 2.0

USE_POP_TERM = False
POP_TERM_KIND = "ln_urban_pop"


LOSS = "soft_l1"
F_SCALE = 1.0
REG_W = 0.01

MAX_NFEV = 200000


plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def to_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

def safe_write_excel(df_out: pd.DataFrame, out_name: str):

    desktop = _floor_output('turkiye')
    out_dir = desktop if desktop.exists() else Path.cwd()
    out_path = out_dir / out_name
    df_out.to_excel(out_path, index=False, engine="openpyxl")
    return out_path

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


year_col = "年份"
gdp_col  = "GDP总量（十亿美元）"
pop_col  = "人口数量（人）"
urb_col  = "城镇化率（%）"


for c in [year_col, gdp_col, pop_col, urb_col]:
    if c not in df.columns:
        raise KeyError(f"找不到列 '{c}'，当前列：{df.columns.tolist()}")

df[year_col] = to_num(df[year_col]).round()
df[gdp_col]  = to_num(df[gdp_col])
df[pop_col]  = to_num(df[pop_col])
df[urb_col]  = to_num(df[urb_col])

df = df.dropna(subset=[year_col, pop_col, urb_col]).copy()
df[year_col] = df[year_col].astype(int)
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


urb01 = np.clip(urb01, 0.01, 0.99)

idx = {y: i for i, y in enumerate(years)}

print(f"✅ 年份范围：{years.min()}–{years.max()}（共 {len(years)} 年）")
print(f"✅ 列：年份='{year_col}', 人口='{pop_col}', 城镇化率='{urb_col}', GDP='{gdp_col}'")


mask_gdp = np.isfinite(gdp_in) & (gdp_in > 0) & np.isfinite(pop) & (pop > 0)
years_gdp = years[mask_gdp]
if len(years_gdp) < 5:
    raise ValueError("可用GDP点太少。请确认 Excel 中已填入 GDP(1989–2024) + GDP预测(2025–2030) 的若干点。")

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


if USE_POP_TERM:
    if POP_TERM_KIND == "ln_pop":
        pop_term_all = np.log(np.maximum(pop, 1e-12))
    elif POP_TERM_KIND == "ln_urban_pop":
        pop_term_all = np.log(np.maximum(pop * urb01, 1e-12))
    else:
        raise ValueError("POP_TERM_KIND 只能是 'ln_pop' 或 'ln_urban_pop'")
else:
    pop_term_all = None

print("\n--- GDP 外推信息 ---")
print(f"锚点(2025–2030)人均GDP对数增速 g0 = {g0*100:.3f}%/年")
print(f"长期收敛增速 G_LONG = {G_LONG*100:.3f}%/年，TAU = {TAU} 年")
print(f"最后一个原始GDP年份 = {last_gdp_year}（之后为外推）")


anchor_years = np.array(sorted(ANCHORS_TR.keys()), dtype=int)
anchor_area  = np.array([ANCHORS_TR[y] for y in anchor_years], dtype=float)


ln_gdppc_a = np.array([ln_gdppc_all[idx[y]] if y in idx else np.nan for y in anchor_years], dtype=float)
U_a        = np.array([urb01[idx[y]]        if y in idx else np.nan for y in anchor_years], dtype=float)

if USE_POP_TERM:
    P_a = np.array([pop_term_all[idx[y]] if y in idx else np.nan for y in anchor_years], dtype=float)
else:
    P_a = None

m_fit = np.isfinite(ln_gdppc_a) & np.isfinite(U_a) & np.isfinite(anchor_area)
if USE_POP_TERM:
    m_fit = m_fit & np.isfinite(P_a)

if m_fit.sum() < 4:
    raise ValueError(f"有效锚点不足：{m_fit.sum()} 个。请检查锚点年份在表中是否都有 GDP/人口/城镇化率。")

x1 = ln_gdppc_a[m_fit]
x2 = U_a[m_fit]
y  = anchor_area[m_fit]
if USE_POP_TERM:
    x3 = P_a[m_fit]

def residuals(params):
    if USE_POP_TERM:
        pred = logistic_cov(params, x1, x2, x3, use_pop=True)
        reg  = REG_W * np.array([params[1], params[3], params[4], params[5]])
    else:
        pred = logistic_cov(params, x1, x2, None, use_pop=False)
        reg  = REG_W * np.array([params[1], params[3], params[4]])
    return np.r_[pred - y, reg]


ymax = float(np.nanmax(y))
L_min = ymax + L_LOWER_MARGIN
if L_min >= L_MAX:
    raise ValueError(
        f"锚点最大值={ymax:.2f}，L_min=锚点最大值+{L_LOWER_MARGIN}={L_min:.2f} 已 >= L_MAX={L_MAX}。\n"
        f"请降低 L_LOWER_MARGIN 或提高 L_MAX（但你要求 L_MAX=60）。"
    )

L0 = float(np.clip(max(35.0, ymax + 10.0), L_min + 0.5, L_MAX - 0.5))
k0 = 1.0
a0 = -1.0
b0 = 0.5
c0 = 2.0

if USE_POP_TERM:
    d0 = 0.0
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
    max_nfev=MAX_NFEV
)
p_hat = res.x

if USE_POP_TERM:
    A_pred = logistic_cov(p_hat, ln_gdppc_all, urb01, pop_term_all, use_pop=True)
    L_hat, k_hat, a_hat, b_hat, c_hat, d_hat = p_hat
else:
    A_pred = logistic_cov(p_hat, ln_gdppc_all, urb01, None, use_pop=False)
    L_hat, k_hat, a_hat, b_hat, c_hat = p_hat


anchor_col = np.full(len(years), np.nan, dtype=float)
for yk, vk in ANCHORS_TR.items():
    if yk in idx:
        anchor_col[idx[yk]] = vk

out = pd.DataFrame({
    "年份": years,
    "人口": pop,
    "城镇化率(0-1)": urb01,
    "GDP_原始输入(十亿美元)": gdp_in,
    "GDP_外推(十亿美元)": gdp_ext,
    "GDP_合并(十亿美元)": gdp_merged,
    "人均GDP(GDPpc)": gdp_pc_all,
    "ln(人均GDP)": ln_gdppc_all,
    "锚点_城镇人均住宅面积(㎡/人)": anchor_col,
    "预测_城镇人均住宅面积_BAU_Logistic(㎡/人)": A_pred
})

out_path = safe_write_excel(out, "土耳其_BAU_GDP外推_城镇人均住宅面积_Logistic_Lmax60_无强约束_1989-2100.xlsx")

print("\n================== DONE ==================")
print(f"GDP外推：以 2025–2030 的人均GDP增速为锚点；长期收敛到 {G_LONG*100:.2f}%/y，TAU={TAU}")
print(f"Logistic（不固定L，但 L<=60，无强约束）参数：")
print(f"  L_hat={float(L_hat):.6f} (<= {L_MAX})")
print(f"  k_hat={float(k_hat):.6f}")
print(f"  a_hat={float(a_hat):.6f}")
print(f"  b_hat={float(b_hat):.6f}  (lnGDPpc)")
print(f"  c_hat={float(c_hat):.6f}  (Urbanization)")
if USE_POP_TERM:
    print(f"  d_hat={float(d_hat):.6f}  ({POP_TERM_KIND})")
print(f"拟合：loss={LOSS}, status={res.status}, cost={res.cost:.6f}, nfev={res.nfev}")
print(f"📁 已导出：{out_path}")
print("==========================================\n")


plt.figure(figsize=(10, 6))
plt.plot(years, A_pred, linewidth=2.2, label="预测_城镇人均住宅面积(BAU Logistic)")
plt.scatter(anchor_years, anchor_area, color="red", s=60, zorder=5, label="锚点(观测)")
plt.xlabel("年份")
plt.ylabel("城镇人均住宅面积 (㎡/人)")
plt.title("土耳其：BAU模式 城镇人均住宅面积 Logistic 拟合与预测（1989–2100）")
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
m_in = np.isfinite(gdp_in) & (gdp_in > 0)
plt.scatter(years[m_in], gdp_in[m_in], s=18, alpha=0.7, label="GDP_原始输入")
plt.plot(years, gdp_ext, linewidth=2.0, label="GDP_外推")
plt.plot(years, gdp_merged, linewidth=2.2, label="GDP_合并")
plt.xlabel("年份")
plt.ylabel("GDP (十亿美元)")
plt.title("土耳其：GDP 外推与合并（1989–2100）")
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend()
plt.tight_layout()
plt.show()
