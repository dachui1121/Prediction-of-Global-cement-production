from cement_bau.floor_area_runtime import floor_input as _floor_input, floor_output as _floor_output, floor_anchors as _floor_anchors
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from pathlib import Path


import matplotlib.pyplot as plt


ANCHORS_IR = _floor_anchors('iran')


excel_path = str(_floor_input('iran', 'macro'))
sheet_name = "Sheet1"


G_LONG = 0.012
TAU    = 25.0


L_MAX = 60.0
L_LOWER_MARGIN = 2.0


LOSS = "soft_l1"
F_SCALE = 1.0
REG_W = 0.25

YEAR_MIN = 1989
YEAR_MAX = 2100

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

def standardize(x, mean=None, std=None):
    x = np.asarray(x, float)
    if mean is None:
        mean = np.nanmean(x)
    if std is None:
        std = np.nanstd(x)
    std = std if (np.isfinite(std) and std > 0) else 1.0
    return (x - mean) / std, mean, std


def logistic_cov(params, x1, x2, x3):
    L, a, b, c, d = params
    z = a + b * x1 + c * x2 + d * x3
    return L / (1.0 + np.exp(-z))


df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
df.columns = df.columns.astype(str).str.strip()
df = df.dropna(how="all").replace([np.inf, -np.inf], np.nan)

year_col = "年份"
gdp_col  = "GDP总量（十亿美元）"
pop_col  = "人口数量（人）"
urb_col  = "城镇化率"

for c in [year_col, gdp_col, pop_col, urb_col]:
    if c not in df.columns:
        raise KeyError(f"找不到列 '{c}'，当前列：{df.columns.tolist()}")

df[year_col] = to_num(df[year_col]).round().astype("Int64")
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
idx = {y: i for i, y in enumerate(years)}


mask_gdp = np.isfinite(gdp_in) & (gdp_in > 0) & np.isfinite(pop) & (pop > 0)
years_gdp = years[mask_gdp]
if len(years_gdp) < 5:
    raise ValueError("可用GDP点太少，请确认已填入历史GDP + 2025–2030 GDP锚点。")

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
    gdp_pc[i] = gdp_pc[i-1] * np.exp(gt)

gdp_ext = gdp_pc * pop
gdp_merged = gdp_in.copy()
gdp_merged[~np.isfinite(gdp_merged)] = gdp_ext[~np.isfinite(gdp_merged)]

gdp_pc_all = gdp_merged / np.maximum(pop, 1e-12)
ln_gdppc_all = np.log(np.maximum(gdp_pc_all, 1e-30))
ln_pop_all = np.log(np.maximum(pop, 1e-12))


anchor_years = np.array(sorted(ANCHORS_IR.keys()), dtype=int)
anchor_area  = np.array([ANCHORS_IR[y] for y in anchor_years], dtype=float)

x_lnGDPpc_a = np.array([ln_gdppc_all[idx[y]] if y in idx else np.nan for y in anchor_years], dtype=float)
U_a         = np.array([urb01[idx[y]]       if y in idx else np.nan for y in anchor_years], dtype=float)
x_lnPOP_a   = np.array([ln_pop_all[idx[y]]  if y in idx else np.nan for y in anchor_years], dtype=float)

m_fit = np.isfinite(x_lnGDPpc_a) & np.isfinite(U_a) & np.isfinite(x_lnPOP_a) & np.isfinite(anchor_area)
if m_fit.sum() < 4:
    raise ValueError("有效锚点不足，请检查锚点年份是否在表中都有 GDP/人口/城镇化率。")

x1_raw = x_lnGDPpc_a[m_fit]
x2_raw = U_a[m_fit]
x3_raw = x_lnPOP_a[m_fit]
y      = anchor_area[m_fit]


x1_all_std, m1, s1 = standardize(ln_gdppc_all)
x2_all_std, m2, s2 = standardize(urb01)
x3_all_std, m3, s3 = standardize(ln_pop_all)

x1 = (x1_raw - m1) / s1
x2 = (x2_raw - m2) / s2
x3 = (x3_raw - m3) / s3

ymax = float(np.nanmax(y))
L_min = ymax + L_LOWER_MARGIN
if L_min >= L_MAX:
    raise ValueError("锚点最大值过高导致 L_min>=L_MAX，请调整。")

L0 = float(np.clip(max(35.0, ymax + 10.0), L_min + 0.5, L_MAX - 0.5))
p0 = np.array([L0, -1.0, 1.0, 1.0, 0.0], float)

lb = np.array([L_min, -20.0, 0.0, 0.0, -10.0], float)
ub = np.array([L_MAX,  20.0, 20.0, 20.0,  10.0], float)

def residuals(params):
    pred = logistic_cov(params, x1, x2, x3)
    L, a, b, c, d = params
    reg = REG_W * np.array([a, b, c, d], float)
    return np.r_[pred - y, reg]

res = least_squares(
    residuals, p0, bounds=(lb, ub),
    loss=LOSS, f_scale=F_SCALE,
    max_nfev=200000
)
p_hat = res.x
A_pred = logistic_cov(p_hat, x1_all_std, x2_all_std, x3_all_std)

L_hat, a_hat, b_hat, c_hat, d_hat = p_hat


anchor_col = np.full(len(years), np.nan, dtype=float)
for yk, vk in ANCHORS_IR.items():
    if yk in idx:
        anchor_col[idx[yk]] = vk

out = pd.DataFrame({
    "年份": years,
    "人口(人)": pop,
    "城镇化率(0-1)": urb01,
    "GDP_原始输入(十亿美元)": gdp_in,
    "GDP_外推(十亿美元)": gdp_ext,
    "GDP_合并(十亿美元)": gdp_merged,
    "人均GDP(GDPpc, 十亿美元/人)": gdp_pc_all,
    "ln(人均GDP)": ln_gdppc_all,
    "ln(人口)": ln_pop_all,
    "锚点_人均住宅面积(㎡/人)": anchor_col,
    "预测_人均住宅面积_Logistic_BAU(㎡/人)": A_pred
})

out_path = (_floor_output('iran') / '伊朗_GDP外推_人均住宅面积_Logistic_BAU_1989-2100.xlsx')
out.to_excel(out_path, index=False, engine="openpyxl")
print("DONE ->", out_path)


anchor_pred = []
anchor_err  = []
for yy in anchor_years:
    if yy in idx:
        i = idx[yy]
        pred = float(A_pred[i])
        anchor_pred.append(pred)
        anchor_err.append(pred - ANCHORS_IR[yy])
    else:
        anchor_pred.append(np.nan)
        anchor_err.append(np.nan)

anchor_fit_df = pd.DataFrame({
    "Year": anchor_years,
    "Anchor": [ANCHORS_IR[y] for y in anchor_years],
    "Fitted": anchor_pred,
    "Error(Fitted-Anchor)": anchor_err
})


snap_years = [1989, 2000, 2005, 2020, 2024, 2030, 2050, 2100]
snap = out[out["年份"].isin(snap_years)].copy()
print("\n=== 关键年份快照 ===")
print(snap[["年份","GDP_合并(十亿美元)","人均GDP(GDPpc, 十亿美元/人)","城镇化率(0-1)","预测_人均住宅面积_Logistic_BAU(㎡/人)"]])

print("\n=== 模型参数 ===")
print(f"g0(2025-2030 lnGDPpc growth) = {g0:.6f}")
print(f"G_LONG={G_LONG}, TAU={TAU}")
print(f"L_hat={L_hat:.6f}, a_hat={a_hat:.6f}, b_hat={b_hat:.6f}, c_hat={c_hat:.6f}, d_hat={d_hat:.6f}")
print(f"loss={LOSS}, status={res.status}, cost={res.cost:.6f}, nfev={res.nfev}")


plt.rcParams["axes.unicode_minus"] = False


plt.figure(figsize=(10, 5))
plt.plot(years, gdp_merged, label="GDP 合并")
plt.plot(years, gdp_in, label="GDP 原始输入", linestyle="--")
plt.plot(years, gdp_ext, label="GDP 外推", linestyle=":")
plt.axvline(2030, linestyle="--", linewidth=1)
plt.title("Iran GDP (bn USD): Input vs Extrapolated vs Merged")
plt.xlabel("Year")
plt.ylabel("GDP (bn USD)")
plt.legend()
plt.grid(True, alpha=0.3)


plt.figure(figsize=(10, 5))
plt.plot(years, gdp_pc_all, label="GDP per capita")
plt.title("Iran GDP per capita (GDPpc)")
plt.xlabel("Year")
plt.ylabel("GDPpc (bn USD/person)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.figure(figsize=(10, 5))
plt.plot(years, ln_gdppc_all, label="ln(GDPpc)")
plt.title("ln(GDP per capita)")
plt.xlabel("Year")
plt.ylabel("ln(GDPpc)")
plt.grid(True, alpha=0.3)
plt.legend()


plt.figure(figsize=(10, 5))
plt.plot(years, pop, label="Population")
plt.title("Population")
plt.xlabel("Year")
plt.ylabel("People")
plt.grid(True, alpha=0.3)
plt.legend()

plt.figure(figsize=(10, 5))
plt.plot(years, urb01, label="Urbanization (0-1)")
plt.title("Urbanization Rate")
plt.xlabel("Year")
plt.ylabel("Urbanization (0-1)")
plt.grid(True, alpha=0.3)
plt.legend()


plt.figure(figsize=(10, 5))
plt.plot(years, A_pred, label="Predicted Housing Area (m2/person)")
plt.scatter(anchor_years, [ANCHORS_IR[y] for y in anchor_years], label="Anchors", zorder=3)
plt.title("Per-capita Residential Floor Area (BAU Logistic)")
plt.xlabel("Year")
plt.ylabel("m2/person")
plt.grid(True, alpha=0.3)
plt.legend()


plt.figure(figsize=(8, 4))
plt.bar(anchor_fit_df["Year"].astype(str), anchor_fit_df["Error(Fitted-Anchor)"])
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Anchor Fit Error (Fitted - Anchor)")
plt.xlabel("Anchor Year")
plt.ylabel("m2/person")
plt.grid(True, axis="y", alpha=0.3)


plt.show()
