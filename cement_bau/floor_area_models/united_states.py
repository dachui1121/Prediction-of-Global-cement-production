from cement_bau.floor_area_runtime import floor_input as _floor_input, floor_output as _floor_output, floor_anchors as _floor_anchors
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path
import matplotlib.pyplot as plt


macro_excel_path  = str(_floor_input('united_states', 'macro'))
anchor_excel_path = str(_floor_input('united_states', 'anchors'))

macro_sheet_name  = None
anchor_sheet_name = None

OUT_PATH = (_floor_output('united_states') / '美国_1990-2100_城镇人均住宅面积_Logistic_L120.xlsx')


G_LONG = 0.012
TAU    = 25.0


L_FIXED = 120.0


plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


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

def auto_pick_sheet(excel_path, want_area=False):
    xls = pd.ExcelFile(excel_path, engine="openpyxl")
    best = None
    best_score = -1
    for sh in xls.sheet_names:
        df0 = pd.read_excel(excel_path, sheet_name=sh, engine="openpyxl")
        df0.columns = df0.columns.astype(str).str.strip()

        has_year = any(("Year" in c) or ("YEAR" in c) or ("年份" in c) for c in df0.columns.astype(str))
        if want_area:
            has_area = any(
                (("人均" in c) and (("面积" in c) or ("住宅" in c) or ("居住" in c))) or
                (("AREA" in c.upper()) and ("PER" in c.upper())) or
                (("URBAN" in c.upper()) and ("AREA" in c.upper()))
                for c in df0.columns.astype(str)
            )
            score = int(has_year) + int(has_area)
        else:
            has_pop = any(("人口" in c) or ("POP" in c.upper()) or ("Population" in c) for c in df0.columns.astype(str))
            has_urb = any(("城镇化" in c) or ("城市化" in c) or ("URBAN" in c.upper()) for c in df0.columns.astype(str))
            has_gdp = any(("GDP" in c.upper()) or ("国内生产总值" in c) or ("GDP总量" in c) for c in df0.columns.astype(str))
            score = int(has_year) + int(has_pop) + int(has_urb) + int(has_gdp)

        if score > best_score:
            best_score = score
            best = sh
    return best

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-12)
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_pred[m] - y_true[m]) / y_true[m])) * 100.0)


if macro_sheet_name is None:
    macro_sheet_name = auto_pick_sheet(macro_excel_path, want_area=False)

df = pd.read_excel(macro_excel_path, sheet_name=macro_sheet_name, engine="openpyxl")
df.columns = df.columns.astype(str).str.strip()
df = df.dropna(how="all").replace([np.inf, -np.inf], np.nan)

year_col = pick_col(df, ["年份", "Year", "YEAR"])
pop_col  = pick_col(df, ["人口", "Population", "POP"])
urb_col  = pick_col(df, ["城镇化率", "城市化率", "Urbanization", "Urban rate", "URB"])

gdp_candidates = [c for c in df.columns if ("GDP" in c.upper()) or ("国内生产总值" in c) or ("GDP总量" in c)]
if len(gdp_candidates) == 0:
    raise KeyError(f"未找到GDP列。当前列名：{df.columns.tolist()}")
gdp_col = max(gdp_candidates, key=lambda c: pd.to_numeric(df[c], errors="coerce").notna().sum())

df[year_col] = to_num(df[year_col])
df[pop_col]  = to_num(df[pop_col])
df[urb_col]  = to_num(df[urb_col])
df[gdp_col]  = to_num(df[gdp_col])

df = df.dropna(subset=[year_col, pop_col, urb_col]).copy()
df[year_col] = df[year_col].round().astype(int)
df = df.sort_values(year_col).drop_duplicates(subset=[year_col]).reset_index(drop=True)


ymin, ymax = int(df[year_col].min()), int(df[year_col].max())
if ymin > 1990 or ymax < 2100:
    years_full = np.arange(min(1990, ymin), max(2100, ymax) + 1)
    df_full = pd.DataFrame({year_col: years_full})
    df_full = df_full.merge(df[[year_col, pop_col, urb_col, gdp_col]], on=year_col, how="left")
    df_full[pop_col] = to_num(df_full[pop_col]).interpolate(limit_direction="both")
    df_full[urb_col] = to_num(df_full[urb_col]).interpolate(limit_direction="both")
    df_full[gdp_col] = to_num(df_full[gdp_col])
    df = df_full

years  = df[year_col].to_numpy(int)
pop    = df[pop_col].to_numpy(float)
urb    = df[urb_col].to_numpy(float)
gdp_in = df[gdp_col].to_numpy(float)

urb01 = urb / 100.0 if np.nanmax(urb) > 1.5 else urb.copy()
idx = {y: i for i, y in enumerate(years)}

print(f"✅ 宏观sheet: {macro_sheet_name}")
print(f"✅ 列识别：year={year_col} | pop={pop_col} | urb={urb_col} | gdp={gdp_col}")


if anchor_sheet_name is None:
    anchor_sheet_name = auto_pick_sheet(anchor_excel_path, want_area=True)

dfA = pd.read_excel(anchor_excel_path, sheet_name=anchor_sheet_name, engine="openpyxl")
dfA.columns = dfA.columns.astype(str).str.strip()

yA_col = pick_col(dfA, ["年份", "Year", "YEAR"])

area_candidates = []
for c in dfA.columns.astype(str):
    cu = c.upper()
    if (("人均" in c) and (("面积" in c) or ("住宅" in c) or ("居住" in c))) or \
       (("URBAN" in cu) and ("AREA" in cu)) or \
       (("PER" in cu) and ("AREA" in cu)):
        area_candidates.append(c)
if len(area_candidates) == 0:
    area_candidates = [c for c in dfA.columns.astype(str) if ("面积" in c) or ("AREA" in c.upper())]
if len(area_candidates) == 0:
    raise KeyError("第二个Excel里没找到‘城镇人均住宅面积(㎡/人)’列，请检查列名。")

area_col = max(area_candidates, key=lambda c: pd.to_numeric(dfA[c], errors="coerce").notna().sum())

dfA[yA_col] = to_num(dfA[yA_col])
dfA[area_col] = to_num(dfA[area_col])
dfA = dfA.dropna(subset=[yA_col, area_col]).copy()
dfA[yA_col] = dfA[yA_col].round().astype(int)
dfA = dfA.sort_values(yA_col)

dfA = dfA[(dfA[yA_col] >= 1990) & (dfA[yA_col] <= 2015)].copy()

ANCHORS = {int(y): float(a) for y, a in zip(dfA[yA_col].to_numpy(int), dfA[area_col].to_numpy(float))
           if np.isfinite(a) and a > 0}

if len(ANCHORS) < 4:
    raise ValueError(f"锚点太少（{len(ANCHORS)}个）。建议至少4个年份的面积锚点。")

print(f"✅ 锚点sheet: {anchor_sheet_name} | 面积列: {area_col} | 锚点数量: {len(ANCHORS)}")


mask_gdp = np.isfinite(gdp_in) & (gdp_in > 0) & np.isfinite(pop) & (pop > 0)
years_gdp = years[mask_gdp]
if len(years_gdp) < 5:
    raise ValueError("可用GDP点太少。请确认已填入 GDP(1990–2024) + GDP预测(2025–2030)。")

gdp_pc = np.full_like(pop, np.nan, dtype=float)
gdp_pc[mask_gdp] = gdp_in[mask_gdp] / pop[mask_gdp]
last_gdp_year = int(years_gdp.max())

def growth_from_anchor_period():
    if (2025 in idx) and (2030 in idx):
        i25, i30 = idx[2025], idx[2030]
        if np.isfinite(gdp_pc[i25]) and np.isfinite(gdp_pc[i30]) and gdp_pc[i25] > 0 and gdp_pc[i30] > 0:
            return float((np.log(gdp_pc[i30]) - np.log(gdp_pc[i25])) / (2030 - 2025))
    if (2026 in idx) and (2030 in idx):
        i26, i30 = idx[2026], idx[2030]
        if np.isfinite(gdp_pc[i26]) and np.isfinite(gdp_pc[i30]) and gdp_pc[i26] > 0 and gdp_pc[i30] > 0:
            return float((np.log(gdp_pc[i30]) - np.log(gdp_pc[i26])) / (2030 - 2026))
    return None

g0 = growth_from_anchor_period()
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

print(f"✅ GDP最后可用年份: {last_gdp_year} | g0={g0:.6f} (ln增速/年)")


anchor_years = np.array(sorted(ANCHORS.keys()), dtype=int)
anchor_area  = np.array([ANCHORS[y] for y in anchor_years], dtype=float)

ln_gdppc_a = np.array([ln_gdppc_all[idx[y]] if y in idx else np.nan for y in anchor_years], dtype=float)
U_a        = np.array([urb01[idx[y]] if y in idx else np.nan for y in anchor_years], dtype=float)

m_fit = np.isfinite(ln_gdppc_a) & np.isfinite(U_a) & np.isfinite(anchor_area)
if m_fit.sum() < 4:
    raise ValueError(f"有效锚点不足：{m_fit.sum()} 个（锚点年份可能在宏观表里缺人口/城镇化/GDP）。")

def logistic_fixed_L(X, k, a, b, c):
    ln_gdppc, U = X
    z = k * (a + b * ln_gdppc + c * U)
    z = np.clip(z, -60, 60)
    return L_FIXED / (1.0 + np.exp(-z))

X_fit = np.vstack([ln_gdppc_a[m_fit], U_a[m_fit]])
y_fit = anchor_area[m_fit]

p0 = [1.0, -5.0, 1.0, 1.0]
bounds = ([1e-3, -100.0, 0.0, 0.0], [50.0, 100.0, 50.0, 50.0])

(k_hat, a_hat, b_hat, c_hat), _ = curve_fit(
    logistic_fixed_L, X_fit, y_fit, p0=p0, bounds=bounds, maxfev=40000
)

A_pred = logistic_fixed_L(np.vstack([ln_gdppc_all, urb01]), k_hat, a_hat, b_hat, c_hat)


A_fit_on_anchors = logistic_fixed_L(np.vstack([ln_gdppc_a[m_fit], U_a[m_fit]]), k_hat, a_hat, b_hat, c_hat)
rmse_val = rmse(y_fit, A_fit_on_anchors)
mape_val = mape(y_fit, A_fit_on_anchors)

print(f"✅ Logistic参数: k={k_hat:.6f}, a={a_hat:.6f}, b={b_hat:.6f}, c={c_hat:.6f} | L={L_FIXED}")
print(f"✅ 锚点拟合误差: RMSE={rmse_val:.3f} ㎡/人 | MAPE={mape_val:.2f}%")


anchor_col = np.full(len(years), np.nan, dtype=float)
for yk, vk in ANCHORS.items():
    if yk in idx:
        anchor_col[idx[yk]] = vk

out = pd.DataFrame({
    "年份": years,
    "人口": pop,
    "城镇化率(0-1)": urb01,
    "GDP_原始输入": gdp_in,
    "GDP_外推": gdp_ext,
    "GDP_合并(已知用输入,未知用外推)": gdp_merged,
    "人均GDP(GDPpc)": gdp_pc_all,
    "锚点_城镇人均住宅面积(㎡/人)": anchor_col,
    "预测_城镇人均住宅面积_Logistic(㎡/人)": A_pred
})

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
out.to_excel(OUT_PATH, index=False, engine="openpyxl")


plt.figure(figsize=(12, 6))
plt.plot(years, gdp_merged, linewidth=2.0, label="GDP（输入+外推合并）")
plt.scatter(years[np.isfinite(gdp_in)], gdp_in[np.isfinite(gdp_in)], s=18, alpha=0.7, label="GDP输入点")
plt.xlabel("年份")
plt.ylabel("GDP（Excel原单位）")
plt.title("US GDP total: input + extrapolation to 2100")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(years, A_pred, linewidth=2.2, label=f"预测面积（Logistic, L={L_FIXED}）")
plt.scatter(anchor_years, anchor_area, s=55, marker="o", label="真实锚点(1990-2015)")

mask_anchor_range = (years >= anchor_years.min()) & (years <= anchor_years.max())
plt.plot(years[mask_anchor_range], A_pred[mask_anchor_range], linewidth=3.0, alpha=0.55, label="锚点区间拟合曲线")

plt.xlabel("年份")
plt.ylabel("城镇人均住宅面积（㎡/人）")
plt.title(f"US urban per-capita residential area (Logistic S-curve, L={L_FIXED})\n"
          f"Fit on anchors: RMSE={rmse_val:.3f} ㎡/人, MAPE={mape_val:.2f}%")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

print("✅ 完成")
print(f"📁 已导出: {OUT_PATH}")
