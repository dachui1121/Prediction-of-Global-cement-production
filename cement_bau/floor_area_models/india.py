from cement_bau.floor_area_runtime import floor_input as _floor_input, floor_output as _floor_output, floor_anchors as _floor_anchors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from pathlib import Path


plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


excel_path = str(_floor_input('india', 'macro'))
sheet_name = "Sheet1"

df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
df.columns = df.columns.astype(str).str.strip()
df = df.dropna(how="all").copy()
df = df.replace([np.inf, -np.inf], np.nan)


def pick_col(df_, keywords, required=True):
    for kw in keywords:
        for c in df_.columns:
            if c == kw or (kw in c):
                return c
    if required:
        raise KeyError(f"找不到列（关键词 {keywords}），当前列名：{df_.columns.tolist()}")
    return None

year_col = pick_col(df, ["年份", "Year"])
pop_col  = pick_col(df, ["印度人口数", "人口", "Population"])
area_col = pick_col(df, ["人均住宅面积", "住宅面积", "Housing area"])


df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
df[pop_col]  = pd.to_numeric(df[pop_col], errors="coerce")
df[area_col] = pd.to_numeric(df[area_col], errors="coerce")

df = df.dropna(subset=[year_col, pop_col]).copy()
df[year_col] = df[year_col].round().astype(int)
df = df.sort_values(year_col).drop_duplicates(subset=[year_col]).reset_index(drop=True)

years = df[year_col].to_numpy(dtype=int)
POP_all = df[pop_col].to_numpy(dtype=float)
areas_all = df[area_col].to_numpy(dtype=float)


mask_fit = np.isfinite(POP_all) & np.isfinite(areas_all)
if mask_fit.sum() < 4:
    raise ValueError(f"可用于拟合的有效点太少：{mask_fit.sum()} 个。请检查 {area_col} 是否有历史数据。")

POP_fit = POP_all[mask_fit]
areas_fit = areas_all[mask_fit]


def logistic_population(pop, L, k, t0):
    return L / (1 + np.exp(-k * (pop - t0)))


L0 = float(np.nanmax(areas_fit) * 1.2)
t00 = float(np.nanmedian(POP_fit))
k0 = 1.0 / max(1.0, float(np.nanstd(POP_fit)))

p0 = [L0, k0, t00]


pop_min, pop_max = float(np.nanmin(POP_fit)), float(np.nanmax(POP_fit))
bounds = (
    [0.0, 0.0, pop_min - (pop_max - pop_min) * 2.0],
    [200.0, 10.0, pop_max + (pop_max - pop_min) * 2.0]
)

popt, _ = curve_fit(
    logistic_population, POP_fit, areas_fit,
    p0=p0, bounds=bounds, maxfev=20000
)
L_fit, k_fit, t0_fit = popt


areas_pred_all = logistic_population(POP_all, L_fit, k_fit, t0_fit)


result_df = pd.DataFrame({
    "年份": years,
    "人口": POP_all,
    "原始人均住宅面积(㎡/人)": areas_all,
    "Logistic拟合/预测人均住宅面积(㎡/人)": areas_pred_all
})


plt.figure(figsize=(12, 7))
plt.scatter(years[mask_fit], areas_fit, color="red", s=50, label="历史数据点（用于拟合）", zorder=5)
plt.plot(years, areas_pred_all, color="blue", linewidth=2.2,
         label=f"人口→面积 Logistic\nL={L_fit:.2f}, k={k_fit:.4g}, t0={t0_fit:.3g}")
plt.xlabel("年份")
plt.ylabel("人均住宅面积（㎡/人）")
plt.title("印度人均住宅面积：基于人口的 Logistic 拟合与预测（1989–2100）")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


desktop_path = str(_floor_output('india'))
output_path = os.path.join(desktop_path, "印度人均住宅面积_人口Logistic_1989-2100_拟合预测.xlsx")
result_df.to_excel(output_path, index=False, engine="openpyxl")

print(f"✅ 拟合完成：L={L_fit:.2f}, k={k_fit:.6g}, t0={t0_fit:.6g}")
print(f"✅ 输出已保存：{output_path}")
print(f"（提示）用于拟合的有效历史点数：{mask_fit.sum()}；总年份数：{len(years)}")
print(f"识别到列名：年份='{year_col}', 人口='{pop_col}', 面积='{area_col}'")
