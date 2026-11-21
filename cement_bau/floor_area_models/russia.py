from cement_bau.floor_area_runtime import floor_input as _floor_input, floor_output as _floor_output, floor_anchors as _floor_anchors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import expit
from numpy.linalg import lstsq


ANCHORS_RU = _floor_anchors('russia')


excel_path = str(_floor_input('russia', 'macro'))
sheet_name = "Sheet1"


G_LONG = 0.012
TAU    = 25.0


L_MAX = 60.0
L_LOWER_MARGIN = 2.0

USE_POP_TERM = True
POP_TERM_KIND = "ln_pop"

LOSS = "soft_l1"
F_SCALE = 1.0

REG_W = 0.02
B_MIN = 0.05

K_UB = 4.0
YEAR_MIN = 1989
YEAR_MAX = 2100
AUTO_GDP_UNIT_SCALE = True


SCALE_START = 1989
SCALE_END   = 2030


L_TARGET = 45.0
L_SIGMA  = 8.0
L_W      = 1.0


C_LO, C_HI = -2.0, 2.0
D_LO, D_HI = -2.0, 2.0


KZ_CLIP = 20.0


SAT_MAX_2100 = 0.99
SAT_SIGMA    = 0.01


def pick_col(df_, keywords, required=True):
    cols = df_.columns.astype(str).tolist()
    for kw in keywords:
        if kw in cols:
            return kw
    for kw in keywords:
        for c in cols:
            if kw in str(c):
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
    return float(np.clip(g, -0.05, 0.15))

def zscore_with_idx(full_arr, idx_fit):
    v = np.asarray(full_arr, float)
    aa = v[idx_fit]
    mu = np.nanmean(aa)
    sd = np.nanstd(aa)
    if (not np.isfinite(sd)) or sd < 1e-12:
        sd = 1.0
    return (v - mu) / sd, mu, sd

def logistic_cov(params, x1, x2, x3, use_pop=True):

    L, k, a, b, c, d = params
    z  = a + b * x1 + c * x2 + d * x3
    kz = np.clip(k * z, -KZ_CLIP, KZ_CLIP)
    return L * expit(kz), kz, z

def at_bound(val, lo, hi, tol=1e-6):
    return (abs(val - lo) < tol) or (abs(val - hi) < tol)


df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
df.columns = df.columns.astype(str).str.strip()
df = df.dropna(how="all").replace([np.inf, -np.inf], np.nan)

year_col = pick_col(df, ["年份", "Year", "year"])
pop_col  = pick_col(df, ["人口", "Population", "pop"])
urb_col  = pick_col(df, ["城镇化率", "城市化率", "Urbanization", "urban"])
gdp_col  = pick_col(df, ["GDP（十亿美元）", "GDP(十亿美元)", "GDP总量", "GDP", "gdp"])

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

if AUTO_GDP_UNIT_SCALE:
    if ("十亿" in str(gdp_col)) or ("billion" in str(gdp_col).lower()):
        gdp_in = gdp_in * 1e9

idx = {y: i for i, y in enumerate(years)}
print(f"✅ 年份范围：{years.min()}–{years.max()}（{len(years)}年）")
print(f"✅ 识别列：year={year_col}, pop={pop_col}, urban={urb_col}, gdp={gdp_col}")


mask_gdp = np.isfinite(gdp_in) & (gdp_in > 0) & np.isfinite(pop) & (pop > 0)
years_gdp = years[mask_gdp]
if len(years_gdp) < 5:
    raise ValueError("可用GDP点太少，请确认历史GDP + 2025–2030 锚点")

gdp_pc = np.full_like(pop, np.nan, dtype=float)
gdp_pc[mask_gdp] = gdp_in[mask_gdp] / pop[mask_gdp]
last_gdp_year = int(years_gdp.max())

def growth_from_2025_2030():
    if (2025 in idx) and (2030 in idx):
        i25, i30 = idx[2025], idx[2030]
        if np.isfinite(gdp_pc[i25]) and np.isfinite(gdp_pc[i30]) and gdp_pc[i25] > 0 and gdp_pc[i30] > 0:
            return float((np.log(gdp_pc[i30]) - np.log(gdp_pc[i25])) / 5.0)
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
ln_gdppc_all = np.log(np.maximum(gdp_pc_all, 1e-12))

if USE_POP_TERM:
    if POP_TERM_KIND == "ln_pop":
        pop_term_all = np.log(np.maximum(pop, 1e-12))
    elif POP_TERM_KIND == "ln_urban_pop":
        pop_term_all = np.log(np.maximum(pop * urb01, 1e-12))
    else:
        raise ValueError("POP_TERM_KIND 只能是 ln_pop 或 ln_urban_pop")
else:
    pop_term_all = np.zeros_like(pop)


anchor_years = np.array(sorted(ANCHORS_RU.keys()), dtype=int)
anchor_area  = np.array([ANCHORS_RU[y] for y in anchor_years], dtype=float)

missing = [y for y in anchor_years if y not in idx]
if missing:
    raise ValueError(f"锚点年份缺失：{missing}")
anchor_idx = np.array([idx[y] for y in anchor_years], dtype=int)


scale_idx = np.where((years >= SCALE_START) & (years <= min(SCALE_END, years.max())))[0]
if len(scale_idx) < 10:
    scale_idx = np.where(np.isfinite(ln_gdppc_all) & np.isfinite(urb01) & np.isfinite(pop_term_all))[0]

ln_z, mu1, sd1 = zscore_with_idx(ln_gdppc_all, scale_idx)
U_z,  mu2, sd2 = zscore_with_idx(urb01,       scale_idx)
P_z,  mu3, sd3 = zscore_with_idx(pop_term_all, scale_idx)

print(f"✅ 标准化尺度窗口：{years[scale_idx].min()}–{years[scale_idx].max()}，Urban_sd={sd2:.6g}")


X = np.column_stack([np.ones(len(anchor_idx)), U_z[anchor_idx], P_z[anchor_idx]])
coef, *_ = lstsq(X, ln_z[anchor_idx], rcond=None)
ln_orth = ln_z - (coef[0] + coef[1] * U_z + coef[2] * P_z)


x1 = ln_orth[anchor_idx]
x2 = U_z[anchor_idx]
x3 = P_z[anchor_idx]
y  = anchor_area.copy()

ymax = float(np.nanmax(y))
L_min = ymax + L_LOWER_MARGIN
if L_min >= L_MAX:
    raise ValueError("L_min 已 >= L_MAX，请调 L_MAX 或 L_LOWER_MARGIN")


i2100 = idx[2100] if 2100 in idx else None

def residuals(params):
    pred, kz_anchor, _ = logistic_cov(params, x1, x2, x3, use_pop=True)


    reg = REG_W * np.array([params[1], params[4], params[5]])


    regL = (L_W * (params[0] - L_TARGET) / L_SIGMA) if (L_W > 0) else 0.0


    if i2100 is not None:
        _, kz2100, _ = logistic_cov(params, ln_orth[i2100], U_z[i2100], P_z[i2100], use_pop=True)
        s2100 = expit(np.clip(kz2100, -KZ_CLIP, KZ_CLIP))
        regSat = max(0.0, (s2100 - SAT_MAX_2100) / SAT_SIGMA)
    else:
        regSat = 0.0

    return np.r_[pred - y, reg, regL, regSat]


L0 = float(np.clip(max(L_TARGET, ymax + 10.0), L_min + 0.5, L_MAX - 0.5))
p0 = np.array([L0, 1.2, 0.0, max(0.6, B_MIN), 0.6, 0.0], float)

lb = np.array([L_min, 1e-3, -20.0, B_MIN, C_LO, D_LO], float)
ub = np.array([L_MAX, K_UB,  20.0,  5.0, C_HI, D_HI], float)

res = least_squares(residuals, p0, bounds=(lb, ub), loss=LOSS, f_scale=F_SCALE, max_nfev=200000)
p_hat = res.x
L_hat, k_hat, a_hat, b_hat, c_hat, d_hat = p_hat


A_pred, kz_all, z_all = logistic_cov(p_hat, ln_orth, U_z, P_z, use_pop=True)


diff_tail = np.max(np.abs(np.diff(A_pred[-50:])))
print("\n=== 参数（最终）===")
print(f"L_hat={L_hat:.6f}   (range [{L_min:.2f}, {L_MAX:.2f}])  at_bound={at_bound(L_hat, L_min, L_MAX)}")
print(f"k_hat={k_hat:.6f}   (<= {K_UB})                        at_bound={at_bound(k_hat, 1e-3, K_UB)}")
print(f"a_hat={a_hat:.6f}")
print(f"b_hat={b_hat:.6f}   (lnGDPpc_orth)                     at_bound={at_bound(b_hat, B_MIN, 5.0)}")
print(f"c_hat={c_hat:.6f}   (Urban_z)                          at_bound={at_bound(c_hat, C_LO, C_HI)}")
print(f"d_hat={d_hat:.6f}   ({POP_TERM_KIND}_z)                at_bound={at_bound(d_hat, D_LO, D_HI)}")
print(f"status={res.status}, cost={res.cost:.6f}, nfev={res.nfev}")
print(f"tail max|ΔA| (last 50 years) = {diff_tail:.8f}")

if i2100 is not None:
    s2100 = expit(np.clip(kz_all[i2100], -KZ_CLIP, KZ_CLIP))
    print(f"2100: expit(kz)={s2100:.6f}  (constraint <= {SAT_MAX_2100})")


anchor_col = np.full(len(years), np.nan, dtype=float)
for yy, vv in ANCHORS_RU.items():
    anchor_col[idx[yy]] = vv

out = pd.DataFrame({
    "年份": years,
    "人口": pop,
    "城镇化率(0-1)": urb01,
    "GDP_合并(美元口径)": gdp_merged,
    "人均GDP(GDPpc)": gdp_pc_all,
    "ln(人均GDP)": ln_gdppc_all,
    "lnGDPpc_z(尺度窗)": ln_z,
    "Urban_z(尺度窗)": U_z,
    "PopTerm_z(尺度窗)": P_z,
    "lnGDPpc_orth": ln_orth,
    "kz_all": kz_all,
    "锚点_人均住宅面积": anchor_col,
    "预测_人均住宅面积_Logistic": A_pred
})

out_path = (_floor_output('russia') / '俄罗斯_Logistic_去特征裁剪_含远期约束_1989-2100.xlsx')
out.to_excel(out_path, index=False, engine="openpyxl")
print(f"\n📁 已导出：{out_path}")


plt.figure(figsize=(11, 6))
plt.plot(years, A_pred, linewidth=2, label="预测：人均住宅面积 (修复版)")
plt.scatter(anchor_years, anchor_area, s=60, marker="o", label="锚点", zorder=5)

plt.axhline(L_hat, linestyle="--", linewidth=1.5, label=f"L_hat={L_hat:.2f}")
plt.title("Russia - Per-capita Residential Floor Area Fit & Forecast (1989–2100)")
plt.xlabel("Year")
plt.ylabel("Area (m²/person)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


for yy in [2030, 2040, 2050, 2070, 2090, 2100]:
    if yy in idx:
        i = idx[yy]
        print(f"{yy}: A={A_pred[i]:.4f}, kz={kz_all[i]:.3f}, ln_orth={ln_orth[i]:.3f}, U_z={U_z[i]:.3f}, P_z={P_z[i]:.3f}")
