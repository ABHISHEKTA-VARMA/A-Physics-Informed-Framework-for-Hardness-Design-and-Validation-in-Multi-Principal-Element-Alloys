import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

STATIC_FILE = Path("STATIC_RAW.csv")
ML_FILE = Path("output/top_alloys_for_FEM.csv")
OUT_DIR = Path("output/ml_fem_validation")
OUT_DIR.mkdir(exist_ok=True)

sns.set(style="whitegrid", context="talk")

df_static = pd.read_csv(STATIC_FILE)
df_ml = pd.read_csv(ML_FILE)

df_static.columns = df_static.columns.str.strip()
df_ml.columns = df_ml.columns.str.strip()

# FEM hardness from indentation
h_corr = 0.84 * df_static["Depth_h_mm"]
df_static["FEM_HV"] = df_static["Force_P_N"] / (24.5 * (h_corr**2) * 9.807)

# merge datasets
df = pd.merge(df_ml, df_static, left_on="Alloy_ID", right_on="Alloy")

# use physics-corrected hardness
df = df.rename(columns={"HV_Physics": "Pred_HV"})

# metrics
df["error"] = df["FEM_HV"] - df["Pred_HV"]
df["abs_error"] = np.abs(df["error"])

mae = df["abs_error"].mean()
rmse = np.sqrt(np.mean(df["error"]**2))
rho, _ = spearmanr(df["Pred_HV"], df["FEM_HV"])

reg = LinearRegression().fit(df["Pred_HV"].values.reshape(-1,1), df["FEM_HV"])
slope, intercept = reg.coef_[0], reg.intercept_

# ordering for plots
df["Alloy_num"] = df["Alloy_ID"].str.extract(r'(\d+)').astype(int)
df_sorted = df.sort_values("Alloy_num")

print("\nValidation metrics")
print(f"MAE        : {mae:.2f} HV")
print(f"RMSE       : {rmse:.2f} HV")
print(f"Spearman   : {rho:.3f}")
print(f"Fit        : y = {slope:.3f}x + {intercept:.2f}")

# agreement
plt.figure(figsize=(7,7))

plt.scatter(
    df["Pred_HV"],
    df["FEM_HV"],
    s=100,
    edgecolor="black",
    linewidth=0.6,
    alpha=0.85
)

lims = [
    min(df["Pred_HV"].min(), df["FEM_HV"].min()),
    max(df["Pred_HV"].max(), df["FEM_HV"].max())
]

plt.plot(lims, lims, linestyle="--")

x_line = np.linspace(lims[0], lims[1], 100)
plt.plot(x_line, slope*x_line + intercept)

plt.fill_between(lims, np.array(lims)-20, np.array(lims)+20, alpha=0.12)

plt.text(
    0.98, 0.02,
    f"MAE = {mae:.1f} HV\nRMSE = {rmse:.1f} HV\nSpearman = {rho:.2f}",
    transform=plt.gca().transAxes,
    ha="right",
    va="bottom",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
)

plt.xlabel("Physics-corrected Hardness (HV)")
plt.ylabel("FEM Hardness (HV)")

plt.tight_layout()
plt.savefig(OUT_DIR / "fig1_agreement.png", dpi=600)
plt.close()

# deviation
colors = [
    "#4C72B0" if abs(e) <= 20 else "#DD8452"
    for e in df_sorted["error"]
]

plt.figure(figsize=(9,5))
plt.bar(df_sorted["Alloy_ID"], df_sorted["error"], color=colors)

plt.axhline(0, color="black", linewidth=1)
plt.axhline(20, linestyle="--", linewidth=1)
plt.axhline(-20, linestyle="--", linewidth=1)

plt.ylabel("FEM - Predicted (HV)")
plt.xticks(rotation=30, ha="right")

plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_deviation.png", dpi=600)
plt.close()

# trend
plt.figure(figsize=(9,5))

plt.plot(
    df_sorted["Alloy_ID"],
    df_sorted["Pred_HV"],
    marker="o",
    linewidth=2,
    label="Predicted"
)

plt.plot(
    df_sorted["Alloy_ID"],
    df_sorted["FEM_HV"],
    marker="s",
    linewidth=2,
    label="FEM"
)

plt.xticks(rotation=30, ha="right")
plt.ylabel("Hardness (HV)")
plt.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "fig3_trend.png", dpi=600)
plt.close()

df.to_csv(OUT_DIR / "final_dataset.csv", index=False)

print("\nSaved:", OUT_DIR)
