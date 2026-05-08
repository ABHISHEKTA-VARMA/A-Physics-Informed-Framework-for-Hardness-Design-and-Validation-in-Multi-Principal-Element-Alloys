import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import linregress


# CONFIG

INPUT_FILE = "/content/CREEP_COMPILED_DATASET.csv"
OUTPUT_DIR = "/content/output_creep"

os.makedirs(OUTPUT_DIR, exist_ok=True)

R = 8.314
sns.set(style="whitegrid", context="talk")

print("INPUT_FILE:", INPUT_FILE)
print("OUTPUT_DIR:", OUTPUT_DIR)


# LOAD + CLEAN

df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip().str.lower()

df = df.rename(columns={
    "name": "alloy",
    "temperature": "temp",
    "force": "creep",
    "deformation": "strain"
})

df["alloy"] = df["alloy"].str.upper()


# TEMPERATURE TERMS

df["T_K"] = df["temp"] + 273.15
df["inv_T"] = 1 / df["T_K"]


# ROOM TEMP REFERENCE

df_rt = df[df["temp"] == 27][["alloy", "creep"]].rename(columns={"creep": "creep_rt"})
df = df.merge(df_rt, on="alloy")

df["F_norm"] = df["creep"] / df["creep_rt"]


#  ACTIVATION ENERGY (ARRHENIUS)

rows = []
for alloy, g in df.groupby("alloy"):
    slope, intercept, r_value, _, _ = linregress(
        g["inv_T"].values,
        np.log(g["creep"].values)
    )
    Q = abs(slope * R)
    rows.append({
        "alloy": alloy,
        "Q_J_per_mol": Q,
        "R2": r_value**2
    })

Q_df = pd.DataFrame(rows).sort_values("alloy")

print("\n ACTIVATION ENERGY (ARRHENIUS)")
print(Q_df)


#  EFFECTIVE ACTIVATION INDEX

F_mean = df_rt["creep_rt"].mean()
Q_df = Q_df.merge(df_rt, on="alloy")

Q_df["Q_effective"] = Q_df["Q_J_per_mol"] * (Q_df["creep_rt"] / F_mean)

print("\n EFFECTIVE ACTIVATION INDEX")
print(Q_df[["alloy", "Q_effective"]])

# MODEL UPDATE

df = df.merge(Q_df[["alloy", "Q_effective"]], on="alloy")

df["creep_model"] = df["creep_rt"] * np.exp(
    -df["Q_effective"] / (R * df["T_K"])
)

df["model_norm"] = df["creep_model"] / df["creep_rt"]


# HARDNESS PROXY

df["hv"] = df["creep"] / 9.807
df["hv_rt"] = df["creep_rt"] / 9.807
df["hv_norm"] = df["hv"] / df["hv_rt"]


#  TEMPERATURE SENSITIVITY

sens_rows = []
for alloy, g in df.groupby("alloy"):
    slope = np.polyfit(g["temp"], g["creep"], 1)[0]
    sens_rows.append({"alloy": alloy, "dF_dT": slope})

sens_df = pd.DataFrame(sens_rows).sort_values("alloy")


#  PLOTS

# 1. FORCE vs TEMP
plt.figure(figsize=(8,6))
sns.lineplot(data=df, x="temp", y="creep", hue="alloy", marker="o")
plt.title("Force Degradation with Temperature")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/force_vs_temp.png", dpi=300)
plt.close()

# 2. UNIVERSAL CURVE
plt.figure(figsize=(8,6))
sns.lineplot(data=df, x="temp", y="F_norm", estimator="mean", marker="o")
plt.title("Universal Degradation Curve")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/master_curve.png", dpi=300)
plt.close()

# 3. ARRHENIUS
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="inv_T", y=np.log(df["creep"]))
sns.regplot(x=df["inv_T"], y=np.log(df["creep"]), scatter=False, color="red")
plt.title("Arrhenius Behavior")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/arrhenius.png", dpi=300)
plt.close()

# 4. HARDNESS
plt.figure(figsize=(8,6))
sns.lineplot(data=df, x="temp", y="hv", hue="alloy", marker="o")
plt.title("Hardness Degradation")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/hardness.png", dpi=300)
plt.close()

# 5. HEATMAP
pivot = df.pivot_table(index="alloy", columns="temp", values="hv")
plt.figure(figsize=(8,6))
sns.heatmap(pivot, annot=True, cmap="viridis")
plt.title("Hardness Map")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/heatmap.png", dpi=300)
plt.close()

# 6. SENSITIVITY
plt.figure(figsize=(8,6))
sns.barplot(data=sens_df, x="alloy", y="dF_dT")
plt.xticks(rotation=45)
plt.title("Temperature Sensitivity (dF/dT)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sensitivity.png", dpi=300)
plt.close()

# 7. Q COMPARISON
plt.figure(figsize=(8,6))
x = np.arange(len(Q_df))
w = 0.35

plt.bar(x - w/2, Q_df["Q_J_per_mol"], w, label="Activation Energy")
plt.bar(x + w/2, Q_df["Q_effective"], w, label="Effective Index")

plt.xticks(x, Q_df["alloy"], rotation=45)
plt.title("Activation Energy vs Effective Index")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Q_comparison.png", dpi=300)
plt.close()

# 8. COLLAPSE vs SEPARATION
plt.figure(figsize=(8,6))

# collapse
for _, g in df.groupby("alloy"):
    plt.plot(g["temp"], g["F_norm"], color="black", alpha=0.3)

# separation
for alloy, g in df.groupby("alloy"):
    plt.plot(g["temp"], g["model_norm"], label=alloy)

plt.title("Collapse vs Separation (Key Result)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Normalized Response")
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/collapse_vs_separation.png", dpi=300)
plt.close()


# SPREAD METRIC

spread_raw = df.groupby("temp")["F_norm"].std().mean()
spread_model = df.groupby("temp")["model_norm"].std().mean()

print("\n SPREAD METRIC")
print(f"Raw collapse spread: {spread_raw:.4f}")
print(f"Model separation spread: {spread_model:.4f}")

# SAVE

df.to_csv(f"{OUTPUT_DIR}/final_dataset.csv", index=False)
Q_df.to_csv(f"{OUTPUT_DIR}/activation_energy.csv", index=False)
sens_df.to_csv(f"{OUTPUT_DIR}/sensitivity.csv", index=False)

print("\n Output saved to:", OUTPUT_DIR)
