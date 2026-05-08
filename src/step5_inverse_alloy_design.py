import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

out_dir = Path("output")
out_dir.mkdir(exist_ok=True)

model_path = Path("output/plots_step5_nested/best_model.pkl")
feat_path = Path("output/plots_step5_nested/feature_names.pkl")

if not model_path.exists():
    model_path = Path("output/plots_step5_hybrid/best_model.pkl")
    feat_path = Path("output/plots_step5_hybrid/feature_names.pkl")

if not model_path.exists():
    raise FileNotFoundError("No trained model found. Run Step 5 first.")

model = joblib.load(model_path)
train_features = joblib.load(feat_path)

print("Using model:", model_path)
print("Feature count:", len(train_features))

elements = ["Al", "Co", "Cr", "Ni", "Nb", "Mo", "Ta"]

props = {
    "Al":{"r":1.43,"chi":1.61,"Tm":933,"VEC":3,"G":26},
    "Co":{"r":1.25,"chi":1.88,"Tm":1768,"VEC":9,"G":75},
    "Cr":{"r":1.28,"chi":1.66,"Tm":2180,"VEC":6,"G":115},
    "Ni":{"r":1.24,"chi":1.91,"Tm":1728,"VEC":10,"G":76},
    "Nb":{"r":1.46,"chi":1.60,"Tm":2750,"VEC":5,"G":38},
    "Mo":{"r":1.39,"chi":2.16,"Tm":2896,"VEC":6,"G":126},
    "Ta":{"r":1.46,"chi":1.50,"Tm":3290,"VEC":5,"G":69},
}

rng = np.random.default_rng(42)
samples = rng.dirichlet(np.ones(len(elements)) * 1.2, size=15000)
design_df = pd.DataFrame(samples, columns=elements)

design_df["refractory"] = design_df["Nb"] + design_df["Mo"] + design_df["Ta"]

design_df = design_df[
    (design_df["refractory"] > 0.45) &
    (design_df["Cr"] > 0.08)
].reset_index(drop=True)

if len(design_df) == 0:
    raise ValueError("Empty design space after physics filter")

print("After physics filter:", len(design_df))

def compute_descriptors(df):
    def wavg(p):
        return sum(df[e] * props[e][p] for e in elements)

    d = pd.DataFrame(index=df.index)

    d["r_avg"] = wavg("r")
    d["chi_avg"] = wavg("chi")
    d["Tm_avg"] = wavg("Tm")
    d["VEC_avg"] = wavg("VEC")
    d["G_avg"] = wavg("G")

    d["delta"] = 100 * np.sqrt(
        sum(df[e] * (1 - props[e]["r"] / d["r_avg"])**2 for e in elements)
    )

    d["delta_sq"] = d["delta"]**2
    d["G_delta"] = d["G_avg"] * d["delta"]
    d["elastic_energy"] = d["G_avg"] * d["delta"]**2
    d["bond_energy_proxy"] = d["chi_avg"] * d["Tm_avg"]
    d["G_Tm"] = d["G_avg"] * d["Tm_avg"]
    d["delta_VEC"] = d["delta"] * d["VEC_avg"]

    return d.replace([np.inf, -np.inf], np.nan).fillna(0)

desc = compute_descriptors(design_df)

X_design = pd.DataFrame(0.0, index=design_df.index, columns=train_features)

for el in elements:
    col = f"ELEM_{el}"
    if col in X_design.columns:
        X_design[col] = design_df[el]

for col in desc.columns:
    if col in X_design.columns:
        X_design[col] = desc[col]

X_design = X_design[train_features]

if X_design.isnull().values.any():
    raise ValueError("NaN detected in model input")

missing = [c for c in train_features if c not in X_design.columns]
if len(missing) > 0:
    raise ValueError(f"Missing features: {missing}")

design_df["HV_ML"] = model.predict(X_design)

G_mean = desc["G_avg"].mean()

design_df["HV_Physics"] = (
    design_df["HV_ML"] *
    (desc["G_avg"] / G_mean)
)

design_df = design_df[
    (design_df["HV_ML"] > 460) &
    (design_df["HV_ML"] < 700) &
    (design_df["refractory"] > 0.50) &
    (design_df["Cr"] > 0.10)
].reset_index(drop=True)

print("After ML filtering:", len(design_df))

if len(design_df) < 50:
    print("Fallback triggered")
    design_df = design_df.sort_values("HV_Physics", ascending=False).head(200)

design_df["cluster_key"] = (
    (design_df[elements] * 10)
    .round()
    .astype(int)
    .astype(str)
    .agg("-".join, axis=1)
)

design_df = design_df.drop_duplicates("cluster_key").drop(columns=["cluster_key"])

top = design_df.sort_values("HV_Physics", ascending=False).reset_index(drop=True)

if len(top) >= 10 and top["HV_Physics"].nunique() >= 5:
    bins = pd.qcut(top["HV_Physics"], 5, duplicates="drop")
    top = top.groupby(bins).head(2)

top = top.head(10).copy()

top["Alloy_ID"] = ["Alloy_" + str(i+1) for i in range(len(top))]

def format_comp(row):
    return ", ".join([f"{e}:{row[e]:.2f}" for e in elements if row[e] > 0.01])

top["Composition"] = top.apply(format_comp, axis=1)

top["Signature"] = (
    top[elements]
    .round(3)
    .astype(str)
    .agg("-".join, axis=1)
)

top["Use_for_FEM"] = True

top = top.sort_values("Alloy_ID").reset_index(drop=True)

top.to_csv(out_dir / "top_alloys_for_FEM.csv", index=False)

plt.figure(figsize=(6, 4))
sns.histplot(design_df["HV_Physics"], bins=40, kde=True)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "design_distribution_corrected.png", dpi=600)
plt.close()

design_df.to_csv(out_dir / "full_design_space.csv", index=False)

print("\nTop ranked alloys:")
print(top[["Alloy_ID", "HV_ML", "HV_Physics", "Composition"]])

print("\nInverse design completed")
