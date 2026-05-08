import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.stats.outliers_influence import variance_inflation_factor

np.random.seed(42)
sns.set_style("whitegrid")

# PATHS

INPUT_PATH = Path("/content/output/HEA_descriptor_dataset.csv")
SHAP_PATH = Path("/content/output/plots_step8_shap_final/shap_importance_final.csv")

OUTPUT_DIR = Path("output/STEP9_FINAL_FIGURES")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LOAD DATA

df = pd.read_csv(INPUT_PATH)

target = "PROPERTY: HV"
df = df[df[target].notnull() & (df[target] > 0)].copy()

df = df.select_dtypes(include=[np.number]).copy()

# LOAD SHAP FEATURES

shap_df = pd.read_csv(SHAP_PATH)

# remove element features (important fix)
shap_df = shap_df[~shap_df["Feature"].str.startswith("ELEM_")]

# top features
TOP_N = 10
top_feats = shap_df["Feature"].head(TOP_N).tolist()

# ensure present in df
top_feats = [f for f in top_feats if f in df.columns]

data = df[top_feats + [target]].dropna().copy()

# 1.CORRELATION

corr = data.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    square=True,
    cbar_kws={"shrink": 0.8}
)
plt.title("Correlation Structure (Top SHAP Features)", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_clean.png", dpi=600)
plt.close()

# 2. HARDNESS CORRELATION

hv_corr = corr[target].drop(target).sort_values(ascending=False)

plt.figure(figsize=(6, 4))
sns.barplot(x=hv_corr.values, y=hv_corr.index)

plt.xlabel("Correlation with Hardness (HV)")
plt.title("Top Descriptor Influence on Hardness")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hv_correlation_bar.png", dpi=600)
plt.close()

# 3. VIF

X_vif = data[top_feats].copy()

# remove highly correlated features
corr_matrix = X_vif.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

drop_cols = [col for col in upper.columns if any(upper[col] > 0.9)]
X_vif = X_vif.drop(columns=drop_cols)

# standardize
X_scaled = (X_vif - X_vif.mean()) / X_vif.std()

vif_vals = [
    variance_inflation_factor(X_scaled.values, i)
    for i in range(X_scaled.shape[1])
]

vif_df = pd.DataFrame({
    "Feature": X_scaled.columns,
    "VIF": vif_vals
}).sort_values("VIF", ascending=False)

plt.figure(figsize=(6, 5))
sns.barplot(data=vif_df, x="VIF", y="Feature")

plt.axvline(10, linestyle="--")
plt.title("Multicollinearity Check (VIF)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vif_fixed.png", dpi=600)
plt.close()

# 4. PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[top_feats])

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# explained variance
explained = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained)

plt.figure(figsize=(7, 4))
plt.plot(explained, marker='o', label="Individual")
plt.plot(cum_explained, marker='s', linestyle='--', label="Cumulative")

plt.axhline(0.9, linestyle='--', label="90% variance")

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance")
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_variance.png", dpi=600)
plt.close()

# 5. PCA CLUSTERING

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_pca[:, :2])

plt.figure(figsize=(6, 5))
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels,
    cmap="viridis",
    alpha=0.7
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Material Clustering in Feature Space")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_clusters.png", dpi=600)
plt.close()

# 6. DESIGN MAP

f1, f2 = top_feats[0], top_feats[1]

plt.figure(figsize=(6, 5))
plt.scatter(
    data[f1],
    data[f2],
    c=data[target],
    cmap="viridis",
    alpha=0.8
)

plt.xlabel(f1)
plt.ylabel(f2)

cbar = plt.colorbar()
cbar.set_label("Hardness (HV)")

plt.title("Design Map (Descriptor Space → Hardness)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "design_map.png", dpi=600)
plt.close()

# 7. KEY RELATIONSHIPS

top3 = top_feats[:3]

for feat in top3:
    plt.figure(figsize=(6, 4))
    sns.regplot(x=data[feat], y=data[target], scatter_kws={"alpha":0.5})

    plt.xlabel(feat)
    plt.ylabel("Hardness (HV)")
    plt.title(f"{feat} vs Hardness")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{feat}_vs_HV.png", dpi=600)
    plt.close()

print(" Feature analysis completed ")
