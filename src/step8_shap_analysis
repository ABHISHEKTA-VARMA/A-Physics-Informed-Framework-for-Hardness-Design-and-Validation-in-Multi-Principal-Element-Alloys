import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor

np.random.seed(42)

# PATHS

INPUT_PATH = Path("/content/output/HEA_descriptor_dataset.csv")
MODEL_PATH = Path("/content/output/plots_step5_hybrid/best_model.pkl")
FEATURE_PATH = Path("/content/output/plots_step5_hybrid/feature_names.pkl")

OUTPUT_DIR = Path("output/plots_step8_shap_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LOAD DATA

df = pd.read_csv(INPUT_PATH)
target_col = "PROPERTY: HV"
df = df[df[target_col] > 0].copy()

# LOAD MODEL + FEATURES

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)

X = df[feature_names].copy()
y = df[target_col].values

# OUTLIER REMOVAL

z = np.abs((y - y.mean()) / y.std())
mask = z < 3
X = X.iloc[mask]
y = y[mask]

# APPLY PREPROCESSING FROM PIPELINE

X_proc = model[:-1].transform(X)

# MODEL + EXPLAINER

final_model = model.named_steps["model"]

explainer = shap.TreeExplainer(
    final_model,
    feature_perturbation="tree_path_dependent"
)

shap_exp = explainer(X_proc, check_additivity=False)

shap_values = shap_exp.values
base_value = explainer.expected_value

X_df = pd.DataFrame(X_proc, columns=feature_names)

# FEATURE GROUPING

desc_features = [f for f in feature_names if not f.startswith("ELEM_")]
elem_features = [f for f in feature_names if f.startswith("ELEM_")]

# BEESWARM PLOT

plt.figure()
shap.summary_plot(shap_values, X_df, show=False)
plt.savefig(OUTPUT_DIR / "shap_beeswarm.png", dpi=600, bbox_inches="tight")
plt.close()

# BAR IMPORTANCE

plt.figure()
shap.summary_plot(shap_values, X_df, plot_type="bar", show=False)
plt.savefig(OUTPUT_DIR / "shap_bar.png", dpi=600, bbox_inches="tight")
plt.close()

# IMPORTANCE + VARIABILITY

mean_importance = np.abs(shap_values).mean(axis=0)
std_importance = np.abs(shap_values).std(axis=0)

shap_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance_mean": mean_importance,
    "Importance_std": std_importance
})

shap_df["Type"] = shap_df["Feature"].apply(
    lambda x: "Descriptor" if x in desc_features else "Element"
)

shap_df = shap_df.sort_values("Importance_mean", ascending=False)
shap_df.to_csv(OUTPUT_DIR / "shap_importance_final.csv", index=False)

# STABILITY PLOT

plt.figure(figsize=(8, 10))
plt.errorbar(
    shap_df["Importance_mean"],
    shap_df["Feature"],
    xerr=shap_df["Importance_std"],
    fmt='o'
)
plt.gca().invert_yaxis()
plt.xlabel("Mean |SHAP|")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_importance_stability.png", dpi=600)
plt.close()

# FEATURE TYPE COMPARISON

plt.figure()
sns.boxplot(data=shap_df, x="Type", y="Importance_mean")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "descriptor_vs_element.png", dpi=600)
plt.close()

# DEPENDENCE PLOTS

top_feats = shap_df["Feature"].head(5)

for feat in top_feats:
    shap.dependence_plot(
        feat,
        shap_values,
        X_df,
        interaction_index="auto",
        show=False
    )
    plt.savefig(OUTPUT_DIR / f"dependence_{feat}.png", dpi=600, bbox_inches="tight")
    plt.close()

# INTERACTION SUMMARY

idx_sample = np.random.choice(len(X_proc), size=min(200, len(X_proc)), replace=False)

interaction_values = explainer.shap_interaction_values(X_proc[idx_sample])

plt.figure()
shap.summary_plot(interaction_values, X_df.iloc[idx_sample], show=False)
plt.savefig(OUTPUT_DIR / "shap_interaction_summary.png", dpi=600, bbox_inches="tight")
plt.close()

# LOCAL EXPLANATION

mid = len(X_df) // 2

plt.figure(figsize=(8, 6))
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[mid],
        base_values=base_value,
        data=X_df.iloc[mid],
        feature_names=feature_names
    ),
    max_display=12,
    show=False
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "local_waterfall.png", dpi=600, bbox_inches="tight")
plt.close()

# CORRELATION + VIF

top15 = shap_df["Feature"].head(15)

X_top = X_df[top15]
X_scaled = (X_top - X_top.mean()) / X_top.std()

corr = X_scaled.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation.png", dpi=600)
plt.close()

vif_df = pd.DataFrame({
    "Feature": top15,
    "VIF": [
        variance_inflation_factor(X_scaled.values, i)
        for i in range(len(top15))
    ]
})

vif_df.to_csv(OUTPUT_DIR / "vif.csv", index=False)

# RELATION WITH MODEL OUTPUT

design_check = X_df.copy()
design_check["Predicted_HV"] = model.predict(X)

top_features = shap_df["Feature"].head(3)

for feat in top_features:
    plt.figure()
    plt.scatter(design_check[feat], design_check["Predicted_HV"], alpha=0.6)
    plt.xlabel(feat)
    plt.ylabel("Predicted HV")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"design_link_{feat}.png", dpi=600)
    plt.close()

print("Shap analysis completed")
