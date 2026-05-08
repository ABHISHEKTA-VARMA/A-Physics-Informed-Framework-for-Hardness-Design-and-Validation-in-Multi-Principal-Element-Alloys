import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# output directory

save_dir = Path("output")
save_dir.mkdir(parents=True, exist_ok=True)

# plotting setup

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.major.width"] = 1.1
plt.rcParams["ytick.major.width"] = 1.1

sns.set_style("ticks")
sns.set_context("paper", font_scale=1.35)


# load trained model

nested_model = Path(
    "output/plots_step5_nested/best_model.pkl"
)

nested_features = Path(
    "output/plots_step5_nested/feature_names.pkl"
)

hybrid_model = Path(
    "output/plots_step5_hybrid/best_model.pkl"
)

hybrid_features = Path(
    "output/plots_step5_hybrid/feature_names.pkl"
)

if nested_model.exists():

    model_path = nested_model
    feature_path = nested_features

elif hybrid_model.exists():

    model_path = hybrid_model
    feature_path = hybrid_features

else:
    raise FileNotFoundError(
        "No trained model was found."
    )

model = joblib.load(model_path)
feature_names = joblib.load(feature_path)

print(f"\nModel loaded from: {model_path}")
print(f"Feature count: {len(feature_names)}")

# alloy system

elements = [
    "Al",
    "Co",
    "Cr",
    "Ni",
    "Nb",
    "Mo",
    "Ta"
]

properties = {

    "Al": {
        "r": 1.43,
        "chi": 1.61,
        "Tm": 933,
        "VEC": 3,
        "G": 26
    },

    "Co": {
        "r": 1.25,
        "chi": 1.88,
        "Tm": 1768,
        "VEC": 9,
        "G": 75
    },

    "Cr": {
        "r": 1.28,
        "chi": 1.66,
        "Tm": 2180,
        "VEC": 6,
        "G": 115
    },

    "Ni": {
        "r": 1.24,
        "chi": 1.91,
        "Tm": 1728,
        "VEC": 10,
        "G": 76
    },

    "Nb": {
        "r": 1.46,
        "chi": 1.60,
        "Tm": 2750,
        "VEC": 5,
        "G": 38
    },

    "Mo": {
        "r": 1.39,
        "chi": 2.16,
        "Tm": 2896,
        "VEC": 6,
        "G": 126
    },

    "Ta": {
        "r": 1.46,
        "chi": 1.50,
        "Tm": 3290,
        "VEC": 5,
        "G": 69
    }
}

# Monte Carlo compositional exploration

rng = np.random.default_rng(42)

samples = rng.dirichlet(
    alpha=np.ones(len(elements)) * 1.2,
    size=15000
)

design_space = pd.DataFrame(
    samples,
    columns=elements
)

design_space["refractory_fraction"] = (
    design_space["Nb"] +
    design_space["Mo"] +
    design_space["Ta"]
)

design_space = design_space.loc[
    (design_space["refractory_fraction"] > 0.45) &
    (design_space["Cr"] > 0.08)
].reset_index(drop=True)

print(
    f"Design space after physical filtering: "
    f"{len(design_space)}"
)

# descriptor construction

descriptor_df = pd.DataFrame(
    index=design_space.index
)

for prop in ["r", "chi", "Tm", "VEC", "G"]:

    descriptor_df[f"{prop}_avg"] = sum(
        design_space[el] * properties[el][prop]
        for el in elements
    )

descriptor_df["delta"] = 100 * np.sqrt(

    sum(
        design_space[el] *
        (
            1 -
            properties[el]["r"] /
            descriptor_df["r_avg"]
        )**2

        for el in elements
    )
)

descriptor_df["delta_sq"] = (
    descriptor_df["delta"]**2
)

descriptor_df["G_delta"] = (
    descriptor_df["G_avg"] *
    descriptor_df["delta"]
)

descriptor_df["elastic_energy"] = (
    descriptor_df["G_avg"] *
    descriptor_df["delta_sq"]
)

descriptor_df["bond_energy_proxy"] = (
    descriptor_df["chi_avg"] *
    descriptor_df["Tm_avg"]
)

descriptor_df["G_Tm"] = (
    descriptor_df["G_avg"] *
    descriptor_df["Tm_avg"]
)

descriptor_df["delta_VEC"] = (
    descriptor_df["delta"] *
    descriptor_df["VEC_avg"]
)

descriptor_df["elastic_aniso"] = (
    descriptor_df["G_delta"] /
    (descriptor_df["r_avg"] + 1e-9)
)

descriptor_df["G_var"] = (
    descriptor_df["G_avg"] *
    descriptor_df["delta_sq"]
)

descriptor_df = descriptor_df.replace(
    [np.inf, -np.inf],
    np.nan
).fillna(0)

# prepare model input

X = pd.DataFrame(
    0.0,
    index=design_space.index,
    columns=feature_names
)

for el in elements:

    feature = f"ELEM_{el}"

    if feature in X.columns:
        X[feature] = design_space[el]

for col in descriptor_df.columns:

    if col in X.columns:
        X[col] = descriptor_df[col]

X = X[feature_names]

if X.isnull().values.any():

    raise ValueError(
        "NaN values detected in model input."
    )

# hardness prediction

design_space["HV_ML"] = model.predict(X)

g_reference = descriptor_df["G_avg"].mean()

design_space["HV_Physics"] = (

    design_space["HV_ML"] *
    (
        descriptor_df["G_avg"] /
        g_reference
    )
)

# physics-based filtering

design_space = design_space.loc[
    (design_space["HV_ML"] > 460) &
    (design_space["HV_ML"] < 700) &
    (design_space["refractory_fraction"] > 0.50) &
    (design_space["Cr"] > 0.10)
].reset_index(drop=True)

print(
    f"Filtered candidate count: "
    f"{len(design_space)}"
)

# remove near-duplicate alloys

design_space["signature"] = (

    (design_space[elements] * 10)
    .round()
    .astype(int)
    .astype(str)
    .agg("-".join, axis=1)
)

design_space = (

    design_space
    .drop_duplicates("signature")
    .drop(columns="signature")
    .reset_index(drop=True)
)

# rebuild descriptor matrix after filtering

descriptor_filtered = pd.DataFrame(
    index=design_space.index
)

for prop in ["r", "chi", "Tm", "VEC", "G"]:

    descriptor_filtered[f"{prop}_avg"] = sum(
        design_space[el] * properties[el][prop]
        for el in elements
    )

descriptor_filtered["delta"] = 100 * np.sqrt(

    sum(
        design_space[el] *
        (
            1 -
            properties[el]["r"] /
            descriptor_filtered["r_avg"]
        )**2

        for el in elements
    )
)

descriptor_filtered["delta_sq"] = (
    descriptor_filtered["delta"]**2
)

descriptor_filtered["G_delta"] = (
    descriptor_filtered["G_avg"] *
    descriptor_filtered["delta"]
)

descriptor_filtered["elastic_energy"] = (
    descriptor_filtered["G_avg"] *
    descriptor_filtered["delta_sq"]
)

descriptor_filtered["elastic_aniso"] = (
    descriptor_filtered["G_delta"] /
    (descriptor_filtered["r_avg"] + 1e-9)
)

descriptor_filtered["G_var"] = (
    descriptor_filtered["G_avg"] *
    descriptor_filtered["delta_sq"]
)

descriptor_filtered = descriptor_filtered.fillna(0)

# top alloy extraction

ranked = (

    design_space
    .sort_values(
        "HV_Physics",
        ascending=False
    )
    .reset_index(drop=True)
)

if ranked["HV_Physics"].nunique() >= 5:

    groups = pd.qcut(
        ranked["HV_Physics"],
        q=5,
        duplicates="drop"
    )

    ranked = ranked.groupby(groups).head(2)

top_alloys = ranked.head(10).copy()

top_alloys["Alloy_ID"] = [

    f"Alloy_{i+1}"
    for i in range(len(top_alloys))
]


def composition_string(row):

    return ", ".join(

        [
            f"{el}:{row[el]:.2f}"
            for el in elements
            if row[el] > 0.01
        ]
    )


top_alloys["Composition"] = top_alloys.apply(
    composition_string,
    axis=1
)

top_alloys["Use_for_FEM"] = True

top_alloys = (

    top_alloys
    .sort_values(
        "HV_Physics",
        ascending=False
    )
    .reset_index(drop=True)
)

# save csv outputs

design_space.to_csv(
    save_dir / "full_design_space.csv",
    index=False
)

top_alloys.to_csv(
    save_dir / "top_alloys_for_FEM.csv",
    index=False
)

# Fig. 5(a)
# hardness distribution

fig, ax = plt.subplots(
    figsize=(8.2, 5.4)
)

sns.histplot(
    design_space["HV_Physics"],
    bins=42,
    kde=True,
    edgecolor="white",
    linewidth=0.8,
    ax=ax
)

ax.set_xlabel(
    "Physics-Corrected Hardness (HV)"
)

ax.set_ylabel("Frequency")

ax.tick_params(direction="in")

sns.despine()

plt.tight_layout()

plt.savefig(
    save_dir /
    "Fig5a_hardness_distribution.png",
    dpi=800,
    bbox_inches="tight"
)

plt.close()

# Fig. 5(b)
# descriptor-space distribution


descriptor_columns = [

    "r_avg",
    "chi_avg",
    "Tm_avg",
    "VEC_avg",
    "G_avg",
    "delta",
    "G_delta",
    "elastic_energy",
    "elastic_aniso",
    "G_var"
]

scaled_data = StandardScaler().fit_transform(
    descriptor_filtered[descriptor_columns]
)

pca = PCA(n_components=2)

projection = pca.fit_transform(
    scaled_data
)

pca_frame = pd.DataFrame({

    "PC1": projection[:, 0],
    "PC2": projection[:, 1],
    "HV_Physics":
        design_space["HV_Physics"].values
})

fig, ax = plt.subplots(
    figsize=(7.8, 6)
)

scatter = ax.scatter(
    pca_frame["PC1"],
    pca_frame["PC2"],
    c=pca_frame["HV_Physics"],
    s=30,
    alpha=0.85,
    linewidths=0
)

colorbar = plt.colorbar(scatter)

colorbar.set_label(
    "Physics-Corrected Hardness (HV)"
)

ax.set_xlabel(
    f"PC1 "
    f"({pca.explained_variance_ratio_[0]*100:.1f}%)"
)

ax.set_ylabel(
    f"PC2 "
    f"({pca.explained_variance_ratio_[1]*100:.1f}%)"
)

ax.tick_params(direction="in")

sns.despine()

plt.tight_layout()

plt.savefig(
    save_dir /
    "Fig5b_descriptor_space_distribution.png",
    dpi=800,
    bbox_inches="tight"
)

plt.close()

# Fig. 5(c)
# compositional clustering

cluster_model = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=25
)

cluster_labels = cluster_model.fit_predict(
    design_space[elements]
)

cluster_frame = pd.DataFrame({

    "PC1": projection[:, 0],
    "PC2": projection[:, 1],
    "Cluster": cluster_labels.astype(str)
})

fig, ax = plt.subplots(
    figsize=(7.8, 6)
)

sns.scatterplot(
    data=cluster_frame,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="deep",
    s=40,
    alpha=0.90,
    linewidth=0.25,
    edgecolor="black",
    ax=ax
)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

ax.tick_params(direction="in")

legend = ax.legend(
    title="Cluster",
    frameon=True
)

try:

    handles = legend.legend_handles

except AttributeError:

    handles = legend.legendHandles

for handle in handles:
    handle.set_alpha(1)

sns.despine()

plt.tight_layout()

plt.savefig(
    save_dir /
    "Fig5c_compositional_clustering.png",
    dpi=800,
    bbox_inches="tight"
)

plt.close()

# Fig. 5(d)
# ranked alloy systems

rank_plot = (

    top_alloys
    .sort_values(
        "HV_Physics",
        ascending=True
    )
)

fig, ax = plt.subplots(
    figsize=(8.2, 5.6)
)

bars = ax.barh(
    rank_plot["Alloy_ID"],
    rank_plot["HV_Physics"],
    edgecolor="black",
    linewidth=0.7
)

for bar in bars:

    value = bar.get_width()

    ax.text(
        value + 2,
        bar.get_y() +
        bar.get_height() / 2,
        f"{value:.1f}",
        va="center",
        fontsize=10
    )

ax.set_xlabel(
    "Physics-Corrected Hardness (HV)"
)

ax.set_ylabel(
    "Candidate Alloy"
)

ax.tick_params(direction="in")

sns.despine()

plt.tight_layout()

plt.savefig(
    save_dir /
    "Fig5d_ranked_alloys.png",
    dpi=800,
    bbox_inches="tight"
)

plt.close()



# summary output


print("\nTop candidate alloys:\n")

print(

    top_alloys[
        [
            "Alloy_ID",
            "HV_ML",
            "HV_Physics",
            "Composition"
        ]
    ]
)

print("\nInverse-design workflow completed.")
print("Figures and CSV files exported.")
