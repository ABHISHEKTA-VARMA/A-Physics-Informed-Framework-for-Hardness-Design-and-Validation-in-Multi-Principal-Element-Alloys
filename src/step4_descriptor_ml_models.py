import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import VarianceThreshold

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor


RANDOM_STATE = 42

inp = Path("output/HEA_descriptor_dataset.csv")
out_dir = Path("output/plots_step5_hybrid")
out_dir.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 13})
sns.set_style("whitegrid")


df = pd.read_csv(inp)

target = "PROPERTY: HV"
df = df[df[target].notnull() & (df[target] > 0)].copy()

X = df.drop(columns=[target, "Source"], errors="ignore")
X = X.select_dtypes(include=np.number)

y = df[target].values

# Feature cleaning

cols0 = X.columns.tolist()

vt = VarianceThreshold(1e-8)
X_np = vt.fit_transform(X)

cols_kept = [c for c, k in zip(cols0, vt.get_support()) if k]
X = pd.DataFrame(X_np, columns=cols_kept, index=df.index)

corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_cols = [c for c in upper.columns if any(upper[c] > 0.95)]
X = X.drop(columns=drop_cols)

# CV

outer = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
inner = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# Params

xgb_param = {
    "model__n_estimators": [500, 700],
    "model__max_depth": [4, 5],
    "model__learning_rate": [0.03, 0.05],
    "model__subsample": [0.7, 0.85],
    "model__colsample_bytree": [0.7, 0.85]
}

et_param = {
    "model__n_estimators": [700, 900],
    "model__max_depth": [None, 15],
    "model__max_features": ["sqrt", None]
}

# Models

models = {
    "Extra Trees": ("et", et_param),
    "Random Forest": RandomForestRegressor(
        n_estimators=800,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=RANDOM_STATE
    ),
    "XGBoost": ("xgb", xgb_param)
}


colors = {
    "Extra Trees": "#ff7f0e",
    "Random Forest": "#1f77b4",
    "Gradient Boosting": "#2ca02c",
    "XGBoost": "#d62728"
}

# Train

rows = []
cv_preds = {}
feat_store = {}

for name, info in models.items():

    r2_s, rmse_s, mae_s = [], [], []
    y_full = np.zeros(len(y))
    imps = []

    for tr, te in outer.split(X):

        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]

        # outlier filtering
        z = np.abs((y_tr - y_tr.mean()) / (y_tr.std() + 1e-9))
        mask = z < 3
        X_tr = X_tr.iloc[mask]
        y_tr = y_tr[mask]

        if isinstance(info, tuple):

            tag, grid = info

            if tag == "xgb":
                base = XGBRegressor(
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=0
                )
            else:
                base = ExtraTreesRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )

            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", base)
            ])

            search = RandomizedSearchCV(
                pipe,
                param_distributions=grid,
                n_iter=10,
                scoring="r2",
                cv=inner,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )

            search.fit(X_tr, y_tr)
            model = search.best_estimator_

        else:
            model = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", info)
            ])
            model.fit(X_tr, y_tr)

        yp = model.predict(X_te)
        y_full[te] = yp

        r2_s.append(r2_score(y_te, yp))
        rmse_s.append(np.sqrt(mean_squared_error(y_te, yp)))
        mae_s.append(mean_absolute_error(y_te, yp))

        m = model.named_steps["model"]
        if hasattr(m, "feature_importances_"):
            imps.append(m.feature_importances_)

    cv_preds[name] = y_full

    rows.append([
        name,
        np.mean(r2_s),
        np.mean(rmse_s),
        np.mean(mae_s)
    ])

    if imps:
        feat_store[name] = np.mean(imps, axis=0)

# Results

res = pd.DataFrame(
    rows,
    columns=["Model", "R2", "RMSE", "MAE"]
).sort_values("R2", ascending=False).reset_index(drop=True)

res.to_csv(out_dir / "model_results.csv", index=False)

# Plots

for metric in ["R2", "RMSE", "MAE"]:

    vals = res.set_index("Model")[metric]
    vals = vals.sort_values(ascending=(metric != "R2"))

    plt.figure(figsize=(6, 4))
    plt.bar(vals.index, vals.values,
            color=[colors[m] for m in vals.index])

    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}.png", dpi=600)
    plt.close()

# parity

plt.figure(figsize=(7, 7))
for name, yp in cv_preds.items():
    plt.scatter(y, yp, s=18, alpha=0.5,
                color=colors[name], label=name)

lims = [y.min(), y.max()]
plt.plot(lims, lims, "k--")

plt.xlabel("Experimental HV")
plt.ylabel("Predicted HV")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "parity_all.png", dpi=600)
plt.close()


# residuals

plt.figure(figsize=(7, 4))
for name, yp in cv_preds.items():
    sns.kdeplot(y - yp, label=name, linewidth=2)

plt.xlabel("Residual (HV)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "residuals.png", dpi=600)
plt.close()


# feature importance

best = res.iloc[0]["Model"]

if best in feat_store:
    imp = pd.Series(feat_store[best], index=X.columns)
    imp = imp.sort_values(ascending=False).head(15)

    plt.figure(figsize=(6, 4))
    imp[::-1].plot(kind="barh",
                   color=colors[best],
                   edgecolor="black")

    plt.xlabel("Importance")
    plt.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=600)
    plt.close()


# heatmap

corr_df = X.copy()
corr_df["HV"] = y

top = corr_df.corr()["HV"].abs().sort_values(ascending=False).iloc[1:16].index

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_df[top.tolist() + ["HV"]].corr(),
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    square=True
)

plt.tight_layout()
plt.savefig(out_dir / "correlation_heatmap.png", dpi=600)
plt.close()


print("Saved:", out_dir)
print("Best:", best)

# Save optimized model

if best == "XGBoost":
    final_model = XGBRegressor(
        n_estimators=700,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )

elif best == "Extra Trees":
    final_model = ExtraTreesRegressor(
        n_estimators=900,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

elif best == "Random Forest":
    final_model = RandomForestRegressor(
        n_estimators=800,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

else:
    final_model = GradientBoostingRegressor(
        random_state=RANDOM_STATE
    )


final_pipeline = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", final_model)
])

final_pipeline.fit(X, y)

joblib.dump(final_pipeline, out_dir / "best_model.pkl")
joblib.dump(list(X.columns), out_dir / "feature_names.pkl")

print("Model and feature metadata exported")
