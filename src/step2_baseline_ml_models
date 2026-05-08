import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor


def bootstrap_r2(y_true, y_pred, n=500, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    n_s = len(y_true)

    for _ in range(n):
        idx = rng.choice(n_s, n_s, replace=True)
        vals.append(r2_score(y_true[idx], y_pred[idx]))

    return np.percentile(vals, [2.5, 97.5])


RANDOM_STATE = 42

inp = Path("output/MASTER_HV_DATASET_STEP1_LOCKED.csv")
out_dir = Path("output/plots_step3")
out_dir.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 13})
sns.set_style("whitegrid")


df = pd.read_csv(inp)

target = "PROPERTY: HV"
elem_cols = [c for c in df.columns if c.startswith("ELEM_")]

X = df[elem_cols].values
y = df[target].values


models = {
    "Extra Trees": Pipeline([
        ("m", ExtraTreesRegressor(
            n_estimators=900,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]),
    "Random Forest": Pipeline([
        ("m", RandomForestRegressor(
            n_estimators=800,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]),
    "Gradient Boosting": Pipeline([
        ("m", GradientBoostingRegressor(random_state=RANDOM_STATE))
    ]),
    "XGBoost": Pipeline([
        ("m", XGBRegressor(
            objective="reg:squarederror",
            n_estimators=800,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        ))
    ])
}


colors = {
    "Extra Trees": "#ff7f0e",
    "Random Forest": "#1f77b4",
    "Gradient Boosting": "#2ca02c",
    "XGBoost": "#d62728"
}


y_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


rows = []
cv_preds = {}
feat_imp = {}

for name, pipe in models.items():

    r2_s, rmse_s, mae_s = [], [], []
    y_full = np.zeros_like(y, dtype=float)
    imps = []

    for tr, te in kf.split(X, y_bins):

        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        pipe.fit(X_tr, y_tr)
        yp = pipe.predict(X_te)

        y_full[te] = yp

        r2_s.append(r2_score(y_te, yp))
        rmse_s.append(np.sqrt(mean_squared_error(y_te, yp)))
        mae_s.append(mean_absolute_error(y_te, yp))

        m = pipe.named_steps["m"]
        if hasattr(m, "feature_importances_"):
            imps.append(m.feature_importances_)

    cv_preds[name] = y_full

    ci_l, ci_h = bootstrap_r2(y, y_full)

    rows.append([
        name,
        np.mean(r2_s),
        np.mean(rmse_s),
        np.mean(mae_s),
        ci_l,
        ci_h
    ])

    if imps:
        feat_imp[name] = np.mean(imps, axis=0)


res = pd.DataFrame(
    rows,
    columns=["Model", "R2", "RMSE", "MAE", "CI_low", "CI_high"]
).sort_values("R2", ascending=False)

res.to_csv(out_dir / "model_results.csv", index=False)


# Metric Plots

for metric in ["R2", "RMSE", "MAE"]:

    vals = res.set_index("Model")[metric]

    if metric == "R2":
        vals = vals.sort_values(ascending=False)
    else:
        vals = vals.sort_values(ascending=True)

    plt.figure(figsize=(6, 4))
    plt.bar(vals.index, vals.values,
            color=[colors[m] for m in vals.index])

    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}.png", dpi=600)
    plt.close()


# Prediction Parity

plt.figure(figsize=(7, 7))

for name, yp in cv_preds.items():
    plt.scatter(y, yp, s=18, alpha=0.5,
                color=colors[name], label=name)

lims = [y.min(), y.max()]
plt.plot(lims, lims, "k--")

plt.xlabel("Experimental HV")
plt.ylabel("Predicted HV")
plt.legend(frameon=True)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "parity_all.png", dpi=600)
plt.close()


# Residual Distribution

plt.figure(figsize=(7, 4))

for name, yp in cv_preds.items():
    sns.kdeplot(y - yp, label=name, linewidth=2)

plt.xlabel("Residual (HV)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "residuals.png", dpi=600)
plt.close()


# Feature Importance

best = res.iloc[0]["Model"]

if best in feat_imp:

    imp = pd.Series(feat_imp[best], index=elem_cols)
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


print(f"Results saved in {out_dir}")
