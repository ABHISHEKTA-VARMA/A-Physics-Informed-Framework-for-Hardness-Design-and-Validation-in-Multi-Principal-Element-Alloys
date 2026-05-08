import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor


inp = Path("output/HEA_descriptor_dataset.csv")
out_dir = Path("output/step5b_consistency")
out_dir.mkdir(exist_ok=True)

sns.set_style("whitegrid")


df = pd.read_csv(inp)

target = "PROPERTY: HV"
df = df[df[target].notnull() & (df[target] > 0)].copy()

X = df.drop(columns=[target, "Source"], errors="ignore")
X = X.select_dtypes(include=np.number)

X = X.loc[:, X.notnull().any()]
X = X.loc[:, (X != 0).any(axis=0)]

y = df[target].values

if len(X) < 50:
    raise ValueError("Dataset too small for stable cross-validation")


seeds = [0, 21, 42, 77, 100]
rows = []


for s in seeds:

    y_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)

    r2_s, rmse_s, mae_s, dummy_s = [], [], [], []

    for tr, te in kf.split(X, y_bins):

        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]

        z = np.abs((y_tr - y_tr.mean()) / (y_tr.std() + 1e-9))
        mask = z < 3

        X_tr = X_tr.iloc[mask]
        y_tr = y_tr[mask]

        model = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ExtraTreesRegressor(
                n_estimators=800,
                max_features="sqrt",
                random_state=s,
                n_jobs=-1
            ))
        ])

        model.fit(X_tr, y_tr)
        yp = model.predict(X_te)

        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_tr, y_tr)
        yd = dummy.predict(X_te)

        r2_s.append(r2_score(y_te, yp))
        rmse_s.append(np.sqrt(mean_squared_error(y_te, yp)))
        mae_s.append(mean_absolute_error(y_te, yp))
        dummy_s.append(r2_score(y_te, yd))

    rows.append([
        s,
        np.mean(r2_s),
        np.std(r2_s),
        np.mean(rmse_s),
        np.mean(mae_s),
        np.mean(dummy_s)
    ])


cons = pd.DataFrame(
    rows,
    columns=["Seed", "R2_mean", "R2_std", "RMSE", "MAE", "Dummy_R2"]
)

cons["R2_gain"] = cons["R2_mean"] - cons["Dummy_R2"]

cons.to_csv(out_dir / "consistency_results.csv", index=False)


summary = pd.DataFrame({
    "Metric": ["R2", "RMSE", "MAE"],
    "Mean": [
        cons["R2_mean"].mean(),
        cons["RMSE"].mean(),
        cons["MAE"].mean()
    ],
    "Std": [
        cons["R2_mean"].std(),
        cons["RMSE"].std(),
        cons["MAE"].std()
    ]
})

summary.to_csv(out_dir / "summary.csv", index=False)


plt.figure(figsize=(6, 4))
plt.plot(cons["Seed"], cons["R2_mean"], marker="o")
plt.xlabel("Random seed")
plt.ylabel("R2")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "r2_stability.png", dpi=600)
plt.close()


perm_scores = []
rng = np.random.default_rng(42)

for i in range(50):

    y_perm = rng.permutation(y)
    y_perm_bins = pd.qcut(y_perm, q=5, labels=False, duplicates="drop")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)

    fold_scores = []

    for tr, te in kf.split(X, y_perm_bins):

        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y_perm[tr], y_perm[te]

        model = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ExtraTreesRegressor(
                n_estimators=800,
                max_features="sqrt",
                random_state=i,
                n_jobs=-1
            ))
        ])

        model.fit(X_tr, y_tr)
        yp = model.predict(X_te)

        fold_scores.append(r2_score(y_te, yp))

    perm_scores.append(np.mean(fold_scores))


plt.figure(figsize=(6, 4))
sns.histplot(perm_scores, bins=12)
plt.xlabel("Permutation R2")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "permutation_test.png", dpi=600)
plt.close()


print("Saved:", out_dir)
