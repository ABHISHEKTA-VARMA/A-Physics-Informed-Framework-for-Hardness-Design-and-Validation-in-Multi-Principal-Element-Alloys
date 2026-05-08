import pandas as pd
import hashlib
from pathlib import Path
from datetime import datetime


INPUT_PATH = Path("MASTER_HV_DATASET.csv")

out_dir = Path("output")
out_dir.mkdir(exist_ok=True)

locked_path = out_dir / "MASTER_HV_DATASET_STEP1_LOCKED.csv"
meta_path = out_dir / "dataset_metadata.txt"


df = pd.read_csv(INPUT_PATH)

required = ["PROPERTY: HV", "Source"]
for c in required:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

elem_cols = [c for c in df.columns if c.startswith("ELEM_")]
if len(elem_cols) == 0:
    raise ValueError("No composition columns found")

if df.isnull().values.any():
    raise ValueError("Missing values present")

if (df["PROPERTY: HV"] <= 0).any():
    raise ValueError("Invalid hardness values")

if (df["PROPERTY: HV"] > 1200).any():
    print("High HV values detected (>1200)")

row_sum = df[elem_cols].sum(axis=1)
bad = df[abs(row_sum - 1) > 1e-6]
if len(bad) > 0:
    raise ValueError(f"{len(bad)} rows not normalized")


df = df.sort_values(by=["PROPERTY: HV", "Source"]).reset_index(drop=True)
df = df.sort_index(axis=1)

df.to_csv(locked_path, index=False)


with open(locked_path, "rb") as f:
    h = hashlib.sha256(f.read()).hexdigest()


with open(meta_path, "w") as f:
    f.write("Dataset metadata\n")
    f.write(f"Timestamp (UTC): {datetime.utcnow().isoformat()}\n")
    f.write(f"SHA256: {h}\n")
    f.write(f"Samples: {len(df)}\n")
    f.write(f"Columns: {len(df.columns)}\n")
    f.write(f"Element features: {len(elem_cols)}\n")
    f.write(
        f"HV range: {df['PROPERTY: HV'].min():.2f} - {df['PROPERTY: HV'].max():.2f}\n\n"
    )

    f.write("Columns:\n")
    for c in df.columns:
        f.write(c + "\n")


print("Saved:", locked_path)
print("Hash:", h)
