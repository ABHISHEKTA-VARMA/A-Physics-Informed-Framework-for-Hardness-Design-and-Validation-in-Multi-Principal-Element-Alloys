import pandas as pd
import re
from pathlib import Path


BORG_PATH = Path("MPEA_dataset.csv")
GORSSE_PATH = Path("Gorsse_HV_FINAL.csv")
GE_PATH = Path("GE_RefractoryAlloyScreeningDataset_FINAL.csv")
OUTPUT_PATH = Path("MASTER_HV_DATASET.csv")


def parse_formula(formula):
    parts = re.findall(r'([A-Z][a-z]*)([0-9]*\.?[0-9]*)', str(formula))

    comp = {}
    for el, val in parts:
        try:
            v = float(val) if val not in ["", None] else 1.0
        except:
            v = 1.0
        comp[f"ELEM_{el}"] = v

    s = sum(comp.values())
    if s > 0:
        for k in comp:
            comp[k] /= s

    return comp


borg = pd.read_csv(BORG_PATH)
borg = borg.dropna(subset=["PROPERTY: HV"])

borg_elem = pd.DataFrame(borg["FORMULA"].apply(parse_formula).tolist()).fillna(0)
borg_elem["PROPERTY: HV"] = borg["PROPERTY: HV"].values
borg_elem["Source"] = "Borg"
borg_elem["HV_origin"] = "Experimental"


gorsse = pd.read_csv(GORSSE_PATH)

gorsse_elem = pd.DataFrame(gorsse["Composition"].apply(parse_formula).tolist()).fillna(0)
gorsse_elem["PROPERTY: HV"] = gorsse["HV"]
gorsse_elem["Source"] = "Gorsse"
gorsse_elem["HV_origin"] = "Experimental"


ge = pd.read_csv(GE_PATH)

element_map = {
    "Hf(at%)": "ELEM_Hf",
    "Mo(at%)": "ELEM_Mo",
    "Nb(at%)": "ELEM_Nb",
    "Re(at%)": "ELEM_Re",
    "Ru(at%)": "ELEM_Ru",
    "Ta(at%)": "ELEM_Ta",
    "Ti(at%)": "ELEM_Ti",
    "W(at%)": "ELEM_W",
    "Zr(at%)": "ELEM_Zr",
}

ge_elem = pd.DataFrame()

for c_old, c_new in element_map.items():
    if c_old in ge.columns:
        ge_elem[c_new] = ge[c_old]

ge_elem = ge_elem.fillna(0)

rs = ge_elem.sum(axis=1)
rs[rs == 0] = 1
ge_elem = ge_elem.div(rs, axis=0)

if "Hardness (GPa)" in ge.columns:
    ge_elem["PROPERTY: HV"] = ge["Hardness (GPa)"] * 102  # GPa → HV
    ge_elem["HV_origin"] = "Derived_from_GPa"
    ge_elem["HV_conversion_note"] = "1 GPa ≈ 102 HV"

ge_elem["Source"] = "GE"


master = pd.concat([borg_elem, gorsse_elem, ge_elem], ignore_index=True)
master = master.fillna(0)

elem_cols = [c for c in master.columns if c.startswith("ELEM_")]
master = master.drop_duplicates(subset=elem_cols + ["PROPERTY: HV"])

rs = master[elem_cols].sum(axis=1)
rs[rs == 0] = 1
master[elem_cols] = master[elem_cols].div(rs, axis=0)

cs = master[elem_cols].sum(axis=1)
bad = master[abs(cs - 1) > 1e-4]

if len(bad) > 0:
    print(len(bad), "rows adjusted during normalization")

master["num_elements"] = (master[elem_cols] > 0).sum(axis=1)
master["HV_unit"] = "kgf/mm^2"

master["processing_note"] = (
    "merged datasets; normalized compositions; duplicates removed; HV converted where needed"
)

master.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)
print("Shape:", master.shape)
