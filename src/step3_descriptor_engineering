import numpy as np
import pandas as pd
from pathlib import Path

inp = Path("output/MASTER_HV_DATASET_STEP1_LOCKED.csv")
out = Path("output/HEA_descriptor_dataset.csv")
out.parent.mkdir(exist_ok=True)

df = pd.read_csv(inp)
if df.empty:
    raise ValueError("Empty dataset")

props = {
    "Al":{"r":1.43,"M":26.98,"chi":1.61,"Tm":933,"VEC":3,"E":70e9,"G":26e9,"K":76e9},
    "Co":{"r":1.25,"M":58.93,"chi":1.88,"Tm":1768,"VEC":9,"E":211e9,"G":75e9,"K":180e9},
    "Cr":{"r":1.28,"M":52.00,"chi":1.66,"Tm":2180,"VEC":6,"E":279e9,"G":115e9,"K":160e9},
    "Fe":{"r":1.26,"M":55.85,"chi":1.83,"Tm":1811,"VEC":8,"E":211e9,"G":82e9,"K":170e9},
    "Ni":{"r":1.24,"M":58.69,"chi":1.91,"Tm":1728,"VEC":10,"E":200e9,"G":76e9,"K":180e9},
    "Ti":{"r":1.47,"M":47.87,"chi":1.54,"Tm":1941,"VEC":4,"E":116e9,"G":44e9,"K":110e9},
    "V":{"r":1.34,"M":50.94,"chi":1.63,"Tm":2183,"VEC":5,"E":128e9,"G":47e9,"K":160e9},
    "Nb":{"r":1.46,"M":92.91,"chi":1.60,"Tm":2750,"VEC":5,"E":105e9,"G":38e9,"K":170e9},
    "Mo":{"r":1.39,"M":95.95,"chi":2.16,"Tm":2896,"VEC":6,"E":329e9,"G":126e9,"K":230e9},
    "Ta":{"r":1.46,"M":180.95,"chi":1.50,"Tm":3290,"VEC":5,"E":186e9,"G":69e9,"K":200e9},
    "Zr":{"r":1.60,"M":91.22,"chi":1.33,"Tm":2128,"VEC":4,"E":88e9,"G":33e9,"K":92e9},
    "Hf":{"r":1.59,"M":178.49,"chi":1.30,"Tm":2506,"VEC":4,"E":78e9,"G":30e9,"K":110e9},
    "W":{"r":1.39,"M":183.84,"chi":2.36,"Tm":3695,"VEC":6,"E":411e9,"G":161e9,"K":310e9},
}

elem_cols = [c for c in df.columns if c.startswith("ELEM_")]
elements_all = [c.replace("ELEM_", "") for c in elem_cols]

supported = [e for e in elements_all if e in props]
if len(supported) == 0:
    raise ValueError("No supported elements")

cols = [f"ELEM_{e}" for e in supported]

comp_full = df[elem_cols].fillna(0)
coverage = comp_full[cols].sum(axis=1)

df["descriptor_coverage"] = coverage
df = df[coverage >= 0.9].reset_index(drop=True)

comp = df[cols].copy()
comp = comp.div(comp.sum(axis=1), axis=0).fillna(0)

elements = supported

def wavg(p):
    return sum(comp[f"ELEM_{e}"] * props[e][p] for e in elements)

def wvar(p):
    m = wavg(p)
    return sum(comp[f"ELEM_{e}"] * (props[e][p] - m) ** 2 for e in elements)

d = pd.DataFrame(index=df.index)
R = 8.314

d["Smix"] = -R * np.sum(comp * np.log(comp + 1e-12), axis=1)

d["r_avg"] = wavg("r")
d["delta"] = 100 * np.sqrt(
    sum(comp[f"ELEM_{e}"] * (1 - props[e]["r"] / d["r_avg"]) ** 2 for e in elements)
)

d["chi_avg"] = wavg("chi")
d["Tm_avg"] = wavg("Tm")
d["VEC_avg"] = wavg("VEC")

d["E_avg"] = wavg("E")
d["G_avg"] = wavg("G")
d["K_avg"] = wavg("K")

d["G_var"] = wvar("G")
d["elastic_aniso"] = d["G_var"] / (d["G_avg"] + 1e-9)

d["Omega"] = (d["Tm_avg"] * d["Smix"]) / (d["delta"] ** 2 + 1e-6)

d["G_delta"] = d["G_avg"] * d["delta"]
d["elastic_energy"] = d["G_avg"] * d["delta"] ** 2
d["yield_proxy"] = d["G_avg"] * (1 + d["delta"] / 100)
d["hardness_core"] = d["G_avg"] * d["delta"] ** 2
d["G_over_E"] = d["G_avg"] / (d["E_avg"] + 1e-9)
d["resistance_index"] = d["G_avg"] * d["delta"] * d["Tm_avg"]

d["G_Tm"] = d["G_avg"] * d["Tm_avg"]
d["delta_VEC"] = d["delta"] * d["VEC_avg"]
d["elastic_entropy"] = d["G_avg"] * d["Smix"]

d["bond_energy_proxy"] = d["chi_avg"] * d["Tm_avg"]
d["M_avg"] = wavg("M")

d["log_G"] = np.log(d["G_avg"])
d["log_E"] = np.log(d["E_avg"])

if d.isnull().values.any():
    raise ValueError("NaN in descriptors")

if np.isinf(d.values).any():
    raise ValueError("Inf in descriptors")

df_out = pd.concat([df, d], axis=1)
df_out.to_csv(out, index=False)

print("Saved:", out)
print("Shape:", df_out.shape)
