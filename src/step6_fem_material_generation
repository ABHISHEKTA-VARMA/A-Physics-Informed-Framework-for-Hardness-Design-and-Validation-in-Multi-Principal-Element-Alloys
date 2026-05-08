import numpy as np
import pandas as pd
from pathlib import Path

print("\nSTEP 7: GENERATING FEM MATERIAL FILES")

INPUT_PATH = Path("output/top_alloys_for_FEM.csv")
OUTPUT_DIR = Path("output/ansys_materials")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_PATH.exists():
    raise FileNotFoundError("Run Step 6 first")

df = pd.read_csv(INPUT_PATH)

if "HV_Physics" not in df.columns:
    raise ValueError("HV_Physics missing. Re-run Step 6")

df = df.sort_values("HV_Physics", ascending=False).reset_index(drop=True)
df["Alloy_ID"] = ["Alloy_" + str(i+1) for i in range(len(df))]

print("Alloys ranked by physics-corrected hardness")

elements = ["Al","Co","Cr","Ni","Nb","Mo","Ta"]

props = {
    "Al":{"E":70e9,"G":26e9,"rho":2700,"r":1.43},
    "Co":{"E":211e9,"G":75e9,"rho":8900,"r":1.25},
    "Cr":{"E":279e9,"G":115e9,"rho":7190,"r":1.28},
    "Ni":{"E":200e9,"G":76e9,"rho":8900,"r":1.24},
    "Nb":{"E":105e9,"G":38e9,"rho":8570,"r":1.46},
    "Mo":{"E":329e9,"G":126e9,"rho":10280,"r":1.39},
    "Ta":{"E":186e9,"G":69e9,"rho":16650,"r":1.46},
}

def wavg(row, key):
    return sum(row.get(e,0) * props[e][key] for e in elements)

def compute_poisson(E, G):
    return np.clip((E/(2*G))-1, 0.22, 0.35)

def compute_delta(row):
    r_avg = sum(row.get(e,0)*props[e]["r"] for e in elements)
    if r_avg == 0:
        return 0
    return 100*np.sqrt(
        sum(row.get(e,0)*(1-props[e]["r"]/r_avg)**2 for e in elements)
    )

def compute_refractory(row):
    return row.get("Nb",0)+row.get("Mo",0)+row.get("Ta",0)

def compute_density(row):
    return sum(row.get(e,0)*props[e]["rho"] for e in elements)

def estimate_yield(G, delta, refractory):
    delta_term = 1 + min(delta/40, 1.2)
    ref_term = 1 + min(refractory, 0.6)
    sigma = 0.0065 * G * delta_term * ref_term
    return np.clip(sigma, 300e6, 1.6e9)

total_strain = np.array([0.002,0.01,0.03,0.06,0.10])

summary = []

for i, row in df.iterrows():

    mat_id = i + 1
    name = row["Alloy_ID"]

    E = wavg(row,"E")
    G = wavg(row,"G")

    if E <= 0 or G <= 0:
        continue

    nu = compute_poisson(E,G)
    rho = compute_density(row)

    delta = compute_delta(row)
    refractory = compute_refractory(row)

    sigma_y = estimate_yield(G,delta,refractory)

    stress = np.array([
        sigma_y,
        sigma_y*1.05,
        sigma_y*1.12,
        sigma_y*1.20,
        sigma_y*1.30
    ])

    stress = np.clip(stress, 300e6, 2.0e9)
    stress = np.maximum.accumulate(stress)

    elastic_strain = stress/E
    plastic_strain = np.maximum(total_strain - elastic_strain,0)

    plastic_strain[0] = max(plastic_strain[0],0.0015)
    plastic_strain = np.maximum.accumulate(plastic_strain)

    for k in range(1,len(plastic_strain)):
        if plastic_strain[k] - plastic_strain[k-1] < 1e-4:
            plastic_strain[k] = plastic_strain[k-1] + 1e-4

    comp = ", ".join([f"{e}:{row[e]:.2f}" for e in elements if row.get(e,0)>0.01])

    apdl = f"""
! MATERIAL: {name}
! Composition: {comp}
! HV_ML: {row['HV_ML']:.2f}
! HV_Physics: {row['HV_Physics']:.2f}

MP,EX,{mat_id},{E:.3e}
MP,PRXY,{mat_id},{nu:.4f}

TB,MISO,{mat_id},,5
TBOPT,MISO,0
"""

    for j,(eps,sig) in enumerate(zip(plastic_strain,stress)):
        apdl += f"TBDATA,{j+1},{eps:.6f},{sig:.3e}\n"

    apdl += f"\nMP,DENS,{mat_id},{rho:.1f}\n"

    with open(OUTPUT_DIR / f"{name}.txt","w") as f:
        f.write(apdl)

    summary.append({
        "Alloy_ID":name,
        "HV_ML":row["HV_ML"],
        "HV_Physics":row["HV_Physics"],
        "E_GPa":E/1e9,
        "nu":nu,
        "Density":rho,
        "Yield_MPa":sigma_y/1e6
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUTPUT_DIR/"material_summary.csv",index=False)

print("\nSTEP 7 COMPLETE")
print("Output:",OUTPUT_DIR)

print("\nFINAL RANKED ALLOYS:")
print(summary_df[["Alloy_ID","HV_ML","HV_Physics","Yield_MPa"]])
