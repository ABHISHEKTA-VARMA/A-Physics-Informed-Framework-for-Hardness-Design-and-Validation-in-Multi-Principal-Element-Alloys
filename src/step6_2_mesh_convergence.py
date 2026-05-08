import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

folder = "./"

force_files = glob.glob(os.path.join(folder, "FORCE *.csv"))
defo_files = glob.glob(os.path.join(folder, "DEFORMATION *.csv"))

def mesh_value(f):
    return float(os.path.basename(f).split()[1].replace(".csv", ""))

force_map = {mesh_value(f): f for f in force_files}
defo_map  = {mesh_value(f): f for f in defo_files}

meshes = sorted(set(force_map) & set(defo_map))

data = []

for m in meshes:
    dfF = pd.read_csv(force_map[m])
    dfD = pd.read_csv(defo_map[m])

    dfF.columns = dfF.columns.str.strip()
    dfD.columns = dfD.columns.str.strip()

    P = np.abs(dfF["Force Reaction (Total) [N]"].values)
    h = np.abs(dfD["Deformation Probe (Y) [mm]"].values)

    i = np.argmax(P)

    Pmax = P[i]
    hmax = h[i]

    d = 6.5 * hmax
    HV = 186.3 / (d ** 2)

    data.append([m, Pmax, hmax, HV])

df = pd.DataFrame(data, columns=["Mesh", "Pmax", "hmax", "HV"])
df = df.sort_values("Mesh", ascending=True).reset_index(drop=True)

df.to_csv("HV_results_final.csv", index=False)

fine = df.head(3)

h1, h2, h3 = fine["Mesh"].values
f1, f2, f3 = fine["HV"].values

r = h2 / h1

e21 = f2 - f1
e32 = f3 - f2

p = abs(np.log(abs(e32 / e21)) / np.log(r))

GCI = 1.25 * abs((f1 - f2) / f1) / (r**p - 1) * 100

HV_ref = f1
low = HV_ref * (1 - GCI / 100)
high = HV_ref * (1 + GCI / 100)

plt.rcParams.update({"font.family": "serif", "font.size": 12})

# Convergence
plt.figure(figsize=(7, 5))
plt.plot(df["Mesh"], df["HV"], marker="o", linewidth=1.6, markersize=5)
plt.xlabel("Mesh size (mm)")
plt.ylabel("Hardness (HV)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Fig1_convergence.png", dpi=600)
plt.close()

# Log plot
plt.figure(figsize=(7, 5))
plt.plot(df["Mesh"], df["HV"], marker="o", linewidth=1.6, markersize=5)
plt.xscale("log")
ticks = df["Mesh"][::2]
plt.gca().set_xticks(ticks)
plt.gca().set_xticklabels([f"{t:.2f}" for t in ticks])
plt.xlabel("Mesh size (mm)")
plt.ylabel("Hardness (HV)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Fig2_log.png", dpi=600)
plt.close()

# GCI
plt.figure(figsize=(7, 5))
plt.plot(df["Mesh"], df["HV"], marker="o", linewidth=1.6, markersize=5, label="Simulation")
plt.axhline(HV_ref, linestyle="--", linewidth=1.4, label="Fine mesh")
mask = df["Mesh"] <= df["Mesh"].iloc[2]
plt.fill_between(df["Mesh"][mask], low, high, alpha=0.08, label="GCI band")
plt.xlabel("Mesh size (mm)")
plt.ylabel("Hardness (HV)")
plt.legend(loc="upper right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Fig3_GCI.png", dpi=600)
plt.close()

print(f"Order p = {p:.3f}")
print(f"GCI = {GCI:.2f}%")
