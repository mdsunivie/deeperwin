#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from deeperwin.run_tools.geometry_database import load_geometries
import seaborn as sns

df_ref = pd.read_csv("/home/mscherbela/runs/references/Motta_et_al_metal_insulator_transition_N40_z_TABC.csv", sep=';')
df_ref["n_atoms"] = 40
df_ref["R"] = 1.8

all_geoms = load_geometries()
fnames = [
    "/home/mscherbela/runs/HChain_evals/HChains12_20.csv",
    "/home/mscherbela/runs/HChain_evals/HChains40.csv",
]
df = pd.concat([pd.read_csv(fname) for fname in fnames])
columns = dict(loc_abs_0="z", geom="geom", weight="weight", epoch="epochs", loc_abs_0_sigma_corr="z_sigma")
df = df[list(columns.keys())].rename(columns=columns)
df.loc[df.epochs == 0, "epochs"] = 4_000
df = df[df.epochs.isin([200_000, 20_000])]
df["is_reuse"] = False
df["model"] = "moon"
df["E"] = np.nan
df["E_weighted"] = np.nan
df["z_weighted"] = df.z * df.weight
df["n_atoms"] = df.geom.apply(lambda g: all_geoms[g].n_atoms * all_geoms[g].periodic.supercell[0])
df["R"] = df.geom.apply(lambda g: all_geoms[g].R[1][0] - all_geoms[g].R[0][0])
df["k"] = df.geom.apply(lambda g: all_geoms[g].periodic.k_twist[0])
df = df.sort_values(["n_atoms", "R", "k"])
df = df[df.n_atoms >= 20]

df_tabc = df.groupby(["model", "is_reuse", "n_atoms", "R", "epochs"]).agg(z_weighted=("z_weighted", "sum"), weight=("weight", "sum")).reset_index()
df_tabc["z TABC"] = df_tabc.z_weighted / df_tabc.weight

pivot = df[df.n_atoms == 20].pivot_table("z", index="k", columns="R", aggfunc="mean")

plt.close("all")
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
sns.lineplot(data=df, x="k", y="z", hue="R", ax=axes[0][0], palette="tab10", style="n_atoms")
sns.lineplot(data=df_tabc, x="R", y="z TABC", hue="n_atoms", ax=axes[1][0], palette="tab10")
sns.lineplot(data=df[df.k==0], x="R", y="z", hue="n_atoms", ax=axes[1][1], palette="tab10")
axes[0][1].plot(df_ref.k, df_ref.z, label="AFQMC (N=40)", marker="o")
axes[0][1].plot(df[(df.R==1.8) & (df.n_atoms==40)].k, df[(df.R==1.8) & (df.n_atoms==40)].z, label="DPE (N=40)", marker="o")
axes[0][1].plot(df[(df.R==1.8) & (df.n_atoms==20)].k, df[(df.R==1.8) & (df.n_atoms==20)].z, label="DPE (N=20)", marker="o")
axes[0][1].plot([0], [0.5559], marker='X', color='red', label="FermiNet (N=40, R=1.8, 50k)")
axes[0][1].set_xlabel("k")
axes[0][1].set_ylabel("z")
axes[0][1].set_title("N=40, R=1.8")
axes[0][1].grid(alpha=0.5)
axes[0][1].set_ylim([-0.05, None])
axes[0][1].legend()
# axes[0][1].pcolormesh(pivot.columns.values, pivot.index.values, pivot.values, cmap="viridis", shading="nearest")
axes[0][0].set_title("k-dependence")
axes[0][0].plot([0], [0.2854], marker='X', color='C1', label="FermiNet (N=40, R=1.4, 50k)")
axes[0][0].plot([0], [0.5559], marker='X', color='C3', label="FermiNet (N=40, R=1.8, 50k)")
axes[1][0].set_title("R-dependence (TABC)")
axes[1][1].set_title("R-dependence (Gamma)")

#%%
