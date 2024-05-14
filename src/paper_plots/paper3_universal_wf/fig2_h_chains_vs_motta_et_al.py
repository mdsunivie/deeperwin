#%%
from deeperwin.run_tools.geometry_database import load_energies, load_geometries, load_datasets
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

def extrapolate_to_tdl(n, E_per_atom, n_min=4, n_max=np.inf, extrapolation_degree=2):
    n = np.array(n)
    include = (n >= n_min) & (n <= n_max)
    E_per_atom = np.array(E_per_atom)[include]
    n = n[include]
    A = (1.0 / n)[:, None] ** np.arange(extrapolation_degree+1)
    coeffs = np.linalg.lstsq(A, E_per_atom, rcond=None)[0]
    return coeffs[0], coeffs


fname_ref = "/home/mscherbela/runs/references/Motta_et_al_2017_Hydrogen_chains.csv"
df_motta = pd.read_csv(fname_ref, sep=";")
df_motta["n_atoms"] = df_motta.n_atoms.str.replace("TDL", "-1").astype(int)
df_ref = df_motta.query("method == 'AFQMC'").rename(columns={"E_per_atom": "E_ref_per_atom"}).drop(columns=["method"])

df_all = load_energies()
all_geoms = load_geometries()

df = df_all[df_all.experiment == "2023-03-09_gao_reuse_Hchains_from64k"]
df["n_atoms"] = df.geom.apply(lambda x: all_geoms[x].n_atoms)
df["R"] = df.geom.apply(lambda x: all_geoms[x].R[1,0] - all_geoms[x].R[0,0]).round(4)
df["E_per_atom"] = df.E / df.n_atoms
df["method"] = df["source"] + "_" + df["epoch"].astype(float).apply("{:.0f}".format)

dpe_tdl_data = []
for epoch in [0, 500, 1000, 2000, 4000]:
    df_tdl = df[(df.method.str.contains("dpe")) & (df.R == 1.8) & (df.epoch == epoch)]
    E_tdl, _ = extrapolate_to_tdl(df_tdl.n_atoms.values, df_tdl.E_per_atom.values)
    dpe_tdl_data.append(dict(method=f"dpe_{epoch}", epoch=epoch, E_per_atom=E_tdl, R=1.8, n_atoms=-1))
df = pd.concat([df, pd.DataFrame(dpe_tdl_data), df_motta], ignore_index=True)

df = pd.merge(df, df_ref, on=["n_atoms", "R"], how="left")
df["error_per_atom"] = (df.E_per_atom - df.E_ref_per_atom) * 1000

#%%
plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

methods_tdl = df[df.n_atoms == -1].sort_values("error_per_atom").method.unique()
methods_H10 = df[df.n_atoms == 10].sort_values("error_per_atom").method.unique()
methods = [m for m in methods_tdl if m in methods_H10]
df_H10 = df.query("(R == 1.8) and (n_atoms == 10) and (method in @methods)")
methods = df_H10.sort_values("error_per_atom").method.unique()
methods = [m for m in methods if m not in ["dpe_0", "UHF", "dpe_1000", "dpe_2000", "AFQMC"]]

for ax, n_atoms in zip(axes, [10, -1]):
    df_filt = df.query("(R == 1.8) and (n_atoms == @n_atoms) and (method in @methods)").groupby("method").mean()
    x = np.arange(len(methods))
    for x_, method in enumerate(methods):
        if method not in df_filt.index:
            continue
        ax.barh([x_], [df_filt.loc[method].error_per_atom], height=0.8, color='C1' if 'dpe' in method else 'C0')
    ax.set_yticks(x)
    ax.set_yticklabels(methods)
    ax.set_xlabel("$(E - E_\\mathrm{AFQMC}) / \\mathrm{atom}\, [mHa]$")
    ax.set_xlim(0, 4)
    if n_atoms == 10:
        ax.set_title(r"$R=1.8,\, N_\mathrm{atoms} = 10$")
    else:
        ax.set_title(r"$R=1.8,\, N_\mathrm{atoms} \rightarrow \infty$")
    fig.tight_layout()

#%%

plt.close("all")
fig, (ax_n_atoms, ax_extrapolation, ax_finetuning) = plt.subplots(1, 3, figsize=(14, 6))

E_tdl_ref = df_ref.query("n_atoms == -1 and R == 1.8").E_ref_per_atom.values[0]

epochs = [500, 1000, 2000, 4000]

E_tdl = []
for i, epoch in enumerate(epochs):
    df_dpe = df[(df.method.str.contains("dpe")) & (df.R == 1.8) & (df.epoch == epoch) & (df.n_atoms > 0)]
    df_dpe = df_dpe.sort_values("n_atoms")

    E, coeffs = extrapolate_to_tdl(df_dpe.n_atoms.values, df_dpe.E_per_atom.values)
    E_tdl.append(E)
    print(f"E_dpe ({epoch:4d}) = {E_tdl[-1]:.5f} mHa/atom, errror = {(E_tdl[-1] - E_tdl_ref)*1000:.1f} mHa/atom")
    ax_n_atoms.plot(df_dpe.n_atoms, df_dpe.E_per_atom, color=f"C{i}", label=f"dpe ({epoch} fine-tuning steps)")
    ax_n_atoms.axhline(E_tdl[-1], color=f"C{i}")

    ax_extrapolation.plot(1 / df_dpe.n_atoms, df_dpe.E_per_atom, color=f"C{i}", linestyle="None", marker="o")
    x_fit = np.linspace(np.min(1 / df_dpe.n_atoms), np.max(1 / df_dpe.n_atoms), 100)
    y_fit = coeffs[0] + coeffs[1] * x_fit + coeffs[2] * x_fit**2
    ax_extrapolation.plot(x_fit, y_fit, color=f"C{i}", alpha=0.5)
    ax_extrapolation.plot([0], coeffs[0], color=f"C{i}", marker='s')
    ax_extrapolation.plot([0], [E_tdl_ref], color="k", marker='s')

ax_n_atoms.axhline(E_tdl_ref, color="k", linestyle="--", label="AFQMC")
ax_n_atoms.legend()
E_tdl = np.array(E_tdl)

ax_finetuning.plot(epochs, (E_tdl - E_tdl_ref) * 1000, marker="o")
ax_finetuning.set_xlabel("fine-tuning steps")
ax_finetuning.set_xlabel("$(E - E_\\mathrm{AFQMC}) / \\mathrm{atom}\, [mHa]$")

# %%
