# %%
from deeperwin.run_tools.geometry_database import load_energies, load_geometries, load_datasets
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from paper_plots.paper3_universal_wf.plot_utils import plot_df
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid")
save_figs = True

E_MRCI_THERMODYNAMIC_LIMIT_AFQMC = -0.56569

all_geometries = load_geometries()
hashes_in_order = load_datasets()["HChain_equidist_2-28_1.80_14geoms"].get_hashes()
experiments = [
    "HF_HChains_6-31G**",
    "Gao_etal_2023_HChain_extensivity_Moon",
    "Gao_etal_2023_HChain_extensivity_FermiNet",
    "2023-03-09_gao_reuse_Hchains_from64k",
]


def get_n_atoms(geom_hash):
    return hashes_in_order.index(geom_hash) * 2 + 2


df_full = load_energies()
df = df_full[df_full.geom.isin(hashes_in_order) & df_full.experiment.isin(experiments)]
df["n_atoms"] = df.geom.apply(get_n_atoms)
df.loc[df.source == "dpe", "experiment"] = df.experiment + "_" + df.epoch.astype(float).apply("{:.0f}".format)
df["epoch"] = df.epoch.fillna(0)

color_dict = dict()
lw_dict = dict()
label_dict = dict()
marker_dict = dict()
for e in df.experiment.unique():
    if "Moon" in e:
        color_dict[e] = "C1"
        label_dict[e] = "GLOBE+Moon, zero-shot"
        marker_dict[e] = "^"
    elif "FermiNet" in e:
        color_dict[e] = "C2"
        label_dict[e] = "GLOBE+FermiNet, zero-shot"
        marker_dict[e] = "v"
    elif e.startswith("HF_"):
        color_dict[e] = "dimgray"
        label_dict[e] = "Hartree-Fock"
        marker_dict[e] = "s"
    else:
        color_dict[e] = "C0"
        label_dict[e] = "Ours"
        n_finetune = int(e.split("_")[-1])
        if n_finetune == 0:
            label_dict[e] += ", zero-shot"
        else:
            label_dict[e] += f", {n_finetune} steps"
        lw_dict[e] = 2.5
        marker_dict[e] = "o"
ls_dict = {"2023-03-09_gao_reuse_Hchains_from64k_0": "-", "2023-03-09_gao_reuse_Hchains_from64k_500": ":"}
df["E_per_atom"] = df.E / df.n_atoms
df = df.sort_values(["experiment", "n_atoms"])

plt.close("all")
fig = plt.figure(figsize=(13, 7), tight_layout=True)
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.8])
ax_n_atoms = fig.add_subplot(gs[0, :])
ax_H10 = fig.add_subplot(gs[1, 0])
ax_TDL = fig.add_subplot(gs[1, 1])


plot_df(
    ax_n_atoms,
    df,
    x="n_atoms",
    y="E_per_atom",
    experiment="experiment",
    color_dict=color_dict,
    lw_dict=lw_dict,
    ls_dict=ls_dict,
    label_dict=label_dict,
    marker_dict=marker_dict,
    experiment_order=experiments[:3]
    + ["2023-03-09_gao_reuse_Hchains_from64k_0", "2023-03-09_gao_reuse_Hchains_from64k_500"],
    zorder=10,
)
ax_n_atoms.axhline(
    E_MRCI_THERMODYNAMIC_LIMIT_AFQMC,
    color="brown",
    label="Motta et al., AFQMC, n$_{\\mathrm{atoms}}$$\\rightarrow \\infty$",
    ls="-",
)

ax_n_atoms.set_xlabel("nr of H-atoms")
ax_n_atoms.set_ylabel("energy per atom / Ha")
ax_n_atoms.set_title("H$_6$, H$_{10}$ $\\rightarrow$ H-chains", fontsize=16)
ax_n_atoms.set_xticks(np.arange(2, 30, 2))
ax_n_atoms.legend(loc="upper left", framealpha=1.0)
for n in [6, 10]:
    ax_n_atoms.axvline(n, label="Shared pre-training", color="k", ls="--", zorder=-1)


df_zeroshot = df.query("n_atoms == 12 and epoch==0")[["experiment", "E_per_atom"]]
df_zeroshot["error"] = (df_zeroshot["E_per_atom"] - df.query("n_atoms==8")["E_per_atom"].min()) * 1000
print(df_zeroshot)


## Comparison to Motta et al.
def extrapolate_to_tdl(n, E_per_atom, n_min=4, n_max=np.inf, extrapolation_degree=2):
    n = np.array(n)
    include = (n >= n_min) & (n <= n_max)
    E_per_atom = np.array(E_per_atom)[include]
    n = n[include]
    A = (1.0 / n)[:, None] ** np.arange(extrapolation_degree + 1)
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
df["R"] = df.geom.apply(lambda x: all_geoms[x].R[1, 0] - all_geoms[x].R[0, 0]).round(4)
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

methods_tdl = df[df.n_atoms == -1].sort_values("error_per_atom").method.unique()
methods_H10 = df[df.n_atoms == 10].sort_values("error_per_atom").method.unique()
methods = [m for m in methods_tdl if m in methods_H10]
df_H10 = df.query("(R == 1.8) and (n_atoms == 10) and (method in @methods)")
methods = df_H10.sort_values("error_per_atom").method.unique()
methods = [m for m in methods if m not in ["dpe_0", "UHF", "dpe_1000", "dpe_2000", "AFQMC"]]
method_labels = []
for m in methods:
    if m.startswith("dpe"):
        method_labels.append(f"Ours, {m.split('_')[1]} steps")
    else:
        method_labels.append(m)

for ax, n_atoms in zip([ax_H10, ax_TDL], [10, -1]):
    df_filt = df.query("(R == 1.8) and (n_atoms == @n_atoms) and (method in @methods)").groupby("method").mean()
    x = np.arange(len(methods))
    for x_, method in enumerate(methods):
        label = method
        ax.barh([x_], [df_filt.loc[method].error_per_atom], height=0.8, color="C0" if "dpe" in method else "dimgray")
    ax.set_yticks(x)
    ax.set_yticklabels(method_labels)
    ax.set_xlabel("$(E - E_\\mathrm{AFQMC})\,/\,\\mathrm{atom}\,/\,\\mathrm{mHa}$")
    ax.set_xlim(0, 3.7)
    ax.grid(False, axis="y")
    ax.axvline(1.6, color="k", linestyle="--", linewidth=1, zorder=-1)
    ax.text(1.65, 0, "chem.\nacc.", ha="left", va="center", fontsize=10)
    if n_atoms == 10:
        ax.set_title(r"$R=1.8,\, N_\mathrm{atoms} = 10$", fontsize=16)
    else:
        ax.set_title(r"$R=1.8,\, N_\mathrm{atoms} \rightarrow \infty$", fontsize=16)
    fig.tight_layout()

for ax, label in zip([ax_n_atoms, ax_H10, ax_TDL], "abc"):
    ax.text(0.0, 1.01, f"{label}", transform=ax.transAxes, weight="bold", ha="left", va="bottom")
fig.tight_layout()
if save_figs:
    fig.savefig(
        "/home/mscherbela/ucloud/results/03_paper_unversal_wavefuncion/figures/HChains_extensivity_and_MottaEtAl.png",
        dpi=400,
        bbox_inches="tight",
    )
    fig.savefig(
        "/home/mscherbela/ucloud/results/03_paper_unversal_wavefuncion/figures/HChains_extensivity_and_MottaEtAl.pdf",
        bbox_inches="tight",
    )


# %%
