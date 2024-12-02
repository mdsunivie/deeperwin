# %%
from deeperwin.run_tools.geometry_database import load_energies, load_geometries, load_datasets
import seaborn as sns
import matplotlib.pyplot as plt
from paper_plots.paper3_universal_wf.plot_utils import plot_df
import numpy as np

sns.set_theme(style="whitegrid")
save_figs = True

E_MRCI_THERMODYNAMIC_LIMIT_MRCIQ = -0.56655

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
        label_dict[e] = "Our work"
        n_finetune = int(e.split("_")[-1])
        if n_finetune == 0:
            label_dict[e] += ", zero-shot"
        else:
            label_dict[e] += f", {n_finetune} fine-tuning steps"
        lw_dict[e] = 2.5
        marker_dict[e] = "o"
ls_dict = {"2023-03-09_gao_reuse_Hchains_from64k_0": "-", "2023-03-09_gao_reuse_Hchains_from64k_500": ":"}
df["E_per_atom"] = df.E / df.n_atoms
df = df.sort_values(["experiment", "n_atoms"])


for animation in [0, 1, 2]:
    plt.close("all")
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))

    if animation == 0:
        df_plot = df[df.source.isin(["HF"])]
    elif animation == 1:
        df_plot = df[df.source.isin(["HF", "Gao_etal_2023"])]
    elif animation == 2:
        df_plot = df

    plot_df(
        ax,
        df_plot,
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
    ax.axhline(
        E_MRCI_THERMODYNAMIC_LIMIT_MRCIQ,
        color="brown",
        label=f"MRCI+Q, n$_{{\\mathrm{{atoms}}}}$$\\rightarrow \\infty$: {E_MRCI_THERMODYNAMIC_LIMIT_MRCIQ:.3f} Ha",
        ls="-",
    )

    ax.set_xlabel("nr of H-atoms")
    ax.set_ylabel("energy per atom / Ha")
    ax.set_title("H$_6$, H$_{10}$ $\\rightarrow$ H-chains", fontsize=16)
    ax.set_xticks(np.arange(2, 30, 2))
    ax.legend(loc="upper left", framealpha=1.0)
    ax.set_ylim([-0.58, -0.42])
    for n in [6, 10]:
        ax.axvline(n, label="Shared pre-training", color="k", ls="--", zorder=-1)

    fig.tight_layout()
    if save_figs:
        fig.savefig(
            f"/home/mscherbela/Nextcloud/PhD/talks_and_conferences/2023_04_CMU/figures/HChains_extensivity_{animation}.png",
            dpi=400,
            bbox_inches="tight",
        )
# %%
