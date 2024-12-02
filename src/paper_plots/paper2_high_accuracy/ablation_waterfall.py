import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pandas as pd
from paper_plots.paper2_high_accuracy.waterfall import plot_waterfall
import re

molecules = ["Ethene", "N2_bond_breaking", "K"]
metric = "error_eval_50k"
titles = dict(Ethene="Ethene", N2_bond_breaking="$N_2$, $d=$4.0 bohr", K="K atom")

settings = [
    "01_fermi_iso",
    "02_fermi_iso_fulldet",
    "03_fermi_iso_fulldet_hp",
    "04_fermi_iso_fulldet_hp_emb",
    "05_dpe11",
    "06_dpe11_init",
]
labels_delta = [
    "Dense\nDeterminant",
    "Hyper-\nparameters",
    "SchNet-like\nembedding",
    "Local input-\nfeatures",
    "Envelope\nInitialization",
]
E_ref = dict(
    Ethene=-78.588800,  # CCSD(T)
    N2_bond_breaking=-109.202042,  # experiment
    P=-341.259000,  # experiment
    K=-599.9195443359196,
)
best_variational = dict(
    Ethene=(-78.5844, "[Pfau 2020]\n200k epochs"),  # FermiNet block-det, 200k epochs'
    # N2_bond_breaking=(-109.194, '[Lin 2021], 200k epochs'), #  FermiNet\n1 full-det, 200k epochs,
    N2_bond_breaking=(-109.198817224834, "[Ren 2022]\n400k epochs"),
    P=(-341.2578, "[Spencer 2021]"),
)  # FermiNet 512, 300k

color_other_improvements = get_cmap("inferno")(0.35)
color_our_improvements = get_cmap("inferno")(0.65)
color_summary = [get_cmap("inferno")(0.05), get_cmap("viridis")(0.85)]

colors_delta = [
    color_other_improvements,
    color_our_improvements,
    color_our_improvements,
    color_our_improvements,
    color_our_improvements,
    color_our_improvements,
]
ablation_energies_raw = pd.read_csv("/home/mscherbela/tmp/ablation_v2.csv", sep=";")
ablation_energies_raw.rename(columns={"category": "settings"}, inplace=True)
ablation_energies_raw = ablation_energies_raw[ablation_energies_raw.settings.isin(settings)]
ablation_energies_raw["E_ref"] = ablation_energies_raw.molecule.map(E_ref)
for c in list(ablation_energies_raw):
    if re.match(r"E_(eval|smooth).*\d*k", c):
        ablation_energies_raw["error" + c[1:]] = (ablation_energies_raw[c] - ablation_energies_raw["E_ref"]) * 1e3


ablation_energies = (
    ablation_energies_raw.groupby(["molecule", "settings"])
    .agg(
        error_mean=(metric, "mean"),
        error_std=(metric, "std"),
        error_min=(metric, "min"),
        error_max=(metric, "max"),
        count=(metric, "count"),
    )
    .reset_index()
)
ablation_energies.sort_values(["molecule", "settings"], inplace=True)
molecules_df = pd.DataFrame([dict(molecule=m, settings=s) for m in molecules for s in settings])
ablation_energies = pd.merge(molecules_df, ablation_energies, "left", on=["molecule", "settings"])

plt.close("all")
fig, axes = plt.subplots(1, len(molecules), dpi=100, figsize=(9.6, 4.4), sharey=True)
for ind_mol, (ax, molecule) in enumerate(zip(axes, molecules)):
    ablation_mol = ablation_energies[ablation_energies.molecule == molecule]
    errors = list(np.array(ablation_mol["error_mean"]))
    uncertainties = list(ablation_mol["error_std"])
    plot_waterfall(
        errors,
        label_start="FermiNet\nisotropic",
        labels_delta=labels_delta,
        label_end="This work",
        # summary_points=[(1, 'All published\nimprovements')], #(4, 'Improved\narchitecture')
        color_delta=colors_delta,
        color_summary=color_summary,
        y_err=uncertainties,
        ylim=[min(np.nanmin(errors) * 1.1 - 0.5, 0), min(max(np.nanmax(errors) * 1.1, 0.2), 80)],
        ax=ax,
        label_rotation=0,
        value_position="center",
        horizontal=True,
        textcolors=["w"] * 2 + ["k"] * (len(settings) - 1),
    )

    # ax.plot([np.nan], [np.nan], color='k', ls='--', label=f"Best published\nvariational energy" )
    if molecule in best_variational:
        ax.axvline(
            1e3 * (best_variational[molecule][0] - E_ref[molecule]),
            color="k",
            ls="--",
            label=f"{best_variational[molecule][1]}",
        )

    ax.set_xlabel("$E - E_\\mathrm{ref}$ / mHa", fontsize=12)
    ax.axvline(0, color="k")
    ax.grid(axis="x", alpha=0.5)
    ax.set_title(titles.get(molecule, molecule))
    if ind_mol == 2:
        ax.bar([np.nan], [np.nan], color=color_summary[0], label="FermiNet")
        ax.bar([np.nan], [np.nan], color=color_other_improvements, label="Improvements by\nearlier work")
        ax.bar([np.nan], [np.nan], color=color_our_improvements, label="Our improvements")
        ax.bar([np.nan], [np.nan], color=color_summary[1], label="This work")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

fig.tight_layout()
fig_fname = f"/home/mscherbela/ucloud/results/02_paper_figures_high_acc/waterfalls/{'_'.join(molecules)}_{metric}.png"
fig.savefig(fig_fname, bbox_inches="tight")
fig.savefig(fig_fname.replace(".png", ".pdf"), bbox_inches="tight")
