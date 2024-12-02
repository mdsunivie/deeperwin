# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from deeperwin.run_tools.geometry_database import load_energies, load_geometries, load_datasets, get_all_geometries
import matplotlib.gridspec as gridspec
from deeperwin.utils.utils import KCAL_PER_MOL_IN_HARTREE

sns.set_theme(style="whitegrid")

all_geoms = load_geometries()
df_all = load_energies()
df_ref = df_all.query("source == 'orca_CCSD(T)_CBS_234' or experiment == 'Gerard_et_al_2022_N2_bond_breaking'")
df_ref = df_ref.query("molecule != 'N2' or experiment == 'Gerard_et_al_2022_N2_bond_breaking'")
df_ref = df_ref[["geom", "E"]]
# df_ref = df_all.query("source == 'orca_CCSD(T)_ccpCVQZ'")[["geom", "E"]]
# df_ref = df_all.query("source == 'orca_RHF_ccpCVQZ'")[["geom", "E"]]
df_ref.rename(columns={"E": "E_ref"}, inplace=True)
experiment_name = "2023-05-01_699torsion_nc_by_std_256k_failure_cases"
n_pretrain_variational = 256_000
datasets = [
    ("N2_stretching_GerardEtAl2022", "$N_2$ dissociation", "bond length / bohr"),
    ("C3H4_dist_rotated_20geoms", "Propadiene global rotation", "rotation angle / deg"),
    # ("Cylobutadiene_transition_6geoms", "Cyclobutadiene transition", "Reaction coordinate / a.u."),
    ("C3H4_rotated_torsion_angle_90degrees", "Propadiene torsion", "torsion angle / deg"),
]
datasets_ids = [d[0] for d in datasets]

all_datasets = load_datasets()
all_geometries = load_geometries()
geom_hashes_per_dataset = {d: all_datasets[d].get_hashes() for d in datasets_ids}
geom_hashes = [g.hash for g in get_all_geometries(datasets_ids)]

df = df_all.query(
    "geom in @geom_hashes and experiment == @experiment_name and n_pretrain_variational == @n_pretrain_variational"
)
df = pd.merge(df, df_ref, on="geom", how="left")
df["error"] = (df.E - df.E_ref) * 1000
df["xaxis_Cylobutadiene_transition_6geoms"] = np.nan
df["xaxis_N2_stretching_GerardEtAl2022"] = np.nan
df["xaxis_C3H4_rotated_torsion_angle_90degrees"] = np.nan
df["xaxis_C3H4_dist_rotated_20geoms"] = np.nan

for i, r in df.iterrows():
    for d in datasets_ids:
        if r["geom"] in geom_hashes_per_dataset[d]:
            if d == "Cylobutadiene_transition_6geoms":
                df.loc[i, f"xaxis_{d}"] = 1 - float(r["geom_comment"].split("_")[-1])
            if d == "N2_stretching_GerardEtAl2022":
                g = all_geometries[r["geom"]]
                df.loc[i, f"xaxis_{d}"] = g.R[1, 0] - g.R[0, 0]
            if d == "C3H4_rotated_torsion_angle_90degrees":
                if "C3H4_rotated_torsion" in r["geom_comment"]:
                    df.loc[i, f"xaxis_{d}"] = float(r.geom_comment.split("_d")[1])
                else:
                    df.loc[i, f"xaxis_{d}"] = 0.0
            if d == "C3H4_dist_rotated_20geoms":
                if "C3H4_dist_rotated" in r["geom_comment"]:
                    df.loc[i, f"xaxis_{d}"] = float(r.geom_comment.replace("C3H4_dist_rotated_", "").replace("deg", ""))
                else:
                    df.loc[i, f"xaxis_{d}"] = 0.0

plt.close("all")
fig = plt.figure(figsize=(12, 7))
gs = gridspec.GridSpec(2, len(datasets), fig, height_ratios=[1, 1])
ax_bicyclo = plt.subplot(gs[0, :])
axes = [plt.subplot(gs[1, i]) for i in range(gs.ncols)]

colors = {0: "lightsalmon", 1000: "coral", 4000: "red"}
i = 0
for (d, title, xlabel), ax in zip(datasets, axes):
    xaxis = f"xaxis_{d}"
    # Referene data
    pivot_ref = df.query(f"{xaxis}.notna()").groupby(xaxis).mean().reset_index()
    if d == "N2_stretching_GerardEtAl2022":
        ref_label = "Gerard et al. (2022)"
        E0_ref = pivot_ref.E_ref.iloc[1]
    else:
        ref_label = "CCSD(T)/CBS"
        E0_ref = pivot_ref.E_ref.iloc[0]

    # Our data
    for epoch, marker in zip([0, 4000], ["o", "s"]):
        df_filt = df.query(f"epoch == @epoch and {xaxis}.notna()")
        pivot = df_filt.groupby(xaxis).mean().reset_index()
        if len(pivot) == 0:
            continue
        if "N2" in d:
            E0 = pivot.E.iloc[1]
        else:
            E0 = pivot.E.iloc[0]
        if epoch == 0:
            label = "Our work (zero-shot)"
        else:
            label = f"Our work ({epoch/1000:.0f}k fine-tuning)"
        ax.plot(
            pivot[xaxis],
            (pivot.E - E0) * 1000,
            # yerr=pivot.E_sigma*1000,
            label=label,
            color=colors[epoch],
            marker=marker,
            # capsize=3,
            zorder=2,
        )

    if d == "C3H4_dist_rotated_20geoms":
        ax.axhline(0, color="black", linestyle="-", label="exact", zorder=1)
        ax.set_ylim([-1.5, 10])
    else:
        ax.plot(
            pivot_ref[xaxis],
            (pivot_ref.E_ref - E0_ref) * 1000,
            color="black",
            linestyle="-",
            label=ref_label,
            marker="^",
            zorder=1,
        )
    if d == "Cylobutadiene_transition_6geoms":
        delta_E_ferminet = (154.676480107767 - 154.661385285574) * 1000
        ax.plot(
            [0, 1],
            [0, delta_E_ferminet],
            color="gray",
            linestyle="none",
            label="FermiNet 200k steps",
            marker="s",
            markersize=6,
        )

    img_fname = (
        f"/home/mscherbela/ucloud/results/04_paper_better_universal_wf/figures/renders/relative_energies_mol{i+1}.png"
    )
    if d == "N2_stretching_GerardEtAl2022":
        img_pos = [0.6, -0.05, 0.35, 0.5]
    if d == "C3H4_dist_rotated_20geoms":
        img_pos = [0.6, 0.3, 0.4, 0.5]
    if d == "C3H4_rotated_torsion_angle_90degrees":
        img_pos = [0.5, -0.1, 0.5, 0.5]

    image_ax = ax.inset_axes(img_pos)
    image_ax.imshow(plt.imread(img_fname))
    image_ax.axis("off")

    i += 1
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("$E - E_0$ / mHa")
    # handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc="upper left")
    ax.ticklabel_format(useOffset=False)

fig.tight_layout()


# %%
experiments_Bicyclobutane = [
    # "2023-05-01_699torsion_nc_by_std_174k_reuseshared_Bicyclobutane"
    "2023-05-01_699torsion_nc_by_std_256k_2keval_reuseshared_Bicyclobutane"
]
bicyclo_geoms = all_datasets["Kinal_Piecuch_Bicyclobutane"].get_geometries()
bicyclo_hashes = [g.hash for g in bicyclo_geoms]
geom0 = "5b132a54986fbc4eff405d51c74a2cb2"

# Data from Better, Faster Fermionic Neural Networks
structures = ["con_TS", "dis_TS", "g-but", "gt-TS", "t-but"]
ferminet_hartree = np.array([40.2, 57.7, -25.3, -22.5, -28.4]) * KCAL_PER_MOL_IN_HARTREE
ferminet_hartree_10k = (
    np.array([40.4, 58.6, -25.2, -22.2, -27.9]) * KCAL_PER_MOL_IN_HARTREE
    + np.array([-0.28, -3.92, -2.95, -4.75, -1.93]) * KCAL_PER_MOL_IN_HARTREE
)
dmc_hartree = np.array([40.4, 58.6, -25.2, -22.2, -27.9]) * KCAL_PER_MOL_IN_HARTREE
ccsd_t_hartree = np.array([40.4, 21.8, -25.1, -22.3, -28.0]) * KCAL_PER_MOL_IN_HARTREE
ZVPE_hartree = (np.array([53.35, 52.22, 54.93, 54.56, 54.96]) - 56.01) * KCAL_PER_MOL_IN_HARTREE
ZVPE_correction = {s: zvpe for s, zvpe in zip(structures, ZVPE_hartree)}
ZVPE_correction[geom0] = 0
data_bicyclo = []
for structure, ferminet_10k, ferminet, dmc, ccsd_t in zip(
    structures, ferminet_hartree_10k, ferminet_hartree, dmc_hartree, ccsd_t_hartree
):
    data_bicyclo.append(dict(structure=structure, source="CCSD(T)_Kinal", error=ccsd_t * 1000))
    data_bicyclo.append(dict(structure=structure, source="DMC_Kinal", error=dmc * 1000))
    data_bicyclo.append(dict(structure=structure, source="FermiNet", error=ferminet * 1000))
    data_bicyclo.append(dict(structure=structure, source="FermiNet_10k", error=ferminet_10k * 1000))

# DeepErwin data
df_bicyclo = df_all.query("geom in @bicyclo_hashes and experiment in @experiments_Bicyclobutane")
df_bicyclo["structure"] = df_bicyclo.geom.apply(lambda g: all_geoms[g].comment.split("Bicyclobutane_")[1])
# Compute energy differences relative to geom0
for experiment in df_bicyclo.experiment.unique():
    for epoch in df_bicyclo.epoch.unique():
        filter = (df_bicyclo.experiment == experiment) & (df_bicyclo.epoch == epoch)
        if filter.sum() == 0:
            continue
        E_0 = df_bicyclo[filter & (df_bicyclo.geom == geom0)].E.mean()
        df_bicyclo.loc[filter, "error"] = (df_bicyclo[filter].E - E_0) * 1000
        df_bicyclo.loc[filter, "error"] += df_bicyclo[filter].structure.map(ZVPE_correction) * 1000


df_bicyclo = pd.concat([df_bicyclo, pd.DataFrame(data_bicyclo)], ignore_index=True)
df_bicyclo["method"] = df_bicyclo.source + df_bicyclo.epoch.apply(lambda x: f"_{x:.0f}" if np.isfinite(x) else "")
metadata_bicyclo = [
    ("dpe_0", "Our work: zero-shot", colors[0]),
    # ("dpe_1000", "Our work: 1k steps",, colors[1000])),
    ("dpe_4000", "Our work: 700 steps per geom.", colors[4000]),
    ("DMC_Kinal", "DMC (Kinal et al.)", "k"),
    ("FermiNet", "FermiNet: 200k steps per geom.", "navy"),
    ("FermiNet_10k", "FermiNet: 10k steps per geom.", "C0"),
    ("CCSD(T)_Kinal", "CCSD(T) (Kinal et al.)", "dimgray"),
]
structures_in_order = ["dis_TS", "con_TS", "gt-TS", "g-but", "t-but"]

sns.barplot(
    df_bicyclo.query("structure != 'bicbut'"),
    x="structure",
    hue="method",
    y="error",
    palette={k: c for k, l, c in metadata_bicyclo},
    hue_order=[k for k, l, c in metadata_bicyclo],
    order=structures_in_order,
    ax=ax_bicyclo,
)
handles, labels = ax_bicyclo.get_legend_handles_labels()
ax_bicyclo.legend_.remove()
ax_bicyclo.legend(handles, [l for k, l, c in metadata_bicyclo], ncols=2, loc="center right")
ax_bicyclo.set_xlabel(None)
ax_bicyclo.set_ylabel(r"$E - E_\mathrm{Bicyclobutane}$ / mHa")
ax_bicyclo.set_title("Conformers of bicyclobutane")
ax_bicyclo.axhline(0, color="k")
ax_bicyclo.set_ylim(-70, None)
ax_bicyclo.minorticks_on()
ax_bicyclo.grid(which="minor", axis="y", linestyle=":", linewidth=0.5, color="lightgrey")

# Load images and display them left to right on top of ax_bicyclo, with no visible axes
for i, s in enumerate(structures_in_order):
    image_path = f"/home/mscherbela/ucloud/results/04_paper_better_universal_wf/figures/renders/{s}.png"
    # image_path = f"/Users/leongerard/ucloud/Shared/results/04_paper_better_universal_wf/figures/renders/{s}.png"
    image = plt.imread(image_path)
    xpos_min = 0.1
    xpos_max = 0.92
    ypos = 0.65
    image_ax_width = 0.5
    image_ax_height = image_ax_width * 860 / 1133
    xpos = xpos_min + i * (xpos_max - xpos_min) / (len(structures_in_order) - 1) - image_ax_width / 2
    image_ax = ax_bicyclo.inset_axes([xpos, ypos, image_ax_width, image_ax_height])
    image_ax.imshow(image)
    image_ax.axis("off")

for ax, letter in zip([ax_bicyclo, *axes], "abcd"):
    ax.text(x=0, y=1.01, s=letter, transform=ax.transAxes, fontsize=14, fontweight="bold", ha="left", va="bottom")

fig.tight_layout(h_pad=2)
fname = "/home/mscherbela/ucloud/results/04_paper_better_universal_wf/figures/fig5_relative_energies.png"
# fname = "/Users/leongerard/ucloud/Shared/results/04_paper_better_universal_wf/figures/fig3_vs_indep_and_bicyclo.png"
fig.savefig(fname, dpi=300, bbox_inches="tight")
fig.savefig(fname.replace(".png", ".pdf"), bbox_inches="tight")

# %%
pivot_for_latex = df_bicyclo.query("structure != 'bicbut' and method != 'dpe_1000'").pivot_table(
    index="structure", columns="method", values="error", aggfunc="mean"
)
latex = pivot_for_latex.to_latex(float_format="%.1f", na_rep="---")
print(latex)

print(pivot_for_latex["dpe_0"] - pivot_for_latex["DMC_Kinal"])
print(pivot_for_latex["dpe_4000"] - pivot_for_latex["DMC_Kinal"])
print(pivot_for_latex["FermiNet_10k"] - pivot_for_latex["DMC_Kinal"])


# %%
