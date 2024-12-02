# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from deeperwin.run_tools.geometry_database import load_energies, load_geometries, load_datasets, get_all_geometries

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
    # ("C3H4_dist_rotated_20geoms", "Propadiene global rotation", "rotation angle / deg"),
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
fig, axes = plt.subplots(2, 1, figsize=(5, 6))

colors = {0: "lightsalmon", 1000: "coral", 4000: "red"}
i = 0
for (d, title, xlabel), ax in zip(datasets, axes):
    xaxis = f"xaxis_{d}"
    # Referene data
    pivot_ref = df.query(f"{xaxis}.notna()").groupby(xaxis).mean().reset_index()
    if d == "N2_stretching_GerardEtAl2022":
        ref_label = "Gerard et al. (2022)"
        E0_ref = pivot_ref.E_ref.iloc[1]
        img_fname = (
            "/home/mscherbela/ucloud/results/04_paper_better_universal_wf/figures/renders/relative_energies_mol1.png"
        )
    else:
        ref_label = "CCSD(T)/CBS"
        E0_ref = pivot_ref.E_ref.iloc[0]
        img_fname = (
            "/home/mscherbela/ucloud/results/04_paper_better_universal_wf/figures/renders/relative_energies_mol3.png"
        )

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

fig.tight_layout(h_pad=2)
fname = "/home/mscherbela/ucloud/results/04_paper_better_universal_wf/figures/fig5_relative_energies_poster.png"
fig.savefig(fname, dpi=300, bbox_inches="tight")
fig.savefig(fname.replace(".png", ".pdf"), bbox_inches="tight")

# %%
