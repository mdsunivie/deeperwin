# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from deeperwin.run_tools.geometry_database import load_energies, load_geometries, load_datasets
from deeperwin.utils.plotting import set_ticks_with_SI_postfix

sns.set_theme(style="whitegrid")

all_geoms = load_geometries()
all_datasets = load_datasets()
df_all = load_energies()
df_ref = df_all.query("source == 'orca_CCSD(T)_CBS_234'")[["geom", "E"]]
df_ref.rename(columns={"E": "E_ref"}, inplace=True)

experiments = [
    ("2023-05-05_tao_phisnet_indep", "Our work (no pre-training)", "C1", "--", "o"),
    # ("2023-04-27_reuse_midimol_699torsions_158k", "Our work (pre-trained 174k)", "pink", "-", "o"),
    ("2023-05-01_699torsion_nc_by_std_256k", "Our work (pre-trained)", "C1", "-", "o"),
    # ("2023-05-01_699torsion_nc_by_std_480k", "Our work (pre-trained 480k)", "red", "-", "o"),
    ("2023-02-26_18x20_500karxiv_v10_500k", "Scherbela et al. (2023)", "C0", "-", "s"),
    ("2023-05-08_dpe4_regression", "Gerard et al. (2022)", "C4", "--", "^"),
    ("2023-05-08_psiformer_indep", "PsiFormer", "C2", "--", "v"),
    # ("HF_CBS_234", "HF CBS", "gray", ":"),
    # ("CCSD(T)_ccpCVDZ", "CCSD(T)-2Z", "cornflowerblue", ":"),
    # ("CCSD(T)_ccpCVTZ", "CCSD(T)-3Z", "royalblue", ":"),
    # ("CCSD(T)_ccpCVQZ", "CCSD(T)-4Z", "navy", ":"),
    ("HF_CBS_234", "HF-CBS", None, None, None),
    ("CCSD(T)_ccpCVDZ", "CC-2Z", None, None, None),
    ("CCSD(T)_ccpCVTZ", "CC-3Z", None, None, None),
    ("CCSD(T)_ccpCVQZ", "CC-4Z", None, None, None),
]
experiment_names = [e[0] for e in experiments]

df_all = df_all[df_all.experiment.isin(experiment_names)]
# df_all = df_all.query("(source != 'dpe') or (epoch == 0000)")

datasets = {
    2: "Test_Set_2heavy_atoms_4geoms_dist",
    3: "TinyMol_CNO_rot_dist_test_out_of_distribution_4geoms",
    4: "Kinal_Piecuch_Bicyclobutane_without_dis_TS",
    5: "Test_Set_5heavy_atoms_no_ring_4geoms_dist",
    6: "Test_Set_6heavy_atoms_no_ring_4geoms_dist",
    7: "Test_Set_7heavy_atoms_no_ring_4geoms_dist",
}
included_geoms = []
for dataset_name in datasets.values():
    included_geoms += all_datasets[dataset_name].get_hashes()

df_all = df_all[df_all.geom.isin(included_geoms)]
df_all["n_heavy"] = df_all.geom.apply(lambda x: all_geoms[x].n_heavy_atoms)
df_all = pd.merge(df_all, df_ref, on="geom", how="left")
df_all["error"] = (df_all.E - df_all.E_ref) * 1000
df_all["epoch"].fillna(0, inplace=True)
df = (
    df_all.groupby(["n_heavy", "experiment", "source", "method", "epoch"])
    .error.agg(["mean", "std", "count"])
    .reset_index()
)

color_cc = "gray"
ls_cc = "-"

plt.close("all")
fig, axes = plt.subplots(1, 3, figsize=(10, 4.0), sharey=True, sharex=False, width_ratios=[1, 1, 1.4])

axes_n_heavy = axes[:2]
ax_finetune = axes[2]
for ind_epoch, (ax, epochs) in enumerate(zip(axes_n_heavy, [0, 4000])):
    for exp_name, label, color, ls, marker in experiments:
        df_filt = df.query("experiment == @exp_name and n_heavy >= 3")
        df_filt = df_filt[
            ((df_filt.source == "dpe") & (df_filt.method == "reuse")) | (df_filt.source.str.contains("orca"))
        ]
        is_dpe = ("midimol" in exp_name) or ("v10" in exp_name) or ("nc_by_std" in exp_name)
        if is_dpe:
            df_filt = df_filt.query("epoch == @epochs")
        if len(df_filt) == 0:
            continue
        ax.errorbar(
            df_filt.n_heavy,
            df_filt["mean"],
            yerr=df_filt["std"] if is_dpe else None,
            label=label if is_dpe else None,
            color=color if is_dpe else color_cc,
            ls=ls if is_dpe else ls_cc,
            capsize=5,
            lw=2.5 if is_dpe else None,
            marker=marker,
            markersize=8,
            zorder=None if is_dpe else 0,
        )
        if not is_dpe:
            ax.text(6.5, df_filt["mean"].max() * 1.02, label, color=color_cc, fontsize=9, ha="center", va="bottom")
    ax.axhline(0, color=color_cc, ls=ls_cc, zorder=0)
    ax.text(6.5, 0, "CC-CBS", color=color_cc, fontsize=9, ha="center", va="bottom")

    ax.set_xticks([3, 4, 5, 6, 7])
    ax.set_xlim([3, 7])
    ax.set_xlabel("Nr. of heavy atoms")
    if ind_epoch == 0:
        ax.set_yscale("symlog", linthresh=10)
        ax.set_yticks(np.concatenate([[-10, 0], 10 ** np.arange(1, 5 + 1)]))
        ax.set_ylabel("$E - E_\\mathrm{CCSD(T)-CBS}$ / mHa", labelpad=0)
        ax.set_ylim([-1, 100_000])
        ax.set_title("Zero-shot")
    else:
        ax.set_title(f"After {epochs//1000}k fine-tuning steps")

    color_arrows = "dimgray"
    for x in [5]:
        dfmax = df[df.experiment.str.contains("v10_500k") & (df.n_heavy == x) & (df.epoch == epochs)]
        dfmin = df[df.experiment.str.contains("699torsion_nc_by_std_256k") & (df.n_heavy == x) & (df.epoch == epochs)]
        assert (len(dfmax) == 1) and (len(dfmin) == 1)
        ymax = dfmax["mean"].iloc[0]
        ymin = dfmin["mean"].iloc[0]
        improvement = ymax / ymin
        ax.arrow(
            x + 0.05,
            ymax,
            0,
            ymin - ymax,
            color=color_arrows,
            head_width=0.2,
            head_length=ymin * 0.6,
            length_includes_head=True,
            width=0.02,
            zorder=10,
        )

        if ind_epoch == 0:
            y_text = 6000
            improvement = f"{np.round(improvement, -1):.0f}x"
        else:
            y_text = np.sqrt(ymax * ymin)
            improvement = f"{improvement:.1f}x"
        ax.text(x + 0.1, y_text, improvement, fontsize=10, color=color_arrows, zorder=10)


df_finetune = df.query("n_heavy == 3")
df_finetune = df_finetune.query("method in ['reuse', 'indep']")
df_finetune = df_finetune.groupby(["experiment", "epoch"]).mean().reset_index()

for experiment, label, color, linestyle, marker in experiments:
    df_filt = df_finetune.query("experiment == @experiment")
    is_ref = "HF_CBS" in experiment or "CCSD(T)" in experiment
    if is_ref:
        error = df_filt.iloc[0]["mean"]
        ax_finetune.axhline(error, color=color_cc, linestyle=ls_cc)
        ax_finetune.text(
            np.sqrt(32000 * 16000), error * 1.05, label, color=color_cc, fontsize=10, ha="center", va="bottom"
        )
    else:
        ax_finetune.errorbar(
            df_filt.epoch,
            df_filt["mean"],
            yerr=df_filt["std"],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=7,
            capsize=3,
            lw=2.0,
            zorder=10,
        )
ax_finetune.axhline(0, color=color_cc, ls=ls_cc, zorder=0)
ax_finetune.text(425, 0, "CC-CBS", color=color_cc, fontsize=9, ha="center", va="bottom")

ax_finetune.set_xlabel("Optimization steps")
ax_finetune.set_xscale("asinh", linear_width=1500)
ax_finetune.set_xlim([0, 32_000])
set_ticks_with_SI_postfix(ax_finetune, "x", [0, 1000, 2000, 4000, 8000, 16_000, 32_000])
ax_finetune.legend(loc="upper left", handlelength=3.0, fontsize=9.5, bbox_to_anchor=(0.1, 1.0))
ax_finetune.set_title("Test set with 3 heavy atoms")


for ax in axes:
    ax.grid(color="gray", alpha=0.3, ls=":", zorder=-1)

fig.tight_layout()
for ax, letter in zip(axes, "abc"):
    ax.text(x=0, y=1.01, s=letter, transform=ax.transAxes, fontsize=14, fontweight="bold", ha="left", va="bottom")

# fname = "/home/mscherbela/ucloud/results/04_paper_better_universal_wf/figures/fig3_absolute_energies.png"
# fig.savefig(fname, dpi=300, bbox_inches="tight")
# fig.savefig(fname.replace(".png", ".pdf"), bbox_inches="tight")


# %%
