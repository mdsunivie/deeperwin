
#%%
import pandas as pd
from deeperwin.run_tools.geometry_database import load_energies, load_geometries, load_datasets
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm
from paper_plots.paper3_universal_wf.plot_utils import plot_df, format_with_SI_postfix
import numpy as np
sns.set_theme(style="whitegrid")
save_figs = True

all_geometries = load_geometries()
all_datasets = load_datasets()
geoms_small = all_datasets["TinyMol_CNO_rot_dist_test_in_distribution_30geoms"].get_geometries(all_geometries, all_datasets)
geoms_large = all_datasets["TinyMol_CNO_rot_dist_test_out_of_distribution_40geoms"].get_geometries(all_geometries, all_datasets)
geoms_small = [g.hash for g in geoms_small]
geoms_large = [g.hash for g in geoms_large]


def digits_to_subscript(s):
    s_out = ""
    for c in s:
        if c.isdigit():
            s_out += f"$_{c}$"
        else:
            s_out += c
    return s_out

MOLECULES_IN_TRAINING_SET = ["C2H4", "COH2", "CNH"]
MOLECULES_OUTSIDE_TRAINING_SET = ["C3H4", "CO2", "CN2H2", "CNOH"]
# PRETRAINING_EPOCHS = 256_000
PRETRAINING_EPOCHS = 500_000
REFERENCE_METHOD = "E_CCSD(T)_CBS"
all_molecules = MOLECULES_IN_TRAINING_SET + MOLECULES_OUTSIDE_TRAINING_SET
all_geometries = load_geometries()
experiments = ["2023-03-01_gao_shared_TinyMol", "2023-03-06_tinymol_v10_ablation_n_pretrain"]

df_full = load_energies()
df_ref = df_full[df_full.experiment.str.startswith("HF_CBS") | df_full.experiment.str.startswith("CCSD(T)")].pivot_table(index="geom", columns="experiment", values="E", aggfunc="mean")
df_ref.columns = ["E_" + c for c in list(df_ref)]
df_ref["E_ref"] = df_ref[REFERENCE_METHOD]
for c in list(df_ref):
    df_ref[c.replace("E_", "error_")] = 1000*(df_ref[c] - df_ref[REFERENCE_METHOD])
df_ref = df_ref.reset_index()
df_ref = df_ref[df_ref.geom.isin(geoms_small) | df_ref.geom.isin(geoms_large)]
df_ref["is_small"] = df_ref.geom.isin(geoms_small)
conventional_errors = df_ref.groupby(["is_small"]).mean()
#%%
df = df_full.query("experiment in @experiments")
filter_shared = df.method == 'shared'
filter_reuse = df.method == 'reuseshared'
filter_reuse = filter_reuse & (df.n_pretrain_variational == PRETRAINING_EPOCHS)
filter_reuse = filter_reuse & (df.reuse_from.str.endswith("shared_tinymol_v10"))
df = df[filter_shared | filter_reuse]
pivot = pd.pivot(df, index=["molecule", "geom", "epoch"], columns="method", values="E").reset_index()
pivot["delta_E_mHa"] = (pivot["shared"] - pivot["reuseshared"]) * 1000
pivot["molecule_in_training"] = pivot["molecule"].isin(MOLECULES_IN_TRAINING_SET)
pivot = pivot.merge(df_ref, "left", "geom")
pivot["error_shared"] = (pivot["shared"] - pivot["E_ref"]) * 1000
pivot["error_reuseshared"] = (pivot["reuseshared"] - pivot["E_ref"]) * 1000


color_dict, marker_dict = dict(), dict()
for i, m in enumerate(MOLECULES_IN_TRAINING_SET):
    color_dict[m] = matplotlib.cm.get_cmap("tab20b")(i)
    marker_dict[m] = "o"
for i, m in enumerate(MOLECULES_OUTSIDE_TRAINING_SET):
    color_dict[m] = matplotlib.cm.get_cmap("tab20b")(i+4)
    marker_dict[m] = "^"
label_dict = {m:m for m in all_molecules}
label_dict.update(**dict(COH2="CH2O", CNH="HCN", CNH2="CH2N2", CNOH="HNCO"))

plt.close("all")
plot_kwargs = dict(color_dict = {True: "C0", False: "C1"},
                   marker_dict = {True: 'o', False: 's'})
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plot_df(ax,
        pivot.query("epoch > 0"),
        x="epoch",
        y="error_shared",
        experiment="molecule_in_training",
        **plot_kwargs,
        ls="--",
        label_dict={True: "Train from scratch: small compounds", False: "Train from scratch: large compounds"}
        )
plot_df(ax,
        pivot.query("epoch > 0"),
        x="epoch",
        y="error_reuseshared",
        experiment="molecule_in_training",
        **plot_kwargs,
        ls="-",
        lw=2,
        label_dict={True: "Fine-tune base-model: small compounds", False: "Fine-tune base-model: large compounds"}
        )
ax.set_xlabel("Optimization steps")
ax.set_ylabel("$E - E_\\mathrm{CCSD(T)/CBS}$ / mHa")
ax.set_yticks(10.0**np.array([1, 2, 3, 4, 5, 6]))
ax.set_xscale("log")
ax.set_yscale("log")

# ax.axhline(conventional_errors.loc[True, "error_HF_CBS"], label="HF (CBS)", color='C0')
# ax.axhline(conventional_errors.loc[False, "error_HF_CBS"], label="HF (CBS)", color='C1')
ax.axhline(conventional_errors.loc[True, "error_CCSD(T)_ccpCVTZ"], label="CCSD(T) (cc-pVTZ)", color='C0', ls=':')
ax.axhline(conventional_errors.loc[False, "error_CCSD(T)_ccpCVTZ"], label="CCSD(T) (cc-pVTZ)", color='C1', ls=':')
ax.legend(loc="upper right", handlelength=3.5)


# ax.set_ylim([-5, 300])
x_ticks = [250, 500, 1000, 2000, 4000, 8000, 16_000, 32_000]
ax.set_xlim([min(x_ticks)*0.95, max(x_ticks)*1.05])
ax.set_xticks(x_ticks)
ax.xaxis.set_major_formatter(lambda x, pos: format_with_SI_postfix(x))
fig.tight_layout()


plt.arrow(0.29, 0.75, 0, -0.39,
          length_includes_head=True,
          width=0.009,
          color='dimgray',
          transform=ax.transAxes,
          zorder=20)
plt.text(0.275, 0.5, "~60x lower\nenergy error", transform=ax.transAxes, ha="right")
# plt.arrow(0.7, 0.42, 0, -0.21,
#           length_includes_head=True,
#           width=0.007,
#           color='dimgray',
#           transform=ax.transAxes,
#           zorder=20)
# plt.text(0.685, 0.275, "~8x", transform=ax.transAxes, ha="right")
plt.arrow(0.85, 0.35, -0.59, 0,
          length_includes_head=True,
          width=0.007,
          color='dimgray',
          transform=ax.transAxes,
          zorder=20)
plt.text(0.5, 0.35, "~20x faster", transform=ax.transAxes, ha="center", va='bottom')
plt.arrow(0.99, 0.24, 0, -0.1,
          length_includes_head=True,
          width=0.007,
          color='dimgray',
          transform=ax.transAxes,
          zorder=20)
plt.text(0.98, 0.19, "~3x", transform=ax.transAxes, ha="right")

if save_figs:
    fig.savefig(f"/home/mscherbela/ucloud/results/03_paper_unversal_wavefuncion/figures/eval_foundational_model_v3.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"/home/mscherbela/ucloud/results/03_paper_unversal_wavefuncion/figures/eval_foundational_model_v3.pdf", bbox_inches="tight")

# %%
