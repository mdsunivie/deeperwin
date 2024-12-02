# %%
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
geoms_small = all_datasets["TinyMol_CNO_rot_dist_test_in_distribution_30geoms"].get_geometries(
    all_geometries, all_datasets
)
geoms_large = all_datasets["TinyMol_CNO_rot_dist_test_out_of_distribution_40geoms"].get_geometries(
    all_geometries, all_datasets
)
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
df_ref = df_full[df_full.experiment.str.startswith("HF_CBS") | df_full.experiment.str.startswith("CCSD")]
df_ref = df_ref[df_ref.geom.isin(geoms_small) | df_ref.geom.isin(geoms_large)]
df_ref = df_ref.pivot_table(index="geom", columns="experiment", values="E")
methods = list(df_ref)
df_ref.columns = ["E_" + c for c in list(df_ref)]
df_ref["E_ref"] = df_ref[REFERENCE_METHOD]
for m in methods:
    df_ref[f"error_{m}"] = 1000 * (df_ref[f"E_{m}"] - df_ref[REFERENCE_METHOD])
df_ref = df_ref.reset_index()
df_ref = df_ref[df_ref.geom.isin(geoms_small) | df_ref.geom.isin(geoms_large)]
df_ref["is_small"] = df_ref.geom.isin(geoms_small)
conventional_errors = df_ref.groupby(["is_small"])[[f"error_{m}" for m in methods]].mean()
# %%
include_foundationa_model = True

df = df_full.query("experiment in @experiments")
filter_shared = df.method == "shared"
filter_reuse = df.method == "reuseshared"
filter_reuse = filter_reuse & (df.n_pretrain_variational == PRETRAINING_EPOCHS)
filter_reuse = filter_reuse & (df.reuse_from.str.endswith("shared_tinymol_v10"))
df = df[filter_shared | filter_reuse]
pivot = pd.pivot(df, index=["molecule", "geom", "epoch"], columns="method", values="E").reset_index()
pivot["delta_E_mHa"] = (pivot["shared"] - pivot["reuseshared"]) * 1000
pivot["molecule_in_training"] = pivot["molecule"].isin(MOLECULES_IN_TRAINING_SET)
pivot = pivot.merge(df_ref, "left", "geom")
pivot["error_shared"] = (pivot["shared"] - pivot["E_ref"]) * 1000
pivot["error_reuseshared"] = (pivot["reuseshared"] - pivot["E_ref"]) * 1000
pivot = pivot[~pivot["molecule_in_training"]]


color_dict, marker_dict = dict(), dict()
for i, m in enumerate(MOLECULES_IN_TRAINING_SET):
    color_dict[m] = matplotlib.cm.get_cmap("tab20b")(i)
    marker_dict[m] = "o"
for i, m in enumerate(MOLECULES_OUTSIDE_TRAINING_SET):
    color_dict[m] = matplotlib.cm.get_cmap("tab20b")(i + 4)
    marker_dict[m] = "^"
label_dict = {m: m for m in all_molecules}
label_dict.update(**dict(COH2="CH2O", CNH="HCN", CNH2="CH2N2", CNOH="HNCO"))

plt.close("all")
plot_kwargs = dict(color_dict={True: "C0", False: "C1"}, marker_dict={True: "o", False: "s"})
fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
plot_df(
    ax,
    pivot.query("epoch > 0"),
    x="epoch",
    y="error_shared",
    experiment="molecule_in_training",
    **plot_kwargs,
    ls="-",
    color="C0",
    label_dict={True: "Full training: in distrib.", False: "Full training from scratch"},
)
if include_foundationa_model:
    plot_df(
        ax,
        pivot.query("epoch > 0"),
        x="epoch",
        y="error_reuseshared",
        experiment="molecule_in_training",
        **plot_kwargs,
        ls="-",
        lw=2,
        color="C2",
        label_dict={True: "Fine-tuning: in distrib.", False: "Re-using pre-trained weights"},
    )
if include_foundationa_model:
    ax.set_xlabel("Optimization steps")
else:
    ax.set_xlabel("Training steps")
ax.set_ylabel("mean energy error / mHa")
ax.set_yticks(10.0 ** np.array([1, 2, 3, 4, 5, 6]))
ax.set_xscale("log")
ax.set_yscale("log")

# ax.axhline(conventional_errors.loc[True, "error_HF_CBS"], label="HF (CBS)", color="C0")
# ax.axhline(conventional_errors.loc[False, "error_HF_CBS"], label="HF (CBS)", color="C1")
# ax.axhline(conventional_errors.loc[True, "error_CCSD(T)_ccpCVTZ"], label="CCSD(T) (cc-pVTZ)", color="C0", ls=":")
ax.axhline(conventional_errors.loc[False, "error_CCSD(T)_ccpCVTZ"], label="CCSD(T) (cc-pVTZ)", color="dimgray", ls=":")
ax.legend(loc="upper right", handlelength=3.5)


# ax.set_ylim([-5, 300])
x_ticks = [250, 500, 1000, 2000, 4000, 8000, 16_000, 32_000]
ax.set_xlim([min(x_ticks) * 0.95, max(x_ticks) * 1.05])
ax.set_ylim([7, 50_000])
ax.set_xticks(x_ticks)
ax.xaxis.set_major_formatter(lambda x, pos: format_with_SI_postfix(x))
fig.tight_layout()


if save_figs:
    fig.savefig(
        f"/home/mscherbela/ucloud/results/defense/tao_reuse{include_foundationa_model}_only_OOD.png",
        dpi=400,
        bbox_inches="tight",
    )

# %%
