# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
from deeperwin.run_tools.geometry_database import load_energies, load_geometries, load_datasets
from paper_plots.paper3_universal_wf.plot_utils import plot_df, format_with_SI_postfix
import numpy as np


def get_n_compounds(reuse_from_name):
    if reuse_from_name.endswith("3x20"):
        return 3
    elif reuse_from_name.endswith("9x20"):
        return 9
    else:
        return 18


def get_model_size(orbitals_name):
    if orbitals_name.endswith("_S"):
        return 0
    elif orbitals_name.endswith("_M"):
        return 1
    else:
        return 2


save_figs = True
REFERENCE_METHOD = "CCSD(T)_CBS"
N_EPOCHS_FINETUNE = 4_000

all_geometries = load_geometries()
all_datasets = load_datasets()
in_distr_geometries = [
    g.hash
    for g in all_datasets["TinyMol_CNO_rot_dist_test_in_distribution_30geoms"].get_geometries(
        all_geometries, all_datasets
    )
]
out_of_distr_geometries = [
    g.hash
    for g in all_datasets["TinyMol_CNO_rot_dist_test_in_distribution_30geoms"].get_geometries(
        all_geometries, all_datasets
    )
]

experiments = ["2023-03-06_tinymol_v10_ablation_n_pretrain"]
df_full = load_energies()
df_ref = df_full[df_full.experiment == REFERENCE_METHOD][["geom", "E"]].rename(columns=dict(E="E_ref"))
df_full = df_full.query("experiment in @experiments")
df_full = df_full.merge(df_ref, "left", "geom")
df_full["in_distribution"] = df_full["geom"].isin(in_distr_geometries)
df_full["n_compounds"] = df_full["reuse_from"].apply(get_n_compounds)
df_full["model_size"] = df_full["orbitals"].apply(get_model_size)


df = df_full.query("epoch == @N_EPOCHS_FINETUNE")
df["error"] = (df["E"] - df["E_ref"]) * 1000
plt.close("all")
fig, axes = plt.subplots(3, 1, figsize=(5, 10))
axes = axes.flatten()
ax_model, ax_data, ax_epochs = axes
titles = ["Model size", "Data size", "Pretraining steps"]

plot_kwargs = dict(
    label_dict={True: "Small molecules", False: "Large molecules"},
    color_dict={True: "C0", False: "C1"},
    marker_dict={True: "o", False: "s"},
)

####### Model size ############
plot_df(
    ax_model,
    df.query("n_pretrain_variational == 256000 and n_compounds == 18"),
    x="model_size",
    y="error",
    experiment="in_distribution",
    **plot_kwargs,
)
ax_model.set_xticks([0, 1, 2])
ax_model.set_xticklabels(["S", "M", "L"])
ax_model.set_xlabel("Model size")

####### Data size ############
plot_df(
    ax_data,
    df.query("n_pretrain_variational == 256000 and model_size == 2"),
    x="n_compounds",
    y="error",
    experiment="in_distribution",
    **plot_kwargs,
)
ax_data.set_xticks([3, 9, 18])
ax_data.set_xlabel("Nr. of compounds in training set")


####### Pretraining steps ############
plot_df(
    ax_epochs,
    df.query("n_compounds == 18 and model_size == 2"),
    x="n_pretrain_variational",
    y="error",
    experiment="in_distribution",
    **plot_kwargs,
)
ax_epochs.set_xscale("log", subs=[])
ax_epochs.set_xticks([64_000, 128_000, 256_000, 512_000])
ax_epochs.xaxis.set_major_formatter(lambda x, pos: format_with_SI_postfix(x))
ax_epochs.set_xlabel("Nr. of pretraining steps")


for ax, title in zip(axes, titles):
    ax.minorticks_off()
    ax.legend(loc="center right")
    ax.set_title(title, fontweight="bold")
    # ax.set_ylabel("$E_\mathrm{VMC} - E_\mathrm{CCSD(T)}$ / mHa")
    ax.set_ylabel("$E - E_\\mathrm{CCCSD(T)-CBS}$ / mHa")
    ax.set_ylim([0, None])
    # ax.set_yscale("log")

fig.suptitle(f"Energy error after {N_EPOCHS_FINETUNE//1000}k finetuning-steps")
fig.tight_layout()
fig.text(0.03, 0.93, "a.", fontsize=16, fontweight="bold")
fig.text(0.03, 0.61, "b.", fontsize=16, fontweight="bold")
fig.text(0.03, 0.30, "c.", fontsize=16, fontweight="bold")
if save_figs:
    fig.savefig(
        "/home/mscherbela/ucloud/results/03_paper_unversal_wavefuncion/figures/ablation_foundational_model.png",
        dpi=400,
        bbox_inches="tight",
    )
    fig.savefig(
        "/home/mscherbela/ucloud/results/03_paper_unversal_wavefuncion/figures/ablation_foundational_model.pdf",
        bbox_inches="tight",
    )

# %%
n_epochs_pre = [64_000, 128_000, 256_000, 360_000, 500_000]
n_compounds = [3, 9, 18]
model_sizes = [0, 1, 2]
n_epochs_fine = [0, 250, 1000, 4000, 8000, 16000, 32000]
n_runs_in_distr = [
    np.zeros([len(n_epochs_pre), len(n_epochs_fine)]),
    np.zeros([len(n_compounds), len(n_epochs_fine)]),
    np.zeros([len(model_sizes), len(n_epochs_fine)]),
]
n_runs_out_of_distr = [np.zeros_like(x) for x in n_runs_in_distr]
for j, n in enumerate(n_epochs_fine):
    for i, n_pre in enumerate(n_epochs_pre):
        n_runs_in_distr[0][i, j] = len(
            df_full.query(
                "n_pretrain_variational == @n_pre and epoch == @n and in_distribution == True and n_compounds == 18 and model_size == 2"
            )
        )
        n_runs_out_of_distr[0][i, j] = len(
            df_full.query(
                "n_pretrain_variational == @n_pre and epoch == @n and in_distribution == False and n_compounds == 18 and model_size == 2"
            )
        )
    for i, n_comp in enumerate(n_compounds):
        n_runs_in_distr[1][i, j] = len(
            df_full.query(
                "n_pretrain_variational == 256000 and epoch == @n and in_distribution == True and n_compounds == @n_comp and model_size == 2"
            )
        )
        n_runs_out_of_distr[1][i, j] = len(
            df_full.query(
                "n_pretrain_variational == 256000 and epoch == @n and in_distribution == False and n_compounds == @n_comp and model_size == 2"
            )
        )
    for i, mod_size in enumerate(model_sizes):
        n_runs_in_distr[2][i, j] = len(
            df_full.query(
                "n_pretrain_variational == 256000 and epoch == @n and in_distribution == True and n_compounds == 18 and model_size == @mod_size"
            )
        )
        n_runs_out_of_distr[2][i, j] = len(
            df_full.query(
                "n_pretrain_variational == 256000 and epoch == @n and in_distribution == False and n_compounds == 18 and model_size == @mod_size"
            )
        )


fig_count, axes_count = plt.subplots(3, 2, figsize=(10, 10))
for row in range(3):
    for col, ax, counts in zip([0, 1], axes_count[row], [n_runs_in_distr[row], n_runs_out_of_distr[row]]):
        ax.imshow(counts, cmap="viridis", clim=[0, 40] if col else [0, 30])
        if row == 0:
            ax.set_yticks(np.arange(len(n_epochs_pre)))
            ax.set_yticklabels([str(x // 1000) + "k" for x in n_epochs_pre])
            ax.set_ylabel("Pretraining")
        elif row == 1:
            ax.set_yticks(np.arange(3))
            ax.set_yticklabels([3, 9, 18])
            ax.set_ylabel("Nr of compounds")
        elif row == 2:
            ax.set_yticks(np.arange(3))
            ax.set_yticklabels(["S", "M", "L"])
            ax.set_ylabel("Model size")
        ax.set_xticks(np.arange(len(n_epochs_fine)))
        ax.set_xticklabels([str(x) if x < 1000 else str(x // 1000) + "k" for x in n_epochs_fine])
        ax.set_xlabel("Finetuning")
        ax.grid(False)
axes_count[0][0].set_title("30 small molecules")
axes_count[0][1].set_title("40 large molecules")
fig.tight_layout()

for ax in axes_count.flatten():
    ax.axvline(x=n_epochs_fine.index(4000), color="dimgray")

for ax in axes_count[0]:
    ax.axhline(y=n_epochs_pre.index(500_000), color="dimgray")

# %%
# Diagnostic using individual molecules
df_filt = df_full.query("model_size == 2 and n_compounds == 18 and in_distribution == False")
# df_filt = df_full.query("n_pretrain_variational == 256000 and n_compounds == 18")
df_filt = df_filt.query("epoch == 4000")
df_filt["error"] = (df_filt.E - df_filt.E_ref) * 1000

fig, ax = plt.subplots(1, 1, figsize=(14, 9))
sns.lineplot(df_filt, x="n_pretrain_variational", y="error", hue="molecule", ax=ax, marker="o")
# ax.set_xscale("log")
ax.set_yscale("log")
