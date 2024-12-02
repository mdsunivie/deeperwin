# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
from deeperwin.run_tools.geometry_database import load_energies, load_geometries, load_datasets
from paper_plots.paper3_universal_wf.plot_utils import plot_df, format_with_SI_postfix


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
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
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


for ind, (ax, title) in enumerate(zip(axes, titles)):
    ax.minorticks_off()
    if ind == 0:
        ax.legend(loc="upper right")
    ax.set_title(title, fontweight="bold")
    # ax.set_ylabel("$E_\mathrm{VMC} - E_\mathrm{CCSD(T)}$ / mHa")
    ax.set_ylabel("mean energy error / mHa")
    ax.set_ylim([0, None])
    # ax.set_yscale("log")

# fig.suptitle(f"Energy error after {N_EPOCHS_FINETUNE//1000}k finetuning-steps")
fig.tight_layout()
# fig.text(0.03, 0.93, "a.", fontsize=16, fontweight="bold")
# fig.text(0.03, 0.61, "b.", fontsize=16, fontweight="bold")
# fig.text(0.03, 0.30, "c.", fontsize=16, fontweight="bold")
if save_figs:
    fig.savefig(
        "/home/mscherbela/Nextcloud/PhD/talks_and_conferences/2023_04_CMU/test_modelablation_foundational_model.png",
        dpi=400,
        bbox_inches="tight",
    )

# %%
