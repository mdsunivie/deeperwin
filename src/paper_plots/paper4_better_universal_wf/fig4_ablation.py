# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from deeperwin.run_tools.geometry_database import load_energies, load_geometries, load_datasets
sns.set_theme(style="whitegrid")

all_geoms = load_geometries()
all_datasets = load_datasets()
df_all = load_energies()
df_ref = df_all.query("source == 'orca_CCSD(T)_CBS_234'")[["geom", "E"]]
df_ref.rename(columns={"E": "E_ref"}, inplace=True)

#### Fine-tuning #####
experiments = [
               ("2023-02-26_18x20_128karxiv_v10_128k", "Scherbela et al.\n128k pre-training"),
               ("2023-05-ablation3_128k", "+ improved\narchitecture"),
               ("2023-05-ablation4_128k", "+ normal-mode\ndistortions"),
               ("2023-05-ablation5_128k", "+ larger\ntraining set"),
               ("2023-05-01_699torsion_nc_by_std_128k", "+ 8 determinants"),
               ("2023-05-01_699torsion_nc_by_std_256k", "+ 256k pretraining-\nsteps")
               ]
experiments.append((experiments[-1][0], "Our work"))
ablation_nr = np.arange(len(experiments))
experiment_names = [e[0] for e in experiments]
experiment_labels = [e[1] for e in experiments]
geom_hashes = all_datasets["TinyMol_CNO_rot_dist_test_out_of_distribution_4geoms"].get_hashes()

df = df_all.query("geom in @geom_hashes and experiment in @experiment_names")
df = df[((df.experiment == "2023-05-01_699torsion_nc_by_std_256k") & (df.n_pretrain_variational == 256_000)) | ((df.experiment != "2023-05-01_699torsion_nc_by_std_256k") & (df.n_pretrain_variational == 128_000))]
df = pd.merge(df, df_ref, on="geom", how="left")
df["error"] = (df.E - df.E_ref) * 1000
df["epoch"] = df.epoch.fillna(0).astype(int)
df["ablation_nr"] = df["experiment"].map({exp: nr for nr, exp in zip(ablation_nr, experiment_names)})
df = df.groupby(["experiment", "epoch", "geom"]).mean().reset_index()
#%%
plot_settings = [
    (0, [10, 6000], "log"),
    (4000, [0, 20], "linear")
                 ]

bar_width = 0.9
color = "C0"
start_end_color = 'navy'
text_color_inside = "white"
text_color_outside = "black"
x_min = 5

def _round_to_significant_digits(value, n_digits=3):
    digits = np.ceil(np.log10(np.abs(value)))
    decimals = n_digits - int(digits)
    value = np.round(value, decimals)
    decimals = min(max(0, decimals), 1)
    return f"{value:.{decimals}f}"


def get_value_label(E_min, E_max, x_min, x_max, is_delta=False, is_log_scale=False):
    text_color = "white"
    text_ha = "center"

    if is_delta:
        label = f"{_round_to_significant_digits(E_max-E_min)}\n({E/E_prev-1:+.0%})"
    else:
        label = _round_to_significant_digits(E_max)

    if is_log_scale:
        E_min, E_max, x_min, x_max = np.log(E_min), np.log(E_max), np.log(x_min), np.log(x_max)
    full_width = np.abs(x_max - x_min)
    width = np.abs(E_max - E_min)

    if width / full_width > 0.1:
        pos = 0.5 * (E_min + E_max)
    else:
        pos = max(E_min, E_max) + 0.02 * full_width
        text_color = "black"
        text_ha = "left"
    if is_log_scale:
        pos = np.exp(pos)

    return label, pos, text_color, text_ha

plt.close("all")
fig, axes = plt.subplots(1, len(plot_settings), figsize=(11, 3.8), sharey=True)
for ax, epoch, xlim, scale in zip(axes, *zip(*plot_settings)):
    E_prev = xlim[0]
    for ind_ablation, experiment_name in enumerate(experiment_names):
        df_filt = df.query("experiment == @experiment_name and epoch == @epoch")
        print(epoch, ind_ablation, len(df_filt))
        E, E_std = df_filt.error.mean(), df_filt.error.std()
        text_color = text_color_inside
        ha = "center"
        if ind_ablation == 0:
            ax.barh(y=[ind_ablation], width=[E], color=start_end_color, height=bar_width)
        elif ind_ablation == len(experiment_names) - 1:
            ax.barh(y=[ind_ablation], width=[E], color=start_end_color, height=bar_width)
            ax.plot([E_prev, E_prev], [ind_ablation-1+0.5*bar_width, ind_ablation-0.5*bar_width], color='gray')
            E_prev = xlim[0]
        else:
            ax.barh(y=[ind_ablation], width=[E-E_prev], left=[E_prev], color=color, height=bar_width)
            ax.plot([E_prev, E_prev], [ind_ablation-1+0.5*bar_width, ind_ablation-0.5*bar_width], color='gray')

        value_label, value_label_pos, text_color, ha = get_value_label(E_prev, 
                                                                      E, 
                                                                      xlim[0], 
                                                                      xlim[1], 
                                                                      is_delta=ind_ablation not in [0, len(experiment_names)-1],
                                                                      is_log_scale=(scale=="log"))
        ax.text(value_label_pos, ind_ablation, value_label, va="center", ha=ha, color=text_color, fontsize=10)
        E_prev = E

    ax.set_ylim([np.max(ablation_nr) + 0.5, np.min(ablation_nr) - 0.5])
    ax.set_xlabel("$E - E_\mathrm{CCSD(T)}$ / mHa")
    ax.set_xlim(xlim)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", color='gray', alpha=0.2, ls='-', zorder=-1)

    ax.set_yticks(ablation_nr)
    # ax.set_yticklabels(experiment_labels)
    ax.set_yticklabels([])
    ax.set_xscale(scale)

    if epoch == 0:
        ax.set_xticks([10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
        ax.set_title("Zero-shot accuracy")
    else:
        ax.set_title(f"Accuracy after {epoch} fine-tuning steps")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0f}")
    # ax.xaxis.set_minor_formatter(lambda x, pos: "")

fig.tight_layout(rect=[0.13, 0, 1, 1])
for i, label in enumerate(experiment_labels):
    fontweight = None
    ypos_first = 0.92
    ypos_last = 0.07
    ypos = ypos_first + (ypos_last - ypos_first) * i / (len(experiment_labels) - 1)
    if "+" in label:
        label = label.replace("\n", "\n    ")
    if i == 0 or i == (len(experiment_labels) - 1):
        fontweight="bold"
    axes[0].text(-0.35, ypos, label, va="center", ha="left", fontsize=10, transform=axes[0].transAxes, fontweight=fontweight)

for ax, letter in zip(axes, "abc"):
    ax.text(x=0, y= 1.01, s=letter, transform=ax.transAxes, fontsize=14, fontweight="bold", ha="left", va="bottom")

fname = "/home/mscherbela/ucloud/results/04_paper_better_universal_wf/figures/fig4_ablation.png"
fig.savefig(fname, dpi=300, bbox_inches="tight")
fig.savefig(fname.replace(".png", ".pdf"), bbox_inches="tight")


# %%
