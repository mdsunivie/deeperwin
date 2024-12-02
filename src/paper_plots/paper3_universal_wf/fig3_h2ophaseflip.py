# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from deeperwin.run_tools.geometry_database import load_energies

sns.set_theme(style="whitegrid")

# ground_truth = -76.4382 # dpe 100k
ground_truth = -76.437421  # CCSD(T)/CBS_2345
energies = load_energies()

plt.close("all")
fig, axs = plt.subplots(2, 1, figsize=(4, 6))
# Pretraining loss

data_tao = energies[energies.experiment == "h2o_phaseflip_tao_pretraining_last100steps"]
data_truncation = energies[energies.experiment == "h2o_phaseflip_truncation_pretraining_last100steps"]
data_truncation.sort_values(by="geom_comment")

loss_gao = data_tao.sort_values(by="geom_comment").HF_pretraining_loss.to_list()
loss_truncation = data_truncation.sort_values(by="geom_comment").HF_pretraining_loss.to_list()

twist = list(np.arange(0, 45, 2.25))
axs[0].plot(
    twist,
    loss_gao,
    color="C0",
    label="Our work",
    ls="-",
    marker="o",
    lw=2.5,
)
axs[0].plot(
    twist,
    loss_truncation,
    color="C3",
    label="Standard backflow",
    ls="-",
    marker="^",
    lw=2.5,
)
axs[0].vlines(twist[11], 0.000, 0.03, colors="black", label=None, alpha=0.5, ls="dashed")
axs[0].set_ylim([0.0, 0.025])
# axs[0].set_title("HF-pretraining loss", fontsize=16)
axs[0].set_ylabel("HF-pre-training loss")
axs[0].set_xlabel("Rotation angle / °")

# Emean error
data_tao = energies[energies.experiment == "h2o_phaseflip_tao"]
data_truncation = energies[energies.experiment == "h2o_phaseflip_truncation"]

data_tao["error"] = (data_tao.E - ground_truth) * 1000
data_truncation["error"] = (data_truncation.E - ground_truth) * 1000

error_gao, std_gao = (
    data_tao.groupby(by="epoch").agg(np.mean).error.to_list(),
    data_tao.groupby(by="epoch").agg(np.std).error.to_list(),
)
error_truncation, std_truncation = (
    data_truncation.groupby(by="epoch").agg(np.mean).error.to_list(),
    data_tao.groupby(by="epoch").agg(np.std).error.to_list(),
)


from matplotlib.cm import get_cmap

colors = list(get_cmap("seismic")([0.15])) + list(get_cmap("bwr")([0.85]))

eval_epochs = [128, 512, 1024, 2048, 4096, 16384]
axs[1].plot(
    eval_epochs,
    error_gao,
    color="C0",
    label="Our work",
    ls="-",
    marker="o",
    lw=2.5,
)
axs[1].plot(
    eval_epochs,
    error_truncation,
    color="C3",
    label="Standard backflow",
    ls="-",
    marker="^",
    lw=2.5,
)
axs[1].set_xscale("log")

xtick_values = 2 ** np.arange(7, 15)
xtick_labels = [f"$2^{{{np.log2(x):.0f}}}$" for x in xtick_values]
axs[1].set_xticks(xtick_values)
axs[1].set_xticklabels(xtick_labels)
# axs[1].set_yticks(10.0**np.arange(-1, 4))
axs[1].set_ylim([0.5, 10000])
axs[1].set_yscale("log")

for ax, label in zip(axs, "ab"):
    if label != "a":
        ax.legend(loc="upper right")
    ax.text(0, 1.01, f"{label}.", transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")


axs[1].set_ylabel("$E - E_\\mathrm{CCSD(T)-CBS}$ / mHa")
axs[1].set_xlabel("optimization steps")

fig.tight_layout()
fig.savefig(
    "/home/mscherbela/ucloud/results/03_paper_unversal_wavefuncion/figures/fig3_h2o_phaseflip.png",
    dpi=400,
    bbox_inches="tight",
)
fig.savefig(
    "/home/mscherbela/ucloud/results/03_paper_unversal_wavefuncion/figures/fig3_h2o_phaseflip.pdf", bbox_inches="tight"
)
