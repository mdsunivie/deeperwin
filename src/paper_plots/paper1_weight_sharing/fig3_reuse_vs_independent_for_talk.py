import numpy as np
import matplotlib.pyplot as plt
import pickle
from paper_plots.paper1_weight_sharing.box_plot import draw_box_plot

optimizer = "kfac"
cache_fname = f"/home/mscherbela/tmp/data_shared_vs_indep_{optimizer}.pkl"
with open(cache_fname, "rb") as f:
    full_plot_data = pickle.load(f)
del full_plot_data["Methane"]

colors = dict(
    Indep="C0",
    ReuseIndep="C0",
    ReuseShared95="C4",
    ReuseFromIndep="C2",
    ReuseFromIndepSingleLR="darkgreen",
    ReuseFromSmallerShared="brown",
)
markers = dict(
    Indep="o",
    ReuseIndep="s",
    ReuseShared95="^",
    ReuseFromIndep="D",
    ReuseFromIndepSingleLR="s",
    ReuseFromSmallerShared="v",
)
labels = dict(
    Indep="Independent opt.\n(geom. of shared exp.)",
    ReuseIndep="Independent opt.",
    ReuseShared95="Pre-trained by shared opt.",
    ReuseFromIndep="Reuse from indep. opt.",
    ReuseFromIndepSingleLR="Reuse from indep. opt. (single LR)",
    ReuseFromSmallerShared="Pre-trained by shared opt.\nof smaller molecule",
)
titles = dict(
    H4p="$H_4^+$: 16 geometries",
    H6="$H_6$: 23 geometries",
    H10="$H_{10}$: 23 geometries",
    Ethene="Ethene: 20 geometries",
)

show_n_samples = False
include_whiskers = False

plt.close("all")
kind = "all"  # no_smaller, all

if kind == "no_smaller":
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), dpi=100)
else:
    fig, axes = plt.subplots(2, 2, figsize=(8, 9), dpi=100)

for ind_mol, (molecule, plot_data) in enumerate(full_plot_data.items()):
    row, col = ind_mol // 2, ind_mol % 2
    ax = axes[row][col]

    if optimizer == "kfac":
        if kind == "no_smaller":
            curve_types = ["ReuseIndep", "ReuseShared95"]
        else:
            curve_types = ["ReuseIndep", "ReuseShared95", "ReuseFromSmallerShared"]
        ylims = dict(H4p=[-0.5, 6.0], H6=[-0.5, 6.5], H10=[-1, 14.5], Ethene=[-5, 50.0])
    else:
        curve_types = ["Indep", "ReuseShared95"]
        ylims = dict(H4p=[-0.5, 6.0], H6=[-0.5, 9], H10=[-1, 14.5], Ethene=[-5, 60.0])
    for curve_type in curve_types:
        if curve_type not in plot_data:
            continue

        error_mean = np.array(plot_data[curve_type].error_mean)
        lower_bound = np.array(plot_data[curve_type].error_25p)
        upper_bound = np.array(plot_data[curve_type].error_75p)
        lower_whiskers = error_mean - np.array([np.min(e) for e in plot_data[curve_type].errors])
        upper_whiskers = np.array([np.max(e) for e in plot_data[curve_type].errors]) - error_mean

        ax.plot(
            plot_data[curve_type].n_epochs,
            error_mean,
            color=colors[curve_type],
            marker=markers[curve_type],
            label=labels[curve_type],
            markersize=4,
        )
        for x, y in zip(plot_data[curve_type].n_epochs, plot_data[curve_type].errors):
            draw_box_plot(
                ax,
                x,
                y,
                width=1.15,
                color=colors[curve_type],
                alpha=0.6,
                scale="log",
                center_line="mean",
                alpha_outlier=0.2,
                marker=markers[curve_type],
            )

        ax.set_ylim(ylims[molecule])
        ax.set_xlim([50, 19000])
        ax.set_xscale("log")

    ax.grid(alpha=0.5, color="gray")
    ax.axhline(1.6, color="gray", linestyle="--", label="Chem. acc.: 1 kcal/mol")
    # ax.set_ylim([0, ylims[molecule]])
    xticks = 2 ** np.arange(6, 15, 2)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:,d}" for x in xticks])
    ax.set_xticks(2 ** np.arange(6, 15, 1), minor=True)
    ax.set_title(titles[molecule])
    if col == 0:
        ax.set_ylabel("energy rel. to MRCI / mHa")
    if row == 1:
        ax.set_xlabel("training epochs / geometry")
    if kind == "no_smaller":
        if (row == 0) and (col == 0):
            ax.legend(loc="upper right", framealpha=0.9)
    if kind == "all":
        if (row == 1) and (col == 0):
            ax.legend(loc="upper right", framealpha=0.9)
    ax.minorticks_off()
    # ax.text(0.0, 1.02, f"({chr(97+ind_mol)})", dict(fontweight='bold', fontsize=12), transform = ax.transAxes)

if optimizer == "adam":
    fig.suptitle(f"Optimizer: {optimizer.upper()}")
fig.tight_layout()

# for transparent in [True, False]:
#     fname = f"/home/mscherbela/Nextcloud/PhD/talks_and_conferences/2022_IPAM/Talk/images/reuse_{kind}_{'transp' if transparent else ''}.png"
#     # fname = f"/home/mscherbela/Nextcloud/PhD/talks_and_conferences/2022_IPAM/Talk/images/weight_sharing_{'transp' if transparent else ''}.png"
#     fig.savefig(fname, dpi=400, transparent=transparent)
