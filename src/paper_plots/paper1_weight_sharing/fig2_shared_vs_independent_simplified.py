# %%
import matplotlib.pyplot as plt
import numpy as np
import pickle

optimizer = "kfac"
cache_fname = f"/home/mscherbela/tmp/data_shared_vs_indep_{optimizer}.pkl"
with open(cache_fname, "rb") as f:
    full_plot_data = pickle.load(f)
del full_plot_data["Methane"]

colors = dict(Indep="C0", Shared_75="C1", Shared_95="r", ReuseFromIndep="C2")
markers = dict(Indep="s", Shared_75="o", Shared_95="^", ReuseFromIndep="D")
labels = dict(
    Indep="Old method:\nIndependent optimization",
    Shared_75="75% of weights shared",
    Shared_95="New method:\nWeight-sharing",
    ReuseFromIndep="Reuse from indep. opt.",
)
titles = dict(H4p="$H_4^+$: 112 geometries", H6="$H_6$: 49 geometries", H10="$H_{10}$", Ethene="Ethene")
if optimizer == "adam":
    speed_up_arrows = dict(  # H4p=(530, 100),
        H6=(2100, 200), H10=(4096, 420), Ethene=(16384, 1900)
    )
    ylims = dict(H4p=[-0.5, 6.0], H6=[0, 11], H10=[-1, 12], Ethene=[-1, 120])
else:
    speed_up_arrows = dict(  # H4p=(230, 40),
        H6=(512, 64), H10=(1024, 128), Ethene=(16384, 1024)
    )
    ylims = dict(H4p=[-0.5, 6.0], H6=[-0.5, 6.5], H10=[-0.5, 11], Ethene=[-3, 35.0])
    # ylims = dict(H4p=[-0.5, 7], H6=[-0.5,7], H10=[-1,13.0], Ethene=[-5,140.0])
#

show_n_samples = False
include_whiskers = False

plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(8, 5), dpi=100)
if show_n_samples:
    fig_n_samples, axes_n_samples = plt.subplots(2, 2, figsize=(8, 6), dpi=100)

del full_plot_data["H4p"]
del full_plot_data["H6"]
for ind_mol, (molecule, plot_data) in enumerate(full_plot_data.items()):
    ax = axes[ind_mol]

    for curve_type in ["Indep", "Shared_95"]:
        if curve_type not in plot_data:
            continue

        error_mean = np.array(plot_data[curve_type].error_mean)
        lower_bound = np.array(plot_data[curve_type].error_25p)
        upper_bound = np.array(plot_data[curve_type].error_75p)
        lower_whiskers = error_mean - np.array([np.min(e) for e in plot_data[curve_type].errors])
        upper_whiskers = np.array([np.max(e) for e in plot_data[curve_type].errors]) - error_mean

        ax.errorbar(
            plot_data[curve_type].n_epochs,
            error_mean,
            yerr=plot_data[curve_type].error_std,
            capsize=3,
            color=colors[curve_type],
            marker=markers[curve_type],
            label=labels[curve_type],
            markersize=6,
        )
        # for x,y in zip(plot_data[curve_type].n_epochs,  plot_data[curve_type].errors):
        # draw_box_plot(ax, x, y, width=1.15, color=colors[curve_type],
        #               alpha=0.6, scale='log', center_line='mean', alpha_outlier=0.2,
        #               marker=markers[curve_type], draw_whiskers=False, draw_outliers=False)

        ax.set_ylim(ylims[molecule])
        ax.set_xlim([50, 19000])
        ax.set_xscale("log")

    #
    # if molecule in speed_up_arrows:
    #     x1,x2 = speed_up_arrows[molecule]
    #     ax.axvline(x1, color='dimgray', linestyle='-', alpha=1.0, zorder=-1, linewidth=1)
    #     ax.axvline(x2, color='dimgray', linestyle='-', alpha=1.0, zorder=-1, linewidth=1)
    #     y_arrow = ylims[molecule] * 0.8
    #     ax.arrow(x1, y_arrow, x2 - x1, 0, head_width=y_arrow*0.05, head_length=x2*0.3, length_includes_head=True, color='dimgray')
    #     ax.text(np.sqrt(x1*x2), y_arrow * 1.01, f"~{x1/x2:.0f}x", ha='center', va='bottom', fontsize=12,
    #             # bbox=dict(facecolor='white', edgecolor='None')
    #             )

    ax.grid(alpha=0.5, color="gray")
    ax.axhline(1.6, color="gray", linestyle="--", label="Chemical accuracy: 1 kcal/mol")
    # ax.set_ylim([0, ylims[molecule]])
    xticks = 2 ** np.arange(6, 15, 2)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:,d}" for x in xticks])
    ax.set_xticks(2 ** np.arange(6, 15, 1), minor=True)
    ax.set_title(titles[molecule], fontsize=16)
    if ind_mol == 0:
        ax.set_ylabel("error relative to reference / mHa", fontsize=12)
    if ind_mol == 0:
        ax.legend(loc="upper right", framealpha=0.9, labelspacing=0.8)

    ax.set_xlabel("training steps per geometry", fontsize=12)
    ax.minorticks_off()
    ax.text(0.0, 1.02, f"({chr(97+ind_mol)})", dict(fontweight="bold", fontsize=12), transform=ax.transAxes)


if optimizer == "adam":
    fig.suptitle(f"Optimizer: {optimizer.upper()}")
fig.tight_layout()

fname = f"/home/mscherbela/ucloud/results/paper_figures/jax/shared_vs_indep_{optimizer}_simplified.png"
fig.savefig(fname, dpi=400)
fig.savefig(fname.replace(".png", ".pdf"))
