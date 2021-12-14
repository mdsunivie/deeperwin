import numpy as np
import matplotlib.pyplot as plt
import pickle

optimizer = "adam"
cache_fname = f"/home/mscherbela/tmp/data_shared_vs_indep_{optimizer}.pkl"
with open(cache_fname, 'rb') as f:
    full_plot_data = pickle.load(f)
del full_plot_data['Methane']

colors = dict(Indep='C0', ReuseIndep='C0', ReuseShared95='C4', ReuseFromIndep='C2', ReuseFromIndepSingleLR='darkgreen', ReuseFromSmallerShared='brown')
markers = dict(Indep='o', ReuseIndep='s', ReuseShared95='^', ReuseFromIndep='D', ReuseFromIndepSingleLR='s', ReuseFromSmallerShared='v')
labels = dict(Indep='Independent opt.\n(geom. of shared exp.)', ReuseIndep='Independent opt.',
              ReuseShared95='Pre-trained by shared opt.', ReuseFromIndep="Reuse from indep. opt.", ReuseFromIndepSingleLR="Reuse from indep. opt. (single LR)",
              ReuseFromSmallerShared="Pre-trained by shared opt.\nof smaller molecule")
titles = dict(H4p='$H_4^+$: 16 geometries', H6='$H_6$: 23 geometries', H10='$H_{10}$: 23 geometries', Ethene='Ethene: 20 geometries')
ylims = dict(H4p=6.0, H6=6.0, H10=10.0, Ethene=30.0)

show_n_samples = True

plt.close("all")
fig, axes = plt.subplots(2,2, figsize=(8,6), dpi=200)
if show_n_samples:
    fig_n_samples, axes_n_samples = plt.subplots(2,2, figsize=(8,6), dpi=200)

for ind_mol, (molecule,plot_data) in enumerate(full_plot_data.items()):
    row, col = ind_mol // 2, ind_mol % 2
    ax = axes[row][col]

    if optimizer == 'kfac':
        curve_types = ['ReuseIndep', 'ReuseShared95', 'ReuseFromSmallerShared']
    else:
        curve_types = ['Indep', 'ReuseShared95']
    for curve_type in curve_types:
        if curve_type not in plot_data:
            continue
        ax.semilogx(plot_data[curve_type].n_epochs,
                    plot_data[curve_type].error_mean,
                    color=colors[curve_type],
                    marker=markers[curve_type],
                    label=labels[curve_type])
        ax.fill_between(plot_data[curve_type].n_epochs,
                        plot_data[curve_type].error_25p,
                        plot_data[curve_type].error_75p,
                        alpha=0.3,
                        color=colors[curve_type])
        ax.set_ylim([-0.1 * ylims[molecule], ylims[molecule]])
        ax.set_xlim([64, 16384])


        if show_n_samples:
            axes_n_samples[row][col].semilogx(plot_data[curve_type].n_epochs,
                                              plot_data[curve_type].n_samples,
                                              color=colors[curve_type])

    ax.grid(alpha=0.5, color='gray')
    ax.axhline(1.6, color='gray', linestyle='--', label="Chem. acc.: 1 kcal/mol")
    # ax.set_ylim([0, ylims[molecule]])
    xticks = 2 ** np.arange(6, 15, 2)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:,d}" for x in xticks])
    ax.set_xticks(2 ** np.arange(6, 15, 1), minor=True)
    ax.set_title(titles[molecule])
    if col == 0:
        ax.set_ylabel("energy error / mHa")
    if row == 1:
        ax.set_xlabel("training epochs / geometry")
    if (row == 0) and (col == 0):
        ax.legend(loc='upper right', framealpha=0.9)
    ax.minorticks_off()
if optimizer == 'adam':
    fig.suptitle(f"Optimizer: {optimizer.upper()}")
fig.tight_layout()

fname = f"/home/mscherbela/ucloud/results/paper_figures/jax/reuse_vs_indep_{optimizer}.png"
fig.savefig(fname, dpi=400, bbox_inches='tight')
fig.savefig(fname.replace(".png", ".pdf"))