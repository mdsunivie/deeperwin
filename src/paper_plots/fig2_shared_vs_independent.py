import matplotlib.pyplot as plt
import numpy as np
import pickle

optimizer = 'adam'
cache_fname = f"/home/mscherbela/tmp/data_shared_vs_indep_{optimizer}.pkl"
with open(cache_fname, 'rb') as f:
    full_plot_data = pickle.load(f)
del full_plot_data['Methane']

colors = dict(Indep='C0', Shared_75='C1', Shared_95='r', ReuseFromIndep='C2')
markers = dict(Indep='s', Shared_75='o', Shared_95='^', ReuseFromIndep='D')
labels = dict(Indep='Independent opt.', Shared_75='75% of weights shared', Shared_95='95% of weights shared', ReuseFromIndep="Reuse from indep. opt.")
titles = dict(H4p='$H_4^+$: 112 geometries', H6='$H_6$: 49 geometries', H10='$H_{10}$: 49 geometries', Ethene='Ethene: 30 geometries')
ylims = dict(H4p=6.0, H6=6.0, H10=10.0, Ethene=30.0)
if optimizer == "adam":
    speed_up_arrows = dict(H4p=(530, 100),
                           H6=(2100, 200),
                           H10=(4096, 420),
                           Ethene=(16384, 1900))
else:
    speed_up_arrows = dict(H4p=(230, 40),
                           H6=(600, 80),
                           H10=(1100, 150),
                           Ethene=(13500, 1024))

show_n_samples = True

plt.close("all")
fig, axes = plt.subplots(2,2, figsize=(8,6), dpi=200)
if show_n_samples:
    fig_n_samples, axes_n_samples = plt.subplots(2,2, figsize=(8,6), dpi=200)

for ind_mol, (molecule,plot_data) in enumerate(full_plot_data.items()):
    row, col = ind_mol // 2, ind_mol % 2
    ax = axes[row][col]

    for curve_type in ['Indep', 'Shared_75', 'Shared_95']:
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

    if molecule in speed_up_arrows:
        x1,x2 = speed_up_arrows[molecule]
        ax.axvline(x1, color=colors['Indep'], linestyle=':', alpha=1.0)
        ax.axvline(x2, color=colors['Shared_95'], linestyle=':', alpha=1.0)
        y_arrow = ylims[molecule] * 0.8
        ax.arrow(x1, y_arrow, x2 - x1, 0, head_width=y_arrow*0.05, head_length=x2*0.3, length_includes_head=True, color='dimgray')
        ax.text(np.sqrt(x1*x2), y_arrow * 1.01, f"~{x1/x2:.0f}x", ha='center', va='bottom', fontsize=12,
                # bbox=dict(facecolor='white', edgecolor='None')
                )

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

fname = f"/home/mscherbela/ucloud/results/paper_figures/jax/shared_vs_indep_{optimizer}.png"
fig.savefig(fname, dpi=400)
fig.savefig(fname.replace(".png", ".pdf"))

