import numpy as np
import matplotlib.pyplot as plt
import pickle
from paper_plots.paper1_weight_sharing.box_plot import draw_box_plot

optimizer = "kfac"
cache_fname = f"/home/mscherbela/tmp/data_shared_vs_indep_{optimizer}.pkl"
with open(cache_fname, 'rb') as f:
    full_plot_data = pickle.load(f)
plot_data = full_plot_data['Ethene']

colors = dict(Indep='C0', ReuseIndep='C0', ReuseShared95='C4', ReuseFromIndep='C2', ReuseFromIndepSingleLR='darkgreen', Shared_95='r')
markers = dict(Indep='o', ReuseIndep='s', ReuseShared95='s', ReuseFromIndep='s', ReuseFromIndepSingleLR='o', Shared_95='o')
labels = dict(Indep='Independent opt.', ReuseIndep='Independent opt. (20 geom.)',
              ReuseShared95='Reuse from 95%-shared opt.', ReuseFromIndepSingleLR="Full-weight-reuse from indep. opt.",
              Shared_95='Shared opt. (95% shared)')
titles = dict(H4p='$H_4^+$: 16 geometries', H6='$H_6$: 23 geometries', H10='$H_{10}$: 23 geometries', Ethene='Ethene: 20 geometries')

show_n_samples = True

plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(7,4), dpi=200)

for curve_type in ['Indep', 'ReuseFromIndepSingleLR', 'ReuseShared95', 'Shared_95']:
    ax.semilogx(plot_data[curve_type].n_epochs,
                plot_data[curve_type].error_mean,
                color=colors[curve_type],
                marker=markers[curve_type],
                label=labels[curve_type])
    for x, y in zip(plot_data[curve_type].n_epochs, plot_data[curve_type].errors):
        draw_box_plot(ax, x, y, width=1.15, color=colors[curve_type],
                      alpha=0.6, scale='log', center_line='mean', alpha_outlier=0.2,
                      marker=markers[curve_type])

    # ax.fill_between(plot_data[curve_type].n_epochs,
    #                 plot_data[curve_type].error_25p,
    #                 plot_data[curve_type].error_75p,
    #                 alpha=0.3,
    #                 color=colors[curve_type])

ax.set_ylim([-6, 25])
ax.set_xlim([50, 19000])
ax.grid(alpha=0.5, color='gray')
ax.axhline(1.6, color='gray', linestyle='--', label="Chem. acc.: 1 kcal/mol")
# ax.set_ylim([0, ylims[molecule]])
xticks = 2 ** np.arange(6, 15, 2)
ax.set_xticks(xticks)
ax.set_xticklabels([f"{x:,d}" for x in xticks])
ax.set_xticks(2 ** np.arange(6, 15, 1), minor=True)
ax.set_ylabel("energy rel. to MRCI / mHa")
ax.set_xlabel("training epochs / geometry")
ax.set_title("Twisted and stretched Ethene")
ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
ax.minorticks_off()
fig.tight_layout()

fname = f"/home/mscherbela/ucloud/results/paper_figures/jax/reuse_from_indep_{optimizer}.png"
fig.savefig(fname, dpi=400, bbox_inches='tight')
fig.savefig(fname.replace(".png", ".pdf"))