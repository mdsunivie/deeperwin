import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import get_cmap

df = pd.read_excel("/home/mscherbela/tmp/total_energies.ods", engine="odf")
reference = "E_all"

for key in ['var', 'all', 'dpe50', 'dpe100']:
    df[f'error_{key}'] = np.array((df[f'E_{key}'] - df[reference]) * 1e3, float)

def get_method_footnote(method_name):
    if 'FermiNet DMC' in method_name:
        return "c"
    elif method_name == 'DMC':
        return "b"
    elif "MRF" in method_name:
        return "d"
    elif method_name.startswith("FermiNet"):
        return "a"
    return None

df['method_var_footnote'] = df.method_var.apply(get_method_footnote)

molecules =  [["F", "Ne", "P", "S"],
             ["Cl", "Ar", "NH3", "CO"],
              ["H2O", "N2_4.0", "Ethene", "Cyclobutadiene"],
             ["K", "Fe", "Glycine", "Benzene"]]
for col in molecules:
    for ind_mol, molecule in enumerate(col):
        df.loc[df.molecule == molecule, 'sorting_key'] = ind_mol

labels = {"N2_4.0": "N$_2$\n$^{d=4.0}$",
          "NH3": "NH$_3$",
          "H2O": "H$_2$O",
          "Ethene": "C$_2$H$_4$",
          "Cyclobutadiene": "C$_4$H$_4$",
          # "Cyclobutadiene": "Cyclobuta-\ndiene",
          "Benzene": "C$_6$H$_6$",
          # "Glycine": "C$_2$H$_5$NO$_2$"
          }
colors = get_cmap("viridis")([0.0, 0.8, 0.95])
text_colors = ['w', 'k', 'k']

def barh_with_text(ax, x, y, text_color='k', y_range=None, method_footnotes=None, **barh_kwargs):
    y_range = y_range or (np.max(y) - np.min(y))
    value_is_zero = np.abs(y) < 0.01*y_range

    y_bar = np.array(y)
    bottom = np.zeros_like(y_bar)
    y_bar[value_is_zero] = 0.01*y_range
    bottom[value_is_zero] = -0.5*y_bar[value_is_zero]

    ax.barh(x, y_bar, left=bottom, zorder=3, **barh_kwargs)
    if y_range < 50:
        format_string = "{:.1f}"
    else:
        format_string = "{:.0f}"
    for i, (x_, y_) in enumerate(zip(x, y)):
        if not (np.isfinite(x_) and np.isfinite(y_)):
            continue
        text = format_string.format(y_)
        if method_footnotes is not None:
            text += f" $^{{({method_footnotes[i]})}}$"
        if np.abs(y_) < 0.15 * y_range:
            y_text = max(y_, 0) + y_range * 0.02
            ha = 'left'
            text_color_ = 'k'
        else:
            y_text = y_ / 2
            ha = 'center'
            text_color_ = text_color
        ax.text(y_text, x_, text, color=text_color_, va='center', ha=ha)

n_molecules_per_col = max([len(m) for m in molecules])
bar_width = 0.25
plt.close("all")
fig, axes = plt.subplots(1,len(molecules), figsize=(10,5), dpi=100)
for col, ax in enumerate(axes):
    x = np.arange(len(molecules[col]))
    df_filt = df[df.molecule.isin(molecules[col])].sort_values('sorting_key')
    y_values = df_filt[['error_var', 'error_dpe50', 'error_dpe100']].values
    y_range = np.nanmax(y_values) - np.nanmin(y_values)
    if col == 0:
        ylims = [-0.1, 2.3]
        y_range = ylims[1]-ylims[0]
        axes[col].set_xlim(ylims)
    elif col == 2:
        ylims = [-2.2, 5.5]
        y_range = ylims[1] - ylims[0]
        axes[col].set_xlim(ylims)

    barh_with_text(ax, x-bar_width, df_filt.error_var.values,
                   y_range=y_range,
                   height=bar_width, color=colors[0], text_color=text_colors[0],
                   method_footnotes=df_filt.method_var_footnote.values,
                   label="Best previously published variational results")
    barh_with_text(ax, x, df_filt.error_dpe50.values,
                   y_range=y_range,
                   height=bar_width, color=colors[1], text_color=text_colors[1],
                   label="This work, 50k steps")
    barh_with_text(ax, x+bar_width, df_filt.error_dpe100.values,
                   y_range=y_range,
                   height=bar_width, color=colors[2], text_color=text_colors[2],
                   label="This work, 100k steps")
    ax.set_yticks([])

    molecule_labels = [labels.get(m, m) for m in molecules[col]]
    # ax.set_yticklabels(molecule_labels)
    ax.set_yticklabels([])
    ax.set_ylim([-2*bar_width, (n_molecules_per_col-1) + 2*bar_width])
    xlims = ax.get_xlim()
    rel_label_pos = [0.85, 0.65, 0.8, 0.65][col]
    for i, m in enumerate(molecule_labels):
        y_text = xlims[0] + rel_label_pos * (xlims[1] - xlims[0])
        ax.text(y_text, i, m, ha='left', va='top', fontsize=12, fontweight='bold')

    ax.axvline(0, color='gray', ls='--')
    ax.set_xlabel(r"$E - E_\mathrm{ref}$ / mHa", fontsize=12)
    ax.invert_yaxis()

fig.tight_layout()
fig.subplots_adjust(top=0.93, left=0.003, right=0.99, wspace=0.03)
axes[0].legend(ncol=3, loc=(0, 1.02), fontsize=10)#, frameon=False, borderpad=0)
fname = "/home/mscherbela/ucloud/results/02_paper_figures_high_acc/total_energies.pdf"
fig.savefig(fname, bbox_inches='tight')
fig.savefig(fname.replace(".pdf", ".png"), dpi=400, bbox_inches='tight')

