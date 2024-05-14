import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import matplotlib.ticker as ticker
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

bond_lengths_dpe = np.array([1.60151171, 2.068, 2.13534895, 2.66918618, 3.20302342, 3.73686066, 4.0, 4.27069789, 4.80453513, 5.33837237])
E_exp_interp_dpe = np.array([-109.223164839783,-109.542171381462, -109.539664059474, -109.415670069546, -109.286612732924, -109.218059591226, -109.20204171463, -109.192461246356, -109.183845499452, -109.180767545077])

dpe_data = pd.read_csv("/home/mscherbela/tmp/N2_sweep.csv", sep=';')
dpe_data['ind_geom'] = dpe_data.name.apply(lambda x: int(re.search(r"_(\d{4})", x)[1]))
dpe_data['bond_length'] = dpe_data.ind_geom.apply(lambda x: bond_lengths_dpe[x])
dpe_data['energy_exp_interp'] = dpe_data.ind_geom.apply(lambda x: E_exp_interp_dpe[x])

data = pd.read_csv("/home/mscherbela/runs/references/N2/n2_with_exp_reference.csv")
data = data[['method', 'bond_length', 'energy', 'energy_exp_interp']]
data['method'] = data.method.map({'CCSD': 'ccsd',
                                  'CCSD(T)': 'ccsdt',
                                  'FermiNet, 16 dets': 'fermi_block_16_200k',
                                  'FermiNet, 32 dets': 'fermi_block_32_200k',
                                  'FermiNet, 64 dets': 'fermi_block_64_200k',
                                  'FermiNet_DMC_Ren2022': 'fermi_dmc',
                                  'expt': 'experiment',
                                  'r12-MR-ACPF': 'mrci'})
data = data[(data.bond_length >= 1.6) & (data.bond_length <= 5.4)]
data = data[(data.method != "fermi_block_32_200k") | data.bond_length.apply(lambda x: np.min(np.abs(x - bond_lengths_dpe)) < 0.05)]

for n in [50, 100]:
    key = f'E_eval_{n}k'
    filter = dpe_data.category.apply(lambda x: ('dpe' in x) and ('init' in x))
    df_dpe = dpe_data[filter].groupby(['bond_length']).agg(
        energy=(key, 'mean'), energy_std=(key, 'std'), n_runs=(key, 'count'), energy_exp_interp=('energy_exp_interp', 'mean')).reset_index()
    df_dpe['method'] = f'dpe_{n}k'
    data = pd.concat([data, df_dpe], ignore_index=True)

    df_dpe = dpe_data[dpe_data.category.str.startswith('fermi')].groupby(['bond_length']).agg(
        energy=(key, 'mean'), energy_std=(key, 'std'), n_runs=(key, 'count'), energy_exp_interp=('energy_exp_interp', 'mean')).reset_index()
    df_dpe['method'] = f'fermi_full_{n}k'
    data = pd.concat([data, df_dpe], ignore_index=True)

# data_exp = data[data.method == 'experiment'][['ind_geom', 'energy']].rename(columns=dict(energy='E_expt'))
# data = pd.merge(data, data_exp, 'left', 'ind_geom')
data['delta_to_exp'] = data.energy - data.energy_exp_interp

data = data[~data.method.isin(['HF', 'CCSD'])]


labels = dict(ccsdt='$\\mathbf{CCSD(T)}$',
              fermi_block_32_200k='$\\mathbf{FermiNet}$ $\\mathbf{block}$-$\\mathbf{det}$, 200k epochs',
              fermi_full_50k='$\\mathbf{FermiNet}$ $\\mathbf{dense}$-$\\mathbf{det}$, 50k epochs',
              fermi_full_100k='$\\mathbf{FermiNet}$ $\\mathbf{dense}$-$\\mathbf{det}$, 100k epochs',
              fermi_dmc='$\\mathbf{FermiNet\\ DMC}$, 400k epochs',
              mrci='$\\mathbf{r12}$-$\\mathbf{MR}$-$\\mathbf{ACPF}$',
              dpe_50k='$\\mathbf{This\\ work}$, 50k epochs',
              dpe_100k='$\\mathbf{This\\ work}$, 100k epochs',
)
# colors = dict(ccsdt='indigo', mrci='tab:brown',
#               fermi_block_32_200k='C0', fermi_full_50k='mediumseagreen', fermi_full_100k='green',
#               dpe_50k='C1', dpe_100k='red', fermi_dmc='gray')
# colors = [get_cmap("Paired")(i) for i in [0, 3, 4, 5, 6, 7, 9, 11]]
colors = list(get_cmap("inferno")(np.linspace(0.1, 0.67, 6))) + list(get_cmap("viridis")([0.8, 0.95]))
colors = {k:c for k,c in zip(labels, colors)}
text_colors = ['w'] * 6 + ['k'] * 2
marker_symbols = {k:v for k,v in zip(labels, ['d', 'h', 'v', '^', '*', 'x', 'o', 's'])}


method_df = data.groupby(['method']).agg({'delta_to_exp': ['min', 'max']})
method_df['rel_error'] = 1e3 * (method_df[('delta_to_exp', 'max')] - method_df[('delta_to_exp', 'min')])
#
# binding_energy_errors = []
# for method in method_df.index:
#     E1 = data.loc[(data.ind_geom == 9) & (data.method == method), 'energy'].values
#     E9 = data.loc[(data.ind_geom == 1) & (data.method == method), 'energy'].values
#     if (len(E1) == 1) and (len(E9) == 1):
#         method_df.loc[method, 'binding_energy'] = (E9 - E1)[0] * 1e3
# method_df['binding_energy_error'] = method_df['binding_energy'] - float(method_df.loc['experiment', 'binding_energy'])

def _wrap_lines(s, max_length=16, stop_char=','):
    tokens = s.split(stop_char)
    out = ""
    line_length = 0
    for token in tokens:
        if (line_length == 0) or line_length + len(token) < max_length:
            out += token
            line_length += len(token)
        else:
            out = out[:-1] + "\n" + token.lstrip()
            line_length = 0
        out += stop_char
    return out[:-1]

plt.close("all")
label_fontsize=11
title_fontsize=12
methods = list(labels.keys())
fig, (ax_sweep, ax_methods) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=100, gridspec_kw={'width_ratios': [1.5, 1]})
ax_sweep_inset = inset_axes(ax_sweep, width="100%", height="100%", loc='upper left',
                   bbox_to_anchor=(0.50,1-0.29,.45,.25), bbox_transform=ax_sweep.transAxes)

for method in methods:
    df_filt = data[data.method == method]
    if (~df_filt.delta_to_exp.isnull()).sum() == 0:
        continue
    lw = 2 if 'dpe' in method else 1
    ax_sweep.errorbar(df_filt.bond_length,
                  df_filt.delta_to_exp * 1e3,
                  label=labels.get(method, method),
                  color=colors.get(method, 'gray'),
                  marker=marker_symbols[method],
                  lw=None,
                  yerr=df_filt.energy_std*1e3,
                  capsize=2,
                  zorder=10 if 'dpe' in method else None,
                  ms=6)

ax_sweep.axhline(0, color='dimgray', label='Experiment', lw=2)

ax_sweep.grid(alpha=0.5)
ax_sweep.set_xlabel("bond length / bohr", fontsize=label_fontsize)
ax_sweep.set_ylabel(r"$E - E_\mathrm{experiment}$ / mHa", fontsize=label_fontsize)
ax_sweep.set_title("Energy error for $N_2$ dissociation curve", fontsize=title_fontsize)
ax_sweep.set_ylim([-1.5, 20.0])

ax_sweep.set_xticks([2,3,4,5])
ax_sweep.set_yticks([0,5,10,15,20])
# ax_sweep.set_xticks([1.5, 2.5, 3.5, 4.5, 5.5])
# ax_sweep.set_xticklabels([1.5, 2.5, 3.5, 4.5, 5.5])
# ax_sweep.legend(loc='upper right')
#ax_sweep_inset.set_xlabel("bond length")

ax_sweep_inset.plot(data[data['method']=='experiment'].bond_length, data[data['method']=='experiment'].energy, color='dimgray')

ax_sweep_inset.set_title("$E_\mathrm{experiment}$ / Ha", pad=-1)
#ax_sweep_inset.set_ylabel("E / Ha")
ax_sweep_inset.grid(alpha=0.5)
for item in ([ax_sweep_inset.title, ax_sweep_inset.xaxis.label, ax_sweep_inset.yaxis.label] +
             ax_sweep_inset.get_xticklabels() + ax_sweep_inset.get_yticklabels()):
    item.set_fontsize(9)

ax_sweep_inset.set_xticks([2, 3, 4, 5])
#ax_sweep_inset.set_xticks([2.0, 2.8, 3.6, 4.4, 5.2])
# ax_sweep_inset.set_xticks([1.5, 2.5, 3.5, 4.5, 5.5])
# ax_sweep_inset.set_xticklabels([])
#ax_sweep_inset.set_yticklabels([])
#ax_sweep_inset.set_yticks([])

x_methods = np.arange(len(methods))
rel_erros = [method_df.rel_error.loc[m] for m in methods]
# rel_erros = [method_df.binding_energy_error.loc[m] for m in methods]

ax_methods.barh(x_methods, rel_erros, color=[colors[m] for m in methods], zorder=3)

for i, value, text_color in zip(x_methods, rel_erros, text_colors):
    ax_methods.text(1.8, i, f"{value:2.1f}", ha='right', va='center', color=text_color)
    ax_methods.plot([0.3], [i], marker=marker_symbols[methods[i]], color=text_color, zorder=10)
ax_methods.axvline(0, color='k', zorder=20)

ax_methods.set_yticks(x_methods)
ax_methods.set_yticklabels([_wrap_lines(labels.get(m, m)) for m in methods])
ax_methods.grid(axis='x', which='major', alpha=0.5, zorder=0)
ax_methods.grid(axis='x', which='minor', alpha=0.5, zorder=0, ls='--', lw=0.5)
ax_methods.set_xlabel("Rel. error: $\\Delta E_\\mathrm{max} - \\Delta E_\\mathrm{min}$ / mHa", fontsize=label_fontsize)
ax_methods.set_title("Relative errors", fontsize=title_fontsize)
ax_methods.set_xlim([0, 10])
ax_methods.set_ylim([-0.5, len(methods)-0.5])
ax_methods.invert_yaxis()
ax_methods.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
ax_methods.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
fig.tight_layout()
fig.subplots_adjust(wspace=0.50)

fname = "/home/mscherbela/ucloud/results/02_paper_figures_high_acc/N2_dissociation.png"
fig.savefig(fname, bbox_inches='tight', dpi=600)
fig.savefig(fname.replace('.png', '.pdf'), bbox_inches='tight')




