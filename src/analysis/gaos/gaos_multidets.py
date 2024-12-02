import pandas as pd
from deeperwin.run_tools.load_wandb_data import load_wandb_data
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

def namefilter_func(name: str):
    if name.startswith("n_dets_"):
        return True
    if name.startswith("det_"):
        return True
    if name.startswith("hp_1det_bdiag"):
        return True
    if name.startswith("ref_2023-01-25_N2_bond_breaking"):
        return True
    if name.startswith("gaosymm_2023-01-25_N2_bond_breaking"):
        return True
    return False

tmp_fname = "/home/mscherbela/tmp/runs/gaos_multidet.csv"
df_all = load_wandb_data(
    "gao", tmp_fname, run_name_filter_func=namefilter_func, load_opt_energies=True, save=True, load_fast=True)
E_NEURIPS = dict(N2_bond_breaking={50: -109.1984024, 100: -109.1987459},
                 P={50:-341.2577515, 100: -341.2583587}
                 )

#%%
mapping = {
    "pre_training.n_epochs": ("n_pretraining", "Int64"),
    "model.orbitals.n_determinants": ("n_dets", "Int64"),
    "model.orbitals.determinant_schema": ("determinant_schema", str),
    "model.orbitals.transferable_atomic_orbitals.symmetrize_exponent_mlp": ("symmetry", "boolean"),
    "model.orbitals.transferable_atomic_orbitals.backflow_width": ("backflow_width", "Int64"),
    "model.orbitals.transferable_atomic_orbitals.backflow_depth": ("backflow_depth", "Int64"),
    "model.orbitals.transferable_atomic_orbitals.use_el_ion_embeddings": ("use_el_ion", "boolean"),
    "model.orbitals.transferable_atomic_orbitals.basis_set": ("basis_set", str),
}
for k, (name, dtype) in mapping.items():
    df_all[name] = df_all[k].astype(dtype)
df_all['use_el_ion'].fillna(False, inplace=True)
df_all['model'] = 'GAO'
df_all.loc[df_all.backflow_width.isnull(), 'model'] = 'DPE'
df_all['method'] = df_all.model
df_all.loc[(df_all.model == 'GAO') & df_all.symmetry, 'method'] = 'GAO symmetrized'
df_all.loc[df_all.basis_set == "nan", 'basis_set'] = df_all.loc[df_all.basis_set == "nan", "pre_training.baseline.basis_set"]

molecule = "N2_bond_breaking"
energy_metric = "E_smooth_8k"
# energy_metric = "E_mean_50k"
colors = ["C0", "C1", "coral"]

E0 = E_NEURIPS[molecule][50]
df_all['energy_error'] = (df_all[energy_metric] - E0) * 1000

global_filter = df_all.molecule == molecule
# global_filter &= (df_all.basis_set == '6-31G**')
global_filter = global_filter & ((df_all.model == "DPE") | (~df_all.use_el_ion & (df_all.backflow_width == 256)))
filters = [global_filter & (df_all.determinant_schema == 'block_diag'),
           global_filter & (df_all.determinant_schema == 'full_det'),]
labels = ["Block diagonal", "Full Determinant"]

plt.close("all")
######################################################################
##### Nr of dets vs method ###########################################
######################################################################
fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
for ax, filter, label in zip(axes, filters, labels):
    df_filt = df_all[filter]
    g = sns.barplot(data=df_filt,
                    x='n_dets',
                    y='energy_error',
                    hue='method',
                    estimator="median",
                    ax=ax,
                    errorbar=lambda x: (x.min(), x.max()),
                    capsize=0.1,
                    palette=colors,
                    zorder=1,
                    width=0.7,
                    hue_order=["DPE", "GAO", "GAO symmetrized"],
                    )
    for p in g.patches:
        ax.text(p.get_x() + 0.5 * p.get_width(), 1.0, f"{p.get_height():.1f}", color='k', ha='center')
    sns.move_legend(g, 'upper right')
    ax.set_title(label)
    ax.set_xlabel("Nr determinants")
    ax.set_ylabel(f"{energy_metric} vs. DPE NIPS 50k / mHa")
    ax.legend(loc='upper right')
    # ax.grid(alpha=0.5, axis='y')
fig.suptitle(molecule)
fig.tight_layout()

#%% ##################################################################
##### Nr of dets vs method ###########################################
######################################################################
plt.close("all")
molecule = "N2_bond_breaking"
energy_metrics = ["E_smooth_10k", "E_mean_20k"]
n_colors = 6
palette = {2 ** i: sns.color_palette("hls", n_colors=n_colors)[i] for i in range(n_colors)}
E0 = E_NEURIPS[molecule][50]

fig, axes = plt.subplots(1, len(energy_metrics), figsize=(17, 8), sharey=True)
for ind_epoch, (energy_metric, ax) in enumerate(zip(energy_metrics, axes)):
    df_all['energy_error'] = (df_all[energy_metric] - E0) * 1000
    df_filt = df_all[global_filter & (df_all.model == "GAO") & (df_all.determinant_schema == 'full_det')].copy()
    df_filt['n_pretraining'] = df_filt.n_pretraining.astype(float)
    df_filt['n_dets'] = df_filt.n_dets.astype(int)
    df_dpe = df_all[global_filter & (df_all.method == "DPE") & (df_all.n_pretraining == 4000)].copy()
    df_dpe['n_dets'] = df_dpe['n_dets'].astype(int)
    df_dpe = df_dpe.groupby(['determinant_schema', 'n_dets']).energy_error.mean().reset_index()

    g_scatter = sns.scatterplot(data=df_filt,
                        x='n_pretraining',
                        y='energy_error',
                        hue='n_dets',
                        style='method',
                        style_order=["GAO", "GAO symmetrized"],
                        ax=ax,
                        palette=palette,
                        s=100,
                        )
    pivot = df_filt.groupby(['n_dets', 'n_pretraining']).agg(energy_error=('energy_error', 'mean')).reset_index()
    # g_lines = sns.lineplot(data=pivot,
    #                     x='n_pretraining',
    #                     y='energy_error',
    #                     hue='n_dets',
    #                     ax=ax,
    #                     palette=palette,
    #                  alpha=0.5,
    #                  legend=None
    #                     )

    ax.set_xscale("log")
    tick_values = [1000, 2000, 4000, 8000, 16000]
    ax.set_xticks(tick_values)
    ax.set_xticklabels([str(x) for x in tick_values])
    ax.set_title(f"{energy_metric} vs. DPE NIPS 50k / mHa")
    ax.set_ylim([0, 30])
    for i, row in df_dpe[df_dpe.determinant_schema == 'block_diag'].iterrows():
        ax.axhline(row.energy_error, label=f"DPE block-diag {row.n_dets:.0f} dets", color=palette[row.n_dets], ls='--')
    for i, row in df_dpe[df_dpe.determinant_schema == 'full_det'].iterrows():
        ax.axhline(row.energy_error, label=f"DPE full det {row.n_dets:.0f} dets", color=palette[row.n_dets], ls='-')
    handles_and_labels = [(h, l) for (h,l) in zip(*ax.get_legend_handles_labels()) if 'DPE' in l]

    sns.move_legend(g_scatter, loc='upper left')
    sns_legend = ax.legend_
    ax.legend(*zip(*handles_and_labels), loc='upper right')
    ax.add_artist(sns_legend)


fig.suptitle(f"{molecule}\nToo much pre-training leads to block-diag wavefunction?")
fig.tight_layout()
fig_fname = f"/home/mscherbela/ucloud/results/GAOs_multidet_{molecule}.png"
fig.savefig(fig_fname, dpi=400)
fig.savefig(fig_fname.replace(".png", ".pdf"))


