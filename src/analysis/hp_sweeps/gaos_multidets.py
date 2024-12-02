import pandas as pd
from deeperwin.run_tools.load_wandb_data import load_wandb_data
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def namefilter_func(name: str):
    if name.startswith("n_dets_"):
        return True
    if name.startswith("det_"):
        return True
    if name.startswith("hp_1det_bdiag"):
        return True

tmp_fname = "/home/mscherbela/tmp/runs/gaos_multidet.csv"
df_all = load_wandb_data(
    "gao", tmp_fname, run_name_filter_func=namefilter_func, load_opt_energies=True, save=True, load_fast=True)
df_all = df_all[~df_all.E_smooth_20k.isnull()]

#%%
mapping = {
    "pre_training.n_epochs": ("n_pretraining", "Int64"),
    "model.orbitals.n_determinants": ("n_dets", "Int64"),
    "model.orbitals.determinant_schema": ("determinant_schema", str),
    "model.orbitals.transferable_atomic_orbitals.symmetrize_exponent_mlp": ("symmetry", bool),
    "model.orbitals.transferable_atomic_orbitals.backflow_width": ("backflow_width", "Int64"),
    "model.orbitals.transferable_atomic_orbitals.backflow_depth": ("backflow_depth", "Int64"),
    "model.orbitals.transferable_atomic_orbitals.use_el_ion_embeddings": ("use_el_ion", bool),
    "model.orbitals.transferable_atomic_orbitals.basis_set": ("basis_set", str),
}
for k, (name, dtype) in mapping.items():
    df_all[name] = df_all[k].astype(dtype)
df_all['model'] = 'GAO'
df_all.loc[df_all.backflow_width.isnull(), 'model'] = 'DPE'
df_all['method'] = df_all.model
df_all.loc[(df_all.model == 'GAO') & df_all.symmetry, 'method'] = 'GAO symmetrized'
df_all.loc[df_all.basis_set == "nan", 'basis_set'] = df_all.loc[df_all.basis_set == "nan", "pre_training.baseline.basis_set"]

molecule = "N2_bond_breaking"
energy_metric = "E_smooth_20k"
# energy_metric = "E_mean_50k"
colors = ["C0", "C1", "coral"]
E_NEURIPS = dict(N2_bond_breaking={50: -109.1984024, 100: -109.1987459},
                 P={50:-341.2577515, 100: -341.2583587}
                 )
E0 = E_NEURIPS[molecule][50]
df_all['energy_error'] = (df_all[energy_metric] - E0) * 1000

global_filter = df_all.molecule == molecule
# global_filter &= (df_all.basis_set == '6-31G**')
global_filter = global_filter & ((df_all.model == "DPE") | (~df_all.use_el_ion & (df_all.backflow_width == 256)))
filters = [global_filter & (df_all.determinant_schema == 'block_diag'),
           global_filter & (df_all.determinant_schema == 'full_det'),]
labels = ["Block diagonal", "Full Determinant"]

plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)

for ax, filter, label in zip(axes, filters, labels):
    df_filt = df_all[filter]
    g = sns.barplot(data=df_filt,
                x='n_dets',
                y='energy_error',
                hue='method',
                ax=ax,
                errorbar=lambda x: (x.min(), x.max()),
                capsize=0.1,
                palette=colors,
                zorder=1,
                width=0.7,
                hue_order=["DPE", "GAO", "GAO symmetrized"]
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



# df_filt = df_filt.mean()
# axes[0].bar([ind_method], 1000*(df_filt.E_smooth_20k-E0), color=colors[ind_method], label=method)
#
# axes[0].set_title("Block-diagonal")
# for ax in axes:
#     ax.legend()

