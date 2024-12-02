import pandas as pd
from deeperwin.run_tools.load_wandb_data import load_wandb_data
import matplotlib.pyplot as plt
import numpy as np
import re
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
    if name.startswith("gasymm_4fd_pre_N2_bond_breaking"):
        return True
    return False

def get_off_diag_mode(row):
    if (row['pre_training.off_diagonal_mode'] is not None) and (str(row['pre_training.off_diagonal_mode']).lower() != "nan"):
        return row['pre_training.off_diagonal_mode']
    if (row['pre_training.block_diag_only'] is not None) and (str(row['pre_training.block_diag_only']).lower() != "nan"):
        if row['pre_training.block_diag_only']:
            return "ignore"
        else:
            return "reference"
    return "reference"

def get_method(row):
    if row.off_diagonal_mode == 'exponential':
        return f"exponential_{row.off_diagonal_scale:.1f}"
    else:
        return row.off_diagonal_mode

tmp_fname = "/home/mscherbela/tmp/runs/gao_pretraining_off_diag.csv"
df_all = load_wandb_data(
    "gao", tmp_fname, run_name_filter_func=namefilter_func, load_opt_energies=True, save=True, load_fast=True)

restart_filt = df_all.name == "gasymm_4fd_pre_N2_bond_breaking_16000_exponential_0.1_rep2_from15000"
df_all.loc[restart_filt, "pre_training.n_epochs"] = 16000
df_all.loc[restart_filt, "pre_training.off_diagonal_mode"] = "exponential"
df_all.loc[restart_filt, "pre_training.off_diagonal_scale"] = 0.1

mapping = {
    "pre_training.n_epochs": ("n_pretraining", int),
    "model.orbitals.n_determinants": ("n_dets", int),
    "model.orbitals.determinant_schema": ("determinant_schema", str),
    "model.orbitals.transferable_atomic_orbitals.symmetrize_exponent_mlp": ("symmetry", "boolean"),
    "model.orbitals.transferable_atomic_orbitals.backflow_width": ("backflow_width", "Int64"),
}
for k, (name, dtype) in mapping.items():
    df_all[name] = df_all[k].astype(dtype)

df_all['model'] = 'GAO'
df_all.loc[df_all.backflow_width.isnull(), 'model'] = 'DPE'
df_all['model_detailed'] = df_all.model
df_all.loc[(df_all.model == 'GAO') & df_all.symmetry, 'model_detailed'] = 'GAO symmetrized'
df_all['off_diagonal_mode'] = df_all.apply(get_off_diag_mode, axis=1)
df_all['off_diagonal_scale'] = df_all['pre_training.off_diagonal_scale'].fillna(1.0).astype(float)
df_all['pre_training_method'] = df_all.apply(get_method, axis=1)

df_neurips = pd.read_csv("/home/mscherbela/runs/references/high_accuracy_references.csv")
df_neurips = df_neurips[['molecule', 'E_DPE_NEURIPS_50k', 'E_DPE_NEURIPS_100k']]
df_all = pd.merge(df_all, df_neurips, 'left', 'molecule')

for c in list(df_all):
    if re.match("E_(smooth|mean)_\d+k", c):
        error_name = "error_" + c[2:]
        df_all[error_name] = (df_all[c] - df_all['E_DPE_NEURIPS_50k']) * 1000

#%%
df_filt = df_all[(df_all.molecule == 'N2_bond_breaking') &
                 (df_all.determinant_schema == 'full_det') &
                 (df_all.n_dets == 4)]
hue_order = []
colors= {'reference':'C0', 'ignore':'C1', 'exponential_1.0':'green', 'exponential_0.1': 'lime'}

plt.close("all")
metrics = ['error_smooth_7k','error_smooth_13k', 'error_mean_20k']
fig, axes = plt.subplots(1,len(metrics), figsize=(18,7), sharey=True)
for ind_ax, (metric, ax) in enumerate(zip(metrics, axes)):
    g = sns.lineplot(df_filt,
                 x='n_pretraining',
                 y=metric,
                 hue='pre_training_method',
                 style='model',
                 markers=dict(GAO='o', DPE='X'),
                 palette=colors,
                 markersize=10,
                 ax=ax,
                 errorbar=lambda x: (x.min(), x.max()),
                err_style='bars',
                 legend=(ind_ax == 0))
    ax.set_title(metric)
    ax.set_xscale('log')
    if ind_ax == 0:
        sns.move_legend(g, 'lower left')

    tick_values = [1000, 2000, 4000, 8000, 16000, 32000]
    ax.set_xticks(tick_values)
    ax.set_xticklabels([str(x) for x in tick_values])
    ax.set_ylim([0, None])

fig.suptitle("4 full-dets\nEnergies vs. NEURIPS 50k / mHa")
fig.tight_layout()
fig_fname = f"/home/mscherbela/ucloud/results/GAOs_offdiag_pretraining.png"
fig.savefig(fig_fname, dpi=400, bbox_inches='tight')
fig.savefig(fig_fname.replace(".png", ".pdf"), bbox_inches='tight')



