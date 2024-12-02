import pandas as pd
from deeperwin.run_tools.load_wandb_data import load_wandb_data
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
from deeperwin.utils.utils import add_value_texts_to_barchart
sns.set_theme()


def name_filter_func(name):
    if name.startswith("ref_2023-01-25"):
        return True
    if name.startswith("gaosymm_2023-01-25"):
        return True
    if name.startswith("gaosymm_exp_2023-01-25"):
        return True
    if name.startswith("gaosymm_exp3_2023-01-25"):
        return True
    return False


def get_method(name):
    if name.startswith("gaosymm_exp3"):
        return "GAO symmetric (exp. pretraining, alpha=3)"
    if name.startswith("gaosymm_exp"):
        return "GAO symmetric (exp. pretraining, alpha=1)"
    if name.startswith("gaosymm"):
        return "GAO symmetric"
    if name.startswith("ref"):
        return "DPE"
    return "Other"

tmp_fname = "/home/mscherbela/tmp/runs/gao_reg_2023-01-25.csv"
df_all = load_wandb_data(
    "gao", tmp_fname, run_name_filter_func=name_filter_func, load_opt_energies=True, save=True, load_fast=True
)
df_all['model'] = 'GAO symm'
df_all.loc[df_all['model.orbitals.transferable_atomic_orbitals.backflow_width'].isnull(), 'model'] = 'DPE'
df_all['method'] = df_all.name.apply(get_method)

df_neurips = pd.read_csv("/home/mscherbela/runs/references/high_accuracy_references.csv")
df_neurips = df_neurips[['molecule', 'E_DPE_NEURIPS_50k', 'E_DPE_NEURIPS_100k']]
df_all = pd.merge(df_all, df_neurips, 'left', 'molecule')
for c in list(df_all):
    if re.match("E_(smooth|mean)_\d+k", c):
        error_name = "error_" + c[2:]
        df_all[error_name] = (df_all[c] - df_all['E_DPE_NEURIPS_50k']) * 1000

#%%
df_mol = df_all.groupby(['molecule']).agg(n_el=('physical.n_electrons', 'mean')).reset_index()
molecules_in_order = df_mol.sort_values('n_el')['molecule'].values

plt.close("all")
fig, axes = plt.subplots(1,2, figsize=(16,7), sharey=True)

metrics = ['error_mean_20k', 'error_mean_50k']
hue_order = ['DPE', 'GAO symmetric', 'GAO symmetric (exp. pretraining, alpha=1)', 'GAO symmetric (exp. pretraining, alpha=3)']
palette = ['C1', 'C0', 'C2', 'lightgreen']
for ax, metric in zip(axes, metrics):
    sns.barplot(df_all, x='molecule', y=metric, hue='method', ax=ax, order=molecules_in_order, hue_order=hue_order, palette=palette)
    ax.set_title(metric)
    ax.set_ylim([0, 5])
add_value_texts_to_barchart(axes, space=0.02, fontsize=10, rotation=90, color='k')
fig.suptitle("4 full-dets\nEnergies vs. NEURIPS 50k / mHa")
fig.tight_layout()
fig_fname = f"/home/mscherbela/ucloud/results/GAOs_regresssion_2023-01-05.png"
fig.savefig(fig_fname, dpi=400, bbox_inches='tight')
fig.savefig(fig_fname.replace(".png", ".pdf"), bbox_inches='tight')


