import pandas as pd
from deeperwin.run_tools.load_wandb_data import load_wandb_data
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
sns.set_theme()


def name_filter_func(name):
    if name.startswith("offdiag_"):
        return True
    return False


tmp_fname = "/home/mscherbela/tmp/runs/2023-01_offdiag_pretraining.csv"
df_all = load_wandb_data(
    "off_diagonal_pretraining", tmp_fname, run_name_filter_func=name_filter_func, load_opt_energies=True, save=True, load_fast=True
)
df_all['n_pretraining'] = df_all['pre_training.n_epochs']
df_all['off_diagonal_mode'] = df_all['pre_training.off_diagonal_mode']

df_neurips = pd.read_csv("/home/mscherbela/runs/references/high_accuracy_references.csv")
df_neurips = df_neurips[['molecule', 'E_DPE_NEURIPS_50k', 'E_DPE_NEURIPS_100k']]
df_all = pd.merge(df_all, df_neurips, 'left', 'molecule')
for c in list(df_all):
    if re.match("E_(smooth|mean)_\d+k", c):
        error_name = "error_" + c[2:]
        df_all[error_name] = (df_all[c] - df_all['E_DPE_NEURIPS_50k']) * 1000
df_all['n_pretraining_k'] = df_all['n_pretraining'] / 1000

#%%

df_filt = df_all[df_all.n_pretraining > 100]

plt.close("all")
fig, axes = plt.subplots(1,2, figsize=(16,7), sharey=True)

metrics = ['error_mean_10k', 'error_mean_20k']
for ax, metric in zip(axes, metrics):
    sns.lineplot(df_filt,
                 x='n_pretraining_k',
                 y=metric,
                 hue='off_diagonal_mode',
                 errorbar=lambda x: (x.min(), x.max()),
                 ax=ax
                 )
    ax.set_title(metric)
    ax.set_xscale("log")
    ax.set_xlabel("Pretraining epochs / k")
    tick_values = [1, 5, 30, 100]
    ax.set_xticks(tick_values)
    ax.set_xticklabels([str(x) for x in tick_values])
    ax.set_ylim([0, None])

fig.suptitle("NH3, DPE, 32 full-dets\nEnergies vs. NEURIPS 50k / mHa")
fig.tight_layout()
fig_fname = f"/home/mscherbela/ucloud/results/off_diagonal_mode_DPE.png"
fig.savefig(fig_fname, dpi=400, bbox_inches='tight')
fig.savefig(fig_fname.replace(".png", ".pdf"), bbox_inches='tight')
