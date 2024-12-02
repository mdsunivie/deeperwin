import pandas as pd
from deeperwin.run_tools.load_wandb_data import load_wandb_data
import matplotlib.pyplot as plt
import numpy as np
import ast
import itertools

tmp_fname = "/home/mscherbela/tmp/runs/regression_2022-12-23.csv"
df_all = load_wandb_data("regression",
                         tmp_fname,
                         run_name_filter_func=lambda name: name.startswith("reg_2022-12-23"),
                         load_opt_energies=True,
                         load_fast=True)

def get_model_size(row):
    if not np.isnan(row['model.embedding.el_transformer.n_heads']):
        return int(row['model.embedding.el_transformer.n_heads'] * row['model.embedding.el_transformer.attention_dim'])
    else:
        n_one_el = row['model.embedding.n_hidden_one_el']
        if isinstance(n_one_el, str):
            n_one_el = ast.literal_eval(n_one_el)
        return int(n_one_el[0])

df_all['model'] = df_all['model.embedding.name']
df_all['model_size'] = df_all.apply(get_model_size, axis=1)
df_all = df_all.copy()
# %%
df_ref = pd.read_csv("/home/mscherbela/runs/references/high_accuracy_references.csv")
pivot = pd.pivot_table(df_all, index="molecule", columns=["model", "model_size"], values=["E_mean_50k", "E_mean_100k"],
                       aggfunc="mean")
column_names = {}
for n_epoch, model, model_size in itertools.product([50, 100], ["dpe4", "transformer"], [256, 512]):
    column_names[(f'E_mean_{n_epoch}k', model, model_size)] = f'E_{model}_{model_size}_{n_epoch}k'
pivot.columns = [column_names[c] for c in pivot.columns]
pivot = pd.merge(pivot, df_ref, how='left', on='molecule')

methods = ['dpe4_256_50k', 'dpe4_256_100k', 'dpe4_512_50k', 'dpe4_512_100k',
           'transformer_256_50k', 'transformer_256_100k', 'transformer_512_50k',
           'PSIFORMER_256_200k', 'PSIFORMER_512_200k']

def get_color(method):
    small = '256' in method
    if 'transformer' in method:
        return 'lightblue' if small else 'C0'
    if 'dpe4' in method:
        return 'salmon' if small else 'C1'
    if 'PSIFORMER' in method:
        return 'lightgreen' if small else 'C2'
    return 'gray'

plt.close("all")
molecules = ["NH3", "Ethene", "N2_bond_breaking", "K", "Cyclobutadiene"]
fig, axes = plt.subplots(1, 5, figsize=(17,6), sharey=True)
for mol, ax in zip(molecules, axes.flatten()):
    df = pivot[pivot.molecule == mol].iloc[0]
    for i, method in enumerate(methods):
        error = (df['E_' + method] - df['E_DPE_NEURIPS_100k']) * 1e3
        if not np.isnan(error):
            ax.barh([i], [error], color=get_color(method), zorder=3)
            ax.text(-2, i, f"{error:.1f}")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlim([-5, 5])
    ax.set_ylim([len(methods)-0.5, -0.5])
    ax.set_title(mol)
    ax.xaxis.grid(True, alpha=0.5)
    ax.set_xlabel("error rel. to NIPS-DPE 100k / mHa")
    ax.axvline(0, color='k', zorder=10)
fig.tight_layout()
fig.savefig("/home/mscherbela/ucloud/results/reg_2022-12-23.png", dpi=400)
fig.savefig("/home/mscherbela/ucloud/results/reg_2022-12-23.pdf")



