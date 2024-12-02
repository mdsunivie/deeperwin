import pandas as pd
from deeperwin.run_tools.load_wandb_data import load_wandb_data
import matplotlib.pyplot as plt
import numpy as np

tmp_fname = "/home/mscherbela/tmp/runs/hp_transformer_2022-12.csv"
df_all = load_wandb_data(
    "transformer", tmp_fname, run_name_filter_func=lambda name: name.startswith("hp_"), load_opt_energies=True, save=True, load_fast=True
)
df_all = df_all[~df_all.E_mean_50k.isnull()]


def get_param_variation(exp_name):
    return exp_name.split("N2_bond_breaking")[0][3:-1]


df_all["param_variation"] = df_all["experiment_name"].apply(get_param_variation)

mapping = {
    "optimization.optimizer.learning_rate": ("lr", float),
    "optimization.optimizer.lr_schedule.decay_time": ("lr_decay_time", int),
    "pre_training.n_epochs": ("n_pretrain", int),
    "model.embedding.el_transformer.mlp_depth": ("mlp_depth", int),
    "model.embedding.el_transformer.n_heads": ("n_heads", int),
    "model.embedding.el_transformer.use_layer_norm": ("layer_norm", bool),
}
params = [x[0] for x in mapping.values()]
defaults = dict(lr_decay_time=6000, n_pretrain=4000, n_heads=4, layer_norm=False, mlp_depth=0, lr=0.1)

for k, v in mapping.items():
    df_all[v[0]] = df_all[k].astype(v[1])
df_all = df_all.groupby(["molecule"] + params).agg(dict(E_mean_50k=("mean", "min", "max"), E_mean_100k=("mean", "min", "max"))).reset_index()
# %%
molecule = "N2_bond_breaking"
E_NEURIPS = {50: -109.1984024, 100: -109.1987459}
E_ref = E_NEURIPS[100]
pivot = df_all[df_all.molecule == "N2_bond_breaking"]
bar_width = 0.4

plt.close("all")
fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
for param_name, ax in zip(params, axes.flatten()):
    filter = np.ones(len(df_all), bool)
    for p in defaults:
        if p != param_name:
            filter = np.logical_and(filter, df_all[p] == defaults[p])
    df = df_all[(df_all.molecule == molecule) & filter]
    x_labels = [str(x_) for (x_) in df[param_name]]
    mean_50 = (df[("E_mean_50k", "mean")] - E_ref) * 1000
    yerr_50_neg = (df[("E_mean_50k", "min")] - E_ref) * 1000 - mean_50
    yerr_50_pos = (df[("E_mean_50k", "max")] - E_ref) * 1000 - mean_50
    mean_100 = (df[("E_mean_100k", "mean")] - E_ref) * 1000
    yerr_100_neg = (df[("E_mean_100k", "min")] - E_ref) * 1000 - mean_100
    yerr_100_pos = (df[("E_mean_100k", "max")] - E_ref) * 1000 - mean_100
    ind_default = x_labels.index(str(defaults[param_name]))

    error_bar_style = dict(lw=2, capsize=5, capthick=1)
    x = np.arange(len(df))
    ax.bar(
        x - bar_width,
        mean_50,
        width=bar_width,
        color="gray",
        zorder=3,
        align="edge",
        label="50k epochs",
        yerr=(yerr_50_neg.abs(), yerr_50_pos),
        error_kw=dict(**error_bar_style, ecolor='k')
    )
    ax.bar(x,
           mean_100,
           width=bar_width,
           color="C0",
           zorder=3,
           align="edge",
           label="100k epochs",
           yerr=(yerr_100_neg.abs(), yerr_100_pos),
           error_kw=dict(**error_bar_style, ecolor='navy')
           )
    ax.axvspan(ind_default - 0.5, ind_default + 0.5, label="default settings", color="green", alpha=0.2)
    for x_, e_50, e_100 in zip(x, mean_50, mean_100):
        ax.text(x_ - 0.5 * bar_width, 0.1, f"{e_50:.1f}", color="k", ha="center", va="bottom")
        ax.text(x_ + 0.5 * bar_width, 0.1, f"{e_100:.1f}", color="navy", ha="center", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.axhline((E_NEURIPS[50] - E_ref) * 1000, linestyle="--", color="darkgray", label="NIPS DPE 50k")
    ax.axhline(0, linestyle="-", color="k", zorder=5)

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    order = [2, 3, 0, 1]
    ax.legend([legend_handles[o] for o in order], [legend_labels[o] for o in order])
    ax.set_xlabel(param_name)
    ax.set_ylim([-0.5, 6])
    ax.set_ylabel("error rel. to 100k NIPS-DPE / mHa")
    ax.grid(axis="y", alpha=0.5, zorder=0)

fig.suptitle(f"Hyperparameter-tuning for {molecule}", fontsize=16)
fig.tight_layout()
fig.savefig(f"/home/mscherbela/ucloud/results/transformer_hp_tuning_{molecule}.png", dpi=400)
fig.savefig(f"/home/mscherbela/ucloud/results/transformer_hp_tuning_{molecule}.pdf")
