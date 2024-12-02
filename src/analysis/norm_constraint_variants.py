import pandas as pd
from deeperwin.run_tools.load_wandb_data import load_wandb_data
import matplotlib.pyplot as plt
import numpy as np

tmp_fname = "/home/mscherbela/tmp/runs/2023-01_norm_constraint.csv"
df_all = load_wandb_data(
    "norm_constraint", tmp_fname, run_name_filter_func=lambda name: name.startswith("nc2_"), load_opt_energies=True, save=True, load_fast=True
)
df_all = df_all[~df_all.E_mean_20k.isnull()]

def get_param_variation(row):
    s = row['norm_constraint_mode']
    if row['warmup'] > 0:
        s += f"_{row['norm_constraint']}_warmup"
    return s

mapping = {
    "optimization.optimizer.norm_constraint_mode": ("norm_constraint_mode", str),
    "optimization.optimizer.norm_constraint": ("norm_constraint", float),
    "optimization.optimizer.lr_schedule.warmup": ("warmup", int),
}
for k, v in mapping.items():
    df_all[v[0]] = df_all[k].astype(v[1])
df_all["param_variation"] = df_all.apply(get_param_variation, axis=1)

params = [p[0] for p in mapping.values()]
param_variations = df_all.param_variation.unique()
df_all = df_all.groupby(["molecule", "param_variation"] + params).agg(dict(E_mean_50k=("mean", "min", "max"), E_mean_20k=("mean", "min", "max"))).reset_index()
# %%
molecule = "N2_bond_breaking"
E_NEURIPS = dict(N2_bond_breaking={50: -109.1984024, 100: -109.1987459},
                 P={50:-341.2577515, 100: -341.2583587}
                 )[molecule]
E_ref = E_NEURIPS[100]
bar_width = 0.4

plt.close("all")
fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
for param_variation, ax in zip(param_variations, axes.flatten()):
    filter = df_all.molecule == molecule
    filter = filter & (param_variation == df_all.param_variation)
    df = df_all[filter]
    param_name = 'warmup' if 'warmup' in param_variation else 'norm_constraint'
    x_labels = [str(x_) for (x_) in df[param_name]]
    mean_20 = (df[("E_mean_20k", "mean")] - E_ref) * 1000
    yerr_20_neg = (df[("E_mean_20k", "min")] - E_ref) * 1000 - mean_20
    yerr_20_pos = (df[("E_mean_20k", "max")] - E_ref) * 1000 - mean_20
    mean_50 = (df[("E_mean_50k", "mean")] - E_ref) * 1000
    yerr_50_neg = (df[("E_mean_50k", "min")] - E_ref) * 1000 - mean_50
    yerr_50_pos = (df[("E_mean_50k", "max")] - E_ref) * 1000 - mean_50

    error_bar_style = dict(lw=2, capsize=5, capthick=1)
    x = np.arange(len(df))
    ax.bar(x-bar_width,
           mean_20,
           width=bar_width,
           color="gray",
           zorder=3,
           align="edge",
           label="20k epochs",
           # yerr=(yerr_20_neg.abs(), yerr_20_pos),
           # error_kw=dict(**error_bar_style, ecolor='navy')
           )
    ax.bar(
        x,
        mean_50,
        width=bar_width,
        color="C0",
        zorder=3,
        align="edge",
        label="50k epochs",
        # yerr=(yerr_50_neg.abs(), yerr_50_pos),
        # error_kw=dict(**error_bar_style, ecolor='k')
    )

    for x_, e_20, e_50 in zip(x, mean_20, mean_50):
        ax.text(x_ - 0.5 * bar_width, 0.1, f"{e_20:.1f}", color="k", ha="center", va="bottom")
        ax.text(x_ + 0.5 * bar_width, 0.1, f"{e_50:.1f}", color="navy", ha="center", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.axhline((E_NEURIPS[50] - E_ref) * 1000, linestyle="--", color="b", label="NIPS DPE 50k")
    ax.axhline(0, linestyle="-", color="k", zorder=5)

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    order = [1, 2, 0]
    ax.legend([legend_handles[o] for o in order], [legend_labels[o] for o in order], loc='upper right')
    ax.set_xlabel(param_name)
    ax.set_title(param_variation)
    ax.set_ylim([-0.5, 4])
    ax.set_ylabel("error rel. to 100k NIPS-DPE / mHa")
    ax.grid(axis="y", alpha=0.5, zorder=0)

fig.suptitle(f"Norm-constraint modes for {molecule}", fontsize=16)
fig.tight_layout()
fig.savefig(f"/home/mscherbela/ucloud/results/norm_constraint_mode_{molecule}.png", dpi=400)
fig.savefig(f"/home/mscherbela/ucloud/results/norm_constraint_mode_{molecule}.pdf")
