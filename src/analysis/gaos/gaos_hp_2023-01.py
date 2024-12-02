import pandas as pd
from deeperwin.run_tools.load_wandb_data import load_wandb_data
import matplotlib.pyplot as plt
import numpy as np

tmp_fname = "/home/mscherbela/tmp/runs/2023-01_hp_gaos.csv"
df_all = load_wandb_data(
    "gao", tmp_fname, run_name_filter_func=lambda name: name.startswith("hp_1det_bdiag"), load_opt_energies=True, save=True, load_fast=True
)
df_all = df_all[~df_all.E_mean_20k.isnull()]

mapping = {
    "pre_training.n_epochs": ("n_pretraining", int),
    "model.orbitals.transferable_atomic_orbitals.backflow_width": ("backflow_width", int),
    "model.orbitals.transferable_atomic_orbitals.backflow_depth": ("backflow_depth", int),
    "model.orbitals.transferable_atomic_orbitals.use_el_ion_embeddings": ("use_el_ion", bool),
    "model.orbitals.transferable_atomic_orbitals.basis_set": ("basis_set", str),
}
default_values = dict(n_pretraining=20_000, backflow_width=256, backflow_depth=2, use_el_ion=False, basis_set="STO-6G")

for k, (name, dtype) in mapping.items():
    df_all[name] = df_all[k]
    if dtype in [int, bool]:
        df_all.loc[df_all[name].isnull(), name] = 0
    df_all[name] = df_all[name].astype(dtype)

params = [p[0] for p in mapping.values()]
is_ref = df_all.name.str.contains("ref")
df_ref = df_all.loc[is_ref, :]
df_all = df_all.loc[~is_ref, :]
for param_name in params:
    df_all[f'include_in_{param_name}'] = True
    for other_param_name, v in default_values.items():
        if other_param_name == param_name:
            continue
        df_all[f'include_in_{param_name}'] = df_all[f'include_in_{param_name}'] & (df_all[other_param_name] == v)

# %%
molecule = "N2_bond_breaking"
df_DPE_ref = df_ref.groupby("molecule").agg(dict(E_mean_50k=("mean", "min", "max"), E_mean_20k=("mean", "min", "max")))

E_NEURIPS = dict(N2_bond_breaking={50: -109.1984024, 100: -109.1987459},
                 P={50:-341.2577515, 100: -341.2583587}
                 )
# E_ref = E_NEURIPS[100]
E_ref = df_DPE_ref.loc[molecule, ('E_mean_50k', 'mean')]
y_lims = dict(N2_bond_breaking=[0, 4], Ethene=[0,8])[molecule]

bar_width = 0.4

plt.close("all")
fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
for param_name, ax in zip(params, axes.flatten()):
    filter = df_all.molecule == molecule
    filter = filter & df_all[f'include_in_{param_name}']
    df = df_all[filter].groupby(param_name).agg(dict(E_mean_50k=("mean", "min", "max"), E_mean_20k=("mean", "min", "max"))).reset_index()

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
           yerr=(yerr_20_neg.abs(), yerr_20_pos),
           error_kw=dict(**error_bar_style, ecolor='k')
           )
    ax.bar(
        x,
        mean_50,
        width=bar_width,
        color="C0",
        zorder=3,
        align="edge",
        label="50k epochs",
        yerr=(yerr_50_neg.abs(), yerr_50_pos),
        error_kw=dict(**error_bar_style, ecolor='b')
    )

    for x_, e_20, e_50 in zip(x, mean_20, mean_50):
        ax.text(x_ - 0.5 * bar_width, y_lims[0]+0.1, f"{e_20:.1f}", color="k", ha="center", va="bottom")
        ax.text(x_ + 0.5 * bar_width, y_lims[0]+0.1, f"{e_50:.1f}", color="navy", ha="center", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.axhline((df_DPE_ref.loc[molecule, ('E_mean_20k', 'mean')] - E_ref) * 1000, linestyle="--", color="k", label="DPE 1det 20k")
    # ax.axhline((df_DPE_ref.loc[molecule, ('E_mean_50k', 'mean')] - E_ref) * 1000, linestyle="--", color="b",
    #            label="DPE 1det 50k")

    ax.axhline(0, linestyle="-", color="k", zorder=5)

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    order = [1, 2, 0]
    ax.legend([legend_handles[o] for o in order], [legend_labels[o] for o in order], loc='upper right')
    ax.set_xlabel(param_name)
    ax.set_title(param_name)
    ax.set_ylim(y_lims)
    ax.set_ylabel("error rel. to 50k DPE\n1 block-diag det / mHa")
    ax.grid(axis="y", alpha=0.5, zorder=0)

fig.suptitle(f"Generalized atomic orbitals {molecule}\n1 block-diag. determinant", fontsize=16)
fig.tight_layout()
fig.savefig(f"/home/mscherbela/ucloud/results/2023-01-19-hp_gaos_{molecule}.png", dpi=400)
fig.savefig(f"/home/mscherbela/ucloud/results/2023-01-19-hp_gaos_{molecule}.pdf")
