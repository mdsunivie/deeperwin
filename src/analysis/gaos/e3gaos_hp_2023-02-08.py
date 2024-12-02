import pandas as pd
from deeperwin.run_tools.load_wandb_data import load_wandb_data
import matplotlib.pyplot as plt
import numpy as np

tmp_fname = "/home/mscherbela/tmp/runs/2023-02-08_hp_e3gaos.csv"
df_all = load_wandb_data(
    "e3gao", tmp_fname, run_name_filter_func=lambda name: name.startswith("e3gao_hp_"), load_opt_energies=True, save=True, load_fast=True
)
# df_all = df_all[~df_all.E_mean_errork.isnull()]

mapping = {
    "pre_training.n_epochs": ("n_pretraining", int),
    "model.orbitals.transferable_atomic_orbitals.hidden_irreps_backflow": ("backflow_irreps", str),
    "model.orbitals.transferable_atomic_orbitals.hidden_irreps_envelope": ("envelope_irreps", str),
    "model.orbitals.transferable_atomic_orbitals.backflow_depth": ("backflow_depth", int),
    "model.orbitals.transferable_atomic_orbitals.envelope_depth": ("envelope_depth", int),
}
default_values = dict(n_pretraining=4000,
                      envelope_irreps="32x0e",
                      backflow_irreps="64x0e+32x1o",
                      backflow_depth=2,
                      envelope_depth=2
                      )

for k, (name, dtype) in mapping.items():
    df_all[name] = df_all[k]
    if dtype in [int, bool]:
        df_all.loc[df_all[name].isnull(), name] = 0
    df_all[name] = df_all[name].astype(dtype)

params = [m[0] for m in mapping.values()]
for param_name in params:
    df_all[f'include_in_{param_name}'] = True
    for other_param_name, v in default_values.items():
        if other_param_name == param_name:
            continue
        df_all[f'include_in_{param_name}'] = df_all[f'include_in_{param_name}'] & (df_all[other_param_name] == v)

# %%
molecule = "Be"
dpe_ref_energies = dict(Be=-14.667351884555815, B=-24.653799299430847)
E_ref = dpe_ref_energies[molecule]
y_lims = dict(Be=[0, 1], B=[0,2.5])[molecule]
metric = "E_mean_10k"

bar_width = 0.8
plt.close("all")
fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
for param_name, ax in zip(params, axes.flatten()):
    filter = df_all.molecule == molecule
    filter = filter & df_all[f'include_in_{param_name}']
    df = df_all[filter].groupby(param_name).agg({metric: ("mean", "min", "max")}).reset_index()

    x_labels = [str(x_) for (x_) in df[param_name]]
    mean_error = (df[(metric, "mean")] - E_ref) * 1000
    yerr_neg = (df[(metric, "min")] - E_ref) * 1000 - mean_error
    yerr_pos = (df[(metric, "max")] - E_ref) * 1000 - mean_error

    error_bar_style = dict(lw=2, capsize=5, capthick=1)
    x = np.arange(len(df))
    ax.bar(x - 0.5*bar_width,
           mean_error,
           width=bar_width,
           color="gray",
           zorder=3,
           align="edge",
           label="20k epochs",
           yerr=(yerr_neg.abs(), yerr_pos),
           error_kw=dict(**error_bar_style, ecolor='k')
           )

    for x_, y_ in zip(x, mean_error):
        ax.text(x_, y_lims[0]+0.1, f"{y_:.1f}", color="k", ha="center", va="bottom")


    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.axhline(0, linestyle="-", color="k", zorder=5)

    # legend_handles, legend_labels = ax.get_legend_handles_labels()
    # order = [1, 2, 0]
    # ax.legend([legend_handles[o] for o in order], [legend_labels[o] for o in order], loc='upper right')
    ax.set_xlabel(param_name)
    ax.set_title(param_name)
    ax.set_ylim(y_lims)
    ax.set_ylabel("error rel. to DPE / mHa")
    ax.grid(axis="y", alpha=0.5, zorder=0)

fig.suptitle(f"E3 Generalized atomic orbitals {molecule}", fontsize=16)
fig.tight_layout()
fig.savefig(f"/home/mscherbela/ucloud/results/2023-02-08-hp_e3gaos_{molecule}.png", dpi=400)
fig.savefig(f"/home/mscherbela/ucloud/results/2023-02-08-hp_e3gaos_{molecule}.pdf")
