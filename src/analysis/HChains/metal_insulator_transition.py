#%%
import wandb
from deeperwin.run_tools.geometry_database import load_geometries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from deeperwin.utils.plotting import get_discrete_colors_from_cmap
import re
from numpy.polynomial.polynomial import polyfit, polyval
import itertools

def get_localization_length(z, N):
    return (1/(2*np.pi)) * np.sqrt(-N * np.log(z**2))

df_ref = pd.read_csv("/home/mscherbela/runs/references/Motta_et_al_metal_insulator_transition.csv", sep=';')
df_ref["id_string"] = df_ref.apply(lambda row: f"{row.method}, N={row.n_atoms}, {row.source}", axis=1)
all_geoms = load_geometries()

api = wandb.Api()
runs = api.runs("schroedinger_univie/tao_HChains")

def filter_func(run):
    if re.match(r"MIT_Shared10-26_5keval_\d+_100k", run.name):
        return True
    # if re.match(r"MIT_Shared10-26_eval_\d+_50k", run.name):
    #     return True
    if re.match(r"MIT_Shared10-26_Reuse38_eval_\d+_5k", run.name):
        return True
    if re.match(r"MIT_Shared10-26_Reuse50_eval_\d+_5k", run.name):
        return True
    if re.match(r"MIT_HChainPBC22_FermiNetIndep_\d{4}", run.name):
        return True
    # if re.match(r"MIT_Shared12-28_2keval_\d+_50k", run.name):
    #     return True
    # if re.match(r"MIT_HChainPBC16_FermiNetIndep_\d+", run.name):
    #     return True
    return False
    
runs = [r for r in runs if filter_func(r)]

all_data = []
for run in runs:
    if not "loc_abs_0" in run.summary_metrics:
        continue # not evaluated yet

    # geometry
    geom = run.config["physical.comment"].split('__')[0]
    g = all_geoms[geom]

    # kgrid
    match = re.match(".*(\d+kgrid).*", run.name)
    if match:
        kgrid = int(match.groups()[0].replace("kgrid", ""))
    else:
        kgrid = 1

    # Weight for TABC
    weight = run.config.get("physical.weight_for_shared")
    weight = float(weight if weight is not None else 1.0)

    # Optimization epoch
    epoch = run.summary_metrics.get("opt_n_epoch")
    if epoch is None:
        epoch = run.summary_metrics["opt_epoch"]
    if re.match(r"MIT_Shared10-26_Reuse(38|50)_eval_\d+_5k", run.name):
        epoch = 5000

    # energy
    E = run.summary_metrics.get("E_mean", np.nan)

    is_reuse = "reuse" in run.name.lower()

    all_data.append(
        dict(
            geom=geom,
            R=g.R[1][0] - g.R[0][0],
            name=run.name,
            kgrid=kgrid,
            weight=weight,
            E=E,
            E_weighted=E * weight,
            n_atoms=run.config["physical.periodic.supercell"][0] * 2,
            z=run.summary_metrics["loc_abs_0"],
            z_weighted=run.summary_metrics["loc_abs_0"] * weight,
            use_ion_ion_features=run.config.get("model.embedding.use_ion_ion_features", False),
            epochs=epoch,
            basis_set=run.config["baseline.basis_set"],
            model=run.config["model.name"],
            is_reuse=is_reuse,
        )
    )
df = pd.DataFrame(all_data)
#%%
csv_fnames = [
    "/home/mscherbela/runs/HChain_evals/HChains12_20.csv",
    "/home/mscherbela/runs/HChain_evals/HChains40.csv",
]
extra_data = pd.concat([pd.read_csv(fname) for fname in csv_fnames])
columns = dict(loc_abs_0="z", geom="geom", weight="weight", epoch="epochs")
extra_data = extra_data[list(columns.keys())].rename(columns=columns)
extra_data["is_reuse"] = False
extra_data["model"] = "moon"
extra_data["E"] = np.nan
extra_data["E_weighted"] = np.nan
extra_data["z_weighted"] = extra_data.z * extra_data.weight
extra_data["n_atoms"] = extra_data.geom.apply(lambda g: all_geoms[g].n_atoms * all_geoms[g].periodic.supercell[0])
extra_data["R"] = extra_data.geom.apply(lambda g: all_geoms[g].R[1][0] - all_geoms[g].R[0][0])
extra_data.loc[extra_data.epochs == 0, "epochs"] = 4_000
extra_data = extra_data[extra_data.epochs.isin([200_000, 20_000])]
extra_data.loc[extra_data.n_atoms >=30, "is_reuse"] = True
df = pd.concat([df, extra_data])

#%%
groupings = ["model", "is_reuse", "n_atoms", "R", "epochs"]
pivot = df.groupby(groupings).agg(z_weighted=("z_weighted", "sum"),
                                E_weighted=("E_weighted", "sum"),
                                weight=("weight", "sum"),
                                # epochs=("epochs", "mean")
                                ).reset_index()
pivot["filled_shell"] = ((pivot.n_atoms - 2) % 4) == 0
pivot["z"] = pivot.z_weighted / pivot.weight
pivot["E"] = pivot.E_weighted / pivot.weight
pivot["loc_lambda"] = get_localization_length(pivot.z, pivot.n_atoms)
df_ref["loc_lambda"] = get_localization_length(df_ref.z, df_ref.n_atoms)

df_fn = pivot[pivot.model == "ferminet"]

plt.close("all")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

df_ref_filt = df_ref[df_ref.source.str.contains("Motta")]
id_strings = df_ref_filt.id_string.unique()

for (ax_z, ax_lambda) in axes:
    colors_ref = ["k", "gray", "darkgreen", "C2"]
    for ax, metric in zip([ax_z, ax_lambda], ["z", "loc_lambda"]):
        for id_string, color in zip(id_strings, colors_ref):
            ax.plot(df_ref[df_ref.id_string==id_string].R, 
                    df_ref[df_ref.id_string==id_string][metric], 
                    color=color, 
                    marker='o', 
                    ls='--', 
                    label=id_string)
        for n_atoms in df_fn.n_atoms.unique():
            ax.plot(df_fn[df_fn.n_atoms == n_atoms].R, 
                    df_fn[df_fn.n_atoms == n_atoms][metric], 
                    color="cyan", 
                    marker='o', 
                    ls='--', 
                    label=f"FN, N={n_atoms}, 50k steps per geom.")

for is_filled, is_reuse in itertools.product([False, True], [False, True]):
    df_filt1 = pivot[(pivot.is_reuse == is_reuse) & 
                     (pivot.filled_shell == is_filled) &
                     (pivot.model == "moon")]
    if is_filled:
        ax_z, ax_lambda = axes[0]
    else:
        ax_z, ax_lambda = axes[1]
    for epochs in sorted(df_filt1.epochs.unique()):
        df_filt = df_filt1[df_filt1.epochs == epochs]
        if len(df_filt) == 0:
            continue
        n_atom_values = sorted(df_filt.n_atoms.unique())
        cmap = {(True, False): "Blues",
                (True, True): "Purples",
                (False, False): "Reds",
                (False, True): "Oranges"}
        cmap = cmap[(is_filled, is_reuse)]
        colors = get_discrete_colors_from_cmap(len(n_atom_values), cmap, 0.5, 0.9)
        for ind_n_atom, n_atoms in enumerate(n_atom_values):
            df_plot = df_filt[df_filt.n_atoms == n_atoms]
            epochs = df_plot.epochs.mean()
            if is_reuse:
                label = f'DPE: N={n_atoms}, {df_plot.epochs.mean()//1000:.0f}k fine-tuning steps'
            elif (not is_reuse) and (ind_n_atom == 2):
                label = f"DPE: N={min(n_atom_values)}-{max(n_atom_values)}, {epochs//1000:.0f}k shared opt steps"
            else:
                label = None

            marker = 's' if is_reuse else 'D'
            # ls = '-' if (is_reuse or (epochs == 100_000)) else '--'
            ls = '-'
            color = colors[ind_n_atom]
    
            for ax, metric in zip([ax_z, ax_lambda], ["z", "loc_lambda"]):
                ax.plot(df_plot.R, 
                        df_plot[metric],
                        color=color, 
                        marker=marker,
                        ls=ls,
                        label=label,
                        )
            try:
                ind_label = 5 if is_filled else 3
                text_box = ax_z.text(df_plot.R.iloc[ind_label]+0.07, 
                        df_plot.z.iloc[ind_label], 
                        f"N={n_atoms}", 
                        color=color, 
                        ha="left", 
                        va="center")
                text_box.set_bbox(dict(facecolor='white', alpha=0.9, edgecolor='none'))
            except IndexError:
                pass
        
for ax in axes.flatten():
    ax.legend()
    ax.set_xlabel("atom spacing R / bohr")
    ax.set_xlim([0.95, 3.65])
    ax.axvline(1.7, color='gray', alpha=0.5)
for ax_z, ax_lambda in axes:
    ax_z.axhline(0, color='k', zorder=-1)
    ax_z.axhline(1, color='k', zorder=-1)
    ax_lambda.axhline(0, color='k', zorder=-1)
    ax_lambda.axhline(1, color='gray', zorder=-1)

    ax_lambda.set_ylim([0, 4.0])
    ax_z.set_ylim([-0.05, 1.05])
    ax_z.set_ylabel("$|z_N|$")
    ax_z.set_title("Complex localization measure")
    ax_lambda.set_ylabel("$\lambda / R$")
    ax_lambda.set_title("Localization length")

fig.suptitle("Metal-insulator transition in H-Chain")
fig.tight_layout()
fname = "/home/mscherbela/ucloud/results/HChain_metal_insulator_transition_TABC.png"
fig.savefig(fname, dpi=200, bbox_inches="tight")
fig.savefig(fname.replace(".png", ".pdf"), bbox_inches="tight")

#%%
fig_ext, (ax_ext, ax_zinf) = plt.subplots(1, 2, figsize=(10, 8))

poly_degree = 2
n_inv_plot = np.linspace(0, 0.1, 100)

R_values = sorted(df.R.unique())
colors = get_discrete_colors_from_cmap(len(R_values))

N_inf = 500

z_inf = []
for i, R in enumerate(R_values):
    # df_R = df[(df.R == R) & (df.n_atoms >= 14)]
    df_R = df[(df.R == R) & (df.n_atoms % 4 == 0) & (df.n_atoms < 40)]
    n_inv = 1/df_R.n_atoms
    coeffs = polyfit(n_inv, df_R.z, poly_degree)
    z_fit = polyval(n_inv_plot, coeffs)
    ax_ext.plot(n_inv, df_R.z, marker='o', label=f"R={R:.2f}", ls="none", color=colors[i])
    ax_ext.plot(n_inv_plot, z_fit, label=None, color=colors[i], alpha=0.8)
    z_inf.append(polyval(1/N_inf, coeffs))
z_inf = np.array(z_inf)
lambda_inf = get_localization_length(z_inf, N_inf)
ax_ext.set_xlabel("$1/N$")
ax_ext.set_ylabel("$|z_N|$")
ax_ext.legend()

ax_z.plot(R_values, z_inf, color="peru", marker='o', label="Extrapolation to N={N_inf}")
ax_lambda.plot(R_values, lambda_inf, color="peru", marker='o', label="Extrapolation to N={N_inf}")
ax_zinf.set_xlabel("atom spacing R / bohr")
ax_zinf.set_ylabel("$|z_\infty|$")





