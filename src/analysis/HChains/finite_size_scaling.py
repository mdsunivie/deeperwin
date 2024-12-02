# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb
from numpy.polynomial.polynomial import polyfit, polyval
from deeperwin.utils.plotting import get_discrete_colors_from_cmap
import re

def fit_tdl_extrapolation(N_inv, E, n_points=500):
    coeffs = polyfit(N_inv, E, 2)
    x = np.linspace(0, 0.1, n_points)
    E_fit = polyval(x, coeffs)
    return x, E_fit

api = wandb.Api()
runs = api.runs("schroedinger_univie/tao_HChains")
def filter_func(run):
    # if run.name.startswith("FSS_100k_HChainPBC_4-22_1.80_"):
    #     return True
    if run.name.startswith("FSS_NoTwistBug_4-22_1.80_4kgrid_100k_"):
        return True
    if run.name.startswith("FSS_NoTwistBug_4-22_1.80_1kgrid_50k_"):
        return True
    if re.match(r"FSS_4kgrid_50k_ReuseShared_HChainPBC\d+_1.80_4kgrid_\d+", run.name):
        return True
    if re.match(r"FSS_1kgrid_50k_ReuseIndep_.*__HChainPBC\d+_1.80_k=0.000", run.name):
        return True
    return False


runs = [r for r in runs if filter_func(r)]
all_data = []
for run in runs:
    weight = run.config.get("physical.weight_for_shared")
    weight = weight or 1.0
    twist = run.config["physical.periodic.k_twist"][0]
    if ('gamma' in run.name.lower()) or ('1kgrid' in run.name.lower()):
        k_grid = 1
    elif '4kgrid' in run.name.lower():
        k_grid = 4
    else:
        k_grid = run.name.split("_")[-2].replace("kgrid", "")
    all_data.append(
        dict(
            name=run.name,
            k_grid = int(k_grid),
            E=run.summary_metrics["E_mean"],
            n_atoms=run.config["physical.periodic.supercell"][0] * 2,
            twist=twist,
            weight=weight,
            E_weighted=run.summary_metrics["E_mean"] * weight,
            epochs=run.summary_metrics.get("opt_n_epoch", run.summary_metrics["opt_epoch"]),
            is_reuse=run.config.get("reuse.path") is not None 
            # E_gamma = run.summary_metrics["opt_E_mean_smooth"] if twist == 0 else np.nan,
        )
    )
df = pd.DataFrame(all_data)
# df = df[df.n_atoms >= 6]

df_ref = pd.read_csv("/home/mscherbela/runs/references/Motta_et_al_finite_size_scaling_webplotdigitizer.csv", sep=';')
# %%
pivot = df.groupby(["k_grid", "n_atoms", "is_reuse"]).agg(E_weighted=("E_weighted", "sum"),
                                                          weight=("weight", "sum"),
                                                          epochs=("epochs", "mean")).reset_index()
# pivot["E_gamma_per_atom"] = pivot["E_gamma"] / pivot["n_atoms"]
pivot["E_weighted_per_atom"] = pivot["E_weighted"] / (pivot["n_atoms"] * pivot["weight"])
pivot["N_inv"] = 1 / pivot["n_atoms"]

plt.close("all")
fig, ax = plt.subplots(figsize=(5, 4*1.075))
# ax_inset = axis.inset_axes([0.12, 0.1, 0.4, 0.3])

ref_methods = sorted(df_ref.method.unique())
ref_methods = [m for m in ref_methods if "OBC" not in m]
colors_ref = get_discrete_colors_from_cmap(len(ref_methods), "Greys", 1.0, 0.4)
for method, color in zip(ref_methods, colors_ref):
    df_filt = df_ref[df_ref.method==method]
    df_filt = df_filt[df_filt.N_atoms < 1000]
    x, E_fit = fit_tdl_extrapolation(1/df_filt.N_atoms, df_filt.E)
    label = method.replace(" w/ PBC", "")
    ax.scatter(1/df_filt.N_atoms, df_filt.E, color=color, marker='o', label=label)
    print(f"{method:<20}: {E_fit[0] * 1000:.2f} mHa")
    ax.plot(x, E_fit, color=color, label=None)

E_afqmc = -0.56569
sigma_afqmce = 0.3e-3
ax.scatter([0], [E_afqmc], marker='+', s=100,color='brown', label="AFQMC")
ax.fill_between([-1, 1], [E_afqmc-sigma_afqmce]*2, [E_afqmc+sigma_afqmce]*2, color='brown', alpha=0.3)


# for ind_k_grid, k_grid in enumerate(pivot.k_grid.unique()):
for ind_k_grid, k_grid in enumerate([1, 4]):
    color = f"C{ind_k_grid}"
    for is_reuse in [False, True]:
        df_filt = pivot[(pivot.k_grid==k_grid) & (pivot.is_reuse == is_reuse)]
        label = "Ours"
        if k_grid > 1:
            label += " (TABC)"
        else:
            label += " ($\Gamma$-point)"
        # if is_reuse:
        #     label += " (reuse)"
        # else:
        #     label += " (shared opt.)"
        if is_reuse:
            label=None
        ax.scatter(df_filt.N_inv, df_filt.E_weighted_per_atom, marker='s', color=color, label=label, facecolors='none' if is_reuse else None)
    for modulo in [0, 2, None]:
        if (k_grid == 1) and modulo is None:
            continue
        elif (k_grid > 1) and modulo is not None:
            continue
        df_extrapolation = pivot[(pivot.k_grid==k_grid) & (pivot.n_atoms >= 10) & (pivot.epochs > 0)]
        if modulo is not None:
            df_extrapolation = df_extrapolation[df_extrapolation.n_atoms % 4 == modulo]
        x, E_fit = fit_tdl_extrapolation(df_extrapolation.N_inv, df_extrapolation.E_weighted_per_atom)
        method = f"DPE k_grid={k_grid}"
        print(f"{method:<20}: {E_fit[0] * 1000:.2f} mHa")
        ax.plot(x, E_fit, color=color, label=None)
    # ax.plot(pivot["N_inv"], pivot["E_gamma_per_atom"], "s-", label=r"DPE $\Gamma$", color='C2', lw=lw, ms=ms)
    # ax.plot(pivot["N_inv"], pivot["E_weighted_per_atom"], "s-", label="DPE TABC", color='C1', lw=lw, ms=ms)

ax.text(0.037, -0.5633, "Unfilled shells\n$N_\\mathrm{atoms}=4n$", color='C0', ha="left", va='center')
ax.text(0.047, -0.570, "Filled shells\n$N_\\mathrm{atoms}=4n+2$", color='C0', ha="left", va='center')
ax.text(0, 1.0, "a", transform=ax.transAxes, ha="left", va="bottom", fontsize=16, fontweight="bold")


# ax.text(0.05, -0.5636, "Unfilled shells\n$N_\\mathrm{atoms}=4n$", color='C0', ha="left", va='center')
# ax.text(0.065, -0.5683, "Filled shells\n$N_\\mathrm{atoms}=4n+2$", color='C0', ha="left", va='center')


ax.legend()
ax.set_xlabel(r"$1/N_\mathrm{atoms}$")
ax.set_ylabel("energy per atom / Ha")
ax.set_ylim([-0.573, -0.562])
ax.set_xlim([-0.002, 0.102])
# ax.set_title("Extrapolation of energy to thermodynamic limit")

fig.tight_layout()
fname = "/home/mscherbela/ucloud/results/05_paper_solids/figures/HChains_finite_size_scaling.png"
fig.savefig(fname, dpi=200, bbox_inches='tight')
fig.savefig(fname.replace(".png", ".pdf"), bbox_inches='tight')




