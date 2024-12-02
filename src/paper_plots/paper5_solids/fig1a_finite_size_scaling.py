# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deeperwin.utils.plotting import get_discrete_colors_from_cmap
from scipy.optimize import curve_fit

df = pd.read_csv("plot_data/fig_1a_Hchain_energy.csv")
df["weight"] = df["weight"].fillna(1.0)
df["is_reuse"] = df["is_reuse"].fillna(False)
df["E_weighted"] = df["E"] * df["weight"]
pivot = (
    df[df.method == "DeepErwin"]
    .groupby(["k_grid", "n_atoms", "is_reuse"])
    .agg(
        E_weighted=("E_weighted", "sum"),
        weight=("weight", "sum"),
        epochs=("epochs", "mean"),
        E_sigma=("E_sigma", "mean"),
    )
    .reset_index()
)
pivot["E_weighted_per_atom"] = pivot["E_weighted"] / (pivot["n_atoms"] * pivot["weight"])
pivot["E_sigma_per_atom"] = pivot["E_sigma"] / pivot["n_atoms"]
pivot["N_inv"] = 1 / pivot["n_atoms"]


def polynomial(x, a, b, c):
    return a + b * x + c**x**2


def fit_tdl_extrapolation(N_inv, E, n_points=500):
    coeffs, pcov = curve_fit(polynomial, N_inv, E)
    x = np.linspace(0, 0.1, n_points)
    E_fit = polynomial(x, *coeffs)
    return x, E_fit, np.sqrt(pcov[0, 0])


plt.close("all")
fig, ax = plt.subplots(figsize=(5, 4))

ref_methods = ["LR-DMC w/ PBC", "DeepSolid"]
colors_ref = get_discrete_colors_from_cmap(len(ref_methods), "Greys", 1.0, 0.4)
for method, color in zip(ref_methods, colors_ref):
    df_filt = df[df.method == method]
    df_filt = df_filt[df_filt.n_atoms < 1000]
    x, E_fit, sigma_Einf = fit_tdl_extrapolation(1 / df_filt.n_atoms, df_filt.E)
    label = method.replace(" w/ PBC", "")
    ax.scatter(1 / df_filt.n_atoms, df_filt.E, color=color, marker="o", label=label)
    print(f"{method:<20}: ({E_fit[0] * 1000:.2f} +- {sigma_Einf*1000:.2f}) mHa")
    ax.plot(x, E_fit, color=color, label=None)

E_afqmc = -0.56569
sigma_afqmce = 0.3e-3
ax.scatter([0], [E_afqmc], marker="+", s=100, color="brown", label="AFQMC")
ax.fill_between([-1, 1], [E_afqmc - sigma_afqmce] * 2, [E_afqmc + sigma_afqmce] * 2, color="brown", alpha=0.3)


for ind_k_grid, k_grid in enumerate([1, 4]):
    color = f"C{ind_k_grid}"
    for is_reuse in [False, True]:
        df_filt = pivot[(pivot.k_grid == k_grid) & (pivot.is_reuse == is_reuse)]
        label = "Ours"
        if k_grid > 1:
            label += " (TABC)"
        else:
            label += " ($\Gamma$-point)"
        if is_reuse:
            label = None
        ax.scatter(
            df_filt.N_inv,
            df_filt.E_weighted_per_atom,
            marker="s",
            color=color,
            label=label,
            facecolors="none" if is_reuse else None,
        )
    for modulo in [0, 2, None]:
        if (k_grid == 1) and modulo is None:
            continue
        elif (k_grid > 1) and modulo is not None:
            continue
        df_extrapolation = pivot[(pivot.k_grid == k_grid) & (pivot.n_atoms >= 10) & (pivot.epochs > 0)]
        if modulo is not None:
            df_extrapolation = df_extrapolation[df_extrapolation.n_atoms % 4 == modulo]
        x, E_fit, sigma_Einf = fit_tdl_extrapolation(df_extrapolation.N_inv, df_extrapolation.E_weighted_per_atom)
        method = f"DPE k_grid={k_grid}"
        print(f"{method:<20}: ({E_fit[0] * 1000:.2f} +- {sigma_Einf*1000:.2f}) mHa")
        ax.plot(x, E_fit, color=color, label=None)

ax.text(0.037, -0.5633, "Unfilled shells\n$N_\\mathrm{atoms}=4n$", color="C0", ha="left", va="center")
ax.text(0.047, -0.570, "Filled shells\n$N_\\mathrm{atoms}=4n+2$", color="C0", ha="left", va="center")
ax.text(0, 1.0, "a", transform=ax.transAxes, ha="left", va="bottom", fontsize=16, fontweight="bold")

ax.legend()
ax.set_xlabel(r"$1/N_\mathrm{atoms}$")
ax.set_ylabel("energy per atom / Ha")
ax.set_ylim([-0.573, -0.562])
ax.set_xlim([-0.002, 0.102])

fig.tight_layout()
fname = "plot_output/fig1a_HChains_finite_size_scaling.png"
fig.savefig(fname, dpi=200, bbox_inches="tight")
fig.savefig(fname.replace(".png", ".pdf"), bbox_inches="tight")
