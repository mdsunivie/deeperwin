# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def birch_murnaghan_fct(x, E_0, V_0, B_0, B_0_):
    x_cubic = x**3  # make lc in bohr to cubical quantity
    return E_0 + ((9 * V_0 * B_0) / 16) * (
        ((V_0 / x_cubic) ** (2 / 3) - 1) ** (3) * B_0_
        + (((V_0 / x_cubic) ** (2 / 3) - 1) ** 2 * (6 - 4 * (V_0 / x_cubic) ** (2 / 3)))
    )


def curve_fitting(x, y, bounds=([-250, 200, -np.inf, -np.inf], [-100, 800, np.inf, np.inf])):
    popt, pcov = curve_fit(birch_murnaghan_fct, x, y, bounds=bounds)
    return popt, pcov


energy_lih_far_apart = 7.47798 + 0.4997908  # energy of Li & H
cohesive_energy_fct = lambda x: (x + energy_lih_far_apart) * 1000  # convert to millihartree
geometries = [6.4, 6.8, 7.2, 7.6, 7.9, 8.3, 8.7, 9.1]  # grid of lattice constants

data_fname = "plot_data/fig_3_lih_data.csv"
df = pd.read_csv(data_fname)

# our work
df_ourwork = df[df.method == "moon_tao_lih222"]
df_ourwork_lih333 = df[df.method == "moon_tao_lih333"]

# deepsolid
df_deepsolid = df[df.method == "deepsolid_lih222"]
df_deepsolid_lih333 = df[df.method == "deepsolid_lih333"]


### Prepare data
# Add to our work per twist calculation the structure factor correction + zpve
df_ourwork["E_mean_corrected"] = df_ourwork.E_mean + df_ourwork.sfc_correction + df_ourwork.zpve

# Perform per lattice constant twist averaging, weighted by the twist weight
df_ourwork = (
    df_ourwork.groupby(by="geom").apply(lambda x: np.average(x.E_mean_corrected, weights=x.tabc_weight)).reset_index()
)
df_ourwork.rename(columns={"geom": "geom", 0: "E_mean_corrected"}, inplace=True)

# Add to DeepSolid energies hf correction + zpve
df_deepsolid["E_mean_corrected"] = df_deepsolid.E_mean + df_deepsolid.hf_correction + df_deepsolid.zpve

# Our work LiH 3x3x3: Add str. fac. correction + zpve
df_ourwork_lih333["E_mean_corrected"] = (
    df_ourwork_lih333.E_mean + df_ourwork_lih333.sfc_correction + df_ourwork_lih333.zpve
)
# Perform twist averaging
energy_our_work_lih333 = np.average(df_ourwork_lih333.E_mean_corrected, weights=df_ourwork_lih333.tabc_weight)

# DeepSolid LiH 3x3x3: Add hf correction + zpve
df_deepsolid_lih333["E_mean_corrected"] = (
    df_deepsolid_lih333.E_mean + df_deepsolid_lih333.hf_correction + df_deepsolid_lih333.zpve
)

# cohesive energy in mHa
df_deepsolid["E_coh"] = cohesive_energy_fct(df_deepsolid.E_mean_corrected)
df_ourwork["E_coh"] = cohesive_energy_fct(df_ourwork.E_mean_corrected)

cohesive_our_work_lih333 = cohesive_energy_fct(energy_our_work_lih333)
cohesive_deepsolid_lih333 = cohesive_energy_fct(df_deepsolid_lih333["E_mean_corrected"])

### Plot data
x_axis = "geom"
y_axis = "E_coh"

data = [
    (df_deepsolid, "DeepSolid (2x2x2)", "black"),
    (df_ourwork, "Our work (2x2x2)", "C1"),
]

x_linspace = np.linspace(min(geometries), max(geometries), 200)

fig, ax = plt.subplots(1, 1, figsize=(5, 4))

# Plot Final results
for df, name, color in data:
    x, y = np.array(df[x_axis]), np.array(df[y_axis])

    popt, pcov = curve_fitting(x, y)
    ax.scatter(x, y, color=color)

    fitted_curve_data = birch_murnaghan_fct(x_linspace, *popt)
    ax.plot(x_linspace, fitted_curve_data, color=color, label=name)
    print(f"Name: {name}, E_0: {popt[0]}, V_0: {popt[1]}, B_0: {popt[2]}")
    if name == "Our work (2x2x2)":
        best_guess = popt[1] ** (1 / 3)

        # Volume: V = lc**3
        # dV/dlc = 3lc**2
        # dlc = dV / (3lc**2)
        # dV represent the change / uncertainty in volume and this is converted to the change in lattice constant
        sigma = pcov[1, 1] ** (1 / 2) / (3 * best_guess**2)
    else:
        ax.scatter(
            [7.6741772396190076],  # ES lattice constant
            [cohesive_deepsolid_lih333],
            color="black",
            # alpha=0.7,
            marker="x",
            label="DeepSolid (3x3x3)",
        )


### Experiment data
# Reference for experimental data:
# Binnie et al., "Bulk and surface energetics of crystalline lithium hydride: Benchmarks from quantum Monte Carlo and quantum chemistry", 2010.
ax.fill_between(
    geometries, -175.6, -174.9, color="darkgrey", label="Experiment"
)  # experimental values go from -175.6mHa to -174.9mHa

ax.vlines(7.6741772396190076, -140, -200, color="darkgrey", linestyles="--")

ax.vlines(
    best_guess,
    -140,
    -200,
    color="C1",
    # alpha=0.5,
    linestyles="--",
)

ax.scatter(
    [7.6741772396190076],  # ES lattice constant
    [cohesive_our_work_lih333],
    color="C1",
    alpha=0.7,
    marker="x",
    label="Our work (3x3x3)",
)
ax.set_ylabel("Cohesive energy / mHa")
ax.set_xlabel(r"Lattice constant / $a_0$")
ax.legend(loc="upper right")
ax.set_ylim([-180, -140])
ax.set_xlim([min(geometries) - 0.05, max(geometries) + 0.05])

fname = "plot_output/lih_twist_avg.png"
fig.savefig(fname, dpi=300, bbox_inches="tight")
fig.savefig(fname.replace(".png", ".pdf"), bbox_inches="tight")
fig.tight_layout()
