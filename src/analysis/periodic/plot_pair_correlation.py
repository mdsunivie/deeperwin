#%%
import numpy as np
import matplotlib.pyplot as plt
from deeperwin.utils.plotting import get_discrete_colors_from_cmap

N_atoms = 16

# ind_geoms = np.arange(4)
ind_geoms = np.array([0, 1, 5, 7])
all_data = []
for ind_geom in ind_geoms:
    fname = f"/home/mscherbela/tmp/structure_factors/FN_StructureFac_HChain16_{ind_geom:02d}_50k/rho_2el_rel_050000.npz"
    raw_data = np.load(fname)
    
    all_data.append({})
    xmin, xmax = raw_data["r_min"][0], raw_data["r_max"][0]
    n_grid = raw_data["n_grid"][0]
    all_data[-1]["rho_same"] = raw_data["rho"][0, 0, :, 0, 0] + raw_data["rho"][1, 1, :, 0, 0]
    all_data[-1]["rho_diff"] = raw_data["rho"][1, 0, :, 0, 0] + raw_data["rho"][0, 1, :, 0, 0]
    all_data[-1]["x"] = (np.arange(n_grid) / n_grid) * (xmax - xmin) + xmin
    spacing = [1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 3.0, 3.6][ind_geom]
    all_data[-1]["spacing"] = spacing
    # L = spacing * N_atoms
    # G = 2 * np.pi / L
    # all_data[-1]["kx"] = k_vecs[is_only_x, 0]
    # all_data[-1]["kx_frac"] = all_data[-1]["kx"] / G

plt.close("all")
fig, axes = plt.subplots(1,2, figsize=(9, 5))
# colors = get_discrete_colors_from_cmap(2, "Blues", 0.4, 0.9) + get_discrete_colors_from_cmap(2, "Reds", 0.4, 0.9) 
colors = get_discrete_colors_from_cmap(len(all_data), "plasma", 0.0, 0.7)
for ind_geom, data in enumerate(all_data):
    a = data["spacing"]
    plt_kwargs = dict(label=f"R={data['spacing']:.1f}", color=colors[ind_geom])
    axes[0].plot(data["x"] / a, data["rho_same"], **plt_kwargs)
    axes[1].plot(data["x"] / a, data["rho_diff"], **plt_kwargs)
axes[0].set_title(r"$\rho(x_1 - x_2)$ same spin")
axes[1].set_title(r"$\rho(x_1 - x_2)$ different spin")

for ax in axes:
    ax.legend()
    for n in range(-8, 8+1):
        ax.axvline(n, ls="--", color="grey", alpha=0.2)
    ax.set_xlabel("$\Delta x / R$")
    ax.set_ylabel("Hist. count")

fig.suptitle("Pair correlation function (FermiNet)")
fig.tight_layout()
fig_fname = "/home/mscherbela/ucloud/results/HChain_pair_correlation_fn.png"
fig.savefig(fig_fname, dpi=200, bbox_inches="tight")
fig.savefig(fig_fname.replace(".png", ".pdf"), bbox_inches="tight")


