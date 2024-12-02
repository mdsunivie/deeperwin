#%%
import numpy as np
import matplotlib.pyplot as plt
from deeperwin.utils.plotting import get_discrete_colors_from_cmap

N_atoms = 16

# ind_geoms = np.arange(4)
ind_geoms = np.array([0, 1, 5, 7])
all_data = []
for ind_geom in ind_geoms:
    fname = f"/home/mscherbela/tmp/structure_factors/FN_StructureFac_HChain16_{ind_geom:02d}_50k/structure_factors_050000.npz"
    raw_data = np.load(fname)

    k_vecs = raw_data["k_vecs"]
    is_only_x = np.all(k_vecs[:, 1:] == 0, axis=1) & (k_vecs[:, 0] >= 0)
    all_data.append({})
    for key in ["rho_k_up", "rho_k_dn", "rho_k_el", "rho_k_sp", "S_k_up", "S_k_dn", "S_k_el", "S_k_sp"]:
        all_data[-1][key] = raw_data[key][is_only_x]
    spacing = [1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 3.0, 3.6][ind_geom]
    L = spacing * N_atoms
    G = 2 * np.pi / spacing
    all_data[-1]["spacing"] = spacing
    all_data[-1]["kx"] = k_vecs[is_only_x, 0]
    all_data[-1]["kx_frac"] = all_data[-1]["kx"] / G


# kx = k_vecs[is_only_x, 0]
# period = (2 * np.pi / kx) / spacing

def kfrac_to_period(kfrac):
    return 1 / kfrac

def period_to_kfrac(period):
    return 1 / period

plt.close("all")
fig, axes = plt.subplots(1, 1, figsize=(6, 4))
axes = np.atleast_2d(axes)

colors = get_discrete_colors_from_cmap(len(all_data), "plasma", 0.0, 0.7)
for ind_geom, data in enumerate(all_data):
    color = colors[ind_geom]
    for ind_metric, metric in enumerate(["S"]):
        for ind_particle, particle in enumerate(["sp"]):
            key = f"{metric}_k_{particle}"
            ax = axes[ind_metric, ind_particle]
            if metric == "rho":
                # ax.plot(kx, np.real(data[key]), label="real", marker='o')
                # ax.plot(kx, np.imag(data[key]), label="imag", marker='o')
                ax.semilogy(data["kx_frac"], np.abs(data[key]), marker='o', ms=2, label=f"R={data['spacing']:.1f}", color=color)
                ax.set_ylim([1e-3, None])
            else:
                ax.plot(data["kx_frac"], data[key], marker='o', ms=2, label=f"R={data['spacing']:.1f}", color=color)
            ax.set_title(key)
            ax.set_xlabel(r"kx / ($2\pi / R$)")
            secax = ax.secondary_xaxis("top", functions=(kfrac_to_period, period_to_kfrac))
            secax.set_xlabel("period / a")
            period_ticks = [0.5, 2/3, 1, 2, 4]
            secax.set_xticks(period_ticks)
            secax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.0f}" if np.round(x) == x else f"{x:.2f}"))
            for x in period_ticks:
                ax.axvline(period_to_kfrac(x), ls="--", color="grey", alpha=0.2)
            ax.legend()
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylabel("S(k)")
            # ax.plot(period, np.abs(data[key]), label=key)
fig.tight_layout()
fig_fname = "/home/mscherbela/ucloud/results/HChain_structure_factors.png"
fig.savefig(fig_fname, dpi=200, bbox_inches="tight")
fig.savefig(fig_fname.replace(".png", ".pdf"), bbox_inches="tight")







