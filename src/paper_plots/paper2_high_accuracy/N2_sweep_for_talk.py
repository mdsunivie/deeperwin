import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

bond_lengths_dpe = np.array(
    [1.60151171, 2.068, 2.13534895, 2.66918618, 3.20302342, 3.73686066, 4.0, 4.27069789, 4.80453513, 5.33837237]
)
E_exp_interp_dpe = np.array(
    [
        -109.223164839783,
        -109.542171381462,
        -109.539664059474,
        -109.415670069546,
        -109.286612732924,
        -109.218059591226,
        -109.20204171463,
        -109.192461246356,
        -109.183845499452,
        -109.180767545077,
    ]
)

# dpe_data = pd.read_csv("/Users/leongerard/Desktop/plot_data/N2_sweep.csv", sep=';')
dpe_data = pd.read_csv("/home/mscherbela/tmp/N2_sweep.csv", sep=";")
dpe_data["ind_geom"] = dpe_data.name.apply(lambda x: int(re.search(r"_(\d{4})", x)[1]))
dpe_data["bond_length"] = dpe_data.ind_geom.apply(lambda x: bond_lengths_dpe[x])
dpe_data["energy_exp_interp"] = dpe_data.ind_geom.apply(lambda x: E_exp_interp_dpe[x])
dpe_data["energy_error"] = (dpe_data.E_eval_100k - dpe_data.energy_exp_interp) * 1000
dpe_data = (
    dpe_data.groupby("bond_length")
    .agg(
        energy_mean=("E_eval_100k", "mean"),
        energy_std=("E_eval_100k", "std"),
        energy_error_mean=("energy_error", "mean"),
        energy_error_std=("energy_error", "std"),
    )
    .reset_index()
)

data = pd.read_csv("/home/mscherbela/runs/references/N2/n2_with_exp_reference.csv")
data = data[["method", "bond_length", "energy", "energy_exp_interp"]]
data["method"] = data.method.map(
    {
        "CCSD": "ccsd",
        "CCSD(T)": "ccsdt",
        "FermiNet, 16 dets": "fermi_block_16_200k",
        "FermiNet, 32 dets": "fermi_block_32_200k",
        "FermiNet, 64 dets": "fermi_block_64_200k",
        "FermiNet_DMC_Ren2022": "fermi_dmc",
        "expt": "experiment",
        "r12-MR-ACPF": "mrci",
    }
)
data = data[data.method == "ccsdt"]


fontsize = 11
plt.close("all")
fig, axes = plt.subplots(1, 2, dpi=100, figsize=(7, 4))
axes[0].plot(data.bond_length, data.energy_exp_interp, color="k", label="Experiment")
axes[0].plot(data.bond_length, data.energy, color="C0", label="CCSD(T)")
axes[0].errorbar(
    dpe_data.bond_length,
    dpe_data.energy_mean,
    yerr=dpe_data.energy_std,
    color="C1",
    label="DeepErwin",
    marker="o",
    ls="None",
)
axes[0].set_xlabel("bond length / bohr", fontsize=fontsize)
axes[0].set_title("energy / Ha", fontsize=fontsize)
axes[0].yaxis.set_major_locator(plt.MaxNLocator(4))
axes[0].legend(loc="lower right")
axes[0].set_ylim([None, -109.1])


axes[1].axhline(0, color="k")
axes[1].plot(data.bond_length, (data.energy - data.energy_exp_interp) * 1000, color="C0", label="CCSD(T)")
axes[1].errorbar(
    dpe_data.bond_length,
    dpe_data.energy_error_mean,
    yerr=dpe_data.energy_error_std,
    color="C1",
    label="Deep-learning-based VMC: DeepErwin",
    marker="o",
    capsize=2,
)
axes[1].set_xlabel("bond length / bohr", fontsize=fontsize)
axes[1].set_title("energy error / mHa", fontsize=fontsize)
axes[1].set_ylim([None, 30])
fig.tight_layout()
fig.subplots_adjust(wspace=0.32)
fig.savefig("/home/mscherbela/ucloud/results/02_paper_figures_high_acc/N2_simplified.png", dpi=500, bbox_inches="tight")
fig.savefig("/home/mscherbela/ucloud/results/02_paper_figures_high_acc/N2_simplified.pdf", bbox_inches="tight")
