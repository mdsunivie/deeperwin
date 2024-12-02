#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deeperwin.run_tools.geometry_database import load_energies

df_all = load_energies()
experiments = [("2023-10-06_MoonTAO_shared_Hbcc_rep1", "C1", "MoonTAO shared (rep1)"),
               ("2023-10-06_MoonTAO_shared_LiH_rep1", "C1", "MoonTAO shared (rep1)"),
               ("2023-10-06_MoonTAO_shared_Hbcc_rep2", "C3", "MoonTAO shared (rep2)"),
               ("2023-10-06_MoonTAO_shared_LiH_rep2", "C3", "MoonTAO shared (rep2)"),
               ("2023-10-06_ferminet", 'gray', "FermiNet indep"),
               ]
colors = {e[0]: e[1] for e in experiments}
labels = {e[0]: e[2] for e in experiments}
experiments = [e[0] for e in experiments]


df_all = df_all[df_all.experiment.isin(experiments)]
df_ref = df_all[(df_all.embedding == "ferminet") & (df_all.epoch == 20_000)][["geom", "E"]]
df_ref = df_ref.rename(columns={"E": "E_ref"})
df_all = df_all.merge(df_ref, on="geom", how="left")
df_all["error"] = (df_all.E - df_all.E_ref) * 1000
df_all["lattice_const"] = df_all.geom_comment.str.extract("a=([0-9.]+)").astype(float)
df_all = df_all.sort_values(["molecule", "experiment", "lattice_const", "epoch_geom"])

#%%
plt.close("all")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ind_mol, molecule in enumerate(["LiH_bcc_2x2x2", "H_bcc_2x2x2"]):
    df_mol = df_all[df_all.molecule == molecule]
    for exp in df_mol.experiment.unique():
        for ind_geom, geom in enumerate(df_mol.geom.unique()):
            df_exp = df_mol[(df_mol.experiment == exp) & (df_mol.geom == geom)]
            axes[0, ind_mol].semilogx(df_exp.epoch_geom / 1000,
                                      df_exp.error, 
                                      color=colors[exp], 
                                      label=labels[exp] if ind_geom == 0 else None)
    
    x_ticks = [2, 4, 10, 20, 50]
    axes[0, ind_mol].set_xticks(x_ticks, [str(x) for x in x_ticks])
    axes[0, ind_mol].set_title(molecule)
    axes[0, ind_mol].set_xlabel("Steps per geometry / k")
    axes[0, ind_mol].grid(alpha=0.5)
    axes[0, ind_mol].set_ylabel("E - E FermiNet, 20k / mHa")
    axes[0, ind_mol].legend()

    for exp in df_mol.experiment.unique():
        df_exp = df_mol[(df_mol.experiment == exp)]
        steps_max = df_exp.epoch_geom.max()
        df_exp = df_exp[(df_exp.epoch_geom == steps_max)]
        axes[1, ind_mol].plot(df_exp.lattice_const,
                              df_exp.E, 
                              color=colors[exp], 
                              label=labels[exp] + f" ({steps_max/1000:.0f}k steps)")
        axes[1, ind_mol].set_xlabel("conv. lattice const / bohr")
        axes[1, ind_mol].set_ylabel("E / Ha")
        axes[1, ind_mol].grid(alpha=0.5)
        axes[1, ind_mol].legend()

fig.tight_layout()
fig.savefig("/home/mscherbela/ucloud/results/shared_vs_indep_periodic.png", dpi=300, bbox_inches="tight")


