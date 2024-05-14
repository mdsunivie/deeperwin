# %%
import pandas as pd
from deeperwin.run_tools.geometry_database import load_energies

experiments_to_include = [
    "2023-05-05_tao_phisnet_indep",
    "2023-05-01_699torsion_nc_by_std_256k",
    "2023-02-26_18x20_500karxiv_v10_500k",
    "2023-05-08_dpe4_regression",
    "2023-05-08_psiformer_indep",
    "HF_CBS_234",
    "CCSD(T)_ccpCVDZ",
    "CCSD(T)_ccpCVTZ",
    "CCSD(T)_ccpCVQZ",
    "2023-02-26_18x20_128karxiv_v10_128k",
    "2023-05-ablation3_128k",
    "2023-05-ablation4_128k",
    "2023-05-ablation5_128k",
    "2023-05-01_699torsion_nc_by_std_128k",
    "2023-05-01_699torsion_nc_by_std_256k",
    "2023-05-01_699torsion_nc_by_std_256k_failure_cases",
    "2023-05-01_699torsion_nc_by_std_256k_2keval_reuseshared_Bicyclobutane",
    "ePBE0+MBD_QM7X",
    "2023-05-11_reuse_midimol_699torsions_256k_largescale_qm7",
]

columns_to_drop = [
    "n_moelcules_shared",
    "batch_size",
    "wandb_url",
    "n_pretrain_HF",
    "energy_type",
    "embedding",
    "orbitals",
    "reuse_from",
    "n_pretrain_variational",
    "n_shared_molecules",
    "epoch_geom",
]

df_all = load_energies()
df = df_all[df_all["experiment"].isin(experiments_to_include)]
df = df.drop(columns=columns_to_drop)
df = df.to_csv("/home/mscherbela/tmp/vmc_on_a_budget_energies.csv", sep=",", index=False)


# %%
