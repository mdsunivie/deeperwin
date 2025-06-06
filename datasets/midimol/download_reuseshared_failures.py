# %%
import pandas as pd
from deeperwin.run_tools.geometry_database import append_energies
from deeperwin.run_tools.load_wandb_data import load_full_history
from wandb import Api
import numpy as np

if __name__ == "__main__":
    api = Api()
    all_runs = api.runs("schroedinger_univie/gao_shared_tinymol")
    run_names = [
        "reuse_midimol_2023-05-01_699torsion_nc_by_std_256k_C3H4_dist_rotated_20geoms_C3H4_dist_rotated_20geoms",
        "reuse_midimol_2023-05-01_699torsion_nc_by_std_256k_C3H4_rotated_torsion_angle_90degrees_C3H4_rotated_torsion_angle_90degrees",
        "reuse_midimol_2023-05-01_699torsion_nc_by_std_256k_Cylobutadiene_transition_6geoms_Cylobutadiene_transition_6geoms",
        "reuse_midimol_2023-05-01_699torsion_nc_by_std_256k_N2_stretching_GerardEtAl2022_N2_stretching_GerardEtAl2022",
    ]

    runs = []
    for name in run_names:
        for r in all_runs:
            runs += [r for r in all_runs if r.name.startswith(name)]
    print(f"Found {len(runs)} matching runs")

    metadata = dict(
        experiment="2023-05-01_699torsion_nc_by_std_256k_failure_cases",
        source="dpe",
        method="reuseshared",
        embedding="gnn_phisnet",
        orbitals="phisnet_8fd",
        reuse_from="699torsions_nc_by_std_256k",
        n_pretrain_variational=256_000,
        n_shared_molecules=6,
        batch_size=2048,
    )

    data_energy = []
    for i, r in enumerate(runs):
        print(f"Loading run {i+1}/{len(runs)}")
        config, history = load_full_history(r)
        if "E_mean" not in list(history):
            continue
        eval_history = history[~history.E_mean.isnull()][["opt_epoch", "opt_n_epoch", "E_mean", "E_mean_sigma"]]
        for i, row in eval_history.reset_index().iterrows():
            if np.isnan(row.opt_n_epoch):
                total_epochs = row.opt_epoch
                geom_epochs = None
            else:
                total_epochs = row.opt_n_epoch
                geom_epochs = row.opt_epoch
            data_energy.append(
                dict(
                    geom=config["geom_hash"],
                    geom_comment=config["physical.comment"].split("__")[1],
                    molecule=config["molecule"],
                    n_pretrain_HF=config["pre_training.n_epochs"],
                    energy_type="eval",
                    epoch=total_epochs,
                    epoch_geom=geom_epochs,
                    E=row.E_mean,
                    E_sigma=row.E_mean_sigma,
                    wandb_url=config["wandb_url"],
                    **metadata,
                )
            )
    df = pd.DataFrame(data_energy)
    full_df = append_energies(df)

# %%
