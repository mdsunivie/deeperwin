import pandas as pd
from deeperwin.run_tools.geometry_database import append_energies
from deeperwin.run_tools.load_wandb_data import load_full_history
from wandb import Api
import numpy as np

if __name__ == "__main__":
    api = Api()
    runs = api.runs("schroedinger_univie/gao_shared_CH")
    runs = [r for r in runs if r.name.startswith("gaoloc_shared_ch_20xC2H4_v2_withCache")]
    print(f"Found {len(runs)} matching runs")

    metadata = dict(
        experiment="2023-02-23_gao_shared_20xC2H4_v2",
        source="dpe",
        method="shared",
        embedding="dpe256",
        orbitals="gao_4fd",
        reuse_from=None,
        n_pretrain_variational=0,
        n_shared_molecules=20,
    )

    data_energy = []
    for i, r in enumerate(runs):
        print(f"Loading run {i+1}/{len(runs)}")
        config, history = load_full_history(r)
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
    append_energies(df)
