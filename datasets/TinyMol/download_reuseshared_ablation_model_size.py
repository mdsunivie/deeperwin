import pandas as pd
from deeperwin.run_tools.geometry_database import append_energies
from deeperwin.run_tools.load_wandb_data import load_full_history
from wandb import Api
import numpy as np
import re

if __name__ == "__main__":
    api = Api()
    runs = api.runs("schroedinger_univie/gao_shared_tinymol")

    def get_model_size(run_name):
        m = re.match(r"reuseshared_tm_18x20_(S|M|XL)v(\d+)_255k_TinyMol_CNO_rot_dist", run_name)
        if m is None:
            return None
        else:
            return m.group(1)

    runs = [r for r in runs if get_model_size(r.name) is not None]
    print(f"Found {len(runs)} matching runs")

    metadata = dict(
        experiment="2023-03-24_tinymol_v20_ablation",
        source="dpe",
        method="reuseshared",
        embedding="dpe256",
        n_shared_molecules=10,
        batch_size=2048,
        n_pretrain_variational=256000,
    )

    data_energy = []
    for i, r in enumerate(runs):
        model_size = get_model_size(r.name)
        orbitals = f"gao_4fd_{model_size}"
        reuse_from = f"256kshared_tinymol_{model_size}v10"
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
                    reuse_from=reuse_from,
                    orbitals=orbitals,
                    **metadata,
                )
            )
    df = pd.DataFrame(data_energy)
    full_df = append_energies(df)
