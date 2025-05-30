# %%
import pandas as pd
from deeperwin.run_tools.geometry_database import append_energies
from deeperwin.run_tools.load_wandb_data import load_full_history
from wandb import Api

if __name__ == "__main__":
    api = Api()
    runs = api.runs("schroedinger_univie/tao_periodic_shared")
    # runs = [r for r in runs if r.name.startswith("ferminet_indep_LiH_5geom_gamma_H_bcc_2x2x2_5geom_gamma")]
    runs = [r for r in runs if r.name.startswith("ferminet_LiH_bcc_2x2x2_5geom_around_eq_gamma")]
    metadata = dict(
        experiment="2023-10-06_ferminet",
        source="dpe",
        method="indep",
        embedding="ferminet",
        orbitals="ferminet_32fd",
        reuse_from=None,
        n_pretrain_variational=0,
        n_shared_molecules=1,
        batch_size=2048,
    )
    print(f"Found {len(runs)} matching runs")

    data_energy = []
    for i, r in enumerate(runs):
        print(f"Loading run {i+1}/{len(runs)}")
        config, history = load_full_history(r)
        if "E_mean" not in list(history):
            continue
        eval_history = history[~history.E_mean.isnull()][["opt_epoch", "E_mean", "E_mean_sigma"]]
        for i, row in eval_history.reset_index().iterrows():
            total_epochs = row.opt_epoch
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
