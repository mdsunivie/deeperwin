# %%
import pandas as pd
from deeperwin.run_tools.geometry_database import append_energies
from deeperwin.run_tools.load_wandb_data import load_full_history
from wandb import Api


def round_to_divisible_by_n(x, n=2):
    if (x % n) == 0:
        return x
    else:
        return (x + n) - (x % n)


if __name__ == "__main__":
    api = Api()
    runs = api.runs("schroedinger_univie/gao_shared_hchains")
    runs = [r for r in runs if r.name.startswith("gaoloc_hchains_v3_reuse_from64k_to4k")]

    metadata = dict(
        experiment="2023-03-09_gao_reuse_Hchains_from64k",
        source="dpe",
        method="reuse",
        embedding="dpe256",
        orbitals="gao_4fd",
        reuse_from="64kshared_24geoms_H6_H10",
        n_pretrain_variational=64_000,
    )

    data_energy = []
    for i, r in enumerate(runs):
        print(f"Loading run {i+1}/{len(runs)}")
        config, history = load_full_history(r)
        if "E_mean" not in list(history):
            continue
        eval_history = history[~history.E_mean.isnull()][["opt_epoch", "E_mean", "E_mean_sigma"]]
        for i, row in eval_history.iterrows():
            data_energy.append(
                dict(
                    geom=config["geom_hash"],
                    geom_comment=config["physical.comment"].split("__")[1],
                    molecule=config["molecule"],
                    n_pretrain_HF=config["pre_training.n_epochs"],
                    energy_type="eval",
                    epoch=round_to_divisible_by_n(row.opt_epoch, 5),
                    E=row.E_mean,
                    E_sigma=row.E_mean_sigma,
                    wandb_url=config["wandb_url"],
                    **metadata,
                )
            )

    df = pd.DataFrame(data_energy)
    full_df = append_energies(df)

# %%
