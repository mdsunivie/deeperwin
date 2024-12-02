# %%
import pandas as pd
from deeperwin.run_tools.geometry_database import append_energies
from deeperwin.run_tools.load_wandb_data import load_full_history
from wandb import Api
import re

if __name__ == "__main__":
    api = Api()
    runs = api.runs("schroedinger_univie/gao_shared_tinymol")

    # runs = [r for r in runs if r.name.startswith("reuse_32kopt_tinymol_v10_2023-02-26_18x20_128karxiv_v10")]
    # metadata = dict(experiment="2023-02-26_18x20_128karxiv_v10_128k",
    #                 source="dpe",
    #                 method="reuse",
    #                 embedding="dpe4",
    #                 orbitals="gao_4fd",
    #                 reuse_from="18x20_v10_128k",
    #                 n_pretrain_variational=128_000,
    #                 n_shared_molecules=1,
    #                 batch_size=2048,
    #                 )

    runs = [r for r in runs if re.match(r"reuse_midimol_2023-05-01_699torsion_nc_by_std_480k_\dheavy.*", r.name)]
    metadata = dict(
        experiment="2023-05-01_699torsion_nc_by_std_480k",
        source="dpe",
        method="reuse",
        embedding="gnn_phisnet",
        orbitals="phisnet_8fd",
        reuse_from="699torsions_nc_by_std_480k",
        n_pretrain_variational=480_000,
        n_shared_molecules=1,
        batch_size=2048,
    )

    # runs = [r for r in runs if re.match(r"reuse_tinymol_v10_2023-02-26_18x20_500karxiv_v10_\dheavy.*", r.name)]
    # metadata = dict(experiment="2023-02-26_18x20_500karxiv_v10_500k",
    #                 source="dpe",
    #                 method="reuse",
    #                 embedding="dpe4",
    #                 orbitals="gao_4fd",
    #                 reuse_from="18x20_v10_500k",
    #                 n_pretrain_variational=500_000,
    #                 n_shared_molecules=1,
    #                 batch_size=2048,
    #                 )

    ########################################
    ######## ABLATION STUDIES ##############
    ########################################

    # runs = [r for r in runs if r.name.startswith("reuse_midimol_2023-05-08_ablation2a_phisnet_without_nodes_128k")]
    # metadata = dict(experiment="2023-05-ablation2a_128k",
    #                 source="dpe",
    #                 method="reuse",
    #                 embedding="gnn",
    #                 orbitals="phisnet_4fd",
    #                 reuse_from="ablation2a_128k",
    #                 n_pretrain_variational=128_000,
    #                 n_shared_molecules=1,
    #                 batch_size=2048,
    #                 )

    # runs = [r for r in runs if r.name.startswith("reuse_midimol_2023-04-29_ablation3_phisnet_128k")]
    # metadata = dict(experiment="2023-05-ablation3_128k",
    #                 source="dpe",
    #                 method="reuse",
    #                 embedding="gnn_phisnet",
    #                 orbitals="phisnet_4fd",
    #                 reuse_from="ablation3_128k",
    #                 n_pretrain_variational=128_000,
    #                 n_shared_molecules=1,
    #                 batch_size=2048,
    #                 )

    # runs = [r for r in runs if r.name.startswith("reuse_midimol_2023-05-08_ablation3_phisnet_rep2_128k")]
    # metadata = dict(experiment="2023-05-ablation3_128k_rep2",
    #                 source="dpe",
    #                 method="reuse",
    #                 embedding="gnn_phisnet",
    #                 orbitals="phisnet_4fd",
    #                 reuse_from="ablation3_128k",
    #                 n_pretrain_variational=128_000,
    #                 n_shared_molecules=1,
    #                 batch_size=2048,
    #                 )

    # runs = [r for r in runs if r.name.startswith("reuse_midimol_2023-05-08_ablation4_distortion_128k")]
    # metadata = dict(experiment="2023-05-ablation4_128k",
    #                 source="dpe",
    #                 method="reuse",
    #                 embedding="gnn_phisnet",
    #                 orbitals="phisnet_4fd",
    #                 reuse_from="ablation4_128k",
    #                 n_pretrain_variational=128_000,
    #                 n_shared_molecules=1,
    #                 batch_size=2048,
    #                 )

    # runs = [r for r in runs if r.name.startswith("reuse_midimol_2023-05-08_ablation5_compounds_128k")]
    # metadata = dict(experiment="2023-05-ablation5_128k",
    #                 source="dpe",
    #                 method="reuse",
    #                 embedding="gnn_phisnet",
    #                 orbitals="phisnet_4fd",
    #                 reuse_from="ablation5_128k",
    #                 n_pretrain_variational=128_000,
    #                 n_shared_molecules=1,
    #                 batch_size=2048,
    #                 )
    # print(f"Found {len(runs)} matching runs")

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
