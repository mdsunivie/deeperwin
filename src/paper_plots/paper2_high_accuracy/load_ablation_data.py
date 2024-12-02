import os
import numpy as np
import wandb
import pandas as pd
from gql import gql

api = wandb.Api(timeout=30)

eval_epochs = np.array([50, 100, 200, 300]) * 1000
smooth_epochs = np.array([10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300]) * 1000


def get_config_settings(config):
    use_baseline = config.get("model.orbitals.baseline_orbitals.use_bf_factor", False)
    full_det = config["model.orbitals.use_full_det"]
    dpe_hp = abs(config["optimization.schedule.decay_time"] - 6000) < 1e-3
    use_schnet = config.get("model.embedding.use_schnet_features", False)
    use_local = config["model.features.use_local_coordinates"]
    use_init = config.get("model.orbitals.envelope_orbitals.initialization", "constant") != "constant"
    off_diag_init = config.get("model.orbitals.envelope_orbitals.initialization_off_diag", "constant") == "copy"

    if (
        (not use_baseline)
        and (not full_det)
        and (not dpe_hp)
        and (not use_schnet)
        and (not use_local)
        and (not use_init)
    ):
        return "01_fermi_iso"
    elif (not use_baseline) and full_det and (not dpe_hp) and (not use_schnet) and (not use_local) and (not use_init):
        return "02_fermi_iso_fulldet"
    elif (not use_baseline) and full_det and dpe_hp and (not use_schnet) and (not use_local) and (not use_init):
        return "03_fermi_iso_fulldet_hp"
    elif (not use_baseline) and full_det and dpe_hp and use_schnet and (not use_local) and (not use_init):
        return "04_fermi_iso_fulldet_hp_emb"
    elif (not use_baseline) and full_det and dpe_hp and use_schnet and use_local and (not use_init):
        return "05_dpe11"
    elif (not use_baseline) and full_det and dpe_hp and use_schnet and use_local and use_init and (not off_diag_init):
        return "06_dpe11_init"
    elif (not use_baseline) and full_det and dpe_hp and use_schnet and use_local and use_init and off_diag_init:
        return "06b_dpe11_init_offdiag"
    return "other"


def get_runs(project_name, entity="schroedinger_univie"):
    query = gql(
        """
    query Runs($project: String!, $entity: String!) {
        project(name: $project, entityName: $entity) {
            name
            runCount
            runs(first: 10000)
            {
                edges {
                    node {
                            name
                            host
                        }
                    }
            }
        }
    }
    """
    )

    variables = {"entity": entity, "project": project_name}
    result = api.client.execute(query, variable_values=variables)
    runs = [edge["node"] for edge in result["project"]["runs"]["edges"]]
    assert len(runs) == result["project"]["runCount"]
    return runs


def get_cluster(hostname):
    if "vsc3" in hostname:
        return "vsc3"
    elif "dgx" in hostname:
        return "dgx"
    elif "gpu1-mat" in hostname:
        return "hgx"
    else:
        return "other"


if "scherbela" in os.getlogin():
    fname = "/home/mscherbela/tmp/ablation_energies.csv"
else:
    fname = "/Users/leongerard/Desktop/data_schroedinger/ablation_energies.csv"

if os.path.isfile(fname):
    existing_data = pd.read_csv(fname)
else:
    existing_data = pd.DataFrame([], columns=["id", "state"])
existing_data.set_index("id", inplace=True)

data = []
runs = get_runs("ablation")
for i, run_dict in enumerate(runs):
    run = api.run(f'schroedinger_univie/ablation/{run_dict["name"]}')
    if (i % 5) == 0:
        print(f"{i} / {len(runs)}")
    if not run.name.startswith("ablation_"):
        continue
    if "opt_error_smooth" not in run.summary:
        continue

    if run.id in existing_data.index:
        if existing_data.loc[run.id].state in ["finished", "crashed"]:
            print(f"Skipping finished/crashed run: {run.name}")
            continue
        else:
            print("Run in existing data was not finished: reloading it")
            existing_data.drop(index=run.id, inplace=True)

    molecule = run.config["physical.name"]
    settings = run.name.split("ablation_")[1].split("_" + molecule)[0]
    repetition = int(run.name.split("rep")[-1])
    d = dict(
        id=run.id,
        name=run.name,
        state=run.state,
        molecule=molecule,
        settings=get_config_settings(run.config),
        n_epochs=run.summary["opt_epoch"],
        error_smooth_latest=run.summary["opt_error_smooth"],
        repetition=repetition,
        cluster=get_cluster(run_dict["host"]),
    )

    opt_error_smooth = [row for row in run.scan_history(keys=["opt_error_smooth", "opt_epoch"], page_size=10_000)]
    # smoothed error
    for row in opt_error_smooth:
        n_epoch = row["opt_epoch"] + 1
        if n_epoch in smooth_epochs:
            d[f"error_smooth_{n_epoch//1000}k"] = row.get("opt_error_smooth", np.nan)

    # intermed error
    intermed_error = [
        row
        for row in run.scan_history(keys=["opt_epoch", "error_intermed_eval", "sigma_intermed_eval"], page_size=10_000)
    ]
    for el in intermed_error:
        d[f"error_eval_{el['opt_epoch']//1000}k"] = el["error_intermed_eval"]
        d[f"sigma_eval_{el['opt_epoch']//1000}k"] = el["sigma_intermed_eval"]

    # final eval error
    if "error_eval" in run.summary:
        n_epochs = run.summary["opt_epoch"] + 1
        d[f"error_eval_{n_epochs // 1000}k"] = run.summary["error_eval"]
        d[f"sigma_eval_{n_epochs // 1000}k"] = run.summary["sigma_error_eval"]
        d["eval_error_final"] = run.summary["error_eval"]
        d["eval_sigma_final"] = run.summary["sigma_error_eval"]
        d["error_smooth_final"] = run.summary["opt_error_smooth"]

    # Add repetition number
    data.append(d)

df = pd.DataFrame(data)
df = pd.concat([df, existing_data.reset_index()], axis=0, ignore_index=True)

columns_to_aggregate = [f"error_eval_{n//1000}k" for n in eval_epochs] + [
    f"error_smooth_{n//1000}k" for n in smooth_epochs
]
columns_to_aggregate = [c for c in columns_to_aggregate if c in list(df)]
aggs = {k: "count" for k in columns_to_aggregate}
for cluster in df.cluster.unique():
    df[f"running_on_{cluster}"] = (df.state == "running") & (df.cluster == cluster)
    aggs[f"running_on_{cluster}"] = "sum"
df_overview = df.groupby(["molecule", "settings"]).agg(aggs).reset_index()
molecules_df = pd.DataFrame(
    [dict(molecule=m, settings=s) for m in df_overview.molecule.unique() for s in sorted(df_overview.settings.unique())]
)
df_overview = pd.merge(molecules_df, df_overview, "left", ["molecule", "settings"])
df_overview.fillna(0, inplace=True)

df.to_csv(fname, index=False)
df_overview.rename(columns={k: k.split("_")[-1] if "smooth" in k else k for k in list(df_overview)}, inplace=True)
df_overview.rename(columns={k: k.split("_")[-1] if "running_on" in k else k for k in list(df_overview)}, inplace=True)
df_overview.to_csv(fname.replace(".csv", "_overview.csv"), index=False)
df_overview.to_clipboard(index=False, sep=";")
