import os
import numpy as np
import wandb
import pandas as pd
from gql import gql
import re

api = wandb.Api(timeout=30)


def _get_runs(project_name, entity="schroedinger_univie"):
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
                            displayName
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


def _get_cluster(hostname):
    if "vsc3" in hostname:
        return "vsc3"
    if "vsc5" in hostname:
        return "vsc5"
    elif "dgx" in hostname:
        return "dgx"
    elif "gpu1-mat" in hostname:
        return "hgx"
    else:
        return "other"


def load_wandb_data(project, fname=None, category_func=None, run_name_filter_func=None, save=True):
    SMOOTH_EPOCHS = np.arange(1, 301) * 1000 - 2

    run_category_func = category_func or (lambda run: "default")
    run_name_filter_func = run_name_filter_func or (lambda run_name: True)

    if fname is not None:
        if os.path.isfile(fname):
            existing_data = pd.read_csv(fname, sep=";")
        else:
            existing_data = pd.DataFrame([], columns=["id", "state"])
        existing_data.set_index("id", inplace=True)

    data = []
    runs = _get_runs(project)
    for i, run_dict in enumerate(runs):
        run_id = run_dict["name"]
        if (i % 5) == 0:
            print(f"{i+1} / {len(runs)}")
        if not run_name_filter_func(run_dict["displayName"]):
            continue

        if run_dict["name"] in existing_data.index:
            if existing_data.loc[run_id].state in ["finished", "crashed"]:
                print(f"Skipping finished/crashed run: {run_dict['displayName']}")
                continue
            else:
                print("Run in existing data was not finished: reloading it")
                existing_data.drop(index=run_id, inplace=True)
        run = api.run(f'schroedinger_univie/{project}/{run_dict["name"]}')

        try:
            molecule = run.config["physical.name"]
            n_el = run.config.get("physical.n_electrons")
            Z = run.config["physical.Z"]
            charge = sum(Z) - n_el
            if charge != 0:
                molecule += f"{charge:+d}"
        except KeyError:
            print("Could not retrieve physical-config: Skipping")
            continue

        # Add repetition number
        if "rep" in run.name:
            repetition = int(run.name.split("rep")[-1].split("_")[0])
        else:
            repetition = 1
        d = dict(
            id=run.id,
            name=run.name,
            state=run.state,
            molecule=molecule,
            category=run_category_func(run),
            n_epochs=run.summary.get("opt_epoch", 0),
            error_smooth_latest=run.summary.get("opt_error_smooth", np.nan),
            repetition=repetition,
            cluster=_get_cluster(run_dict["host"]),
        )

        full_history = [row for row in run.scan_history(page_size=10_000)]
        # smoothed error
        for row in full_history:
            if ("opt_epoch" in row) and ("opt_E_mean_smooth" in row):
                n_epoch = row["opt_epoch"] + 1
                if n_epoch in SMOOTH_EPOCHS:
                    n = int(np.round(n_epoch / 1000))
                    d[f"error_smooth_{n}k"] = row.get("opt_error_smooth", np.nan)
                    d[f"E_smooth_{n}k"] = row.get("opt_E_mean_smooth", np.nan)

        # intermed error
        # intermed_error = [row for row in run.scan_history(keys=["opt_epoch", "E_intermed_eval_mean", "E_intermed_eval_std"], page_size=10_000)]
        for row in full_history:
            if ("opt_epoch" in row) and ("E_intermed_eval_mean" in row):
                n = int(np.round(row["opt_epoch"] / 1000))
                if (run.id == "s69fimge") and (n == 50):
                    continue  # exclude broken run
                # d[f"error_eval_{n}k"] = el['error_intermed_eval']
                d[f"E_eval_{n}k"] = row["E_intermed_eval_mean"]
                d[f"sigma_eval_{n}k"] = float(row["E_intermed_eval_std"]) / np.sqrt(
                    run.config["optimization.intermediate_eval.n_epochs"]
                )

        # final eval error
        if "E_mean" in run.summary:
            n_epochs = run.summary.get("opt_epoch")
            if n_epochs is None:
                n_epochs = run.config.get("optimization.n_epochs", 0) + run.config.get("optimization.n_epochs_prev", 0)
            else:
                n_epochs += 1
            n = int(np.round(n_epochs / 1000))
            d[f"E_eval_{n}k"] = run.summary.get("E_mean", np.nan)
            d[f"error_eval_{n}k"] = run.summary.get("error_eval", np.nan)
            d[f"sigma_eval_{n}k"] = run.summary.get("sigma_error_eval", np.nan)
            d["E_eval_final"] = run.summary.get("E_mean", np.nan)
            d["eval_error_final"] = run.summary.get("error_eval", np.nan)
            d["eval_sigma_final"] = run.summary.get("sigma_error_eval", np.nan)
            d["error_smooth_final"] = run.summary.get("opt_error_smooth", np.nan)
        data.append(d)
    df = pd.DataFrame(data)
    df = pd.concat([df, existing_data.reset_index()], axis=0, ignore_index=True)
    if (fname is not None) and save:
        df.to_csv(fname, index=False, sep=";")
    return df


def build_overview(df, eval_epochs=None, smooth_epochs=None):
    df = df.copy()
    columns_to_delete = []
    if eval_epochs is not None:
        for c in list(df):
            if re.match(R"E_eval_\d*k", c) and int(c.split("_")[-1][:-1]) not in eval_epochs:
                columns_to_delete.append(c)
        for n in eval_epochs:
            k = f"E_eval_{n}k"
            if k not in list(df):
                df[k] = np.nan
    if smooth_epochs is not None:
        for c in list(df):
            if re.match(R"E_smooth_\d*k", c) and (int(c.split("_")[-1][:-1]) not in smooth_epochs):
                columns_to_delete.append(c)
        for n in smooth_epochs:
            k = f"E_smooth_{n}k"
            if k not in list(df):
                df[k] = np.nan
    df = df.drop(columns=columns_to_delete)

    eval_columns = sorted([c for c in list(df) if re.findall(r"E_eval_\d*k", c)], key=lambda x: (len(x), x))
    smooth_columns = sorted([c for c in list(df) if re.findall(r"E_smooth_\d*k", c)], key=lambda x: (len(x), x))
    columns_to_aggregate = eval_columns + smooth_columns
    aggs = {k: "count" for k in columns_to_aggregate}
    df["finished"] = df.state == "finished"
    df["crashed"] = (df.state != "finished") & (df.state != "running")
    for cluster in ["vsc3", "dgx", "hgx"]:
        df[cluster] = (df.state == "running") & (df.cluster == cluster)
        aggs[cluster] = "sum"
    aggs["finished"] = "sum"
    aggs["crashed"] = "sum"
    df_overview = df.groupby(["molecule", "category"]).agg(aggs).reset_index()
    df_overview.fillna(0, inplace=True)

    df_overview.rename(columns={k: k.split("_")[-1] if "smooth" in k else k for k in list(df_overview)}, inplace=True)
    df_overview.to_clipboard(index=False, sep=";")
    return df_overview


def load_raw_data(run_id, add_id=True):
    run = api.run(run_id)

    # Optimization
    opt_history = [
        row
        for row in run.scan_history(keys=["opt_epoch", "opt_E_mean_unclipped", "opt_E_std_unclipped"], page_size=10_000)
    ]
    df_opt = pd.DataFrame(opt_history)
    df_opt["metric_type"] = "opt"

    # Evaluation (incl. intermediate eval)
    intermed_history = [
        row
        for row in run.scan_history(keys=["opt_epoch", "E_intermed_eval_mean", "E_intermed_eval_std"], page_size=10_000)
    ]
    df_eval = pd.DataFrame(intermed_history)
    df_eval["E_intermed_eval_std"] = df_eval["E_intermed_eval_std"] / np.sqrt(
        run.config["optimization.intermediate_eval.n_epochs"]
    )
    df_eval.rename(columns=dict(E_intermed_eval_mean="E_eval", E_intermed_eval_std="sigma_eval"), inplace=True)
    if "E_mean" in run.summary:
        eval_data = pd.DataFrame(
            [
                dict(
                    opt_epoch=df_opt.opt_epoch.max() + 1,
                    E_eval=run.summary["E_mean"],
                    sigma_eval=run.summary["E_mean_sigma"],
                )
            ]
        )
        df_eval = pd.concat([df_eval, eval_data], ignore_index=True)
    df_eval["metric_type"] = "eval"

    # Meta data
    df = pd.concat([df_eval, df_opt], ignore_index=True, axis=0)
    if add_id:
        df["id"] = run.id
        df["project"] = run.project
    return df


if __name__ == "__main__":
    df = load_wandb_data(
        "ablation_v2",
        fname="/home/mscherbela/tmp/ablation_v2.csv",
        run_name_filter_func=lambda x: "ablation_06_dpe11_fulldet_init_K_512_STO-6G_4000_0.004_rep2" in x,
        save=False,
    )
