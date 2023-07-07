#!/usr/bin/env python
from wandb import Api
import wandb
import argparse
import numpy as np

def copy_run(
    source_run, target_project, target_name=None, prefix="", entity="schroedinger_univie", convert=False, add_reference=False, verbose=False
):
    source_run = source_run.replace("https://wandb.ai/", "")
    if "?workspace" in source_run:
        source_run = source_run[:source_run.index("?workspace")]
    api = Api()
    source_run = api.run(source_run)

    target_name = target_name or prefix + source_run.name
    target_run = wandb.init(entity=entity, project=target_project, name=target_name)
    target_run.config.update(source_run.config)
    target_run.config["original_run"] = source_run.url

    non_summary_keys = set()
    row_nr = 0
    for row in source_run.scan_history():
        keys_to_log = [k for k in row if not k.startswith("_")]
        for k in keys_to_log:
            non_summary_keys.add(k)
        metrics_to_log = {k: row[k] for k in keys_to_log}
        if convert:
            metrics_to_log = convert_old_metrics(metrics_to_log)
        if add_reference:
            metrics_to_log = add_ref_energy(source_run.config.get("physical.name"), metrics_to_log)
        target_run.log(metrics_to_log)
        row_nr += 1
        if verbose and (row_nr % 2000) == 0:
            print(f"Uploading row {row_nr:6d}...")

    for k in source_run.summary.keys():
        if (k in non_summary_keys) or k.startswith("_"):
            continue
        target_run.summary[k] = source_run.summary[k]
    wandb.finish()


def add_ref_energy(molecule, metrics):
    E_NEURIPS2022 = dict(
        O=-75.06700553,
        F=-99.73372628,
        Ne=-128.9376113,
        P=-341.2583587,
        S=-398.1097395,
        Cl=-460.1485168,
        Ar=-527.5418717,
        K=-599.919544,
        Fe=-1263.649813,
        H2O=-76.43818434,
        NH3=-56.56377709,
        CO=-113.3242538,
        N2=-109.5413829,
        N2_bond_breaking=-109.1987459,
        Ethene=-78.58709528,
        Benzene=-232.2266754,
        Glycine=-284.4328458,
        Cyclobutadiene=-154.6790764,
    )
    if (molecule in E_NEURIPS2022) and ("opt_E_mean_smooth" in metrics):
        metrics["error_smooth_NEURIPS22"] = (metrics["opt_E_mean_smooth"] - E_NEURIPS2022[molecule]) * 1e3
    return metrics


def convert_old_metrics(metrics):
    metrics = {k: v if v is not None else np.nan for k, v in metrics.items()}
    if "opt_E_std" in metrics:
        metrics["opt_E_var_clipped"] = metrics["opt_E_std"] ** 2
        metrics["opt_E_var"] = metrics["opt_E_std_unclipped"] ** 2
    if "opt_E_mean_unclipped" in metrics:
        metrics["opt_E_mean_clipped"] = metrics["opt_E_mean"]
        metrics["opt_E_mean"] = metrics["opt_E_mean_unclipped"]
    if "eval_E_std" in metrics:
        # for eval there is no clipping => eval_E == eval_E_unclipped
        metrics["eval_E_var"] = metrics["eval_E_std"] ** 2
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Wandb identifier of run to copy, e.g. my_entity/my_project/3j7cyfux")
    parser.add_argument("target_project", help="Name of project to copy into")
    parser.add_argument(
        "--target_name", default=None, help="Name of target run. Defaults to prefix+original_name, e.g. copy_of_original_run_name"
    )
    parser.add_argument("--prefix", default="copy_of_", help="Prefix to auto-generate target name")
    parser.add_argument("--entity", default="schroedinger_univie", help="User/entity to copy into")
    parser.add_argument("--quiet", action="store_true", help="Do not print progress during copying")
    parser.add_argument(
        "--no-conversion", action="store_true", help="Do not convert metrics of old runs to equivalent new metrics (e.g. std -> var)"
    )
    parser.add_argument("--no-reference", action="store_true", help="Do not add reference energies and corresponding errors")

    args = parser.parse_args()
    copy_run(
        args.source,
        args.target_project,
        args.target_name,
        args.prefix,
        args.entity,
        not args.no_conversion,
        not args.no_reference,
        not args.quiet,
    )

if __name__ == "__main__":
    main()
