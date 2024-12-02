import pathlib
import subprocess
import numpy as np

import pandas as pd

import wandb


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--entity", default="schroedinger_univie")
    parser.add_argument("-E", "--prev_epoch", type=int, default=0)
    parser.add_argument("-p", "--project", default="tao_periodic_reuse")
    parser.add_argument("--add_prev_last_epoch", action="store_true")
    parser.add_argument("--geom_id", type=int, default=0)
    parser.add_argument("--parsed", action="store_true", help="use already parsed GPU.out files")
    # parser.add_argument("-i", "--run_id", default=None)
    parser.add_argument("--name_prefix", default="m")
    parser.add_argument("--group_prefix", default="manual_upload_")
    parser.add_argument("fname", nargs="+", help="GPU.out(s) to upload")
    args = parser.parse_args()
    return args


def get_parsed_path(fname):
    file = pathlib.Path(fname)
    return str(file.parent / (file.stem + "_parsed.csv"))


def convert_to_csv(gpu_out_path, *extra_args):
    run = subprocess.run(["python", "-m", "deeperwin.run_tools.logfile_to_csv_simple", gpu_out_path, *extra_args])
    if run.returncode != 0:
        import sys

        print(run)
        print("ERROR: Error in logfile_to_csv. See above")
        sys.exit(run.returncode)


def merge_parsed(parsed_fnames, add_prev_last_epoch=False):
    data = []
    df_prev = None
    for parsed_fname in parsed_fnames:
        df = pd.read_csv(parsed_fname)
        if len(data) > 0 and add_prev_last_epoch:
            df_prev = data[-1]
            df.epoch += df_prev.epoch.iloc[-1]

            for geom_id in df.geom_id.unique():
                if np.isnan(geom_id):
                    continue
                prev_epoch_per_geom = df_prev.epoch_per_geom[df_prev.geom_id == geom_id].iloc[-1]
                df.loc[df.geom_id == geom_id, "epoch_per_geom"] += prev_epoch_per_geom
        data.append(df)
    merged = pd.concat(data)
    return merged.reset_index(drop=True)


def read_physical_name_from_GPU_out(gpu_out_path):
    with open(gpu_out_path) as infile:
        txt = infile.read()
    ind1 = txt.find("physical.name=")
    expr = txt[ind1:].split(maxsplit=1)[0]
    experiment_name = expr.split("=", maxsplit=1)[1].replace(";", "")
    return experiment_name


def read_experiment_name_from_GPU_out(gpu_out_path):
    with open(gpu_out_path) as infile:
        txt = infile.read()
    ind1 = txt.find("experiment_name=")
    expr = txt[ind1:].split(maxsplit=1)[0]
    experiment_name = expr.split("=")[1].replace(";", "")
    return experiment_name


if __name__ == "__main__":
    args = get_args()

    for fname in args.fname:
        convert_to_csv(fname)

    parsed_paths = args.fname if args.parsed else [get_parsed_path(fname) for fname in args.fname]
    df = merge_parsed(parsed_paths, args.add_prev_last_epoch)

    name = read_experiment_name_from_GPU_out(args.fname[0])

    if "geom_id" not in df or np.all(df.geom_id.isna()):
        groups = ((None, df),)
    else:
        groups = df.groupby("geom_id")

    for geom_id, item in groups:
        print(f"Uploading geom_id {geom_id}")
        wandb_name = f"{args.name_prefix}{name}"
        wandb_name += f"_{geom_id}" if geom_id is not None else ""
        wandb.init(
            entity=args.entity,
            project=args.project,
            name=wandb_name,
            group=f"{args.group_prefix}{name}",
        )
        for index, line in item.T.items():
            wandb.log(line.loc[~line.isna()].to_dict())
        wandb.finish()
