#!/usr/bin/env python
from typing import Optional
import pandas as pd
import argparse

METRICS_BLACKLIST = {"opt_epoch", "opt_damping"}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="", required=False)
    parser.add_argument(
        "--blacklist",
        nargs="*",
    )
    # parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("fname")
    args = parser.parse_args()
    return args


def try_convert(val, to_type):
    try:
        return to_type(val)
    except:
        print("Could not parse this line: ", val)
        raise


def parse_data(data, blacklist: Optional[set] = None):
    blacklist = (blacklist or set()) | METRICS_BLACKLIST
    tokens = data.split("; ")
    data_dict = dict()
    for t in tokens:
        key, value = t.split("=")
        if key in blacklist:
            continue
        elif key in ["geom_id", "n_epoch", "opt_n_epoch"]:
            value = try_convert(value, int)
        else:
            value = try_convert(value, float)
        data_dict[key] = value
    return data_dict


if __name__ == "__main__":
    args = get_args()

    full_data = None
    fname_out = args.output or args.fname.replace(".out", "_parsed.csv")
    full_data = []
    with open(args.fname) as f:
        for i, line in enumerate(f):
            is_pre_epoch = "dpe          INFO     pre Epoch" in line
            is_opt_epoch = "dpe          INFO     opt Epoch" in line
            is_eval_epoch = "dpe          INFO     eval Epoch" in line
            if is_opt_epoch or is_pre_epoch or is_eval_epoch:
                ep_per_geom, data = line.split(": ")
                ep_per_geom = int(ep_per_geom.split("Epoch")[1])
                data = parse_data(data, args.blacklist)
                full_data.append(data)
        df = pd.DataFrame(full_data)
        if "epoch" in list(df):
            print(f"{df.epoch.max()/1000:.0f}k epochs in total")
        df.to_csv(fname_out, index=False)
