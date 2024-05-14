#!/usr/bin/env python
import pandas as pd
import numpy as np
import argparse

def parse_data(data, blacklist=None):
    blacklist = blacklist or set()
    tokens = data.split("; ")
    data_dict = dict()
    geom_id = None
    epoch = None
    for t in tokens:
        key, value = t.split("=")
        if key in blacklist:
            continue
        if key == "geom_id":
            geom_id = int(value)
        elif key == "n_epoch" or key == "opt_n_epoch":
            epoch = int(value)
        else:
            try:
                data_dict[key] = float(value)
            except:
                print("Could not parse this line: " + data)
                raise
    return epoch, geom_id, data_dict


def get_nr_of_geometries(fname, n_lines_max=1_000_000):
    found_geom_ids = set()

    n_lines = 0
    with open(fname) as f:
        for line in f:
            if "geom_id=" in line:
                id = int(line.split("geom_id=")[1].split(";")[0])
                found_geom_ids.add(id)
            n_lines += 1
            if n_lines > n_lines_max:
                break
    return max(found_geom_ids) + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-geoms", type=int, default=0)
    parser.add_argument("--output", default="", required=False)
    parser.add_argument("--averaging", type=int, default=50, required=False)
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--one_line_per_epoch", default=False, action="store_true")
    parser.add_argument("fname", nargs="+")
    args = parser.parse_args()

    for fname in args.fname:
        fname_out = args.output or fname.replace(".out", "_parsed.csv")
        n_geoms = args.n_geoms or get_nr_of_geometries(fname)
        averaging = args.averaging
        METRICS_BLACKLIST = {"opt_error_E_mean",
                             "opt_error_E_mean_smooth",
                             'opt_mcmc_stepsize',
                             'opt_mcmc_acc_rate',
                             'opt_mcmc_max_age',
                             'opt_damping',
                             'opt_mcmc_delta_r_mean',
                             'opt_mcmc_delta_r_median'}

        print(f"Parsing {fname} with {n_geoms} geometries...")
        values_latest = dict()
        values_average = dict()
        full_data = []
        with open(fname) as f:
            for i,line in enumerate(f):
                is_pre_epoch = "dpe          INFO     pre Epoch" in line
                is_opt_epoch = "dpe          INFO     opt Epoch" in line
                if is_pre_epoch or is_opt_epoch:
                    ep_per_geom, data = line.split(": ")
                    ep_per_geom = int(ep_per_geom.split("Epoch")[1])
                    epoch, geom_id, data = parse_data(data, METRICS_BLACKLIST)
                    data["epoch_per_geom"] = ep_per_geom
                else:
                    continue

                if epoch == 0:
                    values_latest = {k: np.zeros(n_geoms) for k in data}
                    values_average = {k: np.zeros(n_geoms) for k in data}
                    n_values = np.zeros(n_geoms, int)

                n_values[geom_id] += 1
                for k,v in data.items():
                    values_latest[k][geom_id] = v
                    values_average[k][geom_id] += v

                if (epoch+1) % averaging == 0:
                    if args.one_line_per_epoch:
                        iterator = [geom_id]
                    else:
                        iterator = range(n_geoms)
                    for ind_g in iterator:
                        data_row = dict(epoch=epoch+1, geom_id=ind_g, phase="pre" if is_pre_epoch else "opt")
                        for k in values_average:
                            data_row[k] = values_average[k][ind_g] / n_values[ind_g]
                        full_data.append(data_row)

                    n_values = np.minimum(n_values, 1)
                    for k in values_average:
                      values_average[k][:] = values_latest[k]

                if (epoch >0) and (epoch % 50_000 == 0) and args.verbose:
                    print(f'{"pre" if is_pre_epoch else "opt"}-epoch {epoch//1000}k')

        df = pd.DataFrame(full_data)
        if "epoch" in list(df):
            print(f"{df.epoch.max()/1000:.0f}k epochs in total")
        df.to_csv(fname_out, index=False)

