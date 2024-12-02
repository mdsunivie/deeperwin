#!/usr/bin/env python
import h5py
import argparse
import numpy as np

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=str)
parser.add_argument("output_file", type=str)
parser.add_argument("n_samples", type=int)
args = parser.parse_args()
assert args.input_file != args.output_file, "Input and output file must be distinct"

with h5py.File(args.output_file, "w") as f_out:
    with h5py.File(args.input_file, "r") as f_in:
        keys = list(f_in.keys())
        n_keys_orig = len(keys)
        keys = np.random.choice(keys, args.n_samples, replace=False)
        print(f"Copying {args.n_samples}/{n_keys_orig}...")
        for k in keys:
            f_in.copy(k, f_out, expand_external=True)
