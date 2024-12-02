from deeperwin.checkpoints import load_run, save_run, RunData

import argparse

parser = argparse.ArgumentParser(description="Merge checkpoints")
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--params", type=str, default=None)
parser.add_argument("--fixed_params", type=str, default=None)
parser.add_argument("--opt_state", type=str, default=None)
parser.add_argument("--mcmc_state", type=str, default=None)
parser.add_argument("--clipping_state", type=str, default=None)
parser.add_argument("--output", type=str, default=None)

# Get all filenames which must be loaded
args = parser.parse_args()
all_fnames = ["config", "params", "fixed_params", "opt_state", "mcmc_state", "clipping_state"]
fnames = {k: getattr(args, k) for k in all_fnames}
fnames = {k: v for k, v in fnames.items() if v is not None}

# Load all unique required checkpoints
data = {fname: load_run(fnames[fname]) for fname in set(fnames.values)}

# Assemble new checkpoint
new_data = RunData()
for key, fname in fnames:
    src_data = data[fname]
    setattr(new_data, key, getattr(src_data, key))

# Save new checkpoint
save_run(args.output, new_data)
