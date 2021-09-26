"""
CLI for wandb sweeps.
"""

#!/usr/bin/env python
import os
import subprocess
import pathlib
import wandb
from deeperwin.configuration import set_with_nested_key
from ruamel.yaml import YAML
from deeperwin.dispatch import *

def get_current_run_nr():
    # find all .yml files with "run" in their name
    run_strs = [d.split('_')[-1].split('.')[0][3:] for d in os.listdir('.') if "run" in d and ".yml" in d]

    # extract integer next to "run", if possible
    run_nrs = [int(s) for s in run_strs if s.isdigit()]

    # return index
    return 0 if len(run_nrs) == 0 else max(run_nrs) + 1

if __name__ == '__main__':
    # Init wandb and get current parameter set
    run = wandb.init()

    # Load base config and adjust based on current run
    yaml = YAML()
    with open("config.yml") as f:
        config = yaml.load(f)
    for k, v in run.config.items():
        config = set_with_nested_key(config, k, v)
    parsed_config = Configuration.parse_obj(config)

    run_nr = get_current_run_nr()

    config = set_with_nested_key(config, "logging.wandb.project", None)
    config = set_with_nested_key(config, "logging.basic.fname", "erwin.log")
    config = set_with_nested_key(config, "experiment_name", f"{parsed_config.experiment_name}_{run_nr:03d}")
    config = set_with_nested_key(config, "computation.dispatch", "local")

    config_fname = f"config_run{run_nr:03d}.yml"

    with open(config_fname, 'w') as f:
        yaml.dump(config, f)

    # Run the main run in the sub-directory
    main_fname = pathlib.Path(__file__).resolve().parent.joinpath("main.py")

    # "Finish" this run before starting the actual run.
    # Otherwise this pseudo-run overwrites the results of the actual run upon exiting this script
    run.finish()
    subprocess.call(["python", main_fname, config_fname], cwd=".")







