#!/usr/bin/env python
"""
CLI for wandb sweeps.
"""

import wandb
import os
from deeperwin.configuration import Configuration, set_with_flattened_key
from ruamel.yaml import YAML
from deeperwin.process_molecule import process_molecule
import re

def _get_current_run_nr():
    directories = "\n".join([d for d in os.listdir() if os.path.isdir(d)])
    run_nrs = [int(s) for s in re.findall(r"\brun_(\d*)", directories)]
    return max(run_nrs) + 1 if run_nrs else 0

def run_sweep_agent():
    # Init wandb and get current parameter set
    run = wandb.init()

    # Load base config and adjust based on current run
    yaml = YAML()
    with open("config.yml") as f:
        config_dict = yaml.load(f)
    config_dict, parsed_config = Configuration.update_configdict_and_validate(config_dict, run.config)

    run_nr = _get_current_run_nr()
    run_dir = f"run_{run_nr:03d}"
    os.mkdir(run_dir)
    os.chdir(run_dir)

    config_dict = set_with_flattened_key(config_dict, "logging.wandb.project", None)
    config_dict = set_with_flattened_key(config_dict, "logging.basic.fname", "erwin.log")
    config_dict = set_with_flattened_key(config_dict, "experiment_name", f"{parsed_config.experiment_name}_{run_nr:03d}")
    config_dict = set_with_flattened_key(config_dict, "dispatch.system", "local")
    config_fname = f"config_run{run_nr:03d}.yml"
    with open(config_fname, 'w') as f:
        yaml.dump(config_dict, f)

    # "Finish" this run before starting the actual run.
    # Otherwise this pseudo-run overwrites the results of the actual run upon exiting this script
    run.finish()
    process_molecule(config_fname)
    os.chdir("..")

if __name__ == '__main__':
    run_sweep_agent()

