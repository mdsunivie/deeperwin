#!/usr/bin/env python3
"""
DeepErwin CLI
"""
import argparse
import itertools
import warnings
from ruamel import yaml
from deeperwin.configuration import set_with_nested_key
from deeperwin.dispatch import *


def main():
    parser = argparse.ArgumentParser(description="JAX implementation of DeepErwin QMC code.")
    parser.add_argument("--input", "-i", default="config.yml", help="Path to input config file")
    parser.add_argument('--parameter', '-p', nargs='+', action='append', default=[])
    parser.add_argument('--force', '-f', action="store_true", help="Overwrite directories if they already exist")
    parser.add_argument('--wandb-sweep', nargs=3, default=[],
                        help="Start a hyperparmeter sweep using wandb, with a given sweep-identifier, number of agents and number of runs per agent, e.g. --wandb-sweep schroedinger_univie/sweep_debug/cpxe3iq9 3 10")
    parser.add_argument('--exclude-param-name', '-e', action="store_true", default=True,
                        help="Do not inject a shortened string for the parameter name into the experiment name, leading to shorter (but less self explanatory) experiment names.")
    parser.add_argument('--dry-run', action='store_true', help="Only set-up the directories and config-files, but do not dispatch the actual calculation.")
    parser.add_argument('--start-time-offset', default=0, type=int, help="Add a delay (given in seconds) to each consecutive job at runtime, to avoid GPU-collisions")
    parser.add_argument('--start-time-offset-first', default=0, type=int, help="Add constant delay (given in seconds) to all jobs at runtime, to avoid GPU-collisions")


    # args = parser.parse_args(
    #    ["config.yml", "-p", "optimization.n_epochs", "100", "-p", "evaluation.n_epochs", "30", "40", "-p",
    #     "experiment_name", "LiveDemoExperiment", "--exclude-param-name"])
    args = parser.parse_args()

    wandb_sweep = len(args.wandb_sweep) == 3

    # load and parse config
    with open(args.input) as f:
        raw_config = yaml.safe_load(f)
    parsed_config = Configuration.parse_obj(raw_config)

    # load cli parameters
    cli_flags = []

    for param_tuple in args.parameter:
        if len(param_tuple) <= 1:
            raise ValueError(f"Each parameter must have at least 1 value. No value for {param_tuple[0]}")
        cli_flags.append([(param_tuple[0], value) for value in param_tuple[1:]])

    # prepare experiment directories
    experiment_dirs = []
    experiment_configs = []
    experiment_config_dicts = []

    for config_changes in itertools.product(*cli_flags):
        config_dict, parsed_config = Configuration.update_config(copy.deepcopy(raw_config), config_changes)
        experiment_config_dicts.append(config_dict)
        experiment_configs.append(parsed_config)

        experiment_dir = build_experiment_name(config_changes, False, parsed_config.experiment_name)
        if parsed_config.dispatch.split_opt is not None:
            if parsed_config.reuse is None: # TODO: change to reuse?
                experiment_dir = f"{experiment_dir}/{experiment_dir}"
            else:
                experiment_dir = f"../{experiment_dir}"
        experiment_dirs.append(setup_experiment_dir(experiment_dir, force=args.force))

    # prepare single job directories
    job_dirs = []
    job_config_dicts = []

    for exp_dir, exp_config_dict, exp_config in zip(experiment_dirs, experiment_config_dicts, experiment_configs):
        if "physical" in exp_config_dict:
            n_molecules = 1 if "changes" not in exp_config_dict["physical"] else len(
                exp_config_dict["physical"]["changes"])
        else:
            n_molecules = 1
        if n_molecules > 1 and wandb_sweep:
            wandb_sweep = False
            warnings.warn(
                ("Wandb sweep only compatible with single molecule computations. Wandb sweep will not be started."))

        if (n_molecules > 1) and (exp_config.optimization.shared_optimization is None) and not wandb_sweep:
            dump_config_dict(exp_dir, exp_config_dict)
            for idx, p in enumerate(exp_config_dict["physical"]["changes"]):
                job_name = idx_to_job_name(idx)
                job_config_dict = copy.deepcopy(exp_config_dict)
                for k in p.keys():
                    job_config_dict["physical"][k] = copy.deepcopy(p[k])
                job_config_dict["physical"]["changes"] = None
                job_config_dict = set_with_nested_key(job_config_dict, "experiment_name",
                                                      "".join(exp_dir.split("/")[-1:]) + "_" + job_name)
                # only final sub-folder name should be part of experiment name
                job_dirs.append(setup_job_dir(exp_dir, job_name))
                job_config_dicts.append(job_config_dict)
        else:
            if wandb_sweep:
                sweep_id, n_agents, n_runs_per_agent = args.wandb_sweep[0], int(args.wandb_sweep[1]), int(
                    args.wandb_sweep[2])
                for n in range(n_agents):
                    job_name = f"agent{n:02d}"

                    job_config_dict = copy.deepcopy(exp_config_dict)
                    job_config_dict = set_with_nested_key(job_config_dict, "experiment_name",
                                                          f"{exp_dir}_{job_name}")

                    job_dirs.append(setup_job_dir(exp_dir, job_name))
                    job_config_dicts.append(job_config_dict)
            else:
                exp_config_dict = set_with_nested_key(exp_config_dict, "experiment_name", "".join(exp_dir.split("/")[-1:]))
                job_dirs.append(exp_dir)
                job_config_dicts.append(exp_config_dict)

    # dispatch runs
    for job_nr, (job_dir, job_config_dict) in enumerate(zip(job_dirs, job_config_dicts)):
        # split job into multiple runs for vsc
        if ('dispatch' in job_config_dict) and (job_config_dict['dispatch'].get('split_opt') is not None) and\
                (job_config_dict.get('reuse') is None):
                job_config_dict['dispatch']['split_opt'] += [job_config_dict['optimization']['n_epochs']]
                if ('evaluation' in job_config_dict) and (job_config_dict['evaluation'].get('n_epochs') is not None):
                    job_config_dict['dispatch']['eval_epochs'] = job_config_dict['evaluation']['n_epochs']
                    job_config_dict['evaluation']['n_epochs'] = 0
                else:
                    job_config_dict['dispatch']['eval_epochs'] = 5000 # setting to default number of epochs in evaluation
                    job_config_dict['evaluation'] = {'n_epochs': 0}
                job_config_dict['optimization']['n_epochs'] = job_config_dict['dispatch']['split_opt'].pop(0)


        # dump config dict
        dump_config_dict(job_dir, job_config_dict)

        # parse job config
        job_config = Configuration.parse_obj(job_config_dict)

        # define which script will run
        if wandb_sweep:
            command = ["python", "-m", "wandb", "agent", "--count", str(n_runs_per_agent), str(sweep_id)]
        elif job_config.optimization.shared_optimization is not None:
            command = ["python", str(get_fname_fullpath("process_molecules_shared.py")), "config.yml"]
        else:
            offset = args.start_time_offset_first + job_nr * args.start_time_offset
            command = ["python", str(get_fname_fullpath("process_molecule.py")), "config.yml", str(offset)]
        if not args.dry_run:
            dispatch_job(command, job_dir, job_config)


if __name__ == '__main__':
    main()
