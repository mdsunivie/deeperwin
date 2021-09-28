"""
DeepErwin CLI
"""

#!/usr/bin/env python3
import argparse
import itertools
import warnings

from deeperwin.configuration import set_with_nested_key
from deeperwin.dispatch import *


def main():
    parser = argparse.ArgumentParser(description="JAX implementation of DeepErwin QMC code.")
    parser.add_argument("config_file", default="config.yml", help="Path to input config file", nargs="?")
    parser.add_argument('--parameter', '-p', nargs='+', action='append', default=[])
    parser.add_argument('--force', '-f', action="store_true", help="Overwrite directories if they already exist")
    parser.add_argument('--wandb-sweep', nargs=3, default=[],
                        help="Start a hyperparmeter sweep using wandb, with a given sweep-identifier, number of agents and number of runs per agent, e.g. --wandb-sweep schroedinger_univie/sweep_debug/cpxe3iq9 3 10")
    parser.add_argument('--exclude-param-name', action="store_true",
                        help="Do not inject a shortened string for the parameter name into the experiment name, leading to shorter (but less self explanatory) experiment names.")

    # args = parser.parse_args(
    #    ["config.yml", "-p", "optimization.n_epochs", "100", "-p", "evaluation.n_epochs", "30", "40", "-p",
    #     "experiment_name", "LiveDemoExperiment", "--exclude-param-name"])
    args = parser.parse_args()

    wandb_sweep = len(args.wandb_sweep) == 3

    # load and parse config
    with open(args.config_file) as f:
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
        experiment_dir = build_experiment_name(
            [changes for i, changes in enumerate(config_changes) if len(cli_flags[i]) > 1], not args.exclude_param_name,
            parsed_config.experiment_name)

        experiment_dirs.append(setup_experiment_dir(experiment_dir, force=args.force))

    # prepare single job directories
    job_dirs = []
    job_config_dicts = []

    for exp_dir, exp_config_dict, exp_config in zip(experiment_dirs, experiment_config_dicts, experiment_configs):
        is_restart = not exp_config.restart is None
        if "physical" in exp_config_dict:
            n_molecules = 1 if "changes" not in exp_config_dict["physical"] else len(
                exp_config_dict["physical"]["changes"])
        else:
            n_molecules = 1
        if n_molecules > 1 and wandb_sweep:
            wandb_sweep = False
            warnings.warn(
                ("Wandb sweep only compatible with single molecule computations. Wandb sweep will not be started."))
        if n_molecules > 1 and is_restart:
            raise Exception("Restarts with multiple molecular geometries are currently not supported.")

        # check if experiment is restart
        if is_restart:
            restart_path = os.path.expanduser(exp_config.restart.path)
            if exp_config.restart.recursive:
                paths_restart = find_runs_rec(restart_path, include_checkpoints=exp_config.restart.checkpoints)
            else:
                paths_restart = [restart_path] if contains_run(restart_path) else []

            if len(paths_restart) == 0:
                warnings.warn((f"No run(s) found in restart directory {restart_path}."))

            for p in paths_restart:
                job_config_dict = copy.deepcopy(exp_config_dict)
                job_config_dict = set_with_nested_key(job_config_dict, "restart.recursive", False)
                job_config_dict = set_with_nested_key(job_config_dict, "restart.path",
                                                      os.path.join(restart_path, p))
                if p == ".":
                    job_dirs.append(exp_dir)
                    job_config_dicts.append(job_config_dict)
                else:
                    job_config_dict = set_with_nested_key(job_config_dict, "experiment_name",
                                                          exp_config.experiment_name + "_" + p)
                    job_dirs.append(setup_job_dir(exp_dir, p))
                    job_config_dicts.append(job_config_dict)
        else:
            if n_molecules > 1 and exp_config.optimization.shared_optimization is None and not wandb_sweep:
                dump_config_dict(exp_dir, exp_config_dict)
                for idx, p in enumerate(exp_config_dict["physical"]["changes"]):
                    job_name = idx_to_job_name(idx)

                    job_config_dict = copy.deepcopy(exp_config_dict)
                    for k in p.keys():
                        job_config_dict["physical"][k] = copy.deepcopy(p[k])
                    job_config_dict["physical"]["changes"] = None
                    job_config_dict = set_with_nested_key(job_config_dict, "experiment_name", exp_dir + "_" + job_name)

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
                    exp_config_dict = set_with_nested_key(exp_config_dict, "experiment_name", exp_dir)

                    job_dirs.append(exp_dir)
                    job_config_dicts.append(exp_config_dict)

    # dispatch runs
    for job_dir, job_config_dict in zip(job_dirs, job_config_dicts):
        # dump config dict
        dump_config_dict(job_dir, job_config_dict)

        # parse job config
        job_config = Configuration.parse_obj(job_config_dict)

        # define which script will run
        if wandb_sweep:
            command = ["python", "-m", "wandb", "agent", "--count", str(n_runs_per_agent), str(sweep_id)]
        elif job_config.optimization.shared_optimization is not None:
            command = ["python", str(get_fname_fullpath("process_molecules_inter.py")), "config.yml"]
        else:
            command = ["python", str(get_fname_fullpath("process_molecule.py")), "config.yml"]

        dispatch_job(command, job_dir, job_config)


if __name__ == '__main__':
    main()
