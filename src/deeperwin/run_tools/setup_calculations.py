import itertools
import warnings
from deeperwin.configuration import set_with_flattened_key, Configuration
import copy
from deeperwin.run_tools.dispatch import dispatch_job, build_experiment_name, setup_experiment_dir, dump_config_dict, idx_to_job_name
from deeperwin.utils import setup_job_dir
import ruamel.yaml as yaml

def setup_calculations(args):
    wandb_sweep = len(args.wandb_sweep) == 3

    # load and parse config
    with open(args.input) as f:
        raw_config = yaml.safe_load(f)
    Configuration.parse_obj(raw_config) # check validity of input config

    # load cli parameters
    all_config_changes = dict()
    for param_tuple in args.parameter:
        if len(param_tuple) <= 1:
            raise ValueError(f"Each parameter must have at least 1 value. No value for {param_tuple[0]}")
        all_config_changes[param_tuple[0]] = param_tuple[1:]

    # prepare experiment directories
    experiment_dirs = []
    experiment_configs = []
    experiment_config_dicts = []

    for changed_config_values in itertools.product(*all_config_changes.values()):
        config_changes = {k: v for k, v in zip(all_config_changes.keys(), changed_config_values)}
        config_dict, parsed_config = Configuration.update_configdict_and_validate(raw_config, config_changes)
        experiment_config_dicts.append(config_dict)
        experiment_configs.append(parsed_config)
        experiment_dir = build_experiment_name(config_changes, parsed_config.experiment_name)
        experiment_dirs.append(setup_experiment_dir(experiment_dir, force=args.force))

    # prepare single job directories
    job_dirs = []
    job_config_dicts = []
    for exp_dir, exp_config_dict, exp_config in zip(experiment_dirs, experiment_config_dicts, experiment_configs):
        if exp_config.physical:
            n_molecules = len(exp_config.physical.changes) if exp_config.physical.changes else 1
        else:
            n_molecules = 1
        if n_molecules > 1 and wandb_sweep:
            wandb_sweep = False
            warnings.warn(
                ("Wandb sweep only compatible with single molecule computations. Wandb sweep will not be started."))

        if (n_molecules > 1) and (exp_config.optimization.shared_optimization is None) and not wandb_sweep:
            # Weight-sharing optimization
            dump_config_dict(exp_dir, exp_config_dict)
            for idx, p in enumerate(exp_config_dict["physical"]["changes"]):
                job_name = idx_to_job_name(idx)
                job_config_dict = copy.deepcopy(exp_config_dict)
                for k in p.keys():
                    job_config_dict["physical"][k] = copy.deepcopy(p[k])
                job_config_dict["physical"]["changes"] = None
                job_config_dict = set_with_flattened_key(job_config_dict, "experiment_name",
                                                      "".join(exp_dir.split("/")[-1:]) + "_" + job_name)
                # only final sub-folder name should be part of experiment name
                job_dirs.append(setup_job_dir(exp_dir, job_name))
                job_config_dicts.append(job_config_dict)
        else:
            if wandb_sweep:
                # WandB sweep
                sweep_id, n_agents, n_runs_per_agent = args.wandb_sweep[0], int(args.wandb_sweep[1]), int(
                    args.wandb_sweep[2])
                for n in range(n_agents):
                    job_name = f"agent{n:02d}"

                    job_config_dict = copy.deepcopy(exp_config_dict)
                    job_config_dict = set_with_flattened_key(job_config_dict, "experiment_name",
                                                          f"{exp_dir}_{job_name}")

                    job_dirs.append(setup_job_dir(exp_dir, job_name))
                    job_config_dicts.append(job_config_dict)
            else:
                # Standard single-molecule calculation
                exp_config_dict = set_with_flattened_key(exp_config_dict, "experiment_name", "".join(exp_dir.split("/")[-1:]))
                job_dirs.append(exp_dir)
                job_config_dicts.append(exp_config_dict)

    # dispatch runs
    for job_nr, (job_dir, job_config_dict) in enumerate(zip(job_dirs, job_config_dicts)):
        # dump config dict
        dump_config_dict(job_dir, job_config_dict)

        # parse job config
        job_config = Configuration.parse_obj(job_config_dict)

        # define which script will run
        if wandb_sweep:
            command = ["python", "-m", "wandb", "agent", "--count", str(n_runs_per_agent), str(sweep_id)]
        elif job_config.optimization.shared_optimization is not None:
            command = "deeperwin run-shared config.yml".split()
        else:
            command = "deeperwin run config.yml".split()
        if not args.dry_run:
            offset = args.start_time_offset_first + job_nr * args.start_time_offset
            dispatch_job(command, job_dir, job_config, offset)


if __name__ == '__main__':
    raise ValueError("Use 'deeperwin setup ...' to call this script")
