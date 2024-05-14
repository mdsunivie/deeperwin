import itertools
from typing import List, Tuple, Dict
import warnings
from deeperwin.configuration import set_with_flattened_key, Configuration, build_flattend_dict, build_physical_configs_from_changes
import copy
from deeperwin.run_tools.dispatch import dispatch_job, build_experiment_name, setup_experiment_dir, dump_config_dict, idx_to_job_name
from deeperwin.run_tools.geometry_database import expand_geometry_list
from deeperwin.utils.utils import setup_job_dir
import ruamel.yaml as yaml

def prepare_single_job(args: List[str], exp_dir: str, exp_config_dict: Dict) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Creates job dirs, job configs dictionaries & CLI command for performing (a single geometry on) a single job.

    :param args: command line arguments passed via cli
    :param exp_dir: directory name of the experiment
    :param exp_config_dict: Dictionary containing the config for experiment
    """
    exp_config_dict = set_with_flattened_key(exp_config_dict, "experiment_name", "".join(exp_dir.split("/")[-1:]))
    job_dirs = [exp_dir]
    job_config_dicts = [exp_config_dict]
    command = "deeperwin run config.yml".split()

    return job_dirs, job_config_dicts, command

def prepare_mulitple_geometries_on_single_job_shared(args: List[str], exp_dir: str, exp_config_dict: Dict) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Creates job dirs & job configs dictionaries for performing multiple geometries on a single job using shared optimization.

    :param args: command line arguments passed via cli
    :param exp_dir: directory name of the experiment
    :param exp_config_dict: Dictionary containing the config for experiment
    """
    job_dirs, job_config_dicts, _ = prepare_single_job(args, exp_dir, exp_config_dict)
    command = "deeperwin run-multiple-shared config.yml".split()
    return job_dirs, job_config_dicts, command

def prepare_multiple_geometries_on_multiple_jobs(args: List[str], exp_dir: str, exp_config_dict: Dict) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Creates job dirs & job configs dictionaries for performing mulitple job on different geometries.

    :param args: command line arguments passed via cli
    :param exp_dir: directory name of the experiment
    :param exp_config_dict: Dictionary containing the config for experiment
    """
    job_dirs = []
    job_config_dicts = []

    dump_config_dict(exp_dir, exp_config_dict)
    physical_config_dicts = build_physical_configs_from_changes(exp_config_dict["physical"], parse=False)
    for idx, phys_config in enumerate(physical_config_dicts):
        job_name = idx_to_job_name(idx)
        job_config_dict = copy.deepcopy(exp_config_dict)
        job_config_dict["physical"] = phys_config
        job_config_dict["experiment_name"] = "".join(exp_dir.split("/")[-1:]) + "_" + job_name
        # only final sub-folder name should be part of experiment name
        job_dirs.append(setup_job_dir(exp_dir, job_name))
        job_config_dicts.append(job_config_dict)

    command = "deeperwin run config.yml".split()

    return job_dirs, job_config_dicts, command

def prepare_wandb_sweep_jobs(args: List[str], exp_dir: str, exp_config_dict: Dict) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Creates job dirs & job configs dictionaries for performing a wandb hyperparameter sweep.

    :param args: command line arguments passed via cli
    :param exp_dir: directory name of the experiment
    :param exp_config_dict: Dictionary containing the config for experiment
    """
    job_dirs = []
    job_config_dicts = []

    sweep_id, n_agents, n_runs_per_agent = args.wandb_sweep[0], int(args.wandb_sweep[1]), int(args.wandb_sweep[2])
    for n in range(n_agents):
        job_name = f"agent{n:02d}"

        job_config_dict = copy.deepcopy(exp_config_dict)
        job_config_dict = set_with_flattened_key(job_config_dict, "experiment_name", f"{exp_dir}_{job_name}")
        job_dirs.append(setup_job_dir(exp_dir, job_name))
        job_config_dicts.append(job_config_dict)

    command = ["python", "-m", "wandb", "agent", "--count", str(n_runs_per_agent), str(sweep_id)]

    return job_dirs, job_config_dicts, command



def setup_calculations(args: List[str]):
    wandb_sweep = len(args.wandb_sweep) == 3

    # load and parse config
    with open(args.input) as f:
        raw_config = yaml.YAML(typ='safe', pure=True).load(f)
    Configuration.model_validate(raw_config) # check validity of input config

    # load cli parameters
    all_config_changes = dict()
    for param_tuple in args.parameter:
        if len(param_tuple) <= 1:
            raise ValueError(f"Each parameter must have at least 1 value. No value for {param_tuple[0]}")
        param_name, *param_values = param_tuple
        if param_name in all_config_changes:
            raise ValueError(f"Each parameter may only be supplied once (but with multiple values if required). Duplicate parameter: {param_name}")
        all_config_changes[param_name] = param_values

    # prepare experiment directories
    experiment_dirs = []
    experiment_configs = []
    experiment_config_dicts = []

    for changed_config_values in itertools.product(*all_config_changes.values()):
        config_changes = {k: v for k, v in zip(all_config_changes.keys(), changed_config_values)}
        config_dict, parsed_config = Configuration.update_configdict_and_validate(raw_config, config_changes)
        experiment_dir = build_experiment_name(config_changes, parsed_config.experiment_name)
        if "physical" in config_dict and isinstance(config_dict['physical'], (str, list)):
            config_dict['physical'] = dict(changes=expand_geometry_list(config_dict['physical']))
            parsed_config = Configuration.model_validate(config_dict)
        try:
            experiment_dirs.append(setup_experiment_dir(experiment_dir, force=args.force))
            experiment_config_dicts.append(config_dict)
            experiment_configs.append(parsed_config)
        except FileExistsError:
            print(f"Skipping existing run-directory: {experiment_dir}")


    # prepare single job directories
    all_job_dirs, all_job_configs, all_commands = [], [], []
    for exp_dir, exp_config_dict, exp_config in zip(experiment_dirs, experiment_config_dicts, experiment_configs):
        if exp_config.physical:
            n_molecules = len(exp_config.physical.changes) if exp_config.physical.changes else 1
        else:
            n_molecules = 1
        
        if n_molecules > 1 and wandb_sweep:
            raise ValueError("Wandb sweep only compatible with single molecule computations. Wandb sweep will not be started.")
        if n_molecules <= 1 and exp_config.optimization.shared_optimization:
            raise ValueError("Can not perform shared optimization since only 1 geometry is defined")

        if (n_molecules > 1) and exp_config.optimization.shared_optimization:
            job_dirs, job_config_dicts, command = prepare_mulitple_geometries_on_single_job_shared(args, exp_dir, exp_config_dict)
        elif (n_molecules > 1) and not exp_config.optimization.shared_optimization:
            job_dirs, job_config_dicts, command = prepare_multiple_geometries_on_multiple_jobs(args, exp_dir, exp_config_dict)
        elif wandb_sweep:
            job_dirs, job_config_dicts, command = prepare_wandb_sweep_jobs(args, exp_dir, exp_config_dict)
        else: # n_molecules == 1
            if "physical" in exp_config_dict:
                exp_config_dict['physical'] = exp_config.physical.create_geometry_list(exp_config_dict['physical'].get('changes'))[0].dict()
            job_dirs, job_config_dicts, command = prepare_single_job(args, exp_dir, exp_config_dict)

        for job_dir, job_config_dict in zip(job_dirs, job_config_dicts):
            dump_config_dict(job_dir, job_config_dict)
            job_config = Configuration.model_validate(job_config_dict)
            all_job_dirs.append(job_dir)
            all_job_configs.append(job_config)
            all_commands.append(command)

    # dispatch runs
    for job_nr, (job_dir, job_config, command) in enumerate(zip(all_job_dirs, all_job_configs, all_commands)):
        offset = args.start_time_offset_first + job_nr * args.start_time_offset
        dispatch_job(command, job_dir, job_config, offset, args.dry_run)

if __name__ == '__main__':
    raise ValueError("Use 'deeperwin setup ...' to call this script")
