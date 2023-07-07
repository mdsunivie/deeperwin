import itertools
from typing import List, Tuple, Dict
from deeperwin.configuration import set_with_flattened_key, PhisNetTrainingConfiguration
from deeperwin.run_tools.dispatch import dispatch_job, build_experiment_name, setup_experiment_dir, dump_config_dict, idx_to_job_name
import ruamel.yaml as yaml



def setup_calculations(args: List[str]):
    # load and parse config
    with open(args.input) as f:
        raw_config = yaml.safe_load(f)
    PhisNetTrainingConfiguration.parse_obj(raw_config) # check validity of input config

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
    for changed_config_values in itertools.product(*all_config_changes.values()):
        config_changes = {k: v for k, v in zip(all_config_changes.keys(), changed_config_values)}
        config_dict, parsed_config = PhisNetTrainingConfiguration.update_configdict_and_validate(raw_config, config_changes)
        experiment_dir = build_experiment_name(config_changes, parsed_config.experiment_name)
        config_dict = set_with_flattened_key(config_dict, "experiment_name", "".join(experiment_dir.split("/")[-1:]))
        parsed_config = PhisNetTrainingConfiguration.parse_obj(config_dict)
        try:
            experiment_dirs.append(setup_experiment_dir(experiment_dir, force=args.force))
            dump_config_dict(experiment_dir, config_dict)
            experiment_configs.append(parsed_config)
        except FileExistsError:
            print(f"Skipping existing run-directory: {experiment_dir}")


    # dispatch runs
    for job_nr, (job_dir, job_config) in enumerate(zip(experiment_dirs, experiment_configs)):
        dispatch_job("deeperwin train-phisnet config.yml".split(), job_dir, job_config, 0, args.dry_run)

if __name__ == '__main__':
    raise ValueError("Use 'deeperwin setup-phisnet ...' to call this script")
