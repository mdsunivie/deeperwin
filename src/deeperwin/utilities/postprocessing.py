import json
import os
import re
import pandas as pd

import deeperwin.main
from deeperwin.utilities import erwinConfiguration
from deeperwin.utilities.utils import sortTogether, MOLECULES_IN_DEFAULT_ORDER


def findAllRuns(root_dir = "~/runs", config_filter={}, sortby=None, require_completed=True):
    """
    Iteratively traverses a given directory and finds all directories that contain DeepErwin runs.

    DeepErwin runs are being identified by containing a file 'erwin.log' and 'config.json'.

    Args:
        root_dir (str): Starting directory to look for runs
        config_filter (dict): dict of key/value pairs that must match the configuration of a run to be included (e.g. {'optimization.n_epochs':1000})
        sortby (str): Config-dict key to use for sorting the results
        require_completed (bool): If True, only include runs that have completed (i.e. that have at least an erwin.log file)

    Returns:
        tuple containing

        - **run_dirs** (list of str): List of all directory names that contain DeepErwin runs
        - **configs** (list of configs): List of all config objects corresponding to these runs
    """
    run_dirs = []
    configs = []
    for dir, dirs, files in os.walk(os.path.expanduser(root_dir)):
        config_string = ""
        if ('config.json' in files):# and ('walkers_train.npy' in files):
            with open(os.path.join(dir, 'config.json')) as f:
                config_string = f.read()
        elif ('erwin.log' in files) and not require_completed:
            with open(os.path.join(dir, 'erwin.log')) as f:
                match = re.search("erwin - DEBUG - {.*}", f.read())
                if match is not None:
                    config_string = match.group(0)[16:]
        else:
            continue

        if config_string == "":
            continue

        try:
            config_dict = json.loads(config_string)
        except json.decoder.JSONDecodeError:
            print(f"Could not decode file: {os.path.join(dir, 'config.json')}")
            raise
        for key, val in config_filter.items():
            if config_dict.get(key) != val:
                break
        else:
            run_dirs.append(dir)
            configs.append(erwinConfiguration.DefaultConfig.build_from_dict(config_dict, allow_new_keys=True))

    if sortby is not None:
        if isinstance(sortby, (tuple, list)):
            sort_keys = [tuple(c.get_with_nested_key(s) for s in sortby) for c in configs]
        else:
            sort_keys = [c.get_with_nested_key(sortby) for c in configs]
        run_dirs, configs = sortTogether(sort_keys, run_dirs, configs)[1:]
    return run_dirs, configs


def loadRuns(root_dir, data_getters=None, get_full_history=True, get_full_config=True, modify_config_fn=None, ignore_history_keys=None, history_only=True, exclude_paths=None):
    """
    Iteratively traverse a directory, find all DeepErwin runs and assemble their data in a pandas DataFrame.

    Args:
        root_dir (str): Starting directory
        data_getters (dict): Dict that maps a string key to a function. Each function takes a WaveFunction as an argument and returns some value that will be stored in the dataframe under the corresponding key.
        get_full_history (bool): If True, the full history (e.g. energy, learning-rate, etc.) of the full training curve are included as lists. Setting to False can signficantly accelerate loading large datasets.
        get_full_config (bool): If True all config options are included as columns of the DataFrame
        modify_config_fn (function): A function that takes a config object, modifys it and returns it before the contents are written into the DataFrame. Useful for mapping renamed config options
        ignore_history_keys (iterable of str): History entries that should not be loaded. Ignoring large history items can accelerate loading.
        history_only (bool): If True, only loads config and history, but does not reload the actual model. Significantly speeds up loading of data, but prevents access to model weights or similar data.
        exclude_paths (list of str): List of paths to exclude from loading

    Returns:
        (pd.DataFrame): DataFrame containing one row per run
    """
    if data_getters is None:
        data_getters = {}

    if ignore_history_keys is None:
        ignore_history_keys = set()
    else:
        ignore_history_keys = set(ignore_history_keys)

    dirs = findAllRuns(root_dir)[0]
    data = []

    for d in dirs:
        if (exclude_paths is not None) and (os.path.normpath(d) in exclude_paths):
            continue
        wf = deeperwin.main.WaveFunction.load(d, history_only=history_only, modify_config_fn=modify_config_fn, ignore_history_keys=ignore_history_keys)
        summary = wf.get_summary(data_getters)
        summary['path'] = d
        summary['timestamp'] = os.path.getmtime(os.path.join(d, 'erwin.log'))
        if get_full_history:
            for k in wf.history:
                summary[k] = wf.history[k]
        if get_full_config:
            config_dict = wf.config.get_as_dict()
            for k,v in config_dict.items():
                summary[k.replace('.','_')] = v
        data.append(summary)
    df = pd.DataFrame(data)
    if len(df) > 0:
        df['element'] = pd.Categorical(df.name, categories=MOLECULES_IN_DEFAULT_ORDER)
    return df