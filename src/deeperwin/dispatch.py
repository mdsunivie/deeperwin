"""
Dispatch management for different machines (clusters, local, ...)
"""

import bz2
import copy
import logging
import os
import pickle
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from ruamel import yaml

from deeperwin.configuration import Configuration


def load_from_file(fname):
    with bz2.open(fname, 'rb') as f:
        return pickle.load(f)


def save_to_file(fname, **data):
    with bz2.open(fname, 'wb') as f:
        pickle.dump(data, f)


def load_all_runs(root_dir, max_history_length=50):
    full_data = []
    for fname in Path(root_dir).rglob("results.bz2"):
        raw_data = load_from_file(fname)
        data = dict(raw_data['config'])
        for k, v in raw_data['metrics'].items():
            if isinstance(v, list) and (max_history_length is not None) and (len(v) > max_history_length):
                continue
            data[k] = v
        data['dirname'] = fname.parent.name
        full_data.append(data)
    return pd.DataFrame(full_data)


def prepare_checkpoints(run_path, chkpt_epochs, config: Configuration):
    checkpoints = {}
    for epoch in chkpt_epochs:
        chkpt_name = f"chkpt{epoch:06d}"
        chkpt_dir = os.path.join(run_path, chkpt_name)
        if os.path.exists(chkpt_dir):
            logging.warning(f"Directory {chkpt_dir} already exists. Results might be overwritten.")
        else:
            os.mkdir(chkpt_dir)
        config_chkpt = copy.deepcopy(config)
        config_chkpt.optimization.n_epochs = epoch
        config_chkpt.evaluation.n_epochs = 0
        config_chkpt.experiment_name = f"{config_chkpt.experiment_name}_{chkpt_name}"
        config_chkpt.save(os.path.join(chkpt_dir, "full_config.yml"))
        checkpoints[epoch] = chkpt_dir
    return checkpoints


def is_checkpoint(path):
    return 'chkpt' in os.path.split(path)[1].lower()


def load_run(path):
    results = load_from_file(os.path.join(path, "results.bz2"))
    parsed_config = Configuration.load(os.path.join(path, "full_config.yml"))
    return results, parsed_config


def contains_run(path):
    # check for full config and results.bz2
    if not os.path.exists(os.path.join(path, "full_config.yml")):
        return False
    if not os.path.exists(os.path.join(path, "results.bz2")):
        return False
    return True


def find_runs_rec(path, include_checkpoints=False):
    run_paths = []
    paths_to_check = [p[0] for p in os.walk(path)]
    for p in paths_to_check:
        if contains_run(p) and (include_checkpoints or not is_checkpoint(p)):
            run_paths.append(os.path.relpath(p, start=path))
    return run_paths


def idx_to_job_name(idx):
    return f"{idx:04d}"


def dump_config_dict(directory, config_dict):
    with open(Path(directory).joinpath('config.yml'), 'w') as f:
        yaml.YAML().dump(Configuration._to_prettified_yaml(config_dict), f)


def setup_experiment_dir(directory, force=False):
    if os.path.isdir(directory):
        if force:
            shutil.rmtree(directory)
        else:
            raise FileExistsError(f"Could not create experiment {directory}: Directory already exists.")
    os.makedirs(directory)
    return directory


def setup_job_dir(parent_dir, name):
    job_dir = os.path.join(parent_dir, name)
    if os.path.exists(job_dir):
        logging.warning(f"Directory {job_dir} already exists. Results might be overwritten.")
    else:
        os.makedirs(job_dir)
    return job_dir


def shorten_parameter_name(name):
    return "".join([s[:2] for s in name.replace("_", "").split(".")])


def build_experiment_name(parameters, include_param_shorthand, basename=""):
    s = [basename] if len(basename) > 0 else []
    for name, value in parameters:
        if include_param_shorthand:
            s.append(f"{shorten_parameter_name(name)}-{value}")
        else:
            s.append(value)
    return "_".join(s)


def get_fname_fullpath(fname):
    return Path(__file__).resolve().parent.joinpath(fname)


def dispatch_to_local(command, run_dir, config: Configuration):
    subprocess.run(command, cwd=run_dir)


def dispatch_to_vsc3(command, run_dir, config: Configuration):
    time_in_minutes = duration_string_to_minutes(config.dispatch.time)
    queue = 'gpu_rtx2080ti' if config.dispatch.queue == "default" else config.dispatch.queue
    jobfile_content = get_jobfile_content_vsc3(' '.join(command), config.experiment_name, queue,
                                               time_in_minutes, config.dispatch.conda_env)

    with open(os.path.join(run_dir, 'job.sh'), 'w') as f:
        f.write(jobfile_content)
    subprocess.run(['sbatch', 'job.sh'], cwd=run_dir)


def dispatch_to_vsc4(command, run_dir, config: Configuration):
    time_in_minutes = duration_string_to_minutes(config.dispatch.time)
    queue = 'mem_0096' if config.dispatch.queue == "default" else config.dispatch.queue
    jobfile_content = get_jobfile_content_vsc4(' '.join(command), config.experiment_name, queue,
                                               time_in_minutes)

    with open(os.path.join(run_dir, 'job.sh'), 'w') as f:
        f.write(jobfile_content)
    subprocess.run(['sbatch', 'job.sh'], cwd=run_dir)


def append_nfs_to_fullpaths(command):
    ret = copy.deepcopy(command)
    for idx, r in enumerate(ret):
        if r.startswith("/") and os.path.exists(r):
            ret[idx] = "/nfs" + r
    return ret


def dispatch_to_dgx(command, run_dir, config: Configuration):
    command = append_nfs_to_fullpaths(command)
    time_in_minutes = duration_string_to_minutes(config.dispatch.time)
    src_dir = "/nfs"+str(Path(__file__).resolve().parent.parent)
    jobfile_content = get_jobfile_content_dgx(' '.join(command), config.experiment_name,
                                              "/nfs" + str(os.path.abspath(run_dir)),
                                              time_in_minutes, config.dispatch.conda_env,
                                              src_dir)

    with open(os.path.join(run_dir, 'job.sh'), 'w') as f:
        f.write(jobfile_content)
    subprocess.run(['sbatch', 'job.sh'], cwd=run_dir)


def duration_string_to_minutes(s):
    match = re.search("([0-9]*[.]?[0-9]+)( *)(.*)", s)
    if match is None:
        raise ValueError(f"Invalid time string: {s}")
    amount, unit = float(match.group(1)), match.group(3).lower()
    if unit in ['d', 'day', 'days']:
        return int(amount * 1440)
    elif unit in ['h', 'hour', 'hours']:
        return int(amount * 60)
    elif unit in ['m', 'min', 'minute', 'minutes', '']:
        return int(amount)
    elif unit in ['s', 'sec', 'second', 'seconds']:
        return int(amount / 60)
    else:
        raise ValueError(f"Invalid unit of time: {unit}")


def get_jobfile_content_vsc4(command, jobname, queue, time):
    return f"""#!/bin/bash
#SBATCH -J {jobname}
#SBATCH -N 1
#SBATCH --partition {queue}
#SBATCH --qos {queue}
#SBATCH --output CPU.out
#SBATCH --time {time}

module purge
{command}"""


def get_jobfile_content_vsc3(command, jobname, queue, time, conda_env):
    return f"""#!/bin/bash
#SBATCH -J {jobname}
#SBATCH -N 1
#SBATCH --partition {queue}
#SBATCH --qos {queue}
#SBATCH --output GPU.out
#SBATCH --time {time}
#SBATCH --gres=gpu:1

module purge
module load cuda/10.1.168
source /opt/sw/x86_64/glibc-2.17/ivybridge-ep/anaconda3/5.3.0/etc/profile.d/conda.sh
conda activate {conda_env}

export CUDA_VISIBLE_DEVICES="0"
{command}"""


def get_jobfile_content_dgx(command, jobname, jobdir, time, conda_env, src_dir):
    return f"""#!/bin/bash
#SBATCH -J {jobname}
#SBATCH -N 1
#SBATCH --output GPU.out
#SBATCH --time {time}
#SBATCH --gres=gpu:1
#SBATCH --chdir {jobdir}

export CONDA_ENVS_PATH="/nfs$HOME/.conda/envs:$CONDA_ENVS_PATH"
source /opt/anaconda3/etc/profile.d/conda.sh 
conda activate {conda_env}
export PYTHONPATH="{src_dir}"
export WANDB_API_KEY=$(grep -Po "(?<=password ).*" /nfs$HOME/.netrc)
export CUDA_VISIBLE_DEVICES="0"
{command}"""


def dispatch_job(fname, job_dir, config):
    dispatch_to = config.dispatch.system
    if dispatch_to == "auto":
        dispatch_to = "local"
        if os.path.exists("/etc/slurm/slurm.conf"):
            with open('/etc/slurm/slurm.conf') as f:
                slurm_conf = f.readlines()
                if 'slurm.vda.univie.ac.at' in ''.join(slurm_conf):
                    dispatch_to = "dgx"
        if 'HPC_SYSTEM' in os.environ:
            dispatch_to = os.environ["HPC_SYSTEM"].lower()  # vsc3 or vsc4
    logging.info(f"Dispatching command {' '.join(fname)} to: {dispatch_to}")
    globals()[f"dispatch_to_{dispatch_to}"](fname, job_dir, config)
