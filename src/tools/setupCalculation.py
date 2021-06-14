#!/usr/bin/env python3
import argparse
import json
import itertools
import os
import stat
import subprocess
import shutil
from deeperwin.utilities import erwinConfiguration
import numpy as np

def makeExecutable(fname):
    """
    Equivalent to chmod +x fname
    """
    os.chmod(fname, os.stat(fname).st_mode | stat.S_IXUSR | stat.S_IXGRP)

def getNumberOfCores(cluster, partition):
    """
    Return hardcoded number of CPU-cores for a given cluster (e.g. vsc3, vsc4, dgx) and partition. Useful for estimating CPUh cost.
    """
    if cluster == 'vsc3':
        if 'plus' in partition:
            return 10
        else:
            return 16
    elif cluster == 'vsc4':
        return 24
    else:
        return 24

def setupCalculation(command, job_dir, job_name, partition, qos, cluster, submit=False, time_in_hours=5, conda='tfgpu', n_gpu=1, model=None):
    job_name = job_name.replace(" ", "_")
    job_fname = os.path.join(job_dir, 'job.sh')
    with open(job_fname, 'w') as f:
        if cluster == "vsc3":
            gres_str = "\n#SBATCH --gres gpu:1" if partition.startswith('gpu') else ""
            sbatch_content = f"""#!/bin/bash\n#SBATCH -J {job_name}\n#SBATCH -N 1\n#SBATCH --partition {partition}\n#SBATCH --qos {qos}{gres_str}\n#SBATCH --output GPU.out\n#SBATCH --time {int(time_in_hours * 60)}\n\nmodule purge\nmodule load cuda/11.0.2\nsource /opt/sw/x86_64/glibc-2.17/ivybridge-ep/anaconda3/5.3.0/etc/profile.d/conda.sh\nconda activate {conda}\n\n{command}"""
        elif cluster == "vsc4":
            if model == "PauliNet":
                print("Running PauliNet")
                sbatch_content = f"""#!/bin/bash\n#SBATCH -J {job_name}\n#SBATCH -N 1\n#SBATCH --partition {partition}\n#SBATCH --qos {qos}\n#SBATCH --output CPU.out\n#SBATCH --time {int(time_in_hours * 60)}\n\nmodule purge\nsource ~/deepqmc2/bin/activate\n\n{command}"""
            else:
                sbatch_content = f"""#!/bin/bash\n#SBATCH -J {job_name}\n#SBATCH -N 1\n#SBATCH --partition {partition}\n#SBATCH --qos {qos}\n#SBATCH --output CPU.out\n#SBATCH --time {int(time_in_hours * 60)}\n\nmodule purge\nsource ~/pytf/bin/activate\n\n{command}"""
        elif cluster == "dgx":
            sbatch_content = f"""#!/bin/bash\n#SBATCH -J {job_name}\n#SBATCH -N 1\n#SBATCH --output GPU.out\n#SBATCH --time {int(time_in_hours * 60)}\n#SBATCH --gres gpu:{n_gpu}\n\n{command}"""
        else:
            raise ValueError(f"Invalid cluster name: {cluster}")
        f.write(sbatch_content)
    makeExecutable(job_fname)
    if submit:
        run_command = ['sbatch']
        if cluster == 'dgx':
            run_command += ['--chdir', f'/nfs{job_dir}']
        run_command += ['job.sh']
        subprocess.run(run_command, cwd=job_dir)
    return time_in_hours * getNumberOfCores(cluster, partition)

def getCodePath(branch):
    """
    Get path to deeperwin code for a given 'branch'.

    A branch does not necessarily correspond to a GIT branch, but corresponds to any subdirectory that the code has been cloned into.
    These directories should be named '~/develop/deeperwin_{branch}'

    Args:
        branch (str): Name of 'branch' to use for running this calculation

    Returns:
        (str): Full path to branch
    """
    if branch == '':
        path = os.path.expanduser('~/develop/deeperwin')
    else:
        path = os.path.expanduser(f'~/develop/deeperwin_{branch}')
    assert os.path.isdir(path), f"Code does not exist: {path}"
    return path

def buildCommandForDGX(job_dir, code_dir, job_name):
    user_id = os.getuid()
    command = f"nvidia-docker run --name {job_name} --user {user_id}:{user_id}"
    command += " --rm --shm-size=100g --ulimit memlock=-1 --ulimit stack=67108864"
    command += f" -v /nfs{os.path.realpath(job_dir)}:/workspace -v /nfs{os.path.realpath(code_dir)}:/deeperwin deeperwin/tf-nightly-gpu"
    return command


def get_cluster():
    """
    Return string that identifies the current system the code is running on:
    Returns:
        (str): ID-string: either 'vsc3', 'vsc4', 'dgx'
    """
    if 'HPC_SYSTEM' in os.environ:
        return os.environ["HPC_SYSTEM"].lower() # vsc3 or vsc4
    else:
        return 'dgx'

if __name__ == '__main__':
    partition_mapping = dict(gpu='gpu_gtx1080single', rtx='gpu_rtx2080ti', v100='gpu_v100', k20='gpu_k20m',
                             gtxmulti='gpu_gtx1080multi', gpu_gtx1080multi='gpu_gtx1080multi',
                             gpu_gtx1080single='gpu_gtx1080single', gpu_rtx2080ti='gpu_rtx2080ti', gpu_v100='gpu_v100',
                             gpu_k20m='gpu_k20m', gpu_gtx1080amd='gpu_gtx1080amd', gtxamd='gpu_gtx1080amd', cpu64='mem_0064', cpu128='mem_0128', cpu256='mem_0256',
                             cpu64p='vsc3plus_0064', cpu256p='vsc3plus_0256', mem_0064='mem_0064', mem_0128='mem_0128',
                             mem_0256='mem_0256', vsc3plus_0064='vsc3plus_0064', vsc3plus_0256='vsc3plus_0256',
                             mem_0096="mem_0096", mem_0384="mem_0384", mem_0768="mem_0768")
    qos_mapping = dict(gpu_gtx1080single='gpu_gtx1080single', gpu_rtx2080ti='gpu_rtx2080ti',
                       gpu_gtx1080multi='gpu_gtx1080multi',
                       gpu_k20m='gpu_k20m', gpu_v100='gpu_v100', gpu_gtx1080amd='gpu_gtx1080amd', mem_0064='normal_0064', mem_0128='normal_0128',
                       mem_0256='normal_0256',
                       vsc3plus_0064='vsc3plus_0064', vsc3plus_0256='vsc3plus_0256', mem_0096="mem_0096",
                       mem_0384="mem_0384", mem_0768="mem_0768")

    parser = argparse.ArgumentParser()
    parser.add_argument("input", default="config.in", nargs='?',
                        help="Base configuration file to use (e.g. config.in). Must be a JSON-file of a config-dict. Final config is a combination of base-config-file, defaults and parameters set explicitly with -p flag")
    parser.add_argument("-p", "--param", nargs='+', action='append', default=[],
                        help="Specify config-parameter to be set or sweeped, by specificing config-key and value. Example: -p optimization.n_epochs 500. If multiple values are given, multiple calculations will be set-up (e.g. -p optimization.learning_rate 0.01 0.05 will set-up 2 calculations).")
    parser.add_argument("--plinspace", nargs='+', action='append', default=[],
                        help="Same as param, but sweeps parameters linearly from min to max in N steps. Example: --plinspace optimization.n_epochs 500 2000 5")
    parser.add_argument("-v", "--variable", nargs='+', action='append', default=[],
                        help="Variables to be substituted in config-parameters. Same usage as -p, but instead of directly setting confi-values, it substitutes $VARIABLENAMES in the config.in (or the passed -p parameters)")
    parser.add_argument("--vlinspace", nargs='+', action='append', default=[],
                        help="Same as --variable, but sweeps parameters linearly from min to max in N steps. Example: --vlinspace VARIABLENAME 500 2000 5")
    parser.add_argument("--vrange", nargs='+', action='append', default=[],
                        help="Same as --variable, but sweeps parameters linearly using range (start, stop, step) Example: --vrange VARIABLENAME 0 11 1")
    parser.add_argument("-n", "--name", default="",
                        help="Job-name prefix to be used for calculation directory and slurm job-name")
    parser.add_argument("-f", "--force", action='store_true',
                        help="Overwrite calculation directory and files if they already exist")
    parser.add_argument("-s", "--submit", action='store_true', help="Directly submit jobs to SLURM job queue")
    parser.add_argument("-t", "--time", default=24.0, type=float, help="Time limit (in hours) for SLURM jobs")
    parser.add_argument("--nocheck", action='store_true',
                        help="Do not check whether parameters are valid config-parameters")
    parser.add_argument("--partition", default="gpu_gtx1080single", help="Partition and to use for SLURM.",
                        choices=partition_mapping.keys())
    vsc4_partition_default = "mem_0096"
    parser.add_argument("--qos", default=None, help="QOS to use for SLURM.")
    parser.add_argument("--conda", default="tf_nightly", help="Specify which conda environment to activate before running the job")
    parser.add_argument("--branch", default='', help="Specifiy alternative code-version to run. Code must reside in ~/develop/deeperwin_{branch}")
    args = parser.parse_args()

    cluster = get_cluster()

    if cluster == "vsc4" and args.partition == parser.get_default("partition"):
        partition = vsc4_partition_default
    else:
        partition = args.partition

    partition = partition_mapping[partition]
    if args.qos is None:
        qos = qos_mapping[partition]
    else:
        qos = args.qos
        if qos != qos_mapping[partition]:
            print("WARNING: QOS does not match the default mapping between partition and qos.")

    with open(args.input) as f:
        base_config_dict = json.load(f)

    param_names = [p[0] for p in args.param]
    param_values = [p[1:] for p in args.param]
    variable_names = [p[0] for p in args.variable]
    variable_values = [p[1:] for p in args.variable]

    for l in args.plinspace:
        assert len(l) == 4, "--plinspace arguments must have exactly 4 values: option, value_min, value_max, n_steps"
        param_names.append(l[0])
        param_values.append(np.round(np.linspace(float(l[1]), float(l[2]), int(l[3])), decimals=6))
    for l in args.vlinspace:
        assert len(l) == 4, "--vlinspace arguments must have exactly 4 values: option, value_min, value_max, n_steps"
        variable_names.append(l[0])
        variable_values.append(np.round(np.linspace(float(l[1]), float(l[2]), int(l[3])), decimals=6))
    for l in args.vrange:
        assert (len(l) == 4) or (len(l) == 3), "--vrange arguments must have 3 or 4 values: option, start, stop, [stepsize]"
        variable_names.append(l[0])
        if len(l) == 4:
            variable_values.append(range(int(l[1]), int(l[2]), int(l[3])))
        elif len(l) == 3:
            variable_values.append(range(int(l[1]), int(l[2])))


    n_params = len(args.param)
    n_variables = len(args.variable)

    for p in args.param + args.variable:
        if len(p) < 2:
            raise ValueError("No name or value specified for parameter or variable")

    cpu_hours = 0
    for param_var_combination in itertools.product(*(param_values + variable_values)):
        param_combination = param_var_combination[:n_params]
        var_combination = param_var_combination[n_params:]

        # Update the config-dict with all the parameters specified over the command line
        config_dict = dict(base_config_dict)
        for key, value in zip(param_names, param_combination):
            config_dict[key] = value

        # Substitute all variables (named $VARIABLENAME) in the config dict with their values
        for var_name, var_value in zip(variable_names, var_combination):
            found_variable = False
            for key, value in config_dict.items():
                if (type(value) == str) and ("$" + var_name) in value:
                    config_dict[key] = config_dict[key].replace("$" + var_name, str(var_value))
                    found_variable = True
            assert found_variable, f"Variable not found in input config: {var_name}"

        for key, value in config_dict.items():
            if type(value) == str:
                assert "$" not in value, f"No value specified for this key/value pair: {key}: {value}"

        # Create a folder for the calculation
        job_name = args.name + "_".join(map(str, param_var_combination))
        if job_name == '':
            job_name = 'job'

        # Try to build the configuration and see whether it is valid
        if not args.nocheck:
            _conf = erwinConfiguration.DefaultConfig.build_from_dict(config_dict, convert_type=True,
                                                                     allow_new_keys=False)
            #config_dict['output.tb_path'] = os.path.join(_conf.output.tb_path, job_name)
            _conf.validate()

        job_dir = os.path.realpath(job_name)
        if os.path.isdir(job_dir):
            if args.force:
                shutil.rmtree(job_dir)
            else:
                raise FileExistsError(f"Job-directory already exists: {job_dir}. Skipping this calculation")
        os.makedirs(job_dir)

        # Put the config-file + a sbatch run-script into the calculation folder
        with open(os.path.join(job_dir, 'config.in'), 'w') as f:
            json.dump(config_dict, f, indent=4)

        code_path = getCodePath(args.branch)
        if cluster in ['vsc3', 'vsc4']:
            command = f"python {os.path.join(code_path, 'deeperwin/__main__.py')} config.in"
        elif cluster == 'dgx':
            command = buildCommandForDGX(job_dir, code_path, job_name)

        model = config_dict.get('model', None)
        cpu_hours += setupCalculation(command, job_dir, job_name,
                         partition, qos, cluster, args.submit, args.time, args.conda, model=model)
    print(f"Total CPUh requested: {cpu_hours / 1e3:5.1f}k")
