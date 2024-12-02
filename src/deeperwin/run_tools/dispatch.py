"""
Dispatch management for different machines (clusters, local, ...)
"""

import copy
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from ruamel import yaml
from deeperwin.configuration import Configuration, to_prettified_yaml


def idx_to_job_name(idx):
    return f"{idx:04d}"


def dump_config_dict(directory, config_dict, config_name="config.yml"):
    with open(Path(directory).joinpath(config_name), "w") as f:
        yaml.YAML().dump(to_prettified_yaml(config_dict), f)


def setup_experiment_dir(directory, force=False):
    if os.path.isdir(directory):
        if force:
            shutil.rmtree(directory)
        else:
            raise FileExistsError(f"Could not create experiment {directory}: Directory already exists.")
    os.makedirs(directory)
    return directory


def build_experiment_name(parameters, basename=""):
    s = [basename] if basename else []
    for name, value in parameters.items():
        if name == "experiment_name" or name == "reuse.path":
            continue
        else:
            s.append(value)
    return "_".join(s)


def get_fname_fullpath(fname):
    return Path(__file__).resolve().parent.joinpath(fname)


def dispatch_to_local(command, run_dir, config: Configuration, sleep_in_sec, dry_run=False):
    env = os.environ.copy()
    if dry_run:
        print(f"Dry-run of command: {command}")
    else:
        subprocess.run(command, cwd=run_dir, env=env)


def dispatch_to_local_background(command, run_dir, config: Configuration, sleep_in_sec, dry_run):
    with open(os.path.join(run_dir, "GPU.out"), "w") as f:
        if not dry_run:
            print(f"Dispatching to local_background: {command}")
            subprocess.Popen(
                command,
                stdout=f,
                stderr=f,
                start_new_session=True,
                cwd=run_dir,
                shell=True,
            )
        else:
            print(f"Dry-runing: {command}")


def _save_and_submit_slurm_job(jobfile_content, run_dir=".", dry_run=False):
    with open(os.path.join(run_dir, "job.sh"), "w") as f:
        f.write(jobfile_content)
    if not dry_run:
        subprocess.run(["sbatch", "job.sh"], cwd=run_dir)


def dispatch_to_vsc5(command, run_dir, config: Configuration, sleep_in_sec, dry_run):
    time_in_minutes = duration_string_to_minutes(config.dispatch.time)
    queue_translation = dict(default="zen3_0512_a100x2", a40="zen2_0256_a40x2", a100="zen3_0512_a100x2")
    queue = queue_translation.get(config.dispatch.queue, config.dispatch.queue)
    if config.computation.n_local_devices:
        n_gpus = config.computation.n_local_devices
    elif queue in ["zen3_0512_a100x2", "zen2_0256_a40x2"]:
        n_gpus = 2
    else:
        n_gpus = 1
    n_nodes = config.computation.n_nodes
    if (n_nodes > 1) and (("a100x2" in queue) or ("a40x2" in queue)) and (n_gpus < 2):
        print("You requested multiple multi-GPU nodes, using only 1 GPU each. Are you sure, you want this?")
    jobfile_content = get_jobfile_content_vsc5(
        " ".join(command),
        config.experiment_name,
        queue,
        time_in_minutes,
        config.dispatch.conda_env,
        sleep_in_sec,
        n_gpus,
        n_nodes,
    )
    if (n_gpus == 1) and (n_nodes == 1):
        jobfile_content = jobfile_content.replace("#SBATCH -N", "##SBATCH -N")
    _save_and_submit_slurm_job(jobfile_content, run_dir, dry_run)


def dispatch_to_hgx(command, run_dir, config: Configuration, sleep_in_sec, dry_run):
    time_in_minutes = duration_string_to_minutes(config.dispatch.time)
    queue = "hgx" if config.dispatch.queue == "default" else config.dispatch.queue
    qos = config.dispatch.qos or "normal"
    if config.computation.n_local_devices:
        n_gpus = config.computation.n_local_devices
    else:
        n_gpus = 1
    jobfile_content = get_jobfile_content_hgx(
        " ".join(command),
        config.experiment_name,
        queue,
        qos,
        time_in_minutes,
        config.dispatch.conda_env,
        n_gpus,
        n_nodes=1,
        memory_in_gb=config.dispatch.memory,
    )
    _save_and_submit_slurm_job(jobfile_content, run_dir, dry_run)


def dispatch_to_leonardo(command, run_dir, config: Configuration, sleep_in_sec, dry_run):
    # assert config.computation.n_nodes == 1, "Code-base is currently not tested on multi-node."
    time_in_minutes = duration_string_to_minutes(config.dispatch.time)
    if config.dispatch.qos == "default" or config.dispatch.qos is None:
        if time_in_minutes <= 30:
            qos = "boost_qos_dbg"
        elif time_in_minutes <= (24 * 60):
            qos = "normal"
        else:
            qos = "boost_qos_lprod"
    else:
        qos = config.dispatch.qos
    jobfile_content = get_jobfile_content_leonardo(
        " ".join(command),
        config.experiment_name,
        qos,
        time_in_minutes,
        config.dispatch.conda_env,
        n_nodes=config.computation.n_nodes,
    )
    _save_and_submit_slurm_job(jobfile_content, run_dir, dry_run)


def dispatch_to_vsc4(command, run_dir, config: Configuration, sleep_in_sec, dry_run):
    time_in_minutes = duration_string_to_minutes(config.dispatch.time)
    queue = "mem_0096" if config.dispatch.queue == "default" else config.dispatch.queue
    jobfile_content = get_jobfile_content_vsc4(" ".join(command), config.experiment_name, queue, time_in_minutes)
    _save_and_submit_slurm_job(jobfile_content, run_dir, dry_run)


def dispatch_to_juwels(command, run_dir, config: Configuration, sleep_in_sec, dry_run):
    time_in_minutes = duration_string_to_minutes(config.dispatch.time)

    queue_translation = dict(default="booster")
    queue = queue_translation.get(config.dispatch.queue, config.dispatch.queue)
    n_gpus = config.computation.n_local_devices or 4
    n_nodes = config.computation.n_nodes
    if time_in_minutes > 1440:
        logging.warn("Max time on Juwels is 1 day. This will probably crash")
    if (n_nodes > 1) and (n_gpus < 4):
        logging.warn(
            "You requested multiple multi-GPU nodes, using less than 4 GPUs " "each. Are you sure, you want this?"
        )
    jobfile_content = get_jobfile_content_juwels(
        " ".join(command),
        config.experiment_name,
        queue,
        time_in_minutes,
        config.dispatch.conda_env,
        n_gpus,
        n_nodes,
    )
    _save_and_submit_slurm_job(jobfile_content, run_dir, dry_run)


def dispatch_to_vega(command, run_dir, config: Configuration, sleep_in_sec, dry_run):
    time_in_minutes = duration_string_to_minutes(config.dispatch.time)
    queue = "gpu"
    n_gpus = config.computation.n_local_devices or 4
    n_nodes = config.computation.n_nodes
    if time_in_minutes > 2880:
        print("Max time on vega is 2 days. This will probably crash")
    if (n_nodes > 1) and (n_gpus < 4):
        print("You requested multiple multi-GPU nodes, using less than 4 GPUs " "each. Are you sure, you want this?")
    jobfile_content = get_jobfile_content_vega(
        " ".join(command),
        config.experiment_name,
        queue,
        time_in_minutes,
        config.dispatch.conda_env,
        n_gpus,
        n_nodes,
    )
    _save_and_submit_slurm_job(jobfile_content, run_dir, dry_run)


def dispatch_to_local_slurm(command, run_dir, config: Configuration, sleep_in_sec, dry_run):
    time_in_minutes = duration_string_to_minutes(config.dispatch.time)
    # TODO: Allow other queues
    queue = "booster"
    n_gpus = config.computation.n_local_devices or 1
    n_nodes = config.computation.n_nodes
    if (n_nodes > 1) or (n_gpus > 1):
        raise ValueError("Can only run single-node and single-gpu jobs with local_slurm")
    jobfile_content = get_jobfile_content_local_slurm(
        " ".join(command),
        config.experiment_name,
        queue,
        time_in_minutes,
        config.dispatch.conda_env,
        n_gpus,
        n_nodes,
    )
    _save_and_submit_slurm_job(jobfile_content, run_dir, dry_run)


def dispatch_to_baskerville(command, run_dir, config: Configuration, sleep_in_sec, dry_run):
    time_in_minutes = duration_string_to_minutes(config.dispatch.time)
    # TODO: Allow other queues
    queue = "booster"
    n_gpus = config.computation.n_local_devices or 4
    n_nodes = config.computation.n_nodes
    if time_in_minutes > 4320:
        print("Max time on Baskerville is 3 days. This will probably crash")
    if (n_nodes > 1) and (n_gpus < 4):
        print("You requested multiple multi-GPU nodes, using less than 4 GPUs " "each. Are you sure, you want this?")
    jobfile_content = get_jobfile_content_baskerville(
        " ".join(command),
        config.experiment_name,
        queue,
        time_in_minutes,
        config.dispatch.conda_env,
        n_gpus,
        n_nodes,
    )
    _save_and_submit_slurm_job(jobfile_content, run_dir, dry_run)


def append_nfs_to_fullpaths(command):
    ret = copy.deepcopy(command)
    for idx, r in enumerate(ret):
        if r.startswith("/") and os.path.exists(r):
            ret[idx] = "/nfs" + r
    return ret


def _map_dgx_path(path):
    path = Path(path).resolve()
    if str(path).startswith("/home"):
        return "/nfs" + str(path)
    else:
        return path


def dispatch_to_dgx(command, run_dir, config: Configuration, sleep_in_sec, dry_run):
    command = append_nfs_to_fullpaths(command)
    time_in_minutes = duration_string_to_minutes(config.dispatch.time)
    src_dir = _map_dgx_path(Path(__file__).resolve().parent.parent)
    jobfile_content = get_jobfile_content_dgx(
        " ".join(command),
        config.experiment_name,
        _map_dgx_path(os.path.abspath(run_dir)),
        time_in_minutes,
        config.dispatch.conda_env,
        src_dir,
    )

    with open(os.path.join(run_dir, "job.sh"), "w") as f:
        f.write(jobfile_content)
    if not dry_run:
        subprocess.run(["sbatch", "job.sh"], cwd=run_dir)


def duration_string_to_minutes(s):
    match = re.search("([0-9]*[.]?[0-9]+)( *)(.*)", s)
    if match is None:
        raise ValueError(f"Invalid time string: {s}")
    amount, unit = float(match.group(1)), match.group(3).lower()
    if unit in ["d", "day", "days"]:
        return int(amount * 1440)
    elif unit in ["h", "hour", "hours"]:
        return int(amount * 60)
    elif unit in ["m", "min", "mins", "minute", "minutes", ""]:
        return int(amount)
    elif unit in ["s", "sec", "secs", "second", "seconds"]:
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


def get_jobfile_content_vsc5(command, jobname, queue, time, conda_env, sleep_in_seconds, n_local_gpus, n_nodes):
    return f"""#!/bin/bash
#SBATCH -J {jobname}
#SBATCH -N {n_nodes}
#SBATCH --partition {queue}
#SBATCH --qos {queue}
#SBATCH --output GPU.out
#SBATCH --time {time}
#SBATCH --gres=gpu:{n_local_gpus}
#SBATCH --exclude=n3067-005,n3068-001,n3066-015,n3068-003,n3074-004,n3073-003,n3073-006,n3068-005,n3066-001
#SBATCH --ntasks-per-node 1

export MODULEPATH=/opt/sw/vsc4/VSC/Modules/TUWien:/opt/sw/vsc4/VSC/Modules/Intel/oneAPI:/opt/sw/vsc4/VSC/Modules/Parallel-Environment:/opt/sw/vsc4/VSC/Modules/Libraries:/opt/sw/vsc4/VSC/Modules/Compiler:/opt/sw/vsc4/VSC/Modules/Debugging-and-Profiling:/opt/sw/vsc4/VSC/Modules/Applications:/opt/sw/vsc4/VSC/Modules/p71545::/opt/sw/spack-0.17.1/var/spack/environments/zen3/modules/linux-almalinux8-zen:/opt/sw/spack-0.17.1/var/spack/environments/zen3/modules/linux-almalinux8-zen2:/opt/sw/spack-0.17.1/var/spack/environments/zen3/modules/linux-almalinux8-zen3
module purge
module load cuda/11.8.0-gcc-9.4.0-2bqftyz
source /gpfs/opt/sw/spack-0.17.1/opt/spack/linux-almalinux8-zen3/gcc-11.2.0/miniconda3-4.12.0-ap65vga66z2rvfcfmbqopba6y543nnws/etc/profile.d/conda.sh
conda activate {conda_env}
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NVIDIA_TF32_OVERRIDE=0
export WANDB_DIR="${{HOME}}/tmp"
srun {command}"""


def get_jobfile_content_hgx(command, jobname, queue, qos, time, conda_env, n_local_gpus, n_nodes, memory_in_gb):
    memory_in_mb = memory_in_gb * 1000 if memory_in_gb else 80_000 * n_local_gpus
    return f"""#!/bin/bash
#SBATCH -J {jobname}
#SBATCH -N {n_nodes}
#SBATCH -n 1
#SBATCH --cpus-per-task 8
#SBATCH --partition {queue}
#SBATCH --qos {qos}
#SBATCH --output GPU.out
#SBATCH --time {time}
#SBATCH --gres=gpu:{n_local_gpus}
{f"#SBATCH --mem={memory_in_mb}"}

# if $HOME/develop/deeperwin_jaxtest/.venv exists, activate it, else fall back to conda
if [ -d $HOME/develop/deeperwin_jaxtest/.venv ]; then
    source $HOME/develop/deeperwin_jaxtest/.venv/bin/activate
else
    source /opt/anaconda3/etc/profile.d/conda.sh
    conda activate {conda_env}
fi
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NVIDIA_TF32_OVERRIDE=0
export WANDB_DIR="${{HOME}}/tmp"
srun {command}
"""


def get_jobfile_content_leonardo(command, jobname, qos, time, conda_env, n_nodes):
    return f"""#!/bin/bash
#SBATCH -J {jobname}
#SBATCH -N {n_nodes}
#SBATCH -p boost_usr_prod
#SBATCH --qos {qos}
#SBATCH -A L-AUT_Sch-Hoef
#SBATCH --output GPU.out
#SBATCH --time {time}
#SBATCH --gres=gpu:4
#SBATCH --mem=256000
#SBATCH --cpus-per-task 8

# if $HOME/develop/deeperwin_jaxtest/.venv exists, activate it, else fall back to conda
if [ -d $HOME/develop/deeperwin_jaxtest/.venv ]; then
    source $HOME/develop/deeperwin_jaxtest/.venv/bin/activate
else
    source $HOME/envs/{conda_env}/bin/activate
fi
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NVIDIA_TF32_OVERRIDE=0
export WANDB_MODE=offline
srun {command}
"""


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
export WANDB_DIR="/nfs${{HOME}}/tmp"
export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1
{command}"""


def get_jobfile_content_juwels(
    command,
    jobname,
    queue,
    time,
    conda_env,
    n_local_gpus,
    n_nodes,
    user_email="h.sutterud21@imperial.ac.uk",
    account="neuralwf",
    env_dir="$ENVS",
):
    assert queue == "booster"
    return f"""#!/bin/bash
#SBATCH -J {jobname}
#SBATCH -A {account}
#SBATCH -N {n_nodes}
#SBATCH -t {time}
#SBATCH --partition={queue}
#SBATCH --gres=gpu:{n_local_gpus}
#SBATCH --output GPU.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user={user_email}

module purge
module load Stages/2022
module load GCC/11.2.0 Python/3.9.6 CUDA/11.5 cuDNN/8.3.1.22-CUDA-11.5

source {env_dir}/{conda_env}/{conda_env}/bin/activate

export WANDB_MODE=offline
echo "remember to sync wandb after job is done!"

srun --nodes={n_nodes} \
     --gres=gpu:{n_local_gpus} \
     --export=ALL \
     --output=GPU.out \
     --error=GPU.out \
     {command}"""


def get_jobfile_content_vega(command, jobname, queue, time, conda_env, n_local_gpus, n_nodes):
    assert queue == "gpu"
    return f"""#!/bin/bash
#SBATCH --partition={queue}
#SBATCH --nodes={n_nodes}
#SBATCH --time={time}
#SBATCH --gres=gpu:{n_local_gpus}
#SBATCH --mem=256GB

module purge
module load Python/3.9.6-GCCcore-11.2.0
module load git

env_dir=$ENVS/{conda_env}
source $env_dir/bin/activate

export NVIDIA_TF32_OVERRIDE=0
export WANDB_DIR="${{HOME}}/tmp"

{command}"""


def get_jobfile_content_local_slurm(
    command,
    jobname,
    queue,
    time,
    conda_env,
    n_local_gpus,
    n_nodes,
    user_email="h.sutterud21@imperial.ac.uk",
):
    return f"""#!/bin/bash
#SBATCH -J {jobname}
#SBATCH -N {n_nodes}
#SBATCH -t {time}
#SBATCH --output GPU.out
#SBATCH --mem 15G
#SBATCH --cpus-per-task 8

conda activate {conda_env}

{command}"""


def get_jobfile_content_baskerville(
    command,
    jobname,
    queue,
    time,
    conda_env,
    n_local_gpus,
    n_nodes,
    user_email="h.sutterud21@imperial.ac.uk",
):
    return f"""#!/bin/bash
#SBATCH -J {jobname}
#SBATCH -N {n_nodes}
#SBATCH --qos epsrc
#SBATCH -t {time}
#SBATCH --output GPU.out
#SBATCH --gres gpu:{n_local_gpus}
#SBATCH --mem-per-gpu 40G
#SBATCH --cpus-per-gpu=4
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user={user_email}

module purge; module load baskerville
module load Python/3.9.5-GCCcore-10.3.0 CUDA/11.3.1 cuDNN/8.2.1.32-CUDA-11.3.1

# Put the ENVS alias in this file
source ~/.bash_aliases

env_dir=$ENVS/{conda_env}/{conda_env}
source $env_dir/bin/activate

export NVIDIA_TF32_OVERRIDE=0

{command}"""


def _determine_hpc_sytstem():
    machine_name = os.uname()[1]
    if machine_name == "gpu1-mat":  # HGX
        return "hgx"
    elif "leonardo" in machine_name:
        return "leonardo"
    elif os.path.exists("/etc/slurm/slurm.conf"):
        with open("/etc/slurm/slurm.conf") as f:
            slurm_conf = f.readlines()
            if "slurm.vda.univie.ac.at" in "".join(slurm_conf):
                return "dgx"
    elif os.environ.get("HOSTNAME", "").startswith("l5"):
        return "vsc5"
    elif "HPC_SYSTEM" in os.environ:
        return os.environ["HPC_SYSTEM"].lower()  # vsc3 or vsc4
    elif "vega" in machine_name:
        return "vega"
    else:
        return "local"


def dispatch_job(command, job_dir, config, sleep_in_sec, dry_run=False):
    dispatch_to = config.dispatch.system
    if dispatch_to == "auto":
        dispatch_to = _determine_hpc_sytstem()
    logging.info(f"Dispatching command {' '.join(command)} to: {dispatch_to}")
    dispatch_func = dict(
        local=dispatch_to_local,
        local_background=dispatch_to_local_background,
        local_slurm=dispatch_to_local_slurm,
        vsc4=dispatch_to_vsc4,
        vsc5=dispatch_to_vsc5,
        dgx=dispatch_to_dgx,
        hgx=dispatch_to_hgx,
        leonardo=dispatch_to_leonardo,
        juwels=dispatch_to_juwels,
        baskerville=dispatch_to_baskerville,
        vega=dispatch_to_vega,
    )[dispatch_to]
    dispatch_func(command, job_dir, config, sleep_in_sec, dry_run)
