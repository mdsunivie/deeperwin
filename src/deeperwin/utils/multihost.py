import logging
import os
import subprocess
import jax
from jax.lib import xla_bridge


def disable_slave_loggers(logger):
    if jax.process_index() > 0:
        logger.debug(f"DeepErwin: Disabling logging for process {jax.process_index()}")
        logging.disable()


def configure_hardware(
    config,
) -> None:
    if config.computation.n_nodes > 1:
        init_multi_host_on_slurm(config.computation.n_nodes)

    if not config.computation.n_local_devices:
        config.computation.n_local_devices = jax.local_device_count()
    else:
        assert jax.local_device_count() == config.computation.n_local_devices

    used_hardware = xla_bridge.get_backend().platform
    if config.computation.require_gpu and (used_hardware == "cpu"):
        raise ValueError("Required GPU, but no GPU available: Aborting.")

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Process: {jax.process_index()}; Used hardware: {used_hardware}; Local device count: {jax.local_device_count()}; Global device count: {jax.device_count()}")


def init_multi_host_on_slurm(n_nodes=None):
    for key in ['SLURM_JOB_NODELIST', 'SLURM_NODEID', 'SLURM_JOB_NUM_NODES']:
        assert key in os.environ, "Multi-host only implemented for SLURM"
    process_count = int(os.environ["SLURM_JOB_NUM_NODES"])
    if n_nodes:
        assert process_count == n_nodes, "Number of nodes requested does not match number of SLURM nodes"

    r = subprocess.run(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
                       capture_output=True, encoding="utf-8")
    master_hostname = r.stdout.split("\n")[0]
    process_id = int(os.environ["SLURM_NODEID"])
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        local_device_ids = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        local_device_ids = None
    print(f"Initializing process {process_id} / {process_count} with local_devices={local_device_ids}")
    jax.distributed.initialize(master_hostname + ":8080", process_count, process_id, local_device_ids)
    assert jax.process_index() == process_id, f"jax.process_index() != process_id; {jax.process_index()} != {process_id}"
