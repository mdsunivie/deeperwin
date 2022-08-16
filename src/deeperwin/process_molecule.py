#!/usr/bin/env python3
"""
CLI to process a single molecule.
"""
import logging
import os
import sys
from deeperwin.configuration import Configuration
import ruamel.yaml as yaml

def process_molecule(config_file):
    with open(config_file, "r") as f:
        raw_config = yaml.YAML().load(f)
    config: Configuration = Configuration.parse_obj(raw_config)

    # Set environment variable to control jax behaviour before importing jax
    if config.computation.disable_tensor_cores:
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    if config.computation.force_device_count and config.computation.n_local_devices:
        os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={config.computation.n_local_devices}'

    from jax.config import config as jax_config
    jax_config.update("jax_enable_x64", config.computation.float_precision == "float64")
    import jax
    from deeperwin.utils import getCodeVersion, merge_params, init_multi_host_on_slurm, replicate_across_devices

    if config.computation.n_nodes > 1:
        init_multi_host_on_slurm()

    # These imports can only take place after we have set the jax_config options
    import chex
    if config.computation.disable_jit:
        chex.fake_pmap_and_jit().start()
    from jax.lib import xla_bridge
    from deeperwin.model import build_log_psi_squared
    from deeperwin.optimization import optimize_wavefunction, evaluate_wavefunction, pretrain_orbitals
    from deeperwin.checkpoints import load_data_for_reuse, delete_obsolete_checkpoints
    from deeperwin.loggers import LoggerCollection, build_dpe_root_logger
    import haiku as hk
    import numpy as np

    if not config.computation.n_local_devices:
        config.computation.n_local_devices = jax.local_device_count()
    else:
        assert jax.local_device_count() == config.computation.n_local_devices

    used_hardware = xla_bridge.get_backend().platform
    if config.computation.require_gpu and (used_hardware == "cpu"):
        raise ValueError("Required GPU, but no GPU available: Aborting.")

    # Initialize loggers for logging debug/info messages
    logger = build_dpe_root_logger(config.logging.basic)
    if jax.process_index() > 0:
        logger.debug(f"DeepErwin: Disabling logging for process {jax.process_index()}")
        logging.disable()
    if config.computation.rng_seed is None:
        rng_seed = np.random.randint(2**31, size=())
        config.computation.rng_seed = int(replicate_across_devices(rng_seed)[0])
    rng_seed = config.computation.rng_seed
    np.random.seed(rng_seed)

    logger.debug(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logger.debug(f"Used hardware: {used_hardware}; Local device count: {jax.local_device_count()}; Global device count: {jax.device_count()}")

    # When reusing/restarting and old run: Merge configs and load data
    if config.reuse is not None:
        config, params_to_reuse, fixed_params, mcmc_state, opt_state, clipping_state = load_data_for_reuse(config, raw_config)
    else:
        params_to_reuse, fixed_params, mcmc_state, opt_state, clipping_state = None, None, None, None, None

    # Build wavefunction and initialize parameters
    log_psi_squared, orbital_func, params, fixed_params = build_log_psi_squared(config.model, config.physical, fixed_params, rng_seed)
    if params_to_reuse:
        params = merge_params(params, params_to_reuse)

    # Log config and metadata of run
    if jax.process_index() == 0:
        loggers = LoggerCollection(config.logging, config.experiment_name)
        loggers.on_run_begin()
        if config.logging.wandb:
            import wandb
            config.logging.wandb.id = wandb.run.id

        loggers.log_config(config)
        loggers.log_tags(config.logging.tags)
        loggers.log_param("code_version", getCodeVersion())
        loggers.log_param("n_params", hk.data_structures.tree_size(params))
        if "baseline_energies" in fixed_params:
            loggers.log_metrics(fixed_params["baseline_energies"])
        config.save("full_config.yml")
    else:
        loggers = None

    # STEP 1: Supervised pre-training of wavefunction orbitals
    if config.pre_training and config.pre_training.n_epochs > 0:
        logger.info("Starting pre-training of orbitals...")
        params, _, mcmc_state = pretrain_orbitals(
            orbital_func, mcmc_state, params, fixed_params, config.pre_training, config.physical, config.model,
            rng_seed, loggers,
        )

    # STEP 2: Unsupervised variational wavefunction optimization
    if config.optimization.n_epochs > 0:
        logger.info("Starting optimization...")
        mcmc_state, params, opt_state, clipping_state = optimize_wavefunction(
            log_psi_squared,
            params,
            fixed_params,
            mcmc_state,
            config.optimization,
            config.physical,
            rng_seed,
            loggers,
            opt_state,
            clipping_state,
        )

    # STEP 3: Wavefunction evaluationA
    if config.evaluation.n_epochs > 0:
        logger.info("Starting evaluation...")
        eval_history, mcmc_state = evaluate_wavefunction(
            log_psi_squared,
            params,
            fixed_params,
            mcmc_state,
            config.evaluation,
            config.physical,
            rng_seed,
            loggers,
            config.optimization.n_epochs_total,
        )
    if jax.process_index() == 0:
        loggers.log_checkpoint(config.optimization.n_epochs_total, params, fixed_params, mcmc_state, opt_state, clipping_state)
        delete_obsolete_checkpoints(config.optimization.n_epochs_total, config.optimization.checkpoints)
        loggers.on_run_end()

if __name__ == '__main__':
    process_molecule(sys.argv[1])
