#!/usr/bin/env python3
"""
CLI to process a single molecule.
"""
import os
import sys
import time
if len(sys.argv) > 2:
    try:
        print(f"Sleeping for collision avoidance: {sys.argv[2]} seconds")
        time.sleep(float(sys.argv[2]))
        sys.argv = sys.argv[:-1]
    except ValueError:
        print("Invalid sleeping time for collision avoidance specified. Skipping sleeping.")

from deeperwin.available_gpus import get_free_GPU_id
os.environ['CUDA_VISIBLE_DEVICES'] = get_free_GPU_id()

import argparse
import jax.numpy as jnp
from jax.config import config as jax_config
from jax.lib import xla_bridge
from ruamel import yaml

from deeperwin.configuration import Configuration
from deeperwin.dispatch import prepare_checkpoints, contains_run, load_run
from deeperwin.loggers import LoggerCollection, build_dpe_root_logger
from deeperwin.mcmc import MCMCState, MetropolisHastingsMonteCarlo, resize_nr_of_walkers
from deeperwin.model import build_log_psi_squared
from deeperwin.optimization import optimize_wavefunction, evaluate_wavefunction
from deeperwin.utils import getCodeVersion, prepare_data_for_logging, get_number_of_params, split_trainable_params, merge_trainable_params, unpickle_opt_state

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optimization of wavefunction for a single molecular configuration.")
    parser.add_argument("config_file", default="config.yml", help="Path to input config file", nargs="?")
    args = parser.parse_args()

    # load and parse config
    config = Configuration.load(args.config_file)

    # Initialize loggers for logging debug/info messages
    logger = build_dpe_root_logger(config.logging.basic)

    # update restart config with passed config file
    is_restart = config.restart is not None
    params_from_restart = False
    loaded_mcmc_state = None
    opt_state = None
    clipping_state = None
    if is_restart:
        restart_results, restart_config = load_run(config.restart.path)
        with open(args.config_file) as f:
            raw_config = yaml.safe_load(f)
        config = Configuration.update_config(restart_config.dict(), raw_config.items())[1]
        if config.reuse is not None:
            logger.warning("Restart overrules re-use of weights: Ignoring re-use config-option")
            config.reuse = None
        if config.restart.recursive:
            logger.warning('Ignoring setting restart.recursive.')
        params_from_restart = config.restart.reuse_params
        loaded_mcmc_state = restart_results['weights']['mcmc'] if config.restart.reuse_mcmc_state else None
        if config.restart.reuse_opt_state:
            if 'opt' in restart_results['weights']:
                opt_state = unpickle_opt_state(restart_results['weights']['opt'])
            else:
                logger.warning("No optimizer state stored in loaded restart-file. Re-initializeing opt-state.")
        if config.restart.reuse_clipping_state:
            clipping_state = restart_results['weights'].get('clipping', None)
        config.optimization.n_epochs_prev = restart_config.optimization.n_epochs + restart_config.optimization.n_epochs_prev

    # save full config
    config.save("full_config.yml")

    # init loggers
    loggers = LoggerCollection(config.logging, config.experiment_name)
    loggers.on_run_begin()
    loggers.log_params(config.get_as_flattened_dict())
    loggers.log_tags(config.logging.tags)


    jax_config.update("jax_enable_x64", config.computation.float_precision == "float64")
    jax_config.update("jax_disable_jit", config.computation.disable_jit)
    logger.debug(f"Used hardware: {xla_bridge.get_backend().platform}")

    # Initialization of model
    loggers.log_param("code_version", getCodeVersion())
    _, log_psi_squared, trainable_params, fixed_params = build_log_psi_squared(config.model, config.physical,
                                                                               init_fixed_params=not params_from_restart)
    if params_from_restart:
        fixed_params = restart_results['weights']['fixed']
        trainable_params = restart_results['weights']['trainable']

    if config.reuse is not None:
        data_to_reuse, _ = load_run(config.reuse.path)
        if config.reuse.reuse_trainable_params:
            if config.reuse.reuse_modules is None:
                params_to_reuse = data_to_reuse['weights']['trainable'] # reuse all modules
            else:
                # Only reuse selected modules and split off all unused ones
                params_to_reuse, _ = split_trainable_params(data_to_reuse['weights']['trainable'], config.reuse.reuse_modules)
            logger.debug(f"Reusing {get_number_of_params(params_to_reuse)} weights")
            trainable_params = merge_trainable_params(params_to_reuse, trainable_params)
        loaded_mcmc_state = restart_results['weights']['mcmc'] if config.reuse.reuse_mcmc_state else None

    loggers.log_metrics(dict(E_hf=fixed_params["E_hf"], E_casscf=fixed_params["E_casscf"]))
    loggers.log_param("n_params", get_number_of_params(trainable_params))

    # Initialization of MCMC and restart/reload of parameters
    mcmc = MetropolisHastingsMonteCarlo(config.mcmc)
    mcmc_state = loaded_mcmc_state or MCMCState.initialize_around_nuclei(config.mcmc.n_walkers_opt, config.physical)
    mcmc_state.log_psi_sqr = log_psi_squared(*mcmc_state.model_args, trainable_params, fixed_params)

    # Wavefunction optimization
    if config.optimization.n_epochs > 0:
        logger.info("Starting optimization...")
        checkpoints = prepare_checkpoints(".", config.optimization.checkpoints, config) if len(
            config.optimization.checkpoints) > 0 else {}
        mcmc_state, trainable_params, opt_state, clipping_state = optimize_wavefunction(
            log_psi_squared, trainable_params, fixed_params, mcmc, mcmc_state, config.optimization, checkpoints, loggers, opt_state, clipping_state
        )
    full_data = prepare_data_for_logging(trainable_params, fixed_params, mcmc_state, opt_state, clipping_state)
    loggers.log_weights(full_data)

    # Wavefunction evaluation
    mcmc_state = resize_nr_of_walkers(mcmc_state, config.mcmc.n_walkers_eval)
    if config.evaluation.n_epochs > 0:
        logger.info("Starting evaluation...")
        E_eval, forces_eval, mcmc_state = evaluate_wavefunction(
            log_psi_squared, trainable_params, fixed_params, mcmc, mcmc_state, config.evaluation, loggers
        )

        # Postprocessing
        E_mean = jnp.nanmean(E_eval)
        E_mean_sigma = jnp.nanstd(E_eval) / jnp.sqrt(len(E_eval))

        loggers.log_metrics(dict(E_mean=E_mean, E_mean_sigma=E_mean_sigma), metric_type="eval")
        if config.physical.E_ref is not None:
            error_eval, sigma_eval = 1e3 * (E_mean - config.physical.E_ref), 1e3 * E_mean_sigma
            loggers.log_metrics(dict(error_eval=error_eval, sigma_error_eval=sigma_eval,
                                     error_plus_2_stdev=error_eval + 2 * sigma_eval), force_log=True)
        if forces_eval is not None:
            forces_mean = jnp.nanmean(forces_eval, axis=0)
            loggers.log_metric('forces_mean', forces_mean, force_log=True)
    loggers.on_run_end()
