"""
CLI to process a single molecule.
"""

#!/usr/bin/env python3
import argparse
import logging
import os

import jax.numpy as jnp
from jax.config import config as jax_config
from jax.lib import xla_bridge
from ruamel import yaml

from deeperwin.configuration import Configuration
from deeperwin.dispatch import prepare_checkpoints, contains_run, load_run
from deeperwin.loggers import LoggerCollection
from deeperwin.mcmc import MCMCState, MetropolisHastingsMonteCarlo, resize_nr_of_walkers
from deeperwin.model import build_log_psi_squared, get_number_of_params
from deeperwin.optimization import optimize_wavefunction, evaluate_wavefunction
from deeperwin.utils import getCodeVersion, make_opt_state_picklable

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optimization of wavefunction for a single molecular configuration.")
    parser.add_argument("config_file", default="config.yml", help="Path to input config file", nargs="?")
    args = parser.parse_args()

    # load and parse config
    config = Configuration.load(args.config_file)

    # update restart config with passed config file
    is_restart = config.restart is not None
    params_from_restart = False
    mcmc_from_restart = False
    if is_restart:
        if config.restart.recursive:
            logging.warning('Ignoring setting restart.recursive.')
        if not contains_run(config.restart.path):
            raise Exception("Restart path does not contain a valid run.")
        restart_results, restart_config = load_run(config.restart.path)
        with open(args.config_file) as f:
            raw_config = yaml.safe_load(f)
        config = Configuration.update_config(restart_config.dict(), raw_config.items())[1]
        params_from_restart = config.restart.reuse_params
        mcmc_from_restart = config.restart.reuse_mcmc_state
        config.optimization.n_epochs_prev = restart_config.optimization.n_epochs + restart_config.optimization.n_epochs_prev
    # save full config
    config.save("full_config.yml")

    # init loggers
    loggers = LoggerCollection(config.logging, config.experiment_name)
    loggers.on_run_begin()
    loggers.log_params(config.get_as_flattened_dict())
    loggers.log_tags(config.logging.tags)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if config.computation.use_gpu else "-1"

    jax_config.update("jax_enable_x64", config.computation.float_precision == "float64")
    jax_config.update("jax_disable_jit", config.computation.disable_jit)
    logging.debug(f"Used hardware: {xla_bridge.get_backend().platform}")

    # Initialization of model
    loggers.log_param("code_version", getCodeVersion())
    _, log_psi_squared, trainable_params, fixed_params = build_log_psi_squared(config.model, config.physical,
                                                                               init_fixed_params=not params_from_restart)
    if params_from_restart:
        fixed_params = restart_results['weights']['fixed']
        trainable_params = restart_results['weights']['trainable']
    loggers.log_metrics(dict(E_hf=fixed_params["E_hf"], E_casscf=fixed_params["E_casscf"]))
    loggers.log_param("n_params", get_number_of_params(trainable_params))

    # Initialization of MCMC and restart/reload of parameters
    mcmc = MetropolisHastingsMonteCarlo(config.mcmc)
    mcmc_state = MCMCState.initialize_around_nuclei(config.mcmc.n_walkers_opt,
                                                    config.physical) if not mcmc_from_restart else \
    restart_results['weights']['mcmc']
    mcmc_state.log_psi_sqr = log_psi_squared(*mcmc_state.model_args, trainable_params, fixed_params)

    # Wavefunction optimization
    if config.optimization.n_epochs > 0:
        logging.info("Starting optimization...")
        checkpoints = prepare_checkpoints(".", config.optimization.checkpoints, config) if len(
            config.optimization.checkpoints) > 0 else {}
        mcmc_state, trainable_params, opt_state = optimize_wavefunction(
            log_psi_squared, trainable_params, fixed_params, mcmc, mcmc_state, config.optimization, checkpoints, loggers
        )
    full_data = dict(trainable=trainable_params, fixed=fixed_params, mcmc=mcmc_state)
    if config.logging.log_opt_state and config.optimization.n_epochs > 0:
        full_data['opt'] = make_opt_state_picklable(opt_state)
    loggers.log_weights(full_data)

    # Wavefunction evaluation
    mcmc_state = resize_nr_of_walkers(mcmc_state, config.mcmc.n_walkers_eval)
    if config.evaluation.n_epochs > 0:
        logging.info("Starting evaluation...")
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
                                     error_plus_2_stdev=error_eval + 2 * sigma_eval))
        if forces_eval is not None:
            forces_mean = jnp.nanmean(forces_eval, axis=0)
            loggers.log_metric('forces_mean', forces_mean)
    loggers.on_run_end()
