#!/usr/bin/env python3
"""
CLI to process a single molecule.
"""
import os
import sys
import time
import argparse
from jax.config import config as jax_config
from ruamel import yaml
from deeperwin.configuration import Configuration, update_config
from deeperwin.available_gpus import get_free_GPU_id
import numpy as np

parser = argparse.ArgumentParser(description="Optimization of wavefunction for a single molecular configuration.")
parser.add_argument("config_file", default="config.yml", help="Path to input config file")
parser.add_argument("startup_delay", default=0, help="Sleep for a given nr of seconds before allocating a GPU to avoid collisions", nargs="?", type=int)
args = parser.parse_args()
if args.startup_delay > 0:
    print(f"Sleeping for collision avoidance: {sys.argv[2]} seconds")
    time.sleep(args.startup_delay)

# load and parse config
config = Configuration.load(args.config_file)

os.environ['CUDA_VISIBLE_DEVICES'] = get_free_GPU_id(config.computation.require_gpu)
if config.computation.disable_tensor_cores:
    os.environ['NVIDIA_TF32_OVERRIDE'] = "0"
jax_config.update("jax_enable_x64", config.computation.float_precision == "float64")
jax_config.update("jax_disable_jit", config.computation.disable_jit)

# These imports can only take place after we have set the jax_config options
import jax
import jax.numpy as jnp
from jax.lib import xla_bridge
from deeperwin.loggers import LoggerCollection, build_dpe_root_logger
from deeperwin.mcmc import MCMCState, MetropolisHastingsMonteCarlo, resize_nr_of_walkers
from deeperwin.model import build_log_psi_squared
from deeperwin.optimization import optimize_wavefunction, evaluate_wavefunction, pretrain_orbitals
from deeperwin.utils import getCodeVersion, prepare_data_for_logging, get_number_of_params, split_trainable_params, merge_trainable_params, \
    unpickle_opt_state, load_run

# Initialize loggers for logging debug/info messages
logger = build_dpe_root_logger(config.logging.basic)

# update restart config with passed config file
params_from_restart = False
mcmc_state = None
opt_state = None
clipping_state = None
fixed_params = None

# init loggers
logger.debug(f"Used hardware: {xla_bridge.get_backend().platform}")

if config.reuse is not None:
    data_to_reuse, old_config = load_run(config.reuse.path, True, config.reuse.ignore_extra_settings)
    if config.reuse.reuse_config:
        with open(args.config_file) as f:
            raw_config = yaml.safe_load(f)
        config = update_config(old_config, raw_config)
    if config.reuse.continue_n_epochs:
        config.optimization.n_epochs_prev = old_config.optimization.n_epochs + old_config.optimization.n_epochs_prev
    if config.reuse.skip_burn_in:
        config.mcmc.n_burn_in_opt = 0
    if config.reuse.skip_pretraining:
        config.pre_training = None
    if config.reuse.reuse_opt_state:
        if 'opt' in data_to_reuse['weights']:
            opt_state = unpickle_opt_state(data_to_reuse['weights']['opt'])
            logger.debug(f"Reusing opt-state")
        else:
            logger.warning("No optimizer state stored in loaded restart-file. Re-initializeing opt-state.")
    if config.reuse.reuse_clipping_state:
        clipping_state = data_to_reuse['weights'].get('clipping', None)
        if clipping_state is None:
            logger.warning("No clipping state stored in loaded restart-file. Re-initializing clipping-state.")
    if config.reuse.reuse_trainable_params:
        if config.reuse.reuse_modules is None:
            params_to_reuse = data_to_reuse['weights']['trainable'] # reuse all modules
        else:
            # Only reuse selected modules and split off all unused ones
            params_to_reuse, _ = split_trainable_params(data_to_reuse['weights']['trainable'], config.reuse.reuse_modules)
        logger.debug(f"Reusing {get_number_of_params(params_to_reuse)} weights")
    if config.reuse.reuse_fixed_params:
        fixed_params = data_to_reuse['weights']['fixed']
    if config.reuse.reuse_mcmc_state:
        mcmc_state = data_to_reuse['weights']['mcmc']
        if config.reuse.randomize_mcmc_rng:
            new_seed = np.random.randint(2**16)
            logger.debug(f"Selecting new seed for MCMC rng_state: {new_seed}")
            mcmc_state.rng_state = jax.random.PRNGKey(new_seed)

loggers = LoggerCollection(config.logging, config.experiment_name)
loggers.on_run_begin()
loggers.log_params(config.get_as_flattened_dict())
loggers.log_tags(config.logging.tags)
loggers.log_param("code_version", getCodeVersion())

# Initialization of model
log_psi_squared, orbital_func, trainable_params, fixed_params = build_log_psi_squared(config.model, config.physical, fixed_params)
if config.reuse and config.reuse.reuse_trainable_params:
    trainable_params = merge_trainable_params(params_to_reuse, trainable_params)

# save full config
config.save("full_config.yml")

if config.logging.wandb is not None:
    import wandb
    config.logging.wandb.id = wandb.run.id

if 'baseline_energies' in fixed_params:
    loggers.log_metrics(fixed_params['baseline_energies'])
loggers.log_param("n_params", get_number_of_params(trainable_params))

# Initialization of MCMC and restart/reload of parameters
mcmc = MetropolisHastingsMonteCarlo(config.mcmc)
mcmc_state = mcmc_state or MCMCState.initialize_around_nuclei(config.mcmc.n_walkers_opt, config.physical)
mcmc_state.log_psi_sqr = log_psi_squared(*mcmc_state.model_args, trainable_params, fixed_params)
if config.pre_training and config.pre_training.n_epochs > 0:
    trainable_params, _, _ = pretrain_orbitals(orbital_func, mcmc, mcmc_state, trainable_params, fixed_params,
                                               config.pre_training,
                                               config.physical,
                                               config.model,
                                               loggers)

# Wavefunction optimization
if config.optimization.n_epochs > 0:
    logger.info("Starting optimization...")
    mcmc_state, trainable_params, opt_state, clipping_state = optimize_wavefunction(
        log_psi_squared, trainable_params, fixed_params, mcmc, mcmc_state, config.optimization, config.optimization.checkpoints, loggers, opt_state, clipping_state,
        E_ref=config.physical.E_ref, use_profiler=config.computation.use_profiler
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
    E_mean = np.nanmean(np.array(E_eval, dtype=float))
    E_mean_sigma = np.nanstd(np.array(E_eval, dtype=float)) / np.sqrt(len(E_eval))

    loggers.log_metrics(dict(E_mean=E_mean, E_mean_sigma=E_mean_sigma), metric_type="eval")
    if config.physical.E_ref is not None:
        error_eval, sigma_eval = 1e3 * (E_mean - config.physical.E_ref), 1e3 * E_mean_sigma
        loggers.log_metrics(dict(error_eval=error_eval, sigma_error_eval=sigma_eval,
                                 error_plus_2_stdev=error_eval + 2 * sigma_eval), force_log=True)
    if forces_eval is not None:
        forces_mean = jnp.nanmean(forces_eval, axis=0)
        loggers.log_metric('forces_mean', forces_mean, force_log=True)
loggers.on_run_end()

if config.dispatch.split_opt is not None and len(config.dispatch.split_opt)>0:
    import ruamel
    from dispatch import dump_config_dict, get_fname_fullpath
    import subprocess

    with open(args.config_file) as f:
        data = ruamel.yaml.YAML().load(f)

    next_epoch = config.dispatch.split_opt.pop(0)
    n_epochs_opt = next_epoch - (config.optimization.n_epochs + config.optimization.n_epochs_prev)
    n_epochs_eval = 0 if len(config.dispatch.split_opt) >= 1 else config.dispatch.eval_epochs
    path = os.getcwd()

    config_restart = dict(
        experiment_name=config.experiment_name + f"_{next_epoch}",
        physical=data['physical'],
        mcmc=dict(
            n_burn_in_opt=0
        ),
        pre_training=dict(
            n_epochs=0
        ),
        optimization=dict(
            n_epochs=n_epochs_opt
        ),
        evaluation=dict(
            n_epochs=n_epochs_eval
        ),
        reuse=dict(
            path=path
        ),
        dispatch=dict(
            queue=config.dispatch.queue,
            split_opt=config.dispatch.split_opt,
            eval_epochs=config.dispatch.eval_epochs
        )
    )

    path = "/".join(path.split("/")[:-1])
    config_name = f"config_restart_{next_epoch}.yml"
    dump_config_dict(path, config_restart, config_name)

    command = ["python", str(get_fname_fullpath("main.py")), "-i", "../" + config_name]
    subprocess.run(command)