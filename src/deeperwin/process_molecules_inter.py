"""
CLI to process multiple molecules with interdependent optimization.
"""

import argparse
import copy
import logging
import os
import time
from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jax._src.util import unzip2
from jax.config import config as jax_config
from jax.experimental.optimizers import OptimizerState
from jax.lib import xla_bridge
from jax.tree_util import tree_unflatten, tree_flatten

from deeperwin.configuration import Configuration, SharedOptimizationConfig, OptimizationConfig
from deeperwin.dispatch import idx_to_job_name, setup_job_dir, prepare_checkpoints
from deeperwin.evaluation import evaluate_wavefunction
from deeperwin.loggers import LoggerCollection
from deeperwin.mcmc import MCMCState, MetropolisHastingsMonteCarlo, resize_nr_of_walkers, calculate_metrics
from deeperwin.model import build_log_psi_squared, get_number_of_params
from deeperwin.optimization import build_grad_loss, build_optimizer
from deeperwin.utils import getCodeVersion, make_opt_state_picklable


@dataclass
class WF:
    physical = None
    init_trainable_params = None
    fixed_params = None
    mcmc_state = None
    opt_state = None
    clipping_params: Tuple[float] = (0, 1000)
    checkpoints = {}
    loggers = None
    current_metrics = {}
    n_opt_epochs: int = 0
    age: int = 0


def is_shared_module(key, shared_modules):
    if shared_modules is None:
        return False
    return key in shared_modules


def update_shared_weights(dest_weights, src_weights, shared_modules):
    for mod in src_weights.keys():
        if is_shared_module(mod, shared_modules):
            dest_weights[mod] = src_weights[mod]
    return dest_weights


def init_wf(config: Configuration):
    wfs = []

    physical_configs = config.physical.set_from_changes()
    for i, p in enumerate(physical_configs):  # self.shared_opt_config.config_changes):
        logging.info(f"Init wavefunction {i}...")
        # init WF object
        wf = WF()
        wf.physical = p

        # init parameters
        if i == 0:
            _, log_psi_squared, wf.init_trainable_params, wf.fixed_params = build_log_psi_squared(config.model, p)
        else:
            _, _, wf.init_trainable_params, wf.fixed_params = build_log_psi_squared(config.model, p)

        if i != 0:
            wf.init_trainable_params = update_shared_weights(wf.init_trainable_params, wfs[0].init_trainable_params,
                                                             config.optimization.shared_optimization.shared_modules)

        # MCMC state
        wf.mcmc_state = MCMCState.initialize_around_nuclei(config.mcmc.n_walkers_opt, p)
        wf.mcmc_state.log_psi_sqr = log_psi_squared(*wf.mcmc_state.model_args, wf.init_trainable_params,
                                                    wf.fixed_params)

        # make folder for single WF (stores adjusted config and logger data)
        job_name = idx_to_job_name(i)
        job_dir = setup_job_dir(".", job_name)

        # init loggers
        loggers = LoggerCollection(config.logging, config.experiment_name + "_" + job_name, save_path=job_name,
                                   prefix=f"{job_name}_")
        loggers.on_run_begin()
        # loggers.log_params(config.get_as_flattened_dict())
        loggers.log_tags(config.logging.tags)
        loggers.log_metrics(dict(E_hf=wf.fixed_params["E_hf"], E_casscf=wf.fixed_params["E_casscf"]))
        loggers.log_param("n_params", get_number_of_params(wf.init_trainable_params))
        wf.loggers = loggers

        # save full config for single wavefunction
        config_wf = copy.deepcopy(config)
        config_wf.physical = p
        config_wf.optimization.shared_optimization = None
        config_wf.save(os.path.join(job_dir, "full_config.yml"))

        # prepare checkpoints
        wf.checkpoints = prepare_checkpoints(job_dir, config.optimization.checkpoints, config) if len(
            config.optimization.checkpoints) > 0 else {}
        wfs.append(wf)

    mcmc = MetropolisHastingsMonteCarlo(config.mcmc)
    return log_psi_squared, mcmc, wfs


def warm_up_opt(wfs, mcmc, config: Configuration, log_psi_squared):  # could be moved to init
    grad_loss_func = build_grad_loss(log_psi_squared, config.optimization.clipping)

    for i, wf in enumerate(wfs):
        logging.info(f"Starting warm-up for wf {i}...")

        fixed_params = wf.fixed_params
        trainable_params = wf.init_trainable_params
        mcmc_state = wf.mcmc_state

        mcmc_state = mcmc.run_burn_in_opt(log_psi_squared, (trainable_params, fixed_params), mcmc_state)
        wfs[i].mcmc_state = mcmc_state

        opt_get_params, optimize_epoch, opt_state = build_optimizer(log_psi_squared, grad_loss_func, mcmc,
                                                                    trainable_params, fixed_params, config.optimization,
                                                                    config.mcmc.n_walkers_opt, mcmc_state=None)
        wfs[i].opt_state = opt_state

    return wfs, opt_get_params, optimize_epoch

def update_adam_opt_state(opt_state, params):
    params, tree2 = tree_flatten(params)

    def do_nothing(state, params):
        x, m, v = state
        return params, m, v

    states_flat, tree, subtrees = opt_state
    states = map(tree_unflatten, subtrees, states_flat)
    new_states = map(do_nothing, states, params)
    new_states_flat, subtrees2 = unzip2(map(tree_flatten, new_states))
    return OptimizerState(new_states_flat, tree, subtrees)

def update_kfac_opt_state(opt_state_old, opt_state, params, internal_optimzier):

    if internal_optimzier.name == "adam":
        adam_state = update_adam_opt_state(opt_state_old[0], params)
        opt_state = (adam_state, params, opt_state[-2], opt_state[-1])
    else:
        opt_state = (None, params, opt_state[-2], opt_state[-1])
    return opt_state


def update_bfgs_opt_state(opt_state_old, opt_state, params, internal_optimzier):

    if internal_optimzier.name == "adam":
        adam_state = update_adam_opt_state(opt_state_old[0], params)
        opt_state = (adam_state, opt_state[1])
    else:
        raise("Shared optimization for other internal optimizer than Adam for BFGS is not supported!")
    return opt_state


def update_opt_state(opt_state_old, params, opt_state, opt_config: OptimizationConfig):
    if opt_config.optimizer.name == "adam":
        opt_state = update_adam_opt_state(opt_state_old, params)
    elif opt_config.optimizer.name == "slbfgs":
        opt_state = update_bfgs_opt_state(opt_state_old, opt_state, params, opt_config.optimizer.internal_optimizer)
    elif opt_config.optimizer.name == "kfac":
        opt_state = update_kfac_opt_state(opt_state_old, opt_state, params, opt_config.optimizer.internal_optimizer)
    else:
        raise("Optimizer currently not supported to update opt_state for shared optimization!")
    return opt_state


def update_wf(index, index_next, mcmc_state, opt_state, opt_get_param, clipping_params, wfs, opt_config: OptimizationConfig):
    shared_modules = opt_config.shared_optimization.shared_modules
    # adjust parameters for all wfs - adjust only next optimized wf
    wfs[index].mcmc_state = mcmc_state
    wfs[index].opt_state = opt_state
    wfs[index].clipping_params = clipping_params

    params = opt_get_param(opt_state)
    params_next = opt_get_param(
        wfs[index_next].opt_state)  # tree_unflatten(wfs[index_next].opt_state[1], wfs[index_next].opt_state[0])
    for i, mod in enumerate(["embed", "jastrow", "bf_fac", "bf_shift"]):
        if mod in shared_modules:
            # if mod == 'orbital_bf_fac':
            #     params_next['bf_fac']['up_output'] = params['bf_fac']['up_output']
            #     params_next['bf_fac']['dn_output'] = params['bf_fac']['dn_output']
            # elif mod == 'general_bf_fac':
            #     params_next['bf_fac']['up'] = params['bf_fac']['up']
            #     params_next['bf_fac']['dn'] = params['bf_fac']['dn']
            #     params_next['bf_fac']['scale'] = params['bf_fac']['scale']
            # else:
                params_next[mod] = params[mod]
    wfs[index_next].opt_state = update_opt_state(wfs[index_next].opt_state, params_next, opt_state, opt_config)

    return wfs

def get_index(n_epoch, wfs, config: SharedOptimizationConfig):
    method = config.scheduling_method
    if method == "round_robin":
        return n_epoch % len(wfs)
    elif method == 'stddev':
        wf_ages = n_epoch - jnp.array([wf.age for wf in wfs])
        if jnp.any(n_epoch) < len(wfs):
            index = n_epoch % len(wfs)
        elif jnp.any(wf_ages > config.max_age):
            index = jnp.argmax(wf_ages)
        else:
            stddevs = [wf.current_metrics['E_std'] for wf in wfs]
            index = np.argmax(stddevs)
        return index
    else:
        raise ("Wavefunction scheduler currently not supported.")


def optimize_inter(config: Configuration, wfs, mcmc, log_psi_squared):
    wfs, opt_get_params, optimize_epoch = warm_up_opt(wfs, mcmc, config, log_psi_squared)
    n_wfs = len(wfs)
    index = get_index(0, wfs, config.optimization.shared_optimization)
    t_start = time.time()

    for n_epoch in range(config.optimization.n_epochs * n_wfs):

        # optimize wf[index] for one eppoch
        E_epoch, mcmc_state, opt_state, clipping_params = optimize_epoch(n_epoch, wfs[index].mcmc_state,
                                                                         wfs[index].opt_state,
                                                                         wfs[index].clipping_params,
                                                                         wfs[index].fixed_params)

        # update metrics + epoch counter
        wfs[index].current_metrics = {'E_mean': jnp.nanmean(E_epoch), 'E_std': jnp.nanstd(E_epoch)}
        wfs[index].n_opt_epochs += 1

        # get next indext for optimization
        index_next = get_index(n_epoch, wfs, config.optimization.shared_optimization)

        # update wf[index] with optimization results and wf[next_index] with new shared weights
        wfs = update_wf(index, index_next, mcmc_state, opt_state, opt_get_params, clipping_params, wfs,
                        config.optimization)

        # collect epoch time
        t_end = time.time()

        # check for checkpoints. if any, all wfs have the same checkpoints.
        if n_epoch in wfs[0].checkpoints:
            for wf in wfs:
                if wf.loggers is not None:
                    # update opt_state with current shared weights
                    wf_params = opt_get_params(wf.opt_state)
                    wf_params = update_shared_weights(wf_params, opt_get_params(opt_state),
                                                      config.optimization.shared_optimization.shared_modules)
                    wf.opt_state = update_opt_state(wf.opt_state, wf_params)

                    # create full data dict
                    full_data = dict(trainable=wf_params, fixed=wf.fixed_params, mcmc=wf.mcmc_state,
                                     opt=make_opt_state_picklable(wf.opt_state))

                    # log weights
                    wf.loggers.log_weights(full_data)

                    # log checkpoint
                    logging.info(f"Logging checkpoint to folder {wf.checkpoints[n_epoch]}")
                    wf.loggers.log_checkpoint(wf.checkpoints[n_epoch])

        # log metrics
        if wfs[index].loggers is not None:
            metrics = calculate_metrics(n_epoch, n_wfs, E_epoch, mcmc_state, t_end - t_start, "opt")
            wfs[index].loggers.log_metrics(*metrics)

        # reset clock and update index
        t_start = time.time()
        index = index_next

    # update shared weights from last opt_state for all wfs
    for wf in wfs:
        wf_params = opt_get_params(wf.opt_state)
        wf_params = update_shared_weights(wf_params, opt_get_params(opt_state), config.optimization.shared_optimization.shared_modules)
        wf.opt_state = update_opt_state(wf.opt_state, wf_params, opt_state, config.optimization)

    return wfs, opt_get_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimization of wavefunction for multiple molecular configurations with interdependent optimization.")
    parser.add_argument("config_file", default="config.yml", help="Path to input config file", nargs="?")
    args = parser.parse_args()
    config = Configuration.load(args.config_file)
    config.save("full_config.yml")
    loggers_set = LoggerCollection(config.logging, config.experiment_name)
    loggers_set.on_run_begin()
    loggers_set.log_params(config.get_as_flattened_dict())
    loggers_set.log_tags(config.logging.tags)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if config.computation.use_gpu else "-1"

    jax_config.update("jax_enable_x64", config.computation.float_precision == "float64")
    jax_config.update("jax_disable_jit", config.computation.disable_jit)
    logging.debug(f"Used hardware: {xla_bridge.get_backend().platform}")

    # Initialization of model and MCMC states TODO restart and reload
    loggers_set.log_param("code_version", getCodeVersion())
    log_psi_squared, mcmc, wfs = init_wf(config)

    # Wavefunction optimization
    if config.optimization.n_epochs > 0:
        logging.info("Starting optimization...")
        wfs, opt_get_params = optimize_inter(config, wfs, mcmc, log_psi_squared)

    for wf in wfs:
        wf.loggers.log_weights(dict(trainable=opt_get_params(wf.opt_state), fixed=wf.fixed_params, mcmc=wf.mcmc_state,
                                    opt=make_opt_state_picklable(wf.opt_state)))

    # Wavefunction evaluation
    if config.evaluation.n_epochs > 0:
        logging.info("Starting evaluation...")
        error_set = []
        sigma_set = []
        error_plus_2_stdev_set = []
        for i, wf in enumerate(wfs):
            logging.info(f"Starting evaluation for wavefunction {i}...")
            mcmc_state = resize_nr_of_walkers(wf.mcmc_state, config.mcmc.n_walkers_eval)
            trainable_params = opt_get_params(wf.opt_state)
            E_eval, forces_eval, mcmc_state = evaluate_wavefunction(log_psi_squared,
                                                                    trainable_params,
                                                                    wf.fixed_params,
                                                                    mcmc,
                                                                    mcmc_state,
                                                                    config.evaluation,
                                                                    wf.loggers)
            # Postprocessing
            E_mean = jnp.nanmean(E_eval)
            E_mean_sigma = jnp.nanstd(E_eval) / jnp.sqrt(len(E_eval))

            wf.loggers.log_metrics(dict(E_mean=E_mean, E_mean_sigma=E_mean_sigma), metric_type="eval")
            if wf.physical.E_ref is not None:
                error_eval, sigma_eval = 1e3 * (E_mean - wf.physical.E_ref), 1e3 * E_mean_sigma
                wf.loggers.log_metrics(dict(error_eval=error_eval, sigma_error_eval=sigma_eval,
                                               error_plus_2_stdev=error_eval + 2 * sigma_eval))
                error_set.append(error_eval)
                sigma_set.append(sigma_eval)
                error_plus_2_stdev_set.append(error_eval + 2 * sigma_eval)

            if forces_eval is not None:
                forces_mean = jnp.nanmean(forces_eval, axis=0)
                wf.loggers.log_metric('forces_mean', forces_mean)

        loggers_set.log_metrics(
            dict(error_eval=jnp.nanmean(jnp.array(error_set)), sigma_error_eval=jnp.nanmean(jnp.array(sigma_set)),
                 error_plus_2_stdev=jnp.nanmean(jnp.array(error_plus_2_stdev_set))), metric_type="eval")
    for wf in wfs:
        wf.loggers.on_run_end()
    loggers_set.on_run_end()
