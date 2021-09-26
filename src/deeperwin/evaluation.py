"""
Logic for wavefunction evaluation.
"""

import logging
import time

import jax
import numpy as np
from jax import numpy as jnp

from deeperwin.configuration import EvaluationConfig
from deeperwin.hamiltonian import get_local_energy, calculate_forces
from deeperwin.loggers import DataLogger
from deeperwin.mcmc import MCMCState, MetropolisHastingsMonteCarlo, calculate_metrics


@jax.partial(jax.jit, static_argnums=(0, 1))
def _evaluation_step(log_psi_squared, mcmc, mcmc_state, params):
    mcmc_state = mcmc.run_inter_steps(log_psi_squared, params, mcmc_state, "eval")
    E_loc = get_local_energy(log_psi_squared, *mcmc_state.model_args, *params)
    return mcmc_state, E_loc


def _build_force_polynomial_coefficients(R_core, polynomial_degree):
    j = np.arange(1, polynomial_degree + 1)
    A = R_core ** 2 / (2 + j[np.newaxis, :] + j[:, np.newaxis] + 1)
    b = 1 / (j + 1)
    coeff = np.linalg.solve(A, b)
    coeff = np.reshape(coeff, [-1, 1, 1, 1, 1])
    return coeff


def build_evaluation_step(log_psi_sqr_func, mcmc, eval_config):
    def _evaluation_step(mcmc_state: MCMCState, params):
        mcmc_state = mcmc.run_inter_steps(log_psi_sqr_func, params, mcmc_state)
        E_loc = get_local_energy(log_psi_sqr_func, *mcmc_state.model_args, *params)
        if eval_config.forces is not None:
            poly_coeffs = _build_force_polynomial_coefficients(eval_config.forces.R_core,
                                                               eval_config.forces.polynomial_degree)
            forces = calculate_forces(*mcmc_state.model_args, mcmc_state.log_psi_sqr, log_psi_sqr_func, params,
                                      eval_config.forces, poly_coeffs)
        else:
            forces = None
        return mcmc_state, E_loc, forces

    return jax.jit(_evaluation_step)


def evaluate_wavefunction(
        log_psi_squared,
        trainable_params,
        fixed_params,
        mcmc: MetropolisHastingsMonteCarlo,
        mcmc_state: MCMCState,
        config: EvaluationConfig,
        logger: DataLogger = None
):
    params = (trainable_params, fixed_params)
    logging.debug("Starting burn-in for evaluation...")
    mcmc_state = mcmc.run_burn_in_eval(log_psi_squared, params, mcmc_state)

    _evaluation_step = build_evaluation_step(log_psi_squared, mcmc, config)

    t_start = time.time()
    E_eval_mean = []
    forces_mean = []
    for n_epoch in range(config.n_epochs):
        mcmc_state, E_epoch, forces = _evaluation_step(mcmc_state, (trainable_params, fixed_params))
        t_end = time.time()
        E_eval_mean.append(jnp.mean(E_epoch))
        if logger is not None:
            logger.log_metrics(*calculate_metrics(n_epoch, 1, E_epoch, mcmc_state, (t_end - t_start), "eval"))
            if forces is not None:
                logger.log_metric("forces", forces, n_epoch, "eval")
                forces_mean.append(forces)
        t_start = t_end
    forces_mean = jnp.array(forces_mean) if (len(forces_mean) > 0) else None
    return jnp.array(E_eval_mean), forces_mean, mcmc_state
