"""
Helper functions.
"""

import os
import subprocess

import numpy as np
import scipy.optimize
from jax import numpy as jnp
from jax.experimental import optimizers as jaxopt

from deeperwin.configuration import OptimizationConfig, ClippingConfig


def getCodeVersion():
    """
    Determine the current git commit by calling 'git log -1'.
    Returns:
         (str): Git commit hash and latest commit message
    """
    try:
        path = os.path.dirname(__file__)
        msg = subprocess.check_output(['git', 'log', '-1'], cwd=path, encoding='utf-8')
        return msg.replace('\n', '; ')
    except Exception as e:
        print(e)
        return None


def get_el_ion_distance_matrix(r_el, R_ion):
    """
    Args:
        r_el: shape [N_batch x n_el x 3]
        R_ion: shape [N_ion x 3]
    Returns:
        diff: shape [N_batch x n_el x N_ion x 3]
        dist: shape [N_batch x n_el x N_ion]
    """
    diff = jnp.expand_dims(r_el, -2) - R_ion
    dist = jnp.linalg.norm(diff, axis=-1)
    return diff, dist


def get_full_distance_matrix(r_el):
    """
    Args:
        r_el: shape [n_el x 3]
    Returns:
    """
    diff = jnp.expand_dims(r_el, -2) - jnp.expand_dims(r_el, -3)
    dist = jnp.linalg.norm(diff, axis=-1)
    return dist


def get_distance_matrix(r_el):
    """
    Compute distance matrix omitting the main diagonal (i.e. distance to the particle itself)

    Args:
        r_el: [batch_dims x n_electrons x 3]

    Returns:
        tuple: differences [batch_dims x n_el x (n_el-1) x 3], distances [batch_dims x n_el x (n_el-1)]
    """
    n_el = r_el.shape[-2]
    indices = np.array([[j for j in range(n_el) if j != i] for i in range(n_el)], dtype=int)
    diff = r_el[..., indices, :] - jnp.expand_dims(r_el, -2)
    dist = jnp.linalg.norm(diff, axis=-1)
    return diff, dist


def make_opt_state_picklable(opt_state):
    if isinstance(opt_state, jaxopt.OptimizerState):
        return jaxopt.unpack_optimizer_state(opt_state)
    elif isinstance(opt_state, tuple):
        return tuple(make_opt_state_picklable(x) for x in opt_state)
    elif isinstance(opt_state, list):
        return list(make_opt_state_picklable(x) for x in opt_state)
    elif isinstance(opt_state, dict):
        return {k: make_opt_state_picklable(v) for k, v in opt_state.items()}
    else:
        return opt_state


# Optimization

def build_inverse_schedule(lr_init, lr_decay_time):
    return lambda t: lr_init / (1 + t / lr_decay_time)


def get_builtin_optimizer(optimizer_config: OptimizationConfig, schedule_config, base_lr):
    schedule = build_learning_rate_schedule(base_lr, schedule_config)
    name = optimizer_config.name
    if name == 'adam':
        return jaxopt.adam(schedule, b1=optimizer_config.b1, b2=optimizer_config.b2, eps=optimizer_config.eps)
    else:
        standard_optimizers = dict(rmsprop_momentum=jaxopt.rmsprop_momentum, sgd=jaxopt.sgd)
        return standard_optimizers[optimizer_config.name](schedule)


def build_learning_rate_schedule(base_lr, schedule_config):
    if schedule_config.name == "inverse":
        return build_inverse_schedule(base_lr, schedule_config.decay_time)
    elif schedule_config.name == "fixed":
        return base_lr
    else:
        raise ValueError(f"Unsupported config-value for optimization.schedule.name: {schedule_config.name}")


def calculate_clipping_state(E, config: ClippingConfig):
    if config.center == "mean":
        center = jnp.nanmean(E)
    elif config.center == "median":
        center = jnp.nanmedian(E)
    else:
        raise ValueError(f"Unsupported config-value for optimization.clipping.center: {config.center}")

    if config.width_metric == "std":
        width = jnp.nanstd(E) * config.clip_by
    elif config.width_metric == "mae":
        width = jnp.nanmean(jnp.abs(E - center)) * config.clip_by
    else:
        raise ValueError(f"Unsupported config-value for optimization.clipping.width_metric: {config.width_metric}")
    return center, width


def morse_potential(r, E, a, Rb, E0):
    """
    Returns the Morse potential :math:`E_0 + E (exp{-2x} - 2exp{-x})` where :math:`x = a (r-R_b)`
    """
    x = a * (r - Rb)
    return E * (np.exp(-2 * x) - 2 * np.exp(-x)) + E0


def fit_morse_potential(d, E, p0=None):
    """
    Fits a morse potential to given energy data.
    Args:
        d (list, np.array): array-like list of bondlengths
        E (list, np.array): array-like list of energies for each bond-length
        p0 (tuple): Initial guess for parameters of morse potential. If set to None, parameters will be guessed based on data.
    Returns:
        (tuple): Fit parameters for Morse potential. Can be evaluated using morsePotential(r, *params)
    """
    if p0 is None:
        p0 = (0.1, 1.0, np.mean(d), -np.min(E) + 0.1)
    morse_params = scipy.optimize.curve_fit(morse_potential, d, E, p0=p0)[0]
    return tuple(morse_params)
