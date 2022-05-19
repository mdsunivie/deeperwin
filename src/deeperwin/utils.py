"""
Helper functions.
"""
import bz2
import copy
import logging
import os
import pickle
import subprocess
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import scipy.optimize
from jax import numpy as jnp
from jax.experimental import optimizers as jaxopt
from jax.experimental.optimizers import OptimizerState, optimizer, make_schedule
from jax.tree_util import tree_unflatten, tree_flatten
from jax._src.util import unzip2
from jax.flatten_util import ravel_pytree

from deeperwin.configuration import  ClippingConfig, Configuration
from collections import OrderedDict

LOGGER = logging.getLogger("dpe")

###############################################################################
##################################### Logging  ################################
###############################################################################

def calculate_metrics(epoch_nr: int, E_epoch: jnp.array, E_epoch_unclipped: jnp.array, E_epoch_unclipped_history: List[float],
                      mcmc_state: 'MCMCState', mcmc_state_old: 'MCMCState', time_per_epoch: float, metric_type: str, epoch_per_geometry: int = None, E_ref=None, smoothing=0.05):
    metrics = {}

    n_averaging = int(smoothing * len(E_epoch_unclipped_history))
    E_mean_smooth = E_epoch_unclipped_history[-n_averaging:]
    if len(E_mean_smooth) > 1:
        E_smooth = np.mean(E_mean_smooth)
        metrics[metric_type + "_E_mean_smooth"] = E_smooth
        if E_ref:
            metrics[metric_type + "_error_smooth"] = float(E_smooth - E_ref) * 1e3

    metrics[metric_type + "_E_mean"] = float(jnp.mean(E_epoch))
    if E_ref:
        metrics[metric_type + "_error_E_mean"] = (float(jnp.mean(E_epoch)) - E_ref) * 1e3
    metrics[metric_type + "_E_std"] = float(jnp.std(E_epoch))
    if metric_type == "opt":
        metrics[metric_type + "_E_mean_unclipped"] = float(jnp.mean(E_epoch_unclipped))
        if E_ref:
            metrics[metric_type + "_error_E_mean"] = (float(jnp.mean(E_epoch_unclipped)) - E_ref) * 1e3
        metrics[metric_type + "_E_std_unclipped"] = float(jnp.std(E_epoch_unclipped))
    metrics[metric_type + "_mcmc_stepsize"] = float(mcmc_state.stepsize)
    metrics[metric_type + "_mcmc_acc_rate"] = float(mcmc_state.acc_rate)

    if mcmc_state_old:
        delta_r = np.linalg.norm(mcmc_state.r - mcmc_state_old.r, axis=-1)
        metrics[metric_type + "_mcmc_delta_r_mean"] = float(np.mean(delta_r))
        metrics[metric_type + "_mcmc_delta_r_median"] = float(np.median(delta_r))

    metrics[metric_type + "_mcmc_max_age"] = float(jnp.max(mcmc_state.walker_age))
    metrics[metric_type + "_t_epoch"] = time_per_epoch
    if epoch_per_geometry is None:
        epoch_per_geometry = epoch_nr
    metrics[metric_type + "_epoch_per_geom"] = epoch_per_geometry

    return metrics, int(epoch_nr), metric_type

def getCodeVersion():
    """
    Determine the current git commit by calling 'git log -1'.
    Returns:
         (str): Git commit hash and latest commit message
    """
    try:
        path = os.path.dirname(__file__)
        msg = subprocess.check_output(["git", "log", "-1"], cwd=path, encoding="utf-8")
        return msg.replace("\n", "; ")
    except Exception as e:
        print(e)
        return None


def prepare_data_for_logging(trainable_params, fixed_params, mcmc_state, opt_state=None, clipping_state=None):
    # create full data dict
    full_data = dict(trainable=trainable_params, fixed=fixed_params, mcmc=mcmc_state, clipping=clipping_state)
    if opt_state is not None:
        full_data["opt"] = make_opt_state_picklable(opt_state)
    return full_data


###############################################################################
############################# Model / physics / hamiltonian ###################
###############################################################################


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
        tuple: differences [batch_dims x n_el x n_el x 3], distances [batch_dims x n_el x n_el]
    """
    n_el = r_el.shape[-2]
    diff = r_el[..., None, :, :] - r_el[..., :, None, :]
    diff_padded = diff + jnp.eye(n_el)[..., None]
    dist = jnp.linalg.norm(diff_padded, axis=-1) * (1 - jnp.eye(n_el))
    return diff, dist


###############################################################################
##################################### Optimization ############################
###############################################################################


class PickleableOptState:
    def __init__(self, unpacked_opt_state):
        self.unpacked_opt_state = unpacked_opt_state


def make_opt_state_picklable(opt_state):
    if isinstance(opt_state, jaxopt.OptimizerState):
        return PickleableOptState(jaxopt.unpack_optimizer_state(opt_state))
    elif isinstance(opt_state, tuple):
        return tuple(make_opt_state_picklable(x) for x in opt_state)
    elif isinstance(opt_state, list):
        return list(make_opt_state_picklable(x) for x in opt_state)
    elif isinstance(opt_state, OrderedDict):
        x = OrderedDict()
        for k, v in opt_state.items():
            x[k] = make_opt_state_picklable(v)
        return x
    elif isinstance(opt_state, dict):
        return {k: make_opt_state_picklable(v) for k, v in opt_state.items()}
    else:
        return opt_state


def unpickle_opt_state(pickled_opt_state):
    if isinstance(pickled_opt_state, PickleableOptState):
        return jaxopt.pack_optimizer_state(pickled_opt_state.unpacked_opt_state)
    elif isinstance(pickled_opt_state, tuple):
        return tuple(unpickle_opt_state(x) for x in pickled_opt_state)
    elif isinstance(pickled_opt_state, list):
        return list(unpickle_opt_state(x) for x in pickled_opt_state)
    elif isinstance(pickled_opt_state, OrderedDict):
        x = OrderedDict()
        for k, v in pickled_opt_state.items():
            x[k] = unpickle_opt_state(v)
        return x
    elif isinstance(pickled_opt_state, dict):
        return {k: unpickle_opt_state(v) for k, v in pickled_opt_state.items()}
    else:
        return pickled_opt_state


def build_learning_rate_schedule(base_lr, schedule_config):
    if schedule_config.name == "inverse":
        return build_inverse_schedule(base_lr, schedule_config.decay_time)
    elif schedule_config.name == "fixed":
        return base_lr
    else:
        raise ValueError(f"Unsupported config-value for optimization.schedule.name: {schedule_config.name}")


def build_inverse_schedule(lr_init, lr_decay_time):
    return lambda t: lr_init / (1 + t / lr_decay_time)


def _update_adam_opt_state(opt_state, params):
    params, tree2 = tree_flatten(params)

    def do_nothing(state, params):
        x, m, v = state
        return params, m, v

    states_flat, tree, subtrees = opt_state
    states = map(tree_unflatten, subtrees, states_flat)
    new_states = map(do_nothing, states, params)
    new_states_flat, subtrees2 = unzip2(map(tree_flatten, new_states))
    return OptimizerState(new_states_flat, tree, subtrees)

@optimizer
def adam_inverse(step_size, b1=0.9, b2=0.999, eps=1e-8):
  """Construct optimizer triple for Adam.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to a positive scalar.
    b1: optional, a positive scalar value for beta_1, the exponential decay rate
      for the first moment estimates (default 0.9).
    b2: optional, a positive scalar value for beta_2, the exponential decay rate
      for the second moment estimates (default 0.999).
    eps: optional, a positive scalar value for epsilon, a small constant for
      numerical stability (default 1e-8).

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    m0 = jnp.zeros_like(x0)
    v0 = jnp.zeros_like(x0)
    return x0, m0, v0
  def update(i, g, state):
    x, m, v = state
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    v = (1 - b2) * jnp.square((g - m)) + b2 * v  # Centered second moment estimate.
    mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
    vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
    m_clipped = jnp.clip(mhat, a_min=mhat - 3*vhat, a_max=mhat + 3*vhat)
    x = x - step_size(i) * m_clipped
    return x, m, v
  def get_params(state):
    x, _, _ = state
    return x
  return init, update, get_params




def _update_adam_scaled_opt_state(opt_state, params):
    params, tree2 = tree_flatten(params)

    def do_nothing(state, params):
        x, m, v, scale = state
        return params, m, v, scale

    states_flat, tree, subtrees = opt_state
    states = map(tree_unflatten, subtrees, states_flat)
    new_states = map(do_nothing, states, params)
    new_states_flat, subtrees2 = unzip2(map(tree_flatten, new_states))
    return OptimizerState(new_states_flat, tree, subtrees)


@optimizer
def sgd_scaled(step_size):
    """Construct optimizer triple for stochastic gradient descent with differently scaled gradients for reuse option.

    Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to a positive scalar.

    Returns:
    An (init_fun, update_fun, get_params) triple.
    """
    step_size = make_schedule(step_size)
    def init(x0):
        scale = jnp.ones_like(x0)
        return x0, scale

    def update(i, g, state):
        x, scale = state
        x = x - step_size(i) * scale * g
        return x, scale

    def get_params(state):
        x, _ = state
        return x

    return init, update, get_params


@optimizer
def adam_scaled(step_size, b1=0.9, b2=0.999, eps=1e-8):
    """Construct optimizer triple for Adam.

    Args:
      step_size: positive scalar, or a callable representing a step size schedule
        that maps the iteration index to positive scalar.
      b1: optional, a positive scalar value for beta_1, the exponential decay rate
        for the first moment estimates (default 0.9).
      b2: optional, a positive scalar value for beta_2, the exponential decay rate
        for the second moment estimates (default 0.999).
      eps: optional, a positive scalar value for epsilon, a small constant for
        numerical stability (default 1e-8).

    Returns:
      An (init_fun, update_fun, get_params) triple.
    """
    step_size = make_schedule(step_size)

    def init(x0):
        m0 = jnp.zeros_like(x0)
        v0 = jnp.zeros_like(x0)
        scale = jnp.ones_like(x0)
        return x0, m0, v0, scale

    def update(i, g, state):
        x, m, v, scale = state
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * jnp.square(g) + b2 * v  # Second moment estimate.
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        x = x - scale * step_size(i) * mhat / (jnp.sqrt(vhat) + eps)
        return x, m, v, scale

    def get_params(state):
        x, _, _, _ = state
        return x

    return init, update, get_params


def set_scaled_optimizer_lr(opt_config, opt_state, get_params):
    params_ravel, unravel_func = ravel_pytree(get_params(opt_state))
    scale = jnp.ones(params_ravel.shape)
    scale = unravel_func(scale)

    for mod in scale:
        if mod in opt_config.scaled_modules:
            scale_of_mod, unravel_func = ravel_pytree(scale[mod])
            scale_of_mod = scale_of_mod * opt_config.scale_lr
            scale[mod] = unravel_func(scale_of_mod)

    def _update_adam_scaled_opt_state(opt_state, scale):
        scale, tree2 = tree_flatten(scale)

        def _do_nothing(state, scale):
            x, m, v, _ = state
            return x, m, v, scale

        states_flat, tree, subtrees = opt_state
        states = map(tree_unflatten, subtrees, states_flat)
        new_states = map(_do_nothing, states, scale)
        new_states_flat, subtrees2 = unzip2(map(tree_flatten, new_states))
        return OptimizerState(new_states_flat, tree, subtrees)

    def _update_sgd_scaled_opt_state(opt_state, scale):
        scale, tree2 = tree_flatten(scale)
        def _do_nothing(state, scale):
            x, _ = state
            return x, scale

        states_flat, tree, subtrees = opt_state
        states = map(tree_unflatten, subtrees, states_flat)
        new_states = map(_do_nothing, states, scale)
        new_states_flat, subtrees2 = unzip2(map(tree_flatten, new_states))
        return OptimizerState(new_states_flat, tree, subtrees)

    if opt_config.name == "adam_scaled":
        return _update_adam_scaled_opt_state(opt_state, scale)
    elif opt_config.name == "sgd_scaled":
        return _update_sgd_scaled_opt_state(opt_state, scale)

def get_builtin_optimizer(optimizer_config, schedule_config, base_lr):
    schedule = build_learning_rate_schedule(base_lr, schedule_config)
    name = optimizer_config.name
    if name == "adam":
        return jaxopt.adam(schedule, b1=optimizer_config.b1, b2=optimizer_config.b2, eps=optimizer_config.eps)
    elif name == "adam_scaled":
        init_fn, update_fn, get_params_fn = adam_scaled(schedule, b1=optimizer_config.b1, b2=optimizer_config.b2, eps=optimizer_config.eps)

        def init_and_set_scales_fn(initial_params):
            state = init_fn(initial_params)
            return set_scaled_optimizer_lr(optimizer_config, state, get_params_fn)

        return init_and_set_scales_fn, update_fn, get_params_fn
    elif name == "adam_inverse":
        init_fn, update_fn, get_params_fn = adam_inverse(schedule, b1=optimizer_config.b1, b2=optimizer_config.b2, eps=optimizer_config.eps)
        return init_fn, update_fn, get_params_fn
    elif name == "sgd_scaled":
        init_fn, update_fn, get_params_fn = sgd_scaled(schedule)

        def init_and_set_scales_fn(initial_params):
            state = init_fn(initial_params)
            return set_scaled_optimizer_lr(optimizer_config, state, get_params_fn)

        return init_and_set_scales_fn, update_fn, get_params_fn
    else:
        standard_optimizers = dict(rmsprop_momentum=jaxopt.rmsprop_momentum, sgd=jaxopt.sgd)
        return standard_optimizers[optimizer_config.name](schedule)


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


###############################################################################
########################### Post-processing / analysis ########################
###############################################################################
ANGSTROM_IN_BOHR = 1.88973


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


def save_xyz_file(fname, R, Z, comment=""):
    """Units for R are expected to be Bohr and will be translated to Angstrom in output"""
    assert len(R) == len(Z)
    PERIODIC_TABLE = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
         'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr']

    ANGSTROM_IN_BOHR = 1.88973
    with open(fname, "w") as f:
        f.write(str(len(R)) + "\n")
        f.write(comment + "\n")
        for Z_, R_ in zip(Z, R):
            f.write(
                f"{PERIODIC_TABLE[Z_-1]:3>} {R_[0]/ANGSTROM_IN_BOHR:-.10f} {R_[1]/ANGSTROM_IN_BOHR:-.10f} {R_[2]/ANGSTROM_IN_BOHR:-.10f}\n"
            )


def get_extent_for_imshow(x, y):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return x[0] - 0.5 * dx, x[-1] + 0.5 * dx, y[0] - 0.5 * dy, y[-1] + 0.5 * dy


def get_ion_pos_string(R):
    s = ",".join(["[" + ",".join([f"{x:.6f}" for x in R_]) + "]" for R_ in R])
    return "[" + s + "]"


###############################################################################
##################### Shared optimization / split model #######################
###############################################################################


def remove_module_params_nested(params: dict, module_key: str):
    keys = module_key.split(".")
    model_params = params
    for (i, key) in enumerate(keys):
        if not key in model_params:
            LOGGER.warning(f"Could not find module {module_key}")
            return params
        if i < len(keys) - 1:
            model_params = model_params[key]
    model_params.pop(keys[-1])
    return params


def get_module_params_nested(params: dict, module_key: str):
    keys = module_key.split(".")
    model_params = params
    for key in keys:
        if key not in model_params:
            LOGGER.warning(f"Could not find module {module_key}")
            return None
        model_params = model_params[key]
    return model_params


def set_module_params_nested(params: dict, update_params, module_key: str, make_copy=False):
    if update_params is None:
        return params
    keys = module_key.split(".")
    model_params = params
    for key in keys[:-1]:
        if not key in model_params:
            model_params[key] = {}
        model_params = model_params[key]
    model_params[keys[-1]] = update_params if not make_copy else copy.deepcopy(update_params)
    return params


def merge_trainable_params(shared_params: dict, unique_params: dict):
    merged_params = dict(unique_params)
    for key in shared_params:
        merged_params = set_module_params_nested(merged_params, shared_params[key], key)
    return merged_params


def split_trainable_params(full_trainable_params: dict, shared_modules: List[str]):
    unique_params = dict(full_trainable_params)
    shared_params = {}
    for module in shared_modules:
        shared_params[module] = get_module_params_nested(unique_params, module)
        remove_module_params_nested(unique_params, module)
    return shared_params, unique_params


###############################################################################
##################### Various #######################
###############################################################################


def convert_to_jnp(inp):
    if inp is None:
        return inp
    if isinstance(inp, dict):
        for key in inp:
            inp[key] = convert_to_jnp(inp[key])
        return inp
    if isinstance(inp, list):
        return [convert_to_jnp(x) for x in inp]
    if isinstance(inp, tuple):
        return tuple([convert_to_jnp(x) for x in inp])
    if isinstance(inp, np.ndarray):
        return jnp.array(inp)
    if isinstance(inp, str):
        return inp
    return jnp.array(inp)


def get_number_of_params(nested_params):
    """
    Returns the number of parameters in a nested structure.

    Args:
        nested_params: Possibly nested structure that holds different model parameters.

    """
    if nested_params is None:
        return 0
    if hasattr(nested_params, "shape"):
        return int(np.prod(nested_params.shape))
    elif isinstance(nested_params, dict):
        return sum([get_number_of_params(p) for p in nested_params.values()])
    else:
        return sum([get_number_of_params(p) for p in nested_params])


def load_from_file(fname):
    with bz2.open(fname, 'rb') as f:
        return pickle.load(f)


def save_to_file(fname, **data):
    with bz2.open(fname, 'wb') as f:
        pickle.dump(data, f)


def load_run(path, load_config=True, ignore_extra_settings=False):
    results = load_from_file(os.path.join(path, "results.bz2"))
    if load_config:
        for k in ['code_version', 'n_params', 'tags']:
            results['config'].pop(k, None)
        parsed_config = Configuration.from_flattened_dict(results['config'], ignore_extra_settings)
        return results, parsed_config
    else:
        return results


def load_all_runs(root_dir, max_history_length=50):
    full_data = []
    for fname in Path(root_dir).rglob("results.bz2"):
        raw_data = load_from_file(fname)
        data = dict(raw_data['config'])
        for k, v in raw_data['metrics'].items():
            if isinstance(v, list) and (max_history_length is not None) and (len(v) > max_history_length):
                continue
            data[k] = v
        data['dirname'] = fname.parent.name
        full_data.append(data)
    return pd.DataFrame(full_data)