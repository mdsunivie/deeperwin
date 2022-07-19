"""
Helper functions.
"""
import functools
import logging
import os
import subprocess
import re
import jax
import numpy as np
import scipy.optimize
from jax import numpy as jnp
from deeperwin.configuration import ClippingConfig
import haiku as hk

LOGGER = logging.getLogger("dpe")

###############################################################################
############################# Parallelization  ################################
###############################################################################

pmap = functools.partial(jax.pmap, axis_name="devices")
pmean = functools.partial(jax.lax.pmean, axis_name="devices")
psum = functools.partial(jax.lax.psum, axis_name="devices")

def replicate_across_devices(data, n_devices):
    return jax.tree_map(lambda x: jnp.tile(x, [n_devices] + [1] * x.ndim), data)

def merge_from_devices(data):
    return jax.tree_map(lambda x: x[0], data)


batch_rng_split = jax.vmap(lambda k: jax.random.split(k,2), in_axes=0, out_axes=1)

###############################################################################
##################################### Logging  ################################
###############################################################################

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

def setup_job_dir(parent_dir, name):
    job_dir = os.path.join(parent_dir, name)
    if os.path.exists(job_dir):
        logging.warning(f"Directory {job_dir} already exists. Results might be overwritten.")
    else:
        os.makedirs(job_dir)
    return job_dir

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
    diff = r_el[..., None, :] - R_ion[..., None, :, :]
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
    diff = r_el[..., :, None, :] - r_el[..., None, :, :]
    diff_padded = diff + jnp.eye(n_el)[..., None]
    dist = jnp.linalg.norm(diff_padded, axis=-1) * (1 - jnp.eye(n_el))
    return diff, dist

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

def get_params_filter_func(module_names):
    regex = re.compile("(" + "|".join(module_names) + ")")
    def filter_func(module_name, name, v):
        return len(regex.findall(f"{module_name}/{name}")) > 0
    return filter_func

def split_params(params, module_names):
    return hk.data_structures.partition(get_params_filter_func(module_names), params)

def merge_params(params_old, params_new):
    return hk.data_structures.merge(params_old, params_new)

def get_number_of_params(params):
    return hk.data_structures.tree_size(params)
