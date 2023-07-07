#%%
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
import haiku as hk
import e3nn_jax as e3nn
from typing import Tuple, List

LOGGER = logging.getLogger("dpe")

###############################################################################
############################# Parallelization  ################################
###############################################################################

pmap = functools.partial(jax.pmap, axis_name="devices")
pmean = functools.partial(jax.lax.pmean, axis_name="devices")
psum = functools.partial(jax.lax.psum, axis_name="devices")

def multi_vmap(func, n):
    for n in range(n):
        func = jax.vmap(func)
    return func

def replicate_across_devices(data):
    # Step 1: Tile data across local devices
    data = jax.tree_util.tree_map(lambda x: jnp.tile(x, [jax.local_device_count()] + [1] * jnp.array(x).ndim), data)

    # Step 2: Replace data on each device by data from device 0 on process 0
    def _select_master_data(x):
        x = jax.lax.cond(jax.lax.axis_index("devices") == 0,
                         lambda y: y,
                         lambda y: jax.tree_util.tree_map(jnp.zeros_like, y),
                         x)
        return jax.lax.psum(x, axis_name='devices')
    data = jax.pmap(_select_master_data, axis_name='devices')(data)
    return data

def get_from_devices(data):
    return jax.tree_util.tree_map(lambda x: x[0], data)

def is_equal_across_devices(data):
    full_data = jax.tree_util.tree_map(_pad_data, data)
    full_data = jax.tree_leaves(full_data)
    n_devices = jax.device_count()
    for x in full_data:
        for i in range(1, n_devices):
            if not np.allclose(x[i], x[0]):
                return False
    return True

@functools.partial(jax.pmap, axis_name='i')
def _pad_data(x):
    full_data = jnp.zeros((jax.device_count(), *x.shape), x.dtype)
    full_data = full_data.at[jax.lax.axis_index('i')].set(x)
    return jax.lax.psum(full_data, axis_name='i')

def merge_from_devices(x):
    if x is None:
        return None
    full_data = _pad_data(x)
    # Data is now identical on all local devices; pick the 0th device and flatten
    return full_data[0].reshape((-1, *full_data.shape[3:]))

batch_rng_split = jax.vmap(lambda k: jax.random.split(k,2), in_axes=0, out_axes=1)

def tree_dot(x, y):
    return jax.tree_util.tree_reduce(jnp.add, jax.tree_util.tree_map(lambda x_, y_: jnp.sum(x_ * y_), x, y))

def tree_mul(x, y):
    return jax.tree_util.tree_map(lambda x_, y_: x_ * y_, x, y)

def tree_add(x, y):
    return jax.tree_util.tree_map(lambda x_, y_: x_ + y_, x, y)

def tree_norm(x_as_tree):
  norm_sqr = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: jnp.sum(x**2), x_as_tree)))
  return jnp.sqrt(norm_sqr)


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

def setup_job_dir(parent_dir, name) -> str:
    job_dir = os.path.join(parent_dir, name)
    if os.path.exists(job_dir):
        logging.warning(f"Directory {job_dir} already exists. Results might be overwritten.")
    else:
        os.makedirs(job_dir)
    return job_dir

def get_param_size_summary(params):
    s = "Param breakdown:\n"
    padding = max([len(k) for k in params.keys()])
    format_str = f"{{:<{padding}}} : {{:11,d}}\n"

    for k,v in params.items():
        s += format_str.format(k, hk.data_structures.tree_size(v))
    s += "-" * (padding + 13) + "\n"
    s += format_str.format("Total", hk.data_structures.tree_size(params))
    return s


def prettyprint_param_shapes(params):
    def categorize_param(name):
        if "e3_one_electron_layer" in name:
            return "Embedding: One Electron Stream", 211
        elif ("mlp_el_ion" in name) or ("mlp_same" in name) or ("mlp_diff" in name):
            return "Embedding: 2 particle_stream", 212
        elif "h1_map" in name:
            return "Embedding: H1 Mapping", 213
        elif ("interaction_el_ion" in name) or ("interaction_same" in name) or ("interaction_diff" in name):
            return "Embedding: W Mapping", 214
        elif "ion_embed" in name:
            return "Embedding: Ion", 215
        elif "readout" in name:
            return "Embedding: Readout", 216
        elif name.startswith("wf/orbitals"):
            return "Orbitals", 301
        elif "transformer_embedding/upmapping" in name:
            return "Embedding: Upmapping", 201
        elif "self_attention" in name and "attention_linear_map" in name:
            return "Embedding: Self attention q/k/v", 202
        elif "self_attention" in name and "output." in name:
            return "Embedding: Self attention output", 203
        elif "self_attention" in name and "layer_norm." in name:
            return "Embedding: Self attention layer-norm", 204
        elif "transformer_embedding/mlp" in name:
            return "Embedding: Self attention", 210
        elif "el_el_cusp" in name:
            return "Cusps: Electron-Electron", 500
        else:
            return "Other", 1000

    param_blocks = {}
    sorting_priority = {}
    for module, name, param in hk.data_structures.traverse(params):
        full_name = module + "." + name
        param_cat, sort_key = categorize_param(full_name)
        sorting_priority[param_cat] = sort_key
        if param_cat not in param_blocks:
            param_blocks[param_cat] = 0
        param_blocks[param_cat] += int(np.prod(jnp.shape(param)))
        print(f"{param_cat:<40}: {full_name:<50}: {jnp.shape(param)}")

    print("-"*47)
    for category in sorted(param_blocks.keys(), key=lambda cat: sorting_priority[cat]):
        n_params = param_blocks[category]
        print(f"{category:<40}: {n_params//1000:4d}k")
    print("-"*47)
    print(f"{'Total':<40}: {sum(param_blocks.values())//1000:4d}k")

###############################################################################
############################# Model / physics / hamiltonian ###################
###############################################################################

def without_cache(fixed_params):
    return {k:v for k,v in fixed_params.items() if k != "cache"}

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


def get_distance_matrix(r_el, full=True):
    """
    Compute distance matrix omitting the main diagonal (i.e. distance to the particle itself)
    Args:
        r_el: [batch_dims x n_electrons x 3]
    Returns:
        tuple: differences [batch_dims x n_el x n_el x 3], distances [batch_dims x n_el x n_el]
    """
    n_el = r_el.shape[-2]
    diff = r_el[..., None, :, :] - r_el[..., :, None, :] # r_el[..., :, None, :] - r_el[..., None, :, :]
    if full:
        # Fill diagonal != 0, so there is no problem with the gradients of the norm at r=0
        diff_padded = diff + jnp.eye(n_el)[..., None]
        dist = jnp.linalg.norm(diff_padded, axis=-1) * (1 - jnp.eye(n_el)) # Remove the diagonal elements again
    else:
        rows = [jnp.concatenate([diff[..., i, :i, :], diff[..., i, i + 1:, :]], axis=-2) for i in range(n_el)]
        diff = jnp.stack(rows, axis=-3)
        dist = jnp.linalg.norm(diff, axis=-1)
    return diff, dist


def generate_exp_distributed(rng, batch_shape, k=1.0):
    """Sample 3D points which are distributed spherically symmetrically, with an exponentially decaying radial pdf

    p(r) = r^2 * exp(-k*r)
    """
    xp = np.array([ 0.        ,  0.16646017,  0.2962203 ,  0.42016042,  0.54028054,
        0.6017006 ,  0.66318066,  0.78726079,  0.91426091,  1.04588105,
        1.15018115,  1.25946126,  1.37514138,  1.4998815 ,  1.73462173,
        2.22060222,  2.43890244,  2.66892267,  2.88428288,  3.004163  ,
        3.12246312,  3.23980324,  3.35662336,  3.47338347,  3.59030359,
        3.70764371,  3.82564383,  3.93746394,  4.05024405,  4.16416416,
        4.27938428,  4.3960844 ,  4.51440451,  4.63452463,  4.75662476,
        5.00734501,  5.26824527,  5.54106554,  5.82790583,  6.12646613,
        6.44380644,  6.78324678,  7.14934715,  7.54514755,  7.98038798,
        8.46468846,  9.01334901, 10.13523014, 11.71389171, 20.        ])
    Fp = np.array([0.00000000e+00, 6.78872002e-04, 3.47483629e-03, 9.05128575e-03,
       1.76268412e-02, 2.32836169e-02, 2.98157678e-02, 4.56084288e-02,
       6.52255332e-02, 8.89313535e-02, 1.09892813e-01, 1.33656048e-01,
       1.60527321e-01, 1.91123424e-01, 2.51942878e-01, 3.82805274e-01,
       4.40421447e-01, 4.98732268e-01, 5.50391443e-01, 5.77741958e-01,
       6.03679841e-01, 6.28341184e-01, 6.51818158e-01, 6.74201746e-01,
       6.95532631e-01, 7.15858180e-01, 7.35220119e-01, 7.52589600e-01,
       7.69166627e-01, 7.84977683e-01, 8.00044955e-01, 8.14391869e-01,
       8.28035657e-01, 8.40997278e-01, 8.53296166e-01, 8.75965247e-01,
       8.96197668e-01, 9.14129002e-01, 9.29897296e-01, 9.43441707e-01,
       9.55144213e-01, 9.65128109e-01, 9.73528150e-01, 9.80434071e-01,
       9.86033946e-01, 9.90453671e-01, 9.93834179e-01, 9.97521537e-01,
       9.99334839e-01, 9.99999544e-01])

    key_uniform, key_gaussian = jax.random.split(rng)
    r = jax.random.normal(key_gaussian, shape=(*batch_shape, 3))
    u = jax.random.uniform(key_uniform, shape=batch_shape)
    x = jnp.interp(u, Fp, xp)
    r = (r / np.linalg.norm(r, axis=-1, keepdims=True)) * (x / k)[..., None]
    return r


# def correction(R, R_org):
#     # shape N_ions x N_ions x 3
#     diff = R - R_org

#     # shape N_ions
#     dist = np.linalg.norm(diff, axis=-1, keepdims=True)
#     return np.tanh(dist) * diff * 0.5

# def displacement_fct(modes, eigvals, R_org, R):
#     n_atoms = R_org.shape[-2]
#     nb_modes = modes.shape[-1]
#     rand_dir = np.random.normal(0, 1, (1, nb_modes)) * 0.1

#     displacement = modes * rand_dir
#     displacement = displacement * (1/np.sqrt(eigvals))

#     delta_R = np.sum(displacement, axis=-1).reshape((n_atoms, 3))
#     delta_R -= correction(R, R_org)
#     return delta_R, R + delta_R

# Based on mir-group/nequip
# Adapted from mace_jax (fixed multiplicities)
def tp_out_irreps_with_instructions(
    irreps1: e3nn.Irreps, irreps2: e3nn.Irreps, target_irreps: e3nn.Irreps
) -> Tuple[e3nn.Irreps, List]:
    trainable = True

    # Collect possible irreps and their instructions
    irreps_out_list: List[Tuple[int, e3nn.Irrep]] = []
    instructions = []
    for i, (mul1, ir_in) in enumerate(irreps1):
        for j, (mul2, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                    k = len(irreps_out_list)  # instruction index
                    irreps_out_list.append((mul1*mul2, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))

    # We sort the output irreps of the tensor product so that we can simplify them
    # when they are provided to the second o3.Linear
    irreps_out = e3nn.Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()

    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    return irreps_out, instructions

###############################################################################
########################### Post-processing / analysis ########################
###############################################################################


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


def save_xyz_file(fname, R, Z, comment="", **extra_data):
    """Units for R are expected to be Bohr and will be translated to Angstrom in output"""
    assert len(R) == len(Z)
    PERIODIC_TABLE = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
         'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr']

    if extra_data:
        property_string = " Properties=species:S:1:pos:R:3"
        for key, values in extra_data.items():
            assert len(values) == len(R)
            if values.dtype == float:
                dtype_string = "R"
            elif values.dtype == int:
                dtype_string = "I"
            else:
                raise ValueError("Unsupported dtype for extra data")
            property_string += f":{key}:{dtype_string}:{values.shape[1]}"

    ANGSTROM_IN_BOHR = 1.88973
    with open(fname, "w") as f:
        f.write(str(len(R)) + "\n")
        f.write(comment)
        if extra_data:
            f.write(property_string)
        f.write("\n")
        for i, (Z_, R_) in enumerate(zip(Z, R)):
            f.write(
                f"{PERIODIC_TABLE[Z_-1]:3>} {R_[0]/ANGSTROM_IN_BOHR:-.10f} {R_[1]/ANGSTROM_IN_BOHR:-.10f} {R_[2]/ANGSTROM_IN_BOHR:-.10f}"
            )
            for values in extra_data.values():
                f.write(" " + " ".join([f"{v}" for v in values[i]]))
            f.write("\n")


def get_extent_for_imshow(x, y):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return x[0] - 0.5 * dx, x[-1] + 0.5 * dx, y[0] - 0.5 * dy, y[-1] + 0.5 * dy


def get_ion_pos_string(R):
    s = ",".join(["[" + ",".join([f"{x:.6f}" for x in R_]) + "]" for R_ in R])
    return "[" + s + "]"


def add_value_texts_to_barchart(axes, orient="v", space=.01, format_string = "{:.1f}", at_tip=False, **text_kwargs):
    def _single(_ax):
        if orient == "v":
            for p in _ax.patches:
                h = 0.0 if np.isnan(p.get_height()) else p.get_height()
                _x = p.get_x() + p.get_width() / 2
                if at_tip:
                    _y = p.get_y() + h * (1 + space)
                else:
                    _y = _ax.get_ylim()[0] + space * (_ax.get_ylim()[1] - _ax.get_ylim()[0])
                value = format_string.format(p.get_height())
                _ax.text(_x, _y, value, va='bottom', ha="center", **text_kwargs)
        elif orient == "h":
            for p in _ax.patches:
                w = 0.0 if np.isnan(p.get_width()) else p.get_width()
                if at_tip:
                    _x = p.get_x() + w * (1 + space)
                else:
                    _x = _ax.get_xlim()[0] + space * (_ax.get_xlim()[1] - _ax.get_xlim()[0])
                _y = p.get_y() + p.get_height() - p.get_height() / 2
                value = format_string.format(p.get_width())
                _ax.text(_x, _y, value, va='center', ha="left", **text_kwargs)

    if isinstance(axes, np.ndarray):
        for idx, ax in np.ndenumerate(axes):
            _single(ax)
    else:
        _single(axes)


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

def merge_params(params_init, params_reuse, enforce_equal_n_params=False):
    params_merge = hk.data_structures.merge(params_init, params_reuse)
    if enforce_equal_n_params:
        n_params_init = hk.data_structures.tree_size(params_init)
        n_params_reuse = hk.data_structures.tree_size(params_reuse)
        n_params_merge = hk.data_structures.tree_size(params_merge)
        assert n_params_init == n_params_reuse, f"Nr of initialized and reused params does not match: {n_params_init} != {n_params_reuse}"
        assert n_params_merge == n_params_init, f"Nr of merged and initialized params does not match: {n_params_merge} != {n_params_init}"
    return params_merge


def get_number_of_params(params):
    return hk.data_structures.tree_size(params)


def get_next_geometry_index(
    n_epoch: int,
    geometry_data_stores: List['GeometryDataStore'],
    scheduling_method: str,
    max_age: int,
    n_initial_round_robin_per_geom: int,
    permutation: List[int] = None,
) -> int:
    """
    Suggests next geometry to train on using either
    (1) round_robin -> always pick next geometry in line
    (2) stddev      -> pick pick geometry with highests E_std
    """
    if scheduling_method == "round_robin":
        idx_next = n_epoch % len(geometry_data_stores)
        if permutation is not None:
            return permutation[idx_next]
        else:
            return permutation
    elif scheduling_method == 'stddev':
        if n_epoch < len(geometry_data_stores) * n_initial_round_robin_per_geom:
            return get_next_geometry_index(n_epoch, geometry_data_stores, "round_robin", None, None, permutation)
        wf_ages = n_epoch - jnp.array([geometry_data_store.last_epoch_optimized for geometry_data_store in geometry_data_stores])
        max_age = max_age or int(len(geometry_data_stores) * 1.5)
        if jnp.any(wf_ages > max_age):
            index = jnp.argmax(wf_ages)
        else:
            stddevs = [jnp.sqrt(geometry_data_store.current_metrics['E_var']) for geometry_data_store in geometry_data_stores]
            index = np.argmax(stddevs)
        return index
    else:
        raise NotImplementedError("Wavefunction scheduler currently not supported.")


PERIODIC_TABLE = 'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr'.split()
KCAL_PER_MOL_IN_HARTREE = 0.0015936
ANGSTROM_IN_BOHR = 1.88973

if __name__ == '__main__':
    R = np.eye(3) * 5
    Z = [1,1,2]
    dipole = np.random.normal(size=(3,3))
    save_xyz_file("/home/mscherbela/tmp/test.xyz", R, Z, dipoles=dipole)


# %%
