from dataclasses import dataclass
from typing import Dict, Tuple
import scipy
from jax import numpy as jnp
from deeperwin.configuration import Configuration, PhysicalConfig, DistortionConfig
from deeperwin.loggers import LoggerCollection, WavefunctionLogger
from deeperwin.mcmc import MCMCState
from deeperwin.run_tools.dispatch import idx_to_job_name
from deeperwin.utils.utils import LOGGER, get_el_ion_distance_matrix, setup_job_dir, PERIODIC_TABLE, ANGSTROM_IN_BOHR
from deeperwin.loggers import initialize_training_loggers
import numpy as np
import haiku as hk
import dataclasses

@dataclass
class GeometryDataStore:
    """
    Data class to store all optimization related data for
    each unique geometry/compound seen during training
    """
    idx: int = None
    physical_config: PhysicalConfig = None
    physical_config_original: PhysicalConfig = None
    rotation: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(3))
    spin_state: Tuple[int] = None
    mcmc_state: MCMCState = None
    fixed_params: Dict = None
    clipping_state: Tuple[float, float] = None
    wavefunction_logger: WavefunctionLogger = None
    current_metrics = {}
    n_distortions: int = 0
    n_opt_epochs: int = 0
    n_opt_epochs_last_dist: int = 0
    last_epoch_optimized: int = 0
    weight: float = None


    def init_wave_function_logger(self, config: Configuration) -> None:
        job_name = idx_to_job_name(self.idx)
        job_dir = setup_job_dir(".", job_name)
        loggers = initialize_training_loggers(config, True, self.idx, job_dir, True)
        self.wavefunction_logger = WavefunctionLogger(loggers, prefix="opt", n_step=config.optimization.n_epochs_prev, smoothing=0.05)

def parse_xyz(xyz_content):
    """
    Parse the content of an XYZ file

    Args:
        xyz_content: str; Molecule descriptor in XYZ format: Nr of atoms, comment line, atoms positions in angstrom

    Returns:
        R, Z, comment
    """
    lines = [l.strip() for l in xyz_content.split("\n") if l]
    n_atoms = int(lines[0])
    assert len(lines) >= (n_atoms + 2), "Number of atoms exceeds nr of lines in xyz content"
    comment = lines[1]
    R = []
    Z = []
    for i, l in enumerate(lines[2:]):
        if i == n_atoms:
            break
        atom_type, x, y,z = l.split()
        Z.append(PERIODIC_TABLE.index(atom_type) + 1)
        R.append([float(x) * ANGSTROM_IN_BOHR, float(y) * ANGSTROM_IN_BOHR, float(z) * ANGSTROM_IN_BOHR])
    return np.array(R), np.array(Z, int), comment

def parse_coords(coords_content):
    lines = coords_content.split("\n")
    is_content = False
    R = []
    Z = []
    for l in lines:
        if "$coord" in l:
            is_content = True
            continue
        if "$end" in l:
            break
        if "$" in l:
            continue
        if is_content:
            x, y, z, atom_type = l.split()
            Z.append(PERIODIC_TABLE.index(atom_type.capitalize()) + 1)
            R.append([float(x), float(y), float(z)])
    return np.array(R), np.array(Z, int)

def distort_geometry(g: GeometryDataStore, config: DistortionConfig):
    R_old = np.array(g.physical_config.R)
    R_orig = np.array(g.physical_config_original.R)
    if g.n_distortions < config.reset_every_n_distortions:
        R_new = get_distortion(g.fixed_params['hessian'],
                            R_old,
                            R_orig @ g.rotation,
                            energy_per_mode=config.distortion_energy,
                            min_stiffness=config.min_stiffness,
                            bias_towards_orig=config.bias_towards_orig,
                            min_dist_factor=config.min_distance_factor,
                            )
        g.n_distortions += 1
    else:
        LOGGER.info(f"Resetting geometry {g.idx} to original geometry.")
        g.n_distortions = 0
        R_new = R_orig @ g.rotation

    # Adjust el. pos & ion pos
    r_new = space_warp_coordinate_transform(g.mcmc_state.r, R_new, R_old, config.space_warp)
    U, r_new, R_new = apply_random_global_rotation(r_new, R_new)

    # Update mcmc state & physical config & fixed params & clipping state
    g.physical_config.R = R_new.tolist()
    g.rotation = g.rotation @ U
    g.mcmc_state.r = r_new
    g.mcmc_state.R = R_new
    g.clipping_state = (None, None) # this would do basically psiformer clipping with v18 config

    # Reset distortion count for max age
    g.n_opt_epochs_last_dist = 0
    return g


def get_distortion(hessian, R, R_orig, energy_per_mode=0.05, min_stiffness=0.2, bias_towards_orig=0.1, min_dist_factor=0.8):
    try:
        eigvals, eigvecs = np.linalg.eigh(hessian)
        eigvals = np.maximum(eigvals, min_stiffness)
    except np.linalg.LinAlgError:
        eigvals = np.ones(hessian.shape[0]) * 10
        eigvecs = np.eye(hessian.shape[0])
        LOGGER.warning("Could not diagonalize hessian, using 10x identity matrix instead.")
    n_trials = 100

    # Draw random displacement per normal mode
    dx = np.random.normal(size=[n_trials, len(eigvals)])

    # Scale displacement to have the desired energy per mode => Move more along soft modes
    dx *= np.sqrt(energy_per_mode / eigvals)
    delta_R = np.einsum("ki,ni->nk",eigvecs, dx).reshape([n_trials, -1, 3])

    # Add bias towards the original positions (computed in normal mode basis)
    delta_R += (R_orig - R) * bias_towards_orig

    # Remove global shift
    delta_R -= np.mean(delta_R, axis=-2, keepdims=True)
    R_trial = R + delta_R

    # Compute all interatomic distances and accept only if no atoms get too close
    dist_matrix_orig = np.linalg.norm(R_orig[:, None, :] - R_orig[None, :, :], axis=-1) + np.eye(len(R))
    for i, R in enumerate(R_trial):
        dist_matrix = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1) + np.eye(len(R))
        dist_ratio = dist_matrix / dist_matrix_orig
        if np.all(dist_ratio > min_dist_factor):
            break
    else:
        LOGGER.warning(f"Could not find a distortion that satisfies the distance constraints: {np.min(dist_ratio)}")
    return R


def apply_random_global_rotation(r, R):
    U = scipy.spatial.transform.Rotation.random().as_matrix()
    U = jnp.array(U)
    R_rot = R @ U
    r_rot = r @ U
    return U, r_rot, R_rot


def space_warp_coordinate_transform(r, R, R_orig, space_warp):
    assert (R.ndim == 2) and (R_orig.ndim == 2), f"Shape R={R.shape}, R_orig={R_orig.shape}"
    _, dist = get_el_ion_distance_matrix(r, R)
    delta_R = R - R_orig

    # Shape n_el
    if space_warp == "nearest":
        ind_dist_closest = jnp.argmin(dist, axis=-1, keepdims=False)
        delta_R_per_el = delta_R[ind_dist_closest]
    elif space_warp == "1/r4":
        eps = 1e-8
        weights = 1 / (dist ** 4 + eps) # Shape n_el x n_ion
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True) # Normalize across ions Shape n_el x n_ion
        delta_R_per_el = jnp.einsum("...iI,Id->...id", weights, delta_R) # Shape [batch x n_el x 3]
    return r + delta_R_per_el


if __name__ == '__main__':
    with open("/home/mscherbela/develop/deeperwin_jaxtest/datasets/geometries/HEAT/coord.c2h2") as f:
        content = f.read()
    R, Z = parse_coords(content)

