import itertools
import numpy as np
import jax
import jax.scipy
from jax import numpy as jnp
import chex
from deeperwin.configuration import PeriodicConfig

def get_kpoints_in_sphere(rec_lattice, n_kpoints_min):
    """
    References: https://github.com/deepmind/ferminet/blob/main/ferminet/pbc/envelopes.py
    """
    dk = 1e-5
    # Generate ordinals of the lowest min_kpoints kpoints
    max_k = int(jnp.ceil(n_kpoints_min + dk) ** (1 / 3.))
    ordinals = sorted(range(-max_k, max_k + 1), key=abs)
    ordinals = jnp.asarray(list(itertools.product(ordinals, repeat=3)))

    kpoints = ordinals @ rec_lattice.T
    kpoints = jnp.asarray(sorted(kpoints, key=jnp.linalg.norm))
    k_norms = jnp.linalg.norm(kpoints, axis=1)

    return kpoints[k_norms <= k_norms[n_kpoints_min - 1] + dk]

def get_kpoint_grid(rec_lattice, n_k_per_dim, map_to_first_bz=True):
    n_k_per_dim = np.array(n_k_per_dim, int)
    ordinals = np.meshgrid(*[np.arange(n) for n in n_k_per_dim], indexing='ij')
    ordinals = np.stack(ordinals, axis=-1).reshape(-1, 3)
    k_frac = ordinals / n_k_per_dim
    k_vecs = k_frac @ rec_lattice.T  # rec lattice shape: [dim x vec]
    if map_to_first_bz:
        k_vecs = map_to_first_brillouin_zone(k_vecs, rec_lattice)
    return k_vecs

def cartesian_to_fractional(r, *, inv_lattice):
    return r @ inv_lattice


def fractional_to_cartesian(r_frac, *, lattice):
    return r_frac @ lattice

def is_commensurable(lattice, k_points):
    """
    Args:
        lattice: (3, 3) array
        k_points: (3, k) array
    """
    #  lattice shape: [vec x dim]. Dot product over physical dimension
    phases = lattice.T @ k_points
    return np.allclose(np.cos(phases), 1.0, atol=1e-4) and np.allclose(np.sin(phases), 0.0, atol=1e-4)


def project_into_first_unit_cell(r, lattice, inv_lattice=None, around_origin=False):
    if inv_lattice is None:
        inv_lattice = jnp.linalg.inv(lattice)
    r_frac = cartesian_to_fractional(r, inv_lattice=inv_lattice)
    if around_origin:
        r_frac = jnp.mod(r_frac + 0.5, 1.0) - 0.5
    else:
        r_frac = jnp.mod(r_frac, 1.0)
    r = fractional_to_cartesian(r_frac, lattice=lattice)
    return r

def map_to_first_voronoi_cell(r, lattice, n_max=5):
    """lattice shape: [vec x dim]"""
    r = project_into_first_unit_cell(r, lattice, around_origin=True)
    ndim = r.shape[-1]

    ordinals = np.meshgrid(*[np.arange(-n_max, n_max + 1) for _ in range(ndim)])
    ordinals = np.stack(ordinals, axis=-1).reshape([-1, ndim])
    shifts = ordinals @ lattice  # [n_ordinals x n_vecs] @ [n_vecs x n_dims]
    r_shifted = r + shifts
    r_norm = jnp.linalg.norm(r_shifted, axis=-1)
    r_wigner = r_shifted[jnp.argmin(r_norm, axis=-1), :]
    return r_wigner

def map_to_first_brillouin_zone(k_points, rec_lattice):
    """rec_lattice shape: [dim x vec]"""
    return jax.vmap(map_to_first_voronoi_cell, in_axes=(0, None))(k_points, rec_lattice.T)


@chex.dataclass
class LatticeParams:
    rec: jax.Array = None
    lattice: jax.Array = None
    volume: float = None
    gamma: float = None
    madelung_const: float = None
    lat_vectors: jax.Array = None
    rec_vectors: jax.Array = None
    rec_vec_square: jax.Array = None
    rec_vectors_weights: jax.Array = None
    k_twist: jax.Array = None

    @classmethod
    def from_periodic_config(cls, periodic_config: PeriodicConfig):
        """
        References: https://github.com/deepmind/ferminet/blob/main/ferminet/pbc/hamiltonian.py
        """
        lattice = periodic_config.lattice

        rec = 2 * jnp.pi * jnp.linalg.inv(lattice)
        #  periodic_config.k_twist in units of vec, rec.shape = [dim x vec]
        k_twist = np.array(periodic_config.k_twist) @ rec.T
        volume = jnp.abs(jnp.linalg.det(lattice))

        if periodic_config.gamma_option == "min_rec":
            smallestheight = jnp.amin(1 / jnp.linalg.norm(rec.T / 2 * jnp.pi, axis=1)) # norm over lattice vectors
            gamma = (5.0 / smallestheight) ** 2 # (2.8 / volume ** (1 / 3)) ** 2
        else:
            gamma = (2.8 / volume ** (1 / 3)) ** 2

        # lattice vectors
        ordinals = sorted(range(-periodic_config.truncation_limit, periodic_config.truncation_limit + 1), key=abs) #  ordinals in units of vec, rec.shape = [dim x vec]
        ordinals = np.meshgrid(ordinals, ordinals, ordinals)
        ordinals = np.stack(ordinals, axis=-1).reshape([-1, 3])
        lat_vectors = ordinals @ lattice

        # remove lattice vectors with almost zero contribution
        # shift all ordinals by one closer to the origin to be on the safe side regarding small contributions
        # because we don't want to sample over all electron positions in our supercell
        ordinals_shifted = ordinals - np.sign(ordinals)

        # compute real space ewald without final summation
        displacements = jnp.linalg.norm(ordinals_shifted @ lattice + 0.0001, axis=-1)
        weight = jax.scipy.special.erfc(gamma ** 0.5 * displacements) / displacements

        # remove lat vectors with a contribution less than eps
        bigweight = weight > 1e-12
        lat_vectors = lat_vectors[bigweight]

        lat_vec_norm = jnp.linalg.norm(lat_vectors[1:], axis=-1)

        # reciprocal vectors
        rec_vectors = ordinals[1:] @ rec.T
        rec_vec_square = jnp.einsum('ij,ij->i', rec_vectors, rec_vectors)

        rec_vectors_weight = 4 * jnp.pi * jnp.exp(-rec_vec_square / (4 * gamma))
        rec_vectors_weight /= volume * rec_vec_square
        bigweight = rec_vectors_weight > 1e-12
        rec_vectors, rec_vectors_weight = rec_vectors[bigweight], rec_vectors_weight[bigweight]

        # madelung const
        madelung_const = (
                jnp.sum(jax.scipy.special.erfc(gamma ** 0.5 * lat_vec_norm) / lat_vec_norm)
                - 2 * gamma ** 0.5 / jnp.pi ** 0.5)

        # rec_vectors_weight = 4 * jnp.pi * jnp.exp(-rec_vec_square / (4 * gamma))
        # rec_vectors_weight /= volume * rec_vec_square
        madelung_const += (jnp.sum(rec_vectors_weight) - jnp.pi / (volume * gamma))

        return cls(rec=rec,
                   lattice=lattice,
                   volume=volume,
                   gamma=gamma,
                   madelung_const=madelung_const,
                   lat_vectors=lat_vectors,
                   rec_vectors=rec_vectors,
                   rec_vectors_weights=rec_vectors_weight,
                   k_twist=k_twist)
