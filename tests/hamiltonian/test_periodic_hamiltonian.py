import numpy as np

import jax.numpy as jnp
import jax

from deeperwin.configuration import Configuration, PhysicalConfig, PeriodicConfig, EmbeddingConfigFermiNet
from deeperwin.model.wavefunction import build_log_psi_squared
from deeperwin.hamiltonian import build_local_energy


def test_periodic_hamiltonian():
    """
    Test for the periodic hamiltonian.
    References: https://github.com/deepmind/ferminet/blob/main/ferminet/pbc/tests/features_test.py
    """
    key = jax.random.PRNGKey(42)

    # geometry definition
    spin_state = (6, 5)
    R = jnp.asarray([[0.0, 0.0, 0.2], [1.2, 1.0, -0.2], [2.5, -0.8, 0.6]])
    Z = jnp.asarray([2, 5, 7])

    # build baseline config
    config = Configuration(
        physical=PhysicalConfig(
            R=R,
            Z=Z,
            n_up=spin_state[0],
            n_electrons=sum(spin_state),
            periodic=PeriodicConfig(
                lattice_prim=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], supercell=[1, 1, 1]
            ),
        )
    )
    config.model.embedding = EmbeddingConfigFermiNet(n_hidden_one_el=10, n_hidden_two_el=5, n_iterations=2)

    # build log psi squared & local energy
    log_psi_squared, orbital_func, cache_func, params, fixed_params = build_log_psi_squared(
        config.model, config.physical, None, fixed_params=None, rng_seed=1234
    )
    get_local_energy = build_local_energy(
        log_psi_squared,
        True,  # is complex
        config.physical.periodic is not None,
        False,  # heg background
    )

    # init electron positions
    key, subkey = jax.random.split(key)
    r = jax.random.uniform(subkey, shape=(1, 11, 3))

    # local energy
    loc_energy_1 = get_local_energy(params, spin_state, r, R, Z, fixed_params)

    # Shift a random electron
    key, subkey = jax.random.split(key)
    e_idx = jax.random.randint(subkey, (1,), 0, r.shape[1])
    key, subkey = jax.random.split(key)
    randvec = jax.random.randint(subkey, (3,), 0, 100).astype(jnp.float32)
    r = r.at[0, e_idx].add(randvec)

    # local energy
    loc_energy_2 = get_local_energy(params, spin_state, r, R, Z, fixed_params)

    atol, rtol = 4.0e-3, 4.0e-3
    assert np.isclose(loc_energy_1, loc_energy_2, atol=atol, rtol=rtol)


if __name__ == "__main__":
    test_periodic_hamiltonian()
