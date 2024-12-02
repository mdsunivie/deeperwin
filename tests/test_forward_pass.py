# TODO: have this test for all models (MACE, PauliNet, FermiNet, dpe, transformer, etc)

import jax.numpy as jnp
import jax
import numpy as np

from deeperwin.configuration import Configuration, PhysicalConfig
from deeperwin.model.wavefunction import build_log_psi_squared
from deeperwin.utils.utils import replicate_across_devices
from deeperwin.mcmc import MCMCState


def test_forward_pass():
    """
    Tests if simple forward passes through the model don't cause
    any errors to be thrown
    """
    # construct restricted closed shell configuration for Nitrogen dimer
    config = Configuration()
    config.physical = PhysicalConfig(name="N2")
    config.model.orbitals.determinant_schema = "full_det"

    config.model.embedding.h_one_correlation = 0
    config.model.embedding.use_layer_norm = False
    config.model.embedding.use_h_one_mlp = True
    config.model.embedding.use_average_h_one = True
    config.model.embedding.use_h_one_same_diff = True

    # geometry definition
    R, Z = jnp.array(config.physical.R), jnp.array(config.physical.Z)
    n_up, n_dn = config.physical.n_up, config.physical.n_dn

    # sample random electron coordinates through MCMC state object
    n = 5
    mcmc_state = MCMCState.initialize_around_nuclei(
        n, config.physical, "gaussian", "el_ion_mapping", jax.random.PRNGKey(100)
    )

    # build log_psi_squared
    rng_seed = int(replicate_across_devices(np.array([10]))[0])
    log_psi_squared, _, _, params, fixed_params = build_log_psi_squared(
        config=config.model, physical_config=config.physical, baseline_config=None, fixed_params=None, rng_seed=rng_seed
    )

    # take random coordinates from mcmc state & check symmetry constraint 5 times
    for idx in range(n):
        r_coordinates = mcmc_state.r[idx, :, :]
        log_prob = log_psi_squared(params, n_up, n_dn, r_coordinates, R, Z, fixed_params)
        assert log_prob is not None


if __name__ == "__main__":
    test_forward_pass()
