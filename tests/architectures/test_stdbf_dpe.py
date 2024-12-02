import jax.numpy as jnp
import jax
import numpy as np

from deeperwin.configuration import Configuration
from deeperwin.model.wavefunction import build_log_psi_squared
from deeperwin.utils.utils import replicate_across_devices, get_number_of_params
from deeperwin.mcmc import MCMCState
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "../configs"
CHECKPOINTS_DIR = Path(__file__).parent / "../checkpoints"


def test_stdbf_dpe_architecture():
    config_file = CONFIG_DIR / "config_stdbf_dpe.yml"
    raw_config, config = Configuration.load_configuration_file(config_file)
    assert isinstance(raw_config, dict)

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
    (
        log_psi_squared,
        orbital_func,
        cache_func,
        params,
        fixed_params,
    ) = build_log_psi_squared(
        config=config.model,
        physical_config=config.physical,
        baseline_config=config.baseline,
        fixed_params=None,
        rng_seed=rng_seed,
    )

    # make sure that the number didn't blow up due to new modules
    nb_params = get_number_of_params(params)
    assert nb_params < 2000, f"Nb. of parameters: {get_number_of_params(params)}"

    # take random coordinates from mcmc state and evaluate a forward pass of log psi
    log_prob, phase = log_psi_squared(params, n_up, n_dn, mcmc_state.r, R, Z, fixed_params)
    assert log_prob is not None
    assert log_prob.shape == (n,)
