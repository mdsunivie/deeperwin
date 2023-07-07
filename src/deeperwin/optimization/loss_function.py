import logging
import jax
import jax.numpy as jnp
import kfac_jax
from deeperwin.configuration import ClippingConfig
from deeperwin.hamiltonian import get_local_energy
from deeperwin.utils.utils import pmean, without_cache
import functools

LOGGER = logging.getLogger("dpe")

def init_clipping_state():
    return jnp.array([0.0]).squeeze(), jnp.array([1e5]).squeeze()

def _update_clipping_state(E, clipping_state, clipping_config: ClippingConfig):
    del clipping_state
    center = dict(mean=jnp.nanmean,
                  median=jnp.nanmedian,
                  )[clipping_config.center](E)
    center = pmean(center)
    if clipping_config.width_metric == 'mae':
        width = jnp.nanmean(jnp.abs(E-center))
        width = pmean(width)
    elif clipping_config.width_metric == 'std':
        width = jnp.nanmean((E-center)**2)
        width = jnp.sqrt(pmean(width))
    else:
        raise NotImplementedError(f"Unknown clipping metric: {clipping_config.width_metric}")
    return center, width * clipping_config.clip_by

def _clip_energies(E, clipping_state, clipping_config: ClippingConfig):
    center, width = clipping_state
    if (not clipping_config.from_previous_step) or (center is None):
        center, width = _update_clipping_state(E, clipping_state, clipping_config)

    if clipping_config.name == "hard":
        clipped_energies = jnp.clip(E, center - width, center + width)
    elif clipping_config.name == "tanh":
        clipped_energies = center + jnp.tanh((E - center) / width) * width
    else:
        raise ValueError(f"Unsupported config-value for optimization.clipping.name: {clipping_config.name}")
    new_clipping_state = _update_clipping_state(clipped_energies, clipping_state, clipping_config)
    return clipped_energies, new_clipping_state

def build_value_and_grad_func(log_psi_sqr_func, clipping_config: ClippingConfig):
    """
    Returns a callable that computes the gradient of the mean local energy for a given set of MCMC walkers with respect to the model defined by `log_psi_func`.

    Args:
        log_psi_sqr_func (callable): A function representing the wavefunction model
        clipping_config (ClippingConfig): Clipping hyperparameters
        use_fwd_fwd_hessian (bool): If true, the second partial derivatives required for computing the local energy are obtained with a forward-forward scheme.

    """

    # Build custom total energy jvp. Based on https://github.com/deepmind/ferminet/blob/jax/ferminet/train.py
    @functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
    def total_energy(params, state, spin_state, batch):
        # TODO: why is spin state no integer anymore here now??
        clipping_state = state 
        E_loc = get_local_energy(log_psi_sqr_func, params, spin_state, *batch)
        E_mean = pmean(jnp.nanmean(E_loc))
        E_var = pmean(jnp.nanmean((E_loc - E_mean) ** 2))

        E_loc_clipped, clipping_state = _clip_energies(E_loc, clipping_state, clipping_config)
        E_mean_clipped = pmean(jnp.nanmean(E_loc_clipped))
        E_var_clipped = pmean(jnp.nanmean((E_loc_clipped - E_mean_clipped) ** 2))
        aux = dict(E_mean=E_mean,
                   E_var=E_var,
                   E_mean_clipped=E_mean_clipped,
                   E_var_clipped=E_var_clipped,
                   E_loc_clipped=E_loc_clipped)
        loss = E_mean_clipped
        return loss, (clipping_state, aux)

    @total_energy.defjvp
    def total_energy_jvp(spin_state, primals, tangents):
        params, state, batch = primals
        r, R, Z, fixed_params = batch
        batch_size = batch[0].shape[0]

        loss, (state, stats) = total_energy(params, state, spin_state, batch)
        diff = stats["E_loc_clipped"] - stats["E_mean_clipped"]

        def func(params):
            return log_psi_sqr_func(params, *spin_state, r, R, Z, without_cache(fixed_params))

        log_psi_sqr, tangents_log_psi_sqr = jax.jvp(func, (primals[0],), (tangents[0],))
        kfac_jax.register_normal_predictive_distribution(log_psi_sqr[:, None])  # Register loss for kfac optimizer

        primals_out = loss, (state, stats)
        tangents_out = jnp.dot(tangents_log_psi_sqr, diff) / batch_size, (state, stats)
        return primals_out, tangents_out

    return jax.value_and_grad(total_energy, has_aux=True)