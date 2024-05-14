import logging
import jax
import jax.numpy as jnp
import kfac_jax
from deeperwin.configuration import ClippingConfig
from deeperwin.utils.utils import pmean, without_cache
import functools

LOGGER = logging.getLogger("dpe")

def init_clipping_state(is_complex=False):
    if is_complex:
        return jnp.array(0.0 + 0.0j), jnp.array(1e12 + 1e12j)
    else:
        return jnp.array(0.0), jnp.array(1e12)

def _get_clipping_center_and_width(E, clipping_config: ClippingConfig):
    center = dict(mean=jnp.nanmean, median=jnp.nanmedian)[clipping_config.center](E)
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


def _update_clipping_state(E, clipping_config: ClippingConfig):
    if jnp.iscomplexobj(E):
        center_real, width_real = _get_clipping_center_and_width(E.real, clipping_config)
        center_imag, width_imag = _get_clipping_center_and_width(E.imag, clipping_config)
        if clipping_config.clip_imag_around_0:
            center_imag = 0.0
        return center_real + 1.j * center_imag, width_real + 1.j * width_imag
    else:
        return _get_clipping_center_and_width(E, clipping_config)

def _get_clip_func(name):
    if name == "hard":
        return lambda E, center, width: jnp.clip(E, center - width, center + width)
    elif name == "tanh":
        def _clip(E, center, width):
            if not jnp.iscomplexobj(E):
                return center + jnp.tanh((E - center) / width) * width
            else:
                deviation_real = jnp.tanh((E.real - center.real) / width.real) * width.real
                deviation_imag = jnp.tanh((E.imag - center.imag) / width.imag) * width.imag
                return center + deviation_real + 1.j * deviation_imag
        return _clip
    else:
        raise ValueError(f"Unsupported config-value for optimization.clipping.name: {name}")

def _clip_energies(E, clipping_state, clipping_config: ClippingConfig):
    center, width = clipping_state
    clip_func = _get_clip_func(clipping_config.name)
    if (not clipping_config.from_previous_step) or (center is None):
        center, width = _update_clipping_state(E, clipping_config)

    clipped_energies = clip_func(E, center, width)

    # update clipping center & width
    new_clipping_state = _update_clipping_state(clipped_energies, clipping_config)
    return clipped_energies, new_clipping_state

def build_value_and_grad_func(log_psi_sqr_func, get_local_energy, clipping_config: ClippingConfig, is_complex=False, kfac_register_complex=False):
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
        E_loc = get_local_energy(params, spin_state, *batch)
        E_mean = pmean(jnp.nanmean(E_loc))
        E_var = pmean(jnp.nanmean((E_loc - E_mean) * jnp.conj(E_loc - E_mean)))

        E_loc_clipped, clipping_state = _clip_energies(E_loc, clipping_state, clipping_config)
        E_mean_clipped = pmean(jnp.nanmean(E_loc_clipped))
        E_var_clipped = pmean(jnp.nanmean((E_loc_clipped - E_mean_clipped) * jnp.conj(E_loc_clipped - E_mean_clipped)))
        aux = dict(E_mean=E_mean,
                   E_var=E_var,
                   E_mean_clipped=E_mean_clipped,
                   E_var_clipped=E_var_clipped,
                   E_loc_clipped=E_loc_clipped,
                   E_loc=E_loc)
        loss = E_mean_clipped.real
        return loss, (clipping_state, aux)

    @total_energy.defjvp
    def total_energy_jvp(spin_state, primals, tangents):
        params, state, batch = primals
        r, R, Z, fixed_params = batch
        batch_size = batch[0].shape[0]
        loss, (state, stats) = total_energy(params, state, spin_state, batch)

        if is_complex:
            def func(p):
                # We cannot use the cached TAOs here, because KFAC needs to see the full pass through the network, 
                # including through the TAOs. When using a different optimizer, one could cache here and save potentially a lot of computation.
                return log_psi_sqr_func(p, *spin_state, r, R, Z, without_cache(fixed_params))

            (log_psi_phase, log_psi_sqr), (tan_phase, tan_log_psi_sqr) = jax.jvp(func, (primals[0],), (tangents[0],))
            diff = stats["E_loc_clipped"] - stats["E_mean_clipped"]
            real_term = jnp.dot(tan_log_psi_sqr, diff.real)
            imag_term = 2 * jnp.dot(tan_phase, diff.imag)
            # Effetively register log |psi| here instead of log |psi|^2, which is consistent with FermiNet.
            # This effectively leads to higher relative damping and higher effective learning rates
            kfac_jax.register_normal_predictive_distribution(0.5 * log_psi_sqr[:, None])
            if kfac_register_complex:
                # Register log psi instead of log |psi|
                # log_psi += 1.0j * log_psi_phase
                kfac_jax.register_normal_predictive_distribution(log_psi_phase[:, None])

            primals_out = loss, (state, stats)
            tangents_out = ((real_term + imag_term) / batch_size, (state, stats))
        else:
            def func(params):
                # We cannot use the cached TAOs here, because KFAC needs to see the full pass through the network, 
                # including through the TAOs. When using a different optimizer, one could cache here and save potentially a lot of computation.
                return log_psi_sqr_func(params, *spin_state, r, R, Z, without_cache(fixed_params))[1]

            log_psi_sqr, tangents_log_psi_sqr = jax.jvp(func, (primals[0],), (tangents[0],))
            diff = stats["E_loc_clipped"] - stats["E_mean_clipped"]
            kfac_jax.register_normal_predictive_distribution(0.5 * log_psi_sqr[:, None])  # Register loss for kfac optimizer

            primals_out = loss, (state, stats)
            tangents_out = jnp.dot(tangents_log_psi_sqr, diff) / batch_size, (state, stats)
        return primals_out, tangents_out

    return jax.value_and_grad(total_energy, has_aux=True)
