from typing import Callable, Dict, Tuple, Literal
import jax
import jax.numpy as jnp
import optax
import haiku as hk
from deeperwin.mcmc import MetropolisHastingsMonteCarlo, MCMCState
from deeperwin.configuration import StandardOptimizerConfig
import re


def run_mcmc_with_cache(
    log_psi_sqr_func: Callable,
    cache_func_pmapped: Callable,
    mcmc: MetropolisHastingsMonteCarlo,
    params: Dict,
    spin_state: Tuple[int],
    mcmc_state: MCMCState,
    fixed_params: Dict,
    split_mcmc=True,
    merge_mcmc=True,
    mode: Literal["burnin", "intersteps"] = "intersteps",
):
    if split_mcmc:
        mcmc_state = mcmc_state.split_across_devices()

    if cache_func_pmapped is not None:
        fixed_params["cache"] = cache_func_pmapped(params, *spin_state, *mcmc_state.build_batch(fixed_params))

    if mode == "burnin":
        mcmc_state = mcmc.run_burn_in(log_psi_sqr_func, mcmc_state, params, *spin_state, fixed_params)
    elif mode == "intersteps":
        mcmc_state = mcmc.run_inter_steps(log_psi_sqr_func, mcmc_state, params, *spin_state, fixed_params)
    else:
        raise ValueError(f"Unknown MCMC mode: {mode}")
    if merge_mcmc:
        mcmc_state = mcmc_state.merge_devices()
    return mcmc_state, fixed_params


def build_lr_schedule(base_lr, schedule_config):
    if schedule_config.name == "inverse":

        def get_lr(t):
            lr = base_lr / (1 + (t + schedule_config.offset_time) / schedule_config.decay_time)
            lr = jnp.maximum(schedule_config.minimum, lr)
            if schedule_config.warmup > 0:
                lr *= jnp.minimum((t + 1) / schedule_config.warmup, 1.0)
            return lr

        return get_lr
    elif schedule_config.name == "exponential":

        def get_lr(t):
            lr = base_lr * jnp.exp(-(t + schedule_config.offset_time) / schedule_config.decay_time)
            lr = jnp.maximum(schedule_config.minimum, lr)
            if schedule_config.warmup > 0:
                lr *= jnp.minimum((t + 1) / schedule_config.warmup, 1.0)
            return lr

        return get_lr
    # TODO: test more thoroughly + prefactor should perhaps be model size
    elif schedule_config.name == "noam":
        prefactor = schedule_config.warmup_steps
        return lambda t: base_lr * (
            prefactor**0.5 * min(max(1, t) ** (-0.5), max(1, t) * schedule_config.warmup_steps ** (-1.5))
        )
    elif schedule_config.name == "fixed":
        return lambda t: base_lr
    else:
        raise ValueError(f"Unsupported config-value for optimization.schedule.name: {schedule_config.name}")


def build_optax_optimizer(config: StandardOptimizerConfig):
    lr_schedule = build_lr_schedule(config.learning_rate, config.lr_schedule)
    if config.name in ["adam", "sgd", "rmsprop", "lamb", "lion"]:
        optimizer = getattr(optax, config.name)(lr_schedule)
    else:
        raise ValueError(f"Unknown optimizer: {config.name}")

    # Scale gradients for selected modules
    if hasattr(config, "scaled_modules") and config.scaled_modules:
        regex = re.compile("(" + "|".join(config.scaled_modules) + ")")

        def leaf_filter_func(module_name, name, v):
            return len(regex.findall(f"{module_name}/{name}")) > 0

        optimizer = optax.chain(
            optimizer,
            optax.masked(optax.scale(config.scale_lr), lambda params: hk.data_structures.map(leaf_filter_func, params)),
        )
    return optimizer
