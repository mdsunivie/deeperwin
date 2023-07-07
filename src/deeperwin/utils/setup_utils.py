import jax
from jax import numpy as jnp
import haiku as hk
from typing import Dict, Optional

from deeperwin.configuration import Configuration
from deeperwin.loggers import LoggerCollection
from deeperwin.utils.utils import getCodeVersion
from deeperwin.checkpoints import delete_obsolete_checkpoints


def initialize_training_loggers(
        config: Configuration,
        params: Dict[str, Dict[str, jnp.DeviceArray]],
        fixed_params: Dict[str, Dict[str, jnp.DeviceArray]],
        use_wandb_group: bool = False,
        exp_idx_in_group: Optional[int] = None,
        save_path=".",
        parallel_wandb_logging=False,
) -> LoggerCollection:
    if jax.process_index() == 0:
        loggers = LoggerCollection(config=config.logging, 
                                   name=config.experiment_name,
                                   use_wandb_group=use_wandb_group,
                                   exp_idx_in_group=exp_idx_in_group,
                                   save_path=save_path,
                                   parallel_wandb_logging=parallel_wandb_logging)
        loggers.on_run_begin()
        loggers.log_config(config)
        loggers.log_tags(config.logging.tags)
        loggers.log_param("code_version", getCodeVersion())
        loggers.log_param("n_params", hk.data_structures.tree_size(params))
        if "baseline_energies" in fixed_params:
            loggers.log_metrics(fixed_params["baseline_energies"])
    else:
        loggers = None

    return loggers

def finalize_experiment_run(
    config: Configuration, 
    loggers, 
    params, 
    fixed_params, 
    mcmc_state, 
    opt_state, 
    clipping_state,
    ema_params=None
) -> None:
    if jax.process_index() == 0:
        loggers.log_checkpoint(config.optimization.n_epochs_total, params, fixed_params, mcmc_state, opt_state, clipping_state, ema_params)
        delete_obsolete_checkpoints(config.optimization.n_epochs_total, config.optimization.checkpoints)
        loggers.on_run_end()
