import logging
import os
import pickle
import re

import ruamel.yaml

from deeperwin.configuration import (
    Configuration,
    build_flattend_dict,
    CheckpointConfig,
    PhisNetModelConfig,
    build_physical_configs_from_changes,
)
from deeperwin.mcmc import MCMCState
import jax
import numpy as np
import haiku as hk
import zipfile
from dataclasses import dataclass, fields
from typing import Optional, Any, List, Union
from deeperwin.utils.utils import split_params, without_cache


@dataclass
class RunData:
    config: Optional[Union[Configuration, dict]] = None
    history: Optional[List[dict]] = None
    summary: Optional[dict] = None
    metadata: Optional[dict] = None
    params: Optional[dict] = None
    ema_params: Optional[dict] = None
    fixed_params: Optional[dict] = None
    opt_state: Optional[Any] = None
    mcmc_state: Optional[MCMCState] = None
    clipping_state: Optional[Any] = None


def write_history(f, history, delim=";"):
    keys = set()
    for h in history:
        keys.update(h.keys())
    f.write((delim.join([str(k) for k in keys]) + "\n").encode("utf-8"))
    for h in history:
        line = delim.join([str(h.get(k, "")) for k in keys])
        f.write((line + "\n").encode("utf-8"))


def save_run(fname, data: RunData):
    with zipfile.ZipFile(fname, "w", zipfile.ZIP_BZIP2) as zf:
        if data.config is not None:
            with zf.open("config.yml", "w", force_zip64=True) as f:
                if hasattr(data.config, "save"):
                    data.config.save(f)
                else:
                    ruamel.yaml.YAML().dump(data, f)

        if data.history is not None:
            with zf.open("history.csv", "w", force_zip64=True) as f:
                write_history(f, data.history)
        if data.summary is not None:
            with zf.open("summary.csv", "w", force_zip64=True) as f:
                lines = [f"{k};{v}" for k, v in data.summary.items()]
                f.write("\n".join(lines).encode("utf-8"))
        for field in fields(RunData):
            key = field.name
            value = getattr(data, key)
            if (value is not None) and (key not in ["config", "history", "summary"]):
                with zf.open(key + ".pkl", "w", force_zip64=True) as f:
                    pickle.dump(value, f)


def load_run(fname, parse_config=True, parse_csv=False, load_pkl=True):
    data = RunData()
    with zipfile.ZipFile(fname, "r") as zip:
        fnames = zip.namelist()
        if "config.yml" in fnames:
            with zip.open("config.yml", "r") as f:
                if parse_config:
                    data.config = Configuration.load(f)
                else:
                    data.config = ruamel.yaml.YAML().load(f)
        for field in fnames:
            key, extension = os.path.splitext(field)
            if (extension == ".pkl") and load_pkl:
                with zip.open(field, "r") as f:
                    setattr(data, key, pickle.load(f))
            elif (extension == ".csv") and parse_csv:
                import pandas as pd

                with zip.open(field, "r") as f:
                    setattr(data, key, pd.read_csv(f, sep=";"))
    return data


def load_data_for_reuse(config: Configuration, raw_config):
    logger = logging.getLogger("dpe")
    (params_to_reuse, fixed_params, mcmc_state, opt_state, clipping_state, phisnet_params, map_fixed_params) = (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )

    if config.reuse.path is not None:
        # Load the old data; only parse the config if we want to reuse it, otherwise ignore
        reuse_data = load_run(config.reuse.path, parse_config=config.reuse.reuse_config)

        if config.reuse.reuse_config:
            _, config = config.update_configdict_and_validate(reuse_data.config.dict(), build_flattend_dict(raw_config))

    if config.reuse.continue_n_epochs:
        if reuse_data.metadata:
            config.optimization.n_epochs_prev = reuse_data.metadata.get("n_epochs", 0)
    if config.reuse.skip_burn_in:
        config.optimization.mcmc.n_burn_in = 0
    if config.reuse.skip_pretraining:
        config.pre_training = None
    if config.reuse.reuse_opt_state:
        opt_state = reuse_data.opt_state
    if config.reuse.reuse_clipping_state:
        clipping_state = reuse_data.clipping_state
    if config.reuse.reuse_mcmc_state:
        mcmc_state = reuse_data.mcmc_state
        if config.reuse.randomize_mcmc_rng:
            new_seed = np.random.randint(2**31)
            logger.debug(f"Selecting new seed for MCMC rng_state: {new_seed}")
            mcmc_state.rng_state = jax.random.PRNGKey(new_seed)
    if config.reuse.reuse_fixed_params:
        fixed_params = without_cache(reuse_data.fixed_params)  # fixed params of initial geometry 0
    if config.reuse.reuse_trainable_params:
        params_to_reuse = reuse_data.params
        if config.reuse.reuse_modules is not None:
            # Only reuse selected modules and split off all unused ones
            params_to_reuse, _ = split_params(params_to_reuse, config.reuse.reuse_modules)
        logger.debug(f"Reusing {hk.data_structures.tree_size(params_to_reuse)} weights")
    if config.reuse.reuse_ema_trainable_params:
        params_to_reuse = reuse_data.ema_params
        if config.reuse.reuse_modules is not None:
            # Only reuse selected modules and split off all unused ones
            params_to_reuse, _ = split_params(params_to_reuse, config.reuse.reuse_modules)
        logger.debug(f"Reusing {hk.data_structures.tree_size(params_to_reuse)} ema weights")
    if config.reuse.path_phisnet is not None:
        phisnet_data = load_run(config.reuse.path_phisnet, parse_config=False)
        phisnet_params = phisnet_data.params
        phisnet_params = jax.tree_util.tree_map(jax.numpy.array, phisnet_params)
        logger.debug(
            f"Reusing {hk.data_structures.tree_size(phisnet_params)} PhisNet weights and reusing phisnet config from phisnet checkpoint"
        )
        phisnet_model_config = PhisNetModelConfig.parse_obj(phisnet_data.config["model"])
        config.model.orbitals.transferable_atomic_orbitals.phisnet_model = phisnet_model_config

    if config.optimization.shared_optimization and (
        config.reuse.reuse_fixed_params or config.reuse.reuse_mcmc_state or config.reuse.reuse_clipping_state
    ):
        logger.debug("Reusing for shared opt. fixed params or mcmc state or clipping state.")

        phys_configs = build_physical_configs_from_changes(raw_config["physical"])
        map_fixed_params = {pc.comment.split("_")[0]: (None, None, None) for pc in phys_configs}
        for i in range(len(phys_configs)):
            path = (
                "/".join(config.reuse.path.split("/")[:-2]) + "/" + f"{i:04}" + "/" + config.reuse.path.split("/")[-1]
            )
            g = load_run(path, parse_config=True)
            hash = g.config.physical.comment.split("_")[0]
            map_fixed_params[hash] = (
                without_cache(g.fixed_params) if config.reuse.reuse_fixed_params else None,
                g.mcmc_state if config.reuse.reuse_mcmc_state else None,
                g.clipping_state if config.reuse.reuse_clipping_state else None,
            )

    return (
        config,
        params_to_reuse,
        fixed_params,
        mcmc_state,
        opt_state,
        clipping_state,
        phisnet_params,
        map_fixed_params,
    )


def is_checkpoint_required(n_epoch: int, checkpoint_config: CheckpointConfig):
    if n_epoch in checkpoint_config.additional_n_epochs:
        return True
    can_save = (n_epoch > 0) or checkpoint_config.keep_epoch_0
    if can_save and (n_epoch % checkpoint_config.replace_every_n_epochs == 0):
        return True
    if can_save and (n_epoch % checkpoint_config.keep_every_n_epochs == 0):
        return True
    return False


def delete_obsolete_checkpoints(n_epoch, chkpt_config: CheckpointConfig, prefix="", directory="."):
    fnames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for fname in fnames:
        match = re.match(prefix + r"chkpt(\d+).zip", fname)
        if not match:
            continue
        n = int(match.group(1))
        if (n < n_epoch) and (n % chkpt_config.keep_every_n_epochs) and (n not in chkpt_config.additional_n_epochs):
            logging.getLogger("dpe").debug(f"Deleting old checkpoint: {fname}")
            try:
                os.remove(os.path.join(directory, fname))
            except FileNotFoundError:
                pass  # Checkpoint has already been deleted in some other way
