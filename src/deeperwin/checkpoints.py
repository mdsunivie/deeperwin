import logging
import os
import pickle
import re
import pandas as pd
from deeperwin.configuration import Configuration, build_flattend_dict, CheckpointConfig
from deeperwin.mcmc import MCMCState
import jax
import numpy as np
import haiku as hk
import zipfile
from dataclasses import dataclass, fields
from typing import Optional, Any, List
from deeperwin.utils import split_params


@dataclass
class RunData:
    config: Optional[Configuration] = None
    history: Optional[List[dict]] = None
    summary: Optional[dict] = None
    metadata: Optional[dict] = None
    params: Optional[dict] = None
    fixed_params: Optional[dict] = None
    opt_state: Optional[Any] = None
    mcmc_state: Optional[MCMCState] = None
    clipping_state: Optional[Any] = None


def save_run(fname, data: RunData):
    with zipfile.ZipFile(fname, "w", zipfile.ZIP_BZIP2) as zf:
        if data.config is not None:
            with zf.open("config.yml", "w") as f:
                data.config.save(f)
        if data.history is not None:
            df = pd.DataFrame(data.history)
            with zf.open("history.csv", "w") as f:
                df.to_csv(f, sep=";", index=False)
        if data.summary is not None:
            with zf.open("summary.csv", "w") as f:
                lines = [f"{k};{v}" for k,v in data.summary.items()]
                f.write("\n".join(lines).encode("utf-8"))
        for field in fields(RunData):
            key = field.name
            value = getattr(data, key)
            if (value is not None) and (key not in ['config', 'history', 'summary']):
                with zf.open(key+".pkl", "w") as f:
                    pickle.dump(value, f)

def load_run(fname):
    data = RunData()
    with zipfile.ZipFile(fname, "r") as zip:
        fnames = zip.namelist()
        if "config.yml" in fnames:
            with zip.open("config.yml", "r") as f:
                data.config = Configuration.load(f)
        for field in fnames:
            key, extension = os.path.splitext(field)
            if extension == '.pkl':
                with zip.open(field, "r") as f:
                    setattr(data, key, pickle.load(f))
    return data


def load_data_for_reuse(config: Configuration, raw_config):
    logger = logging.getLogger("dpe")
    params_to_reuse, fixed_params, mcmc_state, opt_state, clipping_state = None, None, None, None, None
    reuse_data = load_run(config.reuse.path)

    if config.reuse.reuse_config:
        _, config = config.update_configdict_and_validate(config.dict(), build_flattend_dict(raw_config))
    if config.reuse.continue_n_epochs:
        if reuse_data.metadata:
            config.optimization.n_epochs_prev = reuse_data.metadata.get('n_epochs', 0)
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
            new_seed = np.random.randint(2**16)
            logger.debug(f"Selecting new seed for MCMC rng_state: {new_seed}")
            mcmc_state.rng_state = jax.random.PRNGKey(new_seed)
    if config.reuse.reuse_fixed_params:
        fixed_params = reuse_data.fixed_params
    if config.reuse.reuse_trainable_params:
        params_to_reuse = reuse_data.params
        if config.reuse.reuse_modules is not None:
            # Only reuse selected modules and split off all unused ones
            params_to_reuse, _ = split_params(params_to_reuse, config.reuse.reuse_modules)
        logger.debug(f"Reusing {hk.data_structures.tree_size(params_to_reuse)} weights")

    return config, params_to_reuse, fixed_params, mcmc_state, opt_state, clipping_state


def is_checkpoint_required(n_epoch, checkpoint_config: CheckpointConfig):
    if n_epoch in checkpoint_config.additional_n_epochs:
        return True
    if (n_epoch > 0) and (n_epoch % checkpoint_config.replace_every_n_epochs == 0):
        return True
    if (n_epoch > 0) and (n_epoch % checkpoint_config.keep_every_n_epochs == 0):
        return True
    return False


def delete_obsolete_checkpoints(n_epoch, chkpt_config: CheckpointConfig):
    fnames = [f for f in os.listdir() if os.path.isfile(f)]
    for fname in fnames:
        match = re.match(r"chkpt(\d+).zip", fname)
        if not match:
            continue
        n = int(match.group(1))
        if (n < n_epoch) and (n % chkpt_config.keep_every_n_epochs) and (n not in chkpt_config.additional_n_epochs):
            logging.getLogger("dpe").debug(f"Deleting old checkpoint: {fname}")
            os.remove(fname)
