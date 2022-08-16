"""
Logging of metrics and weights to local disk and web services.
"""

import logging
import time
import os.path
import sys
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Literal

import jax
import numpy as np
import wandb

from deeperwin.configuration import LoggingConfig, BasicLoggerConfig, LoggerBaseConfig, PickleLoggerConfig, \
    WandBConfig, Configuration
from deeperwin.checkpoints import save_run, RunData


def build_dpe_root_logger(config: BasicLoggerConfig):
    # Step 1: Set up root logger, logging everything to console
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handlers = []
    if config and config.log_to_stdout:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        handlers.append(handler)
    if config and config.fname:
        handler = logging.FileHandler(config.fname, mode='w')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        handlers.append(handler)
    logging.basicConfig(level=config.log_level, handlers=handlers, force=True)

    # Step 2: Set up dpe-logger: Currently no handlers, but will also automatically log into root
    return logging.getLogger("dpe")

class DataLogger(ABC):
    """
    Abstract class representing a generic data logger to store configs and metrics during wavefunction optimization.

    Specific implementations include logging to a log-file, logging a full history to a binary file,
    and logging to web-services such as Weights & Biases.
    """

    def __init__(self, config: LoggerBaseConfig, name, save_path='.'):
        self.logger_config = config
        self.name = name
        self.save_path = save_path

    def _should_skip_epoch(self, epoch):
        if epoch is None:
            return False
        else:
            return (epoch % (self.logger_config.n_skip_epochs + 1)) != 0

    def log_param(self, key, value):
        d = {}
        d[key] = value
        self.log_params(d)

    def log_metric(self, key, value, epoch=None, metric_type: Literal["", "opt", "eval", "pre"] = "", force_log=False):
        d = {}
        d[key] = value
        self.log_metrics(d, epoch, metric_type, force_log)

    def log_tags(self, tags: List[str]):
        self.log_param("tags", tags)

    def log_config(self, config: Configuration):
        self.log_params(config.as_flattened_dict())

    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], epoch=None, metric_type: Literal["", "opt", "eval", "pre"] = "", force_log=False):
        pass

    def log_checkpoint(self, n_epoch, params=None, fixed_params=None, mcmc_state=None, opt_state=None, clipping_state=None):
        pass

    def on_run_end(self):
        pass

    def on_run_begin(self):
        pass

class LoggerCollection(DataLogger):
    """
    List of multiple loggers that replicates logs across all sub-loggers.
    """

    def __init__(self, config: LoggingConfig, name, save_path=".", prefix=''):
        super(LoggerCollection, self).__init__(config, name, save_path)
        self.loggers: List[DataLogger] = self.build_loggers(config, name, save_path, prefix)

    @staticmethod
    def build_loggers(config: LoggingConfig, name, save_path=".", prefix=''):
        loggers = []
        if config.basic is not None:
            loggers.append(BasicLogger(config.basic, name, save_path, prefix))
        if config.pickle is not None:
            loggers.append(PickleLogger(config.pickle, name, save_path, config.log_opt_state))
        if config.wandb is not None:
            loggers.append(WandBLogger(config.wandb, name, save_path, prefix))
        return loggers

    def log_checkpoint(self, n_epoch, params=None, fixed_params=None, mcmc_state=None, opt_state=None, clipping_state=None):
        for l in self.loggers:
            l.log_checkpoint(n_epoch, params, fixed_params, mcmc_state, opt_state, clipping_state)

    def on_run_begin(self):
        for l in self.loggers:
            l.on_run_begin()

    def on_run_end(self):
        for l in self.loggers:
            l.on_run_end()

    def log_config(self, config: Configuration):
        for l in self.loggers:
            l.log_config(config)

    def log_params(self, params: Dict[str, Any]):
        for l in self.loggers:
            l.log_params(params)

    def log_metrics(self, metrics: Dict[str, Any], epoch=None, metric_type: Literal["", "opt", "eval", "pre"] = "", force_log=False):
        for l in self.loggers:
            l.log_metrics(metrics, epoch, metric_type, force_log)

    def log_tags(self, tags: List[str]):
        for l in self.loggers:
            l.log_tags(tags)


class WandBLogger(DataLogger):
    def __init__(self, config: WandBConfig, name, save_path='.', prefix=''):
        super().__init__(config, name, save_path)
        self.prefix = (prefix + "_") if prefix else ""
        self.blacklist = tuple(config.blacklist)

    @staticmethod
    def _convert_metric_datatype(x):
        if x is None:
            return x
        elif isinstance(x, (float, int)):
            return x
        elif hasattr(x, 'shape'):
            n_elements = int(np.prod(x.shape))
            if n_elements > 1:
                return np.array(x)
            else:
                return float(x)
        else:
            # logging.warning(f"Cannot log data-type using WandB: {type(x)}")
            return float(np.mean(x))

    def on_run_begin(self):
        if wandb.run is None:
            if self.logger_config.id is not None and self.logger_config.use_id:
                wandb.init(dir=self.save_path, project=self.logger_config.project, name=self.name,
                           entity=self.logger_config.entity, tags=[], resume='must', id=self.logger_config.id,
                           allow_val_change=True)
            else:
                wandb.init(dir=self.save_path, project=self.logger_config.project, name=self.name, entity=self.logger_config.entity, tags=[])


    def on_run_end(self):
        if wandb.run is not None:
            wandb.run.finish()

    def log_params(self, params):
        wandb.run.config.update(params, allow_val_change=True if self.logger_config.id is not None else False)

    def log_tags(self, tags: List[str]):
        wandb.run.tags = wandb.run.tags + tuple(tags)

    def _log_metrics_async(self, metrics: Dict[str, Any]):
        wandb.run.log(metrics)

    def log_metrics(self, metrics, epoch=None, metric_type: str = "", force_log=False):
        if (not force_log) and self._should_skip_epoch(epoch):
            return
        metrics_prefixed = {self.prefix + k: v for k,v in metrics.items() if not k.endswith(self.blacklist)}
        for key in metrics_prefixed:
            metrics_prefixed[key] = self._convert_metric_datatype(metrics_prefixed[key])
        if epoch is None:
            for k, v in metrics_prefixed.items():
                wandb.run.summary[k] = v
        else:
            epoch_key = f'{metric_type}_epoch' if metric_type else 'epoch'
            metrics_prefixed[epoch_key] = epoch
            wandb.run.log(metrics_prefixed)


class BasicLogger(DataLogger):
    """
    Logger using python built-ing logging module.
    """

    def __init__(self, config: BasicLoggerConfig, name="dpe", save_path=".", prefix=""):
        super().__init__(config, name, save_path)
        formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
        fh = logging.FileHandler(os.path.join(save_path, config.fname), mode='w')
        fh.setFormatter(formatter)
        fh.setLevel(config.log_level)
        logger_name = "dpe" if len(prefix) == 0 else f"dpe.{prefix}"
        self.logger = logging.getLogger(logger_name)
        self.logger.handlers.append(fh)

    def log_metrics(self, metrics: Dict[str, Any], epoch=None, metric_type: Literal["", "opt", "eval", "pre"] = "", force_log=False):
        if (not force_log) and self._should_skip_epoch(epoch):
            return
        msg = "; ".join(f"{key}={value}" for key, value in metrics.items())
        if epoch is not None:
            msg = f"{metric_type} Epoch {epoch:>5d}: " + msg
        self.logger.info(msg)

    def log_params(self, params: Dict[str, Any]):
        msg = "; ".join(f"{key}={value}" for key, value in params.items())
        self.logger.info(msg)

def prepare_checkpoint_directory(epoch, run_path=".", pretraining_checkpoint=False):
    chkpt_name = f"chkpt{epoch:06d}" if not pretraining_checkpoint else f"pretrain_chkpt{epoch:06d}"
    chkpt_dir = os.path.join(run_path, chkpt_name)
    if os.path.exists(chkpt_dir):
        logging.warning(f"Directory {chkpt_dir} already exists. Results might be overwritten.")
    else:
        os.mkdir(chkpt_dir)
    return chkpt_dir

class PickleLogger(DataLogger):
    """
    Logger that stores full history in memory and dumps it as a binary file on exit.
    """

    def __init__(self, config: PickleLoggerConfig, name, save_path=".", log_opt_state=True):
        super().__init__(config, name, save_path)
        self.history = []
        self.summary = dict()
        self.config = None
        self.meta_data = dict()
        self.weights = dict()
        self.log_opt_state = log_opt_state

    def log_config(self, config: Configuration):
        self.config = config

    def log_metrics(self, metrics: Dict[str, Any], epoch=None, metric_type: Literal["", "opt", "eval", "pre"] = "", force_log=False):
        if (not force_log) and self._should_skip_epoch(epoch):
            return
        if epoch is None:
            if metric_type:
                metrics = {metric_type + "_" + k:v for k,v in metrics.items()}
            self.summary.update(metrics)
        else:
            if metric_type:
                epoch_key = f"{metric_type}_epoch"
            else:
                epoch_key = "epoch"
            self.history.append({epoch_key: epoch, **metrics})


    def log_params(self, params: Dict[str, Any]):
        self.meta_data.update(params)

    def log_checkpoint(self, n_epoch, params=None, fixed_params=None, mcmc_state=None, opt_state=None, clipping_state=None):
        data = RunData(config=self.config,
                       metadata=dict(n_epochs=n_epoch, **self.meta_data),
                       history=self.history,
                       summary=self.summary,
                       params=jax.tree_util.tree_map(np.array, params),
                       fixed_params=fixed_params,
                       opt_state=opt_state,
                       mcmc_state=mcmc_state,
                       clipping_state=clipping_state)
        if n_epoch is None:
            fname = os.path.join(self.save_path, f"chkpt.zip")
        else:
            fname = os.path.join(self.save_path, f"chkpt{n_epoch:06d}.zip")
        save_run(fname, data)

class WavefunctionLogger:
    def __init__(self, loggers: LoggerCollection, prefix = "", n_step=0, smoothing=0.05):
        self.loggers = loggers
        self.n_step = n_step
        self.prefix = prefix
        self.smoothing = smoothing
        self.history = {}
        self._time = time.time()
        self._mcmc_state_old = None

    def smooth(self, key, value):
        if key not in self.history:
            self.history[key] = []
        self.history[key].append(value)
        n_averaging = int(self.smoothing * len(self.history[key]))
        samples_for_averaging = self.history[key][-n_averaging:]
        if len(samples_for_averaging) > 0:
            return np.nanmean(samples_for_averaging, axis=0)

    def log_step(self, metrics, E_ref=None, mcmc_state:'MCMCState'=None, opt_stats=None,
                 extra_metrics=None):
        if self.loggers is None:
            return

        # metrics = dict()
        # if 'E_mean_unclipped' in aux_stats:
        #     metrics["E_mean"] = 'E_mean_unclipped'
        # if 'E_mean_clipped' in aux_stats:
        #     metrics['E_mean']

        # metrics["E_std"] = np.nanstd(E_loc_unclipped)

        # if E_loc_clipped is not None:
        #     metrics["E_mean_clipped"] = np.nanmean(E_loc_clipped)
        #     metrics["E_std_clipped"] = np.nanstd(E_loc_clipped)

        if E_ref is not None:
            metrics["error_E_mean"] = (metrics["E_mean"] - E_ref) * 1e3

        if mcmc_state is not None:
            metrics["mcmc_stepsize"] = float(mcmc_state.stepsize)
            metrics["mcmc_acc_rate"] = float(mcmc_state.acc_rate)
            metrics["mcmc_max_age"] = np.max(mcmc_state.walker_age)
            if self._mcmc_state_old:
                delta_r = np.linalg.norm(mcmc_state.r - self._mcmc_state_old.r, axis=-1)
                metrics["mcmc_delta_r_mean"] = np.mean(delta_r)
                metrics["mcmc_delta_r_median"] = np.median(delta_r)
            self._mcmc_state_old = mcmc_state

        if opt_stats is not None:
            for key in ['param_norm', 'grad_norm', 'precon_grad_norm', 'norm_constraint_factor']:
                if key in opt_stats:
                    metrics[key] = opt_stats[key]

        for key in ["E_mean", "error_E_mean", "forces"]:
            if key not in metrics:
                continue
            smoothed = self.smooth(key, metrics[key])
            if smoothed is not None:
                metrics[f"{key}_smooth"] = smoothed

        t = time.time()
        metrics["t_epoch"] = t - self._time
        self._time = t
        metrics = {f"{self.prefix}_{k}": v for k, v in metrics.items()}
        if extra_metrics:
            metrics.update(extra_metrics)

        self.loggers.log_metrics(metrics, self.n_step, self.prefix)
        self.n_step += 1

    def log_summary(self, E_ref=None, epoch_nr=None):
        metrics = dict()
        if "E_mean" in self.history:
            energies = np.array(self.history["E_mean"], dtype=float)
            metrics["E_mean"] = np.nanmean(energies)
            metrics["E_mean_sigma"] = np.nanstd(energies) / np.sqrt(len(energies))
            if E_ref is not None:
                metrics["error_eval"] = 1e3 * (metrics["E_mean"] - E_ref)
                metrics["sigma_error_eval"] = 1e3 * metrics["E_mean_sigma"]
                metrics["error_plus_2_stdev"] = metrics["error_eval"] + 2 * metrics["sigma_error_eval"]
        if "forces_mean" in self.history:
            metrics["forces_mean"] = np.nanmean(self.history["forces_mean"], axis=0)
        if len(metrics) > 0:
            self.loggers.log_metrics(metrics, epoch_nr, "opt", force_log=True)



