"""
Logging of metrics and weights to local disk and web services.
"""

import logging
import os.path
import sys
import copy
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Literal

import numpy as np
import wandb

from deeperwin.configuration import LoggingConfig, BasicLoggerConfig, LoggerBaseConfig, PickleLoggerConfig, \
    WandBConfig, Configuration
from deeperwin.utils import save_to_file


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

    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], epoch=None, metric_type: Literal["", "opt", "eval", "pre"] = "", force_log=False):
        pass

    def log_checkpoint(self, chkpt_dir, n_epoch, pretraining=False):
        pass

    def on_run_end(self):
        pass

    def on_run_begin(self):
        pass

    def log_weights(self, weight_dict):
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

    def log_checkpoint(self, chkpt_dir, n_epoch, pretraining=False):
        for l in self.loggers:
            l.log_checkpoint(chkpt_dir, n_epoch, pretraining=pretraining)

    def on_run_begin(self):
        for l in self.loggers:
            l.on_run_begin()

    def on_run_end(self):
        for l in self.loggers:
            l.on_run_end()

    def log_params(self, params: Dict[str, Any]):
        for l in self.loggers:
            l.log_params(params)

    def log_metrics(self, metrics: Dict[str, Any], epoch=None, metric_type: Literal["", "opt", "eval", "pre"] = "", force_log=False):
        for l in self.loggers:
            l.log_metrics(metrics, epoch, metric_type, force_log)

    def log_tags(self, tags: List[str]):
        for l in self.loggers:
            l.log_tags(tags)

    def log_weights(self, weight_dict):
        for l in self.loggers:
            l.log_weights(weight_dict)


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
        metrics_prefixed = {}
        for k, v in metrics.items():
            if not k.endswith(self.blacklist):
                metrics_prefixed[self.prefix + k] = v
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
        self.metrics = dict(opt_epochs=[], eval_epochs=[], pre_epochs=[])
        self.config = dict()
        self.weights = dict()
        self.log_opt_state = log_opt_state

    def log_metrics(self, metrics: Dict[str, Any], epoch=None, metric_type: Literal["", "opt", "eval", "pre"] = "", force_log=False):
        if (not force_log) and self._should_skip_epoch(epoch):
            return
        for key, value in metrics.items():
            if (key not in self.metrics) and (epoch is not None):
                self.metrics[key] = []
            if epoch is None:
                self.metrics[key] = value
            else:
                self.metrics[key].append(value)
            if (epoch is not None) and metric_type:
                n_epoch_key = f"{metric_type}_epochs"

                if n_epoch_key not in self.metrics:
                    self.metrics[n_epoch_key] = []
                if (len(self.metrics[n_epoch_key]) == 0) or (self.metrics[n_epoch_key][-1] != epoch):
                    self.metrics[n_epoch_key].append(epoch)

    def log_params(self, params: Dict[str, Any]):
        self.config.update(params)

    def log_weights(self, weight_dict):
        # For some unclear reason this deepcopy has caused the code to get stuck for hours without progress.
        # Not copying should be fine since we only store the weights once, but this could in principle be problematic when
        # external code modifies stored weights in-place
        # logging.debug(f"Pickle logger: Creating deepcopy of weight dict; keys = {weight_dict.keys()}")
        # weight_dict = copy.deepcopy(weight_dict)
        logging.debug("Pickle logger: Updating self.weights")
        self.weights.update(weight_dict)
        if ('opt' in self.weights) and (not self.log_opt_state):
            del self.weights['opt']

    def log_checkpoint(self, chkpt_dir, n_epoch=None, pretraining=False):
        config_dict = copy.deepcopy(self.config)
        if n_epoch is not None:
            chkpt_name = f"chkpt{n_epoch:06d}"
            for k in ['code_version', 'n_params', 'tags']:
                config_dict.pop(k, None)
            config_chkpt = Configuration.from_flattened_dict(config_dict)
            config_chkpt.optimization.n_epochs = n_epoch if not pretraining else 0
            config_chkpt.evaluation.n_epochs = 0
            config_chkpt.experiment_name = f"{config_chkpt.experiment_name}_{chkpt_name}"
            config_chkpt.save(os.path.join(chkpt_dir, "full_config.yml"))
            config_dict = config_chkpt.get_as_flattened_dict()
        data = dict(metrics=self.metrics, config=config_dict, weights=self.weights)
        save_to_file(os.path.join(chkpt_dir, self.logger_config.fname), **data)

    def on_run_end(self):
        self.log_checkpoint(self.save_path)

