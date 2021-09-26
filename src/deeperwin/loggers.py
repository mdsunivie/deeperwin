"""
Logging of metrics and weights to local disk and web services.
"""

import logging
import os.path
import sys
import threading
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Literal

import numpy as np
import wandb

from deeperwin.configuration import LoggingConfig, BasicLoggerConfig, LoggerBaseConfig, PickleLoggerConfig, \
    WandBConfig
from deeperwin.dispatch import save_to_file


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

    def log_metric(self, key, value, epoch=None, metric_type: Literal["", "opt", "eval"] = ""):
        d = {}
        d[key] = value
        self.log_metrics(d, epoch, metric_type)

    def log_tags(self, tags: List[str]):
        self.log_param("tags", tags)

    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], epoch=None, metric_type: Literal["", "opt", "eval"] = ""):
        pass

    def log_checkpoint(self, chkpt_dir):
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
            loggers.append(BasicLogger(config.basic, name, save_path))
        if config.pickle is not None:
            loggers.append(PickleLogger(config.pickle, name, save_path))
        if config.wandb is not None:
            loggers.append(WandBLogger(config.wandb, name, save_path, prefix))
        return loggers

    def log_checkpoint(self, chkpt_dir):
        for l in self.loggers:
            l.log_checkpoint(chkpt_dir)

    def on_run_begin(self):
        for l in self.loggers:
            l.on_run_begin()

    def on_run_end(self):
        for l in self.loggers:
            l.on_run_end()

    def log_params(self, params: Dict[str, Any]):
        for l in self.loggers:
            l.log_params(params)

    def log_metrics(self, metrics: Dict[str, Any], epoch=None, metric_type: Literal["", "opt", "eval"] = ""):
        for l in self.loggers:
            l.log_metrics(metrics, epoch, metric_type)

    def log_tags(self, tags: List[str]):
        for l in self.loggers:
            l.log_tags(tags)

    def log_weights(self, weight_dict):
        for l in self.loggers:
            l.log_weights(weight_dict)


class WandBLogger(DataLogger):
    def __init__(self, config: WandBConfig, name, save_path='.', prefix=''):
        super(WandBLogger, self).__init__(config, name, save_path)
        self.run = None
        self.prefix = prefix

    @staticmethod
    def _convert_metric_datatype(x):
        if isinstance(x, (float, int)):
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
            self.run = wandb.init(dir=self.save_path, project=self.logger_config.project, name=self.name,
                                  entity=self.logger_config.entity, tags=[])
        else:
            self.run = wandb.run

    def on_run_end(self):
        if wandb.run is not None:
            self.run.finish()
        else:
            self.run = None

    def log_params(self, params):
        self.run.config.update(params)

    def log_tags(self, tags: List[str]):
        self.run.tags = self.run.tags + tuple(tags)

    def _log_metrics_async(self, metrics: Dict[str, Any]):
        self.run.log(metrics)

    def log_metrics(self, metrics, epoch=None, metric_type: Literal["", "opt", "eval"] = ""):
        metrics_prefix = {}
        for k, v in metrics.items():
            metrics_prefix[self.prefix + k] = v
        if self._should_skip_epoch(epoch):
            return
        for key in metrics_prefix:
            metrics_prefix[key] = self._convert_metric_datatype(metrics_prefix[key])
        if epoch is None:
            for k, v in metrics_prefix.items():
                self.run.summary[k] = v
        else:
            if metric_type == "opt":
                metrics_prefix['opt_epoch'] = epoch
            elif metric_type == 'eval':
                metrics_prefix['eval_epoch'] = epoch
            thread = threading.Thread(target=self._log_metrics_async, args=[metrics_prefix])
            thread.start()


class BasicLogger(DataLogger):
    """
    Logger using python built-ing logging module.
    """

    def __init__(self, config: BasicLoggerConfig, name, save_path="."):
        super(BasicLogger, self).__init__(config, name, save_path)
        handlers = []
        if config.log_to_stdout:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            handlers.append(handler)
        if config.fname is not None:
            handler = logging.FileHandler(os.path.join(save_path, config.fname), mode='w')
            handler.setLevel(logging.DEBUG)
            handlers.append(handler)
        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                            level=config.log_level,
                            handlers=handlers,
                            force=True)

    def log_metrics(self, metrics: Dict[str, Any], epoch=None, metric_type: Literal["", "opt", "eval"] = ""):
        if self._should_skip_epoch(epoch):
            return
        msg = "; ".join(f"{key}={value}" for key, value in metrics.items())
        if epoch is not None:
            msg = f"{metric_type} Epoch {epoch:>5d}: " + msg
        logging.info(msg)

    def log_params(self, params: Dict[str, Any]):
        msg = "; ".join(f"{key}={value}" for key, value in params.items())
        logging.info(msg)


class PickleLogger(DataLogger):
    """
    Logger that stores full history in memory and dumps it as a binary file on exit.
    """

    def __init__(self, config: PickleLoggerConfig, name, save_path="."):
        super(PickleLogger, self).__init__(config, name, save_path)
        self.metrics = dict(opt_epochs=[], eval_epochs=[])
        self.config = dict()
        self.weights = dict()

    def log_metrics(self, metrics: Dict[str, Any], epoch=None, metric_type: Literal["", "opt", "eval"] = ""):
        if self._should_skip_epoch(epoch):
            return
        for key, value in metrics.items():
            if (key not in self.metrics) and (epoch is not None):
                self.metrics[key] = []
            if epoch is None:
                self.metrics[key] = value
            else:
                self.metrics[key].append(value)
            n_epoch_key = "opt_epochs" if metric_type == "opt" else "eval_epochs"
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

    def log_checkpoint(self, chkpt_dir):
        data = dict(metrics=self.metrics, config=self.config, weights=self.weights)
        save_to_file(os.path.join(chkpt_dir, self.logger_config.fname), **data)

    def on_run_end(self):
        self.log_checkpoint(self.save_path)
