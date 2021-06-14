import numpy as np
import tensorflow as tf

from deeperwin.utilities.logging import getLogger
from deeperwin.main import DTYPE

logger = getLogger()


class AdaptiveLRInverse(tf.keras.callbacks.Callback):
    r"""
    Default learning rate scheduler, with the learning rate being roughly ~1/t.

    An inverse learning rate is used and updated for each epoch e by :math:`lr(e) = \frac{lr_0}{1 + \frac{e}{d}}` with decay rate d.
    Per default the base learning rate is set to 1.5e-3 and the decay is set to 1000.
    """
    def __init__(self, optimizer, lr_config, base_lr):
        super().__init__()
        self.optimizer = optimizer

        self.base_lr = base_lr
        self.decay_time = lr_config.decay_time

    # docstr-coverage:inherited
    def on_epoch_begin(self, epoch, logs=None):
        new_lr = self.base_lr / (1 + epoch / self.decay_time)
        logger.debug(f"Epoch {epoch}: Setting new LR: {new_lr:.3e}")
        tf.keras.backend.set_value(self.optimizer.lr, new_lr)

class AdaptiveLRExponential(tf.keras.callbacks.Callback):
    """
    Exponential learning rate scheduler.

    In the first phase the learning rate is increased linearly by a small increment each epoch.
    After this short warm-up period we start to decrease the learning with a exponential decay.
    """
    def __init__(self, optimizer, lr_config, base_lr, n_epochs):
        super().__init__()
        self.optimizer = optimizer
        self.fac_start = lr_config.fac_start
        self.fac_end = lr_config.fac_end
        self.start_lr = base_lr * self.fac_start
        self.end_lr = base_lr * self.fac_end
        self.epochs_warmup = lr_config.epochs_warmup
        n_epochs_exp = n_epochs - self.epochs_warmup
        self.decay_param = np.log(self.fac_start / self.fac_end) / (n_epochs_exp - 1)
        if self.epochs_warmup > 0:
            self.warmup_increment = (self.start_lr - self.end_lr) / self.epochs_warmup

    # docstr-coverage:inherited
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.epochs_warmup:
            new_lr = self.end_lr + epoch*self.warmup_increment
        else:
            new_lr = self.start_lr * np.exp(-self.decay_param * (epoch - self.epochs_warmup))
        logger.debug(f"Epoch {epoch}: Setting new LR: {new_lr:.3e}")
        tf.keras.backend.set_value(self.optimizer.lr, new_lr)

def get_adaptive_lr_callback(optimizer, adaptive_lr_config, base_lr, n_epochs):
    """
    Return a learning rate callback object, based on the config object (in particular the field adaptive_lr_config.name).

    Args:
        optimizer (tf.keras.optimizers.Optimizer): Optimizer object which will be manipulated in the callback
        adaptive_lr_config (AdaptiveLRConfig): Settings about the adaptive learning rate, e.g. stored in config.optimization.adaptiveLR
        base_lr (float): Initial learning rate
        n_epochs (int): Total number of training epochs (relevant for schedulers that aim for a set final learning rate)

    Returns:
        (tf.keras.callbacks.Callback): Callback which adjusts the learning rate
    """
    if adaptive_lr_config.name == 'exponential':
        return AdaptiveLRExponential(optimizer, adaptive_lr_config, base_lr, n_epochs)
    elif adaptive_lr_config.name == 'inverse':
        return AdaptiveLRInverse(optimizer, adaptive_lr_config, base_lr)
    else:
        raise ValueError(f"Unknown adaptiveLR.name: {adaptive_lr_config.name}")

class CreateGraphCallback(tf.keras.callbacks.Callback):
    """
    Callback to profile the tensorflow code.
    """
    def __init__(self, use_profiler, output_path, tb_writer, epoch_nr=1):
        super().__init__()
        self.tb_writer = tb_writer
        self.use_profiler = use_profiler
        self.epoch_nr = epoch_nr
        self.output_path = output_path

    # docstr-coverage:inherited
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.epoch_nr:
            tf.summary.trace_on(graph=True, profiler=self.use_profiler)

    # docstr-coverage:inherited
    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.epoch_nr:
            with self.tb_writer.as_default():
                tf.summary.trace_export(
                    name="graph",
                    step=0,
                    profiler_outdir=self.output_path)

class UpdateClippingCallback(tf.keras.callbacks.Callback):
    """
    During training, the local energies are clipped to control outliers. After every epoch the clipping center and the range are updated
    by using the local energies of the previous iteration.
    """
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    # docstr-coverage:inherited
    def on_epoch_end(self, epoch, logs=None):
        if 'local_energies_train' in logs:
            local_energies = logs['local_energies_train']
            local_energies = np.array(local_energies)
            local_energies = local_energies[~np.isnan(local_energies)]
            if len(local_energies) < 2:
                self.logger.warning("Skipping update of energy clipping range, due to too many nan values")
            else:
                mean = np.mean(local_energies)
                clip_range = self.config.optimization.clip_by * np.mean(np.abs(local_energies-mean))
                clip_range = np.clip(clip_range, self.config.optimization.clip_min, self.config.optimization.clip_max)
                self.model.energy_clipping_center.assign(mean)
                self.model.energy_clipping_range.assign(clip_range)
                self.logger.debug(f"New energy clipping window: {mean:.3f} +- {clip_range:.3f} Ha")

class LogParametersCallback(tf.keras.callbacks.Callback):
    """
    Adds data to a history dictionary and logs information to a tensorboard.
    """
    def __init__(self, wf):
        self.config = wf.config
        self.logger = logger
        self.wf = wf

    # docstr-coverage:inherited
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            self.logger.info(f"Total number of trainable parameters: {self.model.get_number_of_params()}")
        self._addMetaDataToHistory(logs)
        self._write_logs_to_tensorboard(epoch, logs)

    def _addMetaDataToHistory(self, logs):
        # Log model-specific parameters (e.g. internal weights, etc.)
        for k, v in self.model.get_model_params_for_logging().items():
            self.wf.history_add_val(k, v)

    def _write_logs_to_tensorboard(self, epoch, logs=None):
        if self.config.output.tb_path is not None:
            with self.wf.tb_writer.as_default():
                # write history to tensorboard
                for key in logs:
                    if (("train" in key) or ("valid" in key)) and (len(logs[key].shape) == 0):
                        tf.summary.scalar(key, logs[key], step=epoch)
                tf.summary.scalar("lr", self.wf.optimizer._decayed_lr(DTYPE), step=epoch)
                tf.summary.scalar("batch_size", self.model.batch_size, step=epoch)
