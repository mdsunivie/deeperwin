#!/usr/bin/env python3
"""
This module contains the core logic of DeepErwin.

It contains the :class:`~deeperwin.main.WaveFunction` class, which contains methods for optimization of a wavefunction,
as well as evaluation of energies and forces.
This module can be executed directly by passing a configuration file as outlined in the examples. Call it with python main.py -h to see usage.
"""
import argparse
import copy
import json
import os
import importlib
import logging
import shutil
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from shutil import copyfile
from tensorflow.python.keras import backend as K
import tensorboard.plugins.hparams.api
import pickle
if "5.0" not in pickle.compatible_formats:
    import pickle5 as pickle

DTYPE = tf.dtypes.float32
tf.keras.backend.set_floatx(DTYPE.name)

from deeperwin.utilities.logging import getLogger
from deeperwin.utilities.utils import getCodeVersion, TARGET_ENERGIES, apply_module_shortcut_dict, is_in_module, \
    get_hist_range
from deeperwin.utilities import callbacks, erwinConfiguration
from deeperwin.utilities.mcmc import PositionalMetropolisHastings
from deeperwin.utilities.erwinConfiguration import DefaultConfig
from deeperwin.models.base import WFModel

logger = getLogger()
EPSILON = 1e-8

class WaveFunction():
    """
    This class is used as an entry point to the optimization and evaluation of a wavefunction.
    It initializes the Metropolis Hasting algorithm for training, validation and evaluation.
    Handels the optimization and evaluation and as well the training history.
    """

    def __init__(self, config: DefaultConfig, log_to_file=True, history_only=False, **model_kwargs):
        """
        Creates a new WaveFunction instance based on the specified configuration.

        Args:
            config (erwinConfiguration.DefaultConfig): Global config-object. The model will primarily use the fields in config.model
            log_to_file:
            history_only (bool): Used when loading a model for analysis. Only populates the config, but does not actually build the model
            **model_kwargs: Additional arguments passed to the WaveFunction

        Attributes:
            molecule_name (str): Name of the molcule.
            ion_positions (list of coordinates): Ion positions of the molecule in Cartesian coordinates.
            ion_charges (list of int): Ion charges.
            n_electrons (int): Number of electrons.
            n_spinup (int): Number of spinup electrons.
            energy_baseline (float): Used as a reference during optimization. Should be a little bit lower than the actual ground state energy.
            model_name (string): Class name of the model used for representing the log-squared of the wave function.
            model_config (dict): Dictionary containing the configuration of the wave function model.
            model (deeperwin.models.base.WFModel): Model that implements the log-squared of the wave function.
            optimizer (tf.keras.Optimizer): Instance of the optimizer that is used to optimize the wave function.
        """
        self.config = config
        self.config.physical.fill_missing_values_with_defaults()
        self.molecule_name = config.physical.name
        self.n_electrons = config.physical.n_electrons
        self.n_spinup = config.physical.n_spin_up
        self.ion_positions = config.physical.ion_positions
        self.ion_charges = config.physical.ion_charges
        self.ion_positions = tf.Variable(self.ion_positions, trainable=False, dtype=DTYPE, name="ion_position")
        self.ion_charges = tf.Variable(self.ion_charges, trainable=False, dtype=DTYPE, name="ion_charges")

        #self.energy_baseline = tf.Variable(config.physical.energy_baseline, dtype=DTYPE, trainable=False)
        self.model_name = config.model.name
        if log_to_file:
            log_fname = config.output.log_path
        else:
            log_fname = None
        self.logger = getLogger(name=config.output.logger_name, fname=log_fname,
                                file_level=config.output.log_level_file, console_level=config.output.log_level_console)
        self.logger.debug(json.dumps(config.get_as_dict()))

        model_module = importlib.import_module('deeperwin.models.'+config.model.name)
        self.model = getattr(model_module, config.model.name)(config, history_only=history_only, **model_kwargs)

        # optimization, evaluation & mcmc
        self.optimizer = self.get_optimizer()

        self.mh_mcmc = {}
        for key in ["train", "valid", "eval"]:
            self.mh_mcmc[key] = PositionalMetropolisHastings(self.config.integration[key],
                                                             self.config, self.logger)

        self.walkers_mcmc = {"train": None,
                             "eval": None,
                             "valid": None}

        self.history = {"loss_train": [],
                        "loss_valid": [],
                        "mean_energies_train": [],
                        "mean_energies_eval": [],
                        "mean_energies_valid": [],
                        "std_energies_train": [],
                        "std_energies_eval": [],
                        "std_energies_valid": [],
                        }
        self.histograms = {"train": None,
                           "valid": None,
                           "eval": None}
        self.batch_history = {}
        self.callbacks = []
        self.logs = {}
        # evaluation
        self.reset_evaluation()
        self._epoch_last_logging = -1000
        self._observables = []
        self.apply_gradient_callback = self.apply_gradients_default

        if (not history_only) and (config.model.reuse_weights is not None):
            reuse_weights_cfgs = config.model.reuse_weights # type: erwinConfiguration.ReuseWeightsConfig
            if type(config.model.reuse_weights) is not list:
                reuse_weights_cfgs = [reuse_weights_cfgs]

            for reuse_weights_cfg in reuse_weights_cfgs:
                self.reuse_weights(reuse_weights_cfg.reuse_dirs, apply_module_shortcut_dict(reuse_weights_cfg.weights), reuse_weights_cfg.interpolation)

    def get_optimizer(self):
        """
        Initializes the optimizer for training.

        Returns:
            tf.keras.optimizer: Tensorflow Optimizer
        """
        optimizer = tf.keras.optimizers.get(self.config.optimization.optimizer)
        K.set_value(optimizer.lr, self.config.optimization.learning_rate)
        return optimizer

    def get_summary(self, data_getters={}):
        """
        Extract key meta-data of the WaveFunction and return it as a dictionary.

        This function is in particular called when loading runs for post-processing.
        Args:
            data_getters (dict str->function): Dictionary of additional functions that take the wavefunction object and return values that are included in the output dict

        Returns:
            (dict): Dictionary summarizing the WaveFunction, including keys such as name, n_epochs, E_eval, etc.)
        """
        data = dict(name=self.config.physical.name,
                    n_electrons=self.config.physical.n_electrons,
                    lr=self.config.optimization.learning_rate,
                    n_epochs=self.config.optimization.n_epochs,
                    n_walkers=self.config.integration.train.n_walkers,
                    batch_size=self.config.optimization.batch_size,
                    E_eval=self.total_energy,
                    std_E_eval=self.std_err_total_energy)
        data.update(self.model.get_summary())

        if self.config.evaluation.forces.calculate:
            data['forces'] = self.forces
            data['std_forces'] = self.std_err_forces
        if len(self.ion_positions.numpy()) == 2:
            data['bond_length'] = np.linalg.norm(self.ion_positions.numpy()[0] - self.ion_positions.numpy()[1])

        try:
            data['error_eval'] = (data['E_eval'] - TARGET_ENERGIES[data['name']]) * 1000
            data['std_error_eval'] = self.std_err_total_energy * 1000
        except Exception:
            self.logger.warning("Could not calculate evaluation error")
            data['error_eval'] = np.nan

        for key, getter in data_getters.items():
            data[key] = getter(self)
        return data

    def init_run(self):
        """
        Makes sure that the weights of the neural networks are initialized.
        """
        inp = tf.random.normal([1, self.config.physical.n_electrons, 3])
        self.model.log_squared(inp)
        
    def reuse_weights(self, reuse_dirs, module_names, interpolation="nearest"):
        """
        Main function to initialize the weights based on pretrained neural networks.

        Args:
            reuse_dirs (list): Directories to pretrained neural networks for reusing.
            module_names (list): Module names which specify the parts of the network for reusing.
            interpolation (str): Approach to interpolate the weights.
        """
        if type(reuse_dirs) is not list:
            reuse_dirs = [reuse_dirs]
        weights = []
        for reuse_dir in reuse_dirs:
            if os.path.isfile(os.path.join(reuse_dir, 'config.json')):
                wfun = WaveFunction.load(reuse_dir)
                wfun.init_run()
                weights.append([wfun.get_module_weights(module_name) for module_name in module_names])
        if len(weights) == 0:
            raise ValueError("Requested weight-reuse, but could not load any of the specified restart_dirs")
        elif len(weights) == 1:
            reuse_weights = weights[0]
        else:
            logger.warning("Weight interpolation currently not implemented. Only the weights from the first re-use directory will be used.")
            reuse_weights = weights[0]

        self.init_run()  # make sure that variables are initialized
        logger.debug(f"Using pre-trained weights for these modules: " + ", ".join(m for m in module_names))
        for k, module_name in enumerate(module_names):
            self.set_module_weights(module_name, reuse_weights[k])

    def get_module_weights(self, module_name):
        """
        Gets all trainable weights of the neural network for which the name is equal to specific module.

        Args:
            module_name (str): Name of the specific module.

        Returns:
            list: Trainable weights of a neural network which should be used for reinitializing.
        """
        weight_list = []
        for var in super(WFModel, self.model).trainable_variables:
            if is_in_module(var.name, module_name):
                weight_list.append(var.numpy())
        return weight_list

    def set_module_weights(self, module_name, weights):
        """
        Assigns all trainable weights of the neural network weights from pretrained networks.

        Args:
            module_name (str): Name of the module.
            weights (list):  Neural network weights from pretrained networks.
        """
        k = 0 
        for var in super(WFModel, self.model).trainable_variables:
            if is_in_module(var.name, module_name):
                var.assign(weights[k])
                k += 1
        assert k == len(weights)

    def _create_callbacks(self):
        """
        Creates the callbacks for optimization.
        """
        # Learning Rate and Batch Size
        self.callbacks.append(callbacks.get_adaptive_lr_callback(self.optimizer, self.config.optimization.adaptiveLR,
                                                                 self.config.optimization.learning_rate,
                                                                 self.config.optimization.n_epochs))

        # Clipping
        self.callbacks.append(callbacks.UpdateClippingCallback(self.config, self.logger))

        # Logging and debugging
        self.callbacks.append(callbacks.LogParametersCallback(self))
        if self.config.output.create_graph:
            self.callbacks.append(callbacks.CreateGraphCallback(self.config.output.use_profiler,
                                                                self.config.output.tb_path, self.tb_writer))

    def on_train_begin(self, full_reset=False):
        """ Prepares all necessary objects and parameters for optimizing the wave function. Call this function before calling optimize(...)

        Args:
            full_reset (bool): True if the mcmc walker should be reset, when walker from a previous optimization run are used.
        """
        self.logger.info("Starting optimization...")
        self.logger.debug("Initializing MCMC-walkers ...")
        for key in ["train", "valid"]:
            if full_reset or (self.walkers_mcmc[key] is None):
                self.walkers_mcmc[key] = self.init_walkers(self.config.integration[key].n_walkers,
                                                           electron_ion_mapping=self.config.physical.electron_ion_mapping)

            if f"mh_scale_{key}" in self.history.keys():
                self.mh_mcmc[key].init_state(self.model.log_squared, self.walkers_mcmc[key], self.history[f"mh_scale_{key}"][-1])
            else:
                self.mh_mcmc[key].init_state(self.model.log_squared, self.walkers_mcmc[key])

        for key in ["train", "valid"]:
            if self.config.output.compute_histogram:
                self.histograms[key] = list(
                    np.histogram2d([], [], range=get_hist_range(self.ion_positions), bins=self.config.output.histogram_n_bins))
            else:
                self.histograms[key] = None

        # ensure that early stopping callback works
        self.model.stop_training = False

        self.setup_tensorboard()
        self._create_callbacks()
        self.model.batch_size = self.config.optimization.batch_size

        self.reset_evaluation()

        # init callbacks
        for cb in self.callbacks:
            cb.set_model(self.model)

        for key in ['train', 'valid']:
            self.logger.info(f"Starting burn-in {key}...")
            self.walkers_mcmc[key] = self.mh_mcmc[key].step(self.config.integration[key].n_burnin_steps)
            self.logger.debug("Finished burn-in {}: scale={:.4f}, acceptance={:.3f}".format(key,
                                                                                            self.mh_mcmc[key].scale.numpy(),
                                                                                            self.mh_mcmc[key].acceptance_rate.numpy()))


        self.batches_train = tf.data.Dataset.from_tensor_slices(self.walkers_mcmc["train"]).batch(batch_size=self.model.batch_size)
        if self.config.optimization.shuffle:
            self.batches_train = self.batches_train.shuffle(self.walkers_mcmc["train"].shape[0], reshuffle_each_iteration=True)

        self.n_batches_with_nan_gradients = 0

        for cb in self.callbacks:
            cb.on_train_begin(logs=self.logs)

    def _saveDebugInfo(self, local_energies, key):
        """
        Stores in a history dictionary for debugging purpose outlines of mcmc walkers and local energy.

        Args:
            local_energies (tf.Tensor): Clipped local energies.
            key (str): Either train, valid, eval.
        """
        local_energies = local_energies.numpy()
        walkers = self.walkers_mcmc[key].numpy()

        is_nan = np.isnan(local_energies)
        self.history_add_val(f'walkers_nan_{key}', walkers[is_nan, :,:])

        local_energies = local_energies[~is_nan]
        walkers = walkers[~is_nan,:,:]
        indices = np.argsort(local_energies)
        self.history_add_val(f'walkers_max_energy_{key}', walkers[indices[-5:],:,:])
        self.history_add_val(f'energies_max_energy_{key}', local_energies[indices[-5:]])
        self.history_add_val(f'walkers_min_energy_{key}', walkers[indices[:5],:,:])
        self.history_add_val(f'energies_min_energy_{key}', local_energies[indices[:5]])

    def _resample_walkers(self, walkers, n_output):
        """
        Creates the walker positions for evaluation. Either based on the final walker position of the training or randomly sampled.
        Args:
            walkers (tf.Tensor): Walker positions stored in a tf.Tensor.
            n_output (int): Number of walker.

        Returns:
            tf.Tensor: Walker positions.
        """
        if n_output <= walkers.shape[0]:
            return tf.identity(walkers[:n_output])
        else:
            n_rep = n_output // walkers.shape[0]
            n_remainder = n_output % walkers.shape[0]
            return tf.concat([tf.tile(walkers, [n_rep, 1, 1]), walkers[:n_remainder,...]], axis=0)

    # What is with all the inputs of compile_eval?
    def compile_evaluation(self, compute_histogram=True, hist_n_bins=512, hist_range=[[-3, 3], [-3, 3]]):
        """
        Prepares all necessary objects and parameters for evaluating the wave function currently represented by the model. Call this function before calling evaluate(...)
        """
        if self.config.evaluation.reuse_training_walkers and (self.walkers_mcmc["train"] is not None):
            self.walkers_mcmc["eval"] = self._resample_walkers(self.walkers_mcmc["train"],
                                                               self.config.integration.eval.n_walkers)
        else:
            self.walkers_mcmc["eval"] = self.init_walkers(self.config.integration.eval.n_walkers,
                                                          electron_ion_mapping=self.config.physical.electron_ion_mapping)
        self.mh_mcmc["eval"].init_state(self.model.log_squared, self.walkers_mcmc["eval"])

        if self.config.output.compute_histogram:
            self.histograms["eval"] = list(
                np.histogram2d([], [], range=get_hist_range(self.ion_positions), bins=self.config.output.histogram_n_bins))
        else:
            self.histograms["eval"] = None

    def history_add_val(self, key, val):
        """
        Adds the value to the given key to a history dictionary.

        Args:
            key (str): Name of the stored parameter.
            val: Value of the parameter which is mostly a tf.Tensor.
        """
        if key not in self.history:
            self.history[key] = []
        self.history[key].append(val)
        self.logs[key] = val

    def reset_evaluation(self):
        """
        Sets the current walker position to None and deletes old history for the evaluation. To make sure when the wavefunction
        was restarted it starts for the evaluation from zero.
        """
        self.walkers_mcmc["eval"] = None
        self.history["mean_energies_eval"] = []
        self.histograms["eval"] = None

    def _log_local_energies(self, local_energies, key):
        """
        Stores based on the local energy data in a history dictionary, e.g. for the Tensorbaord.

        Args:
            local_energies (tf.Tensor): Local energy.
            key (str): Either train, valid or eval dependent on the stage DeepErwin is currently.

        """
        self.logs['local_energies_' + key] = local_energies
        if 'train' in key:
            self.history_add_val(f"mean_energies_unclipped_{key}", tf.reduce_mean(local_energies).numpy())
            self.history_add_val(f"std_energies_unclipped_{key}", tf.math.reduce_std(local_energies).numpy())
            local_energies = self.model.clip_energies(local_energies)
        local_loss = self.model.local_loss(local_energies)
        loss = tf.reduce_mean(local_loss)
        self.history_add_val(f"mean_energies_{key}", tf.reduce_mean(local_energies).numpy())
        self.history_add_val(f"std_energies_{key}", tf.math.reduce_std(local_energies).numpy())
        self.history_add_val(f"loss_{key}", loss.numpy())

        if self.config.output.store_debug_data:
            self._saveDebugInfo(local_energies, key)

    def are_gradients_nan(self, gradients, epoch, batch):
        """
        Checks if the gradients for the optimization of the wavefunction contain nan values and logs an Exception.

        Args:
            gradients (tf.Tensor): Gradients obtained by taking the derivative of the groundstate energy with respect to the neural network parameters.
            epoch (int): Current number of epoch.
            batch (int): Index of the current batch.

        Returns:
            bool: True if any gradient contains a nan value.
        """
        grad_contains_nan = [tf.reduce_any(tf.math.is_nan(g)) for g in gradients]
        if any(grad_contains_nan):
            self.logger.warning(f"Epoch {epoch}, batch {batch}: Gradient contained nan. Skipping this batch")
            self.n_batches_with_nan_gradients += 1
            if self.n_batches_with_nan_gradients >= self.config.optimization.max_nan_batches:
                self.save('..')
                raise Exception(
                    f"{self.n_batches_with_nan_gradients} batches with nan gradients. Wavefunction has been saved. Aborting now.")
            return True
        return False

    def apply_gradients_default(self, wf_optimizer, gradients, variables):
        """
        Applies the gradient updates to the neural network parameters.

        Args:
            wf_optimizer (tf.keras.optimizer): Tensorflow Optimizer
            gradients (tf.Tensor):
            variables (list): All trainable neural network weights.
        """
        wf_optimizer.apply_gradients(zip(gradients, variables))

    def optimize_epoch(self, ind_epoch):
        """
        Optimization step for one epoch. Computes the loss, gradients and updates the weights. For independet calculations it is called in :meth:`~.WaveFunction.optimize` and for
        parallel training calculation :meth:`~.ParallelTrainer.optimize` calls it.

        Args:
            ind_epoch (int): Number of epoch.
        """
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch=ind_epoch, logs=self.logs)

        batch_energies = []
        for ind_batch, batch in enumerate(self.batches_train):
            grad, loss_val, local_loss, local_energies_unclipped = self.model.grad_loss(batch, tf.constant(True))
            batch_energies.append(local_energies_unclipped)
            self.logger.debug(f"Epoch {ind_epoch}, batch {ind_batch}: loss={loss_val}")
            if not self.are_gradients_nan(grad, ind_epoch, ind_batch):
                self.apply_gradient_callback(self.optimizer, grad, self.model.trainable_variables)
        local_energies = tf.squeeze(tf.concat(batch_energies, axis=0))
        self._log_local_energies(local_energies, 'train')

        self.batches_train = self.updateWalkerPositions('train', self.config.optimization.shuffle)

        # Log information about state of optimization (incl. energies) to history and tensorboard
        if (ind_epoch < self.config.output.no_skip_train_for_initial_epochs) or \
                (ind_epoch - self._epoch_last_logging) > self.config.output.n_skip_train:
            self._calculateValidationEpoch(ind_epoch)

        for cb in self.callbacks:
            cb.on_epoch_end(epoch=ind_epoch, logs=self.logs)
        self.logger.debug(f"Epoch {ind_epoch}: end of epoch")

    def _calculateValidationEpoch(self, ind_epoch):
        """
        Computes the loss with respect to the validation walkers. It is only called every config.output.n_skip_train epochs.

        Args:
            ind_epoch (int): Number of epoch.
        """
        self._epoch_last_logging = ind_epoch

        self.logger.debug(f"Validation epoch: Compute local train and valid energies for history/logging")
        batches_valid = self.updateWalkerPositions('valid', shuffle=False)

        local_energies = tf.squeeze(tf.concat([self.model(batch) for batch in self.batches_train], axis=0))
        self._log_local_energies(local_energies, 'train')
        local_energies = tf.squeeze(tf.concat([self.model(batch) for batch in batches_valid], axis=0))
        self._log_local_energies(local_energies, 'valid')

        E_train, E_valid = self.logs['mean_energies_train'], self.logs['mean_energies_valid']
        self.logger.info(f"Optimization epoch {ind_epoch}: E train: {E_train:.4f}; E valid: {E_valid:.4f}")

    # What is with all the inputs?
    def optimize(self, initial_epoch=0):
        """ Optimizes the model weights of the wave function. Call this function only after :meth:`~.WaveFunction.compile_optimization` or
        if you want to continue a previously started optimiziation. Main work happens in :meth:`~.WaveFunction.optimize_epoch`.

        Args:
            initial_epoch (int): Number of initial epoch. Only important when the wavefunction was restarted and optimization is continued otherwise zero.
        """

        if self.config.optimization.n_epochs == 0:
            return
        self.on_train_begin()

        # Main training loop over all epochs ind_epoch
        for ind_epoch in range(initial_epoch, initial_epoch + self.config.optimization.n_epochs):
            if self.model.stop_training:
                break
            t_start = time.time()
            self.optimize_epoch(ind_epoch)
            t_end = time.time()
            self.logger.debug(f"Epoch {ind_epoch}: Time for this epoch: {t_end - t_start:.3f} sec")
        self.on_train_end()

    def on_train_end(self):
        """
        Calls all callbacks after the optimization.
        """
        for cb in self.callbacks: # function name on train end
            cb.on_train_end(logs=self.logs)

    def setup_tensorboard(self):
        """
        Setups the tensorboard if a path is given in the configs.
        """
        if self.config.output.tb_path is not None:
            tb_path = os.path.expanduser(self.config.output.tb_path)
            if not os.path.exists(Path(tb_path)):
                os.makedirs(Path(tb_path))
            self.tb_writer = tf.summary.create_file_writer(tb_path)
            try:
                with self.tb_writer.as_default():
                    tensorboard.plugins.hparams.api.hparams(self.config.get_hyperparam_dict())
            except Exception:
                self.logger.warning("Could not write configuration to tensorboard")

    def updateWalkerPositions(self, type, shuffle=False, create_batches=True):
        """
        Updates walker position and stores metadata to history dictionary, e.g. for the tensorboard.

        Args:
            type (str): Either train, valid or eval dependent on the stage DeepErwin is currently.
            shuffle (bool): True if the batches of the dataset should be shuffled. Per default it is true for training.
            create_batches (bool): True if the dataset should be split into batches. Per default it is true for training and validation.

        Returns:
            tf.data.Dataset: Current position of mcmc walker.
        """
        self.logger.debug(f"Starting MCMC for walkers {type}...")
        self.walkers_mcmc[type] = self.mh_mcmc[type].step()
        self.history_add_val(f"mh_scale_{type}", self.mh_mcmc[type].scale.numpy())
        self.history_add_val(f"mh_stepsize_{type}", self.mh_mcmc[type].mean_stepsize.numpy())
        self.history_add_val(f"mh_acceptance_rate_{type}", self.mh_mcmc[type].acceptance_rate.numpy())
        self.history_add_val(f"mh_age_mean_{type}", tf.reduce_mean(self.mh_mcmc[type].age).numpy())
        self.history_add_val(f"mh_age_max_{type}", tf.reduce_max(self.mh_mcmc[type].age.numpy()))
        if create_batches:
            batches = tf.data.Dataset.from_tensor_slices(self.walkers_mcmc[type]).batch(
                batch_size=self.model.batch_size)
            if shuffle:
                batches = batches.shuffle(self.walkers_mcmc[type].shape[0], reshuffle_each_iteration=True)
            self.update_histogram(type)
            return batches
        else:
            return None

    def update_histogram(self, type):
        """
        Stores the walker positions from MCMC for the training, validation and evaluation for each epoch.

        Args:
            type (str): Either train, valid or eval
        """
        if self.histograms[type] is not None:
            state_np = np.reshape(self.walkers_mcmc[type].numpy(), [-1, 3])
            self.histograms[type][0] += \
                np.histogram2d(state_np[:, 1], state_np[:, 0], bins=self.histograms[type][1])[0]

    @tf.function
    def calculate_forces_directly(self, inp):
        """
        Calculates forces without antithetic sampling, either by fitting a polynomial to circumvent the infinite variance at the nucleus or without any
        adaptations.

        Args:
            inp (tf.Tensor): Walker positions of the evaluation of shape [batch_size x nb_el x 3].

        Returns:
            (tf.Tensor): Electronic forces of shape [batch_size x n_ions x 3]
        """
        diff_el_ion, r_el_ion = self.model.get_electron_ion_differences(inp)
        if self.config.evaluation.forces.use_polynomial:
            forces = self.model._calculate_forces_polynomial_fit(diff_el_ion, r_el_ion)
        else:
            forces = self.model._calculate_forces(diff_el_ion, r_el_ion)
        return tf.reduce_sum(forces, axis=1) + self.model.ion_ion_forces

    @tf.function
    def calculate_forces_antithetic_sampling(self, inp, log_squared):
        """
        Calculates the forces following closely the work of https://doi.org/10.1103/PhysRevLett.94.036404 by using antithetic sampling and per default fitting a polynomial
        to the force density close to the nuclei.

        Args:
            inp (tf.Tensor): Walker positions of the evaluation of shape [batch_size x nb_el x 3].
            log_squared (tf.Tensor): Evaluation of :meth:`~deeperwin.deeperwin.WFModel.log_squared` with respect to the walker positions.

        Returns:
            (tf.Tensor): Electronic forces of shape [batch_size x n_ions x 3]
        """
        # [batch x electron x ion x xyz]
        diff_el_ion, r_el_ion = self.model.get_electron_ion_differences(inp)
        if self.config.evaluation.forces.use_polynomial:
            forces = self.model._calculate_forces_polynomial_fit(diff_el_ion, r_el_ion)
        else:
            forces = self.model._calculate_forces(diff_el_ion, r_el_ion)

        ind_closest_ion = tf.cast(tf.argmin(tf.squeeze(r_el_ion, axis=3), axis=2),tf.int32)
        d_closest_ion = tf.reduce_min(tf.squeeze(r_el_ion, axis=3), axis=2)
        is_core_electron = d_closest_ion < self.config.evaluation.forces.R_core
        shift = tf.gather(diff_el_ion, ind_closest_ion, batch_dims=2, axis=2)
        inp_flipped_all = inp - 2*shift
        n_ions = self.ion_charges.shape[0]
        ion_mask = tf.equal(tf.range(n_ions, dtype=tf.int32),tf.expand_dims(ind_closest_ion, axis=-1))

        for i in tf.range(self.n_electrons):
            electron_mask = tf.reshape(tf.equal(tf.range(self.n_electrons),i), [1,-1,1])
            electron_mask = tf.logical_and(electron_mask, tf.expand_dims(is_core_electron, axis=2))
            inp_flipped = tf.where(electron_mask, inp_flipped_all, inp)

            log_squared_flipped = self.model.log_squared(inp_flipped)
            weight = tf.exp(log_squared_flipped - log_squared)
            weight = tf.reshape(weight, [-1,1,1])
            factor = tf.where(tf.logical_and(electron_mask, ion_mask), (1-weight)/2, 1.0)
            forces = forces * tf.expand_dims(factor, axis=-1)

        force = tf.reduce_sum(forces, axis=1)
        return force + self.model.ion_ion_forces

    def _is_converged(self, observable_name):
        """
        Checks if the standard deviation criteria for the evaluation energy or the forces are fulfilled and returns a bool, i.e. for the energy :math:`sigma(E_loc) <= epsilon`
        and for the forces :math:`max_i (sigma(F_i)) <= epsilon`.

        Args:
            observable_name (str): Name of the observable. Currently, we have only energy and forces.

        Returns:
            bool: True if the criteria for energies or forces are fulfilled.
        """
        if observable_name == 'energy':
            return self.std_err_total_energy * 1e3 <= self.config.evaluation.target_std_err_mHa
        elif observable_name == 'forces':
            std_error_max = np.max(self.std_err_forces * 1e3)
            return std_error_max <= self.config.evaluation.forces.target_std_err_mHa
        else:
            raise ValueError(f"Unknown observable: {observable_name}")

    def _log_evaluation(self, observable_name, epoch):
        """
        Logs progress of the evaluation for the observables energies and forces.

        Args:
            observable_name (str): Name of the observable. Currently, we have only energy and forces.
            epoch (int): Number of the epoch.
        """
        if observable_name == 'energy':
            self.logger.info(
                f"evaluation iteration {epoch}; mean local energy: {self.logs['mean_energies_eval']:.4f}; total energy: {self.total_energy:.4f} +- {self.std_err_total_energy}")
        elif observable_name == 'forces':
            msg = "evaluation iteration {}; forces: {}; total forces: {} +- {}".format(
                epoch, str(self.logs['mean_forces_eval']).replace('\n', ', '),
                str(self.forces).replace('\n', ', '), str(self.std_err_forces).replace('\n', ', '))
            self.logger.info(msg)
        else:
            raise ValueError(f"Unknown observable: {observable_name}")


    def _add_observable_for_evaluation(self, observable_name):
        """
        Adds a dictionary with necessary information like convergence and evaluation function for each observable to a list.

        Args:
            observable_name (str): Name of the observable. Currently, we have only energy and forces.
        """
        if observable_name == 'energy':
            self._observables.append(dict(name=observable_name, converged=False, func=self._evaluate_energy))
        elif observable_name == 'forces':
            self._observables.append(dict(name=observable_name, converged=False, func=self._evaluate_forces))
        else:
            raise ValueError(f"Unknown observable: {observable_name}")

    def _evaluate_energy(self):
        # extract local energies, mean, and standard deviation for evaluation walkers
        local_energies = self.model(self.walkers_mcmc["eval"])
        self.history_add_val("mean_energies_eval", tf.reduce_mean(local_energies).numpy())
        self.history_add_val("std_energies_eval", tf.math.reduce_std(local_energies).numpy())
        return local_energies

    def _evaluate_forces(self):
        """
        Calls :meth:`~.WaveFunction.calculate_forces_antithetic_sampling` to compute the forces in the evaluation step.

        Returns:
            tf.Tensor: Electronic forces of shape [batch_size x n_ions x 3]
        """
        if self.config.evaluation.forces.use_antithetic_sampling:
            forces = self.calculate_forces_antithetic_sampling(self.walkers_mcmc["eval"],
                                                               self.mh_mcmc["eval"].current_log_prob)
        else:
            forces = self.calculate_forces_directly(self.walkers_mcmc["eval"])
        self.history_add_val("mean_forces_eval", tf.reduce_mean(forces, axis=0).numpy())
        self.history_add_val("std_forces_eval", tf.math.reduce_std(forces, axis=0).numpy())
        return forces

    # evaluates total energy of wavefunction and creates histogram of electron positions in MCMC procedure
    def evaluate(self, epochs=None):
        """
        Evaluates the wave function currently represented by the model weights. Call this function only after :meth:`~.WaveFunction.compile_evaluation`
        or if you want to continue a previously started evaluation. Currently, it can evaluate the groundstate energy and the forces. For the forces the main calculations per step are computed at
        :meth:`~.WaveFunction.calculate_forces_antithetic_sampling` and for the energy :meth:`~.WaveFunction._evaluate_energy`.

        Args:
            epochs (int): Number of evaluation steps.
        """
        if self.walkers_mcmc["eval"] is None:
            raise Exception("Evaluation not properly initialized. Call compile_evaluation first.")

        if epochs is None:
            epochs = self.config.evaluation.n_epochs_max

        self._add_observable_for_evaluation('energy')
        if self.config.evaluation.forces.calculate:
            self._add_observable_for_evaluation('forces')

        # burnin steps for evaluation walkers
        self.logger.info("Starting evaluation...")
        self.logger.debug("Evaluate: Burn-in")
        self.walkers_mcmc["eval"] = self.mh_mcmc["eval"].step(self.config.integration.eval.n_burnin_steps)

        self.logger.debug("Evaluate: Evaluation")
        stop_evaluation = False # Flag for early stopping
        for epoch in range(epochs):
            if stop_evaluation:
                break
            self.updateWalkerPositions('eval', shuffle=False, create_batches=False)

            for observable in self._observables:
                if not observable['converged']:
                    observable['func']()

            if self.config.output.store_walkers_eval:
                self.history_add_val('walkers_eval', self.walkers_mcmc["eval"].numpy())

            # print evaluation progress to stdout
            if (epoch % self.config.output.n_skip_eval) == 0:
                for observable in self._observables:
                    self._log_evaluation(observable['name'], epoch)
                # Check for early stopping of evaluation
                if epoch >= self.config.evaluation.n_epochs_min:
                    for observable in self._observables:
                        if (not observable['converged']) and self._is_converged(observable['name']): # Converged in this epoch
                            self.logger.info(f"Reached target accuracy for {observable['name']}. Stopping evaluation for this observable")
                            observable['converged'] = True
                            if all([o['converged'] for o in self._observables]):
                                self.logger.info(f"Reached target accuracy all observables. Stopping evaluation.")
                                stop_evaluation = True
        else:
            # Only reached when evaluation is not prematurely stopped due to reaching target accuracy
            for observable in self._observables:
                if not observable['converged']:
                    self.logger.warning(f"Did not reach target accuracy for {observable['name']}")
        self.logger.info(f"Total energy: {self.total_energy:.4f} +- {self.std_err_total_energy} Ha")


    def init_walkers(self, n_walkers, electron_ion_mapping=None, stddev=.5):
        """
        Initialization of the walkers by a normal distribution with a mean around the nuclei and a default stddev of 0.5.

        Args:
            n_walkers (int): Number of walker. Default is 2048 for evaluation and training.
            electron_ion_mapping (list): Maps each eletron to a nucleus.
            stddev (float): Float value of the standard deviation. Default is 0.5.

        Returns:
            (tf.Tensor): Walker positions of shape [batch_size x n_el x 3]
        """
        mean = np.array([self.ion_positions[ion] for ion in electron_ion_mapping])
        return tf.random.normal(shape=[n_walkers, self.n_electrons, 3], mean=mean, stddev=stddev, dtype=DTYPE)

    @property
    def total_energy(self):
        """
        Groundstate energy of the wave function .

        Returns:
            float: Ground state energy.
        """
        E_eval = np.array(self.history["mean_energies_eval"])
        if len(E_eval) == 0:
            self.logger.warning("Wavefunction has not been evaluated yet.")
            return np.nan
        return np.mean(E_eval[~np.isnan(E_eval)])


    @property
    def std_err_total_energy(self):
        """
        Monte Carlo error.

        Returns:
            float: Monte Carlo error :math:`sigma(E_loc)/sqrt(n_mcmc)` of shape [1].
        """
        E_eval = np.array(self.history["mean_energies_eval"])
        if len(E_eval) == 0:
            self.logger.warning("Wavefunction has not been evaluated yet.")
            return np.nan
        E_eval = E_eval[~np.isnan(E_eval)]
        return np.std(E_eval) / np.sqrt(len(E_eval))

    @property
    def forces(self):
        """
        Returns the current estimate of the nuclear forces as the average of the current evaluation history.

        Returns:
            (np.array): Numpy array of shape [N x 3] where N is the number of nucleii and 3 is xyz
        """
        if ('mean_forces_eval' not in self.history) or len(self.history["mean_forces_eval"]) == 0:
            self.logger.warning("Wavefunction has not been evaluated yet.")
            return np.ones([self.ion_charges.shape[0], 3]) * np.nan
        F = np.array(self.history["mean_forces_eval"])
        return np.nanmean(F, axis=0)

    @property
    def std_err_forces(self):
        """
        Returns the current empirical std-error of the forces, i.e. std(F)/sqrt(n_samples)

        Returns:
            (np.array): Numpy array of shape [N x 3] where N is the number of nucleii and 3 is xyz
        """
        if ('mean_forces_eval' not in self.history) or len(self.history["mean_forces_eval"]) == 0:
            self.logger.warning("Wavefunction has not been evaluated yet.")
            return np.ones([self.ion_charges.shape[0], 3]) * np.nan
        F = np.array(self.history["mean_forces_eval"])
        not_nan = ~np.any(np.any(np.isnan(F), axis=2), axis=1)
        return np.std(F[not_nan, :,:], axis=0) / np.sqrt(not_nan.sum())

    def save_meta_data(self, path):
        """
        Stores the python code of the models, the helpers functions and a json with all configs.

        Args:
            path (str): Path to store models, helpers and configs.
        """
        self.logger.info(f"Saving source code and config.json to {path}")
        # create save path
        if not os.path.exists(path):
            os.makedirs(path)

        # save source file in models folder
        if self.config.output.copy_source_code and not os.path.exists(Path(path + "/source")):
            os.makedirs(Path(path + "/source"))
            dir = os.path.dirname(os.path.realpath(globals()["__file__"]))

            # save all files in models folder except pycache
            models_dir = dir + "/models/"
            for p in os.listdir(models_dir):
                if os.path.isfile(models_dir + p) and p != "__pycache__":
                    temp = models_dir + p
                    copyfile(temp, Path(path + "/source/" + p))

            # save callbacks.py + main.py files
            copyfile(dir + "/callbacks.py", Path(path + "/source/callbacks.py"))
            copyfile(globals()["__file__"], Path(path + "/source/main.py"))

        # save full config
        with open(Path(path + "/config.json"), "w") as outfile:
            json.dump(self.config.get_as_dict(), outfile, indent=4, separators=(',', ':'))

    def save(self, path):
        """ Saves the wavefunction including model weights, optimization states (MCMC walker), and evaluation states (MCMC walker) to a folder. Important if you want to restart
        the wavefunction, all necessary information will be stored.

        Args:
            path (str): Path of the folder.
        """
        self.logger.info(f"Saving wavefunction to {path}")
        # create save path
        if not os.path.exists(path):
            os.makedirs(path)

        # save model
        chkpt_model = tf.train.Checkpoint(model=self.model)
        chkpt_model.write(path + "/chkpt_model")

        # save optimizer
        if self.optimizer is not None:
            chkpt_optimizer = tf.train.Checkpoint(optimizer=self.optimizer)
            chkpt_optimizer.write(path + "/chkpt_optim")
            with open(Path(path + "/cfg_optim.pickle"), "wb") as outfile:
                pickle.dump(tf.optimizers.serialize(self.optimizer), outfile)

        # save walkers
        for key in ["train", "eval", "valid"]:
            if self.walkers_mcmc[key] is not None:
                np.save(path + "/walkers_" + key + ".npy", self.walkers_mcmc[key].numpy())

        # save history
        if not os.path.exists(Path(path + "/history")):
            os.makedirs(Path(path + "/history"))
        for key in self.history:
            np.save(path + "/history/" + key + ".npy", self.history[key])

        batch_history_fname = os.path.join(path, 'batch_history.pkl')
        with open(batch_history_fname, 'wb') as f:
            pickle.dump(self.batch_history, f, pickle.HIGHEST_PROTOCOL)

        # save histogram
        for key in self.histograms:
            if self.histograms[key] is not None:
                if not os.path.exists(Path(path + "/histograms")):
                    os.makedirs(Path(path + "/histograms"))
                np.save(path + "/histograms/" + key + ".npy", self.histograms[key])

    @classmethod
    def load(cls, path, history_only=False, modify_config_fn=None, ignore_history_keys=None, log_to_file=False) -> 'WaveFunction':
        """ Loads a previously saved wave function from a folder.

        Args:
            path (str): Path of the wave function.
            history_only (bool): If true, don't load the full model (including tf-checkpoints),
            but only the configuration and history, which loads significantly faster and can be sufficient for analysis
            modify_config_fn (function): A function that takes a configuration and returns a modified configuration,
            which is used for restoring the model. Useful to load runs from older models with differently named config params.
            ignore_history_keys (set): To filter certain history files.
            log_to_file (bool): If True, a logger will be initialized with :meth:`logging.getLogger`.

        Returns:
            The wave function as an WaveFunction instance.
        """
        path = os.path.normpath(path)  # removes trailing slashes

        if ignore_history_keys is None:
            ignore_history_keys = set()

        with open(Path(path + "/config.json")) as json_file:
            config_dict = json.load(json_file)
        config = DefaultConfig.build_from_dict(config_dict, allow_new_keys=True)
        if modify_config_fn is not None:
            config = modify_config_fn(config)

        # create instance
        wfun = cls(config, history_only=history_only, log_to_file=log_to_file)

        # load history
        if os.path.exists(Path(path + "/history")):
            for filename in os.listdir(Path(path + "/history")):
                if filename.endswith(".npy") and (filename + '.npy') not in ignore_history_keys:
                    try:
                        wfun.history[filename.split(".")[0]] = list(np.load(path + "/history/" + filename))
                    except ValueError:
                        # Catches rare error 'ValueError: Object arrays cannot be loaded when allow_pickle=False'
                        wfun.history[filename.split('.')[0]] = []

        # Batchwise history
        batch_history_fname = os.path.join(path, 'batch_history.pkl')
        if os.path.isfile(batch_history_fname):
            with open(batch_history_fname, 'rb') as f:
                wfun.batch_history = pickle.load(f)

        if not history_only:
            # load model
            chkpt_model = tf.train.Checkpoint(model=wfun.model)
            chkpt_model.restore(path + "/chkpt_model")

            # load optimizer
            if os.path.exists(Path(path + "/cfg_optim.pickle")):
                with open(Path(path + "/cfg_optim.pickle"), "rb") as file:
                    cfg_optim = pickle.load(file)
                wfun.optimizer = tf.keras.optimizers.deserialize(cfg_optim)
                if os.path.exists(Path(path + "/chkpt_optim")):
                    chkpt_optimizer = tf.train.Checkpoint(optimizer=wfun.model.optimizer)
                    chkpt_optimizer.restore(path + "/chkpt_optim")

            # load walkers, means and std deviations
            for key in ["train", "eval", "valid"]:
                path_walkers = path + "/walkers_" + key + ".npy"
                if os.path.exists(Path(path_walkers)):
                    wfun.walkers_mcmc[key] = tf.convert_to_tensor(np.load(path_walkers), dtype=DTYPE)
                wfun.mh_mcmc[key] = PositionalMetropolisHastings(wfun.config.integration[key], wfun.config, wfun.logger)
                # if wfun.walkers_mcmc[key] is not None:
                #     wfun.mh_mcmc[key].init_state(wfun.model.log_squared, wfun.walkers_mcmc[key])

            # load histogram
            if os.path.exists(Path(path + "/histograms")):
                for filename in os.listdir(Path(path + "/histograms")):
                    if filename.endswith(".npy"):
                        wfun.histograms[filename.split(".")[0]] = list(
                            np.load(path + "/histograms/" + filename, allow_pickle=True))
        return wfun

class ParallelTrainer:
    """
    This class implements the shared optimization functionality.

    It handles all instances of wavefunction for different geometries and choses based on a heuristic which one to optimize.
    Each wavefunction is an instance of the :class:`~deeperwin.main.WaveFunction` class and uses the underlying structure to optimize the weights. Similar to an independet run one has to call
     :meth:`~deeperwin.main.ParallelTrainer.optimize` to train the predefined wavefunctions and :meth:`~deeperwin.main.ParallelTrainer.evaluate` for evaluation.
    """
    def __init__(self, config, user_config_dict = None):
        """
        Initializes based on the given geometry set all wavefunction instances.

        Args:
            config (erwinConfiguration.DefaultConfig): Global config-object.
            user_config_dict (dict or None): User specific changes to the config file. Important when a shared optimization run is restarted based on a previous computation and certain hyperparameters need to be changed.
        """
        self.config = config # type: erwinConfiguration.DefaultConfig
        self.pt_config = config.parallel_trainer
        self.logger = getLogger()
        self.model_config = self.config.model

        self.shared_weights = self.pt_config.shared_weights
        if len(self.shared_weights) != 0:
            self.shared_weights.append("cusp_el_el_weight")
            self.shared_weights.append("cusp_longrange_weight")
        self.shared_weights = apply_module_shortcut_dict(self.shared_weights)
        # wavefunctions[0] == master with all weights; wavefunctions[1:] are "slaves" with references to master weights

        self.wavefunctions = []
        for i,change in enumerate(self.pt_config.config_changes):
            wf_config = copy.deepcopy(config)
            wf_config.update_with_dict(change)

            wf = wf_from_config(wf_config, index = i, user_config_dict = user_config_dict)
            self.wavefunctions.append(wf)
            self.wavefunctions[i].apply_gradient_callback = self._apply_gradients_callback


        self.n_wavefunctions = len(self.wavefunctions)
        self.shared_optimizer = self.wavefunctions[0].get_optimizer()
        self.last_epoch_trained = np.zeros(self.n_wavefunctions, np.int)
        self.n_epochs_trained = np.zeros(self.n_wavefunctions, np.int)

        self.master_model = self.wavefunctions[0].model
        self.subdirs = [f'wf_{i}' for i in range(self.n_wavefunctions)]
        self.weights = [[] for i in range(self.n_wavefunctions)] # list of all model weights for each wavefunction
        self.is_trainable = [] # has same lengths as weights
        self.is_shared = [] # has same lengths as weights
        self.is_shared_trainable = [] # only has len(model.trainable_variables)

        self.callbacks = []

    def save_all(self):
        """
        Saves all necessary data for reusing the trained weights and wavefunction instances.
        """
        for i, (wf, d) in enumerate(zip(self.wavefunctions, self.subdirs)):
            self.get_weights_from_storage(i)
            wf.save(d)

    def save_all_metadata(self):
        """
        Stores the python code of the models, the helpers functions and a json with all configs for each wavefunction. By calling
        :meth:`~deeperwin.main.WaveFunction.save_meta_data`.
        """
        for wf, d in zip(self.wavefunctions, self.subdirs):
            wf.save_meta_data(d)

    def _select_next_wavefunction_to_train(self, epoch):
        """
        Selects which wavefunction to optimize. Currently, three methods are implement. Default is the method based on the standard deviation of the local energy for the wavefunction, where only the wavefunction
        with the highest std. deviation is trained. The other methods are "random", where randomly a wavefunction is choosen and "round-robin", where every wavefunction is optimized equally in fixed manner.

        Args:
            epoch (int): Current epoch

        Returns:
            (int): Index of the wavefunction for optimization

        """
        if self.config.parallel_trainer.scheduling == 'round-robin':
            ind_wf = epoch % self.n_wavefunctions
        elif self.config.parallel_trainer.scheduling == 'random':
            ind_wf = np.random.randint(self.n_wavefunctions)
        elif self.config.parallel_trainer.scheduling == 'stddev':
            ind_wf = self._select_wf_based_on_stddev(epoch)
        self.last_epoch_trained[ind_wf] = epoch
        self.n_epochs_trained[ind_wf] += 1
        return ind_wf

    def _select_wf_based_on_stddev(self, epoch):
        """
        Standard deviation approach for optimization scheduling. Uses "round-robin" for the first 10 epochs and then based on the standard deviation returns an index of a wavefunction for optimization.

        Args:
            epoch (int): Current epoch

        Returns:
            (int): Index of the wavefunction for optimization

        """
        wf_ages = epoch - self.last_epoch_trained
        if np.any(self.n_epochs_trained < 10):
            # Initially use round-robin for 10 epochs per wavefunction
            ind = epoch % self.n_wavefunctions
            self.logger.debug(f"Selecting wf_{ind}, based on initial round-robin")
        elif np.any(wf_ages > self.config.parallel_trainer.scheduling_max_age):
            # Ensure no wavefunction stays untrained for too many epochs
            ind = np.argmax(wf_ages)
            self.logger.debug(f"Selecting wf_{ind}, based on age: {wf_ages[ind]}")
        else:
            # Select wavefunction based on std-dev of energy
            std_devs = [np.nanmean(wf.history['std_energies_train'][-3:]) for wf in self.wavefunctions]
            ind = np.nanargmax(std_devs)
            self.logger.debug(f"Selecting wf_{ind}, based on std-dev: {std_devs[ind]:.3f} Ha")
        return ind

    def _apply_gradients_callback(self, wf_optimizer, gradients, trainable_variables):
        assert len(self.is_shared_trainable) == len(trainable_variables)
        shared_gradients = [g for i,g in enumerate(gradients) if self.is_shared_trainable[i]]
        shared_variables = [v for i,v in enumerate(trainable_variables) if self.is_shared_trainable[i]]
        self.shared_optimizer.apply_gradients(zip(shared_gradients, shared_variables))

        individual_gradients = [g for i,g in enumerate(gradients) if not self.is_shared_trainable[i]]
        individual_variables = [v for i,v in enumerate(trainable_variables) if not self.is_shared_trainable[i]]
        wf_optimizer.apply_gradients(zip(individual_gradients, individual_variables))

    def move_weights_to_storage(self, i):
        """
        Creates a deepcopy of the weights of wavefunction i, so the current status of the none trainable weights are not lost by overwriting.

        Args:
            i (int): Index of the wavefunction
        """
        self.weights[i] = copy.deepcopy(self.wavefunctions[i].model.variables)

    def _is_variable_shared(self, var):
        return any([is_in_module(var.name, module_name) for module_name in self.shared_weights])

    def get_weights_from_storage(self, ind_wf):
        """
        Get all weights from storage and overwrite the current weights in the target model with the weights from storage,
        EXCEPT for weights that are trainable and shared (because those should not be overwritten, but kept between iterations)
        """
        for i, var in enumerate(self.master_model.variables):
            if (not self.is_trainable[i]) or (not self.is_shared[i]):
                var.assign(self.weights[ind_wf][i])

    def _get_number_of_shared_weights(self, master):
        """
        Counts at the beginning of the training all shared weights.

        Args:
            master (~deeperwin.models.base.WFModel): Master wavefunction.

        Returns:
            tuple containing

            - **n_params_shared** (int): All shared weights across every geometry.
            - **n_params_total** (int): Total number of weights of the neural network.
        """
        n_params_shared = 0
        n_params_total = 0
        for i, v in enumerate(master.variables):
            n_params = int(np.sum(np.prod(v.shape)))
            n_params_total += n_params
            if self.is_shared[i]:
                n_params_shared += n_params
        self.config.output.n_weights_total = n_params
        self.config.output.n_weights_shared = n_params_shared
        self.config.output.n_weights_not_shared = n_params - n_params_shared
        return n_params_shared, n_params_total

    def _warmup_models(self):
        """
        Run a single walker through all models to initialize all weights
        """
        for i,wf in enumerate(self.wavefunctions):
            self.logger.debug(f"Running warm-up for wf {i}")
            wf.init_run()
            # init here
            self.move_weights_to_storage(i)

        # Debug assertions
        for i in range(self.n_wavefunctions):
            assert len(self.weights[i]) == len(self.weights[0]), f"Wavefunction {i} does not have same same number of weights as wf 0."
            for j in range(len(self.weights[i])):
                self.weights[0][j].name == self.weights[i][j].name, f"Weight {j} of wf_{i} has a different name than in wf_0."

        self.is_shared = [self._is_variable_shared(v) for v in self.weights[0]]
        self.is_trainable = [v.trainable and not any([is_in_module(v.name, module_name) for module_name in self.master_model.non_trainable_weight_names]) for v in self.weights[0]]
        self.is_shared_trainable = [self._is_variable_shared(v) for v in self.master_model.trainable_variables]

    def _connect_models(self):
        self.master_model = self.wavefunctions[0].model
        for i in range(1, self.n_wavefunctions):
            self.wavefunctions[i].model = self.master_model

    def optimize(self, initial_epoch = 0):
        """
        Optimization function similar to the independent case. All wavefunctions are connected to master wavefunctions to make sure that all shared weights are updated for every geometry in every
        optimization step. Therefore, before each optimization step the none shared weights have to be switch to the corresponding geometry.
        After the parameter updates these weights have to be stored again by using a deepcopy.

        Args:
            initial_epoch (int): Initial epoch, only import when a previous computation is continued.
        """
        self.logger.debug("Starting parallel optifmization...")

        self._warmup_models()
        self._connect_models()

        for i, wf in enumerate(self.wavefunctions):
            self.logger.debug(f"On-train-begin for wavefunction {i}")
            self.get_weights_from_storage(i)
            wf.on_train_begin()
            self.move_weights_to_storage(i)

        self.callbacks.append(callbacks.get_adaptive_lr_callback(self.shared_optimizer,
                                                                 self.config.parallel_trainer.adaptiveLR,
                                                                 self.config.optimization.learning_rate / self.config.parallel_trainer.lr_factor,
                                                                 self.config.optimization.n_epochs))
        for cb in self.callbacks:
            cb.on_train_begin()

        # for ind_epoch in range(initial_epoch, initial_epoch + self.config.optimization.n_epochs):
        for epoch in range(initial_epoch, initial_epoch + self.config.optimization.n_epochs):
            for cb in self.callbacks:
                cb.on_epoch_begin(epoch)
            ind_wf = self._select_next_wavefunction_to_train(epoch)
            self.logger.info(f"Next WF to train: {ind_wf}")
            self.get_weights_from_storage(ind_wf)
            self.wavefunctions[ind_wf].optimize_epoch(epoch)
            if epoch == 0:
                n_weights_shared, n_weights_total = self._get_number_of_shared_weights(self.wavefunctions[ind_wf].model)
                self.logger.info(f"Weights shared: {n_weights_shared}/{n_weights_total}; Not shared: {n_weights_total-n_weights_shared}/{n_weights_total}")
            self.move_weights_to_storage(ind_wf)
            for cb in self.callbacks:
                cb.on_epoch_end(epoch)

        for i,wf in enumerate(self.wavefunctions):
            self.get_weights_from_storage(i)
            wf.on_train_end()
            self.move_weights_to_storage(i)

        for cb in self.callbacks:
            cb.on_train_end()

        self.logger.debug("Optimization finished...")

    def evaluate(self, save_wfs=True):
        """
        Evaluation of all wavefunctions one by one. Similar to the independent case.

        Args:
            save_wfs (bool):

        """
        for ind_wf, wf in enumerate(self.wavefunctions):
            self.get_weights_from_storage(ind_wf)
            self.logger.info(f"Starting evaluation for wf {ind_wf}...")
            self.wavefunctions[ind_wf].compile_evaluation()
            self.wavefunctions[ind_wf].evaluate()
            if save_wfs:
                self.wavefunctions[ind_wf].save(self.subdirs[ind_wf])
            self.move_weights_to_storage(ind_wf)


def wf_from_config(config: DefaultConfig, index=None, user_config_dict=None):
    """
    Creates a wavefunction instance from a config file. Additional options are to restart a wavefunction from a previous run and change certain hyperparameters by using a
    user_config_dict.

    Args:
        config (erwinConfiguration.DefaultConfig): Global config-object.
        index (int): If a shared optimization run is restarted the index stands for a specific geometry.
        user_config_dict (dict): If a wavefunction is restarted with the dictionary certain hyperparameters can be changed for the new optimization run.

    Returns:
        (:meth:`deeperwin.main.WaveFunction`): Instance of a wavefunction
    """
    if config.restart_dir != "":
        if user_config_dict is not None:
            def _modify_config_fn(config_wf):
                config_wf.update_with_dict(user_config_dict, allow_new_keys=True)
                return config_wf
        else:
            modify_config_fn = None
        if index is not None:
            path = os.path.join(config.restart_dir, f"wf_{index}")
        else:
            path = config.restart_dir
        return WaveFunction.load(path, modify_config_fn=_modify_config_fn, log_to_file=True)
    else:
        return WaveFunction(config, name=f"wf_{index}")


def calculate_with_weight_sharing(config: erwinConfiguration.DefaultConfig, user_config_dict = None):
    """
    Optimization and evaluation logic for the shared optimization. It initializes the logger, the Parallel Trainer and calls optimization and evaluation.
    Additionally, it creates the logging path for every wavefunction.

    Args:
        config (erwinConfiguration.DefaultConfig): Global config-object.
        user_config_dict (dict or None): Stores changes of hyperparameters if the run is restarted from a previous calculation.
    """
    logger = getLogger(force_new=True)
    logger.info(f"Starting parallel wavefunction training")

    if config.restart_dir != "":
        p = os.path.join(config.restart_dir, "../../config.in")
        with open(p) as f:
            config_dict = json.load(f)
        config = erwinConfiguration.DefaultConfig.build_from_dict(config_dict)
        initial_epoch = config.optimization.n_epochs
        config.output.code_version = getCodeVersion()
        config.update_with_dict(user_config_dict, allow_new_keys=True)
    else:
        initial_epoch = 0

    # Create calculation subdirectories
    n_wavefunctions = len(config.parallel_trainer.config_changes)
    subdirs = [f'wf_{i}' for i in range(n_wavefunctions)]
    for i,d in enumerate(subdirs):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        config.parallel_trainer.config_changes[i]['output.tb_path'] = d
        config.parallel_trainer.config_changes[i]['output.log_path'] = d + '/erwin.log'
        config.parallel_trainer.config_changes[i]['output.logger_name'] = f'erwin.wf{i}'
    config.validate()

    # Initialize wavefunctions
    pt = ParallelTrainer(config, user_config_dict)
    pt.save_all_metadata()

    # Run core program
    pt.optimize(initial_epoch)
    pt.save_all() # Save intermediate results, in particular the final trained model weights
    pt.evaluate(save_wfs=True)


def calculate_without_weight_sharing(config: erwinConfiguration.DefaultConfig, user_config_dict = None):
    """
    Optimization and evaluation logic for the shared optimization. It initializes the logger, the Parallel Trainer and calls optimization and evaluation.
    Additionally, it creates the logging path for every wavefunction.

    Args:
        config (erwinConfiguration.DefaultConfig): Global config-object.
        user_config_dict (dict or None): Stores changes of hyperparameters if the run is restarted from a previous calculation.
    """
    logger = getLogger(force_new=True, file_level=config.output.log_level_file, console_level=config.output.log_level_console)

    wf = wf_from_config(config, user_config_dict = user_config_dict)

    logger.info(f"Starting calculation for {config.physical.name}: {config.physical.n_electrons} electrons, {config.optimization.n_epochs} epochs")

    config.validate()
    wf.save_meta_data('.')

    wf.optimize()
    wf.save('.')

    wf.compile_evaluation()
    wf.evaluate()
    wf.save('.')


def main():
    # tf.config.run_functions_eagerly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("input", default="config.in", nargs='?',
                        help="Base configuration file to use (e.g. config.in). Must be a JSON-file of a config-dict. Final config is a combination of defaults and parameters set in this file.")
    args = parser.parse_args()

    with open(args.input) as f:
        config_dict = json.load(f)

    config = DefaultConfig.build_from_dict(config_dict)
    config.output.code_version = getCodeVersion()
    if config.parallel_trainer.use:
        if config.restart_dir != "":
            calculate_with_weight_sharing(config, user_config_dict=config_dict)
        else:
            calculate_with_weight_sharing(config, user_config_dict=None)
    else:
        calculate_without_weight_sharing(config, user_config_dict=config_dict)

if __name__ == '__main__':
    main()
