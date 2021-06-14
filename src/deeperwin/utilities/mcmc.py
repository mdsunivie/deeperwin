
import numpy as np
import tensorflow as tf

import deeperwin.utilities.logging
from deeperwin.main import DTYPE

from deeperwin.utilities.erwinConfiguration import DefaultConfig, MCMCConfiguration



class PositionalMetropolisHastings:
    """
    This class contains the Metropolis Hastings algorithm to sample the electron coordinates with respect to :math:`log(\psi^2)`.

    The core functionality lies in :meth:`~PositionalMetropolisHastings._run_steps`, where new steps are proposed and accepted.
    As opposed to a pure vanilla Metropolis Hastings algorithm, this implementation has a dynamic step-size in both time and space:

        * Time: If the acceptance probability is too low, the step-size is reduced (and vice-versa)
        * Space: In regions close to the nucleii (where psi varies rapidly) smaller steps are chosen, compared to regions far from the nucleii. This space-dependant behaviour is hard-coded
    """
    def __init__(self, mcmc_config: MCMCConfiguration, full_config: DefaultConfig, logger=None):
        """
        Initializes the Metropolis Hastings algorithm.

        Args:
            mcmc_config (erwinConfiguration.DefaultConfig.integration.train.MCMCConfiguration): MCMC specific configs. The train has to be replaced by eval or valid for evaluation or validation.
            full_config (erwinConfiguration.DefaultConfig): Global config-object.
            logger (logging.Logger): Logger to dump information. If no logger is provided, the default DeepErwin logger is used
        """
        if logger is None:
            self.logger = deeperwin.utilities.logging.getLogger('erwin')
        else:
            self.logger = logger
        self.config = full_config
        self.mcmc_config = mcmc_config
        self.log_prob_density_fn = None
        self.ion_positions = tf.Variable(self.config.physical.ion_positions, dtype=DTYPE, trainable=False)
        self.ion_charges =  tf.Variable(self.config.physical.ion_charges, dtype=DTYPE, trainable=False)
        self.R_ions = tf.expand_dims(self.ion_positions, axis=0)
        self.ion_core_radius = 1.0 /self.ion_charges
        self.scale = tf.Variable(1.0, trainable=False, dtype=DTYPE)
        self.mean_stepsize = tf.Variable(1.0, trainable=False, dtype=DTYPE)
        self.acceptance_rate = tf.Variable(0.0, trainable=False, dtype=DTYPE)
        self.age = None
        self.current_state = None
        self.current_log_prob = None

    def init_state(self, log_prob_density_fn, state, scale=None):
        """
        Initialize walkers and their log probability density.

        This function calls the log_prob_density_fn once and can therefore be computationally expensive.
        Args:
            log_prob_density_fn (function): Function that computes :math:`\log(p(x)^2)`
            state (np.array, tf.Tensor): Initial state x for all walkers
            scale (float): Initial stepsize scale. If None is given, the default (1.0) is kept

        Returns:
            None
        """
        self.log_prob_density_fn = log_prob_density_fn
        self.current_state = tf.Variable(state)
        self.current_log_prob = tf.Variable(self.log_prob_density_fn(state))
        self.age = tf.Variable(np.zeros(state.shape[0]), dtype=tf.int32, trainable=False)
        if scale != None:
            self.scale.assign(scale)

    def propose(self, state):
        """
        Proposes a new mcmc step which satisfies the detailed balance property.

        Args:
            state (tf.Tensor): Current position of the mcmc walker with shape [nb_walker x nb_electrons x 3]

        Returns:
            (tf.Tensor): New mcmc step of shape [nb_walker x nb_electrons x 3].
            (tf.Tensor): Detailed balance correction of shape [batch_size].

        """
        sigma_i = self._getLocalStepsize(state)
        self.mean_stepsize.assign(tf.reduce_mean(sigma_i))
        delta = tf.random.normal(state.shape) * tf.expand_dims(sigma_i, axis=2)
        new_state = state + delta
        sigma_j = self._getLocalStepsize(new_state)
        delta_sqr = tf.reduce_sum(tf.square(delta), axis=2)

        det_bal_corr = 3*tf.reduce_sum(tf.math.log(sigma_i) - tf.math.log(sigma_j), axis=1)
        det_bal_corr = det_bal_corr + 0.5*tf.reduce_sum(delta_sqr * (1/sigma_i**2 - 1/sigma_j**2), axis=1)
        return new_state, det_bal_corr

    def _getDistanceMatrix(self, state):
        x1n = tf.reduce_sum(tf.square(state), axis=2)
        x2n = tf.reduce_sum(tf.square(self.R_ions), axis=2)
        cross_term = tf.matmul(state, self.R_ions, transpose_b=True)
        dist = tf.expand_dims(x1n, axis=2) + tf.expand_dims(x2n, axis=1) - 2 * cross_term
        return tf.math.sqrt(dist)

    def _getLocalStepsize(self, state):
        """
        Computes for each electron a step size which depends on its distance to the closest ion to encourage smaller steps close to the nuclei.

        Args:
            state (tf.Tensor): Current position of the mcmc walker with shape [nb_walker x nb_electrons x 3]

        Returns:
            (tf.Tensor): Computes local stepsize for each electron. The tensor is of shape [batch_size x n_el]

        """
        if self.mcmc_config.use_local_stepsize:
            d_el_ions = self._getDistanceMatrix(state)
            d_effective = tf.reduce_min(d_el_ions / self.ion_core_radius + self.mcmc_config.local_stepsize_constant, axis=2) # reduce over ions
            return tf.clip_by_value(d_effective * self.scale, self.mcmc_config.min_scale, self.mcmc_config.max_scale)
        else:
            return tf.clip_by_value(tf.ones(state.shape[:2], dtype=DTYPE) * self.scale, self.mcmc_config.min_scale, self.mcmc_config.max_scale)

    def _adjustStepSize(self, acceptance_rate):
        """
        Adjusts the scale for the local step sizes to achieve an average acceptance rate of 50%. Either decreases it if the acceptance rate is too small
        or increases it if it is too big.

        Args:
            acceptance_rate (tf.Tensor): Current average acceptance rate.

        Returns:
            (tf.Tensor): Adjusted new scale
        """
        if acceptance_rate < self.mcmc_config.target_acceptance_rate:
            scale = self.scale / 1.1
        else:
            scale = self.scale * 1.05
        return scale

    @tf.function
    def _run_steps(self, current_state, current_log_prob, n_steps):
        """
        Computes based on the previous walker positions new positions.

        Args:
            current_state (tf.Tensor): Current mcmc walker position of shape [batch_size x n_el x 3]
            current_log_prob (tf.Tensor): Current evaluation of :math:`log(\psi^2)` based on current_state with shape [batch_size].
            n_steps (tf.Tensor): Number of steps the walker should move before the next optimization/ evaluation step.

        Returns:
            (tf.Tensor): New walker position with shape [batch_size x n_el x 3] after n_steps.
            (tf.Tensor): New evaluation of :math:`log(\psi^2)` for accepted walker steps.
        """
        for i in tf.range(n_steps):
            new_state, db_corr = self.propose(current_state)
            new_log_prob = self.log_prob_density_fn(new_state)
            p_accept = tf.exp(new_log_prob - current_log_prob + db_corr)
            accept = p_accept > tf.random.uniform(p_accept.shape, dtype=DTYPE)

            self.acceptance_rate.assign(tf.reduce_mean(tf.cast(accept, dtype=DTYPE)))
            self.scale.assign(self._adjustStepSize(self.acceptance_rate))

            accept = tf.logical_or(accept, self.age > self.mcmc_config.max_age)
            self.age.assign(tf.where(accept, 0, self.age+1))
            current_log_prob = tf.where(accept, new_log_prob, current_log_prob)
            current_state = tf.where(tf.reshape(accept, [-1, 1, 1]), new_state, current_state)
        return current_state, current_log_prob

    def step(self, n_steps=None):
        """
        High level function to run the Metropolis Hastings algorithm for multiple steps. Under the hood, calls the tensorflow function :meth:`_run_steps` which performs the actual computation.

        Args:
            n_steps: Number of steps to take. If None are given, the default number stored in the config (self.mcmc_config.n_inter_steps) is used.
        Returns:
            (tf.Tensor): State vector after n Metropolis Hastings steps
        """
        if n_steps is None:
            n_steps = tf.constant(self.mcmc_config.n_inter_steps, dtype=tf.int32)
        else:
            n_steps = tf.constant(n_steps, dtype=tf.int32)
        current_state, current_log_prob = self._run_steps(self.current_state, self.current_log_prob, n_steps)
        self.current_state.assign(current_state)
        self.current_log_prob.assign(current_log_prob)
        return self.current_state
