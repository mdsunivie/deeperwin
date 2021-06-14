import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
from abc import ABC, abstractmethod

from deeperwin.main import DTYPE
from deeperwin.utilities.utils import apply_module_shortcut_dict, is_in_module
from deeperwin.utilities.logging import getLogger


class WFModel(Model, ABC):
    """
    Implements the core functionality of a TF-based wave function model.
    In particular this class provides:

        * Calculation of the local energy through the :meth:`~deeperwin.models.base.WFModel.call` method
        * Calculation of local forces
        * Gradient calculation of the local energy
        * Calculation of basic features such as pair-wise distances between particles

    To implement a model, inherit from WFModel and implement the abstract method :meth:`~deeperwin.models.base.WFModel.log_squared`.
    For a specific model implementation example, including all network architectures see :class:`~deeperwin.models.DeepErwinModel`.
    """

    def __init__(self, config, name="WaveFunction", model_config=None,
                 **kwargs):
        """
        Create a new Wavefunction Model that can map electron coordinates to a wavefunction density :math:`\log(\psi^2)`.

        Args:
            config (erwinConfiguration.DefaultConfig): Global config-object. The model will primarily use the fields in config.model
            name (str): Name of the wavefunction model. Default is DeepErwinModel.
            model_config (erwinConfiguration.DefaultConfig.model):
            **kwargs: Arguments to be passed to the super() constructor Model() and ABC()
        """
        super().__init__(name=name, **kwargs)
        self.logger = getLogger(config.output.logger_name, config.output.log_path)
        phys = config.physical

        self.config = config
        if model_config is None:
            self.model_config = config.model
        else:
            self.model_config = model_config
        self.n_electrons = phys.n_electrons
        self.n_spinup = phys.n_spin_up
        self.n_spindown = phys.n_electrons - self.n_spinup
        self.ion_positions = tf.Variable(phys.ion_positions, trainable=False, name="ion_positions", dtype=DTYPE)
        self.ion_charges = tf.Variable(phys.ion_charges, trainable=False, name="ion_charges", dtype=DTYPE)
        self.n_ions = len(phys.ion_charges)
        self.n_particles = self.n_ions + self.n_electrons
        self.n_ion_types = len(tf.unique(self.ion_charges).y)

        self.n_particle_types = self.n_ion_types + 2
        self.particle_type_idxs = tf.concat(
            [tf.zeros(self.n_spinup, dtype=tf.int32), tf.ones(self.n_spindown, dtype=tf.int32),
             tf.unique(self.ion_charges).idx + (self.n_particle_types - self.n_ion_types)],
            axis=0).numpy()

        self.particle_type_counts = [np.count_nonzero(self.particle_type_idxs == k) for k in
                                     range(self.n_particle_types)]

        self.diff_comp_mat = self.__construct_diff_comp_mat()
        self.ion_potential = self.__get_ion_potential(self.ion_positions.numpy(), self.ion_charges.numpy())
        self.ion_ion_forces = self.get_ion_ion_forces(self.ion_positions.numpy(), self.ion_charges.numpy())
        self.force_cusp_coeffs = self.get_force_cusp_coefficients()

        self.energy_clipping_center = tf.Variable(0, trainable=False, dtype=DTYPE)
        self.energy_clipping_range = tf.Variable(self.config.optimization.clip_by, trainable=False, dtype=DTYPE)

        # optimization parameters
        self.stop_training = False

        # non_trainable_weight_names
        self.non_trainable_weight_names = apply_module_shortcut_dict(config.model.non_trainable_weights)

    def get_number_of_params(self):
        """
        Calculates the number of trainable variables in the model.

        Returns:
            int: Number of trainable variables.
        """

        return int(np.sum([np.prod(v.shape) for v in self.trainable_variables]))

    @abstractmethod
    def log_squared(self, inp):
        r"""
        Computes :math:`\log(\psi(r)^2)`. Example implementation can be found in :meth:`~deeperwin.models.DeepErwinModel.log_squared`

        Args:
            inp (tf.Tensor): Tensor of shape [batch_size x n_electrons x 3]
        """
        pass

    @tf.function
    def grad_loss(self, inp, training):
        r"""
        Gradient of the loss function :math:`\nabla_\theta E_\theta = \nabla_\theta \mathbb{E}(E_\text{loc})`.
        For the training the function clips the local energies by calling :meth:`~WFModel.clip_energies` to reduce the effect of outliers.

        Args:
            inp (tf.Tensor): Tensor of walker positions of shape [batch_size x n_el x 3].

        Returns:
            (tuple): tuple containing:

                - grad (list): Gradients with respect to the neural network weights. Each entry in the list is of type tf.Tensor.
                - loss_val (tf.Tensor): Total energy of this batch whereas it is based on clipped local loses.
                - local_loss (tf.Tensor): Clipped local energies of shape [batch_size].
                - local_energies_raw (tf.Tensor): Unclipped local energies of shape [batch_size].
        """
        # \nabla_\theta \phi_\theta} - \Braket{E_\text{loc}} \Braket{ \nabla_\theta \phi_\theta}

        self.logger.debug(
            f"Tracing function grad_loss: inp.shape = {inp.shape}; inp.dtype = {inp.dtype}; training={training}")

        with tf.GradientTape() as g:
            log_sqr = tf.squeeze(self.log_squared(inp))
            local_energies_raw = tf.squeeze(self.call(inp))
            local_energies = tf.cond(training, lambda: self.clip_energies(local_energies_raw),
                                     lambda: local_energies_raw)
            local_loss = self.local_loss(local_energies)
            loss_val = tf.reduce_mean(local_loss)
            loss_centered = local_loss - loss_val
            temp = tf.reduce_mean(log_sqr * tf.stop_gradient(loss_centered))
        grad = g.gradient(temp, self.trainable_variables)
        return grad, loss_val, local_loss, local_energies_raw

    # docstr-coverage:excused `nothing much is happening`
    @tf.function(input_signature=(tf.TensorSpec(shape=(None,), dtype=DTYPE),))
    def local_loss(self, local_energies):
        self.logger.debug(
            f"Tracing function local_loss: local_energies.shape = {local_energies.shape}; local_energies.dtype = {local_energies.dtype}")
        return local_energies

    # returns local energy Hpsi/psi
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, 3], dtype=DTYPE),))
    def call(self, inp):
        r"""
        Computes the local energies at given input positions. For a wave function :math:`\phi`, the local energy is defined as
        :math:`V(r) - \frac{1}{4} \nabla_{r}^2 \phi(r) - \frac{1}{8} \left(\nabla_{r} \phi\right)^2(r)` whereas V denotes the potential energy
        with respect to the electron positions r.

        Args:
            inp (tf.Tensor): Tensor of walker positions of shape [batch_size x n_el x 3]. Each walker represents an electron.

        Returns:
            (tf.Tensor): Returns the local energies of shape [batch_size].

        """
        self.logger.debug(f"Tracing function call: id(self) = {id(self)}, inp.shape = {inp.shape}; inp.dtype = {inp.dtype}")
        inp_list = tf.unstack(tf.reshape(inp, [-1, self.n_electrons*3]), axis=-1)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
            g.watch(inp_list)
            with tf.GradientTape(watch_accessed_variables=False) as gg:
                inp_flat = tf.stack(inp_list, axis=1)
                gg.watch(inp_flat)
                y = self.log_squared(tf.reshape(inp_flat, [-1, self.n_electrons, 3]))
            nabla_flat = gg.gradient(y, inp_flat)
            nabla_list = tf.unstack(nabla_flat, axis=1)
        nabla2_flat = tf.stack([g.gradient(nabla_list[k], inp_list[k]) for k in range(self.n_electrons * 3)], axis=1)
        kinetic_energies = tf.reduce_sum(-nabla2_flat / 4.0 - tf.math.square(nabla_flat) / 8.0, axis=1, keepdims=True)

        potential_energies = self.potential_energy(inp)
        local_energies = kinetic_energies + potential_energies
        return local_energies

    # @tf.function
    def _calculate_forces_polynomial_fit(self, diff_el_ion, d_el_ion):
        r"""
        Calculate hellman-feynman forces on ions :math:`Z_m \left(\frac{1}{N} \sum_{k} \sum_i \frac{r^k_i-R_m}{|r^k_i - R_m|^3}\right)`.
        Does not include ion-ion forces.

        Args:
            diff_el_ion (tf.Tensor): Tensor of shape [batch_size x n_el x n_ions x 3].
            d_el_ion: Distance of all electrons to all ions. Tensor of shape [batch_size x n_el x n_ions x 1].

        Returns:
            (tf.Tensor): Tensor of shape [batch_size x n_el x n_ions x 3].

        """
        # forces = [batch x n_el x n_ions x xyz]
        forces_outer = diff_el_ion / d_el_ion**3
        j = tf.range(1, self.config.evaluation.forces.j_max+1, dtype=DTYPE)
        j = tf.reshape(j, [-1, 1, 1, 1, 1])
        force_moments = (diff_el_ion/d_el_ion) * tf.pow(d_el_ion/self.config.evaluation.forces.R_core, j)
        forces_core = tf.reduce_sum(self.force_cusp_coeffs * force_moments, axis=0) # sum over moments
        forces = tf.where(d_el_ion < self.config.evaluation.forces.R_core, forces_core, forces_outer)
        forces = forces * tf.reshape(self.ion_charges, [1, 1, -1, 1])
        return forces

    # Not used?
    def _calculate_forces(self, diff_el_ion, d_el_ion):
        d_el_ion = d_el_ion / tf.tanh(d_el_ion/self.config.evaluation.forces.R_cut)
        return tf.reshape(self.ion_charges, [-1,1]) * diff_el_ion / d_el_ion ** 3

    def clip_energies(self, local_energies):
        """
        The local energies are clipped around the mean of the previous optimization step controlled by the mean distance of
        the previous local energies to their mean. Per default a soft clipping approach is applied through the tanh.
        Clipping range and energy clipping center are set after each epoch in the callback :class:`~deeperwin.utilities.callbacks.UpdateClippingCallback`.

        Args:
            local_energies (tf.Tensor): Local energies of shape [batch_size].

        Returns:
            (tf.Tensor): Clipped local energies of shape [batch_size]. Default soft clipping.
        """
        if self.config.optimization.soft_clipping:
            local_energies = self.energy_clipping_center + self.energy_clipping_range * tf.tanh((local_energies - self.energy_clipping_center) / self.energy_clipping_range)
        else:
            local_energies = tf.clip_by_value(local_energies,
                                              self.energy_clipping_center - self.energy_clipping_range,
                                              self.energy_clipping_center + self.energy_clipping_range)
        return local_energies

    def __construct_diff_comp_mat(self):
        """
        Creates a matrix with 0's and 1's. Every row represents a particle and every column stores the logic to compute the difference of two particles.

        Returns:
            (tf.Tensor): Matrix of shape [n_p x n_p*(n_p - 1)]
        """
        diff_comp_mat = np.zeros([self.n_particles, self.n_particles * (self.n_particles - 1)])
        skips = 0
        for i in range(self.n_particles ** 2):
            if (i // self.n_particles) != (i % self.n_particles):
                diff_comp_mat[(i // self.n_particles), i - skips] += 1
                diff_comp_mat[(i % self.n_particles), i - skips] -= 1
            else:
                skips += 1
        return tf.Variable(diff_comp_mat, dtype=DTYPE, trainable=False)

    def __get_ion_potential(self, ion_positions, ion_charges):
        """
        Computes the ion potential of the Hamiltonian. The remaining parts are added in :meth:`~WFModel.potential_energy`.

        Args:
            ion_positions (numpy.ndarray): Ion positions of shape [nb_ion x 3]
            ion_charges (numpy.ndarray): Ion charges of shape [nb_ion]

        Returns:
            (tf.Tensor): Summation of the distance of each nuclear pair multipled with the nuclear charges.
        """
        ion_potential = np.array(0.0)
        for k in range(self.n_ions):
            for j in range(k + 1, self.n_ions):
                ion_potential += (ion_charges[k] * ion_charges[j]) / np.linalg.norm(
                    ion_positions[k, :] - ion_positions[j, :])
        return tf.Variable(ion_potential, dtype=DTYPE, trainable=False, name="ion_potential")

    def get_ion_ion_forces(self, R, Z):
        """
        Computes the part of the forces independent of the electron positions. Per default the remaining parts are added in
        :meth:`~deeperwin.models.base.WFModel.calculate_forces_antithetic_sampling`

        Args:
            R (numpy.ndarray): Ion positions of shape [nb_ion x 3]
            Z (numpy.ndarray): Ion charges of shape [nb_ion]

        Returns:
            (tf.Tensor): Ion ion forces of shape [nb_ion]
        """
        forces = np.zeros_like(R)
        for i in range(len(R)):
            for j in range(i + 1, len(R)):
                F = Z[i] * Z[j] * (R[i] - R[j]) / (np.linalg.norm(R[i] - R[j])) ** 3
                forces[i] += F
                forces[j] -= F
        return tf.Variable(forces, dtype=DTYPE, trainable=False)

    def get_force_cusp_coefficients(self):
        r"""
        Computes the polynomial coefficients for the polynomial fit correction of the forces by solving a linear system:
        :math:`\sum_{p,l} A_{pl} c_l = b_p` with :math:`A_{pl} = R^2 / (p+l+2)` and :math:`1/(1+p)` whereas R is a predefined radius
        around the ions.

        Returns:
            (tf.Tensor): Coefficients of shape [max_degree of polynomial, 1, 1, 1, 1]
        """
        j = np.arange(1, self.config.evaluation.forces.j_max + 1)
        A = self.config.evaluation.forces.R_core**2 / (2 + j[np.newaxis, :] + j[:, np.newaxis] + 1)
        b = 1 / (j + 1)
        coeff = np.linalg.solve(A, b)
        coeff = np.reshape(coeff, [-1,1,1,1,1])
        return tf.Variable(coeff, dtype=DTYPE, trainable=False)

    def get_electron_ion_differences(self, inp):
        """
        Returns differences and distances of electrons to ions.

        Args:
            inp: Electron positions of shape [batch_size x n_el x 3]

        Returns:
            tuple containing

            - **difference matrix** (tf.Tensor): Difference matrix of the electrons to the ions of shape [batch_size x n_el x n_ion x 3]
            - **distance matrix** (tf.Tensor): Distance matrix of the electrons to the ions of shape [batch_size x n_el x n_ion x 1]

        """
        diff = tf.expand_dims(inp, -2) - self.ion_positions
        dist = tf.math.reduce_euclidean_norm(diff, axis=-1, keepdims=True)
        return diff, dist

    def get_pairwise_differences(self, inp):
        """
        Returns all pairwise distances and differences between all particles (ions and electrons).
        Matrix will first contain all electrons than all ions; electrons will contain first the spin-up electrons,
        then the spin-down electrons.

        Args:
            inp (tf.Tensor): Electron positions of shape [n_inputs, n_electrons, 3].

        Returns:
            tuple containing

             - **distance matrix** (tf.Tensor): All pairwise distances of particles in the molecule. The tensor is of shape [batch_size, 1, self.n_particles, self.n_particles-1].
             - **difference matrix** (tf.Tensor): All pairwise differences between particles in the molecules. The tensor is of shape [batch_size, 3, self.n_particles, self.n_particles-1].
        """
        n_inp = tf.shape(inp)[0]
        all_pos = tf.concat([inp, tf.reshape(tf.tile(self.ion_positions, [n_inp, 1]), [-1, self.n_ions, 3])], axis=1)

        # compute all pairwise differences and distances
        # pw_differences.shape = (batch_size, 1, self.n_particles, self.n_particles-1)
        pw_differences = tf.reshape(tf.linalg.matmul(tf.transpose(all_pos, perm=[0, 2, 1]), self.diff_comp_mat),
                                    [-1, 3, self.n_particles, self.n_particles - 1])

        pw_distances = tf.norm(pw_differences, axis=1, keepdims=True)

        return pw_distances, pw_differences

    def __get_i_e_distances(self, pw_distances):
        """
        Returns all distances between each electron to each ion.

        Args:
            pw_distances (tf.Tensor): Distances of all particles to all the other particles of shape [batch_size x 1 x n_p x n_p]

        Returns:
            (tf.Tensor): Distance tensor of shape [batch_size x 1 x n_nuc x n_el]

        """
        return pw_distances[:, :, (self.n_electrons):(self.n_particles), 0:(self.n_electrons)]

    def __get_e_e_distances(self, pw_distances):
        """
        Extract the matrix of electron-electron distances from the matrix of all pairwise distances.

        Args:
            pw_distances (tf.Tensor): Shape [batch_dims x n_particles x (n_particles - 1)]

        Returns:
            (tf.Tensor): el-el distances; shape [batch_dims x n_el x (n_el - 1)]

        """
        electron_electron_mask = tf.convert_to_tensor(
            np.triu(np.ones((self.n_electrons, self.n_electrons - 1), dtype=np.bool)))
        electron_electron_mask = tf.concat([electron_electron_mask,
                                            tf.zeros(shape=[self.n_particles - self.n_electrons, self.n_electrons - 1],
                                                     dtype=tf.bool)], axis=0)
        electron_electron_mask = tf.concat([electron_electron_mask,
                                            tf.zeros(shape=[self.n_particles, self.n_particles - self.n_electrons],
                                                     dtype=tf.bool)], axis=1)
        return tf.boolean_mask(pw_distances, electron_electron_mask, axis=2)

    def potential_energy(self, inp):
        """
        Computes the potential energy of the Hamiltonian at given inputs. First the electron electron interactions are computed, second the electron
        ion interactions and at the end the ion ion interactions are added.

        Args:
            inp (tf.Tensor): Electron positions with shape [batch_size x n_el x 3].

        Returns:
            (tf.Tensor): Potential energy of shape [batch_size x 1]

        """
        pw_distances, _ = self.get_pairwise_differences(inp)
        electron_electron_distances = self.__get_e_e_distances(pw_distances)
        ion_electron_distances = self.__get_i_e_distances(pw_distances)

        return tf.reduce_sum(
            tf.math.divide(tf.constant(1.0, dtype=DTYPE), electron_electron_distances),
            axis=-1) - tf.reduce_sum(self.ion_charges * tf.reduce_sum(
            tf.math.divide(tf.constant(1.0, dtype=DTYPE), ion_electron_distances), axis=3),
                                     axis=2) + self.ion_potential

    def get_spin_coords(self, full_coords, spin):
        """
        Take a full set of electron coordinates (or features) and
        slice out only the coordinates that belong to the given spin

        Args:
            full_coords (tf.Tensor): Coordinates of all electrons. The shape is [batch_size x n_electrons x -1].
            spin (int): Spin type.

        Returns:
            (tf.Tensor): Coordinates of the electrons to the corresponding spin. The shape is of [batch_size x n_spin x -1]
        """
        assert (full_coords.shape[1] is None) or (full_coords.shape[1] == self.n_electrons), "Dimension 1 is not an electron dimension"
        if spin == 0:
            return full_coords[:, :self.n_spinup, ...]
        elif spin == 1:
            return full_coords[:, self.n_spinup:, ...]
        else:
            raise ValueError(f"Invalid value for spin: {spin}")

    # docstr-coverage:excused ``
    def get_model_params_for_logging(self):
        return dict()

    # docstr-coverage:excused ``
    def get_summary(self):
        return dict()

    @property
    def trainable_variables(self):
        """
        Checks every trainable variable of the neural networks and sets it to trainable=False if it is contained in a list of predefined none trainable weights.
        This is used, when the weights of a previous run are reused and certain modules of the neural network should be fixed and none trainable.

        Returns:
            (list): List of all trainable variables of the neural network.
        """
        ret = []
        for var in super(WFModel, self).trainable_variables:
            if not any([is_in_module(var.name, module_name) for module_name in self.non_trainable_weight_names]):
                ret.append(var)
            else:
                var._trainable = False
        return ret


class SequentialWithResidual(tf.keras.layers.Layer):
    """
    Creates a feed-forward neural network based on given input settings.
    """
    def __init__(self, n_hidden, kernel_initializer, activation, linear_out_layer=True, n_dims_out=1, use_residual=True,
                 use_bias_on_output=True, name="SequentialResidual", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_hidden = n_hidden
        self.use_residual = use_residual

        self.dense_layers = [tf.keras.layers.Dense(units=n_hidden[k], activation=activation,
                                                   name=name + "_sub_dense_" + str(k),
                                                   use_bias=use_bias_on_output or (k != len(self.n_hidden)-1) or linear_out_layer,
                                                   dtype=DTYPE, kernel_initializer=kernel_initializer) for k in
                             range(len(self.n_hidden))]
        output_activation = None if linear_out_layer else activation
        self.out_layer = tf.keras.layers.Dense(units=n_dims_out, activation=output_activation, name=name + "_sub_dense_out", dtype=DTYPE,
                                               use_bias=use_bias_on_output)

    def call(self, inp):
        """
        Feedforward call of a fully connected neural network with the option of residual layers.

        Args:
            inp (tf.Tensor): Input of the network

        Returns:
            (tf.Tensor): Output
        """
        for k in range(len(self.n_hidden)):
            if self.use_residual and k > 0 and self.n_hidden[k] == self.n_hidden[k - 1]:
                inp = self.dense_layers[k](inp) + inp
            else:
                inp = self.dense_layers[k](inp)
        inp = self.out_layer(inp)
        return inp