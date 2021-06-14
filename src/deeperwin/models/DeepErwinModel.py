import deeperwin.models.base
import deeperwin.utilities.utils
from deeperwin.main import DTYPE
import deeperwin.main
import deeperwin.models.CASSCFModel
from deeperwin.utilities.erwinConfiguration import DeepErwinModelConfig
import tensorflow as tf
import numpy as np

class DeepErwinModel(deeperwin.models.base.WFModel):
    """
    This class contains the specific implementation of our DeepErwin Wavefunction-Model.
    Calling the log_squared method of an instance calculates :math:`log(\psi^2)` for given electron coordinates.
    Internally the DeepErwinModel uses 3 core bulding blocks:

    * A Slater-type model that calculates orbitals and determinants
    * An instance of the SimpleSchnet class that calculates electron embeddings
    * Various trainable neural networks that take the electron embeddings as inputs and calculate adjustements (Jastrow factor and backflows) to the result of the Slater-type model

    """
    def __init__(self, config, history_only=False, **model_kwargs):
        """
        Create a new WFModel based on the specified configuration

        Args:
            config (erwinConfiguration.DefaultConfig): Global config-object. The model will primarily use the fields in config.model
            history_only (bool): Used when loading a model for analysis. Only populates the config, but does not actually build the model
            **model_kwargs: Arguments to be passed to the super() constructor WFModel()
        """
        super().__init__(config, **model_kwargs)
        self.config = config
        self.model_config = config.model # type: DeepErwinModelConfig
        if history_only:
            return

        self.slater_model = deeperwin.models.CASSCFModel.CASSCFModel(config)
        self.initializer = deeperwin.utilities.utils.get_initializer_from_config(config)
        self.energy_clipping_center.assign(self.slater_model.total_energy)
        config.output['E_casscf'] = self.slater_model.total_energy
        config.output['E_HF'] = self.slater_model.total_energy_hf

        if self.model_config.embed.use:
            self.embedding = SimpleSchnet(self.config)

        # Backflow factor + shift
        self.backflow_shift_nets, self.backflow_factor_nets, self.general_backflow_factor = self._build_backflow_nets()

        if self.model_config.use_backflow_shift:
            self.backflow_shift_weight = tf.Variable(initial_value=self.config.model.initial_backflow_shift_weight,
                                                     trainable=True, name='backflow_shift_weight', dtype=DTYPE)
        if self.model_config.use_backflow_factor:
            self.backflow_factor_weight = tf.Variable(initial_value=self.config.model.initial_backflow_factor_weight,
                                                      trainable=True, name='backflow_factor_weight', dtype=DTYPE)
        # Jastrow factor
        self.symmetric_fit_nets = self._build_symmetric_nets()

        self.cusp_el_el_weight = tf.Variable(initial_value=0.5, trainable=True,
                                                 constraint=tf.keras.constraints.NonNeg(), dtype=DTYPE,
                                                 name="cusp_el_el_weight")

        # decaying shift
        if self.model_config.decaying_shift.use:
            # use a tensor of shape [1] instead of shape [] to circumvent keras constraints bug for 0-dim variables
            self.decaying_shift_scale = tf.Variable(initial_value=[self.model_config.decaying_shift.initial_scale],
                                                    trainable=self.model_config.decaying_shift.trainable_scale,
                                                    constraint=tf.keras.constraints.MinMaxNorm(0.1, 5.0))
        if self.model_config.use_backflow_shift:
            self.backflow_shift_weight = tf.Variable(initial_value=self.config.model.initial_backflow_shift_weight,
                                                     trainable=True, name='backflow_shift_weight', dtype=DTYPE)
        if self.model_config.use_backflow_factor:
            self.backflow_factor_weight = tf.Variable(initial_value=self.config.model.initial_backflow_factor_weight,
                                                      trainable=True, name='backflow_factor_weight', dtype=DTYPE)
        if self.model_config.use_symmetric_part:
            self.symm_weight = tf.Variable(initial_value=self.config.model.initial_symm_weight, trainable=True,
                                           name='symm_weight', dtype=DTYPE)

    def _build_backflow_nets(self):
        """
        Generates empty neural networks to calculate the backflow factors and backflow shifts.
        At this point model weights are not yet initialized.

        Returns:
            tuple containing

            - **shift_nets** (list): list of 2 elements containing backflow shift networks for spin-up and spin-down
            - **factor_nets** (list of lists): list containing 2 lists: one for spin-up, one for spin-down. Each list contains one neural network for each determinant and orbital. In total there are n_determinants*n_electron neural nets
            - **general_net** (list): list of 2 neural networks (one for spin-up, one for spin-down). Each network takes an electron-embedding as input and outputs a backflow-embedding that is then fed to the backflow-shift and backflow-factor nets
        """
        shift_nets = [None, None]
        if self.model_config.use_backflow_shift:
            for spin in range(2):
                if np.any(self.particle_type_idxs == spin):
                    name = f"backflow_shift_net_{spin}"
                    net = self._build_sequential_net(name, self.model_config.n_hidden_backflow_shift, 3, bias_out=False)
                    shift_nets[spin] = net

        factor_nets = [[],[]]
        general_net = []

        self.ind_orbitals_backflow = [[], []]

        if self.model_config.use_backflow_factor:
            # Std. backflow factor
            self.ind_orbitals_backflow = []
            for spin in range(2):
                if spin == 0:
                    n_elec = self.n_spinup
                else:
                    n_elec = self.n_electrons - self.n_spinup
                # Create for each determinant and each electron a unique factor
                for i in range(n_elec * self.model_config.slatermodel.n_determinants):
                    name = f"backflow_factor_net_spin{spin}_ind_{i}"
                    net = self._build_sequential_net(name, self.model_config.n_hidden_backflow_factor_orbitals, 1)
                    factor_nets[spin].append(net)
                # Create a general backflow factor part
                name = f"backflow_factor_general_net{spin}"
                general_net.append(
                    self._build_sequential_net(name, self.model_config.n_hidden_backflow_factor_general))
                self.ind_orbitals_backflow.append(tf.reshape(tf.constant(np.arange(0, n_elec*self.model_config.slatermodel.n_determinants)), [self.model_config.slatermodel.n_determinants, n_elec]))
        return shift_nets, factor_nets, general_net

    def _build_symmetric_nets(self):
        """
        Generates the networks for the Jastrow factors.

        Returns:
            (list): list containing 2 elements (one each for spin up/down), each being one neural network that maps electron embeddings to a Jastrow factor
        """
        nets = [None, None]
        if self.model_config.use_symmetric_part:
            for spin in range(2):
                if not np.any(self.particle_type_idxs == spin):
                    continue
                name = f"symmetric_fit_net_{spin}"
                nets[spin] = self._build_sequential_net(name, self.config.model.n_hidden_fit, 1)
        return nets

    def _build_sequential_net(self, name, n_hidden, n_linear_out = None, bias_out = True):
        """
        Wrapper function to generate a standard fully connected Sequential Neural network with tanh activation function.
        Depending on the configuration, the model will use or not use residual connections between subsequent layers.

        Args:
            name (str): tensorflow-id to be used for this network
            n_hidden (list): Number of hidden units per layer. If n_linear_out is None, the last entry specifies the output dimension
            n_linear_out (int or None): Output dimension of the network. For this last layer no activation function is applied
            bias_out (bool): True if bias weights should be used in output layer

        Returns:
            (dpe.SequentialWithResidual): Sequential Neural Network
        """
        if n_linear_out is None:
            return deeperwin.models.base.SequentialWithResidual(n_hidden[:-1],
                                                                kernel_initializer=self.initializer,
                                                                activation=tf.tanh, linear_out_layer=False,
                                                                n_dims_out=n_hidden[-1],
                                                                use_residual=self.config.model.use_residual,
                                                                name=name, use_bias_on_output=bias_out)
        else:
            return deeperwin.models.base.SequentialWithResidual(n_hidden,
                                                                kernel_initializer=self.initializer,
                                                                activation=tf.tanh, linear_out_layer=True,
                                                                n_dims_out=n_linear_out,
                                                                use_residual=self.config.model.use_residual,
                                                                name=name, use_bias_on_output=bias_out)

    def get_pairwise_features(self, pairwise_distances, pair_wise_differences):
        """
        Takes pairwise distances between all particles and calculates feature vectors for all pairwise combinations of electrons and other particles.

        Args:
            pairwise_distances (tf.Tensor): Tensor of shape [batch-size x 3 x n_particles x (n_particles-1)], describing all pairwise distances in each cartesian coordinate between particles
            pair_wise_differences (tf.Tensor): [batch-size x 1 x n_particles x (n_particles-1)], describing the norm of all pairwise distances

        Returns:
            (tf.Tensor): Tensor of shape [batch-size x n_electrons x n_particles x n_features]
        """
        EPS = 0.5

        # Transform to format [batch_size x n_electrons x (n_particles-1) x 3]
        diff = tf.transpose(pair_wise_differences, [0, 2, 3, 1])[:, :self.n_electrons, :, :]
        dist = tf.transpose(pairwise_distances, [0, 2, 3, 1])[:, :self.n_electrons, :, :]
        dist_featurs = [1.0 / (dist + EPS)]
        if self.model_config.n_rbf_features > 0:
            dist_featurs.append(self.get_rbf_features(dist))

        dist_features = tf.concat(dist_featurs, axis=-1)
        directional_features = diff / (dist + EPS)**2  # delta of electrons to all other particles
        features = tf.concat([directional_features, dist_features], axis=-1)
        return features

    def get_rbf_features(self, r):
        """
        Maps distances to a set of gaussians, which can then be used as a kind of "one-hot-encoding" of the distance

        Args:
            r (tf.Tensor): Tensor of shape [batch-dimensions x 1] representing pairwise distances

        Returns:
            (tf.Tensor): Tensor of shape [batch-dimensions x n_rbf_features]. The number of features returned is set in the config
        """
        r_rbf_max = 5.0
        q = tf.linspace(0, 1, self.model_config.n_rbf_features)
        q = tf.cast(q, DTYPE)
        mu = q**2 * r_rbf_max
        sigma = r_rbf_max / (self.model_config.n_rbf_features-1) * (2*q+1/(self.model_config.n_rbf_features-1))

        return tf.exp(-((r-mu)/sigma)**2)

    def get_cusp_short_range(self, all_distances):
        r"""
        Returns the electron-electron cusp-correction, i.e. a negative log-squared when electrons come close to each other.
        The correction is given by :math:`\text{weight} \frac{d_{ij}}{(d_{ij}+1}`.
        Args:
            all_distances (tf.Tensor): Tensor of shape [batch_dimensions x 1 x  n_particles x (n_particles-1)]

        Returns:
            (tf.Tensor): Tensor of shape [batch_dimensions]

        """
        d_el_el = all_distances[:,0,:self.n_electrons,:self.n_electrons-1]
        log_sqr = self.cusp_el_el_weight * tf.reduce_sum(d_el_el/(d_el_el+1), axis=[1,2]) / 2 # divide by 2 because each pair is being double-counted
        return log_sqr

    def _calculate_backflow_factor(self, x_spin, spin):
        """
        Calculate the backflow matrix to be multiplied with the orbitals.

        Args:
            x_spin: (tf.Tensor): Electron embedding for all electrons of a given spin. Shape [batch-dimensions x n_el_spin x embedding_dim]
            spin (int): Which spins the given electrons have. 0 = up, 1 = down

        Returns:
            (tf.Tensor): Tensor of shape [batch-dimensions x n_determinants x n_el_spin x n_el_spin (orbitals)]. The last dimension varies along the orbitals, the second last dimension varies along the electron index"""
        # Calculate backflow factor for n_el_spin per determinant
        x_spin = self.general_backflow_factor[spin](x_spin)
        bf = tf.concat([net(x_spin) for net in self.backflow_factor_nets[spin]], axis=-1)
        bf = tf.gather(bf, self.ind_orbitals_backflow[spin], axis=-1)
        # swap dimensions 1,2 so that it becomes [batchsize x n_dets x n_el_spin x n_el_spin (orbitals)]
        bf = tf.transpose(bf, perm=[0, 2, 1, 3])

        return 1.0 + tf.exp(self.backflow_factor_weight) * bf

    def _calculate_shift_decay(self, d_el_ion_spin):
        """
        Computes a decay factor that can then be applied to the backflow shift, in order to ensure not violating the cusp condition.
        The decay for electron i is calculated as :math:`tanh(d_i/s_i)^2`, where :math:`d_i` is the distance to the closest ion for electron i, and s_i is the decay length-scale for this ion.
        The decay length is proportional to 1/Z, because the core-electron-orbitals for these nuclei are also narrower.

        Args:
            d_el_ion_spin (tf.Tensor): Tensor of shape [batch_size x n_el x n_ion]

        Returns:
            (tf.Tensor): Decay factor as tensor of shape [batch_size x n_el x 1]
        """
        # Get distance and charge of closest ion
        ind_closest_ion = tf.argmin(d_el_ion_spin, axis=2)
        Z_ion = tf.gather(self.ion_charges, ind_closest_ion)
        d_to_closest_ion = tf.reduce_min(d_el_ion_spin, axis=2)

        # decay scale ~ 1/Z
        scale = self.decaying_shift_scale / Z_ion
        return tf.expand_dims(tf.tanh((d_to_closest_ion / scale) ** 2), axis=-1)

    def log_squared(self, el_coords):
        """
        Main function of the wavefunction model, which calculates the log of the squared wavefunction for a given configuration of electron coordinates.
        This wavefunction can then be used for MCMC sampling, or for energy evaluation.

        Args:
            el_coords (tf.Tensor): Tensor of shape [batch_size x n_electrons x 3]

        Returns:
            (tf.Tensor): Tensor of shape [batch_size] that contains :math:`log(psi(r)^2)`
        """
        self.logger.debug(f"Tracing log_squared: id(self)={id(self)}, el_coords.shape={el_coords.shape}, el_coords.dtype={el_coords.dtype}")
        # Calculate pairwise distances and use them to calculate a high-dimensional embedding per electron
        all_distances, all_differences = self.get_pairwise_differences(el_coords)

        pair_features = self.get_pairwise_features(all_distances, all_differences)
        x_all = self.embedding(pair_features)

        log_squared = self.get_cusp_short_range(all_distances)

        backflow_shift = []
        backflow_factors = []
        for spin in range(2):
            x_spin = self.get_spin_coords(x_all, spin)
            #d_to_closest_ion = self.get_closest_ion(all_distances,spin)
            # Symmetric part
            if self.model_config.use_symmetric_part:
                # sum over 1-electron contributions
                log_symm = tf.squeeze(tf.reduce_sum(self.symmetric_fit_nets[spin](x_spin), axis=1))
                log_squared = log_squared + tf.exp(self.symm_weight) * log_symm

            # Backflow for antisymmetric part: shift of coordinates
            if self.model_config.use_backflow_shift:
                shift = self.backflow_shift_nets[spin](x_spin)
                if self.model_config.decaying_shift.use:
                    # ensures that the shift is zero at r_electron = R_ion
                    d_el_ion = self.get_spin_coords(
                        tf.squeeze(all_distances[:, :, :self.n_electrons, self.n_electrons - 1:], axis=1), spin)
                    shift = shift * self._calculate_shift_decay(d_el_ion)
                backflow_shift.append(shift)
            # Backflow for antisymmetric part: factor to be multiplied with orbitals in slater-det
            if self.model_config.use_backflow_factor:
                backflow_factors.append(self._calculate_backflow_factor(x_spin, spin))
            else:
                backflow_factors.append(1.0)

        # Assemble backflow for all coordinates (across both spins)
        if len(backflow_shift) > 0:
            backflow_shift = tf.concat(backflow_shift, axis=1)
            el_coords = el_coords + tf.exp(self.backflow_shift_weight) * backflow_shift

        # psi_anti = self.slater_model.get_psi(el_coords, backflow_factors[0], backflow_factors[1])
        log_squared = log_squared + self.slater_model.get_log_psi_squared(el_coords, backflow_factors[0], backflow_factors[1])
        return log_squared

    def get_model_params_for_logging(self):
        """
        Collects several key model parameters and returns them as a dict. Parameters added here will eventually end up in the WaveFunction.history

        Returns:
            (dict): dict of floats or numpy arrays
        """
        data = {}
        for attr in ['cusp_long_range_weight', 'cusp_el_el_weight', 'cusp_el_ion_weight', 'backflow_shift_weight',
                     'backflow_factor_weight', 'symm_weight']:
            if hasattr(self, attr):
                data[attr] = getattr(self, attr).numpy()
        return data

    def get_summary(self):
        """
        Collect several key model parameters to be used in the model-summary and returns them as a dict. #
        The dict contains information on the model architecture, as well as the baseline energy calculated by the underlying Slater-type model

        Returns:
            (dict): Key/Values of architecture parameters
        """
        data = {}
        try:
            data['E_casscf'] = self.slater_model.total_energy
        except AttributeError:
            pass
        return data

class SimpleSchnet(tf.keras.Model):
    """
    Submodel that calculates high-dimensional, interacting electron embeddings from electron coordinates.
    Heavily based on the version of SchNet that has been outlined in https://arxiv.org/abs/1909.08423
    """
    def __init__(self, config: deeperwin.utilities.erwinConfiguration.DefaultConfig, *args, **kwargs):
        """
        Build a new embedding model based on the configuraton. Primarily keys contained in config.model.embed will be used.

        Args:
            config (erwinConfiguration.Defaultconfig): Full global configuration object
            *args: Additional arguments to be passed to the super constructor (tf.keras.Model)
            **kwargs: Additional arguments to be passed to the super constructor (tf.keras.Model)
        """

        super().__init__(*args, **kwargs)
        self.embed_config = config.model.embed
        self.config = config
        self.embed_dim = self.embed_config.embedding_dim
        self.n_iterations = self.embed_config.n_iterations

        ion_charges = np.array(self.config.physical.ion_charges, np.int)
        self.ion_indices = [np.where(ion_charges == Z)[0] for Z in np.unique(ion_charges)]
        self.ion_indices = [tf.Variable(v, trainable=False, dtype=tf.int32) for v in self.ion_indices]
        self.ion_ids = [list(np.unique(ion_charges)).index(Z) for Z in ion_charges]
        self._build_nets()
        self._build_index_matrices()

    def _get_default_net(self, n_hidden, n_out, name=None):
        """
        Helper function to build a fully-connected Sequential network with tanh-activation and no linear output layer

        Args:
            n_hidden (list): Number of hidden nodes per layer
            n_out (int): Output dimension
            name (str): Tensorflow-identification name to be used

        Returns:
            (tf.keras.Sequential): Neural network
        """
        initializer = deeperwin.utilities.utils.get_initializer_from_config(self.config)
        layers = [tf.keras.layers.Dense(n, activation=tf.tanh, kernel_initializer=initializer, name=name+f"_{k}")
                                   for k,n in enumerate(n_hidden + [n_out])]
        return tf.keras.Sequential(layers, name=name)

    def _build_nets(self):
        """
        Builds all neural networks and embedding vectors that are required for SimpleSchnet.
        The specific network dimensions are all specified in self.config.model.embed

        In particular:

        * w_same, w_diff, w_ions: Networks that compute weightings for the contribution of every other particle during the embedding rounds
        * h_same, h_diff: Transformation networks that transform electron embeddings in each round. Their contributions to the next round of embeddings are weighted by the w-networks
        * ion_embeddings: Fixed embedding vectors for each ion-type (instead of transformation networks h)
        * g: Network that transforms the output of each embedding round

        Returns:
            None
        """
        self.w_same = []
        self.w_diff = []
        self.w_ions = []
        self.h_same = []
        self.h_diff = []
        self.ion_embeddings = []
        self.g = []
        self.h_same_init = tf.Variable(np.random.randn(1, self.embed_dim), dtype=tf.float32, trainable=True)
        self.h_diff_init = tf.Variable(np.random.randn(1, self.embed_dim), dtype=tf.float32, trainable=True)
        self.n_unique_ions = len(set(self.config.physical.ion_charges))
        self.ion_embeddings= tf.Variable(np.random.randn(self.n_unique_ions, self.n_iterations, self.embed_dim),
                                               trainable=True, dtype=DTYPE, name=f'embed_ion_embedding')
        for n in range(self.embed_config.n_iterations):
            self.w_same.append(self._get_default_net(self.embed_config.n_hidden_w, self.embed_dim, f'embed_w_same_{n}'))
            self.w_diff.append(self._get_default_net(self.embed_config.n_hidden_w, self.embed_dim, f'embed_w_diff_{n}'))
            self.w_ions.append([self._get_default_net(self.embed_config.n_hidden_w, self.embed_dim, f'embed_w_ions_{n}_{k}')
                           for k in range(self.n_unique_ions)])

            if self.embed_config.use_g_function:
                self.g.append(self._get_default_net(self.embed_config.n_hidden_g, self.embed_dim, f'embed_g_{n}'))

        for n in range(self.embed_config.n_iterations - 1):
            self.h_same.append(self._get_default_net(self.embed_config.n_hidden_h, self.embed_dim, f'embed_h_same_{n}'))
            self.h_diff.append(self._get_default_net(self.embed_config.n_hidden_h, self.embed_dim, f'embed_h_diff_{n}'))


    def _build_index_matrices(self):
        """
        Helper function that generates a matrix of indices that allows to gather all pairs of electrons that are either up/up or down/down.
        Results are stored in self._indices_u_u and self._indices_d_d

        Returns:
            None
        """
        n_el = self.config.physical.n_electrons
        n_up = self.config.physical.n_spin_up
        n_dn = n_el - n_up
        self._indices_u_u = tf.Variable([[j for j in range(n_up) if j != i] for i in range(n_up)],
                                        trainable=False, dtype=tf.int32)
        self._indices_d_d = tf.Variable([[j+n_up for j in range(n_dn) if j != i] for i in range(n_dn)],
                                        trainable=False, dtype=tf.int32)

    def call(self, pairwise_features):
        """
        Computes electron embeddings from a pairwise-feature-matrix for each electron-particle combination.

        Args:
            pairwise_features (tf.Tensor): Tensor of shape [batch_size x n_electrons x n_particles-1 x n_rbf]

        Returns:
            (tf.Tensor): Tensor of shape [batch_size x n_electrons x embedding_dim]
        """
        n_el = self.config.physical.n_electrons
        n_up = self.config.physical.n_spin_up
        r_pairs_u_u = pairwise_features[:, :n_up, :n_up-1,:]
        r_pairs_d_d = pairwise_features[:, n_up:n_el, n_up:n_el-1,:]
        r_pairs_u_d = pairwise_features[:, :n_up, n_up-1:n_el-1,:]
        r_pairs_d_u = pairwise_features[:, n_up:n_el, :n_up, :]
        r_pairs_ions = [tf.gather(pairwise_features, ind + n_el-1, axis=2) for ind in self.ion_indices]
        h_ions = tf.gather(self.ion_embeddings, self.ion_ids, axis=0)

        # Round 0: r -> x
        for n in range(self.n_iterations):
            w_u_u = self.w_same[n](r_pairs_u_u)
            w_u_d = self.w_diff[n](r_pairs_u_d)
            w_d_d = self.w_same[n](r_pairs_d_d)
            w_d_u = self.w_same[n](r_pairs_d_u)
            w_el_ions = tf.concat([w_ions(r) for w_ions, r in zip(self.w_ions[n], r_pairs_ions)], axis=2)

            if n == 0:
                h_same = tf.expand_dims(tf.tile(self.h_same_init, [n_el, 1]), axis=0)
                h_diff = tf.expand_dims(tf.tile(self.h_diff_init, [n_el, 1]), axis=0)
            else:
                h_same = self.h_same[n - 1](x)
                h_diff = self.h_diff[n - 1](x)

            h_u_u = tf.gather(h_same, self._indices_u_u, axis=1)
            h_u_d = tf.expand_dims(h_diff[:,n_up:,:], axis=1)
            h_d_d = tf.gather(h_same, self._indices_d_d, axis=1)
            h_d_u = tf.expand_dims(h_diff[:, :n_up, :], axis=1)

            x_up = tf.reduce_sum(w_u_u * h_u_u, axis=2) + tf.reduce_sum(w_u_d * h_u_d, axis=2)
            x_dn = tf.reduce_sum(w_d_u * h_d_u, axis=2) + tf.reduce_sum(w_d_d * h_d_d, axis=2)
            x = tf.concat([x_up, x_dn], axis=1)
            x = x + tf.reduce_sum(w_el_ions * h_ions[:,n,:], axis=2)
            if self.embed_config.use_g_function:
                x = self.g[n](x)
        return x

if __name__ == '__main__':
    import logging

    tf.config.run_functions_eagerly(True)

    config = deeperwin.utilities.erwinConfiguration.DefaultConfig(quality='minimal')
    config.model = deeperwin.utilities.erwinConfiguration.DeepErwinModelConfig()
    # config.model.embed = erwinConfiguration.SimpleSchnetConfig(use=True, quality='minimal')
    config.model.embed = deeperwin.utilities.erwinConfiguration.SimpleSchnetConfig(use=True)
    config.model.slatermodel.n_determinants = 20
    config.physical.set_by_molecule_name('LiH')
    # config.evaluation.forces.calculate = True
    # config.model.embed_without_loop = False

    wf = deeperwin.main.WaveFunction(config)
    wf.logger.setLevel(logging.DEBUG)

    wf.optimize()
    wf.compile_evaluation()
    wf.evaluate()