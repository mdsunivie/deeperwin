"""
Definitions of wavefunction models (DeepErwin, baseline)
"""
from typing import Callable, Tuple, Union, List, Literal
import logging

import numpy as np
import jax.nn
from jax import numpy as jnp

from deeperwin.configuration import (
    DeepErwinModelConfig,
    InputFeatureConfig,
    EnvelopeOrbitalsConfig,
    BaselineOrbitalsConfig,
    OrbitalsConfig,
    SimpleSchnetConfig,
    PhysicalConfig,
    CASSCFConfig,
    FermiNetConfig,
    SchnetConfig,
    InitializationConfig,
    MolecularOrbitalFeaturesConfig,
EmbeddingConfigType
)
from deeperwin.orbitals import evaluate_molecular_orbitals, get_baseline_solution, get_hartree_fock_solution, split_results_into_spins, \
    eval_atomic_orbitals, fit_orbital_envelopes_to_hartree_fock, get_envelope_exponents_from_atomic_orbitals, get_envelope_exponents_hardcoded, get_envelope_exponents_cisd
from deeperwin.features import build_local_rotation_matrices
from deeperwin.utils import get_el_ion_distance_matrix, get_distance_matrix, convert_to_jnp, get_number_of_params

try:
    from deeperwin.register_curvature import register_repeated_dense
    from deeperwin.kfac_ferminet_alpha.layers_and_loss_tags import register_scale_and_shift
except ImportError:

    def register_repeated_dense(y, X, W, b):
        return y


    def register_scale_and_shift(y, inputs, has_scale, has_shift):
        return y

logger = logging.getLogger("dpe")


###############################################################################
########################## BASIC FUNCTIONALITY ################################
###############################################################################
def scale(X, params, register):
    """
    Computes Y = exp(`params`) * `X`.

    Args:
        X (array): Input
        params (array): Scale
        register (bool): Enables registration of scaling function for the K-FAC algorithm.

    """
    y = jnp.exp(params.squeeze()) * X
    if not register:
        return y
    else:
        return register_scale_and_shift(y, [X, params], has_scale=True, has_shift=False)


def dense_layer(X, W, b, register):
    """
    Computes a single dense linear layer, i.e. `W` * `X` + `b`

    Args:
        X (array): Input
        W (array): Weights
        b (array): Bias
        register (bool): Enables registration of the layer for the K-FAC algorithm.

    """
    y = jnp.dot(X, W)
    if b is not None:
        y += b
    if not register:
        return y
    else:
        return register_repeated_dense(y, X, W, b)


def ffwd_net(params, X, linear_output=False, register=True, activation='tanh'):
    """
    Computes the output of a fully-connected feed-forward neural network with a tanh non-linearity.

    Args:
        params (array): Weights and biases of the network
        X (array): Input
        linear_output (bool): If false, the non-linearity is applied to the output layer.
        register (bool): Enables registration of the single layers for the K-FAC algorithm.

    """
    activation_func = dict(tanh=jnp.tanh, softplus=jax.nn.softplus, elu=jax.nn.elu)[activation]

    for p in params[:-1]:
        X = dense_layer(X, *p, register)
        X = activation_func(X)
    X = dense_layer(X, *params[-1], register)
    if linear_output:
        return X
    else:
        return activation_func(X)


def init_ffwd_net(n_neurons, input_dim, has_output_bias=True, init_config: InitializationConfig = None):
    """
    Initializes the parameters for a fully-connected feed-forward neural network.

    Args:
        n_neurons (list[int]): Number of neurons in each layer
        input_dim (int): Input dimension
        n_parallel (int): Number of independent affine linear mappings that are initialized for each layer. Can be used to build a model that represents multiple wavefunctions.

    Returns:
        array: Initial parameters

    """
    init_config = init_config or InitializationConfig()
    params = []
    for i, output_dim in enumerate(n_neurons):
        if i > 0:
            input_dim = n_neurons[i - 1]

        if init_config.weight_scale == 'glorot':
            scale = np.sqrt(6 / (input_dim + output_dim))  # glorot
        elif init_config.weight_scale == 'glorot-input':
            scale = np.sqrt(2 / input_dim)  # variant used in FermiNet
        else:
            raise ValueError(f"Unknown type of weight initializer: {init_config.weight_scale}")

        if init_config.weight_distribution == 'normal':
            W = np.random.normal(0, scale, [input_dim, output_dim])
        elif init_config.weight_distribution == 'uniform':
            W = np.random.uniform(-scale, scale, [input_dim, output_dim])
        else:
            raise ValueError(f"Unknown type of weight distribution: {init_config.weight_distribution}")

        b = np.random.normal(0, init_config.bias_scale, [output_dim])

        params.append([jnp.array(W), jnp.array(b)])
    if (len(params) > 0) and (not has_output_bias):
        params[-1][1] = None  # remove bias parameters in last layer
    return params


###############################################################################
################### Input features / preprocessing ############################
###############################################################################

def build_preprocessor(config: InputFeatureConfig, physical_config: PhysicalConfig, init_config: InitializationConfig):
    def calculate_input_features(r, R, Z, params, fixed_params):
        diff_el_el, dist_el_el = get_distance_matrix(r)
        diff_el_ion, dist_el_ion = get_el_ion_distance_matrix(r, R)

        if config.use_local_coordinates:
            diff_el_ion = jnp.sum(fixed_params['local_rotations'] * diff_el_ion[..., None, :], axis=-1)

        features_el_el = get_pairwise_features('el_el', diff_el_el, dist_el_el, config, physical_config, params)
        features_el_ion = get_pairwise_features('el_ion', diff_el_ion, dist_el_ion, config, physical_config, params)

        features_el = []
        if config.mo_features:
            features_el.append(_get_mo_input_features(diff_el_ion, dist_el_ion, fixed_params['mo_features'], config.mo_features.normalize_orbitals))
        if config.n_one_el_features > 0:
            features_el.append(jnp.sum(ffwd_net(params['el_ion_aggregation'], features_el_ion), axis=-2))
        if config.concatenate_el_ion_features:
            # concatenate all el-ion features into a long h_one feature
            features_el.append(jnp.reshape(features_el_ion, features_el_ion.shape[:-2] + (-1,)))
        if features_el:
            features_el = jnp.concatenate(features_el, axis=-1)
        else:
            features_el = jnp.ones(r.shape[:-1] + (1,))
        return (diff_el_el, dist_el_el, diff_el_ion, dist_el_ion), (features_el, features_el_el, features_el_ion)

    initial_params = {}
    if config.n_one_el_features > 0:
        _, _, n_el_ion_features = get_nr_of_input_features(config, physical_config)
        n_neurons = config.n_hidden_one_el_features + [config.n_one_el_features]
        initial_params['el_ion_aggregation'] = init_ffwd_net(n_neurons, n_el_ion_features, init_config)

    if config.slater_exponential_factors:
        gamma = [physical_config.n_electrons, physical_config.n_electrons]
        eta = [physical_config.n_electrons, physical_config.n_ions]
        initial_params["slater_exponential_input"] = dict(gamma=jnp.ones(gamma),
                                          eta=jnp.ones(eta))

    return calculate_input_features, initial_params


def get_pairwise_features(pair_type: Literal['el_el', 'el_ion'], differences, dist, config: InputFeatureConfig, phys_config: PhysicalConfig, params):
    """
    Computes pairwise features based on particle distances.

    Args:
        differences (array): Pairwise particle differences (i.e. 3 coordinates per pair)
        dist (array): Pairwise particle distances
        config (DeepErwinModelConfig): Hyperparameters of the DeepErwin model
        dist_feat (bool): Flag that controls the usage of features of the form `dist`^n

    Returns:
        array: Pairwise distance features

    """
    features = []
    if config.use_rbf_features:
        features_rbf = _get_rbf_features(dist, config.n_rbf_features, config.use_sigma_paulinet)
        features.append(features_rbf)
    if config.use_distance_features:
        eps = config.eps_dist_feat
        if len(config.distance_feature_powers) > 0:
            features_dist = jnp.stack(
                [dist ** n if n > 0 else 1 / (dist ** (-n) + eps) for n in config.distance_feature_powers], axis=-1
            )
        features.append(features_dist)

    if (pair_type == 'el_el') and config.use_el_el_differences:
        features.append(differences)
    if (pair_type == 'el_ion') and config.use_el_ion_differences:
        features.append(differences)

    if (pair_type == 'el_el') and config.use_el_el_spin:
        el_shape = differences.shape[:-2] # [batch-dims x n_el]
        spins = np.ones(el_shape)
        spins[..., :phys_config.n_up] = 0
        spin_diff = np.abs(spins[..., None, :] - spins[..., :, None]) * 2 - 1
        features.append(spin_diff[..., None])

    if (pair_type == 'el_el') and config.slater_exponential_factors:
        features_slater_el = (1.0 / params['slater_exponential_input']['gamma']) * (1 - jnp.exp(-params['slater_exponential_input']['gamma']*dist))
        features.append(features_slater_el[..., None])
    if (pair_type == 'el_ion') and config.slater_exponential_factors:
        features_slater_ion = (1.0 / params['slater_exponential_input']['eta']) * (1 - jnp.exp(-params['slater_exponential_input']['eta']*dist))
        features.append(features_slater_ion[..., None])

    return jnp.concatenate(features, axis=-1)


def get_nr_of_input_features(config: InputFeatureConfig, physical_config: PhysicalConfig):
    """
    Calculates the feature dimensions returned from the preprocessor.

    Args:
        config: Configuration of the model

    Returns:
        tuple: Tuple of 2 integers. Nr of features for electrons and nr of features for pairs of particles (el-el or el-ion)
    """

    n_el_el = config.use_rbf_features * config.n_rbf_features + config.use_distance_features * 1 + config.use_el_el_differences * 3 + config.use_el_el_spin + config.slater_exponential_factors * 1
    n_el_ion = config.use_rbf_features * config.n_rbf_features + config.use_distance_features * 1 + config.use_el_ion_differences * 3 + config.slater_exponential_factors * 1

    n_el = config.n_one_el_features
    if config.mo_features:
        n_el += config.mo_features.n_occ + config.mo_features.n_unocc
    if config.concatenate_el_ion_features:
        n_ions = len(physical_config.Z)
        n_el += n_ions * n_el_ion
    if n_el == 0:
        n_el = 1  # constant input-feature
    return n_el, n_el_el, n_el_ion


def _get_rbf_features(dist, n_features, use_sigma_paulinet):
    """
    Computes radial basis features based on Gaussians with different means from pairwise distances. This can be interpreted as a special type of "one-hot-encoding" for the distance between two particles.

    Args:
        dist (array): Pairwise particle distances
        n_features (int): Number of radial basis features
        use_sigma_paulinet (bool): Flag that controls computation of the sigma parameter of the Gaussians

    Returns:
        array: Pairwise radial basis features

    """
    r_rbf_max = 5.0
    q = jnp.linspace(0, 1.0, n_features)
    mu = q ** 2 * r_rbf_max

    if use_sigma_paulinet:
        sigma = (1 / 7) * (1 + r_rbf_max * q)
    else:
        sigma = r_rbf_max / (n_features - 1) * (2 * q + 1 / (n_features - 1))

    dist = dist[..., jnp.newaxis]  # add dimension for features
    return dist ** 2 * jnp.exp(-dist - ((dist - mu) / sigma) ** 2)


def init_fixed_params_for_input_features(config: MolecularOrbitalFeaturesConfig, physical_config: PhysicalConfig):
    ao, hf = get_hartree_fock_solution(physical_config, config.basis_set)
    n_orbitals = len(ao)
    mo_coeffs, energies, occupations = split_results_into_spins(hf)
    mo_coeffs = np.concatenate(mo_coeffs, axis=1)
    energies = np.concatenate(energies)
    occupations = np.concatenate(occupations)
    ind_sort = np.argsort(energies)
    mo_coeffs = mo_coeffs[:, ind_sort]
    occupations = occupations[ind_sort]
    energies = energies[ind_sort]
    n_ao_orbitals, n_mo_orbitals = mo_coeffs.shape

    n_occ = int(np.sum(occupations))
    ind_min = n_occ - config.n_occ
    ind_max = n_occ + config.n_unocc
    selected_mo_coeffs = mo_coeffs[:, max(ind_min, 0):min(ind_max, n_mo_orbitals)]

    # If there are not enough orbitals, pad it with zeros
    if ind_min < 0:
        selected_mo_coeffs = np.concatenate([np.zeros([n_ao_orbitals, -ind_min]), selected_mo_coeffs], axis=1)
    if ind_max >= n_mo_orbitals:
        selected_mo_coeffs = np.concatenate([selected_mo_coeffs, np.zeros([n_ao_orbitals, ind_max - n_orbitals + 1])], axis=1)
    return dict(ao=ao, mo_coeffs=selected_mo_coeffs)


def _get_mo_input_features(el_ion_diff, el_ion_dist, fixed_params, normalize):
    ao_params = fixed_params['ao']
    aos = eval_atomic_orbitals(el_ion_diff, el_ion_dist, ao_params)
    mo_features = aos @ fixed_params['mo_coeffs']

    if normalize:
        mo_features /= (1 + np.linalg.norm(mo_features, axis=-1, keepdims=True))
    return mo_features


###############################################################################
############################# Embedding #######################################
###############################################################################


def build_embedding(config: EmbeddingConfigType, physical_config: PhysicalConfig, init_config: InitializationConfig, n_input_features):
    if config is None:
        return None, None
    n_el, n_up, n_ions = physical_config.n_electrons, physical_config.n_up, len(physical_config.Z)

    _build = dict(simple_schnet=_build_simple_schnet,
                  fermi=_build_ferminet_embedding,
                  schnet=_build_schnet)[config.name]
    return _build(config, init_config, n_el, n_up, n_input_features)


def get_embedding_dim(config: EmbeddingConfigType):
    if config is None:
        return 0,0,0
    if config.name in ['schnet', 'simple_schnet']:
        n_el = config.embedding_dim
        n_el_el = config.embedding_dim
        n_el_ion = config.embedding_dim
    elif config.name == 'fermi':
        n_el = config.n_hidden_one_el[-1]
        n_el_el = config.n_hidden_two_el[-1]
        if config.use_el_ion_stream:
            n_el_ion = config.n_hidden_el_ions[-1]
        else:
            n_el_ion = n_el
    return n_el, n_el_el, n_el_ion


def _mean_or_zero(x, axis):
    mean = jnp.mean(x, axis=axis)
    if x.shape[axis] == 0:
        mean = jnp.zeros_like(mean)
    return mean

def _build_ferminet_embedding(config: FermiNetConfig, init_config: InitializationConfig, n_el, n_up, n_input_features):
    n_features_el, n_features_el_el, n_features_el_ion = n_input_features
    n_dn = n_el - n_up

    def _construct_schnet_features(h_one, h_el_el, h_el_ion, ion_embeddings, params):
        batch_dims = h_one.shape[:-2]

        h_mapped = ffwd_net(params["h_map"], h_one, linear_output=config.use_linear_out) # mapping from lower dim to same dim of h_el_el
        if (not config.use_w_mapping) and (not config.use_h_two_same_diff):
            # Simple case where we do not differentiate between same and different at all (not in input streams and not in the mappings)
            embeddings_el_el = jnp.sum(h_el_el * h_mapped[..., None, :, :], axis=-2)
        else:
            if config.use_h_two_same_diff:
                w_same, w_diff = h_el_el
            else:
                w_same = jnp.concatenate([h_el_el[..., :n_up, :n_up, :].reshape(batch_dims + (n_up * n_up, -1)),
                                          h_el_el[..., n_up:, n_up:, :].reshape(batch_dims + (n_dn * n_dn, -1))], axis=-2)
                w_diff = jnp.concatenate([h_el_el[..., :n_up, n_up:, :].reshape(batch_dims + (n_up * n_dn, -1)),
                                          h_el_el[..., n_up:, :n_up, :].reshape(batch_dims + (n_dn * n_up, -1))], axis=-2)

            if config.use_w_mapping:
                w_same = ffwd_net(params['w_map_same'], w_same, linear_output=config.use_linear_out)
                w_diff = ffwd_net(params['w_map_diff'], w_diff, linear_output=config.use_linear_out)

            w_uu = w_same[..., :n_up * n_up, :].reshape(batch_dims + (n_up, n_up, -1))
            w_ud = w_diff[..., :n_up * n_dn, :].reshape(batch_dims + (n_up, n_dn, -1))
            w_du = w_diff[..., n_up * n_dn:, :].reshape(batch_dims + (n_dn, n_up, -1))
            w_dd = w_same[..., n_up * n_up:, :].reshape(batch_dims + (n_dn, n_dn, -1))
            h_u = h_mapped[..., None, :n_up, :]
            h_d = h_mapped[..., None, n_up:, :]
            emb_up = jnp.sum(w_uu * h_u, axis=-2) + jnp.sum(w_ud * h_d, axis=-2)
            emb_dn = jnp.sum(w_du * h_u, axis=-2) + jnp.sum(w_dd * h_d, axis=-2)
            embeddings_el_el = jnp.concatenate([emb_up, emb_dn], axis=-2)

        if config.use_el_ion_stream:
            h_ion = ffwd_net(params['h_ion'], ion_embeddings)
            embeddings_el_ions = jnp.sum(h_el_ion * h_ion, axis=-2)
            return embeddings_el_el, embeddings_el_ions
        else:
            return embeddings_el_el


    def _construct_symmetric_features(h_one, h_el_el, h_el_ion, ion_embeddings, params):
        if config.use_h_one:
            features = [h_one]
        else:
            features = []

        # Average over all h_ones from 1-el-stream
        if config.use_average_h_one:
            g_one = [jnp.mean(h_one[..., :n_up, :], keepdims=True, axis=-2), jnp.mean(h_one[..., n_up:, :], keepdims=True, axis=-2)]
            features += [jnp.tile(el, [n_el, 1]) for el in g_one]

        # Average over 2-el-stream
        if config.use_average_h_two:
            assert not config.use_h_two_same_diff
            f_pairs_with_up = jnp.mean(h_el_el[..., :n_up, :], axis=-2)
            f_pairs_with_dn = jnp.mean(h_el_el[..., n_up:, :], axis=-2)
            features += [f_pairs_with_up, f_pairs_with_dn]

        # Average of el-ion-stream
        if config.use_el_ion_stream and not config.use_schnet_features:
            features.append(jnp.mean(h_el_ion, axis=-2))

        if config.use_schnet_features:
            if config.use_el_ion_stream:
                h_mult_el_el, h_mult_el_ion = _construct_schnet_features(h_one, h_el_el, h_el_ion, ion_embeddings, params)
                features += [h_mult_el_el, h_mult_el_ion]
            else:
                h_mult_el_el = _construct_schnet_features(h_one, h_el_el, h_el_ion, ion_embeddings,
                                                                         params)
                features += [h_mult_el_el]
        return jnp.concatenate(features, axis=-1)

    residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y


    def _build_h_two_same_diff_featurse(features_el_el):
        batch_dims = features_el_el.shape[:-3]
        h_uu = features_el_el[..., :n_up, :n_up, :].reshape(batch_dims + (n_up * n_up, -1))
        h_ud = features_el_el[..., :n_up, n_up:, :].reshape(batch_dims + (n_up * n_dn, -1))
        h_du = features_el_el[..., n_up:, :n_up, :].reshape(batch_dims + (n_dn * n_up, -1))
        h_dd = features_el_el[..., n_up:, n_up:, :].reshape(batch_dims + (n_dn * n_dn, -1))
        return [jnp.concatenate([h_uu, h_dd], axis=-2), jnp.concatenate([h_ud, h_du], axis=-2)]


    def _call_embed(features_el, features_el_el, features_el_ion, Z, params):
        h_one = features_el
        h_el_ion = features_el_ion
        if config.use_h_two_same_diff:
            h_two = _build_h_two_same_diff_featurse(features_el_el)
        else:
            h_two = features_el_el

        if config.use_schnet_features and config.use_el_ion_stream:
            Z = jnp.transpose(jnp.tile(Z[:, jnp.newaxis], features_el_el.shape[:-3]))
            Z = jnp.reshape(Z, features_el_el.shape[:-3] + (Z.shape[-1], 1))
            ion_embeddings = ffwd_net(params["ion_emb"], Z, linear_output=False, register=True)
            ion_embeddings = jnp.reshape(ion_embeddings[..., jnp.newaxis, :], Z.shape[:-2] + (1, Z.shape[-2], config.n_hidden_two_el[0]))
        else:
            ion_embeddings = None
        for i, (params_one, params_two) in enumerate(zip(params["one_el"][:-1], params['two_el'])):
            h_one_in = _construct_symmetric_features(h_one, h_two, h_el_ion, ion_embeddings, params['schnet_features'][i]
            if config.use_schnet_features else None)

            h_one = residual(h_one_in, ffwd_net(params_one, h_one_in))
            if config.use_h_two_same_diff:
                h_two[0] = residual(h_two[0], ffwd_net(params_two['same'], h_two[0]))
                h_two[1] = residual(h_two[1], ffwd_net(params_two['diff'], h_two[1]))
            else:
                h_two = residual(h_two, ffwd_net(params_two, h_two))

            if (config.use_el_ion_stream and config.use_schnet_features) or \
                    (config.use_el_ion_stream and not config.use_schnet_features):
                h_el_ion = residual(h_el_ion, ffwd_net(params["el_ion"][i], h_el_ion))

        # We have one more 1-electron layer than 2-electron layers: Now apply the last 1-electron layer
        h_one_in = _construct_symmetric_features(h_one, h_two, h_el_ion, ion_embeddings if config.use_schnet_features else None, params['schnet_features'][-1] if config.use_schnet_features else None)
        h_one = residual(h_one_in, ffwd_net(params['one_el'][-1], h_one_in))
        if not config.use_el_ion_stream:
            h_el_ion = h_one[..., jnp.newaxis, :]

        return h_one, h_two, h_el_ion

    # First layer: h, mean(h_up_down), mean(h_up_up), mean(h_up), mean(h_down), mean(h_el_ion), sum(w_el * h_el) + sum(w_ion * h_ion) or sum(w_el * h_el) & sum(w_ion * h_ion)

    boolean_el_ion = 1 if config.use_schnet_features and config.use_el_ion_stream else 0
    one_el_inp_shape = [n_features_el +
                        2 * n_features_el_el * config.use_average_h_two +
                        2 * n_features_el * config.use_average_h_one +
                        n_features_el_ion * config.use_el_ion_stream * (not config.use_schnet_features) +
                        (config.emb_dim * config.use_schnet_features * config.sum_schnet_features * config.use_w_mapping +
                         n_features_el_el * config.use_schnet_features * config.sum_schnet_features * (not config.use_w_mapping)) +
                        ((config.emb_dim + boolean_el_ion*n_features_el_ion) * config.use_schnet_features * (not config.sum_schnet_features) * config.use_w_mapping +
                         (n_features_el_el + boolean_el_ion*n_features_el_ion) * config.use_schnet_features * (not config.sum_schnet_features) * (not config.use_w_mapping))]

    # Other layers:  h, mean(h_up_down), mean(h_up_up), mean(h_up), mean(h_down), sum(w_el * h_el) + sum(w_ion * h_ion) or sum(w_el * h_el) & sum(w_ion * h_ion)
    one_el_inp_shape += [n_el +
                         2 * n_el_el * config.use_average_h_two +
                         2 * n_el * config.use_average_h_one +
                         (config.emb_dim*config.use_schnet_features*config.sum_schnet_features*config.use_w_mapping + n_el_el*config.use_schnet_features*config.sum_schnet_features*(not config.use_w_mapping))  +
                         ((1+boolean_el_ion)*config.emb_dim*config.use_schnet_features*(not config.sum_schnet_features)*config.use_w_mapping + (1+boolean_el_ion)*n_el_el*config.use_schnet_features*(not config.sum_schnet_features)*(not config.use_w_mapping))
                         for n_el, n_el_el in zip(config.n_hidden_one_el[:-1], config.n_hidden_two_el)]

    # Add input size for el_ion stream
    if config.use_el_ion_stream and not config.use_schnet_features:
        for i, n_el_ion in enumerate(config.n_hidden_el_ions):
            one_el_inp_shape[i + 1] += n_el_ion

    two_el_inp_shape = [n_features_el_el] + config.n_hidden_two_el[:-1]

    embed_params = {}
    embed_params['one_el'] = [init_ffwd_net([n_out], n_in, True, init_config) for n_in, n_out in
                              zip(one_el_inp_shape, config.n_hidden_one_el)]

    if config.use_average_h_two or config.use_schnet_features:
        if config.use_h_two_same_diff:
            embed_params['two_el'] = [{'same': init_ffwd_net([n_out], n_in, True, init_config),
                                       'diff': init_ffwd_net([n_out], n_in, True, init_config)}
                                      for n_in, n_out in zip(two_el_inp_shape, config.n_hidden_two_el)]
        else:
            embed_params['two_el'] = [init_ffwd_net([n_out], n_in, True, init_config) for n_in, n_out in
                                  zip(two_el_inp_shape, config.n_hidden_two_el)]

    if config.use_schnet_features:
        if config.use_el_ion_stream:
            embed_params['ion_emb'] = init_ffwd_net([config.n_hidden_two_el[0]], 1, True, init_config)

        one_el_stream = [n_features_el] + config.n_hidden_one_el[:-1]
        two_el_stream = two_el_inp_shape[:1] + config.n_hidden_two_el
        ion_stream = [n_features_el_ion] + config.n_hidden_el_ions

        embed_params['schnet_features'] = []
        for n_in_el, n_in_el_el, n_in_ion in zip(one_el_stream, two_el_stream, ion_stream):
            layer_it = {}
            emb_dim = config.emb_dim if config.use_w_mapping else n_in_el_el
            layer_it['h_map'] = init_ffwd_net([emb_dim], n_in_el, True, init_config)
            layer_it['h_ion'] = init_ffwd_net([n_in_ion], config.n_hidden_two_el[0], True, init_config)
            if config.use_w_mapping:
                layer_it['w_map_same'] = init_ffwd_net([emb_dim], n_in_el_el, True, init_config)
                layer_it['w_map_diff'] = init_ffwd_net([emb_dim], n_in_el_el, True, init_config)
            embed_params['schnet_features'].append(layer_it)


    if (config.use_el_ion_stream and config.use_schnet_features) or (config.use_el_ion_stream and not config.use_schnet_features):
        el_ion_inp_shape = [n_features_el_ion] + config.n_hidden_el_ions[:-1]
        embed_params['el_ion'] = [init_ffwd_net([n_out], n_in, True, init_config) for n_in, n_out in
                                  zip(el_ion_inp_shape, config.n_hidden_el_ions)]  # config.n_hidden_el_ion stream
    return _call_embed, embed_params


def _build_simple_schnet(config: SimpleSchnetConfig, init_config: InitializationConfig, n_el, n_up, n_input_features):
    """
    Builds the electron coordinate embedding of the DeepErwin wavefunction model and initializes the respective trainable parameters.

    Args:
        config (SimpleSchnetConfig): Hyperparameters of the electron coordinate embedding
        n_el (int): Number of electrons
        n_up (int): Number of up-spin electrons
        n_features_pair (int): Input feature dimension
        name (str): Name of the embedding instance

    Returns:
        A tuple (name, call, params) where call is a callable representing the built electron embedding. The dictionary params contains the initial trainable parameters.

    """
    n_features_el, n_features_el_el, n_features_el_ion = n_input_features

    n_dn = n_el - n_up
    indices_u_u = np.array([[j for j in range(n_up) if j != i] for i in range(n_up)], dtype=int)
    indices_d_d = np.array([[j + n_up for j in range(n_dn) if j != i] for i in range(n_dn)], dtype=int)
    emb_dim = config.embedding_dim

    def _call_embed(features_el, features_el_el, features_el_ion, Z, params):
        f_pairs_u_u = features_el_el[..., :n_up, : n_up, :]
        f_pairs_d_d = features_el_el[..., n_up:n_el, n_up: n_el, :]
        f_pairs_u_d = features_el_el[..., :n_up, n_up: n_el, :]
        f_pairs_d_u = features_el_el[..., n_up:n_el, :n_up, :]

        f_pairs_u_u = jnp.reshape(f_pairs_u_u, f_pairs_u_u.shape[:-3] + (n_up**2, n_features_el_el))
        f_pairs_d_d = jnp.reshape(f_pairs_d_d, f_pairs_d_d.shape[:-3] + (n_dn**2, n_features_el_el))
        f_pairs_u_d = jnp.reshape(f_pairs_u_d, f_pairs_u_d.shape[:-3] + (n_up * n_dn, n_features_el_el))
        f_pairs_d_u = jnp.reshape(f_pairs_d_u, f_pairs_d_u.shape[:-3] + (n_up * n_dn, n_features_el_el))
        f_pairs_same = jnp.concatenate([f_pairs_u_u, f_pairs_d_d], axis=-2)
        f_pairs_diff = jnp.concatenate([f_pairs_u_d, f_pairs_d_u], axis=-2)

        Z = jnp.transpose(jnp.tile(Z[:, jnp.newaxis], features_el_el.shape[:-3]))
        Z = jnp.reshape(Z, features_el_el.shape[:-3] + (Z.shape[-1], 1))
        ion_embeddings = ffwd_net(params["ion_emb"], Z, linear_output=False, register=True)
        ion_embeddings = jnp.reshape(ion_embeddings[..., jnp.newaxis, :], Z.shape[:-2] + (1, Z.shape[-2], emb_dim))
        x = features_el
        for n in range(config.n_iterations):
            h_same = ffwd_net(params["h_same"][n], x, linear_output=False)
            h_diff = ffwd_net(params["h_diff"][n], x, linear_output=False)

            h_u_u = h_same[..., jnp.newaxis, :n_up, :]
            h_d_d = h_same[..., jnp.newaxis, n_up:, :]
            h_u_d = h_diff[..., jnp.newaxis, n_up:, :]
            h_d_u = h_diff[..., jnp.newaxis, :n_up, :]

            # h_u_u = h_same[..., indices_u_u, :]
            # h_d_d = h_same[..., indices_d_d, :]
            # h_u_d = h_diff[..., jnp.newaxis, n_up:, :]
            # h_d_u = h_diff[..., jnp.newaxis, :n_up, :]

            if n == 0 or not config.deep_w:
                w_same = ffwd_net(params["w_same"][n], f_pairs_same, linear_output=config.use_linear_layer_w)
                w_diff = ffwd_net(params["w_diff"][n], f_pairs_diff, linear_output=config.use_linear_layer_w)
            else:
                w_same = ffwd_net(params["w_same"][n], w_same, linear_output=config.use_linear_layer_w)
                w_diff = ffwd_net(params["w_diff"][n], w_diff, linear_output=config.use_linear_layer_w)

            batch_dims = w_same.shape[:-2]
            w_u_u = w_same[..., :n_up * n_up, :].reshape(batch_dims + (n_up, n_up, -1))
            w_u_d = w_diff[..., :n_up * n_dn, :].reshape(batch_dims + (n_up, n_dn, -1))
            w_d_u = w_diff[..., n_up * n_dn:, :].reshape(batch_dims + (n_dn, n_up, -1))
            w_d_d = w_same[..., n_up * n_up:, :].reshape(batch_dims + (n_dn, n_dn, -1))

            # w_u_u = jnp.reshape(w_same[..., : (n_up * (n_up - 1)), :], w_same.shape[:-2] + (n_up, n_up - 1, emb_dim))
            # w_d_d = jnp.reshape(w_same[..., (n_up * (n_up - 1)):, :], w_same.shape[:-2] + (n_dn, n_dn - 1, emb_dim))
            # w_u_d = jnp.reshape(w_diff[..., : (n_up * n_dn), :], w_diff.shape[:-2] + (n_up, n_dn, emb_dim))
            # w_d_u = jnp.reshape(w_diff[..., (n_up * n_dn):, :], w_diff.shape[:-2] + (n_dn, n_up, emb_dim))

            if n == 0 or not config.deep_w:
                w_el_ions = ffwd_net(params["w_el_ions"][n], features_el_ion, linear_output=config.use_linear_layer_w)
            else:
                w_el_ions = ffwd_net(params["w_el_ions"][n], w_el_ions, linear_output=config.use_linear_layer_w)

            embeddings_el_el = jnp.concatenate([
                jnp.concatenate([w_u_u * h_u_u, w_u_d * h_u_d], axis=-2),
                jnp.concatenate([w_d_u * h_d_u, w_d_d * h_d_d], axis=-2)
            ], axis=-3)

            embeddings_el_ions = w_el_ions * ion_embeddings

            x = jnp.sum(embeddings_el_el, axis=-2) + jnp.sum(embeddings_el_ions, axis=-2)

            # if config.one_el_input:
            #     x = jnp.concatenate([h_same, h_diff, x], axis=-1)
            # if config.use_res_net:
            #     x = x + ffwd_net(params["g_func"][n], x, linear_output=False)
            # else:
            x = ffwd_net(params["g_func"][n], x, linear_output=False)

        return x, embeddings_el_el, embeddings_el_ions

    embed_params = {}
    shape_w = config.n_hidden_w + [emb_dim]
    shape_h = config.n_hidden_h + [emb_dim]
    shape_g = config.n_hidden_g + [emb_dim]

    if config.deep_w:
        embed_params["w_same"] = [
            init_ffwd_net(shape_w, n_features_el_el, True, init_config) if i == 0 else init_ffwd_net(shape_w, emb_dim, True, init_config) for
            i in range(config.n_iterations)]
        embed_params["w_diff"] = [
            init_ffwd_net(shape_w, n_features_el_el, True, init_config) if i == 0 else init_ffwd_net(shape_w, emb_dim, True, init_config) for
            i in range(config.n_iterations)]
        embed_params["w_el_ions"] = [
            init_ffwd_net(shape_w, n_features_el_ion, True, init_config) if i == 0 else init_ffwd_net(shape_w, emb_dim, True, init_config) for
            i in
            range(config.n_iterations)]

    else:
        embed_params["w_same"] = [init_ffwd_net(shape_w, n_features_el_el, True, init_config) for _ in range(config.n_iterations)]
        embed_params["w_diff"] = [init_ffwd_net(shape_w, n_features_el_el, True, init_config) for _ in range(config.n_iterations)]
        embed_params["w_el_ions"] = [init_ffwd_net(shape_w, n_features_el_ion, True, init_config) for _ in
                                     range(config.n_iterations)]

    embed_params["h_same"] = [init_ffwd_net(shape_h, n_features_el if i == 0 else emb_dim, True, init_config) for i in
                              range(config.n_iterations)]
    embed_params["h_diff"] = [init_ffwd_net(shape_h, n_features_el if i == 0 else emb_dim, True, init_config) for i in
                              range(config.n_iterations)]
    embed_params["ion_emb"] = init_ffwd_net([shape_h[-1]], 1, True, init_config)
    embed_params["g_func"] = [
        init_ffwd_net(shape_g, emb_dim, True, init_config) if not config.one_el_input else init_ffwd_net(shape_g, 3 * emb_dim, True,
                                                                                                         init_config) for _ in
        range(config.n_iterations)]
    return _call_embed, embed_params


def _build_schnet(config: SchnetConfig, init_config: InitializationConfig, n_el, n_up, n_input_features):
    n_features_el, n_features_el_el, n_features_el_ion = n_input_features

    n_dn = n_el - n_up
    indices_u_u = np.array([[j for j in range(n_up) if j != i] for i in range(n_up)], dtype=int)
    indices_d_d = np.array([[j + n_up for j in range(n_dn) if j != i] for i in range(n_dn)], dtype=int)
    emb_dim = config.embedding_dim

    if config.use_res_net:
        residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y
    else:
        residual = lambda x, y: y

    def _call_embed(features_el, features_el_el, features_el_ion, Z, params):
        f_pairs_u_u = features_el_el[..., :n_up, : n_up - 1, :]
        f_pairs_d_d = features_el_el[..., n_up:n_el, n_up: n_el - 1, :]
        f_pairs_u_d = features_el_el[..., :n_up, n_up - 1: n_el - 1, :]
        f_pairs_d_u = features_el_el[..., n_up:n_el, :n_up, :]

        f_pairs_u_u = jnp.reshape(f_pairs_u_u, f_pairs_u_u.shape[:-3] + (n_up * (n_up - 1), n_features_el_el))
        f_pairs_d_d = jnp.reshape(f_pairs_d_d, f_pairs_d_d.shape[:-3] + (n_dn * (n_dn - 1), n_features_el_el))
        f_pairs_u_d = jnp.reshape(f_pairs_u_d, f_pairs_u_d.shape[:-3] + (n_up * n_dn, n_features_el_el))
        f_pairs_d_u = jnp.reshape(f_pairs_d_u, f_pairs_d_u.shape[:-3] + (n_up * n_dn, n_features_el_el))
        f_pairs_same = jnp.concatenate([f_pairs_u_u, f_pairs_d_d], axis=-2)
        f_pairs_diff = jnp.concatenate([f_pairs_u_d, f_pairs_d_u], axis=-2)

        Z = jnp.transpose(jnp.tile(Z[:, jnp.newaxis], features_el_el.shape[:-3]))
        Z = jnp.reshape(Z, features_el_el.shape[:-3] + (Z.shape[-1], 1))
        ion_embeddings = ffwd_net(params["ion_emb"], Z, linear_output=False, register=True)
        ion_embeddings = jnp.reshape(ion_embeddings[..., jnp.newaxis, :], Z.shape[:-2] + (1, Z.shape[-2], emb_dim))

        x = features_el
        h = ffwd_net(params["h"][0], x, linear_output=False)
        for n in range(config.n_iterations):
            h_u_u = h[..., indices_u_u, :]
            h_d_d = h[..., indices_d_d, :]
            h_u_d = h[..., jnp.newaxis, n_up:, :]
            h_d_u = h[..., jnp.newaxis, :n_up, :]

            if n == 0:
                w_same = ffwd_net(params["w_same"][n], f_pairs_same, linear_output=config.use_linear_layer_w)
                w_diff = ffwd_net(params["w_diff"][n], f_pairs_diff, linear_output=config.use_linear_layer_w)
            else:
                w_same = ffwd_net(params["w_same"][n], w_same, linear_output=config.use_linear_layer_w)
                w_diff = ffwd_net(params["w_diff"][n], w_diff, linear_output=config.use_linear_layer_w)

            w_u_u = jnp.reshape(w_same[..., : (n_up * (n_up - 1)), :], w_same.shape[:-2] + (n_up, n_up - 1, emb_dim))
            w_d_d = jnp.reshape(w_same[..., (n_up * (n_up - 1)):, :], w_same.shape[:-2] + (n_dn, n_dn - 1, emb_dim))
            w_u_d = jnp.reshape(w_diff[..., : (n_up * n_dn), :], w_diff.shape[:-2] + (n_up, n_dn, emb_dim))
            w_d_u = jnp.reshape(w_diff[..., (n_up * n_dn):, :], w_diff.shape[:-2] + (n_dn, n_up, emb_dim))

            if n == 0:
                w_el_ions = ffwd_net(params["w_el_ions"][n], features_el_ion, linear_output=config.use_linear_layer_w)
            else:
                w_el_ions = ffwd_net(params["w_el_ions"][n], w_el_ions, linear_output=config.use_linear_layer_w)

            embeddings_el_el = jnp.concatenate([
                jnp.concatenate([w_u_u * h_u_u, w_u_d * h_u_d], axis=-2),
                jnp.concatenate([w_d_u * h_d_u, w_d_d * h_d_d], axis=-2)
            ], axis=-3)

            embeddings_el_ions = w_el_ions * ion_embeddings
            embeddings_el_el_sum = jnp.sum(embeddings_el_el, axis=-2)
            embeddings_el_ions_sum = jnp.sum(embeddings_el_ions, axis=-2)

            if config.one_el_input:
                if config.use_concatenation:
                    x = jnp.concatenate([h, embeddings_el_el_sum, embeddings_el_ions_sum], axis=-1)
                else:
                    x = jnp.sum(embeddings_el_el, axis=-2) + jnp.sum(embeddings_el_ions, axis=-2)
                    x = jnp.concatenate([h, x], axis=-1)
            else:
                if config.use_concatenation:
                    x = jnp.concatenate([embeddings_el_el_sum, embeddings_el_ions_sum], axis=-1)
                else:
                    x = jnp.sum(embeddings_el_el, axis=-2) + jnp.sum(embeddings_el_ions, axis=-2)

            h_next = ffwd_net(params["h"][n + 1], x, linear_output=False)
            h = residual(h, h_next)

        return h, embeddings_el_el, embeddings_el_ions

    embed_params = {}
    shape_w = config.n_hidden_w + [emb_dim]
    shape_h = config.n_hidden_h + [emb_dim]

    embed_params["w_same"] = [
        init_ffwd_net(shape_w, n_features_el_el, True, init_config) if i == 0 else init_ffwd_net(shape_w,
                                                                                                emb_dim,
                                                                                                True,
                                                                                                init_config)
        for i in range(config.n_iterations)]
    embed_params["w_diff"] = [
        init_ffwd_net(shape_w, n_features_el_el, True, init_config) if i == 0 else init_ffwd_net(shape_w,
                                                                                                emb_dim,
                                                                                                True,
                                                                                                init_config)
        for i in range(config.n_iterations)]
    embed_params["w_el_ions"] = [
        init_ffwd_net(shape_w, n_features_el_el, True, init_config) if i == 0 else init_ffwd_net(shape_w,
                                                                                                emb_dim,
                                                                                                True,
                                                                                                init_config)
        for i in
        range(config.n_iterations)]

    if config.one_el_input:
        if config.use_concatenation:
            embed_params["h"] = [
                init_ffwd_net(shape_h, n_features_el if i == 0 else 3 * emb_dim, True, init_config)
                for i in range(config.n_iterations + 1)]
        else:
            embed_params["h"] = [init_ffwd_net(shape_h, n_features_el if i == 0 else 2 * emb_dim, True, init_config)
                                 for i in range(config.n_iterations + 1)]
    else:
        if config.use_concatenation:
            embed_params["h"] = [
                init_ffwd_net(shape_h, n_features_el if i == 0 else 2 * emb_dim, True, init_config)
                for i in range(config.n_iterations + 1)]
        else:
            embed_params["h"] = [init_ffwd_net(shape_h, n_features_el if i == 0 else emb_dim, True, init_config)
                                 for i in range(config.n_iterations + 1)]

    embed_params["ion_emb"] = init_ffwd_net(shape_h, 1, True, init_config)

    return _call_embed, embed_params


###############################################################################
############################## Backflow #######################################
###############################################################################
def calculate_shift_decay(d_el_ion, Z, decaying_parameter):
    """
    Computes the scaling factor ensuring that the contribution of the backflow shift decays in the proximity of a nucleus.

    Args:
        d_el_ion (array): Pairwise electron-ion distances
        Z (array): Nuclear charges
        decaying_parameter (array): Decaying parameters (same length as `Z`)

    Returns:
        array: Scalings for each electron

    """
    scale = decaying_parameter / Z
    scaling = jnp.prod(jnp.tanh((d_el_ion / scale) ** 2), axis=-1)
    return scaling


def build_backflow_shift(config: BaselineOrbitalsConfig, init_config: InitializationConfig, emb_dims: Tuple[int]):
    """
    Builds the backflow shift of the DeepErwin wavefunction model and initializes the respective trainable parameters.

    Args:
        config (DeepErwinModelConfig): Hyperparameters of the DeepErwin model
        n_el (int): Number of electrons
        name (str): Name of the backflow shift instance

    Returns:
        A tuple (name, call, params) where call is a callable representing the built backflow shift. The dictionary params contains the initial trainable parameters.
    """

    def _calc_shift(x, pair_embedding, nn_params, diff, dist):
        n_particles = diff.shape[-2]
        x_tiled = jnp.tile(jnp.expand_dims(x, axis=-2), (n_particles, 1))
        features = jnp.concatenate([x_tiled, pair_embedding], axis=-1)
        shift = ffwd_net(nn_params, features, linear_output=True)
        shift_weights = shift / (1 + dist[..., jnp.newaxis] ** 3)
        return jnp.sum(shift_weights * diff, axis=-2)

    def _call_bf_shift(diff_el_el, dist_el_el, diff_el_ion, dist_el_ion, emb_el, emb_el_el, emb_el_ion, Z, params):
        shift_towards_electrons = _calc_shift(emb_el, emb_el_el, params["w_el"], diff_el_el, dist_el_el)
        shift_towards_ions = _calc_shift(emb_el, emb_el_ion, params["w_ion"], diff_el_ion, dist_el_ion)
        if config.use_trainable_shift_decay_radius:
            shift_decay = calculate_shift_decay(dist_el_ion, Z, params["scale_decay"])
        else:
            shift_decay = calculate_shift_decay(dist_el_ion, Z, 1.0)
        shift = (shift_towards_electrons + shift_towards_ions) * shift_decay[..., jnp.newaxis]
        if config.use_trainable_scales:
            shift = scale(shift, params["scale_el"], register=config.register_scale)
        return shift

    bf_params = {}
    emb_dim_el, emb_dim_el_el, emb_dim_el_ion = emb_dims
    input_dim_el_el = emb_dim_el + emb_dim_el_el
    input_dim_el_ion = emb_dim_el + emb_dim_el_el

    bf_params["w_el"] = init_ffwd_net(config.n_hidden_bf_shift + [config.output_shift], input_dim_el_el, True, init_config)
    bf_params["w_ion"] = init_ffwd_net(config.n_hidden_bf_shift + [config.output_shift], input_dim_el_ion, True, init_config)

    if config.use_trainable_scales:
        bf_params["scale_el"] = jnp.array([-3.5])
    if config.use_trainable_shift_decay_radius:
        bf_params["scale_decay"] = jnp.array([0.5])
    return _call_bf_shift, bf_params


def build_backflow_factor(config: BaselineOrbitalsConfig, init_config: InitializationConfig, n_electrons, n_up, n_dets,
                          emb_dims: Tuple[int]):
    """
    Builds the backflow factor of the DeepErwin wavefunction model and initializes the respective model parameters. Note that this function yields a single callable but specific initial parameters for different determinants, orbitals, and spins.

    Args:
        config (DeepErwinModelConfig): Hyperparameters of the DeepErwin model
        n_electrons (int): Number of electrons
        n_up (int): Number of up-spin electrons
        name (str): Name of the backflow factor instance

    Returns:
        A tuple (name, call, params) where call is a callable representing the built backflow factor. The dictionary params contains the initial trainable parameters.

    """
    n_dn = n_electrons - n_up
    input_dim = emb_dims[0]  # Only the elctron embedding will be used

    def _call_bf_factor(embeddings, params):
        embeddings_up = embeddings[..., :n_up, :]
        embeddings_dn = embeddings[..., n_up:, :]

        if len(config.n_hidden_bf_factor) != 0:
            embeddings_up = ffwd_net(params["general"]["up"], embeddings_up, linear_output=False)
            embeddings_dn = ffwd_net(params["general"]["dn"], embeddings_dn, linear_output=False)

        bf_up = ffwd_net(params["orbital"]["up_output"], embeddings_up, linear_output=True)
        bf_dn = ffwd_net(params["orbital"]["dn_output"], embeddings_dn, linear_output=True)

        bf_up = jnp.reshape(bf_up, bf_up.shape[:-2] + (n_dets * n_up * n_up,))
        bf_dn = jnp.reshape(bf_dn, bf_dn.shape[:-2] + (n_dets * n_dn * n_dn,))

        bf = jnp.concatenate([bf_up, bf_dn], axis=-1)
        if config.use_trainable_scales:
            bf = scale(bf, params["general"]["scale"], register=config.register_scale)
        bf = config.bf_factor_constant_bias + bf

        # output-shape: [batch x n_up x n_dets x n_up_orb]
        bf_up = jnp.reshape(bf[..., :n_dets * n_up * n_up], bf_up.shape[:-1] + (n_up, n_dets, n_up))
        bf_dn = jnp.reshape(bf[..., n_dets * n_up * n_up:], bf_dn.shape[:-1] + (n_dn, n_dets, n_dn))

        bf_up = jnp.swapaxes(bf_up, -3, -2)  # output-shape: [batch x n_dets x n_up x n_up_orb]
        bf_dn = jnp.swapaxes(bf_dn, -3, -2)
        return bf_up, bf_dn

    bf_params = dict(general=dict(), orbital=dict())
    bf_params["general"]["up"] = init_ffwd_net(config.n_hidden_bf_factor, input_dim, True, init_config)
    bf_params["general"]["dn"] = init_ffwd_net(config.n_hidden_bf_factor, input_dim, True, init_config)
    if config.use_trainable_scales:
        bf_params["general"]["scale"] = jnp.array([-2.0])

    if len(config.n_hidden_bf_factor) != 0:
        input_dim = config.n_hidden_bf_factor[-1]

    bf_params["orbital"]["up_output"] = init_ffwd_net([n_dets * n_up], input_dim, config.use_bf_factor_bias, init_config)
    bf_params["orbital"]["dn_output"] = init_ffwd_net([n_dets * n_dn], input_dim, config.use_bf_factor_bias, init_config)

    return _call_bf_factor, bf_params


###############################################################################
############################## Orbitals #######################################
###############################################################################
def _build_baseline_orbital_net(config: OrbitalsConfig, physical_config: PhysicalConfig, init_config, n_dets, emb_dims):
    n_el, n_up, R, Z = physical_config.get_basic_params()

    def _call(r, diff_dist, embeddings, params, fixed_params):
        if config.baseline_orbitals.use_bf_shift:
            backflow_shift = bf_shift_call(*diff_dist, *embeddings, Z, params['bf_shift'])
            r = r + backflow_shift
        diff_el_ion, dist_el_ion = get_el_ion_distance_matrix(r, R)

        # Evaluate atomic and molecular orbitals for every determinant
        mo_matrix_up, mo_matrix_dn = get_baseline_slater_matrices(diff_el_ion, dist_el_ion, fixed_params,
                                                                  config.use_full_det)
        if config.baseline_orbitals.use_bf_factor:
            backflow_factor_up, backflow_factor_dn = bf_factor_call(embeddings[0], params['bf_fac'])
            mo_matrix_up *= backflow_factor_up
            mo_matrix_dn *= backflow_factor_dn
        return mo_matrix_up, mo_matrix_dn

    initial_params = {}
    if config.baseline_orbitals.use_bf_factor:
        bf_factor_call, initial_params['bf_fac'] = build_backflow_factor(config.baseline_orbitals, init_config, n_el,
                                                                         n_up, n_dets, emb_dims)
    if config.baseline_orbitals.use_bf_shift:
        bf_shift_call, initial_params['bf_shift'] = build_backflow_shift(config.baseline_orbitals, init_config,
                                                                         emb_dims)
    return _call, initial_params


def _build_envelope_orbitals(config: EnvelopeOrbitalsConfig, physical_config: PhysicalConfig, init_config, n_dets, full_det, emb_dims):
    n_el, n_up, _, Z = physical_config.get_basic_params()
    n_dn = n_el - n_up
    n_ions = len(Z)
    input_dim = emb_dims[0]

    output_size_up = n_dets * (n_el if full_det else n_up)
    output_size_dn = n_dets * (n_el if full_det else n_dn)
    env_function = dict(isotropic_exp=_isotropic_envelope)[config.envelope_type]

    def _call_envelope_orb(emb_el, dist_el_ion, params):
        embeddings_up = emb_el[..., :n_up, :]
        embeddings_dn = emb_el[..., n_up:, :]

        if len(config.n_hidden_env_factor) != 0:
            embeddings_up = ffwd_net(params["general"]["up"], embeddings_up, linear_output=False)
            embeddings_dn = ffwd_net(params["general"]["dn"], embeddings_dn, linear_output=False)


        bf_up = ffwd_net(params["orbital"]["up_output"], embeddings_up, linear_output=True)
        bf_dn = ffwd_net(params["orbital"]["dn_output"], embeddings_dn, linear_output=True)
        # output-shape: [batch x n_up x (n_dets * n_up_orb)]

        bf_up = jnp.swapaxes(bf_up.reshape(bf_up.shape[:-1] + (n_dets, -1)), -3, -2)
        bf_dn = jnp.swapaxes(bf_dn.reshape(bf_dn.shape[:-1] + (n_dets, -1)), -3, -2)
        # output - shape: [batch x n_dets x n_up x n_up_orb]

        # Modify molecular orbitals using additive backflow with envelope
        mo_matrix_up, mo_matrix_dn = env_function(dist_el_ion, n_dets, **params["envelope"], n_up=n_up, n_el=n_el)
        mo_matrix_up *= bf_up
        mo_matrix_dn *= bf_dn
        return mo_matrix_up, mo_matrix_dn

    initial_params = dict(general=dict(), orbital=dict())

    if config.n_hidden_env_factor:
        initial_params["general"]["up"] = init_ffwd_net(config.n_hidden_env_factor, input_dim, True, init_config)
        initial_params["general"]["dn"] = init_ffwd_net(config.n_hidden_env_factor, input_dim, True, init_config)
        input_dim = config.n_hidden_env_factor[-1]

    initial_params["orbital"]["up_output"] = init_ffwd_net([output_size_up], input_dim, config.use_bf_add_bias, init_config)
    initial_params["orbital"]["dn_output"] = init_ffwd_net([output_size_dn], input_dim, config.use_bf_add_bias, init_config)

    if config.initialization == 'constant':
        c_up, c_dn = jnp.ones([n_ions, n_dets * n_up]), jnp.ones([n_ions, n_dets * n_dn])
        alpha_up, alpha_dn = jnp.ones([n_ions, n_dets * n_up]), jnp.ones([n_ions, n_dets * n_dn])
    # elif config.initialization == 'hf_fit':
    #     screening = 0.7
    #     (c_up, c_dn), (alpha_up, alpha_dn) = fit_orbital_envelopes_to_hartree_fock(physical_config)
    #     initial_params["envelope"] = dict(
    #         c_up=jnp.tile(c_up, [1, n_dets]),
    #         c_dn=jnp.tile(c_dn, [1, n_dets]),
    #         alpha_up=jnp.tile(alpha_up, [1, n_dets]) * screening,
    #         alpha_dn=jnp.tile(alpha_dn, [1, n_dets]) * screening
    #     )
    elif config.initialization == 'analytical':
        (c_up, c_dn), (alpha_up, alpha_dn) = get_envelope_exponents_from_atomic_orbitals(physical_config)
        alpha_up = jnp.tile(alpha_up, [1, n_dets])
        alpha_dn = jnp.tile(alpha_dn, [1, n_dets])
        c_up = jnp.tile(c_up, [1, n_dets])
        c_dn = jnp.tile(c_dn, [1, n_dets])

    # elif config.initialization == 'analytical_random':
    #     (c_up, c_dn), (alpha_up, alpha_dn) = get_envelope_exponents_from_atomic_orbitals(physical_config)
    #     alpha_up = jnp.tile(alpha_up, [1, n_dets])
    #     alpha_dn = jnp.tile(alpha_dn, [1, n_dets])
    #     alpha_up = alpha_up * np.random.normal(1, 0.2, alpha_up.shape) + np.random.normal(0, 0.5, alpha_up.shape)
    #     alpha_dn = alpha_dn * np.random.normal(1, 0.2, alpha_dn.shape) + np.random.normal(0, 0.5, alpha_dn.shape)
    #     initial_params["envelope"] = dict(c_up=jnp.tile(c_up, [1, n_dets]), c_dn=jnp.tile(c_dn, [1, n_dets]),
    #                                       alpha_up=alpha_up, alpha_dn=alpha_dn
    #                                       )
    # elif config.initialization == 'hardcoded':
    #     initial_params["envelope"] = get_envelope_exponents_hardcoded(physical_config, n_dets)
    elif config.initialization == 'cisd':
        (c_up, c_dn), (alpha_up, alpha_dn) = get_envelope_exponents_cisd(physical_config, n_dets)
    else:
        raise NotImplementedError(f"Unknown initialization method for envelopes: {config.initialization}")
    if full_det:
        c_up, c_dn = _add_full_det_offdiagonal_blocks(c_up, c_dn, n_dets, config.initialization_off_diag,  1.0)
        alpha_up, alpha_dn = _add_full_det_offdiagonal_blocks(alpha_up, alpha_dn, n_dets, config.initialization_off_diag, 1.0)
    initial_params["envelope"] = dict(c_up=c_up, c_dn=c_dn, alpha_up=alpha_up, alpha_dn=alpha_dn)

    return _call_envelope_orb, initial_params

def _add_full_det_offdiagonal_blocks(x_up, x_dn, n_dets, method, constant=1.0):
    batch_shape = x_up.shape[:-1]
    x_up = x_up.reshape(batch_shape + (n_dets, -1))
    x_dn = x_dn.reshape(batch_shape + (n_dets, -1))
    n_up, n_dn = x_up.shape[-1], x_dn.shape[-1]

    if method == 'constant':
        x_up_padded = jnp.concatenate([x_up, jnp.ones(batch_shape + (n_dets, n_dn)) * constant], axis=-1)
        x_dn_padded = jnp.concatenate([jnp.ones(batch_shape + (n_dets, n_up)) * constant, x_dn], axis=-1)
    elif method == 'copy':
        x_up_padded = jnp.concatenate([x_up, x_dn], axis=-1)
        x_dn_padded = jnp.concatenate([x_up, x_dn], axis=-1)
    else:
        raise ValueError(f"Unsupported initialization method for off-diagonal elements of envelopes: {method}")

    x_up_padded = x_up_padded.reshape(batch_shape + (-1,))
    x_dn_padded = x_dn_padded.reshape(batch_shape + (-1,))
    return x_up_padded, x_dn_padded


def build_orbital_net(config: OrbitalsConfig, physical_config: PhysicalConfig, init_config: InitializationConfig,
                      emb_dims: Tuple[int]) -> Tuple[Callable[..., Tuple[jnp.array, jnp.array]], dict]:
    initial_params = {}

    if config.envelope_orbitals:
        env_orb_call, initial_params['env_orb'] = _build_envelope_orbitals(config.envelope_orbitals,
                                                                           physical_config,
                                                                           init_config,
                                                                           config.n_determinants,
                                                                           config.use_full_det,
                                                                           emb_dims)
    if config.baseline_orbitals:
        baseline_orb_call, initial_params['baseline_orb'] = _build_baseline_orbital_net(config,
                                                                                        physical_config,
                                                                                        init_config,
                                                                                        config.n_determinants,
                                                                                        emb_dims)

    def _call(r, diff_dist, embeddings, params, fixed_params):
        # Modify molecular orbitals using additive backflow with envelope
        if config.envelope_orbitals:
            emb_el = embeddings[0]
            dist_el_ion = diff_dist[3]
            mo_matrix_up, mo_matrix_dn = env_orb_call(emb_el, dist_el_ion, params['env_orb'])
        else:
            mo_matrix_up, mo_matrix_dn = 0.0, 0.0

        if config.baseline_orbitals:
            mo_up_bl, mo_dn_bl = baseline_orb_call(r, diff_dist, embeddings, params['baseline_orb'], fixed_params)
            mo_matrix_up += mo_up_bl
            mo_matrix_dn += mo_dn_bl
        return mo_matrix_up, mo_matrix_dn

    return _call, initial_params


def get_baseline_slater_matrices(diff_el_ion, dist_el_ion, fixed_params, full_det):
    atomic_orbitals, ao_cusp_params, mo_cusp_params, mo_coeff, ind_orb, ci_weights = fixed_params
    n_dets, n_up= ind_orb[0].shape
    n_dn = ind_orb[1].shape[1]
    n_ao = mo_coeff[0].shape

    mo_matrix_up = evaluate_molecular_orbitals(
        diff_el_ion[..., :n_up, :, :], dist_el_ion[..., :n_up, :], atomic_orbitals, mo_coeff[0], ao_cusp_params, mo_cusp_params[0]
    )
    mo_matrix_dn = evaluate_molecular_orbitals(
        diff_el_ion[..., n_up:, :, :], dist_el_ion[..., n_up:, :], atomic_orbitals, mo_coeff[1], ao_cusp_params, mo_cusp_params[1]
    )
    # 1) Select orbitals for each determinant => [(batch) x n_el x n_det x n_orb]
    # 2) Move determinant axis forward => [(batch) x n_det x n_el x n_orb]
    mo_matrix_up = jnp.moveaxis(mo_matrix_up[..., ind_orb[0]], -2, -3)
    mo_matrix_dn = jnp.moveaxis(mo_matrix_dn[..., ind_orb[1]], -2, -3)

    if full_det:
        batch_shape = mo_matrix_up.shape[:-2]
        mo_matrix_up = jnp.concatenate([mo_matrix_up, jnp.zeros(batch_shape + (n_up, n_dn))], axis=-1)
        mo_matrix_dn = jnp.concatenate([jnp.zeros(batch_shape + (n_dn, n_up)), mo_matrix_dn], axis=-1)


    # CI weights need to go somewhere; could also multiply onto mo_dn, should yield same results
    ci_weights = ci_weights[:, None, None]
    ci_weights_up = jnp.abs(ci_weights)**(1/n_up)

    # adjust sign of first col to match det sign
    ci_weights_up *= jnp.concatenate([jnp.sign(ci_weights), jnp.ones([n_dets, 1, mo_matrix_up.shape[-1]-1])], axis=-1)
    mo_matrix_up *= ci_weights_up
    return mo_matrix_up, mo_matrix_dn


def evaluate_sum_of_determinants(mo_matrix_up, mo_matrix_dn, full_det):
    LOG_EPSILON = 1e-8

    if full_det:
        mo_matrix = jnp.concatenate([mo_matrix_up, mo_matrix_dn], axis=-2)
        sign_total, log_total = jnp.linalg.slogdet(mo_matrix)
    else:
        sign_up, log_up = jnp.linalg.slogdet(mo_matrix_up)
        sign_dn, log_dn = jnp.linalg.slogdet(mo_matrix_dn)
        log_total = log_up + log_dn
        sign_total = sign_up * sign_dn
    log_shift = jnp.max(log_total, axis=-1, keepdims=True)
    psi = jnp.exp(log_total - log_shift) * sign_total
    psi = jnp.sum(psi, axis=-1)  # sum over determinants
    log_psi_sqr = 2 * (jnp.log(jnp.abs(psi) + LOG_EPSILON) + jnp.squeeze(log_shift, -1))
    return log_psi_sqr


def _isotropic_envelope(el_ion_dist, n_det, alpha_up, alpha_dn, c_up, c_dn, n_up, n_el):
    """
    Evaluates evelopes (=baseline orbital) consisting of isotropic decaying exponentials centered on each nucleus.

    Args:
        el_ion_dist: [... x n_el x n_ions]
        alpha: [n_dets x n_orbitals x n_ions]
        c: [n_dets x n_orbitals x n_ions]

    Returns:
        mo_matrix_up, mo_matrix_dn
    """

    d_up = el_ion_dist[..., :n_up, :, jnp.newaxis]  # [batch x el_up x ion x  1 (orb)]
    d_dn = el_ion_dist[..., n_up:, :, jnp.newaxis]

    y_up = jax.nn.softplus(alpha_up) * d_up  # [el_up x ion x (orb * dets)]
    y_dn = jax.nn.softplus(alpha_dn) * d_dn
    exponent_up = register_scale_and_shift(y_up, [d_up, alpha_up], has_scale=True, has_shift=False)
    exponent_dn = register_scale_and_shift(y_dn, [d_dn, alpha_dn], has_scale=True, has_shift=False)

    z_up = c_up * jnp.exp(-exponent_up)
    z_dn = c_dn * jnp.exp(-exponent_dn)

    sum_up = register_scale_and_shift(z_up, [jnp.exp(-exponent_up), c_up], has_scale=True, has_shift=False)
    sum_dn = register_scale_and_shift(z_dn, [jnp.exp(-exponent_dn), c_dn], has_scale=True, has_shift=False)
    mo_up = jnp.sum(sum_up, axis=-2)  # sum over ions
    mo_dn = jnp.sum(sum_dn, axis=-2)  # sum over ions

    mo_up = jnp.reshape(mo_up, mo_up.shape[:-1] + (n_det, -1))  # [batch x el_up x dets x orb]
    mo_dn = jnp.reshape(mo_dn, mo_dn.shape[:-1] + (n_det, -1))
    return jnp.moveaxis(mo_up, -2, -3), jnp.moveaxis(mo_dn, -2, -3)  # Move determinant axis to the front


###############################################################################
##################### Corrections: Jastrow, el-el-cusps########################
###############################################################################


def build_jastrow_factor(config: DeepErwinModelConfig, n_up, emb_dims: Tuple[int]):
    """
    Builds the Jastrow factor of the DeepErwin wavefunction model and initializes the respective model parameters.

    Args:
        config (DeepErwinModelConfig): Hyperparameters of the DeepErwin model
        n_up (int): Number of up-spin electrons
        name (str): Name of the Jastrow factor instance

    Returns:
        A tuple (name, call, params) where call is a callable representing the built Jastrow factor. The dictionary params contains the initial trainable parameters.

    """

    def _call_jastrow(embeddings_el, embeddings_el_el, embeddings_el_ion, params):
        del embeddings_el_el, embeddings_el_ion

        if config.differentiate_spins_jastrow:
            jastrow_up = ffwd_net(params["up"], embeddings_el[..., :n_up, :], linear_output=True)
            jastrow_dn = ffwd_net(params["dn"], embeddings_el[..., n_up:, :], linear_output=True)
            jastrow = jnp.sum(jastrow_up, axis=(-2, -1)) + jnp.sum(jastrow_dn, axis=(-2, -1))
        else:
            jastrow = ffwd_net(params["w"], embeddings_el, linear_output=True)
            jastrow = jnp.sum(jastrow, axis=(-2, -1))
        return jastrow

    jas_params = {}

    input_dim = emb_dims[0]
    if config.differentiate_spins_jastrow:
        jas_params["up"] = init_ffwd_net(config.n_hidden_jastrow + [1], input_dim, False, config.initialization)
        jas_params["dn"] = init_ffwd_net(config.n_hidden_jastrow + [1], input_dim, False, config.initialization)
    else:
        jas_params["w"] = init_ffwd_net(config.n_hidden_jastrow + [1], input_dim, False, config.initialization)
    return _call_jastrow, jas_params


def el_el_cusp(el_el_dist, n_electrons: int, n_up: int):
    factor = np.ones([n_electrons, n_electrons]) * 0.5
    factor[:n_up, : n_up] = 0.25
    factor[n_up:, n_up:] = 0.25

    r_cusp_el_el = 1.0
    A = r_cusp_el_el ** 2
    B = r_cusp_el_el
    # No factor 0.5 here, e.g. when comparing to NatChem 2020, [doi.org/10.1038/s41557-020-0544-y], because:
    # A) We double-count electron-pairs because we take the full distance matrix (and not only the upper triangle)
    # B) We model log(psi^2)=2*log(|psi|) vs log(|psi|) int NatChem 2020, i.e. the cusp correction needs a factor 2
    return -jnp.sum(factor * A / (el_el_dist + B), axis=[-2, -1])


###############################################################################
############################## Total model ####################################
###############################################################################

def build_log_psi_squared(config: DeepErwinModelConfig, phys_config: PhysicalConfig, fixed_params=None):
    """
    Builds log(psi(.)^2) for a wavefunction psi that is based on the DeepErwin model and initializes the respective trainable and fixed parameters.

    Args:
        config (DeepErwinModelConfig): Hyperparameters of the DeepErwin model
        phys_config (PhysicalConfig): Description of the molecule
        init_fixed_params (bool): If false, fixed parameters will not be initialized. In particular, this includes the computation of CASSCF baseline results.
        name (str): Name of the model instance

    Returns:
        A tuple (name, call, trainable_params, fixed_params) where call is a callable representing the built model. The dictionaries trainable_params and fixed_params contain the respective initial model parameters.

    """
    n_electrons, n_up, n_ions = phys_config.n_electrons, phys_config.n_up, len(phys_config.Z)
    feature_dims = get_nr_of_input_features(config.features, phys_config)
    emb_dims = get_embedding_dim(config.embedding)
    initial_fixed_params = fixed_params or init_model_fixed_params(config, phys_config)

    initial_trainable_params = {}
    call_preprocess, initial_trainable_params['input'] = build_preprocessor(config.features, phys_config, config.initialization)
    call_embed, initial_trainable_params['embed'] = build_embedding(config.embedding, phys_config, config.initialization, feature_dims)
    call_orbitals, initial_trainable_params['orbitals'] = build_orbital_net(config.orbitals, phys_config, config.initialization, emb_dims)
    logger.debug(f"Number of parameters: {get_number_of_params(initial_trainable_params)}")

    if config.use_jastrow:
        jastrow_call, initial_trainable_params['jastrow'] = build_jastrow_factor(config, n_up, emb_dims)

    def _call_slater_orbitals(r, R, Z, params, fixed_params, aux_output=False):
        # Preprocessing: Calculate basic features from input coordinates
        Z = jnp.array(Z, dtype=float)

        # Preprocessing: Calculate basic features from input coordinates
        diff_dist, features = call_preprocess(r, R, Z, params['input'], fixed_params['input'])
        # diff_dist is a tuple containing (diff_el_el, dist_el_el, diff_el_ion, dist_el_ion)
        # features is a tuple containing (features_el, features_el_el, features_el_ion)


        if config.embedding:
            embeddings = call_embed(*features, Z, params['embed'])
            # embeddings is a tuple containing (embeddings_el, embeddings_el_el, embeddings_el_ion)
        else:
            embeddings = None

        # Slater matrix/determinant of orbitals
        output = call_orbitals(r, diff_dist, embeddings, params['orbitals'], fixed_params['orbitals'])
        if aux_output:
            output += (diff_dist, features, embeddings)
        return output

    def _call(r, R, Z, params, fixed_params):
        mo_up, mo_dn, diff_dist, features, embeddings = _call_slater_orbitals(r, R, Z, params, fixed_params, aux_output=True)
        log_psi_sqr = evaluate_sum_of_determinants(mo_up, mo_dn, config.orbitals.use_full_det)

        # Jastrow factor to the total wavefunction
        if config.use_jastrow:
            log_psi_sqr += jastrow_call(*embeddings, params['jastrow'])

        # Electron-electron-cusps
        if config.use_el_el_cusp_correction:
            log_psi_sqr += el_el_cusp(diff_dist[1], n_electrons, n_up)

        return log_psi_sqr

    return _call, _call_slater_orbitals, initial_trainable_params, initial_fixed_params


def init_model_fixed_params(config: DeepErwinModelConfig, physical_config: PhysicalConfig):
    """
    Computes CASSCF baseline solution for DeepErwin model and initializes fixed parameters.

    Args:
        casscf_config (CASSCFConfig): CASSCF hyperparmeters
        physical_config (PhysicalConfig): Description of the molecule

    Returns:
        dict: Initial fixed parameters
    """
    fixed_params = dict(input={}, orbitals=None, baseline_energies=dict(E_ref=physical_config.E_ref))

    logger.debug("Calculating baseline solution...")

    if config.orbitals.baseline_orbitals:
        cas_config = config.orbitals.baseline_orbitals.baseline
        fixed_params['orbitals'], (E_hf, E_casscf) = get_baseline_solution(physical_config, cas_config, config.orbitals.n_determinants)
        fixed_params['baseline_energies'].update(dict(E_hf=E_hf, E_casscf=E_casscf))
        logger.debug(f"Finished baseline calculation: E_casscf={E_casscf:.6f}")

    if config.features.mo_features:
        fixed_params['input']['mo_features'] = init_fixed_params_for_input_features(config.features.mo_features, physical_config)
    if config.features.use_local_coordinates:
        fixed_params['input']['local_rotations'] = build_local_rotation_matrices(physical_config)

    return convert_to_jnp(fixed_params)


if __name__ == "__main__":
    pass
