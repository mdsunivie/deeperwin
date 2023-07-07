from typing import Tuple, List
import numpy as np
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import haiku as hk
import functools
from deeperwin.utils.utils import tp_out_irreps_with_instructions, multi_vmap
from deeperwin.model.definitions import Edge, Embeddings, InputFeatures, DiffAndDistances
from deeperwin.model.e3nn_utils import Linear, NamedE3Linear, norm_nonlinearity, get_irreps_up_to_lmax, to_irreps_array
from deeperwin.configuration import EmbeddingConfigE3MPNN, MLPConfig
from deeperwin.model.mlp import MLP, _get_normalization_factors


class Interaction(hk.Module):
    def __init__(self, n_nodes_out, irreps_node, irreps_edge, irreps_out, use_radial_mlp, mlp_depth=None, mlp_width=None, name=None):
        super().__init__(name=name)
        self.n_nodes_out = n_nodes_out
        irreps_mid, instructions = tp_out_irreps_with_instructions(e3nn.Irreps(irreps_node),
                                                                   e3nn.Irreps(irreps_edge),
                                                                   e3nn.Irreps(irreps_out))
        self.tp = e3nn.FunctionalTensorProduct(irreps_node, irreps_edge, irreps_mid, instructions)
        n_weights = 0
        for instruction in self.tp.instructions:
            if instruction.has_weight:
                n_weights += np.prod(instruction.path_shape)

        n_hidden = [mlp_width] * mlp_depth if use_radial_mlp else []
        self.weight_MLP = hk.nets.MLP(n_hidden + [n_weights], activation=jax.nn.silu, activate_final=False)


    def __call__(self, features_node, features_edge, features_weights, edges):
        batch_size = features_node.shape[:-2]
        src = jnp.array([e.src for e in edges], int)
        tgt = jnp.array([e.tgt for e in edges], int)

        weights = self.weight_MLP(features_weights)
        tp = multi_vmap(self.tp.left_right, len(batch_size)+1) # vmap over batch-dim and edges (+1)
        messages = tp(weights, features_node[..., src, :], features_edge) # [batch x edges x feat]

        message = e3nn.IrrepsArray.zeros(self.tp.irreps_out, leading_shape=(*batch_size, self.n_nodes_out))
        message = message.at[..., tgt, :].add(messages)                           # [batch x n_nodes_out (n_el) x feat]
        return message


class E3OneElectronLayer(hk.Module):
    def __init__(self, irreps_out, skip_connection, scalar_activation, use_trainable_res_weight, name=None):
        super().__init__(name=name)
        self.irreps_out = irreps_out
        self.skip_connection = skip_connection
        self.scalar_activation = dict(tanh=jnp.tanh, gelu=jax.nn.gelu)[scalar_activation]
        self.use_trainable_res_weight = use_trainable_res_weight

    def __call__(self, features_old, *messages):
        features_el = e3nn.concatenate([features_old, *messages])
        features_el = NamedE3Linear(self.irreps_out, "linear")(features_el)
        features_el = norm_nonlinearity(features_el, self.scalar_activation)
        if self.skip_connection and self.use_trainable_res_weight:
            w = hk.get_parameter("residual_weight", [], init=lambda s, d: jnp.ones(s, d) * 0.1)
            features_el = w * features_el

        if self.skip_connection == "no":
            return features_el
        if self.skip_connection == 'linear':
            return features_el + NamedE3Linear(self.irreps_out, "skip_conn")(features_old)
        if self.skip_connection == 'residual':
            if features_old.shape == features_el.shape:
                return features_el + features_old
            else:
                return features_el
        if self.skip_connection == 'residual_or_linear':
            if features_old.shape == features_el.shape:
                return features_el + features_old
            else:
                return features_el + NamedE3Linear(self.irreps_out, "skip_conn")(features_old)
        raise NotImplementedError("Unknown type of skip connection")


class E3ChannelMapping(hk.Module):
    def __init__(self, channels_out, name=None):
        super().__init__(name=name)
        self.channels_out = channels_out

    def __call__(self, x):
        target_irreps = get_irreps_up_to_lmax(x.irreps.lmax, self.channels_out)
        return Linear(target_irreps,
                           path_normalization="element",
                           gradient_normalization="element")(x)


def generate_edges(n_el: int, n_up: int, n_ion: int) -> Tuple[List[Edge], List[Edge], List[Edge]]:
    n_dn = n_el - n_up
    same, diff, el_ion = [], [], []

    spin = [0] * n_up + [1]*n_dn
    for i in range(n_el):
        for j in range(n_el):
            if i==j:
                continue
            if spin[i] == spin[j]:
                same.append(Edge(i,j))
            else:
                diff.append(Edge(i,j))
        for j in range(n_ion):
            el_ion.append(Edge(i, j))
    return same, diff, el_ion

class EquivariantMPNNEmbedding(hk.Module):
    def __init__(self, config: EmbeddingConfigE3MPNN, mlp_config: MLPConfig, name=None):
        super().__init__(name=name)
        self.config = config
        self.mlp_config = mlp_config
        self.interaction_dim = self.config.interaction_dim or self.config.n_hidden_two_el

        self.irreps_interaction = get_irreps_up_to_lmax(config.l_max, self.interaction_dim)
        self.L_max = self.config.L_max or self.config.l_max
        self.irreps_el = []
        self.mlp_same = []
        self.mlp_diff = []
        self.mlp_el_ion = []
        self.one_el_layer = []
        for n in range(config.n_iterations):
            if n == (config.n_iterations - 1):
                # Only scalar features in last layer
                irreps_el = e3nn.Irreps(f"{config.n_hidden_one_el}x0e")
            else:
                irreps_el = get_irreps_up_to_lmax(config.L_max, self.config.n_hidden_one_el)

            two_particle_mlp = functools.partial(MLP, [config.n_hidden_two_el], self.mlp_config, linear_out=False)
            self.mlp_same.append(two_particle_mlp(name="mlp_same"))
            self.mlp_diff.append(two_particle_mlp(name="mlp_diff"))
            self.mlp_el_ion.append(two_particle_mlp(name="mlp_el_ion"))
            self.one_el_layer.append(E3OneElectronLayer(irreps_el,
                                                        self.config.skip_connection,
                                                        self.config.scalar_activation_one_el,
                                                        self.config.use_trainable_residual_weight))
            self.irreps_el.append(irreps_el)

    def _radial_cutoff_long_range(self, r):
        x = r * (1.0 / self.config.r_cut_sh_long_range)
        x = jnp.where(x < 1.0, x, 1.0)
        return jnp.cos((0.5 * np.pi) * x)

    def _radial_decay_short_range(self, r):
        return jnp.tanh(r * (1.0 / self.config.r_cut_sh_short_range))

    def _spherical_harmonics(self, diff, dist, l_max=None):
        if l_max is None:
            l_max = self.config.l_max
        Y = e3nn.spherical_harmonics(get_irreps_up_to_lmax(l_max),
                                     e3nn.IrrepsArray("1o", diff),
                                     normalize=self.config.normalize_spherical_harmonics)
        if self.config.r_cut_sh_short_range:
            factor = self._radial_decay_short_range(dist)[..., None]
            if self.config.cut_l0_short_range:
                Y = Y * factor
            elif Y.irreps.lmax > 0:
                Y_scalar = Y[..., :1]
                Y_vector = Y[..., 1:]
                Y = e3nn.concatenate([Y_scalar, Y_vector * factor], axis=-1)
        if self.config.r_cut_sh_long_range:
            factor = self._radial_cutoff_long_range(dist)[..., None]
            Y = Y * factor
        return Y

    def __call__(self, diff_dist: DiffAndDistances, features: InputFeatures, n_up: int):
        n_el, n_ion = diff_dist.diff_el_ion.shape[-3:-1]
        normalization_el_el, normalization_el_ion = _get_normalization_factors(self.config.neighbor_normalization,
                                                                               n_el,
                                                                               n_ion)
        # Slices of feature vectors depending on spin
        assert diff_dist.dist_el_el.shape[-1] == diff_dist.dist_el_el.shape[-2], "Must use full el-el distance matrix"
        edges_same, edges_diff, edges_el_ion = generate_edges(n_el, n_up, n_ion)
        features_same = jnp.stack([features.el_el[..., edge.tgt, edge.src, :] for edge in edges_same], axis=-2)
        features_diff = jnp.stack([features.el_el[..., edge.tgt, edge.src, :] for edge in edges_diff], axis=-2)
        features_el_ion = jnp.stack([features.el_ion[..., edge.tgt, edge.src, :] for edge in edges_el_ion], axis=-2)
        features_el = e3nn.IrrepsArray(f"{features.el.shape[-1]}x0e", features.el)

        # Spherical harmonics Y, to be used in the interactions
        #TODO: compute spherical harmonics only for the edges given in the respective lists
        sh_el_el = self._spherical_harmonics(diff_dist.diff_el_el, diff_dist.dist_el_el)
        sh_el_ion = self._spherical_harmonics(diff_dist.diff_el_ion, diff_dist.dist_el_ion)
        sh_same = e3nn.stack([sh_el_el[..., edge.tgt, edge.src, :] for edge in edges_same], axis=-2)
        sh_diff = e3nn.stack([sh_el_el[..., edge.tgt, edge.src, :] for edge in edges_diff], axis=-2)
        sh_el_ion = e3nn.stack([sh_el_ion[..., edge.tgt, edge.src, :] for edge in edges_el_ion], axis=-2)

        # Map the one-hot-encoded ion features to an independent vector for each iteration
        input_features_ion = to_irreps_array(features.ion)

        if self.config.create_el_features_from_ions:
            #TODO: adapt to use edges
            sh = self._spherical_harmonics(diff_dist.diff_el_ion,
                                           diff_dist.dist_el_ion,
                                           l_max=input_features_ion.irreps.lmax)
            products = e3nn.tensor_product(input_features_ion[..., None, :, :], sh, filter_ir_out=["0e"])
            features_el = e3nn.concatenate([
                features_el,
                e3nn.sum(products, axis=-2) * normalization_el_ion
            ]).simplify()

        input_features_ion = input_features_ion.filtered(get_irreps_up_to_lmax(self.config.l_max))


        embeddings_el = []
        for n in range(self.config.n_iterations):
            # 2-electron stream: Radial weights for interactions
            if not self.config.use_radial_mlp:
                features_same = self.mlp_same[n](features_same)
                features_diff = self.mlp_diff[n](features_diff)
                features_el_ion = self.mlp_el_ion[n](features_el_ion)

            h1_mapped = E3ChannelMapping(self.interaction_dim, name="h1_map")(features_el)
            features_ion = E3ChannelMapping(self.interaction_dim, name="ion_embed")(input_features_ion)

            # # 2 particle interactions: Tensor product of w(r_ij) * Y x f_j; no mixing of channels
            interaction_same = Interaction(n_el,
                                           h1_mapped.irreps,
                                           sh_el_el.irreps,
                                           self.irreps_el[n],
                                           self.config.use_radial_mlp,
                                           self.config.radial_mlp_depth,
                                           self.config.radial_mlp_width,
                                           name="interaction_same")
            interaction_diff = Interaction(n_el,
                                           h1_mapped.irreps,
                                           sh_el_el.irreps,
                                           self.irreps_el[n],
                                           self.config.use_radial_mlp,
                                           self.config.radial_mlp_depth,
                                           self.config.radial_mlp_width,
                                           name="interaction_diff")
            interaction_el_ion = Interaction(n_el,
                                             features_ion.irreps,
                                             sh_el_ion.irreps,
                                             self.irreps_el[n],
                                             self.config.use_radial_mlp,
                                             self.config.radial_mlp_depth,
                                             self.config.radial_mlp_width,
                                             name="interaction_el_ion"
                                             )

            message_same = interaction_same(h1_mapped, sh_same, features_same, edges_same)
            message_diff = interaction_diff(h1_mapped, sh_diff, features_diff, edges_diff)
            message_el_ion = interaction_el_ion(features_ion, sh_el_ion, features_el_ion, edges_el_ion)
            message_el_el = (message_same + message_diff) * normalization_el_el
            message_el_ion = message_el_ion * normalization_el_ion

            if self.config.use_msg_mapping:
                message_el_el = NamedE3Linear(self.irreps_el[n], "msg_el_el")(message_el_el)
                message_el_ion = NamedE3Linear(self.irreps_el[n], "msg_el_ion")(message_el_ion)
            features_el = self.one_el_layer[n](features_el, message_el_ion, message_el_el)

            if self.config.output_intermediate_features or (n == (self.config.n_iterations - 1)):
                embeddings_el.append(features_el.filtered(["0e", "0o"]).array)

        features_el = jnp.concatenate(embeddings_el, axis=-1)
        if self.config.readout_mlp_depth > 0:
            n_neurons = [self.config.readout_mlp_width] * self.config.readout_mlp_depth
            features_el = MLP(n_neurons, self.mlp_config, name="readout")(features_el)

        return Embeddings(features_el,
                          features_ion.array,
                          None,
                          None
                          )
