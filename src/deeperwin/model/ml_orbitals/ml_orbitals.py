# %%
import e3nn_jax as e3nn
from deeperwin.model.e3nn_utils import (
    E3LinearChannelMixing,
    NamedE3Linear,
    get_irreps_up_to_lmax,
    _filter_scalar,
    _filter_vector,
    E3ChannelNorm,
)
from deeperwin.orbitals import (
    OrbitalParamsHF,
    build_pyscf_molecule_from_physical_config,
    _get_atomic_orbital_basis_functions,
    localize_orbitals,
)
from deeperwin.model.mlp import MLP
from deeperwin.configuration import MLPConfig
from deeperwin.utils.utils import get_param_size_summary
import logging

import haiku as hk
import jax.numpy as jnp
import jax
from typing import Optional
import numpy as np
import math

_ALL_PARITIES = False
LOGGER = logging.getLogger("dpe")


def scalar_activation(x: e3nn.IrrepsArray):
    scalars = x.filter(_filter_scalar)
    vectors = x.filter(_filter_vector)
    # TODO: swish instead of SiLU?
    scalars_act = jax.nn.silu(scalars.array)
    scalars = e3nn.IrrepsArray(scalars.irreps, scalars_act)
    return e3nn.concatenate([scalars, vectors], axis=-1)


def get_radial_basis(r, scale, N_basis, r_cut):
    k = np.arange(N_basis)
    log_factorial = np.zeros(N_basis)
    log_factorial[1:] = np.cumsum(np.log(k[1:]))
    log_binomials = log_factorial[N_basis - 1] - log_factorial - log_factorial[::-1]

    eps_at_r_equal_0 = 1e-8
    x = r[..., None] / scale + eps_at_r_equal_0
    ln_f = log_binomials - k * x + (N_basis - k - 1) * jnp.log1p(-jnp.exp(-x))

    x_cut = jnp.clip(r[..., None] / r_cut, 0, 1 - 1e-6)
    cutoff = jnp.where(x_cut < 1.0, jnp.exp(-(x_cut**2) / (1 - x_cut**2)), 0.0)
    return jnp.exp(ln_f) * cutoff


class E3Selfmix(hk.Module):
    def __init__(self, L_out: int, name: Optional[str] = None):
        super().__init__(name)
        self.L_out = L_out
        self.irreps_out = get_irreps_up_to_lmax(L_out, all_parities=_ALL_PARITIES)
        self.lin = NamedE3Linear(self.irreps_out, with_bias=True)

    def __call__(self, x: e3nn.IrrepsArray):
        """
        Args:
            x: (..., channels, lm)
        """
        n_channels = x.shape[-2]

        # Tensorproduct with itself
        # TODO: replace with tensor_square or some implementation that is awaare of this symmetry?
        # z = e3nn.tensor_square(x, filter_ir_out=self.irreps_out)
        z = e3nn.tensor_product(x, x, filter_ir_out=self.irreps_out)
        z = self.lin(z)

        # Split irreps into the ones that have overlap with the input and the ones that don't
        z_small = z.filter(x.irreps)
        z_large = z.filter([ir for ir in z.irreps if ir not in x.irreps])
        k = hk.get_parameter("k", [n_channels, z_small.irreps.num_irreps], init=jnp.ones)
        z = e3nn.concatenate([z_small + k * x.filter(z_small.irreps), z_large], axis=-1)
        return z


class E3SphLinear(hk.Module):
    def __init__(
        self,
        L_out: Optional[int] = None,
        n_channels_out: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.L_out = L_out
        self.n_channels_out = n_channels_out

    def __call__(self, x: e3nn.IrrepsArray):
        L = self.L_out or x.irreps.lmax
        n_channels_out = self.n_channels_out or x.shape[-2]

        x = E3Selfmix(L)(x)
        x = E3LinearChannelMixing(n_channels_out)(x)
        return x


class E3ResidualBlock(hk.Module):
    def __init__(self, L: Optional[int] = None, name: Optional[str] = None):
        super().__init__(name)
        self.L = L

    def __call__(self, x: e3nn.IrrepsArray):
        L = self.L if self.L is not None else x.irreps.lmax
        z = E3ChannelNorm()(x)
        z = scalar_activation(z)
        z = E3SphLinear(L)(z)
        z = scalar_activation(z)
        z = E3SphLinear(L)(z)
        z = z + x.filter(z.irreps)
        return z


class E3ResidualWithLinearOut(hk.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.res = E3ResidualBlock()
        self.sph_lin = E3SphLinear()

    def __call__(self, x: e3nn.IrrepsArray):
        x = self.res(x)
        x = scalar_activation(x)
        x = self.sph_lin(x)
        return x


class E3PairMix(hk.Module):
    def __init__(self, L_out: Optional[int] = None, name: Optional[str] = None):
        super().__init__(name)
        self.L_out = L_out

    def __call__(self, x: e3nn.IrrepsArray, y: e3nn.IrrepsArray, dist_features: jnp.array):
        n_channels = x.shape[-2]
        if self.L_out is None:
            L = x.irreps.lmax
        else:
            L = self.L_out
        irreps_out = get_irreps_up_to_lmax(L, all_parities=_ALL_PARITIES)

        tp = e3nn.FunctionalFullyConnectedTensorProduct(x.irreps, y.irreps, irreps_out)
        weight_shapes = [ins.path_shape for ins in tp.instructions if ins.has_weight]
        n_weights_per_channel = int(sum([np.prod(shape) for shape in weight_shapes]))

        # Compute TP weights as linear mapping of radial features
        weights = hk.Linear(n_weights_per_channel * n_channels, with_bias=True)(dist_features)
        weights = weights.reshape(weights.shape[:-1] + (n_channels, n_weights_per_channel))

        # vmap weights and inputs over channels
        tp_func = hk.vmap(tp.left_right, in_axes=(0, 0, 0), out_axes=0, split_rng=False)

        # vmap input over batch dimensions
        n_batch_dims = x.ndim - 2
        for _ in range(n_batch_dims):
            tp_func = hk.vmap(tp_func, in_axes=(0, 0, 0), out_axes=0, split_rng=False)

        return tp_func(weights, x, y)


class E3PhisNetInteraction(hk.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def __call__(
        self,
        nodes_rec: e3nn.IrrepsArray,
        nodes_snd: e3nn.IrrepsArray,
        edge_ind: jax.Array,
        Y_edge: jax.Array,
        dist_feat: jax.Array,
    ):
        """
        Args:
            nodes: (n_nodes, channels, lm)
            edge_ind: (n_edges, 2)
            Y_edge: (n_edges, 1, lm)
            dist_feat: (n_edges, n_dist_features)
        """
        neighbors = nodes_snd[edge_ind[:, 0]]
        a = self._get_interaction_a(neighbors, Y_edge, dist_feat)
        b = self._get_interaction_b(neighbors, Y_edge, dist_feat)
        # TODO: Normalize by 1/sqrt(<n_neighbors>)?
        nodes_rec = nodes_rec.at[edge_ind[:, 1]].add(a)
        nodes_rec = nodes_rec.at[edge_ind[:, 1]].add(b)
        return nodes_rec

    def _get_interaction_a(self, neighbor: e3nn.IrrepsArray, Y: e3nn.IrrepsArray, dist_feat: jax.Array):
        L = neighbor.irreps.lmax
        n_channels = neighbor.shape[-2]
        scalars = neighbor.filter("0e").array
        n_irreps = neighbor.irreps.num_irreps

        Y = E3SphLinear(L, n_channels)(Y)
        weights = hk.Linear(n_channels * n_irreps, with_bias=True)(dist_feat)
        weights = weights.reshape(weights.shape[:-1] + (n_channels, n_irreps))
        weights *= scalars
        return weights * Y

    def _get_interaction_b(self, neighbor: e3nn.IrrepsArray, Y: e3nn.IrrepsArray, dist_feat: jax.Array):
        L = neighbor.irreps.lmax
        n_channels = neighbor.shape[-2]

        Y = E3SphLinear(L, n_channels)(Y)
        z = E3PairMix(L)(neighbor, Y, dist_feat)
        return z


class E3PhisNetLayer(hk.Module):
    def __init__(self, L_out: int, n_channels_out: int, name: Optional[str] = None):
        super().__init__(name)
        self.L_out = L_out
        self.n_channels_out = n_channels_out

        self.res_in = E3ResidualBlock()
        self.res_update = E3ResidualBlock()
        self.res_out = E3ResidualBlock()
        self.mapping_snd = E3ResidualWithLinearOut()
        self.mapping_rec = E3ResidualWithLinearOut()
        self.mapping_update = E3ResidualWithLinearOut()
        self.interaction = E3PhisNetInteraction()

    def __call__(
        self,
        nodes: e3nn.IrrepsArray,
        edge_ind: jax.Array,
        Y: e3nn.IrrepsArray,
        dist_features: jax.Array,
    ):
        """
        Args:
            nodes: (n_nodes, channels, lm)
            edge_ind: (n_edges, 2)
            R: (n_nodes, 3)
            dist_features: (n_edges, n_dist_features)
        """
        nodes = self.res_in(nodes)
        nodes_snd = self.mapping_snd(nodes)
        nodes_rec = self.mapping_rec(nodes)

        update = self.interaction(nodes_rec, nodes_snd, edge_ind, Y, dist_features)
        update = self.mapping_update(update)
        nodes = self.res_update(nodes + update)

        nodes = self.res_out(nodes)
        return nodes


class E3TensorExpansion(hk.Module):
    def __init__(self, l1_max, l2_max, l3_max, name: Optional[str] = None):
        super().__init__(name)

        self.cg = dict()
        for l1 in range(l1_max + 1):
            for l2 in range(l2_max + 1):
                for l3 in range(l3_max + 1):
                    self.cg[(l1, l2, l3)] = e3nn.clebsch_gordan(l1, l2, l3)

    def __call__(self, l1, l2, x: e3nn.IrrepsArray):
        assert len(x.irreps) == 1
        assert x.irreps[0].mul == 1
        l3 = x.irreps[0].ir.l
        return jnp.einsum("ijm,...m", self.cg[l1, l2, l3], x.array)


class E3PhisNetMatrixBlock(hk.Module):
    def __init__(
        self,
        irreps_row: e3nn.Irreps,
        irreps_col: e3nn.Irreps,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.irreps_row = e3nn.Irreps(irreps_row)
        self.irreps_col = e3nn.Irreps(irreps_col)
        self.irreps_in = e3nn.tensor_product(irreps_row, irreps_col)
        self.expansion = E3TensorExpansion(self.irreps_row.lmax, self.irreps_col.lmax, self.irreps_in.lmax)

    def _call_no_batch(self, x: e3nn.IrrepsArray):
        assert x.irreps == self.irreps_in
        M = jnp.zeros([self.irreps_row.dim, self.irreps_col.dim])

        offsets = {ir.ir: 0 for ir in self.irreps_in}
        ind_row = 0
        for ir_row in self.irreps_row:
            for n in range(ir_row.mul):
                ind_col = 0
                for ir_col in self.irreps_col:
                    for m in range(ir_col.mul):
                        block = jnp.zeros([ir_row.ir.dim, ir_col.ir.dim])
                        for ir_in in e3nn.tensor_product(str(ir_row.ir), str(ir_col.ir)):
                            offset = offsets[ir_in.ir]
                            x_selected = x.filter(ir_in.ir)[..., offset : offset + ir_in.ir.dim]
                            block += self.expansion(ir_row.ir.l, ir_col.ir.l, x_selected)
                            offsets[ir_in.ir] += ir_in.ir.dim
                        M = M.at[
                            ind_row : ind_row + ir_row.ir.dim,
                            ind_col : ind_col + ir_col.ir.dim,
                        ].set(block)
                        ind_col += ir_col.ir.dim
                ind_row += ir_row.ir.dim
        return M

    def __call__(self, x: e3nn.IrrepsArray):
        n_batch_dims = x.ndim - 1

        func = self._call_no_batch
        for _ in range(n_batch_dims):
            func = hk.vmap(func, split_rng=False)
        return func(x)


class E3PhisNetHamiltonian(hk.Module):
    def __init__(self, irreps_basis, name: Optional[str] = None):
        super().__init__(name)
        self.irreps_basis = e3nn.Irreps(irreps_basis)
        self.block_mapping = E3PhisNetMatrixBlock(self.irreps_basis, self.irreps_basis)
        lmax = self.block_mapping.irreps_in.lmax

        self.res_center = E3ResidualBlock()
        self.res_neighbor = E3ResidualBlock()

        self.pairmix_offdiag = E3PairMix()
        self.res_offdiag = E3ResidualBlock(lmax)
        self.res_diag = E3ResidualBlock(lmax)

        self.diag_norm = E3ChannelNorm(name="norm_diag")
        self.offdiag_norm = E3ChannelNorm(name="norm_offdiag")
        self.lin_diag = NamedE3Linear(
            self.block_mapping.irreps_in, keep_zero_outputs=True, with_bias=True, name="lin_diag"
        )
        self.lin_offdiag = NamedE3Linear(
            self.block_mapping.irreps_in, keep_zero_outputs=True, with_bias=True, name="lin_offdiag"
        )

    def __call__(
        self,
        nodes: e3nn.IrrepsArray,
        edge_ind: jax.Array,
        Y: e3nn.IrrepsArray,
        dist_features: jax.Array,
    ):
        c = self.res_center(nodes)
        n = self.res_neighbor(nodes)
        # TODO: add message passing (sums over j,k in Eq. 17 and Eq. 18); Current implementation corresponds to the ablation "simple pair features" and increases prediction errors on Ethanol from 77 to 121 µHa on F
        diag_features = self.res_diag(c)
        offdiag_features = self.pairmix_offdiag(c[edge_ind[:, 1]], n[edge_ind[:, 0]], dist_features)
        offdiag_features = self.res_offdiag(offdiag_features)

        diag_features = self.diag_norm(diag_features)
        offdiag_features = self.offdiag_norm(offdiag_features)
        diag_features = self.lin_diag(diag_features.axis_to_mul())
        offdiag_features = self.lin_offdiag(offdiag_features.axis_to_mul())

        diag_blocks = self.block_mapping(diag_features)
        offdiag_blocks = self.block_mapping(offdiag_features)
        return diag_blocks, offdiag_blocks


class E3PhisNetOverlap(hk.Module):
    def __init__(self, irreps_basis: str, L: int, n_channels: int, name: Optional[str] = None):
        super().__init__(name)
        self.irreps_basis = e3nn.Irreps(irreps_basis)

        self.sphlin_diag = E3SphLinear(L, n_channels)
        self.sphlin_offdiag = E3SphLinear(L, n_channels)
        self.pairmix = E3PairMix()

        self.res_diag = E3ResidualBlock()
        self.res_offdiag = E3ResidualBlock()

        self.block_mapping = E3PhisNetMatrixBlock(self.irreps_basis, self.irreps_basis)
        self.lin_diag = NamedE3Linear(
            self.block_mapping.irreps_in, keep_zero_outputs=True, with_bias=True, name="lin_diag"
        )
        self.lin_offdiag = NamedE3Linear(
            self.block_mapping.irreps_in, keep_zero_outputs=True, with_bias=True, name="lin_offdiag"
        )

    def __call__(
        self,
        nodes: e3nn.IrrepsArray,
        edge_ind: jax.Array,
        Y: e3nn.IrrepsArray,
        dist_features: jax.Array,
    ):
        features_diag = self.sphlin_diag(nodes)
        features_diag = self.res_diag(features_diag)
        # features_diag = scalar_activation(features_diag)
        features_diag = self.lin_diag(features_diag.axis_to_mul())
        blocks_diag = self.block_mapping(features_diag)

        nodes_snd = nodes[edge_ind[:, 0]]
        nodes_rec = nodes[edge_ind[:, 1]]
        # The PhisNet paper only uses Y; The PhisNet code uses nodes_snd for L=0 and Y for L>0; no explanation is given
        features_snd = e3nn.concatenate([Y, nodes_snd], axis=-2)
        features_offdiag = self.pairmix(nodes_rec, self.sphlin_offdiag(features_snd), dist_features)
        features_offdiag = self.res_offdiag(features_offdiag)
        # features_offdiag = scalar_activation(features_offdiag)
        features_offdiag = self.lin_offdiag(features_offdiag.axis_to_mul())
        blocks_offdiag = self.block_mapping(features_offdiag)
        return blocks_diag, blocks_offdiag


def _assemble_symmetric_matrix(diag_blocks, offdiag_blocks, edge_ind):
    n_basis = diag_blocks.shape[-1]
    n_nodes = diag_blocks.shape[-3]
    batch_dims = diag_blocks.shape[:-3]
    H = jnp.zeros(batch_dims + (n_nodes, n_nodes, n_basis, n_basis))
    H = H.at[..., np.arange(n_nodes), np.arange(n_nodes), :, :].set(diag_blocks)
    H = H.at[..., edge_ind[:, 1], edge_ind[:, 0], :, :].set(offdiag_blocks)
    # TODO clean up
    H = jnp.swapaxes(H, -2, -3)
    H = H.reshape(batch_dims + (n_nodes * n_basis, n_nodes * n_basis))
    H = (H + H.swapaxes(-1, -2)) * 0.5
    H = H.reshape(batch_dims + (n_nodes, n_basis, n_nodes, n_basis))
    H = jnp.swapaxes(H, -2, -3)
    return H


class E3PhisNet(hk.Module):
    def __init__(
        self,
        irreps_basis: str,
        n_iterations: int,
        L: int,
        n_channels: int,
        Z_max: int,
        n_rbf_features: int,
        r_max: float,
        r_scale: float,
        force_overlap_diag_to_one: bool = True,
        predict_overlap: bool = True,
        predict_hamiltonian: bool = True,
        predict_core_hamiltonian: bool = True,
        predict_density: bool = False,
        predict_energy: bool = False,
        predict_forces: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.irreps_basis = e3nn.Irreps(irreps_basis)
        self.n_iterations = n_iterations
        self.n_channels = n_channels
        self.L = L
        self.irreps = get_irreps_up_to_lmax(L, all_parities=_ALL_PARITIES)
        self.Z_max = Z_max
        self.n_rbf_features = n_rbf_features
        self.r_max = r_max
        self.r_scale = r_scale
        self.force_overlap_diag_to_one = force_overlap_diag_to_one
        self.predict_overlap = predict_overlap
        self.predict_hamiltonian = predict_hamiltonian
        self.predict_core_hamiltonian = predict_core_hamiltonian
        self.predict_density = predict_density
        self.predict_energy = predict_energy
        self.predict_forces = predict_forces
        self.predict_node_embeddings = (
            self.predict_hamiltonian
            or self.predict_core_hamiltonian
            or self.predict_density
            or self.predict_energy
            or self.predict_forces
        )
        # Atomic energies estimated using Hartree-Fock with the 6-311G basis set to roughly remove the large energy offset
        self.ATOMIC_ENERGIES = jnp.array(
            [
                -0.4998098152732818,
                -2.8598954245692076,
                -7.432026442600989,
                -14.57187393722457,
                -24.52701999144317,
                -37.59864430938627,
                -54.25588944760885,
                -74.67792595819445,
                -99.39415794702747,
                -128.52255305398998,
                -161.8459261506474,
                -199.606555987365,
                -241.87001546761337,
                -288.78930946826847,
                -340.61564987313955,
                -397.41530095059136,
                -459.46950407029686,
                -526.8066262302902,
            ]
        )

        if self.predict_node_embeddings:
            self.embedding_layers = [E3PhisNetLayer(L, n_channels) for _ in range(n_iterations)]
        if self.predict_hamiltonian:
            self.hamiltonian_mapping = E3PhisNetHamiltonian(irreps_basis, name="H")
        if self.predict_core_hamiltonian:
            self.core_hamiltonian_mapping = E3PhisNetHamiltonian(irreps_basis, name="H_core")
        if self.predict_density:
            self.density_mapping = E3PhisNetHamiltonian(irreps_basis, name="rho")
        if self.predict_overlap:
            self.overlap_mapping = E3PhisNetOverlap(irreps_basis, L, n_channels)
        if self.predict_energy or self.predict_forces:
            self.energy_layernorm = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True, name="energy_layernorm"
            )
            self.energy_mlp = MLP(
                [n_channels, n_channels, 1], MLPConfig(activation="silu"), linear_out=True, name="energy"
            )

    def embed_nodes(self, Z: jax.Array):
        # TODO: Initializing with 0 and then running all computations on these empty features in the first layer is a bit wasteful; Can we do better?
        vectors = jnp.zeros(Z.shape + (self.n_channels, (self.L + 1) ** 2 - 1))
        scalars = hk.Embed(self.Z_max, self.n_channels)(Z - 1)
        nodes = jnp.concatenate([scalars[..., None], vectors], axis=-1)
        nodes = e3nn.IrrepsArray(self.irreps, nodes)
        return nodes

    def embed_edges(self, diff: jax.Array, dist: jax.Array):
        Y = e3nn.spherical_harmonics(get_irreps_up_to_lmax(self.L, all_parities=_ALL_PARITIES), diff, normalize=True)
        Y = Y[..., None, :]  # Add channel dimension
        dist_features = get_radial_basis(dist, self.r_scale, self.n_rbf_features, self.r_max)
        return Y, dist_features

    def _get_ion_ion_energy(self, dist: jax.Array, Z: jax.Array, edge_ind: jax.Array):
        eps = 1e-6
        Zi = Z.at[edge_ind[:, 0]].get(mode="fill", fill_value=0)
        Zj = Z.at[edge_ind[:, 1]].get(mode="fill", fill_value=0)
        E_pot = Zi * Zj / (dist + eps)
        return jnp.sum(E_pot) * 0.5

    def _call_no_batch(self, R: jax.Array, Z: jax.Array, edge_ind: jax.Array):
        R = jnp.array(R, jnp.float32)
        Z = jnp.array(Z, jnp.int32)
        edge_ind = jnp.array(edge_ind, jnp.int32)
        n_ions_max = R.shape[0]
        Ri = R.at[edge_ind[:, 1]].get(mode="fill", fill_value=1000)
        Rj = R.at[edge_ind[:, 0]].get(mode="fill", fill_value=0)
        diff = Ri - Rj
        dist = jnp.linalg.norm(diff, axis=-1)

        # Lookup embedding of nodes and edges
        node_features = self.embed_nodes(Z)
        Y, dist_features = self.embed_edges(diff, dist)

        outputs = dict()
        if self.predict_overlap:
            # Overlap is calculated directly from the node features, without a deep embedding
            diag_blocks_S, offdiag_blocks_S = self.overlap_mapping(node_features, edge_ind, Y, dist_features)

            # Help overlap to be positive definite, by forcing diagonal elements to be 1
            if self.force_overlap_diag_to_one:
                n_basis = diag_blocks_S.shape[-1]
                diag_blocks_S = diag_blocks_S * (1 - jnp.eye(n_basis)) + jnp.eye(n_basis)
            outputs["S"] = _assemble_symmetric_matrix(diag_blocks_S, offdiag_blocks_S, edge_ind)

        if self.predict_node_embeddings:
            # Deep embedding of ions
            node_embeddings = None
            for layer in self.embedding_layers:
                node_features = E3ChannelNorm()(node_features)
                node_features = layer(node_features, edge_ind, Y, dist_features)
                if node_embeddings is None:
                    node_embeddings = node_features
                else:
                    node_embeddings = node_embeddings + node_features
            if len(self.embedding_layers) == 0:
                node_embeddings = node_features  # Only relevant for debugging, when no embedding layers are used
            outputs["atom_embeddings"] = node_embeddings

        if self.predict_hamiltonian:
            diag_blocks_H, offdiag_blocks_H = self.hamiltonian_mapping(node_embeddings, edge_ind, Y, dist_features)
            outputs["H"] = _assemble_symmetric_matrix(diag_blocks_H, offdiag_blocks_H, edge_ind)
        if self.predict_core_hamiltonian:
            diag_blocks_H, offdiag_blocks_H = self.core_hamiltonian_mapping(node_embeddings, edge_ind, Y, dist_features)
            outputs["H_core"] = _assemble_symmetric_matrix(diag_blocks_H, offdiag_blocks_H, edge_ind)
        if self.predict_density:
            diag_blocks_rho, offdiag_blocks_rho = self.density_mapping(node_embeddings, edge_ind, Y, dist_features)
            outputs["rho"] = _assemble_symmetric_matrix(diag_blocks_rho, offdiag_blocks_rho, edge_ind)
        if self.predict_energy or self.predict_forces:
            scalar_features = node_embeddings.filter("0e").array.reshape([n_ions_max, self.n_channels])
            scalar_features = self.energy_layernorm(scalar_features)
            atomic_energies = jnp.where(Z > 0, self.ATOMIC_ENERGIES[Z - 1], 0.0)
            atomic_energies += self.energy_mlp(scalar_features).squeeze(axis=-1)
            energy = jnp.sum(atomic_energies)
            # energy += self._get_ion_ion_energy(dist, Z, edge_ind)
            outputs["energy"] = energy
        else:
            energy = None
        return energy, outputs

    def __call__(self, R: jax.Array, Z: jax.Array, edge_ind: jax.Array):
        if self.predict_forces:
            def func(R_, Z_, edge_ind_):
                (E, outputs), forces = jax.value_and_grad(self._call_no_batch, argnums=0, has_aux=True)(
                    R_, Z_, edge_ind_
                )
                # Note that the output of this function is actually MINUS the forces
                outputs["forces"] = forces
                return E, outputs

        else:
            func = self._call_no_batch

        n_batch_dims = Z.ndim - 1
        for _ in range(n_batch_dims):
            func = hk.vmap(func, in_axes=(0, 0, 0), out_axes=(0, 0), split_rng=False)
        return func(R, Z, edge_ind)


def solve_roothaan_equations(H, S, eps=1e-8, eps_mode="sqrt_eps", canonicalize=False):
    # Orthogonalize basis
    if eps_mode.lower() == "tikhonov":
        S = S + eps * jnp.eye(S.shape[0])
        s, U = jnp.linalg.eigh(S)
        s = 1 / jnp.sqrt(s)
    elif eps_mode.lower() == "zero":
        s, U = jnp.linalg.eigh(S)
        s = jnp.where(s > eps, 1 / jnp.sqrt(s), 0.0)
    elif eps_mode.lower() == "sqrt_eps":
        s, U = jnp.linalg.eigh(S)
        s = jnp.where(s > eps, 1 / jnp.sqrt(s), 1 / jnp.sqrt(eps))
    else:
        raise ValueError(f"eps_mode {eps_mode} not recognized")
    X = U * s
    H_orth = X.T @ H @ X

    # Solve orthogonal eigenvalue problem
    mo_energy, mo_coeff_orth = jnp.linalg.eigh(H_orth)

    # Transform back to non-orhtogonal basis
    mo_coeff = X @ mo_coeff_orth

    # Canonlicalize sign of orbitals
    if canonicalize:
        sign = jnp.sign(jnp.sum(mo_coeff, axis=-2, keepdims=True))
        mo_coeff = mo_coeff * sign
    return mo_energy, mo_coeff


def _build_dummy_features(rng, Z_max, N_ions_max):
    edge_ind = np.array([[i, j] for i in range(N_ions_max) for j in range(N_ions_max) if i != j], int)
    rng, rng1, rng2 = jax.random.split(rng, 3)
    R = jax.random.normal(rng1, [N_ions_max, 3])
    Z = jax.random.randint(rng2, [N_ions_max], 1, Z_max + 1)
    diff = R[edge_ind[:, 1]] - R[edge_ind[:, 0]]
    dist = jnp.linalg.norm(diff, axis=1)
    return R, Z, edge_ind, diff, dist


def build_phisnet_model(
    phisnet_params, phisnet_config, irreps_basis, N_ions_max, rng_seed, with_hessian=True, with_forces=False
):
    rng, rng1, rng2 = jax.random.split(jax.random.PRNGKey(rng_seed), 3)

    model = hk.without_apply_rng(
        hk.transform(
            lambda R_, Z_, edge_ind_: E3PhisNet(
                irreps_basis=irreps_basis or phisnet_config.irreps_basis,
                n_iterations=phisnet_config.n_iterations,
                L=phisnet_config.L,
                n_channels=phisnet_config.n_channels,
                Z_max=phisnet_config.Z_max,
                n_rbf_features=phisnet_config.n_rbf_features,
                r_max=phisnet_config.r_cutoff,
                r_scale=phisnet_config.r_scale,
                force_overlap_diag_to_one=phisnet_config.force_overlap_diag_to_one,
                predict_overlap=phisnet_config.predict_S,
                predict_hamiltonian=phisnet_config.predict_H,
                predict_core_hamiltonian=phisnet_config.predict_H_core,
                predict_density=phisnet_config.predict_rho,
                predict_energy=phisnet_config.predict_energy,
                predict_forces=phisnet_config.predict_forces and with_forces,
            )(R_, Z_, edge_ind_)
        )
    )

    R, Z, edges, _, _ = _build_dummy_features(rng2, phisnet_config.Z_max, N_ions_max)
    params_init = model.init(rng1, R, Z, edges)
    print(get_param_size_summary(params_init))
    assert hk.data_structures.tree_size(phisnet_params) == hk.data_structures.tree_size(params_init), (
        f"Nb. of PhisNet params {hk.data_structures.tree_size(phisnet_params)} for reuse don't"
        f" align with model size {hk.data_structures.tree_size(params_init)}!"
    )

    phisnet_model = lambda R, Z, edge_ind: model.apply(phisnet_params, R, Z, edge_ind)
    if with_hessian:
        phisnet_model = build_get_hessian(phisnet_model, params_baked_in=True)

        def phisnet_model_with_hessian(R, Z, edge_ind):
            H, output = phisnet_model(R, Z, edge_ind)
            output["hessian"] = H
            return output

        return jax.jit(phisnet_model_with_hessian)
    else:
        return jax.jit(lambda R, Z, e: phisnet_model(R, Z, e)[1])


def build_get_hessian(model_func, params_baked_in=False):
    """Returns a function that computes the Hessian of the model_func with respect to the nuclear coordinates R.

    Args:
        model_func: Function that takes as input the parameters, the nuclear coordinates R, the nuclear charges Z, and the edge indices.
        params_baked_in: If True, the model_func takes as input only the nuclear coordinates R, the nuclear charges Z, and the edge indices.
    """

    if params_baked_in:

        def get_hessian(R, Z, edges):
            R_flat = R.reshape(R.shape[:-2] + (-1,))
            func = lambda R_flat: model_func(R_flat.reshape(R_flat.shape[:-1] + (-1, 3)), Z, edges)
            hessian, aux = jax.hessian(func, has_aux=True)(R_flat)
            return hessian, aux

    else:

        def get_hessian(params, R, Z, edges):
            R_flat = R.reshape(R.shape[:-2] + (-1,))
            func = lambda R_flat: model_func(params, R_flat.reshape(R_flat.shape[:-1] + (-1, 3)), Z, edges)
            hessian, aux = jax.hessian(func, has_aux=True)(R_flat)
            return hessian, aux

    return get_hessian


# TODO put into function & align with train_phisnet.py
def _build_phisnet_features(R, Z, N_ions_max, nb_orbitals_per_Z):
    max_per_orb_type = {}
    for _, orbs in nb_orbitals_per_Z.items():
        for orb_typ, nb_orb in orbs.items():
            if orb_typ in max_per_orb_type.keys() and max_per_orb_type[orb_typ] < nb_orb:
                max_per_orb_type[orb_typ] = nb_orb
            else:
                max_per_orb_type[orb_typ] = nb_orb
    N_basis = sum(max_per_orb_type.values())

    edge_connectivities = []
    for n in range(N_ions_max + 1):
        indices = [(i, j) for i in range(n) for j in range(n) if i != j]
        edge_connectivities.append(np.array(indices, dtype=np.int32))

    Z_pad = np.zeros([N_ions_max], dtype=np.int32)
    R_pad = np.zeros([N_ions_max, 3], dtype=np.float32)
    diff_pad = np.zeros([N_ions_max * (N_ions_max - 1), 3])
    edge_con = np.ones([N_ions_max * (N_ions_max - 1), 2], dtype=np.int32) * N_ions_max + 1

    n_atoms = len(Z)
    n_edges = n_atoms * (n_atoms - 1)
    R_pad[:n_atoms] = R
    Z_pad[:n_atoms] = Z
    edge_con[:n_edges] = edge_connectivities[n_atoms]
    diff_pad[:n_edges] = R_pad[edge_con[:n_edges, 0], :] - R_pad[edge_con[:n_edges, 1], :]
    dist_pad = np.linalg.norm(diff_pad, axis=-1)

    mask = np.zeros([N_ions_max, N_ions_max, N_basis, N_basis], dtype=np.float32)
    offset_row = 0
    for row, Z_row in enumerate(Z):
        orb_per_row = nb_orbitals_per_Z[Z_row]
        norb_row, norb_row_src = 0, 0
        for orb_type_row, nb_orb_per_type_row in orb_per_row.items():
            offset_col = 0
            slice_row_tgt = slice(norb_row, nb_orb_per_type_row + norb_row)

            for col, Z_col in enumerate(Z):
                orb_per_col = nb_orbitals_per_Z[Z_col]

                norb_col, norb_col_src = 0, 0
                for orb_type_col, nb_orb_per_type_col in orb_per_col.items():
                    slice_col_tgt = slice(norb_col, nb_orb_per_type_col + norb_col)
                    mask[row, col, slice_row_tgt, slice_col_tgt] = 1

                    norb_col += max_per_orb_type[orb_type_col]
                    norb_col_src += nb_orb_per_type_col
                offset_col += sum(orb_per_col.values())
            norb_row += max_per_orb_type[orb_type_row]
            norb_row_src += nb_orb_per_type_row
        offset_row += sum(orb_per_row.values())
    return R_pad, Z_pad, edge_con, diff_pad, dist_pad, mask


def unpad_matrices(mask, *matrices):
    """Unpad a matrix by removing zero rows and columns."""
    # Reshape matrix from [N, N, b, b] to [N, b, N, b] and then flatten to []
    mask = jnp.moveaxis(mask, -3, -2).flatten()
    output_size = int(mask.sum())
    n_basis_total = math.isqrt(output_size)
    assert n_basis_total**2 == output_size

    matrices = [jnp.moveaxis(m, -3, -2).flatten() for m in matrices]
    return [m[mask > 0].reshape([n_basis_total, n_basis_total]) for m in matrices]


def get_edge_connectivity(n_atoms):
    """Returns the edge connectivity matrix for a given number of atoms."""
    indices = [(i, j) for i in range(n_atoms) for j in range(n_atoms) if i != j]
    return np.array(indices, dtype=np.int32)


def get_phisnet_solution(
    physical_config, phisnet_model, basis_set, localization_method: str, N_ions_max, nb_orbitals_per_Z, atomic_orbitals=None
):
    n_el, n_up, R, Z = physical_config.get_basic_params()
    n_dn = n_el - n_up
    assert n_up == n_dn, "Only closed-shell systems are currently supported by PhisNet-style TAOs"
    assert R.ndim == 2 and Z.ndim == 1, f"R.shape = {R.shape}, Z.shape = {Z.shape}"
    N_ions = Z.shape[0]

    R_pad, Z_pad, edge, _, _, mask = _build_phisnet_features(R, Z, N_ions_max, nb_orbitals_per_Z)
    outputs = phisnet_model(R_pad, Z_pad, edge)

    # Atomic embeddings for Deep-Erwin embedding
    node_embeddings = outputs["atom_embeddings"]
    node_embeddings = np.reshape(np.array(node_embeddings.array)[:N_ions], newshape=Z.shape[:1] + (-1,))

    # Predicted energy and hessian for sampling of geometries
    E_hf = outputs.get("energy", np.nan)
    hessian = outputs.get("hessian")
    if hessian is not None:
        hessian = hessian[: N_ions * 3, : N_ions * 3]

    # Compute mo_coeffs from Fock- and overlap-matrix for Transferrable Atomic Orbitals
    H_pred, S_pred = outputs.get("H"), outputs.get("S")
    mo_coeff, mo_energies = None, None
    if (H_pred is not None) and (S_pred is not None):
        H_pred, S_pred = unpad_matrices(mask, H_pred, S_pred)
        mo_energies, mo_coeff = solve_roothaan_equations(H_pred, S_pred)

        # Converting to numpy array to do in-place operations
        mo_coeff, mo_energies = np.array(mo_coeff), np.array(mo_energies)
        if localization_method:
            mol = build_pyscf_molecule_from_physical_config(physical_config, basis_set)
            mo_coeff[:, :n_up], mo_energies[:n_up] = localize_orbitals(
                mol, mo_coeff[:, :n_up], localization_method, mo_energies[:n_up]
            )
        mo_coeff = [mo_coeff, mo_coeff]
        mo_energies = [mo_energies, mo_energies]

    # Remaining info for feature generation for TAOs
    if atomic_orbitals is None:
        molecule = build_pyscf_molecule_from_physical_config(physical_config, basis_set)
        atomic_orbitals = _get_atomic_orbital_basis_functions(molecule)
    ind_orbitals = [
        np.tile(np.arange(n_up), 1).reshape((-1, n_up)),
        np.tile(np.arange(n_dn), 1).reshape((-1, n_dn)),
    ]

    orbital_params = OrbitalParamsHF(
        atomic_orbitals=atomic_orbitals,
        mo_coeff=mo_coeff,
        mo_energies=mo_energies,
    )
    return orbital_params, node_embeddings, hessian, dict(E_phisnet=E_hf)


if __name__ == "__main__":
    # Generate dummy data
    n_nodes = 2
    Z_max = 9
    irreps_basis = "2x0e+1x1o"

    rng = jax.random.PRNGKey(0)
    edge_ind = np.array([[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j], int)
    rng, rng1, rng2 = jax.random.split(rng, 3)
    R = jax.random.normal(rng1, [n_nodes, 3])
    Z = jax.random.randint(rng2, [n_nodes], 1, Z_max + 1)
    diff = R[edge_ind[:, 1]] - R[edge_ind[:, 0]]
    dist = jnp.linalg.norm(diff, axis=1)

    rng, rng1 = jax.random.split(rng)
    model = hk.without_apply_rng(
        hk.transform(
            lambda Z_, edge_ind_, diff_, dist_: E3PhisNet(
                irreps_basis, n_iterations=1, L=2, n_channels=7, Z_max=Z_max, n_rbf_features=5, r_max=10.0, r_scale=4.0
            )(Z_, edge_ind_, diff_, dist_)
        )
    )

    params = model.init(rng1, Z, edge_ind, diff, dist)
    nodes, H, S = model.apply(params, Z, edge_ind, diff, dist)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print("H.shape", H.shape)
    print("S.shape", S.shape)
    print("end")


# %%
