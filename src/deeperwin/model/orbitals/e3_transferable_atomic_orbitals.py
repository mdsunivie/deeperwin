import jax
import numpy as np
from deeperwin.configuration import E3TransferableAtomicOrbitalsConfig, MLPConfig, E3SymmetricTensorProductMLPConfig, E3OrbitalTransformerConfig
from deeperwin.model.definitions import Embeddings, DiffAndDistances
from deeperwin.model.mlp import MLP, get_rbf_features
import haiku as hk
import e3nn_jax as e3nn
import jax.numpy as jnp
from deeperwin.model.e3nn_utils import (
    GVP,
    swapaxes_e3,
    symmetrize_e3_func,
    antisymmetrize_e3_func,
    NamedE3Linear,
    TensorProductNet,
    _tile_channels,
    tile_e3,
    E3MACEGNN,
)
from deeperwin.model.e3nn_utils import get_irreps_up_to_lmax
from typing import Optional, Dict



class OrbitalAttention(hk.Module):
    """
    Attention across orbitals for each atom.

    Args:
        n_iterations: Number of iterations of attention
        attention_irreps: Irreps of the q,k,v of the attention
    """

    def __init__(self, attention_irreps, n_heads, name=None):
        super().__init__(name=name)
        self.attention_irreps = e3nn.Irreps(attention_irreps)
        self.n_heads = n_heads
        self._attn_normalization = 1.0 / np.sqrt(e3nn.Irreps(attention_irreps).num_irreps)


    def __call__(self, orb_features):
        # Atom-wise attention
        q = NamedE3Linear(self.attention_irreps * self.n_heads, name=f"q")(orb_features)
        k = NamedE3Linear(self.attention_irreps * self.n_heads, name=f"k")(orb_features)
        v = NamedE3Linear(self.attention_irreps * self.n_heads, name=f"v")(orb_features)
        q = q.mul_to_axis(self.n_heads, -2)
        k = k.mul_to_axis(self.n_heads, -2)
        v = v.mul_to_axis(self.n_heads, -2)
        weight = e3nn.elementwise_tensor_product(q[..., :, None, :, :], k[..., None, :, :, :], filter_ir_out=["0e"]).array
        weight = jnp.sum(weight, axis=-1)  # inner product
        weight = jax.nn.softmax(weight * self._attn_normalization, axis=-2)
        update = e3nn.sum(weight[..., None] * v[..., None, :, :, :], axis=-3)
        update = update.axis_to_mul(-2)

        # Resdidual
        if orb_features.irreps == update.irreps:
            orb_features += update
        else:
            orb_features = update
        return orb_features
    

class EquivariantAttention(hk.Module):
    def __init__(self, attention_irreps, n_heads:int, name=None):
        super().__init__(name=name)
        self.attention_irreps = e3nn.Irreps(attention_irreps)
        self.n_heads = n_heads
        self._attn_normalization = 1.0 / np.sqrt(e3nn.Irreps(attention_irreps).num_irreps)
        self.n_hidden_edge_mlp = [64, 64]

    def __call__(self, node_feat: e3nn.IrrepsArray, edge_attr: e3nn.IrrepsArray, edge_sh: e3nn.IrrepsArray):
        # Edge-wise attention
        q = NamedE3Linear(self.attention_irreps * self.n_heads, name=f"q")(node_feat)
        k = NamedE3Linear(self.attention_irreps * self.n_heads, name=f"k")(node_feat)
        v = NamedE3Linear(self.attention_irreps * self.n_heads, name=f"v")(node_feat)
        q = q.mul_to_axis(self.n_heads, -2)
        k = k.mul_to_axis(self.n_heads, -2)
        v = v.mul_to_axis(self.n_heads, -2)

        # Compute additional weights that depend on the edge features e_ij, <x_i, Y_ij>, <x_j, Y_ij>
        edge_feat_rec = e3nn.tensor_product(node_feat[..., :, None, :], edge_sh, filter_ir_out=["0e"]).array
        edge_feat_snd = e3nn.tensor_product(node_feat[..., None, :, :], edge_sh, filter_ir_out=["0e"]).array
        edge_feat = jnp.concatenate([edge_attr, edge_feat_rec, edge_feat_snd], axis=-1)
        edge_weight = MLP(self.n_hidden_edge_mlp + [self.n_heads], MLPConfig(), name="edge_mlp")(edge_feat)

        weight = e3nn.elementwise_tensor_product(q[..., :, None, :, :], k[..., None, :, :, :], filter_ir_out=["0e"]).array
        weight = jnp.sum(weight, axis=-1) * self._attn_normalization + edge_weight
        weight = jax.nn.softmax(weight, axis=-2)
        update = e3nn.sum(weight[..., :, :, :, None] * v[..., None, :, :, :], axis=-3)
        update = update.axis_to_mul(-2)

        # Resdidual
        if node_feat.irreps == update.irreps:
            node_feat += update
        else:
            node_feat = update
        return node_feat
    

class OrbitalTransformer(hk.Module):
    def __init__(self, config: E3OrbitalTransformerConfig, name: Optional[str] = None):
        super().__init__(name)
        self.config = config
        self.attention_irreps = e3nn.Irreps(config.attention_irreps)
        self.l_max = self.attention_irreps.lmax
        self.rbf_n_features = 32
        self.rbf_width = 5.0

        self.orb_attn = []
        self.ion_attn = []
        for n in range(config.n_iterations):
            self.orb_attn.append(OrbitalAttention(self.attention_irreps, config.n_heads, name=f"orb_attn_{n}"))
            self.ion_attn.append(EquivariantAttention(self.attention_irreps, config.n_heads, name=f"ion_attn_{n}"))


    def get_positional_encodings(self, n_orbitals, n_occ):
        encodings = []

        # Encoding of orbital "position" (i.e. energetic ordering)
        pos_orb = (np.arange(n_orbitals) - n_occ) / n_orbitals
        freq = np.pi * np.arange(1, self.config.orb_order_encoding_dim+1)
        feat_pos = np.sin(pos_orb[:, None] * freq[None, :])
        encodings.append(feat_pos)
        feat_pos = np.cos(pos_orb[:, None] * freq[None, :])
        encodings.append(feat_pos)

        # Encoding of orbital occupation
        feat_occ = np.concatenate([np.ones(n_occ), -np.ones(n_orbitals - n_occ)])
        encodings.append(feat_occ[:, None])

        # TODO: include orbital energy as feature?

        # Add constant to serve as bias-term
        # encodings.append(jnp.ones([n_orbitals, 1]))
        return np.concatenate(encodings, axis=-1)


    def _call_no_batch(self, orb_features: e3nn.IrrepsArray, ion_ion_diff, ion_ion_dist, n_occ):
        """
        Args:
            orb_features: (ion, orbital, features (lm))    
            ion_ion_dist: (n_ions, n_ions)
            ion_ion_diff: (n_ions, n_ions, 3)
            n_occ: Number of occupied orbitals
        """
        edge_Y = e3nn.spherical_harmonics(get_irreps_up_to_lmax(self.l_max), ion_ion_diff, normalize=False)
        edge_attr = get_rbf_features(ion_ion_dist, self.rbf_n_features, self.rbf_width)

        for ion_attn, orb_attn in zip(self.ion_attn, self.orb_attn):
            # Attention across ions, vmapped over orbitals
            orb_features = hk.vmap(ion_attn, in_axes=(1, None, None), out_axes=1, split_rng=False)(orb_features, edge_attr, edge_Y)
            # Attention across orbitals, vmapped over ions
            orb_features = hk.vmap(orb_attn, in_axes=(0,), out_axes=0, split_rng=False)(orb_features)
        orb_features = orb_features[..., :n_occ, :]
        return orb_features
    
    
    def __call__(self, orb_features: e3nn.IrrepsArray, ion_ion_diff, ion_ion_dist, n_occ):
        # Add orbital ranking encoding
        n_orbitals = orb_features.shape[-2]
        orb_encoding = self.get_positional_encodings(n_orbitals, n_occ)
        orb_encoding = jnp.tile(orb_encoding, orb_features.shape[:-2] + (1,1))
        orb_encoding = e3nn.IrrepsArray(f"{orb_encoding.shape[-1]}x0e", orb_encoding)
        orb_features = e3nn.concatenate([orb_features, orb_encoding], axis=-1)

        # vmap over batch dimensions
        f = self._call_no_batch
        n_batch_dims = orb_features.ndim - 3
        for _ in range(n_batch_dims):
            f = hk.vmap(f, in_axes=(0, 0, 0, None), out_axes=0, split_rng=False)
        return f(orb_features, ion_ion_diff, ion_ion_dist, n_occ)


class E3TransferableAtomicOrbitals(hk.Module):
    """
    SE(3)-equivariant orbitals, based on Hartree-Fock orbital coefficients.

    Args:
        config: Configuration of the model
        n_dets: Number of determinants
        determinant_schema: Schema of the determinants. Currently only "full_det" is supported.
        name: Name of the module
    """

    def __init__(
        self,
        config: E3TransferableAtomicOrbitalsConfig,
        n_dets: int,
        determinant_schema: str,
        name="e3_generalized_atomic_orbitals",
    ):
        super().__init__(name=name)
        self.n_dets = n_dets
        self.determinant_schema = determinant_schema
        self.config = config
        # TODO: implement or assert in config
        assert (
            determinant_schema == "full_det"
        ), f"E3-TAO currently not implemented for determinant_schema {determinant_schema}"

    @hk.experimental.name_like("__call__")
    def get_exponents_and_backflows(
        self, orbital_features: e3nn.IrrepsArray, n_up, n_dn, el_emb_dim, tile_dims, ion_ion_diff, ion_ion_dist
    ):
        """
        Calculate the exponents and backflows for given orbitals.

        Args:
            orbital_features: Orbital features
            n_up: Number of up electrons
            n_dn: Number of down electrons
            el_emb_dim: Dimension of the electron embedding
            tile_dims: Batch-dimension across which all operations are tiled
            ion_ion_diff: 3D-difference vectors between ions
            ion_ion_dist: distances between ions
        """
        exponents = [None, None]
        backflows = [None, None]

        if self.config.n_virtual_orbitals is None:
            n_virt_orb = max(n_up, n_dn)
        else:
            n_virt_orb = self.config.n_virtual_orbitals
        orbital_features = orbital_features[..., : max(n_up, n_dn) + n_virt_orb, :]

        # KFAC requires all operations to have the same leading (batch)-dimension.
        # Therefore If the data has a batch-dim, we tile the orbital features across samples
        # TODO: remove tiling, as soon as kfac supports non-batched operations
        # TODO: Currently need to convert from e3nn array to jnp array and back again for tiling
        # TODO: It should be possible to vmap without moving the orbital dim to the front (with swapaxes_e3) and moving it back afterwards.
        #       However, e3nn currently behaves weirdly, when the vmap is applied to a dim other than 0.
        orbital_features = tile_e3(
            orbital_features, (*tile_dims, 1, 1, 1, 1)
        )  # [batch x spin x ions x orbitals x features]
        # orbital_features = swapaxes_e3(orbital_features, -3, -2)  # [batch x spin x orbitals x ions x features]
        # gnn = E3MACEGNN(self.config.orb_feature_gnn)
        # for _ in range(2):
        #     gnn = jax.vmap(gnn, in_axes=[0, None, None])  # vmap over spin and orbitals
        # for _ in range(len(tile_dims)):
        #     gnn = jax.vmap(gnn, in_axes=[0, 0, 0])  # vmap over batch-dim
        # orbital_features = gnn(orbital_features, ion_ion_diff, ion_ion_dist)
        # orbital_features = swapaxes_e3(orbital_features, -3, -2)  # [batch x spin x ions x orbitals x features]

        for spin, n_occ in enumerate([n_up, n_dn]):
            orb_trans = OrbitalTransformer(self.config.orb_transformer, name=f"orb_transf_{spin}")
            orb_features = orbital_features[..., spin, :, :, :]  # [batch x ions x orbitals x features
            orb_features = orb_trans(orb_features, ion_ion_diff, ion_ion_dist, n_occ)
            orb_features = orb_features[..., :n_occ, :]
            exponents[spin] = E3TAOExponents(
                self.config.envelope, self.n_dets, self.config.symmetrize_exponent_mlp, f"envelope_{spin}"
            )(orb_features)

            backflows[spin] = E3TAOBackflow(
                self.config.backflow,
                self.config.backflow_l_max_out,
                el_emb_dim,
                self.n_dets,
                self.config.antisymmetrize_backflow_mlp,
                f"backflow_{spin}",
            )(orb_features)
            # [batch x ions x orb x same / diff x dets x features(emb_dim * lm)]
        return exponents, backflows

    def get_fixed_rank_backflow(self, spin, ind_orb, ranks, n_ions, el_emb_dim, batch_dims):
        features = [
            e3nn.IrrepsArray("0e", jnp.ones(1)),
            e3nn.IrrepsArray("1o", jnp.array([1.0, 0, 0])),
            e3nn.IrrepsArray("1o", jnp.array([0, 1.0, 0])),
            e3nn.IrrepsArray("1o", jnp.array([0, 0, 1.0])),
        ]
        features = e3nn.concatenate([features[i] for i in ranks], axis=-1)
        features = tile_e3(
            features,
            (
                *batch_dims,
                1,
            ),
        )
        bf = NamedE3Linear(
            get_irreps_up_to_lmax(1, n_ions * 2 * self.n_dets * el_emb_dim),
            keep_zero_outputs=True,
            name=f"bf_e3lin_{spin}_{ind_orb}",
        )(features)
        bf = bf.mul_to_axis(factor=n_ions)
        bf = bf.mul_to_axis(factor=2)
        bf = bf.mul_to_axis(factor=self.n_dets)
        return bf

    def __call__(
        self,
        diff_dist: DiffAndDistances,
        embeddings: Embeddings,
        n_ions: int,
        n_up: int,
        n_dn: int,
        orbital_features: e3nn.IrrepsArray,  # [spin, n_ions, n_orbitals, feature_dim],
        orbital_energies: jnp.ndarray,  # [spin, n_orbitals]
        cache: Optional[Dict] = None,
    ):
        emb_el = embeddings.el
        emb_dim = emb_el.shape[-1]
        batch_dims = emb_el.shape[:-2]

        if cache is None:
            exponents, backflows = self.get_exponents_and_backflows(
                orbital_features,
                n_up,
                n_dn,
                emb_el.shape[-1],
                batch_dims,
                diff_dist.diff_ion_ion,
                diff_dist.dist_ion_ion,
            )
        else:
            exponents, backflows = cache["exponents"], cache["backflows"]

        Y_el_ion = e3nn.spherical_harmonics(
            get_irreps_up_to_lmax(backflows[0].irreps.lmax), diff_dist.diff_el_ion, normalize=False
        )

        mos = []
        for spin, (exponent, backflow) in enumerate(zip(exponents, backflows)):
            if spin == 0:
                slice_same = slice(None, n_up)
                slice_diff = slice(n_up, None)
            else:
                slice_same = slice(n_up, None)
                slice_diff = slice(None, n_up)

            # [batch x ions x orb x spin-type x dets]
            envelope_same = jnp.exp(
                -exponent[..., None, :, :, 0, :] * diff_dist.dist_el_ion[..., slice_same, :, None, None]
            )
            envelope_diff = jnp.exp(
                -exponent[..., None, :, :, 1, :] * diff_dist.dist_el_ion[..., slice_diff, :, None, None]
            )

            # Input shapes: [batch x el (spin) x ion x orb x det x features (incl lm)]
            bf_same = e3nn.tensor_product(
                backflow[..., None, :, :, 0, :, :],  # add electron dim
                Y_el_ion[..., slice_same, :, None, None, :],  # add orbital, det dim
                filter_ir_out=["0e"],
            ).array
            bf_diff = e3nn.tensor_product(
                backflow[..., None, :, :, 1, :, :],  # add electron dim
                Y_el_ion[..., slice_diff, :, None, None, :],  # add orbital, same/diff, det dim
                filter_ir_out=["0e"],
            ).array
            bf_same = bf_same.reshape(bf_same.shape[:-1] + (emb_dim, -1)).sum(axis=-1)  # sum over l
            bf_diff = bf_diff.reshape(bf_diff.shape[:-1] + (emb_dim, -1)).sum(axis=-1)  # sum over l

            # I = ions, k = orbitals, i = electrons (same/diff), d = determinants, a = embedding dimension
            mo_same = jnp.einsum("...iIkda,...ia,...iIkd->...dik", bf_same, emb_el[..., slice_same, :], envelope_same)
            mo_diff = jnp.einsum("...iIkda,...ia,...iIkd->...dik", bf_diff, emb_el[..., slice_diff, :], envelope_diff)
            mos.append((mo_same, mo_diff))

        mo_up = jnp.concatenate([mos[0][0], mos[1][1]], axis=-1)
        mo_dn = jnp.concatenate([mos[0][1], mos[1][0]], axis=-1)
        return mo_up, mo_dn


class E3TAOExponents(hk.Module):
    """
    Compute exponents for the Transferrable Atomic Orbitals (TAO).

    Args:
        e3tp_config: Configuration for the E(3) tensor product network.
        n_dets: Number of determinants.
    """

    def __init__(self, e3tp_config: E3SymmetricTensorProductMLPConfig, n_dets, symmetrize=True, name=None):
        super().__init__(name=name)
        output_dim = 2 * n_dets  # same/diff x n_dets
        self._n_dets = n_dets
        self.mlp = TensorProductNet(
            e3tp_config.depth,
            e3tp_config.l_max,
            e3tp_config.n_channels,
            e3tp_config.order,
            l_max_out=0,
            use_activation=e3tp_config.use_activation,
            activate_final=False,
            use_layer_norm=False,
            scalar_activation=e3tp_config.scalar_activation,
            gate_activation_even=e3tp_config.gate_activation,
        )
        self.lin_out = NamedE3Linear(f"{output_dim}x0e", name="lin_out")
        if symmetrize:
            self.mlp = symmetrize_e3_func(self.mlp)

    def __call__(self, mo_coeffs: e3nn.IrrepsArray):
        exponents = self.mlp(mo_coeffs)
        exponents = self.lin_out(exponents).array
        exponents = exponents.reshape(
            exponents.shape[:-1] + (2, self._n_dets)
        )  # [(batch) x ion x orb x same/diff x dets]
        exponents = jax.nn.softplus(exponents)
        return exponents


class E3TAOBackflow(hk.Module):
    def __init__(
        self,
        e3tp_config: E3SymmetricTensorProductMLPConfig,
        l_max_out: int,
        el_emb_dim: int,
        n_dets: int,
        antisymmetrize=True,
        name=None,
    ):
        super().__init__(name=name)
        irreps_out = get_irreps_up_to_lmax(l_max_out, 2 * n_dets * el_emb_dim)
        self.n_dets = n_dets
        self.mlp = GVP(["128x0e+128x1o"]*2, 
        activate_final=False,
        )
        # self.mlp = TensorProductNet(
        #     e3tp_config.depth,
        #     e3tp_config.l_max,
        #     e3tp_config.n_channels,
        #     e3tp_config.order,
        #     l_max_out,
        #     e3tp_config.use_activation,
        #     activate_final=False,
        #     use_layer_norm=False,
        #     scalar_activation=e3tp_config.scalar_activation,
        #     gate_activation_even=e3tp_config.gate_activation,
        # )
        self.lin_out = NamedE3Linear(irreps_out, name="lin_out")
        if antisymmetrize:
            self.mlp = antisymmetrize_e3_func(self.mlp)

    def __call__(self, mo_coeffs: e3nn.IrrepsArray):
        bf = self.mlp(mo_coeffs)  # [(batch) x ion x orbital x features]
        bf = self.lin_out(bf)
        bf = bf.mul_to_axis(2 * self.n_dets)
        irreps = bf.irreps

        bf = bf.array.reshape(bf.shape[:-2] + (2, self.n_dets, -1))
        bf = e3nn.IrrepsArray(irreps, bf)  # [batch x ions x orb x same/diff x dets x features (emb_dim * lm)]
        return bf
