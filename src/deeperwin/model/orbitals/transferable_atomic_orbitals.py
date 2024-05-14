from typing import Optional, Dict
import numpy as np
import haiku as hk
import jax
from jax import numpy as jnp
from deeperwin.configuration import MLPConfig, TransferableAtomicOrbitalsConfig, OrbFeatureDenseGNNConfig
from deeperwin.model import MLP, antisymmetrize, symmetrize, DiffAndDistances, Embeddings
from deeperwin.model.mlp import get_rbf_features
from deeperwin.model.gnn import DenseGNN, ScaleFeatures
from deeperwin.utils.utils import build_complex

class TAOBackflow(hk.Module):
    def __init__(
        self,
        width,
        depth,
        emb_dim,
        determinant_schema,
        n_dets,
        antisymmetrize,
        mlp_config: MLPConfig,
        is_complex,
        output_width=None,
        name=None,
    ):
        super().__init__(name=name)
        self.n_hidden = [width] * depth
        if output_width and self.n_hidden:
            self.n_hidden[-1] = output_width
        self._n_dets = n_dets
        self._output_dim = n_dets * emb_dim
        self._determinant_schema = determinant_schema
        self._mlp_config = mlp_config
        self._antisymmetrize = antisymmetrize
        self._is_complex = is_complex
        if self._determinant_schema == "full_det":
            self._output_dim *= 2
        if is_complex:
            self._output_dim *= 2

    def __call__(self, x):
        bf_func = MLP(self.n_hidden, self._mlp_config, linear_out=False)
        if self._antisymmetrize:
            bf_func = antisymmetrize(bf_func)
        bf = bf_func(x)
        bf = hk.Linear(self._output_dim, with_bias=False, name="lin_out")(bf)
        if self._determinant_schema == "full_det":
            bf = bf.reshape(bf.shape[:-1] + (2, self._n_dets, -1))
        else:
            bf = bf.reshape(bf.shape[:-1] + (self._n_dets, -1))
        if self._is_complex:
            bf = build_complex(bf)
        return bf


class TAOExponents(hk.Module):
    def __init__(
        self,
        width,
        depth,
        determinant_schema,
        n_dets,
        generate_prefactors,
        symmetrize,
        use_squared_input,
        mlp_config: MLPConfig,
        name=None,
    ):
        super().__init__(name=name)
        self._width = width
        self._depth = depth
        self._n_dets = n_dets
        self._determinant_schema = determinant_schema
        self._mlp_config = mlp_config
        self._symmetrize = symmetrize
        self._output_dim = n_dets
        self._use_squared_input = use_squared_input
        self._generate_prefactors = generate_prefactors
        self.output_shape = []
        if determinant_schema == "full_det":
            self.output_shape.append(2)
        self.output_shape.append(n_dets)
        if generate_prefactors:
            self.output_shape.append(2)
        self.output_shape = tuple(self.output_shape)
        self._output_dim = int(np.prod(self.output_shape))

    def __call__(self, x):
        if self._use_squared_input:
            z = x**2
            x = jnp.concatenate([x, z], axis=-1)
        exp_func = MLP([self._width] * self._depth + [self._output_dim], self._mlp_config, linear_out=True)
        if self._symmetrize:
            exp_func = symmetrize(exp_func)

        output = exp_func(x)
        output = output.reshape(output.shape[:-1] + self.output_shape)  # [input_dims x (spin) x dets x (exp/prefac)]
        # TODO: this bias of +1 could be put into the initialization of the last layer bias
        if self._generate_prefactors:
            output = output.at[..., 0].set(jax.nn.softplus(1 + output[..., 0]))
            output = output.at[..., 1].add(1.0)
        else:
            output = jax.nn.softplus(1 + output)
        return output


class OrbitalFeatureGNN(hk.Module):
    def __init__(self, config: OrbFeatureDenseGNNConfig, mlp_config: MLPConfig, n_edge_features, name=None):
        super().__init__(name=name)
        self.config = config
        self.mlp_config = mlp_config
        self.gnn = DenseGNN(config, mlp_config)
        self.n_edge_features = n_edge_features

    def __call__(
        self,
        orb_features: jax.Array,
        ion_ion_diff: jax.Array,
        ion_ion_dist: jax.Array,
    ):
        orb_features = jnp.moveaxis(orb_features, -2, -3)  # [... x ion x orb x features] -> [... orb x ion x features]
        # Calculate edge features as (rbf(|r|), rbf(|r|) * x, rbf(|r|) * y, rbf(|r|) * z)
        if self.config.mlp_edge_depth > 0:
            ion_ion_edges = jnp.concatenate([ion_ion_diff, ion_ion_dist[..., None]], axis=-1)
            ion_ion_edges = MLP([self.n_edge_features] * self.config.mlp_edge_depth, self.mlp_config, linear_out=False)(
                ion_ion_edges
            )
            ion_ion_edges *= ScaleFeatures(self.config.exp_scaling, self.n_edge_features, name="ion_ion_orbitals")(
                ion_ion_dist[..., None]
            )
        else:
            rbfs = get_rbf_features(ion_ion_dist, n_features=self.n_edge_features, r_max=5.0)
            diff_features = jnp.concatenate([ion_ion_diff, jnp.ones_like(ion_ion_dist)[..., None]], axis=-1)
            ion_ion_edges = rbfs[..., None, :] * diff_features[..., :, None]
            ion_ion_edges = ion_ion_edges.reshape([*ion_ion_edges.shape[:-2], -1])
        ion_ion_edges = ion_ion_edges[..., None, :, :, :]  # add dummy-dim for orbitals
        if self.config.antisymmetrize:
            ion_ion_edges = ion_ion_edges[..., None, :, :, :]  # add dummy-dim for antisymmetrization
            orb_features = antisymmetrize(lambda nodes: self.gnn(nodes, edges=ion_ion_edges)[0], tmp_axis=-4)(
                orb_features
            )
        else:
            orb_features = self.gnn(orb_features, edges=ion_ion_edges)[0]
        orb_features = jnp.moveaxis(orb_features, -3, -2)  # => [... x ion x orb x features]
        return orb_features


class TransferableAtomicOrbitals(hk.Module):
    def __init__(
        self,
        config: TransferableAtomicOrbitalsConfig,
        n_dets: int,
        determinant_schema: str,
        is_complex: bool,
        name="generalized_atomic_orbitals",
    ):
        super().__init__(name=name)
        self.n_dets = n_dets
        self.determinant_schema = determinant_schema
        self.config = config
        self.sep_ion_sums = config.use_separate_ion_sum_for_envelopes
        self.is_complex = is_complex
        # TODO: implement or assert in config
        assert determinant_schema in [
            "full_det"
        ], f"TAO currently not implemented for determinant_schema {determinant_schema}"

    @hk.experimental.name_like("__call__")
    def get_exponents_and_backflows(
        self, orbital_features: jax.Array, n_up, n_dn, el_emb_dim, tile_dims, ion_ion_diff, ion_ion_dist
    ):
        exponents = [None, None]
        backflows = [None, None]

        backflow_final_hidden = None
        backflow_dim = el_emb_dim
        backflow_n_dets = self.n_dets
        envelope_n_dets = self.n_dets

        for spin, (orb_features, n_orbitals) in enumerate(zip(orbital_features, [n_up, n_dn])):
            orb_features = orb_features[..., :n_orbitals, :]

            # KFAC requires all operations to have the same leading (batch)-dimension.
            # Therefore If the data has a batch-dim, we tile the orbital features across samples
            # TODO: remove tiling, as soon as kfac supports non-batched operations
            orb_features = jnp.tile(orb_features, (*tile_dims, 1, 1, 1))  # [batch x ions x orbitals x features]

            if self.config.orb_feature_gnn and self.config.orb_feature_gnn.n_iterations > 0:
                orb_features = OrbitalFeatureGNN(
                    self.config.orb_feature_gnn,
                    self.config.mlp,
                    self.config.n_rbf_features_orb_gnn,
                    name=f"orbital_gnn_{spin}",
                )(
                    orb_features,
                    ion_ion_diff,
                    ion_ion_dist,
                )

            # Backflow function f(c)
            backflow_func = TAOBackflow(
                self.config.backflow_width,
                self.config.backflow_depth,
                backflow_dim,
                self.determinant_schema,
                backflow_n_dets,
                self.config.antisymmetrize_backflow_mlp,
                self.config.mlp,
                self.is_complex,
                backflow_final_hidden,
                name=f"backflow_{spin}",
            )
            backflows[spin] = backflow_func(orb_features)  # [batch x ions x orb x spin-type x dets x emb_dm]

            # Exponents g(c)
            if self.config.use_exponentials:
                envelope_func = TAOExponents(
                    self.config.envelope_width,
                    self.config.envelope_depth,
                    self.determinant_schema,
                    envelope_n_dets,
                    self.config.use_separate_ion_sum_for_envelopes,
                    self.config.symmetrize_exponent_mlp,
                    self.config.use_squared_envelope_input,
                    self.config.mlp,
                    name=f"exponent_{spin}",
                )
                exponents[spin] = envelope_func(orb_features)  # [batch x ions x orb x spin-type x dets]
        return exponents, backflows

    def _get_scalar_product(self, embeddings, backflows, slice_same, slice_diff):
        b_same = backflows[..., :, :, 0, :, :]
        b_diff = backflows[..., :, :, 1, :, :]
        if self.config.use_el_ion_embedding:
            h_same = embeddings.el_ion[..., slice_same, :, :]
            h_diff = embeddings.el_ion[..., slice_diff, :, :]
        else:
            h_same = embeddings.el[..., slice_same, :]
            h_diff = embeddings.el[..., slice_diff, :]
        
        
        if self.sep_ion_sums:
            if self.config.use_el_ion_embedding:
                # Sum over ions and embedding dimension
                mo_same = jnp.einsum("...Ikda,...iIa->...dik", b_same, h_same)
                mo_diff = jnp.einsum("...Ikda,...iIa->...dik", b_diff, h_diff)
            else:
                # Sum over ions first, then over embedding dimension
                mo_same = jnp.einsum("...kda,...ia->...dik", jnp.sum(b_same, axis=-4), h_same)
                mo_diff = jnp.einsum("...kda,...ia->...dik", jnp.sum(b_diff, axis=-4), h_diff)
        else:
            if self.config.use_el_ion_embedding:
                # Sum over embedding dimension
                mo_same = jnp.einsum("...Ikda,...iIa->...diIk", b_same, h_same)
                mo_diff = jnp.einsum("...Ikda,...iIa->...diIk", b_diff, h_diff)
            else:
                # Sum over embedding dimension
                mo_same = jnp.einsum("...Ikda,...ia->...diIk", b_same, h_same)
                mo_diff = jnp.einsum("...Ikda,...ia->...diIk", b_same, h_diff)
        return mo_same, mo_diff
    
    def _get_envelope(self, diff_dist, exponent, slice_same, slice_diff):
        dist_el_ion_same = diff_dist.dist_el_ion[..., slice_same, :]
        dist_el_ion_diff = diff_dist.dist_el_ion[..., slice_diff, :]

        if self.sep_ion_sums:
            exponent, prefac = exponent[..., 0], exponent[..., 1]

        # => [batch x el x ion x orb x dets]
        exponent_same = exponent[..., None, :, :, 0, :] * dist_el_ion_same[..., :, :, None, None]
        exponent_diff = exponent[..., None, :, :, 1, :] * dist_el_ion_diff[..., :, :, None, None]
        env_same = jnp.exp(-exponent_same)
        env_diff = jnp.exp(-exponent_diff)
        if self.sep_ion_sums:
            env_same *= prefac[..., None, :, :, 0, :]
            env_diff *= prefac[..., None, :, :, 1, :]
            # => [batch x el x orb x det] => [batch x det x el x orb]
            env_same = np.moveaxis(jnp.sum(env_same, axis=-3), -1, -3)
            env_diff = np.moveaxis(jnp.sum(env_diff, axis=-3), -1, -3)
        else:
            # => [batch x det x el x ion x orb]
            env_same = np.moveaxis(env_same, -1, -4)
            env_diff = np.moveaxis(env_diff, -1, -4)
        return env_same, env_diff
        
            

    def __call__(
        self,
        diff_dist: DiffAndDistances,
        embeddings: Embeddings,
        n_ions: int,
        n_up: int,
        n_dn: int,
        orbital_features: jnp.array,  # [n_ions, n_orbitals, feature_dim]
        cache: Optional[Dict] = None,
    ):
        batch_dims = embeddings.el.shape[:-2]
        embedding_dim = embeddings.el_ion.shape[-1] if self.config.use_el_ion_embedding else embeddings.el.shape[-1]

        if cache is None:
            exponents, backflows = self.get_exponents_and_backflows(
                orbital_features,
                n_up,
                n_dn,
                embedding_dim,
                batch_dims,
                diff_dist.diff_ion_ion,
                diff_dist.dist_ion_ion,
            )
        else:
            exponents, backflows = cache["exponents"], cache["backflows"]

        # orb_features are of shape [n_ions x n_orbitals (n_up, n_dn) x feature_dim]
        mos = []
        for spin, (exponent, backflow) in enumerate(zip(exponents, backflows)):
            if spin == 0:
                slice_same, slice_diff = slice(None, n_up), slice(n_up, None)
            else:
                slice_same, slice_diff = slice(n_up, None), slice(None, n_up)

            mo_same, mo_diff = self._get_scalar_product(embeddings, backflow, slice_same, slice_diff)
            if self.config.use_exponentials:
                env_same, env_diff = self._get_envelope(diff_dist, exponent, slice_same, slice_diff)
                mo_same *= env_same
                mo_diff *= env_diff
            if not self.sep_ion_sums:
                # sum over ions
                mo_same = jnp.sum(mo_same, axis=-2)
                mo_diff = jnp.sum(mo_diff, axis=-2)
            mos.append((mo_same, mo_diff))

        # mo_up and mo_dn now refer to the up/dn-ORBITALS evaluated for all electrons,
        # i.e. they currently contain two block-columns of the slater matrix
        # The subsequent determinant evaluation assumes that mo_up and mo_dn refer to up/dn-ELECTRONS,
        # i.e. two block-rows of the slater matrix
        # We therefore reshuffle the blocks accordingly
        # TODO: refactor all orbitals to output mo_up and mo_dn orbital-blocks instead of electron-blocks,
        #  and get rid of this reshuffling

        # mos is a list of 2 tuples:
        # the 1st index corresponds to the orbital block (0=up / 1=down),
        # the 2nd index corresponds to being on the diagonal or off-diagonal (0=diag=same / 1=offdiag/diff)
        if self.determinant_schema == "full_det":
            mo_up = jnp.concatenate([mos[0][0], mos[1][1]], axis=-1)
            mo_dn = jnp.concatenate([mos[0][1], mos[1][0]], axis=-1)
        elif self.determinant_schema == "block_diag":
            mo_up = mos[0][0]
            mo_dn = mos[1][0]
        return mo_up, mo_dn
