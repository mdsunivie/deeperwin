import haiku as hk
import jax
from jax import numpy as jnp
from deeperwin.configuration import MLPConfig, TransferableAtomicOrbitalsConfig, OrbFeatureDenseGNNConfig
from deeperwin.model import MLP, antisymmetrize, symmetrize, DiffAndDistances, Embeddings
from deeperwin.model.mlp import get_rbf_features
from deeperwin.model.gnn import DenseGNN
from typing import Optional, Dict

class TAOBackflow(hk.Module):
    def __init__(self, width, depth, emb_dim, determinant_schema, n_dets, antisymmetrize, mlp_config: MLPConfig, output_width=None, name=None):
        super().__init__(name=name)
        self.n_hidden = [width] * depth
        if output_width and self.n_hidden:
            self.n_hidden[-1] = output_width
        self._n_dets = n_dets
        self._output_dim = n_dets * emb_dim
        self._determinant_schema = determinant_schema
        self._mlp_config = mlp_config
        self._antisymmetrize = antisymmetrize
        if self._determinant_schema == "full_det":
            self._output_dim *= 2

    def __call__(self, x):
        bf_func = MLP(self.n_hidden,
                      self._mlp_config,
                      linear_out=False,
                      residual=False)
        if self._antisymmetrize:
            bf_func = antisymmetrize(bf_func)
        bf = bf_func(x)
        bf = hk.Linear(self._output_dim, with_bias=False, name="lin_out")(bf)
        if self._determinant_schema == "full_det":
            bf = bf.reshape(bf.shape[:-1] + (2, self._n_dets, -1))
        else:
            bf = bf.reshape(bf.shape[:-1] + (self._n_dets, -1))
        return bf


class TAOExponents(hk.Module):
    def __init__(self, width, depth, determinant_schema, n_dets, symmetrize, use_prefactors, use_squared_input, mlp_config: MLPConfig, name=None):
        super().__init__(name=name)
        self._width = width
        self._depth = depth
        self._n_dets = n_dets
        self._determinant_schema = determinant_schema
        self._use_prefactors = use_prefactors
        self._mlp_config = mlp_config
        self._symmetrize = symmetrize
        self._output_dim = n_dets
        self._use_squared_input = use_squared_input
        if determinant_schema == "full_det":
            self._output_dim *= 2
        if use_prefactors:
            self._output_dim *= 2

    def __call__(self, x):
        if self._use_squared_input:
            # z = hk.Linear(self._width)(x)**2
            z = x**2
            x = jnp.concatenate([x, z], axis=-1)
        exp_func = MLP([self._width] * self._depth + [self._output_dim],
                       self._mlp_config,
                       linear_out=True,
                       residual=False)
        if self._symmetrize:
            exp_func = symmetrize(exp_func)
        output = exp_func(x)
        if self._determinant_schema == "full_det":
            output = output.reshape(output.shape[:-1] + (2, -1)) # [spin x (dets * exp/prefac)]
        if self._use_prefactors:
            output = output.reshape(output.shape[:-1] + (self._n_dets, 2))
            exponents = output[..., 0]
            prefacs = output[..., 1]
        else:
            exponents = output
            prefacs = None

        exponents = jax.nn.softplus(exponents)
        return exponents, prefacs


class OrbitalFeatureGNN(hk.Module):
    def __init__(self, config: OrbFeatureDenseGNNConfig, mlp_config: MLPConfig, n_edge_features, name=None):
        super().__init__(name=name)
        self.gnn = DenseGNN(config, mlp_config)
        self.n_edge_features = n_edge_features

    def __call__(self, orb_features: jax.Array, ion_ion_diff: jax.Array, ion_ion_dist: jax.Array, ):
        orb_features = jnp.moveaxis(orb_features, -2, -3) # [... x ion x orb x features] -> [... orb x ion x features]
        # Calculate edge features as (rbf(|r|), rbf(|r|) * x, rbf(|r|) * y, rbf(|r|) * z)
        rbfs = get_rbf_features(ion_ion_dist, n_features=self.n_edge_features, r_max=5.0)
        ion_ion_edges = jnp.concatenate([rbfs,
                                         rbfs * ion_ion_diff[..., 0:1],
                                         rbfs * ion_ion_diff[..., 1:2],
                                         rbfs * ion_ion_diff[..., 2:3],
                                         ], axis=-1)
        ion_ion_edges = ion_ion_edges[..., None, None, :, :, :] # add dummy-dim for orbitals and antisymmetrization
        orb_features = antisymmetrize(lambda nodes: self.gnn(nodes, edges=ion_ion_edges)[0],
                                      tmp_axis=-4)(orb_features)
        orb_features = jnp.moveaxis(orb_features, -3, -2)
        return orb_features


class TransferableAtomicOrbitals(hk.Module):
    def __init__(self,
                 config: TransferableAtomicOrbitalsConfig,
                 mlp_config: MLPConfig,
                 n_dets: int,
                 determinant_schema: str,
                 name="generalized_atomic_orbitals"):
        super().__init__(name=name)
        self.n_dets = n_dets
        self.determinant_schema = determinant_schema
        self.config = config
        self.mlp_config_backflow = mlp_config.copy()
        self.mlp_config_envelope = mlp_config.copy()
        if self.config.backflow_activation:
            self.mlp_config_backflow.activation = self.config.backflow_activation
        if self.config.envelope_activation:
            self.mlp_config_envelope.activation = self.config.envelope_activation
        # TODO: implement or assert in config
        assert determinant_schema in ["full_det"], f"TAO currently not implemented for determinant_schema {determinant_schema}"

    @hk.experimental.name_like("__call__")
    def get_exponents_and_backflows(self, orbital_features: jax.Array, n_up, n_dn, el_emb_dim, tile_dims, ion_ion_diff, ion_ion_dist, get_envelopes=True, get_backflows=True):
        exponents = [None, None]
        prefacs = [None, None]
        backflows = [None, None]

        backflow_final_hidden = None
        if self.config.product_mode == "full":
            backflow_dim = el_emb_dim
            backflow_n_dets = self.n_dets
            envelope_n_dets = self.n_dets
        elif self.config.product_mode == "tucker":
            backflow_dim = self.config.tucker_rank_orb
            backflow_n_dets = 1
            envelope_n_dets = self.n_dets
        elif self.config.product_mode == "cpd":
            backflow_dim = self.config.product_rank
            envelope_n_dets = self.config.product_rank
            backflow_n_dets = 1
        elif self.config.product_mode == "el_downmap":
            backflow_dim = self.config.product_rank
            envelope_n_dets = self.n_dets
            backflow_n_dets = 1
        elif self.config.product_mode == "orb_downmap":
            backflow_dim = el_emb_dim
            backflow_n_dets = self.n_dets
            envelope_n_dets = self.n_dets
            backflow_final_hidden = self.config.product_rank


        for spin, (orb_features, n_orbitals) in enumerate(zip(orbital_features, [n_up, n_dn])):
            orb_features = orb_features[..., :n_orbitals, :]

            # KFAC requires all operations to have the same leading (batch)-dimension.
            # Therefore If the data has a batch-dim, we tile the orbital features across samples
            # TODO: remove tiling, as soon as kfac supports non-batched operations
            orb_features = jnp.tile(orb_features, (*tile_dims, 1, 1, 1))   # [batch x ions x orbitals x features]

            if self.config.orb_feature_gnn and self.config.orb_feature_gnn.n_iterations > 0:
                orb_features = OrbitalFeatureGNN(self.config.orb_feature_gnn,
                                                 self.mlp_config_backflow,
                                                 self.config.n_rbf_features_orb_gnn,
                                                 name=f"orbital_gnn_{spin}")(orb_features,
                                                                             ion_ion_diff,
                                                                             ion_ion_dist,
                                                                             )

            if get_envelopes:
                envelope_func = TAOExponents(self.config.envelope_width,
                                             self.config.envelope_depth,
                                             self.determinant_schema,
                                             envelope_n_dets,
                                             self.config.symmetrize_exponent_mlp,
                                             self.config.use_prefactors,
                                             self.config.use_squared_envelope_input,
                                             self.mlp_config_envelope,
                                             name=f"exponent_{spin}")
                exponents[spin], prefacs[spin] = envelope_func(orb_features) # [batch x ions x orb x spin-type x dets]
            if get_backflows:
                backflow_func = TAOBackflow(self.config.backflow_width,
                                            self.config.backflow_depth,
                                            backflow_dim,
                                            self.determinant_schema,
                                            backflow_n_dets,
                                            self.config.antisymmetrize_backflow_mlp,
                                            self.mlp_config_backflow,
                                            backflow_final_hidden,
                                            name=f"backflow_{spin}")
                backflows[spin] = backflow_func(orb_features)       # [batch x ions x orb x spin-type x dets x emb_dm]
        return exponents, backflows, prefacs


    def __call__(self,
                 diff_dist: DiffAndDistances,
                 embeddings: Embeddings,
                 n_ions: int,
                 n_up: int,
                 n_dn: int,
                 orbital_features: jnp.array,  # [n_ions, n_orbitals, feature_dim]
                 cache: Optional[Dict] = None,
                 ):
        emb_el = embeddings.el
        if self.config.el_feature_dim is not None:
            emb_el = hk.Linear(self.config.el_feature_dim, name="lin_el_embedding")(emb_el)

        batch_dims = emb_el.shape[:-2]
        if cache is None:
            exponents, backflows, prefacs = self.get_exponents_and_backflows(orbital_features,
                                                                              n_up,
                                                                              n_dn,
                                                                              emb_el.shape[-1],
                                                                              batch_dims,
                                                                              diff_dist.diff_ion_ion,
                                                                              diff_dist.dist_ion_ion,
                                                                              )
        else:
            exponents, backflows, prefacs = cache['exponents'], cache['backflows'], cache['prefacs']

        # orb_features are of shape [n_ions x n_orbitals (n_up, n_dn) x feature_dim]
        mos = []
        for spin, (exponent, backflow, prefac) in enumerate(zip(exponents, backflows, prefacs)):
            if spin == 0:
                slice_same = slice(None, n_up)
                slice_diff = slice(n_up, None)
            else:
                slice_same = slice(n_up, None)
                slice_diff = slice(None, n_up)

            emb_same = emb_el[..., slice_same, :]
            emb_diff = emb_el[..., slice_diff, :]

            dist_el_ion_same = diff_dist.dist_el_ion[..., slice_same, :]
            dist_el_ion_diff = diff_dist.dist_el_ion[..., slice_diff, :]

            # envelope shapes: [batch x el x ion x orbital x same/diff x dets]
            envelope_same = jnp.exp(-exponent[..., None, :, :, 0, :] * dist_el_ion_same[..., :, :, None, None])
            envelope_diff = jnp.exp(-exponent[..., None, :, :, 1, :] * dist_el_ion_diff[..., :, :, None, None])
            if prefac is not None:
                envelope_same *= prefac[..., None, :, :, 0, :]
                envelope_diff *= prefac[..., None, :, :, 1, :]

            bf_same = backflow[..., :, :, 0, :, :]
            bf_diff = backflow[..., :, :, 1, :, :]


            if self.config.product_mode == "cpd":
                # In principle there is one big tensor which has input dimensions [el_embedding_dim, orbital_embedding_dim] and maps it to [n_determinants]
                # This 3D tensor is decomposed using a canonic polyadic decompositon (CPD) into a sum of products of 2D matrices
                # Each component (electron, orbital, determinant) is mapped onto a rank index (denoted a in the einsum) and then summed over
                mos_same_diff = []
                for same_diff, bf, emb, envelope in zip(
                    ["same", "diff"],
                    [bf_same, bf_diff],
                    [emb_same, emb_diff],
                    [envelope_same, envelope_diff]
                    ):
                    bf = bf.squeeze(axis=-2) # squeeze out the determinant dimension => [batch x ion x orbital x orb_feature]
                    mo = jnp.sum(envelope * bf[..., None, :, :, :], axis= -3) # sum over ions
                    emb_mapped = hk.Linear(self.config.product_rank, name=f"cpd_spin{spin}_{same_diff}_embedding")(emb)
                    mo = mo * emb_mapped[..., None, :] # add dummy orbital dimension to electron embedding
                    mo = hk.Linear(self.n_dets, with_bias=False, name=f"cpd_spin{spin}_{same_diff}_determinants")(mo)    # map from rank to to determinant dimension
                    mo = jnp.moveaxis(mo, -1, -3) # move determinant dimension to the front
                    mos_same_diff.append(mo)
                mos.append(mos_same_diff)
            elif self.config.product_mode == "tucker":
                emb_same_mapped = hk.Linear(self.config.tucker_rank_el, name=f"tucker_el_{spin}_same")(emb_same)
                emb_diff_mapped = hk.Linear(self.config.tucker_rank_el, name=f"tucker_el_{spin}_diff")(emb_diff)
                tucker_same = hk.get_parameter(f"tucker_same_s{spin}",
                                               [self.config.tucker_rank_el,
                                                self.config.tucker_rank_orb,
                                                self.n_dets],
                                                init=hk.initializers.VarianceScaling(1.0, "fan_avg"))
                tucker_diff = hk.get_parameter(f"tucker_diff_s{spin}",
                                [self.config.tucker_rank_el,
                                self.config.tucker_rank_orb,
                                self.n_dets],
                                init=hk.initializers.VarianceScaling(1.0, "fan_avg"))
                mo_same = jnp.einsum("...ia,...Ikb,abd,...iIkd->...dik", emb_same_mapped, bf_same.squeeze(axis=-2), tucker_same, envelope_same)
                mo_diff = jnp.einsum("...ia,...Ikb,abd,...iIkd->...dik", emb_diff_mapped, bf_diff.squeeze(axis=-2), tucker_diff, envelope_diff)
                mos.append((mo_same, mo_diff))
            elif self.config.product_mode in ["full", "orb_downmap"]:
                # sum over embedding dimension and ions:
                # I = ions, k = orbitals, i = electrons, d = determinants, a = embedding dimension
                mo_same = jnp.einsum("...ia,...Ikda,...iIkd->...dik", emb_same, bf_same, envelope_same)
                mo_diff = jnp.einsum("...ia,...Ikda,...iIkd->...dik", emb_diff, bf_diff, envelope_diff)
                mos.append((mo_same, mo_diff))
            elif self.config.product_mode == "el_downmap":
                emb_same_mapped = hk.Linear(self.n_dets * self.config.product_rank, with_bias=False, name=f"el_downmap_{spin}_same")(emb_same)
                emb_diff_mapped = hk.Linear(self.n_dets * self.config.product_rank, with_bias=False, name=f"el_downmap_{spin}_diff")(emb_diff)
                emb_same_mapped = emb_same_mapped.reshape(emb_same_mapped.shape[:-1] + (self.n_dets, self.config.product_rank))
                emb_diff_mapped = emb_diff_mapped.reshape(emb_diff_mapped.shape[:-1] + (self.n_dets, self.config.product_rank))
                mo_same = jnp.einsum("...ida,...Ika,...iIkd->...dik", emb_same_mapped, bf_same.squeeze(axis=-2), envelope_same)
                mo_diff = jnp.einsum("...ida,...Ika,...iIkd->...dik", emb_diff_mapped, bf_diff.squeeze(axis=-2), envelope_diff)
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
        if self.determinant_schema == 'full_det':
            mo_up = jnp.concatenate([mos[0][0], mos[1][1]], axis=-1)
            mo_dn = jnp.concatenate([mos[0][1], mos[1][0]], axis=-1)
        elif self.determinant_schema == 'block_diag':
            mo_up = mos[0][0]
            mo_dn = mos[1][0]
        return mo_up, mo_dn
