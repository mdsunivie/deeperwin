"""
File containing the wavefunction model & function to build the model
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import haiku as hk
import haiku.experimental
import jax
import jax.numpy as jnp
import numpy as np

from deeperwin.configuration import BaselineConfigType, JastrowConfig, MLPConfig, ModelConfig, PhysicalConfig
from deeperwin.model.definitions import Embeddings, InputFeatures, WavefunctionDefinition
from deeperwin.model.embeddings import AxialTransformerEmbedding, FermiNetEmbedding, GNNEmbedding, TransformerEmbedding, MoonEmbedding
from deeperwin.model.input_features import InputPreprocessor
from deeperwin.model.ml_orbitals.ml_orbitals import get_phisnet_solution
from deeperwin.model.mlp import MLP
from deeperwin.model.orbitals import OrbitalNet
from deeperwin.orbitals import get_atomic_orbital_descriptors, get_baseline_solution, get_envelope_exponents_from_atomic_orbitals, get_n_basis_per_Z
from deeperwin.utils.periodic import LatticeParams, get_kpoints_in_sphere
from deeperwin.utils.utils import get_distance_matrix, get_param_size_summary, get_periodic_distance_matrix, periodic_norm

LOGGER = logging.getLogger("dpe")


class JastrowFactor(hk.Module):
    def __init__(self, config: JastrowConfig, mlp_config: MLPConfig, name=None):
        super().__init__(name=name)
        self.config = config
        self.mlp_config = mlp_config

    def __call__(self, embeddings: Embeddings, n_up: int):
        if self.config.differentiate_spins:
            jastrow_up = MLP(self.config.n_hidden + [1], self.mlp_config, linear_out=True, output_bias=False, name="up")(
                embeddings.el[..., : n_up, :])
            jastrow_dn = MLP(self.config.n_hidden + [1], self.mlp_config, linear_out=True, output_bias=False, name="dn")(
                embeddings.el[..., n_up:, :])
            jastrow = jnp.sum(jastrow_up, axis=(-2, -1)) + jnp.sum(jastrow_dn, axis=(-2, -1))
        else:
            jastrow = MLP(self.config.n_hidden + [1], linear_out=True, output_bias=False, name="mlp")(embeddings.el)
            jastrow = jnp.sum(jastrow, axis=(-2, -1))
        return jastrow


def evaluate_sum_of_determinants(mo_matrix_up, mo_matrix_dn):
    LOG_EPSILON = 1e-8

    if (mo_matrix_up.shape[-1] != mo_matrix_up.shape[-2]) and (mo_matrix_dn.shape[-1] != mo_matrix_dn.shape[-2]):
        # determinant_schema is full_det or restricted_closed_shell
        mo_matrix = jnp.concatenate([mo_matrix_up, mo_matrix_dn], axis=-2)
        sign_total, log_total = jnp.linalg.slogdet(mo_matrix)
    else:
        # determinant schema is block diagonal
        sign_up, log_up = jnp.linalg.slogdet(mo_matrix_up)
        sign_dn, log_dn = jnp.linalg.slogdet(mo_matrix_dn)
        log_total = log_up + log_dn
        sign_total = sign_up * sign_dn

    log_shift = jnp.max(log_total, axis=-1, keepdims=True)
    psi = jnp.exp(log_total - log_shift) * sign_total
    psi = jnp.sum(psi, axis=-1)  # sum over determinants

    log_psi_sqr = 2 * (jnp.log(jnp.abs(psi) + LOG_EPSILON) + jnp.squeeze(log_shift, -1))
    phase = jnp.angle(psi)
    return phase, log_psi_sqr


class Wavefunction(hk.Module):
    def __init__(
        self, 
        config: ModelConfig, 
        wavefunction_definition: WavefunctionDefinition, 
        name: str = "wf"
    ) -> None:
        """
        Args:
            config:                   Model Configuration
            phys_config:              Physical Configuration, in the case of multiple compounds, this should be the 
                                      physical configuration with the largest amount of electrons
            max_n_particles:          NamedTuple containing the maximum amount of ions, up_el, dn_el that the wavefunction
                                      model should consider
            train_multiple_compounds: Whether to train on multiple or single compound
            Name: 
        """
        super().__init__(name=name)
        self.config = config
        self.wavefunction_definition = wavefunction_definition

        self.input_preprocessor = InputPreprocessor(
            config=config.features,
            mlp_config=config.mlp,
            wavefunction_definition=wavefunction_definition,
            name="input"
        )
        if self.config.jastrow:
            self.jastrow = JastrowFactor(self.config.jastrow, self.config.mlp)
        else:
            self.jastrow = None

        self.complex_wf = config.complex_wf
        self.orb_net = OrbitalNet(self.config.orbitals,
                                  self.config.mlp,
                                  complex_wf=self.complex_wf,
                                  name="orbitals",
                                  )

    def __call__(self, n_up: int, n_dn: int, r, R, Z, fixed_params: Optional[Dict] = None):
        fixed_params = fixed_params or {}
        diff_dist, features = self.input_preprocessor(n_up, n_dn, r, R, Z, fixed_params)

        if self.config.embedding.name == "gnn" and self.config.embedding.ion_gnn.name == "phisnet_ion_emb":
            features_ion = fixed_params['transferable_atomic_orbitals']['features_ion_phisnet']
            features = InputFeatures(features.el, features_ion, features.el_el, features.el_ion, features.ion_ion)

        embeddings = self._calculate_embedding(diff_dist, features, n_up)
        mo_up, mo_dn = self._calculate_orbitals(diff_dist, embeddings, fixed_params, Z.shape[-1], n_up, n_dn)
        phase, log_psi_sqr = evaluate_sum_of_determinants(mo_up, mo_dn)

        # Jastrow factor to the total wavefunction
        if self.jastrow:
            log_psi_sqr += self.jastrow(embeddings, n_up)

        # Electron-electron-cusps
        if self.config.use_el_el_cusp_correction:
            log_psi_sqr += self._el_el_cusp(diff_dist.dist_el_el, n_up)

        return phase, log_psi_sqr

    def get_slater_matrices(self, n_up, n_dn, r, R, Z, fixed_params: Optional[Dict] = None):
        assert n_up + n_dn == r.shape[-2] # assert down & up electrons equal total amount of electrons

        fixed_params = fixed_params or {}
        diff_dist, features = self.input_preprocessor(n_up, n_dn, r, R, Z, fixed_params)
        if self.config.embedding.name == "gnn" and self.config.embedding.ion_gnn.name == "phisnet_ion_emb":
            features_ion = fixed_params['transferable_atomic_orbitals']['features_ion_phisnet']
            features = InputFeatures(features.el, features_ion, features.el_el, features.el_ion, features.ion_ion)

        embeddings = self._calculate_embedding(diff_dist, features, n_up)
        mo_up, mo_dn = self._calculate_orbitals(diff_dist, embeddings, fixed_params, Z.shape[-1], n_up, n_dn)
        return mo_up, mo_dn

    @haiku.experimental.name_like("__call__")
    def _calculate_embedding(self, diff_dist, features, n_up):
        if self.config.embedding.name in ["ferminet", "dpe4"]:
            return FermiNetEmbedding(self.config.embedding,
                                     self.config.mlp)(features, n_up)
        elif self.config.embedding.name == "moon":
            return MoonEmbedding(self.config.embedding)(diff_dist, features, n_up)
        if self.config.embedding.name == "gnn":
            return GNNEmbedding(self.config.embedding,
                                self.config.mlp)(diff_dist, features, n_up)
        elif self.config.embedding.name == "transformer":
            return TransformerEmbedding(self.config.embedding,
                                        self.config.mlp)(features, n_up)
        elif self.config.embedding.name == "axial_transformer":
            return AxialTransformerEmbedding(self.config.embedding, self.config.mlp)(features, n_up)
        else:
            raise ValueError(f"Unknown embedding: {self.config.embedding.name}")

    @haiku.experimental.name_like("__call__")
    def _calculate_orbitals(self, diff_dist, embeddings, fixed_params, n_ions, n_up, n_dn):
        return self.orb_net(diff_dist,
                            embeddings,
                            fixed_params,
                            n_ions,
                            n_up,
                            n_dn)

    def _calculate_cache(self, n_up: int, n_dn: int, r, R, Z, fixed_params: Optional[Dict] = None):
        cache = dict()
        if not self.config.use_cache:
            return None
        
        # TODO: Refactor this somehow so we can call something like self.input_preprocessor.get_distance(R)
        if (fixed_params is not None) and fixed_params.get("periodic") is not None:
            diff_ion_ion, dist_ion_ion = get_periodic_distance_matrix(R, 
                                                                      fixed_params["periodic"].lattice, 
                                                                      fixed_params['periodic'].rec / (2*np.pi))
        else:
            diff_ion_ion, dist_ion_ion = get_distance_matrix(R)

        if self.config.orbitals.transferable_atomic_orbitals and self.config.orbitals.transferable_atomic_orbitals.name == "taos":
            cache['taos'] = dict()
            if self.config.embedding.name == "axial_transformer":
                emb_dim = self.config.embedding.output_dim
            elif self.config.embedding.name == "transformer":
                attention_value_dim = self.config.embedding.el_transformer.attention_value_dim or self.config.embedding.el_transformer.attention_dim
                emb_dim = self.config.embedding.el_transformer.attention_output_dim or attention_value_dim * self.config.embedding.el_transformer.n_heads
            elif self.config.embedding.name == "gnn":
                emb_dim = self.config.embedding.gnn.message_passing.node_dim
            elif self.config.embedding.name == "moon":
                emb_dim = self.config.embedding.output_dim
            else:
                emb_dim = self.config.embedding.n_hidden_one_el[-1]
            exp, bf = self.orb_net.taos.get_exponents_and_backflows(
                fixed_params['transferable_atomic_orbitals']['features'],
                n_up,
                n_dn,
                emb_dim,
                (),
                diff_ion_ion,
                dist_ion_ion
                )
            cache['taos']['exponents'] = exp
            cache['taos']['backflows'] = bf
        return cache


    def _el_el_cusp(self, el_el_dist, n_up):
        # # No factor 0.5 here, e.g. when comparing to NatChem 2020, [doi.org/10.1038/s41557-020-0544-y], because:
        # # A) We double-count electron-pairs because we take the full distance matrix (and not only the upper triangle)
        # # B) We model log(psi^2)=2*log(|psi|) vs log(|psi|) int NatChem 2020, i.e. the cusp correction needs a factor 2
        alpha_same = hk.get_parameter("el_el_cusp_same", [], init=jnp.ones)
        alpha_diff = hk.get_parameter("el_el_cusp_diff", [], init=jnp.ones)
        factor_same = -0.25
        factor_diff = -0.5

        flat_shape = el_el_dist.shape[:-2] + (-1,)
        if el_el_dist.shape[-1] == el_el_dist.shape[-2]:
            # Full el-el-distance matrix (including distance to itself, which is 0)
            dist_same = jnp.concatenate([el_el_dist[..., :n_up, :n_up].reshape(flat_shape),
                                         el_el_dist[..., n_up:, n_up:].reshape(flat_shape)], axis=-1)
            dist_diff = jnp.concatenate([el_el_dist[..., :n_up, n_up:].reshape(flat_shape),
                                         el_el_dist[..., n_up:, :n_up].reshape(flat_shape)], axis=-1)
        else:
            dist_same = jnp.concatenate([el_el_dist[..., :n_up, :n_up-1].reshape(flat_shape),
                                         el_el_dist[..., n_up:, n_up:].reshape(flat_shape)], axis=-1)
            dist_diff = jnp.concatenate([el_el_dist[..., :n_up, n_up-1:].reshape(flat_shape),
                                         el_el_dist[..., n_up:, :n_up].reshape(flat_shape)], axis=-1)
        cusp_same = jnp.sum(alpha_same ** 2 / (alpha_same + dist_same), axis=-1)
        cusp_diff = jnp.sum(alpha_diff ** 2 / (alpha_diff + dist_diff), axis=-1)
        return factor_same * cusp_same + factor_diff * cusp_diff


    def init_for_multitransform(self):
        return self.__call__, (self.__call__, self.get_slater_matrices, self._calculate_embedding, self._calculate_orbitals, self._calculate_cache)



def build_log_psi_squared(
    config: ModelConfig, 
    physical_config: Union[PhysicalConfig, List[PhysicalConfig]],  
    baseline_config: BaselineConfigType,
    fixed_params: Dict, 
    rng_seed: Any,
    phisnet_model=None,
    N_ions_max=None,
) -> Tuple[Callable, Callable, Callable, Dict, Dict]:
    if isinstance(physical_config, PhysicalConfig):
        _phys_config = physical_config
    else:
        _phys_config = physical_config[0]

    # Initialize fixed model parameters
    fixed_params = fixed_params or init_model_fixed_params(config, _phys_config, baseline_config, phisnet_model, N_ions_max)

    # construct definition of the wavefunction model
    wavefunction_definition = construct_wavefunction_definition(config, physical_config)

    # Build model
    model = hk.multi_transform(lambda: Wavefunction(config, wavefunction_definition).init_for_multitransform())

    # Initialized trainable parameters using a dummy batch
    n_el, _, R, Z = _phys_config.get_basic_params()
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(rng_seed), 2)
    r = jax.random.normal(rng1, [1, n_el, 3])
    params = model.init(rng2, _phys_config.n_up, _phys_config.n_dn, r, R, Z, fixed_params)

    # if single geometry check for potential analytical initialization of orbitals
    params = check_orbital_intialization(config, physical_config, params)

    # Remove rng-argument (replace by None) and move parameters to back of function
    log_psi_sqr = lambda params, n_up, n_dn, *batch: model.apply[0](params, None, n_up, n_dn, *batch)
    get_slater_mat = lambda params, n_up, n_dn, *batch: model.apply[1](params, None, n_up, n_dn, *batch)
    get_cache = lambda params, n_up, n_dn, *batch: model.apply[4](params, None, n_up, n_dn, *batch)

    LOGGER.debug(get_param_size_summary(params))
 
    return log_psi_sqr, get_slater_mat, get_cache, params, fixed_params

def construct_wavefunction_definition(
    config: ModelConfig,
    physical_config: Union[PhysicalConfig, List[PhysicalConfig]],
) -> WavefunctionDefinition:
    """
    Construct definition for the wavefunction model
    """
    if config.Z_max is None:
        if isinstance(physical_config, PhysicalConfig):
            Z_max = max(physical_config.Z)
        else:
            Z_max = int(max([int(max(phys_config.Z)) for phys_config in physical_config]))
    else:
        Z_max = config.Z_max

    if config.Z_min is None:
        if isinstance(physical_config, PhysicalConfig):
            Z_min = min(physical_config.Z)
        else:
            Z_min = int(min([int(min(phys_config.Z)) for phys_config in physical_config]))
    else:
        Z_min = config.Z_min

    return WavefunctionDefinition(
        Z_max=Z_max,
        Z_min=Z_min
    )

def check_orbital_intialization(
    config: ModelConfig,
    physical_config: Union[PhysicalConfig, List[PhysicalConfig]],
    params: Dict[str, Dict[str, jnp.ndarray]]
) -> Dict[str, Dict[str, jnp.ndarray]]:
    if isinstance(physical_config, PhysicalConfig) and config.orbitals.envelope_orbitals and config.orbitals.envelope_orbitals.initialization == "analytical":
        weights, alphas = get_envelope_exponents_from_atomic_orbitals(
            physical_config, pad_full_det=((config.orbitals.determinant_schema == "full_det") or (config.orbitals.determinant_schema == "restricted_closed_shell"))
        )
        params['wf/orbitals/envelope_orbitals']['alpha_dn'] = jnp.tile(alphas[0], [1, config.orbitals.n_determinants])
        params['wf/orbitals/envelope_orbitals']['alpha_up'] = jnp.tile(alphas[1], [1, config.orbitals.n_determinants])
        params['wf/orbitals/envelope_orbitals']['weights_dn'] = jnp.tile(weights[0], [1, config.orbitals.n_determinants])
        params['wf/orbitals/envelope_orbitals']['weights_up'] = jnp.tile(weights[1], [1, config.orbitals.n_determinants])
    return params


def init_model_fixed_params(model_config: ModelConfig, 
                            physical_config: PhysicalConfig,
                            baseline_config: BaselineConfigType,
                            phisnet_model=None, 
                            N_ions_max=None, 
                            atomic_orbitals=None):
    """
    Computes CASSCF baseline solution for DeepErwin model and initializes fixed parameters.

    Args:
        casscf_config (CASSCFConfig): CASSCF hyperparmeters
        physical_config (PhysicalConfig): Description of the molecule

    Returns:
        dict: Initial fixed parameters
    """
    fixed_params = dict(input={}, baseline_energies=dict(E_ref=physical_config.E_ref))
    is_periodic = physical_config.periodic is not None
    _, n_up, R, Z = physical_config.get_basic_params()

    if is_periodic:
        lattice_params = LatticeParams.from_periodic_config(physical_config.periodic)
        fixed_params['periodic'] = lattice_params

    if baseline_config:
        nuc_features = None
        if baseline_config.name == "phisnet":
            atom_types = baseline_config.atom_types or set(Z)
            nb_orbitals_per_Z = get_n_basis_per_Z(baseline_config.basis_set, tuple(atom_types), is_periodic)
            orbital_params, nuc_features, hessian, energies = get_phisnet_solution(physical_config,
                                                                                phisnet_model,
                                                                                baseline_config.basis_set,
                                                                                baseline_config.localization,
                                                                                N_ions_max,
                                                                                nb_orbitals_per_Z,
                                                                                atomic_orbitals)
        else:
            orbital_params, energies = get_baseline_solution(physical_config, baseline_config)
        fixed_params["baseline_orbitals"] = orbital_params
        fixed_params["baseline_energies"].update(energies)
        LOGGER.debug("Finished baseline calculation: " + ", ".join([f"{k}={v:.6f}" for k,v in energies.items()]))
        if is_periodic:
            LOGGER.debug(f"Occupied k-points (baseline, spin up): {orbital_params.k_points[0].T[:n_up]}")


    # Compute orbital coefficients as descriptors for Transferable Atomic Orbitals
    tao_config = model_config.orbitals.transferable_atomic_orbitals
    if tao_config:
        atom_types = tao_config.atom_types or getattr(baseline_config, "atom_types", None) or set(Z)
        nb_orbitals_per_Z = get_n_basis_per_Z(baseline_config.basis_set, 
                                              tuple(atom_types), 
                                              is_periodic, 
                                              baseline_config.pyscf_options.exp_to_discard)
        orbital_features = get_atomic_orbital_descriptors(orbital_params,
                                                          R,
                                                          Z,
                                                          lattice_params.lattice if is_periodic else None,
                                                          atom_types,
                                                          nb_orbitals_per_Z,
                                                          model_config.orbitals.transferable_atomic_orbitals
                                                 )
        if is_periodic:
            k_twist_frac = np.array(physical_config.periodic.k_twist) if physical_config.periodic.k_twist is not None else np.zeros(3)
            twist_features = []
            if tao_config.twist_encoding is not None and "concat" in tao_config.twist_encoding:
                twist_features.append(np.tile(k_twist_frac, [*orbital_features[0].shape[:-1], 1]))
            if tao_config.twist_encoding is not None and "periodic" in tao_config.twist_encoding:
                R = np.array(R)
                periodic_twist = np.concatenate([jnp.sin(R @ k_twist_frac)[..., None],
                                                 jnp.cos(R @ k_twist_frac)[..., None]], axis=-1)[..., None, :]
                twist_features.append(np.tile(periodic_twist, [1, orbital_features[0].shape[-2], 1]))
            # Append the twist to each orbital feature (both spin up and down)
            orbital_features = tuple(np.concatenate([o, *twist_features], axis=-1) for o in orbital_features)

        fixed_params['transferable_atomic_orbitals'] = dict(orbitals=orbital_params, 
                                                            features=orbital_features, 
                                                            features_ion_phisnet=nuc_features)

    if model_config.orbitals.periodic_orbitals is not None:
        k_points = get_kpoints_in_sphere(lattice_params.rec, model_config.orbitals.periodic_orbitals.n_k_points_min)
        fixed_params['periodic_orbitals'] = dict(k_point_grid=k_points)

    return jax.tree_util.tree_map(jnp.array, fixed_params)


