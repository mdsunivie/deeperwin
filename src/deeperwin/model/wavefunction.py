"""
File containing the wavefunction model & function to build the model
"""

import haiku.experimental
import jax
import jax.numpy as jnp
from typing import Optional, Dict, Callable, Any, Tuple, Union, List
import haiku as hk
import logging
from deeperwin.configuration import (
    ModelConfig,
    JastrowConfig,
    PhysicalConfig,
    MLPConfig,
    CASSCFConfig
)
from deeperwin.model.input_features import InputPreprocessor
from deeperwin.model.mlp import MLP
from deeperwin.model.embeddings import *
from deeperwin.model.orbitals import OrbitalNet
from deeperwin.model.orbitals.transferable_atomic_orbitals import TransferableAtomicOrbitals
from deeperwin.model.definitions import *
from deeperwin.utils.utils import get_distance_matrix, get_param_size_summary
from deeperwin.orbitals import get_baseline_solution, get_atomic_orbital_descriptors, get_envelope_exponents_from_atomic_orbitals, get_n_basis_per_Z
from deeperwin.model.ml_orbitals.ml_orbitals import get_phisnet_solution
from deeperwin.local_features import build_local_rotation_matrices, build_global_rotation_matrix
from deeperwin.model.e3nn_utils import get_ao_to_irreps_matrix
import e3nn_jax as e3nn


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

    # determinant_schema is full_det or restricted_closed_shell
    if mo_matrix_up.shape[-1] != mo_matrix_up.shape[-2] and mo_matrix_dn.shape[-1] != mo_matrix_dn.shape[-2]:
        mo_matrix = jnp.concatenate([mo_matrix_up, mo_matrix_dn], axis=-2)
        sign_total, log_total = jnp.linalg.slogdet(mo_matrix)
    # determinant schema is block diagonal
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

        self.orb_net = OrbitalNet(self.config.orbitals,
                                  self.config.mlp,
                                  self.wavefunction_definition,
                                  name="orbitals")

    def __call__(self, n_up: int, n_dn: int, r, R, Z, fixed_params: Optional[Dict] = None):
        fixed_params = fixed_params or {}
        diff_dist, features = self.input_preprocessor(n_up, n_dn, r, R, Z, fixed_params.get('input'))

        if self.config.embedding.name == "gnn" and self.config.embedding.ion_gnn.name == "phisnet_ion_emb":
            # TODO switch features.ion from constance to features generated by phisnet in fixed_params
            # TODO make this flag optional to debug
            features_ion = fixed_params['transferable_atomic_orbitals']['features_ion_phisnet']
            features = InputFeatures(features.el, features_ion, features.el_el, features.el_ion, features.ion_ion)

        embeddings = self._calculate_embedding(diff_dist, features, n_up)
        mo_up, mo_dn = self._calculate_orbitals(diff_dist, embeddings, fixed_params, Z.shape[-1], n_up, n_dn)
        log_psi_sqr = evaluate_sum_of_determinants(mo_up, mo_dn)

        # Jastrow factor to the total wavefunction
        if self.jastrow:
            log_psi_sqr += self.jastrow(embeddings, n_up)

        # Electron-electron-cusps
        if self.config.use_el_el_cusp_correction:
            log_psi_sqr += self._el_el_cusp(diff_dist.dist_el_el, n_up)
        return log_psi_sqr

    def get_slater_matrices(self, n_up, n_dn, r, R, Z, fixed_params: Optional[Dict] = None):
        assert n_up + n_dn == r.shape[-2] # assert down & up electrons equal total amount of electrons

        fixed_params = fixed_params or {}
        diff_dist, features = self.input_preprocessor(n_up, n_dn, r, R, Z, fixed_params.get('input'))
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
        if self.config.embedding.name == "gnn":
            return GNNEmbedding(self.config.embedding,
                                self.config.mlp)(diff_dist, features, n_up)
        elif self.config.embedding.name == "dpe1":
            return PauliNetEmbedding(self.config.embedding,
                                     self.config.mlp)(features, n_up)
        elif self.config.embedding.name == "transformer":
            return TransformerEmbedding(self.config.embedding,
                                        self.config.mlp)(features, n_up)
        elif self.config.embedding.name == "e3mpnn":
            return EquivariantMPNNEmbedding(self.config.embedding,
                                            self.config.mlp)(diff_dist, features, n_up)
        elif self.config.embedding.name == "axial_transformer":
            return AxialTransformerEmbedding(self.config.embedding, self.config.mlp)(features, n_up)
        else:
            raise ValueError(f"Unknown embedding: {self.config.embedding.name}")

    @haiku.experimental.name_like("__call__")
    def _calculate_orbitals(self, diff_dist, embeddings, fixed_params, n_ions, n_up, n_dn):
        return self.orb_net(diff_dist,
                            embeddings,
                            fixed_params.get("orbitals"),
                            fixed_params.get("transferable_atomic_orbitals"),
                            fixed_params.get("cache"),
                            n_ions,
                            n_up,
                            n_dn)

    def _calculate_cache(self, n_up: int, n_dn: int, r, R, Z, fixed_params: Optional[Dict] = None):
        cache = dict()
        if not self.config.use_cache:
            return None
        if self.config.orbitals.transferable_atomic_orbitals and self.config.orbitals.transferable_atomic_orbitals.name == "taos":
            cache['taos'] = dict()
            diff, dist = get_distance_matrix(R)
            if self.config.embedding.name == "axial_transformer":
                emb_dim = self.config.embedding.output_dim
            elif self.config.embedding.name == "transformer":
                attention_value_dim = self.config.embedding.el_transformer.attention_value_dim or self.config.embedding.el_transformer.attention_dim
                emb_dim = self.config.embedding.el_transformer.attention_output_dim or attention_value_dim * self.config.embedding.el_transformer.n_heads
            elif self.config.embedding.name == "gnn":
                emb_dim = self.config.embedding.gnn.message_passing.node_dim
            else:
                emb_dim = self.config.orbitals.transferable_atomic_orbitals.el_feature_dim or self.config.embedding.n_hidden_one_el[-1]
            exp, bf, pf = self.orb_net.taos.get_exponents_and_backflows(
                fixed_params['transferable_atomic_orbitals']['features'],
                n_up,
                n_dn,
                emb_dim,
                (),
                diff,
                dist,
                True,
                True)
            cache['taos']['exponents'] = exp
            cache['taos']['backflows'] = bf
            cache['taos']['prefacs'] = pf
        if self.config.orbitals.transferable_atomic_orbitals and self.config.orbitals.transferable_atomic_orbitals.name == "e3_taos":
            cache['e3_taos'] = dict()
            diff, dist = get_distance_matrix(R)
            emb_dim = self.config.embedding.n_hidden_one_el[-1]
            exp, bf = self.orb_net.e3_taos.get_exponents_and_backflows(fixed_params['transferable_atomic_orbitals']['features_e3'],
                                                                       n_up,
                                                                       n_dn,
                                                                       emb_dim,
                                                                       (),
                                                                       diff,
                                                                       dist,
                                                                       )
            cache['e3_taos']['exponents'] = exp
            cache['e3_taos']['backflows'] = bf
        return cache


    def _el_el_cusp(self, el_el_dist, n_up):
        # # No factor 0.5 here, e.g. when comparing to NatChem 2020, [doi.org/10.1038/s41557-020-0544-y], because:
        # # A) We double-count electron-pairs because we take the full distance matrix (and not only the upper triangle)
        # # B) We model log(psi^2)=2*log(|psi|) vs log(|psi|) int NatChem 2020, i.e. the cusp correction needs a factor 2
        alpha_same = hk.get_parameter("el_el_cusp_same", [], init=lambda s,d: jnp.ones(s,d))
        alpha_diff = hk.get_parameter("el_el_cusp_diff", [], init=lambda s,d: jnp.ones(s,d))
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
    fixed_params: Dict, 
    rng_seed: Any,
    phisnet_model,
    N_ions_max,
    nb_orbitals_per_Z
) -> Tuple[Callable, Callable, Callable, Dict, Dict]:
    if isinstance(physical_config, PhysicalConfig):
        _phys_config = physical_config
    else:
        _phys_config = physical_config[0]

    # Initialize fixed model parameters
    fixed_params = fixed_params or init_model_fixed_params(config, _phys_config, phisnet_model, N_ions_max, nb_orbitals_per_Z)

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
    if config.max_n_ions is None:
        if isinstance(physical_config, PhysicalConfig):
            max_n_ions = len(physical_config.Z)
        else:
            max_n_ions = max([len(phys_config.Z) for phys_config in physical_config])
    else:
        max_n_ions = config.max_n_ions

    if config.max_n_up_orbitals is None:
        if isinstance(physical_config, PhysicalConfig):
            max_n_up_orbitals = int(physical_config.n_up)
        else:
            max_n_up_orbitals = max([phys_config.n_up for phys_config in physical_config])
    else:
        max_n_up_orbitals = config.max_n_up_orbitals

    if config.max_n_dn_orbitals is None:
        if isinstance(physical_config, PhysicalConfig):
            max_n_dn_orbitals = int(physical_config.n_dn)
        else:
            max_n_dn_orbitals = max([phys_config.n_dn for phys_config in physical_config])
    else:
        max_n_dn_orbitals = config.max_n_dn_orbitals

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
        max_n_ions=max_n_ions,
        max_n_up_orbitals=max_n_up_orbitals,
        max_n_dn_orbitals=max_n_dn_orbitals,
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

def init_model_fixed_params(config: ModelConfig, physical_config: PhysicalConfig, phisnet_model, N_ions_max, atomic_orbitals=None):
    """
    Computes CASSCF baseline solution for DeepErwin model and initializes fixed parameters.

    Args:
        casscf_config (CASSCFConfig): CASSCF hyperparmeters
        physical_config (PhysicalConfig): Description of the molecule

    Returns:
        dict: Initial fixed parameters
    """
    fixed_params = dict(input={}, baseline_energies=dict(E_ref=physical_config.E_ref))

    # Compute canonical input rotation
    if config.features.coordinates == "local_rot":
        fixed_params["input"]["local_rotations"] = build_local_rotation_matrices(physical_config.R, physical_config.Z)
    elif config.features.coordinates == "global_rot":
        fixed_params["input"]["global_rotation"] = build_global_rotation_matrix(physical_config.R, physical_config.Z)
    elif config.features.coordinates == "cartesian":
        pass
    else:
        raise ValueError(f"Unkonwn coordinate preprocessing: {config.features.coordinates}")

    # Compute baseline solution for PauliNet-style orbtitals (i.e. baseline * backflow)
    if config.orbitals.baseline_orbitals:
        cas_config = config.orbitals.baseline_orbitals.baseline
        LOGGER.debug("Calculating baseline solution...")
        fixed_params["orbitals"], (E_hf, E_casscf) = get_baseline_solution(physical_config, cas_config, config.orbitals.n_determinants)
        fixed_params["baseline_energies"].update(dict(E_hf=E_hf, E_casscf=E_casscf))
        LOGGER.debug(f"Finished baseline calculation: E_casscf={E_casscf:.6f}")

    # Compute orbital coefficients as descriptors for Transferable Atomic Orbitals
    if config.orbitals.transferable_atomic_orbitals:
        # Build descriptors
        atom_types = config.orbitals.transferable_atomic_orbitals.atom_types or set(physical_config.Z)
        nb_orbitals_per_Z = get_n_basis_per_Z(config.orbitals.transferable_atomic_orbitals.basis_set, tuple(atom_types))
        node_features_phisnet = None
        if config.orbitals.transferable_atomic_orbitals.phisnet_model:
            orbital_params, node_features_phisnet, hessian, (E_hf, E_casscf) = get_phisnet_solution(physical_config,
                                                                                                    phisnet_model,
                                                                                                    config.orbitals.transferable_atomic_orbitals.basis_set,
                                                                                                    config.orbitals.transferable_atomic_orbitals.localization,
                                                                                                    N_ions_max,
                                                                                                    nb_orbitals_per_Z,
                                                                                                    atomic_orbitals)
            fixed_params["hessian"] = hessian
        else:
            # Build a baseline solution
            baseline_config = CASSCFConfig(basis_set=config.orbitals.transferable_atomic_orbitals.basis_set,
                                           localization=config.orbitals.transferable_atomic_orbitals.localization,
                                           cusps=None)
            orbital_params, (E_hf, E_casscf) = get_baseline_solution(physical_config, baseline_config, 1)
        fixed_params["baseline_energies"].update(dict(E_hf=E_hf, E_casscf=E_casscf))

        features = get_atomic_orbital_descriptors(orbital_params.mo_coeff,
                                                  physical_config.Z,
                                                  atom_types,
                                                  nb_orbitals_per_Z,
                                                 )
        fixed_params['transferable_atomic_orbitals'] = dict(orbitals=orbital_params, features=features, features_ion_phisnet=node_features_phisnet)
        
        if "e3" in config.orbitals.transferable_atomic_orbitals.name:
            U, irreps = get_ao_to_irreps_matrix(orbital_params.atomic_orbitals)
            fixed_params['transferable_atomic_orbitals']['features_e3'] = e3nn.IrrepsArray(irreps, features @ U)


    return jax.tree_util.tree_map(jnp.array, fixed_params)


