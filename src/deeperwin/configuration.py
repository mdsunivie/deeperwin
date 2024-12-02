# %%
"""
DeepErwin hyperparameter and configuration management.
"""

import copy
from typing import Union, Optional, List, Tuple, Iterable, Any, Dict, Sequence
from typing_extensions import Literal
from pydantic import BaseModel, model_validator
import numpy as np
import pathlib
import ruamel.yaml as yaml
import ruamel.yaml.comments

LocalizationType = Literal[None, "", "none", "boys", "pm"]
ActivationType = Literal["tanh", "silu", "elu", "relu", "gelu"]


class ConfigBaseclass(BaseModel):
    """Base class for all config models"""

    def save(self, file):
        data = to_prettified_yaml(self.dict())
        if isinstance(file, str):
            with open(file, "w") as f:
                ruamel.yaml.YAML().dump(data, f)
        else:
            ruamel.yaml.YAML().dump(data, file)

    @classmethod
    def load(cls, file) -> "ConfigBaseclass":
        if isinstance(file, str):
            with open(file) as f:
                data = ruamel.yaml.YAML().load(f)
        else:
            data = ruamel.yaml.YAML().load(file)
        return cls.model_validate(data)

    def as_flattened_dict(self):
        output_dict = {}
        for label in self.dict():
            subconfig = getattr(self, label)
            if hasattr(subconfig, "as_flattened_dict"):
                subdict = subconfig.as_flattened_dict()
                for sublabel, subvalue in subdict.items():
                    output_dict[f"{label}.{sublabel}"] = subvalue
            else:
                output_dict[label] = subconfig
        return output_dict

    @classmethod
    def from_flattened_dict(cls, config_dict, ignore_extra_settings=False):
        if ignore_extra_settings:

            class IgnoringConfig(cls):
                class Config:
                    extra = "ignore"

            nested_dict = IgnoringConfig.from_flattened_dict(config_dict).dict()
        else:
            nested_dict = build_nested_dict(config_dict)

        config = cls.model_validate(nested_dict)
        return config

    @model_validator(mode="after")
    def disable_unused_children(self):
        for k, v in self:
            if hasattr(v, "use") and not getattr(v, "use"):
                setattr(self, k, None)
        return self

    @classmethod
    def update_configdict_and_validate(cls, config_dict: dict, config_changes: Iterable[Tuple[str, Any]]):
        config_dict = copy.deepcopy(config_dict)
        # First loop: Build an updated dictionary
        for key, value in config_changes.items():
            set_with_flattened_key(config_dict, key, value)
        # Parse the config dict and validate that all parameters are valid
        parsed_config = cls.model_validate(config_dict)

        # Second loop: Update the values using the parsed values to get correct type; Not strictly necessary,
        # but yields nicer looking input-config files
        for key, value in config_changes.items():
            try:
                set_with_flattened_key(config_dict, key, get_with_flattened_key(parsed_config, key))
            except AttributeError:
                print(f"Following key/value no longer present in parsed config. Skipping {key}={value}")
        return config_dict, parsed_config

    class Config:
        """The name 'Config' here is a pydantic term used to config the behaviour of the dataclass.
        It's not a config for the variational monte carlo code"""

        extra = "forbid"


class InitializationConfig(ConfigBaseclass):
    bias_scale: float = 0.0

    weight_scale: Literal["glorot", "glorot-input"] = "glorot"

    weight_distribution: Literal["normal", "uniform"] = "uniform"


class AttentionConfig(ConfigBaseclass):
    n_heads: int = 4
    attention_dim: int = 32
    use_layer_norm: bool = True
    use_residual: bool = True


class MessagePassingConfig(ConfigBaseclass):
    node_dim: int = 256
    edge_dim: int = 32
    use_edge_features_for_gating: bool = True
    use_node_features_for_gating: bool = True
    update_edge_features: bool = True
    weighting: Literal["linear", "softmax"] = "linear"
    aggregation: Literal["sum", "mean"] = "sum"
    activation: Optional[ActivationType] = "silu"


class MessagePassingConfigPhisnetFeatures(ConfigBaseclass):
    node_dim: int = 64
    edge_dim: int = 32
    use_edge_features_for_gating: bool = True
    use_node_features_for_gating: bool = False
    update_edge_features: bool = False
    weighting: Literal["linear", "softmax"] = "linear"
    aggregation: Literal["sum", "mean"] = "sum"
    activation: Optional[ActivationType] = "silu"


class MessagePassingConfigOrbFeatures(MessagePassingConfig):
    node_dim: int = 16
    edge_dim: int = 32
    use_edge_features_for_gating: bool = True
    use_node_features_for_gating: bool = False
    update_edge_features: bool = False
    weighting: Literal["linear", "softmax"] = "linear"
    aggregation: Literal["sum", "mean"] = "sum"
    activation: Optional[ActivationType] = "silu"


class DenseGNNConfig(ConfigBaseclass):
    n_iterations: int = 3
    attention: Optional[AttentionConfig] = AttentionConfig()
    message_passing: Optional[MessagePassingConfig] = MessagePassingConfig()
    use_edges_in_attention: bool = True
    mlp_depth: int = 1
    linear_out: bool = False
    edge_embedding_depth: int = 0
    edge_embedding_width: int = 64
    final_mpnn: bool = False
    use_edge_bias: bool = True


class ExpScaling(ConfigBaseclass):
    name: Literal["exp_scale"] = "exp_scale"
    mean_scaling: float = 10
    std_scaling: float = 1
    nb_feat: int = 32


class OrbFeatureDenseGNNConfig(DenseGNNConfig):
    n_iterations: int = 1
    attention: Optional[AttentionConfig] = None
    message_passing: Optional[MessagePassingConfigOrbFeatures] = MessagePassingConfigOrbFeatures()
    use_edges_in_attention: bool = True
    mlp_depth: int = 1
    linear_out: bool = False
    edge_embedding_width: int = 32
    edge_embedding_depth: int = 1
    mlp_edge_depth: int = 0
    exp_scaling: Union[None, ExpScaling] = ExpScaling()
    antisymmetrize: bool = False


class TransformerConfig(ConfigBaseclass):
    n_iterations: int = 4
    """Number of embedding iterations"""

    n_heads: int = 4
    """Number of self-attention heads"""

    attention_dim: int = 64
    """Dimensionality of queries/keys per head in self-attention"""

    attention_value_dim: Optional[int] = None
    """Dimensionality of values per head in in self-attention. If None is provided, will use attention_dim, i.e. the dimension of queries/keys."""

    attention_output_dim: Optional[int] = None
    """Dimension of one-electron features going into and out-of MLP"""

    mlp_depth: int = 0
    """Number of hidden layers in one-electron-MLP"""

    use_layer_norm: bool = False
    """Apply layer norm between self-attention and one-electron-MLP"""

    # initialize_with_sender_att: bool = False

    residual: bool = True

    final_residual: bool = True

    use_residual_before_lin: bool = False

    combine_attention_blocks: bool = (
        False  # computes ion attention & el attention in serial fashion if there is a sender input
    )


class MLPConfig(ConfigBaseclass):
    activation: Literal["tanh", "silu", "elu", "relu", "gelu"] = "tanh"

    init_bias_scale: float = 0.0

    init_weights_scale: Literal["fan_in", "fan_out", "fan_avg"] = "fan_avg"

    init_weights_distribution: Literal["normal", "truncated_normal", "uniform"] = "uniform"

    use_residual: bool = False

    use_layer_norm: bool = False


class FeedForwardConfig(ConfigBaseclass):
    name: str = "mlp"
    n_layers: int = 1
    n_hidden_dim: int = 32
    mlp_config: MLPConfig = MLPConfig()


class EdgeFeatureConfig(ConfigBaseclass):
    name: str = "edge_feature_transformer"
    # mlp: Optional[FeedForwardConfig] = None
    # attention: Optional[AttentionConfig] = None
    ion_ion: bool = False
    el_ion: bool = False
    el_el: bool = False


class EmbeddingConfigTransformer(ConfigBaseclass):
    name: Literal["transformer"] = "transformer"

    el_transformer: TransformerConfig = TransformerConfig()
    """Configuration for the attention of electrons to other electrons and ions"""

    ion_transformer: Optional[TransformerConfig] = None
    """Configuration for the attention between ions. If None is provided, no ion-ion attention-based embedding is used."""

    edge_feature: Optional[EdgeFeatureConfig] = None


class EmbeddingConfigAxialTranformer(ConfigBaseclass):
    name: Literal["axial_transformer"] = "axial_transformer"

    ion_attention: AttentionConfig = AttentionConfig()

    el_attention: AttentionConfig = AttentionConfig()

    n_iterations: int = 4

    output_dim: int = 64

    embedding_dim: int = 64

    mlp_depth: int = 0

    agg_ion_contribution: bool = False

    axial_attention_ion_per_layer: bool = True


class EmbeddingConfigFermiNet(ConfigBaseclass):
    name: Literal["ferminet"] = "ferminet"

    n_hidden_one_el: Union[List[int], int] = 256
    """Number of hidden neurons per layer for the one electron stream in the FermiNet network"""

    n_hidden_two_el: Union[List[int], int] = 32
    """Number of hidden neurons per layer for the electron-electron stream in the FermiNet network"""

    n_hidden_el_ions: Union[List[int], int] = 32
    """Number of hidden neurons per layer for the electron-ion stream in the FermiNet network"""

    n_iterations: Optional[int] = 4
    """Number iterations to combine features of different particle streams into the one electron stream"""

    use_el_ion_stream: bool = False
    """Whether to generate a second stream of interactions between electrons and ions"""

    use_average_h_one: bool = True
    """Whether to include the average embedding of all electrons as a feature into the 1-el-stream"""

    use_h_one_same_diff: bool = False
    """Whether to average over all 1-el-streams with same/diff spin (relative to the current electron) or average over all up/dn spin streams"""

    initialization: InitializationConfig = InitializationConfig()
    """How to initialize weights and biases"""

    use_average_h_two: bool = True
    """Use the average over the electron-electron stream output as additional feature for the one electron stream"""

    use_h_one: bool = True
    """Use the electron stream output as features for its own stream in the next iteration"""

    use_h_two_same_diff: bool = False
    """Split the electron-electron stream into same- and different spin channel. Effectively turning one stream into two
    seperate streams"""

    use_schnet_features: bool = False
    """Use SchNet convolution features as input to the one electron stream"""

    sum_schnet_features: bool = True
    """Sum electron-electron and electron-ion SchNet convolution features together"""

    emb_dim: int = 128
    """Embedding dimension for the electron-electron and electron-ion SchNet convolution features"""

    use_linear_out: bool = False
    """Use linear layer for mapping particle features to same dimension"""

    use_w_mapping: bool = False
    """Apply neural network for inter-particle features to further process before computing SchNet convolution"""

    use_layer_norm: bool = False
    """Whether to apply layer norm after linear mapping of symmetric features"""

    neighbor_normalization: Literal["sum", "sqrt", "mean"] = "sum"

    use_h_one_mlp: bool = True

    h_one_correlation: int = 0

    use_symmetric_product: bool = True

    downmap_during_product: bool = True

    one_el_skip_conn: bool = True

    use_schnet_bias_feat: bool = True

    schnet_aggregation: str = "sum"

    use_ln_aft_act: bool = False

    use_ln_bef_act: bool = False

    use_deep_schnet_feat: bool = False

    @model_validator(mode="after")
    def _set_n_hidden(self):
        def _make_list(key, length):
            if isinstance(getattr(self, key), int):
                setattr(self, key, [getattr(self, key)] * length)

        _make_list("n_hidden_one_el", self.n_iterations)
        _make_list("n_hidden_two_el", self.n_iterations - 1)
        _make_list("n_hidden_el_ions", self.n_iterations - 1)
        if len(self.n_hidden_one_el) != (len(self.n_hidden_two_el) + 1):
            raise ValueError("Number of layers for 1-el-stream must be one more than nr of layers in 2-el-stream")
        return self

    @model_validator(mode="after")
    def _set_n_iterations(self):
        if self.n_iterations is None:
            try:
                self.n_iterations = len(self.n_hidden_one_el)
            except TypeError as e:
                raise ValueError("n_hidden_one_el must be a list of net-widths if n_iterations is not specified") from e
        return self


class PhisNetIonEmbeddingConfig(ConfigBaseclass):
    name: Literal["phisnet_ion_emb"] = "phisnet_ion_emb"
    ion_width: int = 64
    ion_depth: int = 1
    message_passing: Optional[MessagePassingConfigPhisnetFeatures] = MessagePassingConfigPhisnetFeatures()
    use_edge_bias: bool = False


class IonGNNConfig(DenseGNNConfig):
    name: Literal["ion_gnn"] = "ion_gnn"


class EmbeddingConfigGNN(ConfigBaseclass):
    name: Literal["gnn"] = "gnn"
    gnn: DenseGNNConfig = DenseGNNConfig()
    ion_gnn: Union[None, IonGNNConfig, PhisNetIonEmbeddingConfig] = None

    exp_scaling: Union[None, ExpScaling] = None
    ion_ion_width: int = 64
    ion_ion_depth: int = 3
    el_ion_width: int = 64
    el_ion_depth: int = 3
    el_el_width: int = 64
    el_el_depth: int = 3
    cutoff_type: Literal["constant", "inverse", "exponential", "cosine"] = "constant"
    cutoff_distance: float = 5.0


# class MoonMPNN(ConfigBaseclass):
#     envelope_shape: int = 32
#     nb_layer: int = 2


# class EmbeddingConfigMoon(ConfigBaseclass):
#     name: Literal["moon"] = "moon"
#     n_iterations: int = 3
#     el_el_edge_dim: int = 32
#     el_ion_edge_dim: int = 32
#     nuclei_emb: int = 256
#     output_dim: int = 256

#     el_mpnn: MoonMPNN = MoonMPNN()
#     ion_mpnn: MoonMPNN = MoonMPNN()


class EmbeddingConfigDeepErwin4(EmbeddingConfigFermiNet):
    """DO NOT INTRODUCE NEW FIELDS HERE. This class is only used to provide alternative defaults"""

    name: Literal["dpe4"] = "dpe4"
    n_iterations: int = 4
    n_hidden_one_el: Union[List[int], int] = 256
    n_hidden_two_el: Union[List[int], int] = 32
    n_hidden_el_ions: Union[List[int], int] = 32
    use_el_ion_stream: bool = True
    use_h_two_same_diff: bool = True
    emb_dim: int = 32
    use_w_mapping: bool = True
    use_schnet_features: bool = True
    sum_schnet_features: bool = False
    use_average_h_one: bool = True
    use_average_h_two: bool = False
    use_h_one: bool = True
    use_linear_out: bool = False
    neighbor_normalization: Literal["sum", "sqrt", "mean"] = "mean"


class EmbeddingConfigMoon(ConfigBaseclass):
    name: Literal["moon"] = "moon"
    n_iterations: int = 3
    n_hidden_spatial: List[int] = [16, 8]
    el_el_dim: int = 32
    el_ion_dim_collect: int = 256
    ion_dim: int = 256
    output_dim: int = 256
    activation: ActivationType = "silu"
    initial_scale_spatial: Union[float, List[float]] = 10.0
    initial_scale_output: Union[float, List[float]] = [0.2, 2]
    envelope_power_spatial: int = 2
    envelope_power_output: int = 1
    use_ion_ion_features: bool = False


class CuspCorrectionConfig(ConfigBaseclass):
    use: bool = True
    cusp_type: Literal["mo", "ao"] = "mo"
    """Mode how to calculate cusp-corrected orbitals. 
    'ao' computes a cusp correction for each atomic orbital (i.e. for each basis function), 
    'mo' computes a cusp correction for each molecular orbital (i.e. each solution of the HF-calculation).
    For atoms both cusp_types should be equivalent, but for molecules the simpler 'ao' cusps can in principle not correctly model the cusps that arise from an atomic wavefunction having a finite contribution at a different nucleus.   
    """

    r_cusp_el_ion_scale: float = 1.0
    """Scaling parameter for the electron-ion cusp corrections. No cusp correction is applied outside a radius :code:`r_cusp_el_ion_scale / Z`"""


class PyscfMeanFieldOptionsConfig(ConfigBaseclass):
    chkfile: Optional[str] = None
    """File to save cell and meanfield objects. Only implemented for periodic
    calculations, because pyscf is slow there."""

    density_fit: bool = True
    """Enables density fitting. Only implemented for periodic calculations,
    because default plane wave fitting can quickly run out of memory."""

    df_mesh: Optional[List[int]] = None
    """Mesh for density fitting"""


class PyscfOptionsConfig(ConfigBaseclass):
    verbose: int = 0
    output: Optional[str] = "/dev/null"
    precision: float = 1e-8
    exp_to_discard: Optional[float] = None
    mf_options: PyscfMeanFieldOptionsConfig = PyscfMeanFieldOptionsConfig()


class HartreeFockConfig(ConfigBaseclass):
    name: Literal["hf"] = "hf"
    """Identifier of the baseline calculation. Fixed."""

    basis_set: str = "6-311G"
    """Basis set to use for the Hartree-Fock / CASSCF calculation. See the documentation of pySCF for all available basis-sets."""

    localization: LocalizationType = None

    pyscf_options: PyscfOptionsConfig = PyscfOptionsConfig()
    """Extra options for pyscf."""


class PhisnetBaselineConfig(ConfigBaseclass):
    name: Literal["phisnet"] = "phisnet"

    basis_set: str = "6-311G"

    localization: LocalizationType = None

    atom_types: Optional[List[int]] = None


class PeriodicMeanFieldConfig(ConfigBaseclass):
    name: Literal["periodic_hf", "periodic_dft", "periodic_unrestricted_dft"] = "periodic_hf"

    basis_set: Union[str, Dict] = "6-311G"

    pseudo: Optional[Union[str, Dict]] = None

    k_points: Optional[Union[Sequence[int], Sequence[Sequence[float]]]] = None
    """
    k-points to use for the mean field calculation. Can either be a list of 3D k-points, or a list of 3 integeres, which will be used to generate a gamma-centered k-mesh; 
    If set to None, the k-points will be chosed based on the supercell of the physical config, i.e. for a primitive cell a gamma-calculation is performed, for an [n x n x n] supercell, a n x n x k-mesh is generated
    """

    pyscf_options: PyscfOptionsConfig = PyscfOptionsConfig()
    """Extra options for pyscf."""

    localization: LocalizationType = None


class CASSCFConfig(ConfigBaseclass):
    name: Literal["casscf"] = "casscf"
    """Identifier of the baseline calculation. Fixed."""

    basis_set: str = "6-311G"
    """Basis set to use for the Hartree-Fock / CASSCF calculation. See the documentation of pySCF for all available basis-sets."""

    n_active_electrons: int = 4

    n_active_orbitals: int = 8

    localization: LocalizationType = None

    n_dets: Optional[int] = None

    pyscf_options: PyscfOptionsConfig = PyscfOptionsConfig()
    """Extra options for pyscf."""


BaselineConfigType = Optional[Union[HartreeFockConfig, CASSCFConfig, PeriodicMeanFieldConfig, PhisnetBaselineConfig]]


class InputFeatureConfig(ConfigBaseclass):
    use_rbf_features: bool
    """Whether to build distance features using gaussian functions of the distances"""
    n_rbf_features: int
    """Number of radial basis functions to use as pairwise fature vector"""

    r_cut_bessel: float = 5.0
    """Cut-off radius for the bessel functions"""

    full_el_el_distance_matrix: bool
    """Whether to use a full [n_el x n_el] matrix of distances, including the diagonal 0 terms"""

    use_distance_features: bool
    """Whether to include the absolute value of the distance (and powers of this distance) as input feature"""

    use_el_ion_differences: bool
    """Whether to include electron-ion differences as input features"""

    use_el_el_differences: bool
    """Whether to include electron-electron differences as input features"""

    use_el_spin: bool = False

    # TODO: remove deprecated
    coordinates: Literal["cartesian"] = "cartesian"
    """How to preprocess the input coordinates: Cartesian = unprocessed; global_rot = one global rotation matrix for all electron and ion coordinates; local_rot = one rotation matrix for each ion, affecting all corresponding electron-ion differences"""

    n_ion_ion_rbf_features: int = 32

    n_ion_ion_mlp_features: int = 0

    ion_embed_type: Optional[Literal["lookup", "one-hot", "mlp"]] = "lookup"

    n_ion_features: int = 32

    concatenate_el_ion_features: bool
    """Whether to use concatenated electron-ion features as initialization for h_one; breaks equivariance"""

    # n_hidden_one_el_features: List[int]
    # """Number of hidden nodes for generation of one-electron-features from el-ion-features"""

    use_el_ion_convolution: bool = False

    log_scale_distances: bool = False
    """Scale the difference- and distance-features by a factor of log(1+r)/r"""

    init_as_zeros: bool = False

    # El-el edge & el init with el-el info
    n_el_el_features: int = 32
    n_el_el_layers: int = 2
    el_el_gating_operation: Literal["rbf", "gauss", "none"] = "none"
    exp_decay_el_el_edge: bool = False
    init_with_el_el_feat: bool = False

    # El-ion edge & el init with ion info
    n_el_ion_features: int = 32
    """Generate one-electron features as a sum of el-ion features"""
    n_el_ion_layers: int = 2
    el_ion_gating_operation: Literal["rbf", "gauss", "none"] = "none"
    exp_decay_el_ion_edge: bool = False
    init_with_el_ion_feat: bool = False

    rmax: int = 5
    max_scale_gauss: float = 8.0

    include_twist: Optional[List[Literal["el", "ion", "el_el", "el_ion", "ion_ion"]]] = None

    @model_validator(mode="after")
    def local_coords_concatenation_validation(self):
        if self.coordinates == "local_rot" and not self.concatenate_el_ion_features:
            raise ValueError(
                "Can't use local coordinate system if el_ion_features are not concatenated -> architecture not expressive enough"
            )
        return self


class InputFeatureConfigFermiNet(InputFeatureConfig):
    """DO NOT INTRODUCE NEW FIELDS HERE. This class is only used to provide alternative defaults"""

    name: Literal["ferminet"] = "ferminet"
    use_rbf_features: bool = False
    n_rbf_features: int = 0
    use_distance_features: bool = True
    use_el_ion_differences: bool = True
    use_el_el_differences: bool = True
    # n_el_ion_features: int = 0
    init_with_el_ion_feat: bool = False
    concatenate_el_ion_features: bool = True
    use_el_el_spin: bool = False
    coordinates: Literal["cartesian", "global_rot", "local_rot"] = "cartesian"
    log_scale_distances: bool = False
    use_el_spin: bool = False
    full_el_el_distance_matrix: bool = True
    n_ion_ion_rbf_features: int = 32
    ion_embed_type: Optional[Literal["lookup", "one-hot", "mlp"]] = None


class InputFeatureConfigTransformer(InputFeatureConfigFermiNet):
    """DO NOT INTRODUCE NEW FIELDS HERE. This class is only used to provide alternative defaults"""

    name: Literal["transformer"] = "transformer"
    use_el_el_differences: bool = False
    log_scale_distances: bool = True
    use_el_spin: bool = True
    full_el_el_distance_matrix: bool = False
    n_ion_ion_rbf_features: int = 32


class InputFeatureConfigDPE4(InputFeatureConfigFermiNet):
    """DO NOT INTRODUCE NEW FIELDS HERE. This class is only used to provide alternative defaults"""

    name: Literal["dpe4"] = "dpe4"
    coordinates: Literal["cartesian", "global_rot", "local_rot"] = "cartesian"
    use_el_el_differences: bool = False
    full_el_el_distance_matrix: bool = True
    n_ion_ion_rbf_features: int = 32
    ion_embed_type: Optional[Literal["lookup", "one-hot", "mlp"]] = "lookup"


class InputFeatureConfigMoon(InputFeatureConfigFermiNet):
    name: Literal["moon"] = "moon"
    coordinates: Literal["cartesian", "global_rot", "local_rot"] = "cartesian"
    use_el_el_differences: bool = True
    concatenate_el_ion_features: bool = False
    n_ion_ion_rbf_features: int = 0
    n_ion_features: int = 32
    n_ion_ion_mlp_features: int = 0
    ion_embed_type: Optional[Literal["lookup", "one-hot", "mlp"]] = "lookup"


class EnvelopeOrbitalsConfig(ConfigBaseclass):
    envelope_type: Literal["isotropic_exp"] = "isotropic_exp"
    """Which enveolope to use for the backflow-add term"""

    n_hidden: List[int] = []
    """List of ints, specifying the number of hidden units per layer in the backflow-factor-network."""

    use_bias: bool = False
    """Enable / disable bias terms for the final layer of the backflow-add network"""

    initialization: Literal["constant", "analytical"] = "constant"
    """Use a least-squares fit to initialized the scale- and decay-parameters for the envelopes instead of constant intialization"""


class TransferableAtomicOrbitalEmbedding(ConfigBaseclass):
    use: bool = True

    tao_embed_width: int = 64

    tao_embed_depth: int = 2

    tao_embed_dim: int = 64

    spin_adapted_network: bool = False


class PhisNetModelConfig(ConfigBaseclass):
    n_iterations: int = 1
    n_channels: int = 12
    L: int = 2
    n_rbf_features: int = 32
    r_cutoff: float = 30.0
    r_scale: float = 4.0
    Z_max: int = 9
    irreps_basis: str = "2x0e+1x1o"
    force_overlap_diag_to_one: bool = True
    predict_S: bool = True
    predict_H: bool = True
    predict_H_core: bool = True
    predict_rho: bool = False
    predict_energy: bool = False
    predict_forces: bool = False


class TransferableAtomicOrbitalsConfig(ConfigBaseclass):
    name: Literal["taos"] = "taos"

    envelope_width: int = 64

    envelope_depth: int = 2

    backflow_width: int = 256

    backflow_depth: int = 2

    symmetrize_exponent_mlp: bool = False

    antisymmetrize_backflow_mlp: bool = False

    use_prefactors: bool = False

    use_exponentials: bool = True

    use_el_ion_embedding: bool = False

    use_squared_envelope_input: bool = False

    mlp: MLPConfig = MLPConfig()

    orb_feature_gnn: Optional[OrbFeatureDenseGNNConfig] = None

    n_rbf_features_orb_gnn: int = 16

    phisnet_model: Optional[PhisNetModelConfig] = None

    atom_types: Optional[List[int]] = None

    # twist_encoding: Optional[Literal["concat", "concat_periodic"]] = None
    twist_encoding: Optional[List[Literal["concat", "periodic"]]] = None

    use_separate_ion_sum_for_envelopes: bool = False

    use_atom_positions: bool = False

    use_orbital_positions: bool = False

    use_atom_orbital_diff: bool = False


class PeriodicEnvelopeOrbitalsConfig(ConfigBaseclass):
    use: bool = True

    n_k_points_min: int = 50


class OrbitalsConfig(ConfigBaseclass):
    envelope_orbitals: Optional[EnvelopeOrbitalsConfig] = EnvelopeOrbitalsConfig()
    """Orbitals containing only a simple, parametrized envelope, multiplied by a neural network"""

    transferable_atomic_orbitals: Union[None, TransferableAtomicOrbitalsConfig] = None
    """Orbitals being composed of atom-wise contributions, conditioned on orbital descriptors such as Hartree-Fock basis coefficients"""

    n_determinants: int = 32
    """Number of determinants in the wavefunction model"""

    determinant_schema: Literal["full_det", "block_diag", "restricted_closed_shell"] = "full_det"
    """How to construct slater determinants from a set of predicted (spin) orbitals"""

    periodic_orbitals: Union[None, PeriodicEnvelopeOrbitalsConfig] = None
    """Periodic envelope for solids"""

    use_bloch_envelopes: bool = False


class OrbitalsConfigFermiNet(OrbitalsConfig):
    envelope_orbitals: Optional[EnvelopeOrbitalsConfig] = EnvelopeOrbitalsConfig()
    transferable_atomic_orbitals: Union[None, TransferableAtomicOrbitalsConfig] = None
    n_determinants: int = 32
    determinant_schema: Literal["full_det", "block_diag", "restricted_closed_shell"] = "full_det"


class JastrowConfig(ConfigBaseclass):
    use: bool = True

    n_hidden: List[int] = [40, 40]
    """List of ints, specifying the number of hidden units per layer in the jastrow-network. If not provided, the width and depth set by *net_width* and *net_depth* are used."""

    differentiate_spins: bool = False
    """Use separate functions for J(spin_up) and J(spin_down)"""


class ModelConfig(ConfigBaseclass):
    """Configuration for the primary wavefunction model, which maps electron coordinate to psi"""

    features: InputFeatureConfig
    """Config-options for mapping raw inputs (r,R,Z) to some symmetrized input features"""

    embedding: Union[
        EmbeddingConfigDeepErwin4,
        EmbeddingConfigFermiNet,
        EmbeddingConfigTransformer,
        EmbeddingConfigAxialTranformer,
        EmbeddingConfigMoon,
        None,
    ]
    """Config-options for mapping symmetrized input features to high-dimensional embeddings of electrons"""

    orbitals: OrbitalsConfig
    """Config-options for computing orbitals from embedding vectors of electrons"""

    mlp: MLPConfig = MLPConfig()
    """How to build multi-layer-perceptrons: Activation and how to initialize weights and biases"""

    jastrow: Optional[JastrowConfig]
    """Enable the jastrow-factor, i.e. multiple the total wavefunction by the output of a global neural network"""

    use_el_el_cusp_correction: bool
    """Explicit, additive el-el-cusp correction"""

    disable_determinant: bool = False
    """Flag purely for debugging purposes: Computes an average over all orbitals instead of a determinant, to profile computational cost"""

    max_n_up_orbitals: Optional[int] = None
    """Optional field to request n amount of up orbitals from deeperwin, the orbitals will be truncated down for smaller systems """

    max_n_dn_orbitals: Optional[int] = None
    """Optional field to request n amount of dn orbitals from deeperwin, the orbitals will be truncated down for smaller systems """

    Z_max: Optional[int] = None  # 18=Argon
    """Optional field to define the max Z a model can compute wavefunctions on"""

    Z_min: Optional[int] = 1
    """Optional field to define the min Z a model can compute wavefunctions on"""

    max_n_ions: Optional[int] = None
    """Optional field to define the max number of ions a model can compute wavefunctions on"""

    use_cache: bool = True
    """Cache parts of the network which are constant across electron-samples of the same geometry, such as purely geometry dependent networks."""

    complex_wf: bool = False
    """Use complex-valued output of network."""

    kfac_register_complex: bool = False
    """Register log psi instead of log |psi| with kfac."""


class ModelConfigDeepErwin4(ModelConfig):
    """DO NOT INTRODUCE NEW FIELDS HERE. This class is only used to provide alternative defaults"""

    name: Literal["dpe4"] = "dpe4"
    features: Union[InputFeatureConfigDPE4, InputFeatureConfigFermiNet] = InputFeatureConfigDPE4()
    embedding: Union[
        EmbeddingConfigDeepErwin4,
        EmbeddingConfigFermiNet,
        EmbeddingConfigGNN,
        EmbeddingConfigAxialTranformer,
        EmbeddingConfigMoon,
        None,
    ] = EmbeddingConfigDeepErwin4()
    orbitals: OrbitalsConfigFermiNet = OrbitalsConfigFermiNet()
    jastrow: Optional[JastrowConfig] = None
    use_el_el_cusp_correction: bool = False


class ModelConfigMoon(ModelConfig):
    """DO NOT INTRODUCE NEW FIELDS HERE. This class is only used to provide alternative defaults"""

    name: Literal["moon"] = "moon"
    features: Union[InputFeatureConfigDPE4, InputFeatureConfigFermiNet, InputFeatureConfigMoon] = (
        InputFeatureConfigMoon()
    )
    embedding: Union[
        EmbeddingConfigDeepErwin4,
        EmbeddingConfigFermiNet,
        EmbeddingConfigGNN,
        EmbeddingConfigAxialTranformer,
        EmbeddingConfigMoon,
        None,
    ] = EmbeddingConfigMoon()
    orbitals: OrbitalsConfigFermiNet = OrbitalsConfigFermiNet()
    jastrow: Optional[JastrowConfig] = None
    use_el_el_cusp_correction: bool = False


class ModelConfigFermiNet(ModelConfig):
    """DO NOT INTRODUCE NEW FIELDS HERE. This class is only used to provide alternative defaults"""

    name: Literal["ferminet"] = "ferminet"
    features: InputFeatureConfigFermiNet = InputFeatureConfigFermiNet()
    embedding: Union[EmbeddingConfigFermiNet, EmbeddingConfigDeepErwin4, EmbeddingConfigMoon, None] = (
        EmbeddingConfigFermiNet()
    )
    orbitals: OrbitalsConfigFermiNet = OrbitalsConfigFermiNet()
    jastrow: Optional[JastrowConfig] = None
    use_el_el_cusp_correction: bool = False


class ModelConfigTransformer(ModelConfig):
    """DO NOT INTRODUCE NEW FIELDS HERE. This class is only used to provide alternative defaults"""

    name: Literal["transformer"] = "transformer"
    features: Union[InputFeatureConfigTransformer, InputFeatureConfigFermiNet] = InputFeatureConfigTransformer()
    embedding: Union[
        EmbeddingConfigTransformer,
        EmbeddingConfigFermiNet,
        EmbeddingConfigDeepErwin4,
        EmbeddingConfigAxialTranformer,
        None,
    ] = EmbeddingConfigTransformer()
    orbitals: OrbitalsConfigFermiNet = OrbitalsConfigFermiNet()
    jastrow: Optional[JastrowConfig] = None
    use_el_el_cusp_correction: bool = True


EmbeddingConfigType = Union[
    EmbeddingConfigDeepErwin4,
    EmbeddingConfigFermiNet,
    EmbeddingConfigTransformer,
    EmbeddingConfigMoon,
    None,
]


class MCMCSimpleProposalConfig(ConfigBaseclass):
    name: Literal["normal", "cauchy", "normal_one_el"] = "normal"


class MCMCLangevinProposalConfig(ConfigBaseclass):
    name: Literal["langevin"] = "langevin"

    langevin_scale: float = 1.0

    r_min: float = 0.2

    r_max: float = 2.0


class LocalStepsizeProposalConfig(ConfigBaseclass):
    """Config for a local stepsize proposal rule for MCMC. Stepsize depends on distance to closest nuclei"""

    name: Literal["local", "local_one_el"] = "local"

    r_min: float = 0.1
    """Minimal stepsize for electron move"""

    r_max: float = 1
    """Max stepsize for electron move"""


class MCMCConfig(ConfigBaseclass):
    """Config for Markov-Chain-Monte-Carlo integration"""

    n_inter_steps: int
    """Number of MCMC steps between epochs"""

    n_burn_in: int
    """Number of MCMC steps before starting optimization"""

    max_age: int
    """Maximum number of MCMC steps for which a walker can reject updates during optimization. After having rejected an update max_age times, the walkers is forced to accepet, to avoid getting stuck"""

    stepsize_update_interval: int
    """Number of steps after which the step-size is adjusted"""

    n_walkers: int = 2048
    """Number of walkers for optimization"""

    spin_initialization: Literal["el_ion_mapping", "random", "ordered", "mixed"] = "el_ion_mapping"
    """How to distribute spins across atoms.

    el_ion_mapping: Use the el_ion_mapping provided in PhysicalConfig, which in turn typically defaults to Hundts rule / AFM order.
    random: Randomly distribute spin-up electrons across atoms, and fill the rest with spin-down electrons.
    ordered: Put all spin-up electrons on the first atoms and all spin-down on the last atoms. E.g. yields FM order for Hydrogen chains.
    mixed: 50% of the batch are initialized according to el_ion_mapping and 50% are initialized 'ordered'
    """

    initialization: Literal["gaussian", "exponential"] = "exponential"
    """Initial radial distribution of electrons around ions."""

    target_acceptance_rate: float = 0.5
    """Acceptance-rate that the MCMC-runner is trying to achieve by modifying the stepsize"""

    min_stepsize_scale: float = 1e-2
    """Minimum stepsize. For spatially adaptive stepsize-schemes this only defines a factor which may be modified by the adaptive scheme"""

    max_stepsize_scale: float = 1.0
    """Maximum stepsize. For spatially adaptive stepsize-schemes this only defines a factor which may be modified by the adaptive scheme"""

    proposal: Union[
        MCMCSimpleProposalConfig,
        LocalStepsizeProposalConfig,
        MCMCLangevinProposalConfig,
    ] = MCMCSimpleProposalConfig()
    """Type of proposal function to use for MCMC steps"""

    p_spin_swap: float = 0.0
    """Probability of swapping the spin of two electrons during an MCMC step"""

    p_spin_flip: float = 0.0
    """Probability of flipping the spin of all electrons during an MCMC step"""


class MCMCConfigPreTrain(MCMCConfig):
    n_inter_steps: int = 1
    n_burn_in: int = 0
    stepsize_update_interval: int = 1000
    max_age: int = 20


class MCMCConfigOptimization(MCMCConfig):
    n_inter_steps: int = 20
    n_burn_in: int = 1000
    stepsize_update_interval: int = 100
    max_age: int = 20


class MCMCConfigEvaluation(MCMCConfig):
    n_inter_steps: int = 20
    n_burn_in: int = 500
    stepsize_update_interval: int = 100
    max_age: int = 100


class ClippingConfig(ConfigBaseclass):
    name: Literal["hard", "tanh"] = "tanh"
    width_metric: Literal["std", "mae"] = "std"
    center: Literal["mean", "median"] = "mean"
    from_previous_step: bool = True
    clip_by: float = 5.0
    clip_imag_around_0: bool = False


class ConstantLRSchedule(ConfigBaseclass):
    name: Literal["fixed"] = "fixed"


class ExponentialLRSchedule(ConfigBaseclass):
    name: Literal["exponential"] = "exponential"
    decay_time: float = 10_000.0
    offset_time: float = 0
    warmup: int = 0
    minimum: float = 0.0


class InverseLRScheduleConfig(ConfigBaseclass):
    name: Literal["inverse"] = "inverse"
    decay_time: float = 6000.0
    offset_time: float = 0
    warmup: int = 0
    minimum: float = 0.0


class NoamLRScheduleConfig(ConfigBaseclass):
    name: Literal["noam"] = "noam"
    warmup_steps: int = 20


class _InverseLRScheduleConfigForKFACwithAdam(InverseLRScheduleConfig):
    decay_time: float = 1000
    warmup: int = 0


class StandardOptimizerConfig(ConfigBaseclass):
    name: Literal["adam", "rmsprop_momentum", "sgd", "lion", "lamb"] = (
        "adam"  # add others optimizers that don't need further configs here
    )

    learning_rate: float = 1e-3

    lr_schedule: Union[
        InverseLRScheduleConfig,
        ConstantLRSchedule,
        NoamLRScheduleConfig,
        ExponentialLRSchedule,
    ] = InverseLRScheduleConfig()
    """Schedule for the learning rate decay"""

    scaled_modules: Optional[List[str]] = None
    """List of parameters for which the learning rate is being scaled."""

    scale_lr: float = 1.0
    """Factor which to apply to the learning rates of specified modules"""


class _OptimizerConfigAdamForPretraining(StandardOptimizerConfig):
    name: Literal["adam"] = "adam"
    learning_rate: float = 3e-4
    lr_schedule: Union[
        ConstantLRSchedule,
        InverseLRScheduleConfig,
        NoamLRScheduleConfig,
        ExponentialLRSchedule,
    ] = ConstantLRSchedule()


class _OptimizerConfigSGD(StandardOptimizerConfig):
    name: Literal["adam", "rmsprop_momentum", "sgd"] = "sgd"
    learning_rate: float = 1.0
    lr_schedule: Union[
        InverseLRScheduleConfig,
        ConstantLRSchedule,
        NoamLRScheduleConfig,
        ExponentialLRSchedule,
    ] = ConstantLRSchedule()


class _OptimizerConfigAdamForKFAC(StandardOptimizerConfig):
    name: Literal["adam", "rmsprop_momentum", "sgd"] = "adam"
    learning_rate: float = 2e-3
    lr_schedule: Union[
        _InverseLRScheduleConfigForKFACwithAdam,
        InverseLRScheduleConfig,
        ConstantLRSchedule,
        NoamLRScheduleConfig,
        ExponentialLRSchedule,
    ] = _InverseLRScheduleConfigForKFACwithAdam()


class OptimizerConfigKFAC(ConfigBaseclass):
    name: Literal["kfac"] = "kfac"
    """Identifier. Fixed"""

    learning_rate: float = 0.1

    lr_schedule: Union[
        InverseLRScheduleConfig,
        ConstantLRSchedule,
        NoamLRScheduleConfig,
        ExponentialLRSchedule,
    ] = InverseLRScheduleConfig()
    """Schedule for the learning rate decay"""

    momentum: float = 0.0
    norm_constraint_mode: Literal["fisher", "fisher_scaled", "precond_grad", "precond_grad_scaled"] = "fisher_scaled"
    norm_constraint: float = 3e-3
    scale_nc_by_std_dev: bool = False
    min_clip_nc: float = 3.0
    max_clip_nc: float = 8.0
    damping: float = 1e-3
    l2_reg: float = 0.0
    damping_schedule: Union[
        ConstantLRSchedule,
        InverseLRScheduleConfig,
        NoamLRScheduleConfig,
        ExponentialLRSchedule,
    ] = ConstantLRSchedule()
    estimation_mode: Literal["fisher_gradients", "fisher_exact"] = "fisher_exact"
    register_generic: bool = False
    update_inverse_period: int = 1
    """Period of how often the fisher matrix is being updated (in batches). e.g. update_inverse_period==1 means that it is updated after every gradient step."""

    n_burn_in: int = 0
    min_damping: float = 1e-4
    curvature_ema: float = 0.95
    internal_optimizer: Union[_OptimizerConfigSGD, StandardOptimizerConfig] = _OptimizerConfigSGD()
    """Internal optimizer to use for applying the preconditioned gradients calculated by KFAC. Use SGD for 'pure' KFAC."""


class _OptimizerConfigKFACwithAdam(OptimizerConfigKFAC):
    name: Literal["kfac_adam"] = "kfac_adam"
    learning_rate: float = 1.0
    norm_constraint: float = 6e-5
    lr_schedule: Union[
        _InverseLRScheduleConfigForKFACwithAdam,
        InverseLRScheduleConfig,
        ConstantLRSchedule,
        NoamLRScheduleConfig,
    ] = _InverseLRScheduleConfigForKFACwithAdam()
    internal_optimizer: Union[_OptimizerConfigAdamForKFAC, StandardOptimizerConfig] = _OptimizerConfigAdamForKFAC()


class SRCGOptimizerConfig(ConfigBaseclass):
    name: Literal["srcg"] = "srcg"
    """Identifier of optimizer. Fixed."""

    learning_rate: float = 0.1
    lr_schedule: Union[
        InverseLRScheduleConfig,
        ConstantLRSchedule,
        NoamLRScheduleConfig,
        ExponentialLRSchedule,
    ] = InverseLRScheduleConfig()
    damping: float = 1e-3
    damping_schedule: Union[ConstantLRSchedule, InverseLRScheduleConfig, ExponentialLRSchedule] = ConstantLRSchedule()
    max_update_norm: float = 0.1

    internal_optimizer: Union[_OptimizerConfigSGD, StandardOptimizerConfig] = _OptimizerConfigSGD()
    center_gradients: bool = True
    preconditioner: Optional[Literal["variance"]] = None
    preconditioner_ema: float = 0.0
    preconditioner_batch_size: int = 64
    initial_guess: Literal["previous", "zero", "grad", "preconditioned_grad"] = "previous"
    maxiter: int = 100
    linearize_jvp: bool = True


class ForceEvaluationConfig(ConfigBaseclass):
    use: bool = True
    R_cut: float = 0.1
    R_core: float = 0.5
    use_antithetic_sampling: bool = False


class DensityEvaluationConfig(ConfigBaseclass):
    calculate_density: bool = True
    calculate_pair_density: bool = False
    r_min: List[float] = [0.0, 0.0, 0.0]
    r_max: List[float] = [1.0, 1.0, 1.0]
    n_grid: List[int] = [50, 50, 50]


class EvaluationConfig(ConfigBaseclass):
    opt_epochs: List[int] = []
    """List of optimization epochs at which to run an intermediate evaluation to assess wavefunction accuracy."""

    evaluate_final: bool = True
    """Whether to always run an evaluation at the end of the optimization."""

    mcmc: MCMCConfigEvaluation = MCMCConfigEvaluation()

    n_epochs: int = 500
    calculate_energies: bool = True
    forces: Optional[ForceEvaluationConfig] = None
    localization_metric: Literal[None, "min_abs", "all_abs", "all"] = "min_abs"
    """Controls detail of localization calculation for periodic systems.
    None: Don't calculate localization
    min_abs: minimal absolute value over spatial dimensions.
    all_abs: absolute value for each spatial dimension.
    all: real and imaginary components for each spatial dimension."""

    structure_factor_grid: Optional[List[int]] = None
    """Number of grid points per dimension for structure factor calculation. If None, no structure factor is calculated."""

    density: Optional[DensityEvaluationConfig] = None

    max_batch_size: int = 64

    forward_lap: bool = False
    """Whether to use the forward laplacian method from the flux package to calculate the kinetic energy"""


class DistortionConfig(ConfigBaseclass):
    max_age: int = 20
    """Number of steps before the geometry gets distoreted and rotated"""

    min_stiffness: float = 0.2

    distortion_energy: float = 0.005
    """Mean energy induced by distortions per degree of freedom in Hartree"""

    bias_towards_orig: float = 0.2
    """How much bias to apply, pushing the gometry towards the original geometry. 0.0 means no bias, 1.0 means always use original geometry"""

    min_distance_factor: float = 0.8
    """Minimum interatomic distance to accept, as a factor of the original interatomic distance"""

    reset_every_n_distortions: int = 30
    """How often to reset the distortion to the original geometry"""

    space_warp: Literal["nearest", "1/r4"] = "1/r4"
    """Weighting function, deciding how the electrons should be moved"""

    init_distortion_age: Literal["random", "zero"] = "random"


class SharedOptimizationConfig(ConfigBaseclass):
    use: bool = True

    shared_modules: Optional[List[str]] = None
    """What modules/ parts of the neural network should be shared during weight-sharing"""

    orbital_method: Literal["truncate"] = "truncate"

    scheduling_method: Literal["round_robin", "stddev", "weight", "var_per_el"] = "round_robin"
    """Method to vary between geometries during weight-sharing"""

    n_initial_round_robin_per_geom: int = 10

    max_age: Optional[int] = None
    """Maximal number of epochs a geometry can be not considered during weight-sharing. Preventing a few geometries from requiring all parameter updates for stddev scheduling"""

    distortion: Optional[DistortionConfig] = None


class CheckpointConfig(ConfigBaseclass):
    replace_every_n_epochs: int = 1000
    keep_every_n_epochs: int = 50_000
    additional_n_epochs: List[int] = []
    keep_epoch_0: bool = False
    log_only_zero_geom: bool = False


class OptimizationConfig(ConfigBaseclass):
    mcmc: MCMCConfigOptimization = MCMCConfigOptimization()

    optimizer: Union[
        OptimizerConfigKFAC,
        _OptimizerConfigKFACwithAdam,
        StandardOptimizerConfig,
        SRCGOptimizerConfig,
    ] = OptimizerConfigKFAC()
    """Which optimizer to use and its corresponding sub-options"""

    n_epochs: int = 60_000
    """Number of epochs for wavefunction optimization"""

    n_epochs_prev: int = 0  # if run is restart, this can be used to store number of previous epochs
    """Nr of epochs that this wavefunction has already been optimized. This can be used to store number of previous epochs after a restart"""

    use_batch_reweighting: bool = False
    """Reweight gradients for different samples with the changes in log(psi^2) between batches"""

    checkpoints: CheckpointConfig = CheckpointConfig()

    # checkpoints: List[int] = []
    # """List of epoch-numbers at which a checkpoint-file should be dumped, containing all model weights and MCMC-walkers to allow a restart or evaluation"""

    clipping: ClippingConfig = ClippingConfig()
    """Config for clipping the local energies in the loss function. Clipping significantly improves optimization stability."""

    shared_optimization: Optional[SharedOptimizationConfig] = None
    """Config for shared optimization of multiple wavefunctions using weight-sharing between them"""

    stop_on_nan: bool = True
    """Whether to abort a calculation once an optimization energies reaches nan or +/- inf"""

    params_ema_factor: float = 0.95
    """Factor to regulate the length of memory for trainable parameters"""

    max_batch_size: int = 64

    forward_lap: bool = False
    """Whether to use the forward laplacian method from the flux package to calculate the kinetic energy"""

    @property
    def n_epochs_total(self):
        return self.n_epochs_prev + self.n_epochs

    # @root_validator
    # def scale_lr_for_shared_modules(cls, values):
    #     if values['shared_optimization'] is None:
    #         return values
    #     shared_modules = values['shared_optimization'].shared_modules
    #     optimizer = values['optimizer']
    #
    #     if isinstance(optimizer, AdamScaledOptimizerConfig):
    #         if optimizer.scaled_modules is None:
    #             optimizer.scaled_modules = shared_modules
    #     if hasattr(optimizer, 'internal_optimizer'):
    #         internal_optimizer = optimizer.internal_optimizer
    #         if isinstance(internal_optimizer, AdamScaledOptimizerConfig):
    #             if internal_optimizer.scaled_modules is None:
    #                 internal_optimizer.scaled_modules = shared_modules
    #     return values


class RestartConfig(ConfigBaseclass):
    mode: Literal["restart"] = "restart"

    path: Optional[str] = None
    """Path to a previous calculation directory, containing a results.bz2 file"""

    path_phisnet: Optional[str] = None
    """Path to zip file containing a pretrained PhisNet model"""

    reuse_config: bool = True
    """Whether to re-use the configuration of the original run or whether to start from a new default config"""

    reuse_opt_state: bool = True
    """Reuse state of the optimizer, e.g. momentum, hessian estimates"""

    reuse_mcmc_state: bool = True
    """Whether to re-use the position of MCMC walkers. This may only make sense when calculating a wavefunction for the same (or very similar) geometry"""

    randomize_mcmc_rng: bool = False
    """Whether to change the state of the random-number-generator of the MCMC state. Does not move walkers, but will change their future random walk"""

    reuse_clipping_state: bool = True
    """Whehter to re-use the clipping state for optimization to clip local energy. This may only make sense when calculating a wavefunction for the same (or very similar) geometry"""

    reuse_trainable_params: bool = True
    """Whether to re-use the trainable weights of the neural networks. Use *reuse_modules* to specify only specific parts of the model to be re-used."""

    reuse_ema_trainable_params: bool = False
    """Whether to re-use the trainable weights estimated from a running average of optimization steps. (experimental setting)"""

    reuse_fixed_params: bool = True
    """Whether to re-use fixed (i.e. non-trainable) params"""

    reuse_modules: Optional[List[str]] = None
    """Model-specific list of strings, detailing which submodules should be re-used vs. re-initialized randomly. Names of modules to re-used is the same as the list of modules for weight-sharing-constraints."""

    continue_n_epochs: bool = True
    """Whether to increment n_epochs_prev in the config, to keep track of total epochs trained"""

    skip_burn_in: bool = True
    """Whether to skip burn-in steps for MCMC walkers"""

    skip_pretraining: bool = True
    """Whether to skip supervised pretraining of the wavefunction using HartreeFock"""

    ignore_extra_settings: bool = False
    """Whether to ignore extra config flags that are no longer compatible with the current config"""

    check_param_count: bool = True


class ReuseConfig(RestartConfig):
    """Do not introduce new fields here"""

    mode: Literal["reuse"] = "reuse"
    reuse_config: bool = False
    reuse_opt_state: bool = False
    reuse_mcmc_state: bool = False
    reuse_clipping_state: bool = False
    reuse_trainable_params: bool = True
    reuse_ema_trainable_params: bool = False
    reuse_fixed_params: bool = False
    reuse_modules: Optional[List[str]] = None
    continue_n_epochs: bool = False
    skip_burn_in: bool = False
    skip_pretraining: bool = False
    ignore_extra_settings: bool = False
    check_param_count: bool = True


class PeriodicConfig(ConfigBaseclass):
    lattice_prim: Optional[Sequence[Sequence[float]]] = None
    """Primitive cell lattice; 3x3 matrix. Each row of this matrix is one lattice vector, each column corresponds to a cartesian dimension."""

    supercell: Union[Sequence[int], int] = 1
    """Number of primitive cells in the simulation cell in each (lattice) direction."""

    truncation_limit: int = 100

    gamma_option: Literal["optimized", "min_rec"] = "optimized"

    k_twist: List[float] = [0.0, 0.0, 0.0]
    """Twist vector for (twisted) periodic boundary conditions. 
    The twist vector is given in units of the reciprocal lattice vectors of the supercell, e.g. [0.5, 0.0, 0.0] means that the phase os shifted by pi/L for a lattice constant L in the x-direction."""

    include_heg_background: bool = False

    lattice_const: Optional[float] = None

    lattice_type: Literal[None, "sc", "bcc", "fcc"] = None

    is_expanded: bool = False
    """Whether the properties of the PhysicalConfig describe a primitive cell (False) or whether they have already been expanded to a supercell (True)"""

    R_prim: Optional[Sequence[Sequence[float]]] = None

    Z_prim: Optional[Sequence[int]] = None

    charge_prim: Optional[int] = None

    spin_prim: Optional[int] = None

    @model_validator(mode="after")
    def supercell_int_to_list(self):
        if isinstance(self.supercell, int):
            self.supercell = [self.supercell] * 3
        return self

    @model_validator(mode="after")
    def lattice_from_predefined_lattice(self):
        if (self.lattice_prim is not None) or (self.lattice_type is None):
            return self
        if self.lattice_type == "sc":
            lattice = np.eye(3)
        elif self.lattice_type == "bcc":
            lattice = np.array([1, 1, -1], [1, 1, 1], [1, -1, 1]) / 2
        elif self.lattice_type == "fcc":
            lattice = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]]) / 2
        else:
            raise ValueError(f"Unknown lattice type: {self.lattice_type}")
        self.lattice_prim = (lattice * self.lattice_const).tolist()
        return self

    @property
    def lattice(self) -> np.ndarray:
        """Supercell lattice; 3x3 matrix. Each row of this matrix is one lattice vector, each column corresponds to a cartesian dimension."""
        return np.diag(self.supercell) @ np.array(self.lattice_prim)

    @property
    def n_prim_cells(self) -> int:
        """Number of primitive cells in the simulation cell"""
        return int(np.prod(self.supercell))

    def get_shifts(self):
        ordinals = np.stack(np.meshgrid(*[np.arange(n) for n in self.supercell]), axis=-1).reshape([-1, 3])
        return ordinals @ self.lattice_prim


class PhysicalConfigChange(ConfigBaseclass):
    name: Optional[str] = None
    comment: Optional[str] = None
    R: Optional[List[List[float]]] = None
    Z: Optional[List[int]] = None
    E_ref: Optional[float] = None
    E_ref_source: Optional[str] = None
    el_ion_mapping: Optional[List[int]] = None
    n_electrons: Optional[int] = None
    n_up: Optional[int] = None
    periodic: Optional[PeriodicConfig] = None
    weight_for_shared: Optional[float] = None


class PhysicalConfig(ConfigBaseclass):
    def get_basic_params(self):
        return self.n_electrons, self.n_up, np.array(self.R), np.array(self.Z)

    @staticmethod
    def _get_spin_from_hunds_rule(Z):
        n_orbitals = [
            1,
            1,
            3,
            1,
            3,
            1,
            5,
            3,
            1,
            5,
            3,
            1,
            7,
            5,
        ]  # 1s, 2s, 2p, 3s, 3p, 4s, 5d, ...
        n_electrons = Z
        n_up = 0
        n_dn = 0
        for n_in_orb in n_orbitals:
            n_up += min(n_in_orb, n_electrons)
            n_electrons -= min(n_in_orb, n_electrons)
            n_dn += min(n_in_orb, n_electrons)
            n_electrons -= min(n_in_orb, n_electrons)
            if n_electrons == 0:
                break
        return n_up - n_dn

    @staticmethod
    def _generate_el_ion_mapping(R, Z, n_el, n_up):
        # TODO: This method appears to work well for neutral systems (n_el == sum(Z)),
        # but appears to generate unbalenced distributions for charged molecules
        """Generates a list of electrons each mapped to one of the ions.

        Electrons are mapped, such that local spin is minimzed, by greedily looking for the most unbalanced position
        and placing and opposite spin electron on this site. Local spin is calculated as a weighted sum of all other ions.
        """
        R = np.array(R)
        n_ions = len(Z)
        dist = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
        weights = np.exp(-dist)
        n_per_ion = np.zeros([n_ions, 2], int)
        n_per_spin = [n_up, n_el - n_up]

        for n in range(n_el):
            # Calculate the spin of each ion (up - down) and calculate a weighted 'local' from these raw spins
            local_spin = weights @ (n_per_ion[:, 0] - n_per_ion[:, 1])
            spin_to_assign = np.concatenate([np.zeros(n_ions, dtype=int), np.ones(n_ions, dtype=int)])
            ion_to_assign = np.concatenate([np.arange(n_ions, dtype=int), np.arange(n_ions, dtype=int)])
            resulting_local_spins = []
            for s, ind_ion in zip(spin_to_assign, ion_to_assign):
                change_in_spin = np.zeros(n_ions)
                change_in_spin[ind_ion] = 1 if s == 0 else -1
                resulting_local_spins.append(local_spin + weights @ change_in_spin)

            # Sort them to find the most unbalanced ions
            ind_additon = np.argsort(np.max(np.abs(resulting_local_spins), axis=-1))
            for i in ind_additon:
                spin = spin_to_assign[i]
                ind_ion = ion_to_assign[i]
                if (sum(n_per_ion[ind_ion]) == Z[ind_ion]) or (n_per_spin[spin] == 0):
                    # Do not put more electrons of this spin if there are no more electrons left or the atom is full
                    continue
                n_per_ion[ind_ion, spin] += 1
                n_per_spin[spin] -= 1
                break

        # Convert to list of length electrons
        mapping = []
        for spin in range(2):
            for ind_ion, n in enumerate(n_per_ion[:, spin]):
                mapping = mapping + [ind_ion] * n
        return mapping

    name: Optional[str] = None
    """Name of the molecule to be calculated. If other physical parameters are not specified, this name will be used as a lookup-key to find default values."""

    R: Optional[List[List[float]]] = None
    """List of lists, specifiying the position of all ions. Outer index loops over ions, inder index loops over 3 coordinates XYZ."""

    Z: Optional[List[int]] = None
    """Nuclear charges per ion"""

    n_electrons: Optional[int] = None
    """Total number of electrons in the system"""

    n_up: Optional[int] = None
    """Number of spin-up electrons"""

    el_ion_mapping: Optional[List[int]] = None
    """Initial position of electrons. len(el_ion_mapping) == n_electrons. For each electron the list-entry specifies the index of the nucleus where the electron should be initialized.
    Note that the n_up first entries in this list correpsond to spin-up electrons and the remaining entries correspond to spin-down electrons"""

    E_ref: Optional[float] = None
    """Known ground-truth energy as reference for output. This value is not used in the actual calculation, but only to benchmark the achieved results."""

    E_ref_source: Optional[str] = None
    """Source of included reference energy"""

    comment: Optional[str] = None
    """Optional comment to keep track of molecules, geometries or experiments"""

    changes: Optional[List[PhysicalConfigChange]] = None
    """List of changes to be made to this physical-configuration to obtain different molecules. This can be used to calculate multiple geometries with similar settings, by specifying a base configuration (e.g. charge, spin, CASSCF-settings, etc) and only specifying the changes (e.g. in nuclear geometry) in the changes section."""

    periodic: Optional[PeriodicConfig] = None
    """Config to specify primitive cell and supercell in periodic boundary
    conditions."""

    weight_for_shared: Optional[float] = None
    """How important this specific geometry is for a shared optimization. 
    This can be used to weight geometries differently, e.g. to give more weight (and thus potentially more optimization steps) 
    to geometries that are more difficult to optimize or that are more relevant due to symmetry considerations."""

    @property
    def n_dn(self):
        return self.n_electrons - self.n_up

    @property
    def n_ions(self):
        return len(self.Z)

    @property
    def charge(self):
        return sum(self.Z) - self.n_electrons

    @property
    def spin(self):
        return self.n_up - self.n_dn

    def get_expanded_if_supercell(self):
        if (self.periodic is None) or (self.periodic.is_expanded):
            return self.model_copy()
        else:
            shifts = self.periodic.get_shifts()
            n_el_sc = int(np.round(self.n_electrons * self.periodic.n_prim_cells))

            if self.el_ion_mapping is None:
                el_ion_mapping = None
            else:
                el_ion_mapping_up = self.el_ion_mapping[: self.n_up]
                el_ion_mapping_dn = self.el_ion_mapping[self.n_up :]

                el_ion_mapping = []
                for i in range(self.periodic.n_prim_cells):
                    el_ion_mapping += [n + i * self.n_ions for n in el_ion_mapping_up]
                for i in range(self.periodic.n_prim_cells):
                    el_ion_mapping += [n + i * self.n_ions for n in el_ion_mapping_dn]
            periodic = self.periodic.model_copy()
            periodic.is_expanded = True
            periodic.R_prim = self.R
            periodic.Z_prim = self.Z
            periodic.charge_prim = self.charge
            periodic.spin_prim = self.spin

            return PhysicalConfig(
                name=self.name,
                R=np.concatenate([np.array(self.R) + shift for shift in shifts], axis=0),
                Z=np.tile(self.Z, self.periodic.n_prim_cells),
                n_electrons=n_el_sc,
                el_ion_mapping=el_ion_mapping,
                E_ref=self.E_ref,
                E_ref_source=self.E_ref_source,
                comment=self.comment,
                changes=copy.deepcopy(self.changes),
                periodic=periodic,
            )

    _PERIODIC_TABLE = {
        k: i + 1
        for i, k in enumerate(
            "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr".split(
                " "
            )
        )
    }

    _DEFAULT_MOLECULES = ruamel.yaml.YAML().load(pathlib.Path(__file__).parent.joinpath("molecules.yaml"))

    @model_validator(mode="after")
    def populate_physical_config_from_name(self):
        mol = dict()
        if self.name:
            if self.name in self._PERIODIC_TABLE:
                Z = self._PERIODIC_TABLE[self.name]
                mol = dict(Z=[Z], R=[[0.0, 0.0, 0.0]], spin=self._get_spin_from_hunds_rule(Z))
            elif self.name in self._DEFAULT_MOLECULES:
                mol = self._DEFAULT_MOLECULES[self.name]
        if self.Z is None:
            self.Z = mol.get("Z")
        if self.R is None:
            self.R = mol.get("R")
        if (self.n_electrons is None) and (self.Z is not None):
            charge = mol.get("charge", 0)
            self.n_electrons = sum(self.Z) - charge
        if (self.n_up is None) and (self.n_electrons is not None):
            spin = mol.get("spin", 0)
            self.n_up = (self.n_electrons + spin + 1) // 2  # if there is an extra electrons, assign them to up
        if self.el_ion_mapping is None:
            if "el_ion_mapping" in mol:
                self.el_ion_mapping = mol["el_ion_mapping"]
            elif (
                (self.R is not None)
                and (self.Z is not None)
                and (self.n_electrons is not None)
                and (self.n_up is not None)
            ):
                self.el_ion_mapping = self._generate_el_ion_mapping(self.R, self.Z, self.n_electrons, self.n_up)
        if self.E_ref is None:
            self.E_ref = mol.get("E_ref")
        if self.E_ref_source is None:
            self.E_ref_source = mol.get("E_ref_source")
        return self

    def create_geometry_list(self, phys_changes):
        if phys_changes is None:
            return [self]
        ret = []
        for change in phys_changes:
            change = build_flattend_dict(change)
            p = PhysicalConfig.update_configdict_and_validate(self.model_dump(), change)[1]
            p.changes = None
            ret.append(p)
        return ret


class LoggerBaseConfig(ConfigBaseclass):
    n_skip_epochs: int = 0


class WandBConfig(LoggerBaseConfig):
    project: Optional[str] = "default"
    entity: Optional[str] = None
    id: Optional[str] = None
    use_id: bool = False
    blacklist: List[str] = [
        "update_norm(",
        "precon_grad_norm(",
        "grad_norm(",
        "param_norm(",
    ]


class BasicLoggerConfig(LoggerBaseConfig):
    log_to_stdout: bool = True
    log_level: Union[int, Literal["CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG"]] = "DEBUG"
    sublog_levels: Dict[str, Literal["CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG"]] = {
        "absl": "INFO",
        "h5py": "INFO",
        "jax.interpreters.pxla": "INFO",
        "jax._src.dispatch": "INFO",
        "jax._src.interpreters.pxla": "INFO",
        "jax._src.xla_bridge": "INFO",
        "jax._src.cache_key": "INFO",
        "jax._src.compiler": "INFO",
        "jax._src.compilation_cache": "INFO",
    }
    n_skip_epochs: int = 9
    fname: Optional[str] = "log.out"
    blacklist: List[str] = [
        "update_norm(",
        "precon_grad_norm(",
        "grad_norm(",
        "param_norm(",
    ]


class PickleLoggerConfig(LoggerBaseConfig):
    fname: str = "results.bz2"


class LoggingConfig(ConfigBaseclass):
    tags: List[str] = []
    """List of tags to help mark/identify runs"""

    log_opt_state: bool = True
    """Flag whether to log the full state of the optimizer. Note that this can produce very large output-files, in particular when logging the hessian of 2nd-order-optimizers"""

    basic: Optional[BasicLoggerConfig] = BasicLoggerConfig()
    """Config for basic logger, which will produce human-readable log-files using the python built-in logging module"""

    wandb: Optional[WandBConfig] = None
    """Config for logging to Weights&Biases"""

    pickle: Optional[PickleLoggerConfig] = PickleLoggerConfig()
    """Config for logging to binary files, which contain machine-readable pickle files containg all configs, metrics and model weights"""


class DispatchConfig(ConfigBaseclass):
    system: Literal[
        "local",
        "local_background",
        "local_slurm",
        "vsc3",
        "vsc4",
        "dgx",
        "leonardo",
        "auto",
        "juwels",
        "vega",
        "baskerville",
    ] = "auto"
    """Which compute-cluster to use for this experiment. 'auto' detects whether the code is running on a known compute-cluster and selects the corresponding profile, or otherwise defaults to local execution"""

    queue: Literal[
        "default",
        "jupyter",
        "a40",
        "a100",
        "zen3_0512_a100x2",
        "zen2_0256_a40x2",
        "booster",
    ] = "default"
    """SLURM queue to use for job submission on compute clusters. Not relevant for local execution"""

    qos: Literal[
        None,
        "gpus2",
        "gpus4",
        "gpus6",
        "boost_qos_dbg",
        "boost_qos_bprod",
        "boost_qos_lprod",
        "normal",
    ] = None
    """Quality of Service for SLURM. None assigns the default value for the corresponding partition"""

    time: str = "3day"
    """Maximum job run-time to use for job submission on compute clusters. Not relevant for local execution"""

    conda_env: str = "jax"
    """Conda environment to select in SLURM-job-script. Not relevant for local execution"""

    memory: Optional[int] = None
    """Required memory in GB. Only relevant for SLURM."""

    split_opt: Optional[List[int]] = None

    eval_epochs: Optional[int] = None


class ComputationConfig(ConfigBaseclass):
    use_gpu: bool = True
    """deprecated"""
    require_gpu: bool = False
    n_local_devices: Optional[int] = None
    n_nodes: int = 1
    rng_seed: Optional[int] = None
    force_device_count: bool = False
    disable_jit: bool = False
    float_precision: Literal["float32", "float64"] = "float32"
    disable_tensor_cores: bool = True
    use_profiler: bool = False


class PreTrainingConfig(ConfigBaseclass):
    use: bool = True

    mcmc: MCMCConfigPreTrain = MCMCConfigPreTrain()

    n_epochs: int = 1000
    """Number of pre-training steps to fit DL-wavefunction to a baseline calculation as e.g. Hartree Fock"""

    optimizer: Union[_OptimizerConfigAdamForPretraining, StandardOptimizerConfig] = _OptimizerConfigAdamForPretraining()
    """Optimizer used for pre-training only. Setting will not affect standard optimization"""

    use_only_leading_determinant: bool = True
    """Whether to use only Hartree Fock determinant for pre-training or for each DL-wavefunction determinant a different CAS determinant"""

    sampling_density: Literal["reference", "model"] = "model"
    """Which wavefunction to use for """

    checkpoints: CheckpointConfig = CheckpointConfig()

    off_diagonal_mode: Literal["reference", "ignore", "exponential"] = "reference"
    """How to treat off-diagonal blocks of full determinants during pretraining. 
    reference: learn Hartree-Fock off-diagonals (ie. 0)
    ignore: ignore the off-diagonal orbitals in the pretraining loss, ie. do not train them
    exponential: Train the off-diagonal orbitals to be a sum of exponentials centered on each atom
    """

    off_diagonal_exponent: float = 1.0
    """Exponent to use for off-diagonal orbitals in 'exponential' off-diagonal mode"""

    off_diagonal_scale: float = 1.0
    """Scale to use for off-diagonal orbitals in 'exponential' off-diagonal mode"""

    use_distortions_for_shared_opt: bool = True


# TODO: Generate a simple overview graphics of the main config options; use the schema to autogenerate?
class Configuration(ConfigBaseclass):
    """Root configuration for DeepErwin"""

    physical: Union[None, PhysicalConfig, str, List[str]] = None
    """The physical system/molecule being calculated"""

    pre_training: Optional[PreTrainingConfig] = PreTrainingConfig()
    """Supervised pre-training of the orbitals to fit the baseline orbitals."""

    optimization: OptimizationConfig = OptimizationConfig()
    """The wavefunction optimization"""

    evaluation: EvaluationConfig = EvaluationConfig()
    """The evaluation of the wavefunction (after optimization)"""

    model: Union[
        ModelConfigDeepErwin4,
        ModelConfigFermiNet,
        ModelConfigTransformer,
        ModelConfigMoon,
    ] = ModelConfigDeepErwin4()
    """The actual wavefunction model mapping electron coordinates to psi"""

    baseline: BaselineConfigType = HartreeFockConfig()
    """A conventional baseline calculation, e.g. Hartree-Fock, to use as input to some wavefunction models (e.g. TAOs), as well as for pre-training"""

    logging: LoggingConfig = LoggingConfig()
    """The output of the code, e.g. logging to files, or online-services"""

    computation: ComputationConfig = ComputationConfig()
    """Options regarding computational details such as GPU usage and float precision"""

    dispatch: DispatchConfig = DispatchConfig()
    """Options regarding where the code is being run, i.e. locally vs asynchronysly on a compute-cluster"""

    reuse: Optional[Union[RestartConfig, ReuseConfig]] = None
    """Reuse information from a previosu runt to smartly initialize weights or MCMC walkers."""

    comment: Optional[str] = None
    """Optional coment to keep track of experiments"""

    experiment_name: Optional[str] = "deeperwin_experiment"
    """Experiment name to keep track of experiments"""

    @model_validator(mode="after")
    def experiment_has_name(self):
        if self.experiment_name is None:
            if self.physical is None:
                self.experiment_name = "exp"
            else:
                self.experiment_name = self.physical.name
        return self

    @classmethod
    def load_configuration_file(cls, config_file: str) -> Tuple[Any, "Configuration"]:
        with open(config_file, "r") as f:
            raw_config = yaml.YAML().load(f)
        config: Configuration = cls.model_validate(raw_config)
        return raw_config, config


class CheckpointConfigPhisNet(ConfigBaseclass):
    checkpoint_metric: Literal[
        "test_mo_occ_cosine_dist",
        "test_loss_mo",
        "test_loss",
        "test_mae_energy",
        "test_mae_forces",
    ] = "test_loss"
    every_n_epochs: int = 50


class PhisNetLossWeightsConfig(ConfigBaseclass):
    H: float = 1.0
    S: float = 1.0
    H_core: float = 1.0
    rho: float = 1.0
    energy: float = 1.0
    forces: float = 1.0


class PhisNetTrainingConfiguration(ConfigBaseclass):
    experiment_name: str = "phisnet_experiment"
    comment: str = ""
    model: PhisNetModelConfig = PhisNetModelConfig()
    optimizer: StandardOptimizerConfig = StandardOptimizerConfig()
    data_path: str
    basis_set: str = "STO-6G"
    n_epochs: int = 100
    batch_size: int = 20
    trainingset_size: Union[float, int] = 0.8
    validate_small_every_n_epochs: int = 1
    validate_full_every_n_epochs: int = 20
    lr_schedule_patience: int = 10
    logging: LoggingConfig = LoggingConfig()
    checkpoint: CheckpointConfigPhisNet = CheckpointConfigPhisNet()
    dispatch: DispatchConfig = DispatchConfig()
    computation: ComputationConfig = ComputationConfig()
    max_grad_norm: Optional[float] = 1.0
    eps_roothaan: float = 1e-6
    eps_roothaan_mode: Literal["zero", "sqrt_eps", "tikhonov"] = "zero"
    loss_weights: PhisNetLossWeightsConfig = PhisNetLossWeightsConfig()
    load_checkpoint: Optional[str] = None
    flip_forces: bool = False
    copy_data_to_local: str = ""
    n_atoms_max: int = 0


def get_with_flattened_key(config: Configuration, key):
    if "." not in key:
        return getattr(config, key)
    else:
        tokens = key.split(".")
        parent_key = tokens[0]
        child_key = ".".join(tokens[1:])
        return get_with_flattened_key(getattr(config, parent_key), child_key)


def set_with_flattened_key(config_dict, key, value):
    if not isinstance(value, dict) and "." not in key:
        config_dict[key] = value
    elif isinstance(value, dict):
        if key not in config_dict or config_dict[key] is None:
            config_dict[key] = {}
        for key_child, value_child in value.items():
            set_with_flattened_key(config_dict[key], key_child, value_child)
    else:
        tokens = key.split(".")
        parent_key = tokens[0]
        child_key = ".".join(tokens[1:])
        if (parent_key not in config_dict) or (config_dict[parent_key] is None):
            config_dict[parent_key] = {}
        set_with_flattened_key(config_dict[parent_key], child_key, value)
    return config_dict


def build_nested_dict(flattened_dict):
    root_dict = {}
    for k, value in flattened_dict.items():
        key_tokens = k.split(".")
        d = root_dict
        for i, key in enumerate(key_tokens):
            if i == len(key_tokens) - 1:
                d[key] = value  # reached leaf => store value
                continue
            if key not in d:
                d[key] = {}
            d = d[key]  # traverse further
    return root_dict


def build_flattend_dict(nested_dict):
    flattened_dict = dict()
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            flattened_dict.update({key + "." + k: v for k, v in build_flattend_dict(value).items()})
        else:
            flattened_dict[key] = value
    return flattened_dict


def to_prettified_yaml(data):
    def _convert_to_inline_block(item):
        s = ruamel.yaml.comments.CommentedSeq(item)
        s.fa.set_flow_style()
        return s

    def _convert_data(d):
        if isinstance(d, dict):
            return {k: _convert_data(v) for k, v in d.items()}
        elif isinstance(d, list):
            return _convert_to_inline_block(d)
        else:
            return d

    return _convert_data(data)


def build_physical_configs_from_changes(raw_physical_config_dict, parse=True):
    changes = raw_physical_config_dict.get("changes")
    output_configs = []
    if (changes is None) or len(changes) == 0:
        output_configs = [raw_physical_config_dict]
    else:
        for idx, change_dict in enumerate(changes):
            config_dict = copy.deepcopy(raw_physical_config_dict)
            for k, v in build_flattend_dict(change_dict).items():
                set_with_flattened_key(config_dict, k, v)
            if not config_dict.get("comment"):
                config_dict["comment"] = str(idx)
            config_dict["changes"] = None
            output_configs.append(config_dict)
    if parse:
        output_configs = [PhysicalConfig.model_validate(c) for c in output_configs]
    return output_configs


if __name__ == "__main__":
    import pathlib
    import json

    c = Configuration.model_validate(dict(physical=dict(name="LiH"), model=dict(name="dpe4")))
    p = pathlib.Path(__file__).parent.joinpath("../../sample_configs/config_schema.json").resolve()
    print(f"Updating schema: {str(p)}")
    with open(p, "w") as f:
        f.write(json.dumps(c.model_json_schema()))


# %%
