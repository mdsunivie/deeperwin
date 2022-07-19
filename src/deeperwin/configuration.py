"""
DeepErwin hyperparameter and configuration management.
"""
import copy
from typing import Union, Literal, Optional, List, Tuple, Iterable, Any
import ruamel.yaml.comments
from pydantic import BaseModel, validator, root_validator
import numpy as np
import pathlib

class ConfigBaseclass(BaseModel):
    """Base class for all config models"""

    def save(self, file):
        data = to_prettified_yaml(self.dict())
        if isinstance(file, str):
            with open(file, 'w') as f:
                ruamel.yaml.YAML().dump(data, f)
        else:
            ruamel.yaml.YAML().dump(data, file)

    @classmethod
    def load(cls, file) -> 'ConfigBaseclass':
        if isinstance(file, str):
            with open(file) as f:
                data = ruamel.yaml.YAML().load(f)
        else:
            data = ruamel.yaml.YAML().load(file)
        return cls.parse_obj(data)

    def as_flattened_dict(self):
        output_dict = {}
        for label in self.dict():
            subconfig = getattr(self, label)
            if hasattr(subconfig, 'as_flattened_dict'):
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

        config = cls().parse_obj(nested_dict)
        return config

    @root_validator(allow_reuse=True)
    def disable_unused_children(cls, values):
        for element in values.keys():
            if hasattr(values[element], 'use'):
                if not values[element].use:
                    values[element] = None
        return values

    @classmethod
    def update_configdict_and_validate(cls, config_dict: dict, config_changes: Iterable[Tuple[str, Any]]):
        config_dict = copy.deepcopy(config_dict)
        # First loop: Build an updated dictionary
        for key, value in config_changes.items():
            set_with_flattened_key(config_dict, key, value)
        # Parse the config dict and validate that all parameters are valid
        parsed_config = cls.parse_obj(config_dict)

        # Second loop: Update the values using the parsed values to get correct type; Not strictly necessary,
        # but yields nicer looking input-config files
        for key, value in config_changes.items():
            set_with_flattened_key(config_dict, key, get_with_flattened_key(parsed_config, key))
        return config_dict, parsed_config

    class Config:
        """The name 'Config' here is a pydantic term used to config the behaviour of the dataclass.
        It's not a config for the variational monte carlo code"""
        extra = "forbid"


class InitializationConfig(ConfigBaseclass):
    bias_scale: float = 0.0

    weight_scale: Literal["glorot", "glorot-input"] = "glorot"

    weight_distribution: Literal["normal", "uniform"] = "uniform"


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

    @validator("n_iterations", always=True)
    def _set_n_iterations(cls, n, values):
        if n is None:
            try:
                n = len(values['n_hidden_one_el'])
            except TypeError:
                raise ValueError(f"n_hidden_one_el must be a list of net-widths if n_iterations is not specified")
        return n

    @root_validator
    def _set_n_hidden(cls, values):
        def _make_list(key, len):
            if isinstance(values[key], int):
                values[key] = [values[key]] * len
        _make_list('n_hidden_one_el', values['n_iterations'])
        _make_list('n_hidden_two_el', values['n_iterations']-1)
        _make_list('n_hidden_el_ions', values['n_iterations'] - 1)
        if len(values['n_hidden_one_el']) != (len(values['n_hidden_two_el']) + 1):
            raise ValueError(f"Number of layers for 1-el-stream must be one more than nr of layers in 2-el-stream")
        return values


class EmbeddingConfigDeepErwin4(EmbeddingConfigFermiNet):
    """DO NOT INTRODUCE NEW FIELDS HERE. This class is only used to provide alternative defaults"""
    name: Literal["dpe4"] = "dpe4"
    n_iterations: int = 4
    n_hidden_one_el: Union[List[int], int] = 256
    n_hidden_two_el: Union[List[int], int] = 32
    n_hidden_el_ions: Union[List[int], int] = 32
    use_el_ion_stream: bool = True
    use_h_two_same_diff = True
    emb_dim = 32
    use_w_mapping = True
    use_schnet_features = True
    sum_schnet_features = False
    use_average_h_one = True
    use_average_h_two = False
    use_h_one = True
    use_linear_out = False


class EmbeddingConfigDeepErwin1(ConfigBaseclass):
    name: Literal["dpe1"] = "dpe1"
    """Identifier for this model-part. Fixed."""

    embedding_dim: int = 64
    """Dimensionality of embedding vectors, including intermediate network widths."""

    n_iterations: int = 2
    """Number of embedding iterations for SchNet"""

    n_hidden_w: List[int] = [40, 40]
    """Number of hidden neurons per layer of the w-Network within SchNet. The w-network takes pairwise features between particles as inputs and calculates the weight of each embedding dimension as output. If not specified the width/depth as specified in net_width/net_depth is used"""

    deep_w = False
    """Whether to process inter-particle features further or to use for every SchNet iteration the inter-particle distances"""

    n_hidden_h: List[int] = [40, 40]
    """Number of hidden neurons per layer of the h-Network within SchNet. The h-network takes the embedding of the previous iteration as input and computes a new embedding per particle, which is then weighted using the w-networks. If not specified the width/depth as specified in net_width/net_depth is used"""

    n_hidden_g: List[int] = [40]
    """Number of hidden neurons per layer of the g-Network within SchNet. The g-network applies a final non-linear transformation at the end of each SchNet round. If not specified the width as specified in net_width and depth = net_depth - 1 is used"""

    use_linear_layer_w = True

    use_res_net: bool = False
    """Whether to use residuals for the networks"""

    one_el_input: bool = False


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


class CASSCFConfig(ConfigBaseclass):
    name: Literal["casscf"] = "casscf"
    """Identifier of the baseline calculation. Fixed."""

    basis_set: str = "6-311G"
    """Basis set to use for the Hartree-Fock / CASSCF calculation. See the documentation of pySCF for all available basis-sets."""

    cusps: CuspCorrectionConfig = CuspCorrectionConfig()
    """Settings for the correction of wavefunction cusps when particles come close to each other"""

    only_hf: bool = False
    """Only as HF orbitals as baseline method"""


class InputFeatureConfig(ConfigBaseclass):
    use_rbf_features: bool
    """Whether to build distance features using gaussian functions of the distances"""

    n_rbf_features: int
    """Number of radial basis functions to use as pairwise fature vector"""

    use_distance_features: bool
    """Whether to include the absolute value of the distance (and powers of this distance) as input feature"""

    use_el_ion_differences: bool
    """Whether to include electron-ion differences as input features"""

    use_el_el_differences: bool
    """Whether to include electron-electron differences as input features"""

    use_el_el_spin: bool
    """Whether to include additional information about the spin configurations of the elctrons"""

    use_local_coordinates: bool
    """Whether to construct local coordinate systems by diagonalizing the density matrix, leading to equivariant difference featres"""

    n_one_el_features: int
    """Generate one-electron features as a sum of el-ion features"""

    n_ion_features: int
    """Embed nuclear charges into vectors of length n_ion_faetures"""

    concatenate_el_ion_features: bool
    """Whether to use concatenated electron-ion features as initialization for h_one; breaks equivariance"""

    n_hidden_one_el_features: List[int]
    """Number of hidden nodes for generation of one-electron-features from el-ion-features"""

    eps_dist_feat: float
    """Epsilon to be used when calculating pairwise features with negative exponent n < 0. Feature vector is calculated as :math:`\\frac{1}{dist^{-n} + eps}`"""

    distance_feature_powers: List[int]
    """To calculate the embeddings, pairwise distances between particles are converted into input feature vectors and then fed into the embedding network. 
    These input feature vectors partially consist of radial-basis-functions and monomials of the distance (i.e. r^n). This list specifies which exponents should be used to create the distance monomials.
    Note, that using powers of -1 or +1 can lead to cusps in the input features, which may or may not be desirable.
    """


class InputFeatureConfigDPE1(InputFeatureConfig):
    """DO NOT INTRODUCE NEW FIELDS HERE. This class is only used to provide alternative defaults"""
    name: Literal["dpe1"] = "dpe1"
    use_rbf_features: bool = True
    n_rbf_features: int = 32
    use_distance_features: bool = True
    distance_feature_powers: List[int] = [-1]
    eps_dist_feat:float = 1e-2
    use_el_ion_differences: bool = False
    use_el_el_differences: bool = False
    n_one_el_features:int = 0
    n_hidden_one_el_features: List[int] = []
    concatenate_el_ion_features: bool = False
    use_el_el_spin: bool = False
    use_local_coordinates: bool = False
    n_ion_features = 0


class InputFeatureConfigFermiNet(InputFeatureConfig):
    """DO NOT INTRODUCE NEW FIELDS HERE. This class is only used to provide alternative defaults"""
    name: Literal["ferminet"] = "ferminet"
    use_rbf_features: bool = False
    n_rbf_features: int = 0
    use_distance_features: bool = True
    eps_dist_feat:float = 1e-2
    distance_feature_powers: List[int] = [1]
    use_el_ion_differences: bool = True
    use_el_el_differences: bool = True
    n_one_el_features: int = 0
    n_hidden_one_el_features: List[int] = []
    concatenate_el_ion_features: bool = True
    use_el_el_spin: bool = False
    use_local_coordinates: bool = False
    n_ion_features = 32

class InputFeatureConfigDPE4(InputFeatureConfigFermiNet):
    """DO NOT INTRODUCE NEW FIELDS HERE. This class is only used to provide alternative defaults"""
    name: Literal["dpe4"] = "dpe4"
    use_local_coordinates: bool = True
    use_el_el_differences: bool = False


class EnvelopeOrbitalsConfig(ConfigBaseclass):
    envelope_type: Literal["isotropic_exp"] = "isotropic_exp"
    """Which enveolope to use for the backflow-add term"""

    n_hidden: List[int] = []
    """List of ints, specifying the number of hidden units per layer in the backflow-factor-network."""

    use_bias: bool = False
    """Enable / disable bias terms for the final layer of the backflow-add network"""

    initialization: Literal["constant", "analytical"] = "constant"
    """Use a least-squares fit to initialized the scale- and decay-parameters for the envelopes instead of constant intialization"""



class BaselineOrbitalsConfig(ConfigBaseclass):
    baseline: CASSCFConfig = CASSCFConfig()
    """Config-options for the underlying baseline calculation, typically a Complete-Active-Space Self-Consistent-Field (CASSCF) Calculation"""

    use_trainable_scales: bool = True
    """Scale the network outputs by a trainable float. Clever initialization of this scale can speed-up training, but in principle also introduces redundant parameters."""

    use_bf_shift: bool = True
    """Enable the backflow-shift, i.e. shift all electron coordinates by a neural network before calculating the CASSCF orbitals"""

    use_trainable_shift_decay_radius: bool = True
    """Backflow-shift decays to 0 close the the nuclei to ensure correct cusp conditions. This flag specifies whether the radius in which the shift decays is a trainable parameter or fixed."""

    register_scale = False
    """Whether to register the parameterst that scale the output of the backflow networks as parameters for KFAC optimization"""

    n_hidden_bf_factor: List[int] = [40, 20, 20]
    """List of ints, specifying the number of hidden units per layer in the backflow-factor-network. If not provided, the width and depth set by *net_width* and *net_depth* are used."""

    n_hidden_bf_shift: List[int] = [40, 40]
    """List of ints, specifying the number of hidden units per layer in the backflow-shift-network. If not provided, the width and depth set by *net_width* and *net_depth* are used."""

    use_bf_factor: bool = True
    """Enable the backflow-factor, i.e. multiply the CASSCF orbitals with the output of a neural network"""

    use_bf_factor_bias: bool = True
    """Enable / disable bias terms for the final layer of the backflow-factor network"""

    bf_factor_constant_bias: float = 1.0
    """Bias to apply to the output of the backflow factor, i.e. phi = phi_HF x (backflow + bias). If bias is set to 1.0, NN-output of zero, yields to original HF-orbitals"""

    output_shift: Literal[1, 3] = 1
    """Dimensionality of the output for the backflow-shift network. Can either be scalar or 3-dimensional. Note that only a scalar output ensures rotational equivariance."""


class OrbitalsConfig(ConfigBaseclass):
    baseline_orbitals: Optional[BaselineOrbitalsConfig] = None
    """Orbitals containing a baked-in baselin model (e.g. Hartree-Fock or CASSCF) which gets modified by 2 neural networks: backflow shifts and backflow factors"""

    envelope_orbitals: Optional[EnvelopeOrbitalsConfig] = EnvelopeOrbitalsConfig()
    """Obritals containing only a simple, parametrized envelope, multiplied by a neural network"""

    n_determinants: int = 32
    """Number of determinants in the wavefunction model"""

    use_full_det: bool = True
    """Whether to use full n_el x n_el determinants, or whether to assume a block diagonal structure, 
    represented by det(n_up x n_up)*det(n_dn x n_dn)"""


class OrbitalsConfigFermiNet(OrbitalsConfig):
    baseline_orbitals: Optional[BaselineOrbitalsConfig] = None
    envelope_orbitals: Optional[EnvelopeOrbitalsConfig] = EnvelopeOrbitalsConfig()
    n_determinants = 32
    use_full_det = True

class OrbitalsConfigDPE1(OrbitalsConfig):
    baseline_orbitals: Optional[BaselineOrbitalsConfig] = BaselineOrbitalsConfig()
    envelope_orbitals: Optional[EnvelopeOrbitalsConfig] = None
    n_determinants = 20
    use_full_det = False


class JastrowConfig(ConfigBaseclass):
    use: bool = True

    n_hidden: List[int] = [40, 40]
    """List of ints, specifying the number of hidden units per layer in the jastrow-network. If not provided, the width and depth set by *net_width* and *net_depth* are used."""

    differentiate_spins: bool = False
    """Use separate functions for J(spin_up) and J(spin_down)"""


class MLPConfig(ConfigBaseclass):
    activation: Literal["tanh", "silu", "elu", "relu"] = "tanh"

    init_bias_scale: float = 0.0

    init_weights_scale: Literal["fan_in", "fan_out", "fan_avg"] = "fan_avg"

    init_weights_distribution: Literal["normal", "truncated_normal", "uniform"] = "uniform"


class ModelConfig(ConfigBaseclass):
    """Configuration for the primary wavefunction model, which maps electron coordinate to psi"""

    features: InputFeatureConfig
    """Config-options for mapping raw inputs (r,R,Z) to some symmetrized input features"""

    embedding: Union[EmbeddingConfigDeepErwin4, EmbeddingConfigFermiNet, EmbeddingConfigDeepErwin1, None]
    """Config-options for mapping symmetrized input features to high-dimensional embeddings of electrons"""

    orbitals: OrbitalsConfig
    """Config-options for computing orbitals from embedding vectors of electrons"""

    mlp: MLPConfig = MLPConfig()
    """How to build multi-layer-perceptrons: Activation and how to initialize weights and biases"""

    jastrow: Optional[JastrowConfig]
    """Enable the jastrow-factor, i.e. multiple the total wavefunction by the output of a global neural network"""

    use_el_el_cusp_correction: bool
    """Explicit, additive el-el-cusp correction"""


class ModelConfigDeepErwin1(ModelConfig):
    """DO NOT INTRODUCE NEW FIELDS HERE. This class is only used to provide alternative defaults"""
    name: Literal["dpe1"] = "dpe1"
    features: Union[InputFeatureConfigDPE1, InputFeatureConfigFermiNet] = InputFeatureConfigDPE1()
    embedding: Union[EmbeddingConfigDeepErwin1, EmbeddingConfigDeepErwin4, EmbeddingConfigFermiNet, None] = EmbeddingConfigDeepErwin1()
    orbitals: OrbitalsConfigDPE1 = OrbitalsConfigDPE1()
    jastrow: Optional[JastrowConfig] = JastrowConfig()
    use_el_el_cusp_correction = True

class ModelConfigDeepErwin4(ModelConfig):
    """DO NOT INTRODUCE NEW FIELDS HERE. This class is only used to provide alternative defaults"""
    name: Literal["dpe4"] = "dpe4"
    features: Union[InputFeatureConfigDPE4, InputFeatureConfigFermiNet, InputFeatureConfigDPE1] = InputFeatureConfigDPE4()
    embedding: Union[EmbeddingConfigDeepErwin4, EmbeddingConfigFermiNet, EmbeddingConfigDeepErwin1, None] = EmbeddingConfigDeepErwin4()
    orbitals: OrbitalsConfigFermiNet = OrbitalsConfigFermiNet()
    jastrow: Optional[JastrowConfig] = None
    use_el_el_cusp_correction: bool = False

class ModelConfigFermiNet(ModelConfig):
    """DO NOT INTRODUCE NEW FIELDS HERE. This class is only used to provide alternative defaults"""
    name: Literal["ferminet"] = "ferminet"
    features: Union[InputFeatureConfigFermiNet, InputFeatureConfigDPE1] = InputFeatureConfigFermiNet()
    embedding: Union[EmbeddingConfigFermiNet, EmbeddingConfigDeepErwin4, EmbeddingConfigDeepErwin1, None] = EmbeddingConfigFermiNet()
    orbitals: OrbitalsConfigFermiNet = OrbitalsConfigFermiNet()
    jastrow: Optional[JastrowConfig] = None
    use_el_el_cusp_correction = False

EmbeddingConfigType = Union[EmbeddingConfigDeepErwin4, EmbeddingConfigFermiNet, EmbeddingConfigDeepErwin1, None]

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

    target_acceptance_rate: float = 0.5
    """Acceptance-rate that the MCMC-runner is trying to achieve by modifying the stepsize"""

    min_stepsize_scale: float = 1e-2
    """Minimum stepsize. For spatially adaptive stepsize-schemes this only defines a factor which may be modified by the adaptive scheme"""


    max_stepsize_scale: float = 1.0
    """Maximum stepsize. For spatially adaptive stepsize-schemes this only defines a factor which may be modified by the adaptive scheme"""

    proposal: Union[MCMCSimpleProposalConfig, LocalStepsizeProposalConfig, MCMCLangevinProposalConfig] = MCMCSimpleProposalConfig()
    """Type of proposal function to use for MCMC steps"""

class MCMCConfigPreTrain(MCMCConfig):
    n_inter_steps = 1
    n_burn_in = 0
    stepsize_update_interval = 1000
    max_age = 20

class MCMCConfigOptimization(MCMCConfig):
    n_inter_steps = 20
    n_burn_in = 1000
    stepsize_update_interval = 100
    max_age = 20

class MCMCConfigEvaluation(MCMCConfig):
    n_inter_steps = 20
    n_burn_in = 500
    stepsize_update_interval = 100
    max_age = 100


class ClippingConfig(ConfigBaseclass):
    name: Literal["hard", "tanh"] = "tanh"
    width_metric: Literal["std", "mae"] = "std"
    center: Literal["mean", "median"] = "mean"
    from_previous_step: bool = True
    clip_by: float = 5.0


class ConstantLRSchedule(ConfigBaseclass):
    name: Literal["fixed"] = "fixed"


class InverseLRScheduleConfig(ConfigBaseclass):
    name: Literal["inverse"] = "inverse"
    decay_time: float = 6000.0

class _InverseLRScheduleConfigForKFACwithAdam(InverseLRScheduleConfig):
    decay_time: float = 1000


class StandardOptimizerConfig(ConfigBaseclass):
    name: Literal["adam", "rmsprop_momentum", "sgd"] = "adam"  # add others optimizers that don't need further configs here

    learning_rate: float = 1e-3

    lr_schedule: Union[InverseLRScheduleConfig, ConstantLRSchedule] = InverseLRScheduleConfig()
    """Schedule for the learning rate decay"""

    scaled_modules: Optional[List[str]] = None
    """List of parameters for which the learning rate is being scaled."""

    scale_lr = 1.0
    """Factor which to apply to the learning rates of specified modules"""

class _OptimizerConfigAdamForPretraining(StandardOptimizerConfig):
    name: Literal["adam"] = "adam"
    learning_rate: float = 3e-4
    lr_schedule : Union[ConstantLRSchedule, InverseLRScheduleConfig] = ConstantLRSchedule()

class _OptimizerConfigSGD(StandardOptimizerConfig):
    name: Literal["adam", "rmsprop_momentum", "sgd"] = "sgd"
    learning_rate = 1.0
    lr_schedule: Union[InverseLRScheduleConfig, ConstantLRSchedule] = ConstantLRSchedule()


class _OptimizerConfigAdamForKFAC(StandardOptimizerConfig):
    name: Literal["adam", "rmsprop_momentum", "sgd"] = "adam"
    learning_rate = 2e-3
    lr_schedule: Union[_InverseLRScheduleConfigForKFACwithAdam, InverseLRScheduleConfig, ConstantLRSchedule] = _InverseLRScheduleConfigForKFACwithAdam()


class OptimizerConfigKFAC(ConfigBaseclass):
    name: Literal["kfac"] = "kfac"
    """Identifier. Fixed"""

    learning_rate: float = 0.1

    lr_schedule: Union[InverseLRScheduleConfig, ConstantLRSchedule] = InverseLRScheduleConfig()
    """Schedule for the learning rate decay"""

    momentum: float = 0.0
    norm_constraint: float = 1e-3
    damping: float = 1e-3
    damping_scheduler: bool = False
    estimation_mode: Literal['fisher_gradients', 'fisher_exact'] = 'fisher_exact'
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
    lr_schedule: Union[_InverseLRScheduleConfigForKFACwithAdam, InverseLRScheduleConfig, ConstantLRSchedule] = _InverseLRScheduleConfigForKFACwithAdam()
    internal_optimizer: Union[_OptimizerConfigAdamForKFAC, StandardOptimizerConfig] = _OptimizerConfigAdamForKFAC()


class BFGSOptimizerConfig(ConfigBaseclass):
    name: Literal["slbfgs"] = "slbfgs"
    """Identifier of optimizer. Fixed."""

    internal_optimizer: StandardOptimizerConfig = StandardOptimizerConfig()
    """Configuration for built-in optimizer to update the parameters, usinge the preconditioned gradients calculated by BFGS. 
    Use these internal optimizers to easily implement features like momentum."""

    memory_length: int = 2000
    """Number of gradients to keep in memory to build-up the hessian. Longer memory yields a higher-rank hessian, potentially increasing accuracy, but also slightly increases run-time cost."""

    hessian_regularization: float = 100.0
    """Amount of Tikhonov regularization to apply"""

    norm_constraint: float = 5e-4
    """Maximum of the norm of the preconditioned gradients. If the norm of the preconditioned gradients exceeds this norm_constraint, it will be scaled down to meet this norm."""

    use_variance_reduction: bool = False
    """Recalculate 2 batches at the end of each epoch to reduce stochastic MCMC noise in the gradients. Improves accuracy at the expense of 2 additional batches per epoch"""

    update_hessian_every_n_epochs: int = 1
    """How often to update the hessian. More frequent update (= smaller setting) is preferrable, but can be expensive when using use_variance_reduction=True"""


class IntermediateEvaluationConfig(ConfigBaseclass):
    n_epochs: int = 5000
    """How many evaluation epochs to use for intermediate evaluation."""

    opt_epochs: List[int] = []
    """List of epochs at which to run an intermediate evaluation to accurately assess wavefunction accuracy."""

class SharedOptimizationConfig(ConfigBaseclass):
    use: bool = True

    shared_modules: Optional[List[str]] = None
    """What modules/ parts of the neural network should be shared during weight-sharing"""

    scheduling_method: Union[Literal["round_robin", "stddev"]] = "round_robin"
    """Method to vary between geometries during weight-sharing"""

    max_age: int = 50
    """Maximal number of epochs a geometry can be not considered during weight-sharing. Preventing a few geometries from requiring all parameter updates for stddev scheduling"""


class CheckpointConfig(ConfigBaseclass):
    replace_every_n_epochs: int = 1000
    keep_every_n_epochs: int = 50_000
    additional_n_epochs: List[int] = []


class OptimizationConfig(ConfigBaseclass):
    mcmc: MCMCConfigOptimization = MCMCConfigOptimization()

    optimizer: Union[OptimizerConfigKFAC, _OptimizerConfigKFACwithAdam, StandardOptimizerConfig, BFGSOptimizerConfig] = OptimizerConfigKFAC()
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

    intermediate_eval: Optional[IntermediateEvaluationConfig] = None
    """Config for running intermediate evaluation runs during wavefunction optimization to obtain accurate estimates of current accuracy"""

    shared_optimization: Optional[SharedOptimizationConfig] = None
    """Config for shared optimization of multiple wavefunctions using weight-sharing between them"""

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

    path: str
    """Path to a previous calculation directory, containing a results.bz2 file"""

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

class ReuseConfig(RestartConfig):
    """Do not introduce new fields here"""
    mode: Literal["reuse"] = "reuse"

    reuse_config: bool = False
    reuse_opt_state: bool = False
    reuse_mcmc_state: bool = False
    reuse_clipping_state: bool = False
    reuse_trainable_params: bool = True
    reuse_fixed_params: bool = False
    reuse_modules: Optional[List[str]] = None
    continue_n_epochs: bool = False
    skip_burn_in: bool = False
    skip_pretraining: bool = False
    ignore_extra_settings: bool = False


class ForceEvaluationConfig(ConfigBaseclass):
    use: bool = True
    R_cut: float = 0.1
    R_core: float = 0.5
    use_antithetic_sampling: bool = False


class EvaluationConfig(ConfigBaseclass):
    mcmc: MCMCConfigEvaluation = MCMCConfigEvaluation()
    n_epochs: int = 10_000
    calculate_energies: bool = True
    forces: Optional[ForceEvaluationConfig] = None


class PhysicalConfigChange(ConfigBaseclass):
    R: Optional[List[List[float]]] = None
    E_ref: Optional[float] = None
    comment: Optional[str] = None


class PhysicalConfig(ConfigBaseclass):
    def get_basic_params(self):
        return self.n_electrons, self.n_up, np.array(self.R), np.array(self.Z)

    @staticmethod
    def _get_spin_from_hunds_rule(Z):
        n_orbitals = [1, 1, 3, 1, 3, 1, 5, 3, 1, 5, 3, 1, 7, 5]  # 1s, 2s, 2p, 3s, 3p, 4s, 5d, ...
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

    @property
    def n_dn(self):
        return self.n_electrons - self.n_up

    @property
    def n_ions(self):
        return len(self.Z)

    el_ion_mapping: Optional[List[int]] = None
    """Initial position of electrons. len(el_ion_mapping) == n_electrons. For each electron the list-entry specifies the index of the nucleus where the electron should be initialized.
    Note that the n_up first entries in this list correpsond to spin-up electrons and the remaining entries correspond to spin-down electrons"""

    n_cas_electrons: Optional[int] = None
    """Number of active electrons to include in the CASSCF calculation. Other electrons will be kept fixed in their respective orbitals."""

    n_cas_orbitals: Optional[int] = None
    """Number of active orbitals to include in the CASSCF calculation. Other orbitals are kept as always occupied or always unoccupied, depending on their energy."""

    E_ref: Optional[float] = None
    """Known ground-truth energy as reference for output. This value is not used in the actual calculation, but only to benchmark the achieved results."""

    E_ref_source: Optional[str] = None
    """Source of included reference energy"""

    comment: Optional[str] = None
    """Optional comment to keep track of molecules, geometries or experiments"""
    
    changes: Optional[List[PhysicalConfigChange]] = None
    """List of changes to be made to this physical-configuration to obtain different molecules. This can be used to calculate multiple geometries with similar settings, by specifying a base configuration (e.g. charge, spin, CASSCF-settings, etc) and only specifying the changes (e.g. in nuclear geometry) in the changes section."""

    _PERIODIC_TABLE = {k: i + 1 for i, k in enumerate(
        ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
         'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'])}

    _DEFAULT_MOLECULES = ruamel.yaml.YAML().load(pathlib.Path(__file__).parent.joinpath('molecules.yaml'))

    @validator("R", always=True)
    def populate_R(cls, v, values):
        if v is None:
            if values['name'] in cls._PERIODIC_TABLE:
                return [[0.0, 0.0, 0.0]]
            else:
                return cls._DEFAULT_MOLECULES[values['name']]['R']
        else:
            return v

    @validator("Z", always=True)
    def populate_Z(cls, Z, values):
        if Z is None:
            if values['name'] in cls._PERIODIC_TABLE:
                Z = [cls._PERIODIC_TABLE[values['name']]]
            else:
                Z = cls._DEFAULT_MOLECULES[values['name']]['Z']
        if len(Z) != len(values['R']):
            raise ValueError(
                f"List of nuclear charges Z has length {len(Z)}, but list of ion positions R has length {len(values['R'])}")
        return Z

    @validator("n_electrons", always=True)
    def populate_n_electrons(cls, v, values):
        if v is None:
            if values.get('name') in cls._DEFAULT_MOLECULES:
                charge = cls._DEFAULT_MOLECULES[values['name']].get('charge', 0)
            else:
                charge = 0
            n_electrons = int(sum(values['Z']) - charge)
            return n_electrons
        else:
            return v

    @validator("n_up", always=True)
    def populate_n_up(cls, v, values):
        if v is None:
            if values.get('name') in cls._PERIODIC_TABLE:
                spin = cls._get_spin_from_hunds_rule(values['Z'][0])
            elif (values.get('name') in cls._DEFAULT_MOLECULES):
                spin = cls._DEFAULT_MOLECULES[values['name']].get('spin') or 0
            else:
                spin = 0
            return (values['n_electrons'] + spin + 1) // 2  # if there is an extra electrons, assign them to up
        else:
            return v

    @validator("el_ion_mapping", always=True)
    def populate_el_ion_mapping(cls, v, values):
        if v is None:
            if ('name' in cls._DEFAULT_MOLECULES) and ('el_ion_mapping' in cls._DEFAULT_MOLECULES[values['name']]):
                v = cls._DEFAULT_MOLECULES[values['name']]['el_ion_mapping']
            else:
                v = cls._generate_el_ion_mapping(values['R'], values['Z'], values['n_electrons'], values['n_up'])
        if len(v) != values['n_electrons']:
            raise ValueError(
                f"An initial ion-mapping must be supplied for all electrons. len(el_ion_mapping)={len(v)}, n_electrons={values['n_electrons']}")
        return v

    @validator("n_cas_electrons", always=True)
    def populate_n_cas_electrons(cls, v, values):
        if v is None:
            if values['name'] in cls._DEFAULT_MOLECULES:
                v = cls._DEFAULT_MOLECULES[values['name']].get('cas_n_active_electrons')
        return v

    @validator("n_cas_orbitals", always=True)
    def populate_n_cas_orbitals(cls, v, values):
        if v is None:
            if values['name'] in cls._DEFAULT_MOLECULES:
                v = cls._DEFAULT_MOLECULES[values['name']].get('cas_n_active_orbitals')
        return v

    @validator("E_ref", always=True)
    def populate_E_ref(cls, v, values):
        name = values['name']
        if (v is None) and (name is not None):
            if name in cls._DEFAULT_MOLECULES:
                return cls._DEFAULT_MOLECULES[name].get('E_ref')
        return v

    @validator("E_ref_source", always=True)
    def populate_E_ref_source(cls, v, values):
        name = values['name']
        if (v is None) and (name is not None):
            if name in cls._DEFAULT_MOLECULES:
                return cls._DEFAULT_MOLECULES[name].get('E_ref_source')
        return v

    def set_from_changes(self):
        if self.changes is None:
            return [self]
        ret = []
        for change in self.changes:
            p = PhysicalConfig.update_configdict_and_validate(self.dict(), change.as_flattened_dict())[1]
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
    blacklist: List[str] = []


class BasicLoggerConfig(LoggerBaseConfig):
    log_to_stdout: bool = True
    log_level: Union[int, Literal["CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG"]] = "DEBUG"
    n_skip_epochs: int = 9
    fname: Optional[str] = "log.out"


class PickleLoggerConfig(LoggerBaseConfig):
    fname: str = 'results.bz2'


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
    system: Literal["local", "local_background", "vsc3", "vsc4", "dgx", "auto"] = "auto"
    """Which compute-cluster to use for this experiment. 'auto' detects whether the code is running on a known compute-cluster and selects the corresponding profile, or otherwise defaults to local execution"""

    queue: Literal[
        "default", "vsc3plus_0064", "devel_0128", "gpu_a40dual", "gpu_gtx1080amd", "gpu_gtx1080multi", "gpu_gtx1080single", "gpu_v100", "gpu_k20m", "gpu_rtx2080ti", "jupyter", "normal_binf", "vsc3plus_0256", "mem_0096", "devel_0096", "jupyter", "mem_0096", "mem_0384", "mem_0768"] = "default"
    """SLURM queue to use for job submission on compute clusters. Not relevant for local execution"""

    time: str = "3day"
    """Maximum job run-time to use for job submission on compute clusters. Not relevant for local execution"""

    conda_env: str = "jax"
    """Conda environment to select in SLURM-job-script. Not relevant for local execution"""

    split_opt: Optional[List[int]] = None

    eval_epochs: Optional[int] = None

class ComputationConfig(ConfigBaseclass):
    use_gpu: bool = True
    """deprecated"""
    require_gpu: bool = False
    n_devices: Optional[int] = None
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

    baseline: CASSCFConfig = CASSCFConfig()
    """Physical baseline settings for pre-training"""

# TODO: Generate a simple overview graphics of the main config options; use the schema to autogenerate?
class Configuration(ConfigBaseclass):
    """Root configuration for DeepErwin"""

    physical: Optional[PhysicalConfig]
    """The physical system/molecule being calculated"""

    pre_training: Optional[PreTrainingConfig] = PreTrainingConfig()
    """Supervised pre-training of the orbitals to fit the baseline orbitals."""

    optimization: OptimizationConfig = OptimizationConfig()
    """The wavefunction optimization"""

    evaluation: EvaluationConfig = EvaluationConfig()
    """The evaluation of the wavefunction (after optimization)"""

    model: Union[ModelConfigDeepErwin4, ModelConfigFermiNet, ModelConfigDeepErwin1] = ModelConfigDeepErwin4()
    """The actual wavefunction model mapping electron coordinates to psi"""

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

    @root_validator
    def experiment_has_name(cls, values):
        if values['experiment_name'] is None:
            if values['physical'] is None:
                values['experiment_name'] = "exp"
            else:
                values['experiment_name'] = values["physical"].name
        return values

    @root_validator
    def no_reuse_while_shared(cls, values):
        if (values['optimization'].shared_optimization is not None) and (values['reuse'] is not None):
            raise ValueError("Re-using weights not currently supported for shared optimization")
        return values


def get_with_flattened_key(config: Configuration, key):
    if "." not in key:
        return getattr(config, key)
    else:
        tokens = key.split('.')
        parent_key = tokens[0]
        child_key = ".".join(tokens[1:])
        return get_with_flattened_key(getattr(config, parent_key), child_key)

def set_with_flattened_key(config_dict, key, value):
    if not type(value) == dict and "." not in key:
        config_dict[key] = value
    elif type(value) == dict:
        if key not in config_dict or config_dict[key] is None:
            config_dict[key] = {}
        for key_child, value_child in value.items():
            set_with_flattened_key(config_dict[key], key_child, value_child)
    else:
        tokens = key.split('.')
        parent_key = tokens[0]
        child_key = ".".join(tokens[1:])
        if parent_key not in config_dict:
            config_dict[parent_key] = {}
        set_with_flattened_key(config_dict[parent_key], child_key, value)
    return config_dict

def build_nested_dict(flattened_dict):
    root_dict = {}
    for k, value in flattened_dict.items():
        key_tokens = k.split('.')
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
            flattened_dict.update({key + "." + k: v for k,v in build_flattend_dict(value).items()})
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

if __name__ == '__main__':
    import pathlib
    c = Configuration.parse_obj(dict(physical=dict(name='LiH'), model=dict(name="dpe4")))
    p = pathlib.Path(__file__).parent.joinpath("../../sample_configs/config_schema.json").resolve()
    print("Updating: ", p)
    with open(p, 'w') as f:
        f.write(c.schema_json())
    #
    # c = Configuration.parse_obj(dict(reuse=dict(mode='reuse', reuse_config=False, path="")))
    # print(c.reuse)
