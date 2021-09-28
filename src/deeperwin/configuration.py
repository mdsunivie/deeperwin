"""
DeepErwin hyperparameter and configuration management.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Union, Literal, Optional, List

import pydantic
import ruamel.yaml.comments
from pydantic import BaseModel, validator, root_validator
from pydantic.fields import ModelField


class ConfigModel(BaseModel):
    """Base class for all config models"""

    @staticmethod
    def _to_prettified_yaml(data):
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

    def save(self, fname):
        data = self._to_prettified_yaml(self.dict())
        with open(fname, 'w') as f:
            ruamel.yaml.YAML().dump(data, f)

    @classmethod
    def load(cls, fname) -> 'ConfigModel':
        with open(fname) as f:
            data = ruamel.yaml.YAML().load(f)
            return cls.parse_obj(data)

    def get_as_flattened_dict(self):
        output_dict = {}
        for label in self.dict():
            subconfig = getattr(self, label)
            if hasattr(subconfig, 'get_as_flattened_dict'):
                subdict = subconfig.get_as_flattened_dict()
                for sublabel, subvalue in subdict.items():
                    output_dict[f"{label}.{sublabel}"] = subvalue
            else:
                output_dict[label] = subconfig
        return output_dict

    @root_validator(allow_reuse=True)
    def disable_unused_children(cls, values):
        for element in values.keys():
            if hasattr(values[element], 'use'):
                if not values[element].use:
                    values[element] = None
        return values

    @classmethod
    def update_config(cls, config_dict, config_changes):
        # First loop: Build an updated dictionary
        for key, value in config_changes:
            set_with_nested_key(config_dict, key, value)
        # Parse the config dict and validate that all parameters are valid
        config = cls.parse_obj(config_dict)

        # Second loop: Update the values using the parsed values to get correct type; Not strictly necessary,
        # but yields nicer looking input-config files
        for key, value in config_changes:
            set_with_nested_key(config_dict, key, get_attribute_by_nested_key(config, key))
        return config_dict, config

    class Config:
        """The name 'Config' here is a pydantic term used to config the behaviour of the dataclass.
        It's not a config for the variational monte carlo code"""
        extra = "forbid"


class NetworkConfig(ConfigModel, ABC):
    net_width: Optional[int] = None
    net_depth: Optional[int] = None

    @staticmethod
    @abstractmethod
    def get_default_network_shape(key, width, depth):
        pass

    @validator("*", always=True)
    def _set_embedding_widths(cls, n_hidden, values, field: ModelField):
        if field.name.startswith("n_hidden") and (n_hidden is None):
            return cls.get_default_network_shape(field.name.replace("n_hidden_", ""), values['net_width'],
                                                 values['net_depth'])
        return n_hidden


class SimpleSchnetConfig(NetworkConfig):
    name: Literal["simple_schnet"] = "simple_schnet"
    """Identifier for this model-part. Fixed."""

    embedding_dim: int = 64
    """Dimensionality of embedding vectors, including intermediate network widths."""

    n_iterations: int = 2
    """Number of embedding iterations for SchNet"""

    net_width: Optional[int] = 40
    """Number of hidden units per layers for w,h,g networks within SchNet."""

    net_depth: Optional[int] = 2
    """Depth of hidden layers for w-,h-, and g-networks within SchNet. Note that by default the g-network has only depth 1 instead of depth 2"""


    n_hidden_w: Optional[List[int]] = None
    """Number of hidden neurons per layer of the w-Network within SchNet. The w-network takes pairwise features between particles as inputs and calculates the weight of each embedding dimension as output. If not specified the width/depth as specified in net_width/net_depth is used"""


    n_hidden_h: Optional[List[int]] = None
    """Number of hidden neurons per layer of the h-Network within SchNet. The h-network takes the embedding of the previous iteration as input and computes a new embedding per particle, which is then weighted using the w-networks. If not specified the width/depth as specified in net_width/net_depth is used"""


    n_hidden_g: Optional[Union[List[int]]] = None
    """Number of hidden neurons per layer of the g-Network within SchNet. The g-network applies a final non-linear transformation at the end of each SchNet round. If not specified the width as specified in net_width and depth = net_depth - 1 is used"""

    use_res_net: bool = False
    """Whether to use residuals for the networks"""

    @staticmethod
    def get_default_network_shape(key, width, depth):
        return dict(w=[width] * depth, h=[width] * depth, g=[width] * (depth - 1))[key]


class DummyEmbeddingConfig(ConfigModel):
    name: Literal["dummy"] = "dummy"


class CuspCorrectionConfig(ConfigModel):
    use: bool = True
    cusp_type: Literal["mo", "ao"] = "mo"
    """Mode how to calculate cusp-corrected orbitals. 
    'ao' computes a cusp correction for each atomic orbital (i.e. for each basis function), 
    'mo' computes a cusp correction for each molecular orbital (i.e. each solution of the HF-calculation).
    For atoms both cusp_types should be equivalent, but for molecules the simpler 'ao' cusps can in principle not correctly model the cusps that arise from an atomic wavefunction having a finite contribution at a different nucleus.   
    """

    r_cusp_el_el: float = 1.0
    """Length-scale for the electron-electron cusp correction"""

    r_cusp_el_ion_scale: float = 1.0
    """Scaling parameter for the electron-ion cusp corrections. No cusp correction is applied outside a radius :code:`r_cusp_el_ion_scale / Z`"""


class CASSCFConfig(ConfigModel):
    name: Literal["casscf"] = "casscf"
    """Identifier of the baseline calculation. Fixed."""

    basis_set: str = "6-311G"
    """Basis set to use for the Hartree-Fock / CASSCF calculation. See the documentation of pySCF for all available basis-sets."""

    n_determinants: int = 20
    """Number of determinants of the CASSCF calculation to keep"""

    cusps: CuspCorrectionConfig = CuspCorrectionConfig()
    """Settings for the correction of wavefunction cusps when particles come close to each other"""


class DeepErwinModelConfig(NetworkConfig):
    """Configuration for the primary wavefunction model, which maps electron coordinate to psi"""

    net_width: Optional[int] = 40
    """Width of hidden layers for backflow- and jastrow-networks"""

    net_depth: Optional[int] = 2
    """Depth of hidden layers for backflow- and jastrow-networks"""

    n_hidden_bf_factor: Optional[List[int]] = None
    """List of ints, specifying the number of hidden units per layer in the backflow-factor-network. If not provided, the width and depth set by *net_width* and *net_depth* are used."""

    n_hidden_bf_shift: Optional[List[int]] = None
    """List of ints, specifying the number of hidden units per layer in the backflow-shift-network. If not provided, the width and depth set by *net_width* and *net_depth* are used."""

    n_hidden_jastrow: Optional[List[int]] = None
    """List of ints, specifying the number of hidden units per layer in the jastrow-network. If not provided, the width and depth set by *net_width* and *net_depth* are used."""

    use_bf_factor: bool = True
    """Enable the backflow-factor, i.e. multiply the CASSCF orbitals with the output of a neural network"""

    use_bf_shift: bool = True
    """Enable the backflow-shift, i.e. shift all electron coordinates by a neural network before calculating the CASSCF orbitals"""

    use_jastrow: bool = True
    """Enable the jastrow-factor, i.e. multiple the total wavefunction by the output of a global neural network"""

    output_shift: Literal[1, 3] = 1
    """Dimensionality of the output for the backflow-shift network. Can either be scalar or 3-dimensional. Note that only a scalar output ensures rotational equivariance."""

    distance_feature_powers: List[int] = [-1]
    """To calculate the embeddings, pairwise distances between particles are converted into input feature vectors and then fed into the embedding network. 
    These input feature vectors partially consist of radial-basis-functions and monomials of the distance (i.e. r^n). This list specifies which exponents should be used to create the distance monomials.
    Note, that using powers of -1 or +1 can lead to cusps in the input features, which may or may not be desirable.
    """

    sigma_pauli: bool = True
    """How to calculate the width or the radial-basis-function features: True = Implementation according to PauliNet, False = Implementation according to original DeepErwin"""

    use_trainable_shift_decay_radius: bool = True
    """Backflow-shift decays to 0 close the the nuclei to ensure correct cusp conditions. This flag specifies whether the radius in which the shift decays is a trainable parameter or fixed."""

    n_rbf_features: int = 32
    """Number of radial basis functions to use as pairwise fature vector"""

    eps_dist_feat: float = 1e-2
    """Epsilon to be used when calculating pairwise features with negative exponent n < 0. Feature vector is calculated as :math:`\\frac{1}{dist^{-n} + eps}`"""

    n_pairwise_features: Optional[int] = None
    """Total number of pairwise features to be used as input to the embedding network. Calculated automatically as :code:`n_rbf_features + len(distance_feature_powers)`"""

    embedding: Union[SimpleSchnetConfig, DummyEmbeddingConfig] = SimpleSchnetConfig()
    """Config-options for the electron-embedding"""

    baseline: CASSCFConfig = CASSCFConfig()
    """Config-options for the underlying baseline calculation, typically a Complete-Active-Space Self-Consistent-Field (CASSCF) Calculation"""

    register_scale = False
    """Whether to register the parameterst that scale the output of the backflow networks as parameters for KFAC optimization"""

    @staticmethod
    def get_default_network_shape(key, width, depth):
        return dict(bf_factor=[width] + [width // 2] * depth, bf_shift=[width] * depth, jastrow=[width] * depth)[key]

    @validator("n_pairwise_features", always=True)
    def total_number_of_pairwise_features(cls, n, values):
        n_required = values['n_rbf_features'] + len(values['distance_feature_powers'])
        n = n_required if n is None else n
        if n != n_required:
            raise pydantic.ValidationError(
                f"Number of total pairwise features {n} is inconsistent with n_rbf_features={values['n_rbf_features']} and distance_feature_powers={values['distance_feature_powers']}")
        return n


class MCMCSimpleProposalConfig(ConfigModel):
    name: Literal["normal", "cauchy"] = "normal"


class MCMCLangevinProposalConfig(ConfigModel):
    name: Literal["langevin"] = "langevin"
    langevin_scale: float = 1.0
    r_min: float = 0.2
    r_max: float = 2.0


class MCMCConfig(ConfigModel):
    """Config for Markov-Chain-Monte-Carlo integration"""

    n_walkers_opt: int = 2048
    """Number of walkers for optimization"""

    n_walkers_eval: Optional[int] = None
    """Number of walkers for evaluation"""

    n_inter_steps: int = 10
    """Number of MCMC steps between epochs"""

    n_burn_in_opt: int = 2000
    """Number of MCMC steps before starting optimization"""

    n_burn_in_eval: int = 500
    """Number of MCMC steps before starting evaluation"""

    target_acceptance_rate: float = 0.5
    """Acceptance-rate that the MCMC-runner is trying to achieve by modifying the stepsize"""

    min_stepsize_scale: float = 1e-2
    """Minimum stepsize. For spatially adaptive stepsize-schemes this only defines a factor which may be modified by the adaptive scheme"""

    max_stepsize_scale: float = 1.0
    """Maximum stepsize. For spatially adaptive stepsize-schemes this only defines a factor which may be modified by the adaptive scheme"""

    max_age_opt: int = 20
    """Maximum number of MCMC steps for which a walker can reject updates during optimization. After having rejected an update max_age times, the walkers is forced to accepet, to avoid getting stuck"""

    max_age_eval: int = 20
    """Maximum number of MCMC steps for which a walker can reject updates during evaluation. After having rejected an update max_age times, the walkers is forced to accepet, to avoid getting stuck"""

    proposal: Union[MCMCLangevinProposalConfig, MCMCSimpleProposalConfig] = MCMCSimpleProposalConfig()
    """Type of proposal function to use for MCMC steps"""

    @validator("n_walkers_eval", always=True)
    def populate_n_walkers_eval(cls, n, values):
        return n or values['n_walkers_opt']


class ClippingConfig(ConfigModel):
    name: Literal["hard", "tanh"] = "tanh"
    width_metric: Literal["std", "mae"] = "std"
    center: Literal["mean", "median"] = "mean"
    unclipped_center: bool = False
    clip_by: float = 5.0


class FixedLRSchedule(ConfigModel):
    name: Literal["fixed"] = "fixed"


class InverseLRScheduleConfig(ConfigModel):
    name: Literal["inverse"] = "inverse"
    decay_time: float = 1000.0


class StandardOptimizerConfig(ConfigModel):
    name: Literal["rmsprop_momentum", "sgd"] = "sgd"  # add others optimizers that don't need further configs here
    order: Literal[1] = 1


class AdamOptimizerConfig(StandardOptimizerConfig):
    name: Literal["adam"] = "adam"
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8


class KFACOptimizerConfig(ConfigModel):
    name: Literal["kfac"] = "kfac"
    order: Literal[2] = 2
    momentum: float = 0.0
    norm_constraint: float = 0.001
    damping: float = 0.0005
    damping_scheduler: bool = True
    estimation_mode: str = 'fisher_gradients'
    register_generic: bool = True
    update_inverse_period: int = 1
    decay_time: int = 6000
    decay_time_damping: int = 6000
    min_damping: float = 1e-4
    curvature_ema: float = 0.05
    internal_optimizer: Union[AdamOptimizerConfig, StandardOptimizerConfig] = AdamOptimizerConfig()


class BFGSOptimizerConfig(ConfigModel):
    name: Literal["slbfgs"] = "slbfgs"
    """Identifier of optimizer. Fixed."""

    order: Literal[2] = 2
    """Degree of optimizer. Fixed."""

    internal_optimizer: Union[AdamOptimizerConfig, StandardOptimizerConfig] = AdamOptimizerConfig()
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


class IntermediateEvaluationConfig(ConfigModel):
    n_epochs: int = 500
    opt_epochs: List[int] = []

class SharedOptimizationConfig(ConfigModel):
    use: Literal[True, False] = True
    shared_modules: Optional[List[Literal["embed", "jastrow", "bf_fac", "bf_shift"]]] = None
    scheduling_method: Union[Literal["round_robin", "stddev"]] = "round_robin"
    max_age: int = 50


class OptimizationConfig(ConfigModel):
    optimizer: Union[
        AdamOptimizerConfig, StandardOptimizerConfig, KFACOptimizerConfig, BFGSOptimizerConfig] = AdamOptimizerConfig()
    """Which optimizer to use and its corresponding sub-options"""

    schedule: Union[InverseLRScheduleConfig, FixedLRSchedule] = InverseLRScheduleConfig()
    """Schedule for the learning rate decay"""

    learning_rate: float = 1.5e-3
    """Initial learning rate at epoch 0. Actual learning rate during optimization may be modified through the LR-scheduler"""
    n_epochs: int = 10_000
    """Number of epochs for wavefunction optimization"""

    n_epochs_prev: int = 0  # if run is restart, this can be used to store number of previous epochs
    """Nr of epochs that this wavefunction has already been optimized. This can be used to store number of previous epochs after a restart"""

    batch_size: int = 512
    """Nr of walkers to process in a single backprop step. Reduce this number in case of insufficient GPU-memory"""

    use_batch_reweighting: bool = False
    """Reweight gradients for different samples with the changes in log(psi^2) between batches"""

    checkpoints: List[int] = []
    """List of epoch-numbers at which a checkpoint-file should be dumped, containing all model weights and MCMC-walkers to allow a restart or evaluation"""

    clipping: ClippingConfig = ClippingConfig()
    """Config for clipping the local energies in the loss function. Clipping significantly improves optimization stability."""

    intermediate_eval: Optional[IntermediateEvaluationConfig] = None
    """Config for running intermediate evaluation runs during wavefunction optimization to obtain accurate estimates of current accuracy"""

    shared_optimization: Optional[SharedOptimizationConfig] = None
    """Config for interdependent optimization of multiple wavefunctions using weight-sharing between them"""


class RestartConfig(ConfigModel):
    path: str
    reuse_params: bool = True
    reuse_mcmc_state: bool = True
    recursive: bool = True
    checkpoints: bool = True


class ForceEvaluationConfig(ConfigModel):
    use: bool = False
    R_cut: float = 0.1
    R_core: float = 0.5
    use_polynomial: bool = False
    use_antithetic_sampling: bool = False
    polynomial_degree: int = 4


class EvaluationConfig(ConfigModel):
    n_epochs: int = 2000
    forces: Optional[ForceEvaluationConfig] = None


class PhysicalConfigChange(ConfigModel):
    R: Optional[List[List[float]]] = None
    E_ref: Optional[float] = None
    comment: Optional[str] = None


class PhysicalConfig(ConfigModel):
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

    n_cas_electrons: Optional[int] = None
    """Number of active electrons to include in the CASSCF calculation. Other electrons will be kept fixed in their respective orbitals."""

    n_cas_orbitals: Optional[int] = None
    """Number of active orbitals to include in the CASSCF calculation. Other orbitals are kept as always occupied or always unoccupied, depending on their energy."""

    E_ref: Optional[float] = None
    """Known ground-truth energy as reference for output. This value is not used in the actual calculation, but only to benchmark the achieved results."""

    comment: Optional[str] = None
    """Optional comment to keep track of molecules, geometries or experiments"""
    
    changes: Optional[List[PhysicalConfigChange]] = None
    """List of changes to be made to this physical-configuration to obtain different molecules. This can be used to calculate multiple geometries with similar settings, by specifying a base configuration (e.g. charge, spin, CASSCF-settings, etc) and only specifying the changes (e.g. in nuclear geometry) in the changes section."""

    _PERIODIC_TABLE = {k: i + 1 for i, k in enumerate(
        ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
         'Cl', 'Ar'])}
    _DEFAULT_GEOMETRIES = dict(H2=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]],
                               N2=[[0.0, 0.0, 0.0], [2.06800, 0.0, 0.0]],
                               LiH=[[0.0, 0.0, 0.0], [3.015, 0.0, 0.0]],
                               HChain10=[[1.8 * i, 0.0, 0.0] for i in range(10)],
                               HChain6=[[1.8 * i, 0.0, 0.0] for i in range(6)],
                               Ethene=[[1.26517164, 0, 0], [-1.26517164, 0, 0], [2.328293764, 1.7554138407, 0],
                                       [2.328293764, -1.7554138407, 0],
                                       [-2.328293764, 1.7554138407, 0], [-2.328293764, -1.7554138407, 0]],
                               EtheneBarrier=[[1.26517164, 0, 0], [-1.26517164, 0, 0], [2.328293764, 0, 1.7554138407],
                                              [2.328293764, 0, -1.7554138407],
                                              [-2.328293764, 1.7554138407, 0], [-2.328293764, -1.7554138407, 0]],
                               Cyclobutadiene=[
                                   [-1.4777688, -1.2793472, 0],
                                   [+1.4777688, -1.2793472, 0],
                                   [+1.4777688, +1.2793472, 0],
                                   [-1.4777688, +1.2793472, 0],
                                   [-2.9180622, -2.7226601, 0],
                                   [+2.9180622, -2.7226601, 0],
                                   [+2.9180622, +2.7226601, 0],
                                   [-2.9180622, +2.7226601, 0]
                               ])
    _DEFAULT_CHARGES = dict(H2=[1, 1],
                            N2=[7, 7],
                            LiH=[3, 1],
                            HChain10=[1] * 10,
                            HChain6=[1] * 6,
                            Ethene=[6, 6, 1, 1, 1, 1],
                            EtheneBarrier=[6, 6, 1, 1, 1, 1],
                            Cyclobutadiene=[6, 6, 6, 6, 1, 1, 1, 1])
    _DEFAULT_SPIN = dict(H=1, He=0, Li=1, Be=0, B=1, C=2, N=3, O=2, F=1, Ne=0, Na=1, Mg=0, Al=1, Si=2, P=3, S=2, Cl=1,
                         Ar=0, Ethene=0, Cyclobutadiene=0, HChain6=0, HChain10=0)
    _DEFAULT_EL_ION_MAPPINGS = dict(H2=[0, 1],
                                    N2=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
                                    LiH=[0, 1, 0, 0],
                                    HChain6=[0, 2, 4, 1, 3, 5],
                                    HChain10=[0, 2, 4, 6, 8, 1, 3, 5, 7, 9],
                                    Ethene=[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 4, 3, 5],
                                    Cyclobutadiene=[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 6, 0, 0, 0, 1, 1, 1, 2, 2, 2,
                                                    3, 3, 3, 5, 7]
                                    )
    _DEFAULT_CAS_N_ACTIVE_ORBITALS = dict(He=2, Li=9, Be=8, LiH=10, B=12, C=12, N=12, O=12, F=12,
                                          Ne=12, H2=4, Li2=16, Be2=16, B2=16, C2=16, N2=16,
                                          H2p=1, H3plus=9, H4plus=12, H4Rect=12, HChain6=12, HChain10=10,
                                          Ethene=12, Cyclobutadiene=12)

    _DEFAULT_CAS_N_ACTIVE_ELECTRONS = dict(He=2, Li=3, Be=2, LiH=2, B=3, C=4, N=5, O=6, F=7, Ne=8, H2=2, Li2=2, Be2=4,
                                           B2=6, C2=8, N2=6, H2p=1, H3plus=2, H4plus=3, H4Rect=4, HChain6=6,
                                           HChain10=10, Ethene=12, Cyclobutadiene=12)

    _REFERENCE_ENERGIES = {'He': -2.90372, 'Li': -7.478067, 'Be': -14.66733, 'B': -24.65371, 'C': -37.84471,
                           'N': -54.58882,
                           'O': -75.06655, 'F': -99.7329, 'Ne': -128.9366, 'H2': -1.17448, 'LiH': -8.070548,
                           'N2': -109.5423, 'Li2': -14.9954, 'CO': -113.3255, 'Be2': -29.338, 'B2': -49.4141,
                           'C2': -75.9265, 'O2': -150.3274, 'F2': -199.5304, 'H4Rect': -2.0155,
                           'H3plus': -1.3438355180000001, 'H4plus': -1.8527330000000002, 'HChain6': -3.40583160,
                           'HChain10': -5.6655, 'Ethene': -78.57744597}

    @validator("R", always=True)
    def populate_R(cls, v, values):
        if v is None:
            if values['name'] in cls._PERIODIC_TABLE:
                return [[0.0, 0.0, 0.0]]
            else:
                return cls._DEFAULT_GEOMETRIES[values['name']]
        else:
            return v

    @validator("Z", always=True)
    def populate_Z(cls, Z, values):
        if Z is None:
            if values['name'] in cls._PERIODIC_TABLE:
                Z = [cls._PERIODIC_TABLE[values['name']]]
            else:
                Z = cls._DEFAULT_CHARGES[values['name']]
        if len(Z) != len(values['R']):
            raise pydantic.ValidationError(
                f"List of nuclear charges Z has length {len(Z)}, but list of ion positions R has length {len(values['R'])}")
        return Z

    @validator("n_electrons", always=True)
    def populate_n_electrons(cls, v, values):
        return int(sum(values['Z'])) if v is None else v

    @validator("n_up", always=True)
    def populate_n_up(cls, v, values):
        if v is None:
            spin = cls._DEFAULT_SPIN.get(values['name']) or 0
            return (values['n_electrons'] + spin + 1) // 2  # if there is an extra electrons, assign them to up
        else:
            return v

    @validator("el_ion_mapping", always=True)
    def populate_el_ion_mapping(cls, v, values):
        if v is None:
            if values['name'] in cls._PERIODIC_TABLE:
                v = [0] * values['n_electrons']
            else:
                v = cls._DEFAULT_EL_ION_MAPPINGS[values['name']]
        if len(v) != values['n_electrons']:
            raise pydantic.ValidationError(
                f"An initial ion-mapping must be supplied for all electrons. len(el_ion_mapping)={len(v)}, n_electrons={values['n_electrons']}")
        return v

    @validator("n_cas_electrons", always=True)
    def populate_n_cas_electrons(cls, v, values):
        if v is None:
            return cls._DEFAULT_CAS_N_ACTIVE_ELECTRONS[values['name']]
        else:
            return v

    @validator("n_cas_orbitals", always=True)
    def populate_n_cas_orbitals(cls, v, values):
        if v is None:
            return cls._DEFAULT_CAS_N_ACTIVE_ORBITALS[values['name']]
        else:
            return v

    @validator("E_ref", always=True)
    def populate_E_ref(cls, v, values):
        name = values['name']
        if (v is None) and (name is not None):
            return cls._REFERENCE_ENERGIES.get(name)
        return v

    def set_from_changes(self):
        if self.changes is None:
            return [self]
        ret = []
        for change in self.changes:
            p = PhysicalConfig.update_config(self.dict(), change.get_as_flattened_dict().items())[1]
            p.changes = None
            ret.append(p)
        return ret


class LoggerBaseConfig(ConfigModel):
    n_skip_epochs: int = 0


class WandBConfig(LoggerBaseConfig):
    project: str = "default"
    entity: Optional[str] = None


class BasicLoggerConfig(LoggerBaseConfig):
    log_to_stdout: bool = True
    log_level: Union[int, Literal["CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG"]] = "DEBUG"
    n_skip_epochs: int = 9
    fname: str = "log.out"


class PickleLoggerConfig(LoggerBaseConfig):
    fname: str = 'results.bz2'


class LoggingConfig(ConfigModel):
    tags: List[str] = []
    """List of tags to help mark/identify runs"""

    log_opt_state: bool = False
    """Flag whether to log the full state of the optimizer. Note that this can produce very large output-files, in particular when logging the hessian of 2nd-order-optimizers"""

    basic: Optional[BasicLoggerConfig] = BasicLoggerConfig()
    """Config for basic logger, which will produce human-readable log-files using the python built-in logging module"""

    wandb: Optional[WandBConfig] = None
    """Config for logging to Weights&Biases"""

    pickle: Optional[PickleLoggerConfig] = PickleLoggerConfig()
    """Config for logging to binary files, which contain machine-readable pickle files containg all configs, metrics and model weights"""


class DispatchConfig(ConfigModel):
    system: Literal["local", "vsc3", "vsc4", "dgx", "auto"] = "auto"
    """Which compute-cluster to use for this experiment. 'auto' detects whether the code is running on a known compute-cluster and selects the corresponding profile, or otherwise defaults to local execution"""

    queue: Literal[
        "default", "vsc3plus_0064", "devel_0128", "gpu_a40dual", "gpu_gtx1080amd", "gpu_gtx1080multi", "gpu_gtx1080single", "gpu_v100", "gpu_k20m", "gpu_rtx2080ti", "jupyter", "normal_binf", "vsc3plus_0256", "mem_0096", "devel_0096", "jupyter", "mem_0096", "mem_0384", "mem_0768"] = "default"
    """SLURM queue to use for job submission on compute clusters. Not relevant for local execution"""

    time: str = "1day"
    """Maximum job run-time to use for job submission on compute clusters. Not relevant for local execution"""

    conda_env: str = "jax"
    """Conda environment to select in SLURM-job-script. Not relevant for local execution"""


class ComputationConfig(ConfigModel):
    use_gpu: bool = True
    disable_jit: bool = False
    float_precision: Literal["float32", "float64"] = "float32"


class Configuration(ConfigModel):
    """Root configuration for DeepErwin"""

    physical: Optional[PhysicalConfig]
    """The physical system/molecule being calculated"""

    mcmc: MCMCConfig = MCMCConfig()
    """The Markov-Chain-Monte-Carlo integration"""

    optimization: OptimizationConfig = OptimizationConfig()
    """The wavefunction optimization"""

    evaluation: EvaluationConfig = EvaluationConfig()
    """The evaluation of the wavefunction (after optimization)"""

    model: Union[DeepErwinModelConfig] = DeepErwinModelConfig()
    """The actual wavefunction model mapping electron coordinates to psi"""

    logging: LoggingConfig = LoggingConfig()
    """The output of the code, e.g. logging to files, or online-services"""

    computation: ComputationConfig = ComputationConfig()
    """Options regarding computational details such as GPU usage and float precision"""

    dispatch: DispatchConfig = DispatchConfig()
    """Options regarding where the code is being run, i.e. locally vs asynchronysly on a compute-cluster"""

    restart: Optional[RestartConfig] = None
    """Restarting from a previous calculation, reusing model weights and MCMC walkers"""

    comment: Optional[str] = None
    """Optional coment to keep track of experiments"""

    experiment_name: Optional[str] = None
    """Experiment name to keep track of experiments"""


    @root_validator
    def walkers_divisible_by_batch_size(cls, values):
        n_walkers = values['mcmc'].n_walkers_opt
        batch_size = values['optimization'].batch_size
        if (n_walkers % batch_size) != 0:
            raise ValueError(f"Number of walkers ({n_walkers}) is not divisible by batch-size ({batch_size})")
        return values

    @root_validator
    def experiment_has_name(cls, values):
        if values['experiment_name'] is None:
            values['experiment_name'] = values["physical"].name
        return values


# Helper Functions
def get_attribute_by_nested_key(config: Configuration, key):
    if "." not in key:
        return getattr(config, key)
    else:
        tokens = key.split('.')
        parent_key = tokens[0]
        child_key = ".".join(tokens[1:])
        return get_attribute_by_nested_key(getattr(config, parent_key), child_key)


def set_with_nested_key(config_dict, key, value):
    if "." not in key:
        config_dict[key] = value
    else:
        tokens = key.split('.')
        parent_key = tokens[0]
        child_key = ".".join(tokens[1:])
        if parent_key not in config_dict:
            config_dict[parent_key] = {}
        set_with_nested_key(config_dict[parent_key], child_key, value)
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


if __name__ == '__main__':
    c = Configuration(physical=PhysicalConfig(name='LiH'))
    print(c)
