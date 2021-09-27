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
    net_width: Union[int, None]
    net_depth: Union[int, None]

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
    embedding_dim = 64
    n_iterations = 2
    net_width: Union[int, None] = 40
    net_depth: Union[int, None] = 2
    n_hidden_w: Union[List[int], None]
    n_hidden_h: Union[List[int], None]
    n_hidden_g: Union[List[int], None]
    use_res_net = False

    @staticmethod
    def get_default_network_shape(key, width, depth):
        return dict(w=[width] * depth, h=[width] * depth, g=[width] * (depth - 1))[key]


class DummyEmbeddingConfig(ConfigModel):
    name: Literal["dummy"]


class CuspCorrectionConfig(ConfigModel):
    use = True
    cusp_type: Literal["mo", "ao"] = "mo"
    r_cusp_el_el = 1.0
    r_cusp_el_ion_scale = 1.0


class CASSCFConfig(ConfigModel):
    name: Literal["casscf"] = "casscf"
    basis_set = "6-311G"
    n_determinants = 20
    cusps = CuspCorrectionConfig()


class DeepErwinModelConfig(NetworkConfig):
    net_width: Union[int, None] = 40
    net_depth: Union[int, None] = 2
    n_hidden_bf_factor: Union[List[int], None]
    n_hidden_bf_shift: Union[List[int], None]
    n_hidden_jastrow: Union[List[int], None]
    use_bf_factor = True
    use_bf_shift = True
    use_jastrow = True

    el_el_shift_decay = False
    target_el = False
    sum_first = False
    output_shift: Literal[1, 3] = 1
    distance_feature_powers: List[int] = [-1]
    sigma_pauli = True
    use_trainable_shift_decay_radius = True

    n_rbf_features: int = 32
    eps_dist_feat: float = 1e-2
    n_pairwise_features: Optional[int] = None
    embedding: Union[SimpleSchnetConfig, DummyEmbeddingConfig] = SimpleSchnetConfig()
    baseline = CASSCFConfig()

    register_scale = True

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
    langevin_scale = 1.0
    r_min = 0.2
    r_max = 2.0


class MCMCConfig(ConfigModel):
    n_walkers_opt: int = 2048
    n_walkers_eval: Optional[int] = None
    n_inter_steps = 10
    n_burn_in_opt = 2000
    n_burn_in_eval = 500
    target_acceptance_rate = 0.5
    min_stepsize_scale = 1e-2
    max_stepsize_scale = 1.0
    max_age_opt: int = 20
    max_age_eval: int = 20

    proposal: Union[MCMCLangevinProposalConfig, MCMCSimpleProposalConfig] = MCMCSimpleProposalConfig()

    @validator("n_walkers_eval", always=True)
    def populate_n_walkers_eval(cls, n, values):
        return n or values['n_walkers_opt']


class ClippingConfig(ConfigModel):
    name: Literal["hard", "tanh"] = "tanh"
    width_metric: Literal["std", "mae"] = "std"
    center: Literal["mean", "median"] = "mean"
    unclipped_center = False
    clip_by = 5.0


class FixedLRSchedule(ConfigModel):
    name: Literal["fixed"] = "fixed"


class InverseLRScheduleConfig(ConfigModel):
    name: Literal["inverse"] = "inverse"
    decay_time: float = 500.0


class StandardOptimizerConfig(ConfigModel):
    name: Literal["rmsprop_momentum", "sgd"] = "sgd"  # add others optimizers that don't need further configs here
    order: Literal[1] = 1


class AdamOptimizerConfig(StandardOptimizerConfig):
    name: Literal["adam"] = "adam"
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8


class KFACOptimizerConfig(ConfigModel):
    name: Literal["kfac"] = "kfac"
    order: Literal[2] = 2
    momentum = 0.0
    norm_constraint = 0.001
    damping = 0.0005
    damping_scheduler = True
    estimation_mode = 'fisher_gradients'
    register_generic = True
    update_inverse_period = 1
    decay_time = 6000
    decay_time_damping = 6000
    min_damping = 1e-4
    curvature_ema = 0.05
    internal_optimizer: Union[AdamOptimizerConfig, StandardOptimizerConfig] = AdamOptimizerConfig()


class BFGSOptimizerConfig(ConfigModel):
    name: Literal["slbfgs"] = "slbfgs"
    order: Literal[2] = 2
    internal_optimizer: Union[AdamOptimizerConfig, StandardOptimizerConfig] = AdamOptimizerConfig()
    memory_length = 1000
    hessian_regularization: float = 200.0
    norm_constraint = 1.5e-3
    use_variance_reduction = True
    update_hessian_every_n_epochs = 1


class IntermediateEvaluationConfig(ConfigModel):
    n_epochs = 500
    opt_epochs: List[int] = []


class OptimizationConfig(ConfigModel):
    optimizer: Union[
        AdamOptimizerConfig, StandardOptimizerConfig, KFACOptimizerConfig, BFGSOptimizerConfig] = AdamOptimizerConfig()
    schedule: Union[InverseLRScheduleConfig, FixedLRSchedule] = InverseLRScheduleConfig()
    learning_rate = 1.5e-3
    n_epochs = 10_000
    n_epochs_prev = 0  # if run is restart, this can be used to store number of previous epochs
    batch_size = 512
    use_batch_reweighting = False
    checkpoints: List[int] = []
    clipping = ClippingConfig()
    intermediate_eval: Optional[IntermediateEvaluationConfig] = None
    interdependent = False
    shared_modules: Optional[List[Literal["embed", "jastrow", "bf_fac", "bf_shift"]]] = None

    @root_validator
    def shared_modules_only_with_interdependent(cls, values):
        if values['shared_modules'] is not None:
            if not values["interdependent"]:
                warnings.warn(
                    "Shared optimization is only possible with interdependent optimization. Setting shared_modules to None.")
                values['shared_modules'] = None
        return values


class RestartConfig(ConfigModel):
    path: str
    reuse_params = True
    reuse_mcmc_state = True
    recursive = True
    checkpoints = True


class ForceEvaluationConfig(ConfigModel):
    R_cut = 0.1
    R_core = 0.5
    use_polynomial: bool = False
    use_antithetic_sampling: bool = False
    polynomial_degree: int = 4


class EvaluationConfig(ConfigModel):
    n_epochs = 2000
    forces: Optional[ForceEvaluationConfig] = None


class PhysicalConfigChange(ConfigModel):
    R: Optional[List[List[float]]]
    E_ref: Optional[float]
    comment: Optional[str]


class PhysicalConfig(ConfigModel):
    name: Optional[str]
    R: Optional[List[List[float]]]
    Z: Optional[List[int]]
    n_electrons: Optional[int]
    n_up: Optional[int]
    el_ion_mapping: Optional[List[int]]
    n_cas_electrons: Optional[int]
    n_cas_orbitals: Optional[int]
    E_ref: Optional[float]
    comment: Optional[str]
    changes: Optional[List[PhysicalConfigChange]] = None

    _PERIODIC_TABLE = {k: i + 1 for i, k in enumerate(
        ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
         'Cl', 'Ar'])}
    _DEFAULT_GEOMETRIES = dict(H2=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]],
                               N2=[[0.0, 0.0, 0.0], [2.06800, 0.0, 0.0]],
                               LiH=[[0.0, 0.0, 0.0], [3.015, 0.0, 0.0]],
                               HChain10=[[2.0 * i, 0.0, 0.0] for i in range(10)],
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
    n_skip_epochs = 0


class WandBConfig(LoggerBaseConfig):
    project: Optional[str] = "default"
    entity: Optional[str]


class BasicLoggerConfig(LoggerBaseConfig):
    log_to_stdout = True
    log_level: Union[int, Literal["CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG"]] = "DEBUG"
    n_skip_epochs = 9
    fname: str = "log.out"


class PickleLoggerConfig(LoggerBaseConfig):
    fname = 'results.bz2'


class LoggingConfig(ConfigModel):
    tags: List[str] = []
    log_opt_state: bool = False
    basic: Optional[BasicLoggerConfig] = BasicLoggerConfig()
    wandb: Optional[WandBConfig] = None
    pickle: Optional[PickleLoggerConfig] = PickleLoggerConfig()

    # @classmethod
    # def get_logger_names(cls):
    #     logger_names = []
    #     for k, v in cls.__dict__["__fields__"].items():
    #         if issubclass(v.type_, LoggerBaseConfig):
    #             logger_names.append(k)
    #     return logger_names


class DispatchConfig(ConfigModel):
    system: Literal["local", "vsc3", "vsc4", "dgx", "auto"] = "auto"
    queue: Literal[
        "default", "vsc3plus_0064", "devel_0128", "gpu_a40dual", "gpu_gtx1080amd", "gpu_gtx1080multi", "gpu_gtx1080single", "gpu_v100", "gpu_k20m", "gpu_rtx2080ti", "jupyter", "normal_binf", "vsc3plus_0256", "mem_0096", "devel_0096", "jupyter", "mem_0096", "mem_0384", "mem_0768"] = "default"
    time: str = "1day"
    conda_env: str = "jax"


class ComputationConfig(ConfigModel):
    use_gpu = True
    disable_jit = False
    float_precision: Literal["float32", "float64"] = "float32"


class Configuration(ConfigModel):
    physical: Optional[PhysicalConfig]
    mcmc = MCMCConfig()
    optimization = OptimizationConfig()
    evaluation = EvaluationConfig()
    model: Union[DeepErwinModelConfig] = DeepErwinModelConfig()
    logging = LoggingConfig()
    computation = ComputationConfig()
    dispatch = DispatchConfig()
    restart: Optional[RestartConfig]
    comment: Optional[str]
    experiment_name: str = None

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
    pass

    #
    # parsed_config = Configuration.parse_obj(raw_config)
    # physical_configs = parsed_config.physical.set_from_changes()

    # c = Configuration(physical=PhysicalConfig(name='C'))
    # d = c.get_as_flattened_dict()
    # print(d)
