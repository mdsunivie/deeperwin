"""
This module contains classes modelling all configuration options for DeepErwin.

This includes basic functionality, such as parsing and storing, as well as all default values for all config options.
The options are nested hierarchically, starting at the root class :class:`DefaultConfig`.
"""
import logging
import json
from deeperwin.utilities.logging import getLogger

class Configuration:
    def get_as_dict(self):
        """
        Convert configuration to a flattened dictionary.

        Output will be of the form: {a.b.x: 1, a.b.y: 2, ...}
        Returns:
            (dict): Flattened dict containing full configuration
        """
        config_dict = {}
        for name, property in vars(self).items():
            if isinstance(property, Configuration):
                for name_, property_ in property.get_as_dict().items():
                    config_dict[name + "." + name_] = property_
            else:
                config_dict[name] = property
        return config_dict

    def get_with_nested_key(self, nested_key):
        """
        Retrieve a value by passing a nexted key, where the nesting levels are separated by '.', e.g. get_with_nested_key('optimization.n_epochs')
        Args:
            nested_key (str): Key

        Returns:
            (value): Value of dict
        """
        keys = nested_key.split('.')
        item = self
        for key in keys:
            item = item[key]
        return item

    def set_with_type_conversion(self, key, value, convert_type=True, allow_new_keys=False):
        """
        Sets a value by trying to convert the passed value to the required datatype for the field indicated by key.

        If the passed type is not a string, it will be cast to the dtype of the current value located at the key.
        If the passed type is a string and the target is not a string, eval() will be used to cast it. This allows passing expressions, such as '1+2' to an int field, or '[i for i in range(5)]' to a list field.
        Only pass trusted config-strings to this function, because eval does not provide any security checks!

        Args:
            key (str): Non-nested key (use set_with_nested_key if you have a '.'-separated nested key)
            value: Value to be set
            convert_type (bool): Whether to enable type conversion
            allow_new_keys (bool): Whether to allow create non-existing keys, or to raise a KeyError when key does not exist

        Returns:
            None
        """
        from . import geometries
        if key in self:
            # Key is already existing in configuration => update the value (optionally align types before doing that)
            if convert_type:
                target_type = type(self[key])
                if (type(value) == str) and (target_type != str):
                    try:
                        value = eval(
                            value)  # Warning: this evaluates any python code and thus creates a potential security risk, but the safer ast.literal_eval() does not have enough capabilities.
                    except ValueError:
                        raise ValueError(
                            f"Error when tryping to parse the following key/value config pair {key}: {value}")
                if self[key] is not None and not isinstance(self[key], Configuration):
                    value = target_type(value)
            self[key] = value
        else:
            # Key does not exist yet
            if allow_new_keys:
                self[key] = value
            else:
                raise KeyError(
                    f"Key {key} does not yet exist in configuration. If this was not a typo and you wanted to intentionally add a new property, use allow_new_keys=True")


    def set_with_nested_key(self, nested_key, value, convert_type=True, allow_new_keys=False):
        """
        Iteratively traverses down a configuration along the nested key and sets the value at the leaf.


        Thsi function attempts to convert the given input to the target dtype, including eval() on strings (see :meth:`~deeperwin.utilities.postprocessing.set_with_type_conversion`).
        Args:
            nested_key (str): Key-string, where levels are separted by '.', e.g. 'model.embed.use'
            value: Value to be set
            convert_type (bool): Whether to enable type conversion
            allow_new_keys (bool): Whether to allow create non-existing keys, or to raise a KeyError when key does not exist

        Returns:
            None
        """
        keys = nested_key.split('.')
        if len(keys) == 1:
            # Base case: set the value directly
            self.set_with_type_conversion(keys[0], value, convert_type, allow_new_keys)
        else:
            # Nested key => Create current layer if it does not exist and then recurse
            if (keys[0] not in self) or (self[keys[0]] is None):
                self[keys[0]] = self._get_subclass_from_param_name(keys, value)
            self[keys[0]].set_with_nested_key('.'.join(keys[1:]), value, convert_type, allow_new_keys)

    def _get_subclass_from_param_name(self, keys, value):
        """
        Parses a key/value pair to determine which config-class should be constructed for it.

        Checks whether the key is a 'name' attribute and uses this to to determine the correct class. If no matching class is found, a generic Configuration() object is returned.
        """
        if len(keys) > 0:
            if keys[0] == 'adaptiveLR':
                if len(keys) > 1:
                    if keys[1] == 'name':
                        if value == 'inverse':
                            return InverseLRConfig()
                        elif value == 'exponential':
                            return ExponentialLRConfig()
                        elif value == 'patience':
                            return PatienceLRConfig()
                        else:
                            return Configuration()
        return Configuration()

    def get_hyperparam_dict(self):
        """
        Returns the config as a flattened dictionairy, but converts all types that are not float or int to strings.

        This method is used to pass hyperparameters to the tensorboard log.
        Returns:
            (dict): Full config dict
        """
        d = self.get_as_dict()
        for key in d:
            if not isinstance(d[key], (float, int)):
                d[key] = str(d[key])
        return d

    def __getitem__(self, key):
        return vars(self)[key]

    def __setitem__(self, key, value):
        vars(self)[key] = value

    def __contains__(self, item):
        return item in vars(self)

    def update_with_dict(self, d, convert_type=True, allow_new_keys=False, **kwargs):
        """
        Update values in a Configuration object for the given keys/values in the passed dictionary. Keys not present in the passed dict, will not be changed.

        Args:
            d (dict): Keys are '.'-separated nested keys, values are arbitrary.
            convert_type (bool): Whether to allow magic type conversion
            allow_new_keys (bool): Whether to allow setting new keys
            **kwargs: Additional key/value pairs to be set

        Returns:
            None
        """
        if d is None:
            d = {}
        if 'quality' in kwargs:
            self.setDefaults(kwargs['quality'])
            del kwargs['quality']
        for key, val in kwargs.items():
            d[key] = val

        def priority_func(key):
            """
            Assigns higher priorities to config-keys if they are a name atribute (because they will determine the config class template)
            """
            priority = 10
            if key.endswith('name'):
                priority = 5
            return priority

        entries = [(key.count('.'), priority_func(key), key, d[key]) for key in d]
        for entries in sorted(entries):
            key, value = entries[2:]
            self.set_with_nested_key(key, value, convert_type, allow_new_keys)


    @classmethod
    def build_from_dict(cls, d, convert_type=True, allow_new_keys=False):
        """
        Build a new Configuration object based on a dict of nested key/value pairs. See :meth:`update_with_dict`.

        Args:
            d (dict): Keys are '.'-separated nested keys, values are arbitrary.
            convert_type (bool): Whether to allow magic type conversion
            allow_new_keys (bool): Whether to allow setting new keys

        Returns:
            (Configuration): New config object

        """
        config = cls()
        config.update_with_dict(d, convert_type, allow_new_keys)
        return config

    # docstr-coverage:inherited
    def setDefaults(self, quality='default'):
        """
        Set several config parameters to a pre-defined set of values, indicated by a quality level.

        Subclasses can implement any quality name, but should at least support 'default'
        Args:
            quality (str): Quality level

        Returns:
            None
        """
        if quality == 'default':
            pass
        else:
            raise ValueError(f"Unknown quality type {quality} for configuration class {type(self).__name__}")

    def __str__(self):
        s = ""
        for k, v in self.get_as_dict().items():
            s += f"{k}: {v}\n"
        return s[:-1]  # omit last newline character

    @classmethod
    def load(cls, fname):
        """
        Load a Configuration from a JSON-file that contains a dump of the flattend config-dict (e.g. as given by get_as_dict).
        Args:
            fname (str): Filename to a json file

        Returns:
            (Configuration): Configuration object
        """
        with open(fname) as f:
            config_dict = json.load(f)
        return cls.build_from_dict(config_dict)

class DefaultConfig(Configuration):
    """
    Default configuration for the DeepErwin code package. This class defines both the required config structure, as well as all default values.
    """
    def __init__(self, config_dict=None, **kwargs):
        super().__init__()
        self.restart_dir = ""
        self.physical = PhysicalConfig()
        self.model = DeepErwinModelConfig()
        self.integration = IntegrationConfig()
        self.optimization = OptimizationConfig()
        self.evaluation = EvaluationConfig()
        self.output = OutputConfig()
        self.parallel_trainer = ParallelTrainerConfig()
        self.update_with_dict(config_dict, **kwargs)

    # docstr-coverage:inherited
    @classmethod
    def build_from_dict(cls, d, convert_type=True, allow_new_keys=False, **kwargs):
        config = cls()
        if 'model.name' in d:
            if d['model.name'].lower() == "deeperwinmodel":
                config.model = DeepErwinModelConfig()
        config.update_with_dict(d, convert_type, allow_new_keys)
        return config

    def validate(self, raise_on_invalid=True):
        """
        Checks the configuration for correctness and consistency. This check is not exhaustive!
        Args:
            raise_on_invalid (bool): If True, raises a ValueError if config is incorrect. Otherwise only returns False.

        Returns:
            (bool): Whether passed config is (at least superficially) valid
        """
        is_valid = True
        if self.parallel_trainer.scheduling not in ['round-robin', 'random', 'stddev']:
            getLogger().warning(f"Invalid parallel-trainer wf-selection-scheme: {self.parallel_trainer.scheduling }")
            is_valid = False
        if hasattr(self.model, 'use_linear_transformation_for_embed') and self.model.use_linear_transformation_for_embed:
            getLogger().warning("use_linear_transformation_for_embed is no longer supported")
            is_valid = False
        if hasattr(self.physical, 'el_el_interaction_scale') and self.physical.el_el_interaction_scale != 1.0:
            getLogger().warning("Electron-Electron interaction scale must be 1.0; non-physical interactions are deprecated")
            is_valid = False
        # if hasattr(self.model, 'embed_without_loop') and self.model.embed_without_loop:
        #     if (self.physical.n_spin_up < 2) or (self.physical.n_electrons - self.physical.n_spin_up) < 2:
        #         ut.getLogger().warning("embed_without_loop requires at least 2 up and 2 down electrons. Disabling embed_without_loop now.")
        #         self.model.embed_without_loop = False
        if hasattr(self.parallel_trainer, 'shared_weights'):
            allowed_weights = set(['embedding', 'symmetric', 'backflow_shift', 'backflow_factor_general', 'backflow_factor_orbitals'])
            for w in self.parallel_trainer.shared_weights:
                if w not in allowed_weights:
                    getLogger().warning("Invalid shared weight: " + w)
                    is_valid = False
        if hasattr(self.optimization, 'use_energy_baseline') and self.optimization.use_energy_baseline:
            getLogger().warning("Usage of energy baseline is deprecated and no longer supported")
            is_valid = False
        if self.parallel_trainer.use and ((self.parallel_trainer.adaptiveLR is None) or (self.parallel_trainer.adaptiveLR.name is None)):
            getLogger().warning("No LR schedule provided for shared weights. Using same schedule as individual weights")
            self.parallel_trainer.adaptiveLR = self.optimization.adaptiveLR

        if raise_on_invalid and not is_valid:
            raise ValueError("Found invalid combination of configuration parameters - Aborting")
        return is_valid

    # docstr-coverage:inherited
    def setDefaults(self, quality='default'):
        self.model = DeepErwinModelConfig(quality=quality)
        self.optimization = OptimizationConfig(quality=quality)
        self.integration = IntegrationConfig(quality=quality)
        self.evaluation = EvaluationConfig(quality=quality)

class ParallelTrainerConfig(Configuration):
    def __init__(self, config_dict=None, **kwargs):
        super().__init__()
        self.use = False
        self.scheduling = 'stddev' # 'random', 'round-robin'
        self.config_changes = []
        self.shared_weights = []
        self.lr_factor = 10.0
        self.adaptiveLR = None
        self.scheduling_max_age = 100
        self.update_with_dict(config_dict, **kwargs)

class PhysicalConfig(Configuration):
    def __init__(self, config_dict=None, **kwargs):
        super().__init__()
        self.name = ""
        self.n_electrons = None
        self.n_spin_up = None
        self.ion_charges = []
        self.ion_positions = []
        self.electron_ion_mapping = []
        self.update_with_dict(config_dict, **kwargs)

    def __setitem__(self, key, value):
        if key == 'name':
            try:
                self.set_by_molecule_name(value)
            except ValueError:
                vars(self)[key] = value
                # print(f"Warning: Unknown chemical composition {value}") # Unknown chemical composition
        else:
            vars(self)[key] = value

    def set_by_chemical_element(self, element):
        """
        Sets the physical parameters for a single atom calculation based on the element symbol in the periodic table.

        This function sets the name, ion_charge, ion_positions, number of electrons and spin.
        Currently only light elements up to Neon are implemented.
        Args:
            element (str): Which atom/element to calculate

        Returns:
            (bool): Whether
        """
        elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']
        if element in elements:
            Z = elements.index(element) + 1
            self.energy_baseline = [-0.6, -3.0, -8.0, -15.0, -25.0, -38.0, -55, -75, -100, -129][Z - 1]
            self.ion_charges = [Z]
            self.ion_positions = [[0.0, 0.0, 0.0]]
            self.n_spin_up = [1, 1, 2, 2, 3, 4, 5, 5, 5, 5][Z - 1]
            self.n_electrons = sum(self.ion_charges)
        else:
            raise KeyError(f"Unknown chemical molecule / element: {element}")

    def fill_missing_values_with_defaults(self):
        if self.n_electrons is None:
            self.n_electrons = sum(self.ion_charges)
        if self.n_spin_up is None:
            self.n_spin_up = (self.n_electrons+1)//2

        # Distribute the electrons and spins equally across nuclei
        if len(self.electron_ion_mapping) == 0:
            electrons = self.ion_charges.copy()
            i = 0
            N_ions = len(self.ion_charges)
            while self.n_electrons > sum(electrons):
                electrons[i] += 1
                i = (i + 1) % N_ions
            while self.n_electrons < sum(electrons):
                if electrons[i] > 0:
                    electrons[i] -= 1
                i = (i + 1) % N_ions

            spin_up_ion_mapping = []
            spin_dn_ion_mapping = []
            nr_of_el_up_to_be_assigned = self.n_spin_up
            nr_of_el_dn_to_be_assigned = self.n_electrons - self.n_spin_up

            while sum(electrons) > 0:
                if electrons[i] > 0:
                    electrons[i] -= 1
                    if nr_of_el_up_to_be_assigned >= nr_of_el_dn_to_be_assigned:
                        spin_up_ion_mapping.append(i)
                    else:
                        spin_dn_ion_mapping.append(i)
                i = (i + 1) % N_ions
            self.electron_ion_mapping = spin_up_ion_mapping + spin_dn_ion_mapping

    def set_by_molecule_name(self, molecule):
        """
        Set the physical parameters of a calculation based on the name of a molecule.
        Args:
            molecule (str): Name of molecule or element

        Returns:
            None
        """
        if molecule == "H2":
            self.ion_charges = [1, 1]
            self.ion_positions = [[0.0, 0.0, 0.0], [1.40108, 0.0, 0.0]]
            self.energy_baseline = -1.5
            self.n_spin_up = 1
        elif molecule == "H2p":
            self.ion_charges = [1, 1]
            self.ion_positions = [[0.0, 0.0, 0.0], [1.9, 0.0, 0.0]]
            self.energy_baseline = -0.6
            self.n_spin_up = 1
            self.n_electrons = 1
        elif molecule == "LiH":
            self.ion_charges = [3, 1]
            self.ion_positions = [[0.0, 0.0, 0.0], [3.015, 0.0, 0.0]]
            self.energy_baseline = -8.5
            self.n_spin_up = 2
        elif molecule == "Li2":
            self.ion_charges = [3,3]
            self.ion_positions = [[0.0, 0.0, 0.0], [5.051048595, 0.0, 0.0]]
            self.energy_baseline = -15.0
            self.n_spin_up = 3
            self.electron_ion_mapping = [0,0,1, 0,1,1]
        elif molecule == "Be2":
            self.ion_charges = [4, 4]
            self.ion_positions = [[0.0, 0.0, 0.0], [4.629828672, 0.0, 0.0]]
            self.energy_baseline = -29.3
            self.n_spin_up = 4
            self.electron_ion_mapping = [0,0,1,1, 0,0,1,1]
        elif molecule == "B2":
            self.ion_charges = [5, 5]
            self.ion_positions = [[0.0, 0.0, 0.0], [3.004664322, 0.0, 0.0]]
            self.energy_baseline = -49.3
            self.n_spin_up = 4
            self.electron_ion_mapping = [0,0,1,1, 0,0,0,1,1,1]
        elif molecule == "N2":
            self.ion_charges = [7, 7]
            self.ion_positions = [[0.0, 0.0, 0.0], [2.06800, 0.0, 0.0]]
            self.energy_baseline = -109
            self.n_spin_up = 7
            self.electron_ion_mapping = [0,0,0,0,1,1,1, 0,0,0,1,1,1,1]
        elif molecule == "H4Rect":
            self.ion_charges = [1,1,1,1]
            a = 1.4
            self.ion_positions = [[0,0,0],[a,0,0],[0,a,0],[0,0,0]]
            self.energy_baseline = 0
            self.n_spin_up = 2
        elif molecule == "H4plus":
            self.ion_charges = [1,1,1,1]
            a = 1.4
            self.ion_positions = [[0,0,0],[a,0,0],[0,a,0],[0,0,0]]
            self.energy_baseline = 0
            self.n_spin_up = 2
            self.n_electrons = 3
            self.electron_ion_mapping = [0,3, 1]
        elif molecule == "H3plus":
            self.ion_charges = [1,1,1]
            a = 1.4
            self.ion_positions = [[0,0,0],[a,0,0],[0,a,0]]
            self.energy_baseline = 0
            self.n_spin_up = 1
            self.n_electrons = 2
            self.electron_ion_mapping = [0,1]
        elif molecule.startswith("HChain"):
            n = int(molecule.replace('HChain',''))
            self.ion_charges = [1]*n
            a = 2.0
            self.ion_positions = [[i*a,0,0] for i in range(n)]
            self.energy_baseline = -0.5*n
            self.n_spin_up = (n+1)//2
            self.electron_ion_mapping = list(range(0,n,2)) + list(range(1,n,2)) # initialize antiferromagnetically
        else:
            self.set_by_chemical_element(molecule)

        if self.n_electrons is None:
            self.n_electrons = sum(self.ion_charges)
        self.name = molecule
        if len(self.electron_ion_mapping) == 0:
            for i, charge in enumerate(self.ion_charges):
                self.electron_ion_mapping += [i] * charge


class DeepErwinModelConfig(Configuration):
    def __init__(self, config_dict=None, **kwargs):
        super().__init__()
        self.name = "DeepErwinModel"
        self.use_residual = False
        self.embed = SimpleSchnetConfig()

        self.use_symmetric_part = True
        self.use_backflow_factor = True
        self.use_backflow_shift = True

        self.n_hidden_fit = [40, 40] # Jastrow factor
        self.n_hidden_backflow_factor_general = [40, 40]
        self.n_hidden_backflow_factor_orbitals = []
        self.n_hidden_backflow_shift = [40, 40]

        self.initial_symm_weight = 0.0
        self.initial_backflow_factor_weight = -2.0
        self.initial_backflow_shift_weight = -4.0
        self.decaying_shift = DecayingShiftConfig()

        self.initializer_name = 'glorot_uniform'
        self.initializer_scale = 0.05

        self.reuse_weights = None #type: ReuseWeightsConfig #list or single entry of ReuseWeightsConfig instances

        self.n_rbf_features = 16
        self.slatermodel = CASSCFModelConfig()
        self.non_trainable_weights = []
        self.update_with_dict(config_dict, **kwargs)

    # docstr-coverage:inherited
    def setDefaults(self, quality='default'):
        if quality == 'default':
            pass
        elif quality == 'minimal':
            self.n_hidden_embed = [5]
            self.n_hidden_backflow_shift = [5]
            self.n_hidden_backflow_factor = [5]
            self.n_hidden_fit = [5]
        else:
            raise ValueError(f"Unknown quality type {quality} for configuration class {type(self).__name__}")

class DecayingShiftConfig(Configuration):
    def __init__(self, config_dict=None, **kwargs):
        super().__init__()
        self.name = "DecayingShiftConfig"
        self.use = True
        self.initial_scale = 0.5
        self.trainable_scale = False
        self.update_with_dict(config_dict, **kwargs)

class SimpleSchnetConfig(Configuration):
    def __init__(self, config_dict=None, **kwargs):
        super().__init__()
        self.name = "SimpleSchnet"
        self.use = True
        self.n_iterations = 2
        self.use_g_function = True
        self.embedding_dim = 64
        self.n_hidden_w = [40, 40]
        self.n_hidden_h = [40, 40]
        self.n_hidden_g = [40]
        self.update_with_dict(config_dict, **kwargs)

    # docstr-coverage:inherited
    def setDefaults(self, quality='default'):
        if quality == 'default':
            pass
        elif quality == 'minimal':
            self.n_iterations = 1
            self.embedding_dim = 10
            self.n_hidden_w = [5]
            self.n_hidden_h = [5]
            self.n_hidden_g = [5]
        else:
            raise ValueError(f"Unknown quality type {quality} for configuration class {type(self).__name__}")

class SlaterModelConfig(Configuration):
    def __init__(self, config_dict=None, **kwargs):
        super().__init__()
        self.name = "SlaterModel"
        self.basis = '6-311G'
        self.use_orbital_cusp_correction = True
        self.scale_r_cusp = 1.0
        self.use_regularization = False
        self.regularization_decay = 5.0
        self.svd_log_shift = 1e-8
        self.log_shift = 1e-8
        self.update_with_dict(config_dict, **kwargs)

class CASSCFModelConfig(SlaterModelConfig):
    def __init__(self, config_dict=None, **kwargs):
        super().__init__()
        self.name = "CASSCFModel"
        self.n_determinants = 20
        self.n_active_orbitals = None
        self.n_cas_electrons = None
        self.update_with_dict(config_dict, **kwargs)

    def set_defaults_by_molecule_name(self, molecule):
        """
        Set reasonable, computationally cheap defaults for the number of active electrons and orbitals.

        Defaults have been tested for the default basis set (6-311G), but might be fully adequate for other basis sets.
        Args:
            molecule (str): Name of molecule

        Returns:
            (bool): True if defaults could be set
        """
        defaults = dict(He=(2,2), Li=(9,3), Be=(8,2), LiH=(10,2), B=(12,3), C=(12,4), N=(12,5), O=(12,6), F=(12,7), Ne=(12,8),
                        H2=(4,2), Li2=(16,2), Be2=(16,4), B2=(16,6), C2=(16,8), N2=(16,6),
                        H2p=(1,1), H3plus=(9,2), H4plus=(12,3), H4Rect=(12, 4), HChain10=(10, 10))
        for n in range(2,9+1,2):
            defaults[f'HChain{n}'] = (2*n, n)

        if molecule in defaults:
            self.n_active_orbitals, self.n_cas_electrons = defaults[molecule]
            getLogger().info(f"No CAS settings supplied. Using default for {molecule}: {self.n_active_orbitals} orbitals, {self.n_cas_electrons} electrons")
            return True
        else:
            getLogger().warning(f"No hardcoded CAS defaults available for molecule: {molecule}. Defaulting to n_cas_electrons=n_valence_electrons, n_active_orbitals=2*n_cas_electrons")
            return False

    def set_defaults(self, ion_charges, n_electrons):
        if self.n_cas_electrons is None:
            VALENCE = {1:1, 2:2, 3:1, 4:2, 5:3, 6:4, 7:5, 8:6, 9:7, 12:8} # Mapping nuclear charge to number of valence electrons
            self.n_cas_electrons = min(sum(VALENCE[Z] for Z in ion_charges), n_electrons)
        if self.n_active_orbitals is None:
            self.n_active_orbitals = 2 * self.n_cas_electrons

class IntegrationConfig(Configuration):
    def __init__(self, config_dict=None, **kwargs):
        super().__init__()
        self.train = MCMCConfiguration(n_inter_steps=5, max_age=10)
        self.valid = MCMCConfiguration(n_walkers=512)
        self.eval = MCMCConfiguration(n_walkers=2048, n_burnin_steps=100) # fewer burn-in steps b/c rainingwalker reuse
        self.update_with_dict(config_dict, **kwargs)

    # docstr-coverage:inherited
    def setDefaults(self, quality='default'):
        self.train = MCMCConfiguration(quality=quality)
        self.valid = MCMCConfiguration(quality=quality)
        self.eval = MCMCConfiguration(quality=quality, n_walkers=2048)
        if quality == 'default':
            pass
        elif quality == 'minimal':
            self.eval.n_walkers = 50
        else:
            raise ValueError(f"Unknown quality type {quality} for class {type(self).__name__}")

class MCMCConfiguration(Configuration):
    def __init__(self, config_dict=None, **kwargs):
        super().__init__()
        self.n_walkers = 2048
        self.n_inter_steps = 10
        self.n_burnin_steps = 1000
        self.init_scale = 0.1
        self.max_scale = 0.5
        self.min_scale = 1e-2
        self.max_age = 100
        self.target_acceptance_rate = 0.5
        self.use_local_stepsize = True
        self.local_stepsize_constant = 0.2
        self.update_with_dict(config_dict, **kwargs)

    # docstr-coverage:inherited
    def setDefaults(self, quality='default'):
        if quality == 'default':
            pass
        elif quality == 'minimal':
            self.n_walkers = 100
            self.n_inter_steps = 1
            self.n_burnin_steps = 2
        elif quality == 'high':
            self.n_walkers = 2048
            self.n_inter_steps = 100
            self.n_burnin_steps = 1000
        else:
            raise ValueError(f"Unknown quality type {quality} for configuration class {type(self).__name__}")


class OptimizationConfig(Configuration):
    def __init__(self, config_dict=None, **kwargs):
        super().__init__()
        self.learning_rate = 1.5e-3
        self.n_epochs = 10000
        self.batch_size = 512
        self.clip_by = 5
        self.soft_clipping = True
        self.clip_max = 10.0
        self.clip_min = 0.01
        self.optimizer = "Adam"
        self.shuffle = True
        self.max_nan_batches = 1000
        self.adaptiveLR = InverseLRConfig() #alternative: ExponentialLRConfig()
        self.update_with_dict(config_dict, **kwargs)

    # docstr-coverage:inherited
    def setDefaults(self, quality='default'):
        if quality == 'default':
            pass
        elif quality == 'minimal':
            self.n_epochs = 10
            self.batch_size = 50
        elif quality == 'high':
            self.n_epochs = 2048
            self.batch_size = 128
            self.epochs_per_mcmc_step = 1
        else:
            raise ValueError(f"Unknown quality type {quality} for configuration class {type(self).__name__}")

class AdaptiveLRConfig(Configuration):
    def __init__(self, config_dict=None, **kwargs):
        super().__init__()
        self.name = None
        self.update_with_dict(config_dict, **kwargs)

class PatienceLRConfig(Configuration):
    def __init__(self, config_dict=None, **kwargs):
        super().__init__()
        self.name = 'patience'
        self.use = False
        self.monitor = 'running_avg_mean_energies_train'
        self.factor_dec = 0.5
        self.factor_inc = 1.25
        self.patience_dec = 50
        self.patience_inc = 50
        self.min_delta = 0.02
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.update_with_dict(config_dict, **kwargs)

class ExponentialLRConfig(Configuration):
    def __init__(self, config_dict=None, **kwargs):
        super().__init__()
        self.name = 'exponential'
        self.fac_start = 3.0
        self.fac_end = 0.2
        self.epochs_warmup = 32
        self.update_with_dict(config_dict, **kwargs)

class InverseLRConfig(Configuration):
    def __init__(self, config_dict=None, **kwargs):
        super().__init__()
        self.name = 'inverse'
        self.decay_time = 1000
        self.update_with_dict(config_dict, **kwargs)

class OutputConfig(Configuration):
    def __init__(self, config_dict=None, **kwargs):
        super().__init__()
        self.tb_path = './tb'
        self.log_path = './erwin.log'
        self.logger_name = 'erwin'
        self.log_level_console = logging.INFO
        self.log_level_file = logging.DEBUG
        self.use_profiler = False
        self.create_graph = False
        self.n_skip_train = 100
        self.no_skip_train_for_initial_epochs = 10
        self.n_skip_eval = 25
        self.comment = ""
        self.code_version = "Unknown"
        self.store_debug_data = True
        self.copy_source_code = False
        self.store_walkers_eval = False
        self.compute_histogram = False
        self.histogram_n_bins = 512
        self.E_HF = 0.0
        self.E_casscf = 0.0
        self.update_with_dict(config_dict, **kwargs)

class ForceEvaluationConfig(Configuration):
    def __init__(self, config_dict=None, **kwargs):
        super().__init__()
        self.calculate = True
        self.R_core = 0.5
        self.R_cut = 0.1
        self.target_std_err_mHa = 0.2
        self.use_polynomial = True
        self.use_antithetic_sampling = True
        self.j_max = 4 # maximum degree of polynomial force variance reduction
        self.update_with_dict(config_dict, **kwargs)

class EvaluationConfig(Configuration):
    def __init__(self, config_dict=None, **kwargs):
        super().__init__()
        self.n_epochs_min = 500
        self.n_epochs_max = 2000
        self.target_std_err_mHa = 0.5
        self.reuse_training_walkers = True
        self.forces = ForceEvaluationConfig()
        self.update_with_dict(config_dict, **kwargs)

    # docstr-coverage:inherited
    def setDefaults(self, quality='default'):
        if quality == 'default':
            pass
        elif quality == 'minimal':
            self.n_epochs_min = 5
            self.n_epochs_max = 10
        elif quality == 'high':
            self.target_std_err_mHa = 2.0
            self.n_epochs_max = 20000
            self.n_epochs_min = 2000
        else:
            raise ValueError(f"Unknown quality type {quality} for configuration class {type(self).__name__}")

class ReuseWeightsConfig(Configuration):
    def __init__(self, config_dict=None, **kwargs):
        super().__init__()
        self.reuse_dirs = None #single directory or list of directories that contains wavefunctions
        self.weights = [] #list of weights wich are to be reused
        self.interpolation = "nearest" # specify interpolation mode
        self.update_with_dict(config_dict, **kwargs)


