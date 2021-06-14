import numpy as np
import pyscf
import pyscf.mcscf
import tensorflow as tf
from .SlaterDetModel import SlaterDetModel
from deeperwin import main as dpe
from deeperwin.utilities import erwinConfiguration


def _int_to_spin_tuple(x):
    if type(x) == int:
        return (x,)*2
    else:
        return x

class CASSCFModel(SlaterDetModel):
    """
    Model that wraps the quantum chemistry method CASSCF to provide an initial set of determinants in the the DeepErwinModel.
    The core logic is contained in :meth:`~.CASSCFModel.CASSCFModel.build_determinants_and_weights`, the actual evaluation of the determinants happens in the call method of the base class :class:`~.SlaterDetModel.SlaterDetModel`
    """
    def __init__(self, config, name='CASSCFModel', history_only=False, **kwargs):
        r"""Create a new CASSCFModel that calculates :math:`\log(\psi^2)` based on a sum of slater determinants which are obtained by Complete Active Space Self Consistent Field (CASSCF)"""
        try:
            self.model_config = config.model.slatermodel # type: erwinConfiguration.CASSCFModelConfig
        except AttributeError:
            self.model_config = config.model # type: erwinConfiguration.CASSCFModelConfig
        super().__init__(config, name=name, model_config=self.model_config, **kwargs)
        if (self.model_config.n_active_orbitals is None) or (self.model_config.n_cas_electrons is None):
            if not self.model_config.set_defaults_by_molecule_name(config.physical.name):
                self.model_config.set_defaults(config.physical.ion_charges, config.physical.n_electrons)

        self.n_active_orbitals = self.model_config.n_active_orbitals
        self.n_cas_electrons = self.model_config.n_cas_electrons
        if not history_only:
            self.finishInitialization()

    def _get_orbitals_by_cas_type(self):
        """
        Splits orbitals into fixed orbitals (that are either always occupied, or always unoccupied) and active orbitals (that are occupied in some determinants, and unoccupied in others).

        Returns:
            tuple containing

            - **fixed_orbitals** (list of 2 lists): For each spin, a list of indices of fixed orbitals
            - **active_orbitals** (list of 2 lists): For each spin, a list of indices of active orbitals
        """
        n_core = _int_to_spin_tuple(self.casscf.ncore)
        n_cas = _int_to_spin_tuple(self.casscf.ncas)

        active_orbitals = [list(range(n_core[s], n_core[s]+n_cas[s])) for s in range(2)]
        fixed_orbitals = [list(range(n_core[s])) for s in range(2)]
        return fixed_orbitals, active_orbitals

    def _get_orbital_indices(self):
        """
        Parse the output of the pySCF CASSCF calculation to determine which orbitals are included in which determinant.

        Returns:
            (list of 2 np.arrays): First array for spin-up, second array for spin-down. Each array has shape [N_determinants x n_electrons_of_spin] and contains the indices of occupied orbitals in each determinant.
        """
        fixed, active = self._get_orbitals_by_cas_type()

        nelcas = _int_to_spin_tuple(self.casscf.nelecas)
        occ_up = pyscf.fci.cistring._gen_occslst(active[0], nelcas[0])
        occ_down = pyscf.fci.cistring._gen_occslst(active[1], nelcas[1])

        orbitals_up = []
        orbitals_dn = []
        for o_up in occ_up:
            for o_dn in occ_down:
                orbitals_up.append(fixed[0] + list(o_up))
                orbitals_dn.append(fixed[1] + list(o_dn))
        return [np.array(orbitals_up, dtype=np.int), np.array(orbitals_dn, dtype=np.int)]

    def _truncate_determinants(self, ind_orbitals, ci_weights):
        """
        Sorts the determinants obtained by CASSCF by the absolute value of their CI coefficients and only keeps the most important determinants
        This limits the complexity of the CASSCF model and improves troughput of the surrounding DeepErwinModel.
        The number of determinants to keep is set by *model_config.n_determinants*

        Args:
            ind_orbitals (list of np.array): 2 integer arrays of shape [N x n_electrons_of_spin] that contain the indices of molecular orbitals used in each of the N determinants of that spin
            ci_weights (np.array): Weight of each determinant. Numpy array of shape N, where N is the number of determinants.

        Returns:
            tuple containing

            - **ind_orbitals** (list): Truncated list of orbitals per determinant. List of 2 integer arrays, with same format as input but truncated to N_dets <= N
            - **ci_weights**: (np.array): Truncated determinant weights. Same format as input but truncated to N_dets <= N
        """
        n_dets = self.model_config.n_determinants
        if n_dets < 1:
            return ind_orbitals, ci_weights
        # Find indices of largest determinants
        if n_dets > len(ci_weights):
            self.logger.warning(f"Specified number of determinants ({n_dets}) exceeds determinants in CAS ({len(ci_weights)}). Using all CAS determinants.")
            n_dets = len(ci_weights)
            self.model_config.n_determinants = n_dets
        ind = np.argsort(ci_weights**2)[::-1] # get indices of (absolute) largest ci weights
        ind = ind[:n_dets]

        # Truncate determinants and weights
        ci_weights = ci_weights[ind]
        ci_weights /= np.sum(ci_weights**2) # renormalize
        for spin in range(2):
            # mo_coeff[spin] = mo_coeff[spin][ind]
            ind_orbitals[spin] = ind_orbitals[spin][ind]
        return ind_orbitals, ci_weights

    def build_determinants_and_weights(self):
        """
        Calculates an initial set of Slater-determinants and their corresponding CI-weights based on a CASSCF calculation (Complete Active Space Self Consistent Field).
        Runs a Hartree-Fock calculation to get an initial set of orbitals and then runs a CASSCF calculation based on these orbitals.

        All results are stored in the attributes :attr:`CASSCFModel.hf`, :attr:`CASSCFModel.casscf`, :attr:`CASSCFModel.mo_coeff`, :attr:`CASSCFModel.ind_orbitals`, and :attr:`CASSCFModel.ci_weights`.

        Returns:
            None
        """
        self.hf = pyscf.scf.HF(self.molecule)
        self.hf.verbose = 0  # suppress output to console
        self.logger.debug(f"Solving Hartree-Fock...")
        self.hf.kernel()
        self.logger.info(f"Solved Hartree-Fock: E = {self.hf.e_tot:.3f}")

        if self.n_electrons > 1:
            self.casscf = pyscf.mcscf.UCASSCF(self.hf, self.n_active_orbitals, self.n_cas_electrons)
            self.logger.debug(f"Solving CAS SCF...")
            self.casscf.kernel()
            self.logger.info(f"Solved CAS SCF: E = {self.casscf.e_tot:.3f}")

            # If CASSCF calculation was spin-unrestricted, use same orbitals for both spin channels
            if isinstance(self.casscf.mo_coeff, tuple) or len(self.casscf.mo_coeff.shape) == 3:
                mo_coeff = [self.casscf.mo_coeff[0], self.casscf.mo_coeff[1]]
            else:
                mo_coeff = [self.casscf.mo_coeff, self.casscf.mo_coeff]
            assert len(mo_coeff) == 2
            for m in mo_coeff:
                assert m.shape[0] == m.shape[1]
            ind_orbitals = self._get_orbital_indices()
            self.ind_orbitals, ci_weights = self._truncate_determinants(ind_orbitals, self.casscf.ci.flatten())
        else:
            self.ind_orbitals = [[[0]],[[]]]
            ci_weights = [1.0]
            mo_coeff = [self.hf.mo_coeff, self.hf.mo_coeff]
            self.casscf = None

        self.mo_coeff = [tf.Variable(m, dtype=dpe.DTYPE, trainable=False, name="mo_coeff") for m in mo_coeff]
        self.ind_orbitals = [tf.Variable(ind, trainable=False, name="ind_orbitals") for ind in self.ind_orbitals]
        self.ci_weights = tf.Variable(ci_weights, dtype=dpe.DTYPE, name="ci_weigths")

    @property
    def total_energy(self):
        """Full CASSCF energy, without truncation of determinants"""
        if self.casscf is not None:
            return self.casscf.e_tot
        else:
            return self.hf.e_tot

    @property
    def total_energy_hf(self):
        """Hartree Fock energy (i.e. energy using a single determinant)"""
        return self.hf.e_tot

if __name__ == '__main__':
    tf.config.experimental_run_functions_eagerly(True)

    config = erwinConfiguration.DefaultConfig(quality='minimal')
    config.model = erwinConfiguration.CASSCFModelConfig()
    config.physical.set_by_molecule_name('LiH')
    config.physical.ion_positions = [[0,0,0],[2.7,0,0]]
    config.evaluation.n_epochs_max = 2000
    config.evaluation.n_epochs_min = 2000
    config.output.n_skip_eval = 1
    config.evaluation.target_std_err_mHa = 0.0
    config.integration.eval.use_local_stepsize = False
    wf = dpe.WaveFunction(config)
    wf.model.buildCuspCorrection()

    wf.compile_evaluation()
    wf.evaluate()

    #%%
    # n_burnin = 400
    # E_eval = wf.history['mean_energies_eval'][n_burnin:]
    #
    # plt.close("all")
    # plt.subplot(2,2,1)
    # plt.plot(wf.history['mean_energies_eval'])
    # plt.axhline(np.mean(E_eval), color='k')
    # plt.axhline(wf.model.total_energy_hf, color='r')
    # delta = 1000*(np.mean(E_eval) - wf.model.total_energy_hf)
    # print(delta)
    #
    # plt.subplot(2,2,2)
    # corr = ut.getAutoCorrelation(E_eval)
    # plt.title("autocorr")
    # plt.plot(corr)
    #
    # plt.subplot(2,2,3)
    # plt.title("scale")
    # plt.plot(wf.history['mh_scale_eval'])
    #
    # plt.subplot(2, 2, 4)
    # plt.title("max age")
    # plt.plot(wf.history['mh_age_max_eval'])





    # plt.close("all")
    # elements = ['He', 'Li', 'Be', 'B', 'C', 'LiH']
    # fig, axes = plt.subplots(2,len(elements), figsize=(16,10))
    # for j_element, element in enumerate(elements):
    #     config = erwinConfiguration.DefaultConfig()
    #     config.model = erwinConfiguration.CASSCFModelConfig(basis='6-31G', n_determinants=1)
    #     config.physical.set_by_chemical_element(element)
    #     wf = dpe.WaveFunction(config)
    #     wf.model.buildCuspCorrection()
    #     N = 200
    #     r = np.linspace(-0.5, 0.5, N)
    #     x = np.zeros([N, config.physical.n_electrons, 3]) + 1e-3
    #     x[:,1:,:] += np.random.randn(config.physical.n_electrons-1, 3)
    #     x[:, 0, 0] = r
    #     x = tf.constant(x, dtype=dpe.DTYPE)
    #
    #     for i,cusp,label in zip(range(2), [False, True], ["Original orbitals", "Cusp-corrected orbitals"]):
    #         wf.model.model_config.use_orbital_cusp_correction = cusp
    #         logpsi = wf.model.log_squared(x).numpy()
    #         Eloc = wf.model(x).numpy()
    #
    #         axes[0][j_element].plot(r, logpsi, label=label)
    #         axes[1][j_element].plot(r, Eloc, label=label)
    #         axes[1][j_element].legend()
    #         axes[0][j_element].set_title(element)
    #         if j_element == 0:
    #             axes[0][j_element].set_ylabel("log psi^2", fontsize=14)
    #             axes[1][j_element].set_ylabel("E_loc", fontsize=14)
    #         axes[1][j_element].set_ylim([-100, 50])
    #
    # plt.suptitle("Psi and Eloc as a function of distance electron-nucleus")
    # plt.savefig(ut.RESULTS_DIR + f'/Orbital_cusp_corrections.png')
