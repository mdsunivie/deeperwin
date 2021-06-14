import abc
import numpy as np
import pyscf
import tensorflow as tf

import deeperwin.models.base
from deeperwin import main as dpe
from tensorflow.keras import Model

class GaussianOrbital(Model):
    r"""
    Representation of a single Gaussian-type-orbital, as it is typically used for quantum chemistry basis sets.
    Each orbital has the form :math:`x^{l_x} y^{l_y} z^{l_z} R(r)` where :math:`R(r) = \sum_i c_i exp(-\alpha_i r^2)`.

    The orbitals can be evaluated by :meth:`~models.SlaterDetModel.GaussianOrbital.call`.
    """
    def __init__(self, ind_nucleus, R0, alpha, weights, exponents, label):
        """
        Build a new GTO from given basis set coefficients

        Args:
            ind_nucleus (int): Which nucleus of the molecule this basis set referse to
            R0 (array-like): Position of this nucleus in [x,y,z]
            alpha (array-like): Exponents for the gaussians
            weights: Weights for the individual gaussians. Must be of same length as alpha
            exponents (array of ints): Exponents lx, ly, lz of this orbital. e.g. [0,0,0] corresponds to an s-orbital, [1,0,0] to a px-orbital
            label (str): Type of orbital (e.g. 's', 'p', 'd', etc.)
        """
        super().__init__()
        self.ind_nucleus = ind_nucleus
        self.R0_np = np.array(R0)
        self.alpha_np = np.array(alpha)
        self.weights_np = np.array(weights)
        self.exponents_np = np.array(exponents)
        self._normalize_weights()

        self._R0 = tf.Variable(self.R0_np, dtype=dpe.DTYPE, trainable=False, name = "_R0_GaussianOrbital")
        self._alpha = tf.Variable(self.alpha_np, dtype=dpe.DTYPE, trainable=False, name = "_alpha_GaussianOrbital")
        self._weights = tf.Variable(self.weights_np, dtype=dpe.DTYPE, trainable=False, name = "_weights_GaussianOrbital")
        self._exponents = tf.Variable(self.exponents_np, dtype=dpe.DTYPE, trainable=False, name = "_exponents_GaussianOrbital")
        self.label = label
        self.rc = 0.0
        self.use_sto_cusp = False

    def _normalize_weights(self):
        """Normalizes all weights to ensure that the norm of this orbital is 1"""
        l_tot = np.sum(self.exponents_np)
        fac_alpha = (2 * self.alpha_np / np.pi) ** (3 / 4) * (8 * self.alpha_np) ** (l_tot / 2)

        fac = np.array([np.math.factorial(x) for x in self.exponents_np])
        fac2 = np.array([np.math.factorial(2 * x) for x in self.exponents_np])
        fac_exponent = np.sqrt(np.prod(fac) / np.prod(fac2))
        factor = fac_alpha * fac_exponent
        self.weights_np *= factor

    def _eval_radial_func(self, r, deriv=0):
        """
        Evaluates this orbital at a distance r from the nucleus, which only makes sense for s-type orbitals.
        This function is used when evaluation the orbital-cusp-corrections.

        Args:
            r (float or np.array): Distance from the nucleus where the orbital is evaluated
            deriv (int): Integer in [0,1,2], specifying wether the orbital, its 1st or 2nd derivate should be calculated

        Returns:
            float or np.array: Value of orbital (or its derivative) at given distance
        """
        assert 's' in self.label[2], "Only implemented for s-type orbitals"
        if deriv == 0:
            fac = 1.0
        elif deriv == 1:
            fac = - 2 * self.alpha_np[:, np.newaxis] * r
        elif deriv == 2:
            fac = 4 * (self.alpha_np[:, np.newaxis] * r)**2 - 2 * self.alpha_np[:, np.newaxis]
        phi = np.sum(fac * self.weights_np[:, np.newaxis] * tf.exp(-self.alpha_np[:, np.newaxis] * r**2), axis=0)
        if type(r) == float:
            return phi[0]
        else:
            return phi

    def call(self, x):
        """
        Evaluates the orbital

        Args:
            x (tf.Tensor): [batch_dims x 3] position in cartesian coordinates

        Returns:
            tf.Tensor: Orbital value phi(x)
        """
        x_centered = x - self._R0  # calculate x, relative to the atom position
        r_sqr = tf.reduce_sum(tf.square(x_centered), axis=-1, keepdims=True)
        pre_fac = tf.reduce_prod(x_centered ** self._exponents, axis=-1)
        phi_gto = pre_fac * tf.reduce_sum(self._weights * tf.exp(-self._alpha * r_sqr), axis=-1)
        return phi_gto
#
class SlaterDetModel(deeperwin.models.base.WFModel, abc.ABC):
    """
    Abstract class that defines (almost) parameterless models that are based on classical quantum chemistry methods,
    e.g. Hartree-Fock or Complete-Active-Space Self-consistent-field.
    This method primarily provides:

    * Initialization of basis-functions / orbitals
    * Cusp-correction of orbitals
    * Evaluation of determinants

    Any child-class must in particular provide an implementation of :meth:`~models.SlaterDetModel.SlaterDetModel.build_determinants_and_weights` to provide a set of approximate slater determinants and CI coefficients. This could for example be HF, FCI, CASSCF, etc.

    For a specific implementation example refer to :class:`model.CASSCFModel.CASSCFModel`
    """
    def __init__(self, config, name='SlaterDetModel', model_config=None, **kwargs):
        super().__init__(config, name=name, model_config=model_config, **kwargs)
        self.model_config = model_config # type: erwinConfiguration.SlaterModelConfig
        self.physical_config = config.physical # type: erwinConfiguration.PhysicalConfig
        self.regularization_decay_length = self.model_config.regularization_decay
        self._build_pyscf_molecule()
        self.atomic_orbitals = []
        self.initBasis()

    def finishInitialization(self):
        """
        Method to be called by child classes after their constructor.
        Applies orbital cusp correction and sets the energy clipping based on the high-level energy estimate given by the slater method.

        Returns:
            None
        """
        self.build_determinants_and_weights()
        if self.model_config.use_orbital_cusp_correction:
            self.buildCuspCorrection()
        # self.energy_clipping_center.assign(self.total_energy)

    def _get_STO_polynomial(self, i_nuc, j_mo, spin):
        """
        Builds a polynomial representing a Slater-type-orbital for nucleus i, orbital j, with a given spin.

        The orbital is expressed as :math:`\text{shift} + \text{sign} exp(\sum_k c_k r^k)` and this function computes the polynomial cofficients :math:`c_k`
        Args:
            i_nuc (int): Index of nucleus
            j_mo (int): Index of orbital
            spin (int): Spin up (0) or down (1)

        Returns:
            tuple containing

            - **shift**(float): shift of STO
            - **sign**(float): sign of STO (+- 1.0)
            - **coeffs**(np.array): coeffs :math:`c_k`of polynomial
            - **mo_coeffs_1s**(np.array): Molecular orbital coefficients of other s-type orbitals to be used inside the cusp radius
        """
        mo_coff = self.mo_coeff[spin][:, j_mo]
        R = self.physical_config.ion_positions[i_nuc]
        Z = self.physical_config.ion_charges[i_nuc]
        is_s_type = np.array([1 if (ao.ind_nucleus == i_nuc) and ('s' in ao.label[2]) else 0 for ao in self.atomic_orbitals])
        atomic_orbitals = np.array([ao.call(R) for ao in self.atomic_orbitals])
        mo_coeffs_1s = mo_coff * is_s_type
        mo_coeffs_others = mo_coff * (1-is_s_type)
        phi_others = np.dot(atomic_orbitals, mo_coeffs_others)

        rc = self.r_cusp[i_nuc]
        g_rc = np.zeros(3) #phi(rc), phi'(rc), phi''(rc) for the 1s orbitals
        g_0 = 0.0
        for i, ao in enumerate(self.atomic_orbitals):
            if is_s_type[i] == 1:
                for deriv in range(3):
                    g_rc[deriv] += ao._eval_radial_func(rc, deriv) * mo_coff[i]
                g_0 += (ao._eval_radial_func(0.0) * mo_coff[i]).numpy()
        if (np.abs(g_rc[0]) < 1e-6) or (np.abs(g_0) < 1e-6):
            return 0.0, 0.0, np.zeros(5), np.zeros(len(mo_coeffs_1s))
        else:
            shift, sign, coeffs, El_opt = self._get_optimal_cusp_poly_coefficients(g_0, phi_others, g_rc, Z, rc)
            if El_opt > 100:
                self.logger.warning(f"Potentially poor cusp correction for nucleus {i_nuc}, molecular orbital: {j_mo}; g_rc: {g_rc}; g_0: {g_0}, mo:{mo_coff}")
            return shift, sign, coeffs, mo_coeffs_1s

    def _get_cusp_poly_coefficients(self, phi0, phi_others, g, Z, rc):
        """
        Returns the parameters of a specific Slater-type orbital that matches all boundary conditions at r=0 and r=rc:

        * :math:`\phi(0) = \phi_0`
        * :math:`\phi'(0) = -Z*(\phi_0 + \phi_\text{others})`
        * :math:`\phi(r_c) = R(r_c)`
        * :math:`\phi'(r_c) = R'(r_c)`
        * :math:`\phi''(r_c) = R''(r_c)`

        Args:
            phi0 (float): Enforced value of phi at r=0
            phi_others (float): Sum of other orbitals at r=0
            g (list): Evaluation of radial function R, its 1st and 2nd derivative at r=rc
            Z (float): Nuclear charge
            rc (float): Cusp radius at which boundary conditions shall apply

        Returns:
            tuple containing parameters of the orbital (see :meth:`models.SlaterDetModel._get_cusp_poly_coefficients`)

            - **sign** (float): sign of STO (+- 1.0)
            - **shift** (float): shift of STO
            - **coeffs** (np.array): coeffs :math:`c_k`of polynomial
            - **mo_coeffs_1s** (np.array): Molecular orbital coefficients of other s-type orbitals to be used inside the cusp radius
        """
        sign = np.sign(phi0 - g[0])
        shift = g[0] + 1.0*(g[0] - phi0)

        R_rc = g[0] - shift
        R_0 = phi0 - shift
        x1 = np.log(R_rc/sign)                 # p(rc)
        x2 = g[1]/R_rc                         # p'(rc)
        x3 = g[2]/R_rc                         # p''(rc) + p'(rc)^2
        x4 = -Z*(shift + R_0 + phi_others)/R_0 # p'(0)
        x5 = np.log(R_0 / sign)                # p(0)

        rc2 = rc**2
        rc3 = rc**3
        rc4 = rc**4

        a0 = x5
        a1 = x4
        a2 = 6*x1/rc2 - 3*x2/rc + x3/2 - 3*x4/rc - 6*x5/rc2 - (x2**2)/2
        a3 = -8*x1/rc3 + 5*x2/rc2 - x3/rc + 3*x4/rc2 + 8*x5/rc3 + (x2**2)/rc
        a4 = 3*x1/rc4 - 2*x2/rc3 + 0.5*x3/rc2 - x4/rc3 - 3*x5/rc4 - 0.5*(x2**2)/rc2
        return sign, shift, np.array([a0,a1,a2,a3,a4])

    def _get_sto_Eloc(self, sign, shift, a, r, Z, phi_others):
        """
        Evaluate the local energy of multiple slater-type orbitals with given polynomials for different radii r.
        This function is used when tuning the free parameter phi0 to optimize the local energy of this orbital during cusp correction.

        Args:
            sign (np.array): Sign of orbital; shape [N]
            shift(np.array): Shift of orbital; shape [N]
            a (np.array): Polynomial coefficients of orbital; shape [Nx5]
            r (np.array): Radii where orbital is being evaluated; shape [N_radii]
            Z (float or int): Nuclear charge
            phi_others (float or int): Sum of other orbitals at the nuclear coordinate

        Returns:
            np.array: Array containing the values of STO(r) for each STO; shape [N x N_radii]
        """
        sign = np.reshape(sign, [-1, 1])
        shift = np.reshape(shift, [-1, 1])

        rn = r**np.arange(5)[:, np.newaxis, np.newaxis]
        a1 = np.array([a[1], 2*a[2], 3*a[3], 4*a[4], np.zeros_like(a[0])])
        a2 = np.array([2*a[2], 6*a[3], 12*a[4], np.zeros_like(a[0]), np.zeros_like(a[0])])
        poly = np.sum(a[...,np.newaxis] * rn, axis=0)
        poly1 = np.sum(a1[...,np.newaxis] * rn, axis=0)
        poly2 = np.sum(a2[...,np.newaxis] * rn, axis=0)

        R = sign * np.exp(poly)
        R0 = sign * np.exp(a[0][:, np.newaxis])
        Zeff = Z * (1+phi_others/(shift + R0))

        return -0.5*(R/(shift+R))*(2*poly1/r + poly2 + poly1**2) - Zeff / r


    def _get_optimal_cusp_poly_coefficients(self, phi0_orig, phi_others, g, Z, rc):
        """
        Get optimal coefficients for a slater-type cusp orbital by varying the free parameter phi0 until a cusp orbital with minimal variance of E_loc inside the cusp radius is found.

        Args:
            phi0_orig (float): Value of original (uncorrected) orbital at r=0
            phi_others (float): Sum of other orbitals at r=0
            g (list of lenght 3): Value, 1st and 2nd derivative of original orbital R at r=rc
            Z: Nuclear charge
            rc: Cusp radius

        Returns:
            Parameters of optimal cusp-corrected Slater-type orbital

        """
        delta = g[0] - phi0_orig
        phi0_values = phi0_orig + np.linspace(-2*delta, 2*delta, 500)
        signs, shifts, polys = self._get_cusp_poly_coefficients(phi0_values, phi_others, g, Z, rc)

        r = np.linspace(1e-4, rc, 100)
        El = self._get_sto_Eloc(signs, shifts, polys, r, Z, phi_others)
        El_std = np.nanstd(El, axis=-1)
        ind_opt = np.nanargmin(El_std)
        El_stdmin = El_std[ind_opt]
        return shifts[ind_opt], signs[ind_opt], polys[:,ind_opt], El_stdmin


    def eval_cusp_STO(self, r, rc, polynomial_coeffs, sign, shift):
        """
        Evaluate a slater-type cusp-orbital.
        This function will be called for all electron coordinates where an electron is inside the cusp radius rc of a nucleus.
        The orbital is given by :math:`\phi(r) = shift + sign \exp(\sum_k c_k r^k)`
        The input radii r are clipped at rc to prevent numerical over-/underflows when evaluating the orbital at radii r>rc

        Args:
            r (tf.Tensor): Distance of electron from nucleus
            rc (tf.Tensor): Cusp radius for this nucleus
            polynomial_coeffs (tf.Tensor): Coefficients :math:`c_k` of cusp polynomial
            sign: Sign of orbital
            shift: Offset/shift of orbital

        Returns:
            tf.Tensor: Value of orbital at given radii
        """
        r = tf.minimum(r, rc)
        n = tf.range(5, dtype=dpe.DTYPE)
        r = tf.expand_dims(r, axis=-1) # add dimensions to compute all molecular orbitals in parallel
        rn = tf.expand_dims(r, axis=-1)**n # add dimension for polynomial coefficients
        poly = tf.reduce_sum(rn*polynomial_coeffs, axis=-1)
        return shift + tf.exp(poly) * sign

    def buildCuspCorrection(self):
        """
        Calculates the orbital cusp correction, by calculating an alternative slater-type orbital at each nucleus i for each molecular orbital j.
        These alternative slater-type orbitals will adhere to the Kato-cusp conditions.
        The implementation is based heavily on: J. Chem. Phys. 122, 224322 (2005); https://doi.org/10.1063/1.1940588

        All results are stored in the member variables :attr:`~.SlaterDetModel.sto_polynomial`,
        :attr:`~.SlaterDetModel.mo_coeff_1s`, :attr:`~.SlaterDetModel.sto_sign`, and :attr:`~.SlaterDetModel.sto_shift`

        Returns:
            None
        """
        n_ions = len(self.physical_config.ion_positions)
        self.sto_polynomial = []
        self.mo_coeff_1s = []
        self.sto_sign = []
        self.sto_shift = []
        self.r_cusp = self.model_config.scale_r_cusp / np.maximum(self.ion_charges.numpy(), 2)
        for spin in range(2):
            n_molecular_orbitals = self.mo_coeff[spin].shape[1]
            n_atomic_orbitals = self.mo_coeff[spin].shape[0]
            self.sto_polynomial.append(np.zeros([n_ions, n_molecular_orbitals, 5]))
            self.mo_coeff_1s.append(np.zeros([n_ions, n_molecular_orbitals, n_atomic_orbitals]))
            self.sto_sign.append(np.zeros([n_ions, n_molecular_orbitals]))
            self.sto_shift.append(np.zeros([n_ions, n_molecular_orbitals]))
            for i in range(n_ions):
                for j in range(n_molecular_orbitals):
                    self.sto_shift[spin][i,j], self.sto_sign[spin][i,j], self.sto_polynomial[spin][i,j,:], self.mo_coeff_1s[spin][i,j,:] = self._get_STO_polynomial(i,j, spin)
            self.sto_polynomial[spin] = tf.Variable(self.sto_polynomial[spin], dtype=dpe.DTYPE, trainable=False)
            self.mo_coeff_1s[spin] = tf.Variable(self.mo_coeff_1s[spin], dtype=dpe.DTYPE, trainable=False)
            self.sto_sign[spin] = tf.Variable(self.sto_sign[spin], dtype=dpe.DTYPE, trainable=False)
            self.sto_shift[spin] = tf.Variable(self.sto_shift[spin], dtype=dpe.DTYPE, trainable=False)
        self.r_cusp = tf.Variable(self.r_cusp, dtype=dpe.DTYPE, trainable=False)

    def log_squared(self, inp):
        """
        Evaluate the wavefunction density :math:`\log \psi(r)^2`. This overrides the corresponding abstract method of :class:`deeperwin.WFModel`
        Basically a thin wrapper around get_los_psi_squared + regularization terms (i.e. electron-electron cusp correction)

        Args:
            inp (tf.Tensor): Electron coordinates. Tensor of shape [batch_size x n_elec x 3]

        Returns:
            tf.Tensor: Tensor of shape [batch_size] containing :math:`\log(psi^2)`

        """
        log_sqr = self.get_log_psi_squared(inp)
        if self.model_config.use_regularization:
            log_sqr = log_sqr + self.get_regularization(inp)
        return log_sqr

    @property
    @abc.abstractmethod
    def total_energy(self):
        """
        Estimate of total energy, as given by the approximate quantum chemistry method.
        Other functions might for example use this property to initialize energy clippings during wavefunction optimization.

        Returns:
            float: Energy estimate

        """
        return None

    @abc.abstractmethod
    def build_determinants_and_weights(self):
        """
        To be implemented by each subclass.
        Must find and approximate solution to the Schrödinger equation (e.g. using Hartree Fock, CI, ...) and
        set the internal variables :attr:`~.SlaterDetModel.mo_coeff` (orbitals) and :attr:`~.SlaterDetModel.ci_weights` (CI expansion coefficients, i.e. weights of determinants).

        Returns:
            tuple containing

            - **mo_coeff** (np.array): Coefficients of molecular orbitals
            - **ci_weights** (np.array): Coefficients of determinants
        """
        self.mo_coeff, self.ci_weights = None, None

    def _build_pyscf_molecule(self):
        """
        Convert geometry information encoded in config to a molecule object that can be parsed by pyscf package.

        Returns:
            Molecule object
        """
        self.molecule = pyscf.gto.Mole()
        self.molecule.atom = self._build_pyscf_geometry_string()
        self.molecule.unit = "bohr"
        self.molecule.basis = self.model_config.basis
        self.molecule.cart = True
        self.molecule.spin = 2 * self.physical_config.n_spin_up - self.physical_config.n_electrons
        self.molecule.charge = sum(self.config.physical.ion_charges) - self.config.physical.n_electrons
        self.molecule.output = "tmp_pyscf.log"
        self.molecule.verbose = 0  # suppress output to console
        self.molecule.max_memory = 10e3 # maximum memory in megabytes (i.e. 10e3 = 10GB)
        self.molecule.build()
        return self.molecule

    def _eval_cusp_correction(self, ind_closest_ion, r_closest_ion, spin, atomic_orbitals):
        """
        Calculates a corretion to each molecular orbital and each electron when it is within the cusp radius rc of any nucleus.
        The result is effectively :math:`\phi_\text{cusp-orbital} - phi\text{original orbital}`

        Args:
            ind_closest_ion (tf.Tensor): Integer Tensor of shape [batch_size x n_electrons_of_spin], indexing which of the ions is closest
            r_closest_ion (tf.Tensor): Tensor of shape [batch_size x n_electrons_of_spin], indexing the distance to the closest ion
            spin (int): Spin of given electrons (0=up, 1=down)
            atomic_orbitals (tf.Tensor): Evaluation of all atomic orbitals (GTOs) for each electron; shape [batch_size x n_electrons_of_spin x n_atomic_orbitals]

        Returns:
            tf.Tensor: Correction to molecular orbitals of shape [batch_size x n_electrons_of_spin x n_molecular_orbitals]. Correction is 0 for all electron coordinates that are outside the cusp radius of every nucleus and nonzero otherwise.
        """
        # apply cusp correction here
        r_cusp = tf.gather(self.r_cusp, ind_closest_ion, axis=0)
        polynomial = tf.gather(self.sto_polynomial[spin], ind_closest_ion, axis=0)
        sign = tf.gather(self.sto_sign[spin], ind_closest_ion, axis=0)
        shift = tf.gather(self.sto_shift[spin], ind_closest_ion, axis=0)
        mo_coeffs_s_type = tf.gather(self.mo_coeff_1s[spin], ind_closest_ion, axis=0)
        mo_stype_gtos = tf.reduce_sum(tf.expand_dims(atomic_orbitals, axis=2) * mo_coeffs_s_type, axis=-1)
        mo_cusp_correction = self.eval_cusp_STO(r_closest_ion, r_cusp, polynomial, sign, shift) - mo_stype_gtos
        mo_cusp_correction = tf.where(tf.expand_dims(r_closest_ion < r_cusp, axis=-1), mo_cusp_correction, 0.0)
        return mo_cusp_correction

    def get_log_psi_squared(self, el_coords, backflow_factor_up=1.0, backflow_factor_dn = 1.0):
        """
        Calculate wavefunction density :math:`\log(\psi(r)^2)` for this slater-type model.
        As opposed to :meth:`~.SlaterDetModel.SlaterDetModel.log_squared`, this method does not apply any regularizations or additional cusp corrections beyond the orbital cusp correction.
        This gives the enclosing parent model the option to employ wholistic corrections/regularizations on the total wavefunction.

        Args:
            el_coords (tf.Tensor): Tensor of shape [batch_size x n_electrons x 3]: Electron coordinates
            backflow_factor_up (float or tf.Tensor): Backflow factors to be applied to each of the orbitals of the spin-up determinants. Tensor of shape [batch_size x n_determinants x n_electrons_with_spin_up x n_orbitals_per_determinant]
            backflow_factor_dn (float or tf.Tensor): Backflow factors to be applied to each of the orbitals of the spin-down determinants. Tensor of shape [batch_size x n_determinants x n_electrons_with_spin_down x n_orbitals_per_determinant]

        Returns:
            tf.Tensor: Tensor of shape [batch_size] containing log(psi)^2

        """
        r_el_ion = tf.squeeze(self.get_pairwise_differences(el_coords)[0], axis=1)
        r_el_ion = r_el_ion[:, :self.n_electrons, self.n_electrons-1:]
        ind_closest_ion_all = tf.argmin(r_el_ion, axis=-1) # get index of closest ion for each electron
        # r_closest_ion_all = tf.gather(r_el_ion, ind_closest_ion_all, axis=-1, batch_dims=2)
        r_closest_ion_all = tf.reduce_min(r_el_ion, axis=-1)
        log_dets = []
        signs = []
        for spin in [0,1]:
            x = self.get_spin_coords(el_coords, spin)
            ind_closest_ion = self.get_spin_coords(ind_closest_ion_all, spin)
            r_closest_ion = self.get_spin_coords(r_closest_ion_all, spin)

            if spin == 0:
                bf = backflow_factor_up
            else:
                bf = backflow_factor_dn
            # [batch-size x n_el_spin x n_orbitals]
            atomic_orbitals = tf.stack([ao.call(x) for ao in self.atomic_orbitals], axis=-1)
            molecular_orbitals = atomic_orbitals @ self.mo_coeff[spin]
            if self.model_config.use_orbital_cusp_correction:
                molecular_orbitals = molecular_orbitals + self._eval_cusp_correction(ind_closest_ion, r_closest_ion,
                                                                                     spin, atomic_orbitals)

            # [batch-size x n_el_spin x n_dets x n_el_spin (mos)]
            ind_orbitals = self.ind_orbitals[spin]
            molecular_orbitals = tf.gather(molecular_orbitals, ind_orbitals, axis=-1)
            molecular_orbitals = tf.transpose(molecular_orbitals, [0, 2, 1, 3]) # move determinant index forward
            molecular_orbitals = molecular_orbitals * bf

            # sign, logdet = tf.linalg.slogdet(molecular_orbitals)
            sign, logdet = self.slogdet_via_svd(molecular_orbitals)
            signs.append(sign)
            log_dets.append(logdet)

            # determinants.append(self.calc_det_via_svd(molecular_orbitals))
        # [batch-size x n_dets x 2 (spin)]
        log_dets = tf.reduce_sum(tf.stack(log_dets, axis=-1), axis=-1) # reduce over spins; log-space => sum = product of determinants
        signs = tf.reduce_prod(tf.stack(signs, axis=-1), axis=-1) # reduce over spins
        log_shift = tf.reduce_max(log_dets, axis=1, keepdims=True) # reduce over determinants
        psi = tf.reduce_sum(self.ci_weights * signs * tf.exp(log_dets - log_shift), axis=1)

        log_psi_squared = 2 * tf.math.log(tf.abs(psi)+self.model_config.log_shift) + 2 * tf.squeeze(log_shift, axis=1)
        return log_psi_squared


    def slogdet_via_svd(self, A):
        """
        Calculate sign and logarithm of a determinant by computing its singular value decomposition (SVD).
        To improve numerical stability all singular values are shifted by a small positive shift defined in config.model.svd_log_shift.
        This prevents issues with vanishing determinants.
        
        Args:
            A (tf.Tensor): Tensor of shape [batch_dimensions x n x n]

        Returns:
            tuple containing
            
            - **sign** (tf.Tensor): Sign of determinant as tensor of shape [batch_dimensions], containing +- 1
            - **logdet** (tf.Tensor): Logarithm of absolute value of determinant :math:`log|det(A)|`
        """
        S, U, V = tf.linalg.svd(A, compute_uv=True)
        logdet = tf.reduce_sum(tf.math.log(S + self.model_config.svd_log_shift), axis=-1)
        sign = tf.sign(tf.linalg.det(tf.matmul(U, V)))
        return sign, logdet

    def _build_pyscf_geometry_string(self):
        """
        Build a string that contains nuclear charges and atomic positions to be parsed by pyscf

        Returns:
            str: Multiline string containing a line with 'Z Rx Ry Rz' for each atom
        """
        geometry_string = ""
        for charge, pos in zip(self.physical_config.ion_charges, self.physical_config.ion_positions):
            geometry_string += f"{charge} {pos[0]} {pos[1]} {pos[2]}\n"
        return geometry_string

    def get_regularization(self, inp):
        """
        Computes a regularization term that is supposed to ensure a smooth decay of the wavefunction at long distances.
        In particular this regularization should decrease numerical issues for electrons that are very far from all nucleii and therefore see little to no gradient in the wavefunction due numerical underflows in the determinant calculations.
        However this regularization has been observed to cause issues close to the nuclei, because it is potentially not cusp-free and can therefore cause unwanted energy errors.

        Args:
            inp (tf.Tensor): Tensor of electron coordinates with shape [batch_size x n_electrons x 3]

        Returns:
            tf.Tensor: Tensor of shape [batch_size] containing a log(psi^2) correction t
        """
        dist = self.get_pairwise_differences(inp)[0]
        dist_el_ion = dist[:, 0, self.n_electrons:, :self.n_electrons] # output-shape: batch x ion x electron
        weights = 1/(dist_el_ion+1)**2
        weights = weights / tf.reduce_sum(weights, axis=2, keepdims=True)
        log_sqr = - tf.reduce_sum(weights * dist_el_ion, axis=[1,2]) / self.regularization_decay_length
        return log_sqr

    def initBasis(self):
        """
        Intialize all GTO basis-set coefficients from pyscf basis set

        Returns:
            None
        """
        ao_labels = self.molecule.ao_labels(None)
        self.n_basis = len(ao_labels)

        ind_basis = 0
        for ind_nuc, (element, atom_pos) in enumerate(self.molecule._atom):
            for gto_data in self.molecule._basis[element]:
                l = gto_data[0]
                gto_data = np.array(gto_data[1:])
                alpha = gto_data[:, 0]
                weights = gto_data[:, 1:]
                for ind_contraction in range(weights.shape[1]):
                    for m in range(-l, l + 1):
                        shape_string = ao_labels[ind_basis][3]  # string of the form 'xxy' or 'zz'
                        lxyz = [shape_string.count(x) for x in ['x', 'y', 'z']]
                        self.atomic_orbitals.append(GaussianOrbital(ind_nuc,
                                                                    atom_pos,
                                                                    alpha,
                                                                    weights[:, ind_contraction],
                                                                    lxyz,
                                                                    ao_labels[ind_basis]))
                        ind_basis += 1
        assert len(self.atomic_orbitals) == self.n_basis, "Could not properly construct basis functions. " \
                                             "You probably tried to use a valence-only basis (e.g. cc-pVDZ) for an all-electron calculation."
