"""
Contains the physical baseline model (e.g. CASSCF).

This module provides functionality to calculate a basline solution using pySCF and functions to
"""
import logging

import jax.numpy as jnp
import pyscf
import pyscf.ci
import pyscf.mcscf
import numpy as np
from deeperwin.configuration import PhysicalConfig, CASSCFConfig
from deeperwin.utils import get_el_ion_distance_matrix
from scipy.optimize import curve_fit

logger = logging.getLogger("dpe")

#################################################################################
############################ Orbital functions ##################################
#################################################################################

def eval_gaussian_orbital(el_ion_diff, el_ion_dist, ind_nuc, alpha, weights, angular_momenta):
    """
    Evaluate a single orbital on multiple points in space r.

    Each orbital is evaluated at N points in 3D space.
    Each orbital consist of k gaussians centered around a nucleus at position R.

    Args:
        el_ion_diff: Position where orbital is evaluated, relative to nucleus, shape [3]
        el_ion_dist: norm of difference, i.e. distance between electron position and the nucleus; shape []
        alpha: Gaussian decay coefficients [k]
        weights: Linear expansion coeffs of orbitals [k]
        angular_momenta: Integers describing the power of each cartesian coordinate, shape [3]

    Returns:
        Array of shape [N]
    """

    r_sqr = jnp.expand_dims(el_ion_dist ** 2, axis=-1)  # shape [1]
    pre_fac = jnp.prod(el_ion_diff ** angular_momenta, axis=-1)  # shape []
    phi_gto = pre_fac * jnp.sum(weights * jnp.exp(-alpha * r_sqr), axis=-1)
    return phi_gto


def eval_atomic_orbitals(el_ion_diff, el_ion_dist, orbitals, ao_cusp_params=None):
    """
    Args:
        el_ion_diff: shape [N_batch x n_el x N_ion x 3]
        el_ion_dist: shape [N_batch x n_el x N_ion]
        orbitals:

    Returns:

    """
    outputs = []
    for i,ao in enumerate(orbitals):
        ind_nuc = ao[0]
        diff = el_ion_diff[..., ind_nuc, :]
        dist = el_ion_dist[..., ind_nuc]
        gto = eval_gaussian_orbital(diff, dist, *ao)
        if (ao_cusp_params is not None) and (ao_cusp_params[i] is not None):
            r_c = ao_cusp_params[i][0]
            sto = _eval_cusp_atomic_orbital(dist, *ao_cusp_params[i])
            weight_gto = 0.5 * (jnp.sign(dist - r_c) + 1)
            weight_sto = 1.0 - weight_gto
            gto = weight_gto * gto + weight_sto * sto
        outputs.append(gto)
    return jnp.stack(outputs, axis=-1)

def _eval_cusp_molecular_orbital(dist, r_c, offset, sign, poly):
    dist = jnp.minimum(dist, r_c)
    n = jnp.arange(5)
    r_n = dist[..., jnp.newaxis, jnp.newaxis] ** n
    p_r = jnp.sum(r_n * poly, axis=-1)
    psi = offset + sign*jnp.exp(p_r)
    return psi

def _eval_cusp_atomic_orbital(dist, r_c, offset, sign, poly):
    dist = jnp.minimum(dist, r_c)
    n = jnp.arange(5)
    r_n = dist[..., jnp.newaxis] ** n
    p_r = jnp.dot(r_n, poly)
    psi = offset + sign*jnp.exp(p_r)
    return psi

def evaluate_molecular_orbitals(el_ion_diff, el_ion_dist, atomic_orbitals, mo_coeff, ao_cusp_params=None, mo_cusp_params=None):
    aos = eval_atomic_orbitals(el_ion_diff, el_ion_dist, atomic_orbitals, ao_cusp_params)
    mos = aos @ mo_coeff

    if mo_cusp_params is not None:
        r_c, offset, sign, poly, coeff_1s = mo_cusp_params
        sto_mask = jnp.heaviside(r_c - el_ion_dist, 0.0)[..., jnp.newaxis]
        sto = _eval_cusp_molecular_orbital(el_ion_dist, r_c, offset, sign, poly) - jnp.sum(aos[..., np.newaxis, :, np.newaxis] * coeff_1s, axis=-2)
        sto = jnp.sum(sto*sto_mask, axis=-2)     # sum over all ions, masking only the once that are within r_c of an electron
        mos = mos + sto # subtract GTO-part of 1s orbitals and replace by STO-part
    return mos


#################################################################################
####################### Compile-time helper functions ###########################
#################################################################################

def build_pyscf_molecule(R, Z, charge=0, spin=0, basis_set='6-311G'):
    """
    Args:
        R: ion positions, shape [Nx3]
        Z: ion charges, shape [N]
        charge: integer
        spin: integer; number of excess spin-up electrons
        basis_set (str): basis set identifier

    Returns:
        pyscf.Molecule
    """
    molecule = pyscf.gto.Mole()
    molecule.atom = [[Z_, tuple(R_)] for R_, Z_ in zip(R, Z)]
    molecule.unit = "bohr"
    molecule.basis = basis_set
    molecule.cart = True
    molecule.spin = spin  # 2*n_up - n_down
    molecule.charge = charge
    molecule.output = "/dev/null"
    molecule.verbose = 0  # suppress output to console
    molecule.max_memory = 10e3  # maximum memory in megabytes (i.e. 10e3 = 10GB)
    molecule.build()
    return molecule


def _get_gto_normalization_factor(alpha, angular_momenta):
    l_tot = np.sum(angular_momenta)
    fac_alpha = (2 * alpha / np.pi) ** (3 / 4) * (8 * alpha) ** (l_tot / 2)

    fac = np.array([np.math.factorial(x) for x in angular_momenta])
    fac2 = np.array([np.math.factorial(2 * x) for x in angular_momenta])
    fac_exponent = np.sqrt(np.prod(fac) / np.prod(fac2))
    factor = fac_alpha * fac_exponent
    return factor


def _get_atomic_orbital_basis_functions(molecule):
    """
    Args:
        molecule: pyscf molecule object
    Returns:
        orbitals: tuple of tuple, each containing index of nucleus, nuclear coordinate, exponential decay constant alpha, weights and angular momentum exponents
    """
    ao_labels = molecule.ao_labels(None)
    n_basis = len(ao_labels)
    ind_basis = 0
    atomic_orbitals = []
    for ind_nuc, (element, atom_pos) in enumerate(molecule._atom):
        for gto_data in molecule._basis[element]:
            l = gto_data[0]
            gto_data = np.array(gto_data[1:])
            alpha = gto_data[:, 0]
            weights = gto_data[:, 1:]
            for ind_contraction in range(weights.shape[1]):
                n_orientations = [1, 3, 6][l]
                for m in range(n_orientations): # 1,3,6, ... = number of orbitals per angular momentum
                    shape_string = ao_labels[ind_basis][3]  # string of the form 'xxy' or 'zz'
                    angular_momenta = np.array([shape_string.count(x) for x in ['x', 'y', 'z']], dtype=int)
                    normalization = _get_gto_normalization_factor(alpha, angular_momenta)
                    atomic_orbitals.append([ind_nuc,
                                            alpha,
                                            weights[:, ind_contraction] * normalization,
                                            angular_momenta])
                    ind_basis += 1
    assert len(
        atomic_orbitals) == n_basis, "Could not properly construct basis functions. You probably tried to use a valence-only basis (e.g. cc-pVDZ) for an all-electron calculation."  # noqa
    return atomic_orbitals

def build_pyscf_molecule_from_physical_config(physical_config: PhysicalConfig, basis_set):
    R, Z = np.array(physical_config.R), np.array(physical_config.Z)
    charge = sum(Z) - physical_config.n_electrons
    spin = 2 * physical_config.n_up - physical_config.n_electrons
    return build_pyscf_molecule(R, Z, charge, spin, basis_set)

def get_hartree_fock_solution(physical_config: PhysicalConfig, basis_set):
    molecule = build_pyscf_molecule_from_physical_config(physical_config, basis_set)
    atomic_orbitals = _get_atomic_orbital_basis_functions(molecule)
    hf = pyscf.scf.HF(molecule)
    hf.verbose = 0  # suppress output to console
    hf.kernel()
    return atomic_orbitals, hf


def get_baseline_solution(physical_config: PhysicalConfig, casscf_config: CASSCFConfig, n_dets: int):
    n_el, n_up, R, Z = physical_config.get_basic_params()
    n_dn = n_el - n_up
    atomic_orbitals, hf = get_hartree_fock_solution(physical_config, casscf_config.basis_set)

    if n_dets == 1 or casscf_config.only_hf:
        # Only 1 determinant => no MRCI/CASSCF calculation required
        mo_coeff = hf.mo_coeff
        if len(mo_coeff) != 2:
            mo_coeff = [mo_coeff, mo_coeff]
        ci_weights = np.ones(1) if not casscf_config.only_hf else np.ones(32)
        ind_orbitals = [np.arange(n_up)[None, :], np.arange(n_dn)[None, :]] if not casscf_config.only_hf else\
            [np.tile(np.arange(n_up), n_dets).reshape((-1, n_up)), np.tile(np.arange(n_dn), n_dets).reshape((-1, n_dn))]
        E_hf, E_casscf = hf.e_tot, np.nan
    else:
        # Run CASSCF to get excited determinants
        casscf = pyscf.mcscf.UCASSCF(hf, physical_config.n_cas_orbitals, physical_config.n_cas_electrons)
        casscf.kernel()
        E_hf, E_casscf = hf.e_tot, casscf.e_tot

        mo_coeff = list(casscf.mo_coeff)  # tuple of spin_up, spin_down
        ind_orbitals = _get_orbital_indices(casscf)
        ci_weights = casscf.ci.flatten()

        logger.debug(f"Total nuber of CASSCF-determinants before truncation: {len(ci_weights)}")

        if n_dets < len(ci_weights):
            ind_largest = np.argsort(np.abs(ci_weights))[::-1][:n_dets]
            share_captured = np.sum(ci_weights[ind_largest]**2) / np.sum(ci_weights**2)
            logger.debug(f"Share of CI-weights captured by {n_dets} dets: {share_captured:.3%}")
            ci_weights = ci_weights[ind_largest]
            ci_weights = jnp.array(ci_weights / np.sum(ci_weights ** 2))
            ind_orbitals = ind_orbitals[0][ind_largest], ind_orbitals[1][ind_largest]

    # Calculate cusp-correction-parameters for molecular orbitals
    if casscf_config.cusps.cusp_type == "mo":
        ao_cusp_params = None
        mo_cusp_params = [calculate_molecular_orbital_cusp_params(atomic_orbitals, mo_coeff[i], R, Z, casscf_config.cusps.r_cusp_el_ion_scale) for i in range(2)]
    else:
        ao_cusp_params = [_calculate_ao_cusp_params(*ao, physical_config.Z[ao[0]]) for ao in atomic_orbitals]
        mo_cusp_params = [None, None]

    return (atomic_orbitals, ao_cusp_params, mo_cusp_params, mo_coeff, ind_orbitals, ci_weights), (E_hf, E_casscf)

def get_cisd_leading_determinants(physical_config: PhysicalConfig, n_dets, basis_set="6-31G"):
    molecule = build_pyscf_molecule_from_physical_config(physical_config, basis_set)
    n_up, n_dn = physical_config.n_up, physical_config.n_dn
    hf = pyscf.scf.UHF(molecule)
    hf.verbose = 0  # suppress output to console
    hf.kernel()
    n_mo = hf.mo_coeff.shape[1]
    cisd = pyscf.ci.CISD(hf).run()
    amplitudes = cisd.cisdvec_to_amplitudes(cisd.ci)

    # Ensure that excitations are not double-counted by setting amplitudes of equivalent excitations to 0
    for spin in range(2):
        for row in range(amplitudes[2][spin].shape[0]):
            amplitudes[2][spin][row, :row, :, :] = 0
        for row in range(amplitudes[2][spin].shape[2]):
            amplitudes[2][spin][:, :, row, :row] = 0

    n_up_unocc = n_mo - n_up
    n_dn_unocc = n_mo - n_dn
    coeffs = np.concatenate([[amplitudes[0]]] + [amplitudes[n_exc][spin].flatten() for n_exc in [1,2] for spin in [0,1]])

    # Form a list of tuples, containing: (spin of excitation, list of tuples containing the orbitals to be swapped)
    orbital_indices = [(0, [])]
    orbital_indices += [(0,[(a, b + n_up)]) for a, b in zip(*np.unravel_index(np.arange(n_up * n_up_unocc), [n_up, n_up_unocc]))]
    orbital_indices += [(1, [(a, b + n_dn)]) for a, b in zip(*np.unravel_index(np.arange(n_dn * n_dn_unocc), [n_dn, n_dn_unocc]))]
    orbital_indices += [(0, [(a, c + n_up), (b, d + n_up)]) for a, b, c, d in zip(*np.unravel_index(np.arange(n_up ** 2 * n_up_unocc ** 2),
                                                                                                    [n_up, n_up, n_up_unocc, n_up_unocc]))]
    orbital_indices += [(1, [(a, c + n_dn), (b, d + n_dn)]) for a, b, c, d in zip(*np.unravel_index(np.arange(n_dn ** 2 * n_dn_unocc ** 2),
                                                                                                    [n_dn, n_dn, n_dn_unocc, n_dn_unocc]))]
    ind_largest = np.argsort(np.abs(coeffs))[::-1][:n_dets]
    coeffs_largest = coeffs[ind_largest]
    orbital_indices_largest = [orbital_indices[i] for i in ind_largest]

    mo_indices = [np.tile(np.arange(n), [n_dets, 1]) for n in (n_up, n_dn)]
    for i, (spin, exchanges) in enumerate(orbital_indices_largest):
        for e in exchanges:
            mo_indices[spin][i, e[0]] = e[1]
    return mo_indices, coeffs_largest, hf.mo_coeff


def fit_orbital_envelopes_to_hartree_fock(physical_config: PhysicalConfig):
    # Calculate hartree-fock solution to fit against
    R_ions = np.array(physical_config.R)
    n_ions = len(R_ions)
    n_el_per_spin = [physical_config.n_up, physical_config.n_electrons - physical_config.n_up]
    (atomic_orbitals, _, _, mo_coeff, _, _), _ = get_baseline_solution(physical_config,
                                                                       CASSCFConfig(basis_set="STO-6G"),
                                                                       n_dets=1)

    # Generate atom-centered grid on which to evaluate orbitals for fitting
    n_points_per_atom = 500
    r_sampling = []
    for R in R_ions:
        r = np.exp(np.random.uniform(-2, 1, n_points_per_atom))
        r_angle = np.random.normal(size=[n_points_per_atom, 3])
        r = (r_angle / np.linalg.norm(r_angle, axis=-1, keepdims=True)) * r[:, None]
        r_sampling.append(r + np.array(R))
    r_sampling = np.concatenate(r_sampling, axis=0)

    # Envelope function to use for fitting
    def fit_func(d, *params):
        def softplus(x):
            return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
        c = np.array(params[:n_ions])
        alpha = np.array(params[n_ions:])
        return np.sum(c[None, :] * np.exp(-softplus(alpha[None, :] * d)), axis=-1)

    diff = r_sampling[:, None, :] - R_ions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    p0 = np.ones(2 * n_ions)

    c_values = [[],[]]
    alpha_values = [[],[]]
    for spin in range(2):
        mo_evals = evaluate_molecular_orbitals(diff, dist, atomic_orbitals, mo_coeff[spin])
        for ind_mo in range(n_el_per_spin[spin]):
            mo_ref = np.abs(mo_evals[:, ind_mo])
            p_opt, _ = curve_fit(fit_func, dist, mo_ref, p0)
            c_values[spin].append(p_opt[:n_ions])
            alpha_values[spin].append(p_opt[n_ions:])
    c_values = [np.array(c).T for c in c_values]
    alpha_values = [np.array(a).T for a in alpha_values]
    return c_values, alpha_values


def _get_effective_charge(Z: int, n: int):
    """
    Calculates the approximate effective charge for an electron, using Slater's Rule of shielding.

    Args:
        Z: Nuclear charge of an atom
        n: Principal quantum number of the elctron for which the effective charge is being calculated

    Returns:
        Z_eff (float): effective charge
    """
    shielding = 0
    for n_shell in range(1, n+1):
        n_electrons_in_shell = 2 * n_shell ** 2
        if n_shell == n:
            n_el_in_lower_shells = sum([2*k**2 for k in range(1,n_shell)])
            n_electrons_in_shell = min(Z - n_el_in_lower_shells, n_electrons_in_shell) - 1

        if n_shell == n:
            shielding += n_electrons_in_shell * 0.35
        elif n_shell == (n-1):
            shielding += n_electrons_in_shell * 0.85
        else:
            shielding += n_electrons_in_shell
    return max(Z - shielding, 1)


def _get_atomic_orbital_envelope_exponents(physical_config: PhysicalConfig, basis_set):
    molecule = build_pyscf_molecule_from_physical_config(physical_config, basis_set)
    atomic_orbitals = _get_atomic_orbital_basis_functions(molecule)

    Z_ions = np.array(physical_config.Z)
    n_ions = len(Z_ions)

    n_ao = []
    for ind_nuc in range(n_ions):
        ao_nuc = [ao for ao in atomic_orbitals if ao[0] == ind_nuc]
        nr_of_s_orbitals = len([ao for ao in ao_nuc if sum(ao[-1]) == 0])
        nr_of_p_orbitals = len([ao for ao in ao_nuc if sum(ao[-1]) == 1])
        nr_of_d_orbitals = len([ao for ao in ao_nuc if sum(ao[-1]) == 2])
        assert (nr_of_s_orbitals + nr_of_p_orbitals + nr_of_d_orbitals) == len(ao_nuc), "Initialization currently does support d-orbitals"
        n_ao.append(np.arange(1, nr_of_s_orbitals + 1))
        n_ao.append(np.repeat(np.arange(2, nr_of_p_orbitals // 3 + 2), 3))
        n_ao.append(np.repeat(np.arange(3, nr_of_d_orbitals // 6 + 3), 6))
    n_ao = np.concatenate(n_ao)
    Z_ao = np.array([Z_ions[ao[0]] for ao in atomic_orbitals])
    Z_ao = np.array([_get_effective_charge(Z, n) for Z, n in zip(Z_ao, n_ao)])
    alpha_ao = Z_ao / n_ao
    alpha_ao = alpha_ao * 0.7  # rescale to better fit empirical values
    return alpha_ao

def get_envelope_exponents_from_atomic_orbitals(physical_config: PhysicalConfig, basis_set='6-31G', mo_coeff=None, ind_orbitals=None):
    n_up, n_dn, n_ions = physical_config.n_up, physical_config.n_dn, physical_config.n_ions
    if mo_coeff is None:
        (atomic_orbitals, _, _, mo_coeff, _, _), _ = get_baseline_solution(physical_config, CASSCFConfig(basis_set=basis_set), n_dets=1)
    else:
        atomic_orbitals = _get_atomic_orbital_basis_functions(build_pyscf_molecule_from_physical_config(physical_config, basis_set))
    if ind_orbitals is None:
        ind_orbitals = [np.arange(n_up)[None, :], np.arange(n_dn)[None, :]] # 1 determinant; lowest mos occupied

    alpha_ao = _get_atomic_orbital_envelope_exponents(physical_config, basis_set)
    n_dets = len(ind_orbitals[0])

    ind_nuc_ao = np.array([ao[0] for ao in atomic_orbitals], dtype=int)
    c_values = [np.zeros([n_ions, n_dets, n_up]), np.zeros([n_ions, n_dets, n_dn])]
    alpha_values = [np.zeros([n_ions, n_dets, n_up]), np.zeros([n_ions, n_dets, n_dn])]
    for n_det in range(n_dets):
        for spin in range(2):
            mo_coeff_det = mo_coeff[spin][:, ind_orbitals[spin][n_det]]
            mo_weights = mo_coeff_det ** 2 / np.sum(mo_coeff_det ** 2, axis=0, keepdims=True)
            for ind_mo in range([n_up, n_dn][spin]):
                for ind_nuc in range(n_ions):
                    index_nuc = (ind_nuc_ao == ind_nuc)
                    mo_weights_nuc = mo_weights[:, ind_mo][index_nuc]
                    c_values[spin][ind_nuc, n_det, ind_mo] = np.sum(mo_weights_nuc) + 1e-3
                    alpha_values[spin][ind_nuc, n_det, ind_mo] = (np.dot(mo_weights_nuc, alpha_ao[index_nuc]) + 1e-6) / (np.sum(mo_weights_nuc) + 1e-6)

    for spin in range(2):
        # undo softplus which will be applied in network
        alpha_values[spin] = np.log(np.exp(alpha_values[spin] - 1))
    c_up = jnp.reshape(c_values[0], [n_ions, -1])
    c_dn = jnp.reshape(c_values[1], [n_ions, -1])
    alpha_up = jnp.reshape(alpha_values[0], [n_ions, -1])
    alpha_dn = jnp.reshape(alpha_values[1], [n_ions, -1])
    return (c_up, c_dn), (alpha_up, alpha_dn)

def get_envelope_exponents_hardcoded(physical_config: PhysicalConfig, n_dets):
    n_up, n_dn, n_ions = physical_config.n_up, physical_config.n_dn, physical_config.n_ions

    if physical_config.name == 'P':
        alphas = [(5, [12, 2, 2, 2, 2, -1, -1.5, -1.5, -1.5]),
                  (5, [12, 2.5, 2.5, 2.5, 2.5, -1, -1, -1, -1]),
                  (8, [13, 2.5, 3, 3, 3, 0.5, 0, 0, 0]),
                  (5, [13, 2.5, 2.5, 2.5, 2.5, 0, -0.5, -0.5, -0.5]),
                  (4, [1.5, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5]),
                  (5, [2.5, 1, 1.5, 1.5, 1.5, 0, -0.5, -0.5, -0.5])
                  ]
    elif physical_config.name == 'Cl':
        alphas = [(10, [14, 2.5, 3, 3, 3, 1, 0, 0, 0]),
                  (10, [12, 2, 2, 2, 2, -0.5, -1, -1, -1]),
                  (4, [13, 1.5, 1.5, 1.5, 1.5, -1, -1, -1, -1]),
                  (4, [1.5, 1, 1, 1, 1, 1, 1, 1, 1]),
                  (4, [2.5, 0, 1, 1, 1, 0, -1, -1, -1])
                  ]
    elif physical_config.name == 'O':
        alphas = [(3, [5, 1.3, 0.9, 0.9, 0.9]),
                  (1, [1, 1, 1, 1, 1])
                  ]
    total_weight = np.sum([a[0] for a in alphas])
    alpha_up = []
    alpha_dn = []
    for weight, a in alphas:
        n_repetitions = int(np.round(n_dets * weight / total_weight))
        alpha_up.append(np.tile(np.array(a)[None, :n_up], n_repetitions))
        alpha_dn.append(np.tile(np.array(a)[None, :n_dn], n_repetitions))
    alpha_up = np.concatenate(alpha_up, axis=-1)
    alpha_dn = np.concatenate(alpha_dn, axis=-1)
    assert alpha_up.shape[1] == n_up * n_dets
    assert alpha_dn.shape[1] == n_dn * n_dets

    c_up = jnp.ones([n_ions, n_up * n_dets])
    c_dn = jnp.ones([n_ions, n_dn * n_dets])
    return dict(c_up=c_up, c_dn=c_dn, alpha_up=alpha_up, alpha_dn=alpha_dn)


def get_envelope_exponents_cisd(physical_config, n_dets):
    basis_set = "6-31G"
    ind_orbitals, ci_coeffs, mo_coeff = get_cisd_leading_determinants(physical_config, n_dets, basis_set)
    return get_envelope_exponents_from_atomic_orbitals(physical_config, basis_set, mo_coeff, ind_orbitals)


def _int_to_spin_tuple(x):
    if type(x) == int:
        return (x,) * 2
    else:
        return x

def split_results_into_spins(hf):
    if len(hf.mo_occ.shape) == 2:
        return hf.mo_coeff, hf.mo_energy, hf.mo_occ
    else:
        return [hf.mo_coeff, hf.mo_coeff], [hf.mo_energy, hf.mo_energy], [hf.mo_occ / 2, hf.mo_occ / 2]

def get_orbital_type(atomic_orbital):
    atom_ind, _, _, l = atomic_orbital
    if sum(l) == 0:
        orbital_type = "s"
    elif sum(l) == 1:
        if l[0]:
            orbital_type = "px"
        elif l[1]:
            orbital_type = "py"
        else:
            orbital_type = "pz"
    else:
        orbital_type = "other"
    return f"Nuc{atom_ind}: {orbital_type}"


def get_p_orbital_indices_per_atom(atomic_orbitals):
    current_atom = -1
    p_orb_indices = []
    for i,ao in enumerate(atomic_orbitals):
        if ao[0] != current_atom:
            current_atom = ao[0]
            p_orb_indices.append([None, None, None])
        l = tuple(ao[-1])
        if l == (1, 0, 0):
            p_orb_indices[-1][0] = i
        elif l == (0, 1, 0):
            p_orb_indices[-1][1] = i
        elif l == (0, 0, 1):
            p_orb_indices[-1][2] = i
    return p_orb_indices


def _get_orbitals_by_cas_type(casscf):
    """
    Splits orbitals into fixed orbitals (that are either always occupied, or always unoccupied) and active orbitals (that are occupied in some determinants, and unoccupied in others).

    Returns:
        tuple containing

        - **fixed_orbitals** (list of 2 lists): For each spin, a list of indices of fixed orbitals
        - **active_orbitals** (list of 2 lists): For each spin, a list of indices of active orbitals
    """
    n_core = _int_to_spin_tuple(casscf.ncore)
    n_cas = _int_to_spin_tuple(casscf.ncas)

    active_orbitals = [list(range(n_core[s], n_core[s] + n_cas[s])) for s in range(2)]
    fixed_orbitals = [list(range(n_core[s])) for s in range(2)]
    return fixed_orbitals, active_orbitals


def _get_orbital_indices(casscf):
    """
    Parse the output of the pySCF CASSCF calculation to determine which orbitals are included in which determinant.

    Returns:
        (list of 2 np.arrays): First array for spin-up, second array for spin-down. Each array has shape [N_determinants x n_electrons_of_spin] and contains the indices of occupied orbitals in each determinant.
    """
    fixed, active = _get_orbitals_by_cas_type(casscf)

    nelcas = _int_to_spin_tuple(casscf.nelecas)
    occ_up = pyscf.fci.cistring._gen_occslst(active[0], nelcas[0])
    occ_down = pyscf.fci.cistring._gen_occslst(active[1], nelcas[1])

    orbitals_up = []
    orbitals_dn = []
    for o_up in occ_up:
        for o_dn in occ_down:
            orbitals_up.append(fixed[0] + list(o_up))
            orbitals_dn.append(fixed[1] + list(o_dn))
    return [jnp.array(orbitals_up, dtype=int), jnp.array(orbitals_dn, dtype=int)]

##########################################################################################
##################################### Cusp functions #####################################
##########################################################################################
def _get_local_energy_of_cusp_orbital(r_c, offset, sign, poly, Z, phi_others=0.0):
    r = np.linspace(1e-6, r_c, 100)[:, np.newaxis]
    p_0 = poly[0] + poly[1] * r + poly[2] * r ** 2 + poly[3] * r ** 3 + poly[4] * r ** 4
    p_1 = poly[1] + 2 * poly[2] * r + 3 * poly[3] * r ** 2 + 4 * poly[4] * r ** 3
    p_2 = 2 * poly[2] + 6 * poly[3] * r + 12 * poly[4] * r ** 2
    prefac = sign * np.exp(p_0) / (offset + sign*np.exp(p_0) + phi_others)
    E_l = -prefac * (p_1/r +0.5 *p_2 + 0.5 * p_1**2) - Z/r
    penalty = np.nanvar(E_l, axis=0)
    # penalty = jnp.max(jnp.abs(E_l - E_l[-1]), axis=0)
    return penalty


def _calculate_mo_cusp_params(phi_rc_0, phi_rc_1, phi_rc_2, phi_0, phi_0_others, Z, r_c):
    if np.abs(phi_0) < 1e-6 and np.abs(phi_rc_0) < 1e-6:
        return 0.0, 0.0, np.zeros(5)
    n_cusp_trials = 500
    phi_new_0 = phi_0 * (1.0 + np.concatenate([np.logspace(-2, 1, n_cusp_trials//2), -np.logspace(-2, 1, n_cusp_trials//2)]))

    sign = jnp.sign(phi_new_0 - phi_rc_0)
    offset = 2 * phi_rc_0 - phi_new_0 # = "C"
    phi_rc_shifted = phi_rc_0 - offset # = "R(r_c)"

    x1 = np.log(jnp.abs(phi_rc_shifted))
    x2 = phi_rc_1 / phi_rc_shifted
    x3 = phi_rc_2 / phi_rc_shifted
    x4 = -Z * (phi_new_0 + phi_0_others) / (phi_new_0 - offset)
    x5 = np.log(np.abs(phi_new_0 - offset))

    rc2 = r_c*r_c
    rc3 = rc2 * r_c
    rc4 = rc2 * rc2
    poly = np.array([
        x5,
        x4,
        6*x1/rc2 - 3*x2/r_c + 0.5 * x3 -3*x4/r_c - 6*x5/rc2 - 0.5*x2*x2,
        -8*x1/rc3 + 5*x2/rc2 - x3/r_c + 3*x4/rc2 + 8*x5/rc3 + x2*x2/r_c,
        3*x1/rc4 - 2*x2/rc3 + 0.5*x3/rc2 - x4/rc3 - 3*x5/rc4 - 0.5*x2*x2/rc2
    ])
    E_loc_cusp = _get_local_energy_of_cusp_orbital(r_c, offset, sign, poly, Z, phi_0_others)
    ind_opt = np.nanargmin(E_loc_cusp)
    return offset[ind_opt], sign[ind_opt], poly[:,ind_opt]

def _calculate_ao_cusp_params(ind_nuc, alpha, gto_coeffs, angular_momenta, Z):
    if sum(angular_momenta) != 0:
        return None
    r_c = jnp.minimum(0.5, 1/Z)
    phi_rc_0 = jnp.sum(gto_coeffs * np.exp(-alpha * r_c ** 2))
    phi_rc_1 = jnp.sum(gto_coeffs * (-2 * alpha * r_c) * np.exp(-alpha * r_c ** 2))
    phi_rc_2 = jnp.sum(gto_coeffs * (-2 * alpha + 4 * (alpha * r_c) ** 2) * np.exp(-alpha * r_c ** 2))
    phi_0 = jnp.sum(gto_coeffs)

    n_cusp_trials = 500
    phi_new_0 = phi_0 * np.linspace(-1, 3, n_cusp_trials)

    sign = jnp.sign(phi_new_0 - phi_rc_0)
    offset = 2 * phi_rc_0 - phi_new_0
    phi_shifted = phi_rc_0 - offset

    p0 = jnp.log((phi_new_0 - offset) * sign)
    p1 = -Z * (offset * sign * jnp.exp(-p0) + 1)

    A = jnp.array([[r_c ** 2, r_c ** 3, r_c ** 4],
                   [2 * r_c, 3 * r_c ** 2, 4 * r_c ** 3],
                   [2, 6 * r_c, 12 * r_c ** 2]]
                  )
    b = jnp.array([
        jnp.log(phi_shifted * sign) - p0 - p1 * r_c,
        phi_rc_1 / phi_shifted - p1,
        phi_rc_2 / phi_shifted - (phi_rc_1 / phi_shifted) ** 2
    ])
    poly = jnp.concatenate([jnp.array([p0, p1]), jnp.linalg.solve(A, b)])
    E_loc_cusp = _get_local_energy_of_cusp_orbital(r_c, offset, sign, poly, Z, phi_others=0.0)
    ind_opt = jnp.nanargmin(E_loc_cusp)
    return r_c, offset[ind_opt], sign[ind_opt], poly[:,ind_opt]

def calculate_molecular_orbital_cusp_params(atomic_orbitals, mo_coeff, R, Z, r_cusp_scale):
    n_molecular_orbitals, n_nuclei, n_atomic_orbitals = mo_coeff.shape[1], len(R), len(atomic_orbitals)
    cusp_rc = jnp.minimum(r_cusp_scale / Z, 0.5)
    cusp_offset = np.zeros([n_nuclei, n_molecular_orbitals])
    cusp_sign = np.zeros([n_nuclei, n_molecular_orbitals])
    cusp_poly = np.zeros([n_nuclei, n_molecular_orbitals, 5])
    cusp_1s_coeffs = np.zeros([n_nuclei, n_atomic_orbitals, n_molecular_orbitals])
    for nuc_idx in range(n_nuclei):
        for mo_idx in range(n_molecular_orbitals):
            diff, dist = get_el_ion_distance_matrix(jnp.array(R[nuc_idx]), R)
            ao = eval_atomic_orbitals(diff, dist, atomic_orbitals)
            is_centered_1s = np.array([(a[0] == nuc_idx) and (sum(a[3]) == 0) for a in atomic_orbitals])
            phi_0_1s = (is_centered_1s * ao) @ mo_coeff[:, mo_idx]
            phi_0_others = ((1-is_centered_1s) * ao) @ mo_coeff[:, mo_idx]
            phi_rc_0, phi_rc_1, phi_rc_2 = 0.0, 0.0, 0.0
            r_c = cusp_rc[nuc_idx]
            for i, (_, alpha, weights, _) in enumerate(atomic_orbitals):
                if is_centered_1s[i]:
                    phi_rc_0 += jnp.sum(weights * np.exp(-alpha * r_c ** 2)) * mo_coeff[i, mo_idx]
                    phi_rc_1 += jnp.sum(weights * (-2 * alpha * r_c) * np.exp(-alpha * r_c ** 2)) * mo_coeff[i, mo_idx]
                    phi_rc_2 += jnp.sum(weights * (-2 * alpha + 4 * (alpha * r_c) ** 2) * np.exp(-alpha * r_c ** 2)) * mo_coeff[i, mo_idx]
            cusp_1s_coeffs[nuc_idx, :, mo_idx] = is_centered_1s * mo_coeff[:, mo_idx] # n_nuc x n_atomic_orbitals x n_molec_orbitals
            cusp_offset[nuc_idx, mo_idx], cusp_sign[nuc_idx, mo_idx], cusp_poly[nuc_idx, mo_idx] = _calculate_mo_cusp_params(phi_rc_0,
                                                                                                                             phi_rc_1,
                                                                                                                             phi_rc_2,
                                                                                                                             phi_0_1s,
                                                                                                                             phi_0_others,
                                                                                                                             Z[nuc_idx],
                                                                                                                             r_c)
    return cusp_rc, cusp_offset, cusp_sign, cusp_poly, cusp_1s_coeffs

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import jax
    from configuration import PhysicalConfig, CASSCFConfig, CuspCorrectionConfig
    # physical_config = PhysicalConfig(name='LiH', n_cas_electrons=3, n_cas_orbitals=3)
    # casscf_config_ao = CASSCFConfig(n_determinants=1, basis_set="6311G", cusps=CuspCorrectionConfig(cusp_type="ao"))
    # casscf_config_mo = CASSCFConfig(n_determinants=1, basis_set="6311G", cusps=CuspCorrectionConfig(cusp_type="mo"))
    #
    #
    # R = np.array(physical_config.R)
    # Z = np.array(physical_config.Z)
    #
    # (atomic_orbitals, cusp_params_ao, _, mo_coeff, ind_orbitals, ci_weights), _ = get_baseline_solution(physical_config, casscf_config_ao)
    # (atomic_orbitals, _, cusp_params_mo, mo_coeff, ind_orbitals, ci_weights), _ = get_baseline_solution(physical_config, casscf_config_mo)
    #
    #
    # ind_mo = 0
    # assert ind_mo < mo_coeff[0].shape[-1]
    # def phi_single_electron(r, cusp_params, cusp_type):
    #     r = jnp.expand_dims(r, axis=-2) # add pseudo-axis for multiple electrons
    #     diff = r[..., np.newaxis,:] - R[np.newaxis,...]
    #     dist = jnp.linalg.norm(diff, axis=-1)
    #     if (cusp_params is not None) and (cusp_type == "mo"):
    #         cusp_params = cusp_params[0]
    #     return evaluate_molecular_orbitals(diff, dist, atomic_orbitals, mo_coeff[0], cusp_params, cusp_type)[...,0, ind_mo]
    #
    #
    # def build_kinetic_energy(phi):
    #     def _ekin(r):
    #         eye = jnp.eye(3)
    #         grad_func = jax.grad(phi)
    #
    #         def _loop_body(i, laplacian):
    #             _, G_ii = jax.jvp(grad_func, (r,), (eye[i],))
    #             return laplacian + G_ii[i]
    #
    #         return -0.5 * jax.lax.fori_loop(0, 3, _loop_body, 0.0) / phi(r)
    #     return jax.jit(jax.vmap(_ekin))
    #
    # def calc_potential_energy(r):
    #     _, dist = get_el_ion_distance_matrix(r, R)
    #     E = -jnp.sum(Z/(dist+1e-10), axis=-1)
    #     return E
    #
    # N_samples = 1001
    # r = np.zeros([N_samples, 3])
    # r[:, 0] = np.linspace(-1, 4, N_samples)
    # phi_naive_func = lambda r: phi_single_electron(r, None, "ao")
    # phi_corr_func_ao = lambda r: phi_single_electron(r, cusp_params_ao, "ao")
    # phi_corr_func_mo = lambda r: phi_single_electron(r, cusp_params_mo, "mo")
    #
    # Ekin_naive_func = build_kinetic_energy(phi_naive_func)
    # Ekin_corr_func_ao = build_kinetic_energy(phi_corr_func_ao)
    # Ekin_corr_func_mo = build_kinetic_energy(phi_corr_func_mo)
    # phi_naive = phi_naive_func(r)
    # phi_corr_ao = phi_corr_func_ao(r)
    # phi_corr_mo = phi_corr_func_mo(r)
    #
    #
    # Eloc_naive = Ekin_naive_func(r) + calc_potential_energy(r)
    # Eloc_corr_ao = Ekin_corr_func_ao(r) + calc_potential_energy(r)
    # Eloc_corr_mo = Ekin_corr_func_mo(r) + calc_potential_energy(r)
    #
    # plt.close("all")
    # plt.subplot(2,1,1)
    # plt.plot(r[:, 0], phi_corr_ao, label="Cusp corrected AO")
    # plt.plot(r[:, 0], phi_corr_mo, label="Cusp corrected MO")
    # plt.plot(r[:, 0], phi_naive, '--', label="Naive")
    # plt.grid()
    # plt.legend()
    #
    # plt.subplot(2,1,2)
    # plt.plot(r[:, 0], Eloc_corr_ao, label="Cusp corrected AO", alpha=0.5)
    # plt.plot(r[:, 0], Eloc_corr_mo, label="Cusp corrected MO", alpha=0.5)
    # plt.plot(r[:, 0], Eloc_naive, '--', label="Naive")
    # # plt.ylim([-60, 60])
    # plt.grid()
    # plt.legend()

