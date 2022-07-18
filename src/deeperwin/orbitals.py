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


def get_envelope_exponents_from_atomic_orbitals(physical_config: PhysicalConfig, basis_set='6-31G', pad_full_det=False, leading_dims=None):
    n_up, n_dn, n_ions = physical_config.n_up, physical_config.n_dn, physical_config.n_ions
    (atomic_orbitals, _, _, mo_coeff, _, _), _ = get_baseline_solution(physical_config, CASSCFConfig(basis_set=basis_set), n_dets=1)
    mo_coeff = mo_coeff[0][:, :n_up], mo_coeff[1][:, :n_dn] # keep only occupied orbitals
    alpha_ao = _get_atomic_orbital_envelope_exponents(physical_config, basis_set)

    ind_nuc_ao = np.array([ao[0] for ao in atomic_orbitals], dtype=int)
    c_values = [np.zeros([n_up, n_ions]), np.zeros([n_dn, n_ions])]
    alpha_values = [np.zeros([n_up, n_ions]), np.zeros([n_dn, n_ions])]
    for spin in range(2):
        mo_weights = mo_coeff[spin] ** 2 / np.sum(mo_coeff[spin] ** 2, axis=0) # relative contribution of each ao
        for ind_mo in range([n_up, n_dn][spin]):
            for ind_nuc in range(n_ions):
                index_nuc = (ind_nuc_ao == ind_nuc)
                mo_weights_nuc = mo_weights[:, ind_mo][index_nuc]
                c_values[spin][ind_mo, ind_nuc] = np.sum(mo_weights_nuc) + 1e-3
                alpha_values[spin][ind_mo, ind_nuc] = (np.dot(mo_weights_nuc, alpha_ao[index_nuc]) + 1e-6) / (np.sum(mo_weights_nuc) + 1e-6)

    for spin in range(2):
        # undo softplus which will be applied in network
        alpha_values[spin] = np.log(np.exp(alpha_values[spin] - 1))

    if pad_full_det:
        c_values[0] = np.concatenate([c_values[0], np.ones([n_dn, n_ions])], axis=0)
        alpha_values[0] = np.concatenate([alpha_values[0], np.ones([n_dn, n_ions])], axis=0)
        c_values[1] = np.concatenate([np.ones([n_up, n_ions]), c_values[1]], axis=0)
        alpha_values[1] = np.concatenate([np.ones([n_up, n_ions]), alpha_values[1]], axis=0)
    
    if leading_dims:
        c_values = [np.tile(x, tuple(leading_dims) + (1,1)) for x in c_values]
        alpha_values = [np.tile(x, tuple(leading_dims) + (1,1)) for x in alpha_values]

    return c_values, alpha_values

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

