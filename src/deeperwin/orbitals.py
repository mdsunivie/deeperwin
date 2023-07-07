"""
Contains the physical baseline model (e.g. CASSCF).

This module provides functionality to calculate a basline solution using pySCF and functions to
"""
import logging
import jax
import jax.numpy as jnp
import pyscf
import pyscf.ci
import pyscf.lo
import pyscf.mcscf
import numpy as np
from deeperwin.configuration import PhysicalConfig, CASSCFConfig
from deeperwin.utils.utils import get_el_ion_distance_matrix, generate_exp_distributed, PERIODIC_TABLE
from collections import Counter
import chex
from typing import List, Tuple, Optional, Union, Dict
import dataclasses
import functools

logger = logging.getLogger("dpe")


@chex.dataclass
class AtomicOrbital:
    Z: jnp.array  # 0-dim int
    idx_atom: jnp.array  # 0-dim int
    idx_basis: jnp.array  # 0-dim int
    alpha: jnp.array
    weights: jnp.array
    angular_momenta: jnp.array  # int
    cusp_params: Optional[Tuple[jnp.array]] = None

    @property
    def l_tot(self):
        """Total angular momentum, ie type of orbital (0=s, 1=p, 2=d, ...)"""
        # TODO: This is currently a python sum instead of jnp.sum, because it is used in a traced expression
        #  (the initialization of the backflow envelope exponents); Is this ok? Will that cause issues when changing molecule/geometry?
        return sum(self.angular_momenta)

    @property
    def id_string(self):
        PERIODIC_TABLE = (
            "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr".split()
        )
        atom_type = PERIODIC_TABLE[int(self.Z) - 1] + str(int(self.idx_atom))
        orbital_type = ["s", "p", "d", "f"][self.l_tot]
        for power, axis in zip(self.angular_momenta, ["x", "y", "z"]):
            orbital_type += axis * power
        return atom_type + " " + orbital_type


@chex.dataclass
class OrbitalParams:
    atomic_orbitals: List[AtomicOrbital] = dataclasses.field(default_factory=lambda: [])
    # float; [n_basis x n_orbitals], [n_basis x n_orbitals]
    mo_coeff: Tuple[jnp.array, jnp.array] = (None, None)
    idx_orbitals: Tuple[jnp.array, jnp.array] = (None, None)  # int; [n_dets x n_up], [n_dets x n_dn]
    mo_energies: Tuple[jnp.array, jnp.array] = (None, None)  # float; [n_orbitals], [n_orbitals]
    ci_weights: jnp.array = None
    mo_cusp_params: Union[Tuple[jnp.array], Tuple[None, None]] = (None, None)


#################################################################################
############################ Orbital functions ##################################
#################################################################################


def eval_gaussian_orbital(el_ion_diff, el_ion_dist, ao: AtomicOrbital):
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

    r_sqr = jnp.expand_dims(el_ion_dist**2, axis=-1)  # shape [1]
    pre_fac = jnp.prod(el_ion_diff**ao.angular_momenta, axis=-1)  # shape []
    phi_gto = pre_fac * jnp.sum(ao.weights * jnp.exp(-ao.alpha * r_sqr), axis=-1)
    return phi_gto


def eval_atomic_orbitals(el_ion_diff, el_ion_dist, atomic_orbitals: List[AtomicOrbital]):
    """
    Args:
        el_ion_diff: shape [N_batch x n_el x N_ion x 3]
        el_ion_dist: shape [N_batch x n_el x N_ion]
        atomic_orbitals:

    Returns:

    """
    outputs = []
    for i, ao in enumerate(atomic_orbitals):
        diff = el_ion_diff[..., ao.idx_atom, :]
        dist = el_ion_dist[..., ao.idx_atom]
        gto = eval_gaussian_orbital(diff, dist, ao)
        if (ao.cusp_params is not None) and (ao.cusp_params[i] is not None):
            r_c = ao.cusp_params[i][0]
            sto = _eval_cusp_atomic_orbital(dist, *ao.cusp_params[i])
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
    psi = offset + sign * jnp.exp(p_r)
    return psi


def _eval_cusp_atomic_orbital(dist, r_c, offset, sign, poly):
    dist = jnp.minimum(dist, r_c)
    n = jnp.arange(5)
    r_n = dist[..., jnp.newaxis] ** n
    p_r = jnp.dot(r_n, poly)
    psi = offset + sign * jnp.exp(p_r)
    return psi


def evaluate_molecular_orbitals(el_ion_diff, el_ion_dist, atomic_orbitals, mo_coeff, mo_cusp_params=None):
    aos = eval_atomic_orbitals(el_ion_diff, el_ion_dist, atomic_orbitals)
    mos = aos @ mo_coeff

    if mo_cusp_params is not None:
        r_c, offset, sign, poly, coeff_1s = mo_cusp_params
        sto_mask = jnp.heaviside(r_c - el_ion_dist, 0.0)[..., jnp.newaxis]
        sto = _eval_cusp_molecular_orbital(el_ion_dist, r_c, offset, sign, poly) - jnp.sum(
            aos[..., np.newaxis, :, np.newaxis] * coeff_1s, axis=-2
        )
        sto = jnp.sum(
            sto * sto_mask, axis=-2
        )  # sum over all ions, masking only the once that are within r_c of an electron
        mos = mos + sto  # subtract GTO-part of 1s orbitals and replace by STO-part
    return mos

def get_atomic_orbital_descriptors(
        mo_coeffs,
        Z,
        all_elements: List[int],
        n_basis_per_Z: Dict[int, int],
):
    n_ions = len(Z)
    if isinstance(n_basis_per_Z[Z[0]], dict):
        n_basis_per_Z = {z: sum(n_per_l.values()) for z,n_per_l in n_basis_per_Z.items()}
    offsets = np.array([0] + [n_basis_per_Z[z] for z in all_elements], int)
    offsets = np.cumsum(offsets)
    n_basis_total = offsets[-1]
    slice_tgt = {z: slice(offsets[i], offsets[i + 1]) for i, z in enumerate(all_elements)}

    features_up = np.zeros([n_ions, mo_coeffs[0].shape[1], n_basis_total])
    features_dn = np.zeros([n_ions, mo_coeffs[1].shape[1], n_basis_total])
    ind_src_start = 0
    for i,z in enumerate(Z):
        n_basis = n_basis_per_Z[z]
        features_up[i, :, slice_tgt[z]] = mo_coeffs[0][ind_src_start:ind_src_start + n_basis, :].T
        features_dn[i, :, slice_tgt[z]] = mo_coeffs[1][ind_src_start:ind_src_start + n_basis, :].T
        ind_src_start += n_basis
    return [features_up, features_dn]


# def get_atomic_orbital_descriptors(
#     orbital_params: OrbitalParams,
#     all_elements: List[int],
#     basis_set: str,
#     ao_mapping: Optional[Dict] = None,
#     get_all_atomic_orbitals=False,
# ):
#     # Construct a mapping from element and basis index to flat, zero-padded descriptor vector
#     if ao_mapping is None:
#         all_atomic_orbitals = _get_all_basis_functions(all_elements, basis_set)
#         ao_mapping = {(int(ao.Z), int(ao.idx_basis)): i for i, ao in enumerate(all_atomic_orbitals)}

#     # Get shapes of output matrix
#     n_ions = max([int(ao.idx_atom) for ao in orbital_params.atomic_orbitals]) + 1
#     feature_dim = len(ao_mapping)
#     orbital_descriptors = []
#     for spin, mo_coeff in enumerate(orbital_params.mo_coeff):
#         n_orbitals = mo_coeff.shape[1]
#         features = np.zeros([n_ions, n_orbitals, feature_dim])
#         # Loop over all basis functions used in the molecule and map their mo_coeff values (i.e. rows) to the corresponding feature
#         for idx_ao, ao in enumerate(orbital_params.atomic_orbitals):
#             idx_feature = ao_mapping[(int(ao.Z), int(ao.idx_basis))]
#             features[ao.idx_atom, :, idx_feature] = mo_coeff[idx_ao, :]
#         orbital_descriptors.append(features)

#     if get_all_atomic_orbitals:
#         return orbital_descriptors, all_atomic_orbitals
#     else:
#         return orbital_descriptors


def get_atomic_orbital_descriptors_e3(
    orbital_params: OrbitalParams, all_elements: List[int], basis_set: str, ao_mapping: Optional[Dict] = None
):
    # Construct a mapping from element and basis index to flat, zero-padded descriptor vector
    if ao_mapping is None:
        all_atomic_orbitals = _get_all_basis_functions(all_elements, basis_set)
        ao_mapping = {(int(ao.Z), int(ao.idx_basis)): i for i, ao in enumerate(all_atomic_orbitals)}

    # Get shapes of output matrix
    n_ions = max([int(ao.idx_atom) for ao in orbital_params.atomic_orbitals]) + 1
    feature_dim = len(ao_mapping)
    orbital_descriptors = []
    for spin, mo_coeff in enumerate(orbital_params.mo_coeff):
        n_orbitals = mo_coeff.shape[1]
        features = np.zeros([n_ions, n_orbitals, feature_dim])
        # Loop over all basis functions used in the molecule and map their mo_coeff values (i.e. rows) to the corresponding feature
        for idx_ao, ao in enumerate(orbital_params.atomic_orbitals):
            idx_feature = ao_mapping[(int(ao.Z), int(ao.idx_basis))]
            features[ao.idx_atom, :, idx_feature] = mo_coeff[idx_ao, :]
        orbital_descriptors.append(features)
    return orbital_descriptors


def get_sum_of_atomic_exponentials(dist_el_ion, exponent=1.0, scale=1.0):
    phi_atoms = jnp.exp(-dist_el_ion * exponent)  # [batch x el x ion]
    return jnp.sum(phi_atoms, axis=-1) * scale  # [batch x el]


#################################################################################
####################### Compile-time helper functions ###########################
#################################################################################


def build_pyscf_molecule(R, Z, charge=0, spin=0, basis_set="6-311G"):
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
    molecule.basis = get_basis_set(basis_set, set(Z))
    molecule.cart = True
    molecule.spin = spin  # 2*n_up - n_down
    molecule.charge = charge
    molecule.output = "/dev/null"
    molecule.verbose = 0  # suppress output to console
    # maximum memory in megabytes (i.e. 10e3 = 10GB)
    molecule.max_memory = 10e3
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
    n_basis_functions_total = 0
    atomic_orbitals = []
    for idx_atom, (element, atom_pos) in enumerate(molecule._atom):
        idx_basis_per_atom = 0
        for gto_data in molecule._basis[element]:
            l = gto_data[0]
            gto_data = np.array(gto_data[1:])
            alpha = gto_data[:, 0]
            weights = gto_data[:, 1:]
            for ind_contraction in range(weights.shape[1]):
                n_orientations = [1, 3, 6][l]
                # 1,3,6, ... = number of orbitals per angular momentum
                for m in range(n_orientations):
                    # string of the form 'xxy' or 'zz'
                    shape_string = ao_labels[n_basis_functions_total][3]
                    angular_momenta = np.array([shape_string.count(x) for x in ["x", "y", "z"]], dtype=int)
                    normalization = _get_gto_normalization_factor(alpha, angular_momenta)
                    ao = AtomicOrbital(
                        Z=np.array(molecule.atom_charge(idx_atom), int),
                        idx_atom=np.array(idx_atom, int),
                        idx_basis=idx_basis_per_atom,
                        alpha=alpha,
                        weights=weights[:, ind_contraction] * normalization,
                        angular_momenta=angular_momenta,
                    )
                    atomic_orbitals.append(ao)
                    n_basis_functions_total += 1
                    idx_basis_per_atom += 1
    assert (
        len(atomic_orbitals) == n_basis
    ), "Could not properly construct basis functions. You probably tried to use a valence-only basis (e.g. cc-pVDZ) for an all-electron calculation."  # noqa
    return atomic_orbitals

def _get_orbital_mapping(atomic_orbitals, all_elements):
    orbitals_per_Z = {key: {} for key in all_elements}
    for ao in atomic_orbitals:
        Z = int(ao.Z)
        ang_momentum = ao.l_tot
        if ang_momentum in orbitals_per_Z[Z].keys():
            orbitals_per_Z[Z][ang_momentum] += 1
        else:
            orbitals_per_Z[Z][ang_momentum] = 1

    #nb_orbitals_per_Z = {key: sum(orbitals_per_Z[key].values()) for key in all_elements}
    irrep = []
    for orb_type, nb_orb in orbitals_per_Z[max(all_elements)].items():
        if orb_type == 0:
            irrep.append(f"{nb_orb}x0e")
        elif orb_type == 1:
            irrep.append(f"{nb_orb//3}x1o")
        elif orb_type == 2:
            raise "Not yet implemented for d-type orbitals"
    irrep = "+".join(irrep)

    return orbitals_per_Z, irrep

def _get_all_basis_functions(Z_values: List[int], basis_set):
    """Get all possible basis functions for a given set of chemical elements"""
    # Build a fictional "molecule" with all elements stacked on top of each other
    Z_values = sorted(list(set(Z_values)))
    R = np.zeros([len(Z_values), 3])
    charge = 0
    spin = sum(Z_values) % 2
    molecule = build_pyscf_molecule(R, Z_values, charge, spin, basis_set)
    return _get_atomic_orbital_basis_functions(molecule)


def build_pyscf_molecule_from_physical_config(physical_config: PhysicalConfig, basis_set):
    n_electrons, n_up, R, Z = physical_config.get_basic_params()
    charge = np.sum(Z) - n_electrons
    spin = 2 * n_up - n_electrons
    return build_pyscf_molecule(R, Z, charge, spin, basis_set)


def get_hartree_fock_solution(physical_config: PhysicalConfig, basis_set):
    molecule = build_pyscf_molecule_from_physical_config(physical_config, basis_set) # Bauen basissatz; rausziehen; atomic orbitals= Liste aus orbital mit property l-tot
    atomic_orbitals = _get_atomic_orbital_basis_functions(molecule)
    hf = pyscf.scf.HF(molecule)
    hf.verbose = 0  # suppress output to console
    hf.kernel()
    return atomic_orbitals, hf


def get_basis_set(basis_set: Union[str, Dict], atom_charges=None):
    """ "
    Convert strings of the form H:6-31G**__O:6-311G__default:6-31G to dicts of the same information for pyscf
    """
    if isinstance(basis_set, dict):
        return basis_set
    if isinstance(basis_set, str):
        if "__" in basis_set:
            # Split the basis-set string into a dict of atom-symbols and basis-strings
            atom_basis_strings = basis_set.split("__")
            basis_dict_full = dict()
            for atom_basis in atom_basis_strings:
                atom_symbol, basis = atom_basis.split(":")
                basis_dict_full[atom_symbol] = basis

            # Build a basis-dict, containing only the basis functions which are actually needed
            basis_dict = dict()
            used_atom_symbols = [PERIODIC_TABLE[Z - 1] for Z in set(atom_charges)]
            for atom_symbol in used_atom_symbols:
                if atom_symbol in basis_dict_full:
                    basis_dict[atom_symbol] = basis_dict_full[atom_symbol]
                else:
                    basis_dict[atom_symbol] = basis_dict_full["default"]
            return basis_dict
        return basis_set
    else:
        raise ValueError(f"Unknown dtype of basis-set: {type(basis_set)}")
    
@functools.lru_cache
def get_n_basis_per_Z(basis_set: str, Z_values: tuple):
    """Get the number of basis functions for a given set of chemical elements"""
    basis_dict = get_basis_set(basis_set, Z_values)
    all_basis_functions = _get_all_basis_functions(Z_values, basis_dict)
    n_basis_per_Z = {}
    for ao in all_basis_functions:
        Z = int(ao.Z)
        l_tot = int(ao.l_tot)
        if Z not in n_basis_per_Z:
            n_basis_per_Z[Z] = {}
        if ao.l_tot not in n_basis_per_Z[Z]:
            n_basis_per_Z[Z][l_tot] = 0
        n_basis_per_Z[Z][l_tot] += 1
    return n_basis_per_Z


def localize_orbitals(mol, mo_coeffs_occ, method, orb_energies=None, canonicalize=True):
    if method is None:
        mo_loc = mo_coeffs_occ
    elif method == "boys":
        localizer = pyscf.lo.Boys(mol, mo_coeffs_occ)
        mo_loc = localizer.kernel()
    elif method == "pm":
        localizer = pyscf.lo.PipekMezey(mol, mo_coeffs_occ)
        mo_loc = localizer.kernel()
    elif method == 'cholesky':
        mo_loc = pyscf.lo.cholesky_mos(mo_coeffs_occ)
    elif method == 'qr':
        _, R = np.linalg.qr(mo_coeffs_occ.T)
        mo_loc = R.T
    elif method == "iao":
        mo_loc = pyscf.lo.iao.iao(mol, mo_coeffs_occ)
    elif method == "ibo":
        mo_loc = pyscf.lo.ibo.ibo(mol, mo_coeffs_occ, locmethod="PM").kernel()
    else:
        raise ValueError(f"Unknown orbital localization method: {method}")
    if orb_energies is not None:
        E_loc = get_local_orb_energy_estimate(mo_coeffs_occ, mo_loc, orb_energies)
        mo_loc = mo_loc[:, np.argsort(E_loc)]

    if canonicalize:
        mo_loc = mo_loc * np.sign(np.sum(mo_loc, axis=0))
    
    if orb_energies is None:
        return mo_loc
    else:
        return mo_loc, E_loc


def get_local_orb_energy_estimate(mo_coeff, mo_coeff_loc, mo_E):
    # Sort orbitals according to their energies
    U = np.linalg.lstsq(mo_coeff, mo_coeff_loc, rcond=None)[0]
    weights = U**2
    weights = weights / np.sum(weights, axis=0, keepdims=True)
    local_orb_energies = mo_E @ weights
    return local_orb_energies


def get_baseline_solution(physical_config: PhysicalConfig, casscf_config: CASSCFConfig, n_dets: int):
    n_el, n_up, R, Z = physical_config.get_basic_params()
    n_dn = n_el - n_up
    atomic_orbitals, hf = get_hartree_fock_solution(physical_config, casscf_config.basis_set)

    if n_dets == 1 or casscf_config.only_hf:
        # Only 1 determinant => no MRCI/CASSCF calculation required
        mo_coeff = hf.mo_coeff
        mo_energies = hf.mo_energy
        if mo_coeff.ndim == 2:
            mo_coeff = [mo_coeff, mo_coeff]
            mo_energies = [mo_energies, mo_energies]
        elif mo_coeff.ndim == 3:
            mo_coeff = [mo_coeff[0], mo_coeff[1]]
            mo_energies = [mo_energies[0], mo_energies[0]]
        else:
            raise ValueError("Unknown output shape of pySCF calculation")
        ci_weights = np.ones(1) if not casscf_config.only_hf else np.ones(n_dets)
        if casscf_config.only_hf:
            ind_orbitals = [
                np.tile(np.arange(n_up), n_dets).reshape((-1, n_up)),
                np.tile(np.arange(n_dn), n_dets).reshape((-1, n_dn)),
            ]
        else:
            ind_orbitals = [np.arange(n_up)[None, :], np.arange(n_dn)[None, :]]

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
            share_captured = np.sum(ci_weights[ind_largest] ** 2) / np.sum(ci_weights**2)
            logger.debug(f"Share of CI-weights captured by {n_dets} dets: {share_captured:.3%}")
            ci_weights = ci_weights[ind_largest]
            ci_weights = jnp.array(ci_weights / np.sum(ci_weights**2))
            ind_orbitals = ind_orbitals[0][ind_largest], ind_orbitals[1][ind_largest]

    if casscf_config.localization:
        assert (n_dets == 1) or casscf_config.only_hf, "Orbital localization is potentially incompatible with CASSCF"
        mol = build_pyscf_molecule_from_physical_config(physical_config, casscf_config.basis_set)
        for spin, n_occ in enumerate([n_up, n_dn]):
            mo_coeff[spin][:, :n_occ], mo_energies[spin][:n_occ] = localize_orbitals(
                mol, mo_coeff[spin][:, :n_occ], casscf_config.localization, mo_energies[spin][:n_occ]
            )
            mo_coeff[spin][:, n_occ:], mo_energies[spin][n_occ:] = localize_orbitals(
                mol, mo_coeff[spin][:, n_occ:], casscf_config.localization, mo_energies[spin][n_occ:]
            )

    orbital_params = OrbitalParams(
        atomic_orbitals=atomic_orbitals,
        mo_coeff=mo_coeff,
        idx_orbitals=ind_orbitals,
        mo_energies=mo_energies,
        ci_weights=ci_weights,
    )
    # Calculate cusp-correction-parameters for molecular orbitals
    if casscf_config.cusps and (casscf_config.cusps.cusp_type == "mo"):
        orbital_params.mo_cusp_params = [
            calculate_molecular_orbital_cusp_params(
                atomic_orbitals, mo_coeff[i], R, Z, casscf_config.cusps.r_cusp_el_ion_scale
            )
            for i in range(2)
        ]
    elif casscf_config.cusps and (casscf_config.cusps.cusp_type == "ao"):
        for ao in orbital_params.atomic_orbitals:
            ao.cusp_params = _calculate_ao_cusp_params(ao, physical_config.Z[ao.idx_atom])
    return orbital_params, (E_hf, E_casscf)


def _get_effective_charge(Z: int, n: int, s_lower_shell=0.85, s_same_shell=0.35):
    """
    Calculates the approximate effective charge for an electron, using Slater's Rule of shielding.

    Args:
        Z: Nuclear charge of an atom
        n: Principal quantum number of the elctron for which the effective charge is being calculated

    Returns:
        Z_eff (float): effective charge
    """
    shielding = 0
    for n_shell in range(1, n + 1):
        n_electrons_in_shell = 2 * n_shell**2
        if n_shell == n:
            n_el_in_lower_shells = sum([2 * k**2 for k in range(1, n_shell)])
            n_electrons_in_shell = min(Z - n_el_in_lower_shells, n_electrons_in_shell) - 1

        if n_shell == n:
            shielding += n_electrons_in_shell * s_same_shell
        elif n_shell == (n - 1):
            shielding += n_electrons_in_shell * s_lower_shell
        else:
            shielding += n_electrons_in_shell
    return max(Z - shielding, 1)


def _get_electron_configuration(n_el):
    shells = "1s,2s,2p,3s,3p,4s,3d,4p,5s,4d,5p,6s,4f,5d,6p,5f,6d".split(",")
    n_el_per_shell = dict(s=2, p=6, d=10, f=14)
    el_config = Counter()
    while n_el > 0:
        shell = shells.pop(0)
        n = int(shell[0])
        n_el_in_orb = min(n_el, n_el_per_shell[shell[1]])
        el_config[n] += n_el_in_orb
        n_el -= n_el_in_orb
    return dict(el_config)


def _initialize_walkers_around_atom(rng, R, Z, N_walkers, n_el, n_up):
    el_config = _get_electron_configuration(n_el)
    r = []
    spin = []
    for n, n_el_in_shell in el_config.items():
        subkey, rng = jax.random.split(rng)
        Z_eff = _get_effective_charge(Z, n, 1.0, 0.7)
        exponent = 2 * Z_eff / n
        r.append(generate_exp_distributed(subkey, [N_walkers, n_el_in_shell], exponent))
        n_up_in_shell = max(min(n_el_in_shell // 2, n_up), n_el_in_shell - n_el + n_up)
        n_dn_in_shell = n_el_in_shell - n_up_in_shell
        n_up -= n_up_in_shell
        n_el -= n_el_in_shell
        spin += [0] * n_up_in_shell + [1] * n_dn_in_shell
    r = jnp.concatenate(r, axis=1) + np.array(R)
    is_up = np.array(spin) == 0
    return r[:, is_up, :], r[:, ~is_up, :]


def initialize_walkers_with_exponential_radial_pdf(rng, R, Z, N_walkers, n_el, n_up, el_ion_mapping):
    r_up = []
    r_dn = []
    assert n_el == len(el_ion_mapping), "Number of electrons does not match the number of indices in el_ion_mapping"
    el_ion_mapping_up = np.array(el_ion_mapping)[:n_up]
    el_ion_mapping_dn = np.array(el_ion_mapping)[n_up:]
    for ind_ion, (R_, Z_) in enumerate(zip(R, Z)):
        subkey, rng = jax.random.split(rng)
        n_up_ion = sum(el_ion_mapping_up == ind_ion)
        n_el_ion = sum(el_ion_mapping_dn == ind_ion) + n_up_ion
        if n_el_ion == 0:
            continue
        r_atom_up, r_atom_dn = _initialize_walkers_around_atom(subkey, R_, Z_, N_walkers, n_el_ion, n_up_ion)
        r_up.append(r_atom_up)
        r_dn.append(r_atom_dn)
    return jnp.concatenate(r_up + r_dn, axis=1)


def _get_atomic_orbital_envelope_exponents(physical_config: PhysicalConfig, basis_set):
    molecule = build_pyscf_molecule_from_physical_config(physical_config, basis_set)
    atomic_orbitals = _get_atomic_orbital_basis_functions(molecule)

    Z_ions = np.array(physical_config.Z)
    n_ions = len(Z_ions)

    n_ao = []
    for ind_nuc in range(n_ions):
        ao_nuc = [ao for ao in atomic_orbitals if ao.idx_atom == ind_nuc]
        nr_of_s_orbitals = len([ao for ao in ao_nuc if ao.l_tot == 0])
        nr_of_p_orbitals = len([ao for ao in ao_nuc if ao.l_tot == 1])
        nr_of_d_orbitals = len([ao for ao in ao_nuc if ao.l_tot == 2])
        assert (nr_of_s_orbitals + nr_of_p_orbitals + nr_of_d_orbitals) == len(
            ao_nuc
        ), "Initialization currently does support d-orbitals"
        n_ao.append(np.arange(1, nr_of_s_orbitals + 1))
        n_ao.append(np.repeat(np.arange(2, nr_of_p_orbitals // 3 + 2), 3))
        n_ao.append(np.repeat(np.arange(3, nr_of_d_orbitals // 6 + 3), 6))
    n_ao = np.concatenate(n_ao)
    Z_ao = np.array([ao.Z for ao in atomic_orbitals])
    Z_ao = np.array([_get_effective_charge(Z, n) for Z, n in zip(Z_ao, n_ao)])
    alpha_ao = Z_ao / n_ao
    alpha_ao = alpha_ao * 0.7  # rescale to better fit empirical values
    return alpha_ao


def get_envelope_exponents_from_atomic_orbitals(physical_config: PhysicalConfig, basis_set="6-31G", pad_full_det=False):
    """
    Calculate an initial guess for the envelope parameters using a baseline calculation.

    Envelopes have the form coeff_I exp(-alpha_I * r).

    Returns:
         coeff_values: Tuple of len 2; for each spin has shape [n_ions x n_orb]
         alpha_values: Tuple of len 2; for each spin has shape [n_ions x n_orb]
    """
    n_up, n_dn, n_ions = physical_config.n_up, physical_config.n_dn, physical_config.n_ions
    orbitals, _ = get_baseline_solution(physical_config, CASSCFConfig(basis_set=basis_set, cusps=None), n_dets=1)
    # keep only occupied orbitals
    mo_coeff = orbitals.mo_coeff[0][:, :n_up], orbitals.mo_coeff[1][:, :n_dn]
    alpha_ao = _get_atomic_orbital_envelope_exponents(physical_config, basis_set)

    ind_nuc_ao = np.array([ao.idx_atom for ao in orbitals.atomic_orbitals], dtype=int)
    coeff_values = [np.zeros([n_ions, n_up]), np.zeros([n_ions, n_dn])]
    alpha_values = [np.zeros([n_ions, n_up]), np.zeros([n_ions, n_dn])]
    for spin in range(2):
        # relative contribution of each ao
        mo_weights = mo_coeff[spin] ** 2 / np.sum(mo_coeff[spin] ** 2, axis=0)
        for ind_mo in range([n_up, n_dn][spin]):
            for ind_nuc in range(n_ions):
                index_nuc = ind_nuc_ao == ind_nuc
                mo_weights_nuc = mo_weights[:, ind_mo][index_nuc]
                coeff_values[spin][ind_nuc, ind_mo] = np.sum(mo_weights_nuc) + 1e-3
                alpha_values[spin][ind_nuc, ind_mo] = (np.dot(mo_weights_nuc, alpha_ao[index_nuc]) + 1e-6) / (
                    np.sum(mo_weights_nuc) + 1e-6
                )

    for spin in range(2):
        # undo softplus which will be applied in network
        alpha_values[spin] = np.log(np.exp(alpha_values[spin] - 1))

    if pad_full_det:
        coeff_values[0] = np.concatenate([coeff_values[0], np.ones([n_ions, n_dn])], axis=1)
        alpha_values[0] = np.concatenate([alpha_values[0], np.ones([n_ions, n_dn])], axis=1)
        coeff_values[1] = np.concatenate([np.ones([n_ions, n_up]), coeff_values[1]], axis=1)
        alpha_values[1] = np.concatenate([np.ones([n_ions, n_up]), alpha_values[1]], axis=1)

    return coeff_values, alpha_values


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
    for i, ao in enumerate(atomic_orbitals):
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


def get_orbitals_from_rho(rho, n_occ):
    eigvals, eigvecs = np.linalg.eigh(rho * 0.5)
    eigvals = eigvals[::-1][:n_occ]
    eigvecs = eigvecs[:, ::-1][:, :n_occ]
    M = eigvecs * np.sqrt(eigvals)
    Q,R = np.linalg.qr(M.T)
    return R.T

def align_orbitals(mo, mo_ref, n_occ=None, adjust_scale=False):
    n_orb = mo.shape[-1]
    n_occ = n_occ or n_orb
    mo_unocc = mo[:, n_occ:]

    inner_products = jnp.einsum("bi,bj->ij", mo_ref[:, :n_occ], mo[:, :n_occ])
    max_indices = jnp.argmax(jnp.abs(inner_products), axis=1)
    signs = np.sign(inner_products[np.arange(n_occ), max_indices])

    mo_aligned = mo[:, max_indices]
    mo_aligned *= signs
    if adjust_scale:
        ref_norms = jnp.linalg.norm(mo_ref[:, :n_occ], axis=0)
        mo_norms = jnp.linalg.norm(mo_aligned, axis=0)
        mo_aligned *= ref_norms / mo_norms
    return jnp.concatenate([mo_aligned, mo_unocc], axis=-1)

def get_cosine_dist(mo, mo_ref, eps=1e-8):
    mo_norm = mo / (np.linalg.norm(mo, axis=-2, keepdims=True) + eps)
    mo_ref_norm = mo_ref / (np.linalg.norm(mo_ref, axis=-2, keepdims=True) + eps)
    inner_product = np.einsum("bk,bk->k", mo_norm, mo_ref_norm)
    return 1 - np.mean(inner_product, axis=-1)



##########################################################################################
##################################### Cusp functions #####################################
##########################################################################################
def _get_local_energy_of_cusp_orbital(r_c, offset, sign, poly, Z, phi_others=0.0):
    r = np.linspace(1e-6, r_c, 100)[:, np.newaxis]
    p_0 = poly[0] + poly[1] * r + poly[2] * r**2 + poly[3] * r**3 + poly[4] * r**4
    p_1 = poly[1] + 2 * poly[2] * r + 3 * poly[3] * r**2 + 4 * poly[4] * r**3
    p_2 = 2 * poly[2] + 6 * poly[3] * r + 12 * poly[4] * r**2
    prefac = sign * np.exp(p_0) / (offset + sign * np.exp(p_0) + phi_others)
    E_l = -prefac * (p_1 / r + 0.5 * p_2 + 0.5 * p_1**2) - Z / r
    penalty = np.nanvar(E_l, axis=0)
    # penalty = jnp.max(jnp.abs(E_l - E_l[-1]), axis=0)
    return penalty


def _calculate_mo_cusp_params(phi_rc_0, phi_rc_1, phi_rc_2, phi_0, phi_0_others, Z, r_c):
    if np.abs(phi_0) < 1e-6 and np.abs(phi_rc_0) < 1e-6:
        return 0.0, 0.0, np.zeros(5)
    n_cusp_trials = 500
    phi_new_0 = phi_0 * (
        1.0 + np.concatenate([np.logspace(-2, 1, n_cusp_trials // 2), -np.logspace(-2, 1, n_cusp_trials // 2)])
    )

    sign = jnp.sign(phi_new_0 - phi_rc_0)
    offset = 2 * phi_rc_0 - phi_new_0  # = "C"
    phi_rc_shifted = phi_rc_0 - offset  # = "R(r_c)"

    x1 = np.log(jnp.abs(phi_rc_shifted))
    x2 = phi_rc_1 / phi_rc_shifted
    x3 = phi_rc_2 / phi_rc_shifted
    x4 = -Z * (phi_new_0 + phi_0_others) / (phi_new_0 - offset)
    x5 = np.log(np.abs(phi_new_0 - offset))

    rc2 = r_c * r_c
    rc3 = rc2 * r_c
    rc4 = rc2 * rc2
    poly = np.array(
        [
            x5,
            x4,
            6 * x1 / rc2 - 3 * x2 / r_c + 0.5 * x3 - 3 * x4 / r_c - 6 * x5 / rc2 - 0.5 * x2 * x2,
            -8 * x1 / rc3 + 5 * x2 / rc2 - x3 / r_c + 3 * x4 / rc2 + 8 * x5 / rc3 + x2 * x2 / r_c,
            3 * x1 / rc4 - 2 * x2 / rc3 + 0.5 * x3 / rc2 - x4 / rc3 - 3 * x5 / rc4 - 0.5 * x2 * x2 / rc2,
        ]
    )
    E_loc_cusp = _get_local_energy_of_cusp_orbital(r_c, offset, sign, poly, Z, phi_0_others)
    ind_opt = np.nanargmin(E_loc_cusp)
    return offset[ind_opt], sign[ind_opt], poly[:, ind_opt]


def _calculate_ao_cusp_params(ao: AtomicOrbital, Z):
    if ao.l_tot != 0:
        return None
    r_c = jnp.minimum(0.5, 1 / Z)
    phi_rc_0 = jnp.sum(ao.weights * np.exp(-ao.alpha * r_c**2))
    phi_rc_1 = jnp.sum(ao.weights * (-2 * ao.alpha * r_c) * np.exp(-ao.alpha * r_c**2))
    phi_rc_2 = jnp.sum(ao.weights * (-2 * ao.alpha + 4 * (ao.alpha * r_c) ** 2) * np.exp(-ao.alpha * r_c**2))
    phi_0 = jnp.sum(ao.weights)

    n_cusp_trials = 500
    phi_new_0 = phi_0 * np.linspace(-1, 3, n_cusp_trials)

    sign = jnp.sign(phi_new_0 - phi_rc_0)
    offset = 2 * phi_rc_0 - phi_new_0
    phi_shifted = phi_rc_0 - offset

    p0 = jnp.log((phi_new_0 - offset) * sign)
    p1 = -Z * (offset * sign * jnp.exp(-p0) + 1)

    A = jnp.array([[r_c**2, r_c**3, r_c**4], [2 * r_c, 3 * r_c**2, 4 * r_c**3], [2, 6 * r_c, 12 * r_c**2]])
    b = jnp.array(
        [
            jnp.log(phi_shifted * sign) - p0 - p1 * r_c,
            phi_rc_1 / phi_shifted - p1,
            phi_rc_2 / phi_shifted - (phi_rc_1 / phi_shifted) ** 2,
        ]
    )
    poly = jnp.concatenate([jnp.array([p0, p1]), jnp.linalg.solve(A, b)])
    E_loc_cusp = _get_local_energy_of_cusp_orbital(r_c, offset, sign, poly, Z, phi_others=0.0)
    ind_opt = jnp.nanargmin(E_loc_cusp)
    return r_c, offset[ind_opt], sign[ind_opt], poly[:, ind_opt]


def calculate_molecular_orbital_cusp_params(atomic_orbitals: List[AtomicOrbital], mo_coeff, R, Z, r_cusp_scale):
    n_molecular_orbitals, n_nuclei, n_atomic_orbitals = mo_coeff.shape[1], len(R), len(atomic_orbitals)
    cusp_rc = np.minimum(r_cusp_scale / Z, 0.5)
    cusp_offset = np.zeros([n_nuclei, n_molecular_orbitals])
    cusp_sign = np.zeros([n_nuclei, n_molecular_orbitals])
    cusp_poly = np.zeros([n_nuclei, n_molecular_orbitals, 5])
    cusp_1s_coeffs = np.zeros([n_nuclei, n_atomic_orbitals, n_molecular_orbitals])
    for nuc_idx in range(n_nuclei):
        for mo_idx in range(n_molecular_orbitals):
            diff, dist = get_el_ion_distance_matrix(jnp.array(R[nuc_idx]), R)
            ao = eval_atomic_orbitals(diff, dist, atomic_orbitals)
            is_centered_1s = np.array(
                [(a.idx_atom == nuc_idx) and (sum(a.angular_momenta) == 0) for a in atomic_orbitals]
            )
            phi_0_1s = (is_centered_1s * ao) @ mo_coeff[:, mo_idx]
            phi_0_others = ((1 - is_centered_1s) * ao) @ mo_coeff[:, mo_idx]
            phi_rc_0, phi_rc_1, phi_rc_2 = 0.0, 0.0, 0.0
            r_c = cusp_rc[nuc_idx]
            for i, ao in enumerate(atomic_orbitals):
                if is_centered_1s[i]:
                    phi_rc_0 += np.sum(ao.weights * np.exp(-ao.alpha * r_c**2)) * mo_coeff[i, mo_idx]
                    phi_rc_1 += (
                        np.sum(ao.weights * (-2 * ao.alpha * r_c) * np.exp(-ao.alpha * r_c**2)) * mo_coeff[i, mo_idx]
                    )
                    phi_rc_2 += (
                        np.sum(ao.weights * (-2 * ao.alpha + 4 * (ao.alpha * r_c) ** 2) * np.exp(-ao.alpha * r_c**2))
                        * mo_coeff[i, mo_idx]
                    )
            cusp_1s_coeffs[nuc_idx, :, mo_idx] = (
                is_centered_1s * mo_coeff[:, mo_idx]
            )  # n_nuc x n_atomic_orbitals x n_molec_orbitals
            (
                cusp_offset[nuc_idx, mo_idx],
                cusp_sign[nuc_idx, mo_idx],
                cusp_poly[nuc_idx, mo_idx],
            ) = _calculate_mo_cusp_params(phi_rc_0, phi_rc_1, phi_rc_2, phi_0_1s, phi_0_others, Z[nuc_idx], r_c)
    return cusp_rc, cusp_offset, cusp_sign, cusp_poly, cusp_1s_coeffs
