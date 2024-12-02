"""
Contains the physical baseline model (e.g. CASSCF).

This module provides functionality to calculate a basline solution using pySCF and functions to
"""

from typing import List, Tuple, Optional, Union, Dict, Iterable
import dataclasses
import functools
import pathlib
from collections import Counter
import logging
import jax
import jax.numpy as jnp
import pyscf
import pyscf.pbc.gto
import pyscf.pbc.scf
import pyscf.ci
import pyscf.lo
import pyscf.mcscf
import numpy as np
import chex
from deeperwin.configuration import (
    PhysicalConfig,
    CASSCFConfig,
    BaselineConfigType,
    HartreeFockConfig,
    PeriodicMeanFieldConfig,
    PyscfOptionsConfig,
    TransferableAtomicOrbitalsConfig,
)
from deeperwin.utils import lazy_pyscf as pyscf
from deeperwin.utils.utils import (
    get_el_ion_distance_matrix,
    generate_exp_distributed,
    PERIODIC_TABLE,
    load_periodic_pyscf,
    interleave_real_and_complex,
)
from deeperwin.utils.periodic import is_commensurable, project_into_first_unit_cell
from deeperwin.pbc_orbital_localization import localize_orbitals_pbc

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
        atom_type = PERIODIC_TABLE[int(self.Z) - 1] + str(int(self.idx_atom))
        orbital_type = ["s", "p", "d", "f"][self.l_tot]
        for power, axis in zip(self.angular_momenta, ["x", "y", "z"]):
            orbital_type += axis * power
        return atom_type + " " + orbital_type


@chex.dataclass
class OrbitalParamsHF:
    atomic_orbitals: List[AtomicOrbital] = dataclasses.field(default_factory=lambda: [])
    mo_coeff: Tuple[jnp.array, jnp.array] = (None, None)  # float; [n_basis x n_orbitals], [n_basis x n_orbitals]
    mo_occ: Tuple[jnp.array, jnp.array] = (None, None)  # int; [n_orbs_total], [n_orbs_total]
    mo_energies: Tuple[jnp.array, jnp.array] = (None, None)  # float; [n_orbitals], [n_orbitals]


@chex.dataclass
class OrbitalParamsPeriodicMeanField:
    atomic_orbitals: List[AtomicOrbital] = dataclasses.field(default_factory=lambda: [])
    mo_coeff_pyscf: Tuple[jnp.array, jnp.array] = (
        None,
        None,
    )  # float; [n_orbs_total x n_basis], [n_orbs_total x n_basis]
    mo_coeff: Tuple[jnp.array, jnp.array] = (None, None)  # float; [n_orbs_total x n_basis], [n_orbs_total x n_basis]
    mo_energies: Tuple[jnp.array, jnp.array] = (None, None)  # float; [n_orbs_total], [n_orbs_total]
    mo_occ: Tuple[jnp.array, jnp.array] = (None, None)  # int; [n_orbs_total], [n_orbs_total]
    k_points: Tuple[jnp.array, jnp.array] = (None, None)  # float; [3 x n_orbs_total], [3 x n_orbs_total]
    ind_band: Tuple[jnp.array, jnp.array] = (None, None)  # int; [n_orbs_total], [n_orbs_total]
    orbital_center: Tuple[jnp.array, jnp.array] = (None, None)  # int; [n_orbs_total], [n_orbs_total]
    k_twist: Optional[jnp.array] = None
    # TODO: use different lattices for different orbitals
    # rcut: jnp.ndarray = None
    lattice_vecs: jnp.ndarray = None
    ind_sorted: jnp.ndarray = None
    shift_vecs: jnp.ndarray = None

    def __str__(self) -> str:
        s = ""
        for i, spin in enumerate(["Up", "Down"]):
            s += f"Spin {spin}:\n"
            for ind_orb, (k, E, band, occ) in enumerate(
                zip(self.k_points[i].T, self.mo_energies[i], self.ind_band[i], self.mo_occ[i])
            ):
                s += f"{ind_orb:2d}: k = [{k[0]:+4.3f}, {k[1]:+4.3f}, {k[2]:+4.3f}] | band = {band:2d} | E = {E:+.3f} | occ = {int(occ)}\n"
            s += "-" * 70 + "\n"
        return s


@chex.dataclass
class OrbitalParamsCI:
    atomic_orbitals: List[AtomicOrbital] = dataclasses.field(default_factory=lambda: [])
    mo_coeff: Tuple[jnp.array, jnp.array] = (None, None)  # float; [n_basis x n_orbitals], [n_basis x n_orbitals]
    idx_orbitals: Tuple[jnp.array, jnp.array] = (None, None)  # int; [n_dets x n_up], [n_dets x n_dn]
    ci_weights: jnp.array = None


OrbitalParamsType = Union[OrbitalParamsHF, OrbitalParamsCI, OrbitalParamsPeriodicMeanField]

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


def eval_atomic_orbitals_kpoints(
    el_ion_diff, el_ion_dist, atomic_orbitals: List[AtomicOrbital], lattice_vecs, k_points=None
):
    """Evaluates atomic orbitals on a grid of kpoints, corresponding to what is
    returned by pyscf.pbc.eval_gto.eval_gto. For debugging purposes"""
    print("WARNING, DEPRECATED CODE")
    n_eval = el_ion_diff.shape[0]
    n_aos = len(atomic_orbitals)
    n_kpts = k_points.shape[0]

    if k_points is None:
        k_points = jnp.zeros((1, 3))

    aos = jnp.zeros((n_eval, n_kpts, n_aos), dtype=complex)

    for lattice_vec in lattice_vecs:
        diffs = el_ion_diff - lattice_vec
        dists = jnp.linalg.norm(diffs, axis=-1)
        ao = eval_atomic_orbitals(diffs, dists, atomic_orbitals)
        for ik, k_point in enumerate(k_points):
            aos = aos.at[:, ik].add(jnp.exp(1.0j * k_point @ lattice_vec) * ao)
    return aos


def eval_atomic_orbitals_periodic(
    el_ion_diff, el_ion_dist, atomic_orbitals: List[AtomicOrbital], lattice_vecs, shift_vecs=None, k_twist=None
):
    """shift_vecs: Shift vectors corresponding to the origin of each primitive
    cell in the simulation cell."""
    n_aos = len(atomic_orbitals)
    n_cells = shift_vecs.shape[0]
    *batch_dims, n_atoms, n_dim = el_ion_diff.shape

    if shift_vecs is None:
        shift_vecs = jnp.zeros((1, 3))
    if k_twist is None:
        k_twist = jnp.zeros((3,))

    aos = jnp.zeros((*batch_dims, n_cells, n_aos), dtype=complex)
    el_ion_diff_cell = el_ion_diff.reshape((*batch_dims, n_cells, n_atoms // n_cells, n_dim))

    diffs = el_ion_diff_cell[..., None, :, :, :] - lattice_vecs[:, None, None, :]
    dists = jnp.linalg.norm(diffs, axis=-1)
    phase = jnp.exp(1.0j * k_twist @ lattice_vecs.T)
    aos = eval_atomic_orbitals(diffs, dists, atomic_orbitals) * phase[:, None, None]
    aos = jnp.sum(aos, axis=-3)  # sum over lattice vectors
    aos = aos.reshape((*aos.shape[:-2], -1))  # merge cell and basis dim => [batch x el x basis_total]
    return aos


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


def evaluate_molecular_orbitals(
    el_ion_diff,
    el_ion_dist,
    atomic_orbitals,
    mo_coeff,
    mo_cusp_params=None,
    lattice_vecs=None,
    shift_vecs=None,
    k_twist=None,
):
    if lattice_vecs is None:
        aos = eval_atomic_orbitals(el_ion_diff, el_ion_dist, atomic_orbitals)
    else:
        aos = eval_atomic_orbitals_periodic(
            el_ion_diff,
            el_ion_dist,
            atomic_orbitals=atomic_orbitals,
            lattice_vecs=lattice_vecs,
            shift_vecs=shift_vecs,
            k_twist=k_twist,
        )

    # aos: [batch x el x basis], mo_coeff: [basis x n_orbitals]
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
    orbital_params: Union[OrbitalParamsHF, OrbitalParamsPeriodicMeanField],
    R: jax.Array,
    Z: jax.Array,
    lattice: jax.Array,
    atom_types: Iterable[int],
    n_basis_per_Z,
    tao_config: TransferableAtomicOrbitalsConfig,
):  # return Tuple(jnp.array, jnp.array) each [n_ions x n_orbitals x n_basis_total]
    n_ions = len(Z)
    mo_coeffs = [mo[:, occ > 0] for mo, occ in zip(orbital_params.mo_coeff, orbital_params.mo_occ)]
    mo_dtype = mo_coeffs[0].dtype
    center_of_mass = np.sum(R * Z[:, None], axis=-2) / np.sum(Z, axis=-1)

    if lattice is not None:

        def map_close_to_origin(x):
            # # This offset is a simple hack to make it less likely that orbital positions sit right on the boundary of the unit cell,
            # # leading to potential inconsistencies in downstream models when to positions suddendly changes by a lattice_vec
            # # It shifts all positions slightly in one direction (potentially outside the cell), projects them back and then shifts them back
            offset = 0.5  # bohr, not fractional coords
            x = x - offset
            x = project_into_first_unit_cell(x, lattice, around_origin=True)
            x = x + offset
            return x
    else:

        def map_close_to_origin(x):
            return x

    if isinstance(n_basis_per_Z[Z[0]], dict):
        n_basis_per_Z = {z: sum(n_per_l.values()) for z, n_per_l in n_basis_per_Z.items()}
    offsets = np.array([0] + [n_basis_per_Z[z] for z in atom_types], int)
    offsets = np.cumsum(offsets)
    n_basis_total = offsets[-1]
    slice_tgt = {z: slice(offsets[i], offsets[i + 1]) for i, z in enumerate(atom_types)}

    features = []
    for spin in range(2):
        n_orbitals = mo_coeffs[spin].shape[1]
        feature = np.zeros([n_ions, n_orbitals, n_basis_total], mo_dtype)
        ind_src_start = 0
        for i, z in enumerate(Z):
            n_basis = n_basis_per_Z[z]
            feature[i, :, slice_tgt[z]] = mo_coeffs[spin][ind_src_start : ind_src_start + n_basis, :].T
            ind_src_start += n_basis
        feature = interleave_real_and_complex(feature)

        if tao_config.use_atom_positions:
            rel_atom_pos = map_close_to_origin(R - center_of_mass)
            rel_atom_pos = np.tile(rel_atom_pos[:, None, :], (1, n_orbitals, 1))
            feature = np.concatenate([feature, rel_atom_pos], axis=-1)
        if tao_config.use_orbital_positions:
            rel_orb_pos = map_close_to_origin(orbital_params.orbital_center[spin].T - center_of_mass)
            rel_orb_pos = np.tile(rel_orb_pos[None, :, :], (n_ions, 1, 1))
            feature = np.concatenate([feature, rel_orb_pos], axis=-1)
        if tao_config.use_atom_orbital_diff:
            atom_pos = R[:, None, :]
            orb_pos = orbital_params.orbital_center[spin].T[None, :, :]
            atom_orb_diff = map_close_to_origin(atom_pos - orb_pos)
            feature = np.concatenate([feature, atom_orb_diff], axis=-1)
        features.append(feature)
    return features


def get_sum_of_atomic_exponentials(dist_el_ion, exponent=1.0, scale=1.0):
    phi_atoms = jnp.exp(-dist_el_ion * exponent)  # [batch x el x ion]
    return jnp.sum(phi_atoms, axis=-1) * scale  # [batch x el]


#################################################################################
####################### Compile-time helper functions ###########################
#################################################################################


def build_pyscf_molecule(R, Z, charge=0, spin=0, basis_set="6-311G", pseudo=None, lattice=None, pyscf_options=None):
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
    pyscf_options = dict(pyscf_options) if pyscf_options is not None else {}
    pyscf_options.pop("mf_options", None)
    if lattice is None:
        molecule = pyscf.gto.Mole(**pyscf_options)
    else:
        molecule = pyscf.pbc.gto.Cell(**pyscf_options)
        molecule.a = lattice
    molecule.atom = [[Z_, tuple(R_)] for R_, Z_ in zip(R, Z)]
    molecule.unit = "bohr"
    molecule.basis = get_basis_set(basis_set, set(Z))
    molecule.pseudo = pseudo
    molecule.cart = True
    molecule.spin = spin  # 2*n_up - n_down
    molecule.charge = charge
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
    for idx_atom, (element, _) in enumerate(molecule._atom):
        idx_basis_per_atom = 0
        for gto_data in molecule._basis[element]:
            l = gto_data[0]
            gto_data = np.array(gto_data[1:])
            alpha = gto_data[:, 0]
            weights = gto_data[:, 1:]
            for ind_contraction in range(weights.shape[1]):
                n_orientations = [1, 3, 6, 10][l]
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

    # nb_orbitals_per_Z = {key: sum(orbitals_per_Z[key].values()) for key in all_elements}
    irrep = []
    for orb_type, nb_orb in orbitals_per_Z[max(all_elements)].items():
        if orb_type == 0:
            irrep.append(f"{nb_orb}x0e")
        elif orb_type == 1:
            irrep.append(f"{nb_orb//3}x1o")
        elif orb_type == 2:
            raise NotImplementedError("Not yet implemented for d-type orbitals")
    irrep = "+".join(irrep)

    return orbitals_per_Z, irrep


def _get_all_basis_functions(Z_values: List[int], basis_set, is_periodic=False, pyscf_options=None):
    """Get all possible basis functions for a given set of chemical elements"""
    # Build a fictional "molecule" with all elements stacked on top of each other
    Z_values = sorted(list(set(Z_values)))
    R = np.zeros([len(Z_values), 3])
    charge = 0
    spin = sum(Z_values) % 2
    lattice = np.eye(3) if is_periodic else None
    molecule = build_pyscf_molecule(R, Z_values, charge, spin, basis_set, None, lattice, pyscf_options)
    return _get_atomic_orbital_basis_functions(molecule)


def build_pyscf_molecule_from_physical_config(
    physical_config: PhysicalConfig, basis_set, pseudo=None, use_primitive=False, pyscf_options=None
):
    if (physical_config.periodic is not None) and physical_config.periodic.is_expanded and use_primitive:
        R, Z = np.array(physical_config.periodic.R_prim), np.array(physical_config.periodic.Z_prim)
        charge, spin = physical_config.periodic.charge_prim, physical_config.periodic.spin_prim
    else:
        _, _, R, Z = physical_config.get_basic_params()
        charge, spin = physical_config.charge, physical_config.spin

    if physical_config.periodic is not None:
        if use_primitive:
            lattice = np.array(physical_config.periodic.lattice_prim)
        else:
            lattice = np.array(physical_config.periodic.lattice)
    else:
        lattice = None
    return build_pyscf_molecule(R, Z, charge, spin, basis_set, pseudo, lattice, pyscf_options=pyscf_options)


def get_hartree_fock_solution(
    physical_config: PhysicalConfig, basis_set, pyscf_options: Optional[PyscfOptionsConfig] = None
):
    molecule = build_pyscf_molecule_from_physical_config(physical_config, basis_set, pyscf_options=pyscf_options)
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
def get_n_basis_per_Z(basis_set: str, Z_values: tuple, is_periodic: bool = False, exp_to_discard: float = None):
    """Get the number of basis functions for a given set of chemical elements"""
    basis_dict = get_basis_set(basis_set, Z_values)
    pyscf_options = PyscfOptionsConfig(exp_to_discard=exp_to_discard)
    all_basis_functions = _get_all_basis_functions(Z_values, basis_dict, is_periodic, pyscf_options)
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
    if (method is None) or (method == "") or (method == "none"):
        mo_loc = mo_coeffs_occ
    elif method == "boys":
        localizer = pyscf.lo.Boys(mol, mo_coeffs_occ)
        mo_loc = localizer.kernel()
    elif method == "pm":
        localizer = pyscf.lo.PipekMezey(mol, mo_coeffs_occ)
        mo_loc = localizer.kernel()
    elif method == "cholesky":
        mo_loc = pyscf.lo.cholesky_mos(mo_coeffs_occ)
    elif method == "qr":
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


def get_baseline_solution_HF(physical_config: PhysicalConfig, hf_config: HartreeFockConfig):
    atomic_orbitals, hf = get_hartree_fock_solution(
        physical_config, hf_config.basis_set, pyscf_options=hf_config.pyscf_options
    )
    mo_coeff, mo_energy, mo_occ = split_results_into_spins(hf)

    if hf_config.localization:
        for spin, n_occ in enumerate([physical_config.n_up, physical_config.n_dn]):
            mo_coeff[spin][:, :n_occ], mo_energy[spin][:n_occ] = localize_orbitals(
                hf.mol, mo_coeff[spin][:, :n_occ], hf_config.localization, mo_energy[spin][:n_occ]
            )
            mo_coeff[spin][:, n_occ:], mo_energy[spin][n_occ:] = localize_orbitals(
                hf.mol, mo_coeff[spin][:, n_occ:], hf_config.localization, mo_energy[spin][n_occ:]
            )

    orbital_params = OrbitalParamsHF(
        atomic_orbitals=atomic_orbitals, mo_coeff=mo_coeff, mo_energies=mo_energy, mo_occ=mo_occ
    )
    energies = dict(E_hf=hf.e_tot)
    return orbital_params, energies


def get_baseline_solution_CASSCF(physical_config: PhysicalConfig, casscf_config: CASSCFConfig):
    atomic_orbitals, hf = get_hartree_fock_solution(
        physical_config, casscf_config.basis_set, pyscf_options=casscf_config.pyscf_options
    )
    n_cas_electrons = min(casscf_config.n_active_electrons, physical_config.n_electrons)
    n_cas_orbitals = casscf_config.n_active_orbitals

    casscf = pyscf.mcscf.UCASSCF(hf, n_cas_orbitals, n_cas_electrons)
    casscf.kernel()
    mo_coeff = list(casscf.mo_coeff)  # tuple of spin_up, spin_down
    ind_orbitals = _get_orbital_indices(casscf)
    ci_weights = casscf.ci.flatten()
    logger.debug(f"Total nuber of CASSCF-determinants before truncation: {len(ci_weights)}")

    n_dets = casscf_config.n_dets
    if n_dets < len(ci_weights):
        ind_largest = np.argsort(np.abs(ci_weights))[::-1][:n_dets]
        share_captured = np.sum(ci_weights[ind_largest] ** 2) / np.sum(ci_weights**2)
        logger.debug(f"Share of CI-weights captured by {n_dets} dets: {share_captured:.3%}")
        ci_weights = ci_weights[ind_largest]
        ci_weights = jnp.array(ci_weights / np.sum(ci_weights**2))
        ind_orbitals = ind_orbitals[0][ind_largest], ind_orbitals[1][ind_largest]
    orbital_params = OrbitalParamsCI(
        atomic_orbitals=atomic_orbitals, mo_coeff=mo_coeff, idx_orbitals=ind_orbitals, ci_weights=ci_weights
    )
    energies = dict(E_hf=hf.e_tot, E_casscf=casscf.e_tot)
    return orbital_params, energies


def make_k_points(k_points, cell):
    if k_points is None:
        k_points = np.array([1, 1, 1], int)
    if isinstance(k_points, list):
        k_points = np.array(k_points)
    if k_points.ndim == 1:
        if np.any(k_points.astype(int) != k_points):
            raise ValueError("if kpoints is 1d it should be should be ints specifying a kpoint grid")
        k_points = cell.make_kpts(k_points.astype(int), with_gamma_point=True)
        k_points = np.array(k_points)
    # One CANNOT map the k-points back to the first Brillouin zone at this point,
    # since different k-points would potentially be mapped by different rec vectors
    return k_points


def apply_mf_options(mf, mf_options):
    if mf_options is None:
        return mf
    if mf_options.density_fit:
        mf = mf.density_fit()
    if mf_options.df_mesh is not None:
        mf.with_df.mesh = mf_options.df_mesh
    return mf


def get_periodic_mean_field(
    physical_config: PhysicalConfig, hf_config: PeriodicMeanFieldConfig, return_pyscf_objects=False
):
    # options preprocessing
    pyscf_options = hf_config.pyscf_options
    if pyscf_options is None:
        mf_options = None
        chkfile = ""
    else:
        mf_options = pyscf_options.mf_options
        chkfile = pyscf_options.mf_options.chkfile

    mf = None
    if chkfile and pathlib.Path(chkfile).is_file():
        # TODO:  Check that the loaded cell and mf is correct
        cell, mf = load_periodic_pyscf(chkfile)
    else:
        cell: pyscf.pbc.gto.Cell = build_pyscf_molecule_from_physical_config(
            physical_config, hf_config.basis_set, use_primitive=True, pyscf_options=pyscf_options
        )
    simulation_cell: pyscf.pbc.gto.Cell = build_pyscf_molecule_from_physical_config(
        physical_config, hf_config.basis_set, use_primitive=False, pyscf_options=pyscf_options
    )

    assert cell.spin == 0, "Spin-polarized periodic mean-field not yet implemented"

    atomic_orbitals = _get_atomic_orbital_basis_functions(cell)

    k_points = hf_config.k_points
    if hf_config.k_points is None:
        k_points = physical_config.periodic.supercell

    # k_points: [n_k_points, 3]
    k_points = make_k_points(k_points, cell)
    k_twist = (
        np.array(physical_config.periodic.k_twist) if (physical_config.periodic.k_twist is not None) else np.zeros(3)
    )
    k_twist = k_twist @ simulation_cell.reciprocal_vectors()
    k_points += k_twist
    with np.printoptions(precision=3):
        logger.debug(f"{k_points=}")

    if mf is None:
        if hf_config.name == "periodic_hf":
            mf = pyscf.pbc.scf.KRHF(cell, k_points)
        elif hf_config.name == "periodic_dft":
            mf = pyscf.pbc.scf.KRKS(cell, k_points)
            mf.xc = "pbe,pbe"
        elif hf_config.name == "periodic_unrestricted_dft":
            mf = pyscf.pbc.scf.KUKS(cell, k_points)
            mf.xc = "pbe,pbe"

        mf = apply_mf_options(mf, mf_options)

        mf.chkfile = chkfile
        # mf.max_cycle = 20
        mf.kernel()

    # shift_vecs: [n_shifts, 3]
    shift_vecs = physical_config.periodic.get_shifts()

    # Build a long-list of all orbitals at all k-points, sorted by orbital energy
    mo_coeffs = np.array(mf.mo_coeff)  # [(spin) x n_k_points, n_basis_per_prim_cell, n_orb_per_kpoint]
    mo_occ = np.array(mf.mo_occ)
    mo_energy = np.array(mf.mo_energy)
    if mo_coeffs.ndim == 4:
        mo_coeffs = mo_coeffs[0]  # TODO: This is a hack to get the first spin, should be fixed
        mo_occ = mo_occ[0] * 2
        mo_energy = mo_energy[0]
    mo_coeffs = np.moveaxis(mo_coeffs, -1, -2)  # [n_k_points, n_orb_per_kpoint, n_basis_per_prim_cell]
    n_k_points, n_orb_per_kpoint, n_basis = mo_coeffs.shape
    n_shift = shift_vecs.shape[0]

    # mo_coeffs_supercell: [n_k_points, n_orb_per_kpoint, n_shift, n_basis]
    # exp() has shape: [n_k_points x n_shifts]
    mo_coeffs_supercell = mo_coeffs[:, :, None, :] * np.exp(1j * k_points @ shift_vecs.T)[:, None, :, None]
    mo_coeffs_supercell /= np.sqrt(n_shift)  # normalize to have same norm as mo_coeffs

    mo_coeffs_supercell = mo_coeffs_supercell.reshape([-1, n_basis * n_shift])
    mo_coeffs = mo_coeffs.reshape([-1, n_basis])

    mo_energies_all = np.stack(mo_energy).flatten()
    mo_occ_all = np.stack(mo_occ).flatten()
    ind_sorted = np.argsort(mo_energies_all)

    ind_k, ind_band = np.divmod(ind_sorted, n_orb_per_kpoint)
    mo_coeff_sorted = mo_coeffs[ind_sorted].T  # [n_basis x n_orbitals_total]
    mo_coeff_supercell_sorted = mo_coeffs_supercell[ind_sorted].T  # [n_basis_total x n_orbitals_total]
    mo_energies_sorted = mo_energies_all[ind_sorted]
    mo_occ_sorted = mo_occ_all[ind_sorted]
    k_points_sorted = k_points[ind_k].T
    energies = {
        f"E_{hf_config.name}_prim": mf.e_tot,
        f"E_{hf_config.name}": mf.e_tot * physical_config.periodic.n_prim_cells,
    }

    # Lattice vecs are from the simulation cell
    rcut = pyscf.pbc.gto.eval_gto._estimate_rcut(simulation_cell)

    # TODO: Since there is potentially a different nr of lattice vectors for every geometry, this requires recompilation
    # during shared optimization for every geometry. Would be better to pad this to some reasonable nearest integer or similar to avoid recompilation
    lattice_vecs = pyscf.pbc.gto.eval_gto.get_lattice_Ls(simulation_cell, rcut=rcut.max())

    if hf_config.localization in [None, "", "none"]:
        orbital_center = None
    elif hf_config.localization == "boys":
        mo_coeff_supercell_sorted_occ = mo_coeff_supercell_sorted[:, mo_occ_sorted > 0]
        mo_coeff_supercell_sorted_empty = mo_coeff_supercell_sorted[:, mo_occ_sorted == 0]
        mo_coeff_supercell_sorted_occ, loc_metrics, orbital_center = localize_orbitals_pbc(
            mo_coeff_supercell_sorted_occ, simulation_cell, k_twist
        )
        mo_coeff_supercell_sorted = np.concatenate(
            [mo_coeff_supercell_sorted_occ, mo_coeff_supercell_sorted_empty], axis=1
        )
        energies.update(loc_metrics)
    else:
        raise ValueError(f"Localization method not supported for PBC: {hf_config.localization}")

    orbital_params = OrbitalParamsPeriodicMeanField(
        atomic_orbitals=atomic_orbitals,
        mo_coeff_pyscf=(mo_coeff_sorted, mo_coeff_sorted),
        mo_coeff=(mo_coeff_supercell_sorted, mo_coeff_supercell_sorted),
        mo_energies=(mo_energies_sorted, mo_energies_sorted),
        mo_occ=(mo_occ_sorted // 2, mo_occ_sorted // 2),
        k_points=(k_points_sorted, k_points_sorted),
        orbital_center=(orbital_center, orbital_center),
        ind_band=(ind_band, ind_band),
        k_twist=k_twist,
        lattice_vecs=lattice_vecs,
        ind_sorted=ind_sorted,
        shift_vecs=shift_vecs,
    )
    if return_pyscf_objects:
        return orbital_params, energies, cell, mf
    return orbital_params, energies


def get_baseline_solution(physical_config: PhysicalConfig, pyscf_config: BaselineConfigType):
    if physical_config.periodic is None:
        if pyscf_config.name == "hf":
            return get_baseline_solution_HF(physical_config, pyscf_config)
        elif pyscf_config.name == "casscf":
            return get_baseline_solution_CASSCF(physical_config, pyscf_config)
        else:
            raise ValueError(f"Method not supported for non-periodic system: {pyscf_config.name}")
    else:
        if pyscf_config.name in ["periodic_hf", "periodic_dft", "periodic_unrestricted_dft"]:
            # if pyscf_config.k_points is None:
            #     pyscf_config.k_points = physical_config.periodic.supercell
            return get_periodic_mean_field(physical_config, pyscf_config)
        else:
            raise ValueError(f"Method not supported for periodic system: {pyscf_config.name}")


# TODO: implement for twisted, ie. non-commensurable k-points
def get_supercell_mo_coeffs(mo_coeffs_prim, k_points, lattice_prim, n_supercell):
    """
    Args:
        mo_coeffs_prim: [n_basis x n_orbitals]
        k_points: [3 x n_k_points]
        lattice_prim: np.array, [3 x 3]
        supercell: [int, int, int]
    """
    lattice_prim = np.array(lattice_prim)
    n_supercell = np.array(n_supercell, int)
    lattice_sc = n_supercell @ lattice_prim
    assert is_commensurable(lattice_sc, k_points - k_points[:, :1]), "k-points are not commensurable with supercell"

    ind_shift = np.stack(np.meshgrid(*[np.arange(n) for n in n_supercell]), axis=-1).reshape([-1, 3])
    shifts = ind_shift @ lattice_prim
    n_orbitals = mo_coeffs_prim.shape[-1]
    phase = np.exp(1.0j * (shifts @ k_points))
    mo_coeff_sc = (mo_coeffs_prim[None, :, :] * phase[:, None, :]).reshape([-1, n_orbitals])
    mo_coeff_sc /= np.sqrt(np.prod(n_supercell))  # Fix normalization

    k_points_sc = np.zeros_like(k_points)
    return mo_coeff_sc, k_points_sc


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
    orbitals, _ = get_baseline_solution(physical_config, HartreeFockConfig(basis_set=basis_set))
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
    if isinstance(x, int):
        return (x,) * 2
    else:
        return x


def split_results_into_spins(hf):
    if len(hf.mo_occ.shape) == 2:
        return (hf.mo_coeff[0], hf.mo_coeff[1]), (hf.mo_energy[0], hf.mo_energy[1]), (hf.mo_occ[0], hf.mo_occ[1])
    else:
        return (hf.mo_coeff, hf.mo_coeff), (hf.mo_energy, hf.mo_energy), (hf.mo_occ // 2, hf.mo_occ // 2)


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


# TODO Michael+Leon: delete all of this
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
    # TODO: Will this function be used in periodic calculations?
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
