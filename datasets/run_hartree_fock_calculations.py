from deeperwin.orbitals import build_pyscf_molecule, _get_atomic_orbital_basis_functions
from deeperwin.geometries import ANGSTROM_IN_BOHR
import pyscf
import pyscf.grad
import h5py
import numpy as np
import multiprocessing as mp
import argparse

N_THREADS_PER_CALCULATION = 2


def run_hartree_fock_calculation(R, Z, charge, spin, basis_set, compute_forces, *aux_data):
    molecule = build_pyscf_molecule(R, Z, charge, spin, basis_set)
    atomic_orbitals = _get_atomic_orbital_basis_functions(molecule)

    pyscf.lib.num_threads(N_THREADS_PER_CALCULATION)
    hf = pyscf.scf.HF(molecule)
    hf.verbose = 0  # suppress output to console
    hf.kernel()
    if not hf.converged:
        return None, aux_data

    mo_coeff = hf.mo_coeff
    mo_occ = hf.mo_occ
    mo_energy = hf.mo_energy
    is_rhf = isinstance(hf, pyscf.scf.hf.RHF)
    if is_rhf:
        mo_coeff = np.tile(mo_coeff, [2, 1, 1])
        mo_energy = np.tile(mo_energy, [2, 1])
        mo_occ = np.tile(mo_occ, [2, 1]) / 2

    result = dict(
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        e_tot=hf.e_tot,
        fock_matrix=hf.get_fock(),
        overlap_matrix=hf.get_ovlp(),
        core_hamiltonian=hf.get_hcore(),
        n_el=int(sum(molecule.nelec)),
        n_up=int(molecule.nelec[0]),
        n_dn=int(molecule.nelec[1]),
        atomic_orbitals=[dict(a) for a in atomic_orbitals],
    )

    if compute_forces:
        if is_rhf:
            forces = pyscf.grad.RHF(hf).kernel()
        else:
            forces = pyscf.grad.UHF(hf).kernel()
        result["forces"] = forces
    return result, aux_data


def save_to_hdf5(g: h5py.Group, hf_results: dict):
    for k, values in hf_results.items():
        if isinstance(values, list):
            for i, v in enumerate(values):
                subgroup = g.create_group(f"{k}/{i}")
                save_to_hdf5(subgroup, v)
        elif isinstance(values, dict):
            subgroup = g.create_group(k)
            save_to_hdf5(subgroup, values)
        else:
            if values is not None:
                g.create_dataset(k, data=values)


def process_result(result, n_total):
    HF_result, (ind, calc_name) = result
    if HF_result is None:
        print(f"Not converged: {calc_name}")
        return
    print(f"Saving {ind+1:6d}/{n_total}: {calc_name}")
    if calc_name in f:
        del f[calc_name]
    g = f.create_group(calc_name)
    save_to_hdf5(g, HF_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, help="Path to HDF5 file with geometry dataset")
    parser.add_argument(
        "--n-proc", type=int, default=1, help="Number of processes to spawn for calculations in parallel"
    )
    parser.add_argument("--n-calcs", type=int, default=0, help="Number of geometries to process")
    parser.add_argument("--compute-forces", type=int, default=1, help="Whether to compute forces for each geometry")
    parser.add_argument(
        "--force-recalc",
        action="store_true",
        default=False,
        help="Force the recalculation of a molecule even if it has already been calculated with this basis set",
    )
    parser.add_argument("--basis-sets", nargs="+")
    args = parser.parse_args()
    print(
        f"Running Hartree-Fock calculations using {args.n_proc} processes, with basis sets {args.basis_sets} on the dataset {args.fname}"
    )

    with h5py.File(args.fname, "a") as f:
        arguments = []
        for ind_geom, (geom_id, geom) in enumerate(f.items()):
            R = geom["atXYZ"][...] * ANGSTROM_IN_BOHR
            Z = geom["atNUM"][...]
            n_el = sum(Z)
            spin = n_el % 2
            for basis_set in args.basis_sets:
                calc_name = f"{geom_id}/HF/{basis_set}"
                if (calc_name in f) and not args.force_recalc:
                    continue
                arguments.append((R, Z, 0, spin, basis_set, args.compute_forces > 0, len(arguments), calc_name))
            if args.n_calcs and len(arguments) >= args.n_calcs:
                break
        n_calcs_total = len(arguments)
        print(f"Total nr of calculations: {n_calcs_total}")
        with mp.Pool(args.n_proc) as pool:
            results = [
                pool.apply_async(run_hartree_fock_calculation, a, callback=lambda r: process_result(r, n_calcs_total))
                for a in arguments
            ]
            pool.close()
            pool.join()
