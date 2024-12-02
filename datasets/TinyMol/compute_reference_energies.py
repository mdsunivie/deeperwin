import pyscf.cc
import pyscf.scf
from deeperwin.run_tools.geometry_database import load_geometries, load_datasets
import pyscf
import argparse


def log_energy(geom, method, E):
    print(f"energy_result;{geom.hash};{geom.comment};{method};{E:.8f}", flush=True)


def compute_energies(geom, basis_set, max_theory_level="ccsd(t)"):
    max_theory_level = max_theory_level.lower()
    assert max_theory_level in ["hf", "ccsd", "ccsd(t)"]

    mol = geom.as_pyscf_molecule(basis_set)
    hf = pyscf.scf.HF(mol).run()
    E_HF = hf.e_tot
    log_energy(geom, f"HF_{basis_set}", E_HF)
    if max_theory_level == "hf":
        return

    cc = pyscf.cc.CCSD(hf).run()
    E_CCSD = cc.e_tot
    log_energy(geom, f"CCSD_{basis_set}", E_CCSD)
    if max_theory_level == "ccsd":
        return

    pert_corr = cc.ccsd_t()
    E_CCSDT = E_CCSD + pert_corr
    log_energy(geom, f"CCSD(T)_{basis_set}", E_CCSDT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", help="Geometry hash or dataset to compute use")
    parser.add_argument("--basis-set", nargs="+", help="Basis sets to use")
    parser.add_argument(
        "--theory-level", choices=["HF", "CCSD", "CCSD(T)"], default="CCSD(T)", help="Maximum level of theory to use"
    )
    # args = parser.parse_args("--dataset TinyMol_CNO_rot_dist_C3H4_smiles_C#CC_test_10geoms TinyMol_CNO_rot_dist_C3H4_smiles_C=C=C_10geoms --basis-set ccpCVDZ ccPCVTZ".split())
    args = parser.parse_args()

    all_geometries = load_geometries()
    all_datasets = load_datasets()
    geometries_to_compute = []
    for dataset in args.dataset:
        if dataset in all_geometries:
            geometries_to_compute.apoend(all_geometries[dataset])
        else:
            geometries_to_compute += all_datasets[dataset].get_geometries(all_geometries, all_datasets)

    print(
        f"Computing {args.theory_level} energies for {len(geometries_to_compute)} geometries, using {len(args.basis_set)} basis-sets"
    )
    for basis_set in args.basis_set:
        for geom in geometries_to_compute:
            compute_energies(geom, basis_set, args.theory_level)
