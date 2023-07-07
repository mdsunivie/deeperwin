from deeperwin.run_tools.geometry_database import load_geometries, load_datasets, Geometry
from deeperwin.utils.utils import ANGSTROM_IN_BOHR, PERIODIC_TABLE
import argparse
import os
import shutil
import subprocess
import time

def write_orca_input(geom: Geometry, fname, method, basis_set, frozen_core, n_proc, memory_per_core):
    with open(fname, "w") as f:
        f.write(f"!{method} {'FrozenCore' if frozen_core else 'NoFrozenCore'} {basis_set}\n")

        f.write("%SCF\n")
        f.write("  Convergence tight\n")
        f.write("  maxiter 300\n")
        f.write("  ConvForced 1\n")
        f.write("END\n")
        if 'CCSD' in method:
            f.write("%MDCI\n")
            f.write("  maxiter 300\n")
            f.write("  MaxDIIS 25\n") # Increasing the number of DIIS vectors be stored. Default is 7
            f.write("END\n")
        f.write(f"%MAXCORE {memory_per_core}\n")
        f.write("%PAL\n")
        f.write(f"  nprocs {n_proc}\n")
        f.write("END\n")
        f.write(f"* xyz {geom.charge} {2*geom.spin+1}\n")
        for r, z in zip(geom.R, geom.Z):
            f.write(f" {PERIODIC_TABLE[z-1]} {r[0] / ANGSTROM_IN_BOHR} {r[1] / ANGSTROM_IN_BOHR} {r[2] / ANGSTROM_IN_BOHR}\n")
        f.write("*\n")

def get_orca_results(output_fname):
    results = {}
    with open(output_fname, "r") as f:
        for line in f:
            if ("Total Energy       :" in line) and ("E_hf" not in results):
                results["E_hf"] = float(line.split()[3])
            if ("FINAL SINGLE POINT ENERGY" in line) and ("E_final" not in results):
                results["E_final"] = float(line.split()[4])
    return results

def run_orca(g: Geometry, directory, method, basis_set, frozen_core, n_proc, total_memory, orca_path="orca", clean_calc_dir=True):
    write_orca_input(g, os.path.join(directory, "orca.inp"), method, basis_set, frozen_core, n_proc, int(total_memory*1000/n_proc))
    with open(os.path.join(directory, "orca.out"), "w") as f:
        subprocess.call([orca_path, "orca.inp"], cwd=directory, stdout=f, stderr=f)
    results = get_orca_results(os.path.join(directory, "orca.out"))
    if clean_calc_dir:
        for fname in os.listdir(directory):
            if fname not in ["orca.out", "orca.inp"]:
                os.remove(os.path.join(directory, fname))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--method", type=str, required=False, default="CCSD(T)")
    parser.add_argument("--basis-set", type=str, nargs="+", required=False, default="aug-cc-pVDZ")
    parser.add_argument("--geometries", type=str, nargs="+", default=None, required=False)
    parser.add_argument("--n-proc", type=int, default=16, required=False)
    parser.add_argument("--total-memory", type=int, default=200, required=False, help="Total memory in GB")
    parser.add_argument("--orca-path", type=str, default="/gpfs/data/fs71573/scherbela/orca/orca", required=False, help="Full path to ORCA executable")
    parser.add_argument("--frozen-core", action="store_true", required=False, help="Use frozen core approximation")
    args = parser.parse_args()
    method = args.method
    method_string = method + "_frozen_core" if args.frozen_core else method

    # Load dataset
    all_geometries = load_geometries()
    all_datasets = load_datasets()

    ind_calc = 0
    with open("energies.csv", "w", buffering=1) as energy_file:
        energy_file.write("ind_calc;geom_hash;method;basis_set;E_hf;E_final;duration\n")
        for basis_set in args.basis_set:
            for geom_hash in args.geometries:
                if geom_hash in all_geometries:
                    geoms = [all_geometries[geom_hash]]
                else:
                    geoms = all_datasets[geom_hash].get_geometries(all_geometries, all_datasets)
                
                for g in geoms:
                    geom_hash = g.hash
                    directory = f"{ind_calc:04d}_{geom_hash}"
                    if os.path.isdir(directory):
                        shutil.rmtree(directory)
                    os.makedirs(directory)
                    t0 = time.time()
                    results = run_orca(g, directory, method, basis_set, args.frozen_core, args.n_proc, args.total_memory, args.orca_path)
                    t1 = time.time()
                    energy_file.write(f"{ind_calc};{geom_hash};{method_string};{basis_set};{results.get('E_hf', '')};{results.get('E_final', '')};{t1-t0}\n")
                    ind_calc += 1
