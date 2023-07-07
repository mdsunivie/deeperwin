#!/usr/bin/env python
import h5py
import numpy as np
import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-set", type=str, default="STO-6G")
    parser.add_argument("--output", type=str, default="", help="Path of output file; defaults to same names as input, but ending in .pkl")
    parser.add_argument("fname", type=str, help="Path to HDF5 file with geometry dataset")
    args = parser.parse_args()

    filename = args.fname
    assert filename.endswith(".hdf5"), "Input filename must be a .hdf5 file"
    output_name = args.output or args.fname.replace(".hdf5", ".pkl")
    basis_set = args.basis_set

    chemical_symbols = ['n', 'H', 'He','Li', 'Be', 'B', 'C', 'N', 'O']

    data = []
    with h5py.File(filename, "r") as f:
        print(len(f.keys()))
        for i, mol_key in enumerate(f.keys()):
                mol = f[mol_key]
                if i % 1000 == 0:
                    print(f"Processing {i:5d}...")

                if 'HF' not in mol.keys() or basis_set not in mol['HF'].keys():
                    print(f"No data for: {mol_key}")
                    continue

                mol_data = mol['HF'][basis_set]
                E = np.array(mol_data['e_tot'])
                mo_coeff = np.array(mol_data['mo_coeff'])

                R = np.array(mol['atXYZ']) * 1.8897261258369282  # convert angstrom to bohr
                Z = np.array(mol['atNUM'])

                H = np.array(mol_data['fock_matrix'])
                S = np.array(mol_data['overlap_matrix'])

                if H.ndim == 3:
                    print(mol_key)
                    continue

                atom_types = ''.join([chemical_symbols[i] for i in Z])
                mol = dict(R=R, Z=Z, E=E, H=H, S=S, mo_coeff=mo_coeff)
                if 'core_hamiltonian' in mol_data:
                    mol['H_core'] = np.array(mol_data['core_hamiltonian'])
                if 'forces' in mol_data:
                    mol['forces'] = np.array(mol_data['forces'])
                data.append(mol)
    print(f"Saving {len(data)} calculations")
    with open(output_name, "wb") as f:
        pickle.dump(data, f)