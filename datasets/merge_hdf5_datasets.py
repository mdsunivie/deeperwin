import h5py

input_fnames = [
    "/home/mscherbela/runs/datasets/HEAT_distorted/HEAT_distorted.h5py",
    "/home/mscherbela/runs/datasets/QM7X/qm7x_Zmax10_nelmax36.hdf5",
]
output_fname = "/home/mscherbela/runs/datasets/small_molecules.hdf5"

f_out = h5py.File(output_fname, "w")
for fname_in in input_fnames:
    with h5py.File(fname_in, "r") as f_in:
        print(f"{fname_in:<70}: {len(f_in):6d} entries")
        for key in f_in.keys():
            f_in.copy(key, f_out, expand_external=True)
print(f"{output_fname:<70}: {len(f_out):6d} entries")
f_out.close()
