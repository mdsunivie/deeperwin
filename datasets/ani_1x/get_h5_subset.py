import h5py

# Open 2 h5 files, one for reading and one for writing
# Loop over the keys in the reading file and write the first n_entries to the writing file
fname_in = "ani1xcc_HF_STO6G.h5"
fname_out = "ani1xcc_HF_STO6G_subset.h5"
n_entries = 1000

with h5py.File(fname_in, "r") as f_in, h5py.File(fname_out, "w") as f_out:
    for key in f_in.keys():
        print(f"Copying {key}")
        f_in.copy(key, f_out)
        if len(f_out) >= n_entries:
            break
