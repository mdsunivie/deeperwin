import h5py
import numpy as np

np.random.seed(0)

input_fnames = [f"/storage/schroedinger_datasets/QM7/{i}000.hdf5" for i in range(1, 8 + 1)]
output_fname = "/storage/schroedinger_datasets/QM7/qm7x_Zmax10_nelmax50.hdf5"
DUPLICATES = open("/storage/schroedinger_datasets/QM7/DupMols.dat").read().split("\n")
DUPLICATES = set([d.replace("-opt", "") for d in DUPLICATES])

Z_MAX = 10
N_EL_MAX = 50
N_DISTORTIONS_PER_CONFORMATION = None

n_geoms = 0
all_Z = []
with h5py.File(output_fname, "w") as f_out:
    for fname in input_fnames:
        f = h5py.File(fname, "r")
        for mol_id, mol in f.items():
            conf_ids = [k for k in mol.keys() if k.rsplit("-", 1)[0] not in DUPLICATES]
            if len(conf_ids) == 0:
                continue
            Z = mol[conf_ids[0]]["atNUM"][...]
            if (np.max(Z) > Z_MAX) or np.sum(Z) > N_EL_MAX:
                continue
            assert len(conf_ids) % 101 == 0
            geom = conf_ids[0].rsplit("-", 3)[0]
            i_values = set([c.split("-")[2][1:] for c in conf_ids])
            selected_confs = []
            for i in i_values:
                c_values = set([c.split("-")[3][1:] for c in conf_ids if f"i{i}" in c])
                for c in c_values:
                    selected_confs.append(f"{geom}-i{i}-c{c}-opt")
                    if N_DISTORTIONS_PER_CONFORMATION is None:
                        ind_distortion = np.arange(100)
                    else:
                        ind_distortion = np.random.choice(100, N_DISTORTIONS_PER_CONFORMATION, replace=False)
                    for ind_d in ind_distortion:
                        selected_confs.append(f"{geom}-i{i}-c{c}-d{ind_d+1}")
            n_geoms += len(selected_confs)
            print(fname, n_geoms)
            for conf in selected_confs:
                mol.copy(conf, f_out, name=f"QnnM7X-{conf}")
        f.close()
