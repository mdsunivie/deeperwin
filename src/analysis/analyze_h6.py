from deeperwin.checkpoints import load_from_file
import pathlib
import os
import numpy as np
import pickle
import yaml

REFERENCE_DATA_DIR = '/Users/leongerard/Desktop/parallel_training.pkl'

with open(REFERENCE_DATA_DIR, 'rb') as f:
    reference = pickle.load(f)

reference = reference[reference.name == 'HChain6']
reference = reference[['E_ref', 'physical_ion_positions']]
ion_positions = list(reference.physical_ion_positions)
e_ref = list(reference.E_ref)

ref_dict = {}
for i, pos in enumerate(ion_positions):
    key = str(round(pos[1][0], 1)) +"_"+ str(round(pos[2][0] - pos[1][0], 1))
    ref_dict[key] = e_ref[i]
    if "1.2_2.2" == key:
        print(i)
#%%
directory = '/Users/leongerard/Desktop/JAX_DeepErwin/data/dgx_runs/Eval_BFGS/EvalH6BFGS/'
fnames = [f for f in pathlib.Path(directory).rglob(os.path.join('results.bz2'))]
fnames = sorted(fnames)

folder = ['chkpt000256', 'chkpt000512', 'chkpt001024', 'chkpt002048']

results = {'chkpt000256': [],
           'chkpt000512': [],
           'chkpt001024': [],
           'chkpt002048': [],
           'chkpt004096': []
}
#%%

geom = []
for i, f in enumerate(fnames):
    f_str_split = str(f).split("/")
    f_str_folder = f_str_split[-2]

    if f_str_folder in folder:
        df = load_from_file(f)
        E_eval = df['metrics']['E_mean']
        f_str_config = "/".join(f_str_split[:-1])

        with open(f_str_config + "/full_config.yml", 'r') as stream:
            config = yaml.safe_load(stream)
        pos = config['physical']['R']
        key = str(round(pos[1][0], 1)) + "_" + str(round(pos[2][0] - pos[1][0], 1))

        error = (E_eval - ref_dict[key]) * 1000
        results[f_str_folder].append(error)
    else:
        df = load_from_file(f)
        E_eval = df['metrics']['E_mean']
        f_str_config = "/".join(f_str_split[:-1])

        with open(f_str_config + "/full_config.yml", 'r') as stream:
            config = yaml.safe_load(stream)
        pos = config['physical']['R']
        key = str(round(pos[1][0], 1)) + "_" + str(round(pos[2][0] - pos[1][0], 1))
        if "1.2_2.2" == key:
            print(f)
        geom.append(key)
        error = (E_eval - ref_dict[key])*1000
        results['chkpt004096'].append(error)



print(np.mean(results['chkpt004096']), np.std(results['chkpt004096']))
print(np.mean(results['chkpt002048']), np.std(results['chkpt002048']))
print(np.mean(results['chkpt001024']), np.std(results['chkpt001024']))
print(np.mean(results['chkpt000512']), np.std(results['chkpt000512']))
#%%

geom = set(geom)
ref_geom = set(ref_dict.keys())

print(ref_geom.difference(geom))
#%%
import os
d = "/Users/leongerard/Desktop/JAX_DeepErwin/data/dgx_runs/Eval_BFGS/EvalH6BFGS/"

fnames = [f for f in pathlib.Path(d).rglob(os.path.join('full_config.yml'))]

cancelled_jobs = []

for f in fnames:
    f_split = str(f).split("/")[:-1]
    f_split.append("results.bz2")
    f_path = "/".join(f_split)
    if os.path.exists(f_path):
        pass
    else:
        f_split = str(f).split("/")[-4:-1]

        cancelled_jobs.append("/".join(f_split))
        print("/".join(f_split))

print(len(fnames))
print(cancelled_jobs)
#%%


# h6_dir = '/Users/leongerard/Desktop/H6_Parallel_49geoms.pkl'
#
# with open(h6_dir, 'rb') as f:
#     reference = pickle.load(f)

directory = '/Users/leongerard/Desktop/JAX_DeepErwin/data/dgx_runs/h6_1k_adam_no_shift/'
fnames = [f for f in pathlib.Path(directory).rglob(os.path.join('results.bz2'))]
fnames = sorted(fnames)



df = load_from_file(fnames[0])
df = load_from_file(fnames[0])