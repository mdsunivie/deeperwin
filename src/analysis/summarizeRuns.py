from utils import load_from_file
import pathlib
import os
import numpy as np

REFERENCE_ENERGIES = {'He': -2.90372, 'Li': -7.47806, 'Be': -14.66736, 'B': -24.65391, 'C': -37.845, 'N': -54.5892, 'O': -75.0673, 'F': -99.7339, 'Ne': -128.9376, 'H2': -1.17448, 'LiH': -8.07055, 'N2': -109.5423, 'Li2': -14.9954, 'CO': -113.3255, 'Be2': -29.338, 'B2': -49.4141, 'C2': -75.9265, 'O2': -150.3274, 'F2': -199.5304, 'H4Rect': -2.0155, 'H3plus': -1.3438355180000001, 'H4plus': -1.8527330000000002, 'HChain6': -3.3761900000000002, 'HChain10': -5.6655}
REFERENCE_ENERGIES['H10'] = REFERENCE_ENERGIES['HChain10']

directory = '/users/mscherbela/runs/jaxtest/conv/test9'
fnames = [f for f in pathlib.Path(directory).rglob(os.path.join('results.bz2'))]
fnames = sorted(fnames)
for f in fnames:
    E_eval = load_from_file(f)['E_eval_mean']
    E_eval_mean = np.mean(E_eval)
    E_eval_std = np.std(E_eval) / np.sqrt(len(E_eval))
    name = "/".join(str(f).split('/')[-2:])
    molecule = f.parent.name.split('/')[-1].split('_')[0]
    E_ref = REFERENCE_ENERGIES[molecule]
    print(f"{name:<40}: {1e3*(E_eval_mean - E_ref):+4.1f} +- {1e3*E_eval_std:4.1f}")

