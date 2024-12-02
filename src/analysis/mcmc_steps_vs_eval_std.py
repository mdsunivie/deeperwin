import glob
from utils import load_from_file
import numpy as np
import pandas as pd

fnames = glob.glob("/users/mscherbela/runs/jaxtest/conv/test9/*/results.bz2")
all_data = []
for f in fnames:
    data = load_from_file(f)
    d = data['config']
    E_eval_mean = data['E_eval_mean']
    d['Eval_mean_of_std'] = np.mean(data['E_eval_std'])
    d['Eval_std_of_mean'] = np.mean(data['E_eval_mean'])
    all_data.append(d)
df = pd.DataFrame(all_data)




