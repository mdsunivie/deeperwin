

import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pandas as pd

path = "/Users/leongerard/Desktop/data_schroedinger/O_r5_60000/ferminet_2022_03_24_16:03:26/"

files = [f for f in listdir(path) if isfile(join(path, f))]

width = []
for f in files:
    if f == 'train_stats.csv':
        df = pd.read_csv(path + f)

        #d = np.load(join(path, f))
        #width.append(d['mcmc_width'])


plt.plot(df['energy'])
plt.grid(alpha=0.5)
plt.xlabel("Number of iterations")
plt.ylabel("Std Dev")