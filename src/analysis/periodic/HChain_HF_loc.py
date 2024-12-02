#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/home/mscherbela/tmp/HChain_localization_HF.csv", 
                 sep=";", 
                 header=None)
df.columns = ["status", "N", "R", "loss", "E", "z0_real", "z0_imag", "z0_abs", "z1_real", "z1_imag", "z1_abs", "z2_real", "z2_imag", "z2_abs"]
df["divisible_by_4"] = df["N"] % 4 == 0

plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(12,8))

sns.lineplot(data=df[df.divisible_by_4], 
             x="R", 
             y="loss", 
             hue="N",
             palette="Oranges",
             ax=ax)
sns.lineplot(data=df[~df.divisible_by_4], 
             x="R", 
             y="loss", 
             hue="N",
             palette="Blues",
             ax=ax)