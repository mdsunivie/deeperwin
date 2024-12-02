

import pandas as pd
import numpy as np

def create_changes_string(Rs, E_Refs):
    ret = ''
    for R, E_ref in zip(Rs, E_Refs):
        ret += f'   - R: {R}\n     E_ref: {E_ref}\n     comment: H6_{R[1][0]}_{round(R[2][0] - R[1][0], 1)} \n'
    return ret

def create_changes_string_H4(Rs, E_Refs):
    ret = ''
    for R, E_ref in zip(Rs, E_Refs):
        ret += f'   - R: {R}\n     E_ref: {E_ref}\n     comment: _ \n'
    return ret

def create_changes_string_new_geom(rg):
    ret = ''
    for x in list(rg):
        R = [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [1.4+x, 0.0, 0.0], [2.8+x, 0.0, 0.0], [2.8+2*x, 0.0, 0.0], [5.6+2*x, 0.0, 0.0]]
        ret += f'   - R: {R}\n     E_ref: {0.0}\n     comment: H6_{R[1][0]}_{round(R[2][0] - R[1][0], 1)} \n'
    return ret

def create_H10_from_H6(parametrization, ground_truth):
    ret = ''
    for (a, x, g) in parametrization:
        R = [[0.0, 0.0, 0.0], [a, 0.0, 0.0], [a+x, 0.0, 0.0], [2*a + x, 0.0, 0.0], [2*a + 2*x, 0.0, 0.0], [3*a + 2*x, 0.0, 0.0],
             [3*a+3*x, 0.0, 0.0], [4*a + 3*x, 0.0, 0.0], [4*a + 4*x, 0.0, 0.0], [5*a + 4*x, 0.0, 0.0]]

        ret += f'   - R: {R}\n     E_ref: {ground_truth[float(g)]}\n     comment: H10_{R[1][0]}_{round(R[2][0] - R[1][0], 1)} \n'
    return ret

def get_H6_parametrization(Rs, geom):
    param = []
    for R, g in zip(Rs, geom):
        param.append((R[1][0], round(R[2][0] - R[1][0], 1), g))

    return param

def get_PM_Ref():
    d = "/Users/leongerard/Desktop/Schrodinger/results/P.M. results/energies/H10_additional_energies.out"
    df = pd.read_csv(d, delimiter="\t")
    print(df)

#if __name__ == '__main__':


#%%

d = "/Users/leongerard/Desktop/Schrodinger/results/P.M. results/energies/H10_additional_energies.out"
df = pd.read_csv(d, delimiter="\t")

ground_truth = {}
for i in range(len(df)):
    r = df.iloc[i][0].split()
    ground_truth[float(r[0])] = float(r[-2])

d = "/Users/leongerard/Desktop/Schrodinger/results/P.M. results/energies/H10_energies.out"
df = pd.read_csv(d, delimiter="\t")

for i in range(len(df)):
    r = df.iloc[i][0].split()
    ground_truth[float(r[0])] = float(r[-2])

#%%
d = "/Users/leongerard/ucloud/Shared/deeperwin_datasets/processed/"
df = pd.read_pickle(d + "reuse_vs_serial.pkl.gz")
print(pd.unique(df.method))
df = df[(df.name == 'HChain6') & (df.method == "independent_4096ep")]
# for id in df.keys():
#     print(id)
parametrization = get_H6_parametrization(df.physical_ion_positions, df.geom)
print(create_H10_from_H6(parametrization, ground_truth))