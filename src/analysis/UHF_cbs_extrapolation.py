# %%
import pandas as pd
from deeperwin.run_tools.geometry_database import load_geometries, append_energies
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


all_geoms = load_geometries()
def extrapolate_to_cbs(n_HF, E_HF):
    def get_En_HF(x, E_cbs, a, b):
        return E_cbs + a * np.exp(-b * x)

    p0 = (np.min(E_HF) - 0.1, 0.1, 0.1)
    p_HF, _ = curve_fit(get_En_HF, n_HF, E_HF, p0)
    E_HF_CBS = p_HF[0]
    return E_HF_CBS


def get_basis_size(basis_set):
    if "CVDZ" in basis_set:
        return 2
    elif "CVTZ" in basis_set:
        return 3
    elif "CVQZ" in basis_set:
        return 4
    elif "CV5Z" in basis_set:
        return 5
    return None


# fname = "/home/mscherbela/runs/orca/test_sets_zero_shot/merged.csv"
# fname = "/home/mscherbela/tmp/merged.csv"
# fname = "/home/mscherbela/tmp/6and7_4Z.csv"
fname = "/home/mscherbela/tmp/RHF.csv"

# Remove repeating headers
lines = []
with open(fname) as f:
    for i, line in enumerate(f):
        if (i == 0) or "E_final" not in line:
            lines.append(line)
with open(fname, "w") as f:
    f.writelines(lines)


df = pd.read_csv(fname, sep=";")
df = df[df.basis_set.str.contains("cc-pCV")]
df = df[df.E_final.notnull()]
df = df[df.method == "RHF"]
df["basis_size"] = df.basis_set.apply(get_basis_size)

df = df.groupby(["geom_hash", "basis_set", "basis_size", "method"]).mean().reset_index()
cbs_data = []
for geom in df.geom_hash.unique():
    df_filt = df[df.geom_hash == geom]

    n_HF = df_filt.basis_size.values
    E_HF = df_filt.E_final.values
    E_CBS = extrapolate_to_cbs(n_HF, E_HF)
    cbs_data.append(
                    dict(geom_hash=geom, basis_size=None, basis_set="CBS_2345", E_final=E_CBS, method="RHF")
                )
df = pd.concat([df, pd.DataFrame(cbs_data)], ignore_index=True)

df_out = df[df.basis_set.isin(["CBS_2345", "cc-pCVQZ", "cc-pCV5Z"])]
df_out = df_out[["geom_hash", "basis_set", "E_final", "method"]]
df_out = df_out.rename(columns={"E_final": "E", "geom_hash": "geom"})
df_out["experiment"] = df_out["method"] + "_" + df_out.basis_set.str.replace("-", "")
df_out["source"] = "orca_" + df_out["experiment"]
df_out["method"] = "indep"
df_out['molecule'] = df_out.geom.apply(lambda g: all_geoms[g].name)
df_out['geom_comment'] = df_out.geom.apply(lambda g: all_geoms[g].comment)
df_out.drop(columns=["basis_set"], inplace=True)
append_energies(df_out)

# Unpivot the columns E_hf and E_final
# df_cbs = df_cbs.melt(id_vars=["geom_hash", "basis_set"], value_vars=["E_hf", "E_final"], var_name="E_type", value_name="E")


# df["experiment"] = df.method + "_" + df.basis_size.map({2:"ccpCVDZ", 3:"ccpCVTZ", 4:"ccpCVQZ", "CBS": "CBS"})
# df["method"] = "indep"
# df['epoch'] = 0
# df['method'] = "indep"
# df['source'] = "pyscf_" + df["experiment"]
# df['molecule'] = df.geom.apply(lambda g: all_geometries[g].name)
# df['geom_comment'] = df.geom.apply(lambda g: all_geometries[g].comment)


# %%
