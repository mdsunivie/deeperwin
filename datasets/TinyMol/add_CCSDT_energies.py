import pandas as pd
from deeperwin.run_tools.geometry_database import load_geometries, append_energies
from scipy.optimize import curve_fit
import numpy as np


def extrapolate_to_cbs(n_HF, E_HF, n, E):
    if np.any(np.isnan(E_HF)) or np.any(np.isnan(E)):
        return np.nan, np.nan
    assert np.all(n_HF[:-1] == n)

    def get_En_HF(x, E_cbs, a, b):
        return E_cbs + a * np.exp(-b * x)

    def get_Ecorr_n(x, E_corr_cbs, a):
        return E_corr_cbs + a * x ** (-3)

    p0 = (E_HF[-1] - 0.1, 0.1, 0.1)
    p_HF, _ = curve_fit(get_En_HF, n_HF, E_HF, p0)
    E_HF_CBS = p_HF[0]

    E_corr = E - E_HF[:-1]
    p0 = (E_corr[-1], 0.1)
    p_corr, _ = curve_fit(get_Ecorr_n, n, E_corr, p0)
    E_corr_CBS = p_corr[0]
    return E_HF_CBS, E_HF_CBS + E_corr_CBS


df = pd.read_csv(
    "/home/mscherbela/tmp/ref_energies/merged.out",
    header=0,
    delimiter=";",
    usecols=[1, 2, 3, 4],
    names=["geom", "geom_comment", "experiment", "E"],
)
df["experiment"] = df.experiment.str.replace("ccPCV", "ccpCV")

df = df[df.experiment.str.contains("ccpCV")]
df["basis_set"] = df.experiment.apply(lambda e: e.split("_")[-1])
df["method"] = df.experiment.apply(lambda e: e.split("_")[0])
df["basis_size"] = df.basis_set.map(dict(ccpCVDZ=2, ccpCVTZ=3, ccpCVQZ=4))

pivot = df.pivot("geom", ["method", "basis_size"], "E")
df_cbs = []
for g in pivot.index:
    cbs_energies = []
    methods = ["CCSD", "CCSD(T)"]
    for method in methods:
        df_hf = pivot.loc[g]["HF"]
        df_corr = pivot.loc[g][method]
        E_HF_CBS, E_CBS = extrapolate_to_cbs(df_hf.index.values, df_hf.values, df_corr.index.values, df_corr.values)
        cbs_energies.append(E_CBS)
    df_cbs.append(
        {"geom": g, ("HF", "CBS"): E_HF_CBS, **{(method, "CBS"): E for method, E in zip(methods, cbs_energies)}}
    )
df_cbs = pd.DataFrame(df_cbs).set_index("geom")
pivot = pivot.merge(df_cbs, "left", on="geom")

all_geometries = load_geometries()
df = pivot.reset_index().melt(id_vars=["geom"], value_name="E")
df["experiment"] = df.method + "_" + df.basis_size.map({2: "ccpCVDZ", 3: "ccpCVTZ", 4: "ccpCVQZ", "CBS": "CBS"})
df["method"] = "indep"
df["epoch"] = 0
df["method"] = "indep"
df["source"] = "pyscf_" + df["experiment"]
df["molecule"] = df.geom.apply(lambda g: all_geometries[g].name)
df["geom_comment"] = df.geom.apply(lambda g: all_geometries[g].comment)
df = df.drop(columns=["basis_size"])
all_energies = append_energies(df)
