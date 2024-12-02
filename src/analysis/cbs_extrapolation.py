# %%
import pandas as pd
from deeperwin.run_tools.geometry_database import load_geometries, append_energies
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


all_geoms = load_geometries()
def extrapolate_to_cbs(n_HF, E_HF, n, E):
    if np.any(np.isnan(E_HF)) or np.any(np.isnan(E)):
        return np.nan, np.nan
    HF_indices_for_correlation = np.concatenate([np.where(n_ == n_HF)[0] for n_ in n])

    def get_En_HF(x, E_cbs, a, b):
        return E_cbs + a * np.exp(-b * x)

    def get_Ecorr_n(x, E_corr_cbs, a):
        return E_corr_cbs + a * x ** (-3)

    p0 = (E_HF[-1] - 0.1, 0.1, 0.1)
    p_HF, _ = curve_fit(get_En_HF, n_HF, E_HF, p0)
    E_HF_CBS = p_HF[0]

    E_corr = E - E_HF[HF_indices_for_correlation]
    p0 = (E_corr[-1], 0.1)
    p_corr, _ = curve_fit(get_Ecorr_n, n, E_corr, p0)
    E_corr_CBS = p_corr[0]
    return E_HF_CBS, E_HF_CBS + E_corr_CBS


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
fname = "/home/mscherbela/tmp/energies_h2o.csv"

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

HF_data = []
for i,r in df.iterrows():
    data = dict(r)
    data["E_final"] = data["E_hf"]
    data["method"] = "RHF"
    HF_data.append(data)
df = pd.concat([df, pd.DataFrame(HF_data)], ignore_index=True)
df["basis_size"] = df.basis_set.apply(get_basis_size)
df = df[df.basis_size.notnull()]
df["is_aug"] = df.basis_set.str.contains("aug")

df = df.groupby(["geom_hash", "basis_set", "basis_size", "is_aug", "method"]).mean().reset_index()
cbs_data = []
for geom in df.geom_hash.unique():
    for aug in [True, False]:
        df_filt = df[(df.geom_hash == geom) & (df.is_aug == aug)]
        is_HF = (df_filt.method == "RHF") & (df_filt.basis_size <= 5)
        is_CC = df_filt.method == "CCSD(T)"
        has_ccsdt = ~df_filt.E_final.isnull()

        n_all = df_filt[is_CC].basis_size.values
        E_all = df_filt[is_CC].E_final.values
        n_HF = df_filt[is_HF].basis_size.values
        E_HF = df_filt[is_HF].E_final.values

        if (len(n_HF) < 3) or (len(n_all) < 2):
            continue

        for n_min in [2, 3]:
            for n_max in range(n_min+1, 5+1):
                selection = (n_all >= n_min) & (n_all <= n_max)
                if selection.sum() < (n_max - n_min + 1):
                    continue
                E_HF_CBS, E_CBS = extrapolate_to_cbs(n_HF, E_HF, n_all[selection], E_all[selection])
                basis_set = "CBS_" + "".join(sorted([str(int(n_)) for n_ in n_all[selection]]))
                cbs_data.append(
                    dict(geom_hash=geom, basis_size=None, basis_set=basis_set, is_aug=aug, E_final=E_CBS, method="CCSD(T)")
                )
                if (n_min == 2) and (n_max == 3):
                    cbs_data.append(
                        dict(geom_hash=geom, basis_size=None, basis_set="CBS_234", is_aug=aug, E_final=E_HF_CBS, method="HF")
                    )
df = pd.concat([df, pd.DataFrame(cbs_data)], ignore_index=True)

# %%
is_aug = False
df["n_heavy_atoms"] = df.geom_hash.apply(lambda x: all_geoms[x].n_heavy_atoms)
df = df.sort_values(["n_heavy_atoms", "basis_size"])
geoms_with_cbs = df[df.basis_size.isnull() & (df.is_aug == is_aug)].geom_hash.unique()

plt.close("all")
fig, axes = plt.subplots(6, 7, figsize=(17, 9), sharex=False, sharey=True)
for ind_ax, (ax, geom) in enumerate(zip(axes.flatten(), geoms_with_cbs)):
    df_filt = df[(df.geom_hash == geom) & (df.is_aug == is_aug) & (df.method == "CCSD(T)")]
    df_filt_finite_basis = df_filt[~df_filt.basis_size.isnull() & df_filt.E_final.notnull()]
    E_ref = df_filt_finite_basis.query("basis_size == 3").E_final.iloc[0]
    ax.plot(df_filt_finite_basis.basis_size, (df_filt_finite_basis.E_final - E_ref) * 1000, "o-", label="CCSD(T)")
    for ind_cbs, basis_set in enumerate(["CBS_23", "CBS_234", "CBS_34"]):
            df_subfilt = df_filt[df_filt.basis_set == basis_set]
            if len(df_subfilt) == 0:
                continue
            if basis_set == "CBS_23":
                n_last = 3
                E_last = df_filt_finite_basis.query("basis_size == 3").E_final.iloc[0]
            else:
                n_last = 4
                E_last = df_filt_finite_basis.query("basis_size == 4").E_final.iloc[0]
            ax.plot([n_last, 5], [(E_last - E_ref)*1000, (df_subfilt.E_final.iloc[0] - E_ref) * 1000], label=basis_set, marker="o", color=f"C{ind_cbs+1}")
    g = all_geoms[geom]
    ax.set_title(f"{g.n_heavy_atoms}: {g.name}")
    ax.set_xticks([2, 3, 4, 5])
    ax.set_yticks([-100, 0, 100, 200, 300])
    ax.set_xticklabels(["2Z", "3Z", "4Z", "CBS"])
    if ind_ax == 0:
        ax.legend()
fig.tight_layout()
# %%

df_out = df[(df.basis_set.isin(["CBS_23", "CBS_234", "CBS_34", "cc-pCVDZ", "cc-pCVTZ", "cc-pCVQZ"])) & (df.is_aug == False)]
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
