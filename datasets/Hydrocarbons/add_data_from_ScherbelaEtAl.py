import pandas as pd
from deeperwin.run_tools.geometry_database import load_geometries, load_datasets, load_energies

all_datasets = load_datasets()
all_geometries = load_geometries()
all_energies = load_energies()

df = pd.read_csv("/home/mscherbela/runs/references/reuse_datasets/Fig3_weight_reuse.csv")
df.columns = ["subfigure", "molecule", "method", "epochs_per_geom"] + list(df)[4:]
df_C2H4_shared = df.query("molecule == 'Ethene' and method == 'Pre-trained by shared opt. (violet)'")
df_C2H4_reuse = df.query("molecule == 'Ethene' and method == 'Pre-trained by shared opt. of smaller molecule (red)'")


ds_C2H4 = all_datasets["C2H4_GaoEtAl2023_test_20geoms"]
geometries = [g for g in ds_C2H4.get_geometries(all_geometries, all_datasets)]
geometries = sorted(geometries, key=lambda g: g.comment)

df_mrci = all_energies.query("experiment == 'MRCI_Ethene_20geoms'").set_index("geom")
E_mrci = [df_mrci.loc[g.hash].E for g in geometries]

for df, method in zip([df_C2H4_shared, df_C2H4_reuse], ["shared", "reuse"]):
    df = df.loc[:, ~df.isnull().any()]
    df.columns = list(df)[:4] + list(range(df.shape[1] - 4))
    df = df.melt(
        id_vars=["subfigure", "molecule", "method", "epochs_per_geom"], var_name="geom_nr", value_name="error_mHa"
    )
    n_geoms = df.geom_nr.max() + 1
    df["E_MRCI"] = df.geom_nr.apply(lambda i: E_mrci[i])
    df["E"] = df.E_MRCI + 1e-3 * df.error_mHa
    df["geom"] = df.geom_nr.apply(lambda i: geometries[i].hash)
    df["geom_comment"] = df.geom_nr.apply(lambda i: geometries[i].comment)
    df["source"] = "Scherbela_etal_2022"
    df["experiment"] = f"Scherbela_etal_2022_{method}"
    df["molecule"] = df.molecule.map(dict(Ethene="C2H4", H10="HChain10"))
    df["method"] = method
    df["epoch"] = df["epochs_per_geom"] * n_geoms
    df["batch_size"] = 2048
    df = df.drop(columns=["subfigure", "epochs_per_geom", "geom_nr", "error_mHa", "E_MRCI"])
    # append_energies(df)
