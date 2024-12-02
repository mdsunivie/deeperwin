import pandas as pd
from deeperwin.run_tools.geometry_database import Geometry, load_geometries, append_energies
from deeperwin.utils.utils import PERIODIC_TABLE, ANGSTROM_IN_BOHR
import numpy as np

fname = "/home/mscherbela/runs/datasets/Gao2023/transfer.csv"
df_full = pd.read_csv(fname)
df_full.columns = ["method", "HChain10", "C2H4", "C4H4"]

all_geometries = load_geometries()


def geom_string_to_geometry(s):
    lines = s.split("\n")[1:]
    Z = []
    R = []
    for l in lines:
        tokens = l.replace("[", "").replace("]", "").split(" ")
        tokens = [t for t in tokens if t]
        Z.append(PERIODIC_TABLE.index(tokens[0]) + 1)
        R.append([float(x) for x in tokens[1:]])
    geom = Geometry(R, Z)
    return geom


def add_metadata_HChain(geom):
    d_short = geom.R[1][0] - geom.R[0][0]
    d_long = geom.R[2][0] - geom.R[1][0]
    comment = f"HChain{len(geom.R)}_{d_short:.2f}_{d_long:.2f}"
    geom.comment = comment
    geom.name = f"HChain{len(geom.R)}"
    return geom


def add_metadata_C2H4(geom):
    dCC = np.linalg.norm(geom.R[1] - geom.R[0]) / ANGSTROM_IN_BOHR
    twist = np.arctan2(geom.R[4, 2], geom.R[4, 1]) * 180 / np.pi
    geom.comment = f"C2H4_CC{dCC:.2f}A_{twist:.0f}deg"
    geom.name = "C2H4"
    return geom


def add_metadata_C4H4(geom):
    geom.comment = all_geometries[geom.hash].comment
    geom.name = all_geometries[geom.hash].name
    return geom


molecule = "C4H4"
df = df_full[["method", molecule]]
df.loc[:, "method"] = df.method.map(
    dict(
        direct="Gao_etal_2023_HF_pretraining",
        same="Gao_etal_2023_pretrained_on_similar",
        smaller="Gao_etal_2023_pretrained_on_smaller",
        none="Gao_etal_2023_no_pretraining",
    )
)

all_data = []
geometries = dict()
for _, r in df.iterrows():
    method = r.method
    if pd.isnull(r[molecule]):
        continue
    data = eval(r[molecule])
    for epoch, epoch_data in data.items():
        for geom, geom_data in epoch_data.items():
            geom = geom_string_to_geometry(geom)
            if molecule == "HChain10":
                geom = add_metadata_HChain(geom)
            elif molecule == "C2H4":
                geom = add_metadata_C2H4(geom)
            elif molecule == "C4H4":
                geom = add_metadata_C4H4(geom)
            geometries[geom.hash] = geom
            all_data.append(
                dict(
                    E=geom_data["energy"][0],
                    geom=geom.hash,
                    geom_comment=geom.comment,
                    molecule=geom.name,
                    energy_type="eval",
                    source="Gao_etal_2023",
                    epoch=epoch,
                    experiment=method,
                )
            )
append_energies(pd.DataFrame(all_data))


# geometries = list(geometries.values())
# dataset = GeometryDataset(geometries, name="HChain10_GaoEtAl2023_test_23geoms")

# all_geometries = load_geometries()
# for k,g in all_geometries.items():
#     if g.name == "C2H4" and "old" not in g.comment:
#         g.comment += "_old"
#         geometries.append(g)
# save_geometries(geometries, overwite_existing=False)
# save_datasets(dataset)
