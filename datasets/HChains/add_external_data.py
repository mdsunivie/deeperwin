from deeperwin.run_tools.geometry_database import load_geometries, append_energies
import pandas as pd
import ruamel.yaml

all_geometries = load_geometries()
geometries_by_comment = {g.comment: g for g in all_geometries.values()}
yaml = ruamel.yaml.YAML()

fname = "/home/mscherbela/runs/references/reuse_datasets/H10/finetuning.yml"
with open(fname, "r") as f:
    mrci_data = yaml.load(f)["changes"]


def get_geom_hash(mrci_comment):
    mrci_comment = mrci_comment.replace("H10_", "").split("_")
    d1, d2 = float(mrci_comment[0]), float(mrci_comment[1])
    db_comment = f"HChain10_{d1:.2f}_{d2:.2f}"
    if (d1 == d2) and (db_comment not in geometries_by_comment):
        db_comment = f"HChain10_{d1:.2f}"
    return geometries_by_comment[db_comment].hash


all_data = []
for d in mrci_data:
    all_data.append(dict(E=d["E_ref"], mrci_comment=d["comment"]))
df = pd.DataFrame(all_data)

# %%
df["geom"] = df.mrci_comment.apply(get_geom_hash)
df["geom_comment"] = df["geom"].apply(lambda h: all_geometries[h].comment)
df["molecule"] = "HChain10"
df["epoch"] = None
df["method"] = "indep"
df["source"] = "MRCI"
df["experiment"] = "MRCI_HChain10_23geoms"
#
df = df.drop(columns=["mrci_comment"])
append_energies(df)
