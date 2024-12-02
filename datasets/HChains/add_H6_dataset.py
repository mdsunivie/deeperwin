import ruamel.yaml
from deeperwin.run_tools.geometry_database import Geometry, GeometryDataset, save_datasets, save_geometries

with open("/home/mscherbela/runs/references/reuse_datasets/H10/finetuning.yml") as f:
    data = ruamel.yaml.YAML().load(f)["changes"]

geometries = []
for i, c in enumerate(data):
    n_atoms = len(c["R"])
    name = f"HChain{n_atoms}"
    d_short = c["R"][1][0] - c["R"][0][0]
    d_long = c["R"][2][0] - c["R"][1][0]
    g = Geometry(R=c["R"], Z=[1] * n_atoms, name=name, comment=f"{name}_{d_short:.2f}_{d_long:.2f}")
    geometries.append(g)
dataset = GeometryDataset(geometries, name=f"{name}_test_NatCompSci2022_{len(geometries)}geoms")
save_geometries(geometries)
save_datasets(dataset)
