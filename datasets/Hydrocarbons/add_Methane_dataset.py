import ruamel.yaml
from deeperwin.run_tools.geometry_database import Geometry, GeometryDataset, save_datasets, save_geometries

with open("/home/mscherbela/runs/references/reuse_datasets/Ethene/pretrain_Methane.yml") as f:
    data = ruamel.yaml.YAML().load(f)["changes"]

geometries = []
for i, c in enumerate(data):
    g = Geometry(R=c["R"], Z=[6, 1, 1, 1, 1], name="CH4", comment=f"CH4_rot_dist_{i}")
    geometries.append(g)
dataset = GeometryDataset(geometries, name="CH4_rot_dist_train_NatCompSci2022_20geoms")
save_geometries(geometries)
save_datasets(dataset)
