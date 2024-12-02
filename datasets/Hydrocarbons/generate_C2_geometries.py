# %%
from deeperwin.run_tools.geometry_database import save_datasets, save_geometries, Geometry, GeometryDataset
import numpy as np

d = np.array([2.1, 2.2, 2.3, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.2])
geoms = [
    Geometry(
        R=np.array([[0, 0, 0], [d_, 0, 0]]),
        Z=np.array([6, 6]),
        name="C2",
        comment=f"C2_dimer_{d_:.1f}",
    )
    for d_ in d
]
dataset = GeometryDataset(geoms, name="C2_dimer")
save_geometries(geoms)
save_datasets(dataset)

# %%
