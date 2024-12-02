# %%
from deeperwin.run_tools.geometry_database import load_geometries, GeometryDataset, save_datasets, save_geometries
from scipy.spatial.transform import Rotation
import copy
import numpy as np
import ase.visualize


all_geoms = load_geometries()

# Cyclobutadiene
g0 = all_geoms["510c19f6e48f0c0d80b9824d8182a98e"]
g1 = all_geoms["d3f19e2a53c48d0787e7557d9753127b"]

n_geoms = 6
geometries = []
for x in np.linspace(0, 1.0, n_geoms):
    g = copy.deepcopy(g0)
    g.R = x * g1.R + (1 - x) * g0.R
    g.comment = f"Cylobutadiene_transition_{x:.1f}"
    geometries.append(g)
ds = GeometryDataset(geometries, name=f"Cylobutadiene_transition_{n_geoms}geoms")
geom_dict = {g.hash: g for g in geometries}
save_geometries(geometries)
save_datasets(ds)
ase_geoms = ds.as_ase(all_geometries=geom_dict)
ase.visualize.view(ase_geoms)


# %% C3H4
n_geoms = 20
# U = Rotation.random(num=n_geoms, random_state=1234).as_matrix()
# U[0] = np.eye(3)
np.random.seed(1234)
u = np.random.normal(size=[3])
u /= np.linalg.norm(u)
phi = np.arange(n_geoms) * 2 * np.pi / n_geoms
U = Rotation.from_rotvec(phi[:, None] * u).as_matrix()

geom0 = all_geoms["b2c66515e32b29b4d09cfa60705cd14c"]
geometries = []
for angle, u in zip(phi, U):
    g = copy.deepcopy(geom0)
    g.comment = f"C3H4_dist_rotated_{angle*180/np.pi:.0f}deg"
    g.R = g.R @ u
    geometries.append(g)
ds = GeometryDataset(geometries, name=f"C3H4_dist_rotated_{n_geoms}geoms")
save_geometries(geometries)
save_datasets(ds)
# ase_geoms = ds.as_ase({g.hash: g for g in geometries})
# ase.visualize.view(ase_geoms)

# %%
# N2 bond breaking
# R = [1.601512, 2.068000, 2.135349, 2.669186, 3.203023, 3.736861, 4.000000, 4.270698, 4.804535, 5.338372]
# geom0 = Geometry(R = [[0,0,0], [1.0,0,0]], Z=[7,7], name="N2")
# geometries = []
# for r in R:
#     g = copy.deepcopy(geom0)
#     g.R[1][0] = r
#     g.comment = f"N2_d={r:.6f}"
#     geometries.append(g)
# ds = GeometryDataset(geometries, name="N2_stretching_GerardEtAl2022")
# save_geometries(geometries)
# save_datasets(ds)


# %%
