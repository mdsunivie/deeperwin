# %%
import numpy as np
import matplotlib.pyplot as plt
from deeperwin.run_tools.geometry_database import load_datasets, Geometry, PeriodicLattice


def graphene_geometry():
    lattice = np.array([[4.02795, -2.3255, 0.0], [4.02795, 2.3255, 0.0], [0.0, 0.0, 99.99995]])
    R = [[2.6853, 0.0, 0.0], [5.3706, 0.0, 0.0]]
    Z = [6, 6]
    return R, Z, lattice


R, Z, lat = graphene_geometry()
rec_vecs = np.linalg.inv(2 * lat).T * 2 * np.pi

ds = load_datasets()["deepsolid_benchmarks_C_graphene_2x2x1_19twists"]
geoms_trained = ds.get_geometries()
k_frac_trained = np.array([g.periodic.k_twist for g in geoms_trained])
k_trained = k_frac_trained @ rec_vecs


plt.close("all")

fig, ax = plt.subplots(1, 1)
ax.scatter(k_trained[:, 0], k_trained[:, 1], s=200, color="gray", label="Original calcs")
# for i in range(2):
#     # arrow length includes head
#     ax.arrow(0, 0, rec_vecs[i, 0], rec_vecs[i, 1], color=f"C{i}", width=0.001, length_includes_head=True, head_width=0.02)

ax.axis("equal")

special_points = {
    "G": [0, 0, 0],
    "K": [-1 / 3, -2 / 3, 0],
    "M": [0, -0.5, 0],
}
k_frac_corners = np.array([special_points[p] for p in ["G", "M", "K", "G"]])
n_points_orig = np.array([7, 3, 5])
n_interemed_points = np.array([2, 3, 3])
n_points_new = n_points_orig + (n_points_orig - 1) * n_interemed_points

k_frac_path = []
for k_frac_start, k_frac_end, n_points in zip(k_frac_corners[:-1], k_frac_corners[1:], n_points_new):
    k_frac_path.append(k_frac_start + (k_frac_end - k_frac_start) * np.linspace(0, 1, n_points)[:, None])
k_frac_path = np.concatenate(k_frac_path)
k_path = k_frac_path @ rec_vecs
n_calcs_new = len(k_path) - 1
ax.plot(k_path[:, 0], k_path[:, 1], marker="o", color="red", label=f'"Bandstructure" path ({n_calcs_new} points)')

for label, k_frac in special_points.items():
    k = k_frac @ rec_vecs
    ax.text(k[0], k[1], label, ha="center", va="center", fontsize=16)
ax.legend()


geoms_path = []
g0 = geoms_trained[0]
for k_frac in k_frac_path[:-1]:
    k_string = f"k={k_frac[0]:+.3f},{k_frac[1]:+.3f}"
    g = Geometry(
        R=g0.R,
        Z=g0.Z,
        periodic=PeriodicLattice(
            lattice_prim=g0.periodic.lattice_prim, supercell=g0.periodic.supercell, k_twist=k_frac
        ),
        comment=f"Graphene_2x2_bandstructure_GMKG_43twists_{k_string}",
        name="Graphene_2x2",
    )
    geoms_path.append(g)
# ds_new = GeometryDataset(geometries=geoms_path, name="Graphene_2x2_bandstructure_GMKG_43twists")
# save_datasets(ds_new)
# save_geometries(geoms_path)
