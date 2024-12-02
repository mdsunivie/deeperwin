# %%
from deeperwin.run_tools.geometry_database import load_geometries
import ase.visualize

all_geoms = load_geometries()
geoms = [g for g in all_geoms.values() if "LiH_16at_Gamma" in g.comment]

ase.visualize.view([g.as_ase() for g in geoms])
