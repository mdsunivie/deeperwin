# %%
from deeperwin.run_tools.geometry_database import (
    append_energies,
    Geometry,
    GeometryDataset,
    save_geometries,
    save_datasets,
)
import numpy as np
from deeperwin.utils.utils import ANGSTROM_IN_BOHR
import pandas as pd

source = "Booth_2011_JCP_FCIQMC_cc-pVTZ"
d_angstrom = np.array([0.9, 1.0, 1.1, 1.2, 1.24253, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8])
d_bohr = d_angstrom * ANGSTROM_IN_BOHR
E = [
    -75.436,
    -75.6547,
    -75.7614,
    -75.7987,
    -75.80251,
    -75.7993,
    -75.7798,
    -75.7243,
    -75.6805,
    -75.6454,
    -75.6185,
    -75.5998,
    -75.5881,
    -75.5798,
]

geoms = [
    Geometry(
        R=np.array([[0, 0, 0], [d_, 0, 0]]),
        Z=np.array([6, 6]),
        name="C2",
        comment=f"C2_dimer_{d_:.4f}",
    )
    for d_ in d_bohr
]
dataset = GeometryDataset(geoms, name="C2_dimer_Booth2011_14geoms")
save_geometries(geoms)
save_datasets(dataset)

data = []
for g, E in zip(geoms, E):
    data.append(
        dict(
            E=E,
            geom=g.hash,
            geom_comment=g.comment,
            molecule="C2",
            source=source,
            method="indep",
            experiment=source,
        )
    )
data = pd.DataFrame(data)
append_energies(data)


# %%
