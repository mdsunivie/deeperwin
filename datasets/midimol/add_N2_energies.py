# %%
from deeperwin.run_tools.geometry_database import load_datasets, load_energies, append_energies
import numpy as np
import pandas as pd

ds = load_datasets()["N2_stretching_GerardEtAl2022"]
geoms = ds.get_geometries()
energy_df = load_energies()


energies = [
    -109.1797146,
    -109.1817047,
    -109.1896196,
    -109.1987674,
    -109.2145396,
    -109.284463,
    -109.4144659,
    -109.5387708,
    -109.5413328,
    -109.2229667,
]
distances = [5.338, 4.805, 4.271, 4.000, 3.737, 3.203, 2.669, 2.135, 2.068, 1.602]

energy_data = []
meta_data = dict(
    experiment="Gerard_et_al_2022_N2_bond_breaking",
    energy_type="eval",
    epoch=100_000,
    method="indep",
    source="dpe",
    epoch_geom=100_000,
    embedding="dpe4",
    orbitals="dpe4_32fd",
)
for E, d in zip(energies, distances):
    diffs = [abs(g.R[1, 0] - d) for g in geoms]
    ind_closest = np.argmin(diffs)
    min_diff = diffs[ind_closest]
    g = geoms[ind_closest]
    assert min_diff < 1e-3
    energy_data.append(dict(geom=g.hash, E=E, geom_comment=g.comment, molecule=g.name, **meta_data))
df = pd.DataFrame(energy_data)
append_energies(df)

# %%
