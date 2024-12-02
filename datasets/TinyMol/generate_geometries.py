from deeperwin.utils.utils import PERIODIC_TABLE, ANGSTROM_IN_BOHR
from rdkit.Chem import AllChem as Chem
from deeperwin.run_tools.geometry_database import Geometry, GeometryDataset, save_geometries, save_datasets
from deeperwin.run_tools.geometry_utils import generate_geometry_variants
import itertools
from collections import Counter
import numpy as np

np.random.seed(1234)

elements = ["C", "N", "O"]
bond_types = ["-", "=", "#"]
chain_lengths = 3


def get_sum_formula(Z):
    counts = Counter(Z)
    Z = sorted(counts.keys())
    formula = ""
    for z in Z:
        if z == 1:
            continue
        formula += PERIODIC_TABLE[z - 1]
        if counts[z] > 1:
            formula += str(counts[z])
    if 1 in counts:
        formula += "H"
        if counts[1] > 1:
            formula += str(counts[1])
    return formula


unique_smiles = set()

# 1 heavy atom
for e in elements:
    unique_smiles.add(e)

# 2 heavy atoms
for symbols in itertools.product(elements, bond_types, elements):
    smiles = "".join(symbols)
    m = Chem.MolFromSmiles(smiles)
    if m is not None:
        unique_smiles.add(Chem.MolToSmiles(m))

# 3 heavy atoms
for symbols in itertools.product(elements, bond_types, elements, bond_types, elements):
    smiles = "".join(symbols)
    m = Chem.MolFromSmiles(smiles)
    if m is not None:
        unique_smiles.add(Chem.MolToSmiles(m))

geometries = []
for smiles in unique_smiles:
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    Chem.EmbedMolecule(m, randomSeed=0)
    Chem.MMFFOptimizeMoleculeConfs(m)
    conformers = m.GetConformers()
    if len(conformers) > 1:
        print(smiles, len(conformers))
    for conf in m.GetConformers():
        R = conf.GetPositions()
        R = R - np.mean(R, axis=0, keepdims=True)
        _, _, v = np.linalg.svd(R)
        R = R @ v.T
        R = np.round(R, 5)
        Z = [a.GetAtomicNum() for a in m.GetAtoms()]
        geometries.append(
            Geometry(
                R=R * ANGSTROM_IN_BOHR, Z=Z, name=get_sum_formula(Z), comment=f"{get_sum_formula(Z)}_smiles_{smiles}"
            )
        )


n_el_max = 25
print(f"All molecules                  : {len(geometries)}")
geometries = [g for g in geometries if "+" not in g.name]
print(f"No ionic bonds                 : {len(geometries)}")
geometries = [g for g in geometries if g.n_el <= n_el_max]
print(f"Molecules with <= {n_el_max} electrons : {len(geometries)}")

equilibrium_geometries = sorted(geometries, key=lambda g: (g.n_el, max(g.Z), len(g.Z)))
geometries_train = [g for g in geometries if g.n_el <= 18]
geometries_test = [g for g in geometries if g.n_el > 18]

equilibrium_datasets = []
for geoms in [equilibrium_geometries, geometries_train, geometries_test]:
    n_el_min = min([g.n_el for g in geoms])
    n_el_max = max([g.n_el for g in geoms])
    dataset = GeometryDataset(geoms, name=f"TinyMol_CNO_equilibrium_{n_el_min}-{n_el_max}el_{len(geoms)}geoms")
    print(dataset.name)
    equilibrium_datasets.append(dataset)

# %% Generate distorted datasets
distorted_datasets = []
distorted_geometries = []

train_datasets = []
test_datasets = []
N_distortions = 60
for j, g in enumerate(equilibrium_geometries):
    print(f"Distorting geometry {j}")
    R = generate_geometry_variants(
        g.R, N_distortions, rotate=True, distort=True, include_orig=True, random_state=1234 + j, noise_scale=0.3
    )
    name = g.name
    distorted_geoms = [
        Geometry(R[i], g.Z, g.charge, g.spin, name=name, comment=f"{g.comment}_rot_dist_{i}")
        for i in range(N_distortions)
    ]
    data_train_10 = GeometryDataset(distorted_geoms[:10], name=f"TinyMol_CNO_rot_dist_{g.comment}_10geoms")
    data_train_20 = GeometryDataset(distorted_geoms[:20], name=f"TinyMol_CNO_rot_dist_{g.comment}_20geoms")
    data_train_50 = GeometryDataset(distorted_geoms[:50], name=f"TinyMol_CNO_rot_dist_{g.comment}_50geoms")
    data_test_10 = GeometryDataset(distorted_geoms[-10:], name=f"TinyMol_CNO_rot_dist_{g.comment}_test_10geoms")

    distorted_geometries += distorted_geoms
    distorted_datasets += [data_train_10, data_train_20, data_train_50, data_test_10]
    if g.n_el <= 18:
        train_datasets += [data_train_10, data_train_20, data_train_50]
    else:
        test_datasets += [data_test_10]

data_train_10each = GeometryDataset(datasets=[d for d in train_datasets if len(d.geometries) == 10])
data_train_20each = GeometryDataset(datasets=[d for d in train_datasets if len(d.geometries) == 20])
data_train_50each = GeometryDataset(datasets=[d for d in train_datasets if len(d.geometries) == 50])
data_test_10each = GeometryDataset(datasets=test_datasets)

for d, n, d_type in zip(
    [data_train_10each, data_train_20each, data_train_50each, data_test_10each],
    [10, 20, 50, 10],
    ["train", "train", "train", "test"],
):
    n_compounds = len(d.datasets)
    d.name = f"TinyMol_CNO_rot_dist_{d_type}_{n_compounds}compounds_{n_compounds*n}geoms"

# %%
save_geometries(equilibrium_geometries + distorted_geometries)
save_datasets(equilibrium_datasets + distorted_datasets)
save_datasets([data_train_10each, data_train_20each, data_train_50each, data_test_10each])


# ase.visualize.view([g.as_ase() for g in geometries])
