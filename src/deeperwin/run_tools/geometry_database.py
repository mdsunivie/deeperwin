import numpy as np
from typing import List, Optional, Dict, Union
import hashlib
import pathlib
import pandas as pd
import ase
from deeperwin.run_tools.geometry_utils import BOHR_IN_ANGSTROM
import json

ROUND_R_DECIMALS = 5


def dump_to_json(data, fname):
    def default_encoder(o):
        if hasattr(o, "to_json"):
            return o.to_json()
        return o.__dict__

    with open(fname, "w") as f:
        json.dump(data, f, default=default_encoder, indent=2)

class Geometry:
    def __init__(self, R, Z, charge=0, spin=None, comment="", name=""):
        self.R = np.array(R)
        self.Z = np.array(Z, int)
        self.charge = int(charge)
        if spin is None:
            spin = (sum(Z) - charge) % 2
        self.spin = int(spin)
        self.comment = comment
        self.name = name

    @property
    def n_el(self):
        return int(np.sum(self.Z)) - self.charge
    
    @property
    def n_atoms(self):
        return len(self.Z)
    
    @property
    def n_heavy_atoms(self):
        return len(self.Z[self.Z > 1])

    @property
    def hash(self):
        R = np.round(self.R, decimals=ROUND_R_DECIMALS).astype(float).data.tobytes()
        Z = np.array(self.Z, int).data.tobytes()
        charge = np.array(self.charge, int).data.tobytes()
        spin = np.array(self.spin, int).data.tobytes()
        return hashlib.md5(R + Z + charge + spin).hexdigest()

    @property
    def datset_entry(self):
        return self.hash + "__" + self.comment

    def __len__(self):
        return len(self.R)

    def to_json(self):
        return dict(name=self.name,
                    comment=self.comment,
                    charge=self.charge,
                    spin=self.spin,
                    R=np.array(self.R, float).round(ROUND_R_DECIMALS).tolist(),
                    Z=np.array(self.Z, int).round(ROUND_R_DECIMALS).tolist(),
                    )

    def as_changes_dict(self):
        n_el = int(sum(self.Z) - self.charge)
        return dict(
            R=self.R.tolist(),
            Z=self.Z.tolist(),
            n_electrons = n_el,
            n_up = int(n_el // 2 + self.spin),
            name=self.name,
            comment=self.datset_entry)

    def as_ase(self):
        return ase.Atoms(self.Z, self.R * BOHR_IN_ANGSTROM)

    def as_pyscf_molecule(self, basis_set):
        from deeperwin.orbitals import build_pyscf_molecule
        return build_pyscf_molecule(self.R, self.Z, self.charge, self.spin, basis_set)

    def __repr__(self):
        return f"<Geometry {self.name}, {self.datset_entry}, {self.n_el} el>"



class GeometryDataset:
    def __init__(self, geometries=None, datasets=None, name=""):
        self.name = name
        self.geometries = []
        self.datasets = []
        self.add_dataset(datasets)
        self.add_geometry(geometries)

    def add_geometry(self, geometry):
        if geometry is None:
            return
        if isinstance(geometry, list):
            for g in geometry:
                self.add_geometry(g)
        elif isinstance(geometry, str):
            self.geometries.append(geometry)
        elif isinstance(geometry, Geometry):
            self.add_geometry(geometry.datset_entry)
        else:
            geometry = Geometry(**geometry.__dict__)
            self.add_geometry(geometry)

    def add_dataset(self, dataset):
        if dataset is None:
            return
        if isinstance(dataset, list):
            for d in dataset:
                self.add_dataset(d)
        elif isinstance(dataset, str):
            assert dataset != self.name, f"A dataset cannot include itself: {dataset}"
            self.datasets.append(dataset)
        elif isinstance(dataset, GeometryDataset):
            self.add_dataset(dataset.name)
        else:
            d = GeometryDataset(**dataset.__dict__)
            self.add_dataset(d)


    def get_hashes(self, all_datasets=None):
        geom_hashes = []
        if self.datasets and all_datasets is None:
            all_datasets = load_datasets()
        for d in self.datasets:
            geom_hashes += all_datasets[d].get_hashes(all_datasets)
        geom_hashes += [g.split("__")[0] for g in self.geometries]
        return geom_hashes


    def get_geometries(self, all_geometries=None, all_datasets=None):
        all_geometries = all_geometries or load_geometries()
        geometries = [all_geometries[h] for h in self.get_hashes(all_datasets)]
        return geometries


    def as_ase(self, all_geometries=None):
        geometries = self.get_geometries(all_geometries)
        return [g.as_ase() for g in geometries]

    def get_total_nr_of_geometries(self, all_geometries=None, all_datasets=None):
        return len(self.get_geometries(all_geometries, all_datasets))

    def __repr__(self):
        return f"<GeometryDataset {self.name}: {len(self.geometries)} geometries, {len(self.datasets)} datasets>"


def _get_default_geom_fname():
    return pathlib.Path(__file__).parent.joinpath("../../../datasets/db/geometries.json").resolve()

def _get_default_datasets_fname():
    return pathlib.Path(__file__).parent.joinpath("../../../datasets/db/datasets.json").resolve()

def _get_default_energies_fname():
    return pathlib.Path(__file__).parent.joinpath("../../../datasets/db/energies.csv").absolute()


def load_geometries(geom_db_fname=None) -> Dict[str, Geometry]:
    geom_db_fname = geom_db_fname or _get_default_geom_fname()
    with open(geom_db_fname, "r") as f:
        geometries = json.load(f)
    if geometries is None:
        geometries = dict()
    geometries = {h: Geometry(**g) for h, g in geometries.items()}
    return geometries


def load_datasets(datasets_db_fname=None) -> Dict[str, GeometryDataset]:
    datasets_db_fname = datasets_db_fname or _get_default_datasets_fname()
    with open(datasets_db_fname, "r") as f:
        datasets = json.load(f)
    if datasets is None:
        datasets = dict()
    datasets = {name: GeometryDataset(**d) for name, d in datasets.items()}
    return datasets

def get_all_geometries(dataset_or_geom: Union[List[str], str], all_geoms=None, all_datasets=None):
    """
    Get an identifer of a dataset, a geometry, a list of geometries or a list of datasets and return a list of all geometries within this set.

    Args:
        dataset_or_geom: A dataset name, a geometry name, a list of geometries or a list of datasets. Type: Union[List[str], str]
        all_geoms: A dictionary of all geometries. If None, load from default location. Type: Dict[str, Geometry]
        all_datasets: A dictionary of all datasets. If None, load from default location. Type: Dict[str, GeometryDataset]
    """
    all_geoms = all_geoms or load_geometries()
    all_datasets = all_datasets or load_datasets()

    output_geoms = []
    if not isinstance(dataset_or_geom, str):
        for d in dataset_or_geom:
            output_geoms += get_all_geometries(d, all_geoms, all_datasets)
    else:
        if dataset_or_geom in all_geoms:
            output_geoms.append(all_geoms[dataset_or_geom])
        else:
            output_geoms += all_datasets[dataset_or_geom].get_geometries(all_geoms, all_datasets)
    return output_geoms


def load_energies(energies_db_fname=None):
    energies_db_fname = energies_db_fname or _get_default_energies_fname()
    if pathlib.Path(energies_db_fname).is_file():
        return pd.read_csv(energies_db_fname, delimiter=";")
    else:
        return pd.DataFrame()


def save_energies(energies: pd.DataFrame, energies_db_fname=None):
    energies_db_fname = energies_db_fname or _get_default_energies_fname()
    energies.to_csv(energies_db_fname, sep=";", index=False)


def append_energies(energies: pd.DataFrame, energies_db_fname=None, allow_new_columns=False):
    energies_db_fname = energies_db_fname or _get_default_energies_fname()
    all_energies = load_energies(energies_db_fname)
    n_rows_orig = len(all_energies)

    if not allow_new_columns:
        for c in list(energies):
            if c not in list(all_energies):
                raise ValueError(f"Trying to add a new column: {c}")
    all_energies = pd.concat([all_energies, energies], ignore_index=True)
    columns_for_duplicates = [c for c in list(all_energies) if c not in ["E", "E_sigma"]]
    len_before_dedup = len(all_energies)
    all_energies = all_energies.drop_duplicates(subset=columns_for_duplicates, keep="last")
    n_dup = len_before_dedup - len(all_energies)
    n_added = len(all_energies) - n_rows_orig
    if n_dup:
        print(f"Overwrote {n_dup} duplicate entries with (potentially) updated energies")
    print(f"Added {n_added} new rows to energy database")
    save_energies(all_energies)
    return all_energies


def save_geometries(geometries, geom_db_fname=None, overwite_existing=False):
    geom_db_fname = geom_db_fname or _get_default_geom_fname()
    if isinstance(geometries, list):
        geometries = {g.hash: g for g in geometries}

    if pathlib.Path(geom_db_fname).is_file():
        all_geoms = load_geometries(geom_db_fname)
    else:
        all_geoms = dict()

    n_skipped = 0
    n_added = 0
    for h, g in geometries.items():
        if (h in all_geoms) and (not overwite_existing):
            print(f"Skipping existing geometry: {h}")
            n_skipped += 1
            continue
        else:
            all_geoms[h] = g
            n_added += 1

    dump_to_json(all_geoms, geom_db_fname)
    print(f"Added {n_added} geometries, skipped {n_skipped} existing")



def save_datasets(datasets, dataset_db_fname=None, overwite_existing=False):
    dataset_db_fname = dataset_db_fname or _get_default_datasets_fname()
    if isinstance(datasets, GeometryDataset):
        datasets = [datasets]

    if pathlib.Path(dataset_db_fname).is_file():
        all_datasets = load_datasets(dataset_db_fname)
    else:
        all_datasets = dict()

    n_skipped = 0
    n_added = 0
    for dataset in datasets:
        if dataset.name in all_datasets:
            if overwite_existing:
                print(f"Overwriting existing dataset: {dataset.name}")
            else:
                print(f"Skipping existing dataset: {dataset.name}")
                n_skipped += 1
                continue
        n_added += 1
        all_datasets[dataset.name] = dataset

    dump_to_json(all_datasets, dataset_db_fname)
    print(f"Added {n_added} datasets, skipped {n_skipped} existing")


def expand_geometry_list(geometries, geometry_db=None, datasets_db=None):
    if (geometry_db is None) or isinstance(geometry_db, str):
        geometry_db = load_geometries(geometry_db)
    if (datasets_db is None) or isinstance(datasets_db, str):
        datasets_db = load_datasets(datasets_db)

    result = []
    if isinstance(geometries, list):
        # geometries is a list of strings -> Recurse into each element
        for g in geometries:
            result += expand_geometry_list(g, geometry_db, datasets_db)
    else:
        geom_id = geometries.split("__")[0]
        if geom_id in geometry_db:
            # geom_id is the hash of a geometry -> Load the data
            result.append(geometry_db[geom_id].as_changes_dict())
        elif geom_id in datasets_db:
            # geom_id is actually a dataset_id -> Loop through all geometries in dataset
            for geometry in datasets_db[geom_id].get_geometries(geometry_db):
                result.append(geometry.as_changes_dict())
        else:
            raise KeyError(f"Could not find geometry/dataset: {geom_id}")
    return result


if __name__ == '__main__':
    geom_fname = "/home/mscherbela/tmp/test.json"

    R = np.eye(3)
    Z = [1,2,3]
    g1 = Geometry(R=R, Z=Z, comment="test1")
    g2 = Geometry(R=2*R, Z=Z, comment="test2")
    dump_to_json({g.hash: g for g in [g1, g2]}, geom_fname)

    print("Loading...")
    all_geoms = load_geometries()
    all_datasets = load_datasets()
    print("Done!")

    # dump_to_json(all_geoms, "/home/mscherbela/develop/deeperwin_jaxtest/datasets/db/geometries.json")
    # dump_to_json(all_datasets, "/home/mscherbela/develop/deeperwin_jaxtest/datasets/db/datasets.json")


    # loaded_geoms = load_geometries(geom_fname)
    # loaded_geometries = load_geometries()
    # loaded_datasets = load_datasets()
    # ds = loaded_datasets['ds_dummy']
    # geoms = ds.get_geometries()
    # print(len(geoms))

