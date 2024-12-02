# %%
from deeperwin.configuration import (
    PhysicalConfig,
    CASSCFConfig,
    HartreeFockConfig,
    PeriodicConfig,
    PeriodicMeanFieldConfig,
)
from deeperwin.orbitals import get_baseline_solution
import tempfile
import os
import numpy as np
import pyscf.pbc.gto


def test_casscf_calculation():
    phys_config = PhysicalConfig(name="LiH")
    casscf_config = CASSCFConfig(basis_set="6-311G", n_dets=4)
    with tempfile.TemporaryDirectory(prefix="pyscf") as tmpdirname:
        os.environ["PYSCF_TMPDIR"] = tmpdirname
        orbital_params, energies = get_baseline_solution(phys_config, casscf_config)
    assert energies["E_casscf"] < energies["E_hf"]
    assert len(orbital_params.idx_orbitals[0]) == casscf_config.n_dets
    assert orbital_params.mo_coeff[0].shape == (16, 16)


def test_hf_calculation():
    phys_config = PhysicalConfig(name="LiH")
    hf_config = HartreeFockConfig(basis_set="6-311G")
    with tempfile.TemporaryDirectory(prefix="pyscf") as tmpdirname:
        os.environ["PYSCF_TMPDIR"] = tmpdirname
        orbital_params, energies = get_baseline_solution(phys_config, hf_config)
    assert len(energies) == 1 and "E_hf" in energies
    assert orbital_params.mo_coeff[0].shape == (16, 16)


def test_periodic_pyscf_calculation():
    phys_config = PhysicalConfig(name="He", periodic=PeriodicConfig(lattice_type="sc", lattice_const="2.5"))
    hf_config = PeriodicMeanFieldConfig(basis_set="gth-dzv", pseudo="gth-pade", k_points=[2, 2, 2])

    with tempfile.TemporaryDirectory(prefix="pyscf") as tmpdirname:
        os.environ["PYSCF_TMPDIR"] = tmpdirname
        orbital_params, energies = get_baseline_solution(phys_config, hf_config)
    assert "E_periodic_hf" in energies
    assert orbital_params.mo_coeff[0].shape == (2, 16)  # 2 basis functions x 8 kpoints * 2 bands
    assert orbital_params.mo_energies[0].shape == (16,)  # 8 kpoints * 2 bands
    assert orbital_params.mo_occ[0].shape == (16,)  # 8 kpoints * 2 bands
    assert np.all(orbital_params.mo_occ[0][:8] == 1)  # first orbitals occupied
    assert orbital_params.k_points[0].shape == (3, 16)  # 3D k-vector x 8 kpoints * 2 bands


def _assert_is_close(a, b):
    assert np.isclose(a, b), f"{a} != {b}"


def test_pyscf_k_points():
    a = 1.0
    cell = pyscf.pbc.gto.Cell(atom="He 0 0 0", a=np.eye(3) * a, unit="bohr", basis="gth-szv", pseudo="gth-pade")
    cell.build()
    k_points = cell.make_kpts([1, 1, 2])
    _assert_is_close(k_points[1, 2] * 2 * a, 2 * np.pi)


if __name__ == "__main__":
    test_pyscf_k_points()


# %%
