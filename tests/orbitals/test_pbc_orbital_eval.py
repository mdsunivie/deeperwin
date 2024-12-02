# %%
from deeperwin.configuration import PhysicalConfig, PeriodicConfig, PeriodicMeanFieldConfig
from deeperwin.utils.utils import get_el_ion_distance_matrix
from deeperwin.orbitals import (
    get_baseline_solution,
    build_pyscf_molecule_from_physical_config,
    eval_atomic_orbitals_periodic,
)
import numpy as np
from pyscf.pbc.gto.eval_gto import eval_gto
import pytest


@pytest.fixture(scope="module")
def baseline_result():
    atom_spacing = 2.0
    n_supercells = [1, 1, 2]
    lattice_const_prim = 2 * atom_spacing
    phys = PhysicalConfig(
        R=[[0, 0, 0], [0, 0, 2]],
        Z=[1, 1],
        periodic=PeriodicConfig(
            lattice_prim=np.diag([4, 4, lattice_const_prim]).tolist(), supercell=n_supercells, k_twist=[0, 0, 0.3]
        ),
    )
    phys = phys.get_expanded_if_supercell()
    hf_config = PeriodicMeanFieldConfig(
        basis_set="gth-dzv",
        pseudo="gth-pade",
        k_points=phys.periodic.supercell,
    )

    supercell = build_pyscf_molecule_from_physical_config(
        phys, hf_config.basis_set, hf_config.pseudo, use_primitive=False
    )
    orbital_params, energies = get_baseline_solution(phys, hf_config)
    return orbital_params, supercell, n_supercells


@pytest.fixture(scope="module")
def example_coords(baseline_result):
    orbital_params, supercell, n_supercells = baseline_result
    R = np.array(supercell.atom_coords())

    n_eval = 200
    x = np.linspace(0, 1, n_eval) * 8 * 2
    r = np.array([[0, 0, 0]]) + np.array([0, 0, 1])[None, :] * x[:, None, None]
    diff_el_ion, dist_el_ion = get_el_ion_distance_matrix(r, R)
    dist_el_ion = np.linalg.norm(diff_el_ion, axis=-1)
    return r, R, diff_el_ion, dist_el_ion


@pytest.fixture(scope="module")
def orbitals_jax(baseline_result, example_coords):
    orbital_params, supercell, n_supercells = baseline_result
    r, R, diff, dist = example_coords

    aos_jax = eval_atomic_orbitals_periodic(
        diff,
        dist,
        orbital_params.atomic_orbitals,
        orbital_params.lattice_vecs,
        orbital_params.shift_vecs,
        orbital_params.k_twist,
    )[:, 0, :]
    mos_jax = aos_jax @ orbital_params.mo_coeff[0]
    return aos_jax, mos_jax


@pytest.fixture(scope="module")
def orbitals_pyscf(baseline_result, example_coords):
    orbital_params, supercell, n_supercells = baseline_result
    r, R, diff, dist = example_coords

    aos_pyscf_sc = eval_gto(supercell, "GTOval_cart", r, kpts=orbital_params.k_twist)
    mos_pyscf_sc = aos_pyscf_sc @ orbital_params.mo_coeff[0]
    return aos_pyscf_sc, mos_pyscf_sc


def test_pyscf_orbitals_equal_jax_aos(orbitals_jax, orbitals_pyscf):
    aos_jax, mos_jax = orbitals_jax
    aos_pyscf_sc, mos_pyscf_sc = orbitals_pyscf

    assert np.allclose(aos_jax, aos_pyscf_sc, atol=1e-2)
    assert np.allclose(mos_jax, mos_pyscf_sc, atol=1e-2)


def test_phase_equals_twist(orbitals_jax, baseline_result, example_coords):
    orbital_params, supercell, n_supercells = baseline_result
    r, R, diff, dist = example_coords
    _, mos_jax = orbitals_jax

    phase_expected = np.exp(1j * np.dot(r[-1, 0] - r[0, 0], orbital_params.k_twist))
    phase_mos = mos_jax[-1] / mos_jax[0]
    assert np.allclose(phase_expected, phase_mos, atol=1e-2)
