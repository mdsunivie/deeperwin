# %%
import numpy as np
from deeperwin.configuration import PhysicalConfig, PeriodicConfig
from deeperwin.hamiltonian import (
    get_ewald_el_el_potential_energy,
    get_ewald_el_ion_potential_energy,
    get_ewald_ion_ion_potential_energy,
)
from deeperwin.utils.periodic import LatticeParams, project_into_first_unit_cell
from deeperwin.utils.utils import get_distance_matrix, get_el_ion_distance_matrix


def _get_r_test():
    return np.array([[-0.45298942, 0.78126198, 0.21798746], [-0.9536655, 1.19959636, -0.15530065]])


def _get_physical_config():
    return PhysicalConfig(
        name="He", periodic=PeriodicConfig(lattice_prim=(np.eye(3) * 0.5).tolist(), supercell=[1, 1, 1])
    )


def test_el_el_potential_energy_periodic():
    phys_config = _get_physical_config()
    fixed_params_periodic = LatticeParams.from_periodic_config(phys_config.periodic)
    r = _get_r_test()
    r = project_into_first_unit_cell(r, fixed_params_periodic["lattice"])
    diff_el_el, _ = get_distance_matrix(r, full=True)
    E_el_el = get_ewald_el_el_potential_energy(
        diff_el_el,
        fixed_params_periodic["volume"],
        fixed_params_periodic["rec_vectors"],
        fixed_params_periodic["rec_vectors_weights"],
        fixed_params_periodic["gamma"],
        fixed_params_periodic["lat_vectors"],
        fixed_params_periodic["madelung_const"],
        phys_config.periodic.include_heg_background,
    )
    assert np.isclose(E_el_el.real, 7.0113783)


def test_ion_ion_potential_energy_periodic():
    phys_config = _get_physical_config()
    fixed_params_periodic = LatticeParams.from_periodic_config(phys_config.periodic)
    _, _, R, Z = phys_config.get_basic_params()
    E_ion_ion = get_ewald_ion_ion_potential_energy(
        R,
        Z,
        fixed_params_periodic["volume"],
        fixed_params_periodic["rec_vectors"],
        fixed_params_periodic["rec_vectors_weights"],
        fixed_params_periodic["gamma"],
        fixed_params_periodic["lat_vectors"],
        fixed_params_periodic["madelung_const"],
    )
    assert np.isclose(E_ion_ion.real, 0.0)


def test_el_ion_potential_energy_periodic():
    phys_config = _get_physical_config()
    fixed_params_periodic = LatticeParams.from_periodic_config(phys_config.periodic)
    _, _, R, Z = phys_config.get_basic_params()
    r = _get_r_test()
    r = project_into_first_unit_cell(r, fixed_params_periodic["lattice"])

    diff_el_ion, _ = get_el_ion_distance_matrix(r, R)
    E_el_ion = get_ewald_el_ion_potential_energy(
        diff_el_ion,
        Z,
        fixed_params_periodic["volume"],
        fixed_params_periodic["rec_vectors"],
        fixed_params_periodic["rec_vectors_weights"],
        fixed_params_periodic["gamma"],
        fixed_params_periodic["lat_vectors"],
        fixed_params_periodic["madelung_const"],
    )
    assert np.isclose(E_el_ion.real, -18.826935)


if __name__ == "__main__":
    test_el_el_potential_energy_periodic()
    test_ion_ion_potential_energy_periodic()
    test_el_ion_potential_energy_periodic()

    # import matplotlib.pyplot as plt
    # phys_config = _get_physical_config()
    # fixed_params_periodic = get_fixed_params_periodic(phys_config.periodic)

    # plt.close("all")
    # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # lat_r = fixed_params_periodic["lat_vectors"]
    # lat_k = fixed_params_periodic["rec_vectors"]

    # lat_r = lat_r[lat_r[:, -1] == 0, :]
    # lat_k = lat_k[lat_k[:, -1] == 0, :]
    # axes[0].plot(lat_r[:, 0], lat_r[:, 1], "o")
    # axes[1].plot(lat_k[:, 0], lat_k[:, 1], "o")
    # axes[0].set_aspect("equal")
    # axes[1].set_aspect("equal")
    # axes[0].set_title("Real space")
    # axes[1].set_title("Reciprocal space")


# %%
