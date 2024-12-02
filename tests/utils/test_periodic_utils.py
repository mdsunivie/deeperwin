# %%
from deeperwin.utils.periodic import cartesian_to_fractional, fractional_to_cartesian, project_into_first_unit_cell
from deeperwin.utils.utils import get_periodic_distance_matrix
import numpy as np


def _get_random_lattice():
    while True:
        lattice = np.random.normal(size=(3, 3))
        volume = np.abs(np.linalg.det(lattice))
        if volume > 0.1 and volume < 10:
            # Ensure a reasonable, invertible lattice
            return lattice


def test_fraction_back_and_forth():
    lattice = _get_random_lattice()
    batch_size = 50
    r = np.random.normal(size=(batch_size, 3), scale=10)
    r_frac = cartesian_to_fractional(r, inv_lattice=np.linalg.inv(lattice))
    r_back = fractional_to_cartesian(r_frac, lattice=lattice)
    assert np.allclose(r, r_back, rtol=1e-5, atol=1e-4)


def test_projection_into_first_unit_cell():
    lattice = _get_random_lattice()
    batch_size = 50
    r = np.random.normal(size=(batch_size, 3), scale=3)
    diff, dist = get_periodic_distance_matrix(r, lattice)
    r_first_unit = project_into_first_unit_cell(r, lattice)
    diff_first_unit, dist_first_unit = get_periodic_distance_matrix(r_first_unit, lattice)
    r_frac = cartesian_to_fractional(r_first_unit, inv_lattice=np.linalg.inv(lattice))
    # assert np.all(np.abs(r_frac) <= 0.5)
    assert np.all((r_frac >= 0) & (r_frac <= 1))
    assert np.allclose(dist, dist_first_unit, rtol=1e-4, atol=1e-4)
    assert np.allclose(diff, diff_first_unit, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_projection_into_first_unit_cell()

    # import matplotlib.pyplot as plt

    # def plot2Dlattice(lattice, n_grid=5, ax=None):
    #     if isinstance(n_grid, int):
    #         n_grid = np.arange(-n_grid, n_grid+1)
    #     if ax is None:
    #         ax = plt.gca()
    #     r0 = min(n_grid) * lattice[0]
    #     r1 = max(n_grid) * lattice[0]
    #     for n in n_grid:
    #         shift = lattice[1] * n
    #         ax.plot([(r0 + shift)[0], (r1 + shift)[0]], [(r0 + shift)[1], (r1 + shift)[1]], color='gray', zorder=-1)
    #     r0 = min(n_grid) * lattice[1]
    #     r1 = max(n_grid) * lattice[1]
    #     for n in n_grid:
    #         shift = lattice[0] * n
    #         ax.plot([(r0 + shift)[0], (r1 + shift)[0]], [(r0 + shift)[1], (r1 + shift)[1]], color='gray', zorder=-1)

    # def draw_point_and_repetitions(r, lattice, n_rep, ax=None, color=None):
    #     ax.scatter(r[None, 0], r[None, 1], color=color, s=100)
    #     ind_rep = np.array(np.meshgrid(np.arange(-n_rep, n_rep+1), np.arange(-n_rep, n_rep+1))).T.reshape(-1, 2)
    #     r_rep = ind_rep @ lattice + r
    #     ax.scatter(r_rep[:, 0], r_rep[:, 1], color=color, alpha=0.2)

    # lattice = np.array([[-1.0, 0], [1, 1]])

    # grid_range = np.arange(-3, 3+1)
    # n_grid = np.array(np.meshgrid(grid_range, grid_range)).T.reshape(-1, 2)
    # r_grid = n_grid @ lattice

    # plt.close("all")
    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # # plt.scatter(r_grid[:, 0], r_grid[:, 1], color='gray')
    # plot2Dlattice(lattice, n_grid=grid_range)
    # ax.axis("equal")

    # r0 = np.array([0.2,0.5])
    # r1 = np.array([2.5, 3.2])
    # r1_back = project_into_first_unit_cell(r1, lattice)

    # draw_point_and_repetitions(r0, lattice, 3, ax=ax, color='blue')
    # draw_point_and_repetitions(r1, lattice, 3, ax=ax, color='red')
    # draw_point_and_repetitions(r1_back, lattice, 3, ax=ax, color='green')

    # np.set_printoptions(precision=3, suppress=True)
    # diff01, dist01 = get_periodic_distance_matrix(np.stack([r0, r1], axis=-2), lattice)
    # diff01back, dist01back = get_periodic_distance_matrix(np.stack([r0, r1_back], axis=-2), lattice)
    # print("Diff:")
    # print(diff01[0, 1])
    # print(diff01back[0, 1])

    # print("Dist:")
    # print(dist01[0, 1])
    # print(dist01back[0, 1])

# %%
