# %%
import numpy as np
from deeperwin.run_tools.geometry_database import Geometry
from deeperwin.utils.utils import ANGSTROM_IN_BOHR
from ase.visualize import view


def get_polyethylene_geometry_lapnet(n_rep, periodic):
    """
    Get the geometry of polyethylene chains as used in the LapNet paper: https://arxiv.org/abs/2307.08214
    https://github.com/bytedance/LapNet/blob/main/lapnet/configs/pe.py

    """
    # C-C length
    lcc = 1.55 * ANGSTROM_IN_BOHR
    # C-H length
    lch = 1.09 * ANGSTROM_IN_BOHR
    theta = 109.4712 / 180.0 * 3.1415926

    # repeat config
    dxc = lcc * np.sin(theta / 2.0)
    dyc = lcc * np.cos(theta / 2.0)
    dxh = lch * np.sin(theta / 2.0)
    dyh = lch * np.cos(theta / 2.0)

    pos_C1 = np.array((-0.5 * dxc, -0.5 * dyc, 0.0))
    pos_C2 = np.array((0.5 * dxc, 0.5 * dyc, 0.0))

    pos_H11 = np.array((-0.5 * dxc, -0.5 * dyc - dyh, dxh))
    pos_H12 = np.array((-0.5 * dxc, -0.5 * dyc - dyh, -dxh))
    pos_H21 = np.array((0.5 * dxc, 0.5 * dyc + dyh, dxh))
    pos_H22 = np.array((0.5 * dxc, 0.5 * dyc + dyh, -dxh))
    rep_dis = np.array((2 * dxc, 0, 0))

    R = np.zeros([6 * n_rep, 3])
    Z = np.zeros([6 * n_rep], int)
    for i in range(n_rep):
        R[6 * i, :] = pos_C1 + i * rep_dis
        R[6 * i + 1, :] = pos_C2 + i * rep_dis
        R[6 * i + 2, :] = pos_H11 + i * rep_dis
        R[6 * i + 3, :] = pos_H12 + i * rep_dis
        R[6 * i + 4, :] = pos_H21 + i * rep_dis
        R[6 * i + 5, :] = pos_H22 + i * rep_dis
        Z[6 * i : 6 * i + 6] = [6, 6, 1, 1, 1, 1]
    if periodic:
        lattice = np.eye(3) * 10.0
        lattice[0, :] = rep_dis * n_rep
    else:
        lattice = None
        # Terminate with H-atoms
        dpos_Hl = np.array((-dxh, dyh, 0)) + pos_C1
        dpos_Hr = np.array((dxh, -dyh, 0)) + pos_C2 + rep_dis * (n_rep - 1)
        R = np.concatenate([dpos_Hl[None, :], R, dpos_Hr[None, :]], axis=0)
        Z = np.concatenate([[1], Z, [1]], axis=0)
    return R, Z, lattice


R, Z, lat = get_polyethylene_geometry_lapnet(4, False)

g = Geometry(R, Z)
view(g.as_ase())
