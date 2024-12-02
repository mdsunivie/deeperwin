# %%
from deeperwin.run_tools.geometry_database import (
    save_geometries,
    Geometry,
)
import numpy as np

ANGSTROM_IN_BOHR = 1.8897161646320724

# sp2 cc and ch bond lengths
CC_bond = 1.34 * ANGSTROM_IN_BOHR
CH_bond = 1.086 * ANGSTROM_IN_BOHR
theta = np.deg2rad(120)


def get_cumulene(N, phi_in_deg):
    phi = np.deg2rad(phi_in_deg)
    R_carbon = np.array([CC_bond, 0, 0]) * np.arange(N)[:, None]

    R_left = R_carbon[0]
    R_right = R_carbon[-1]
    R_hydrogen = np.ones([4, 3]) * CH_bond

    R_hydrogen[0] = R_left + CH_bond * np.array([np.cos(theta), np.sin(theta) * np.cos(0), np.sin(theta) * np.sin(0)])
    R_hydrogen[1] = R_left + CH_bond * np.array(
        [np.cos(theta), np.sin(theta) * np.cos(0 + np.pi), np.sin(theta) * np.sin(0 + np.pi)]
    )
    R_hydrogen[2] = R_right - CH_bond * np.array(
        [np.cos(theta), np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)]
    )
    R_hydrogen[3] = R_right - CH_bond * np.array(
        [np.cos(theta), np.sin(theta) * np.cos(phi + np.pi), np.sin(theta) * np.sin(phi + np.pi)]
    )
    return np.concatenate([R_carbon, R_hydrogen]), np.concatenate([np.ones(N) * 6, np.ones(4)])


geometries = []
for n in range(2, 41):
    for phi in [0, 90]:
        R, Z = get_cumulene(n, phi)
        name = f"cumulene_C{n}H4_{phi}deg"
        g = Geometry(R=R, Z=Z, name=name, comment=name)
        geometries.append(g)

save_geometries(geometries)
