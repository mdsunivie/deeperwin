import numpy as np

PERIODIC_TABLE = 'H He Li Be B C N O F Ne'.split()
ANGSTROM_IN_BOHR = 1.88973

def save_xyz_file(fname, R, Z, comment=""):
    with open(fname, 'w') as f:
        f.write(str(len(R)) + '\n')
        f.write(comment + '\n')
        for Z_, R_ in zip(Z, R):
            f.write(f"{PERIODIC_TABLE[Z_-1]:3>} {R_[0]:-.10f} {R_[1]:-.10f} {R_[2]:-.10f}\n")


def get_ethene_molecule(rCC=1.33, rCH=1.08, twist_in_deg=0, HCC_angle_in_deg=121.7, units="bohr"):
    Z = [6,6,1,1,1,1]
    sin_twist,cos_twist = np.sin(twist_in_deg * np.pi / 180), np.cos(twist_in_deg * np.pi / 180)
    phi = (180 - HCC_angle_in_deg) * np.pi / 180
    x_H, y_H = np.cos(phi) * rCH, np.sin(phi) * rCH
    R = np.array([
        [-rCC / 2, 0, 0],
        [+rCC / 2, 0, 0],
        [-rCC / 2 - x_H, +y_H, 0],
        [-rCC / 2 - x_H, -y_H, 0],
        [+rCC / 2 + x_H, +y_H * cos_twist, +y_H * sin_twist],
        [+rCC / 2 + x_H, -y_H * cos_twist, -y_H * sin_twist]
    ])
    if units == "bohr":
        R *= ANGSTROM_IN_BOHR
    elif units == "angstrom":
        pass
    else:
        raise ValueError("unkown units")
    return Z, R

def get_ion_pos_string(R):
    s = ",".join(["["+",".join([f"{x:.6f}" for x in R_]) + "]" for R_ in R])
    return "[" + s + "]"

if __name__ == '__main__':
    import ase
    import ase.visualize
    import ase.units

    # output_dir = '/home/mscherbela/runs/Ethene/references/geometries/'
    # for r_CC in 1.33 + np.linspace(-0.03, 0.03, 3):
    #     for r_CH in [1.08]:
    #         for twist in np.linspace(0, 90, 10):
    #             for HCC_angle in [121.7]:
    #                 Z, R = get_ethene_molecule(r_CC, r_CH, twist, HCC_angle)
    #                 name = f"Ethene_{r_CC:.3f}_{r_CH:.3f}_{HCC_angle:.1f}_{twist:.1f}.xyz"
    #                 save_xyz_file(output_dir + name, R, Z, name)

    for twist in [0, 30, 60, 90]:
        print(get_ion_pos_string(get_ethene_molecule(1.33, 1.08, twist, 121.7, "bohr")[1]))





