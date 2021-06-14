from deeperwin.utilities.erwinConfiguration import PhysicalConfig
import ase
import ase.visualize
import ase.units
import ase.io
import itertools
import numpy as np

def _build_triangle_from_3_sides(d, phi=0):
    """
    Get corners of a triangle that is defined by the length of 3 sides and a global orientation angle
    Args:
        d (iterable): List of length 3, containing the 3 side lengths of the triangle
        phi (float): Orientation of triangle in radiants

    Returns:
        (np.array): Array of shape [3x2]

    """
    cos_theta = (d[0]**2 + d[2]**2 - d[1]**2)/ (2*d[0]*d[2])
    y2 = cos_theta * d[2]
    x2 = np.sqrt(d[2]**2 - y2**2)
    R_raw = np.array([[0,0],
                  [0,d[0]],
                  [x2,y2]])
    R_centered = R_raw - np.mean(R_raw, axis=0)
    # Rotate to position where R[0] is at the bottom
    cos_phi = -R_centered[0,1]/np.linalg.norm(R_centered[0])
    sin_phi = -R_centered[0,0]/np.linalg.norm(R_centered[0])
    rot_mat = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
    # Rotate to specified angle phi
    rot_mat = rot_mat @ np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    R_final = R_centered @ rot_mat
    return R_final

def get_min_distance(R):
    """
    Return the closest distance (euclidean norm) between a set of coordinates R
    Args:
        R (np.array): Coordinate array of shape [Nxd], where N is the number of particles and d is the dimension (e.g. 2 or 3)

    Returns:
        (float): Shortest distance r_ij
    """
    R = np.array(R)
    dist = R[:, np.newaxis, :] - R[np.newaxis, :, :]
    dist = np.linalg.norm(dist, axis=-1) + np.diag(np.nan * np.ones(len(R)))
    return np.nanmin(dist)

class Geometry:
    """
    Template class to represent a nuclear geometry. Subclasses implement parametrizations of specific molecules.
    """
    def __init__(self):
        self.n_electrons = None
        self.electron_ion_mapping = None
        self.ion_positions = np.zeros([0,3])
        self.ion_charges = np.zeros(0)
        self.name = ""
        self.comment = ""

    @property
    def d_min(self):
        """
        Closest distance between atoms within this geometry
        """
        return np.nanmin(self.get_distance_matrix(diag_nan=True))

    def get_distance_matrix(self, diag_nan=False):
        """
        Return the [NxN] distance matrix between each pair of particles
        Args:
            diag_nan (bool): If True, replace the 0s on the main diagonal with nan. This enables finding shortest distances using nanmin

        Returns:
            (np.array): [NxN] distance matrix

        """
        R = np.array(self.ion_positions)
        dist = R[:,np.newaxis,:] - R[np.newaxis,:,:]
        dist_matrix = np.linalg.norm(dist, axis=-1)
        if diag_nan:
            dist_matrix += np.diag(np.nan * np.ones(len(self.ion_positions)))
        return dist_matrix

    @property
    def physical_config(self):
        """
        Builds a PhysicalConfig object to be used for a DeepErwin calculation based on the parameters of this geometry

        Returns:
            (PhysicalConfig): Config to be used for 'config.physical'
        """
        config = PhysicalConfig()
        config.ion_charges = [int(Z) for Z in self.ion_charges]
        config.ion_positions = self.ion_pos_list
        config.n_electrons = self.n_electrons
        config.electron_ion_mapping = self.electron_ion_mapping
        config.n_spin_up = (self.n_electrons+1)//2
        config.name = self.name
        config.energy_baseline = 0
        return config

    def to_ASE_Atoms(self):
        """
        Convertes a geometry to an ASE (atomic simulation environment) object, which allows fast visualization.

        Returns:
            (ase.Atoms): Geometry as ASE object
        """
        geometry = ase.Atoms([ase.Atom(Z, R * ase.units.Bohr) for Z, R in zip(self.ion_charges, self.ion_positions)])
        return geometry

    def to_xyz_file(self, fname, comment=None):
        """
        Writes a geometry to an XYZ file, which can be read by most quantum chemistry codes (e.g. molpro)

        Assumes that the internal untis are bohr, and the xyz file will be written in angstrom.
        Args:
            fname (str): Filename to write the file to. Should the file exist, it will be overwritten.
            comment (str): Comment string to put in the xyz-file comment line. Must not contain newline char

        Returns:
            None
        """
        PERIODIC_TABLE = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne'}
        BOHR_TO_ANGSTROM = 0.5291772105638411
        with open(fname, 'w') as f:
            f.write(f"{len(self.ion_charges)}\n")
            if comment is None:
                f.write("\n")
            else:
                assert '\n' not in comment
                f.write(f"# {comment}\n")
            for Z, pos in zip(self.ion_charges, self.ion_positions):
                pos = np.array(pos) * BOHR_TO_ANGSTROM
                f.write(f"{PERIODIC_TABLE[Z]} {pos[0]:3.6f} {pos[1]:3.6f} {pos[2]:3.6f}\n")

    @property
    def ion_pos_list(self):
        """
        Get coordinates as a list of lists. Useful for JSON config-files that do not support np.arrays.

        Returns:
            (list): list of lists of length 3; e.g. the result can be indexed by result[i][j] with i looping over atoms and j looping of x,y,z
        """
        return [[float(x) for x in R] for R in self.ion_positions]

    @property
    def pos_string(self):
        """
        Get coordinates as fixed precision string, containing a list of lists.
        Returns:
            (str): Position string, e.g. '[[1.000000, 2.000000, 3.000000], [4.000000, 5.000000, 6.000000]]'
        """
        coords = []
        for R in self.ion_positions:
            coords.append("[" + ",".join([f"{x:.6f}" for x in R]) + "]")
        return "[" + ", ".join(coords) + "]"

    def get_pyscf_geometry_string(self):
        """
        Get coordinates and nuclear charges as a string that can be pased to pyscf for initialization of geometry.

        ATTENTION: This geometry string is given in Bohr-units, e.g. when passing it to pyscf make sure that Mole.units is also in Bohr
        Returns:
            (str): Geometry string that can be passed to pyscf.gto.Mole.atom
        """
        s = ""
        for Z, R in zip(self.ion_charges, self.ion_positions):
            s += f"{Z} {R[0]:.6f} {R[1]:.6f} {R[2]:.6f};"
        return s[:-1] #remove last semicolon

    def calculate_HF_energy(self, spin):
        """
        Calculates the Hartree-Fock energy of a geometry using pyscf and the 6-311G basis set.

        Args:
            spin: Total spin of the molecule

        Returns:
            (float): Hartree-Fock energy
        """
        import pyscf
        molecule = pyscf.gto.Mole()
        molecule.verbose = 0
        molecule.unit = 'bohr'
        molecule.atom = self.get_pyscf_geometry_string()
        molecule.basis = '6-311G'
        molecule.charge = sum(self.ion_charges) - self.n_electrons
        molecule.spin = spin
        molecule.build()

        hf = pyscf.scf.RHF(molecule)
        E_hf = hf.kernel()
        return E_hf

class GeometryH4Plus(Geometry):
    """
    Subclass that represents a planar H4+ molecule.
    """
    def __init__(self, d01=1.7760, d02=1.5813, d12=1.7760, phi=0, r3=3.6893):
        """
        This molecule is parametrized by the 3 sides of the H3 triangle and the position of the 4th Hydrogen atom (orientation phi in degrees), distance from center of triangle r3.
        """
        super().__init__()
        self.ion_charges = np.ones(4, dtype=np.int)
        self.n_electrons = 3
        self.electron_ion_mapping = [1,3,2]
        self.name = "H4plus"
        self._set_geometry(d01, d02, d12, phi, r3)

    @classmethod
    def get_minimum_geometry(cls, rotation=0):
        """
        Return the geometry corresponding to the global energtic minimum for H4+.

        Data taken from J. Chem. Phys. 129, 034303 (2008); https://doi.org/10.1063/1.2953571
        Args:
            rotation (int): Integer from 0-2, describing which of the 3 symmetry equivalent rotations to return
        Returns:
            (Geometry): H4+ geometry
        """
        a = 1.7760
        b = 1.5813
        r = 3.6893
        if rotation == 0:
            d = [a,b,a]
            phi = 0
            return cls(*d, phi, r)
        elif rotation == 1:
            d = [a,a,b]
            phi = 123.84028290261426
            return cls(*d, phi, r)
        elif rotation == 2:
            d = [b,a,a]
            phi = 236.15971709738574
            return cls(*d, phi, r)

    @classmethod
    # J. Chem. Phys. 129, 034303 (2008); https://doi.org/10.1063/1.2953571
    def get_saddlepoint1_geometry(cls, d12=1.6454, d23=1.6380, d24=5.1877):
        """
        Return the geometry corresponding to the first saddlepoint of H4+

        Data taken from J. Chem. Phys. 129, 034303 (2008); https://doi.org/10.1063/1.2953571
        Returns:
            (Geometry): H4+ geometry
        """
        d = [1.6380, 1.6454, 1.6454]
        r = 5.5983
        phi = 59.85089209921057 # not exactly 60°, because triangle is distorted
        return cls(*d, phi, r)

    def _set_geometry(self, d01, d02, d12, phi, r3):
        R_H3 = _build_triangle_from_3_sides([d01, d02, d12], phi*np.pi/180)
        R_H = np.array([[0, -r3]])
        R = np.concatenate([R_H3, R_H], axis=0)
        R = np.concatenate([R, np.zeros(4)[:,np.newaxis]], axis=1)
        self.ion_positions = R

class GeometryH4PlusOutOfPlane(Geometry):
    """
    Geometry of a (potentially) non-planar H4+ molecule.

    Parametrized as an equilateral triangle with side length d, and the spherical coordinates of the 4th hydrogen atom (r, phi, theta).
    """
    def __init__(self, d, r, phi, theta):
        super().__init__()
        self.ion_charges = np.ones(4, dtype=np.int)
        self.n_electrons = 3
        self.electron_ion_mapping = [1,3,2]
        self.name = "H4plus"
        self._set_geometry(d, r, phi, theta)

    def _set_geometry(self, d, r, phi, theta):
        angles = np.array([270, 150, 30]) - phi
        angles = np.pi * angles / 180
        theta = np.pi * theta / 180
        r_triangle = d / np.sqrt(3)

        R = np.zeros([4,3])
        R[:3,0] = r_triangle * np.cos(angles) # triangle x
        R[:3,1] = r_triangle * np.sin(angles) # triangle y
        R[3, 1] = -r * np.cos(theta) # extra hydrogen y
        R[3, 2] = r * np.sin(theta) # extra hydrogen z
        self.ion_positions = R
        self.comment = f"d={d:.6f}, r={r:.6f}, phi={phi:.6f}, d={theta:.6f}"

    @classmethod
    def get_PES_dataset(cls):
        """
        Return a standardized, fixed dataset of various H4+ geometries.

        The dataset is given by an equidistant grid of parametrizations that are then filtered by the following criteria:

            * No 2 geometries should be identical/symmetry equivalent
            * Atoms should never by closer than 1.5 Bohr

        Returns:
            (list): List of H4+ geometries

        """
        d_values = np.linspace(1.55, 1.7, 4)
        r_values = np.linspace(3.0, 6.0, 7)
        phi_values = np.linspace(0, 60, 3)
        theta_values = np.linspace(0, 90, 2)

        included_distance_matrix = []
        geometries = []
        for d, r, phi, theta in itertools.product(d_values, r_values, phi_values, theta_values):
            geom = cls(d, r, phi, theta)
            geom.is_ref = True
            d_mat = geom.get_distance_matrix(diag_nan=True)
            if len(included_distance_matrix) > 0:
                max_dist = np.max(np.nanmax(d_mat - np.array(included_distance_matrix), axis=2), axis=1)
                if np.min(max_dist) < 1e-3:
                    continue
            d_min = np.nanmin(d_mat)
            if d_min < 1.5:
                continue
            geometries.append(geom)
            included_distance_matrix.append(d_mat)
        return geometries

    @classmethod
    def get_PES_prediction_dataset(cls):
        """
        Get at hardcoded dataset of H4+ geometries that are not part of the dataset returned by 'get_PES_dataset'.

        d_values = np.linspace(1.55, 1.7, 4)
        r_values = np.linspace(3.0, 5.5, 7)
        phi_values = np.linspace(0, 60, 5)
        theta_values = np.linspace(0, 90, 5)
        d_min = 1.5
        d_max = 4.0
        min_dist_to_reference_set = 0.3

        Returns:
            (list): List of 16 H4+ geometries

        """

        parameters = [(1.55, 3.0, 0.0, 45.0),
         (1.55, 3.0, 0.0, 67.5),
         (1.55, 3.0, 45.0, 45.0),
         (1.55, 3.4166666666666665, 0.0, 67.5),
         (1.55, 3.4166666666666665, 15.0, 45.0),
         (1.55, 3.4166666666666665, 45.0, 45.0),
         (1.55, 3.8333333333333335, 0.0, 67.5),
         (1.55, 3.8333333333333335, 15.0, 22.5),
         (1.55, 3.8333333333333335, 45.0, 0.0),
         (1.55, 3.8333333333333335, 60.0, 45.0),
         (1.55, 4.25, 0.0, 45.0),
         (1.55, 4.25, 0.0, 67.5),
         (1.55, 4.25, 45.0, 0.0),
         (1.6, 4.25, 15.0, 0.0),
         (1.65, 3.0, 60.0, 67.5),
         (1.65, 4.666666666666667, 15.0, 0.0)]
        geometries = [cls(*p) for p in parameters]
        for g in geometries:
            g.is_ref = False
        return geometries


class GeometryH3Plus(Geometry):
    """
    H3+ geometry (i.e. a triangle).

    Parametrized by 3 integers as described in J. Chem. Phys. 108, 2831 (1998); https://doi.org/10.1063/1.475702
    """
    def __init__(self, na=0, nx=0, ny=0):
        super().__init__()
        self.n_tuple = (na,nx,ny)
        self.ion_charges = np.ones(3, dtype=np.int)
        self.n_electrons = 2
        self.electron_ion_mapping = [0,1]
        self._set_geometry(na, nx, ny)
        self.name = "H3plus"
        self.comment = f"na={int(na)}, nx={int(nx)}, ny={int(ny)}"

    # J. Chem. Phys. 108, 2831 (1998); https://doi.org/10.1063/1.475702
    def _set_geometry(self, na=0, nx=0, ny=0):
        s3 = 1.0/np.sqrt(3)
        s6 = 1.0/np.sqrt(6)
        s2 = 1.0/np.sqrt(2)
        A = np.array([[s3, s3, s3],[2*s6, -s6, -s6],[0,-s2, s2]]) / 0.15
        R_tilde = np.linalg.solve(A, np.array([na, nx, ny]))

        beta = 1.3
        R_ref = 1.65
        R12, R13, R23 = R_ref * (1-np.log(1-R_tilde*beta)/beta)
        x3 = (R13**2 + R12**2 - R23**2)/(2*R12)
        y3 = np.sqrt(R13**2 - x3**2)
        self.ion_positions = np.array([[0,0,0],
                                      [R12,0,0],
                                      [x3,y3,0]])

    @classmethod
    def get_PES_dataset(cls):
       """
       Get a fixed dataset of H3+ geometries.

       The dataset corresponds to a grid across the parametrization space, throwing out all geometries that are not in the original dataset [1] and where the minimum distance between atoms <= 1.0 Bohr or >= 3.0 Bohr
       Returns:
            (list): List of H3+ geometries
       """
       dmin = 1.0
       dmax = 3.0
       n_tuples_ref = [(-4, 0, 0), (-3, 0, 0), (-2, 0, 0), (-1, 0, 0), (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0), (0, -4, 0), (0, -3, 0), (0, -2, 0), (0, -1, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0), (-4, -1, 0), (-4, 1, 0), (-3, -2, 0), (-3, -1, 0), (-3, 1, 0), (-3, 2, 0), (-2, -3, 0), (-2, -2, 0), (-2, -1, 0), (-2, 1, 0), (-2, 2, 0), (-2, 3, 0), (-1, -3, 0), (-1, -2, 0), (-1, -1, 0), (-1, 1, 0), (-1, 2, 0), (-1, 3, 0), (1, -4, 0), (1, -3, 0), (1, -2, 0), (1, -1, 0), (1, 1, 0), (1, 2, 0), (1, 3, 0), (2, -4, 0), (2, -3, 0), (2, -2, 0), (2, -1, 0), (2, 1, 0), (2, 2, 0), (2, 3, 0), (3, -4, 0), (3, -3, 0), (3, -2, 0), (3, -1, 0), (3, 1, 0), (3, 2, 0), (4, -3, 0), (4, -2, 0), (4, -1, 0), (4, 1, 0), (4, 2, 0), (5, -1, 0), (5, 1, 0), (-2, 0, 2), (0, 0, 2), (2, 0, 2), (-2, 0, 3), (0, 0, 3), (2, 0, 3), (0, 0, 4)]

       geoms = [cls(na,nx,ny) for na in range(-4,5+1) for nx in range(-4,3+1) for ny in range(0,4+1)]
       is_ref = [g.n_tuple in n_tuples_ref for g in geoms]
       is_good_spacing = [(np.nanmin(g.get_distance_matrix(True)) >= dmin) and (np.max(np.nanmin(g.get_distance_matrix(True), axis=1)) <= dmax) for g in geoms]
       geoms = [g for g,r,s in zip(geoms, is_ref, is_good_spacing) if r or s]
       is_ref = [r for r,s in zip(is_ref,is_good_spacing) if r or s]
       for i,g in enumerate(geoms):
           g.wf_nr = i
           g.is_ref = is_ref[i]
       return geoms


class GeometryH6(Geometry):
    """
    Geometry of a linear H6 chain.
    """
    def __init__(self, ion_pos):
        super().__init__()
        self.n_electrons = 6
        self.name = "HChain6"
        n = 6
        self.ion_charges = [1] * n
        self.ion_positions = ion_pos
        self.n_spin_up = (n + 1) // 2
        self.electron_ion_mapping = list(range(0, n, 2)) + list(range(1, n, 2))  # initialize antiferromagnetically

    @classmethod
    def get_PES_dataset(cls):
        """
        Return a fixed dataset, given by an equidistant grid of the parameters.

        Returns:
            (list): List of 49 H6 geometries
        """
        spacings = [1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
        shifts = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
        dmin = 1

        tuples = [(round(a, 2), round(x, 2)) for a in np.linspace(1.2, 1.8, 7) for x in np.linspace(1.6, 2.8, 7)]
        old_geom = [(round(x - 2 * shift, 2), round(x + 2 * shift, 2)) for x in spacings for shift in shifts if
                       dmin <= round(x - 2 * shift, 2) and (round(x - 2 * shift, 2), round(x + 2 * shift, 2)) not in tuples]
        print(old_geom)
        geom = [cls([[0, 0, 0], [a, 0, 0], [a + x, 0, 0], [2 * a + x, 0, 0], [2 * a + 2 * x, 0, 0], [3 * a + 2 * x, 0, 0]]) for (a, x) in old_geom]
        return geom


class GeometryH10(Geometry):
    """
    Geometry of a linear H10 chain.
    """
    def __init__(self, ion_pos):
        super().__init__()
        self.n_electrons = 10
        self.name = "HChain6"
        n = 10
        self.ion_charges = [1] * n
        self.ion_positions = ion_pos
        self.n_spin_up = (n + 1) // 2
        self.electron_ion_mapping = list(range(0, n, 2)) + list(range(1, n, 2))  # initialize antiferromagnetically

    @classmethod
    def get_PES_dataset(cls):
        """
        Return a fixed dataset, given by an equidistant grid of the parameters.

        Returns:
            (list): List of 49 H10 geometries
        """
        spacings = [1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
        shifts = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
        dmin = 1

        tuples = [(round(a, 2), round(x, 2)) for a in np.linspace(1.2, 1.8, 7) for x in np.linspace(1.6, 2.8, 7)]
        old_geom = [(round(x - 2 * shift, 2), round(x + 2 * shift, 2)) for x in spacings for shift in shifts if
                    dmin <= round(x - 2 * shift, 2) and (
                    round(x - 2 * shift, 2), round(x + 2 * shift, 2)) not in tuples]

        # geom = [cls([[0, 0, 0], [a, 0, 0], [2* a, 0, 0], [3 * a, 0, 0], [4 * a , 0, 0], [5 * a , 0, 0],
        #               [6 * a, 0, 0], [7 * a, 0, 0], [8 * a, 0, 0], [9 * a , 0, 0]] ) for a in [1.4, 1.6, 1.8, 2.0, 2.2, 2.4]]
        geom = [cls([[0, 0, 0], [a, 0, 0], [2 * a, 0, 0], [3 * a, 0, 0], [4 * a, 0, 0], [5 * a, 0, 0],
                     [6 * a, 0, 0], [7 * a, 0, 0], [8 * a, 0, 0], [9 * a, 0, 0]]) for a in
                [1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 3.6]]
        # geom = [cls([[0, 0, 0], [a, 0, 0], [a + x, 0, 0], [2 * a + x, 0, 0], [2 * a + 2 * x, 0, 0], [3 * a + 2 * x, 0, 0], [3 * a + 3 * x, 0, 0]
        #              , [4 * a + 3 * x, 0, 0], [4 * a + 4 * x, 0, 0], [5 * a + 4 * x, 0, 0]]) for (a, x) in old_geom]

        return geom

if __name__ == '__main__':
    import numpy as np

    geoms = GeometryH10.get_PES_dataset()
    print(len(geoms))
    for i, g in enumerate(geoms):
        print("\n")
        print(i)
        print(g.ion_pos_list)
    # d_values = np.linspace(1.55, 1.7, 4)
    # r_values = np.linspace(3.0, 5.5, 7)
    # phi_values = np.linspace(0, 60, 5)
    # theta_values = np.linspace(0, 90, 5)
    #
    # included_distance_matrix = [g.get_distance_matrix(diag_nan=True) for g in GeometryH4PlusOutOfPlane.get_PES_dataset()]
    # geometries = []
    # parameters = []
    # for params in itertools.product(d_values, r_values, phi_values, theta_values):
    #     geom = GeometryH4PlusOutOfPlane(*params)
    #     d_mat = geom.get_distance_matrix(diag_nan=True)
    #     d_min = np.nanmin(d_mat)
    #     if d_min < 1.5:
    #         continue
    #     if len(included_distance_matrix) > 0:
    #         max_dist = np.max(np.nanmax(np.abs(d_mat - np.array(included_distance_matrix)), axis=2), axis=1)
    #         closest_dist = np.min(max_dist)
    #         if closest_dist < 0.3:
    #             continue
    #     d_max = np.max(np.nanmin(d_mat, axis=1))
    #     if d_max > 4:
    #         continue
    #     print(d_max)
    #     geometries.append(geom)
    #     included_distance_matrix.append(d_mat)
    #     parameters.append(params)
    # print(len(geometries))
    #
    # ase_geoms = [g.to_ASE_Atoms() for g in geometries]
    # ase.visualize.view(ase_geoms)


    #
    # ind_in_ref = set([k for k,g in enumerate(GeometryH3Plus.get_PES_dataset()) if g.is_ref])
    # restart_set = [k for k in range(0, 183, 5) if k not in ind_in_ref]
    # print(restart_set)


    # print([g.ion_pos_list for g in GeometryH3Plus.get_PES_dataset() if not g.is_ref][3])
    # geoms = GeometryH3Plus.get_PES_dataset()
    # df = pd.read_csv('/users/mscherbela/runs/xyz_reference_files/H4plus/H4plus_PES/HF_energies.csv', delimiter=';')





