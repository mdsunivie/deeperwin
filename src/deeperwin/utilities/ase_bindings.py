import ase
import ase.visualize
import ase.units
import numpy as np
import ase.calculators.interface

class ASEDummyCalculator(ase.calculators.interface.Calculator):
    """
    Dummy object that acts as a calculator for ASE (atomic simulation environment) but only returns pre-computed fixed values.
    """
    def __init__(self, energy=None, forces=None, **kwargs):
        super(**kwargs)
        self.energy = energy
        self.forces = forces

    def get_forces(self, atoms):
        """
        Returns forces on nucleii
        """
        return self.forces

    def get_potential_energy(self, atoms=None, force_consistent=False):
        """
        Returns total energy of Born-Oppenheimer approximation (electrons: potential + kinetic; nucleii: potential)
        """
        self.energy

    def calculation_required(self, atoms, quantities):
        """
        Hardcoded to False, because values are already hardcoded-
        """
        return False

def _convertToASE(R, Z, energy, forces=None):
    ion_charges = np.array(Z)
    ion_positions = np.array(R)
    geometry = ase.Atoms([ase.Atom(Z, R*ase.units.Bohr) for Z,R in zip(ion_charges, ion_positions)])
    if forces is not None:
        forces = forces * ase.units.Hartree / ase.units.Bohr
    geometry.set_calculator(ASEDummyCalculator(energy, forces))
    return geometry

def getRunAsASE(df_row, scale_forces=1.0):
    """
    Takes a DeepErwin run and converts it to an ASE (atomic simulation environment) object, e.g. for visualization.

    Input units are atomic units (e.g. bohr). Output units are the ASE units (e.g. angstrom)
    Args:
        df_row (pd.Series): Row of a pd.DataFrame as returned by :meth:`deeperwin.utilities.postprocessing.loadRuns`
        scale_forces (float): Scale given forces by this factor to improve visualization (e.g. when forces are too large for pretty plotting)

    Returns:
        (ase.Atoms): Ase Atoms object
    """
    if 'forces' in df_row:
        forces = scale_forces * df_row.forces
    else:
        forces = None
    return _convertToASE(df_row.physical_ion_positions, df_row.physical_ion_charges, df_row.E_eval, forces)