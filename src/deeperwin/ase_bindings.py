import ase
import ase.visualize
import ase.units
import numpy as np
import ase.calculators.interface

def _convertToASE(R, Z, energy=None, forces=None):
    ion_charges = np.array(Z)
    ion_positions = np.array(R)
    geometry = ase.Atoms([ase.Atom(Z, R*ase.units.Bohr) for Z,R in zip(ion_charges, ion_positions)])
    # if forces is not None:
    #     forces = forces * ase.units.Hartree / ase.units.Bohr
    return geometry


if __name__ == '__main__':
    from configuration import PhysicalConfig
    physical_configs = [PhysicalConfig(name=n) for n in 'Ethene EtheneBarrier'.split()]
    geoms = [_convertToASE(phys.R, phys.Z) for phys in physical_configs]
    ase.visualize.view(geoms)
