# %%
"""
Simple ASE writing utilities.
"""
import ase
import numpy as np
from ase.io import write
from typing import Optional, Union, List

# %%
def write_structure(
    filename: str, 
    atoms: ase.Atoms, 
    magmoms: Optional[Union[List, np.ndarray]] = None, 
    **write_kwargs
):
    """Generic ASE writer with native magnetic moment support.

    Parameters
    ----------
    filename : str
        The output path of your file.
    atoms : ase.Atoms
        The ASE Atoms crystal structure object.
    magmoms : list or np.ndarray, optional
        Magnetic moments. Can be scalar values for collinear spins 
        or lists of [mx, my, mz] vectors for non-collinear structures.
    **write_kwargs
        Passed directly to ase.io.write.
    """
    atoms = atoms.copy()

    # If magnetic moments are provided, insert
    if magmoms is not None:
        if np.ndim(magmoms) == 2:
            # 3D magnetic vector moments (Non-collinear)
            atoms.set_initial_magnetic_moments(magmoms)
        else:
            # 1D magnetic moments (Collinear: spin-up/spin-down like 3.2 or -3.2)
            atoms.set_initial_magnetic_moments(magmoms)

    # Route straight to ASE write
    write(filename, atoms, **write_kwargs)

def xyz(filename: str, atoms: ase.Atoms):
    write(filename, atoms, format="xyz")


def cif(filename: str, atoms: ase.Atoms):
    write(filename, atoms, format="cif")


def mcif(filename: str, atoms: ase.Atoms, magmoms: Union[List, np.ndarray]):
    """Write structure to Magnetic CIF (mcif) format.

    Parameters
    ----------
    filename : str
    atoms : ase.Atoms
    magmoms : list or np.ndarray
        Array containing magnetic moments per atom site.
    """
    write_structure(filename, atoms, magmoms=magmoms, format="cif")


def poscar(filename: str, atoms: ase.Atoms):
    write(filename, atoms, format="vasp")