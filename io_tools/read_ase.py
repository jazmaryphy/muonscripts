# %%
import ase
import numpy as np
from ase.io import read

# %%
def read_from_file(filename: str, **read_kwargs) -> ase.Atoms:
    """
    Read a structure file with ase.io.read()

    Parameters
    ----------
    filename : str
        Any structure file ASE can read (.cif, POSCAR/CONTCAR, .xyz,
        Quantum ESPRESSO .pwi/.out, ELK .in/.OUT etc.)
    **read_kwargs :
        Passed through to ase.io.read (e.g. index=-1 for a relaxation
        trajectory, format=... to force a format).

    Returns
    -------
    Atoms object(s) 
    """
    return read(filename, **read_kwargs)