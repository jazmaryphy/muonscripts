# %%
"""
Quantum ESPRESSO readers.
"""
import numpy as np
from io_tools import qe
from constants import constants
from io_tools.read_ase import read_from_file

# %%
def read_in(filename: str):
    """
    Read Quantum ESPRESSO input file.

    Parameters
    ----------
    filename : str

    Returns
    -------
    ase.Atoms
    """
    return read_from_file(filename, format="espresso-in")


def read_out(filename: str, index: int=-1):
    """
    Read Quantum ESPRESSO output structure.

    Parameters
    ----------
    filename : str
    index : int
        Structure index.
        Defaults to final structure.

    Returns
    -------
    ase.Atoms
    """
    return read_from_file(filename, format="espresso-out", index=index)

def parse_in(filename: str):
    """Parses a QE input file and returns a dict with 'namelists' and 'lines'."""
    with open(filename) as f:
        input_string = f.read()

    return {
        "namelists": qe.read_qe_namelists(input_string.lower()),
        "lines": input_string.splitlines()
    }

def read_efg(filename: str):
    """
    Read EFG tensors from Quantum ESPRESSO.

    Parameters
    ----------
    filename : str

    Returns
    -------
    list
        List of 3x3 tensors.
    """
    return qe.read_efg(filename)


def read_xsf_datagrid(filename: str):
    """
    Read a 3D data grid from a Quantum ESPRESSO-produced .xsf file.

    Thin wrapper around `qe.read_qe_xsf_datagrid` -- see that function
    for the full parsing, units, and error-handling documentation. Kept
    as a shorter alias for callers that don't need QE-specific context
    in the name.

    Args:
        filename (str): Path to the .xsf file.

    Returns:
        ndarray with shape (nx, ny, nz): Grid data, Fortran-ordered.
            See `qe.read_qe_xsf_datagrid` for unit conventions.

    Raises:
        OSError, KeyError, ValueError: See `qe.read_qe_xsf_datagrid`.
    """
    return qe.read_qe_xsf_datagrid(filename)