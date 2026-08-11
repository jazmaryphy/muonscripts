# %%
"""
Elk readers.
"""
import ase
import numpy as np
from typing import List
from io_tools.read import read_elk_geom, read_elk_efg

# %%
def read_geom(filename: str) -> ase.Atoms:
    """
    Read Elk geometry.

    Parameters
    ----------
    filename : str

    Returns
    -------
    ase.Atoms
    """
    return read_elk_geom(filename)


def read_efg(filename: str) -> List[np.ndarray]:
    """
    Read EFG tensors from Elk EFG.OUT.

    Returns
    -------
    list
        List of 3x3 EFG tensors.
    """
    return read_elk_efg(filename)