# %%
import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike, NDArray
from typing import Sequence, Literal, Optional, Tuple, Union, Dict, List

from ase import Atoms
from pymatgen.core import Structure

import muon_tools.utils as mutils

# %%
def get_equivalent_sites(
    fcoords: ArrayLike,
    host_lattice: Structure,
    min_distance: float = 0.5,
    symprec: float = 1e-3,
    energies: Optional[ArrayLike] = None,
    e_tol: float = 0.05,
) -> NDArray[np.float64]:
    """Generate symmetry-equivalent muon positions and remove duplicate/near-equivalent sites.

    Parameters
    ----------
    fcoords : (N, 3) array_like
        Input fractional coordinates.
    host_lattice : Structure
        Host crystal structure.
    min_distance : float
        Minimum separation (Å) used for deduplication.
    symprec : float
        Symmetry tolerance.
    energies : array_like, optional
        Site energies.
    e_tol : float
        Energy tolerance for site merging.

    Returns
    -------
    ndarray
        Unique fractional coordinates.
    """

    if isinstance(host_lattice, Atoms):
        from pymatgen.io.ase import AseAtomsAdaptor
        p_st = AseAtomsAdaptor.get_structure(host_lattice)
    else:
        p_st = host_lattice


    positions = mutils.get_equivalent_sites(
        fcoords=fcoords,
        host_lattice=p_st,
        min_distance=min_distance,
        symprec=symprec,
        energies=energies,
        e_tol=e_tol,
    )

    return positions