# %%
from __future__ import annotations

from typing import Optional, Sequence, Tuple
from typing import List, Literal, TypedDict, Union

import numpy as np
import numpy.typing as npt
from ase import Atoms
from pymatgen.core import Lattice, Structure

# %%
#: Anywhere a lattice is needed, a bare Lattice or a Structure (from
#: which `.lattice` is used) are both accepted.
LatticeLike = Union[Lattice, Structure, Atoms]

#: (3,) or (N, 3) array of fractional coordinates. `npt.ArrayLike` covers
#: lists, tuples, and ndarrays alike -- callers aren't forced to pre-wrap
#: plain lists in `np.asarray`.
FracCoords = npt.ArrayLike

#: Which matching atom to treat as "the muon" when more than one site
#: shares `muon_label` -- see `sites.find_muon_index`.
MuonSelect = Literal["first", "last", "unique"]

# %%
def as_lattice(lattice_or_structure: LatticeLike) -> Lattice:
    """
    Accept a `Lattice`, PyMatGen `Structure`, or ASE `Atoms` 
    and return a PyMatGen `Lattice`.
    """

    # 1. PyMatGen Structure or object with .lattice
    if hasattr(lattice_or_structure, "lattice") and isinstance(
        lattice_or_structure.lattice, Lattice
    ):
        return lattice_or_structure.lattice

    # 2. Direct PyMatGen Lattice
    if isinstance(lattice_or_structure, Lattice):
        return lattice_or_structure

    # 3. ASE Atoms object
    if isinstance(lattice_or_structure, Atoms):
        return Lattice(lattice_or_structure.cell.array)

    # 4. Fail fast with a clear error
    raise TypeError(
        f"Expected a Lattice, Structure, or Atoms object, "
        f"got {type(lattice_or_structure).__name__}"
    )


def periodic_distance_matrix(
    lattice_or_structure: LatticeLike, 
    frac_a: FracCoords, 
    frac_b: Optional[FracCoords] = None
) -> np.ndarray:
    """Vectorized true (minimum-image) periodic distance matrix between two
    sets of fractional coordinates, in Angstrom.

    `frac_b=None` compares `frac_a` against itself. This wraps
    `Lattice.get_all_distances`, a single vectorized NumPy call -- the key
    thing that keeps deduplication/pruning fast: it replaces what would
    otherwise be one Python-level distance call per pair of points.
    """
    lat = as_lattice(lattice_or_structure)
    frac_a = np.atleast_2d(frac_a)
    frac_b = frac_a if frac_b is None else np.atleast_2d(frac_b)
    return lat.get_all_distances(frac_a, frac_b)

# %%
def find_muon_index(
    structure: Structure, 
    muon_label: str = "H", 
    which: MuonSelect = "last"
) -> Optional[int]:
    """Index of the muon site in `structure`, identified by species symbol.

    Parameters
    ----------
    which : {"first", "last", "unique"}, default="last"
        'last' matches the convention of appending the muon as an extra
        atom; 'unique' raises if there isn't exactly one match.

    Returns
    -------
    int or None
        None if `muon_label` doesn't appear in `structure` at all.
    """
    matches = [i for i, sp in enumerate(structure.species) if str(sp) == muon_label]
    if not matches:
        return None
    if which == "unique":
        if len(matches) > 1:
            raise ValueError(
                f"{len(matches)} atoms found with label "
                f"'{muon_label}', expected exactly 1."
                )
        return matches[0]
    if which == "first":
        return matches[0]
    if which == "last":
        return matches[-1]
    raise ValueError(
        f"Unknown value for 'which': {which!r}. "
        f"Expected 'first', 'last', or 'unique'."
        )