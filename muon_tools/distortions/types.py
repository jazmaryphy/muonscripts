# %%
"""
Shared type aliases, kept in one place so every module means the same
thing by "a lattice-like object" or "a muon-selection mode" instead of
repeating (and risking drift in) ad-hoc Union types everywhere.
"""

from __future__ import annotations

from typing import List, Literal, TypedDict, Union

import numpy.typing as npt
from ase import Atoms
from pymatgen.core import Lattice, Structure
from pymatgen.core.operations import SymmOp

# %%
#: Anywhere a lattice is needed, a bare Lattice or a Structure (from
#: which `.lattice` is used) are both accepted.
LatticeLike = Union[Lattice, Structure, Atoms]

#: (3,) or (N, 3) array of fractional coordinates. `npt.ArrayLike` covers
#: lists, tuples, and ndarrays alike -- callers aren't forced to pre-wrap
#: plain lists in `np.asarray`.
FracCoords = npt.ArrayLike

#: Per-site magnetic moments passed in by the caller: either (N,) scalar
#: z-moments or (N, 3) non-collinear moment vectors. See
#: `symmetry.get_structural_and_magnetic_ops` for the accepted shapes.
Moments = npt.ArrayLike

#: Which matching atom to treat as "the muon" when more than one site
#: shares `muon_label` -- see `sites.find_muon_index`.
MuonSelect = Literal["first", "last", "unique"]


class SymmetryOpsInfo(TypedDict):
    """Return type of `symmetry.get_structural_and_magnetic_ops`."""
    structural_ops: List[SymmOp]
    magnetic_ops: List[SymmOp]
    is_magnetic: bool