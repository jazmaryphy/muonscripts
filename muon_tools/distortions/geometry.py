# %%
"""
Periodic-boundary geometry helpers.

Nothing in this file knows what a muon is -- it's pure "fractional
coordinates on a periodic lattice" plumbing, used by every other module.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pymatgen.core import Structure, Lattice

from muon_tools.distortions.types import FracCoords, LatticeLike

# %%
def frac_periodic_close(a: FracCoords, b: FracCoords, tol: float) -> bool:
    """True if fractional coordinates `a` and `b` are the same point up to
    a lattice translation (periodic boundary conditions), within `tol`."""
    diff = (np.asarray(a) - np.asarray(b) + 0.5) % 1.0 - 0.5
    return np.all(np.abs(diff) < tol)


def as_lattice(lattice_or_structure: LatticeLike) -> Lattice:
    """Accept either a `Lattice` or a `Structure` (uses its `.lattice`)."""
    return lattice_or_structure.lattice if hasattr(lattice_or_structure, "lattice") else lattice_or_structure


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
def build_matching_supercell(primitive_structure: Structure, target_structure: Structure) -> Structure:
    """Build a supercell of `primitive_structure` matching the lattice of
    `target_structure` (e.g. a DFT relaxation supercell).

    Symmetry operations are basis-dependent: a primitive cell's
    operations don't apply directly to supercell fractional coordinates.
    Use this when your solved host is a primitive cell but your
    relaxations are supercells, so you can pass a properly matching
    `host_structure` to `generate_equivalent_muon_structures`.

    Raises
    ------
    ValueError
        If `target_structure`'s lattice is not an (approximately) integer
        supercell of `primitive_structure`'s.
    """
    matrix = np.dot(
        target_structure.lattice.matrix, np.linalg.inv(primitive_structure.lattice.matrix)
    )
    int_matrix = np.rint(matrix).astype(int)
    if not np.allclose(matrix, int_matrix, atol=1e-4):
        raise ValueError(
            "target_structure's lattice is not an (approximately) integer "
            "supercell of primitive_structure's lattice."
        )
    supercell = primitive_structure.copy()
    supercell.make_supercell(int_matrix)
    return supercell