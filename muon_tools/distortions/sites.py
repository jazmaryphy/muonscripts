# %%
"""
Generating the orbit of equivalent sites under a set of symmetry
operations, deduplicating near-identical points, and identifying which
atom in a structure is the muon.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from muon_tools.distortions.geometry import periodic_distance_matrix
from muon_tools.distortions.types import FracCoords, LatticeLike, MuonSelect

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
            raise ValueError(f"{len(matches)} atoms found with label '{muon_label}', expected exactly 1.")
        return matches[0]
    if which == "first":
        return matches[0]
    if which == "last":
        return matches[-1]
    raise ValueError(f"Unknown value for 'which': {which!r}. Expected 'first', 'last', or 'unique'.")

# %%
def prune_close_positions(
    frac_positions: FracCoords, 
    lattice: LatticeLike, 
    min_distance: float,
    energies: Optional[npt.ArrayLike] = None,
    e_tol: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster/deduplicate a list of fractional positions that are
    physically the same site (within `min_distance`, real periodic
    distance), optionally also requiring energies to agree within `e_tol`.

    For merging positions from INDEPENDENT sources -- e.g. several
    separate DFT relaxations that may have converged to physically the
    same site without landing on numerically identical coordinates. Two
    sites can be spatially close yet genuinely different physical states
    (e.g. a true minimum next to a nearby saddle point); only positions
    both close in space AND close in energy get merged.

    Vectorized: the full pairwise distance matrix is computed once
    (`periodic_distance_matrix`) rather than one distance call per pair
    in a nested Python loop.

    Parameters
    ----------
    frac_positions : (N, 3) array
    lattice : pymatgen.core.Lattice or pymatgen.core.Structure
    min_distance : float [Angstrom]
        Positions closer than this are CANDIDATES for merging.
    energies : (N,) array or None
        If given, candidates are only merged if energies also agree
        within `e_tol`. If None, merging is decided by distance alone.
    e_tol : float [eV], default=0.05

    Returns
    -------
    keep_mask : (N,) bool array
        True for the representative position kept for each cluster (the
        first occurrence, in input order).
    groups : (N,) int array
        `groups[i]` is the index of the representative `i` was merged
        into (`groups[i] == i` if `i` is itself a representative).
    """
    frac_positions = np.asarray(frac_positions, dtype=float)
    n = len(frac_positions)

    dmat = periodic_distance_matrix(lattice, frac_positions)
    close = dmat < min_distance
    if energies is not None:
        energies = np.asarray(energies, dtype=float)
        close &= np.abs(energies[:, None] - energies[None, :]) < e_tol

    groups = np.arange(n)
    keep_mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep_mask[i]:
            continue
        later_matches = np.where(close[i, i + 1:] & keep_mask[i + 1:])[0] + i + 1
        groups[later_matches] = groups[i]
        keep_mask[later_matches] = False

    return keep_mask, groups

# %%
def get_equivalent_sites(
    mu_frac: FracCoords, 
    sym_ops: Sequence[SymmOp], 
    tol: float = 1e-3,
    lattice: Optional[LatticeLike] = None,
    min_distance: Optional[float] = None
) -> np.ndarray:
    """Orbit of fractional position `mu_frac` under symmetry operations
    `ops`, deduplicated (periodic boundary conditions respected).

    Parameters
    ----------
    mu_frac : (3,) array-like
        Fractional coordinates of the site whose orbit (symmetry images)
        is being generated -- typically the relaxed muon position.
    sym_ops : list of pymatgen.core.operations.SymmOp
        Symmetry operations to apply, e.g. `structural_ops` or
        `magnetic_ops` from `get_structural_and_magnetic_ops`. Each `op`
        is applied once via `op.operate(mu_frac)
    tol : float, default=1e-3
        FRACTIONAL-coordinate tolerance for the first, cheap dedup pass.
        Not a physically uniform distance for an anisotropic cell -- fine
        for catching exact/near-exact symmetry-image duplicates, but see
        `lattice`/`min_distance` for the physically-correct cleanup that
        should generally follow it.
    lattice : pymatgen.core.Lattice or pymatgen.core.Structure, optional
        If given together with `min_distance`, a further cleanup pass
        runs via `prune_close_positions`, using the true periodic
        (minimum-image) Cartesian distance -- matters for anisotropic
        cells, where a fixed fractional tolerance corresponds to very
        different real distances along different lattice vectors.
    min_distance : float [Angstrom], optional
        Real-space merge threshold for the `lattice`-based cleanup pass.

    Returns
    -------
    ndarray, shape (n_unique, 3)
        Unique equivalent fractional positions, including `mu_frac` itself.
    """
    mu_frac = np.asarray(mu_frac, dtype=float)
    candidates = np.vstack([op.operate(mu_frac) % 1.0 for op in sym_ops])

    n = len(candidates)
    diff = (candidates[:, None, :] - candidates[None, :, :] + 0.5) % 1.0 - 0.5
    close = np.all(np.abs(diff) < tol, axis=-1)  # (n, n) symmetric bool

    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        keep[np.where(close[i, i + 1:])[0] + i + 1] = False
    sites = candidates[keep]

    if lattice is not None and min_distance is not None and len(sites) > 1:
        keep_mask, _groups = prune_close_positions(
            sites, 
            lattice, 
            min_distance=min_distance, 
            energies=None
        )
        sites = sites[keep_mask]

    return sites