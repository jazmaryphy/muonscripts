# %%
"""
Structural and magnetic symmetry operations of the host crystal.

pymatgen's `SpacegroupAnalyzer` does NOT account for magnetic moments even
if a 'magmom' site property is attached -- it returns identical operations
with or without magmom. So when a magnetic structure is involved, this
module explicitly filters the structural operations down to the subset
that also preserves the magnetic arrangement, by transforming each site's
moment as a pseudovector (m' = det(R) * R @ m -- this picks up an extra
sign flip under improper operations that a normal position vector does
not) and checking it against the moment actually present at the
symmetry-mapped destination atom.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from pymatgen.core import Structure
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from muon_tools.distortions.types import Moments, SymmetryOpsInfo

# %%
def _op_preserves_magnetism(
    op: SymmOp, 
    frac_coords: np.ndarray, 
    magmom: np.ndarray, 
    tol: float = 1e-3
) -> bool:
    """Whether symmetry operation `op` leaves the magnetic arrangement
    `magmom` (per-site (3,) vectors) invariant.

    Vectorized: transforms all sites at once with `operate_multi`, then
    matches each transformed site to its image via a single vectorized
    minimum-image fractional-distance comparison (an (n, n, 3) array),
    instead of an O(n^2) nested Python loop with one closeness check per
    pair of atoms.
    """
    n = len(frac_coords)
    new_p = op.operate_multi(frac_coords) % 1.0

    diff = (new_p[:, None, :] - frac_coords[None, :, :] + 0.5) % 1.0 - 0.5  # (n, n, 3)
    max_abs_diff = np.max(np.abs(diff), axis=-1)  # (n, n)
    match_idx = np.argmin(max_abs_diff, axis=1)
    if not np.all(max_abs_diff[np.arange(n), match_idx] < tol):
        return False  # shouldn't happen for a genuine structural symm op

    R = op.rotation_matrix
    det_r = np.linalg.det(R)
    transformed_m = det_r * (R @ magmom.T).T  # pseudovector transform, all moments at once
    return np.allclose(transformed_m, magmom[match_idx], atol=tol)

# %%
def get_structural_and_magnetic_ops(
    structure: Structure,
    magmom: Optional[Moments] = None,
    symprec: float = 1e-3,
    angle_tolerance: float = 5.0,
) -> SymmetryOpsInfo:
    """Structural (full) and magnetic (moment-preserving subset) symmetry
    operations of `structure`.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        The HOST lattice, WITHOUT the muon -- symmetry should be evaluated
        on the pristine structure, not a muon-perturbed one (the muon
        itself locally breaks symmetry around the site you're trying to
        classify).
    magmom : array-like or None, default=None
        Per-site magnetic moments. Accepts:
        - None: non-magnetic. `magnetic_ops` will equal `structural_ops`.
        - (n_sites,) scalar per site: interpreted as a signed moment along
          the z-axis (common collinear/VASP-MAGMOM convention).
        - (n_sites, 3): full non-collinear moment vectors.
    symprec, angle_tolerance : passed to `SpacegroupAnalyzer`.

    Returns
    -------
    dict with keys "structural_ops", "magnetic_ops" (list of SymmOp), and
    "is_magnetic" (bool).
    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    structural_ops = sga.get_symmetry_operations()

    if magmom is None:
        return {
            "structural_ops": structural_ops, 
            "magnetic_ops": structural_ops, 
            "is_magnetic": False
    }

    magmom = np.asarray(magmom, dtype=float)
    if magmom.ndim == 1:
        magmom = np.column_stack([np.zeros_like(magmom), np.zeros_like(magmom), magmom])

    is_magnetic = bool(np.any(np.abs(magmom) > symprec))
    if not is_magnetic:
        return {
            "structural_ops": structural_ops, 
            "magnetic_ops": structural_ops, 
            "is_magnetic": False
    }

    frac_coords = structure.frac_coords
    magnetic_ops = [
        op for op in structural_ops
        if _op_preserves_magnetism(op, frac_coords, magmom, tol=symprec)
    ]
    return {
        "structural_ops": structural_ops, 
        "magnetic_ops": magnetic_ops, 
        "is_magnetic": True
    }