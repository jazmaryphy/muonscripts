# %%
"""
Robust atom-correspondence recovery via displacement-minimizing assignment.
 
`transplant_distortion` needs `p_st` and `rlx_st` to list host atoms in
exactly the same order -- true automatically when both came from the same
DFT input/output pair, but not guaranteed otherwise (e.g. a structure
re-read/re-written by a different tool with its own ordering convention,
or auto-expanded from a primitive cell via `geometry.build_matching_supercell`).
 
This module recovers that correspondence when it's been lost, by finding
the species-respecting atom assignment that minimizes total periodic
(minimum-image) displacement -- the Hungarian algorithm applied SEPARATELY
within each chemical species group, so atoms are never matched across
species regardless of geometry, no matter how the initial guess looks.
"""
 
from __future__ import annotations
 
import numpy as np
from scipy.optimize import linear_sum_assignment
 
from pymatgen.core import Structure
from muon_tools.distortions.geometry import periodic_distance_matrix

# %%
def reorder_atoms_by_displacement(reference: Structure, target: Structure) -> Structure:
    """Return a copy of `target` with atoms reordered to best match
    `reference`'s atom order, by minimizing total periodic (minimum-image)
    displacement -- separately within each chemical species, so atoms are
    never matched across species.
 
    Parameters
    ----------
    reference : pymatgen.core.Structure
        The atom order to match (e.g. `p_st`, muon-free).
    target : pymatgen.core.Structure
        Structure to reorder (e.g. `rlx_st`, muon-free) -- NOT mutated;
        a reordered copy is returned.
 
    Returns
    -------
    pymatgen.core.Structure
        Copy of `target`, reordered so `result[i]` corresponds to
        `reference[i]` for every i (best displacement-minimizing match
        within each species).
 
    Raises
    ------
    ValueError
        If `reference` and `target` don't share the same lattice, or
        don't have matching per-species atom counts (a genuine species
        mismatch, not just an ordering difference -- reordering can't
        fix that).
    """
    if reference.lattice != target.lattice:
        raise ValueError(
            "reference and target must share the same lattice to compute periodic displacements."
        )
 
    ref_species = [str(sp) for sp in reference.species]
    tgt_species = [str(sp) for sp in target.species]
    ref_counts = {sp: ref_species.count(sp) for sp in set(ref_species)}
    tgt_counts = {sp: tgt_species.count(sp) for sp in set(tgt_species)}
    if ref_counts != tgt_counts:
        raise ValueError(
            f"reference and target have different per-species atom counts -- this is a genuine "
            f"species mismatch, not just an ordering difference, so it can't be fixed by "
            f"reordering: {ref_counts} vs {tgt_counts}."
        )
 
    lattice = reference.lattice
    perm = np.empty(len(reference), dtype=int)
 
    for sp in ref_counts:
        ref_idx = [i for i, s in enumerate(ref_species) if s == sp]
        tgt_idx = [i for i, s in enumerate(tgt_species) if s == sp]
        dmat = periodic_distance_matrix(
            lattice, reference.frac_coords[ref_idx], target.frac_coords[tgt_idx]
        )
        row_ind, col_ind = linear_sum_assignment(dmat)
        for r, c in zip(row_ind, col_ind):
            perm[ref_idx[r]] = tgt_idx[c]
 
    return Structure.from_sites([target[i] for i in perm])