# %%
"""
Top-level workflow: for a single DFT-relaxed muon site, find every
structurally- and/or magnetically-equivalent site and generate an
approximate relaxed structure for each one.

No class here on purpose. The only thing a class would buy us is caching
the symmetry search (`get_structural_and_magnetic_ops`) across repeated
calls for the same host -- and a plain dict does that job just as well,
with far less ceremony:

    from symmetry import get_structural_and_magnetic_ops

    ops_info = get_structural_and_magnetic_ops(host_st, magmom=magmom)  # compute once
    for pristine, relaxed in my_relaxation_results:
        sites = get_structs(
            pristine, relaxed, host_st, ops_info=ops_info  # reused, not recomputed
        )
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from pymatgen.core import Structure

from muon_tools.distortions.geometry import frac_periodic_close
from muon_tools.distortions.transplant import transplant_distortion
from muon_tools.distortions.symmetry import get_structural_and_magnetic_ops
from muon_tools.distortions.types import Moments, MuonSelect, SymmetryOpsInfo
from muon_tools.distortions.sites import get_equivalent_sites, find_muon_index

# %%
@dataclass
class EquivalentSite:
    """One entry from `generate_equivalent_muon_structures`.

    Attributes
    ----------
    frac_pos : (3,) ndarray
        Fractional coordinates of this equivalent site.
    structure : pymatgen.core.Structure or None
        The (approximate) relaxed structure at this site -- None if
        transplanting failed (proximity clash, or no symmetry operation
        connects it to the original site).
    is_original : bool
        True for the input site itself (`structure` is the input
        `rlx_st`, unchanged, not transplanted).
    is_magnetically_equivalent : bool
        True if this site is in the orbit of the original site under the
        MAGNETIC symmetry (trust the transplanted local field); False if
        only structurally equivalent (needs its own relaxation/field
        calculation if magnetism matters).
    """
    frac_pos: np.ndarray
    structure: Optional[Structure]
    is_original: bool
    is_magnetically_equivalent: bool

    @property
    def ok(self) -> bool:
        """Shorthand for `structure is not None` -- did transplanting succeed."""
        return self.structure is not None

# %%
def get_structs(
    p_st: Structure,
    rlx_st: Structure,
    host_st: Structure,
    magmom: Optional[Moments] = None,
    magmom_st: Optional[Structure] = None,
    muon_label: str = "H",
    muon_index: MuonSelect = "last",
    tol: float = 1e-3,
    symprec: float = 1e-3,
    min_distance: Optional[float] = 0.5,
    ops_info: Optional[SymmetryOpsInfo] = None,
    auto_reorder: bool = False,
) -> List[EquivalentSite]:
    """For a single DFT-relaxed muon site, find every structurally- and/or
    magnetically-equivalent site and generate an approximate relaxed
    structure for each (via distortion transplanting), ready for local-
    field calculations (dipolar/EFG) at every equivalent site.

    Parameters
    ----------
    p_st, rlx_st : pymatgen.core.Structure
        Pristine host (muon-free or will be stripped) and relaxed
        structure (with muon), as in `transplant_distortion`.
    host_st : pymatgen.core.Structure
        Structure used to compute symmetry. MUST share the same cell as
        `p_st`/`rlx_st` -- symmetry operations
        are basis-dependent, so a primitive cell's operations don't apply
        directly to supercell fractional coordinates. If your host was
        solved as a primitive cell, build a matching supercell first (see
        `geometry.build_matching_supercell`) and pass that here.
    magmom : array-like or None
        See `get_structural_and_magnetic_ops`. Ignored if `ops_info` is
        given.
    magmom_st : pymatgen.core.Structure, optional
        The structure `magmom` actually refers to (muon-free) -- e.g. the
        primitive/conventional cell, if that's small enough to already
        capture the true magnetic pattern (simple ferromagnetic order,
        for instance). Only needs to be the larger DFT supercell if the
        magnetic order breaks the primitive cell's own periodicity (e.g.
        antiferromagnetic order that doubles it) -- the primitive cell
        can't represent moments finer than its own periodicity, in which
        case `magmom` must be sized to that larger structure instead.
        Defaults to `host_structure` if omitted. Ignored if `ops_info` is
        given.
    min_distance : float [Angstrom] or None, default=0.5
        Passed to `get_equivalent_sites`'s physical-distance cleanup pass.
        Set to None to skip it (fractional-tolerance-only dedup).
    ops_info : dict, optional
        A precomputed result of `get_structural_and_magnetic_ops(host_st, magmom=magmom)`.
        Pass this in when calling this function repeatedly for the same
        host (e.g. several relaxations of the same material) to skip
        redoing the symmetry search every time -- see the module
        docstring for the pattern.
    auto_reorder : bool, default=False
        Passed to `transplant_distortion` -- if True, an atom-order
        mismatch between `pristine_structure` and `relaxed_structure` is
        auto-corrected (displacement-minimizing, species-respecting
        assignment) instead of raising. See `transplant_distortion` for
        the correctness trade-off. Off by default.


    Returns
    -------
    list of EquivalentSite, one per equivalent site (including the
    original).
    """
    # if ops_info is None:
    #     check_st = p_st.copy()
    #     check_idx = find_muon_index(check_st, muon_label=muon_label, which=muon_index)
    #     ops_info = get_structural_and_magnetic_ops(host_st, magmom=magmom, symprec=symprec)

    if ops_info is None:
        if magmom_st is not None:
            check_st = magmom_st.copy()
            check_idx = find_muon_index(check_st, muon_label=muon_label, which=muon_index)
            if check_idx is not None:
                check_st.remove_sites([check_idx])
        else:
            check_st = host_st
        ops_info = get_structural_and_magnetic_ops(
            host_st, check_structure=check_st, magmom=magmom, symprec=symprec
        )

    struct_ops = ops_info["structural_ops"]
    magnetic_ops = ops_info["magnetic_ops"]

    mu_idx = find_muon_index(rlx_st, muon_label=muon_label, which=muon_index)
    if mu_idx is None:
        raise ValueError(f"No atom with muon_label='{muon_label}' found in rlx_st.")
    mu_frac = rlx_st.frac_coords[mu_idx]

    struct_sites = get_equivalent_sites(
        mu_frac, struct_ops, lattice=host_st, tol=tol, min_distance=min_distance
    )
    magnetic_sites = get_equivalent_sites(
        mu_frac, magnetic_ops, lattice=host_st, tol=tol, min_distance=min_distance
    )

    results: List[EquivalentSite] = []
    for site in struct_sites:
        is_original = frac_periodic_close(site, mu_frac, tol)
        is_mag_equiv = any(frac_periodic_close(site, m, tol) for m in magnetic_sites)

        struct: Optional[Structure]
        if is_original:
            struct = rlx_st.copy()
        else:
            ops_for_this_site = magnetic_ops if is_mag_equiv else struct_ops
            struct = transplant_distortion(
                p_st, rlx_st, site, ops_for_this_site,
                muon_label=muon_label, muon_index=muon_index, 
                tol=tol, auto_reorder=auto_reorder,
            )

        results.append(EquivalentSite(
            frac_pos=site, 
            structure=struct,
            is_original=is_original, 
            is_magnetically_equivalent=is_mag_equiv,
        ))

    return results