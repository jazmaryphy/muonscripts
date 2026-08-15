# %%
"""
Transplanting an already-computed relaxation distortion onto a new,
symmetry-equivalent muon site -- so it doesn't need to be re-relaxed from
scratch.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.operations import SymmOp

from muon_tools.distortions.sites import find_muon_index
from muon_tools.distortions.types import FracCoords, MuonSelect
from muon_tools.distortions.reorder import reorder_atoms_by_displacement
from muon_tools.distortions.geometry import frac_periodic_close, build_matching_supercell

# %%
# def transplant_distortion(
#     p_st: Structure,
#     rlx_st: Structure,
#     mupos: FracCoords,
#     sym_ops: Sequence[SymmOp],
#     muon_label: str = "H",
#     muon_index: MuonSelect = "last",
#     tol: float = 1e-3,
# ) -> Optional[Structure]:
#     """Approximate the relaxed structure for a symmetry-equivalent muon
#     site, by transplanting the ALREADY-COMPUTED relaxation distortion
#     (rlx_st - p_st) onto `mupos`, via
#     whichever operation in `sym_ops` maps the original relaxed muon position
#     onto it.

#     Does not mutate its Structure arguments (both are copied first).

#     Parameters
#     ----------
#     p_st : pymatgen.core.Structure
#         Pristine host (muon not yet added, or will be stripped if present).
#     rlx_st : pymatgen.core.Structure
#         Relaxed structure INCLUDING the muon.
#     mupos : (3,) array
#         Fractional coordinates of the target (symmetry-equivalent) site.
#     sym_ops : list of pymatgen SymmOp
#         Operations to search for one mapping the relaxed muon position
#         onto `mupos` -- pass `structural_ops` or `magnetic_ops`
#         from `get_structural_and_magnetic_ops`, depending on which orbit
#         `mupos` came from.
#     tol : float, default=1e-3
#         Matching tolerance (fractional coordinates).

#     Returns
#     -------
#     pymatgen.core.Structure or None
#         The approximate relaxed structure at `mupos`, or None if
#         no matching operation was found, or if placing the muon there
#         would clash with an existing atom (proximity check).
#     """
#     p_st = p_st.copy()
#     rlx_st = rlx_st.copy()

#     # p_st = build_matching_supercell(p_st.copy(), rlx_st.copy())

#     mu_idx_rlxd = find_muon_index(rlx_st, muon_label=muon_label, which=muon_index)
#     if mu_idx_rlxd is None:
#         raise ValueError(f"No atom with muon_label='{muon_label}' found in rlx_st.")
#     mupos_rlx = rlx_st.frac_coords[mu_idx_rlxd]
#     rlx_st.remove_sites([mu_idx_rlxd])

#     mu_idx_prist = find_muon_index(p_st, muon_label=muon_label, which=muon_index)
#     if mu_idx_prist is not None:
#         p_st.remove_sites([mu_idx_prist])

#     if len(rlx_st.frac_coords) != len(p_st.frac_coords):
#         raise ValueError(
#             f"Host atom count mismatch after removing the muon: "
#             f"relaxed={len(rlx_st.frac_coords)}, pristine={len(p_st.frac_coords)}. "
#             f"Are p_st/rlx_st really the same host lattice?"
#         )

#     rlx_species = [str(sp) for sp in rlx_st.species]
#     p_species = [str(sp) for sp in p_st.species]
#     if rlx_species != p_species:
#         first_mismatch = next(i for i in range(len(rlx_species)) if rlx_species[i] != p_species[i])
#         raise ValueError(
#             "p_st and rlx_st have matching atom counts but a DIFFERENT site order "
#             f"(first mismatch at index {first_mismatch}: "
#             f"p_st='{p_species[first_mismatch]}' vs rlx_st='{rlx_species[first_mismatch]}'). "
#             "Per-atom displacement is computed index-by-index, so this would silently produce "
#             "a wrong structure. If p_st was expanded from a primitive cell (e.g. via "
#             "build_matching_supercell), its atom order is not guaranteed to match an "
#             "externally-generated rlx_st -- verify or explicitly reorder p_st to match rlx_st."
#         )

#     matching_op = next(
#         (op for op in sym_ops if frac_periodic_close(op.operate(mupos_rlx)%1.0, mupos, tol)),
#         None,
#     )
#     if matching_op is None:
#         return None  # mupos isn't in the orbit of mupos_rlx under `sym_ops`

#     disp = rlx_st.frac_coords - p_st.frac_coords
#     t_disp = matching_op.operate_multi(disp)

#     new_st = p_st.copy()
#     for i in range(len(new_st)):
#         new_st.translate_sites(i, t_disp[i], frac_coords=True, to_unit_cell=False)

#     try:
#         new_st.append(
#             species=muon_label,
#             coords=mupos,
#             coords_are_cartesian=False,
#             validate_proximity=True,
#             properties={"kind_name": muon_label},
#         )
#     except ValueError:
#         return None  # muon placement clashes with an existing atom

#     return new_st

# %%
def transplant_distortion(
    p_st: Structure,
    rlx_st: Structure,
    mupos: FracCoords,
    sym_ops: Sequence[SymmOp],
    muon_label: str = "H",
    muon_index: MuonSelect = "last",
    tol: float = 1e-3,
    auto_reorder: bool = False,
) -> Optional[Structure]:
    """Approximate the relaxed structure for a symmetry-equivalent muon
    site, by transplanting the ALREADY-COMPUTED relaxation distortion
    (rlx_st - p_st) onto `mupos`, via whichever operation in `sym_ops`
    maps the original relaxed muon position onto it.
 
    Does not mutate its Structure arguments (both are copied first).
 
    IMPORTANT: `p_st` and `rlx_st` must list host atoms in EXACTLY the
    same order (per-atom displacement is computed index-by-index, not by
    nearest-neighbor matching). This holds automatically if both came
    from the same DFT input/output pair. If `p_st` needs to be expanded
    to match `rlx_st`'s cell (e.g. it started life as a primitive cell),
    do that ONCE up front with `geometry.build_matching_supercell` --
    NOT inside this function, and NOT assumed safe -- because the
    resulting atom order is whatever pymatgen's replication algorithm
    produces, which is not guaranteed to match an externally-generated
    `rlx_st` (e.g. a DFT code that groups or orders atoms differently).
    That would silently corrupt every displacement computed here, since
    atom COUNT alone can't detect a reordering. See the check below,
    which does detect it.
 
    Parameters
    ----------
    p_st : pymatgen.core.Structure
        Pristine host (muon not yet added, or will be stripped if present).
    rlx_st : pymatgen.core.Structure
        Relaxed structure INCLUDING the muon.
    mupos : (3,) array
        Fractional coordinates of the target (symmetry-equivalent) site.
    sym_ops : list of pymatgen SymmOp
        CARTESIAN operations to search for one mapping the relaxed muon
        position onto `mupos` -- pass `structural_ops` or `magnetic_ops`
        from `get_structural_and_magnetic_ops`, depending on which orbit
        `mupos` came from.
    tol : float, default=1e-3
        Matching tolerance (fractional coordinates).
    auto_reorder : bool, default=False
        If True, an atom-order mismatch (matching counts, wrong order) is
        NOT an error -- instead, `rlx_st` is automatically reordered to
        match `p_st` via `reorder.reorder_atoms_by_displacement`, which
        finds the species-respecting, periodicity-aware assignment that
        minimizes total displacement (the Hungarian algorithm, run
        separately per species so atoms are never matched across
        species). Off by default because it's a genuine correctness
        trade-off, not a free fix: for a badly-relaxed or very distorted
        structure, the displacement-minimizing assignment might not be
        the physically-correct one. Prefer fixing the ordering upstream
        when you can; use this when you can't (e.g. third-party files
        with an ordering convention you don't control).

    Returns
    -------
    pymatgen.core.Structure or None
        The approximate relaxed structure at `mupos`, or None if no
        matching operation was found, or if placing the muon there would
        clash with an existing atom (proximity check).
 
    Raises
    ------
    ValueError
        If `p_st`/`rlx_st` disagree on host atom count, or agree on
        count but not on per-index species (a strong signal the two
        structures are not in the same atom order -- see above).
    """
    p_st = p_st.copy()
    rlx_st = rlx_st.copy()
 
    lattice_match = np.allclose(
        p_st.lattice.matrix, 
        rlx_st.lattice.matrix, 
        rtol=0, 
        atol=1e-3  # 0.01 Å threshold
    )

    if not lattice_match:
        raise ValueError(
            "p_st and rlx_st have different lattices -- they must be the SAME cell "
            "(the actual DFT relaxation cell) for per-atom displacement to be meaningful."
        )
    lattice = rlx_st.lattice
 
    mu_idx_rlxd = find_muon_index(rlx_st, muon_label=muon_label, which=muon_index)
    if mu_idx_rlxd is None:
        raise ValueError(f"No atom with muon_label='{muon_label}' found in rlx_st.")
    mupos_rlx = rlx_st.frac_coords[mu_idx_rlxd]
    rlx_st.remove_sites([mu_idx_rlxd])
 
    mu_idx_prist = find_muon_index(p_st, muon_label=muon_label, which=muon_index)
    if mu_idx_prist is not None:
        p_st.remove_sites([mu_idx_prist])
 
    if len(rlx_st.frac_coords) != len(p_st.frac_coords):
        raise ValueError(
            f"Host atom count mismatch after removing the muon: "
            f"relaxed={len(rlx_st.frac_coords)}, pristine={len(p_st.frac_coords)}. "
            f"Are p_st/rlx_st really the same host lattice?"
        )
    rlx_species = [str(sp) for sp in rlx_st.species]
    p_species = [str(sp) for sp in p_st.species]
    if rlx_species != p_species:
        if not auto_reorder:
            first_mismatch = next(i for i in range(len(rlx_species)) if rlx_species[i] != p_species[i])
            raise ValueError(
                "p_st and rlx_st have matching atom counts but a DIFFERENT site order "
                f"(first mismatch at index {first_mismatch}: "
                f"p_st='{p_species[first_mismatch]}' vs rlx_st='{rlx_species[first_mismatch]}'). "
                "Per-atom displacement is computed index-by-index, so this would silently produce "
                # "a wrong structure. If p_st was expanded from a primitive cell (e.g. via "
                # "build_matching_supercell), its atom order is not guaranteed to match an "
                # "externally-generated rlx_st -- verify or explicitly reorder p_st to match rlx_st."
                "a wrong structure. Pass auto_reorder=True to recover automatically (minimizes "
                "total displacement, species-respecting), or explicitly reorder p_st/rlx_st yourself."
            )
        rlx_st = reorder_atoms_by_displacement(reference=p_st, target=rlx_st)
 
    mupos_rlx_cart = lattice.get_cartesian_coords(mupos_rlx)
    matching_op = next(
        (op for op in sym_ops
         if frac_periodic_close(lattice.get_fractional_coords(op.operate(mupos_rlx_cart)) % 1.0, mupos, tol)),
        None,
    )
    if matching_op is None:
        return None  # mupos isn't in the orbit of mupos_rlx under `sym_ops`
 
    # Rotate the (Cartesian) displacement field, not the fractional one directly:
    # fractional space is basis-dependent/skewed, so a Cartesian rotation matrix
    # doesn't act correctly on a plain fractional difference vector unless the
    # lattice happens to be orthogonal. apply_rotation_only (no translation --
    # this is a displacement/direction, not a position) is the correct operation.
    disp_cart = lattice.get_cartesian_coords(rlx_st.frac_coords - p_st.frac_coords)
    t_disp_cart = np.array([matching_op.apply_rotation_only(d) for d in disp_cart])
    t_disp_frac = lattice.get_fractional_coords(t_disp_cart)
 
    new_st = p_st.copy()
    for i in range(len(new_st)):
        new_st.translate_sites(i, t_disp_frac[i], frac_coords=True, to_unit_cell=False)
 
    try:
        new_st.append(
            species=muon_label,
            coords=mupos,
            coords_are_cartesian=False,
            validate_proximity=True,
            properties={"kind_name": muon_label},
        )
    except ValueError:
        return None  # muon placement clashes with an existing atom
 
    return new_st