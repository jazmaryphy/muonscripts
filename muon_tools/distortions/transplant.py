# %%
"""
Transplanting an already-computed relaxation distortion onto a new,
symmetry-equivalent muon site -- so it doesn't need to be re-relaxed from
scratch.
"""

from __future__ import annotations

from typing import Optional, Sequence

from pymatgen.core import Structure
from pymatgen.core.operations import SymmOp

from muon_tools.distortions.sites import find_muon_index
from muon_tools.distortions.types import FracCoords, MuonSelect
from muon_tools.distortions.geometry import frac_periodic_close, build_matching_supercell

# %%
def transplant_distortion(
    p_st: Structure,
    rlx_st: Structure,
    mupos: FracCoords,
    sym_ops: Sequence[SymmOp],
    muon_label: str = "H",
    muon_index: MuonSelect = "last",
    tol: float = 1e-3,
) -> Optional[Structure]:
    """Approximate the relaxed structure for a symmetry-equivalent muon
    site, by transplanting the ALREADY-COMPUTED relaxation distortion
    (rlx_st - p_st) onto `mupos`, via
    whichever operation in `sym_ops` maps the original relaxed muon position
    onto it.

    Does not mutate its Structure arguments (both are copied first).

    Parameters
    ----------
    p_st : pymatgen.core.Structure
        Pristine host (muon not yet added, or will be stripped if present).
    rlx_st : pymatgen.core.Structure
        Relaxed structure INCLUDING the muon.
    mupos : (3,) array
        Fractional coordinates of the target (symmetry-equivalent) site.
    sym_ops : list of pymatgen SymmOp
        Operations to search for one mapping the relaxed muon position
        onto `mupos` -- pass `structural_ops` or `magnetic_ops`
        from `get_structural_and_magnetic_ops`, depending on which orbit
        `mupos` came from.
    tol : float, default=1e-3
        Matching tolerance (fractional coordinates).

    Returns
    -------
    pymatgen.core.Structure or None
        The approximate relaxed structure at `mupos`, or None if
        no matching operation was found, or if placing the muon there
        would clash with an existing atom (proximity check).
    """
    p_st = p_st.copy()
    rlx_st = rlx_st.copy()

    p_st = build_matching_supercell(p_st.copy(), rlx_st.copy())

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
        first_mismatch = next(i for i in range(len(rlx_species)) if rlx_species[i] != p_species[i])
        raise ValueError(
            "p_st and rlx_st have matching atom counts but a DIFFERENT site order "
            f"(first mismatch at index {first_mismatch}: "
            f"p_st='{p_species[first_mismatch]}' vs rlx_st='{rlx_species[first_mismatch]}'). "
            "Per-atom displacement is computed index-by-index, so this would silently produce "
            "a wrong structure. If p_st was expanded from a primitive cell (e.g. via "
            "build_matching_supercell), its atom order is not guaranteed to match an "
            "externally-generated rlx_st -- verify or explicitly reorder p_st to match rlx_st."
        )

    matching_op = next(
        (op for op in sym_ops if frac_periodic_close(op.operate(mupos_rlx)%1.0, mupos, tol)),
        None,
    )
    if matching_op is None:
        return None  # mupos isn't in the orbit of mupos_rlx under `sym_ops`

    disp = rlx_st.frac_coords - p_st.frac_coords
    t_disp = matching_op.operate_multi(disp)

    new_st = p_st.copy()
    for i in range(len(new_st)):
        new_st.translate_sites(i, t_disp[i], frac_coords=True, to_unit_cell=False)

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