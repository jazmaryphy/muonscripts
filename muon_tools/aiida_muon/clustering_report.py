# %%
"""Pretty-printing for the output of `get_clustering_after_run` /
`get_clustering_after_run_from_data` (see clustering.py).

Kept as a separate module -- display/reporting logic doesn't need to be
tangled up with the clustering algorithm itself, and this way it's
reusable regardless of which of the two entry points (AiiDA node vs. raw
data) produced the `results` dict.
"""

import numpy as np
from pymatgen.core import Structure
from muon_tools.aiida_muon.clustering import load_workchain_data

# %%
def _supercell_to_unitcell_frac(fcoord, scell_matrix, ucell_matrix):
    """Convert fractional coordinates from a supercell's coordinate system
    into the corresponding unit cell's fractional coordinate system.
 
    Uses the standard convention L_supercell = M @ L_unitcell (as used by
    pymatgen's `Structure.make_supercell`), so:
 
        cartesian        = fcoord @ L_supercell
        frac_unitcell     = cartesian @ inv(L_unitcell)
                          = fcoord @ (L_supercell @ inv(L_unitcell))
 
    Folded into [0, 1) via modulo, since a supercell position generally
    lands outside a single unit cell's [0,1) range before wrapping.

    NOTE: @ is same as np.dot()
    """
    M = np.dot(np.asarray(scell_matrix), np.linalg.inv(np.asarray(ucell_matrix)))
    fcoord_unitcell = np.dot(np.asarray(fcoord), M)

    return fcoord_unitcell % 1.0

# %%
def print_clustering_summary(
    results, 
    input_st=None, 
    use_unitcell=None, 
    precision=5
):
    """Pretty-print `unique_pos` (index, muon fractional coordinates,
    energy difference) and, if present, `mag_inequivalent` (the muon
    positions of any magnetically-inequivalent structures still pending
    calculation) from a clustering result.

    Parameters
    ----------
    results : dict
        Output of `get_clustering_after_run` or
        `get_clustering_after_run_from_data` -- must have 'unique_pos'
        and 'mag_inequivalent' keys (both may be empty).
    input_st : pymatgen.core.Structure or None
        Default=None. The UNIT CELL structure. If None, positions are always
        printed in supercell fractional coordinates, regardless of
        `use_unitcell`.
    use_unitcell : bool or None
        Default=None. Explicit control over which coordinate system to display,
        independent of whether `input_st` happens to be given:
        - None (default) -- AUTO: use unit-cell coordinates if
          `input_st` was given, otherwise supercell coordinates ("do the
          sensible thing" given what you passed).
        - True -- FORCE unit-cell coordinates. Raises ValueError if
          `input_st` isn't also given (nothing to convert with).
        - False -- FORCE supercell coordinates, even if `input_st` is
          given (e.g. you have it handy for another reason, but still
          want to see the raw as-relaxed supercell positions).
    precision : int, default=5
        Decimal places shown for fractional coordinates and energies.

    Notes
    -----
    - Energy differences are computed HERE via `load_workchain_data`,
      relative to the minimum among `unique_pos` -- the dicts inside
      `unique_pos` carry the original, non-shifted `'energy'` value (they
      are the same dicts you originally passed in as `data`, just
      filtered down to the unique sites), so this re-derives the
      min-relative energies the same way `cluster_unique_sites` did
      internally, rather than trusting a stale/absent value.
    - Muon positions for `mag_inequivalent` entries are read directly off
      each pymatgen `Structure` (they are full `Structure` objects, not
      dicts, unlike `unique_pos`), using the same "muon = the atom with
      atomic number 1 (H)" convention as `load_workchain_data`.
    """
    if use_unitcell and input_st is None:
        raise ValueError(
            "use_unitcell=True requires `input_st` (the unit cell Structure) "
            "to compute the supercell->unitcell conversion."
        )
    show_unitcell = (input_st is not None) if use_unitcell is None else use_unitcell
    unitcell_matrix = np.array(input_st.lattice.matrix) if show_unitcell else None
    coord_label = "unitcell frac." if show_unitcell else "supercell frac."
 
    unique_pos = results.get("unique_pos", [])
    mag_inequivalent = results.get("mag_inequivalent", [])
 
    print("\n" + "=" * 78)
    print(f"UNIQUE MUON SITES ({coord_label} coordinates)")
    print("=" * 78)
 
    if not unique_pos:
        print("  (none found)")
    else:
        idx_arr, mu_arr, denrg_arr = load_workchain_data(unique_pos)
 
        if show_unitcell:
            supercell_matrix = Structure.from_dict(unique_pos[0]["rlxd_struct"]).lattice.matrix
            mu_arr = np.array([
                _supercell_to_unitcell_frac(mu, supercell_matrix, unitcell_matrix)
                for mu in mu_arr
            ])
 
        w = 8 + precision  # column width, scales with requested precision
        header = (
            f"{'#':>3} {'idx':>10} "
            f"{'frac_x':>{w}} {'frac_y':>{w}} {'frac_z':>{w}} {'dE (eV)':>{w}}"
        )
        print(header)
        print("-" * len(header))
        for i, (idx, mu, denrg) in enumerate(zip(idx_arr, mu_arr, denrg_arr)):
            print(
                f"{i:>3} {str(idx):>10} "
                f"{mu[0]:>{w}.{precision}f} {mu[1]:>{w}.{precision}f} "
                f"{mu[2]:>{w}.{precision}f} {denrg:>{w}.{precision}f}"
            )
        print("-" * len(header))
        print(f"Total unique sites: {len(unique_pos)}")
 
    print("\n" + "=" * 78)
    print(f"MAGNETICALLY INEQUIVALENT SITES PENDING CALCULATION "
          f"({len(mag_inequivalent)}) ({coord_label} coordinates)")
    print("=" * 78)
 
    if not mag_inequivalent:
        print("  (none -- all magnetically inequivalent sites already covered)")
    else:
        w = 8 + precision
        header = f"{'#':>3} {'frac_x':>{w}} {'frac_y':>{w}} {'frac_z':>{w}}"
        print(header)
        print("-" * len(header))
        for i, struct in enumerate(mag_inequivalent):
            try:
                mu_pos = struct.frac_coords[struct.atomic_numbers.index(1)]
            except (ValueError, AttributeError) as exc:
                print(f"{i:>3}  (could not extract muon position: {exc})")
                continue
            if show_unitcell:
                mu_pos = _supercell_to_unitcell_frac(mu_pos, struct.lattice.matrix, unitcell_matrix)
            print(f"{i:>3} {mu_pos[0]:>{w}.{precision}f} {mu_pos[1]:>{w}.{precision}f} {mu_pos[2]:>{w}.{precision}f}")
        print("-" * len(header))
 
    print("=" * 78)