# %%
"""
Classify a fractional position (e.g. a relaxed muon site) against the
Wyckoff positions of a host structure's space group.

Requires `pyxtal` in addition to this package's usual dependencies --
imported lazily inside `classify_wyckoff_site` so the rest of `distortions`
doesn't need it installed. `pip install pyxtal` if you hit an ImportError
here.

Correctness note (why this isn't just "apply each Wyckoff letter's own
operations to the query and measure how far it moves", which is a natural
first instinct but gives meaningless numbers): a Wyckoff position's own
operation list is only a coset-representative generator for ITS OWN orbit
when applied to a point that already satisfies that position's site-
symmetry constraint. Applied to an arbitrary point instead, it does NOT
measure distance to the nearest true example of that Wyckoff type. Two
methods are used here instead, chosen per Wyckoff letter based on
whether it's actually decorated by a real atom in `structure`:

- OCCUPIED letters (real atoms present): measure the true periodic
  (minimum-image) distance to those actual atoms directly -- exact, no
  symbolic algebra needed, and physically the more relevant question
  ("is my muon near this real atom's site").
- UNOCCUPIED letters (no atom there -- interstitial voids, which is
  often exactly what a muon prefers): there's no real atom to measure
  against, so the query is first PROJECTED onto that Wyckoff position's
  defining sub-manifold (by evaluating its canonical/generator operation
  with the query's own coordinates substituted in -- exact for 0-DOF
  point positions; for positions with free parameters, this adopts the
  query's own coordinate values as the free parameter(s), which is a
  well-defined and standard convention but not a guaranteed globally-
  nearest-point projection for skewed lattices). The projected point's
  full orbit is then generated and the minimum periodic distance from
  the query to any image of it is measured.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from symmetry.utils import FracCoords, MuonSelect
from symmetry.utils import periodic_distance_matrix, find_muon_index

# %%
@dataclass
class WyckoffMatch:
    """One Wyckoff position's classification result for a queried site.

    Attributes
    ----------
    letter : str
        Wyckoff letter (standard ITA convention: 'a' = most special).
    multiplicity : int
        Number of equivalent points per conventional cell.
    occupied : bool
        Whether a real atom in `structure` sits on this Wyckoff position.
    min_distance : float [Angstrom]
        Periodic (minimum-image) distance from the query to the nearest
        actual example of this Wyckoff position -- a real atom if
        `occupied`, otherwise the nearest image of the symmetry-projected
        candidate (see module docstring).
    method : {"real_atom", "symbolic_projection"}
        Which of the two measurement methods was used.
    site_symmetry : str, optional
        Site-symmetry symbol from spglib, if `occupied`.
    """
    letter: str
    multiplicity: int
    occupied: bool
    min_distance: float
    method: str
    site_symmetry: Optional[str] = None

# %%
def classify_wyckoff_site(
    site: FracCoords,
    host_lattice: Structure,
    muon_label: str = "H",
    muon_index: MuonSelect = "last",
    symprec: float = 1e-3,
    angle_tol: float = 5.0,    
) -> List[WyckoffMatch]:
    """Classify fractional position `site` against every Wyckoff
    position of `host_lattice`'s space group.

    Parameters
    ----------
    site : (3,) array-like
        Fractional coordinates to classify (e.g. a relaxed muon site).
    host_lattice : pymatgen.core.Structure
        Host structure (the muon is auto-stripped if present, matching
        `muon_label`/`muon_index`; symmetry should be evaluated on the
        pristine host).
    symprec, angle_tol
        Passed to `SpacegroupAnalyzer` (and to spglib's own space-group
        detection under the hood).

    Returns
    -------
    list of WyckoffMatch, one per Wyckoff position of the space group,
    sorted by `min_distance` ascending (closest first) -- the simplest,
    most intuitive default. Note that high-multiplicity (weakly-
    constrained, close to the general position) Wyckoff letters will
    often show small distances too, since they have fewer constraints to
    satisfy -- that's expected, not a bug. See `nearest_wyckoff_site` for
    a selection rule that prefers the most *specific* nearby match rather
    than just the closest one.
    """
    try:
        from pyxtal.symmetry import Group
    except ImportError as exc:
        raise ImportError(
            "classify_wyckoff_site requires the 'pyxtal' package (pip install pyxtal)."
        ) from exc

    host_lattice = host_lattice.copy()
    mu_idx = find_muon_index(host_lattice, muon_label=muon_label, which=muon_index)
    if mu_idx is not None:
        host_lattice.remove_sites([mu_idx])

    lattice = host_lattice.lattice
    site = np.atleast_2d(site)%1.0

    sga = SpacegroupAnalyzer(host_lattice, symprec=symprec, angle_tolerance=angle_tol)
    dataset = sga.get_symmetry_dataset()
    group = Group(dataset.number)

    occupied_letters = list(dataset.wyckoffs)
    site_symmetry_by_letter = dict(zip(dataset.wyckoffs, dataset.site_symmetry_symbols))
    frac_by_letter: dict = {}
    for i, letter in enumerate(occupied_letters):
        frac_by_letter.setdefault(letter, []).append(host_lattice.frac_coords[i])

    results = []
    for wp in group.Wyckoff_positions:
        if wp.letter in frac_by_letter:
            atom_coords = np.array(frac_by_letter[wp.letter])
            min_distance = float(periodic_distance_matrix(lattice, site, atom_coords).min())
            results.append(WyckoffMatch(
                letter=wp.letter, 
                multiplicity=wp.multiplicity, 
                occupied=True,
                min_distance=min_distance, 
                method="real_atom",
                site_symmetry=site_symmetry_by_letter.get(wp.letter),
            ))
        else:
            projected = wp.ops[0].operate(site) % 1.0
            orbit = np.vstack([op.operate(projected) % 1.0 for op in wp.ops])
            min_distance = float(periodic_distance_matrix(lattice, site, orbit).min())
            results.append(WyckoffMatch(
                letter=wp.letter, 
                multiplicity=wp.multiplicity, 
                occupied=False,
                min_distance=min_distance, 
                method="symbolic_projection",
            ))

    results.sort(key=lambda r: r.min_distance)

    return results

# %%
def nearest_wyckoff_site(
    site: FracCoords,
    host_lattice: Structure,
    min_distance: Optional[float] = 0.5,
    prefer_specific: bool = True,
    max_multi: Optional[int] = None,
    muon_label: str = "H",
    muon_index: MuonSelect = "last",
    symprec: float = 1e-3,
    angle_tol: float = 5.0, 
) -> WyckoffMatch:
    """Convenience wrapper around `classify_wyckoff_site`: just the single
    best match.

    Parameters
    ----------
    min_distance : float [Angstrom] or None, default=0.5
        Restricts which candidates `prefer_specific` is allowed to choose
        among -- without this, "prefer specific" could pick a technically
        more special but physically irrelevant match far from the query
        (e.g. a rare 1-fold site 2 Angstrom away over an obviously-correct
        3-fold site 0.09 Angstrom away). If no match falls within the
        cutoff, the single closest match overall is returned instead
        (never raises just because nothing was that close -- check
        `result.min_distance` yourself if you need to know whether it
        was actually within range). Set to None to disable the cutoff
        and let `prefer_specific` consider every match -- not recommended
        together with `prefer_specific=True` for the reason above.
    prefer_specific : bool, default=True
        Among candidates within `min_distance` (or all candidates, if
        `min_distance` is None), prefer the most specific (lowest
        multiplicity) match rather than simply the closest one -- usually
        the more physically meaningful answer for a relaxed muon site
        that's genuinely near a special position, since high-multiplicity
        (weakly-constrained) Wyckoff letters can show small distances
        too just by having fewer constraints to satisfy. Set False to
        just take the closest match, full stop.
    max_multi : int, optional
        If given, ignore Wyckoff positions with multiplicity above this
        (e.g. exclude the trivial general position, which always matches
        everything at distance 0 and is rarely the answer you want).

    Returns
    -------
    WyckoffMatch
        The selected Wyckoff position (see parameters above for how it's
        chosen).
    """
    matches = classify_wyckoff_site(
        site,
        host_lattice,
        muon_label=muon_label, 
        muon_index=muon_index,
        symprec=symprec, 
        angle_tol=angle_tol,
    )
    if max_multi is not None:
        matches = [m for m in matches if m.multiplicity <= max_multi]
    if not matches:
        raise ValueError(f"No Wyckoff position found with multiplicity <= {max_multi}.")

    candidates = matches
    if min_distance is not None:
        within_cutoff = [m for m in matches if m.min_distance <= min_distance]
        if within_cutoff:
            candidates = within_cutoff
        # else: fall through and rank the full set -- still returns the
        # single closest match overall rather than raising.

    if prefer_specific:
        candidates = sorted(candidates, key=lambda m: (m.multiplicity, m.min_distance))
    return candidates[0]

# %%
def muon_wyckoff_site(
    site: FracCoords,
    host_lattice: Structure,
    min_distance: Optional[float] = 0.5,
    prefer_specific: bool = True,
    prefer_occupied: bool = True,
    max_multi: Optional[int] = None,
    muon_label: str = "H",
    muon_index: MuonSelect = "last",
    symprec: float = 1e-3,
    angle_tol: float = 5.0, 
) -> WyckoffMatch:
    """Convenience wrapper around `classify_wyckoff_site`: returns the best matching Wyckoff site.

    Parameters
    ----------
    site : FracCoords
        Fractional position of the muon site.
    host_lattice : Structure
        Host crystal structure.
    min_distance : float [Å] or None, default=0.5
        Distance cutoff (Å) for considering candidate Wyckoff sites.
        If no site falls within `min_distance`, falls back to evaluating all sites.
    prefer_specific : bool, default=True
        If True, prefers lower multiplicity (more constrained/special) Wyckoff sites.
    prefer_occupied : bool, default=True
        If True, prioritizes Wyckoff positions decorated by real host atoms over 
        symbolic projections onto unoccupied interstitial voids.
    max_multi : int, optional
        Exclude Wyckoff positions with multiplicity above this threshold (e.g., exclude general positions).
    muon_label, muon_index, symprec, angle_tol
        Parameters passed to `classify_wyckoff_site`.

    Returns
    -------
    WyckoffMatch
        The selected Wyckoff position.
    """
    matches = classify_wyckoff_site(
        site,
        host_lattice,
        muon_label=muon_label, 
        muon_index=muon_index,
        symprec=symprec, 
        angle_tol=angle_tol,
    )
    # 1. Filter out positions exceeding max multiplicity
    if max_multi is not None:
        matches = [m for m in matches if m.multiplicity <= max_multi]
    if not matches:
        raise ValueError(f"No Wyckoff position found with multiplicity <= {max_multi}.")

    # 2. Apply distance cutoff filter
    candidates = matches
    if min_distance is not None:
        within_cutoff = [m for m in matches if m.min_distance <= min_distance]
        if within_cutoff:
            candidates = within_cutoff
        # else: fall through and rank the full set -- still returns the
        # single closest match overall rather than raising.

    # 3. Deterministic Sorting Strategy
    if prefer_specific:
        # Sort hierarchy:
        #   1. Occupied status (Real atom > Unoccupied projection, if prefer_occupied=True)
        #   2. Multiplicity (Lower/more specific > Higher)
        #   3. Distance (Closest > Further)
        candidates = sorted(
            candidates, 
            key=lambda m: (
                not m.occupied if prefer_occupied else False, 
                m.multiplicity, 
                m.min_distance
            )
        )
    else:
        # Strictly sort by minimum distance (pure physical proximity)
        candidates = sorted(candidates, key=lambda m: m.min_distance)

    return candidates[0]