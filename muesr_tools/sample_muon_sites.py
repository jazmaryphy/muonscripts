# %%
"""
Sample candidate muon-stopping-site(s).
"""

from __future__ import annotations

from typing import Mapping, Sequence, Optional, Tuple, Dict, List

import numpy as np
import numpy.typing as npt

from pymatgen.core import Structure
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

FracCoords = npt.NDArray[np.float64]  # shape (N, 3), fractional coordinates

# %%
def find_equivalent_positions(
    frac_coords: FracCoords,
    host_lattice: Structure,
    atol: float = 1e-3,
) -> npt.NDArray[np.int32]:
    """
    Creates a list of symmetry equivalent positions for the input structure.
    The output is the same as spg.get_symmetry_dataset()['equivalent_atoms']
    >>> from pymatgen.util.testing import PymatgenTest
    >>> from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    >>> st = PymatgenTest.TEST_STRUCTURES['Li10GeP2S12']
    >>> Niche.find_equivalent_positions(st.frac_coords,st) == SpacegroupAnalyzer(st).get_symmetry_dataset()['equivalent_atoms'] # doctest: +NORMALIZE_WHITESPACE
    array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True])
    """

    lattice = host_lattice.lattice

    # Bring to unit cell
    frac_coords %= 1

    # prepare list of equivalent atoms. -1 mean "not yet analyzed".
    eq_list = np.zeros(len(frac_coords), dtype=np.int32) - 1

    spg = SpacegroupAnalyzer(host_lattice, symprec=atol)

    ops = spg.get_symmetry_operations()

    # This hosts all the equivalent positions obtained for each of the
    # lattice points using all symmetry operations.
    eq_pos = np.zeros([len(ops), len(frac_coords), 3])

    for i, op in enumerate(ops):
        eq_pos[i] = op.operate_multi(frac_coords) % 1

    # Compute equivalence list
    '''
    Miki Bonacci: maybe optimization with np.masked arrays. 
    '''
    for i in range(len(frac_coords)):
        if eq_list[i] >= 0:
            continue

        for j in range(i, len(frac_coords)):
            diff = pbc_shortest_vectors(
                lattice, eq_pos[:, j, :], frac_coords[i]
            ).squeeze()
            if (np.linalg.norm(diff, axis=1) < atol).any():
                eq_list[j] = i

    return eq_list


def prune_atoms_too_close(
    grid_coords: FracCoords,
    host_lattice: Structure,
    min_distance: float,
) -> npt.NDArray[np.bool_]:
    """
    List interstitial positions too close to the lattice points of the
    hosting compounds. Threshold is set by min_distance.
    Parameters
    ----------
    grid_coords:
        An array that contains interstitial points in scaled coordinates.
    atoms: pymatgen.core.Structure
        A pymatgen structure.
    min_distance : list
        Minimum spacing in $A$ between interstitial points and lattice points.
    Returns
    -------
    mask : np.ndarray
        List of interstitial positions not too close to host_lattice lattice points.
    """

    all_differences = pbc_shortest_vectors(
        host_lattice.lattice, grid_coords, host_lattice.frac_coords
    )

    '''
    Miki Bonacci: masked python array? In this way, where the mask is True, position
    should be discarded.

    import numpy.ma as ma
    masked = ma.masked_less(
        np.linalg.norm(all_differences, axis=1), min_distance
        )

    return masked.mask
    '''

    mask = np.ones(len(grid_coords), dtype=bool)
    for i, p in enumerate(all_differences):
        if any(np.linalg.norm(p, axis=1) < min_distance):
            mask[i] = False
    return mask

# %%
def anion_sites(
    structure: Structure,
    n_samples: int = 10000,
    anion_distance: Tuple[float, float] = (0.9, 1.11),
    min_cation_distances: Optional[Mapping[str, float]] = None,
    anion_specie: Sequence[str] = ("O",),
    cation_specie: Optional[Sequence[str]] = None,
    seed: Optional[int] = 42,
    batch_factor: int = 10,
    max_attempts: int = 100,
) -> npt.NDArray[np.float64]:
    """
    Generate candidate muon-stopping sites near an anion sublattice.

    Implements the electrostatic heuristic that a positive muon stops
    close to a region of negative charge density and away from
    positively charged ions: candidate fractional coordinates are
    accepted only if they satisfy

    1. Distance to the *nearest* atom of any species in
       ``anion_specie`` falls within ``anion_distance``.
    2. Distance to *every* cation species listed in
       ``min_cation_distances`` is at or above that species'
       minimum distance (all entries must individually pass --
       logical AND across the dict).

    All distances are in Angstrom. Sampling is by rejection: random
    points are drawn uniformly over the full unit cell in batches and
    filtered against both criteria until ``n_samples`` candidates are
    accepted or ``max_attempts`` batches are exhausted.

    This targets ionic/oxide-like hosts with a single chemically
    meaningful anion group (e.g. O, F, N, S, Cl -- whichever is passed
    via ``anion_specie``); see the module docstring for scope and
    limitations (mixed anion sublattices, metallic hosts, sampling
    efficiency, symmetry).

    Parameters
    ----------
    structure
        Host crystal structure (should NOT already contain a muon /
        implanted species -- this function only samples empty-space
        candidate positions, it does not add or remove sites).
    n_samples
        Number of *accepted* candidate sites to return.
    anion_distance
        ``(min, max)`` allowed distance (Angstrom) to the nearest
        anion atom. Must satisfy ``0 <= min < max``.
    min_cation_distances
        Mapping of cation species symbol -> minimum allowed distance
        (Angstrom) from that species. Every key must correspond to a
        species actually present in ``structure`` (excluding
        ``anion_specie``); missing species raise ``ValueError`` up
        front. If ``None``, no cation-exclusion constraint is applied.
        Example: ``{"Ba": 1.0, "Na": 1.0, "Os": 1.0}``.
    anion_specie
        Species pooled together to define "nearest anion" (e.g.
        ``("O",)`` for an oxide, ``("O", "F")`` to treat O and F as
        equivalent for this purpose -- NOT to apply separate windows
        to each).
    cation_specie
        Species treated as cations for validating
        ``min_cation_distances`` keys. If ``None``, defaults to every
        species in ``structure`` not listed in ``anion_specie``.
    seed
        Seed for ``numpy.random.default_rng``; ``None`` for
        nondeterministic sampling.
    batch_factor
        Candidate points drawn per batch, as a multiple of the number
        of samples still needed. Higher values trade memory for fewer
        (but larger) rejection-sampling batches.
    max_attempts
        Maximum number of batches to draw before giving up.

    Returns
    -------
    numpy.ndarray of numpy.float64, shape (n_samples, 3)
        Fractional coordinates of accepted candidate muon sites.

    Raises
    ------
    ValueError
        If any parameter is invalid, if a requested species is absent
        from ``structure``, or if ``n_samples`` candidates could not
        be generated within ``max_attempts`` batches (the distance
        constraints may be too restrictive for this structure/cell
        size -- consider loosening ``min_cation_distances``,
        widening ``anion_distance``, or increasing
        ``batch_factor``/``max_attempts``).
    """

    # Validate scalar/shape arguments
    if n_samples <= 0:
        raise ValueError("'n_samples' must be positive.")

    if len(anion_distance) != 2:
        raise ValueError("'anion_distance' must contain exactly two values.")

    anion_min, anion_max = map(float, anion_distance)

    if anion_min < 0 or anion_max <= anion_min:
        raise ValueError("'anion_distance' must satisfy 0 <= min < max.")

    if batch_factor <= 0:
        raise ValueError("'batch_factor' must be positive.")

    if max_attempts <= 0:
        raise ValueError("'max_attempts' must be positive.")

    # Resolve species -> fractional coordinates
    anion_specie = tuple(anion_specie)

    if cation_specie is None:
        cation_specie = tuple(
            sorted(
                {
                    site.species_string
                    for site in structure
                    if site.species_string not in anion_specie
                }
            )
        )
    else:
        cation_specie = tuple(cation_specie)

    anion_indices: List[int] = [
        i for i, site in enumerate(structure) if site.species_string in anion_specie
    ]
    if not anion_indices:
        raise ValueError(f"No atoms found for species {anion_specie}.")

    anion_frac: FracCoords = structure.frac_coords[anion_indices]

    cation_frac: Dict[str, FracCoords] = {}
    for specie in cation_specie:
        indices = [i for i, site in enumerate(structure) if site.species_string == specie]
        if not indices:
            raise ValueError(f"No atoms with species '{specie}' were found in the structure.")
        cation_frac[specie] = structure.frac_coords[indices]

    # min_cation_distances: fail loudly up front for any species that
    # isn't actually available, rather than silently skipping it.
    if min_cation_distances is not None:
        for specie, distance in min_cation_distances.items():
            if specie not in cation_frac:
                raise ValueError(f"Cation species '{specie}' is not present in the structure.")
            if float(distance) < 0:
                raise ValueError(f"Minimum distance for {specie} cannot be negative.")

    # Random number generator
    rng = np.random.default_rng(seed)
    candidates: List[FracCoords] = []

    target = n_samples
    attempts = 0

    # Generate candidates
    while len(candidates) < target:

        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                f"Could only generate {len(candidates)} candidate sites after "
                f"{max_attempts} batches. The distance constraints may be too "
                "restrictive."
            )

        n_batch = max(batch_factor * (target - len(candidates)), batch_factor)
        frac = rng.random((n_batch, 3))

        # Distance to nearest anion -- vectorized, fractional coords.
        anion_dist_matrix = structure.lattice.get_all_distances(frac, anion_frac)
        anion_distances = anion_dist_matrix.min(axis=1)

        valid = (anion_distances >= anion_min) & (anion_distances <= anion_max)
        if not np.any(valid):
            continue

        frac_valid = frac[valid]

        # Cation constraints -- vectorized; a candidate is kept only if
        # it passes EVERY specie's constraint (logical AND).
        keep = np.ones(len(frac_valid), dtype=bool)
        if min_cation_distances is not None:
            for specie, min_distance in min_cation_distances.items():
                min_distance = float(min_distance)
                coords = cation_frac[specie]

                dist_matrix = structure.lattice.get_all_distances(frac_valid, coords)

                d_cat = dist_matrix.min(axis=1)
                keep &= d_cat >= min_distance

        candidates.extend(frac_valid[keep])

    return np.asarray(candidates[:target], dtype=float)

# %%
def target_sites(
    structure: Structure,
    target_coords: npt.ArrayLike,
    n_samples: int = 10000,
    target_distance: Tuple[float, float] = (0.9, 1.11),
    min_species_distances: Optional[Mapping[str, float]] = None,
    seed: Optional[int] = 42,
    batch_factor: int = 10,
    max_attempts: int = 100,
) -> npt.NDArray[np.float64]:
    """
    Generate candidate sites near arbitrary fractional coordinates.

    Candidate fractional coordinates are accepted if they satisfy:
    1. Distance to the *nearest* coordinate in ``target_coords`` falls
       within ``target_distance`` [min, max].
    2. Distance to *every* species in ``min_species_distances`` is at or
       above that species' minimum distance (logical AND).

    Parameters
    ----------
    structure
        Host crystal structure. Used for lattice geometry and periodic
        boundary conditions (and to evaluate species exclusion distances).
    target_coords
        Fractional coordinate(s) around which to sample. Can be a single 1D
        array-like of shape (3,) or a 2D array-like of shape (N, 3).
        The coordinates do NOT need to correspond to existing sites in ``structure``.
    n_samples
        Number of accepted candidate sites to return.
    target_distance
        ``(min, max)`` allowed distance (Angstrom) to the nearest target
        coordinate. Must satisfy ``0 <= min < max``.
    min_species_distances
        Mapping of species symbol -> minimum allowed distance (Angstrom)
        from that species present in ``structure``. Example: ``{"Li": 1.5, "O": 0.8}``.
    seed
        Seed for ``numpy.random.default_rng``.
    batch_factor
        Candidate points drawn per batch as a multiple of remaining samples needed.
    max_attempts
        Maximum number of batches to draw before raising an error.

    Returns
    -------
    numpy.ndarray of shape (n_samples, 3)
        Fractional coordinates of accepted candidate sites within the unit cell.
    """
    # 1. Validation & Inputs Preparation
    if n_samples <= 0:
        raise ValueError("'n_samples' must be positive.")

    if len(target_distance) != 2:
        raise ValueError("'target_distance' must contain exactly two values.")

    t_min, t_max = map(float, target_distance)
    if t_min < 0 or t_max <= t_min:
        raise ValueError("'target_distance' must satisfy 0 <= min < max.")

    # Format and wrap target coordinates into a 2D array (N, 3) mod 1
    target_frac = np.atleast_2d(target_coords).astype(np.float64) % 1.0
    if target_frac.ndim != 2 or target_frac.shape[1] != 3:
        raise ValueError("'target_coords' must have shape (3,) or (N, 3).")

    # 2. Map existing species in host structure for exclusion constraints
    species_frac: Dict[str, npt.NDArray[np.float64]] = {}
    if min_species_distances is not None:
        for specie, distance in min_species_distances.items():
            if float(distance) < 0:
                raise ValueError(f"Minimum distance for {specie} cannot be negative.")

            indices = [
                i for i, site in enumerate(structure) if site.species_string == specie
            ]
            if not indices:
                raise ValueError(f"Species '{specie}' is not present in the structure.")

            species_frac[specie] = structure.frac_coords[indices]

    # 3. Rejection Sampling Loop
    rng = np.random.default_rng(seed)
    candidates: List[npt.NDArray[np.float64]] = []

    target = n_samples
    attempts = 0

    while len(candidates) < target:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                f"Could only generate {len(candidates)} candidate sites after "
                f"{max_attempts} batches. The distance constraints may be too "
                "restrictive."
            )

        n_batch = max(batch_factor * (target - len(candidates)), batch_factor)
        frac = rng.random((n_batch, 3))

        # Target distance condition (Periodic Boundary Conditions handled by pymatgen lattice)
        target_dist_matrix = structure.lattice.get_all_distances(frac, target_frac)
        target_distances = target_dist_matrix.min(axis=1)

        valid = (target_distances >= t_min) & (target_distances <= t_max)
        if not np.any(valid):
            continue

        frac_valid = frac[valid]

        # Species exclusion conditions
        keep = np.ones(len(frac_valid), dtype=bool)
        if min_species_distances is not None:
            for specie, min_distance in min_species_distances.items():
                min_distance = float(min_distance)
                coords = species_frac[specie]

                dist_matrix = structure.lattice.get_all_distances(frac_valid, coords)
                d_specie = dist_matrix.min(axis=1)
                keep &= d_specie >= min_distance

        candidates.extend(frac_valid[keep])

    return np.asarray(candidates[:target], dtype=np.float64)