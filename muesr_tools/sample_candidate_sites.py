# %%
from __future__ import annotations

from typing import Mapping, Sequence, Optional, Tuple

import numpy as np
import numpy.typing as npt

from pymatgen.core import Structure
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# %%
def find_equivalent_positions(frac_coords, host_lattice, atol=1e-3):
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


def prune_atoms_too_close(grid_coords, host_lattice, min_distance):
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
def sample_candidate_sites(
    structure: Structure,
    n_samples: int = 10000,
    O_distance: tuple[float, float] = (0.9, 1.11),
    min_cation_distances: Mapping[str, float] | None = None,
    O_specie: Sequence[str] = ("O",),
    cation_specie: Sequence[str] | None = None,
    seed: int | None = 42,
    batch_factor: int = 10,
    max_attempts: int = 100,
) -> npt.NDArray[np.float64]:
    """
    Generate candidate muon stopping sites.

    Candidate sites satisfy:

    1. Distance to the nearest Observed/`O` is within ``O_distance``.
    2. Distance to every specified cation species is larger than its
       corresponding minimum distance (ALL species in
       ``min_cation_distances`` must individually be satisfied -- a
       candidate is only accepted if EVERY entry in the dict passes).

    All distances are in Angstrom.

    Parameters
    ----------
    structure: Structure
        Pymatgen crystal structure.
    n_samples: int
        Number of accepted candidate sites.
    O_distance: tuple(float, float)
        Minimum and maximum allowed distance to the nearest oxygen.
    min_cation_distances: dict
        Minimum distance for each cation species.
        Example
        -------
        {
            "Sr": 1.5,
            "Zn": 1.5,
            "Re": 1.5,
        }
        If None, no cation-distance constraint is applied.
    O_specie: tuple(str)
        Species considered to Observe/generate positions.
    cation_specie: tuple(str)
        Species to use as cations. If None, all non-`O_specie` species
        are treated as cations.
    seed: int
        Random seed.
    batch_factor: int
        Number of random points generated per requested sample.
    max_attempts: int
        Maximum number of batches before stopping.

    Returns
    -------
    numpy.ndarray
        Fractional coordinates with shape ``(n_samples, 3)``.

    Raises
    ------
    ValueError
        If the requested number of candidate sites cannot be generated.
    """

    # Validate input
    if n_samples <= 0:
        raise ValueError("'n_samples' must be positive.")

    if len(O_distance) != 2:
        raise ValueError(
            "'O_distance' must contain exactly two values."
        )

    O_min, O_max = map(float, O_distance)

    if O_min < 0 or O_max <= O_min:
        raise ValueError(
            "'O_distance' must satisfy "
            "0 <= min < max."
        )

    if batch_factor <= 0:
        raise ValueError("'batch_factor' must be positive.")

    if max_attempts <= 0:
        raise ValueError("'max_attempts' must be positive.")

    # Build species indices
    O_specie = tuple(O_specie)

    if cation_specie is None:
        cation_specie = tuple(
            sorted(
                {
                    site.species_string
                    for site in structure
                    if site.species_string not in O_specie
                }
            )
        )
    else:
        cation_specie = tuple(cation_specie)

    # Extract Fractional coordinates
    O_indices = [
        i
        for i, site in enumerate(structure)
        if site.species_string in O_specie
    ]

    if not O_indices:
        raise ValueError(
            f"No atoms found for species {O_specie}."
        )

    O_frac = structure.frac_coords[O_indices]

    cation_frac = {}

    for specie in cation_specie:

        indices = [
            i
            for i, site in enumerate(structure)
            if site.species_string == specie
        ]

        if not indices:
            raise ValueError(
                f"No atoms with species '{specie}' "
                "were found in the structure."
            )

        cation_frac[specie] = structure.frac_coords[indices]

    #
    # If min_cation_distances is given, validate every requested species
    # actually has coordinates available -- fail loudly up front rather
    # than silently skipping a species with no constraint applied.
    #
    if min_cation_distances is not None:
        for specie in min_cation_distances:
            if specie not in cation_frac:
                raise ValueError(
                    f"Cation species '{specie}' "
                    "is not present in the structure."
                )
            if float(min_cation_distances[specie]) < 0:
                raise ValueError(
                    f"Minimum distance for {specie} "
                    "cannot be negative."
                )

    # Random number generator
    rng = np.random.default_rng(seed)

    candidates: list[npt.NDArray[np.float64]] = []

    target = n_samples
    attempts = 0

    # Generate candidates
    while len(candidates) < target:

        attempts += 1

        if attempts > max_attempts:
            raise ValueError(
                f"Could only generate {len(candidates)} "
                f"candidate sites after {max_attempts} batches. "
                "The distance constraints may be too restrictive."
            )

        n_batch = max(
            batch_factor * (target - len(candidates)),
            batch_factor,
        )

        # Random fractional coordinates
        frac = rng.random((n_batch, 3))

        # Distance to nearest oxygen -- vectorized, fractional coords

        #
        # (n_batch, n_O) matrix of periodic distances, one call instead
        # of a Python loop over every candidate x every oxygen atom
        #
        O_dist_matrix = structure.lattice.get_all_distances(frac, O_frac)
        O_distances = O_dist_matrix.min(axis=1)

        valid = (O_distances >= O_min) & (O_distances <= O_max)

        if not np.any(valid):
            continue

        frac_valid = frac[valid]

        # 
        # Cation constraints -- vectorized, fractional coords.
        # A candidate is kept only if it passes EVERY specie's
        # constraint (logical AND across all entries in the dict).
        # 
        keep = np.ones(len(frac_valid), dtype=bool)

        if min_cation_distances is not None:
            for specie, min_distance in min_cation_distances.items():
                min_distance = float(min_distance)
                coords = cation_frac[specie]

                # (n_valid, n_cation_atoms) periodic distance matrix
                dist_matrix = structure.lattice.get_all_distances(
                    frac_valid, coords
                )
                d_cat = dist_matrix.min(axis=1)

                keep &= d_cat >= min_distance

        accepted = frac_valid[keep]

        candidates.extend(accepted)

    return np.asarray(candidates[:target], dtype=float,)