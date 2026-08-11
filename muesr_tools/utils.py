# %%
import spglib
import numpy as np
import numpy.typing as npt
from collections import defaultdict
from typing import Sequence, Optional, Union, Dict, List

from ase import Atoms
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from dataclasses import dataclass

# %%
@dataclass(slots=True)
class LocalFieldResults:
    """Container for local field contributions."""

    # Field contributions
    total: npt.NDArray[np.float64]
    dipolar: npt.NDArray[np.float64]
    lorentz: npt.NDArray[np.float64]
    contact: npt.NDArray[np.float64]
    dipolar_tot: npt.NDArray[np.float64]

    # Field magnitudes
    total_norm: npt.NDArray[np.float64]
    dipolar_norm: npt.NDArray[np.float64]
    lorentz_norm: npt.NDArray[np.float64]
    contact_norm: npt.NDArray[np.float64]
    dipolar_tot_norm: npt.NDArray[np.float64]
    
    # Metadata
    s_axis: npt.NDArray[np.float64]
    muon_positions: npt.NDArray[np.float64]

    # Optional structural correction
    dipolar_correction: npt.NDArray[np.float64]
    dipolar_correction_norm: npt.NDArray[np.float64]

# %%
def _label_kinds(kind_symbols: Dict[int, str]) -> Dict[int, str]:
    """
    Helper to generate distinct labels for symmetry kinds.
    
    If an element symbol appears in multiple symmetry-inequivalent kinds, 
    appends a 1-based index (e.g., 'Fe1', 'Fe2'). Otherwise, uses the plain symbol ('O').
    """
    symbol_counts = defaultdict(int)
    for symbol in kind_symbols.values():
        symbol_counts[symbol] += 1

    labels = {}
    current_indices = defaultdict(int)

    for kind_id, symbol in kind_symbols.items():
        if symbol_counts[symbol] > 1:
            current_indices[symbol] += 1
            labels[kind_id] = f"{symbol}{current_indices[symbol]}"
        else:
            labels[kind_id] = symbol

    return labels


def get_atom_kinds(
    structure: Union[Structure, Atoms], 
    symprec: float = 1e-3
) -> Dict[str, List[int]]:
    """
    Group atoms into symmetry-equivalent kinds and label them by element.

    Works with both pymatgen.core.Structure and ase.Atoms objects.

    Args:
        structure (Union[Structure, Atoms]): Structure to analyze.
        symprec (float): Symmetry-detection tolerance for symmetry operations.

    Returns:
        Dict[str, List[int]]: Mapping from kind label to the list of 0-based 
            atom indices belonging to that symmetry-equivalent kind. 
            Labels are plain element symbols if unique (e.g., "O"), or suffixed 
            with a 1-based index if multiple inequivalent sites exist (e.g., "Fe1", "Fe2").

    Raises:
        TypeError: If input structure type is not pymatgen Structure or ASE Atoms.
        RuntimeError: If symmetry evaluation fails for ASE structures.
    """
    # 1. Extract equivalent site array and species symbols based on input type
    if isinstance(structure, Structure):
        analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
        equiv = analyzer.get_symmetry_dataset().equivalent_atoms
        symbols = [site.specie.symbol for site in structure]

    elif isinstance(structure, Atoms):
        cell = (
            structure.get_cell()[:],
            structure.get_scaled_positions(),
            structure.get_atomic_numbers(),
        )
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)

        if dataset is None:
            raise RuntimeError(
                f"spglib could not determine a symmetry dataset for this "
                f"structure at symprec={symprec}."
            )

        equiv = (
            dataset.equivalent_atoms
            if hasattr(dataset, "equivalent_atoms")
            else dataset["equivalent_atoms"]
        )
        symbols = structure.get_chemical_symbols()

    else:
        raise TypeError(
            f"Unsupported structure type: {type(structure)}. "
            "Must be a pymatgen Structure or ASE Atoms object."
        )

    # 2. Group atom indices by their representative (kind) index
    kind_dict: Dict[int, List[int]] = defaultdict(list)
    for i, k in enumerate(equiv):
        kind_dict[int(k)].append(i)

    # 3. Determine element symbols for each representative kind
    kind_symbols = {k: symbols[k] for k in sorted(kind_dict)}

    # 4. Generate labeled output mapping
    labels = _label_kinds(kind_symbols)
    return {labels[k]: kind_dict[k] for k in sorted(kind_dict)}

# %%
def check_site_distances(
    structure: Structure,
    sites: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Calculate nearest-neighbour distances from each candidate
    site to every chemical species in the structure.

    Returns
    -------
    dict[str, np.ndarray]
        For each species, an array containing the nearest distance
        from every candidate site to that species.
    """

    species = list(dict.fromkeys(
        site.species_string
        for site in structure
    ))

    nearest_distances = {}

    print("\nSite-distance diagnostics:")

    for specie in species:

        indices = [
            i
            for i, site in enumerate(structure)
            if site.species_string == specie
        ]

        coords = structure.frac_coords[indices]

        distances = structure.lattice.get_all_distances(
            sites,
            coords,
        )

        nearest = distances.min(axis=1)

        nearest_distances[specie] = nearest

        print(
            f"{specie:>4s}: "
            f"min = {nearest.min():.4f} Å, "
            f"max = {nearest.max():.4f} Å, "
            f"mean = {nearest.mean():.4f} Å, "
            f"median = {np.median(nearest):.4f} Å"
        )

    return nearest_distances

# %%
def assign_label(index: str) -> str:
    """Assign a label to the muon site."""
    if index < 26:
        return chr(65 + index) 
    elif index < 52:
        return chr(65 + index - 26) + "'" 
    else:
        return chr(65 + index - 52) + "''"

# %%
def _extract_muon(
    structure: Structure,
    muon_pos: Sequence[float] | npt.NDArray[np.floating] | None,
    muon_label: str = "H",
) -> tuple[Structure, npt.NDArray[np.float64]]:
    """
    Extract the muon position and return the host structure.

    Exactly one of the following must be true:

        1. structure contains exactly one muon
        2. muon_pos is supplied

    Parameters
    ----------
    structure
        Structure with or without the muon.
    muon_pos
        Optional fractional coordinates.
    muon_label
        Species used to represent the muon.

    Returns
    -------
    host_structure
        Structure with the muon removed.
    muon_position
        Fractional coordinates.
    """

    structure = structure.copy()

    muon_indices = [
        i
        for i, site in enumerate(structure)
        if site.specie.symbol == muon_label
    ]

    # Case: muon already inside structure
    if len(muon_indices) == 1:

        if muon_pos is not None:
            raise ValueError(
                "Muon supplied twice. "
                "The structure already contains an H atom and "
                "'muon_pos' was also provided."
            )

        idx = muon_indices[0]

        pos = np.asarray(structure[idx].frac_coords)

        structure.remove_sites([idx])

        return structure, pos

    # Multiple muons
    if len(muon_indices) > 1:
        raise ValueError(
            f"Found {len(muon_indices)} '{muon_label}' atoms. "
            "Only one implanted muon is expected."
        )

    # No muon inside structure
    if muon_pos is None:
        raise ValueError(
            "No muon found in the structure.\n"
            "Either include exactly one H atom or provide "
            "'muon_pos'."
        )

    pos = np.asarray(muon_pos, dtype=float).reshape(-1)

    if pos.size != 3:
        raise ValueError(
            "'muon_pos' must contain exactly three "
            "fractional coordinates."
        )

    return structure, pos


def _validate_muon_positions(
    muon_positions: Sequence[float] | npt.NDArray[np.floating],
) -> npt.NDArray[np.float64]:

    positions = np.asarray(muon_positions, dtype=float)

    if positions.ndim == 1:
        if positions.size != 3:
            raise ValueError(
                "'muon_positions' must contain 3 fractional coordinates."
            )
        positions = positions.reshape(1, 3)

    elif positions.ndim == 2:
        if positions.shape[1] != 3:
            raise ValueError(
                "'muon_positions' must have shape (N, 3)."
            )

    else:
        raise ValueError(
            "'muon_positions' must be a 1D or 2D array."
        )

    if not np.all(np.isfinite(positions)):
        raise ValueError(
            "'muon_positions' contains non-finite values."
        )

    return positions

# %%
def _validate_sphere_radius(radius: int) -> int:
    """
    Validate supercell radius.
    """

    radius = int(radius)

    if radius <= 0:
        raise ValueError(
            "'sphere_r' must be a positive integer."
        )

    return radius
    

def _validate_k_vector(
    k: Optional[Sequence[float] | npt.NDArray[np.floating]],
) -> npt.NDArray[np.float64]:
    """
    Validate a magnetic propagation vector.

    Parameters
    ----------
    k
        None or an array-like object containing exactly three components.

    Returns
    -------
    numpy.ndarray
        A (3,) float64 array.

    Raises
    ------
    ValueError
        If `k` does not contain exactly three components.
    """
    if k is None:
        return np.zeros(3, dtype=float)

    k = np.asarray(k, dtype=float).reshape(-1)

    if k.size != 3:
        raise ValueError(
            f"'k' must contain exactly 3 components, got {k.size}."
        )

    return k

    
def _validate_magmoms(
    structure: Structure,
    magmoms: Sequence[float] | npt.NDArray[np.floating],
) -> npt.NDArray[np.float64]:
    """
    Validate magnetic moments.

    Accepts either (N,) & (N,3)

    where N is the number of atoms.
    """
    magmoms = np.asarray(magmoms, dtype=float)
    nsites = len(structure)

    if magmoms.ndim == 1:
        if magmoms.size != nsites:
            raise ValueError(
                f"Expected {nsites} magnetic moments, "
                f"received {magmoms.size}."
            )

    elif magmoms.ndim == 2:
        if magmoms.shape != (nsites, 3):
            raise ValueError(
                f"Expected magnetic moments of shape "
                f"({nsites},3), got {magmoms.shape}."
            )

    else:
        raise ValueError(
            "'magmoms' must have shape (N,) or (N,3)."
        )

    return magmoms