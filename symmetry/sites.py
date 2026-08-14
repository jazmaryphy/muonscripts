# %%
import spglib
import numpy as np
import numpy.typing as npt
from collections import defaultdict
from numpy.typing import ArrayLike, NDArray
from typing import Sequence, Literal, Optional, Tuple, Union, Dict, List

from ase import Atoms
from pymatgen.core import Structure
from pymatgen.symmetry import analyzer
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

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

# %%
def find_defect_index(
    structure: Structure,
    defect_label: str="H",
    which: Literal["first", "last", "unique"] = "last"
) -> Optional[int]:
    """
    Return the index of a defect atom in a structure.

    The defect is identified by its species label. If multiple atoms with
    the same label are present, the ``which`` argument determines which
    index is returned.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        Structure containing the defect atom.
    defect_label : str, default="H"
        Species label used to identify the defect.
    which : {"first", "last", "unique"}, default="last"
        Selection rule when multiple matching atoms exist.

        - ``"first"`` : return the first matching atom.
        - ``"last"`` : return the last matching atom.
        - ``"unique"`` : require exactly one matching atom, otherwise
          raise a ``ValueError``.

    Returns
    -------
    int or None
        Index of the defect atom, or ``None`` if no matching atom is
        found.

    Raises
    ------
    ValueError
        If ``which="unique"`` and zero or multiple matching atoms are
        found.
    """
    matches = [
        i
        for i, site in enumerate(structure)
        if site.specie.symbol == defect_label
    ]

    if which == "unique":
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one '{defect_label}' atom, "
                f"found {len(matches)}."
            )
        return matches[0]

    if not matches:
        return None

    if which == "first":
        return matches[0]

    if which == "last":
        return matches[-1]

    raise ValueError(
        f"Unknown value for 'which': {which!r}. "
        "Expected 'first', 'last', or 'unique'."
    )

    
def get_muon_index(structure: Structure) -> Optional[int]:
    return find_defect_index(structure, defect_label="H", which="last")

# %%
def prune_too_close_pos(
    frac_positions: ArrayLike,
    host_lattice: Structure,
    min_distance: float,
    energies: Optional[ArrayLike] = None,
    e_tol: float = 0.05,
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Return indices and mappings of atoms too close to another one in the cell.

    If energies are not passed, only inter-atomic distance is considered.
    Otherwise both conditions (distance and same energy) must be verified.

    Parameters
    ----------
    frac_positions : numpy.array
        The N x 3 array containing scaled atomic positions.

    host_lattice : pymatgen.core.Structure
        The lattice structure. Only its lattice property is used.

    min_distance: float
         Minimum distance in Angstrom between atoms. Atoms closer than this
         will be considered the same unless they have different energy associated.

    energies: list or numpy.array
         Energy (or any other scalar property) associated with positions
         reported in frac_positions.

    e_tol: float
        Absolute difference between the scalar property associated with atomic sites.

    Returns
    -------
    s_idx : ndarray
        An array of integers. If the value of the item equals its index, 
        the atom is unique. If the value is -1, it is close to a prior atom.
    mapping : ndarray
        An array of integers mapping each atom to the index of the first 
        matching unique atom that satisfied the distance/scalar conditions.

    Suggestions:
                 1. modify -1 into the index of the first atom that matched the conditions
                    on energy and distance. -> this is the mapping
                 2. change `energies` into `scalar_value` to make it more general.

    """

    # energies and tolerance should be in eV
    lattice = host_lattice.lattice

    s_idx = np.arange(len(frac_positions))
    mapping = np.arange(len(frac_positions)) + 1  
    mapping[0] = 1

    for i, pi in enumerate(frac_positions):
        for j, pj in enumerate(frac_positions):
            if j > i:
                diff = pbc_shortest_vectors(lattice, pi, pj).squeeze()
                # print(i,j,diff,np.linalg.norm(diff, axis=0))
                if (energies is not None) and (len(energies) == len(frac_positions)):
                    if (np.linalg.norm(diff, axis=0) < min_distance) and (
                        abs(energies[i] - energies[j]) < e_tol
                    ):
                        s_idx[j] = -1
                        
                        mapping[j] = mapping[i]
                        #print(i,j,mapping)
                else:
                    if np.linalg.norm(diff, axis=0) < min_distance:
                        s_idx[j] = -1
                        mapping[j] = mapping[i]

    return s_idx, mapping


def get_equivalent_sites(
    fcoords: ArrayLike,
    host_lattice: Structure,
    min_distance: float = 0.5,
    symprec: float = 1e-3,
    energies: Optional[ArrayLike] = None,
    e_tol: float = 0.05,
) -> NDArray[np.float64]:
    """Generate symmetry-equivalent muon positions and remove duplicate/near-equivalent sites.

    Parameters
    ----------
    fcoords : (N, 3) array_like
        Input fractional coordinates.
    host_lattice : Structure
        Host crystal structure.
    min_distance : float
        Minimum separation (Å) used for deduplication.
    symprec : float
        Symmetry tolerance.
    energies : array_like, optional
        Site energies.
    e_tol : float
        Energy tolerance for site merging.

    Returns
    -------
    ndarray
        Unique fractional coordinates.
    """

    fcoords = np.atleast_2d(fcoords) % 1.0

    spg = analyzer.SpacegroupAnalyzer(host_lattice, symprec=symprec,)
    ops = spg.get_symmetry_operations(cartesian=False)

    replicas = np.vstack([
        op.operate_multi(fcoords) % 1.0
        for op in ops
    ])

    # If energies are provided, they must be duplicated symmetrically to match replicas
    replica_energies = None
    if energies is not None:
        flat_energies = np.asarray(energies).flatten()
        replica_energies = np.tile(flat_energies, len(ops))


    idx, _ = prune_too_close_pos(
        replicas,
        host_lattice.copy(),
        min_distance,
        energies=replica_energies,
        e_tol=e_tol,
    )

    return replicas[idx == np.arange(len(replicas))]

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
    host_lattice: Structure,
    sites: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Calculate nearest-neighbour distances from each candidate
    site to every chemical species in the host_lattice.

    Returns
    -------
    dict[str, np.ndarray]
        For each species, an array containing the nearest distance
        from every candidate site to that species.
    """

    species = list(dict.fromkeys(
        site.species_string
        for site in host_lattice
    ))

    nearest_distances = {}

    print("\nSite-distance diagnostics:")

    for specie in species:

        indices = [
            i
            for i, site in enumerate(host_lattice)
            if site.species_string == specie
        ]

        coords = host_lattice.frac_coords[indices]

        distances = host_lattice.lattice.get_all_distances(
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