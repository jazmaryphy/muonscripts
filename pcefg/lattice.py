# %%
"""Lattice and crystallographic site symmetry utilities."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Sequence, Union

import spglib
import numpy as np
from ase import Atoms
import numpy.typing as npt
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# %%
def get_site_info(
    atoms: Atoms,
    position_or_index: Union[npt.ArrayLike, int],
    coords_are_cartesian: bool = False,
    atol: float = 1e-3,
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    Optional[int],
    Optional[str],
]:
    """Resolve a target site into Cartesian/fractional coordinates and site metadata.

    Args:
        atoms: ASE `Atoms` object representing the crystal lattice.
        position_or_index: Target position given as fractional/Cartesian coordinates,
            or an integer atom index in `atoms`.
        coords_are_cartesian: If True, treats coordinate input as Cartesian (Å).
            If False, treats input as fractional unit-cell coordinates [0, 1).
        atol: Position tolerance in Ångströms to identify a matching atom index/symbol.

    Returns:
        Tuple containing:
            - cart_pos: Cartesian position array in Å, shape `(3,)`.
            - frac_pos: Fractional position array, shape `(3,)`.
            - site_index: Matched atom index in structure, or None if off-lattice.
            - site_symbol: Matched chemical symbol, or None if off-lattice.
    """
    site_index: Optional[int] = None
    site_symbol: Optional[str] = None

    # Option A: Passed an integer atom index directly
    if isinstance(position_or_index, (int, np.integer)):
        site_index = int(position_or_index)
        site_symbol = str(atoms.symbols[site_index])
        cart_pos = atoms.positions[site_index].copy()
        frac_pos = atoms.get_scaled_positions()[site_index].copy()

    # Option B: Passed a coordinate vector (fractional or Cartesian)
    else:
        pos_arr = np.asarray(position_or_index, dtype=np.float64)
        if coords_are_cartesian:
            cart_pos = pos_arr
            frac_pos = atoms.cell.scaled_positions(cart_pos)
        else:
            frac_pos = pos_arr
            cart_pos = atoms.cell.cartesian_positions(frac_pos)

        # Spatial lookup: check if coordinates match an existing atom site
        if len(atoms) > 0:
            distances = np.linalg.norm(atoms.positions - cart_pos, axis=1)
            min_idx = int(np.argmin(distances))
            if distances[min_idx] <= atol:
                site_index = min_idx
                site_symbol = str(atoms.symbols[min_idx])

    return cart_pos, frac_pos, site_index, site_symbol

# %%
def _label_kinds(
    kind_symbols: Dict[int, str]
) -> Dict[int, str]:
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


def get_site_labels(atoms: Atoms) -> List[str]:
    """Generate per-atom site labels incorporating crystallographic site distinction.

    Uses ASE's `spacegroup_kinds` array if populated (e.g., from CIF import), falling
    back to plain element symbols if symmetry information is absent.

    Args:
        atoms: Input ASE `Atoms` object.

    Returns:
        List of per-atom site strings (e.g., `['Cu1', 'Cu2', 'O', 'O']`).
    """
    symbols = np.array(atoms.get_chemical_symbols())
    kinds = atoms.arrays.get("spacegroup_kinds")

    if kinds is None:
        return list(symbols)

    labels = np.empty(len(symbols), dtype=object)

    for sym in set(symbols):
        elem_mask = symbols == sym
        elem_kinds = kinds[elem_mask]

        # Unique kinds for this element, in order of first appearance
        seen: List[int] = []
        for k in elem_kinds:
            if k not in seen:
                seen.append(k)
        kind_to_num = {k: i + 1 for i, k in enumerate(seen)}

        multi_site = len(seen) > 1
        for idx, k in zip(np.where(elem_mask)[0], elem_kinds):
            labels[idx] = f"{sym}{kind_to_num[k]}" if multi_site else sym

    return list(labels)


def check_charges_cover_atoms(
    atoms: Atoms,
    charges: Dict[str, float],
    strict: bool = False,
) -> List[str]:
    """Verify that every atom in `atoms` maps to an entry in `charges`.

    Args:
        atoms: Input ASE `Atoms` structure.
        charges: Mapping of element symbols or site labels to formal charges.
        strict: If True, raises ValueError if any unresolved sites are present.

    Returns:
        List of missing site/element label strings.

    Raises:
        ValueError: If `strict=True` and unassigned charges exist.
    """
    site_labels = get_site_labels(atoms)
    symbols = atoms.get_chemical_symbols()

    missing = sorted(
        {
            label
            for label, sym in zip(site_labels, symbols)
            if charges.get(label, charges.get(sym)) is None
        }
    )

    if missing and strict:
        raise ValueError(
            f"charges dict does not cover these site(s), atoms would be "
            f"silently dropped from the replicated lattice: {missing}. "
            f"Add an entry for each (either the site label, e.g. 'O2', "
            f"or the plain element symbol, e.g. 'O')."
        )

    # workaround for weird behavior of charges strings: i.e np.str_('symbol)
    missing = [f"{symbol}" for symbol in missing]
    return missing