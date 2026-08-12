# %%
import spglib
import numpy as np
import numpy.typing as npt
from collections import defaultdict
from typing import Sequence, Optional, Union, Dict, List

from ase import Atoms
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

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
def get_site_labels(atoms: Atoms) -> List[str]:
    """Per-atom label distinguishing inequivalent sites of the same element.

    Uses ASE's `spacegroup_kinds` array (populated automatically when
    reading a CIF with symmetry via `ase.io.read`) to detect when an
    element occupies more than one crystallographically distinct site.

    - Elements with a single site keep their plain symbol (e.g. 'Ba', 'Y').
    - Elements with multiple distinct sites get a numbered suffix, in
      order of first appearance in `atoms` (e.g. 'Cu1', 'Cu2', 'O1', 'O2').

    If `atoms` has no `spacegroup_kinds` array (e.g. it wasn't read from a
    CIF, or symmetry info wasn't kept), every atom just gets its plain
    element symbol -- i.e. site-level distinction is unavailable and
    charge lookups fall back to per-element behaviour automatically.
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
        seen = []
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
    charges: Dict, 
    strict: bool =False
) -> List[str]:
    """Check that every atom in `atoms` resolves to a charge in `charges`.

    Mirrors the same lookup order used by `_replicate_lattice` in
    point_charge.py (site label first, e.g. 'O2', then plain element
    symbol, e.g. 'O') and reports any atoms that would silently resolve
    to `None` (and therefore be DROPPED from the replicated lattice
    rather than raising an error).

    Parameters
    ----------
    strict : bool, default=False
        If True, raise a ValueError listing the unresolved site labels.
        If False, just return the list of unresolved labels (empty list
        means everything is covered) so the caller can decide what to do
        (e.g. print a warning).

    Returns
    -------
    list of str
        Unique site labels (e.g. ['O2']) present in `atoms` that do NOT
        resolve to a charge, either directly or via the plain-element
        fallback.
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