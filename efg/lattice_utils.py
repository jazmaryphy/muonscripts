# %%
import numpy as np

# %%
def get_site_labels(atoms):
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


def check_charges_cover_atoms(atoms, charges, strict=False):
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