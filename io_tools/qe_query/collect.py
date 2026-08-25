# %%
"""Collect QE relaxation-type (or any-type) outputs from a folder: find them
(`qe_discover`), classify + check convergence (`qe_classify`), extract a
system signature (`qe_signature`), group by system, filter down to what you
want, and (optionally) load the relaxed structures + energies via ASE.

This is the only module in the `io_tools` QE split that depends on the
other three -- everything else is independently importable/testable.
"""

import collections
from pathlib import Path

from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

from io_tools.qe_query.discover import find_qe_files
from io_tools.qe_query.classify import (
    _is_good,
    check_qe_convergence,
    _normalize_calc_types,
    detect_qe_calc_type_from_input,
    detect_qe_calc_type_from_output,
)
from io_tools.qe_query.signature import (
    _get_input_system_signature,
    _get_output_system_signature,
)
from io_tools.qe import read_qe_namelists

# %%
def iter_paths(data):
    """Yield paths from either a flat list of paths, or a dict
    {signature: [paths, ...]} as returned by `get_relax_after_run(group_by_system=True)`.

    This is what lets downstream loaders accept EITHER a full grouped
    result, a single group's file list (a "subset"), or any other plain
    list of paths, without the caller needing to flatten anything first.
    """
    if isinstance(data, dict):
        for paths in data.values():
            yield from paths
    else:
        yield from data

# %%
### Scan + classify + group

# %%
def get_relax_after_run(
    folder,
    pattern="*.out",
    check_input=False,
    calc_types=("relax", "vc-relax"),
    group_by_system=True,
    verbose=True,
):
    """Collect QE outputs matching `calc_types`, filtered for a "good"
    convergence outcome, from `folder`.

    Parameters
    ----------
    check_input : bool, default=False
        True: require a matching .in file, classify from the parsed
        namelist (reliable, via `detect_qe_calc_type_from_input`).
        False: classify from .out text alone (via
        `detect_qe_calc_type_from_output` -- experimental, validated only
        for relax/vc-relax so far).
    calc_types : str, iterable of str, or None, default=('relax', 'vc-relax')
        Which QE `calculation` values to keep. A single string is accepted
        too, e.g. calc_types='scf'. None keeps every calculation type
        (filters on convergence only).
    group_by_system : bool, default=True
        If True, returns {signature: [out_path, ...]} where `signature` is
        a cheap (nat, ntyp, species) tuple -- outputs from different
        systems in the same folder are kept separate automatically, with
        no prior knowledge of what systems are present required. If
        False, returns a flat list of out_paths instead.

    Returns
    -------
    dict {signature: [Path, ...]}  (group_by_system=True)
    or  list of Path                (group_by_system=False)
    """
    calc_types = _normalize_calc_types(calc_types)
    results = []  # list of (out_path, signature)

    if check_input:
        pairs = find_qe_files(folder, pattern=pattern, check_input=True)
        for out_p, in_p in pairs:
            try:
                input_string = Path(in_p).read_text(errors="ignore")
                namelists = read_qe_namelists(input_string.lower())
            except Exception as exc:
                if verbose:
                    print(f"Skipping {out_p.name}: could not parse {in_p.name} ({exc})")
                continue

            calc_info = detect_qe_calc_type_from_input(namelists=namelists)
            if not calc_info["is_pw_input"]:
                if verbose:
                    print(f"Skipping {out_p.name}: {calc_info['error']}")
                continue
            if calc_types is not None and calc_info["calc_type"] not in calc_types:
                continue

            conv_info = check_qe_convergence(out_p)
            if not _is_good(conv_info):
                if verbose:
                    print(f"Skipping {out_p.name}: {conv_info}")
                continue

            sig = _get_input_system_signature(in_p)
            if sig is None:
                if verbose:
                    print(
                        f"Skipping {out_p.name}: could not determine a system "
                        f"signature from {in_p.name} (missing/incomplete &SYSTEM)"
                    )
                continue
            results.append((out_p, sig))

    else:
        out_files = find_qe_files(folder, pattern=pattern)
        for out_p in out_files:
            calc_info = detect_qe_calc_type_from_output(out_p)
            if calc_types is not None and calc_info.get("calc_type") not in calc_types:
                continue

            conv_info = check_qe_convergence(out_p)
            if not _is_good(conv_info):
                if verbose:
                    print(f"Skipping {out_p.name}: {conv_info}")
                continue

            sig = _get_output_system_signature(out_p)
            results.append((out_p, sig))

    if not group_by_system:
        return [p for p, _sig in results]

    groups = collections.defaultdict(list)
    for p, sig in results:
        groups[sig].append(p)
    return dict(groups)

# %%
### Filter + summarize

# %%
def select_groups(groups, species=None, nat=None, ntyp=None, min_runs=None):
    """Filter a `get_relax_after_run(..., group_by_system=True)` result down
    to the system(s) you actually want, using whichever criteria you know.

    Parameters
    ----------
    groups : dict {(nat, ntyp, species): [Path, ...]}
    species : iterable of str, optional
        Keep only groups whose species set matches exactly (order and
        case don't matter -- matched via a sorted, lowercased tuple, the
        same normalization the signatures already use).
    nat : int or (int, int), optional
        Keep only groups with this exact atom count, or with `nat` inside
        the given (min, max) inclusive range.
    ntyp : int, optional
        Keep only groups with exactly this many distinct species.
    min_runs : int, optional
        Drop groups with fewer than this many converged runs -- useful to
        skip one-off/singleton systems you don't have enough statistics
        on yet.

    Returns
    -------
    dict, same shape as `groups`, containing only the matching entries.
    """
    if species is not None:
        species = tuple(sorted(s.lower() for s in species))
    if isinstance(nat, int):
        nat = (nat, nat)

    selected = {}
    for sig, paths in groups.items():
        g_nat, g_ntyp, g_species = sig

        if species is not None and g_species != species:
            continue
        if nat is not None:
            if g_nat is None or not (nat[0] <= g_nat <= nat[1]):
                continue
        if ntyp is not None and g_ntyp != ntyp:
            continue
        if min_runs is not None and len(paths) < min_runs:
            continue

        selected[sig] = paths

    return selected


def print_relax_summary(groups, sort_by="count"):
    """Pretty-print the output of `get_relax_after_run(..., group_by_system=True)`.

    Parameters
    ----------
    groups : dict {(nat, ntyp, species): [Path, ...]}
    sort_by : 'count' or 'nat', default='count'
        'count' -> largest groups (most files) first.
        'nat'   -> largest systems (most atoms) first.
    """
    if not groups:
        print("No matching, converged files found.")
        return

    def formula_str(species):
        if not species:
            return "(unknown composition)"
        return "-".join(s.capitalize() for s in species)

    rows = []
    for (nat, ntyp, species), paths in groups.items():
        rows.append((formula_str(species), nat, ntyp, len(paths)))

    key = (lambda r: -r[3]) if sort_by == "count" else (lambda r: -(r[1] or 0))
    rows.sort(key=key)

    formula_w = max(len(r[0]) for r in rows)
    for formula, nat, ntyp, n in rows:
        nat_str = str(nat) if nat is not None else "?"
        ntyp_str = str(ntyp) if ntyp is not None else "?"
        print(f"  {formula:<{formula_w}}  nat ={nat_str:>4}  ntyp ={ntyp_str:>2}  "
              f"-> {n} converged run{'s' if n != 1 else ''}")

    total = sum(r[3] for r in rows)
    print(f"  {'-' * (formula_w + 30)}")
    print(f"  {len(rows)} distinct system(s), {total} converged run(s) total")

# %%
### Load structures + energies

# %%
def load_relax_from_folder(
    folder,
    pattern="*.out",
    check_input=False,
    calc_types="relax",
    group_by_system=True,
    pymatgen=True,
    verbose=True,
    **select_kwargs,
):
    """Scan `folder` for converged QE outputs matching `calc_types`, filter
    them via `select_kwargs` (forwarded to `select_groups`), and load the
    relaxed structure + energy for each match.

    Parameters
    ----------
    **select_kwargs :
        Passed through to `select_groups` (e.g. species=['H','V','Si'],
        nat=161 or (min_atoms, max_atoms), ntyp=3, min_runs=1).
        Default: species=None, nat=None, ntyp=None, min_runs=None.

    Returns
    -------
    list of dict
        Each dict has keys 'idx' (file stem), 'rlxd_struct' (pymatgen
        Structure if `pymatgen=True`, else ASE Atoms), and 'energy' (eV).
    """
    groups = get_relax_after_run(
        folder,
        pattern=pattern,
        check_input=check_input,
        calc_types=calc_types,
        group_by_system=group_by_system,
        verbose=verbose,
    )
    subset = select_groups(groups, **select_kwargs)

    rlxd_results = []

    for path in iter_paths(subset):
        path = Path(path)

        try:
            atoms = read(path, index=-1)  # final (relaxed) image only
        except Exception as exc:
            if verbose:
                print(f"Skipping {path.name}: could not read structure ({exc})")
            continue

        try:
            energy = atoms.get_potential_energy()
        except Exception as exc:
            if verbose:
                print(f"Skipping {path.name}: no energy available ({exc})")
            continue

        rlxs = atoms.copy()
        if pymatgen:
            rlxs = AseAtomsAdaptor.get_structure(atoms)

        rlxd_results.append({
            "idx": path.stem,
            "rlxd_struct": rlxs.as_dict(),
            "energy": energy,
        })

    return rlxd_results

# %%
### FUTURE CODES

# %%
def load_scf_from_folder():
    """FUTURE"""
    return


def load_nscf_from_folder():
    """FUTURE"""
    return