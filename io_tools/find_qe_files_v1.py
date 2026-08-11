# %%
"""List Quantum Espresso output files in a folder, with optional 1:1 matching
against their corresponding input files, and an efficient path for picking
out only specific files from a folder containing many.
"""
import collections
from ase.io import read
from pathlib import Path
from typing import Iterable
from io_tools.qe import read_qe_namelists
from io_tools.base import find_first_index
from pymatgen.io.ase import AseAtomsAdaptor

# %%
RELAX_LIKE = {"relax", "vc-relax"}
MD_LIKE = {"md", "vc-md"}
VARIABLE_CELL = {"vc-relax", "vc-md"}
STATIC = {"scf", "nscf", "bands"}

PW_START = "Program PWSCF"

def _classify(calc):
    calc = None if calc is None else str(calc).strip().strip("'\"").lower()
    return {
        "calc_type": calc,
        "relax": calc in RELAX_LIKE | MD_LIKE,
        "vc_relax": calc in VARIABLE_CELL,
        "static": calc in STATIC,
    }


def _is_good(conv_info):
    return conv_info["converged"] and conv_info["job_done"] and not conv_info["crashed"]


def _has_marker(text, lines, case_sensitive=False):
    """
    Check whether one or more strings exist in an iterable of lines.

    Parameters
    ----------
    text : str or iterable of str
        String or strings to search for. If multiple strings are given,
        the function returns True if any of them is found.
    lines : iterable of str
        Lines to search.
    case_sensitive : bool, default=False
        Whether the search is case-sensitive.

    Returns
    -------
    bool
        True if at least one search string is found.
    """
    if isinstance(text, str):
        text = (text,)

    if not case_sensitive:
        text = tuple(t.lower() for t in text)
        return any(
            any(t in line.lower() for t in text)
            for line in lines
        )

    return any(
        any(t in line for t in text)
        for line in lines
    )

def _normalize_calc_types(calc_types):
    """
    Normalize calculation types to a tuple.

    Parameters
    ----------
    calc_types : str, iterable of str, or None
        Calculation type(s). ``None`` means any calculation type.

    Returns
    -------
    tuple[str] or None
        Normalized calculation types.
    """
    if calc_types is None:
        return None

    if isinstance(calc_types, str):
        return (calc_types,)

    if not isinstance(calc_types, collections.abc.Iterable):
        raise TypeError(
            "`calc_types` must be a string, an iterable of strings, or None."
        )

    return tuple(calc_types)


def _iter_paths(data):
    """Yield paths from either a flat list of paths, or a dict
    {signature: [paths, ...]} as returned by `load_relax(group_by_system=True)`.
 
    This is what lets `load_subset_data` accept EITHER a full grouped
    result, a single group's file list (a "subset"), or any other plain
    list of paths, without the caller needing to flatten anything first.
    """
    if isinstance(data, dict):
        for paths in data.values():
            yield from paths
    else:
        yield from data

# %%
def find_qe_files(
    folder,
    pattern: str = "*.out",
    names: Iterable[str] | None = None,
    out_ext: str = ".out",
    check_input: bool = False,
    input_ext: str = ".in",
    strict: bool = False,
    recursive: bool = False,
):
    """Find Quantum Espresso output files, optionally paired 1:1 with inputs.

    Two ways to select which .out files you get, pick whichever fits:

    1. `pattern` (default `'*.out'`) -- any glob pattern, e.g. `'relax_*.out'`
       or `'Cu*.out'`. Good when you want "everything that looks like X"
       and don't know the exact filenames ahead of time. This scans the
       whole folder (glob/rglob), so cost scales with the TOTAL number of
       files in the folder.

    2. `names` -- an explicit list of base filenames (stem, i.e. WITHOUT
       the .out/.in extension), e.g. `['calc_001', 'calc_017', 'calc_204']`.
       When given, `pattern` is ignored entirely. This is the efficient
       option when the folder has thousands of .out files but you only
       want a handful: instead of listing/globbing the whole directory,
       it builds each path directly (`folder / f"{name}{out_ext}"`) and
       just checks existence -- cost scales with `len(names)`, not with
       how many files are in the folder.

    Parameters
    ----------
    folder : str or Path
        Directory to search.
    pattern : str, default='*.out'
        Glob pattern used to find output files. Ignored if `names` is given.
    names : iterable of str, optional
        Explicit base filenames (no extension) to look for directly.
        Efficient targeted selection -- see above.
    out_ext : str, default='.out'
        Extension of the output files (used both to interpret `pattern`'s
        results and to build paths when `names` is given).
    check_input : bool, default=False
        If True, for every output file found, also look for a matching
        input file `folder / f"{stem}{input_ext}"`. Any output file
        WITHOUT a matching input file is dropped (enforces a strict
        one-to-one out<->in correspondence, matched by identical base
        filename/stem).
    input_ext : str, default='.in'
        Extension of the input files to match against.
    strict : bool, default=False
        - With `names`: if True, raise `FileNotFoundError` if any
          requested output file doesn't exist (instead of silently
          skipping it).
        - With `check_input=True`: if True, raise `FileNotFoundError`
          listing any output files that have no matching input file,
          instead of silently dropping them (with a printed warning).
    recursive : bool, default=False
        If True and using `pattern`, search subdirectories too (`rglob`
        instead of `glob`). Ignored if `names` is given.

    Returns
    -------
    list of Path
        If `check_input=False`: sorted list of matching output file paths
        (or in the order given by `names`, if `names` was used).
    list of (Path, Path) tuples
        If `check_input=True`: (output_path, input_path) pairs, one per
        output file that had a matching input file.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a directory")

    if names is not None:
        # Targeted lookup: build paths directly, don't scan the directory.
        out_files = []
        for name in names:
            stem = Path(name).stem if Path(name).suffix else str(name)
            candidate = folder / f"{stem}{out_ext}"
            if candidate.is_file():
                out_files.append(candidate)
            elif strict:
                raise FileNotFoundError(f"Requested output file not found: {candidate}")
            # else: silently skip missing requested files (non-strict default)
    else:
        globber = folder.rglob if recursive else folder.glob
        out_files = sorted(globber(pattern))

    if not check_input:
        return out_files

    matched = []
    unmatched = []
    for out_path in out_files:
        in_path = out_path.with_name(out_path.stem + input_ext)
        if in_path.is_file():
            matched.append((out_path, in_path))
        else:
            unmatched.append(out_path)

    if unmatched:
        msg = (
            f"{len(unmatched)} output file(s) have no matching '{input_ext}' "
            f"input file and were dropped: " + ", ".join(p.name for p in unmatched)
        )
        if strict:
            raise FileNotFoundError(msg)
        print(f"WARNING: {msg}")

    return matched

# %%
def detect_qe_calc_type_from_input(in_path=None, namelists=None):
    """Classify a QE run's `calculation` type from its parsed namelists.

    Provide exactly one of `namelists` (already-parsed dict from
    `read_qe_namelists`) or `in_path` (a .in file to read and parse here).
    See `classify_qe_run` if you need to fall back to output-only
    detection when no input file is available at all.
    """  
    if namelists is None:
        if in_path is None:
            raise ValueError("Provide either `namelists` or `in_path`.")
        try:
            input_string = Path(in_path).read_text(errors="ignore")
            namelists = read_qe_namelists(input_string.lower())
        except Exception as exc:  # deliberately broad: any parser/IO failure -> "not PW input"
            result = _classify(None)
            result.update(
                source="pw_input", is_pw_input=False, error=f"{type(exc).__name__}: {exc}"
            )
            return result
 
    control = namelists.get("control") if isinstance(namelists, dict) else None
    if control is None:
        result = _classify(None)
        result.update(source="pw_input", is_pw_input=False, error="No &CONTROL namelist found")
        return result
 
    calc = control.get("calculation")  # None if &CONTROL exists but 'calculation' wasn't set
    result = _classify(calc)
    result.update(source="pw_input", is_pw_input=True, error=None)

    return result
 

def detect_qe_calc_type_from_output(out_path):
    """Best-effort `calculation`-type classification from .out text alone.
 
    EXPERIMENTAL: only validated for relax/vc-relax so far. Prefer
    `detect_qe_calc_type_from_input` whenever an input file is available.
    """
    out_path = Path(out_path)
    with out_path.open(errors="ignore") as f:
        lines = f.readlines()

    ## 1) Converged relax/vc-relax: "Begin final coordinates" block
    try:
        final_kw = "Begin final coordinates"
        start = find_first_index(final_kw, lines)
        coord_kw = "ATOMIC_POSITIONS"
        index = find_first_index(coord_kw, lines, start=start)
        try:
            end = find_first_index("End final coordinates", lines, start=start)
        except KeyError:
            end = len(lines)

        # has_cell_params = any("CELL_PARAMETERS" in line for line in lines[start:end])
        has_cell_params = _has_marker("CELL_PARAMETERS", lines[start:end])
        calc = "vc-relax" if has_cell_params else "relax"

        result = _classify(calc)
        result["source"] = "pw_output"              
        # result["confidence"] = "high"            # to check later
        return result
    except KeyError:
        pass  # no final-coordinates block -> keep looking

    ## EXPERIMENTAL 
    ## I suggests only use when interested of relax/vc-rlx
    has_dynamics = _has_marker("Entering Dynamics", lines)
    has_new_cell_volume = _has_marker("new unit-cell volume", lines)
    is_bfgs_converged = _has_marker("bfgs converged", lines)
    has_final_enthalpy = _has_marker("Final enthalpy", lines)
    is_nscf = _has_marker(
        ("Non-self-consistent Calculation", "non-self-consistent"),
        lines,
    )
    is_band_structure = _has_marker(
        ("Band Structure Calculation", "Band Symmetry"),
        lines,
    )
    n_total_energy_steps = sum(
        1 for line in lines if line.strip().startswith("!") and "total energy" in line
    )

    # --- 2) MD-type: "Entering Dynamics:" iteration headers ---
    if has_dynamics:
        calc = "vc-md" if has_new_cell_volume else "md"

        result = _classify(calc)
        result["source"] = "pw_output"              
        # result["confidence"] = "high"            # to check later
        return result

    # --- 3) Relax that didn't reach "Begin final coordinates" (e.g. cut off) ---
    if is_bfgs_converged or n_total_energy_steps > 1:
        calc = "vc-relax" if (has_new_cell_volume or has_final_enthalpy) else "relax"

        result = _classify(calc)
        result["source"] = "pw_output"              
        # result["confidence"] = "medium"            # to check later
        return result

    # --- 4) Static run: try to tell scf/nscf/bands apart, lower confidence ---
    if is_band_structure:
        calc, confidence = "bands", "medium"
    elif is_nscf:
        calc, confidence = "nscf", "medium"
    elif n_total_energy_steps >= 1:
        calc, confidence = "scf", "low"  # default assumption for scf
    else:
        result = _classify(None)
        result["source"] = "pw_output"              
        # result["confidence"] = "none"            # to check later
        return result

    result = _classify(calc)
    result["source"] = "pw_output"              
    # result["confidence"] = confidence            # to check later (TRUST CONFIDENCE)
    return result

# %%
def classify_qe_run(out_path=None, in_path=None, namelists=None):
    """Classify a QE run, using whatever information is available.

    Priority: `namelists` > `in_path` (parsed via `read_qe_namelists`) >
    `out_path` (text-heuristic fallback, for runs with no input file at
    all -- e.g. when you didn't use `check_input=True` in `find_qe_files`
    and ended up with orphaned .out files).
    """
    if namelists is not None or in_path is not None:
        return detect_qe_calc_type_from_input(in_path=in_path, namelists=namelists)

    if out_path is not None:
        return detect_qe_calc_type_from_output(out_path)

    raise ValueError("Provide at least one of `namelists`, `in_path`, or `out_path`.")


def check_qe_convergence(out_path, block_size=4096):
    """Memory-efficient, multi-job aware QE convergence checker.
    
    Reads the file backward from the bottom up to evaluate the final state 
    of the file safely, without loading the whole file into RAM.
    """
    out_path = Path(out_path)
    
    # Flags to return
    job_done = False
    crashed = False
    timed_out = False
    scf_converged = None
    ionic_converged = None

    # Flags to stop searching backward once we find the latest status
    found_scf = False
    found_ionic = False
    hit_job_boundary = False

    # Read file backward efficiently
    with out_path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        remainder = ""
        offset = file_size

        while offset > 0 and not hit_job_boundary:
            to_read = min(block_size, offset)
            offset -= to_read
            f.seek(offset)
            
            # Decode chunk safely and combine with previous remainder
            chunk = f.read(to_read).decode("utf-8", errors="ignore")
            data = chunk + remainder
            lines = data.splitlines()
            
            # Save the incomplete first line for the next iteration step
            remainder = lines[0] if offset > 0 else ""
            lines_to_process = lines[1:] if offset > 0 else lines

            # Process this chunk's lines backward
            for line in reversed(lines_to_process):
                if PW_START in line:
                    # Reached the beginning of the most recent run -- every
                    # flag found so far belongs to it; nothing further back
                    # is relevant. Stop immediately, don't process/read more.
                    hit_job_boundary = True
                    break

                # Global termination flags
                if "JOB DONE." in line:
                    job_done = True
                if "Error in routine" in line or "%%%%%%%%%%%%%%%%" in line:
                    crashed = True
                if "Maximum CPU time exceeded" in line:
                    timed_out = True

                # Ionic Convergence status (Stops at the last calculation block)
                if not found_ionic:
                    if "bfgs converged in" in line:
                        ionic_converged = True
                        found_ionic = True
                    elif "The maximum number of steps has been reached" in line:
                        ionic_converged = False
                        found_ionic = True

                # SCF Convergence status (Stops at the last calculation block)
                if not found_scf:
                    if "convergence has been achieved" in line:
                        scf_converged = True
                        found_scf = True
                    elif "convergence NOT achieved" in line:
                        scf_converged = False
                        found_scf = True

                # Multi-job isolation safeguard:
                # If we hit an older job's end banner, stop searching for convergence flags
                if "End of self-consistent calculation" in line and found_ionic:
                    # This prevents mixing flags from older concatenated runs
                    pass

            # Early exit optimization: stop reading if all latest markers are found
            if found_scf and found_ionic and (job_done or crashed or timed_out):
                break

    converged = ionic_converged if ionic_converged is not None else scf_converged

    return {
        "job_done": job_done,
        "crashed": crashed,
        "timed_out": timed_out,
        "scf_converged": scf_converged,
        "ionic_converged": ionic_converged,
        "converged": converged,
    }

# %%
def _get_input_system_signature(in_path):
    """Giving in_path parsed to namelists dict: (nat, ntyp, species).
 
    Returns None if `namelists` doesn't look like a genuine PW input --
    no &SYSTEM section at all, or &SYSTEM present but missing nat/ntyp.
    This is deliberate: returning a fake (None, None, None) tuple instead
    would make `load_relax`'s grouping silently merge together every
    unrelated file that hits this case, as if they were one system.
    Callers should treat `None` as "skip this file", not as a group key.
    """
    input_string = Path(in_path).read_text(errors="ignore")
    namelists = read_qe_namelists(input_string.lower())
    lines = input_string.splitlines()

    if not isinstance(namelists, dict):
        return None
 
    system = namelists.get("system")
    if system is None:
        return None  # no &SYSTEM section -> not recognizable as a PW input
 
    nat = system.get("nat")
    ntyp = system.get("ntyp")

    if nat is None or ntyp is None:
        return None  # &SYSTEM present but incomplete -> no reliable signature
 
    species = None
    if lines is not None:
        names = []
        try:
            index = find_first_index("ATOMIC_POSITIONS", lines)
            for i in range(index + 1, index + 1 + nat):
                row_split = lines[i].split()
                names.append(row_split[0])

            if len(names) == int(nat):
                species =  tuple(sorted(set(names)))
        except (IndexError, ValueError):
            species = None
 
    return (nat, ntyp, species)
 
 
def _get_output_system_signature(out_path):
    """Signature read straight from the .out file header, for when there's
    no matching .in file at all.
    """
    nat = ntyp = None

    out_path = Path(out_path)
    with out_path.open(errors="ignore") as f:
        lines = f.readlines()

    nat_index = find_first_index("number of atoms/cell", lines)
    nat = int(lines[nat_index].split()[-1])

    ntyp_index = find_first_index("number of atomic types", lines, start=nat_index)
    ntyp = int(lines[ntyp_index].split()[-1])

    return (nat, ntyp, None)

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
                lines = input_string.splitlines()
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
def select_groups(groups, species=None, nat=None, ntyp=None, min_runs=None):
    """Filter a `get_relax_after_run(..., group_by_system=True)` result down to the
    system(s) you actually want, using whichever criteria you know.

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
    """Pretty-print the output of `load_relax(..., group_by_system=True)`.
 
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
def load_relax_from_folder(
    folder,
    pattern="*.out",
    check_input=False,
    calc_types="relax",
    group_by_system=True,
    pymatgen=True,
    verbose=True,
    **select_kwargs
):
    """_summary_

    **select_kwargs :
        Passed through to select_groups (e.g. species='H V Si' for atomic
        types, nat=161 or (min_atoms, max_atoms) for total number of atoms, 
        ntyp=3 for number of distinct atoms, min_runs=1 for minimum number
        of converged calculations).
        Default: species=None, nat=None, ntyp=None, min_runs=None
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
    idx = rlxs = energy = None
    
    for path in _iter_paths(subset):
        path = Path(path)

        # print(" ", path.name)
        # print(path.stem, path)

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

        idx = path.stem
        rlxs =  atoms.copy()
        if pymatgen:
            rlxs = AseAtomsAdaptor.get_structure(atoms)

        rlxd_result = {
            'idx': idx,
            'rlxd_struct': rlxs,
            'energy': energy
        }
        rlxd_results.append(rlxd_result)


    return rlxd_results

# %%
def load_scf_from_folder():
    """FUTURE"""
    return

def load_nscf_from_folder():
    """FUTURE"""
    return