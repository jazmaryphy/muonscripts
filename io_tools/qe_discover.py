# %%
"""Find Quantum Espresso output files in a folder, with optional 1:1 matching
against their corresponding input files, and an efficient path for picking
out only specific files from a folder containing many.

This module deliberately knows NOTHING about the CONTENTS of any QE file
-- no namelist parsing, no calculation-type detection, no convergence
checking. It only ever looks at the FILESYSTEM (which files exist, which
extensions they have). That's what keeps it dependency-light and easy to
reuse or test in isolation from the rest of `io_tools`.
"""

from pathlib import Path
from typing import Iterable

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