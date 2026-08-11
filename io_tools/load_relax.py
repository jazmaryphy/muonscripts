# %%
"""Load relaxed structures + energies from a folder of QE outputs.

Usable two ways:

1. IMPORTED, in another script or notebook:

    from load_relax import load_relax

    results = load_relax(
        "/path/to/folder",
        check_input=True,
        calc_types="relax",
        species=["Na", "Cl"],   # forwarded to select_groups
    )

2. RUN DIRECTLY from the command line (see `--help`, or the examples in
   each argparse argument's help text below):

    python load_relax.py /path/to/folder --species Na Cl --output results.json
"""

import json
import pickle
import argparse
from pathlib import Path
from io_tools.find_qe_files import select_groups
from io_tools.find_qe_files import get_relax_after_run
from io_tools.find_qe_files import print_relax_summary
from io_tools.find_qe_files import load_relax_from_folder

# %%
### Importable function

# %%
def load_relax(
    path,
    pattern="*.out",
    check_input=False,
    calc_types="relax",
    group_by_system=True,
    pymatgen=True,
    verbose=True,
    **select_kwargs
):
    """Scan `path` for converged QE outputs matching `calc_types`, filter
    them down via `select_kwargs` (forwarded to `select_groups` -- e.g.
    `species=['Na','Cl']`, `nat=8`, `ntyp=2`, `min_runs=3`), and load the
    relaxed structure + energy for each match.

    Returns
    -------
    list of dict
        Each dict has keys 'idx' (file stem), 'rlxd_struct' (pymatgen
        Structure if `pymatgen=True`, else ASE Atoms), and 'energy' (eV).
    """
    results = load_relax_from_folder(
        folder=path,
        pattern=pattern,
        check_input=check_input,
        calc_types=calc_types,
        group_by_system=group_by_system,
        pymatgen=pymatgen,
        verbose=verbose,
        **select_kwargs,
    )
    return results

# %%
### Save / load results -- pickle (simplest, full-fidelity, Python-only) or
### JSON (portable, human-readable, round-trips via each library's own
### native (de)serialization)

# %%
def _infer_format(path, fmt):
    if fmt is not None:
        return fmt
    suffix = Path(path).suffix.lstrip(".").lower()
    inferred = {"pkl": "pickle", "pickle": "pickle", "json": "json"}.get(suffix)
    if inferred is None:
        raise ValueError(
            f"Could not infer format from suffix '{Path(path).suffix}'; "
            f"pass fmt='pickle' or fmt='json' explicitly."
        )
    return inferred


def _structure_to_json_safe(struct):
    """Convert one relaxed-structure object into something json.dump can
    write, using each library's OWN native round-trippable serialization
    (not a lossy summary) -- `Structure.as_dict()` for pymatgen (the same
    convention as the original `load_workchain_data`'s
    `Structure.from_dict`), or `ase.io.jsonio.encode` for ASE Atoms.
    """
    if hasattr(struct, "as_dict"):  # pymatgen Structure (and most pymatgen objects)
        return {"__pymatgen__": struct.as_dict()}

    from ase import Atoms
    if isinstance(struct, Atoms):
        from ase.io.jsonio import encode
        return {"__ase__": encode(struct)}

    raise TypeError(
        f"Don't know how to JSON-serialize a structure of type {type(struct)}. "
        f"Use fmt='pickle' instead for full-fidelity storage of arbitrary objects."
    )


def _structure_from_json_safe(entry):
    """Inverse of `_structure_to_json_safe`."""
    if "__pymatgen__" in entry:
        from pymatgen.core import Structure
        return Structure.from_dict(entry["__pymatgen__"])
    if "__ase__" in entry:
        from ase.io.jsonio import decode
        return decode(entry["__ase__"])
    raise ValueError(f"Unrecognized structure entry, got keys: {list(entry.keys())}")


def save_relax_results(results, path, fmt=None):
    """Save `load_relax(...)` output to disk.

    Parameters
    ----------
    results : list of dict
        Output of `load_relax` -- each dict has 'idx', 'rlxd_struct',
        'energy'.
    path : str or Path
        Output file path.
    fmt : 'pickle', 'json', or None, default=None
        If None, inferred from `path`'s suffix ('.pkl'/'.pickle' ->
        pickle, '.json' -> json).

    Notes
    -----
    - pickle: stores structure objects as-is, byte-for-byte. Simplest,
      always works regardless of structure type, but binary, Python-only,
      and can break across incompatible pymatgen/ASE version upgrades.
    - json: human-readable and portable, but only supports pymatgen
      Structure and ASE Atoms as the 'rlxd_struct' value (raises
      TypeError for anything else -- use pickle in that case).
    """
    path = Path(path)
    fmt = _infer_format(path, fmt)

    if fmt == "pickle":
        with path.open("wb") as f:
            pickle.dump(results, f)
    elif fmt == "json":
        json_safe = [
            {**r, "rlxd_struct": _structure_to_json_safe(r["rlxd_struct"])}
            for r in results
        ]
        with path.open("w") as f:
            json.dump(json_safe, f, indent=2)
    else:
        raise ValueError(f"fmt must be 'pickle' or 'json', got {fmt!r}")

    print(f"Saved {len(results)} result(s) to {path} ({fmt})")


def load_relax_results(path, fmt=None):
    """Inverse of `save_relax_results` -- load a previously-saved result list."""
    path = Path(path)
    fmt = _infer_format(path, fmt)

    if fmt == "pickle":
        with path.open("rb") as f:
            return pickle.load(f)
    elif fmt == "json":
        with path.open() as f:
            raw = json.load(f)
        return [
            {**r, "rlxd_struct": _structure_from_json_safe(r["rlxd_struct"])}
            for r in raw
        ]
    else:
        raise ValueError(f"fmt must be 'pickle' or 'json', got {fmt!r}")

# %%
### argparse CLI

# %%
def _existing_dir(value):
    """argparse `type=` validator: rejects a non-existent/non-directory
    path with a clean error message AT PARSE TIME, instead of a deep
    traceback later once the scan actually starts.
    """
    p = Path(value)
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {value}")
    return p


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Load converged, relaxed QE structures from a folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples
--------
# Preview only -- see what's in the folder, load nothing, write nothing
python load_relax.py /path/to/folder --calc-type any --preview

# Load one specific material, save to JSON (portable, human-readable)
python load_relax.py /path/to/folder --species Na Cl --output results.json

# Same, but as a dry run first -- prints what it WOULD do, saves nothing
python load_relax.py /path/to/folder --species Na Cl --output results.json --dry-run

# Save as pickle instead (needed if pymatgen=False and you want ASE Atoms
# stored exactly as-is; JSON is supported for ASE Atoms too via
# ase.io.jsonio, but pickle is the simplest full-fidelity option)
python load_relax.py /path/to/folder --species Na Cl --ase-only --output results.pkl
""",
    )
    p.add_argument("folder", type=_existing_dir,
                    help="Folder containing QE .out (and, if --check-input, .in) files.")
    p.add_argument("--pattern", default="*.out")

    p.add_argument("--check-input", dest="check_input", action="store_true", default=False,
                    help="Require a matching .in file; classify via the namelist (reliable).")
    p.add_argument("--no-check-input", dest="check_input", action="store_false",
                    help="Classify from .out text alone (default; experimental).")

    p.add_argument("--calc-type", nargs="+", default=["relax"],
                    help="QE calculation type(s) to keep, or 'any' for no filtering.")

    p.add_argument("--species", nargs="+", default=None)
    p.add_argument("--nat", type=int, default=None)
    p.add_argument("--nat-range", nargs=2, type=int, default=None, metavar=("MIN", "MAX"))
    p.add_argument("--ntyp", type=int, default=None)
    p.add_argument("--min-runs", type=int, default=None)

    p.add_argument("--pymatgen", dest="pymatgen", action="store_true", default=True,
                    help="Store structures as pymatgen Structure objects (default).")
    p.add_argument("--ase-only", dest="pymatgen", action="store_false",
                    help="Store structures as ASE Atoms instead of converting to pymatgen.")

    p.add_argument("--preview", action="store_true",
                    help="Show grouped summary only (ignores all filters); load/save nothing.")
    p.add_argument("--dry-run", action="store_true",
                    help="Run the full selection + load, print what would be saved, "
                         "but never actually write --output.")

    p.add_argument("--output", type=Path, default=None,
                    help="Save results here (.json or .pkl/.pickle; format inferred "
                         "from the suffix unless --format is given).")
    p.add_argument("--format", choices=["pickle", "json"], default=None)

    return p

# %%
def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    calc_types = None if [c.lower() for c in args.calc_type] == ["any"] else args.calc_type

    if args.preview:
        groups = get_relax_after_run(
            args.folder, pattern=args.pattern, check_input=args.check_input,
            calc_types=calc_types, verbose=False,
        )
        print(f"\nPreview: {len(groups)} system group(s) found "
              f"(selection filters ignored in preview mode)\n")
        print_relax_summary(groups)
        return groups

    nat = tuple(args.nat_range) if args.nat_range else args.nat
    results = load_relax(
        args.folder,
        pattern=args.pattern,
        check_input=args.check_input,
        calc_types=calc_types,
        pymatgen=args.pymatgen,
        verbose=True,
        species=args.species, nat=nat, ntyp=args.ntyp, min_runs=args.min_runs,
    )

    print(f"\nLoaded {len(results)} relaxed structure(s).")

    if args.output:
        if args.dry_run:
            fmt = _infer_format(args.output, args.format)
            print(f"[dry run] Would save {len(results)} result(s) to "
                  f"{args.output} ({fmt}) -- nothing written")
        else:
            save_relax_results(results, args.output, fmt=args.format)

    return results


if __name__ == "__main__":
    main()