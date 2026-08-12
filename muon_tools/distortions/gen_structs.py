# %%
"""
Command-line entry point.

    python -m distortions.gen_structs --pristine pristine.cif --relaxed relaxed.cif \
        --muon-label H --out-dir ./equivalent_sites

(or, from inside the package directory: `python gen_structs.py ...`)
"""

from __future__ import annotations

import os
import json
import argparse
from typing import Optional, Sequence

import numpy as np
from pymatgen.core import Structure

from muon_tools.distortions.distortions import get_structs

# %%
def _read_magmom(path: Optional[str]) -> Optional[np.ndarray]:
    return None if path is None else np.loadtxt(path)


def _read_structure(path: str) -> Structure:
    """Read a structure from any format pymatgen understands (cif, POSCAR,
    etc.); falls back to ASE (then converts) for formats pymatgen can't
    parse directly, e.g. Quantum ESPRESSO output (`*.out` / `espresso-out`)."""
    try:
        return Structure.from_file(path)
    except Exception:
        pass
    try:
        from ase.io import read as ase_read
        from pymatgen.io.ase import AseAtomsAdaptor
        atoms = ase_read(path, index=-1)
        return AseAtomsAdaptor.get_structure(atoms)
    except Exception as exc:
        raise ValueError(f"Could not read structure from '{path}' with either pymatgen or ASE.") from exc

# %%
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate approximate relaxed structures for every symmetry-equivalent muon site."
    )
    parser.add_argument("--pristine", required=True, help="Pristine host structure file (muon-free).")
    parser.add_argument("--relaxed", required=True, help="Relaxed structure file (includes the muon).")
    parser.add_argument("--host_lattice", default=True,
                         help="Structure used for the symmetry search (must share the pristine/relaxed cell). "
                              "Defaults to --pristine.")
    parser.add_argument("--magmom", default=None,
                         help="Optional path to a per-site magnetic moments text file: one line per HOST atom "
                              "(same count and order as --pristine/--host, muon excluded), either one number "
                              "per line (z-axis moment, e.g. VASP MAGMOM convention) or three space-separated "
                              "numbers per line 'mx my mz' (non-collinear). Blank lines and '#' comments allowed.")
    parser.add_argument("--muon-label", default="H", help="Species symbol identifying the muon (default: H).")
    parser.add_argument("--muon-index", default="last", choices=["first", "last", "unique"],
                         help="Which matching atom to treat as the muon if several share --muon-label.")
    parser.add_argument("--tol", type=float, default=1e-3, help="Fractional-coordinate matching tolerance.")
    parser.add_argument("--symprec", type=float, default=1e-3, help="Symmetry-detection precision.")
    parser.add_argument("--min-distance", type=float, default=0.5,
                         help="Real-space (Angstrom) merge threshold for near-duplicate orbit points. "
                              "Pass a negative number to disable this cleanup pass.")
    parser.add_argument("--out-dir", required=True, help="Directory to write output structures + summary into.")
    parser.add_argument("--out-format", default="cif", help="pymatgen output format for structures (default: cif).")
    
    return parser

# %%
def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    p_st = _read_structure(args.pristine)
    rlx_st = _read_structure(args.relaxed)
    host_st = _read_structure(args.host_lattice) if args.host_lattice else p_st
    magmom = _read_magmom(args.magmom)
    min_distance = None if args.min_distance is not None and args.min_distance < 0 else args.min_distance

    results = get_structs(
        p_st, rlx_st, host_st,
        magmom=magmom, muon_label=args.muon_label, muon_index=args.muon_index,
        tol=args.tol, symprec=args.symprec, min_distance=min_distance,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    summary = []
    for i, site in enumerate(results):
        entry = {
            "index": i,
            "frac_pos": site.frac_pos.tolist(),
            "is_original": bool(site.is_original),
            "is_magnetically_equivalent": bool(site.is_magnetically_equivalent),
            "ok": bool(site.ok),
        }
        if site.structure is not None:
            fname = f"site_{i:03d}.{args.out_format}"
            site.structure.to(filename=os.path.join(args.out_dir, fname), fmt=args.out_format)
            entry["filename"] = fname
        summary.append(entry)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    n_ok = sum(s.ok for s in results)
    print(f"Found {len(results)} equivalent site(s), wrote {n_ok} structure(s) to {args.out_dir}")
    print(f"Summary: {os.path.join(args.out_dir, 'summary.json')}")

# %%
if __name__ == "__main__":
    main()