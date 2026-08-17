# %%
#!/usr/bin/env python3
"""
Utility module for streaming PWscf (Quantum ESPRESSO) output files 
into pymatgen Structure objects or ASE Atoms objects via pwo2xsf.sh.

usage:  
    python pwo2cif.py -i pw.out

    will create pw.cif file for the latest coordinates
or with 
    python pwo2cif.py -i pw.out -s pw2xsf.sh -f -lc[-oc, -ic, -a, ac] -o pw.cif

where -s is the path of the QE pw2xsf.sh executable, default, is same path as pwo2cif.py 
"""

import os
import sys
import argparse
import subprocess
from io import StringIO
from pathlib import Path
from typing import Union, Literal

from ase import Atoms
from ase.io import read as read_ase
from pymatgen.core import Structure

# Type alias for pwo2xsf.sh extraction flags
PwoFlag = Literal[
    "-lc", "--latestcoor",
    "-oc", "--optcoor",
    "-ic", "--inicoor",
    "-a",  "--animxsf",
    "-ac", "--animcoor"
]

# chmod +x /path/of/the/file/pwo2xsf.sh

# %%
def _check_script_executable(sh_path: Path) -> None:
    """Check if script exists and is executable, raising helpful errors if not."""
    if not sh_path.is_file():
        raise FileNotFoundError(f"pwo2xsf.sh script not found at: {sh_path}")
    
    if not os.access(sh_path, os.X_OK):
        raise PermissionError(
            f"Permission denied to execute script: '{sh_path}'\n"
            f"Please run the following command in your terminal to grant permissions:\n\n"
            f"    chmod +x {sh_path}\n"
        )

# %%
def load_pwo_as_structure(
    pwo_filepath: Union[str, Path],
    flag: PwoFlag = "-lc",
    script_path: Union[str, Path] = "pwo2xsf.sh"
) -> Structure:
    """
    Executes pwo2xsf.sh on a PWscf output file and parses the stdout stream
    directly into a pymatgen Structure object without saving temporary files.
    """
    pwo_path = Path(pwo_filepath).expanduser().resolve()
    sh_path = Path(script_path).expanduser().resolve()

    if not pwo_path.is_file():
        raise FileNotFoundError(f"PWscf output file not found: {pwo_path}")
    
    _check_script_executable(sh_path)

    cmd = [str(sh_path), flag, str(pwo_path)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )

    xsf_string = result.stdout
    return Structure.from_str(xsf_string, fmt="xsf")

# %%
def load_pwo_as_atoms(
    pwo_filepath: Union[str, Path],
    flag: PwoFlag = "-lc",
    script_path: Union[str, Path] = "pwo2xsf.sh"
) -> Atoms:
    """
    Executes pwo2xsf.sh on a PWscf output file and parses the stdout stream
    directly into an ASE Atoms object using an in-memory buffer.
    """
    pwo_path = Path(pwo_filepath).expanduser().resolve()
    sh_path = Path(script_path).expanduser().resolve()

    if not pwo_path.is_file():
        raise FileNotFoundError(f"PWscf output file not found: {pwo_path}")
    
    _check_script_executable(sh_path)

    cmd = [str(sh_path), flag, str(pwo_path)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )

    xsf_stream = StringIO(result.stdout)
    atoms: Atoms = read_ase(xsf_stream, format="xsf")
    return atoms

# %%
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert Quantum ESPRESSO output (.out/.pwo) to pymatgen/ASE structures or structure files via pwo2xsf.sh."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="Path to Quantum ESPRESSO output file."
    )
    parser.add_argument(
        "-s", "--script",
        default="pwo2xsf.sh",
        type=Path,
        help="Path to pwo2xsf.sh executable (default: pwo2xsf.sh)."
    )
    parser.add_argument(
        "-f", "--flag",
        default="-lc",
        choices=["-lc", "--latestcoor", "-oc", "--optcoor", "-ic", "--inicoor", "-a", "--animxsf", "-ac", "--animcoor"],
        help="Coordinate extraction flag for pwo2xsf.sh (default: -lc)."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Optional output structure filepath (e.g. structure.cif, structure.xsf)."
    )
    return parser.parse_args()

# %%
# def main():
#     args = parse_arguments()

#     try:
#         structure = load_pwo_as_structure(
#             pwo_filepath=args.input,
#             flag=args.flag,
#             script_path=args.script
#         )
#         print(f"Successfully loaded structure from: {args.input}")
#         print(f"Formula: {structure.composition.reduced_formula}")
#         print(f"Number of sites: {len(structure)}")

#         if args.output:
#             args.output.parent.mkdir(parents=True, exist_ok=True)
#             structure.to(filename=str(args.output))
#             print(f"Saved output to: {args.output}")

#     except (PermissionError, FileNotFoundError, subprocess.CalledProcessError) as err:
#         print(f"\nERROR: {err}", file=sys.stderr)
#         sys.exit(1)

# %%
def main():
    args = parse_arguments()

    try:
        structure = load_pwo_as_structure(
            pwo_filepath=args.input,
            flag=args.flag,
            script_path=args.script
        )
        print(f"Successfully loaded structure from: {args.input}")
        print(f"Formula: {structure.composition.reduced_formula}")
        print(f"Number of sites: {len(structure)}")

        # Default output path: same folder and stem as input file + .cif
        output_file = (
            args.output.expanduser().resolve()
            if args.output
            else args.input.with_suffix(".cif")
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        structure.to(filename=str(output_file))
        print(f"Saved output to: {output_file}")

    except (PermissionError, FileNotFoundError, subprocess.CalledProcessError) as err:
        print(f"\nERROR: {err}", file=sys.stderr)
        sys.exit(1)

# %%
if __name__ == "__main__":
    main()
