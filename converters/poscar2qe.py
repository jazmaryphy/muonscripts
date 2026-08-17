# %%
#!/usr/bin/env python3
"""
Convert VASP POSCAR / CONTCAR files to Quantum ESPRESSO CELL_PARAMETERS 
and ATOMIC_POSITIONS format.
"""

import sys
import argparse
from pathlib import Path
from collections import Counter

# %%
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert a VASP POSCAR file to Quantum ESPRESSO CELL_PARAMETERS and ATOMIC_POSITIONS."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="Path to input POSCAR file"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file path (default: <input_stem>.qe)"
    )
    parser.add_argument(
        "-s", "--scale-cell",
        action="store_true",
        default=False,
        help="Apply the POSCAR scale factor to lattice vectors when scale > 0 (default: False)"
    )
    return parser.parse_args()

# %%
def read_poscar(filename: Path, scale_cell: bool = False):
    """
    Parse a VASP POSCAR/CONTCAR file supporting both VASP 4 and VASP 5 formats,
    selective dynamics, and Cartesian/Direct coordinates.

    Parameters
    ----------
    filename : Path
        Path to POSCAR file.
    scale_cell : bool, optional
        Whether to scale lattice vectors by the positive scale factor on line 2.
        If False (default), raw cell vectors are used as-is unless scale < 0.
    """
    with open(filename, "r") as f:
        # Strip comments and empty lines while maintaining order
        raw_lines = [line.strip() for line in f if line.strip()]

    if len(raw_lines) < 7:
        raise ValueError("Input file is too short to be a valid POSCAR.")

    # 1. Scale factor / volume target
    scale = float(raw_lines[1])

    # 2. Parse lattice vectors
    raw_lattice = [
        [float(x) for x in raw_lines[i].split()[:3]]
        for i in range(2, 5)
    ]

    if scale > 0:
        if scale_cell:
            lattice = [[x * scale for x in vec] for vec in raw_lattice]
        else:
            lattice = raw_lattice
    else:
        # Scale < 0 represents target volume |scale|
        # Volume V = a1 . (a2 x a3)
        a1, a2, a3 = raw_lattice
        det_vol = abs(
            a1[0]*(a2[1]*a3[2] - a2[2]*a3[1]) -
            a1[1]*(a2[0]*a3[2] - a2[2]*a3[0]) +
            a1[2]*(a2[0]*a3[1] - a2[1]*a3[0])
        )
        vol_factor = (abs(scale) / det_vol) ** (1.0 / 3.0)
        lattice = [[x * vol_factor for x in vec] for vec in raw_lattice]

    # 3. Handle species and atom counts (VASP 4 vs VASP 5 compatibility)
    line5_parts = raw_lines[5].split()
    line6_parts = raw_lines[6].split()

    if line5_parts[0].isdigit():
        # VASP 4 format: Line 5 contains atom counts; no species header
        counts = [int(x) for x in line5_parts]
        species = [f"X{i+1}" for i in range(len(counts))]
        line_idx = 6
    else:
        # VASP 5 format: Line 5 contains species symbols, line 6 contains counts
        species = line5_parts
        counts = [int(x) for x in line6_parts]
        line_idx = 7

    if len(species) != len(counts):
        raise ValueError("Number of atomic species labels does not match count list.")

    natoms = sum(counts)

    # Expand species labels per atom
    atom_labels = []
    for sp, count in zip(species, counts):
        atom_labels.extend([sp] * count)

    # 4. Handle optional Selective Dynamics
    coord_header = raw_lines[line_idx]
    if coord_header.lower().startswith('s'):
        line_idx += 1
        coord_header = raw_lines[line_idx]

    line_idx += 1  # Move past coordinate type line

    # 5. Extract Atomic Positions
    coords = []
    for i in range(natoms):
        if line_idx + i >= len(raw_lines):
            raise ValueError(f"Expected {natoms} atomic coordinates but found only {len(coords)}.")
        
        parts = raw_lines[line_idx + i].split()
        xyz = [float(x) for x in parts[:3]]
        coords.append(xyz)

    return lattice, coord_header, atom_labels, coords


def write_qecoords(output_file: Path, lattice, coordinate_type: str, atom_labels, coords):
    """
    Write lattice vectors and atomic positions in Quantum ESPRESSO format.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    coord_lower = coordinate_type.lower()
    if coord_lower.startswith("d") or (coord_lower.startswith("c") and not coord_lower.startswith("cart")):
        qe_coord_type = "ATOMIC_POSITIONS (crystal)\n"
    elif coord_lower.startswith("cart"):
        qe_coord_type = "ATOMIC_POSITIONS (angstrom)\n"
    else:
        raise ValueError(f"Unrecognized coordinate type in POSCAR: '{coordinate_type}'")

    with open(output_file, "w") as out:
        # Cell Parameters
        out.write("CELL_PARAMETERS (angstrom)\n")
        for vec in lattice:
            out.write(f"        {vec[0]:18.10f} {vec[1]:18.10f} {vec[2]:18.10f}\n")

        out.write("\n")

        # Atomic Positions
        out.write(qe_coord_type)
        for label, xyz in zip(atom_labels, coords):
            out.write(f"{label:<8s} {xyz[0]:18.10f} {xyz[1]:18.10f} {xyz[2]:18.10f}\n")

# %%
def main():
    args = parse_arguments()

    input_file = args.input.expanduser().resolve()
    if not input_file.is_file():
        print(f"ERROR: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    output_file = (
        args.output.expanduser().resolve() 
        if args.output 
        else input_file.with_suffix(".qecoord")
    )

    print(f"Reading : {input_file}")

    try:
        lattice, coord_type, atom_labels, coords = read_poscar(
            input_file, 
            scale_cell=args.scale_cell
        )

        write_qecoords(output_file, lattice, coord_type, atom_labels, coords)

        counts = Counter(atom_labels)
        species_info = ", ".join(f"{sp}({counts[sp]})" for sp in dict.fromkeys(atom_labels))

        print(f"Writing : {output_file}")
        print(f"# Atoms : {len(atom_labels)}")
        print(f"Species : {species_info}")
        print(f"Scaled  : {args.scale_cell}")
        print("Done.")

    except Exception as err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)

# %%
if __name__ == "__main__":
    main()