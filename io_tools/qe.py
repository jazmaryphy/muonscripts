# %%
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union, Sequence

import copy
import warnings
import itertools
import numpy as np
from ase import Atoms
from constants import constants
from io_tools.base import DFTCoordinates, fortran_value, find_first_index

qe_coord_types = ["crystal", "bohr", "angstrom", "alat"]

# constants.BOHR_TO_ANGSTROM
MHZ_TO_KHZ = 1000
BOHR_TO_ANGSTROM = constants.BOHR_RADIUS/constants.ANGSTROM
EFG_CONVERSION_SI = constants.EFG_AMU_TO_SI

# %%
class PWCoordinates(DFTCoordinates):
    """
    Coordinates of the system from the PW data of Quantum Espresso. Subclass of the DFTCoordinates.

    With initiallization reads either output or input of PW module of QE.

    Args:
        filename (str): name of the PW input or output.
        pwfile (str):
            Name of PW input or output file.
            If the file doesn't have proper extension, parameter pw_type should indicate the type.
        pwtype (str): Type of the coord_f. if not listed, will be inferred from extension of pwfile.
        to_angstrom (bool): True if automatically convert the units of ``cell`` and ``coordinates`` to Angstrom.

    """

    def __init__(self, filename, pwtype=None, to_angstrom=False):
        super().__init__()

        if not pwtype:
            pwtype = filename.split(".")[-1]

        if pwtype == "in":
            self.parse_input(filename, to_angstrom=to_angstrom)
        elif pwtype == "out":
            self.parse_output(filename, to_angstrom=to_angstrom)

        else:
            raise TypeError("Unsupported pw_type! Only .out or .in are supported")

    def parse_output(self, filename, to_angstrom=False):
        """
        Method to read coordinates of atoms from PW output into the PWCoordinates instance.

        Args:
            filename (str): the name of the output file.
            to_angstrom (bool): True if automatically convert the units of ``cell`` and ``coordinates`` to Angstrom.

        Returns:
            None

        """
        with open(filename) as f:
            lines = f.readlines()
        alat_index = find_first_index("lattice parameter (alat)", lines)

        self.alat = float(lines[alat_index].split()[-2]) * BOHR_TO_ANGSTROM

        cell_index = find_first_index("crystal axes", lines) + 1
        cell = []
        for index in range(cell_index, cell_index + 3):
            row_split = lines[index].split()
            cell_row = [float(x) for x in row_split[-4:-1]]
            cell.append(cell_row)

        self.cell = np.asarray(cell).T
        self.cell_units = "alat"

        # read coordinates

        names = []
        coordinates = []
        try:
            final_kw = "Begin final coordinates"
            start = find_first_index(final_kw, lines)
            coord_kw = "ATOMIC_POSITIONS"
            index = find_first_index(coord_kw, lines, start=start)
            coord_units = get_ctype(lines[index])

            while True:
                try:
                    index += 1

                    row_split = lines[index].split()
                    name = row_split[0]
                    crow = [float(x) for x in row_split[1:]]

                    names.append(name)
                    coordinates.append(crow)
                except (IndexError, ValueError):
                    break

        except KeyError:
            coord_units = "alat"

            coord_kw = "Cartesian axes"
            index = find_first_index(coord_kw, lines) + 2
            while True:
                try:
                    index += 1

                    row_split = lines[index].split()
                    name = row_split[1]
                    crow = [float(x) for x in row_split[-4:-1]]

                    names.append(name)
                    coordinates.append(crow)

                except IndexError:
                    break

        self.coordinates_units = coord_units
        self.coordinates = np.asarray(coordinates)
        self.names = np.array(names, dtype="<U16")

        if to_angstrom:
            self.to_angstrom(inplace=True)
        return self

    def parse_input(self, filename, to_angstrom=False):
        """
        Method to read coordinates of atoms from PW input into the PWCoordinates instance.

        Args:
            filename (str): the name of the output file.
            to_angstrom (bool): True if automatically convert the units of ``cell`` and ``coordinates`` to Angstrom.

        """
        with open(filename) as f:
            input_string = f.read()

        namelists = read_qe_namelists(input_string.lower())
        lines = input_string.splitlines()

        alat = namelists["system"].get("celldm(1)", None)

        if alat is None:
            alat = namelists["system"].get("a", None)
        else:
            alat *= BOHR_TO_ANGSTROM

        cell = cell_from_system(namelists["system"])
        if cell is not None:
            cell_units = "bohr"
        else:
            index = find_first_index("CELL_PARAMETERS", lines)
            cell_units = get_ctype(lines[index])
            self.cell_units = cell_units
            cell = []

            for _ in range(3):
                index += 1
                line = lines[index]
                cell.append([float(x) for x in line.split()])

            cell = np.asarray(cell).T

            if cell_units == "alat" and alat is None:
                raise ValueError("alat was not found")

            if alat is None:
                assert cell_units in ["bohr", "angstrom"]
                alat = cell[0, 0] * BOHR_TO_ANGSTROM ** (cell_units == "bohr")

        self.cell = cell
        self.cell_units = cell_units
        self.alat = alat

        index = find_first_index("ATOMIC_POSITIONS", lines)
        coord_units = get_ctype(lines[index])

        names = []
        coords = []

        for i in range(index + 1, index + 1 + namelists["system"]["nat"]):
            row_split = lines[i].split()
            names.append(row_split[0])
            coords.append([float(x) for x in row_split[1:]])

        self.names = np.array(names, dtype="<U16")
        self.coordinates = np.asarray(coords)
        self.coordinates_units = coord_units

        if to_angstrom:
            self.to_angstrom(inplace=True)


def cell_from_system(sdict):
    """
    Function to obtain cell from namelist SYSTEM read from PW input.

    Args:
        sdict (dict): Dictinary generated from namelist SYSTEM of PW input.

    Returns:
        ndarray with shape (3,3):
            Cell is 3x3 matrix with entries::

                [[a_x b_x c_x]
                 [a_y b_y c_y]
                 [a_z b_z c_z]],

            where a, b, c are crystallographic vectors,
            and x, y, z are their coordinates in the cartesian reference frame.

    """
    ibrav = sdict.get("ibrav", None)
    if ibrav == 0:
        return None
    params = ["a", "b", "c", "cosab", "cosac", "cosbc"]
    celldm = [sdict.get(f"celldm({i + 1})", 0) for i in range(6)]
    if not any(celldm):
        abc = [sdict.get(a, 0) for a in params]
        celldm = celldms_from_abc(ibrav, abc)

    if not any(celldm):
        return None

    if ibrav == 1:
        cell = np.eye(3) * celldm[0]
        return cell

    elif ibrav == 2:
        v1 = celldm[0] / 2 * np.array([-1, 0, 1])
        v2 = celldm[0] / 2 * np.array([0, 1, 1])
        v3 = celldm[0] / 2 * np.array([-1, 1, 0])

    elif ibrav == 3:
        v1 = celldm[0] / 2 * np.array([1, 1, 1])
        v2 = celldm[0] / 2 * np.array([-1, 1, 1])
        v3 = celldm[0] / 2 * np.array([-1, -1, 1])

    elif ibrav == -3:
        v1 = celldm[0] / 2 * np.array([-1, 1, 1])
        v2 = celldm[0] / 2 * np.array([1, -1, 1])
        v3 = celldm[0] / 2 * np.array([1, 1, -1])

    elif ibrav == 4:
        v1 = celldm[0] * np.array([1, 0, 0])
        v2 = celldm[0] * np.array([-1 / 2, np.sqrt(3) / 2, 0])
        v3 = celldm[0] * np.array([0, 0, celldm[2]])

    elif ibrav == 5:
        term_1 = np.sqrt(1 + 2 * celldm[3])
        term_2 = np.sqrt(1 - celldm[3])
        v1 = celldm[0] * np.array([term_2 / np.sqrt(2), -term_2 / np.sqrt(6), term_1 / np.sqrt(3)])
        v2 = celldm[0] * np.array([0, term_2 * np.sqrt(2 / 3), term_1 / np.sqrt(3)])
        v3 = celldm[0] * np.array([-term_2 / np.sqrt(2), -term_2 / np.sqrt(6), term_1 / np.sqrt(3)])

    elif ibrav == -5:
        term_1 = np.sqrt(1 + 2 * celldm[3])
        term_2 = np.sqrt(1 - celldm[3])
        v1 = celldm[0] * np.array([(term_1 - 2 * term_2) / 3, (term_1 + term_2) / 3, (term_1 + term_2) / 3])
        v2 = celldm[0] * np.array([(term_1 + term_2) / 3, (term_1 - 2 * term_2) / 3, (term_1 + term_2) / 3])
        v3 = celldm[0] * np.array([(term_1 + term_2) / 3, (term_1 + term_2) / 3, (term_1 - 2 * term_2) / 3])

    elif ibrav == 6:
        v1 = celldm[0] * np.array([1, 0, 0])
        v2 = celldm[0] * np.array([0, 1, 0])
        v3 = celldm[0] * np.array([0, 0, celldm[2]])
    elif ibrav == 7:
        v1 = celldm[0] / 2 * np.array([1, -1, celldm[2]])
        v2 = celldm[0] / 2 * np.array([1, 1, celldm[2]])
        v3 = celldm[0] / 2 * np.array([-1, -1, celldm[2]])
    elif ibrav == 8:
        v1 = celldm[0] * np.array([1, 0, 0])
        v2 = celldm[0] * np.array([0, celldm[1], 0])
        v3 = celldm[0] * np.array([0, 0, celldm[2]])
    elif ibrav == 9:
        v1 = celldm[0] / 2 * np.array([1, celldm[1], 0])
        v2 = celldm[0] / 2 * np.array([-1, celldm[1], 0])
        v3 = celldm[0] * np.array([0, 0, celldm[2]])

    elif ibrav == -9:
        v1 = celldm[0] / 2 * np.array([1, -celldm[1], 0])
        v2 = celldm[0] / 2 * np.array([+1, celldm[1], 0])
        v3 = celldm[0] * np.array([0, 0, celldm[2]])

    elif ibrav == 91:
        v1 = celldm[0] * np.array([1, 0, 0])
        v2 = celldm[0] / 2 * np.array([0, celldm[1], -celldm[2]])
        v3 = celldm[0] / 2 * np.array([0, celldm[1], celldm[2]])
    elif ibrav == 10:
        v1 = celldm[0] / 2 * np.array([1, 0, celldm[2]])
        v2 = celldm[0] / 2 * np.array([1, celldm[1], 0])
        v3 = celldm[0] / 2 * np.array([0, celldm[1], celldm[2]])
    elif ibrav == 11:
        v1 = celldm[0] / 2 * np.array([1, celldm[1], celldm[2]])
        v2 = celldm[0] / 2 * np.array([-1, celldm[1], celldm[2]])
        v3 = celldm[0] / 2 * np.array([-1, -celldm[1], celldm[2]])
    elif ibrav == 12:
        sen = np.sqrt(1 - celldm[3] ** 2)
        v1 = celldm[0] * np.array([1, 0, 0])
        v2 = celldm[0] * np.array([celldm[1] * celldm[3], celldm[1] * sen, 0])
        v3 = celldm[0] * np.array([0, 0, celldm[2]])
    elif ibrav == -12:
        sen = np.sqrt(1 - celldm[4] ** 2)
        v1 = celldm[0] * np.array([1, 0, 0])
        v2 = celldm[0] * np.array([0, celldm[1], 0])
        v3 = celldm[0] * np.array([celldm[2] * celldm[4], 0, celldm[2] * sen])

    elif ibrav == 13:
        sen = np.sqrt(1 - celldm[3] ** 2)
        v1 = celldm[0] / 2 * np.array([1, 0, -celldm[2]])
        v2 = celldm[0] * np.array([celldm[1] * celldm[3], celldm[1] * sen, 0])
        v3 = celldm[0] / 2 * np.array([1, 0, celldm[2]])

    elif ibrav == -13:
        sen = np.sqrt(1 - celldm[4] ** 2)
        v1 = celldm[0] / 2 * np.array([1, celldm[1], 0])
        v2 = celldm[0] / 2 * np.array([-1, celldm[1], 0])
        v3 = celldm[0] * np.array([celldm[2] * celldm[4], 0, celldm[2] * sen])

    elif ibrav == 14:
        singam = np.sqrt(1 - celldm[5] ** 2)
        term = 1 + 2 * celldm[3] * celldm[4] * celldm[5] - celldm[3] ** 2 - celldm[4] ** 2 - celldm[5] ** 2
        term = np.sqrt(term / (1 - celldm[5] ** 2))

        v1 = celldm[0] * np.array([1, 0, 0])
        v2 = celldm[0] * np.array([celldm[1] * celldm[5], celldm[1] * singam, 0])
        v3 = celldm[0] * np.array(
            [celldm[2] * celldm[4], celldm[2] * (celldm[3] - celldm[4] * celldm[5]) / singam, celldm[2] * term]
        )
    else:
        raise ValueError("Unsupported ibrav")

    cell = np.stack([v1, v2, v3], axis=1)

    return cell


def celldms_from_abc(ibrav, abc_list):
    """
    Obtain celldms from ibrav value and a, b, c, cosab, cosac, cosbc parameters.

    Using ibrav value and abc parameters from PW input generate celldm array, necessary to construct cell parameters.
    For details about abc and ibrav values see PW input documentation.

    Args:
        ibrav (int): ibrav parameter of PW input.
        abc_list (list): List, of 6 parameters:  a, b, c, cosab, cosac, cosbc

    Returns:
        celldm (list): list of 6 values, from which cell can be generated.

    """
    a, b, c, cosab, cosac, cosbc = abc_list

    celldm = [0.0] * 6
    celldm[0] = a / BOHR_TO_ANGSTROM
    celldm[1] = b / a
    celldm[2] = c / a

    if ibrav in [0, 14]:

        celldm[3] = cosbc
        celldm[4] = cosac
        celldm[5] = cosab

    elif ibrav in [-12, -13]:
        celldm[3] = 0.0
        celldm[4] = cosac
        celldm[5] = 0.0

    elif ibrav in [-5, 5, 12, 13]:
        celldm[3] = cosab
        celldm[4] = 0.0
        celldm[5] = 0.0

    return celldm


def read_gipaw_tensors(lines, keyword=None, start=None, conversion=1):
    """
    Helper function to read GIPAW tensors from the list of lines.

    Args:
        lines (list of str): List of strings contraining lines from the file. Output of open(file).readlines().
        keyword (str): Keyword in the line which indicates the beginning of the tensor data block.
        start (int): Index of the line which indicates the beginning of the tensor data block.
        conversion (float): Conversion factor from GIPAW units to the ones, used in this package.

    Returns:
        ndarray with shape (n, 3, 3): Array of tensors.

    """
    if keyword is not None:
        start = find_first_index(keyword, lines)

    if start is None:
        raise ValueError

    all_tensors = []

    n = start
    while True:
        n += 1
        tensor = []
        try:
            for _ in range(3):
                line = lines[n]
                tensor.append([float(x) * conversion for x in line.split()[2:]])
                n += 1

        except (ValueError, IndexError):
            break

        all_tensors.append(tensor)

    return all_tensors


def read_hyperfine(filename, spin=1):
    """
    Function to read hyperfine couplings from GIPAW output.

    Args:
        filename (str): Name of the GIPAW hyperfine output.
        spin (float): Spin of the central spin. Default 1.

    Returns:
        tuple: Tuple containing:

            * *ndarray with shape (n,)*: Array of Fermi contact terms.
            * *ndarray with shape (n, 3,3)*: Array of spin dipolar hyperfine tensors.

    """
    conversion = MHZ_TO_KHZ / (2 * spin)

    with open(filename) as f:
        lines = f.readlines()

    dipol_keyword = "total dipolar (symmetrized)"
    contact_keyword = "Fermi contact in MHz"

    dipolars = read_gipaw_tensors(lines, dipol_keyword, conversion=conversion)
    start = find_first_index(contact_keyword, lines) + 2

    contacts = []

    for index in range(start, start + len(dipolars)):
        line = lines[index]
        # divided by spin, b/c NI in the GIPAW code
        cont = float(line.split()[-1]) * conversion
        contacts.append(cont)

    return np.asarray(contacts), np.asarray(dipolars)


def read_efg(filename):
    """
    Function to read electric field gradient tensors from GIPAW output.

    Args:
        filename (str): Name of the GIPAW EFG-containing output.

    Returns:
        ndarray with shape (n, 3,3): Array of EFG tensors.

    """
    efg_kw = "total EFG (symmetrized)"
    with open(filename) as f:
        lines = f.readlines()

    tensors = read_gipaw_tensors(lines, keyword=efg_kw, conversion=EFG_CONVERSION_SI)

    return np.asarray(tensors)


def read_qe_namelists(input_string):
    """
    Read Fortran-like namelists from the large string.

    Args:
        input_string (str): String representation of the QE input file.

    Returns:
        dict: Dictionary, containing dicts for each namelist found in the input string.

    """
    namelists = {}

    for block in input_string.split("/\n"):

        if not "&" in block:
            # means we are out of namelists
            break

        lines = [s.strip() for s in block.splitlines() if (s.strip() and s[0] != "!")]
        index = find_first_index("&", lines)
        block_name = lines[index].strip("&\t ")

        namelists[block_name] = {}
        for row in lines[index + 1 :]:
            for pair in row.split(","):
                if not pair.strip():
                    continue
                name, _, value = (x.strip(",\t ").lower() for x in pair.partition("="))
                namelists[block_name][name] = fortran_value(value)

    return namelists


def get_ctype(lin):
    """
    Get coordinates type from the line of QE input/output.

    Args:
        str: Line from QE input/output containing string with coordinates type.

    Returns:
        str: type of the coordinates.
    """

    try:
        coord_type = next(filter(lambda x: x in lin.lower(), qe_coord_types))
    except IndexError:
        raise ValueError(f"{lin} type is not supported.\nAllowed types: ", " ".join(*qe_coord_types))

    return coord_type

# %%
def read_qe_xsf_point(
    filename: str, 
    index: tuple[int, int, int]
) -> tuple[float, tuple[int, int, int]]:
    """
    Read a single value from an XSF 3D datagrid without parsing the rest
    of the grid.

    Reads only as far into the file as needed to reach the requested
    index (Fortran/x-fastest order), instead of parsing every point and
    discarding all but one -- dramatically faster when only one or a
    few values are needed from a large grid.

    Args:
        filename (str): Path to the .xsf file.
        index (tuple[int, int, int]): Zero-based (i, j, k) grid index.

    Returns:
        tuple[float, tuple[int, int, int]]: The value at `index`, and
            the grid's (nx, ny, nz) dimensions (returned so callers can
            cheaply verify two grids share the same shape without a
            second full read).

    Raises:
        OSError: If the file cannot be opened or read.
        KeyError: If no BEGIN_BLOCK_DATAGRID_3D block is found.
        ValueError: If the dimensions line is malformed, `index` is out
            of bounds, or the file ends before reaching `index`.
    """
    try:
        f = open(filename)
    except OSError as e:
        raise OSError(f"Could not open XSF file '{filename}': {e}") from e

    try:
        for line in f:
            if "BEGIN_BLOCK_DATAGRID_3D" in line:
                break
        else:
            raise KeyError(f"No BEGIN_BLOCK_DATAGRID_3D block found in '{filename}'.")

        next(f)  # grid name line
        next(f)  # BEGIN_DATAGRID_3D_<name> line

        try:
            dims = tuple(int(x) for x in next(f).split())
            nx, ny, nz = dims
        except (StopIteration, ValueError) as e:
            raise ValueError(f"Could not parse grid dimensions in '{filename}': {e}") from e

        i, j, k = index
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            raise ValueError(f"Index {index} out of bounds for grid of shape {dims}.")

        target = i + nx * (j + ny * k)  # Fortran-order flat index

        for _ in range(4):  # origin + 3 grid-vector lines
            next(f)

        def tokens():
            for line in f:
                yield from line.split()

        try:
            value = float(next(itertools.islice(tokens(), target, target + 1)))
        except StopIteration as e:
            raise ValueError(
                f"'{filename}' ended before reaching grid index {index} "
                f"(flat position {target})."
            ) from e

        return value, dims
    finally:
        f.close()

# %%
def read_qe_xsf_datagrid(filename: str):
    """
    Read the first 3D data grid from an XCrySDen (.xsf) file.

    Parses a single BEGIN_BLOCK_DATAGRID_3D ... END_DATAGRID_3D block and
    returns the grid values reshaped to (nx, ny, nz). Only the first such
    block in the file is read; files containing multiple stacked grids
    (e.g. separate spin channels in one file) require calling this on
    each block separately, or extending this function to accept a block
    index / name.

    Values are returned exactly as stored in the file, in whatever units
    the code that wrote it used. The XSF format itself does not encode
    units. For charge/spin density grids from Quantum ESPRESSO's pp.x,
    this is typically electrons per cubic Bohr radius (e/bohr^3, i.e.
    Hartree atomic units) -- verify against the source code for other
    DFT packages.

    Args:
        filename (str): Path to the .xsf file.

    Returns:
        ndarray with shape (nx, ny, nz): Grid data, Fortran-ordered so
            that data[i, j, k] corresponds to the grid point that varies
            fastest along x, matching the storage order used by the XSF
            format.

    Raises:
        OSError: If the file cannot be opened or read.
        KeyError: If no BEGIN_BLOCK_DATAGRID_3D or matching
            END_DATAGRID_3D block is found in the file.
        ValueError: If the grid-dimensions line is missing, malformed,
            or the number of parsed values doesn't match nx * ny * nz.
    """
    try:
        with open(filename) as f:
            lines = f.readlines()
    except OSError as e:
        raise OSError(f"Could not read XSF file '{filename}': {e}") from e

    try:
        block_idx = find_first_index("BEGIN_BLOCK_DATAGRID_3D", lines)
    except KeyError as e:
        raise KeyError(
            f"No BEGIN_BLOCK_DATAGRID_3D block found in '{filename}'."
        ) from e

    try:
        dims = tuple(int(x) for x in lines[block_idx + 3].split())
        nx, ny, nz = dims
    except IndexError as e:
        raise ValueError(
            f"'{filename}' is truncated: expected a grid-dimensions line "
            f"3 lines after BEGIN_BLOCK_DATAGRID_3D (around line {block_idx})."
        ) from e
    except ValueError as e:
        raise ValueError(
            f"Could not parse grid dimensions on line {block_idx + 3} "
            f"of '{filename}': {lines[block_idx + 3]!r}"
        ) from e

    data_start = block_idx + 8
    try:
        data_end = find_first_index("END_DATAGRID_3D", lines, start=data_start)
    except KeyError as e:
        raise KeyError(
            f"No matching END_DATAGRID_3D found in '{filename}' "
            f"after line {data_start}."
        ) from e

    raw_text = " ".join(lines[data_start:data_end])
    try:
        values = np.fromstring(raw_text, dtype=float, sep=" ", count=nx * ny * nz)
    except ValueError as e:
        raise ValueError(
            f"Could not parse grid data as floats in '{filename}' "
            f"between lines {data_start} and {data_end}."
        ) from e

    if values.size != nx * ny * nz:
        raise ValueError(
            f"Expected {nx*ny*nz} grid points ({nx}x{ny}x{nz}) in "
            f"'{filename}' but parsed {values.size} values."
        )

    return values.reshape(dims, order="F")

# %%
def build_species_rename_map(
    atoms: Atoms, 
    kind_names: Sequence[str]
) -> Dict[str, str]:
    """Map ASE's own auto-generated QE species labels (e.g. 'Co', 'Co1')
    to the desired kind_name scheme (e.g. 'Co1', 'Co2').

    Parameters
    ----------
    atoms : ase.Atoms
        The exact Atoms object passed to `ase.io.write(..., format="espresso-in", ...)`,
        with `set_initial_magnetic_moments` already called on it.
    kind_names : sequence of str
        Per-atom kind names, same order as `atoms` (e.g. from
        `new_structure.site_properties["kind_name"]`).

    Returns
    -------
    dict mapping ASE's label -> desired label, ready for `relabel_qe_species`.
    """
    magmoms = atoms.get_initial_magnetic_moments()
    seen: dict = {}
    rename_map: Dict[str, str] = {}
    for symbol, magmom, kind in zip(atoms.get_chemical_symbols(), magmoms, kind_names):
        key = (symbol, magmom)
        if key not in seen:
            count_so_far = sum(1 for s, _ in seen if s == symbol)
            ase_label = symbol if count_so_far == 0 else f"{symbol}{count_so_far}"
            seen[key] = ase_label
            rename_map[ase_label] = kind
    return rename_map

def relabel_qe_species(
    filename: str, 
    rename_map: Dict[str, str]
) -> None:
    """Rewrite only the species-label TOKEN (first whitespace-separated
    field) on ATOMIC_SPECIES and ATOMIC_POSITIONS lines of a QE input
    file, using `rename_map`. Every other line (namelists, K_POINTS,
    CELL_PARAMETERS, comments, blank lines) is left untouched.
    """
    lines = open(filename).read().splitlines(keepends=True)
    out = []
    in_species_or_positions = False
    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith(("ATOMIC_SPECIES", "ATOMIC_POSITIONS")):
            in_species_or_positions = True
            out.append(line)
            continue
        if upper.startswith(("K_POINTS", "CELL_PARAMETERS", "CONSTRAINTS",
                              "OCCUPATIONS", "ATOMIC_FORCES", "ADDITIONAL_K_POINTS")):
            in_species_or_positions = False
            out.append(line)
            continue
        if in_species_or_positions and stripped:
            parts = line.split(None, 1)
            label = parts[0]
            if label in rename_map:
                new_label = rename_map[label]
                if len(parts) > 1:
                    out.append(f"{new_label:<4}{parts[1]}")
                else:
                    out.append(f"{new_label}{line[len(line.rstrip()):]}")
                continue
        out.append(line)
    with open(filename, "w") as f:
        f.writelines(out)