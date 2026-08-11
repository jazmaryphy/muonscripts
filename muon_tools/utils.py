# %%
import numpy as np
from typing import Literal, Optional, Tuple
from numpy.typing import ArrayLike, NDArray

from pymatgen.core import Structure
from pymatgen.symmetry import analyzer
from pymatgen.util.coord import pbc_shortest_vectors

# %%
def _pbc_close(a, b, tol=1e-3):
    """Check if fractional coordinates match under periodic boundary conditions."""
    diff = (np.asarray(a) - np.asarray(b) + 0.5) % 1.0 - 0.5
    return np.all(np.abs(diff) < tol)

# %%
def get_supercell_matrix(
    input_structure: Structure,
    target_structure: Structure
) -> np.ndarray:
    """
    Determine the integer supercell transformation matrix that maps
    `input_structure` onto `target_structure`.

    Parameters
    ----------
    input_structure : pymatgen.core.Structure
        Primitive (or smaller) structure.

    target_structure : pymatgen.core.Structure
        Supercell structure.

    Returns
    -------
    ndarray, shape (3, 3)
        Integer supercell transformation matrix.
    """
    matrix = (
        np.dot(target_structure.lattice.matrix, np.linalg.inv(input_structure.lattice.matrix))
    )

    matrix = np.rint(matrix).astype(int)

    # if not np.allclose(matrix,
    #                    target_structure.lattice.matrix
    #                    @ np.linalg.inv(input_structure.lattice.matrix)):
    #     raise ValueError(
    #         "Target lattice is not an integer supercell of the input lattice."
    #     )

    return matrix

# %%
def find_defect_index(
    structure: Structure,
    defect_label: str="H",
    which: Literal["first", "last", "unique"] = "last"
) -> Optional[int]:
    """
    Return the index of a defect atom in a structure.

    The defect is identified by its species label. If multiple atoms with
    the same label are present, the ``which`` argument determines which
    index is returned.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        Structure containing the defect atom.
    defect_label : str, default="H"
        Species label used to identify the defect.
    which : {"first", "last", "unique"}, default="last"
        Selection rule when multiple matching atoms exist.

        - ``"first"`` : return the first matching atom.
        - ``"last"`` : return the last matching atom.
        - ``"unique"`` : require exactly one matching atom, otherwise
          raise a ``ValueError``.

    Returns
    -------
    int or None
        Index of the defect atom, or ``None`` if no matching atom is
        found.

    Raises
    ------
    ValueError
        If ``which="unique"`` and zero or multiple matching atoms are
        found.
    """
    matches = [
        i
        for i, site in enumerate(structure)
        if site.specie.symbol == defect_label
    ]

    if which == "unique":
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one '{defect_label}' atom, "
                f"found {len(matches)}."
            )
        return matches[0]

    if not matches:
        return None

    if which == "first":
        return matches[0]

    if which == "last":
        return matches[-1]

    raise ValueError(
        f"Unknown value for 'which': {which!r}. "
        "Expected 'first', 'last', or 'unique'."
    )


def get_muon_index(structure: Structure) -> Optional[int]:
    return find_defect_index(structure, defect_label="H", which="last")

# %%
def prune_too_close_pos(
    frac_positions: ArrayLike,
    host_lattice: Structure,
    min_distance: float,
    energies: Optional[ArrayLike] = None,
    e_tol: float = 0.05,
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Return indices and mappings of atoms too close to another one in the cell.

    If energies are not passed, only inter-atomic distance is considered.
    Otherwise both conditions (distance and same energy) must be verified.

    Parameters
    ----------
    frac_positions : numpy.array
        The N x 3 array containing scaled atomic positions.

    host_lattice : pymatgen.core.Structure
        The lattice structure. Only its lattice property is used.

    min_distance: float
         Minimum distance in Angstrom between atoms. Atoms closer than this
         will be considered the same unless they have different energy associated.

    energies: list or numpy.array
         Energy (or any other scalar property) associated with positions
         reported in frac_positions.

    e_tol: float
        Absolute difference between the scalar property associated with atomic sites.

    Returns
    -------
    s_idx : ndarray
        An array of integers. If the value of the item equals its index, 
        the atom is unique. If the value is -1, it is close to a prior atom.
    mapping : ndarray
        An array of integers mapping each atom to the index of the first 
        matching unique atom that satisfied the distance/scalar conditions.

    Suggestions:
                 1. modify -1 into the index of the first atom that matched the conditions
                    on energy and distance. -> this is the mapping
                 2. change `energies` into `scalar_value` to make it more general.

    """

    # energies and tolerance should be in eV
    lattice = host_lattice.lattice

    s_idx = np.arange(len(frac_positions))
    mapping = np.arange(len(frac_positions)) + 1  
    mapping[0] = 1

    for i, pi in enumerate(frac_positions):
        for j, pj in enumerate(frac_positions):
            if j > i:
                diff = pbc_shortest_vectors(lattice, pi, pj).squeeze()
                # print(i,j,diff,np.linalg.norm(diff, axis=0))
                if (energies is not None) and (len(energies) == len(frac_positions)):
                    if (np.linalg.norm(diff, axis=0) < min_distance) and (
                        abs(energies[i] - energies[j]) < e_tol
                    ):
                        s_idx[j] = -1
                        
                        mapping[j] = mapping[i]
                        #print(i,j,mapping)
                else:
                    if np.linalg.norm(diff, axis=0) < min_distance:
                        s_idx[j] = -1
                        mapping[j] = mapping[i]

    return s_idx, mapping


def get_equivalent_sites(
    fcoords: ArrayLike,
    host_lattice: Structure,
    min_distance: float = 0.5,
    symprec: float = 1e-3,
    energies: Optional[ArrayLike] = None,
    e_tol: float = 0.05,
) -> NDArray[np.float64]:
    """Generate symmetry-equivalent muon positions and remove duplicate/near-equivalent sites.

    Parameters
    ----------
    fcoords : (N, 3) array_like
        Input fractional coordinates.
    host_lattice : Structure
        Host crystal structure.
    min_distance : float
        Minimum separation (Å) used for deduplication.
    symprec : float
        Symmetry tolerance.
    energies : array_like, optional
        Site energies.
    e_tol : float
        Energy tolerance for site merging.

    Returns
    -------
    ndarray
        Unique fractional coordinates.
    """

    fcoords = np.atleast_2d(fcoords) % 1.0

    spg = analyzer.SpacegroupAnalyzer(host_lattice, symprec=symprec,)
    ops = spg.get_symmetry_operations(cartesian=False)

    replicas = np.vstack([
        op.operate_multi(fcoords) % 1.0
        for op in ops
    ])

    # If energies are provided, they must be duplicated symmetrically to match replicas
    replica_energies = None
    if energies is not None:
        flat_energies = np.asarray(energies).flatten()
        replica_energies = np.tile(flat_energies, len(ops))


    idx, _ = prune_too_close_pos(
        replicas,
        host_lattice.copy(),
        min_distance,
        energies=replica_energies,
        e_tol=e_tol,
    )

    return replicas[idx == np.arange(len(replicas))]