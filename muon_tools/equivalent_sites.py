# %%
import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry import analyzer
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.electronic_structure.core import Magmom

# %%
def prune_too_close_pos(
    frac_positions, host_lattice, min_distance, energies=None, e_tol=0.05
    ):
    """
    Returns index of atom too close to another one in the cell.

    If energies are not passed, only inter-atomic distance is considered.
    Otherwise both conditions (distance and same energy) must be verified.

    Parameters
    ----------
    frac_positions : numpy.array
        The nAtoms x 3 array containing scaled atomic positions.

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
    np.array
        A list of integers.
        If the value of the item equals its index, the atoms is not within
        `min_distance` from others (or the energy threshold is not satisfied).
        If the value is -1, the atom (and possibly the energy) is close to another
        one in the cell.

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

    # frac_positions = np.delete(frac_positions,s_idx,0) #use if append
    # frac_positions = frac_positions[s_idx == np.arange(len(frac_positions))]
    return s_idx, mapping

# %%
def find_equivalent_muons(
    host_lattice,
    frac_coords,
    min_distance=0.5,
    a_tol=1e-3,
    energies=None,
    e_tol=0.05,
    per_site=False,
):
    """
    Generate symmetry-equivalent positions and prune those too close
    in real space.

    If per_site=True, symmetry replicas are generated and pruned
    independently for each input site.
    """

    frac_coords = np.atleast_2d(frac_coords) % 1.0
    spg = analyzer.SpacegroupAnalyzer(host_lattice, symprec=a_tol)
    ops = spg.get_symmetry_operations(cartesian=False)

    if not per_site:
        # Global mode (all sites together)
        all_pos = np.vstack([
            op.operate_multi(frac_coords) % 1.0
            for op in ops
        ]) % 1.0

        idx, mapping = prune_too_close_pos(
            all_pos,
            host_lattice.copy(),
            min_distance,
            energies=energies,
            e_tol=e_tol,
        )
        #print(f"idx: {idx}")
        #print(f"mapping: {mapping}")

        return all_pos[idx == np.arange(len(all_pos))]

    # Per-site mode
    unique_pos_per_site = []

    for i, fc in enumerate(frac_coords):
        # Generate replicas for ONE site
        replicas = np.vstack([
            op.operate(fc) % 1.0
            for op in ops
        ])

        # Optional per-site energies
        site_energies = None
        if energies is not None:
            site_energies = np.full(len(replicas), energies[i])

        idx, mapping = prune_too_close_pos(
            replicas,
            host_lattice.copy(),
            min_distance,
            energies=site_energies,
            e_tol=e_tol,
        )
        #print(f"idx: {idx}")
        #print(f"mapping: {mapping}")

        unique_pos_per_site.append(
            replicas[idx == np.arange(len(replicas))]
        )

    return unique_pos_per_site

# %%
def get_equivalent_sites(
    fcoords,
    host_lattice,
    min_distance=0.5,
    symprec=1e-3,
    energies=None,
    e_tol=0.05,
):
    """
    Generate symmetry-equivalent muon positions and remove
    duplicate/near-equivalent sites.

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

    spg = analyzer.SpacegroupAnalyzer(
        host_lattice,
        symprec=symprec,
    )

    ops = spg.get_symmetry_operations(cartesian=False)

    replicas = np.vstack([
        op.operate_multi(fcoords) % 1.0
        for op in ops
    ])

    idx, _ = prune_too_close_pos(
        replicas,
        host_lattice.copy(),
        min_distance,
        energies=energies,
        e_tol=e_tol,
    )

    return replicas[idx == np.arange(len(replicas))]

# %%
def find_defect_index(
    structure,
    defect_label="H",
    which="last",
):
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


def find_muon_index(structure):
    return find_defect_index(structure, defect_label="H", which="last")

# %%
import numpy as np
from pymatgen.core import Structure


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


# %%
import os
dpath='/home/misah/projects/muonscripts/DATA/UCoGe'

# %%
filename = 'UCoGe.cif'
filename = os.path.join(dpath, filename)
st = Structure.from_file(filename)

st

# %%
siteA = [0.56409,       0.08332,       0.24623]

find_equivalent_muons(st, siteA)
get_equivalent_sites(siteA, st)

# %%
siteB = [0.94777,       0.16786,       0.24452]
# siteB = [0.0,       0.0,       0.5]

find_equivalent_muons(st, siteB)
get_equivalent_sites(siteB, st)
get_equivalent_sites(siteB, st, 0.5)

# %%
sc_matrix = [
    [2, 0, 0], 
    [0, 2, 0], 
    [0, 0, 3]
    ]

sitr = [0.28204,       0.04166,      0.08208]
stc = st.copy()
stc.make_supercell(sc_matrix)
pos = get_equivalent_sites(sitr, stc, 0.5)
print(len(pos), '==', 4*12)
pos

# %%
pos = get_equivalent_sites(sitr, st, 0.5)
print(len(pos), '==', 4*12)
pos

# %%
subpath = os.path.join(dpath, "query/mu+")
subpath

# %%
from ase.io import read as read_from_file
from pymatgen.io.ase import AseAtomsAdaptor

file = '1.out' # highest energy site
file = '4.out' # lowest energy site
file = os.path.join(subpath, file)
final_struct = read_from_file(file, format="espresso-out", index=-1)
final_struct2 =  final_struct.copy()

muon_idx = np.flatnonzero(final_struct2.numbers == 1)
if len(muon_idx) != 1:
    raise ValueError(f"Expected exactly one muon, found {len(muon_idx)}.")
muon_idx = muon_idx.item()
muon_position_rlx = final_struct2.get_scaled_positions()[muon_idx]
del final_struct2[muon_idx]

final_struct2, muon_position_rlx

# %%
from muon_tools.aiida_muon.distortions import get_distortions, get_struct_wt_distortions

rlxd_stc = AseAtomsAdaptor.get_structure(final_struct)
# get_equivalent_sites(sitr, st, 0.5)

# find_defect_index
scmatrix = get_supercell_matrix(st, rlxd_stc)

init_supc2 = st.copy()
init_supc2.make_supercell(scmatrix)

mu_idx = find_muon_index(rlxd_stc)
mupos_rlx = rlxd_stc.frac_coords[mu_idx]

mupos_rlx
equiv_pos = get_equivalent_sites(mupos_rlx, st)
nwp = equiv_pos[0]

# nw_st = get_struct_wt_distortions(
#     init_supc2.copy(),
#     rlxd_stc.copy(),
#     nwp,
#     st,
# )

# mu_idx, nw_st.num_sites, rlxd_stc.num_sites
# nw_st.frac_coords-rlxd_stc.frac_coords

new_st = []
for nwp in equiv_pos:
    nw_st = get_struct_wt_distortions(
        init_supc2.copy(),
        rlxd_stc.copy(),
        nwp,
        st,
    )
    new_st.append(nw_st)

len(new_st)


idx=2
equiv_pos[idx], new_st[idx].frac_coords-rlxd_stc.frac_coords

# %%
# def periodic_difference(a, b):
#     """Minimum-image fractional-coordinate difference `a - b`, wrapped to
#     [-0.5, 0.5) in every component (periodic boundary conditions)."""
#     return (np.asarray(a) - np.asarray(b) + 0.5) % 1.0 - 0.5


# def periodic_equal(a, b, tol=1e-3):
#     """True if fractional coordinates `a` and `b` are the same point up to
#     a lattice translation, within `tol`."""
#     return np.all(np.abs(periodic_difference(a, b)) < tol)

# %%
1/8

# %%


# %%
"""Find structurally- and/or magnetically-equivalent muon sites in a
crystal, and generate approximate relaxed structures for each equivalent
site by "transplanting" an already-computed DFT relaxation's distortion
pattern onto it via the appropriate symmetry operation -- avoiding the
need to re-relax every symmetry-equivalent site from scratch.

Standalone reimplementation/generalization of aiida-muon's
`get_struct_wt_distortions` (https://github.com/positivemuon/aiida-muon),
with two changes:

1. EXPLICIT structural-vs-magnetic handling. pymatgen's
   `SpacegroupAnalyzer` does NOT automatically account for magnetic
   moments even if a 'magmom' site property is attached (verified: it
   returns identical operations with or without magmoms) -- so magnetic
   consistency is checked HERE explicitly, by transforming each site's
   moment as a pseudovector (m' = det(R) * R @ m -- picks up an extra
   sign flip under improper operations that a normal position vector
   does not) and checking it matches the moment actually present at the
   symmetry-mapped destination atom. Passing `magmom=None` naturally
   collapses this to "all structural operations are also magnetic
   operations", so the SAME code path handles non-magnetic materials
   correctly without a separate branch.

2. Muon identification is controllable (`muon_label`/`muon_index`),
   addressing the original docstring's own flagged caveat ("This is
   probably a problem when H atoms are already present").

The resulting per-site structures are ordinary ASE/pymatgen structures
with the muon included as an atom, so they plug directly into a
dipolar-field or point-charge-EFG calculation (e.g. `point_charge_EFG`/
`build_point_charge_efg_neighbors` from earlier in this project) to get
the local field at every equivalent site from a single relaxation.
"""

import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# ---------------------------------------------------------------------------
# 1. Symmetry operations: structural (always) + magnetic (if magmom given)
# ---------------------------------------------------------------------------

def _frac_periodic_close(a, b, tol):
    """True if fractional coordinates `a` and `b` are the same point up to
    a lattice translation (periodic boundary conditions), within `tol`."""
    diff = (np.asarray(a) - np.asarray(b) + 0.5) % 1.0 - 0.5
    return np.all(np.abs(diff) < tol)


def _op_preserves_magnetism(op, frac_coords, magmom, tol=1e-3):
    """Whether symmetry operation `op` leaves the magnetic arrangement
    `magmom` (per-site (3,) vectors) invariant -- i.e. maps every atom to
    another atom carrying the correctly pseudovector-transformed moment.
    """
    R = op.rotation_matrix
    det_r = np.linalg.det(R)
    for p, m in zip(frac_coords, magmom):
        new_p = op.operate(p) % 1.0
        matches = [j for j, q in enumerate(frac_coords) if _frac_periodic_close(new_p, q, tol)]
        if not matches:
            return False  # shouldn't happen for a genuine structural symm op
        transformed_m = det_r * (R @ m)
        if not np.allclose(transformed_m, magmom[matches[0]], atol=tol):
            return False
    return True


def get_structural_and_magnetic_ops(structure, magmom=None, symprec=1e-3, angle_tolerance=5.0):
    """Structural (full) and magnetic (moment-preserving subset) symmetry
    operations of `structure`.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        The HOST lattice, WITHOUT the muon -- symmetry should be
        evaluated on the pristine structure, not a muon-perturbed one
        (the muon itself locally breaks symmetry around the site you're
        trying to classify).
    magmom : array-like or None, default=None
        Per-site magnetic moments. Accepts:
        - None: non-magnetic. `magnetic_ops` will equal `structural_ops`.
        - (n_sites,) scalar per site: interpreted as a signed moment
          along the z-axis (common collinear/VASP-MAGMOM convention).
        - (n_sites, 3): full non-collinear moment vectors.
    symprec, angle_tolerance : passed to `SpacegroupAnalyzer`.

    Returns
    -------
    dict with keys:
        "structural_ops" : list of pymatgen SymmOp
        "magnetic_ops"    : list of pymatgen SymmOp (subset of the above)
        "is_magnetic"     : bool -- whether a non-trivial `magmom` was given
    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    structural_ops = sga.get_symmetry_operations()

    if magmom is None:
        return {"structural_ops": structural_ops, "magnetic_ops": structural_ops, "is_magnetic": False}

    magmom = np.asarray(magmom, dtype=float)
    if magmom.ndim == 1:
        magmom = np.column_stack([np.zeros_like(magmom), np.zeros_like(magmom), magmom])

    is_magnetic = bool(np.any(np.abs(magmom) > symprec))
    if not is_magnetic:
        return {"structural_ops": structural_ops, "magnetic_ops": structural_ops, "is_magnetic": False}

    frac_coords = structure.frac_coords
    magnetic_ops = [
        op for op in structural_ops
        if _op_preserves_magnetism(op, frac_coords, magmom, tol=symprec)
    ]
    return {"structural_ops": structural_ops, "magnetic_ops": magnetic_ops, "is_magnetic": True}


# ---------------------------------------------------------------------------
# 2. Orbit of a muon site under a set of operations
# ---------------------------------------------------------------------------

def get_equivalent_sites(mu_frac, ops, tol=1e-3, lattice=None, min_distance=None):
    """Orbit of fractional position `mu_frac` under symmetry operations
    `ops`, deduplicated (periodic boundary conditions respected).

    Parameters
    ----------
    tol : float, default=1e-3
        FRACTIONAL-coordinate tolerance used for the first, cheap dedup
        pass while building the orbit. Note this is NOT a physically
        uniform distance for an anisotropic cell -- the same `tol`
        corresponds to a tiny real distance along a short lattice vector
        but a much larger one along a long vector. Fine as a first pass
        (catches exact/near-exact symmetry-image duplicates cheaply), but
        see `lattice`/`min_distance` below for the physically-correct
        cleanup that should generally follow it.
    lattice : pymatgen.core.Lattice or pymatgen.core.Structure, optional
        If given (together with `min_distance`), an ADDITIONAL cleanup
        pass is run via `prune_close_positions`, using the TRUE periodic
        (minimum-image) Cartesian distance rather than raw fractional
        difference. This matters for anisotropic cells: e.g. two points
        differing by 0.005 in fractional coordinate along a 1 A axis are
        only 0.005 A apart physically (should be merged), but along a
        20 A axis they'd be 0.1 A apart (should probably stay separate)
        -- a fixed fractional `tol` can't distinguish these cases, but a
        real-distance check can. None (default): skip this pass, keep
        the old fractional-only behaviour.
    min_distance : float [Angstrom], optional
        Real-space merge threshold for the `lattice`-based cleanup pass.
        Only used if `lattice` is also given.

    Returns
    -------
    list of (3,) ndarray
        Unique equivalent fractional positions, including `mu_frac` itself.
    """
    sites = []
    for op in ops:
        new_p = op.operate(mu_frac) % 1.0
        if not any(_frac_periodic_close(new_p, s, tol) for s in sites):
            sites.append(new_p)

    if lattice is not None and min_distance is not None and len(sites) > 1:
        keep_mask, _groups = prune_close_positions(
            np.array(sites), lattice, min_distance=min_distance, energies=None
        )
        sites = [s for s, keep in zip(sites, keep_mask) if keep]

    return sites


# ---------------------------------------------------------------------------
# 3. Muon identification (configurable, unlike the hard-coded "first H")
# ---------------------------------------------------------------------------

def find_muon_index(structure, muon_label="H", which="last"):
    """Index of the muon site in `structure`, identified by species symbol.

    See `which` options in load_subset_data.py's `find_muon_index` (same
    semantics) -- 'last' (default, matches the convention of appending
    the muon as an extra atom), 'first', or 'unique' (raise if ambiguous).
    Returns None if `muon_label` doesn't appear in `structure` at all.
    """
    matches = [i for i, sp in enumerate(structure.species) if str(sp) == muon_label]
    if not matches:
        return None
    if which == "unique":
        if len(matches) > 1:
            raise ValueError(
                f"{len(matches)} atoms found with label '{muon_label}', expected exactly 1."
            )
        return matches[0]
    if which == "first":
        return matches[0]
    return matches[-1]  # 'last' (default)


# ---------------------------------------------------------------------------
# 4. Distortion transplanting (cleaned-up, non-mutating version)
# ---------------------------------------------------------------------------

def transplant_distortion(prist_stc, rlxd_stc, target_mupos, ops, muon_label="H", muon_index="last", tol=1e-3):
    """Approximate the relaxed structure for a symmetry-equivalent muon
    site, by transplanting the ALREADY-COMPUTED relaxation distortion
    (rlxd_stc - prist_stc) onto `target_mupos` via whichever operation in
    `ops` maps the original relaxed muon position onto it.

    Unlike the original aiida-muon function, this does NOT mutate
    `prist_stc`/`rlxd_stc` in place (both are copied first), and muon
    identification is controllable via `muon_label`/`muon_index` rather
    than always taking the first H atom.

    Parameters
    ----------
    prist_stc : pymatgen.core.Structure
        Pristine host (muon not yet added, or will be stripped if present).
    rlxd_stc : pymatgen.core.Structure
        Relaxed structure INCLUDING the muon.
    target_mupos : (3,) array
        Fractional coordinates of the target (symmetry-equivalent) site.
    ops : list of pymatgen SymmOp
        Symmetry operations to search for one mapping the relaxed muon
        position onto `target_mupos` -- pass `structural_ops` or
        `magnetic_ops` from `get_structural_and_magnetic_ops` depending
        on which orbit `target_mupos` came from.
    muon_label, muon_index : see `find_muon_index`.
    tol : float, default=1e-3
        Matching tolerance (fractional coordinates).

    Returns
    -------
    pymatgen.core.Structure or None
        The approximate relaxed structure at `target_mupos`, or None if
        no matching operation was found, or if placing the muon there
        would clash with an existing atom (proximity check).
    """
    prist_stc = prist_stc.copy()
    rlxd_stc = rlxd_stc.copy()

    mu_idx_rlxd = find_muon_index(rlxd_stc, muon_label=muon_label, which=muon_index)
    if mu_idx_rlxd is None:
        raise ValueError(f"No atom with muon_label='{muon_label}' found in rlxd_stc.")
    mupos_rlx = rlxd_stc.frac_coords[mu_idx_rlxd]
    rlxd_stc.remove_sites([mu_idx_rlxd])

    mu_idx_prist = find_muon_index(prist_stc, muon_label=muon_label, which=muon_index)
    if mu_idx_prist is not None:
        prist_stc.remove_sites([mu_idx_prist])

    if len(rlxd_stc.frac_coords) != len(prist_stc.frac_coords):
        raise ValueError(
            f"Host atom count mismatch after removing the muon: "
            f"rlxd={len(rlxd_stc.frac_coords)}, prist={len(prist_stc.frac_coords)}. "
            f"Are prist_stc/rlxd_stc really the same host lattice?"
        )

    matching_op = None
    for op in ops:
        if _frac_periodic_close(op.operate(mupos_rlx) % 1.0, target_mupos, tol):
            matching_op = op
            break
    if matching_op is None:
        return None  # target_mupos isn't in the orbit of mupos_rlx under `ops`

    disp = rlxd_stc.frac_coords - prist_stc.frac_coords
    t_disp = matching_op.operate_multi(disp)

    nw_stc = prist_stc.copy()
    for i in range(len(nw_stc)):
        nw_stc.translate_sites(i, t_disp[i], frac_coords=True, to_unit_cell=False)

    try:
        nw_stc.append(
            species=muon_label,
            coords=target_mupos,
            coords_are_cartesian=False,
            validate_proximity=True,
            properties={"kind_name": muon_label},
        )
    except ValueError:
        return None  # muon placement clashes with an existing atom

    return nw_stc


# ---------------------------------------------------------------------------
# 5b. Pruning near-duplicate positions from INDEPENDENT sources
#     (e.g. several separately-relaxed muon sites converging to the same
#     physical site) -- distinct from get_equivalent_sites' internal
#     symmetry-orbit dedup, since here energies matter too.
# ---------------------------------------------------------------------------

def prune_close_positions(frac_positions, lattice, min_distance, energies=None, e_tol=0.05):
    """Cluster/deduplicate a list of fractional positions that are
    physically the same site (within `min_distance` of each other via the
    true periodic/minimum-image distance), optionally also requiring
    their associated energies to agree within `e_tol`.

    This is for a DIFFERENT situation than `get_equivalent_sites`'
    internal deduplication: that one merges points known to be exact
    symmetry images of each other (generated from a single orbit, so a
    tight numerical tolerance is appropriate and no energy check is
    needed -- they're the same site by construction). This function is
    for merging positions from INDEPENDENT sources -- e.g. several
    separate DFT relaxations (possibly from different starting guesses,
    or from re-relaxing the candidate structures `generate_equivalent_
    muon_structures` produces) that may have converged to physically the
    same site without landing on numerically identical coordinates.

    The energy check matters: two sites can sit spatially close together
    (within `min_distance`) yet be genuinely DIFFERENT physical states
    (e.g. a true minimum next to a nearby saddle point) -- in that case
    you do NOT want to merge them just because they're geometrically
    close. Only positions that are BOTH close in space AND close in
    energy get merged; a close-in-space-but-different-energy pair is
    kept as two distinct sites. Mirrors aiida-muon's
    `prune_too_close_pos`, generalized here to work as a standalone
    utility independent of any particular workflow's data shape.

    Parameters
    ----------
    frac_positions : (N, 3) array
        Fractional coordinates to deduplicate.
    lattice : pymatgen.core.Lattice or pymatgen.core.Structure
        Used to compute the true periodic (minimum-image) distance. A
        Structure is accepted too (its `.lattice` is used), matching
        `host_lattice` usage elsewhere in this project.
    min_distance : float [Angstrom]
        Positions closer than this are CANDIDATES for merging.
    energies : (N,) array or None, default=None
        If given, candidates are only actually merged if their energies
        also agree within `e_tol`. If None, merging is decided by
        distance alone.
    e_tol : float [eV], default=0.05

    Returns
    -------
    keep_mask : (N,) bool array
        True for the representative position kept for each cluster (the
        first occurrence, in input order); False for positions merged
        into an earlier representative.
    groups : (N,) int array
        `groups[i]` gives the index of the representative that position
        `i` was merged into (`groups[i] == i` if `i` is itself a
        representative).
    """
    from pymatgen.util.coord import pbc_shortest_vectors

    lat = lattice.lattice if hasattr(lattice, "lattice") else lattice
    frac_positions = np.asarray(frac_positions)
    n = len(frac_positions)

    groups = np.arange(n)
    keep_mask = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep_mask[i]:
            continue  # i itself was already merged into an earlier group
        for j in range(i + 1, n):
            if not keep_mask[j]:
                continue
            diff = pbc_shortest_vectors(lat, frac_positions[i], frac_positions[j]).squeeze()
            if np.linalg.norm(diff) >= min_distance:
                continue
            if energies is not None and abs(energies[i] - energies[j]) >= e_tol:
                continue  # close in space but energetically distinct -> keep separate
            groups[j] = groups[i]
            keep_mask[j] = False

    return keep_mask, groups


# ---------------------------------------------------------------------------
# 6. Top-level orchestrator
# ---------------------------------------------------------------------------

def generate_equivalent_muon_structures(
    prist_stc, rlxd_stc, host_structure, magmom=None,
    muon_label="H", muon_index="last", tol=1e-3, symprec=1e-3,
    min_distance=0.05,
):
    """For a single DFT-relaxed muon site, find every structurally- and/or
    magnetically-equivalent site and generate an approximate relaxed
    structure for each (via distortion transplanting), ready for local-
    field calculations (dipolar/EFG) at every equivalent site.

    Parameters
    ----------
    prist_stc, rlxd_stc : pymatgen.core.Structure
        Pristine host (muon-free or will be stripped) and relaxed
        structure (with muon), as in `transplant_distortion`.
    host_structure : pymatgen.core.Structure
        Structure used to compute symmetry -- normally the same as
        `prist_stc` (muon-free host), kept as a separate argument in case
        you want to supply e.g. a primitive-cell version of the host for
        the symmetry search while `prist_stc`/`rlxd_stc` are supercells.
    magmom : array-like or None
        See `get_structural_and_magnetic_ops`. None -> non-magnetic;
        both site classifications collapse to the same thing.
    muon_label, muon_index : see `find_muon_index`.
    min_distance : float [Angstrom] or None, default=0.05
        Passed to `get_equivalent_sites` as its physical-distance cleanup
        pass (see there) -- catches near-duplicate orbit points that a
        purely fractional-coordinate tolerance can miss on an anisotropic
        cell. Set to None to skip this pass (fractional-tolerance-only
        behaviour).

    Returns
    -------
    list of dict, one per equivalent site (INCLUDING the original site),
    each with keys:
        "frac_pos" : (3,) ndarray -- fractional position of this site
        "structure" : pymatgen.core.Structure or None -- the (approximate)
            relaxed structure at this site (None if transplanting failed
            -- e.g. a genuine proximity clash; check this before using).
        "is_original" : bool -- True for the input site itself (returned
            as `rlxd_stc` unchanged, not transplanted).
        "is_magnetically_equivalent" : bool -- True if this site is in the
            orbit of the ORIGINAL site under the MAGNETIC symmetry (i.e.
            you can trust the transplanted local field to resemble the
            original site's); False if it's only structurally equivalent
            (found via the full, magnetism-ignoring symmetry) -- these
            need their own independent relaxation/field calculation if
            magnetism actually matters for what you're computing, exactly
            the "mag_inequivalent" concept from aiida-muon's clustering
            workflow, but exposed explicitly here per-site.
    """
    ops_info = get_structural_and_magnetic_ops(host_structure, magmom=magmom, symprec=symprec)
    structural_ops = ops_info["structural_ops"]
    magnetic_ops = ops_info["magnetic_ops"]

    mu_idx = find_muon_index(rlxd_stc, muon_label=muon_label, which=muon_index)
    if mu_idx is None:
        raise ValueError(f"No atom with muon_label='{muon_label}' found in rlxd_stc.")
    mu_frac = rlxd_stc.frac_coords[mu_idx]

    structural_sites = get_equivalent_sites(
        mu_frac, structural_ops, tol=tol, lattice=host_structure, min_distance=min_distance
    )
    magnetic_sites = get_equivalent_sites(
        mu_frac, magnetic_ops, tol=tol, lattice=host_structure, min_distance=min_distance
    )

    results = []
    for site in structural_sites:
        is_original = _frac_periodic_close(site, mu_frac, tol)
        is_mag_equiv = any(_frac_periodic_close(site, m, tol) for m in magnetic_sites)

        if is_original:
            structure = rlxd_stc.copy()
        else:
            ops_for_this_site = magnetic_ops if is_mag_equiv else structural_ops
            structure = transplant_distortion(
                prist_stc, rlxd_stc, site, ops_for_this_site,
                muon_label=muon_label, muon_index=muon_index, tol=tol,
            )

        results.append({
            "frac_pos": site,
            "structure": structure,
            "is_original": is_original,
            "is_magnetically_equivalent": is_mag_equiv,
        })

    return results


# ---------------------------------------------------------------------------
# 7. Class interface: caches the (expensive) symmetry search once per host
#    structure, so repeated calls for different muon sites on the SAME
#    host don't redundantly recompute it. Thin wrapper around everything
#    above -- no logic is duplicated, so all the correctness/perf testing
#    already done on the functions carries over unchanged.
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional


@dataclass
class EquivalentSite:
    """One entry from `MuonSiteFinder.generate_structures` / the standalone
    `generate_equivalent_muon_structures` function.

    Attributes
    ----------
    frac_pos : (3,) ndarray
        Fractional coordinates of this equivalent site.
    structure : pymatgen.core.Structure or None
        The (approximate) relaxed structure at this site -- None if
        transplanting the distortion here failed (proximity clash, or no
        symmetry operation connects it to the original site).
    is_original : bool
        True for the input site itself (`structure` is `rlxd_stc`
        unchanged, not transplanted).
    is_magnetically_equivalent : bool
        True if this site is in the orbit of the original site under the
        MAGNETIC symmetry (trust the transplanted local field); False if
        only structurally equivalent (needs its own relaxation/field
        calculation if magnetism matters) -- the "mag_inequivalent"
        concept from aiida-muon's clustering workflow, exposed per-site.
    """
    frac_pos: np.ndarray
    structure: Optional[object]
    is_original: bool
    is_magnetically_equivalent: bool

    @property
    def ok(self):
        """Shorthand for `structure is not None` -- did transplanting succeed."""
        return self.structure is not None


class MuonSiteFinder:
    """Find structurally- and magnetically-equivalent muon sites in a host
    crystal, and generate approximate relaxed structures for each one.

    The symmetry search (`get_structural_and_magnetic_ops`, the expensive
    part -- a `SpacegroupAnalyzer` call) is computed ONCE, lazily, on
    first use, and cached for the lifetime of the instance -- so reuse one
    `MuonSiteFinder` across every muon site/relaxation you have for a
    given host structure, rather than constructing a new one (or calling
    the standalone `generate_equivalent_muon_structures` function) each
    time, which would redundantly redo the symmetry search every call.

    Example
    -------
        finder = MuonSiteFinder(host_structure, magmom=magmom)

        # cheap: reuses the same cached symmetry search for every call
        for prist_stc, rlxd_stc in my_relaxation_results:
            for site in finder.generate_structures(prist_stc, rlxd_stc):
                if site.ok:
                    ... # feed site.structure into point_charge_EFG, etc.
    """

    def __init__(self, host_structure, magmom=None, symprec=1e-3, angle_tolerance=5.0):
        self.host_structure = host_structure
        self.magmom = magmom
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
        self._ops_info = None  # populated lazily by `ops_info`

    @property
    def ops_info(self):
        """dict with 'structural_ops', 'magnetic_ops', 'is_magnetic' --
        computed on first access, cached thereafter."""
        if self._ops_info is None:
            self._ops_info = get_structural_and_magnetic_ops(
                self.host_structure, magmom=self.magmom,
                symprec=self.symprec, angle_tolerance=self.angle_tolerance,
            )
        return self._ops_info

    @property
    def structural_ops(self):
        return self.ops_info["structural_ops"]

    @property
    def magnetic_ops(self):
        return self.ops_info["magnetic_ops"]

    @property
    def is_magnetic(self):
        return self.ops_info["is_magnetic"]

    def equivalent_sites(self, mu_frac, magnetic=False, tol=1e-3, min_distance=0.05):
        """Orbit of `mu_frac` under this host's structural (default) or
        magnetic symmetry. See the standalone `get_equivalent_sites` for
        parameter details."""
        ops = self.magnetic_ops if magnetic else self.structural_ops
        return get_equivalent_sites(
            mu_frac, ops, tol=tol, lattice=self.host_structure, min_distance=min_distance
        )

    def transplant(self, prist_stc, rlxd_stc, target_mupos, magnetic=True,
                    muon_label="H", muon_index="last", tol=1e-3):
        """Approximate relaxed structure at `target_mupos`, via
        `transplant_distortion` using this host's cached symmetry
        operations (magnetic by default -- see `magnetic`)."""
        ops = self.magnetic_ops if magnetic else self.structural_ops
        return transplant_distortion(
            prist_stc, rlxd_stc, target_mupos, ops,
            muon_label=muon_label, muon_index=muon_index, tol=tol,
        )

    def generate_structures(self, prist_stc, rlxd_stc, muon_label="H", muon_index="last",
                             tol=1e-3, min_distance=0.05):
        """All structurally-/magnetically-equivalent sites for the muon in
        `rlxd_stc`, each with an approximate relaxed `Structure` (or None
        on failure). Same result as the standalone
        `generate_equivalent_muon_structures`, just using this instance's
        cached symmetry operations instead of recomputing them.

        Returns
        -------
        list of EquivalentSite
        """
        mu_idx = find_muon_index(rlxd_stc, muon_label=muon_label, which=muon_index)
        if mu_idx is None:
            raise ValueError(f"No atom with muon_label='{muon_label}' found in rlxd_stc.")
        mu_frac = rlxd_stc.frac_coords[mu_idx]

        structural_sites = self.equivalent_sites(mu_frac, magnetic=False, tol=tol, min_distance=min_distance)
        magnetic_sites = self.equivalent_sites(mu_frac, magnetic=True, tol=tol, min_distance=min_distance)

        results = []
        for site in structural_sites:
            is_original = _frac_periodic_close(site, mu_frac, tol)
            is_mag_equiv = any(_frac_periodic_close(site, m, tol) for m in magnetic_sites)

            if is_original:
                structure = rlxd_stc.copy()
            else:
                structure = self.transplant(
                    prist_stc, rlxd_stc, site, magnetic=is_mag_equiv,
                    muon_label=muon_label, muon_index=muon_index, tol=tol,
                )

            results.append(EquivalentSite(
                frac_pos=site, structure=structure,
                is_original=is_original, is_magnetically_equivalent=is_mag_equiv,
            ))

        return results

    @staticmethod
    def prune(frac_positions, lattice, min_distance, energies=None, e_tol=0.05):
        """See the standalone `prune_close_positions` -- exposed here too
        as a convenience (it's a `staticmethod` since it doesn't need any
        cached symmetry state, unlike the rest of this class)."""
        return prune_close_positions(frac_positions, lattice, min_distance, energies=energies, e_tol=e_tol)

    def __repr__(self):
        n_struct = len(self.ops_info["structural_ops"]) if self._ops_info else "?"
        n_mag = len(self.ops_info["magnetic_ops"]) if self._ops_info else "?"
        return (
            f"MuonSiteFinder(host={self.host_structure.composition.reduced_formula}, "
            f"structural_ops={n_struct}, magnetic_ops={n_mag}, is_magnetic={self.is_magnetic})"
        )


# %%


# %%
import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def _pbc_close(a, b, tol=1e-3):
    """Check if fractional coordinates match under periodic boundary conditions."""
    diff = (np.asarray(a) - np.asarray(b) + 0.5) % 1.0 - 0.5
    return np.all(np.abs(diff) < tol)


def get_symmetry_operations(host_structure, magmom=None, symprec=1e-3):
    """
    Find structural and magnetic symmetry operations for a host crystal.
    
    magmom: None (non-magnetic), or (N,) scalar array, or (N,3) Cartesian vectors.
    """
    sga = SpacegroupAnalyzer(host_structure, symprec=symprec)
    struct_ops = sga.get_symmetry_operations()

    if magmom is None or not np.any(np.abs(magmom) > symprec):
        return struct_ops

    # Convert scalar moments (0, 0, m_z) to 3D Cartesian vectors if needed
    magmom = np.asarray(magmom, dtype=float)
    if magmom.ndim == 1:
        magmom = np.column_stack([np.zeros_like(magmom), np.zeros_like(magmom), magmom])

    lat_T = host_structure.lattice.matrix.T
    lat_T_inv = np.linalg.inv(lat_T)
    frac_coords = host_structure.frac_coords

    magnetic_ops = []
    for op in struct_ops:
        # Transform rotation matrix from fractional space to Cartesian space: R_cart = L^T @ R_frac @ (L^T)^-1
        R_cart = lat_T @ op.rotation_matrix @ lat_T_inv
        det_R = np.linalg.det(R_cart)

        is_valid = True
        for p, m in zip(frac_coords, magmom):
            new_p = op.operate(p) % 1.0
            match_idx = next((j for j, q in enumerate(frac_coords) if _pbc_close(new_p, q, symprec)), None)
            
            if match_idx is None:
                is_valid = False
                break
                
            # Pseudovector transformation: m' = det(R) * (R @ m)
            transformed_m = det_R * (R_cart @ m)
            if not np.allclose(transformed_m, magmom[match_idx], atol=symprec):
                is_valid = False
                break

        if is_valid:
            magnetic_ops.append(op)

    return magnetic_ops


def transplant_distortion(
    prist_stc, 
    rlxd_stc, 
    target_mupos, 
    host_structure, 
    magmom=None, 
    muon_label="H", 
    tol=1e-3
):
    """
    Transplant host distortion from a DFT-relaxed muon calculation onto target_mupos.
    
    Returns a new Structure with the transplanted relaxed lattice + muon at target_mupos,
    or None if target_mupos is not symmetry-equivalent or placement collides.
    """
    prist_stc = prist_stc.copy()
    rlxd_stc = rlxd_stc.copy()

    # Find and remove muon (defaults to last site matching muon_label)
    mu_rlx_indices = [i for i, s in enumerate(rlxd_stc.species) if str(s) == muon_label]
    if not mu_rlx_indices:
        raise ValueError(f"Muon label '{muon_label}' not found in relaxed structure.")
    mu_idx = mu_rlx_indices[-1]
    
    mu_pos_rlx = rlxd_stc.frac_coords[mu_idx]
    rlxd_stc.remove_sites([mu_idx])

    prist_mu_indices = [i for i, s in enumerate(prist_stc.species) if str(s) == muon_label]
    if prist_mu_indices:
        prist_stc.remove_sites([prist_mu_indices[-1]])

    # Get valid symmetry operations
    ops = get_symmetry_operations(host_structure, magmom=magmom, symprec=tol)

    # Find operation mapping original relaxed muon position -> target_mupos
    matching_op = next((op for op in ops if _pbc_close(op.operate(mu_pos_rlx) % 1.0, target_mupos, tol)), None)
    if matching_op is None:
        return None

    # Calculate displacements and map them via the symmetry operation
    displacements = rlxd_stc.frac_coords - prist_stc.frac_coords
    transformed_displacements = matching_op.operate_multi(displacements)

    # Apply displacements to pristine host
    new_stc = prist_stc.copy()
    for i in range(len(new_stc)):
        new_stc.translate_sites(i, transformed_displacements[i], frac_coords=True, to_unit_cell=False)

    # Append muon at target location
    try:
        new_stc.append(
            species=muon_label,
            coords=target_mupos,
            coords_are_cartesian=False,
            validate_proximity=True,
            properties={"kind_name": muon_label},
        )
    except ValueError:
        return None  # Collision detection safeguard

    return new_stc


def get_all_equivalent_structures(prist_stc, rlxd_stc, host_structure, magmom=None, muon_label="H", tol=1e-3):
    """
    Generates relaxed structures for ALL symmetry-equivalent sites in the unit cell.
    """
    ops = get_symmetry_operations(host_structure, magmom=magmom, symprec=tol)
    
    mu_idx = [i for i, s in enumerate(rlxd_stc.species) if str(s) == muon_label][-1]
    mu_pos_rlx = rlxd_stc.frac_coords[mu_idx]

    # Build unique orbit of equivalent fractional positions
    equivalent_sites = []
    for op in ops:
        site = op.operate(mu_pos_rlx) % 1.0
        if not any(_pbc_close(site, existing, tol) for existing in equivalent_sites):
            equivalent_sites.append(site)

    # Generate structures for each site in orbit
    results = []
    for site in equivalent_sites:
        stc = transplant_distortion(
            prist_stc, rlxd_stc, target_mupos=site, host_structure=host_structure,
            magmom=magmom, muon_label=muon_label, tol=tol
        )
        if stc is not None:
            results.append({"site": site, "structure": stc})

    return results

# %%
sites = get_all_equivalent_structures(
    init_supc2.copy(), rlxd_stc.copy(), host_structure=st, magmom=None,
)
 
len(sites)
idx=2
sites[idx]['site'], sites[idx]['structure'].frac_coords-rlxd_stc.frac_coords

# %%
magmoms = [2.0, -2.0, 0.0, 0.0]

magmoms = np.array([
    [ 1.5,  1.5,  0.0],   # Atom 0: moment pointing in x-y plane
    [-1.5, -1.5,  0.0],   # Atom 1: opposing moment
    [ 0.0,  0.0,  2.1],   # Atom 2: moment along z
    [ 0.0,  0.0,  0.0],   # Atom 3: non-magnetic
])

np.any(np.abs(magmoms) > 1e-3)

# %%
# # result = finder.generate_structures(prist_stc, rlxd)


# sites = generate_equivalent_muon_structures(
#     init_supc2.copy(), rlxd_stc.copy(), host_structure=st, magmom=None,
# )
 
# len(sites)
# sites[1]['structure'].frac_coords-rlxd_stc.frac_coords


