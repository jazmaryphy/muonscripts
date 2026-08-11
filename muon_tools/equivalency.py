# %%
import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry import analyzer
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.electronic_structure.core import Magmom

from muon_tools.utils import _pbc_close, get_supercell_matrix, get_muon_index

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

    return s_idx, mapping


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

    spg = analyzer.SpacegroupAnalyzer(host_lattice, symprec=symprec,)

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
def get_symmetry_operations(
    host_lattice, 
    magmom=None, 
    symprec=1e-3
    ):
    """
    Find structural and magnetic symmetry operations for a host crystal.
    
    magmom: None (non-magnetic), or (N,) scalar array, or (N,3) Cartesian vectors.
    """
    sga = analyzer.SpacegroupAnalyzer(host_lattice, symprec=symprec)
    struct_ops = sga.get_symmetry_operations()

    if magmom is None or not np.any(np.abs(magmom) > symprec):
        return struct_ops

    # Convert scalar moments (0, 0, m_z) to 3D Cartesian vectors if needed
    magmom = np.asarray(magmom, dtype=float)
    if magmom.ndim == 1:
        magmom = np.column_stack([np.zeros_like(magmom), np.zeros_like(magmom), magmom])

    lat_T = host_lattice.lattice.matrix.T
    lat_T_inv = np.linalg.inv(lat_T)
    frac_coords = host_lattice.frac_coords

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

# %%
def transplant_distortion(
    prist_stc, 
    rlxd_stc, 
    n_mupos, 
    ipt_st, 
    magmom=None
):
    """
    Transplant host distortion from a DFT-relaxed muon calculation onto target_mupos.
    
    Returns a new Structure with the transplanted relaxed lattice + muon at target_mupos,
    or None if target_mupos is not symmetry-equivalent or placement collides.
    """
    prist_stc = prist_stc.copy()
    rlxd_stc = rlxd_stc.copy()

    # incase of prist_stc is unitcell structure
    scell_matrix = get_supercell_matrix(prist_stc, rlxd_stc)
    prist_stc.make_supercell(scell_matrix)

    tol=1e-3

    # get and remove relaxed muon position
    mu_idx = get_muon_index(rlxd_stc)
    if not mu_idx:
        raise ValueError(f"Muon label 'H' not found in relaxed structure.")
    mupos_rlx = rlxd_stc.frac_coords[mu_idx]
    rlxd_stc.remove_sites([mu_idx])

    # remove initial muon position from prist_stc
    mu_idx = get_muon_index(prist_stc)
    if mu_idx:
        prist_stc.remove_sites([mu_idx])

    assert len(rlxd_stc.frac_coords) == len(prist_stc.frac_coords)

    # Get valid symmetry operations
    ops = get_symmetry_operations(ipt_st, magmom=magmom, symprec=tol)

    # Find operation mapping original relaxed muon position -> target_mupos
    matching_op = next((op for op in ops if _pbc_close(op.operate(mupos_rlx ) % 1.0, n_mupos, tol)), None)
    if matching_op is None:
        return None

    # Calculate displacements and map them via the symmetry operation
    disp = rlxd_stc.frac_coords - prist_stc.frac_coords
    t_disp = matching_op.operate_multi(disp)

    # Apply displacements to pristine host
    new_stc = prist_stc.copy()
    for i in range(len(new_stc)):
        new_stc.translate_sites(i, t_disp[i], frac_coords=True, to_unit_cell=False)

    # Append muon at target location
    try:
        new_stc.append(
            species='H',
            coords=n_mupos,
            coords_are_cartesian=False,
            validate_proximity=True,
            properties={"kind_name": 'H'},
        )
    except ValueError:
        return None  # Collision detection safeguard

    return new_stc

# %%
def get_equiv_struct_wt_distortions(
    prist_stc, 
    rlxd_stc, 
    host_lattice, 
    magmom=None, 
):
    """
    Generates relaxed structures for ALL symmetry-equivalent sites in the unit cell.
    """
    muon_label="H"
    tol=1e-3

    # ops = get_symmetry_operations(host_lattice, magmom=magmom, symprec=tol)

    # get relaxed muon position
    mu_idx = get_muon_index(rlxd_stc)
    if not mu_idx:
        raise ValueError(f"Muon label 'H' not found in relaxed structure.")
    mupos_rlx = rlxd_stc.frac_coords[mu_idx]


    # # Build unique orbit of equivalent fractional positions
    # equiv_pos = []
    # for op in ops:
    #     pos = op.operate(mupos_rlx) % 1.0
    #     if not any(_pbc_close(pos, existing, tol) for existing in equiv_pos):
    #         equiv_pos.append(pos)

    # get equivalent muon pos
    equiv_pos = get_equivalent_sites(
        mupos_rlx,
        host_lattice.copy(),
        min_distance=0.5,
        symprec=tol,
        # energies=None,
        # e_tol=0.05,
    )

    # Generate structures for each site in orbit
    results = []
    for idx, pos in enumerate(equiv_pos):
        stc = transplant_distortion(
            prist_stc.copy(), rlxd_stc.copy(), pos, ipt_st=host_lattice.copy(), magmom=magmom,
        )
        if stc is not None:
            results.append({"idx": idx, "mupos": pos, "struct": stc})
            # results.append(stc)

    return results