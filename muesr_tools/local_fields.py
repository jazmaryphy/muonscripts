# %%
from __future__ import annotations
from typing import Optional, Sequence
import numpy.typing as npt

import numpy as np
# from ase import Atoms  # ase Atoms does not work with muESR, muESR Atoms instead
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.electronic_structure.core import Magmom

from muesr.core import Sample
from muesr.core.atoms import Atoms
from muesr.engines.clfc import find_largest_sphere, locfield

from muon_tools.utils import get_equivalent_sites

from muesr_tools.utils import (
    LocalFieldResults, 
    _extract_muon, 
    _validate_magmoms, 
    _validate_k_vector,
    _validate_sphere_radius, 
    _validate_muon_positions,
)

# %%
def pfields(
    structure: Structure,
    magmoms: Sequence[float] | npt.NDArray[np.floating],
    muon_pos: Sequence[float] | npt.NDArray[np.floating] | None = None,
    sphere_r: int = 100,
    k: Sequence[float] | npt.NDArray[np.floating] | None = None,
    cont_field: float = 0.0,
    include_equivalent_sites: bool = False,
    muon_label: str = "H",
) -> LocalFieldResults:
    """
    Calculate local magnetic fields at one or more muon stopping sites.

    The muon may be supplied either

    1. as an atom (default "H") inside ``structure``, or
    2. explicitly through ``muon_pos``.

    Exactly one of these options must be used.

    Parameters
    ----------
    structure
        Host crystal structure. May optionally contain one muon atom.
    magmoms
        Magnetic moments for the host atoms.
        Accepted shapes are: (N,) & (N,3)
        where N is the number of host atoms (excluding the muon).
    muon_pos
        Fractional coordinates of the muon when the supplied structure
        does not already contain it.
    sphere_r
        Supercell radius used for the dipolar field calculation.
    k
        Magnetic propagation vector.
    cont_field
        Contact field value.
    include_equivalent_sites
        If True, calculate the field at every symmetry-equivalent muon site.
    muon_label
        Species used to identify the implanted muon.
    Returns
    -------
    LocalFieldResults
        Calculated magnetic fields.
    """

    # Input validation
    host_lattice, muon_pos = _extract_muon(structure, muon_pos, muon_label)
    k = _validate_k_vector(k)
    sphere_r = _validate_sphere_radius(sphere_r)
    magmoms = _validate_magmoms(host_lattice.copy(), magmoms)

    p_st = host_lattice.copy() # a copy of structure

    # Assign magnetic moments
    for i, m in enumerate(magmoms):
        p_st[i].properties["magmom"] = Magmom(m)
    
    # for site, moment in zip(p_st, magmoms):
    #     site.properties["magmom"] = Magmom(moment)

    # Gen fourier comp. magnetic moments in complex form
    moments = p_st.site_properties["magmom"]
    fc_sup = np.zeros([len(moments), 3], dtype=complex)
    for i, m in enumerate(moments):
        # fc_sup[i] = m.get_moment_relative_to_crystal_axes(p_scst.lattice).astype(complex)
        fc_sup[i] = m.get_moment().astype(complex)

    # get the s_axis for transforming the contact field that is isotropic
    s_axis = Magmom.get_suggested_saxis(moments)

    # Build MUESR sample
    # to start dipolar calculations

    smp = Sample()

    # get structure from pymatgen-->ase_atoms-->Muesr_atoms
    ase_atom = AseAtomsAdaptor.get_atoms(host_lattice)
    # smp.cell = ase_atom  #raise TypeError('Cell is invalid.') for MnO.mcif
    atoms = Atoms(
        symbols=ase_atom.symbols,
        scaled_positions=ase_atom.get_scaled_positions(),
        cell=ase_atom.cell,
        pbc=True,
    )
    smp.cell = atoms

    # add muon site and its symmetry equivalent, if True
    smp._reset(muon=True)
    if include_equivalent_sites:
        muon_pos = get_equivalent_sites(muon_pos, host_lattice)
    else:
        muon_pos = np.atleast_2d(muon_pos)

    for pos in muon_pos:
        smp.add_muon(pos)

    smp.new_mm()
    smp.mm.k = k
    smp.mm.fc = fc_sup
    # smp.mm.fc_set(fc_sup, coord_system=2)
    # smp.current_mm_idx=0

    # containder, assumed cubic: [sphere_r] * 3 = [sphere_r, sphere_r, sphere_r]
    radius = find_largest_sphere(smp, [sphere_r] * 3)

    # Local field calculation
    results = locfield(smp, "s", [sphere_r] * 3, radius)

    nmu = len(smp.muons)

    B_dip = np.zeros((nmu, 3))
    B_lor = np.zeros((nmu, 3))
    B_con = np.zeros((nmu, 3))
    B_tot = np.zeros((nmu, 3))

    B_dip_lor = np.zeros((nmu, 3))

    B_dip_norm = np.zeros(nmu) 
    B_lor_norm = np.zeros(nmu) 
    B_con_norm = np.zeros(nmu) 
    B_tot_norm = np.zeros(nmu)

    B_dip_lor_norm = np.zeros(nmu) 

    for i, res in enumerate(results):
        B_dip[i] = res.D
        B_lor[i] = res.L
        B_con[i] = s_axis * cont_field
        B_tot[i] = B_dip[i] + B_lor[i] + (s_axis * cont_field)

        B_dip_lor[i] = B_dip[i]+B_lor[i]


        B_dip_norm[i] = np.linalg.norm(B_dip[i])
        B_lor_norm[i] = np.linalg.norm(B_lor[i])
        B_con_norm[i] = np.linalg.norm(B_con[i])
        B_tot_norm[i] = np.linalg.norm(B_tot[i])

        B_dip_lor_norm[i] = np.linalg.norm(B_dip_lor[i])

    return LocalFieldResults(
        total=B_tot,
        dipolar=B_dip,
        lorentz=B_lor,
        contact=B_con,
        dipolar_tot=B_dip_lor,
        
        total_norm=B_tot_norm,
        dipolar_norm=B_dip_norm,
        lorentz_norm=B_lor_norm,
        contact_norm=B_con_norm,
        dipolar_tot_norm=B_dip_lor_norm,

        s_axis=s_axis,
        muon_positions=np.asarray(smp.muons),

        dipolar_correction=np.zeros_like(B_dip),
        dipolar_correction_norm=np.zeros_like(B_dip_norm)
    )

# %%
def multisite_pfields(
    structure: Structure,
    magmoms: Sequence[float] | npt.NDArray[np.floating],
    muon_positions: Sequence[float] | npt.NDArray[np.floating],
    sphere_r: int = 100,
    k: Sequence[float] | npt.NDArray[np.floating] | None = None,
    cont_field: float = 0.0,
) -> LocalFieldResults:
    """
    Calculate local magnetic fields for multiple muon stopping sites.

    The muon may be supplied either

    1. as an atom (default "H") inside ``structure``, or
    2. explicitly through ``muon_pos``.

    Exactly one of these options must be used.

    Parameters
    ----------
    structure
        Host crystal structure without the muon.
    magmoms 
        Magnetic moments for the host atoms. 
        Accepted shapes are ``(N,)`` and ``(N, 3)``, where ``N`` 
        is the number of atoms in ``structure``.
    muon_positions
        Fractional coordinates of the muon stopping sites.
        Accepted shapes are ``(3,)`` for a single site or ``(N, 3)``
        for multiple sites.
    sphere_r
        Supercell radius used for the dipolar field calculation.
    k
        Magnetic propagation vector with three components.
        If ``None``, ``k = (0, 0, 0)``.
    cont_field
        Contact field magnitude. The field direction is determined
        from the suggested magnetic spin axis.

    Returns
    -------
    LocalFieldResults
        Calculated local-field contributions for all muon sites.
    """

    # Input validation
    host_lattice = structure.copy()
    k = _validate_k_vector(k)
    sphere_r = _validate_sphere_radius(sphere_r)
    magmoms = _validate_magmoms(host_lattice.copy(), magmoms)
    muon_positions = _validate_muon_positions(muon_positions)

    p_st = host_lattice.copy() # a copy of structure

    # Assign magnetic moments
    for i, m in enumerate(magmoms):
        p_st[i].properties["magmom"] = Magmom(m)
    
    # for site, moment in zip(p_st, magmoms):
    #     site.properties["magmom"] = Magmom(moment)

    # Gen fourier comp. magnetic moments in complex form
    moments = p_st.site_properties["magmom"]
    fc_sup = np.zeros([len(moments), 3], dtype=complex)
    for i, m in enumerate(moments):
        # fc_sup[i] = m.get_moment_relative_to_crystal_axes(p_scst.lattice).astype(complex)
        fc_sup[i] = m.get_moment().astype(complex)

    # get the s_axis for transforming the contact field that is isotropic
    s_axis = Magmom.get_suggested_saxis(moments)

    # Build MUESR sample
    # to start dipolar calculations

    smp = Sample()

    # get structure from pymatgen-->ase_atoms-->Muesr_atoms
    ase_atom = AseAtomsAdaptor.get_atoms(host_lattice)
    # smp.cell = ase_atom  #raise TypeError('Cell is invalid.') for MnO.mcif
    atoms = Atoms(
        symbols=ase_atom.symbols,
        scaled_positions=ase_atom.get_scaled_positions(),
        cell=ase_atom.cell,
        pbc=True,
    )
    smp.cell = atoms

    for mupos in muon_positions:
        smp.add_muon(mupos)

    smp.new_mm()
    smp.mm.k = k
    smp.mm.fc = fc_sup
    # smp.mm.fc_set(fc_sup, coord_system=2)
    # smp.current_mm_idx=0

    # containder, assumed cubic: [sphere_r] * 3 = [sphere_r, sphere_r, sphere_r]
    radius = find_largest_sphere(smp, [sphere_r] * 3)

    # Local field calculation
    results = locfield(smp, "s", [sphere_r] * 3, radius)

    nmu = len(smp.muons)

    B_dip = np.zeros((nmu, 3))
    B_lor = np.zeros((nmu, 3))
    # B_con = np.zeros((nmu, 3))

    for i, result in enumerate(results):
        B_dip[i] = result.D
        B_lor[i] = result.L

        # B_lor[i] = s_axis * cont_field

    # B_con[:] = s_axis * cont_field

    # Contact field is the same for all muon sites 
    B_con = np.broadcast_to(s_axis * cont_field, (nmu, 3)).copy()

    # Dipolar + Lorentz contribution
    B_dip_lor = B_dip + B_lor
    # Total Field
    B_tot = B_dip_lor.copy() + B_con.copy()

    # Field norms
    B_dip_norm = np.linalg.norm(B_dip, axis=1)
    B_lor_norm = np.linalg.norm(B_lor, axis=1)
    B_con_norm = np.linalg.norm(B_con, axis=1)

    B_dip_lor_norm = np.linalg.norm(B_dip_lor, axis=1)

    B_tot_norm = np.linalg.norm(B_tot, axis=1)

    return LocalFieldResults(
        total=B_tot,
        dipolar=B_dip,
        lorentz=B_lor,
        contact=B_con,
        dipolar_tot=B_dip_lor,
        
        total_norm=B_tot_norm,
        dipolar_norm=B_dip_norm,
        lorentz_norm=B_lor_norm,
        contact_norm=B_con_norm,
        dipolar_tot_norm=B_dip_lor_norm,

        s_axis=s_axis,
        muon_positions=np.asarray(smp.muons),

        dipolar_correction=np.zeros_like(B_dip),
        dipolar_correction_norm=np.zeros_like(B_dip_norm)
    )

# %%
def rfields(
    p_st: Structure,
    magmoms: Sequence[float] | npt.NDArray[np.floating],
    sc_mat: Sequence[Sequence[int]] | npt.NDArray[np.integer],
    r_supst: Structure,
    cont_field: float = 0.0,
    muon_label: str = "H",
) -> LocalFieldResults:
    """
    Compute the dipolar and total local magnetic field using MUESR
    code (10.7566/JPSCP.21.011052)..

    Parameters
    ----------
    p_st
        Pristine host structure.
    magmoms
        Magnetic moments for the host atoms.
    sc_mat
        Supercell transformation matrix.
    r_supst
        Relaxed supercell containing the implanted muon.
    cont_field
        Contact field value from DFT at the muon site in r_supst.

    Returns
    -------
    LocalFieldResults
        Calculated magnetic fields.
    """

    # set-up required parameters
    assert p_st.num_sites == len(magmoms)

    # Assign magnetic moments
    for i, m in enumerate(magmoms):
        p_st[i].properties["magmom"] = Magmom(m)

    p_scst = p_st.copy()
    p_scst.make_supercell(sc_mat)

    # get and remove muon site
    muon_pos, muon_label = None, 'H'
    r_supst, muon_pos = _extract_muon(r_supst.copy(), muon_pos, muon_label)

    # center the supercell around the muon for no-PBC cases. muon at (0.5,0.5,0.5)
    r_supst.translate_sites(
        range(r_supst.num_sites), 0.5-muon_pos, frac_coords=True, to_unit_cell=True
    )
    p_scst.translate_sites(
        range(p_scst.num_sites), 0.5-muon_pos, frac_coords=True, to_unit_cell=True
    )

    # Gen supercell fourier comp. mag moments in complex form
    moments = p_scst.site_properties["magmom"]
    fc_sup = np.zeros([len(moments), 3], dtype=complex)
    for i, m in enumerate(moments):
        # fc_sup[i] = m.get_moment_relative_to_crystal_axes(p_scst.lattice).astype(complex)
        fc_sup[i] = m.get_moment().astype(complex)

    # get the s_axis for transforming the contact field that is isotropic
    s_axis = Magmom.get_suggested_saxis(moments)

    # start dipolar calculations

    smp = Sample()

    # get structure from pymatgen-->ase_atoms-->Muesr_atoms
    ase_atom_p = AseAtomsAdaptor.get_atoms(p_scst)
    # smp.cell = ase_atom  #raise TypeError('Cell is invalid.') for MnO.mcif
    atoms_p = Atoms(
        symbols=ase_atom_p.symbols,
        scaled_positions=ase_atom_p.get_scaled_positions(),
        cell=ase_atom_p.cell,
        pbc=True,
    )
    smp.cell = atoms_p

    smp.new_mm()
    smp.mm.k = np.array([0.0, 0.0, 0.0])
    smp.mm.fc = fc_sup
    # smp.mm.fc_set(fc_sup, coord_system=2)
    # smp.current_mm_idx=0

    # smp.add_muon(muon_pos+0.5-muon_pos)
    smp.add_muon([0.5, 0.5, 0.5])

    # find the largest radius 
    radius = find_largest_sphere(smp, [50, 50, 50])

    # compute B in full(50x50x50 supercell) 'f' in the pristine structre
    r_f_ps = locfield(smp, "s", [50, 50, 50], radius)

    # compute B only within the supercell 's'  using the pristine structre,
    # To include muon induced relaxation effects
    radius_n = np.min(r_supst.lattice.abc)
    r_s_ps = locfield(smp, "s", [50, 50, 50], radius_n)

    # change the cell to the relaxed
    # smp.cell = AseAtomsAdaptor.get_atoms(r_supst)
    ase_atom_r = AseAtomsAdaptor.get_atoms(r_supst)
    atoms_r = Atoms(
        symbols=ase_atom_r.symbols,
        scaled_positions=ase_atom_r.get_scaled_positions(),
        cell=ase_atom_r.cell,
        pbc=True,
    )
    smp.cell = atoms_r

    # compute B only within the supercell 's' the using the relaxed structre
    r_s_rlx = locfield(smp, "s", [50, 50, 50], radius_n)

    # field contibutions
    dip_relax = r_s_rlx[0].D - r_s_ps[0].D  # correction/change to dip due to perturbations
    B_dip = r_f_ps[0].D + dip_relax         # total dipolar
    B_lor = r_f_ps[0].L
    B_con = s_axis * cont_field
    B_tot = B_dip + B_lor + B_con

    B_dip_lor = B_dip + B_lor

    B_dip_norm = np.linalg.norm(B_dip)
    B_lor_norm = np.linalg.norm(B_lor)
    B_con_norm = np.linalg.norm(B_con)
    B_tot_norm = np.linalg.norm(B_tot)

    B_dip_lor_norm = np.linalg.norm(B_dip_lor)

    dip_relax_norm = np.linalg.norm(dip_relax)

    return LocalFieldResults(
        total=B_tot[np.newaxis],
        dipolar=B_dip[np.newaxis],
        lorentz=B_lor[np.newaxis],
        contact=B_con[np.newaxis],
        dipolar_tot=B_dip_lor[np.newaxis],

        total_norm=np.array([B_tot_norm]),
        dipolar_norm=np.array([B_dip_norm]),
        lorentz_norm=np.array([B_lor_norm]),
        contact_norm=np.array([B_con_norm]),
        dipolar_tot_norm=np.array([B_dip_lor_norm]),

        s_axis=s_axis,
        muon_positions=np.atleast_2d(muon_pos),

        dipolar_correction=dip_relax[np.newaxis],
        dipolar_correction_norm=np.array([dip_relax_norm])
    )