# %%
import numpy as np
from constants import constants
from efg.utils import get_omegaQ_mu, quadrupole_frequencies

# %%
def _get_site_labels(atoms):
    """Per-atom label distinguishing inequivalent sites of the same element.
 
    Uses ASE's `spacegroup_kinds` array (populated automatically when
    reading a CIF with symmetry via `ase.io.read`) to detect when an
    element occupies more than one crystallographically distinct site.
 
    - Elements with a single site keep their plain symbol (e.g. 'Ba', 'Y').
    - Elements with multiple distinct sites get a numbered suffix, in
      order of first appearance in `atoms` (e.g. 'Cu1', 'Cu2', 'O1', 'O2').
 
    If `atoms` has no `spacegroup_kinds` array (e.g. it wasn't read from a
    CIF, or symmetry info wasn't kept), every atom just gets its plain
    element symbol -- i.e. site-level distinction is unavailable and
    charge lookups fall back to per-element behaviour automatically.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    kinds = atoms.arrays.get("spacegroup_kinds")
 
    if kinds is None:
        return list(symbols)
 
    labels = np.empty(len(symbols), dtype=object)
 
    for sym in set(symbols):
        elem_mask = symbols == sym
        elem_kinds = kinds[elem_mask]
 
        # Unique kinds for this element, in order of first appearance
        seen = []
        for k in elem_kinds:
            if k not in seen:
                seen.append(k)
        kind_to_num = {k: i + 1 for i, k in enumerate(seen)}
 
        multi_site = len(seen) > 1
        for idx, k in zip(np.where(elem_mask)[0], elem_kinds):
            labels[idx] = f"{sym}{kind_to_num[k]}" if multi_site else sym
 
    return list(labels)

 
def _check_charges_cover_atoms(atoms, charges, strict=False):
    """Check that every atom in `atoms` resolves to a charge in `charges`.
 
    Mirrors the same lookup order used in `_replicate_lattice`
    (site label first, e.g. 'O2', then plain element symbol, e.g. 'O')
    and reports any atoms that would silently resolve to `None` (and
    therefore be DROPPED from the replicated lattice rather than raising
    an error).
 
    Parameters
    ----------
    strict : bool, default=False
        If True, raise a ValueError listing the unresolved site labels.
        If False, just return the list of unresolved labels (empty list
        means everything is covered) so the caller can decide what to do
        (e.g. print a warning).
 
    Returns
    -------
    list of str
        Unique site labels (e.g. ['O2']) present in `atoms` that do NOT
        resolve to a charge, either directly or via the plain-element
        fallback.
    """
    site_labels = _get_site_labels(atoms)
    symbols = atoms.get_chemical_symbols()
 
    unresolved = sorted(
        {
            label
            for label, sym in zip(site_labels, symbols)
            if charges.get(label, charges.get(sym)) is None
        }
    )
 
    if unresolved and strict:
        raise ValueError(
            f"charges dict does not cover these site(s), atoms would be "
            f"silently dropped from the replicated lattice: {unresolved}. "
            f"Add an entry for each (either the site label, e.g. 'O2', "
            f"or the plain element symbol, e.g. 'O')."
        )
 
    return unresolved

# %%
def _replicate_lattice(atoms, charges, sphere_radius, exclude_indices=()):
    """Generate (position[m], charge[e]) for every periodic image of
    every atom in `atoms` within `sphere_radius` of the origin's unit cell
    (replicated enough times to cover `sphere_radius` in every direction, from
    the TRUE cell matrix -- works for any crystal system, not just
    orthorhombic).
 
    `charges` supports two levels of keys, so you don't have to specify a
    site-resolved charge for every element:
 
    - Per-element key, e.g. {'O': -2, 'Ba': 2, 'Y': 3} -- applies to every
      atom of that element regardless of which crystallographic site it
      sits on.
    - Per-site key, e.g. {'Cu1': 1, 'Cu2': 2} -- applies only to atoms on
      that specific site (as identified by `get_site_labels`), letting you
      give chemically/crystallographically distinct atoms of the SAME
      element different charges (e.g. Cu1+ at the 1a site vs a different
      formal charge at the 2g site in YBa2Cu3O6-type structures).
 
    You can mix both in the same dict, e.g. {'Cu1': 1, 'Cu2': 2, 'O': -2,
    'Ba': 2, 'Y': 3}: per-site keys are checked first, and any element
    without a per-site key present in `charges` falls back to its plain
    element-symbol entry.
    """
    # check charges
    missing = _check_charges_cover_atoms(atoms, charges=charges, strict=False)
    if missing :
        raise ValueError(
            f"ERROR: <charges={charges}> does not cover whole atoms/structure, "
            f"charges of: {missing} missing!"
        )

    cell = np.array(atoms.get_cell())      # (3,3), Angstrom, ASE convention
    positions = atoms.get_positions()      # (N,3), Angstrom
    symbols = atoms.get_chemical_symbols()
    site_labels = _get_site_labels(atoms)   # e.g. 'Cu1'/'Cu2' where sites differ

    # how many unit cells to replicate in each direction to cover `sphere_radius`
    cell_norms = np.linalg.norm(cell, axis=1)
    n_reps = np.ceil(sphere_radius / constants.ANGSTROM / cell_norms).astype(int) + 1

    pts, qs = [], []
    for na in range(-n_reps[0], n_reps[0] + 1):
        for nb in range(-n_reps[1], n_reps[1] + 1):
            for nc in range(-n_reps[2], n_reps[2] + 1):
                shift = na * cell[0] + nb * cell[1] + nc * cell[2]
                # for idx, (p, s) in enumerate(zip(positions, symbols)):                   # <-- OLD
                for idx, (p, s, label) in enumerate(zip(positions, symbols, site_labels)):  # <-- NEW 
                    if idx in exclude_indices:
                        continue

                    # q = charges.get(s)                                          # <-- OLD
                    
                    # per-site charge takes priority; fall back to per-element
                    q = charges.get(label, charges.get(s))                        # <-- NEW
                    if q is None or q == 0:
                        continue

                    pts.append((p + shift) * constants.ANGSTROM)   # -> metres
                    qs.append(q * constants.ELEMENTARY_CHARGE)     # -> Coulomb
    return np.array(pts), np.array(qs)


def _single_charge_EFG(charge_position, site_position, charge):
    d = site_position - charge_position
    r2 = np.dot(d, d)
    r5 = r2 ** 2.5
    V = charge / (4 * np.pi * constants.EPSILON0) * (3 * np.outer(d, d) - r2 * np.eye(3)) / r5
    return V

# %%
def muon_point_charge_EFG(mu_position, site_position):
    """The MUON's own contribution to the EFG at `site_position`, kept
    separate from point_charge_EFG() (the lattice/environment part) to
    match undi.py's split into 'EFGTensor' (lattice) vs 'OmegaQmu'
    (muon) -- see undi.py's muon_induced_efg()."""
    mu_position = np.asarray(mu_position) * constants.ANGSTROM
    site_position = np.asarray(site_position) * constants.ANGSTROM
    return _single_charge_EFG(mu_position, site_position, +1 * constants.ELEMENTARY_CHARGE)


def point_charge_EFG(
        atoms, 
        site_position, 
        charges, 
        sphere_radius=30,
        exclude_indices=(), 
        extra_charges=None, 
        gamma_sternheimer=0.0, 
        verbose=True
    ):
    """Point-charge EFG tensor [V/m^2] at `site_position`, to any ASE structure 
    and any sphere_radius.

    Parameters
    ----------
    atoms : ase.Atoms
        The (possibly already muon-relaxed) crystal structure. Should
        NOT include the muon itself as one of its atoms -- pass that
        separately via `extra_charges` (or just compute its
        contribution with muon_point_charge_EFG() below and add the two
        tensors, which keeps the two physically distinct contributions
        -- lattice vs. muon -- separately labeled, exactly the
        convention undi.py expects: 'EFGTensor' for the lattice part,
        'OmegaQmu' computed independently for the muon part).
    site_position : (3,) array [Angstrom]
        Cartesian position of the nucleus to evaluate the EFG at (does
        NOT need to be one of the atoms in `atoms` -- e.g. a candidate
        muon site).
    charges : dict {chemical_symbol: formal_charge_in_units_of_e}
        e.g. {'Na': +1, 'F': -1} for NaF, {'Ca': +2, 'F': -1} for CaF2.
        Species not listed are treated as having zero charge (skipped).
    sphere_radius : float [Angstrom]
        Real-space sphere_radius radius for the lattice sum. Use check_convergence() 
        below to verify it's large enough for your structure before trusting 
        the result. Default 30 Angstrom.
    exclude_indices : iterable of int
        Indices (into `atoms`) to exclude from the bulk sum -- e.g. the
        1-2 ions closest to a candidate muon site if you plan to add
        their RELAXED positions back in via `extra_charges`.
    extra_charges : list of (position[Angstrom], charge[e]) or None
        Any additional point charges not in `atoms` -- e.g. relaxed
        neighbour positions, or the muon itself if you want it folded
        into a single combined tensor rather than kept separate.
    gamma_sternheimer : float, optional
        Sternheimer antishielding factor, V_total = V_lattice*(1+gamma).
        Default 0.0 (no antishielding correction, i.e. the bare
        point-charge lattice sum).
    verbose : bool
        Print the number of charges summed over. Default True.

    Returns
    -------
    (3,3) ndarray, EFG tensor [V/m^2], symmetric and (up to numerical
    noise) traceless.
    """
    sphere_radius_m = sphere_radius*constants.ANGSTROM      # -> metres
    site = np.asarray(site_position) * constants.ANGSTROM   # -> metres
    pts, qs = _replicate_lattice(atoms, charges, sphere_radius_m, exclude_indices=exclude_indices)

    if extra_charges:
        extra_pts = np.array([p for p, q in extra_charges]) * constants.ANGSTROM
        extra_qs = np.array([q for p, q in extra_charges]) * constants.ELEMENTARY_CHARGE
        pts = np.vstack([pts, extra_pts]) if len(pts) else extra_pts
        qs = np.concatenate([qs, extra_qs]) if len(qs) else extra_qs

    d = pts - site
    r2 = np.sum(d * d, axis=1)
    keep = (r2 > 1e-24) & (r2 < sphere_radius_m ** 2)      # drop self-coincident + beyond sphere_radius_m
    d, r2, qs_k = d[keep], r2[keep], qs[keep]
    r5 = r2 ** 2.5

    n_charges = len(qs_k)
    if verbose:
        print(
            f"---- point charge EFG: lattice sum for {n_charges} charges "
            f"within sphere radius {sphere_radius:.3f} Å ----"
        )
        # print(f"---- point charge EFG: lattice sum for {n_charges:e} charges within sphere {sphere_radius:.3f} Å ----")

    V = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            delta_ab = 1.0 if a == b else 0.0
            V[a, b] = np.sum(qs_k * (3 * d[:, a] * d[:, b] - r2 * delta_ab) / r5)
    V *= 1.0 / (4 * np.pi * constants.EPSILON0)
    V *= (1 - gamma_sternheimer)   # <-- new: old --> (1+gamma) convention
    return V


def diagonalize_EFG(tensor, quadrupole_moment):
    """Diagonalize the EFG tensor into its Principal Axis System (PAS).

    Convention (standard NQR/NMR, |Vzz| >= |Vyy| >= |Vxx|, Abragam 1961):
        eta = |Vxx - Vyy| / |Vzz|          (asymmetry parameter, 0<=eta<=1)
        chi = |Vzz * e * Q / h|            (quadrupole coupling constant, Hz)

    Parameters
    ----------
    tensor : (3,3) array [V/m^2]
    quadrupole_moment : float [m^2]

    Returns
    -------
    V_aa : (3,) ndarray
        Principal EFG components [V m^-2] ordered as
            [Vxx, Vyy, Vzz]
        with
            |Vzz| >= |Vyy| >= |Vxx|.
    P : (3, 3) ndarray
        Matrix whose columns are the normalized principal-axis vectors
        corresponding to [Vxx, Vyy, Vzz]. The third column therefore
        gives the principal Vzz axis.
    chi_q : float
        Quadrupole coupling constant [MHz].
    eta : float
        EFG asymmetry parameter,
            eta = |Vxx - Vyy| / |Vzz|,
        satisfying 0 <= eta <= 1 for a properly ordered EFG tensor.
    """
    
    # evals, evecs = np.linalg.eig(tensor)    
    evals, evecs = np.linalg.eigh(tensor)                     # eigh: correct for real symmetric
    order = np.argsort(-np.abs(evals))                        # descending by |value|
    Vzz, Vyy, Vxx = evals[order]
    V_aa = np.array([Vxx, Vyy, Vzz])   

    # D = np.diag(V_aa)                              

    P = evecs[:, order][:, ::-1]                              # Principal axes vector

    scale = np.abs(evals).max()                               # scale parameter to set tolerance
    # eta = float(np.abs(Vxx - Vyy) / np.abs(Vzz)) if Vzz != 0 else 0.0               # Asymmetry parameter of the EFG tensor
    eta = float(np.abs(Vxx - Vyy) / np.abs(Vzz)) if abs(Vzz) > 1e-6*scale else 0.0  # Asymmetry parameter of the EFG tensor

    chi_q = float(np.abs(Vzz * constants.ELEMENTARY_CHARGE * quadrupole_moment / constants.PLANCK_H) )   # Quadrupolar constant, in Hz or s^-1
    chi_q *= 1e-6   # Quadrupolar constant, in MHz
    # return D, P, eta, chi_q, Vzz
    return V_aa, P, chi_q, eta                                                                                    

# %%
def compute_efg(
    atoms, 
    probe_position, 
    atomic_charges, 
    sphere_radius,
    gamma_sternheimer=0.0, 
    exclude_indices=(), 
    extra_charges=None,
    coords_are_cartesian=True,
    nuclear_spin=None,
    quadrupole_moment=None,
    verbose=True
):
    """
    coords_are_cartesian : bool
        Set to True if you are providing coordinates in Cartesian coordinates. 
        Defaults to True.
    """
    # check type of coordinates
    if not coords_are_cartesian:
        probe_position = np.dot(probe_position, atoms.get_cell())
        if extra_charges:
            extra_charges = [(np.dot(p, atoms.get_cell()).tolist(), q) for p, q in extra_charges]

    tensor =  point_charge_EFG(
        atoms,
        probe_position,
        charges=atomic_charges,
        sphere_radius=sphere_radius,
        extra_charges=extra_charges,
        exclude_indices=exclude_indices,
        gamma_sternheimer=gamma_sternheimer,
        verbose=verbose,
    )

    V_aa = principal_axes = chi = eta = None
    Vxx = Vyy = Vzz = None
    nu_z = nu_q = None

    if quadrupole_moment is not None:
        V_aa, principal_axes, chi, eta = diagonalize_EFG(
            tensor, quadrupole_moment=quadrupole_moment)

        Vxx, Vyy, Vzz = V_aa

        if nuclear_spin is not None:
            props = quadrupole_frequencies(I=nuclear_spin, Q=quadrupole_moment, Vzz=Vzz, eta=eta)
            nu_z, nu_q = props.get('nu_z_MHz'), props.get('nu_Q_MHz')

    results = {
        "Vxx": Vxx,
        "Vyy": Vyy,
        "Vzz": Vzz,
        "eta": eta,
        "V_aa": V_aa,
        "nu_z_MHz": nu_z,
        "nu_Q_MHz": nu_q,
        "chi_Q_MHz": chi,
        "EFG_tensor": tensor,
        "principal_axes": principal_axes,
        "probe_index": None,    # TODO later
        "probe_symbol": None,   # TODO later
        "probe_position": np.dot(probe_position, np.linalg.inv(atoms.get_cell()))%1.0
    }

    if verbose:
        _pretty_print_efg(results=results)
    return results


def _pretty_print_efg(results):
    """
    Pretty-print EFG analysis results.

    Any key with value None is silently skipped.
    """
    print("\n" + "=" * 70)
    #
    probe_pos = results.get("probe_position")
    if probe_pos is not None:
        label = f"atom {results['probe_index']} ({results['probe_symbol']})" \
            if results.get("probe_index") is not None and results.get("probe_symbol") is not None \
            else "probe site"
        print(f"EFG analysis for {label} at frac coord. "
              f"({probe_pos[0]:.4f}, {probe_pos[1]:.4f}, {probe_pos[2]:.4f})")
    #
    print("=" * 70)
    #
    scalar_fields = [
        ("Vzz",         "V/m^2"),
        ("Vyy",         "V/m^2"),
        ("Vxx",         "V/m^2"),
        ("eta",         "(unitless)"),
        ("chi_Q_MHz",   "MHz"),
        ("nu_z_MHz",    "MHz"),
        ("nu_Q_MHz",    "MHz"),
    ]

    for key, unit in scalar_fields:
        val = results.get(key)
        if val is None:
            continue

        if key == "eta":
            print(f"{key:<12} = {val: .8f} {unit}")
        elif key in ("chi_Q_MHz", "nu_z_MHz", "nu_Q_MHz"):
            print(f"{key:<12} = {val: .8f} {unit}")
        else:
            print(f"{key:<12} = {val: .8e} {unit}")

    V = results.get("EFG_tensor")

    if V is not None:
        print(f"\nEFG tensor V_ab (V/m^2) =")
        print("-" * 70)
        for row in V:
            print(" [ " + ", ".join(f"{x: .8e}" for x in row) + " ]")
        print("-" * 70)

        print(f"Trace(V_ab) = {np.trace(V): .5e}")
        print(f"Symmetric   = {np.allclose(V, V.T)}")

    P = results.get("principal_axes")

    if P is not None:
        print(f"\nprincipal axes (unitless) = ")
        print("-" * 70)
        for row in P:
            print(" [ " + ", ".join(f"{x: .8e}" for x in row) + " ]")
        print("-" * 70)
    print("=" * 70)

# %%
def build_point_charge_efg_neighbors(
    atoms,
    neighbors,
    muon_position,
    atomic_charges,
    include_nuclear_efg=True,
    include_muon_induced_efg=False,
    remove_efg_noise=False,
    efg_noise_threshold=1e-8,
    efg_factor=1.0,
    sphere_radius=50.0,
    gamma_sternheimer=0.0,
    exclude_indices=(),
    extra_charges=None,
    efg_verbose=False,
    include_muon=True,
):
    """
    Build a quadrupolar neighbour cluster with point-charge EFG tensors.

    Selects nuclei with non-zero quadrupolar interaction from an UNDI
    cluster and optionally computes their electric field gradient (EFG)
    tensors using a point-charge lattice model. The resulting list is
    compatible with UNDI input format.

    Parameters
    ----------
    atoms : ase.Atoms
        Crystal structure used for the point-charge EFG calculation.
    neighbors : iterable of dict
        UNDI atom dictionaries containing nuclear information.
        Each atom dictionary must contain at least:
        ``Position``, ``Label``, ``Spin`` and
        ``ElectricQuadrupoleMoment``.
    muon_position : array-like, shape (3,)
        Fractional coordinates of the muon in the unit cell.
    atomic_charges : dict
        Mapping between atomic species and effective charges used in
        the point-charge EFG calculation.
        Example: ``{"V": +5, "O": -2}``
    include_nuclear_efg : bool, default=True
        If True, compute the EFG tensor generated by the surrounding
        nuclear charge distribution. If False, no EFG tensor is added
        to the atom dictionaries.
    include_muon_induced_efg : bool, default=False
        If True, compute the muon-induced quadrupolar interaction
        contribution and store it as ``OmegaQmu``.
    remove_efg_noise : bool, default=True
        If True, remove numerical noise from the computed EFG tensor by
        setting very small components relative to the largest tensor
        element to zero.
    efg_noise_threshold : float, default=1e-8
        Relative threshold used for EFG noise removal. Components
        satisfying
            abs(EFG_ij) < efg_noise_threshold * max(abs(EFG))
        are set to zero.
    efg_factor : float, optional
        Scale factor applied to every supplied EFG tensor.
    sphere_radius : float, default=50.0
        Radius of the spherical region (in Angstrom) used for the
        point-charge lattice summation.
    gamma_sternheimer : float, default=0.0
        Sternheimer antishielding factor applied to the lattice EFG tensor.
    exclude_indices : iterable of int, default=()
        Indices of atoms in ``atoms`` to exclude from the point-charge
        lattice summation. 
    extra_charges : sequence, optional
        Additional point charges to include in the EFG calculation. The
        format is the same as accepted by ``compute_efg()`` or ``point_charge_EFG()`.
        By default, no extra point charges are included.
    efg_verbose : bool, default=False
        Print additional information during EFG calculations.
    include_muon : bool, default=True
        If True, ensure a muon entry is present in the output.
        If False, remove any muon entry from the output.

    Returns
    -------
    list of dict
        UNDI-compatible quadrupolar neighbour cluster. The muon is
        inserted as the first entry.

        Each nuclear entry contains:
        ``Position``
            Position relative to the muon (meters).
        ``Label``
            Nuclear isotope label.
        ``Spin``
            Nuclear spin.
        ``ElectricQuadrupoleMoment``
            Nuclear quadrupole moment (m^2).
        ``EFGTensor``
            Electric field gradient tensor (V/m^2), only present when
            ``include_nuclear_efg=True``.
        ``OmegaQmu``
            Muon-induced quadrupolar coupling, only present when
            ``include_muon_induced_efg=True``.

    Notes
    -----
    The input ``neighbors`` is not modified. A copy of each atom dictionary
    is created before adding computed quantities.

    Only nuclei with spin I > 1/2 contribute to the quadrupolar cluster.
    """
    quadrupolar_cluster = []
    muon = None

    cell = atoms.get_cell()
    inverse_cell = np.linalg.inv(cell)

    for atom in neighbors:

        # Avoid modifying input dictionaries
        atom = atom.copy()

        # Skip muon here and handle later
        if atom.get("Label") == "mu":
            muon = atom
            continue

        spin = atom.get("Spin", 0.0)
        is_quadrupolar = spin > 0.5001

        # # Only quadrupolar nuclei: I > 1/2
        # if spin <= 0.5001:
        #     continue

        position = atom["Position"]

        # Nuclear EFG from point-charge lattice, AND,
        # ONLY quadrupolar nuclei: I > 1/2
        if include_nuclear_efg and is_quadrupolar:

            quadrupole_moment = atom["ElectricQuadrupoleMoment"]

            # Position of nucleus in fractional coordinates
            # relative to the original unit cell
            nuclear_cartesian = (
                np.dot(muon_position, cell) + position / constants.ANGSTROM
            )

            probe_position = np.dot(nuclear_cartesian, inverse_cell)%1

            result = compute_efg(
                atoms,
                probe_position=probe_position,
                atomic_charges=atomic_charges,
                sphere_radius=sphere_radius,
                gamma_sternheimer=gamma_sternheimer,
                exclude_indices=exclude_indices,
                extra_charges=extra_charges,
                coords_are_cartesian=False,
                nuclear_spin=spin,
                quadrupole_moment=quadrupole_moment,
                verbose=efg_verbose,
            )

            tensor = np.asarray(result["EFG_tensor"], dtype=float)

            # Remove numerical noise
            if remove_efg_noise:
                max_element = np.max(np.abs(tensor))
                if max_element > 0:
                    tensor[np.abs(tensor) < efg_noise_threshold * max_element] = 0.0

            tensor *= efg_factor 
            atom["EFGTensor"] = tensor

        # Muon induced quadrupolar interaction
        # Not sure about its accuracy
        if include_muon_induced_efg and is_quadrupolar:
            distance = np.linalg.norm(position)
            atom["OmegaQmu"] = get_omegaQ_mu(I=spin, Q=quadrupole_moment, r=distance)

        quadrupolar_cluster.append(atom)

    if include_muon:
        if muon is None:
            muon = {
                "Position": np.zeros(3),
                "Label": "mu",
                "Spin": 0.5,
                "Gamma": constants.MUON_GYROMAGNETIC_RATIO,
            }
        else:
            muon = muon.copy()
            muon["Position"] = np.zeros(3)

        quadrupolar_cluster.insert(0, muon)

    return quadrupolar_cluster

# %%
def _check_convergence(atoms, site_position, charges, quadrupole_moment=1.0e-28, cutoffs_ang=(10, 15, 20, 25, 30, 40)):
    """Print V_zz (the dominant, largest-|.| eigenvalue) as a function
    of cutoff radius, so you can see for yourself where it stops
    changing, instead of trusting an arbitrary fixed shell count."""
    print(f"{'cutoff (Å)':>14} {'Vzz (V/m^2)':>16}")
    prev = None
    for c_ang in cutoffs_ang:
        V = point_charge_EFG(atoms, site_position, charges, cutoff=c_ang)
        V_aa, _, _, _ = diagonalize_EFG(V, quadrupole_moment=quadrupole_moment)
        _, _, Vzz = V_aa
        rel_change = "" if prev is None else f"  (delta={100*abs(Vzz-prev)/abs(Vzz):.2f}%)"
        print(f"{c_ang:>14.1f} {Vzz:>16.4e}{rel_change}")
        prev = Vzz

        
def sphere_radius_convergence(
        atoms, 
        site_position, 
        charges, 
        exclude_indices=(), 
        extra_charges=None, 
        quadrupole_moment=1.0e-28,
        sphere_radius_list=None, 
        gamma_sternheimer=0.0,
        conv_thr=1e-3, 
        sphere_radius_step=10.0, 
        sphere_radius_max=100.0,
        num_conv_streak=3, 
        ax=None
    ):
    """Check (and optionally plot) convergence of the point-charge EFG
    real-space sum with sphere_radius.

    IMPORTANT (learned from a real, non-monotonic convergence curve):
    this sum can oscillate shell-to-shell rather than decrease smoothly
    -- a single low-error point does NOT mean you've converged; the
    curve can (and does, in practice) rise again at the next radius.
    So the stopping rule requires `num_conv_streak` CONSECUTIVE pairwise
    comparisons to all be below `conv_thr`, not just the most recent
    one -- a single lucky dip won't trigger early stopping anymore.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        If given, plot log10(relative error) vs sphere_radius onto this
        Axes. None (default): no plot, just the printed table + arrays.
    num_conv_streak : int
        Number of consecutive pairwise comparisons that must all satisfy
        conv_thr before stopping the auto-extension. Default 3.
    """
    if sphere_radius_list is None:
        sphere_radius_list = [10, 15, 20, 25, 30, 40]
    sphere_radius_list = list(sphere_radius_list)

    def _vzz(r):
        V = point_charge_EFG(
                atoms, 
                site_position, 
                charges, 
                sphere_radius=r,
                exclude_indices=exclude_indices,
                extra_charges=extra_charges,
                gamma_sternheimer=gamma_sternheimer
            )
        V_aa, _, _, _ = diagonalize_EFG(V, quadrupole_moment=quadrupole_moment)
        _, _, Vzz = V_aa
        return Vzz

    Vzz_values = [_vzz(r) for r in sphere_radius_list]

    def _is_converged():
        if len(Vzz_values) < num_conv_streak + 1:
            return False
        recent = Vzz_values[-(num_conv_streak+1):]
        diffs = [abs(recent[i+1]-recent[i])/max(abs(recent[i+1]), 1e-300) for i in range(num_conv_streak)]
        return all(d < conv_thr for d in diffs)

    while sphere_radius_list[-1] < sphere_radius_max and not _is_converged():
        sphere_radius_list.append(sphere_radius_list[-1] + sphere_radius_step)
        Vzz_values.append(_vzz(sphere_radius_list[-1]))

    Vzz_values = np.array(Vzz_values)
    best = Vzz_values[-1]
    rel_error = np.abs(Vzz_values - best) / max(abs(best), 1e-300)

    print(f"{'radius (Å)':>12} {'Vzz (V/m^2)':>16} {'rel. error vs best':>18}")
    for r, v, e in zip(sphere_radius_list, Vzz_values, rel_error):
        print(f"{r:>12.1f} {v:>16.4e} {e:>18.2e}")
    if not _is_converged() and sphere_radius_list[-1] >= sphere_radius_max:
        print(f"  WARNING: reached sphere_radius_max={sphere_radius_max} without "
              f"{num_conv_streak} consecutive sustained-converged points --.")
        print(f"  The sum may not actually be converged. Consider raising "
              f"sphere_radius_max or plotting to inspect visually.")

    if ax is not None:
        mask = rel_error > 0
        ax.semilogy(np.array(sphere_radius_list)[mask], rel_error[mask], 'o-')
        ax.set_xlabel('sphere radius (Angstrom)')
        ax.set_ylabel('relative error vs. largest-radius estimate')
        ax.set_title('EFG (Vzz): real-space sum convergence')
        ax.grid(True, which='both', alpha=0.3)

    return sphere_radius_list, Vzz_values, rel_error

# %%
def atom_dict_EFG(
        atoms, 
        site_position, 
        label, 
        spin, 
        gamma, 
        quadrupole_moment,
        charges, 
        sphere_radius=30, 
        muon_position=None, 
        exclude_indices=(), 
        extra_charges=None, 
        gamma_sternheimer=0.0, 
        verbose=False
    ):
    """One-call convenience: compute the lattice EFG (and, if
    mu_position is given, the muon-induced contribution too, kept in
    the SEPARATE 'OmegaQmu' key -- never merged into 'EFGTensor', to
    avoid double-counting, per gen_efg_structure.py's docstring
    earlier in this thread) and package everything as an undi.py atom
    dict, ready to drop into a shell list for shell_scaling.py /
    MuonNuclearInteraction.
    atoms : ase.Atoms
        The (possibly already muon-relaxed) crystal structure. Should
        NOT include the muon itself as one of its atoms -- pass that
        separately via `extra_charges` (or just compute its
        contribution with muon_point_charge_EFG() below and add the two
        tensors, which keeps the two physically distinct contributions
        -- lattice vs. muon -- separately labeled, exactly the
        convention undi.py expects: 'EFGTensor' for the lattice part,
        'OmegaQmu' computed independently for the muon part).
    site_position : (3,) array [Angstrom]
        Cartesian position of the nucleus to evaluate the EFG at (does
        NOT need to be one of the atoms in `atoms` -- e.g. a candidate
        muon site).
    label : str
        Label for the atom/site, e.g. 'F1' or 'Na2'.
    spin : float
        Nuclear spin quantum number I of the nucleus at `site_position`.
    gamma : float [rad/s/T]
        Gyromagnetic ratio of the nucleus at `site_position`.
    quadrupole_moment : float [m^2]
        Nuclear electric quadrupole moment Q of the nucleus at `site_position`.
    charges : dict {chemical_symbol: formal_charge_in_units_of_e}
        e.g. {'Na': +1, 'F': -1} for NaF, {'Ca': +2, 'F': -1} for CaF2.
        Species not listed are treated as having zero charge (skipped).
    sphere_radius : float [Angstrom]
        Real-space cutoff radius for the lattice sum. Unlike POCSI's
        fixed shell count `s`, this is a physical distance -- use
        check_convergence() below to verify it's large enough for your
        structure before trusting the result. Default 30 Angstrom.
    muon_position : (3,) array [cartesian] or None
        If given, the muon's position (in cartesian coordinates) to compute its
        contribution to the EFG at `site_position` and store it in the
        'OmegaQmu' key of the returned dict, separate from the lattice
        EFG tensor in 'EFGTensor' (to avoid double-counting, per
        gen_efg_structure.py's docstring earlier in this thread).
    exclude_indices : iterable of int
        Indices (into `atoms`) to exclude from the bulk sum -- e.g. the
        1-2 ions closest to a candidate muon site if you plan to add
        their RELAXED positions back in via `extra_charges`.
    extra_charges : list of (position[Angstrom], charge[e]) or None
        Any additional point charges not in `atoms` -- e.g. relaxed
        neighbour positions, or the muon itself if you want it folded
        into a single combined tensor rather than kept separate.
    gamma_sternheimer : float, optional
        Sternheimer antishielding factor, V_total = V_lattice*(1+gamma).
        Default 0.0 (no antishielding correction, i.e. the bare 
        point-charge lattice sum).
    verbose : bool
        Print the number of charges summed over. Default False.
    Returns
    -------
    dict with keys:
        'Position' : (3,) array [Angstrom]
            Cartesian position of the nucleus (same as `site_position`).
        'Label' : str
            Label for the atom/site, e.g. 'F1' or 'Na2'.
        'Spin' : float
            Nuclear spin quantum number I of the nucleus at `site_position`.
        'Gamma' : float [rad/s/T]
            Gyromagnetic ratio of the nucleus at `site_position`.
        'ElectricQuadrupoleMoment' : float [m^2]
            Nuclear electric quadrupole moment Q of the nucleus at `site_position`.
        'EFGTensor' : (3,3) array [V/m^2]
            Lattice EFG tensor at `site_position`, symmetric and (up to numerical noise) traceless.
        'OmegaQmu' : float [rad/s] (only present if `mu_position` is not None)
            Muon-induced EFG frequency omegaQmu for the nucleus at `site_position`, computed from the distance to the muon 
            at `mu_position` and the nuclear quadrupole moment.
    """
    V_lattice = point_charge_EFG(
        atoms, 
        site_position, 
        charges, 
        sphere_radius=sphere_radius,
        exclude_indices=exclude_indices,
        extra_charges=extra_charges,
        gamma_sternheimer=gamma_sternheimer,
        verbose=verbose
    )

    d = {
        'Position': np.asarray(site_position) * constants.ANGSTROM,
        'Label': label,
        'Spin': spin,
        'Gamma': gamma,
        'ElectricQuadrupoleMoment': quadrupole_moment,
        'EFGTensor': V_lattice,
    }
    if muon_position is not None:
        d_mu = np.linalg.norm(np.asarray(site_position) - np.asarray(muon_position)) * constants.ANGSTROM
        d['OmegaQmu'] = get_omegaQ_mu(spin, quadrupole_moment, d_mu)
    return d