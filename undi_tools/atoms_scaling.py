# %%
"""
atoms_scaling.py
===================
Generic, N-shell, per-chemical-species version of the CaF2-specific
exclusive_shell_atoms()/rescale_shells() pair. Handles:

  - any number of shells (not just nn/nnn -- nn/nnn/nnnn/... all work
    the same way)
  - each shell containing a MIX of different atomic species (e.g. NaF's
    nn shell has both F and Na neighbors, at genuinely different
    distances, each needing its own fit parameter -- exactly the
    paper's r_F_nn=1.198 Ang / r_Na_nn=2.31 Ang two-parameter nn shell)
  - two different physical SCALING MODES per shell:
      'radius'          -- renormalize to an absolute fitted distance,
                            direction preserved (what you'd use for an
                            nn-type shell whose true bond length is a
                            fit parameter, e.g. r_nn or r_Na_nn)
      'multiplicative'  -- scale the existing position by a dimensionless
                            factor (what you'd use for a zeta-type shell,
                            which approximates everything beyond it and
                            should preserve that shell's own internal,
                            possibly non-uniform geometry rather than
                            collapse it to one fixed radius -- see Eq. 2
                            of PRL 125,087201 for why zeta multiplies
                            rather than renormalizes)

===========================================================================
UNDI-FORMAT ATOM DICT -- the data structure this whole pipeline runs on
===========================================================================
Every atom (including the muon) is a plain Python dict:

    {
        'Position': np.ndarray, shape (3,), SI units [metres]
                     -- e.g. an F 1.36 Angstrom from the muon is
                     np.array([0, 0, 1.36e-10]), NOT np.array([0,0,1.36]).
        'Label':     str, chemical symbol ('F', 'Na', 'Mn', ...) or the
                     literal string 'mu' for the muon. undi.py uses this
                     to auto-populate Spin/Gamma/ElectricQuadrupoleMoment
                     from isotopes.py if you don't set them explicitly.
        'Spin':      float, nuclear spin quantum number I
                     (0.5 for F/H, 1.5 for Na, 2.5 for Mn-55, 0 for mu-
                     coupled spin-0 species you'd normally just omit).
        'Gamma':     float, gyromagnetic ratio [rad/s/T].

        # OPTIONAL, only needed for quadrupolar (I > 1/2) nuclei -- see
        # undi.py's quadrupolar_interaction() / muon_induced_efg():
        'ElectricQuadrupoleMoment': float, nuclear quadrupole moment [m^2]
        'EFGTensor':  np.ndarray (3,3), the nucleus's OWN/lattice electric
                      field gradient tensor [V/m^2] (from the crystal
                      environment, NOT the muon -- see gen_efg_structure.py
                      earlier in this thread for how to build this)
        'OmegaQmu':   float [rad/s], the MUON-INDUCED EFG contribution
                      (from the muon's own +e point charge -- computed
                      separately from EFGTensor to avoid double-counting,
                      see gen_efg_structure.py's docstring)
    }

A "shell" is just a list of these dicts (the muon is never included in
a shell list -- it's always handled separately as a single dict).

A "cumulative raw shell list" (the convention your NN_ATOMS/NNN_ATOMS
already follow) means: raw_shells[0] is the innermost shell (optionally
still including the muon -- it gets filtered out either way);
raw_shells[1] is shell-0-plus-shell-1 combined together;
raw_shells[2] is shells 0+1+2 combined; and so on. This mirrors exactly
how a neighbor-shell finder like neighbor_shells.get_neighbor_shells()
or a distance-sorted CIF lookup naturally produces output -- "everything
within cutoff N" naturally nests "everything within cutoff N-1" inside
it. partition_into_shells() below turns that cumulative representation
into clean, DISJOINT shells.
===========================================================================
"""

import numpy as np
from copy import deepcopy

# %%
from constants import constants
from io_tools.read_ase import read_from_file
from undi_tools.second_moments import zero_field_distribution_powder

import sys
sys.set_int_max_str_digits(100000)

# %%
def normalize_cutoff(cutoff, species):
    """Accept either a single scalar cutoff (Angstrom) applied uniformly
    to every symbol in `species`, or an already-built per-species dict
    (any subset of `species`, or extras -- not validated against
    `species`, so you CAN pass cutoffs for symbols not in this
    particular shell without it being an error), and always return a
    proper {symbol: cutoff_in_Angstrom} dict -- the format
    gen_neighbour_atoms()/populate_undi_atoms() actually expect.

    Parameters
    ----------
    cutoff : float or dict {symbol: float}
        A single number ("use this cutoff for every species I care
        about here") or a dict ("use these specific per-species
        cutoffs").
    species : str or list of str
        The symbol(s) this cutoff applies to when `cutoff` is a scalar.
        Ignored (but harmless to pass) when `cutoff` is already a dict.

    Returns
    -------
    dict {symbol: cutoff_in_Angstrom}

    Examples
    --------
    >>> normalize_cutoff(4.0, 'F')
    {'F': 4.0}
    >>> normalize_cutoff(4.0, ['F', 'Na'])
    {'F': 4.0, 'Na': 4.0}
    >>> normalize_cutoff({'F': 2.0, 'Na': 2.5}, ['F', 'Na'])
    {'F': 2.0, 'Na': 2.5}
    """
    if isinstance(cutoff, dict):
        return dict(cutoff)   # already dict-shaped -- just copy, don't mutate caller's dict
    if isinstance(species, str):
        species = [species]
    if not isinstance(cutoff, (int, float, np.integer, np.floating)):
        # NOTE: np.isscalar('oops') is True (strings count as scalars to
        # numpy!), so that check alone would silently accept garbage
        # here -- explicit numeric-type check instead.
        raise TypeError(
            f"cutoff must be a number or a dict {{symbol: cutoff}}, got {type(cutoff)}"
        )
    return {s: float(cutoff) for s in species}

# %%
def exclusive_atoms(reference_atoms, combined_atoms, rtol=1e-4):
    """Atoms in `combined_atoms` that are NOT already in `reference_atoms`
    (matched by position; 'mu' entries on either side are ignored).

    `rtol` is scaled internally by the smallest reference distance --
    NOT a hardcoded absolute tolerance (an absolute atol silently breaks
    for SI-unit/metre-scale positions, since interatomic distances are
    themselves ~1e-10; see the atol=1e-5 bug earlier in this thread)."""
    ref_pos = np.array([a['Position'] for a in reference_atoms if a['Label'] != 'mu'])
    if len(ref_pos) == 0:
        return deepcopy([a for a in combined_atoms if a['Label'] != 'mu'])

    atol = rtol * np.linalg.norm(ref_pos, axis=1).min()
    result = []
    for atom in combined_atoms:
        if atom['Label'] == 'mu':
            continue
        pos = atom['Position']
        is_in_reference = np.any(np.all(np.isclose(pos, ref_pos, atol=atol), axis=1))
        if not is_in_reference:
            result.append(deepcopy(atom))
    return result


def partition_into_shells(mu_atom, raw_shells, rtol=1e-4):
    """Turn a CUMULATIVE list of raw shells (see module docstring) into
    clean, pairwise-DISJOINT shells: shells[0] is raw_shells[0] minus the
    muon; shells[1] is raw_shells[1] minus everything in shells[0] (and
    the muon); shells[2] is raw_shells[2] minus everything in
    shells[0]+shells[1]; and so on -- for ANY number of shells, not just
    the nn/nnn pair.

    Parameters
    ----------
    mu_atom : dict
        The muon atom.
    raw_shells : list of (list of dict)
        raw_shells[i] = every atom out to the i-th cutoff, combined
        (i.e. raw_shells[-1] is your next-next-...-nearest neighbor (NN...N) "everything so
        far" list; raw_shells[0] is your nearest neighbor (NN) atoms).
    rtol : float
        Passed to exclusive_atoms().

    Returns
    -------
    list of (list of dict) : disjoint shells, same length as raw_shells.
    """
    shells = []
    for raw in raw_shells:
        reference = [mu_atom] + [a for shell in shells for a in shell]
        shells.append(exclusive_atoms(reference, raw, rtol=rtol))
    return shells


def rescale_shell(shell_atoms, scale, mode='radius', default=None):
    """Rescale positions of atoms in one shell, PER CHEMICAL SPECIES.

    Parameters
    ----------
    shell_atoms : list of dict
        One shell's atoms (muon excluded).
    scale : float OR dict {species_label: float}
        A single float applies the same scale to every atom in the
        shell regardless of species (fine for a chemically pure shell,
        e.g. CaF2's all-fluorine nnn shell). A dict applies a DIFFERENT
        scale per species -- e.g. {'F': 1.198, 'Na': 2.31} for a mixed
        nn shell, exactly the paper's NaF fit (two independent nn
        distances for the two different neighbor species in the same
        shell). Species present in the shell but absent from the dict
        use `default`.
    mode : 'radius' or 'multiplicative'
        'radius': renormalize each atom's position to have norm=scale
            (scale given in Angstrom, converted to metres via `ANGSTROM`),
            direction preserved. Use for a shell whose ABSOLUTE bond
            length is itself a fit parameter (nn-type shells).
        'multiplicative': multiply the existing position by the
            dimensionless factor `scale` (no unit conversion, no
            renormalization). Use for a zeta-type shell that proxies
            everything beyond it and should keep its own internal
            geometry rather than collapse to one fixed radius.
    default : float or None
        Scale for species present in `shell_atoms` but not listed as a
        key in a dict `scale`. None (default) leaves that species'
        position completely unchanged -- useful if you only want to
        move SOME species in a mixed shell and leave the rest fixed at
        their ideal-lattice geometry.

    Returns
    -------
    list of dict : a NEW list (does not mutate `shell_atoms`).
    """
    if mode not in ('radius', 'multiplicative'):
        raise ValueError(f"mode must be 'radius' or 'multiplicative', got {mode!r}")

    out = deepcopy(shell_atoms)
    for atom in out:
        species = atom['Label']
        if isinstance(scale, dict):
            if species not in scale:
                if default is None:
                    continue  # leave this species' position untouched
                s = default
            else:
                s = scale[species]
        else:
            s = scale  # single scalar -> applies to every species alike

        if mode == 'radius':
            r = np.linalg.norm(atom['Position'])
            if r > 0:
                atom['Position'] = (atom['Position'] / r) * constants.ANGSTROM * s
        else:  # 'multiplicative'
            atom['Position'] = atom['Position'] * s
    return out

# %%
def scale_neighbors(
    neighbors,
    scale=1.0,
    mode="multiplicative",
    default=None,
    include_muon=True,
):
    """Rescale positions of atoms in a shell/neighbor cluster, PER CHEMICAL SPECIES,
    with independent control over the muon.
 
    This merges the old `rescale_shell` (per-species scaling engine, muon
    excluded from input) and `scale_neighbors_positions` (single global
    factor, muon pinned to the origin) into one function.
 
    Parameters
    ----------
    neighbors : iterable of dict
        Atoms (may or may not include a muon entry with ``Label == 'mu'``).
        Not mutated -- a new list is returned.
    scale : float OR dict {species_label: float}, default=1.0
        A single float applies the same scale to every non-muon atom
        regardless of species (fine for a chemically pure shell, e.g.
        CaF2's all-fluorine nnn shell, or for the old global
        `scale_factor` use case). A dict applies a DIFFERENT scale per
        species -- e.g. {'F': 1.198, 'Na': 2.31} for a mixed nn shell
        (the paper's NaF fit with two independent nn distances for the
        two neighbor species in the same shell). Species present in the
        cluster but absent from the dict use `default`.
    mode : 'radius' or 'multiplicative', default='multiplicative'
        'radius': renormalize each atom's position to have norm=scale
            (scale given in Angstrom, converted via `constants.ANGSTROM`),
            direction preserved. Use when the shell's ABSOLUTE bond
            length is itself a fit parameter (nn-type shells).
        'multiplicative': multiply the existing position by the
            dimensionless factor `scale` (no unit conversion, no
            renormalization). Use for a zeta-type shell that proxies
            everything beyond it, or to reproduce the old
            `scale_neighbors_positions` global-scale-factor behaviour.
    default : float or None, default=None
        Scale for species present in `neighbors` but not listed as a key
        in a dict `scale`. None (default) leaves that species' position
        completely unchanged -- useful if you only want to move SOME
        species in a mixed shell and leave the rest fixed at their
        ideal-lattice geometry. Ignored when `scale` is a single float.
    include_muon : bool, default=True
        If True, ensure a muon entry is present in the output, pinned at
        the origin (never rescaled). If False, remove any muon entry
        from the output.
 
    Returns
    -------
    list of dict
        A NEW list of atoms (does not mutate `neighbors`).
    """
    if mode not in ("radius", "multiplicative"):
        raise ValueError(f"mode must be 'radius' or 'multiplicative', got {mode!r}")
 
    out = []
    muon = None
 
    for atom in deepcopy(neighbors):
        species = atom.get("Label")
 
        if species == "mu":
            muon = atom
            continue
 
        if isinstance(scale, dict):
            if species not in scale:
                if default is None:
                    out.append(atom)  # leave this species' position untouched
                    continue
                s = default
            else:
                s = scale[species]
        else:
            s = scale  # single scalar -> applies to every species alike
 
        if mode == "radius":
            r = np.linalg.norm(atom["Position"])
            if r > 0:
                atom["Position"] = (atom["Position"] / r) * constants.ANGSTROM * s
        else:  # 'multiplicative'
            atom["Position"] = np.asarray(atom["Position"]) * s
 
        out.append(atom)
 
    if include_muon:
        if muon is None:
            muon = {
                "Position": np.zeros(3),
                "Spin": 0.5,
                "Label": "mu",
                "Gamma": constants.MUON_GYROMAGNETIC_RATIO,
            }
        else:
            muon["Position"] = np.zeros(3)
        out.insert(0, muon)
 
    return out
  
 
def rescale_shell(shell_atoms, scale, mode="radius", default=None):
    """Backward-compatible alias for `scale_neighbors` (muon excluded, as before)."""
    return scale_neighbors(
        shell_atoms, scale=scale, mode=mode, default=default, include_muon=False
    )
 
 
def scale_neighbors_positions(neighbors, scale_factor=1.0, include_muon=True):
    """Backward-compatible alias for `scale_neighbors` (global multiplicative scale)."""
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive.")
    return scale_neighbors(
        neighbors,
        scale=scale_factor,
        mode="multiplicative",
        default=None,
        include_muon=include_muon,
    )

# %%
def build_scaled_cluster(mu_atom, raw_shells, radius_params=None, scale_params=None, rtol=1e-4):
    """Generic N-shell cluster builder -- the direct replacement for
    the old CaF2-only build_fit_cluster(r_nn, zeta).

    Parameters
    ----------
    mu_atom : dict
        The muon atom.
    raw_shells : list of (list of dict)
        CUMULATIVE raw shells (see module docstring / partition_into_shells).
        Any number of shells: [nn] for one shell, [nn, nnn] for two
        (the CaF2 case), [nn, nnn, nnnn] for three, etc.
    radius_params : None, float, dict, or list
        Radius-mode scaling (see rescale_shell). A bare float or dict
        applies to shell 0 ONLY (this is the backward-compatible form:
        old code's single r_nn=1.17 becomes radius_params=1.17, or for
        NaF's mixed nn shell, radius_params={'F':1.198,'Na':2.31}).
        A list applies one entry per shell, e.g.
        [{'F':1.198,'Na':2.31}, None, None] to only scale shell 0, or
        [r_nn, None, None] etc. -- pass None for any shell you don't
        want radius-scaled.
    scale_params : None, float, dict, or list
        Multiplicative (zeta-type) scaling, same per-shell convention as
        radius_params. Old code's single zeta=0.92 becomes
        scale_params=[None, 0.92] (shell 0 untouched, shell 1 scaled) --
        note this is DIFFERENT from radius_params' "applies to shell 0
        only" default, because zeta conventionally scales the OUTER
        shell, not the inner one; be explicit with a list to avoid
        ambiguity for your specific case.
    rtol : float
        Passed to partition_into_shells().

    Returns
    -------
    list of dict : mu_atom + every scaled shell, concatenated -- ready
    for undi.MuonNuclearInteraction(atoms).

    Examples
    --------
    CaF2 (old 2-parameter r_nn/zeta case, chemically pure shells):
        atoms = build_scaled_cluster(
            mu_atom, [nn_raw, nnn_raw],
            radius_params=[r_nn, None],
            scale_params=[None, zeta])

    NaF (mixed-species nn shell, two independent nn distances):
        atoms = build_scaled_cluster(
            mu_atom, [nn_raw, nnn_raw],
            radius_params=[{'F': r_F_nn, 'Na': r_Na_nn}, None],
            scale_params=[None, zeta])

    Three shells (nn, nnn, nnnn), only the first two get fit parameters,
    the outermost is left at its ideal-lattice geometry:
        atoms = build_scaled_cluster(
            mu_atom, [nn_raw, nnn_raw, nnnn_raw],
            radius_params=[r_nn, None, None],
            scale_params=[None, zeta, None])
    """
    shells = partition_into_shells(mu_atom, raw_shells, rtol=rtol)
    n = len(shells)

    def _as_per_shell_list(params):
        if params is None:
            return [None] * n
        if isinstance(params, list):
            return params + [None] * (n - len(params))
        return [params] + [None] * (n - 1)   # bare float/dict -> shell 0 only

    radius_list = _as_per_shell_list(radius_params)
    scale_list = _as_per_shell_list(scale_params)

    out = [deepcopy(mu_atom)]
    for shell, r_param, s_param in zip(shells, radius_list, scale_list):
        if r_param is not None:
            shell = rescale_shell(shell, r_param, mode='radius')
        if s_param is not None:
            shell = rescale_shell(shell, s_param, mode='multiplicative')
        out += shell
    return out


def get_muon_atom(cluster):
    """ONE canonical place to extract the muon -- used by every function
    below instead of each reimplementing `[a for a in c if a['Label']=='mu']`
    inline (the duplication in the original compute_zeta_cluster/
    only_muon_cluster pair is exactly the kind of thing that silently
    diverges if the muon-matching convention ever changes)."""
    mu_atoms = [a for a in cluster if a['Label'] == 'mu']
    if len(mu_atoms) != 1:
        raise ValueError(f"Expected exactly 1 muon atom, found {len(mu_atoms)}")
    return mu_atoms[0]


def only_muon_cluster(cluster):
    """Return a new single-atom list containing just the muon from `cluster`."""
    return [deepcopy(get_muon_atom(cluster))]


def compute_zeta_cluster(inf_cluster, two_shells, rtol=1e-4):
    mu_atom = [a for a in two_shells[0] if a['Label'] == 'mu']
    shells = partition_into_shells(mu_atom[0], two_shells, rtol=rtol)

    nn, nnn = shells[0], shells[1]

    inf_S2 = zero_field_distribution_powder(inf_cluster)
    nn_S2  = zero_field_distribution_powder(mu_atom+nn)
    nnn_S2 = zero_field_distribution_powder(mu_atom+nnn)

    return (nnn_S2 / (inf_S2 - nn_S2)) ** (1 / 6)


def compute_zeta_cluster(inf_cluster, shells, rtol=1e-4):
    """Generalized N-shell zeta computation 
    (Eq. 2, J.M Wilkinson et al., PRL 125, 087201 (2020)).

    The LAST shell in `shells` is the one zeta scales (proxying
    everything beyond it); every earlier shell is treated as exact.
    [nn_raw, nnn_raw] reproduces the original CaF2 2-shell formula
    exactly; [nn_raw, nnn_raw, nnnn_raw] pushes the boundary out one
    shell further; [nn_raw] alone does a single-shell zeta fit.

    Parameters
    ----------
    inf_cluster : list of dict
        Large reference cluster/lattice for sigma^2_inf (never used to
        build the actual Hamiltonian).
    shells : list of (list of dict)
        CUMULATIVE raw shells (partition_into_shells convention).
    second_moment_fn : callable
        Your compute_vanvleck_second_moment (injected so this stays
        testable without depending on your specific implementation).
    rtol : float
        Passed to partition_into_shells().

    Returns
    -------
    zeta : float
    """
    mu_atom = get_muon_atom(shells[0])
    disjoint_shells = partition_into_shells(mu_atom, shells, rtol=rtol)

    *inner_shells, outer_shell = disjoint_shells
    inner_flat = [a for shell in inner_shells for a in shell]

    inf_S2   = zero_field_distribution_powder(inf_cluster)
    inner_S2 = zero_field_distribution_powder([mu_atom] + inner_flat) if inner_flat else 0.0
    outer_S2 = zero_field_distribution_powder([mu_atom] + outer_shell)

    denom = inf_S2 - inner_S2
    if denom <= 0:
        raise ValueError("inf_S2 <= inner_S2: check your lattice cutoff/positions.")
    return (outer_S2 / denom) ** (1 / 6)

# %%
# def compute_zeta_from_file(filename, muon_position, shell_cutoffs, species, 
#                            inf_cutoff=40, rtol=1e-4, **gen_kwargs):
#     """Convenience composition (NOT a merge) of the file-reading layer and
#     the pure-math layer: builds the cumulative shells and the reference
#     lattice from a structure file via YOUR gen_neighbour_atoms() +
#     populate_undi_atoms(), then hands them to compute_zeta_cluster()
#     unchanged. Keeps the two layers independently testable/reusable
#     while still giving you the one-call convenience for the common case.

#     Parameters
#     ----------
#     filename : str
#     muon_position : (3,) fractional coordinates
#     shell_cutoffs : list of (float or dict)
#         One entry per cumulative shell. Each entry can be a single
#         number (applied to every symbol in `species` via
#         normalize_cutoff) or an explicit {symbol: cutoff} dict for
#         mixed-species shells (e.g. NaF's nn shell needing different F
#         and Na distances) -- e.g. [4.0, 8.0] for a simple F-only
#         two-shell case (matches your real call's cutoffs={'F': 4.0}
#         shape once normalized), or
#         [{'F': 2.0}, {'F': 4.0, 'Na': 5.0}] for a mixed case.
#     species : str or list of str
#         Which symbol(s) a scalar entry in `shell_cutoffs` expands to
#         (ignored for entries that are already dicts). Default 'F' to
#         match your real usage; pass e.g. ['F', 'Na'] for NaF.
#     inf_cutoff : float
#         Large cutoff [Angstrom] for the sigma^2_inf reference lattice --
#         matches your real gen_neighbour_atoms(..., inf_cutoff=20.0) call.
#     """
#     shells = []
#     for cutoff in shell_cutoffs:
#         cutoff_dict = normalize_cutoff(cutoff, species)
#         data = gen_neighbour_atoms(filename, muon_position, cutoffs=cutoff_dict, 
#                                    inf_cutoff=inf_cutoff, **gen_kwargs)
#         shells.append(populate_undi_atoms(data))

#     inf_cutoff_dict = normalize_cutoff(inf_cutoff, species)
#     inf_data = gen_neighbour_atoms(filename, muon_position, cutoffs=inf_cutoff_dict, 
#                                    inf_cutoff=inf_cutoff, **gen_kwargs)
#     inf_cluster = populate_undi_atoms(inf_data)

#     return compute_zeta_cluster(inf_cluster, shells, rtol=rtol)