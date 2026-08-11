# %%
import numpy as np
from ase import Atom
from ase.neighborlist import neighbor_list
from ase.data import chemical_symbols, atomic_numbers

# %%
from constants import constants
from undi_tools.isotopes_deprecated import Element

# %%
def get_isotopes(Z):
    """
    Isotope lookup (bridges isotopes.py <-> ASE atomic numbers)
    Magnetic isotopes of the element with atomic number Z.

    Returns a list of (abundance_percent, spin, g_factor) tuples, one per
    magnetic (I>0) isotope, taken from the EasySpin database in isotopes.py.
    """
    symbol = chemical_symbols[Z]
    element = Element(symbol)
    return [
        (iso.abundance, iso.spin, iso.g_factor)
        for iso in element.magnetic_isotopes]


def _convert_atomic_cutoff_keys_to_atomic_numbers(cutoff):
    """
    Accepts a dict keyed by element symbol ('F', 'Ca', ...), atomic number
    (9, 20, ...), or a mix of both, and returns an equivalent dict keyed by
    atomic number, which is what ASE's neighbour list needs internally.

    Example
    -------
    normalize_cutoffs({'F': 40, 'Na': 25})  ->  {9: 40, 11: 25}
    """
    _cutoff = {}
    for key, value in cutoff.items():
        if isinstance(key, str):
            try:
                Z = atomic_numbers[key]
            except KeyError:
                raise ValueError(f"Unknown element symbol '{key}' in cutoff")
        else:
            Z = int(key)
        _cutoff[Z] = value
    return _cutoff


def _add_muon(atoms, muon_position):
    """Return a copy of `atoms` with a muon (H atom) in fractional coordinates 
    inserted at the end
    """
    atoms_mu = atoms.copy()
    atoms_mu.extend(Atom('H', [0.,0.,0.]))
    # update muon position
    pos = atoms_mu.get_scaled_positions()
    pos[-1] = muon_position
    atoms_mu.set_scaled_positions(pos)
    return atoms_mu


def _count_muon_atoms(atoms):
    """
    Count the muon (H) pseudo-atoms in `atoms`.
 
    Raises
    ------
    ValueError
        If no muon is present - i.e. compute_second_moments would silently
        divide by zero. Use add_muon() to insert one first.
    """
    tot_H = np.count_nonzero(atoms.get_atomic_numbers() == 1)
    if tot_H == 0:
        raise ValueError(
            "No muon (H) pseudo-atom found in structure. "
            "Use _add_muon() to insert one first."
        )
    return tot_H

# %%
def second_moments_fn(atoms, muon_position=None, cutoff=None):
    """
    Compute the isotope-averaged Van Vleck second moment at each muon (H) site
    in `atoms`, split by contributing element.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure containing the lattice atoms PLUS one (or more) muon
        pseudo-atoms of species 'H' (Z=1) at the site(s) of interest.
        If more than one H atom is present, the result is the average
        second moment per muon site.
    muon_position : (3,) array-like, optional
        Fractional coordinates of a muon site to insert before computing
        (via _add_muon()), for convenience so callers don't need a separate
        _add_muon() call first. If `atoms` already contains muon(s) and
        muon_position is also given, this one is added on top of them.
        Default None: `atoms` must already contain at least one 'H' atom.
    cutoff : dict {symbol_or_atomic_number: cutoff_in_Angstrom}
        Per-species real-space cutoff for the 1/r^6 lattice sum, e.g.
        {'F': 40, 'Ca': 20} or {9: 40, 20: 20} (symbols and atomic numbers
        can be mixed freely). Since the sum converges as ~1/R^3 (surface
        term), a cutoff of 20-40 Angstrom is normally already converged to
        <0.1%. Defaults to constants.MAX_CUTOFF_DISTANCE (40 Angstrom) for
        any species not given explicitly. Default None (NOT {}, to avoid
        the mutable-default-argument trap) -- an empty dict is created
        fresh inside the function on each call instead.


    Returns
    -------
    dict {chemical_symbol: sigma^2_species}  in (rad / second)^2
    """
    if muon_position is not None:
        atoms = _add_muon(atoms, muon_position)
 
    tot_H = _count_muon_atoms(atoms)
 
    cutoff = {} if cutoff is None else dict(cutoff)   # fresh dict every call
    cutoff = _convert_atomic_cutoff_keys_to_atomic_numbers(cutoff)
 
    # isotope-averaged <gamma^2 I(I+1)> for each element present
    species_avg = {}
    for e in np.unique(atoms.get_atomic_numbers()):
        if e == 1:
            continue
        species_avg[e] = 0.0
        for abundance, spin, g_factor in get_isotopes(e):
            species_avg[e] += (
                (abundance / 100) * spin * (spin + 1)
                * (constants.NUCLEAR_MAGNETON_OVER_HBAR * g_factor) ** 2
            )

    # lattice sums over 1/r^6, split by element
    specie_contribs = {}
    for e in np.unique(atoms.get_atomic_numbers()):
        if e == 1:
            continue
        d_ang = neighbor_list(
            'd', atoms, cutoff={(1, int(e)): cutoff.get(e, constants.MAX_CUTOFF_DISTANCE)}
        )
        d_m = d_ang * constants.ANGSTROM      # convert to metres for the 1/r^6 sum
        # the 0.5 avoids double counting: neighbor_list returns each
        # muon-nucleus pair twice (once from each end) when a cutoff is
        # specified for the (1, e) atomic-number pair.
        r6sum = 0.5 * np.sum(d_m ** -6)
        specie_contribs[chemical_symbols[e]] = (
            species_avg[e] * r6sum * constants.SECOND_MOMENT_PREFACTOR / tot_H
            )   # (rad/s)^2
 
    return specie_contribs

# %%
def _pretty_print(contributions, sigma2_total):
    """Print per-species and total second-moment contributions.
    `sigma2_total` must be in (rad/s)^2, matching second_moments()'s
    return convention."""
    print("Species contributions to sigma^2 (rad^2 / s^2):")

    width = max((len(s) for s in contributions), default=5)
    width = max(width, len('Total'))

    # for symbol, val in contributions.items():
    #     print(f"  {symbol:>{width}s}: {val: .6f}")
    # print(f"  {'Total':>{width}s}: {sigma2_total: .6f}")

    # Loop over species and print absolute + percentage contributions
    for symbol, val in contributions.items():
        pct = (val / sigma2_total) * 100 if sigma2_total > 0 else 0.0
        print(f"  {symbol:>{width}s}: {val:20.6f} (rad^2/s^2) | {pct:7.3f}%")

    print(f"  {'Total':>{width}s}: {sigma2_total:20.6f} (rad^2/s^2) | 100.000%")

    # sigma_rad_s = np.sqrt(sigma2_total)
    # # *1e-6 : rad/s -> rad/us   (Route A, verified against rad/s->Hz->MHz)
    # # /2pi  : rad/us -> cycles/us, which IS MHz (1 cycle/us = 1e6 cycles/s)
    # freq_MHz = sigma_rad_s * 1.0e-6 / (2 * np.pi)
    # print(f"  sigma^2 = {sigma2_total:.6f} rad^2/s^2"
    #       f"  sigma = sqrt(sigma^2) = {sigma_rad_s:.6f} rad/s"
    #       f"  =  {freq_MHz:.6f} MHz")

    return


def _shell_sum(atoms, species_avg, rmin, rmax, cutoff_search):
    """Isotope-weighted sum of <gamma^2 I(I+1)>/r^6 over all environment
    atoms lying in the shell rmin <= r < rmax (any species)."""
    tot_H = _count_muon_atoms(atoms)
    total = 0.0
    for e, avg in species_avg.items():
        d = neighbor_list('d', atoms, cutoff={(1, int(e)): cutoff_search})
        d = d[(d >= rmin) & (d < rmax)]
        total += avg * 0.5 * np.sum(d ** -6)
    return total


def _species_avg_gamma2II1(atoms):
    tot_H = _count_muon_atoms(atoms)
    out = {}
    for e in np.unique(atoms.get_atomic_numbers()):
        if e == 1:
            continue
        out[e] = sum(
            (ab / 100) * I * (I + 1) * (constants.NUCLEAR_MAGNETON_OVER_HBAR * g) ** 2
            for ab, I, g in get_isotopes(e)
        )
    return out


def compute_zeta_shell_sum(atoms, r_nn_max, r_nnn_max, r_inf=60.0):
    """
    Compute the scaling parameter zeta of Eq. (2) [J. M. Wilkinson PhysRevLett.125.087201 (2020)] / Eq. (6) [SI]:

        sigma_inf^2 = sigma_nn^2 + (2/3)(mu0/4pi)^2 hbar^2 gamma_mu^2
                      * sum_{j in nnn} gamma_j^2 I_j(I_j+1) / (zeta * r_j)^6

    i.e. zeta rescales the distances of the explicitly-included "next
    nearest neighbour" shell so that its contribution alone reproduces the
    contribution of every more-distant shell out to convergence.

        zeta^-6 = (sigma_inf^2 - sigma_nn^2) / sigma_nnn^2 (unscaled)
        zeta    = [ sigma_nnn^2 / (sigma_inf^2 - sigma_nn^2) ] ^ (1/6)

    Parameters
    ----------
    atoms : ase.Atoms
        Structure with the muon inserted as an 'H' pseudo-atom (see
        add_muon()).
    r_nn_max : float
        Outer radius (Angstrom) of the "nearest neighbour" shell(s) that
        are treated exactly (unscaled) in the explicit cluster Hamiltonian.
    r_nnn_max : float
        Outer radius (Angstrom) of the "next-nearest-neighbour" shell -
        i.e. atoms with r_nn_max <= r < r_nnn_max are the ones whose
        coupling gets scaled by 1/zeta^6 to act as a proxy for the rest
        of the (infinite) lattice.
    r_inf : float
        Cutoff radius (Angstrom) used to approximate the fully converged
        lattice sum. 40-60 Angstrom is normally converged to <0.1%.

    Returns
    -------
    zeta : float
    diagnostics : dict with S_nn, S_nnn, S_inf (the underlying sums,
        proportional to sigma^2 up to the shared `factor` prefactor -
        zeta itself doesn't depend on that prefactor since it cancels
        in the ratio).
    """
    species_avg = _species_avg_gamma2II1(atoms)
    S_nn  = _shell_sum(atoms, species_avg, 0.0, r_nn_max, r_inf)
    S_nnn = _shell_sum(atoms, species_avg, r_nn_max, r_nnn_max, r_inf)
    S_inf = _shell_sum(atoms, species_avg, 0.0, r_inf, r_inf)

    zeta = (S_nnn / (S_inf - S_nn)) ** (1 / 6)
    return zeta, {"S_nn": S_nn, "S_nnn": S_nnn, "S_inf": S_inf}