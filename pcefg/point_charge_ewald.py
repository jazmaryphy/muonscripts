#!/usr/bin/env python3
"""
efg_ewald.py

Periodic point-charge Electric Field Gradient (EFG) calculations using
Ewald summation.

Features
--------
* Periodic EFG using Ewald summation
* Arbitrary ASE crystal cells
* Muon contribution handled separately
* Explicit relaxed-neighbour embedding
* Sternheimer correction
* PAS (principal axis system) analysis
* Ewald parameter convergence checking

Theory
------
The EFG tensor is

    V_ab = d²Φ/(dx_a dx_b)

For a point charge q:

    V_ab =
        q/(4*pi*eps0)
        (3 r_a r_b - r² δ_ab)/r⁵

For a crystal:

    V = V_real + V_recip

using Ewald decomposition.

Recommended μSR workflow
------------------------
    V_total = V_bulk + V_local + V_mu

where

V_bulk
    Infinite perfect crystal (Ewald).

V_local
    Explicit relaxed neighbours around muon.

V_mu
    Direct muon contribution.

Units
-----
Coordinates : Angstrom
Charges     : units of e
EFG         : V/m²
"""

import numpy as np
from scipy.special import erfc

##############################################################################
# CONSTANTS
##############################################################################

EPSILON0 = 8.8541878128e-12
ELEMENTARY_CHARGE = 1.602176634e-19
PLANCK_H = 6.62607015e-34
ANGSTROM = 1e-10

##############################################################################
# GENERAL UTILITIES
##############################################################################

def reciprocal_vectors(cell):
    return 2.0 * np.pi * np.linalg.inv(cell).T


def check_charge_neutrality(atoms, charges):
    qtot = sum(charges.get(s, 0.0) for s in atoms.get_chemical_symbols())

    if abs(qtot) > 1e-10:
        raise ValueError(
            f"Unit cell is not neutral. Total charge = {qtot:.6f} e"
        )

##############################################################################
# POINT CHARGE EFG
##############################################################################

def single_charge_EFG(charge_position, site_position, charge):
    d = site_position - charge_position
    r2 = np.dot(d, d)

    if r2 < 1e-30:
        return np.zeros((3, 3))

    r5 = r2**2.5

    return charge * (3*np.outer(d, d) - r2*np.eye(3)) \
        / (4*np.pi*EPSILON0*r5)


def muon_point_charge_EFG(mu_position, site_position):
    mu = np.asarray(mu_position) * ANGSTROM
    site = np.asarray(site_position) * ANGSTROM

    return single_charge_EFG(
        mu,
        site,
        ELEMENTARY_CHARGE
    )

##############################################################################
# REAL SPACE IMAGES
##############################################################################

def generate_real_space_images(
    atoms,
    charges,
    cutoff_m,
    exclude_indices=()
):

    cell = np.array(atoms.get_cell()) * ANGSTROM
    pos = atoms.get_positions() * ANGSTROM
    sym = atoms.get_chemical_symbols()

    lengths = np.linalg.norm(cell, axis=1)

    nmax = np.ceil(cutoff_m / lengths).astype(int) + 2

    pts = []
    qs = []

    for na in range(-nmax[0], nmax[0] + 1):
        for nb in range(-nmax[1], nmax[1] + 1):
            for nc in range(-nmax[2], nmax[2] + 1):

                shift = na*cell[0] + nb*cell[1] + nc*cell[2]

                if np.linalg.norm(shift) > cutoff_m + np.max(lengths):
                    continue

                for i, (p, s) in enumerate(zip(pos, sym)):
                    if i in exclude_indices:
                        continue

                    q = charges.get(s, 0.0)

                    if q == 0:
                        continue

                    pts.append(p + shift)
                    qs.append(q * ELEMENTARY_CHARGE)

    return np.asarray(pts), np.asarray(qs)

##############################################################################
# REAL SPACE EWALD TERM
##############################################################################

def real_space_efg(d, q, eta):

    r2 = np.sum(d*d, axis=1)
    r = np.sqrt(r2)

    outer = np.einsum("ni,nj->nij", d, d)

    T = 3*outer - r2[:, None, None]*np.eye(3)

    A = erfc(eta*r) / r**5

    B = (
        2*eta/np.sqrt(np.pi)
        * np.exp(-(eta*r)**2)
        * (1/r**4 + 2*eta**2/r**2)
    )

    V = np.sum(
        q[:, None, None]
        * T
        * (A[:, None, None] + B[:, None, None]),
        axis=0
    )

    return V / (4*np.pi*EPSILON0)

##############################################################################
# RECIPROCAL SPACE EWALD TERM
##############################################################################

def reciprocal_space_efg(
    atoms,
    charges,
    site_m,
    eta,
    gmax
):

    cell = np.array(atoms.get_cell()) * ANGSTROM
    vol = abs(np.linalg.det(cell))

    Gbasis = reciprocal_vectors(cell)

    pos = atoms.get_positions() * ANGSTROM
    sym = atoms.get_chemical_symbols()

    lengths = np.linalg.norm(Gbasis, axis=1)
    ng = np.ceil(gmax / lengths).astype(int) + 1

    V = np.zeros((3, 3))

    for h in range(-ng[0], ng[0] + 1):
        for k in range(-ng[1], ng[1] + 1):
            for l in range(-ng[2], ng[2] + 1):

                if h == 0 and k == 0 and l == 0:
                    continue

                G = h*Gbasis[0] + k*Gbasis[1] + l*Gbasis[2]

                Gnorm = np.linalg.norm(G)

                if Gnorm > gmax:
                    continue

                sf = 0.0 + 0.0j

                for p, s in zip(pos, sym):

                    q = charges.get(s, 0.0)

                    if q == 0:
                        continue

                    sf += (
                        q * ELEMENTARY_CHARGE
                        * np.exp(-1j*np.dot(G, p - site_m))
                    )

                damp = np.exp(-Gnorm**2/(4*eta**2))

                V += (
                    (4*np.pi/vol)
                    * damp
                    * sf.real
                    * np.outer(G, G)
                    / Gnorm**2
                )

    V /= EPSILON0

    V -= np.trace(V)/3*np.eye(3)

    return V

##############################################################################
# MAIN EWALD EFG
##############################################################################

def point_charge_EFG_ewald(
    atoms,
    site_position,
    charges,
    exclude_indices=(),
    extra_charges=None,
    gamma_sternheimer=0.0,
    eta=None,
    real_cutoff=15.0,
    gmax=40.0,
    verbose=False
):

    check_charge_neutrality(atoms, charges)

    site = np.asarray(site_position) * ANGSTROM

    cell = np.array(atoms.get_cell()) * ANGSTROM
    volume = abs(np.linalg.det(cell))

    if eta is None:
        eta = 5.6 / volume**(1/3)

    cutoff_m = real_cutoff * ANGSTROM
    gmax_m = gmax / ANGSTROM

    pts, qs = generate_real_space_images(
        atoms,
        charges,
        cutoff_m,
        exclude_indices
    )

    if extra_charges:

        ext_pos = np.array([p for p, q in extra_charges]) * ANGSTROM
        ext_q = np.array([q for p, q in extra_charges]) * ELEMENTARY_CHARGE

        pts = np.vstack([pts, ext_pos])
        qs = np.concatenate([qs, ext_q])

    d = pts - site

    r2 = np.sum(d*d, axis=1)
    mask = r2 > 1e-24

    d = d[mask]
    qs = qs[mask]

    V_real = real_space_efg(d, qs, eta)

    V_recip = reciprocal_space_efg(
        atoms,
        charges,
        site,
        eta,
        gmax_m
    )

    V = V_real + V_recip

    V = 0.5 * (V + V.T)

    V -= np.trace(V)/3*np.eye(3)

    V *= (1 + gamma_sternheimer)

    if verbose:
        print(
            f"eta={eta:.3e}  "
            f"real_cutoff={real_cutoff:.1f} Å  "
            f"gmax={gmax:.1f} Å⁻¹"
        )

    return V

##############################################################################
# TOTAL MUON EFG
##############################################################################

def total_muon_EFG(
    atoms,
    site_position,
    mu_position,
    charges,
    exclude_indices=(),
    extra_charges=None,
    gamma_sternheimer=0.0,
    **kwargs
):

    V_bulk = point_charge_EFG_ewald(
        atoms,
        site_position,
        charges,
        exclude_indices=exclude_indices,
        extra_charges=extra_charges,
        gamma_sternheimer=gamma_sternheimer,
        **kwargs
    )

    V_mu = muon_point_charge_EFG(
        mu_position,
        site_position
    )

    return V_bulk + V_mu

##############################################################################
# PAS ANALYSIS
##############################################################################

def diagonalize_EFG(tensor, quadrupole_moment):

    eigvals, eigvecs = np.linalg.eigh(tensor)

    order = np.argsort(-np.abs(eigvals))

    Vzz, Vyy, Vxx = eigvals[order]

    eta = np.abs(Vxx - Vyy)/np.abs(Vzz)

    chi = abs(
        Vzz
        * ELEMENTARY_CHARGE
        * quadrupole_moment
        / PLANCK_H
    )

    return {
        "Vxx": Vxx,
        "Vyy": Vyy,
        "Vzz": Vzz,
        "eta": eta,
        "chi_MHz": chi*1e-6
    }

##############################################################################
# CONVERGENCE TEST
##############################################################################

def ewald_convergence(
    atoms,
    site_position,
    charges,
    quadrupole_moment=1e-28
):

    print("\nEwald convergence test\n")

    settings = [
        (0.15, 10, 20),
        (0.15, 15, 30),
        (0.20, 15, 40),
        (0.25, 20, 50),
    ]

    previous = None

    print(
        f"{'eta':>8}"
        f"{'rcut':>8}"
        f"{'gmax':>8}"
        f"{'Vzz':>18}"
        f"{'delta':>14}"
    )

    for etaA, rc, gm in settings:

        V = point_charge_EFG_ewald(
            atoms,
            site_position,
            charges,
            eta=etaA/ANGSTROM,
            real_cutoff=rc,
            gmax=gm
        )

        result = diagonalize_EFG(
            V,
            quadrupole_moment
        )

        Vzz = result["Vzz"]

        delta = ""
        if previous is not None:
            delta = f"{100*abs(Vzz-previous)/abs(Vzz):.3f}%"

        print(
            f"{etaA:8.3f}"
            f"{rc:8.1f}"
            f"{gm:8.1f}"
            f"{Vzz:18.5e}"
            f"{delta:>14}"
        )

        previous = Vzz

##############################################################################
# EXAMPLE
##############################################################################

if __name__ == "__main__":

    from ase.build import bulk

    print("\nBuilding NaCl test crystal\n")

    atoms = bulk(
        "NaCl",
        crystalstructure="rocksalt",
        a=5.64
    )

    charges = {
        "Na": +1,
        "Cl": -1
    }

    site = atoms.positions[0]

    muon_position = np.array([
        1.2,
        1.2,
        1.2
    ])

    ewald_convergence(
        atoms,
        site,
        charges
    )

    V = total_muon_EFG(
        atoms,
        site,
        muon_position,
        charges,
        real_cutoff=20,
        gmax=50,
        verbose=True
    )

    print("\nEFG tensor [V/m²]\n")
    print(V)

    print("\nSymmetry check")
    print(np.allclose(V, V.T))

    print("\nTrace check")
    print(np.trace(V))

    result = diagonalize_EFG(
        V,
        quadrupole_moment=1e-28
    )

    print("\nPrincipal axis system\n")

    for k, v in result.items():
        print(f"{k:<10s} : {v:.6e}")