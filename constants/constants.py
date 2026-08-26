# %%
"""Physical constants, unit conversion factors, and derived quantities.

This module provides physical constants, unit conversions, and derived factors
for electronic structure, nuclear quadrupolar resonance (NQR), point-charge EFG
calculations, and muon spin rotation (muSR) physics.

Where applicable, fundamental constants are sourced directly from standard SI
definitions and CODATA via `scipy.constants`.
"""

from typing import Dict
import numpy as np
import scipy.constants as const

# %% [markdown]
# Gyromagnetic Ratios

# %%
GAMMAS: Dict[str, float] = {
    "mu": 2.0 * np.pi * 135.53881e6,
    "F": 2.0 * np.pi * 40.053e6,
    "H": 2.0 * np.pi * 42.577e6,
    "V": 2.0 * np.pi * 11.212944e6,
}
"""Dict[str, float]: Gyromagnetic ratios ($\gamma$) for selected probes/nuclei in rad s^-1 T^-1."""

# %% [markdown]
# Mathematical Constants

# %%
PI: float = np.pi
"""float: Archimedes' constant $\pi$."""

TWOPI: float = 2.0 * np.pi
"""float: Circle constant $2\pi$ (tau)."""

# %% [markdown]
# Fundamental SI Constants (via scipy.constants)

# %%
MU0: float = const.mu_0
"""float: Vacuum magnetic permeability $\mu_0$ in T m A^-1 (or N A^-2)."""

H_PLANCK: float = const.h
"""float: Planck constant $h$ in J s."""

PLANCK_H: float = const.h
"""float: Alias for Planck constant $h$ in J s."""

PLANCK_CONSTANT: float = const.h
"""float: Alias for Planck constant $h$ in J s."""

HBAR: float = const.hbar
"""float: Reduced Planck constant $\hbar = h / (2\pi)$ in J s."""

PLANCK2PI: float = const.hbar
"""float: Alias for reduced Planck constant $\hbar$ in J s."""

PLANCK_HBAR: float = const.hbar
"""float: Alias for reduced Planck constant $\hbar$ in J s."""

EPSILON0: float = const.epsilon_0
"""float: Vacuum electric permittivity $\varepsilon_0$ in F m^-1 (or A^2 s^4 kg^-1 m^-3)."""

ELEMENTARY_CHARGE: float = const.e
"""float: Elementary charge $e$ in Coulombs (C = A s)."""

AVOGADRO_CONSTANT: float = const.N_A
"""float: Avogadro constant $N_A$ in mol^-1."""

BOHR_MAGNETON: float = const.value("Bohr magneton")
"""float: Bohr magneton $\mu_B$ in J T^-1."""

BOHR_RADIUS: float = const.value("Bohr radius")
"""float: Bohr radius $a_0$ in meters (m)."""

BOLTZMANN_CONSTANT: float = const.k
"""float: Boltzmann constant $k_B$ in J K^-1."""

SPEED_OF_LIGHT: float = const.c
"""float: Speed of light in vacuum $c$ in m s^-1."""

SPEED_OF_LIGHT_IN_VACUUM: float = const.c
"""float: Alias for speed of light in vacuum $c$ in m s^-1."""

RYDBERG_CONSTANT: float = const.Rydberg
"""float: Rydberg constant $R_\infty$ in m^-1."""

# %% [markdown]
# Distance & Calculation Cutoff Parameters

# %%
ANGSTROM: float = 1e-10
"""float: 1 Angstrom ($\text{\AA}$) in meters (m)."""

ANGTOM: float = 1e-10
"""float: Alias for 1 Angstrom ($\text{\AA}$) in meters (m)."""

ANGSTROM_TO_METER: float = 1e-10
"""float: Conversion factor from Angstroms to meters (m / $\text{\AA}$)."""

BOHR_TO_ANGSTROM: float = BOHR_RADIUS / ANGSTROM
"""float: Conversion factor from Bohr radii ($a_0$) to Angstroms ($\text{\AA}$)."""

MAX_CUTOFF_DISTANCE: float = 40.0
"""float: Default maximum summation cutoff radius in Angstroms ($\text{\AA}$)."""

# %% [markdown]
# Hartree Energy Conversions & Relationships

# %%
HARTREE_ENERGY: float = const.value("Hartree energy")
"""float: Hartree energy $E_h$ in Joules (J)."""

HARTREE_JOULE: float = HARTREE_ENERGY
"""float: Alias for Hartree energy in Joules (J)."""

HARTREE_JOULE_RELATIONSHIP: float = HARTREE_ENERGY
"""float: Conversion relationship from Hartree to Joules (J / $E_h$)."""

HARTREE_ENERGY_EV: float = const.value("Hartree energy in eV")
"""float: Hartree energy $E_h$ in electronvolts (eV)."""

HARTREE_ELECTRON_VOLT: float = HARTREE_ENERGY_EV
"""float: Alias for Hartree energy in electronvolts (eV)."""

HARTREE_HERTZ: float = HARTREE_ENERGY / H_PLANCK
"""float: Hartree energy equivalent frequency in Hertz (Hz)."""

HARTREE_HERTZ_RELATIONSHIP: float = HARTREE_HERTZ
"""float: Conversion relationship from Hartree to Hertz (Hz / $E_h$)."""

HARTREE_KELVIN: float = HARTREE_ENERGY / BOLTZMANN_CONSTANT
"""float: Hartree energy equivalent temperature in Kelvin (K)."""

HARTREE_KELVIN_RELATIONSHIP: float = HARTREE_KELVIN
"""float: Conversion relationship from Hartree to Kelvin (K / $E_h$)."""

HARTREE_KILOGRAM: float = HARTREE_ENERGY / (SPEED_OF_LIGHT**2)
"""float: Hartree energy mass equivalent in kilograms (kg)."""

HARTREE_KILOGRAM_RELATIONSHIP: float = HARTREE_KILOGRAM
"""float: Conversion relationship from Hartree to mass in kg (kg / $E_h$)."""

# %% [markdown]
# Electronvolt Conversions & Relationships

# %%
ELECTRON_VOLT_JOULE: float = const.eV
"""float: 1 electronvolt (eV) in Joules (J)."""

ELECTRON_VOLT_JOULE_RELATIONSHIP: float = const.eV
"""float: Conversion relationship from eV to Joules (J / eV)."""

ELECTRON_VOLT_KELVIN: float = const.eV / BOLTZMANN_CONSTANT
"""float: 1 electronvolt equivalent temperature in Kelvin (K)."""

ELECTRON_VOLT_KELVIN_RELATIONSHIP: float = ELECTRON_VOLT_KELVIN
"""float: Conversion relationship from eV to Kelvin (K / eV)."""

ELECTRON_VOLT_KILOGRAM: float = const.eV / (SPEED_OF_LIGHT**2)
"""float: 1 electronvolt mass equivalent in kilograms (kg)."""

ELECTRON_VOLT_KILOGRAM_RELATIONSHIP: float = ELECTRON_VOLT_KILOGRAM
"""float: Conversion relationship from eV to mass in kg (kg / eV)."""

# %% [markdown]
# Kelvin Conversions & Relationships

# %%
KELVIN_JOULE: float = BOLTZMANN_CONSTANT
"""float: 1 Kelvin equivalent energy in Joules (J)."""

KELVIN_JOULE_RELATIONSHIP: float = BOLTZMANN_CONSTANT
"""float: Conversion relationship from Kelvin to Joules (J / K)."""

KELVIN_ELECTRON_VOLT: float = BOLTZMANN_CONSTANT / const.eV
"""float: 1 Kelvin equivalent energy in electronvolts (eV)."""

KELVIN_ELECTRON_VOLT_RELATIONSHIP: float = KELVIN_ELECTRON_VOLT
"""float: Conversion relationship from Kelvin to eV (eV / K)."""

KELVIN_HARTREE: float = BOLTZMANN_CONSTANT / HARTREE_ENERGY
"""float: 1 Kelvin equivalent energy in Hartree ($E_h$)."""

KELVIN_HARTREE_RELATIONSHIP: float = KELVIN_HARTREE
"""float: Conversion relationship from Kelvin to Hartree ($E_h$ / K)."""

KELVIN_HERTZ: float = BOLTZMANN_CONSTANT / H_PLANCK
"""float: 1 Kelvin equivalent frequency in Hertz (Hz)."""

KELVIN_HERTZ_RELATIONSHIP: float = KELVIN_HERTZ
"""float: Conversion relationship from Kelvin to Hertz (Hz / K)."""

KELVIN_INVERSE_METER: float = BOLTZMANN_CONSTANT / (H_PLANCK * SPEED_OF_LIGHT)
"""float: 1 Kelvin equivalent wavenumber in inverse meters (m^-1)."""

KELVIN_INVERSE_METER_RELATIONSHIP: float = KELVIN_INVERSE_METER
"""float: Conversion relationship from Kelvin to inverse meters (m^-1 / K)."""

KELVIN_KILOGRAM: float = BOLTZMANN_CONSTANT / (SPEED_OF_LIGHT**2)
"""float: 1 Kelvin mass equivalent in kilograms (kg)."""

KELVIN_KILOGRAM_RELATIONSHIP: float = KELVIN_KILOGRAM
"""float: Conversion relationship from Kelvin to mass in kg (kg / K)."""

# %% [markdown]
# Kilogram Conversions & Relationships

# %%
KILOGRAM_JOULE: float = SPEED_OF_LIGHT**2
"""float: Mass-energy equivalence for 1 kg ($E = mc^2$) in Joules (J)."""

KILOGRAM_JOULE_RELATIONSHIP: float = KILOGRAM_JOULE
"""float: Conversion relationship from kg to mass-energy in Joules (J / kg)."""

KILOGRAM_ELECTRON_VOLT: float = (SPEED_OF_LIGHT**2) / const.eV
"""float: Mass-energy equivalence for 1 kg in electronvolts (eV)."""

KILOGRAM_ELECTRON_VOLT_RELATIONSHIP: float = KILOGRAM_ELECTRON_VOLT
"""float: Conversion relationship from kg to energy in eV (eV / kg)."""

KILOGRAM_HARTREE: float = (SPEED_OF_LIGHT**2) / HARTREE_ENERGY
"""float: Mass-energy equivalence for 1 kg in Hartree ($E_h$)."""

KILOGRAM_HARTREE_RELATIONSHIP: float = KILOGRAM_HARTREE
"""float: Conversion relationship from kg to Hartree ($E_h$ / kg)."""

KILOGRAM_HERTZ: float = (SPEED_OF_LIGHT**2) / H_PLANCK
"""float: Mass-energy equivalent frequency for 1 kg in Hertz (Hz)."""

KILOGRAM_HERTZ_RELATIONSHIP: float = KILOGRAM_HERTZ
"""float: Conversion relationship from kg to Hertz (Hz / kg)."""

KILOGRAM_INVERSE_METER: float = SPEED_OF_LIGHT / HBAR
"""float: Mass-energy equivalent wavenumber for 1 kg in inverse meters (m^-1)."""

KILOGRAM_INVERSE_METER_RELATIONSHIP: float = KILOGRAM_INVERSE_METER
"""float: Conversion relationship from kg to inverse meters (m^-1 / kg)."""

KILOGRAM_KELVIN: float = (SPEED_OF_LIGHT**2) / BOLTZMANN_CONSTANT
"""float: Mass-energy equivalent temperature for 1 kg in Kelvin (K)."""

KILOGRAM_KELVIN_RELATIONSHIP: float = KILOGRAM_KELVIN
"""float: Conversion relationship from kg to Kelvin (K / kg)."""

# %% [markdown]
# EFG & Interaction Prefactors

# %%
EFG_AMU_TO_SI: float = 9.7173624424e21
"""float: Conversion factor from atomic units of EFG ($e/a_0^3$) to SI units (V m^-2)."""

ELEMENTARY_CHARGE_OVER_HBAR: float = ELEMENTARY_CHARGE / HBAR
"""float: Ratio of elementary charge to reduced Planck constant $e / \hbar$ in A J^-1 (s^-1 V^-1)."""

MUON_GYROMAGNETIC_RATIO: float = GAMMAS["mu"]
"""float: Gyromagnetic ratio of the positive muon $\gamma_\mu$ in rad s^-1 T^-1."""

NUCLEAR_MAGNETON_OVER_HBAR: float = TWOPI * 7.622593285e6
"""float: Factor converting nuclear g-factor $g_I$ to gyromagnetic ratio $\gamma$ in rad s^-1 T^-1."""

SECOND_MOMENT_PREFACTOR: float = (
    (2.0 / 3.0)
    * (MU0 / (4.0 * np.pi)) ** 2
    * HBAR**2
    * MUON_GYROMAGNETIC_RATIO**2
)
"""float: Prefactor $(2/3) (\mu_0 / 4\pi)^2 \hbar^2 \gamma_\mu^2$ for van Vleck second moment calculations."""

SPIN_DENSITY_AU_TO_TESLA: float = (
    (2.0 / 3.0) * MU0 * BOHR_MAGNETON / (BOHR_RADIUS**3)
)
"""float: Prefactor converting atomic unit spin density ($e/a_0^3$) into Fermi contact field in Tesla (T / ($e/a_0^3$))."""