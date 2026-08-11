# %%
"""
Physical constants
"""
import numpy as np

# %%
GAMMAS = {
    "mu": 2*np.pi*135.53881e6,
    "F":  2*np.pi*40.053e6,
    "H":  2*np.pi*42.577e6,
    "V":  2*np.pi*11.212944e6,
} # units # (sT)^-1

# %%
PI    = np.pi
TWOPI = 2*PI

# %%
# SI Constants

MU0      = 4 * PI * 1.0e-7                            # T m A^-1
H_PLANCK = 6.62607015e-34                             # J s
PLANCK_H = 6.62607015e-34                             # J s
HBAR     = H_PLANCK/TWOPI                             # J s
PLANCK2PI = H_PLANCK/TWOPI                            # J s
PLANCK_HBAR = H_PLANCK/TWOPI                          # J s
EPSILON0 = 8.8541878128e-12                           # ampere^2 ⋅ kilogram^-1 ⋅ meter^−3 ⋅ second^4 (A^2 kg^-1 m^-3 s^4)
ANGTOM   = 1e-10                                      # meter
ANGSTROM = 1e-10                                      # meter
BOHR_TO_ANGSTROM    = 5.29177210544e-01               # angstrom
ANGSTROM_TO_METER   = 1e-10                           # meter
ELEMENTARY_CHARGE   = 1.602176634e-19                 # Coulomb = ampere ⋅ second
EFG_AMU_TO_SI       = 9.7173624424e21                 # V m^-2 (9.717366650590785e21) (V m^-2)
MAX_CUTOFF_DISTANCE = 40                              # in angstrom
PLANCK_CONSTANT     = 6.62607015e-34                  # J s
AVOGADRO_CONSTANT   = 6.02214076e23                   # mol^-1
BOHR_MAGNETON       = 9.2740100657e-24                # J T^-1 
BOHR_RADIUS         = 5.29177210544e-11               # m
BOLTZMANN_CONSTANT  = 1.380649e-23                    # J K^-1
HARTREE_ENERGY      = 4.3597447222060e-18             # J
HARTREE_ENERGY_EV   = 27.211386245981                 # eV
HARTREE_JOULE_RELATIONSHIP = 4.3597447222060e-18      # J
HARTREE_JOULE = 4.3597447222060e-18                   # J
HARTREE_HERTZ_RELATIONSHIP = 6.5796839204999e15       # Hz
HARTREE_HERTZ = 6.5796839204999e15                    # Hz
ELEMENTARY_CHARGE_OVER_HBAR = 1.519267447e15          # A J^-1
HARTREE_ELECTRON_VOLT = 27.211386245981               # eV
HARTREE_KELVIN_RELATIONSHIP = 3.1577502480398e5       # K
HARTREE_KELVIN = 3.1577502480398e5                    # K
HARTREE_KILOGRAM_RELATIONSHIP = 4.8508702095419e-35   # kg
HARTREE_KILOGRAM = 4.8508702095419e-35                # kg
ELECTRON_VOLT_JOULE_RELATIONSHIP = 1.602176634e-19    # J
ELECTRON_VOLT_JOULE = 1.602176634e-19                 # J
ELECTRON_VOLT_KELVIN_RELATIONSHIP = 1.160451812e4     # K
ELECTRON_VOLT_KELVIN = 1.160451812e4                  # K
ELECTRON_VOLT_KILOGRAM_RELATIONSHIP = 1.782661921e-36 # kg
ELECTRON_VOLT_KILOGRAM = 1.782661921e-36              # kg
KELVIN_ELECTRON_VOLT_RELATIONSHIP = 8.617333262e-5    # eV
KELVIN_ELECTRON_VOLT = 8.617333262e-5                 # eV
KELVIN_HARTREE_RELATIONSHIP = 3.1668115634564e-6      # E_h
KELVIN_HARTREE = 3.1668115634564e-6                   # E_h
KELVIN_HERTZ_RELATIONSHIP = 2.083661912e10            # Hz
KELVIN_HERTZ = 2.083661912e10                         # Hz
KELVIN_INVERSE_METER_RELATIONSHIP = 69.50348004       # m^-1
KELVIN_INVERSE_METER = 69.50348004                    # m^-1
KELVIN_JOULE_RELATIONSHIP = 1.380649e-23              # J
KELVIN_JOULE = 1.380649e-23                           # J
KELVIN_KILOGRAM_RELATIONSHIP = 1.536179187e-40        # kg
KELVIN_KILOGRAM = 1.536179187e-40                     # kg
KILOGRAM_ELECTRON_VOLT_RELATIONSHIP = 5.609588603e35  # eV
KILOGRAM_ELECTRON_VOLT  = 5.609588603e35              # eV
KILOGRAM_HARTREE_RELATIONSHIP = 2.0614857887415e34    # E_h
KILOGRAM_HARTREE = 2.0614857887415e34                 # E_h
KILOGRAM_HERTZ_RELATIONSHIP = 1.356392489e50          # Hz
KILOGRAM_HERTZ = 1.356392489e50                       # Hz
KILOGRAM_INVERSE_METER_RELATIONSHIP = 4.524438335e41  # m^-1
KILOGRAM_INVERSE_METER = 4.524438335e41               # m^-1
KILOGRAM_JOULE_RELATIONSHIP = 8.987551787e16          # J
KILOGRAM_JOULE = 8.987551787e16                       # J
KILOGRAM_KELVIN_RELATIONSHIP = 6.509657260e39         # K
KILOGRAM_KELVIN = 6.509657260e39                      # K
RYDBERG_CONSTANT = 10973731.568157                    # m^-1
SPEED_OF_LIGHT_IN_VACUUM = 299792458                  # m s^-1



# Gyromagnetic Ratios
MUON_GYROMAGNETIC_RATIO = GAMMAS["mu"]                # rad s^-1 T^-1

# nuclear magneton / hbar — converts a tabulated nuclear g-factor into
# a gyromagnetic ratio: gamma_j [rad/s/T] = g_j * MU_N_OVER_HBAR
NUCLEAR_MAGNETON_OVER_HBAR = TWOPI*7.622593285e6      # rad s^-1 T^-1

# Second Moment Prefactor
# #(2/3)(μ_0/4pi)^2 (hbar^2 × gamma_mu^2) 
# SECOND_MOMENT_PREFACTOR = 5.37402139e-5 # angstrom instead of m (old)
SECOND_MOMENT_PREFACTOR = (
    (2 / 3)
    * (MU0 / (4 * np.pi)) ** 2
    * HBAR ** 2
    * MUON_GYROMAGNETIC_RATIO ** 2
)

# Fermi contact field prefactor: converts a spin density difference
# delta_s(0) = rho_up(0) - rho_dw(0), given in atomic units (electrons per
# cubic Bohr radius, e/bohr^3), into the isotropic Fermi contact
# hyperfine field B_c in tesla, via B_c = SPIN_DENSITY_AU_TO_TESLA * delta_s(0).
#
# Derived from B_c = (2*mu_0/3) * mu_B * delta_s(0), with mu_0, mu_B in SI
# and delta_s(0) converted from e/bohr^3 to e/m^3 via division by a0**3.
# C = (2 × μ0 / 3) × μ_B / a0^3 = 52.430351
SPIN_DENSITY_AU_TO_TESLA = (2.0/3) * MU0 * BOHR_MAGNETON / BOHR_RADIUS**3 # Tesla per (e/bohr^3)