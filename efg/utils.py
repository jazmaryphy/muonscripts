# %%
import numpy as np
from constants import constants

# %%
def Vzz_for_unit_charge_at_distance(r):
    """
    Axial EFG for a unit point charge (+e).

    Parameters
    ----------
    r : float
        Distance from charge [m].

    Returns
    -------
    float
        Vzz [V/m^2].
    """
    if r <= 0:
        raise ValueError("Distance must be positive")

    Vzz = (2./(4 * np.pi * constants.EPSILON0)) * (constants.ELEMENTARY_CHARGE / (r**3))
    return Vzz


def gen_radial_EFG(charge_position, site_position, charge=constants.ELEMENTARY_CHARGE, Vzz=None):
    """
    Point-charge EFG tensor at `site_position` due to a point charge
    sitting at `charge_position` (e.g. the muon), or an externally
    supplied Vzz (e.g. from DFT) instead of the point-charge model.

    Parameters
    ----------
    charge_position : (3,) ndarray
        Position of the point charge generating the field [m].
    site_position : (3,) ndarray
        Position of the nucleus where the EFG is evaluated [m].
    charge : float, optional
        Source charge [C] (default +e, i.e. the muon).
    Vzz : float, optional
        Axial EFG component [V/m^2]. If not supplied, computed from the
        point-charge model at the given `charge` and distance.

    Returns
    -------
    (3,3) ndarray
        EFG tensor [V/m^2] at `site_position`.
    """
    x = site_position - charge_position
    r = np.linalg.norm(x)
    if r <= 0:
        raise ValueError("charge_position and site_position are the same point.")
    x_hat = x / r
    if Vzz is None:
        Vzz = (charge / constants.ELEMENTARY_CHARGE) * Vzz_for_unit_charge_at_distance(r)
    return 0.5 * Vzz * (3 * np.outer(x_hat, x_hat) - np.eye(3)) # NOT -0.5*


def _get_omegaQ_mu(I, Q, r, gamma_sternheimer=0.0):   
    """Muon-induced EFG frequency omega_Q,mu [rad/s] for a nucleus of
    spin I and quadrupole moment Q [m^2] at distance r [m] from the
    muon's +e point charge, including the Sternheimer antishielding
    factor (1+gamma)

    Parameters
    ----------
    I : float
        Nuclear spin.
    Q : float
        Nuclear quadrupole moment [m^2].
    r : float
        Muon-nucleus distance [m].
    gamma_sternheimer : float, optional
        Sternheimer antishielding factor. Default is 0.0.

    Returns
    -------
    float
        Angular quadrupole frequency [rad/s].
    """ 
    return -(1 - gamma_sternheimer) * (1 / constants.HBAR) * (1.0 / (4 * np.pi * constants.EPSILON0)) \
        * (3 * constants.ELEMENTARY_CHARGE ** 2 * Q) / (2 * I * (2 * I - 1) * r ** 3)


def get_omegaQ_mu(I, Q, r):   
    """Muon-induced EFG frequency omega_Q,mu [rad/s] for a nucleus of
    spin I and quadrupole moment Q [m^2] at distance r [m] from the
    muon's +e point charge.

    Parameters
    ----------
    I : float
        Nuclear spin.
    Q : float
        Nuclear quadrupole moment [m^2].
    r : float
        Muon-nucleus distance [m].

    Returns
    -------
    float
        Angular quadrupole frequency [rad/s].
    """ 
    return _get_omegaQ_mu(I, Q, r, gamma_sternheimer=0.0)


def EFG_from_omegaq_PAS(omegaq, eta, m, I, Q):
    """
    Construct an EFG tensor in its principal-axis system (PAS)
    from a quadrupole frequency.

    Reference: From "Nuclear Quadrupole Resonance Spectroscopy" by 
    Hand and Das, Solid State Physics Supplement 1, the NQR transition frequency "omega"
    for a spin "S" with electric field gradient principal component "V_zz" and quadrupole 
    moment "Q".

    Parameters
    ----------
    omegaq : float
        Quadrupole frequency [rad/s].
    eta : float
        Asymmetry parameter.
    m : float
        Magnetic quantum number.
    I : float
        Nuclear spin.
    Q : float
        Nuclear quadrupole moment [m^2].

    Returns
    -------
    ndarray
        3x3 EFG tensor in PAS.
    """
    
    A = omegaq / (3 * (2 * np.abs(m) + 1)/constants.HBAR)  # Convert from rad/s to Hz  
    Vzz = A * (4 * I * (2 * I - 1))/(Q * constants.ELEMENTARY_CHARGE)

    Vxx = +0.5 * Vzz * (eta - 1)
    Vyy = -0.5 * Vzz * (eta + 1)

    return np.diag([Vxx, Vyy,Vzz])


def nu_Q(I, Q, Vzz):
    """
    Quadrupole frequency in MHz.

    Parameters
    ----------
    I : float
        Nuclear spin.
    Q : float
        Quadrupole moment [m^2].
    Vzz : float
        Principal EFG component [V/m^2].

    Returns
    -------
    float
        Quadrupole frequency in MHz.
    """

    return (3 * constants.ELEMENTARY_CHARGE * Q * Vzz / ( 2 * I * (2 * I - 1) * constants.PLANCK_H) * 1e-6)


def quadrupole_frequencies(I, Q, Vzz, eta):
    """
    Compute nu_x, nu_y, nu_z and nu_Q in MHz.

    Parameters
    ----------
    I : float
        Nuclear spin.
    Q : float
        Quadrupole moment [m^2].
    Vzz : float
        Principal EFG component [V/m^2].
    eta : float
        Asymmetry parameter.

    Returns
    -------
    dict
        Dictionary containing the computed frequencies in MHz.
    """
    prefactor = 3 * constants.ELEMENTARY_CHARGE * Q / (2 * I * (2 * I - 1) * constants.PLANCK_H) * 1e-6  # MHz

    Vxx = +0.5 * Vzz * (eta - 1)
    Vyy = -0.5 * Vzz * (eta + 1)

    nu_x_MHz = prefactor * Vxx
    nu_y_MHz = prefactor * Vyy
    nu_z_MHz = prefactor * Vzz

    nu_Q_MHz = abs(nu_z_MHz * np.sqrt(1 + eta**2 / 3))

    return {"Vxx": Vxx, "Vyy": Vyy, "Vzz": Vzz, "eta": eta, "nu_Q_MHz": nu_Q_MHz, \
            "nu_x_MHz": nu_x_MHz, "nu_y_MHz": nu_y_MHz, "nu_z_MHz": nu_z_MHz,}