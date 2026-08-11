# %%
import warnings
import numpy as np

# %%
def zero_field_distribution_powder(atoms):
    """Calculates gamma_mu ^2 DeltaG ^2, where gamma_mu is the
        gyromagnetic ratio of the muon and Delta^2 is the variance of
        a Gaussian field distribution, using the secular approximation
        for the dipolar interaction. Powder averaging and zero external
        field are intended.

    Returns
    -------
    float
        gamma_mu ^2 DeltaG ^2. See above.
    """

    plank2pi = 1.0545718E-34 #joule second
    mu_0 = 0.0000012566371 # (kilogram meter) ∕ (ampere^2 × second^2)
    r = 0.
    gamma_mu = 0.
    pos_mu = None
    for atom in atoms:
        if atom['Label'] == 'mu':
            gamma_mu = atom['Gamma']
            pos_mu   = atom['Position']
            break
    else:
        warnings.warn('Multiple muons?! Only using last one in list')

    for atom in atoms:
        if atom['Label'] == 'mu':
            continue
        I = atom['Spin']
        gamma = atom['Gamma']
        r3 = np.linalg.norm(atom['Position'] - pos_mu)**3

        r += 2. * (mu_0/(4*np.pi))**2    * \
            ((gamma * plank2pi) / r3)**2 * \
            0.33333333333 * (I * (I+1))

    return r * (gamma_mu**2)

# %%
# ## Legacy alias
# vanvleck_second_moment = zero_field_distribution_powder
# van_vleck_second_moment = zero_field_distribution_powder
# compute_vanvleck_second_moment = zero_field_distribution_powder