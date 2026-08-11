# %%
from __future__ import annotations
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Tuple
from collections import defaultdict

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from scipy.stats import gaussian_kde
from scipy.integrate import simpson

# %%
from constants import constants
from muesr_tools.local_fields import pfields, rfields, multisite_pfields
from muesr_tools.sample_candidate_sites import (
    prune_atoms_too_close, 
    sample_candidate_sites, 
    find_equivalent_positions
)

GAMMA_MU = constants.MUON_GYROMAGNETIC_RATIO/constants.TWOPI
GAMMA_MU *=1e-6 # MHz/T 

# %%
try:
    import emcee
    HAVE_EMCEE = True
except ImportError:
    HAVE_EMCEE = False

# %%


# %%
def get_atom_kinds_pymatgen(structure: Structure) -> dict[str, list[int]]:
    """
    Group atoms into symmetry-equivalent kinds and label them by element.

    Args:
        structure (pymatgen.Structure): Structure to analyze.

    Returns:
        dict[str, list[int]]: Mapping from kind label to the list of
            0-based atom indices (matching `structure`) belonging to
            that symmetry-equivalent kind. The label is the plain
            element symbol (e.g. "Fe") if the structure has only one
            symmetry-inequivalent site of that element, or the symbol
            suffixed with a 1-based index (e.g. "Fe1", "Fe2", ...) if
            there are several inequivalent sites of the same element --
            matching the kind-labeling convention used in QE input files.
    """
    analyzer = SpacegroupAnalyzer(structure)
    equiv = analyzer.get_symmetry_dataset().equivalent_atoms

    # group atom indices by their representative (kind) index
    kind_dict: dict[int, list[int]] = defaultdict(list)
    for i, k in enumerate(equiv):
        kind_dict[int(k)].append(i)

    # element symbol for each kind, taken from its representative atom
    kind_symbols = {k: structure[k].specie.symbol for k in kind_dict}

    # how many distinct kinds share each element symbol
    symbol_counts: dict[str, int] = defaultdict(int)
    for sym in kind_symbols.values():
        symbol_counts[sym] += 1

    # assign labels: plain symbol if the element has only one kind,
    # else symbol + 1-based index, ordered by representative atom index
    labeled: dict[str, list[int]] = {}
    symbol_running_count: dict[str, int] = defaultdict(int)
    for k in sorted(kind_dict):
        sym = kind_symbols[k]
        if symbol_counts[sym] == 1:
            label = sym
        else:
            symbol_running_count[sym] += 1
            label = f"{sym}{symbol_running_count[sym]}"
        labeled[label] = kind_dict[k]

    return labeled

# %%
def calc_freqdistrib(
    st: Structure,
    magmoms: np.ndarray,
    muon_sites: np.ndarray,
    lorentz_factor: float = 0.0,
    k: Optional[Sequence[float]] = None,
    cont_field: float = 0.0,
    sphere_radius: int = 100
) -> np.ndarray:
    """
    Compute precession frequency per unit moment \nu/\mu (MHz / \mu_B).
    """
    result = multisite_pfields(
        structure=st.copy(),  # Fixed structure argument reference
        magmoms=magmoms,
        muon_positions=muon_sites,
        sphere_r=sphere_radius,
        k=k,
        cont_field=cont_field
    )

    # Total Dipolar + Lorentz field norm in Tesla per \mu_B
    B = result.dipolar_norm + result.lorentz_norm * lorentz_factor

    # Convert Tesla per \mu_B to MHz per \mu_B
    nu_per_mu = B * GAMMA_MU
    return nu_per_mu

# %%
def plot_distribution(ax, nu_per_mu):

    ax.hist(nu_per_mu, bins=80, density=True, alpha=0.5)

    xs = np.linspace(np.min(nu_per_mu), np.max(nu_per_mu), 1000)

    kde = gaussian_kde(nu_per_mu)

    ax.plot(xs, kde(xs),lw=2)

    # ax.set_xlabel(r'$\nu/\mu$ (MHz/$\mu_B$)')
    # ax.set_ylabel('Probability Density')

    return ax


def plot_posterior(ax, mu_grid, posterior):

    ax.plot(mu_grid, posterior, lw=2)

    # ax.set_xlabel(r'$\mu$ ($\mu_B$)')
    # ax.set_ylabel(r'$g(\mu|\nu)$')

    return ax

# %% [markdown]
# BAYESIAN ESTIMATOR

# %%
class BayesianMomentEstimator:
    """
    Fast Bayesian moment estimator for determining g(\mu | \{\nu_i\}).
    """
    def __init__(
        self, 
        nu_per_mu: np.ndarray, 
        num_kde_points: int = 1000
    ):
        self.nu_per_mu = np.asarray(nu_per_mu)
        self.kde = gaussian_kde(self.nu_per_mu)
        
        # Pre-evaluate KDE over grid for fast interpolation
        self.x_grid = np.linspace(
            np.min(self.nu_per_mu)*0.8, np.max(self.nu_per_mu)*1.2, num_kde_points
            )
        self.f_grid = self.kde(self.x_grid)

    def _f_val(
        self, 
        x: np.ndarray
    ) -> np.ndarray:
        """Fast 1D linear interpolation of f(\nu/\mu)."""
        return np.interp(x, self.x_grid, self.f_grid, left=0.0, right=0.0)

    def likelihood_single(
        self, 
        nu: float, 
        dnu: float, 
        mu_grid: np.ndarray
    ) -> np.ndarray:
        """
        Computes likelihood \int_{\nu-dnu}^{\nu+dnu} (1/\mu) f(\nu'/\mu) d\nu'
        """
        out = np.zeros_like(mu_grid, dtype=float)
        nu_points = np.linspace(nu - dnu, nu + dnu, 50)
        
        for i, mu in enumerate(mu_grid):
            if mu <= 0:
                continue
            integrand = (1.0 / mu) * self._f_val(nu_points / mu)
            out[i] = simpson(integrand, x=nu_points)
        return out

    def posterior(
        self, 
        frequencies: Sequence[float], 
        errors: Sequence[float], 
        mu_grid: np.ndarray
    ) -> np.ndarray:
        """
        Computes posterior distribution g(\mu | \{\nu_i\}).
        """
        post = np.ones_like(mu_grid, dtype=float)
        for nu, dnu in zip(frequencies, errors):
            post *= self.likelihood_single(nu, dnu, mu_grid)

        norm = simpson(post, x=mu_grid)
        if norm > 0:
            post /= norm
        return post

    @staticmethod
    def MAP(
        mu_grid: np.ndarray, 
        posterior: np.ndarray
    ) -> float:
        return float(mu_grid[np.argmax(posterior)])

    @staticmethod
    def credible_interval(
        mu_grid: np.ndarray, 
        posterior: np.ndarray, 
        confidence: float = 0.68
    ) -> Tuple[float, float]:
        cdf = np.cumsum(posterior)
        if cdf[-1] > 0:
            cdf /= cdf[-1]
        lower = (1.0 - confidence) / 2.0
        upper = 1.0 - lower
        lo = float(np.interp(lower, cdf, mu_grid))
        hi = float(np.interp(upper, cdf, mu_grid))
        return lo, hi

# %% [markdown]
# MCMC

# %%
class MomentMCMC:

    def __init__(
            self,
            nu_per_mu,
            frequencies,
            frequency_errors):

        if not HAVE_EMCEE:
            raise ImportError(
                "emcee must be installed."
            )

        self.freqs = np.asarray(frequencies)

        self.errors = np.asarray(frequency_errors)

        self.kde = gaussian_kde(nu_per_mu)

    def log_prior(self, theta):

        mu = theta[0]

        if 0.01 < mu < 2.0:
            return 0.0

        return -np.inf


    def log_likelihood(self, theta):

        mu = theta[0]

        logL = 0.0

        for nu, err in zip(self.freqs, self.errors):

            xmin = (nu - err) / mu
            xmax = (nu + err) / mu

            P = self.kde.integrate_box_1d(xmin, xmax)

            if P <= 0:
                return -np.inf

            logL += np.log(P / mu)

        return logL


    def log_probability(self, theta):

        lp = self.log_prior(theta)

        if not np.isfinite(lp):
            return -np.inf

        return lp + self.log_likelihood(theta)


    def run(self, nwalkers=32, nsteps=10000):

        ndim = 1

        p0 = np.random.uniform(0.05, 1.0, size=(nwalkers, ndim))

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability)

        sampler.run_mcmc(p0, nsteps, progress=True)

        return sampler