# %%
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.stats import gaussian_kde

# %%
try:
    import emcee
    HAVE_EMCEE = True
except ImportError:
    HAVE_EMCEE = False

# %%
class BayesianMomentEstimator:
    """
    g(mu | {nu_i}) is obtained from f(nu/mu), the pdf of precession
    frequency-per-unit-moment sampled at candidate muon sites.

    Single frequency:
        g(mu|nu) ~ (1/mu) f(nu/mu)

    Multiple frequencies:
        g(mu|{nu_i}) ~ prod_i  Integral_{nu_i-dnu_i}^{nu_i+dnu_i} f(nu'/mu) dnu'

    pdf_method
    ----------
    "histogram" (default)
        Piecewise-constant f(nu/mu) with an exact piecewise-linear CDF.
        Used for the fast, exact grid-based posterior in `posterior()`.
    "kde"
        Smooth Gaussian-KDE estimate of f(nu/mu). Not needed for the
        grid posterior, but useful as a continuous, differentiable
        log-density when sampling with emcee (see `log_prob`).
    """

    def __init__(
        self, 
        nu_per_mu, 
        mu_max=1.0, 
        bins=200,
        pdf_method="histogram", 
        num_kde_points=4000
    ):
        self.nu_per_mu = np.asarray(nu_per_mu, dtype=float)
        if self.nu_per_mu.ndim != 1:
            raise ValueError("nu_per_mu must be 1-D.")
        if len(self.nu_per_mu) < 2:
            raise ValueError("Need at least two samples.")
        if pdf_method not in {"histogram", "kde"}:
            raise ValueError("pdf_method must be 'histogram' or 'kde'.")

        self.mu_max = float(mu_max)
        self.pdf_method = pdf_method

        # histogram representation (always built; cheap, and used
        # by the fast grid-based posterior regardless of pdf_method)
        self.hist_pdf, self.bin_edges = np.histogram(
            self.nu_per_mu, bins=bins, density=True
        )
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.cdf = np.concatenate(
            ([0.0], np.cumsum(self.hist_pdf * np.diff(self.bin_edges)))
        )

        # optional smooth KDE representation
        if pdf_method == "kde":
            self.kde = gaussian_kde(self.nu_per_mu)
            self.x_grid = np.linspace(
                max(self.nu_per_mu.min() * 0.5, 0.0),
                self.nu_per_mu.max() * 1.5,
                num_kde_points,
            )
            self.f_grid = self.kde(self.x_grid)
            self.cdf_grid = np.zeros_like(self.x_grid)
            self.cdf_grid[1:] = np.cumsum(
                0.5 * (self.f_grid[1:] + self.f_grid[:-1])
                * np.diff(self.x_grid)
            )
            if self.cdf_grid[-1] > 0:
                self.cdf_grid /= self.cdf_grid[-1]


    # f(nu/mu) evaluation (for plotting the *input* distribution)
    def pdf_value(self, x):
        """Evaluate f(nu/mu) using whichever representation was built."""
        x = np.asarray(x, dtype=float)
        if self.pdf_method == "kde":
            return np.interp(x, self.x_grid, self.f_grid, left=0.0, right=0.0)

        result = np.zeros_like(x, dtype=float)
        inside = (x >= self.bin_edges[0]) & (x <= self.bin_edges[-1])
        if np.any(inside):
            idx = np.clip(
                np.searchsorted(self.bin_edges, x[inside], side="right") - 1,
                0, len(self.hist_pdf) - 1,
            )
            result[inside] = self.hist_pdf[idx]
        return result
    

    def plot_distribution(
        self, 
        ax=None, 
        label=None, 
        show_kde=True,
        color=None, 
        **kwargs
    ):
        """
        Plot the raw simulated f(nu/mu) distribution -- i.e. the
        histogram of precession frequency per unit moment obtained
        from the dipolar-field calculation at all candidate muon
        sites. This is the *input* to Bayes' theorem (Blundell et al., 
        Physica Procedia 2011), not the posterior.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        ax.hist(
            self.nu_per_mu, bins=self.bin_edges, density=True,
            histtype="stepfilled", alpha=0.35, color=color,
            label=label, **kwargs
        )

        if show_kde:
            kde = gaussian_kde(self.nu_per_mu)
            x = np.linspace(self.nu_per_mu.min(), self.nu_per_mu.max(), 500)
            ax.plot(x, kde(x), color=color, lw=1.8)

        # ax.set_xlabel(r"$\nu/\mu$  (MHz $\mu_B^{-1}$)")
        # ax.set_ylabel(r"$f(\nu/\mu)$")

        return ax


    # CDF of f(nu/mu) (grid-based posterior machinery)
    def _f_cdf(self, x):
        x = np.asarray(x, dtype=float)
        if self.pdf_method == "kde":
            return np.interp(x, self.x_grid, self.cdf_grid, left=0.0, right=1.0)

        result = np.zeros_like(x, dtype=float)
        below = x <= self.bin_edges[0]
        above = x >= self.bin_edges[-1]
        middle = ~(below | above)
        if np.any(middle):
            xm = x[middle]
            idx = np.clip(
                np.searchsorted(self.bin_edges, xm, side="right") - 1,
                0, len(self.hist_pdf) - 1,
            )
            result[middle] = (
                self.cdf[idx] + self.hist_pdf[idx] * (xm - self.bin_edges[idx])
            )
        result[above] = 1.0
        return result
    

    def likelihood_single(
        self, 
        nu, 
        dnu, 
        mu_grid
    ):
        mu_grid = np.asarray(mu_grid, dtype=float)
        L = np.zeros_like(mu_grid)
        valid = (mu_grid > 0) & (mu_grid <= self.mu_max)
        mu = mu_grid[valid]
        lower, upper = (nu - dnu) / mu, (nu + dnu) / mu
        L[valid] = mu * (self._f_cdf(upper) - self._f_cdf(lower))
        return L
    

    def posterior(
        self, 
        frequencies, 
        errors, 
        mu_grid
    ):
        """Exact grid-based posterior (recommended: fast, no convergence
        diagnostics needed for this 1-parameter problem)."""
        frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        errors = np.atleast_1d(np.asarray(errors, dtype=float))
        mu_grid = np.asarray(mu_grid, dtype=float)

        if frequencies.shape != errors.shape:
            raise ValueError("frequencies and errors must match in shape.")

        post = np.ones_like(mu_grid)
        for nu, dnu in zip(frequencies, errors):
            post *= self.likelihood_single(nu, dnu, mu_grid)

        norm = simpson(post, x=mu_grid)
        if norm <= 0:
            raise ValueError("Posterior could not be normalized -- check "
                              "that mu_grid/mu_max overlap the range "
                              "implied by nu_per_mu.")
        return post / norm
    

    @staticmethod
    def MAP(mu_grid, posterior):
        return float(mu_grid[np.argmax(posterior)])
    

    @staticmethod
    def credible_interval(mu_grid, posterior, confidence=0.68):
        cdf = np.zeros_like(posterior)
        cdf[1:] = np.cumsum(
            0.5 * (posterior[1:] + posterior[:-1]) * np.diff(mu_grid)
        )
        if cdf[-1] > 0:
            cdf /= cdf[-1]
        lo = np.interp((1 - confidence) / 2, cdf, mu_grid)
        hi = np.interp(1 - (1 - confidence) / 2, cdf, mu_grid)
        return float(lo), float(hi)
    


    # Optional: log-posterior + emcee sampler
    def log_prob(
        self, 
        mu, 
        frequencies, 
        errors
    ):
        """
        log g(mu | {nu_i}) up to a constant, uniform prior on
        (0, mu_max]. Used only by the optional emcee sampler --
        the grid-based `posterior()` above is the recommended,
        exact route for this 1-D problem.
        """
        mu = float(mu) if np.ndim(mu) == 0 else float(mu[0])
        if not (0.0 < mu <= self.mu_max):
            return -np.inf

        frequencies = np.atleast_1d(frequencies)
        errors = np.atleast_1d(errors)

        logp = 0.0
        for nu, dnu in zip(frequencies, errors):
            L = self.likelihood_single(nu, dnu, np.array([mu]))[0]
            if L <= 0:
                return -np.inf
            logp += np.log(L)
        return logp

    def posterior_mcmc(
        self, 
        frequencies, 
        errors, 
        n_walkers=32,
        n_steps=3000, 
        burn_in=500, 
        mu0=None, 
        seed=0
    ):
        """
        Cross-check the grid posterior with an emcee ensemble sampler.
        Not required for this 1-D problem (grid quadrature is exact
        and much cheaper) but provided for extension to multi-parameter
        models.
        """
        if not HAVE_EMCEE:
            raise ImportError(
                "emcee is not installed. `pip install emcee` to use "
                "posterior_mcmc(); the grid-based posterior() above "
                "does not require it."
            )

        if self.pdf_method != "kde":
            raise ValueError(
                "posterior_mcmc requires pdf_method='kde' (the "
                "histogram pdf is piecewise-constant with hard zero "
                "plateaus, which trips up emcee's stretch move -- "
                "instantiate this estimator with pdf_method='kde')."
            )

        rng = np.random.default_rng(seed)

        if mu0 is None:
            # rough starting guess: peak of the nu/mu distribution
            # mapped through the first observed frequency
            mu0 = float(np.atleast_1d(frequencies)[0] / self.x_grid[
                np.argmax(self.f_grid)
            ])
            mu0 = min(max(mu0, 1e-3), self.mu_max * 0.9)

        ndim = 1
        # small, well-scaled ball around mu0, on a relative (not
        # absolute) scale so it works whether mu0 is 0.05 or 0.5
        spread = max(1e-3, 0.05 * mu0)
        p0 = mu0 + spread * rng.standard_normal((n_walkers, ndim))
        p0 = np.clip(p0, 1e-4, self.mu_max - 1e-4)

        sampler = emcee.EnsembleSampler(
            n_walkers, ndim, lambda theta: self.log_prob(theta, frequencies, errors)
        )
        sampler.run_mcmc(p0, n_steps, progress=False)

        chain = sampler.get_chain(discard=burn_in, flat=True)[:, 0]
        
        return chain, sampler