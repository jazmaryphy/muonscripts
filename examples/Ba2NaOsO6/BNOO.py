# %%
from __future__ import annotations

import copy
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Sequence, Optional

from pymatgen.core import Structure

# %%
import sys
from pathlib import Path
ROOT = Path.cwd().parent.parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT)) 

from muonscripts.constants import constants

from muonscripts.muesr_tools.local_fields import multisite_pfields
from muonscripts.muesr_tools.utils import check_site_distances, get_atom_kinds
from muonscripts.muesr_tools.sample_muon_sites import anion_sites, target_sites

from muonscripts.muesr_tools.bayesian import BayesianMomentEstimator

from muonscripts.plot_tools.muon import fancy_style_axes

GAMMA_MU = constants.MUON_GYROMAGNETIC_RATIO/constants.TWOPI
GAMMA_MU *=1e-6 # MHz/T 

# %%


# %%
import os
dpath='./'

# %%
filename = 'BNOO.cif'
filename = os.path.join(dpath, filename)
p_st = Structure.from_file(filename)
p_st
print(p_st)

# %%
# Identify atom kinds

atm_kinds = get_atom_kinds(p_st)

print("\nAtom kinds:")
for kind, indices in atm_kinds.items():
    print(f"{kind}: {indices}")


print("\nOs indices:")
print(atm_kinds["Os"])

# %%
# Inspect Os positions
#
# IMPORTANT:
# This lets us verify the AFM assignment later.

print("\nOs fractional coordinates:")
 
for i in atm_kinds["Os"]:
    print(
        f"index = {i:3d}   "
        f"frac = {p_st.frac_coords[i]}"
    )

# %%
# Magnetic moment directions
#
# All moments have magnitude 1.
# Therefore the resulting frequency distribution is nu / mu.

dir_111 = np.array([1.0/np.sqrt(3), 1.0/np.sqrt(3), 1.0/np.sqrt(3)])

dir_001 = np.array([0.0, 0.0, 1.0])

print("\nMoment magnitudes:")
print("|dir_111| =", np.linalg.norm(dir_111))
print("|dir_001| =", np.linalg.norm(dir_001))

# %%
# AFM signs used in the present reconstruction
#
# We should verify these against the actual magnetic structure
# once the Os positions / paper figure are checked.

afm_signs = np.array(
    [-1.0, 1.0, 1.0, -1.0]
)

if len(atm_kinds["Os"]) != len(afm_signs):
    raise ValueError(
        "Number of Os sites does not match "
        "the number of AFM signs."
    )


# %%
# Construct magnetic structures

magmoms_FM111  = np.zeros((len(p_st), 3))
magmoms_AFM111 = np.zeros_like(magmoms_FM111)
magmoms_AFM001 = np.zeros_like(magmoms_FM111)


# FM [111]
magmoms_FM111[atm_kinds["Os"]] = dir_111

# AFM [111]
magmoms_AFM111[atm_kinds["Os"]] = afm_signs[:, None]* dir_111

# AFM [001]
magmoms_AFM001[atm_kinds["Os"]] = afm_signs[:, None]* dir_001

# %%
# Check magnetic moments

for label, moments in {
    "FM111": magmoms_FM111,
    "AFM111": magmoms_AFM111,
    "AFM001": magmoms_AFM001,
}.items():

    print(f"\n{label}")

    for i in atm_kinds["Os"]:
        print(
            f"Os {i:3d}: "
            f"{moments[i]}   "
            f"|m| = {np.linalg.norm(moments[i]):.6f}"
        )

# %% [markdown]
# sample sites and compute freqs/muB for each magnetic configurations

# %%
min_cation_distances={
    "Ba": 2.1,
    "Na": 1.63,
    "Os": 2.4,
}  # old

min_cation_distances={
    "Ba": 1.0,
    "Na": 1.0,
    "Os": 1.0,
} # new

seed_no = 42
n_samples = 20000
O_distance = (0.9, 1.10)

candidate_sites = anion_sites(
    p_st.copy(),
    n_samples=n_samples,

    anion_specie=("O",),
    anion_distance=O_distance,

    min_cation_distances=min_cation_distances,

    batch_factor=50,

    seed=seed_no,
)

print(
    f"\nGenerated "
    f"{len(candidate_sites)} "
    f"candidate muon sites."
)

# %%
# Add positions to original structure
p_stc = p_st.copy()
for pos in candidate_sites:
    p_stc.append('H', pos)

file='BNOO_randomsamples_muonsites.cif'
file = os.path.join(dpath, file)

p_stc = p_stc.copy()
p_stc = Structure(
    p_stc.lattice, 
    p_stc.species, 
    p_stc.frac_coords,
    site_properties=p_stc.site_properties,   
)

p_stc.to(file)
print(
    f"Saved candidate sites to "
    f"{file}"
)

# %%
distances = check_site_distances(
    p_st,
    candidate_sites,
)

# %%
# plt.hist(distances["O"], bins=50)
# distances["O"]

# %%


# %% [markdown]
# load results:

# %%
# Load the dictionary back into memory in binary read mode ('rb')
filename = "BNOO_freqs.pkl"
filename = os.path.join(dpath, filename)
with open(filename, "rb") as f:
    loaded_BNOO_freqs = pickle.load(f)

loaded_BNOO_freqs.keys()

# %%


# %%
def get_nu_values(data, label, field_type="dipolar"):
    """
    Calculates muon precession frequencies (MHz) for a given magnetic field contribution.

    Parameters
    ----------
    data : dict
        Dictionary containing 'fields' and 'parameters'.
    label : str
        Magnetic state label (e.g., "FM111", "AFM001").
    field_type : str, optional
        Field component to extract: 'dipolar', 'total', 'lorentz', 
        'dipolar_tot', or 'contact'. Default is 'dipolar'.

    Returns
    -------
    nu_values : np.ndarray
        Magnitudes converted to frequency in MHz.
    """
    valid_fields = {"dipolar", "total", "lorentz", "dipolar_tot", "contact"}
    if field_type not in valid_fields:
        raise ValueError(
            f"Invalid field_type '{field_type}'. Expected one of {valid_fields}"
        )

    result = data["fields"][label]
    
    # Dynamically access field array (e.g., result.dipolar or result.total)
    field = getattr(result, field_type)
    
    # Calculate magnitude along spatial axes
    field_norm = np.linalg.norm(field, axis=1)
    
    # Convert Tesla -> MHz
    nu_values = field_norm * data["parameters"]["gamma_mu"]
    
    return nu_values

# Get default dipolar frequencies
nu_values = get_nu_values(loaded_BNOO_freqs, "FM111")

# Get total field frequencies
nu_values = get_nu_values(loaded_BNOO_freqs, "FM111", field_type="total")

# Get dipolar + lorentz frequencies
nu_values = get_nu_values(loaded_BNOO_freqs, "AFM001", field_type="dipolar_tot")

# %%


# %%
mu_grid = np.linspace(1e-4, 1.0, 4000)

# A. J. Steele et al., Phys. Rev. B 84, 144416 (2011): nu(0)=3.9(1) MHz,
# eta = nu2/nu1 = 0.4(5) -> nu2 = eta*nu1 = 1.56 MHz
expt_freqs = [3.9, 1.56]
expt_errs = [0.1, 0.2]

styles = {
    "FM111": ("r-", "FM [111]"),
    "AFM111": ("b--", "AFM [111]"),
    "AFM001": ("c-.", "AFM [001]"),
}

fontsize=16

# %%
field_type='dipolar'
# field_type='total'

estimators = {
    label: BayesianMomentEstimator(
        get_nu_values(loaded_BNOO_freqs, label, field_type=field_type)
        ) for label in styles
    }

fig_in, ax_in = plt.subplots(figsize=(6, 4))
for label, (style, leglabel) in styles.items():
    color = style[0]
    estimators[label].plot_distribution(ax=ax_in, label=leglabel, color=color)


fancy_style_axes(ax_in)
ax_in.set_xlim(0, 80)
ax_in.legend(frameon=False)
ax_in.legend(
    loc="best", 
    # bbox_to_anchor=(0.5, 1.001), 
    fontsize=fontsize,
    ncol=1, frameon=False, handlelength=1.9, columnspacing=1.2,
)
ax_in.set_title(r"Simulated $f(\nu/\mu)$ at candidate muon sites")
ax_in.set_xlabel(r"$\nu/\mu$  (MHz $\mu_B^{-1}$)", fontsize=fontsize)
ax_in.set_ylabel(r"$f(\nu/\mu)$", fontsize=fontsize)

plt.tight_layout()
fig_name = "field_distributions.png"
fig_name = os.path.join(dpath, fig_name)
plt.savefig(fig_name)
plt.show()

# %%
field_type='dipolar'
# field_type='total'

fig, ax = plt.subplots(figsize=(6, 4))
results = {}
for label, (style, leglabel) in styles.items():
    estimator = BayesianMomentEstimator(get_nu_values(loaded_BNOO_freqs, label, field_type=field_type))
    post = estimator.posterior(expt_freqs, expt_errs, mu_grid)
    map_val = estimator.MAP(mu_grid, post)
    ci = estimator.credible_interval(mu_grid, post)
    results[label] = (map_val, ci)
    ax.plot(mu_grid, post, style, lw=2, label=leglabel)
    print(f"{label:8s}  MAP = {map_val:.3f} muB   68% CI = "
          f"[{ci[0]:.3f}, {ci[1]:.3f}] muB")

fancy_style_axes(ax)
ax.set_xlabel(r"$\mu$ ($\mu_B$)", fontsize=fontsize)
ax.set_ylabel(r"$g(\mu|\{\nu_i\})$ ($\mu_B^{-1}$)", fontsize=fontsize)
# ax.set_ylim(0.0, 10)
ax.set_xlim(0.0, 0.8)
ax.legend(frameon=False)
ax.legend(
    loc="best", 
    # bbox_to_anchor=(0.5, 1.001), 
    fontsize=fontsize,
    ncol=1, frameon=False, handlelength=1.9, columnspacing=1.2,
)

plt.tight_layout()
fig_name = "PDF_moments.png"
fig_name = os.path.join(dpath, fig_name)
plt.savefig(fig_name)
plt.show()

# %%


# %%
try:
    import emcee
    HAVE_EMCEE = True
except ImportError:
    HAVE_EMCEE = False

# %%
if HAVE_EMCEE:
    print("\nemcee available -- running MCMC cross-check for FM111...")
    # emcee needs the smooth KDE representation of f(nu/mu); the
    # histogram version has hard zero plateaus that break the sampler.
    est_fm = BayesianMomentEstimator(loaded_BNOO_freqs["FM111"], pdf_method="kde")
    chain, sampler = est_fm.posterior_mcmc(
        expt_freqs, expt_errs, mu0=results["FM111"][0], seed=1
    )

    print(f"  emcee : mean = {chain.mean():.3f} muB, "
          f"std = {chain.std():.3f} muB, "
          f"median = {np.median(chain):.3f} muB, "
          f"mean acceptance fraction = "
          f"{np.mean(sampler.acceptance_fraction):.2f}")
    print(f"  grid  : MAP  = {results['FM111'][0]:.3f} muB, "
          f"68% CI = [{results['FM111'][1][0]:.3f}, "
          f"{results['FM111'][1][1]:.3f}] muB")

    fig_mc, ax_mc = plt.subplots(figsize=(6, 4))
    ax_mc.plot(mu_grid, estimators["FM111"].posterior(expt_freqs, expt_errs, mu_grid),
               "r-", lw=2, label="grid (exact)")
    ax_mc.hist(chain, bins=80, density=True, histtype="step", lw=2,
               color="k", label="emcee")
    ax_mc.set_xlabel(r"$\mu$ ($\mu_B$)")
    ax_mc.set_ylabel(r"$g(\mu|\{\nu_i\})$ ($\mu_B^{-1}$)")
    ax_mc.set_xlim(0.0, 0.8)
    ax_mc.legend(frameon=False)
    plt.tight_layout()
else:
    print("\nemcee not installed -- skipping MCMC cross-check "
          "(not needed: the grid posterior above is already exact "
          "for this 1-D problem).")

plt.show()

# %%