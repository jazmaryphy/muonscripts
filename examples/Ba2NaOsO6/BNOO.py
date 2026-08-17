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
from muonscripts.muesr_tools.sample_candidate_sites import sample_anion_muon_sites

from muonscripts.muesr_tools.bayesian import BayesianMomentEstimator

from muonscripts.plot_tools.muon import fancy_style_axes

GAMMA_MU = constants.MUON_GYROMAGNETIC_RATIO/constants.TWOPI
GAMMA_MU *=1e-6 # MHz/T 

# %%


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

    # Total Dipolar + Lorentz field in Tesla per \mu_B
    B = result.dipolar + result.lorentz * lorentz_factor
    B_norm = np.linalg.norm(B, axis=1) 

    # Convert Tesla -> MHz i.e (Tesla /mu_B -> MHz /mu_B)
    nu = B_norm * GAMMA_MU

    return nu

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

candidate_sites = sample_anion_muon_sites(
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
# result = multisite_pfields(
#     structure=p_st.copy(),
#     magmoms=magmoms_FM111,
#     muon_positions=candidate_sites,
#     sphere_r=100,
# )

# np.linalg.norm(result.lorentz, axis=1) 
# B = result.dipolar*0 + result.lorentz
# B, np.linalg.norm(B, axis=1) 

# %% [markdown]
# check how long to compute freqs
# 
# UNCOMMENT LINES

# %%
# import time

# for n in [10, 50, 100, 200]:

#     sites = candidate_sites[:n]

#     t0 = time.perf_counter()

#     result = multisite_pfields(
#         structure=p_st.copy(),
#         magmoms=magmoms_FM111,
#         muon_positions=sites,
#         sphere_r=100,
#     )

#     elapsed = time.perf_counter() - t0

#     print(
#         f"{n:6d} muons : "
#         f"{elapsed:8.2f} s  "
#         f"({elapsed / n:.4f} s/muon)"
#     )

# result.dipolar_norm
# result.lorentz_norm * GAMMA_MU

# %% [markdown]
# UNCOMMENT LINES BELOW:
# 
# Calculate nu / mu

# %%
sphere_radius=80
lorentz_factor=0

# %%
# print("\nCalculating FM [111]...")

# nu_per_mu_FM111 = calc_freqdistrib(
#     st=p_st.copy(),
#     magmoms=magmoms_FM111,
#     muon_sites=candidate_sites,
#     lorentz_factor=lorentz_factor,
#     sphere_radius=sphere_radius
# )

# %%
# print("\nCalculating AFM [111]...")

# nu_per_mu_AFM111 = calc_freqdistrib(
#     st=p_st.copy(),
#     magmoms=magmoms_AFM111,
#     muon_sites=candidate_sites,
#     lorentz_factor=lorentz_factor,
#     sphere_radius=sphere_radius
# )

# %%
# print("\nCalculating AFM [001]...")

# nu_per_mu_AFM001 = calc_freqdistrib(
#     st=p_st.copy(),
#     magmoms=magmoms_AFM001,
#     muon_sites=candidate_sites,
#     lorentz_factor=lorentz_factor,
#     sphere_radius=sphere_radius
# )

# %%
# # Frequency-distribution diagnostics

# frequency_distributions = {
#     "FM111": nu_per_mu_FM111,
#     "AFM111": nu_per_mu_AFM111,
#     "AFM001": nu_per_mu_AFM001,
# }


# for label, values in frequency_distributions.items():
#     print(f"\n{label}")
#     print(f"  min    = {values.min():.6f}")
#     print(f"  max    = {values.max():.6f}")
#     print(f"  mean   = {values.mean():.6f}")
#     print(f"  median = {np.median(values):.6f}")
#     print(f"  std    = {values.std():.6f}")

# %% [markdown]
# saved results:

# %%
# BNOO_freqs = {
#     "FM111": nu_per_mu_FM111,
#     "AFM111": nu_per_mu_AFM111,
#     "AFM001": nu_per_mu_AFM001,

#     "sample_sites": candidate_sites,

#     "structure_sites": p_stc,

#     # Store parameters so we know exactly how the data were generated.
#     "parameters": {
#         "n_samples": n_samples,
#         "O_distance": O_distance,
#         "min_cation_distances": min_cation_distances,
#         "sphere_radius": sphere_radius,
#         "seed": seed_no,
#         "lorentz_factor": lorentz_factor,
#     },
# }

# filename = "BNOO_freqs.pkl"
# filename = os.path.join(dpath, filename)
# with open(filename, "wb") as f:
#     pickle.dump(BNOO_freqs, f)

# print(
#     f"\nSaved frequency distributions to "
#     f"{filename}"
# )


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
estimators = {label: BayesianMomentEstimator(loaded_BNOO_freqs[label]) for label in styles}

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
fig, ax = plt.subplots(figsize=(6, 4))
results = {}
for label, (style, leglabel) in styles.items():
    estimator = BayesianMomentEstimator(loaded_BNOO_freqs[label])
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