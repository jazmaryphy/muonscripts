# %%
from __future__ import annotations

import copy
import pickle
import numpy as np
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

GAMMA_MU = constants.MUON_GYROMAGNETIC_RATIO/constants.TWOPI
GAMMA_MU *=1e-6 # MHz/T 

# %%


# %%
def calc_multisite_fields(
    st: Structure,
    magmoms: np.ndarray,
    muon_positions: np.ndarray,
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
        muon_positions=muon_positions,
        sphere_r=sphere_radius,
        k=k,
        cont_field=cont_field
    )

    # # Total Dipolar + Lorentz field in Tesla per \mu_B
    # B = result.dipolar + result.lorentz * lorentz_factor
    # B_norm = np.linalg.norm(B, axis=1) 

    # # Convert Tesla -> MHz i.e (Tesla /mu_B -> MHz /mu_B)
    # nu = B_norm * GAMMA_MU

    return result

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
    print("-" * 55)

    for i in atm_kinds["Os"]:
        m = moments[i]
        norm = np.linalg.norm(m)

        # Print 3D vector [mx, my, mz] with aligned signs
        if len(m) == 3:
            vec_str = f"[{m[0]:+8.4f}, {m[1]:+8.4f}, {m[2]:+8.4f}]"
        # Fallback for scalar/collinear moments
        else:
            vec_str = f"{m[0]:+8.4f}"

        print(f"Os {i:3d}: {vec_str}  |m| = {norm:7.4f}")

# %% [markdown]
# sample sites and compute freqs/muB for each magnetic configurations

# %%
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


# %%


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
# Test with FM

result = calc_multisite_fields(
    st=p_st.copy(),
    magmoms=magmoms_FM111,
    muon_positions=candidate_sites[0:2],
    sphere_radius=sphere_radius
)

# result contains field contributions in Tesla /mu_B
# result.total        is total field
# result.dipolar      is dipolar contributions
# result.lorentz      is lorentz contributions
# result.dipolar_tot  is dipolar+lorentz contributions
# result.contact      is contact contributions
# *_norm              are the magnitudes

Bdip = result.dipolar
Bdip_norm = np.linalg.norm(Bdip, axis=1)
# # Convert Tesla -> MHz i.e (Tesla /mu_B -> MHz /mu_B)
nu = Bdip_norm * GAMMA_MU
Bdip, Bdip_norm, nu

# %%
print("\nCalculating FM [111]...")

results_FM111 = calc_multisite_fields(
    st=p_st.copy(),
    magmoms=magmoms_FM111,
    muon_positions=candidate_sites,
    sphere_radius=sphere_radius
)

# %%
print("\nCalculating AFM [111]...")

results_AFM111 = calc_multisite_fields(
    st=p_st.copy(),
    magmoms=magmoms_AFM111,
    muon_positions=candidate_sites,
    sphere_radius=sphere_radius
)

# %%
print("\nCalculating AFM [001]...")

results_AFM001 = calc_multisite_fields(
    st=p_st.copy(),
    magmoms=magmoms_AFM001,
    muon_positions=candidate_sites,
    sphere_radius=sphere_radius
)

# %%
# Field diagnostics
field_contributions = {
    "FM111": results_FM111,
    "AFM111": results_AFM111,
    "AFM001": results_AFM001,
}

# Dictionary to store structured results per label
fields_diagnostics = {}

for label, res in field_contributions.items():
    Bdip = res.dipolar
    Bdip_norm = np.linalg.norm(Bdip, axis=1)
    
    # Convert Tesla -> MHz i.e (Tesla /mu_B -> MHz /mu_B)
    nu_values = Bdip_norm * GAMMA_MU

    # Store in nested dictionary
    fields_diagnostics[label] = {
        "results": res,
        "Bdip": Bdip,
        "Bdip_norm": Bdip_norm,
        "nu_values": nu_values,
        "stats": {
            "min": nu_values.min(),
            "max": nu_values.max(),
            "mean": nu_values.mean(),
            "median": np.median(nu_values),
            "std": nu_values.std(),
        },
    }

    # Print summary
    print(f"\n{label}")
    print("-" * 25)
    for stat_name, stat_val in fields_diagnostics[label]["stats"].items():
        print(f"  {stat_name:<6s} = {stat_val:.6f}")

# %% [markdown]
# saved results:

# %%
BNOO_freqs = {
    # "FM111":  results_FM111,
    # "AFM111": results_AFM111,
    # "AFM001": results_AFM001,

    "fields": field_contributions,
    "fields_diagnostics": fields_diagnostics,

    "sample_sites": candidate_sites,

    "host_lattice": p_st.copy(),

    "structure_sites": p_stc,

    # Store parameters so we know exactly how the data were generated.
    "parameters": {
        "n_samples": n_samples,
        "O_distance": O_distance,
        "min_cation_distances": min_cation_distances,
        "sphere_radius": sphere_radius,
        "seed": seed_no,
        "lorentz_factor": lorentz_factor,
        'gamma_mu': GAMMA_MU
    },
}

filename = "BNOO_freqs.pkl"
filename = os.path.join(dpath, filename)
with open(filename, "wb") as f:
    pickle.dump(BNOO_freqs, f)

print(
    f"\nSaved frequency distributions to "
    f"{filename}"
)

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