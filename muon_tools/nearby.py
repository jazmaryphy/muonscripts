# %%
import string
from collections import defaultdict
from typing import Dict, List, Optional, Union

import re
import numpy as np
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# %%
def int_to_roman(n: int) -> str:
    """Convert a 1-based integer to Roman numerals dynamically."""
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syb = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    roman_num = ""
    i = 0
    while n > 0:
        for _ in range(n // val[i]):
            roman_num += syb[i]
            n -= val[i]
        i += 1
    return roman_num


def index_to_letter(idx: int) -> str:
    """Convert a 0-based group index to uppercase letters (0 -> 'A', 25 -> 'Z', 26 -> 'AA')."""
    result = ""
    while idx >= 0:
        result = string.ascii_uppercase[idx % 26] + result
        idx = (idx // 26) - 1
    return result

# %%
def nearby_atom(
    label: str,
    fcoords: Union[List, np.ndarray],
    host_lattice: Union[Structure, Atoms],
    energies: Optional[Union[List, np.ndarray]] = None,
    symprec: float = 1e-3,
) -> Dict[str, List[Dict]]:
    """
    Finds nearest host target atoms and groups candidate muon fcoords by
    their symmetry kind label ('O1', 'O2', etc.).
    """
    fcoords = np.atleast_2d(fcoords)

    if energies is None:
        # energies = np.zeros(len(fcoords))
        energies_list = [None] * len(fcoords)
    else:
        # energies = np.asarray(energies, dtype=float)
        energies_list = np.asarray(energies, dtype=float).tolist()

    if isinstance(host_lattice, Atoms):
        from pymatgen.io.ase import AseAtomsAdaptor
        p_st = AseAtomsAdaptor.get_structure(host_lattice)
    else:
        p_st = host_lattice

    analyzer = SpacegroupAnalyzer(p_st, symprec=symprec)
    equiv = analyzer.get_symmetry_dataset().equivalent_atoms

    kind_repr = defaultdict(list)
    for idx, eq_id in enumerate(equiv):
        if p_st[idx].specie.symbol == label:
            kind_repr[eq_id].append(idx)

    if not kind_repr:
        raise ValueError(f"Species '{label}' not found in the input structure.")

    # FIX: Use 'kind_name' instead of overwriting 'label'
    kind_labels = {}
    for rank, eq_id in enumerate(sorted(kind_repr.keys()), start=1):
        kind_name = f"{label}{rank}" if len(kind_repr) > 1 else label
        kind_labels[eq_id] = kind_name

    atom_index_to_kind = {}
    for eq_id, indices in kind_repr.items():
        for idx in indices:
            atom_index_to_kind[idx] = eq_id

    # FIX: 'label' is still clean ('O') here
    label_idx = [idx for idx in range(len(p_st)) if p_st[idx].specie.symbol == label]
    label_fcoords = p_st.frac_coords[label_idx]

    raw_groups = defaultdict(list)

    for i, frac_pos in enumerate(fcoords):
        all_dists = p_st.lattice.get_all_distances(frac_pos, label_fcoords)[0]
        min_idx_in_subset = np.argmin(all_dists)
        
        nearest_atom_idx = label_idx[min_idx_in_subset]
        nearest_distance = all_dists[min_idx_in_subset]
        kind_id = atom_index_to_kind[nearest_atom_idx]

        host_atom_symbol = p_st[nearest_atom_idx].specie.symbol
        host_atom_label = f"{host_atom_symbol}_{nearest_atom_idx}"
        host_kind_label = kind_labels[kind_id]

        # Key by string label ('O1', 'O2', etc.)
        raw_groups[host_kind_label].append({
            "frac_coord": frac_pos % 1.0,
            # "energy": energies[i],
            "energy": energies_list[i],
            "nearest_atom_idx": nearest_atom_idx,
            "host_atom_label": host_atom_label,
            "host_kind_label": host_kind_label,
            "d_to_host": nearest_distance
        })

    # for key in raw_groups:
    #     raw_groups[key].sort(key=lambda x: x["energy"])

    # Sort safely: if energy is None, default to 0 so sort doesn't crash
    for key in raw_groups:
        raw_groups[key].sort(
            key=lambda x: x["energy"] if x["energy"] is not None else 0
        )

    return dict(raw_groups)

# %%
def _group_fmt(
    label: str, 
    nearby: Dict[str, List[Dict]]
) -> Dict[str, List[Dict]]:
    """
    Formats raw groups by sorting groups according to their minimum energy candidate
    (or by natural group key ordering if energy is None), re-assigning group keys 
    (μ-O1, μ-O2, ...) and site labels (A_{I}, B_{I}, ...).
    """
    
    def get_group_sort_key(k: str):
        first_energy = nearby[k][0].get("energy")
        if first_energy is not None:
            # Sort primarily by energy
            return (0, first_energy)
        else:
            # Fallback: Extract numeric suffix for natural sort ('O1' -> 1, 'O10' -> 10)
            digits = re.findall(r'\d+', k)
            key_number = int(digits[0]) if digits else 0
            return (1, key_number, k)

    # 1. Sort group keys ('O1', 'O2', ...) safely and naturally
    sorted_nearby_keys = sorted(nearby.keys(), key=get_group_sort_key)

    output = {}

    # 2. Assign μ-O1, μ-O2... based on ordering
    for group_rank, raw_key in enumerate(sorted_nearby_keys, start=1):
        group_key = f"μ-{label}{group_rank}" if len(sorted_nearby_keys) > 1 else f"μ-{label}"
        letter = index_to_letter(group_rank - 1)
        
        candidates = nearby[raw_key]
        fmt_candidates = []
        
        for site_rank, item in enumerate(candidates, start=1):
            subscript = int_to_roman(site_rank)
            item["site_label"] = f"{letter}_{{{subscript}}}"
            fmt_candidates.append(item)
            
        output[group_key] = fmt_candidates

    return output

# %%
def get_group_nearby_atom(
    label: str,
    fcoords: Union[List, np.ndarray],
    host_lattice: Union[Structure, Atoms],
    energies: Optional[Union[List, np.ndarray]] = None,
    symprec: float = 1e-3,
) -> Dict[str, List[Dict]]:
    """Main wrapper function combining search and formatting."""

    nearby_ = nearby_atom(
        label=label,
        fcoords=fcoords,
        host_lattice=host_lattice,
        energies=energies,
        symprec=symprec,
    )

    return  _group_fmt(label=label, nearby=nearby_)

# %%
def print_muon_table(
    group_nearby: Dict[str, List[Dict]],
    energy_unit: str = "meV"
) -> None:
    """
    Helper function to print candidate muon sites in publication format with:
    - All columns centered.
    - Site group labels (μ-O1, μ-O2, etc.) vertically centered within their group rows.
    - Dynamic energy column handling (omits or displays 'N/A' cleanly if energy is None).
    """
    # Check if any site in the dictionary contains valid energy data
    has_energies = any(
        site.get('energy') is not None 
        for sites in group_nearby.values() 
        for site in sites
    )

    # Header title dynamically including the energy unit
    energy_header = f"ΔE ({energy_unit})"

    # Adjust layout width and header based on energy presence
    if has_energies:
        header = f"\n{'Sites':^10} | {'Label':^10} | {'Fractional coord.':^24} | {energy_header:^12}"
        divider = "=" * 63
        row_divider = "-" * 63
    else:
        header = f"\n{'Sites':^10} | {'Label':^10} | {'Fractional coord.':^24}"
        divider = "=" * 48
        row_divider = "-" * 48

    print(header)
    print(divider)

    for group_key, sites in group_nearby.items():
        num_sites = len(sites)
        # Calculate the middle row index for vertical centering
        middle_idx = num_sites // 2 if num_sites % 2 != 0 else (num_sites // 2) - 1

        for i, site in enumerate(sites):
            # Print group_key only on the middle row of the group
            site_group_str = group_key if i == middle_idx else ""
            
            pos = site['frac_coord']
            pos_str = f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
            
            if has_energies:
                e_val = site.get('energy')
                energy_str = f"{e_val:.0f}" if e_val is not None else "N/A"
                print(f"{site_group_str:^10} | {site['site_label']:^10} | {pos_str:^24} | {energy_str:^12}")
            else:
                print(f"{site_group_str:^10} | {site['site_label']:^10} | {pos_str:^24}")
            
        print(row_divider)