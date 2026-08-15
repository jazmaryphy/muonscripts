# %%
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union, Sequence

import copy
import numpy as np
from ase import Atoms
from ase.io import write
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.electronic_structure.core import Magmom
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer

# %%
def get_collinear_mag_kindname(
    p_st: Structure, 
    magmom: Sequence[Any], 
    half: bool = True
) -> Tuple[Structure, Dict[str, float]]:
    """
    Provides kind names for magnetically distinct species for spin-polarized
    calculations with Quantum ESPRESSO / AiiDA using pymatgen structures.

    Parameters
    ----------
    p_st : Structure
        Structure to be analyzed.
    magmom : Sequence[Any]
        Corresponding magnetic moments for the structure sites (floats, lists, or Magmom objects).
    half : bool, optional
        If True, normalizes non-zero magnetic moments to +/-0.5 (default is True).

    Returns
    -------
    Tuple[Structure, Dict[str, float]]
        - pymatgen Structure with 'kind_name' site properties attached.
        - Dictionary mapping kind names to starting magnetization values.
    """
    assert len(p_st) == len(magmom), "Structure site count must match magnetic moments length."

    # 1. Project magnetic moments onto collinear z-axis
    coll_m, _ = Magmom.get_consistent_set_and_saxis(magmom)
    st_work: Structure = p_st.copy()
    for i, m in enumerate(coll_m):
        mtm = Magmom(m).get_00t_magmom_with_xyz_saxis()
        st_work[i].properties["magmom"] = Magmom([0.0, 0.0, mtm[2]])

    # 2. Analyze collinear magnetic structure
    p_st2 = CollinearMagneticStructureAnalyzer(st_work, make_primitive=False)
    
    # remove this check later
    try:
        assert p_st2.is_magnetic and p_st2.is_collinear
    except AssertionError:
        print("Warning: Structure is not magnetic or collinear.")

    st_spin: Structure = p_st2.get_structure_with_spin()

    # 3. Process site kinds and starting magnetizations
    kind_values: List[str] = []
    magnetic_elements_kinds: Dict[str, Dict[str, float]] = {}
    start_mag_dict: Dict[str, float] = {}

    for site in st_spin:
        spin: float = float(getattr(site.specie, "spin", 0.0))
        element: str = site.specie.element.symbol

        if not np.isclose(spin, 0.0):
            kinds_for_element = magnetic_elements_kinds.setdefault(element, {})

            # Match against existing spins for this element
            for kind, kind_spin in kinds_for_element.items():
                if np.isclose(spin, kind_spin):
                    kind_name = kind
                    break
            else:
                # Create a new kind name (e.g., 'Fe1', 'Fe2')
                kind_name = f"{element}{len(kinds_for_element) + 1}"
                kinds_for_element[kind_name] = spin
                
                # Compute starting magnetization
                mag_val = np.sign(spin) * 0.5 if half else spin
                start_mag_dict[kind_name] = round(float(mag_val), 2)

            kind_values.append(kind_name)
        else:
            # Assign standard element symbol for non-magnetic sites
            kind_values.append(element)

    return p_st.copy(site_properties={"kind_name": kind_values}), start_mag_dict

# %%
def make_collinear_getmag_kind(
    p_st: Structure, 
    magmom: Sequence[Union[float, List[float], Magmom]], 
    half: bool = True
) -> Dict[str, Any]:
    """
    Calls 'get_collinear_mag_kindname' to make magnetic moments collinear
    and assign site kind names relevant for spin-polarized calculations.

    Parameters
    ----------
    p_st : Structure
        Input pymatgen Structure instance.
    magmom : Sequence[Union[float, List[float], Magmom]]
        Sequence of magnetic moments (floats, 3D vectors, or Magmom objects).
    half : bool, optional
        If True, normalizes non-zero starting magnetizations to +/-0.5 (default is True).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "struct_magkind": pymatgen Structure with 'kind_name' site properties.
        - "start_mag_dict": Dictionary of starting magnetizations per kind.
    """
    # Convert list/array elements to Magmom objects if not already converted
    magmoms = [m if isinstance(m, Magmom) else Magmom(m) for m in magmom]

    # Process collinear structure and starting magnetization dictionary
    st_k, st_m_dict = get_collinear_mag_kindname(p_st, magmoms, half=half)

    return {
        "struct_magkind": st_k,
        "start_mag_dict": st_m_dict
    }

# %%
def reassign_kinds(
    structure: Structure, 
    kind_list: Sequence[str]
) -> Structure:
    """
    Reassigns kind names on a pymatgen Structure instance using a list of kind names.

    Parameters
    ----------
    structure : Structure
        The input pymatgen Structure object.
    kind_list : Sequence[str]
        A list or sequence of kind names corresponding to each site or unique species.

    Returns
    -------
    Structure
        A new pymatgen Structure object with updated 'kind_name' site properties.
    """
    assert len(structure) == len(kind_list), (
        "Length of kind_list must match the number of sites in the structure."
    )

    new_structure = structure.copy()

    # Update site properties with the new kind names
    new_structure.add_site_property("kind_name", list(kind_list))

    return new_structure


def create_starting_mag_kindlist(
    structure: Structure, 
    start_mg_dict: Dict[str]
) -> List:
    kind_names = structure.site_properties["kind_name"]
    per_atom_magmom = [start_mg_dict.get(k, 0.0) for k in kind_names]
    return per_atom_magmom