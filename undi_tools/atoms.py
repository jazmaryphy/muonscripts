# %%
import logging
import numpy as np
from ase.io import read
from ase.atom import Atom
from copy import deepcopy
from fractions import Fraction
from ase.neighborlist import neighbor_list

# %%

from constants import constants
from io_tools.read_ase import read_from_file
from undi_tools.isotopes import Element as element

import sys
sys.set_int_max_str_digits(100000)

# %%
def splitIsotope(s):
    """This function separates the isotope number and the element name.

    Parameters
    ----------
    s : str
        input string, e.g. "63Cu" becomes ('63', 'Cu')

    Returns
    -------
    tuple
        (isotope number, element name)
    """
    return (''.join(filter(str.isdigit, s)) or None,
            ''.join(filter(str.isalpha, s)) or None)


def _validate_optional_dict(name, obj):
    """Return an empty dict for None, otherwise validate that *value* is a dict."""
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise TypeError(
            f"{name} must be None or dict, got {type(obj).__name__}"
        )
    return obj

# %%
def build_undi_neighbors(
    atoms,
    muon_position,
    cutoffs,
    inf_cutoff=10.0,
    efg_tensors=None,
    efg_factor=1.0,
    gamma_overrides=None,
    isotope_overrides=None,
    quadrupole_moments_overrides=None,
    verbose_neighbors=True
):
    """
    Build an UNDI neighbour list from an ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        Crystal structure.
    muon_position : array-like, shape (3,)
        Fractional crystal coordinates of the muon.
    cutoffs : dict
        Species-dependent cutoff distances in Angstrom.
    inf_cutoff : float, optional
        Large cutoff supplied to ASE when constructing the neighbour
        list. The actual selection is subsequently performed using
        ``cutoffs``.
    efg_tensors : sequence of (3, 3) array_like or None, optional
        Precomputed EFG tensors corresponding to ``atoms``.
    efg_factor : float, optional
        Scale factor applied to every supplied EFG tensor.
    gamma_overrides : dict or None, optional
        Mapping from isotope labels or atomic symbols to nuclear
        gyromagnetic ratios.
    isotope_overrides : dict or None, optional
        Mapping from atomic symbols to isotope labels.
    quadrupole_moments_overrides : dict or None, optional
        Mapping from isotope labels or atomic symbols to electric
        quadrupole moments.
    verbose_neighbors: bool, optional
        If True, print neighbor informations. Default. True

    Returns
    -------
    list of dict
        UNDI neighbour specification.
    """

    # Never modify the user's Atoms object
    atoms = atoms.copy()

    # Validate optional parameter inputs expected as dictionary or None
    isotope_overrides = _validate_optional_dict(
        "isotope_overrides",
        isotope_overrides,
    )

    gamma_overrides = _validate_optional_dict(
        "gamma_overrides",
        gamma_overrides,
    )

    quadrupole_moments_overrides = _validate_optional_dict(
        "quadrupole_moments_overrides",
        quadrupole_moments_overrides,
    )

    # # Terrible idea, make sure user SUPPLIED
    # cutoffs = _validate_optional_dict(
    #     "cutoffs",
    #     cutoffs,
    # )

    # Validate supplied EFG tensors once
    if efg_tensors is not None:
        if len(efg_tensors) != len(atoms):
            raise ValueError(
                f"Expected {len(atoms)} EFG tensors, "
                f"got {len(efg_tensors)}."
            )

    # append muon
    atoms.extend(Atom("H", [0, 0, 0]))                   # Use "H" as muon label, maybe an issue H containing structure: TODO 
    # update muon position
    scaled_positions = atoms.get_scaled_positions()
    scaled_positions[-1] = muon_position
    atoms.set_scaled_positions(scaled_positions)

    muon_index = len(atoms) - 1                          # muon index, the last atoms

    ai, aj, D = neighbor_list("ijD", atoms, inf_cutoff)  # very large cutoff to get all possible interactions.
                                                         # Actual selection is done below
    neighbors = []

    for i in range(len(D)):
        if not (ai[i] == muon_index):
            continue
        
        symbol = atoms[aj[i]].symbol

        d = D[i]
        distance = np.linalg.norm(d)
        cutoff = cutoffs.get(symbol, 0)
        if distance > cutoff:
            continue
        
        # if symbol in cutoffs.keys():  # #--> OLD
        if symbol in cutoffs:
            pos = D[i] * constants.ANGSTROM
            pos_str = f"[{d[0]: 8.4f}, {d[1]: 8.4f}, {d[2]: 8.4f}]"
            if verbose_neighbors:
                print(
                    f"Adding atom #{aj[i]:<4d}"
                    f"{symbol:<4s}"
                    f" with position (\u00c5): {pos_str}"
                    f" and distance: {distance:.4f} \u00c5"
                )

            # insert "pre-computed" EFG
            efg_tensor = None
            if efg_tensors is not None:

                efg_tensor = np.asarray(efg_tensors[aj[i]]).copy()

                if efg_tensor.shape != (3, 3):
                    raise ValueError(
                        f"EFG tensor for atom #{aj[i]} has shape "
                        f"{efg_tensor.shape}, expected (3,3)."
                    )

                efg_tensor *= efg_factor

                if verbose_neighbors:
                    print(
                        f"Using supplied EFG tensor for atom #{aj[i]} "
                        f"(scaled by {efg_factor:g})"
                    )

            # Override isotope label if requested/necessary
            original_symbol = symbol
            symbol = isotope_overrides.get(symbol, symbol)

            if symbol != original_symbol:
                print(f"[override] isotope: {original_symbol} -> {symbol}")

            # Override/insert nuclear gyromagnetic moments for atom 'symbol'
            gamma = gamma_overrides.get(symbol)
            if gamma is not None:
                print(
                    f"[override/insert] "
                    f"nuclear gyromagnetic ratio for {symbol}: "
                    f"{gamma: .6e} rad/(sT)"
                )

            # Override/insert electric quadrupole moments for atom 'symbol'
            quadrupole_moment  = quadrupole_moments_overrides.get(symbol)
            if quadrupole_moment  is not None:
                print(
                    f"[override/insert] "
                    f"quadrupole moment for {symbol}: "
                    f"{quadrupole_moment: .6e} m^2"
                )

            neighbor = {
                "Position": pos,
                "Label": symbol,
            }

            if gamma is not None:
                neighbor["Gamma"] = gamma

            if efg_tensor is not None:
                neighbor["EFGTensor"] = efg_tensor

            if quadrupole_moment is not None:
                neighbor["ElectricQuadrupoleMoment"] = quadrupole_moment

            neighbors.append(neighbor)


    # INSERT muon data "Label" "mu" at first index in "neighbors" i.e zeroth
    # The muon position is at origin i.e [0, 0, 0]
    # since all atomic positions above are relative to the "muon_position"
    neighbors.insert(
        0,
        {
            "Position": np.zeros(3),
            "Label": "mu",
            "Spin": 0.5,
            "Gamma": constants.MUON_GYROMAGNETIC_RATIO,
        },
    )

    return neighbors


def build_undi_neighbors_from_file(
    filename,
    muon_position,
    cutoffs,
    inf_cutoff=10.0,
    efg_tensors=None,
    efg_factor=1.0,
    gamma_overrides=None,
    isotope_overrides=None,
    quadrupole_moments_overrides=None,
    verbose_neighbors=True,
    **read_kwargs,
):
    atoms = read_from_file(filename, **read_kwargs)

    return build_undi_neighbors(
        atoms=atoms,
        muon_position=muon_position,
        cutoffs=cutoffs,
        inf_cutoff=inf_cutoff,
        efg_tensors=efg_tensors,
        efg_factor=efg_factor,
        gamma_overrides=gamma_overrides,
        isotope_overrides=isotope_overrides,
        quadrupole_moments_overrides=quadrupole_moments_overrides,
        verbose_neighbors=verbose_neighbors
    )

# %%
def complete_undi_neighbors(neighbors, logger=None, log_level=""):
    """
    Populate UNDI format atomic information.

    Parameters
    ----------
    neighbors: list of dict
        Example:
        [
          {"Position": ..., "Label": "mu"},
          {"Position": ..., "Label": "F"}
        ]
    logger : logging.Logger, optional
    log_level : str, optional

    Returns
    -------
    neighbors : list
        Modified neighbors list.
    """
    # Make own copy to avoid overwriting of internal elements
    neighbors = deepcopy(neighbors)

    logger = logger or logging.getLogger(__name__)

    if log_level:
        try:
            logger.setLevel(getattr(logging, log_level.upper()))
        except:
            logger.warning("Invalid logging level")

    for i, atom in enumerate(neighbors):
        spin  = atom.get('Spin', None)
        label = atom.get('Label', None)
        pos   = atom.get('Position', None)
        gamma = atom.get('Gamma', None)
        quadrupole_moment = atom.get('ElectricQuadrupoleMoment', None)

        # validation
        if pos is None:
            raise ValueError(f"Position needed for atom {i}")

        # assign values 
        # 1. For Muon
        if label == "mu":
            if spin:
                if spin != 0.5:
                    logger.warning("Warning, muon spin already set differs from 0.5!!")
            else:
                neighbors[i]["Spin"] = 0.5

            neighbors[i]['Gamma'] = constants.GAMMAS[label]

        # 1. For Nuclear isotopes
        else:
            A, Symbol = splitIsotope(label)
            e = element(Symbol)
            # explicit isotope
            if A:
                A = int(A)
                for isotope in e.isotopes:
                    if isotope.mass_number == A:
                        break
                else:
                    raise ValueError('Isotope {} for atom {} not found.'.format(A, Symbol))

            # choose most abundant magnetic isotope
            else:

                max_ab = -1.
                l = -1
                for is_n, isotope in enumerate(e.isotopes):
                    if isotope.abundance is None:
                        continue
                    if isotope.abundance > max_ab:
                        l = is_n
                        max_ab = isotope.abundance
                # Select isotope with highest abundance
                isotope = e.isotopes[l]

                level = logging.WARNING if max_ab < 0.99 else logging.INFO
                logger.log(level, 'Using most abundand isotope for {}, i.e. {}{}, {} abundance'.format(label, isotope.mass_number, e.symbol, max_ab))

            # Spin
            # check if overriding spin
            mendeelev_spin = float(Fraction(isotope.spin))
            if spin:
                if spin != mendeelev_spin:
                    logger.warning("Warning, overriding spin for {}".format(label))

            else:
                if mendeelev_spin == 0:
                    raise RuntimeError("Isotope with " + str(isotope) + " with spin 0 not allowed. Specify a different isotope or remove this nucleus.")
                neighbors[i]['Spin'] = mendeelev_spin

            # Gamma
            # check if overriding gamma
            if gamma:
                logger.warning("Warning, overriding gamma for {}".format(label))

            else:
                if isotope.g_factor:
                    neighbors[i]['Gamma'] = isotope.g_factor * 7.622593285e6 * 2. * np.pi  #  \mu_N /h, is 7.622593285(47) MHz/T that in turn is equal to  γ_n / (2 π g_n)
                else:
                    logger.error("g_factor missing in mendeleev library, you need to specify it by hand")
                    raise RuntimeError("Missing gammma.")

            # Quadrupole
            if quadrupole_moment is None:
                neighbors[i]['ElectricQuadrupoleMoment'] = isotope.quadrupole_moment * 1e-28 # m^2
            else:
                neighbors[i]['ElectricQuadrupoleMoment'] = quadrupole_moment
                logger.warning("Warning, overriding quadrupole moment for {}".format(label))
    
    return neighbors


def neighbors_hilbert_dimension(neighbors):
    """
    Compute the Hilbert-space dimension.

    Parameters
    ----------
    neighbors : list of dict

    Returns
    -------
    int
    """
    Hdim = 1

    for atom in neighbors:
        spin = atom.get("Spin")
        if spin is None:
            raise ValueError(f"Spin missing for atom {atom['Label']}")
        Hdim *= int(round(2 * spin + 1))
    return Hdim

# %%
# def gen_neighbour_atoms(
#     filename, 
#     muon_position, 
#     cutoffs, 
#     inf_cutoff=10.0, 
#     efg_tensors=None,
#     efg_factor=1.0,
#     gamma_overrides=None,
#     isotope_overrides=None,
#     quadrupole_moments_overrides=None,
#     **read_kwargs
# ):
#     """
#     Neighbor muon atoms format accepted as UNDI input

#     Parameters
#     ----------
#     filename : str
#         Structure file readable by ASE.
#     muon_position : array-like, shape (3,)
#         Fractional crystal coordinates of the muon.
#     cutoffs : dict
#         Species-dependent cutoff distances in Angstrom.
#         Example: {"Ca": 3.0, "F":1.5}
#     inf_cutoff : float, optional
#         Large cutoff supplied to ASE when constructing the neighbour
#         list. The actual selection is subsequently performed using
#         ``cutoffs``. Default is 10 Å.
#     efg_tensors : sequence of (3, 3) array_like or None, optional
#         Precomputed electric field gradient (EFG) tensors in units of V m^-2 
#         associated with the atoms in ``atoms``.
#         The tensors must be ordered identically to the ASE ``Atoms`` object,
#         such that ``efg_tensors[i]`` corresponds to ``atoms[i]``. 
#         If None (default), no user-supplied EFG tensors are used.
#     efg_factor : float, optional
#         Each EFG tensor associated with a selected atom is retrieved and scaled 
#         by ``efg_factor`` before being assigned to the corresponding nuclear-spin
#         object. Default is 1.0
#     gamma_overrides : dict or None, optional
#         Mapping from isotope labels or atomic symbols to nuclear
#         gyromagnetic ratio [rad/(sT)]
#         Examples:  {'17O': 1.00e6}
#         Override the quadrupole moment of 17O
#             {'O': 1.00e6}
#         Override the value for all oxygen atoms
#         If None (default), quadrupole moments are obtained from the
#         default nuclear-data source (isotope.py).
#     isotope_overrides : dict or None, optional
#         Mapping from atomic symbols to isotope-specific labels.
#         Example: {'Si': '29Si', 'O': '17O'}
#         overrite O and Si to  17O and 29Si isotope:
#         If None (default), no isotope substitutions are applied.
#     quadrupole_moments_overrides : dict or None, optional
#         Mapping from isotope labels or atomic symbols to electric
#         quadrupole moments [m^2].
#         Examples:  {'17O': -2.558e-30}
#         Override the quadrupole moment of 17O
#             {'O': -2.558e-30}
#         Override the value for all oxygen atoms
#         If None (default), quadrupole moments are obtained from the
#         default nuclear-data source (isotope.py).
#     **read_kwargs :
#         Additional keyword arguments passed directly to ``ase.io.read``.

#     Returns
#     -------
#     list of dict
#         Example:
#         [
#             {
#                 "Position": array([0., 0., 0.]),
#                 "Label": "mu"
#             },
#             {
#                 "Position": array([2.7e-10, 0., 1.3e-10]),
#                 "Label": "F"
#             }
#         ]
#     """
#     def _validate_dict(name, obj):
#         """Return an empty dict for None, otherwise validate that *value* is a dict."""
#         if obj is None:
#             return {}
#         if not isinstance(obj, dict):
#             raise TypeError(
#                 f"{name} must be None or dict, got {type(obj).__name__}"
#             )
#         return obj


#     atoms = read_from_file(filename, **read_kwargs)

#     # Validate optional parameter inputs expected as dictionary or None
#     isotope_overrides = _validate_dict(
#         "isotope_overrides",
#         isotope_overrides,
#     )

#     gamma_overrides = _validate_dict(
#         "gamma_overrides",
#         gamma_overrides,
#     )

#     quadrupole_moments_overrides = _validate_dict(
#         "quadrupole_moments_overrides",
#         quadrupole_moments_overrides,
#     )

#     # # Terrible idea, make sure user SUPPLIED
#     # cutoffs = _validate_dict(
#     #     "cutoffs",
#     #     cutoffs,
#     # )

#     # Validate supplied EFG tensors once
#     if efg_tensors is not None:
#         if len(efg_tensors) != len(atoms):
#             raise ValueError(
#                 f"Expected {len(atoms)} EFG tensors, "
#                 f"got {len(efg_tensors)}."
#             )

        
#     # append muon
#     atoms.extend(Atom("H", [0, 0, 0]))                   # Use "H" as muon label, maybe an issue H containing structure: TODO 
#     # update muon position
#     scaled_positions = atoms.get_scaled_positions()
#     scaled_positions[-1] = muon_position
#     atoms.set_scaled_positions(scaled_positions)

#     muon_index = len(atoms) - 1                          # muon index, the last atoms

#     ai, aj, D = neighbor_list("ijD", atoms, inf_cutoff)  # very large cutoff to get all possible interactions.
#                                                          # Actual selection is done below

#     data = []

#     for i in range(len(D)):
#         if not (ai[i] == muon_index):
#             continue
        
#         symb = atoms[aj[i]].symbol

#         d = D[i]
#         distance = np.linalg.norm(d)
#         cutoff = cutoffs.get(symb, 0)
#         if distance > cutoff:
#             continue
        
#         # if symb in cutoffs.keys():  # #--> OLD
#         if symb in cutoffs:
#             pos = D[i] * constants.ANGSTROM
#             pos_str = f"[{d[0]: 8.4f}, {d[1]: 8.4f}, {d[2]: 8.4f}]"
#             print(
#                 f"Adding atom #{aj[i]:<4d}"
#                 f"{symb:<4s}"
#                 f" with position (Å): {pos_str}"
#                 f" and distance: {distance:.4f} Å"
#             )

#             # insert "pre-computed" EFG
#             efg_tensor = None
#             if efg_tensors is not None:

#                 efg_tensor = np.asarray(efg_tensors[aj[i]]).copy()

#                 if efg_tensor.shape != (3, 3):
#                     raise ValueError(
#                         f"EFG tensor for atom #{aj[i]} has shape "
#                         f"{efg_tensor.shape}, expected (3,3)."
#                     )

#                 efg_tensor *= efg_factor

#                 print(
#                     f"Using supplied EFG tensor for atom #{aj[i]} "
#                     f"(scaled by {efg_factor:g})"
#                 )

#             # Override isotope label if requested/necessary
#             old_symb = symb
#             symb = isotope_overrides.get(symb, symb)
#             if symb != old_symb:
#                 print(f"[override] isotope: {old_symb} -> {symb}")

#             # Override/insert nuclear gyromagnetic moments for atom 'symb'
#             gamma = gamma_overrides.get(symb)
#             if gamma is not None:
#                 print(
#                     f"[override/insert] "
#                     f"nuclear gyromagnetic ratio for {symb}: "
#                     f"{gamma:.6e} rad/(sT)"
#                 )

#             # Override/insert electric quadrupole moments for atom 'symb'
#             quad_mom = quadrupole_moments_overrides.get(symb)
#             if quad_mom is not None:
#                 print(
#                     f"[override/insert] "
#                     f"quadrupole moment for {symb}: "
#                     f"{quad_mom:.6e} m^2"
#                 )

#             data.append(
#                 {
#                     "Position": pos,
#                     "Label": symb,
#                     "Gamma": gamma,
#                     "ElectricQuadrupoleMoment": quad_mom,
#                     "EFGTensor": efg_tensor,
#                 }
#             )

#     # INSERT muon data "Label" "mu" at first index in "data" i.e zeroth
#     # The muon position is at origin i.e [0, 0, 0]
#     # since all atomic positions above are relative to the "muon_position"
#     data.insert(0, 
#                     {'Position': np.zeros(3),  
#                      'Label': 'mu',
#                      'Gamma': constants.MUON_GYROMAGNETIC_RATIO,
#                      },
#                 )
#     return data

# %%
# def populate_undi_atoms(atoms, logger=None, log_level=""):
#     """
#     Populate UNDI format atomic information.

#     Parameters
#     ----------
#     atoms : list of dict
#         Example:
#         [
#           {"Position": ..., "Label": "mu"},
#           {"Position": ..., "Label": "F"}
#         ]
#     logger : logging.Logger, optional
#     log_level : str, optional

#     Returns
#     -------
#     atoms : list
#         Modified atom list.
#     """
#     # Make own copy to avoid overwriting of internal elements
#     atoms = deepcopy(atoms)

#     logger = logger or logging.getLogger(__name__)

#     if log_level:
#         try:
#             logger.setLevel(getattr(logging, log_level.upper()))
#         except:
#             logger.warning("Invalid logging level")

#     Hdim = 1

#     for i, atom in enumerate(atoms):
#         spin  = atom.get('Spin', None)
#         label = atom.get('Label', None)
#         pos   = atom.get('Position', None)
#         gamma = atom.get('Gamma', None)
#         quadrupole_moment = atom.get('ElectricQuadrupoleMoment', None)

#         # validation
#         if pos is None:
#             raise ValueError(f"Position needed for atom {i}")

#         # assign values 
#         # 1. For Muon
#         if label == "mu":
#             if spin:
#                 if spin != 0.5:
#                     logger.warning("Warning, muon spin already set differs from 0.5!!")
#             else:
#                 atoms[i]["Spin"] = 0.5

#             atoms[i]['Gamma'] = constants.GAMMAS[label]

#         # 1. For Nuclear isotopes
#         else:
#             A, Symbol = splitIsotope(label)
#             e = element(Symbol)
#             # explicit isotope
#             if A:
#                 A = int(A)
#                 for isotope in e.isotopes:
#                     if isotope.mass_number == A:
#                         break
#                 else:
#                     raise ValueError('Isotope {} for atom {} not found.'.format(A, Symbol))

#             # choose most abundant magnetic isotope
#             else:

#                 max_ab = -1.
#                 l = -1
#                 for is_n, isotope in enumerate(e.isotopes):
#                     if isotope.abundance is None:
#                         continue
#                     if isotope.abundance > max_ab:
#                         l = is_n
#                         max_ab = isotope.abundance
#                 # Select isotope with highest abundance
#                 isotope = e.isotopes[l]

#                 level = logging.WARNING if max_ab < 0.99 else logging.INFO
#                 logger.log(level, 'Using most abundand isotope for {}, i.e. {}{}, {} abundance'.format(label, isotope.mass_number, e.symbol, max_ab))

#             # Spin
#             # check if overriding spin
#             mendeelev_spin = float(Fraction(isotope.spin))
#             if spin:
#                 if spin != mendeelev_spin:
#                     logger.warning("Warning, overriding spin for {}".format(label))

#             else:
#                 if mendeelev_spin == 0:
#                     raise RuntimeError("Isotope with " + str(isotope) + " with spin 0 not allowed. Specify a different isotope or remove this nucleus.")
#                 atoms[i]['Spin'] = mendeelev_spin

#             # Gamma
#             # check if overriding gamma
#             if gamma:
#                 logger.warning("Warning, overriding gamma for {}".format(label))

#             else:
#                 if isotope.g_factor:
#                     atoms[i]['Gamma'] = isotope.g_factor * 7.622593285e6 * 2. * np.pi  #  \mu_N /h, is 7.622593285(47) MHz/T that in turn is equal to  γ_n / (2 π g_n)
#                 else:
#                     logger.error("g_factor missing in mendeleev library, you need to specify it by hand")
#                     raise RuntimeError("Missing gammma.")

#             # Quadrupole
#             if quadrupole_moment is None:
#                 atoms[i]['ElectricQuadrupoleMoment'] = isotope.quadrupole_moment * 1e-28 # m^2
#             else:
#                 atoms[i]['ElectricQuadrupoleMoment'] = quadrupole_moment
#                 logger.warning("Warning, overriding quadrupole moment for {}".format(label))

#             # is below needed
#             # atoms[i]["Label"] = isotope.label

#         # increase Hilbert space dimension
#         # Hdim *= (2*atoms[i]['Spin']+1)

#     # # Check Hilber space dimension makes sense
#     # if np.abs(Hdim - np.rint(Hdim)) > 1e-10:
#     #     raise RuntimeError("Something is very bad in the setup")
#     # else:
#     #     Hdim = int(Hdim)

#     # logger.info("Hilbert space is {} dimensional".format(Hdim))
    
#     return atoms


# def atoms_hilbert_space_dimension(atoms):
#     """
#     Compute the Hilbert-space dimension.

#     Parameters
#     ----------
#     atoms : list of dict

#     Returns
#     -------
#     int
#     """
#     Hdim = 1

#     for atom in atoms:
#         spin = atom.get("Spin")
#         if spin is None:
#             raise ValueError(f"Spin missing for atom {atom['Label']}")
#         Hdim *= int(round(2 * spin + 1))
#     return Hdim


# # Legacy alias
# populate_atoms = populate_undi_atoms

# %%
# def exclusive_shell_atoms(reference_atoms, combined_atoms, rtol=1e-4):
#     """Atoms in `combined_atoms` that are NOT also in `reference_atoms`
#     (matched by position, muon entries ignored on both sides). Used both
#     to strip the muon out of the nn shell, and to strip (muon+nn) out of
#     a combined (muon+nn+nnn) list to recover the true, disjoint nnn shell.
 
#     `rtol` is scaled internally by the smallest reference distance --
#     NOT a hardcoded absolute tolerance (an absolute atol silently
#     breaks for SI-unit/metre-scale positions, since interatomic
#     distances are themselves ~1e-10).
#     """
#     ref_pos = np.array([a['Position'] for a in reference_atoms if a['Label'] != 'mu'])
#     if len(ref_pos) == 0:
#         return deepcopy([a for a in combined_atoms if a['Label'] != 'mu'])
 
#     atol = rtol * np.linalg.norm(ref_pos, axis=1).min()
#     result = []
#     for atom in combined_atoms:
#         if atom['Label'] == 'mu':
#             continue
#         pos = atom['Position']
#         is_in_reference = np.any(np.all(np.isclose(pos, ref_pos, atol=atol), axis=1))
#         if not is_in_reference:
#             result.append(deepcopy(atom))
#     return result

 
# def rescale_shells(nn_atoms, nnn_atoms, r_nn=None, zeta=None, rtol=1e-4):
#     """Given the two fit parameters (r_nn in Angstrom, zeta dimensionless),
#     return (mu_atoms, nn_only, nnn_only) with the nn shell renormalized
#     to radius r_nn (direction preserved) and the nnn shell scaled by zeta.
#     Pass r_nn=None / zeta=None to leave that shell at its original
#     (ideal-lattice) geometry."""
#     nn_atoms, nnn_atoms = deepcopy(nn_atoms), deepcopy(nnn_atoms)
 
#     mu_atoms = [atom for atom in nn_atoms if atom['Label'] == 'mu']
#     if len(mu_atoms) != 1:
#         raise ValueError(f"Expected 1 muon, found {len(mu_atoms)}")
 
#     nn_only  = exclusive_shell_atoms(mu_atoms, nn_atoms,  rtol=rtol)
#     nnn_only = exclusive_shell_atoms(nn_atoms, nnn_atoms, rtol=rtol)
 
#     if r_nn is not None:
#         for atom in nn_only:
#             r = np.linalg.norm(atom['Position'])
#             if r > 0:
#                 atom['Position'] = (atom['Position'] / r) * constants.ANGSTROM * r_nn
 
#     if zeta is not None:
#         for atom in nnn_only:
#             atom['Position'] = atom['Position'] * zeta
 
#     return mu_atoms, nn_only, nnn_only