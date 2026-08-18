# %%
"""
One entry point for writing a QE pw.x input file from a pymatgen
Structure -- handles the magnetic and non-magnetic cases uniformly, so
callers don't need to branch on it themselves.
 
Magnetic path (magmom given):
    1. get_collinear_mag_kindname (see magnetism.py) projects magmom onto
       a collinear z-axis and assigns kind names (e.g. "Co1"/"Co2").
    2. Those per-atom moments are set on the ASE Atoms object directly
       (atoms.set_initial_magnetic_moments) -- this is what lets ASE both
       auto-split same-element atoms into separate QE species AND write
       the correct starting_magnetization(i) itself; see the discussion
       in this conversation for why hand-building either of those is
       fragile/wrong.
    3. ASE's own auto-generated species labels ("Co", "Co1", ...) are
       relabeled to match the kind_name convention ("Co1", "Co2", ...)
       via relabel_qe_species.
 
Non-magnetic path (magmom=None): a plain write, no magnetic bookkeeping.
 
Either way, optionally reformats the result into the aligned/
"centralized" namelist style via qe_format.
 
Namelist contents follow ase.io.espresso.write_espresso_in's own
convention -- a single `input_data` dict of {NAMELIST: {key: value}} --
rather than one parameter per namelist. Whatever you pass is merged
ON TOP OF DEFAULT_QE_NAMELISTS (a copy, per call -- the module-level
constant itself is never mutated): existing default keys you don't
mention are kept, keys you do mention are overridden, and namelist
sections not in the defaults at all (e.g. &FCP) are added as-is.
"""

from __future__ import annotations
 
import copy
from typing import Any, Dict, Optional, Sequence, Tuple

from ase.io import write
from ase.io.espresso import write_espresso_in

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from io_tools.qe_format import relabel_qe_species
from io_tools.qe_format import build_species_rename_map 
from io_tools.qe_format import format_qe_file_centralized
from symmetry.magnetism import make_collinear_getmag_kind

# %%
#: Baseline namelist contents used whenever a given key isn't supplied in
#: `input_data`. Copied (never mutated) per call -- see `_merge_input_data`.
DEFAULT_QE_NAMELISTS: Dict[str, Dict[str, Any]] = {
    "CONTROL": {
        "nstep": 300,
    },
    "SYSTEM": {
        "degauss": 0.01,
        "ecutrho": 600.0,
        "ecutwfc": 60.0,
        "occupations": "smearing",
        "smearing": "gaussian",
    },
    "ELECTRONS": {
        "conv_thr": 1e-07,
        "electron_maxstep": 800,
        "mixing_beta": 0.3,
        "mixing_mode": "local-TF",
    },
    "IONS": {},
    "CELL": {},
}

# %%
def _merge_input_data(
    input_data: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Deep-copy `DEFAULT_QE_NAMELISTS` and update it, section by section,
    with whatever's in `input_data` -- a per-section `dict.update`, not a
    full replace, so unmentioned default keys survive. Sections present
    in `input_data` but not in the defaults (e.g. &FCP) are added as new
    sections."""
    merged = copy.deepcopy(DEFAULT_QE_NAMELISTS)
    if input_data:
        for section, entries in input_data.items():
            merged.setdefault(section, {})
            merged[section].update(entries)
    return merged

# %%
def write_pw_input(
    filename: str,
    structure: Structure,
    magmom: Optional[Sequence[Any]] = None,
    pseudopotentials: Dict[str, str] = None,
    half: bool = True,
    input_data: Optional[Dict[str, Dict[str, Any]]] = None,
    kpts: Optional[Tuple[int, int, int]] = None,
    kspacing: Optional[float] = None,
    koffset: Tuple[int, int, int] = (0, 0, 0),
    crystal_coordinates: bool = True,
    centralized: bool = True,
    indent: int = 3,
) -> None:
    """Write a QE pw.x input file, magnetic or not.
 
    Parameters
    ----------
    filename : str
         A file to which the input is written.
    structure : pymatgen.core.Structure
        Structure to write (muon-free host, or with muon included --
        whatever you want in the file; no muon-specific handling here).
    pseudopotentials : dict
        Keyed by REAL element symbol (e.g. {"Co": "...", "F": "..."}) --
        NOT by kind name, regardless of whether `magmom` is given.
    magmom : sequence, optional
        Per-atom magnetic moments (floats, 3-vectors, or Magmom objects),
        same order as `structure`. None (default) -> non-magnetic write,
        nothing magnetic gets touched. Given -> collinear kind-name
        magnetization is computed and applied automatically; `nspin=2`
        is set in SYSTEM unless `input_data['SYSTEM']['nspin']` already
        specifies one.
    half : bool, default=True
        Passed to `make_collinear_getmag_kind` -- normalize non-zero
        moments to +/-0.5 (typical QE convention) rather than using the
        raw physical moment as `starting_magnetization`.
    input_data : dict, optional
        `{NAMELIST_NAME: {key: value, ...}, ...}` -- same shape ASE's own
        `write_espresso_in` expects. Merged ON TOP of
        `DEFAULT_QE_NAMELISTS` (see module docstring): unmentioned
        default keys are kept, mentioned ones are overridden. Do NOT put
        `starting_magnetization(i)` in here when `magmom` is given --
        it's derived and written automatically.
    kpts, kspacing, koffset, crystal_coordinates
        Passed straight through to `ase.io.write`. `kspacing` (if not
        None) takes priority over `kpts` -- that's ASE's own precedence,
        not something this function adds, so you can leave `kpts` at its
        default (gamma kpts) and just set `kspacing` to switch modes.
    centralized : bool, default=True
        Reformat the written file into the right-aligned/"centralized"
        namelist style afterwards.
    indent : int, default=3
        Left margin (spaces) for `centralized` formatting.
    """
    atoms = AseAtomsAdaptor.get_atoms(structure)
    namelists = _merge_input_data(input_data)
 
    rename_map: Optional[Dict[str, str]] = None
    qe_pseudos = dict(pseudopotentials) if pseudopotentials else {}

    if magmom is not None:
        rst = make_collinear_getmag_kind(structure, magmom=magmom, half=half)
        new_structure = rst["struct_magkind"]
        start_mg_dict = rst["start_mag_dict"]
        kind_names = new_structure.site_properties["kind_name"]
 
        atoms = AseAtomsAdaptor.get_atoms(new_structure)
        per_atom_mag = [start_mg_dict.get(k, 0.0) for k in kind_names]
        atoms.set_initial_magnetic_moments(per_atom_mag)
 
        # Ensure SYSTEM namelist has nspin set unless user explicitly provided one
        if "SYSTEM" not in namelists:
            namelists["SYSTEM"] = {}
        namelists["SYSTEM"].setdefault("nspin", 2)

        rename_map = build_species_rename_map(atoms, kind_names)

        # # --- FIX: Map element-level pseudopotentials to kind names ---
        # # For example, maps pseudopotentials["Fe"] to pseudopotentials["Fe1"], ["Fe2"], etc.
        # for site, kind in zip(new_structure, kind_names):
        #     elem = site.specie.symbol
        #     if elem in qe_pseudos and kind not in qe_pseudos:
        #         qe_pseudos[kind] = qe_pseudos[elem]

    else:
        # 1. Force-clear initial magnetic moments on the ASE Atoms instance
        # so ASE's writer does not auto-generate starting_magnetization(i)
        if atoms.has('initial_magmoms'):
            atoms.set_initial_magnetic_moments([0.0] * len(atoms))

        # 2. Strip any accidental magnetic parameters from the SYSTEM namelist
        system_namelist = namelists.get("SYSTEM", {})
        system_namelist.pop("nspin", None)
        
        # Remove any lingering starting_magnetization keys
        keys_to_remove = [k for k in system_namelist if k.startswith("starting_magnetization")]
        for key in keys_to_remove:
            system_namelist.pop(key, None)
 
    write_espresso_in(
        filename,
        atoms=atoms,
        # format="espresso-in",
        input_data=namelists,
        pseudopotentials=qe_pseudos,
        kspacing=kspacing,
        kpts=kpts,
        koffset=koffset,
        crystal_coordinates=crystal_coordinates,
    )
 
    if rename_map is not None:
        relabel_qe_species(filename, rename_map)
 
    if centralized:
        format_qe_file_centralized(filename, indent=indent)