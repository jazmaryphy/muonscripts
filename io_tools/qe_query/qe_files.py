# %%
"""Backward-compatible re-export shim.

`find_qe_files.py` used to contain everything (file-finding, calc-type
classification, convergence checking, signature extraction, and
grouping/filtering/loading) in one file. It has been split into four
focused modules -- `qe_discover`, `qe_classify`, `qe_signature`,
`qe_collect` -- for easier independent testing/reuse.

This file re-exports the same public names from their new locations, so
existing code doing e.g. `from io_tools.find_qe_files import select_groups`
keeps working unchanged. New code should import directly from the
specific submodule instead (e.g. `from io_tools.qe_collect import
select_groups`) -- this shim is a migration aid, not the long-term home
for anything.
"""

from io_tools.qe_discover import find_qe_files

from io_tools.qe_classify import (
    RELAX_LIKE,
    MD_LIKE,
    VARIABLE_CELL,
    STATIC,
    PW_START,
    detect_qe_calc_type_from_input,
    detect_qe_calc_type_from_output,
    classify_qe_run,
    check_qe_convergence,
)

from io_tools.qe_signature import (
    _get_input_system_signature,
    _get_output_system_signature,
)

from io_tools.qe_collect import (
    iter_paths,
    get_relax_after_run,
    select_groups,
    print_relax_summary,
    load_relax_from_folder,
    load_scf_from_folder,
    load_nscf_from_folder,
)

__all__ = [
    "STATIC", 
    "MD_LIKE", 
    "PW_START",
    "RELAX_LIKE", 
    "VARIABLE_CELL", 
    "iter_paths",
    "select_groups", 
    "find_qe_files",
    "classify_qe_run", 
    "print_relax_summary",
    "get_relax_after_run", 
    "check_qe_convergence",
    "load_scf_from_folder", 
    "load_nscf_from_folder",
    "load_relax_from_folder", 
    "detect_qe_calc_type_from_input", 
    "detect_qe_calc_type_from_output",
]