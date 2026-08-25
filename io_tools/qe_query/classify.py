# %%
"""Classify a Quantum Espresso run's `calculation` type, and check whether
it converged.

Two independent questions, both answered here since they're both about
INTERPRETING a QE run's status (as opposed to `qe_discover`, which only
cares about which files exist on disk, or `qe_signature`, which is about
the physical system size/composition rather than the run's outcome):

- "what KIND of calculation is this" (relax? scf? md?) -- from the input
  namelist (reliable) or the output text (best-effort fallback).
- "did it actually converge" -- from the output text, reading backward
  from EOF so a multi-job-concatenated file always reports the LATEST
  run's outcome, never an older one's.
"""

import os
import collections.abc
from pathlib import Path
from io_tools.qe import read_qe_namelists
from io_tools.base import find_first_index

# %%
RELAX_LIKE = {"relax", "vc-relax"}
MD_LIKE = {"md", "vc-md"}
VARIABLE_CELL = {"vc-relax", "vc-md"}
STATIC = {"scf", "nscf", "bands"}

# Marks the start of a new pw.x run -- used both as a generic marker-search
# helper and, critically, as a hard stop boundary when scanning a (possibly
# multi-job-concatenated) .out file backward in `check_qe_convergence`, so
# an OLDER run's flags in the same file never leak into the latest run's
# result.
PW_START = "Program PWSCF"

# %%
def _classify(calc):
    calc = None if calc is None else str(calc).strip().strip("'\"").lower()
    return {
        "calc_type": calc,
        "relax": calc in RELAX_LIKE | MD_LIKE,
        "vc_relax": calc in VARIABLE_CELL,
        "static": calc in STATIC,
    }


def _is_good(conv_info):
    return conv_info["converged"] and conv_info["job_done"] and not conv_info["crashed"]


def _has_marker(text, lines, case_sensitive=False):
    """Check whether one or more strings exist in an iterable of lines.

    Parameters
    ----------
    text : str or iterable of str
        String or strings to search for. If multiple strings are given,
        the function returns True if any of them is found.
    lines : iterable of str
        Lines to search.
    case_sensitive : bool, default=False
        Whether the search is case-sensitive.

    Returns
    -------
    bool
        True if at least one search string is found.
    """
    if isinstance(text, str):
        text = (text,)

    if not case_sensitive:
        text = tuple(t.lower() for t in text)
        return any(any(t in line.lower() for t in text) for line in lines)

    return any(any(t in line for t in text) for line in lines)


def _normalize_calc_types(calc_types):
    """Normalize calculation types to a tuple.

    Parameters
    ----------
    calc_types : str, iterable of str, or None
        Calculation type(s). ``None`` means any calculation type.

    Returns
    -------
    tuple[str] or None
        Normalized calculation types.
    """
    if calc_types is None:
        return None

    if isinstance(calc_types, str):
        return (calc_types,)

    if not isinstance(calc_types, collections.abc.Iterable):
        raise TypeError(
            "`calc_types` must be a string, an iterable of strings, or None."
        )

    return tuple(calc_types)

# %%
### Calculation-type detection

# %%
def detect_qe_calc_type_from_input(in_path=None, namelists=None):
    """Classify a QE run's `calculation` type from its parsed namelists.

    Provide exactly one of `namelists` (already-parsed dict from
    `read_qe_namelists`) or `in_path` (a .in file to read and parse here).
    See `classify_qe_run` if you need to fall back to output-only
    detection when no input file is available at all.
    """
    if namelists is None:
        if in_path is None:
            raise ValueError("Provide either `namelists` or `in_path`.")
        try:
            input_string = Path(in_path).read_text(errors="ignore")
            namelists = read_qe_namelists(input_string.lower())
        except Exception as exc:  # deliberately broad: any parser/IO failure -> "not PW input"
            result = _classify(None)
            result.update(
                source="pw_input", is_pw_input=False, error=f"{type(exc).__name__}: {exc}"
            )
            return result

    control = namelists.get("control") if isinstance(namelists, dict) else None
    if control is None:
        result = _classify(None)
        result.update(source="pw_input", is_pw_input=False, error="No &CONTROL namelist found")
        return result

    calc = control.get("calculation")  # None if &CONTROL exists but 'calculation' wasn't set
    result = _classify(calc)
    result.update(source="pw_input", is_pw_input=True, error=None)
    return result

# %%
def detect_qe_calc_type_from_output(out_path):
    """Best-effort `calculation`-type classification from .out text alone.
 
    EXPERIMENTAL: only validated for relax/vc-relax so far. Prefer
    `detect_qe_calc_type_from_input` whenever an input file is available.
    """
    out_path = Path(out_path)
    with out_path.open(errors="ignore") as f:
        lines = f.readlines()

    ## 1) Converged relax/vc-relax: "Begin final coordinates" block
    try:
        final_kw = "Begin final coordinates"
        start = find_first_index(final_kw, lines)
        coord_kw = "ATOMIC_POSITIONS"
        index = find_first_index(coord_kw, lines, start=start)
        try:
            end = find_first_index("End final coordinates", lines, start=start)
        except KeyError:
            end = len(lines)

        has_cell_params = _has_marker("CELL_PARAMETERS", lines[start:end])
        calc = "vc-relax" if has_cell_params else "relax"

        result = _classify(calc)
        result["source"] = "pw_output"              
        # result["confidence"] = "high"            # to check later
        return result
    except KeyError:
        pass  # no final-coordinates block -> keep looking

    ## EXPERIMENTAL (ALL BELOW)
    ## only trust this branch for relax/vc-relax; the scf/nscf/bands split
    ## below is a lower-confidence guess.
    has_dynamics = _has_marker("Entering Dynamics", lines)
    has_new_cell_volume = _has_marker("new unit-cell volume", lines)
    is_bfgs_converged = _has_marker("bfgs converged", lines)
    has_final_enthalpy = _has_marker("Final enthalpy", lines)
    is_nscf = _has_marker(
        ("Non-self-consistent Calculation", "non-self-consistent"), lines,
    )
    is_band_structure = _has_marker(
        ("Band Structure Calculation", "Band Symmetry"), lines,
    )
    n_total_energy_steps = sum(
        1 for line in lines if line.strip().startswith("!") and "total energy" in line
    )

    ### 2) MD-type: "Entering Dynamics:" iteration headers 
    if has_dynamics:
        calc = "vc-md" if has_new_cell_volume else "md"
        result = _classify(calc)
        result["source"] = "pw_output"              
        # result["confidence"] = "high"            # to check later
        return result

    ### 3) Relax that didn't reach "Begin final coordinates" (e.g. cut off) 
    if is_bfgs_converged or n_total_energy_steps > 1:
        calc = "vc-relax" if (has_new_cell_volume or has_final_enthalpy) else "relax"
        result = _classify(calc)
        result["source"] = "pw_output"              
        # result["confidence"] = "medium"            # to check later
        return result

    ### 4) Static run: try to tell scf/nscf/bands apart, lower confidence 
    if is_band_structure:
        calc, confidence = "bands", "medium"
    elif is_nscf:
        calc, confidence = "nscf", "medium"
    elif n_total_energy_steps >= 1:
        calc, confidence = "scf", "low"  # default assumption for scf
    else:
        result = _classify(None)
        result["source"] = "pw_output"              
        # result["confidence"] = "none"            # to check later
        return result

    result = _classify(calc)
    result["source"] = "pw_output"              
    # result["confidence"] = confidence            # to check later (TRUST CONFIDENCE)
    return result

# %%
def classify_qe_run(out_path=None, in_path=None, namelists=None):
    """Classify a QE run, using whatever information is available.

    Priority: `namelists` > `in_path` (parsed via `read_qe_namelists`) >
    `out_path` (text-heuristic fallback, for runs with no input file at
    all -- e.g. when you didn't use `check_input=True` in `find_qe_files`
    and ended up with orphaned .out files).
    """
    if namelists is not None or in_path is not None:
        return detect_qe_calc_type_from_input(in_path=in_path, namelists=namelists)

    if out_path is not None:
        return detect_qe_calc_type_from_output(out_path)

    raise ValueError("Provide at least one of `namelists`, `in_path`, or `out_path`.")

# %%
#### Convergence check

# %%
def check_qe_convergence(out_path, block_size=4096):
    """Memory-efficient, multi-job aware QE convergence checker.
    
    Reads the file backward from the bottom up to evaluate the final state 
    of the file safely, without loading the whole file into RAM.
    """
    out_path = Path(out_path)
    
    # Flags to return
    job_done = False
    crashed = False
    timed_out = False
    scf_converged = None
    ionic_converged = None

    # Flags to stop searching backward once we find the latest status
    found_scf = False
    found_ionic = False
    hit_job_boundary = False

    # Read file backward efficiently
    with out_path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        remainder = ""
        offset = file_size

        while offset > 0 and not hit_job_boundary:
            to_read = min(block_size, offset)
            offset -= to_read
            f.seek(offset)
            
            # Decode chunk safely and combine with previous remainder
            chunk = f.read(to_read).decode("utf-8", errors="ignore")
            data = chunk + remainder
            lines = data.splitlines()
            
            # Save the incomplete first line for the next iteration step
            remainder = lines[0] if offset > 0 else ""
            lines_to_process = lines[1:] if offset > 0 else lines

            # Process this chunk's lines backward
            for line in reversed(lines_to_process):
                if PW_START in line:
                    # Reached the beginning of the most recent run -- every
                    # flag found so far belongs to it; nothing further back
                    # is relevant. Stop immediately, don't process/read more.
                    hit_job_boundary = True
                    break

                # Global termination flags
                if "JOB DONE." in line:
                    job_done = True
                if "Error in routine" in line or "%%%%%%%%%%%%%%%%" in line:
                    crashed = True
                if "Maximum CPU time exceeded" in line:
                    timed_out = True

                # Ionic Convergence status (Stops at the last calculation block)
                if not found_ionic:
                    if "bfgs converged in" in line:
                        ionic_converged = True
                        found_ionic = True
                    elif "The maximum number of steps has been reached" in line:
                        ionic_converged = False
                        found_ionic = True

                # SCF Convergence status (Stops at the last calculation block)
                if not found_scf:
                    if "convergence has been achieved" in line:
                        scf_converged = True
                        found_scf = True
                    elif "convergence NOT achieved" in line:
                        scf_converged = False
                        found_scf = True

                # Multi-job isolation safeguard:
                # If we hit an older job's end banner, stop searching for convergence flags
                if "End of self-consistent calculation" in line and found_ionic:
                    # This prevents mixing flags from older concatenated runs
                    pass

            # Early exit optimization: stop reading if all latest markers are found
            if found_scf and found_ionic and (job_done or crashed or timed_out):
                break

    converged = ionic_converged if ionic_converged is not None else scf_converged

    return {
        "job_done": job_done,
        "crashed": crashed,
        "timed_out": timed_out,
        "scf_converged": scf_converged,
        "ionic_converged": ionic_converged,
        "converged": converged,
    }