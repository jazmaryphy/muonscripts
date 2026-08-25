# %%
"""Extract a cheap "system signature" (nat, ntyp, species) from a QE input
or output file -- enough to distinguish different materials/systems from
each other WITHOUT loading full structures (via ASE or otherwise).

This is deliberately separate from `qe_classify` (which answers "what kind
of run is this / did it converge") -- signature extraction is about the
PHYSICAL SYSTEM (how big is it, what's it made of), independent of what
kind of calculation was run on it.
"""

from pathlib import Path
from io_tools.qe import read_qe_namelists
from io_tools.base import find_first_index

# %%
def _get_input_system_signature(in_path):
    """Given `in_path`, parse its namelists and return (nat, ntyp, species).

    Returns None if the file doesn't look like a genuine PW input -- no
    &SYSTEM section at all, or &SYSTEM present but missing nat/ntyp. This
    is deliberate: returning a fake (None, None, None) tuple instead would
    make grouping code silently merge together every unrelated file that
    hits this case, as if they were one system. Callers should treat
    `None` as "skip this file", not as a group key.
    """
    input_string = Path(in_path).read_text(errors="ignore")
    namelists = read_qe_namelists(input_string.lower())
    lines = input_string.splitlines()

    if not isinstance(namelists, dict):
        return None

    system = namelists.get("system")
    if system is None:
        return None  # no &SYSTEM section -> not recognizable as a PW input

    nat = system.get("nat")
    ntyp = system.get("ntyp")

    if nat is None or ntyp is None:
        return None  # &SYSTEM present but incomplete -> no reliable signature

    species = None
    if lines is not None:
        names = []
        try:
            index = find_first_index("ATOMIC_POSITIONS", lines)
            for i in range(index + 1, index + 1 + nat):
                row_split = lines[i].split()
                names.append(row_split[0])

            if len(names) == int(nat):
                species = tuple(sorted(set(names)))
        # NOTE: find_first_index raises KeyError (not IndexError/ValueError)
        # when "ATOMIC_POSITIONS" isn't found at all -- that case MUST be
        # caught here too, or a malformed/incomplete .in file crashes the
        # whole batch instead of just leaving species=None for this file.
        except (IndexError, ValueError, KeyError):
            species = None

    return (nat, ntyp, species)

# %%
def _get_output_system_signature(out_path):
    """Signature read straight from the .out file header, for when there's
    no matching .in file at all.

    Returns None if the atom-count marker isn't found at all (truncated /
    malformed / wrong-code file) -- never raises, so one bad file can't
    crash a batch scan.
    """
    out_path = Path(out_path)
    with out_path.open(errors="ignore") as f:
        lines = f.readlines()

    try:
        nat_index = find_first_index("number of atoms/cell", lines)
        nat = int(lines[nat_index].split()[-1])
    except (KeyError, ValueError, IndexError):
        return None

    try:
        ntyp_index = find_first_index("number of atomic types", lines, start=nat_index)
        ntyp = int(lines[ntyp_index].split()[-1])
    except (KeyError, ValueError, IndexError):
        return (nat, None, None)  # got nat at least; still useful for grouping

    return (nat, ntyp, None)