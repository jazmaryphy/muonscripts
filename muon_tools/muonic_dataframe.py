# %%
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

# %%
def _assign_label(index: int) -> str:
    """
    Map a 0-based energy rank to a letter label: A..Z, A'..Z', A''..Z''.

    Args:
        index (int): 0-based rank after sorting by energy.

    Returns:
        str: Letter label.

    Raises:
        ValueError: If `index` exceeds the 78 labels this scheme supports
            (26 letters x 3 tiers). Beyond that, `chr()` would silently
            produce non-letter characters instead of a sensible label.
    """
    if index < 26:
        return chr(65 + index)
    elif index < 52:
        return chr(65 + index - 26) + "'"
    elif index < 78:
        return chr(65 + index - 52) + "''"
    raise ValueError(
        f"Cannot assign a label for rank {index}: only 78 labels "
        "(A-Z, A'-Z', A''-Z'') are supported."
    )


def _validate_sc_matrix(sc_matrix) -> np.ndarray:
    """
    Validate a supercell transformation matrix.

    Assumes the convention supercell_lattice = sc_matrix @ unitcell_lattice
    (row-vector lattice matrices), matching pymatgen's
    `Structure.make_supercell`.

    Args:
        sc_matrix: Array-like, expected shape (3, 3).

    Returns:
        np.ndarray: Validated (3, 3) float array.

    Raises:
        ValueError: If `sc_matrix` cannot be interpreted as a (3, 3)
            numeric matrix, or is singular (not invertible -- a
            singular matrix means supercell -> unit-cell coordinates
            aren't well-defined).
    """
    sc_matrix = np.asarray(sc_matrix, dtype=float)
    if sc_matrix.shape != (3, 3):
        raise ValueError(f"'sc_matrix' must have shape (3, 3), got {sc_matrix.shape}.")
    if np.isclose(np.linalg.det(sc_matrix), 0.0):
        raise ValueError("'sc_matrix' is singular (determinant ~ 0); not invertible.")
    return sc_matrix


def _add_vector(entry: dict, prefix: str, vec, norm) -> None:
    """Add {prefix}_x/_y/_z/_norm columns for a field vector to `entry`."""
    x, y, z = np.round(vec, 4)
    entry[f"{prefix}_x"] = x
    entry[f"{prefix}_y"] = y
    entry[f"{prefix}_z"] = z
    entry[f"{prefix}_norm"] = round(float(norm), 4)
    

def get_dataframe(
    results: List[Dict[str, Any]], 
    sc_matrix: Optional[list] = None
)-> List[pd.DataFrame]:
    """INFO"""

    has_scmat=False
    if sc_matrix is not None:
        has_scmat=True
        sc_matrix = _validate_sc_matrix(sc_matrix)

    has_fields = False

    all_structures = {}

    for i, result in enumerate(results, start=1):
        idx = result["idx"]
        entry = {}

        entry["structure_id"] = idx
        entry["label"] = i
        entry["tot_energy"] = result["energy"]

        mu_pos = np.round(np.array(result["rlxd_struct"]["sites"][-1]["abc"]), 3)

        entry["muon_position_sc"] = mu_pos

        if has_scmat:
            entry["muon_position_uc"] = (np.dot(mu_pos, sc_matrix)%1).round(3)

        if "fields" in result:
            has_fields=True
            fr = result["fields"]  # LocalFieldResults dataclass

            def _add_vector(entry, prefix, vec, norm):
                x, y, z = np.round(vec, 4)
                entry[f"{prefix}_x"] = x
                entry[f"{prefix}_y"] = y
                entry[f"{prefix}_z"] = z
                entry[f"{prefix}_norm"] = round(float(norm), 4)

            _add_vector(entry, "Btot", fr.total[0], fr.total_norm[0])
            _add_vector(entry, "Bdip", fr.dipolar[0], fr.dipolar_norm[0])
            _add_vector(entry, "Blor", fr.lorentz[0], fr.lorentz_norm[0])
            _add_vector(entry, "Bdip_tot", fr.dipolar_tot[0], fr.dipolar_tot_norm[0])
            _add_vector(entry, "Bcon", fr.contact[0], fr.contact_norm[0])

        all_structures[idx] = entry

    # exporting the dataframe for all sites;
    df_all = pd.DataFrame.from_dict(all_structures)
    # sort
    df_all = df_all.sort_values("tot_energy", axis=1)

    # deltaE
    # round deltaE to integer meV   
    df_all.loc["delta_E"] = ((df_all.loc["tot_energy"] - df_all.loc["tot_energy"].min())*1000).astype(int)

    # redefine the "label" to be letters from A to Z
    df_all.loc["label"] = [_assign_label(i) for i in range(len(df_all.columns))]
    
    # then swap row and columns (for sure can be done already above for df, but useful to keep the same order before this point)
    df_all = df_all.transpose()
    df_all = df_all.reset_index(drop=True)
    df_all.index +=  1

    cols = {
        "structure_id": df_all["structure_id"],
        "label": df_all["label"],
        # "R_mu (frac. coords.)": (
        #     list(df_all["muon_position_uc"]) if has_scmat else list(df_all["muon_position_sc"])
        # ),
        "muon_position": (
            list(df_all["muon_position_uc"]) if has_scmat else list(df_all["muon_position_sc"])
        ),
        "delta_E_meV": df_all["delta_E"],
    }

    if has_fields:
        cols["B_T"] = list(zip(df_all["Btot_x"], df_all["Btot_y"], df_all["Btot_z"]))
        cols["|B_T|"] = df_all["Btot_norm"]
        cols["B_dip"] = list(zip(df_all["Bdip_tot_x"], df_all["Bdip_tot_y"], df_all["Bdip_tot_z"]))
        cols["|B_dip|"] = df_all["Bdip_tot_norm"]
        cols["|B_c|"] = df_all["Bcon_norm"]
    
    df = pd.DataFrame(cols)

    return df, df_all