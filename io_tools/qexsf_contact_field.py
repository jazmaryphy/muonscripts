# %%
import numpy as np
from io_tools import qe
from constants import constants

# %%
def get_muon_contact_field(
    up_file: str,
    down_file: str,
    index: tuple[int, int, int] = (0, 0, 0),
    strict: bool = True,
) -> float | None:
    """
    Compute the isotropic Fermi contact hyperfine field from a pair of
    spin-resolved XSF density grids.

    Reads only the single grid point needed from each file, rather than
    parsing the full 3D grid -- see `_read_xsf_point`.

    Args:
        up_file (str): Path to the spin-up density .xsf file.
        down_file (str): Path to the spin-down density .xsf file.
        index (tuple[int, int, int]): Grid index (i, j, k) at which to
            evaluate the spin density difference. Defaults to (0, 0, 0) --
            verify this is actually the muon site for your workflow.
        conversion (float): Conversion factor from spin density (e/bohr^3)
            to Fermi contact field in tesla. Default 52.430351.
        strict (bool): If True, a failure reading either file is
            re-raised. If False, it's logged and None is returned.

    Returns:
        float | None: Contact hyperfine field in tesla. None if
            `strict` is False and either file failed to read.

    Raises:
        ValueError: If the spin-up and spin-down grids have different
            dimensions.
        OSError, KeyError, ValueError: From `read_xsf_point`, if strict.

    NOTE: FASTER, we dont need the entire .xsf datagrid
    """
    try:
        up_rho, up_dims = qe.read_qe_xsf_point(up_file, index)
    except (OSError, KeyError, ValueError) as e:
        msg = f"Failed to read spin-up file '{up_file}': {e}"
        if strict:
            raise type(e)(msg) from e
        # logger.warning(msg)
        return None

    try:
        dw_rho, dw_dims = qe.read_qe_xsf_point(down_file, index)
    except (OSError, KeyError, ValueError) as e:
        msg = f"Failed to read spin-down file '{down_file}': {e}"
        if strict:
            raise type(e)(msg) from e
        # logger.warning(msg)
        return None

    if up_dims != dw_dims:
        raise ValueError(
            f"Spin-up grid shape {up_dims} ('{up_file}') does not match "
            f"spin-down grid shape {dw_dims} ('{down_file}')."
        )

    sp_density = up_rho - dw_rho
    return sp_density, sp_density * constants.SPIN_DENSITY_AU_TO_TESLA


def get_muon_contact_field_from_datagrid(
    up_file: str,
    down_file: str,
    index: tuple[int, int, int] = (0, 0, 0),
    strict: bool = True,
) -> float | None:
    """
    Compute the isotropic Fermi contact hyperfine field from a pair of
    spin-resolved XSF density grids.

    Uses B_c = (2*mu_0/3) * mu_B * delta_s(0), with the spin density
    difference delta_s = rho_up - rho_dw evaluated at a single grid point.

    Args:
        up_file (str): Path to the spin-up density .xsf file.
        down_file (str): Path to the spin-down density .xsf file.
        index (tuple[int, int, int]): Grid index (i, j, k) at which to
            evaluate the spin density difference. Defaults to (0, 0, 0) --
            this is only the muon/nucleus position if the DFT code that
            produced the grid centered its first grid point there. Verify
            this against your workflow rather than assuming it.
        strict (bool): If True (default), a failure reading either file
            is re-raised immediately. If False, the failure is logged
            as a warning and the function returns None instead of
            raising, so a caller looping over many muon sites can skip
            this one and continue.

    Returns:
        float | None: Contact hyperfine field in tesla, at `index`.
            None if `strict` is False and either file failed to read.

    Raises:
        ValueError: If the spin-up and spin-down grids have different
            shapes (i.e. were not evaluated on the same real-space grid).
        OSError, KeyError, ValueError: Whatever `read_qe_xsf_datagrid`
            raises for either file, if `strict` is True.

    NOTE: SLOW as its reading the entire xsf datagrid
    """
    try:
        up = qe.read_qe_xsf_datagrid(up_file)
    except (OSError, KeyError, ValueError) as e:
        msg = f"Failed to read spin-up file '{up_file}': {e}"
        if strict:
            raise type(e)(msg) from e
        # logger.warning(msg)
        return None

    try:
        dw = qe.read_qe_xsf_datagrid(down_file)
    except (OSError, KeyError, ValueError) as e:
        msg = f"Failed to read spin-down file '{down_file}': {e}"
        if strict:
            raise type(e)(msg) from e
        # logger.warning(msg)
        return None

    if up.shape != dw.shape:
        raise ValueError(
            f"Spin-up grid shape {up.shape} ('{up_file}') does not match "
            f"spin-down grid shape {dw.shape} ('{down_file}'); they must "
            "share the same real-space grid to be compared point-by-point."
        )

    sp_density = up[index] - dw[index]
    return sp_density, sp_density * constants.SPIN_DENSITY_AU_TO_TESLA