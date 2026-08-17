# %%
from copy import deepcopy
from typing import Any, List, Dict, Union, Optional
import numpy as np
from numpy.typing import NDArray

from undi_tools.powder import powder_average, PowderMethod

# Simple, flexible type alias for undi atomic neighbor lists
NeighborList = List[Dict[str, Any]]

# %%
def get_signal(
    neighbors: NeighborList,
    tlist: NDArray[np.float64],
    method: PowderMethod = "single",
    orientation: Optional[Union[List[float], NDArray[np.float64]]] = None,
    **celio_kwargs: Any
) -> NDArray[np.float64]:
    """
    Calculates P_z(t) using Celio's method in UNDI for a single orientation or full powder grids.

    Parameters
    ----------
    neighbors : NeighborList
        List of dictionaries defining atomic sites (e.g., [{'Position': array(...), 'Label': 'mu'}, ...]).
        Must contain 'Position' and 'Label'.
    tlist : NDArray[np.float64]
        1D array of time evaluation points in seconds.
    method : PowderMethod, default="single"
        Orientational grid scheme:
          - 'single'   : Single direction vector or 3x3 rotation matrix (default: [0, 0, 1]).
          - 'xyz'      : Fast 3-orthogonal directions average ([1,0,0], [0,1,0], [0,0,1]).
          - 'zcw'      : Zaremba-Conroy-Wolfsberg powder grid.
          - 'sobol'    : Quasi-random low-discrepancy sampling.
          - 'midpoint' : Midpoint Euler grid.
    orientation : Optional[Union[List[float], NDArray[np.float64]]], default=None
        Single vector (e.g., [0, 0, 1]) or 3x3 rotation matrix.
        Used when `method="single"`. Defaults to [0, 0, 1] if method is "single" and orientation is None.
    verbose : bool, default=True
        If True, prints an informational log regarding the selected sampling method.
    **celio_kwargs : Any
        Celio algorithm parameters (e.g., k, nrep, single_precision, algorithm, index, undi_path).

    Returns
    -------
    NDArray[np.float64]
        Polarization signal array matching the length of `tlist`.
    """
    try:
        from undi import MuonNuclearInteraction
    except (ImportError, ModuleNotFoundError):
        import sys
        if 'undi_path' in celio_kwargs:
            sys.path.append(celio_kwargs.pop('undi_path'))
        from undi import MuonNuclearInteraction

    # Input validation & conflict resolution
    if method == "single":
        if orientation is None:
            orientation = [0.0, 0.0, 1.0]
        celio_kwargs["orientation"] = orientation
        print(f"[INFO] Running single-orientation signal evaluation along: {orientation}")
            
    else:
        if orientation is not None:
            raise ValueError(
                f"Custom 'orientation' ({orientation}) was provided, "
                f"but method is set to '{method}'. "
                f"Set method='single' to evaluate a custom orientation, "
                f"or remove 'orientation' to run grid scheme '{method}'."
            )
        print(f"[INFO] Running powder average using scheme: '{method}'")

    # Extract Celio options
    k: int = celio_kwargs.pop('k', 1)
    nrep: int = celio_kwargs.pop('nrep', 1)
    single_precision: bool = celio_kwargs.pop('single_precision', True)
    algorithm: str = celio_kwargs.pop('algorithm', 'fast')
    log_level: str = celio_kwargs.pop('log_level', 'warning')

    def celio_single_orientation(
        orient: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculates signal for either a 1D direction vector or a 3x3 rotation matrix."""
        ns = MuonNuclearInteraction(deepcopy(neighbors), log_level=log_level)
        
        if orient.ndim == 1:
            ns.translate_rotate_sample_vec(orient)
        else:
            ns.translate_rotate_sample(orient)
        
        orient_signal = np.zeros(len(tlist), dtype=np.float64)
        for _ in range(nrep):
            orient_signal += ns.celio_on_steroids(
                tlist,
                k=k,
                single_precision=single_precision,
                progress=False,
                algorithm=algorithm
            )
        return orient_signal / nrep

    return powder_average(
        celio_single_orientation,
        method=method,
        **celio_kwargs
    )