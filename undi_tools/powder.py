# %%
#!/usr/bin/env python3
"""
powder.py

Utilities for powder averaging and orientational sampling in spectroscopy 
and quantum dynamics simulations (NMR, EPR, μSR, etc.).

Conventions
-----------
ZYZ Euler angles are used throughout.

All grid generators yield tuples of:
    (rotation_matrix: NDArray[np.float64], weight: float)

where weight includes the solid-angle Jacobian sin(beta).

Author: Muhammad Maikudi Isah

References
----------
Euler grids:
    M.H. Levitt, Spin Dynamics, 2nd Ed., Wiley (2008)
Monte Carlo:
    N. Metropolis and S. Ulam, JASA 44, 335 (1949)
Sobol:
    I.M. Sobol, USSR Comput. Math. Math. Phys. 7, 86 (1967)
ZCW:
    A. Zaremba, M. Conroy, W. Wolfsberg, J. Chem. Phys. 60, 1154 (1974)
Fibonacci sphere:
    Swinbank and James, Q.J.R. Meteorol. Soc. 122, 1769 (1996)
"""

from typing import Callable, Generator, List, Literal, Optional, Tuple, Union, Any
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

try:
    from scipy.stats.qmc import Sobol
    HAS_SOBOL: bool = True
except ImportError:
    HAS_SOBOL: bool = False

# Type alias for supported powder averaging methods
PowderMethod = Literal[
    "single", "step", "midpoint", "random", "fibonacci", "sobol", "zcw", "xyz"
]

# %%
def euler_rotation(
    alpha: float, 
    beta: float, 
    gamma: float
) -> NDArray[np.float64]:
    """
    Constructs a 3x3 ZYZ Euler rotation matrix.

    Parameters
    ----------
    alpha : float
        First Euler angle (around Z axis) in radians.
    beta : float
        Second Euler angle (around Y axis) in radians.
    gamma : float
        Third Euler angle (around Z axis) in radians.

    Returns
    -------
    NDArray[np.float64]
        A 3x3 rotation matrix.
    """
    return R.from_euler("zyz", [alpha, beta, gamma], degrees=False).as_matrix()


def rotate_tensor(
    tensor: NDArray[np.float64], 
    rotation: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Rotates a rank-2 Cartesian tensor.

    T' = R * T * R^T

    einstein summation convention:
    T' = R * T * R^T  ==>  T'_ij = R_ik * T_kl * R_jl

    Parameters
    ----------
    tensor : NDArray[np.float64]
        3x3 Cartesian tensor.
    rotation : NDArray[np.float64]
        3x3 orthogonal rotation matrix.

    Returns
    -------
    NDArray[np.float64]
        Rotated 3x3 tensor.
    """
    # rot = rotation @ tensor @ rotation.T
    # rot = np.einsum("ik,kl,jl->ij", rotation, tensor, rotation)
    rot = np.matmul(rotation, np.matmul(tensor, rotation.T))
    return rot

# %%
def powder_avg_single(
    orientation: Union[List[float], NDArray[np.float64]]
) -> Generator[Tuple[NDArray[np.float64], float], None, None]:
    """
    Yields a single user-defined orientation (1D vector or 3x3 rotation matrix).

    Parameters
    ----------
    orientation : Union[List[float], NDArray[np.float64]]
        A 3D direction vector shape (3,) or a 3x3 rotation matrix shape (3, 3).

    Yields
    ------
    orient : NDArray[np.float64]
        The user-defined orientation.
    weight : float
        Unit weight (1.0).
    """
    orient_arr: NDArray[np.float64] = np.asarray(orientation, dtype=np.float64)
    if orient_arr.shape not in [(3,), (3, 3)]:
        raise ValueError(
            f"Single orientation must have shape (3,) or (3, 3), "
            f"got {orient_arr.shape}"
        )
    
    yield orient_arr, 1.0

# %%
def powder_avg_xyz(
) -> Generator[Tuple[NDArray[np.float64], float], None, None]:
    """
    Generates unit vectors along the three principal Cartesian axes [100], [010], [001].

    Note
    ----
    This is not a true powder average. It provides a simple 3-direction 
    orientational sample used for quick testing or highly symmetric systems.

    Yields
    ------
    direction : NDArray[np.float64]
        1D array representing a Cartesian direction vector.
    weight : float
        Integration weight (1.0 per direction).
    """
    directions: List[NDArray[np.float64]] = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    weight: float = 1.0
    for d in directions:
        yield d, weight

# %%
def powder_avg_step(
    Na: int, 
    Nb: int, 
    Ng: int
) -> Generator[Tuple[NDArray[np.float64], float], None, None]:
    """
    Generates a uniform Euler-angle grid (ZYZ convention).

    Parameters
    ----------
    Na : int
        Number of grid points for alpha [0, 2π).
    Nb : int
        Number of grid points for beta [0, π).
    Ng : int
        Number of grid points for gamma [0, 2π).

    Yields
    ------
    rotation_matrix : NDArray[np.float64]
        3x3 ZYZ rotation matrix.
    weight : float
        Integration weight including sin(beta) solid angle term.
    """
    alpha: NDArray[np.float64] = np.linspace(0, 2 * np.pi, Na, endpoint=False)
    beta: NDArray[np.float64] = np.linspace(0, np.pi, Nb, endpoint=False)
    gamma: NDArray[np.float64] = np.linspace(0, 2 * np.pi, Ng, endpoint=False)

    dalpha: float = alpha[1] - alpha[0] if Na > 1 else 2 * np.pi
    dbeta: float = beta[1] - beta[0] if Nb > 1 else np.pi
    dgamma: float = gamma[1] - gamma[0] if Ng > 1 else 2 * np.pi

    for a in alpha:
        for b in beta:
            sinb: float = float(np.sin(b))
            for g in gamma:
                weight: float = sinb * dalpha * dbeta * dgamma
                yield euler_rotation(a, b, g), weight

# %%
def powder_avg_midpoint(
    Na: int, 
    Nb: int, 
    Ng: int
) -> Generator[Tuple[NDArray[np.float64], float], None, None]:
    """
    Generates a midpoint Euler grid.

    More accurate than `powder_avg_step` because it samples interval midpoints,
    avoiding singularity issues at beta = 0.

    Parameters
    ----------
    Na : int
        Number of alpha steps.
    Nb : int
        Number of beta steps.
    Ng : int
        Number of gamma steps.

    Yields
    ------
    rotation_matrix : NDArray[np.float64]
        3x3 rotation matrix.
    weight : float
        Integration weight.
    """
    dalpha: float = 2 * np.pi / Na
    dbeta: float = np.pi / Nb
    dgamma: float = 2 * np.pi / Ng

    alpha: NDArray[np.float64] = (np.arange(Na) + 0.5) * dalpha
    beta: NDArray[np.float64] = (np.arange(Nb) + 0.5) * dbeta
    gamma: NDArray[np.float64] = (np.arange(Ng) + 0.5) * dgamma

    for a in alpha:
        for b in beta:
            sinb: float = float(np.sin(b))
            for g in gamma:
                weight: float = sinb * dalpha * dbeta * dgamma
                yield euler_rotation(a, b, g), weight

# %%
def powder_avg_random(
    N: int
) -> Generator[Tuple[NDArray[np.float64], float], None, None]:
    """
    Generates uniform random 3D rotations (Monte Carlo integration).

    Parameters
    ----------
    N : int
        Number of random orientations to generate.

    Yields
    ------
    rotation_matrix : NDArray[np.float64]
        Random 3x3 rotation matrix.
    weight : float
        Uniform weight factor (4π / N).
    """
    weight: float = 4 * np.pi / N
    for _ in range(N):
        yield R.random().as_matrix(), weight

# %%
def powder_avg_fibonacci(
    N: int
) -> Generator[Tuple[NDArray[np.float64], float], None, None]:
    """
    Generates uniform spherical orientations using a Fibonacci lattice.

    Parameters
    ----------
    N : int
        Number of sample points on the sphere.

    Yields
    ------
    rotation_matrix : NDArray[np.float64]
        3x3 rotation matrix aligning [0, 0, 1] to the Fibonacci point.
    weight : float
        Uniform weight factor (4π / N).
    """
    golden_ratio: float = (np.sqrt(5.0) + 1.0) / 2.0
    weight: float = 4 * np.pi / N

    for k in range(N):
        z: float = 1.0 - (2.0 * k + 1.0) / N
        phi: float = 2 * np.pi * k / golden_ratio
        r: float = np.sqrt(max(0.0, 1.0 - z * z))

        direction: NDArray[np.float64] = np.array([r * np.cos(phi), r * np.sin(phi), z])
        rotation: NDArray[np.float64] = R.align_vectors([direction], [[0.0, 0.0, 1.0]])[0].as_matrix()

        yield rotation, weight

# %%
def powder_avg_sobol(
    N: int
) -> Generator[Tuple[NDArray[np.float64], float], None, None]:
    """
    Generates orientations using a Sobol quasi-random low-discrepancy sequence.

    Convergence
    -----------
    O(N^-1), outperforming standard Monte Carlo sampling.

    Parameters
    ----------
    N : int
        Number of orientations.

    Yields
    ------
    rotation_matrix : NDArray[np.float64]
        3x3 rotation matrix.
    weight : float
        Uniform weight factor (4π / N).

    Raises
    ------
    RuntimeError
        If `scipy.stats.qmc.Sobol` is unavailable in the environment.
    """
    if not HAS_SOBOL:
        raise RuntimeError("Sobol generator unavailable. Ensure SciPy >= 1.7 is installed.")

    sampler = Sobol(d=3, scramble=True)
    pts: NDArray[np.float64] = sampler.random(N)
    weight: float = 4 * np.pi / N

    for p in pts:
        alpha: float = 2 * np.pi * p[0]
        beta: float = float(np.arccos(1.0 - 2.0 * p[1]))
        gamma: float = 2 * np.pi * p[2]

        yield euler_rotation(alpha, beta, gamma), weight

# %%
def fibonacci_numbers(n: int) -> List[int]:
    """
    Generates Fibonacci sequence up to index n.
    """
    f: List[int] = [1, 1]
    while len(f) < n:
        f.append(f[-1] + f[-2])
    return f


def powder_avg_zcw(
    index: int = 12
) -> Generator[Tuple[NDArray[np.float64], float], None, None]:
    """
    Generates a Zaremba-Conroy-Wolfsberg (ZCW) orientational grid.

    Parameters
    ----------
    index : int, default=12
        Fibonacci index determining grid density:
          - index 8  -> 34 orientations
          - index 10 -> 89 orientations
          - index 12 -> 233 orientations
          - index 13 -> 377 orientations

    Yields
    ------
    rotation_matrix : NDArray[np.float64]
        3x3 rotation matrix.
    weight : float
        Uniform weight factor (4π / N_points).
    """
    fib: List[int] = fibonacci_numbers(index + 1)
    npts: int = fib[-1]
    fprev: int = fib[-2]

    weight: float = 4 * np.pi / npts

    for k in range(npts):
        z: float = 2.0 * (k + 0.5) / npts - 1.0
        theta: float = float(np.arccos(z))
        phi: float = 2 * np.pi * k * fprev / npts

        rot: NDArray[np.float64] = euler_rotation(phi, theta, 0.0)
        yield rot, weight

# %%
def directional_average(
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    directions: List[NDArray[np.float64]]
) -> NDArray[np.float64]:
    """
    Computes an unweighted average of a function over a list of direction vectors.

    Parameters
    ----------
    func : Callable[[NDArray[np.float64]], NDArray[np.float64]]
        Function accepting a 1D direction vector and returning a scalar or NDArray.
    directions : List[NDArray[np.float64]]
        List of 3D unit vectors.

    Returns
    -------
    NDArray[np.float64]
        Directionally averaged result.
    """
    signal: Optional[NDArray[np.float64]] = None

    for d in directions:
        value: NDArray[np.float64] = np.asarray(func(d), dtype=np.float64)
        if signal is None:
            signal = np.zeros_like(value)
        signal += value

    if signal is None:
        raise ValueError("Directions list cannot be empty.")

    return signal / len(directions)

# %%
def powder_average(
    func: Callable[[NDArray[np.float64]], Union[float, NDArray[np.float64]]],
    method: PowderMethod = "zcw",
    **kwargs: Any
) -> NDArray[np.float64]:
    """
    Averages an arbitrary orientational function over a spherical powder grid.

    Parameters
    ----------
    func : Callable[[NDArray[np.float64]], Union[float, NDArray[np.float64]]]
        Callable accepting a (3, 3) rotation matrix and returning a scalar or NDArray.
    method : PowderMethod, default="zcw"
        Grid generation method:
          - 'zcw'       : Zaremba-Conroy-Wolfsberg grid (requires `index: int`, default=12)
          - 'midpoint'  : Midpoint Euler grid (requires `Na`, `Nb`, `Ng`)
          - 'step'      : Step Euler grid (requires `Na`, `Nb`, `Ng`)
          - 'random'    : Uniform Monte Carlo sampling (requires `N`)
          - 'fibonacci' : Fibonacci sphere sampling (requires `N`)
          - 'sobol'     : Quasi-random Sobol sequence (requires `N`)
          - 'xyz'       : Cartesian directional average (3 points)
    **kwargs : Any
        Parameters passed to the underlying grid generator (e.g., `index`, `N`, `Na`, `Nb`, `Ng`).

    Returns
    -------
    NDArray[np.float64]
        Powder-averaged result.

    Example
    -------
    >>> result = powder_average(
    ...     func=lambda rmat: gen_signal(atoms, rmat, nrep=6),
    ...     method="zcw",
    ...     index=12
    ... )
    """
    if method == "step":
        grid = powder_avg_step(kwargs["Na"], kwargs["Nb"], kwargs["Ng"])
    elif method == "midpoint":
        grid = powder_avg_midpoint(kwargs["Na"], kwargs["Nb"], kwargs["Ng"])
    elif method == "random":
        grid = powder_avg_random(kwargs["N"])
    elif method == "fibonacci":
        grid = powder_avg_fibonacci(kwargs["N"])
    elif method == "sobol":
        grid = powder_avg_sobol(kwargs["N"])
    elif method == "zcw":
        grid = powder_avg_zcw(kwargs.get("index", 8))
    elif method == "xyz":
        grid = powder_avg_xyz()
    elif method == "single":
        if "orientation" not in kwargs:
            raise ValueError(
                f"Method 'single' requires an 'orientation' "
                f"argument (vector or matrix)."
                )
        grid = powder_avg_single(kwargs["orientation"])
    else:
        raise ValueError(
            f"Unknown powder averaging method '{method}'. "
            f"Supported options: 'zcw', 'midpoint', 'step', "
            f"'random', 'fibonacci', 'sobol', 'xyz'."
        )

    result: Optional[NDArray[np.float64]] = None
    total_weight: float = 0.0

    for rot, w in grid:
        value: NDArray[np.float64] = np.asarray(func(rot), dtype=np.float64)

        if result is None:
            result = np.zeros_like(value, dtype=np.float64)

        result += w * value
        total_weight += w

    if result is None or total_weight == 0.0:
        raise RuntimeError("Powder grid yielded no points or zero accumulated weight.")

    return result / total_weight