# %%
"""Point-charge (PC) lattice summation and Electric Field Gradient (EFG) tensor calculations."""

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import matplotlib.axes
import numpy as np
from ase import Atoms
import numpy.typing as npt
from numpy.typing import NDArray

from efg.utils import get_omegaQ_mu, quadrupole_frequencies
from efg.lattice import check_charges_cover_atoms, get_site_labels, get_site_info

from constants.constants import ANGSTROM, ELEMENTARY_CHARGE, EPSILON0, H_PLANCK, MUON_GYROMAGNETIC_RATIO

# %%
def _replicate_lattice(
    atoms: Atoms,
    charges: dict[str, float],
    sphere_radius_m: float,
    exclude_indices: Sequence[int] = (),
    validate_charges: bool = True,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate positions and charges for periodic images within a cutoff sphere.

    Replicates unit cell positions across periodic boundaries to encompass a spherical
    cutoff region of radius `sphere_radius_m` around the origin.

    Args:
        atoms: Crystal structure containing atomic positions and cell parameters.
        charges: Mapping of element symbols or site labels to formal charges in units of e.
        sphere_radius_m: Spherical cutoff radius in meters.
        exclude_indices: 0-based indices of atoms in `atoms` to exclude from sum.
        validate_charges: If True, raises ValueError if `charges` does not cover all atoms.

    Returns:
        Tuple containing:
            - **pts**: Cartesian coordinates of replicated charges in meters, shape `(N, 3)`.
            - **qs**: Charge values in Coulombs, shape `(N,)`.

    Raises:
        ValueError: If `validate_charges=True` and any atom lacks an assigned charge.
    """
    if validate_charges:
        missing = check_charges_cover_atoms(atoms, charges=charges, strict=False)
        if missing:
            raise ValueError(
                f"ERROR: <charges={charges}> does not cover whole atoms/structure, "
                f"charges of: {missing} missing!"
            )
        # if missing:
        #     raise ValueError(
        #         f"Missing charge specification for site(s)/species: {missing}"
        #     )

    cell: npt.NDArray[np.float64] = np.array(atoms.get_cell())
    positions: npt.NDArray[np.float64] = atoms.get_positions()
    symbols: npt.NDArray[np.str_] = np.array(atoms.get_chemical_symbols())
    site_labels: npt.NDArray[np.str_] = np.array(get_site_labels(atoms))

    # Resolve each atom's charge ONCE (per-site charge takes priority,
    # falls back to per-element) -- this does NOT depend on which
    # periodic image an atom sits in
    q_per_atom: list[Optional[float]] = [
        charges.get(label, charges.get(sym))
        for label, sym in zip(site_labels, symbols)
    ]

    exclude_mask = np.zeros(len(atoms), dtype=bool)
    if exclude_indices:
        exclude_mask[list(exclude_indices)] = True

    nonzero_mask = np.array([q is not None and q != 0 for q in q_per_atom])
    keep_atom = nonzero_mask & ~exclude_mask

    if not np.any(keep_atom):
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)

    kept_positions = positions[keep_atom]
    kept_charges = (
        np.array([q_per_atom[i] for i in np.where(keep_atom)[0]], dtype=float)
        * ELEMENTARY_CHARGE
    )

    # how many unit cells to replicate in each direction to cover `sphere_radius`
    sphere_radius_ang = sphere_radius_m / ANGSTROM
    cell_norms = np.linalg.norm(cell, axis=1)
    n_reps = np.ceil(sphere_radius_ang / cell_norms).astype(int) + 1

    na = np.arange(-n_reps[0], n_reps[0] + 1)
    nb = np.arange(-n_reps[1], n_reps[1] + 1)
    nc = np.arange(-n_reps[2], n_reps[2] + 1)
    nx, ny, nz = np.meshgrid(na, nb, nc, indexing="ij")

    n_ints = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1)
    shifts = np.dot(n_ints, cell)

    pts = (kept_positions[None, :, :] + shifts[:, None, :]).reshape(-1, 3) * ANGSTROM
    qs = np.tile(kept_charges, len(shifts))
    return pts, qs

# %%
def _efg_tensor_from_charges(
    pts: npt.NDArray[np.float64],
    qs: npt.NDArray[np.float64],
    site_position_m: npt.NDArray[np.float64],
    sphere_radius_m: float,
    gamma_sternheimer: float = 0.0,
    verbose: bool = False,
) -> npt.NDArray[np.float64]:
    """Compute 3x3 EFG tensor at a target site from predefined point charges.

    Args:
        pts: Point charge Cartesian coordinates in meters, shape `(N, 3)`.
        qs: Point charges in Coulombs, shape `(N,)`.
        site_position_m: Evaluation position in meters, shape `(3,)`.
        sphere_radius_m: Maximum distance cutoff in meters.
        gamma_sternheimer: Sternheimer antishielding factor.
        verbose: If True, prints diagnostic summation info.

    Returns:
        Symmetric 3x3 EFG tensor in V/m^2.
    """
    d = pts - site_position_m
    r2 = np.sum(d * d, axis=1)
    keep = (r2 > 1e-24) & (r2 < sphere_radius_m**2)

    d, r2, qs_k = d[keep], r2[keep], qs[keep]
    r5 = r2**2.5

    if verbose:
        print(
            f"Point-charge EFG: summing {len(qs_k)} charges "
            f"within radius {sphere_radius_m / ANGSTROM:.3f} Å."
        )

    w = qs_k / r5
    v_tensor = 3.0 * np.einsum("i,ia,ib->ab", w, d, d) - np.eye(3) * np.sum(w * r2)
    v_tensor *= 1.0 / (4.0 * np.pi * EPSILON0)
    v_tensor *= 1.0 - gamma_sternheimer
    return v_tensor


def point_charge_EFG(
    atoms: Atoms,
    site_position: npt.ArrayLike,
    charges: dict[str, float],
    sphere_radius: float = 50.0,
    exclude_indices: Sequence[int] = (),
    extra_charges: Optional[Sequence[Tuple[npt.ArrayLike, float]]] = None,
    coords_are_cartesian: bool = False,
    gamma_sternheimer: float = 0.0,
    verbose: bool = True,
) -> npt.NDArray[np.float64]:
    """Calculate point-charge EFG tensor [V/m^2] at a specified location.

    Args:
        atoms: Periodic crystal structure as an ASE `Atoms` instance.
        site_position: Evaluation coordinate, shape `(3,)`.
            Interpreted as Cartesian (Å) if `coords_are_cartesian=True`,
            or fractional unit-cell coordinates if `coords_are_cartesian=False`.
        charges: Mapping of species or site labels to formal charges in units of e.
        sphere_radius: Spherical summation cutoff radius in Ångströms.
        exclude_indices: Indices of structure atoms to omit from summation.
        extra_charges: Sequence of additional `(position, charge_e)` pairs.
            Positions respect `coords_are_cartesian`.
        coords_are_cartesian: If True, inputs are treated as Cartesian (Å).
            If False (default), inputs are treated as fractional coordinates [0, 1).
        gamma_sternheimer: Sternheimer antishielding factor.
        verbose: If True, prints summation info.

    Returns:
        Symmetric traceless 3x3 EFG tensor in V/m^2.
    """
    site_pos_arr = np.asarray(site_position, dtype=np.float64)

    # Convert site_position to Cartesian (Å) if fractional
    if not coords_are_cartesian:
        cart_site = atoms.cell.cartesian_positions(site_pos_arr)
    else:
        cart_site = site_pos_arr

    # Convert Cartesian position to meters
    site_m = cart_site * ANGSTROM
    sphere_radius_m = sphere_radius * ANGSTROM

    # Replicate structure point charges
    pts, qs = _replicate_lattice(
        atoms, charges, sphere_radius_m, exclude_indices=exclude_indices
    )

    # Handle extra charges with matching coordinate conversion
    if extra_charges:
        extra_pts_list = []
        extra_qs_list = []

        for p, q in extra_charges:
            p_arr = np.asarray(p, dtype=np.float64)
            if not coords_are_cartesian:
                cart_p = atoms.cell.cartesian_positions(p_arr)
            else:
                cart_p = p_arr

            extra_pts_list.append(cart_p * ANGSTROM)
            extra_qs_list.append(q * ELEMENTARY_CHARGE)

        extra_pts = np.array(extra_pts_list, dtype=np.float64)
        extra_qs = np.array(extra_qs_list, dtype=np.float64)

        pts = np.vstack([pts, extra_pts]) if len(pts) else extra_pts
        qs = np.concatenate([qs, extra_qs]) if len(qs) else extra_qs

    # Compute and return 3x3 tensor
    return _efg_tensor_from_charges(
        pts, qs, site_m, sphere_radius_m, gamma_sternheimer, verbose=verbose
    )


def diagonalize_EFG(
    tensor: npt.NDArray[np.float64],
    quadrupole_moment: Optional[float] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Optional[float], float]:
    """Diagonalize EFG tensor into Principal Axis System (PAS) ordered components.

    Calculates principal values according to |Vzz| >= |Vyy| >= |Vxx|.

    Args:
        tensor: 3x3 symmetric EFG tensor in V/m^2.
        quadrupole_moment: Nuclear quadrupole moment Q in m^2 (optional).

    Returns:
        Tuple containing:
            - **V_aa**: Sorted principal components `[Vxx, Vyy, Vzz]` in V/m^2.
            - **P**: 3x3 matrix of normalized principal axes eigenvectors.
            - **chi_q**: Quadrupole coupling constant chi_Q in MHz.
            - **eta**: Asymmetry parameter eta in [0, 1].
    """
    evals, evecs = np.linalg.eigh(tensor)
    order = np.argsort(-np.abs(evals))
    v_zz, v_yy, v_xx = evals[order]
    v_aa = np.array([v_xx, v_yy, v_zz])

    p_matrix = evecs[:, order][:, ::-1]

    scale = np.abs(evals).max()
    eta = float(np.abs(v_xx - v_yy) / np.abs(v_zz)) if abs(v_zz) > 1e-6 * scale else 0.0

    chi_q: Optional[float] = None
    if quadrupole_moment is not None:
        chi_q = float(np.abs(v_zz * ELEMENTARY_CHARGE * quadrupole_moment / H_PLANCK))
        chi_q *= 1e-6  # Convert Hz to MHz

    return v_aa, p_matrix, chi_q, eta

# %%
def compute_efg(
    atoms: Atoms,
    probe_position: npt.ArrayLike,
    atomic_charges: dict[str, float],
    sphere_radius: float,
    gamma_sternheimer: float = 0.0,
    exclude_indices: Sequence[int] = (),
    extra_charges: Optional[Sequence[Tuple[npt.ArrayLike, float]]] = None,
    coords_are_cartesian: bool = True,
    nuclear_spin: Optional[float] = None,
    quadrupole_moment: Optional[float] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compute EFG and nuclear quadrupole properties.

    Args:
        atoms: Crystal structure as an ASE `Atoms` instance.
        probe_position: Target position in Ångströms (if Cartesian) or unit-cell 
            units (if fractional).
        atomic_charges: Mapping of elements/species/site labels to formal charges 
            in units of e.
        sphere_radius: Summation sphere radius in Ångströms.
        gamma_sternheimer: Sternheimer antishielding factor.
        exclude_indices: Indices of atoms in `atoms` to exclude from the lattice sum.
        extra_charges: Additional explicit point charges as `(pos, charge)` tuples.
        coords_are_cartesian: If True, treats `probe_position` as Cartesian (Å). 
            If False, treats `probe_position` as fractional coordinates [0, 1).
        nuclear_spin: Nuclear spin quantum number I.
        quadrupole_moment: Spectroscopic electric quadrupole moment Q in m^2.
        verbose: If True, prints calculation summaries.

    Returns:
        Dictionary containing the raw 3x3 tensor, principal components (Vxx, Vyy, Vzz), 
        asymmetry parameter (eta), and quadrupole coupling constant (chi_Q in MHz).
        and so on...
    """
    # Extract probe info
    cart_pos, frac_pos, probe_index, probe_symbol = get_site_info(
        atoms=atoms,
        position_or_index=probe_position,
        coords_are_cartesian=coords_are_cartesian,
        atol=1e-3,
    )

    # compute raw EFG tensor
    tensor = point_charge_EFG(
        atoms=atoms,
        site_position=probe_position,
        charges=atomic_charges,
        sphere_radius=sphere_radius,
        extra_charges=extra_charges,
        exclude_indices=exclude_indices,
        coords_are_cartesian=coords_are_cartesian,
        gamma_sternheimer=gamma_sternheimer,
        verbose=verbose,
    )

    # post-processing (Diagonalization, eta, chi's)
    v_aa, principal_axes, chi, eta = diagonalize_EFG(
        tensor, quadrupole_moment=quadrupole_moment
    )
    v_xx, v_yy, v_zz = v_aa
    nu_z = nu_q = None

    if quadrupole_moment is not None and nuclear_spin is not None:
        props = quadrupole_frequencies(
            I=nuclear_spin, Q=quadrupole_moment, Vzz=v_zz, eta=eta
        )
        nu_z, nu_q = props.get("nu_z_MHz"), props.get("nu_Q_MHz")


    results = {
        "Vxx": v_xx,
        "Vyy": v_yy,
        "Vzz": v_zz,
        "eta": eta,
        "V_aa": v_aa,
        "nu_z_MHz": nu_z,
        "nu_Q_MHz": nu_q,
        "chi_Q_MHz": chi,
        "EFG_tensor": tensor,
        "principal_axes": principal_axes,
        "probe_index": probe_index,
        "probe_symbol": probe_symbol,
        "probe_position": frac_pos,
    }

    if verbose:
        _pretty_print_efg(results=results)
    return results

# %%
def _pretty_print_efg(results: dict[str, Any]) -> None:
    """Pretty-prints EFG evaluation results table."""
    print("\n" + "=" * 70)
    probe_pos = results.get("probe_position")
    if probe_pos is not None:
        label = (
            f"atom {results['probe_index']} ({results['probe_symbol']})"
            if results.get("probe_index") is not None and results.get("probe_symbol") is not None
            else "probe site"
        )
        print(
            f"EFG analysis for {label} at frac coord. "
            f"({probe_pos[0]:.4f}, {probe_pos[1]:.4f}, {probe_pos[2]:.4f})"
        )
    print("=" * 70)

    scalar_fields = [
        ("Vzz", "V/m^2"),
        ("Vyy", "V/m^2"),
        ("Vxx", "V/m^2"),
        ("eta", "(unitless)"),
        ("chi_Q_MHz", "MHz"),
        ("nu_z_MHz", "MHz"),
        ("nu_Q_MHz", "MHz"),
    ]

    for key, unit in scalar_fields:
        val = results.get(key)
        if val is None:
            continue
        if key == "eta" or key in ("chi_Q_MHz", "nu_z_MHz", "nu_Q_MHz"):
            print(f"{key:<12} = {val: .8f} {unit}")
        else:
            print(f"{key:<12} = {val: .8e} {unit}")

    v_tensor = results.get("EFG_tensor")
    if v_tensor is not None:
        print("\nEFG tensor V_ab (V/m^2) =")
        print("-" * 70)
        for row in v_tensor:
            print(" [ " + ", ".join(f"{x: .8e}" for x in row) + " ]")
        print("-" * 70)
        print(f"Trace(V_ab) = {np.trace(v_tensor): .5e}")
        print(f"Symmetric   = {np.allclose(v_tensor, v_tensor.T)}")

    p_axes = results.get("principal_axes")
    if p_axes is not None:
        print("\nprincipal axes (unitless) = ")
        print("-" * 70)
        for row in p_axes:
            print(" [ " + ", ".join(f"{x: .8e}" for x in row) + " ]")
        print("-" * 70)
    print("=" * 70)

# %%
def sphere_radius_convergence(
    atoms: Atoms,
    site_position: npt.ArrayLike,
    charges: dict[str, float],
    exclude_indices: Sequence[int] = (),
    extra_charges: Optional[Sequence[Tuple[npt.ArrayLike, float]]] = None,
    quadrupole_moment: float = 1.0e-28,
    sphere_radius_list: Optional[Sequence[float]] = None,
    gamma_sternheimer: float = 0.0,
    conv_thr: float = 1e-3,
    sphere_radius_step: float = 10.0,
    sphere_radius_max: float = 100.0,
    num_conv_streak: int = 3,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> Tuple[list[float], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Evaluate convergence of real-space EFG summation over varying radii.

    Args:
        atoms: Crystal structure as ASE `Atoms`.
        site_position: Target Cartesian evaluation position in Ångströms.
        charges: Atomic/site formal charge mapping in units of e.
        exclude_indices: Indices of atoms excluded from summation.
        extra_charges: Additional point charge tuples `(position, charge)`.
        quadrupole_moment: Quadrupole moment Q in m^2.
        sphere_radius_list: Initial list of radii to evaluate in Ångströms.
        gamma_sternheimer: Antishielding factor.
        conv_thr: Relative difference convergence threshold.
        sphere_radius_step: Increment step in Ångströms when extending radii.
        sphere_radius_max: Maximum search cutoff radius in Ångströms.
        num_conv_streak: Number of consecutive evaluations required to declare convergence.
        ax: Optional Matplotlib Axes to plot log relative error vs radius.

    Returns:
        Tuple containing:
            - **radii**: List of evaluated radii in Ångströms.
            - **vzz_values**: Array of Vzz values corresponding to radii.
            - **rel_errors**: Relative errors normalized against final estimate.
    """
    radii = list(sphere_radius_list) if sphere_radius_list is not None else [10.0, 15.0, 20.0, 25.0, 30.0, 40.0]

    build_radius = max(sphere_radius_max + sphere_radius_step, max(radii))
    build_radius_m = build_radius * ANGSTROM
    pts_full, qs_full = _replicate_lattice(
        atoms, charges, build_radius_m, exclude_indices=exclude_indices
    )

    site_m = np.asarray(site_position, dtype=np.float64) * ANGSTROM

    if extra_charges:
        extra_pts = np.array([p for p, _ in extra_charges], dtype=np.float64) * ANGSTROM
        extra_qs = np.array([q for _, q in extra_charges], dtype=np.float64) * ELEMENTARY_CHARGE
        pts_full = np.vstack([pts_full, extra_pts]) if len(pts_full) else extra_pts
        qs_full = np.concatenate([qs_full, extra_qs]) if len(qs_full) else extra_qs

    def _vzz(r: float) -> float:
        r_m = r * ANGSTROM
        v_tensor = _efg_tensor_from_charges(
            pts_full, qs_full, site_m, r_m, gamma_sternheimer, verbose=False
        )
        v_aa, _, _, _ = diagonalize_EFG(v_tensor, quadrupole_moment=quadrupole_moment)
        return float(v_aa[2])

    vzz_values = [_vzz(r) for r in radii]

    def _is_converged() -> bool:
        if len(vzz_values) < num_conv_streak + 1:
            return False
        recent = vzz_values[-(num_conv_streak + 1):]
        diffs = [
            abs(recent[i + 1] - recent[i]) / max(abs(recent[i + 1]), 1e-300)
            for i in range(num_conv_streak)
        ]
        return all(d < conv_thr for d in diffs)

    while radii[-1] < sphere_radius_max and not _is_converged():
        next_r = radii[-1] + sphere_radius_step
        radii.append(next_r)
        vzz_values.append(_vzz(next_r))

    vzz_arr = np.array(vzz_values)
    best = vzz_arr[-1]
    rel_error = np.abs(vzz_arr - best) / max(abs(best), 1e-300)

    print(f"{'radius (Å)':>12} {'Vzz (V/m^2)':>16} {'rel. error vs best':>18}")
    for r, v, e in zip(radii, vzz_arr, rel_error):
        print(f"{r:>12.1f} {v:>16.4e} {e:>18.2e}")

    if not _is_converged() and radii[-1] >= sphere_radius_max:
        print(
            f"WARNING: reached sphere_radius_max={sphere_radius_max} without "
            f"{num_conv_streak} consecutive sustained-converged points."
        )

    if ax is not None:
        mask = rel_error > 0
        ax.semilogy(np.array(radii)[mask], rel_error[mask], "o-")
        ax.set_xlabel("sphere radius (Å)")
        ax.set_ylabel("relative error vs. largest-radius estimate")
        ax.set_title("EFG (Vzz): real-space sum convergence")
        ax.grid(True, which="both", alpha=0.3)

    return radii, vzz_arr, rel_error

# %%
def build_point_charge_efg_neighbors(
    atoms: Atoms,
    neighbors: Iterable[Dict[str, Any]],
    muon_position: Sequence[float] | NDArray[np.float64],
    atomic_charges: Mapping[str, float],
    include_nuclear_efg: bool = True,
    include_muon_induced_efg: bool = False,
    remove_efg_noise: bool = True,
    efg_noise_threshold: float = 1e-8,
    efg_factor: float = 1.0,
    sphere_radius: float = 50.0,
    gamma_sternheimer: float = 0.0,
    exclude_indices: Iterable[int] = (),
    extra_charges: Optional[Sequence[Tuple[Sequence[float] | NDArray[np.float64], float]]] = None,
    efg_verbose: bool = False,
    include_muon: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build quadrupolar neighbours with point-charge EFG tensors compatible with UNDI.
    """
    quadrupolar_neighbors: List[Dict[str, Any]] = []
    muon: Optional[Dict[str, Any]] = None

    cell = atoms.get_cell()

    pts: Optional[NDArray[np.float64]] = None
    qs: Optional[NDArray[np.float64]] = None
    sphere_radius_m = sphere_radius * ANGSTROM

    pts = qs = None
    if include_nuclear_efg:
        pts, qs = _replicate_lattice(
            atoms, atomic_charges, sphere_radius_m, exclude_indices=exclude_indices
        )
        if extra_charges:
            extra_pts = np.array([p for p, q in extra_charges]) * ANGSTROM
            extra_qs = np.array([q for p, q in extra_charges]) * ELEMENTARY_CHARGE
            pts = np.vstack([pts, extra_pts]) if pts is not None and len(pts) else extra_pts
            qs = np.concatenate([qs, extra_qs]) if qs is not None and len(qs) else extra_qs

    for atom_dict in neighbors:
        atom = atom_dict.copy()

        # Extract or skip muon
        if atom.get("Label") == "mu":
            muon = atom
            continue

        spin = float(atom.get("Spin", 0.0))
        is_quadrupolar = spin > 0.5001

        position = np.asarray(atom["Position"])

        if include_nuclear_efg and is_quadrupolar:
            quadrupole_moment = float(atom["ElectricQuadrupoleMoment"])

            # Position of nucleus relative to original unit cell
            nuclear_cartesian = (
                np.dot(muon_position, cell) + position / ANGSTROM
            )
            site_m = nuclear_cartesian * ANGSTROM

            tensor = _efg_tensor_from_charges(
                pts, qs, site_m, sphere_radius_m, gamma_sternheimer, verbose=efg_verbose
            )

            if remove_efg_noise:
                max_element = float(np.max(np.abs(tensor)))
                if max_element > 0:
                    tensor[np.abs(tensor) < efg_noise_threshold * max_element] = 0.0

            tensor *= efg_factor
            atom["EFGTensor"] = tensor

        if include_muon_induced_efg and is_quadrupolar:
            quadrupole_moment = float(atom["ElectricQuadrupoleMoment"])
            distance = float(np.linalg.norm(position))
            atom["OmegaQmu"] = get_omegaQ_mu(I=spin, Q=quadrupole_moment, r=distance)

        quadrupolar_neighbors.append(atom)

    if include_muon:
        if muon is None:
            muon = {
                "Position": np.zeros(3),
                "Label": "mu",
                "Spin": 0.5,
                "Gamma": MUON_GYROMAGNETIC_RATIO,
            }
        else:
            muon = muon.copy()
            muon["Position"] = np.zeros(3)

        quadrupolar_neighbors.insert(0, muon)

    return quadrupolar_neighbors

# %%
class PointChargeEFG:
    """Calculator interface for evaluating Electric Field Gradients using point charges.

    Attributes:
        atoms (Atoms): The periodic crystal structure.
        charges (dict[str, float]): Formal atomic or site charges in units of e.
        sphere_radius (float): Default summation cutoff sphere radius in Ångströms.
        gamma_sternheimer (float): Sternheimer antishielding factor.
        exclude_indices (Tuple[int, ...]): Atomic indices excluded from lattice summations.
        properties (dict[str, Any]): Calculator inputs and target site metadata.
    """

    def __init__(
        self,
        atoms: Atoms,
        charges: dict[str, float],
        sphere_radius: float = 50.0,
        gamma_sternheimer: float = 0.0,
        exclude_indices: Sequence[int] = (),
    ) -> None:
        """Initialize the PointChargeEFG calculator."""
        self.atoms = atoms
        self.charges = charges
        self.sphere_radius = sphere_radius
        self.gamma_sternheimer = gamma_sternheimer
        self.exclude_indices = tuple(exclude_indices)

        # Storage for calculation outputs and metadata
        self.results: dict[str, Any] = {}
        self.properties: dict[str, Any] = {
            "charges": self.charges,
            "sphere_radius": self.sphere_radius,
            "gamma_sternheimer": self.gamma_sternheimer,
            "exclude_indices": self.exclude_indices,
        }


    @property
    def efg_tensor(self) -> Optional[npt.NDArray[np.float64]]:
        """Return the 3x3 EFG tensor [V/m^2]."""
        return self.results.get("EFG_tensor")

    @property
    def principal_axes(self) -> Optional[npt.NDArray[np.float64]]:
        """Return the principal axis eigenvectors."""
        return self.results.get("principal_axes")

    @property
    def v_xx(self) -> Optional[float]:
        """Return the principal Vxx component."""
        return self.results.get("Vxx")

    @property
    def v_yy(self) -> Optional[float]:
        """Return the principal Vyy component."""
        return self.results.get("Vyy")

    @property
    def v_zz(self) -> Optional[float]:
        """Return the principal Vzz component."""
        return self.results.get("Vzz")

    @property
    def eta(self) -> Optional[float]:
        """Return the asymmetry parameter eta."""
        return self.results.get("eta")

    @property
    def nu_z_mhz(self) -> Optional[float]:
        """Return nu_z in MHz."""
        return self.results.get("nu_z_MHz")

    @property
    def nu_q_mhz(self) -> Optional[float]:
        """Return nu_Q in MHz."""
        return self.results.get("nu_Q_MHz")

    @property
    def chi_q_mhz(self) -> Optional[float]:
        """Return quadrupolar coupling constant chi_Q in MHz."""
        return self.results.get("chi_Q_MHz")


    def compute_at(
        self,
        position: Union[npt.ArrayLike, int],
        coords_are_cartesian: bool = False,
        nuclear_spin: Optional[float] = None,
        quadrupole_moment: Optional[float] = None,
        gamma_sternheimer: Optional[float] = None,
        extra_charges: Optional[Sequence[Tuple[npt.ArrayLike, float]]] = None,
        verbose: bool = False,
        atol: float = 1e-3,
    ) -> dict[str, Any]:
        """Compute the EFG tensor and quadrupolar properties at a specific position."""
        # Resolve gamma: use override if provided, else fall back to instance default
        gamma = self.gamma_sternheimer if gamma_sternheimer is None else gamma_sternheimer
        # Resolve target site coordinates and metadata
        cart_pos, frac_pos, probe_idx, probe_sym = get_site_info(
            atoms=self.atoms,
            position_or_index=position,
            coords_are_cartesian=coords_are_cartesian,
            atol=atol,
        )

        # Update input & site metadata
        self.properties.update(
            {
                "probe_position": position,
                "cartesian_position": cart_pos,
                "fractional_position": frac_pos,
                "probe_index": probe_idx,
                "probe_symbol": probe_sym,
                "coords_are_cartesian": coords_are_cartesian,
                "gamma_sternheimer": gamma,
                "nuclear_spin": nuclear_spin,
                "quadrupole_moment": quadrupole_moment,
            }
        )

        # Compute properties using Cartesian coordinates
        res = compute_efg(
            atoms=self.atoms,
            probe_position=cart_pos,
            atomic_charges=self.charges,
            sphere_radius=self.sphere_radius,
            gamma_sternheimer=gamma,
            exclude_indices=self.exclude_indices,
            extra_charges=extra_charges,
            coords_are_cartesian=True,  # Standardized to Cartesian, since "cart_pos" used
            nuclear_spin=nuclear_spin,
            quadrupole_moment=quadrupole_moment,
            verbose=verbose,
        )

        self.results = res
        return self.results

    def get_raw_tensor(
        self,
        position: Union[npt.ArrayLike, int],
        coords_are_cartesian: bool = False,
        gamma_sternheimer: Optional[float] = None,
        extra_charges: Optional[Sequence[Tuple[npt.ArrayLike, float]]] = None,
        atol: float = 1e-3,
    ) -> npt.NDArray[np.float64]:
        """Compute raw 3x3 EFG matrix [V/m^2] without full property extraction."""
        # Resolve gamma: use override if provided, else fall back to instance default
        gamma = self.gamma_sternheimer if gamma_sternheimer is None else gamma_sternheimer

        # Standardize position input via lattice helper
        cart_pos, frac_pos, probe_idx, probe_sym = get_site_info(
            atoms=self.atoms,
            position_or_index=position,
            coords_are_cartesian=coords_are_cartesian,
            atol=atol,
        )

        # Update site metadata
        self.properties.update(
            {
                "probe_position": position,
                "cartesian_position": cart_pos,
                "fractional_position": frac_pos,
                "probe_index": probe_idx,
                "probe_symbol": probe_sym,
                "coords_are_cartesian": coords_are_cartesian,
                "gamma_sternheimer": gamma,
            }
        )

        tensor = point_charge_EFG(
            atoms=self.atoms,
            site_position=cart_pos,
            charges=self.charges,
            sphere_radius=self.sphere_radius,
            exclude_indices=self.exclude_indices,
            extra_charges=extra_charges,
            coords_are_cartesian=True,  # Standardized to Cartesian, since "cart_pos" used
            gamma_sternheimer=gamma,
            verbose=False,
        )

        # Update results dict
        self.results.update(
            {
                "EFG_tensor": tensor,
            }
        )
        # self.results = {"EFG_tensor": tensor}

        return tensor


    def print_summary(self) -> None:
        """Print a structured summary of calculation results and site metadata."""
        if not self.results:
            print("PointChargeEFG: No calculation results available.")
            return

        props = self.properties
        symbol = props.get("probe_symbol", "N/A")
        idx = props.get("probe_index", "N/A")
        frac_pos = props.get("fractional_position")
        cart_pos = props.get("cartesian_position")

        # Format positions cleanly
        frac_str = (
            f"({frac_pos[0]: 8.5f}, {frac_pos[1]: 8.5f}, {frac_pos[2]: 8.5f})"
            if frac_pos is not None
            else "N/A"
        )
        cart_str = (
            f"({cart_pos[0]: 8.5f}, {cart_pos[1]: 8.5f}, {cart_pos[2]: 8.5f})"
            if cart_pos is not None
            else "N/A"
        )

        print("=" * 65)
        print(f"  EFG CALCULATION SUMMARY: Site {symbol} (Index {idx})")
        print("=" * 65)

        print("\n-- Site Metadata & Inputs --")
        print(f"  Fractional Pos : {frac_str}")
        print(f"  Cartesian Pos  : {cart_str}")
        print(f"  Sum Radius     : {props.get('sphere_radius', 'N/A')} Å")
        print(f"  Sternheimer G  : {props.get('gamma_sternheimer', 0.0): .4f}")
        if props.get("nuclear_spin") is not None:
            print(f"  Nuclear Spin I : {props.get('nuclear_spin')}")
        if props.get("quadrupole_moment") is not None:
            print(f"  Quadrupole Q   : {props.get('quadrupole_moment'): .4e} m^2")

        # EFG Matrix Output
        if self.efg_tensor is not None:
            print("\n-- Raw EFG Tensor V_ij (V/m^2) --")
            for row in self.efg_tensor:
                print(
                    f"  ( {row[0]: 14.6e}  {row[1]: 14.6e}  {row[2]: 14.6e} )"
                )

        # Principal Diagonal & Asymmetry (Safely handle None if get_raw_tensor was used)
        if self.v_zz is not None and self.v_xx is not None and self.v_yy is not None:
            print("\n-- Principal Diagonal & Asymmetry --")
            print(
                f"  Vxx = {self.v_xx: 13.6e} V/m^2 | Vyy = {self.v_yy: 13.6e} V/m^2 | Vzz = {self.v_zz: 13.6e} V/m^2"
            )
            if self.eta is not None:
                print(f"  Asymmetry Parameter (eta) : {self.eta: .5f}")

        # Quadrupolar Frequencies (Only prints if quadrupole_moment was provided)
        if self.chi_q_mhz is not None:
            print("\n-- Quadrupolar Frequencies --")
            print(
                f"  Cq  (Quadrupolar Coupling)       : {self.chi_q_mhz: .6f} MHz"
            )
            if self.nu_q_mhz is not None:
                print(
                    f"  nu_Q                             : {self.nu_q_mhz: .6f} MHz"
                )
            if self.nu_z_mhz is not None:
                print(
                    f"  nu_z                             : {self.nu_z_mhz: .6f} MHz"
                )

        print("=" * 65)


    def check_radius_convergence(
        self,
        position: Union[npt.ArrayLike, int],
        coords_are_cartesian: bool = True,
        quadrupole_moment: float = 1.0e-28,
        gamma_sternheimer: Optional[float] = None,
        sphere_radius_max: float = 100.0,
        sphere_radius_step: float = 10.0,
        conv_thr: float = 1e-3,
        ax: Optional[matplotlib.axes.Axes] = None,
        atol: float = 1e-3,
    ) -> Tuple[list[float], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Evaluate how Vzz converges relative to real-space summation radius."""
        # Resolve gamma: use override if provided, else fall back to instance default
        gamma = self.gamma_sternheimer if gamma_sternheimer is None else gamma_sternheimer

        cart_pos, frac_pos, probe_idx, probe_sym = get_site_info(
            atoms=self.atoms,
            position_or_index=position,
            coords_are_cartesian=coords_are_cartesian,
            atol=atol,
        )

        self.properties.update(
            {
                "probe_position": position,
                "cartesian_position": cart_pos,
                "fractional_position": frac_pos,
                "probe_index": probe_idx,
                "probe_symbol": probe_sym,
                "gamma_sternheimer": gamma,
            }
        )

        return sphere_radius_convergence(
            atoms=self.atoms,
            site_position=cart_pos,
            charges=self.charges,
            exclude_indices=self.exclude_indices,
            quadrupole_moment=quadrupole_moment,
            gamma_sternheimer=gamma,
            conv_thr=conv_thr,
            sphere_radius_step=sphere_radius_step,
            sphere_radius_max=sphere_radius_max,
            ax=ax,
        )