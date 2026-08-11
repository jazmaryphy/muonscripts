# %%
from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

# %%
DEFAULT_BAR_COLORS = {
    "B_tot": "#4472C4",  # blue
    "B_dip": "#DAA520",  # gold
    "B_lor": "#70AD47",  # green
    "B_con": "#F1A7A2",  # salmon/red
}

REF_MARKERS = ["o", "^", "s", "P", "X", "v", "*", "H"]
REF_COLORS = [
    "gray", "darkorange", "purple", "seagreen", 
    "brown", "teal", "slateblue", "crimson"
    ]

# %%
def style_axes(ax, fontsize=12):
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.minorticks_on()

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=6,
        width=1.0,
        labelsize=fontsize,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3,
        width=1.0,
    )


# %%
def field_bar(
    ax: Axes,
    labels: Sequence[str],
    B_tot: Sequence[float],
    B_dip: Sequence[float] | None = None,
    B_lor: Sequence[float] | None = None,
    B_con: Sequence[float] | None = None,
    energy: Sequence[float] | None = None,
    e_ypos: float | None = None,
    e_unit: str | None = 'meV',
    e_font: float | None = None,
    B_exp: float | Sequence[float] | None = None,
    be_fmt: str = 'D',
    be_size: float = 60,
    be_color: str = 'black',
    be_size_r: float = 0.65,
    g_width: float = 0.8,
    bar_width: float = 0.25,
    colors: dict[str, str] | None = None,
) -> Axes:
    """
    Plot grouped bars for one or more field components per muon site.

    Args:
        ax: Axes to draw on.
        labels: x-axis category labels, one per site/group.
        B_tot: Total field per site. Always plotted.
        B_dip, B_lor, B_con: Optional additional series. A series is
            plotted only if given; None skips it entirely.
        g_width: Total width occupied by all bars within one group
            (adjacent group centers are 1 apart). Controls the gap
            between groups. Ignored if `bar_width` is set.
        bar_width: Explicit width for each individual bar, overriding
            `group_width`-based auto-sizing.
        colors: Optional overrides for series colors, keyed by series
            name ("B_tot", "B_dip", "B_lor", "B_con") -- not by plot
            position, so it stays correct regardless of which series
            are actually active in a given call. Unspecified keys fall
            back to `DEFAULT_COLORS`. Example: colors={"B_con": "crimson"}
            changes only the contact-field color.

    Returns:
        Axes: the same `ax`, for chaining.
    """
    # fig, ax = plt.subplots(figsize=figsize)

    n = len(labels)

    # validate size
    if energy is not None and len(energy) != n:
        raise ValueError(f"'energy' must match 'labels' in length ({n}), got {len(energy)}.")
    if not (0.0 < be_size_r < 1.0):
        raise ValueError(f"'be_size_r' must be in (0, 1), got {be_size_r}.")

    # set experimental/observed field, same unit as calc. fields
    b_exp_list: list[float] = []
    if B_exp is not None:
        b_exp_list = [B_exp] if np.isscalar(B_exp) else list(B_exp)
        if len(b_exp_list) == 0:
            raise ValueError("'B_exp' must contain at least one value.")
        if len(b_exp_list) > n:
            raise ValueError(
                f"'B_exp' has {len(b_exp_list)} values but there are only "
                f"{n} groups; cannot assign more experimental values than groups."
            )

    m = len(b_exp_list)
    # group_entries[g]: list of (value, is_primary) assigned to group g
    group_entries: list[list[tuple[float, bool]]] = [[] for _ in range(n)]
    if m > 0:
        if m <= n:
            for i, val in enumerate(b_exp_list):
                group_entries[i].append((val, True))
            if m < n:
                ref_val = b_exp_list[0]
                for g in range(m, n):
                    group_entries[g].append((ref_val, False))
        else:
            for i, val in enumerate(b_exp_list):
                g = i % n
                group_entries[g].append((val, i < n))
        
    x = np.arange(n)

    # series = [("B_tot", r"$B_\mathrm{T}$", B_tot)]
    series = [("B_tot", r"$B_{\mu}$", B_tot)]
    if B_dip is not None:
        series.append(("B_dip", r"$B_\mathrm{dip}$", B_dip))
    if B_lor is not None:
        series.append(("B_lor", r"$B_\mathrm{L}$", B_lor))
    if B_con is not None:
        series.append(("B_con", r"$B_\mathrm{c}$", B_con))

    color_map = {**DEFAULT_BAR_COLORS, **(colors or {})}

    n_series = len(series)
    width = bar_width if bar_width is not None else g_width / n_series

    # center the group of n_series bars on each integer x position
    offsets = (np.arange(n_series) - (n_series - 1) / 2) * width
    offset_by_key = dict(zip((k for k, _, _ in series), offsets))

    # plot bars for each field contributions
    for offset, (key, label, values) in zip(offsets, series):
        ax.bar(
            x + offset, values, width=width,
            color=color_map[key], label=label, zorder=3,
        )

    # set labels texts
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # add energy as text, default in unit of meV
    if energy is not None:
        if e_ypos is None:
            ymin, ymax = ax.get_ylim()
            e_ypos = 0.015 * (ymax - ymin)  # just above y=0,

        fontsize=e_font if e_font else 10
        for xi, e in zip(x, energy):
            ax.text(xi, e_ypos, e, ha="center", va="bottom",
                     fontsize=fontsize, zorder=6)
        # set unit, default meV
        if e_unit:
            ax.text(x[-1] + 0.30, e_ypos, e_unit, ha="left", va="bottom",
                     fontsize=fontsize-2, zorder=6)

    ## OLDER VERSION:
    if b_exp_list:
        be_offset = offset_by_key["B_tot"]
        for i, be in enumerate(b_exp_list):
            ax.scatter(
                x[i] + be_offset, be,
                marker=be_fmt, color=be_color, s=be_size,
                zorder=7, # label=(r"$B_\mathrm{exp}$" if i == 0 else None),
            )


    # ## EXPERIMENTAL: 
    # if m > 0:
    #     be_offset = offset_by_key["B_tot"]
    #     primary_labeled = secondary_labeled = False
    #     for g in range(n):
    #         secondary_count = 0
    #         for be, is_primary in group_entries[g]:
    #             if is_primary:
    #                 marker, color, size = be_fmt, be_color, be_size
    #                 lbl = r"$B_\mathrm{exp}$" if not primary_labeled else None
    #                 primary_labeled = True
    #                 be_alpha=1.0
    #                 z = 7
    #             else:
    #                 cyc = secondary_count % len(REF_MARKERS)
    #                 marker, color = REF_MARKERS[cyc], REF_COLORS[cyc]
    #                 size = be_size * (be_size_r ** (secondary_count + 1))
    #                 # lbl = r"$B_\mathrm{exp}$ (extra)" if not secondary_labeled else None
    #                 lbl = None
    #                 secondary_labeled = True
    #                 secondary_count += 1
    #                 be_alpha=0.0
    #                 z = 6
    #             # ax.scatter(x[g] + be_offset, be, marker=marker, 
    #             #            color=color, s=size, zorder=z)
    #             face_rgba = mcolors.to_rgba(color, alpha=be_alpha)
    #             ax.scatter(x[g] + be_offset, be, marker=marker, color=face_rgba,
    #                     s=size, edgecolors="black", linewidths=0.6, zorder=z)
        
    return ax