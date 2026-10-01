"""Colourblind-safe journal figure theme.

Ported from ``notebooks/_generators/nm_style.py``, which was written for the
SOW48 figures and is the style every tutorial figure in this package now uses.

WHY A THEME MODULE
------------------
The alternative house style in this repo is a saturated 8-hue deck palette on an
off-white surface with bold titles and explanatory subtitles.  That is the right
register for a slide and the wrong one for a figure that has to survive a
caption, so the conventions here are stated explicitly:

* **Text is small and uniform.**  7 pt tick labels, 8 pt axis labels, 8.5 pt bold
  panel letters.  No bold headline titles, no subtitle paragraphs -- a figure's
  explanation belongs in its caption.
* **Ink is black on white.**  Spines are 0.6 pt black, and only left+bottom exist.
* **Colour is used sparingly and never decoratively.**  A single muted blue
  carries the default series; vermillion appears only where a contrast is the
  point.
* **Panel letters are lowercase bold at the top-left**, outside the axes.

The palette is Okabe-Ito, the standard colourblind-safe qualitative set for
scientific figures.  Hues are used in fixed order, never cycled.  Note the one
documented exception in :func:`categorical_palette`, which trades CVD-safety for
identity when there are more categories than safe hues -- it says so, and the
docstring explains how to mitigate it.

USAGE
-----
    from pseudoembed.plotting import style as S
    S.apply()
    fig, ax = plt.subplots(figsize=S.FIG_1COL)
    S.panel_letter(ax, "a")
    S.despine(ax)
"""

from __future__ import annotations

import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- palette
# Okabe-Ito, colourblind-safe, fixed order.  BLUE is the default single series;
# VERMILLION is the contrast hue; the rest are for the rare >2-series panel.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
SKY = "#56B4E9"
PURPLE = "#CC79A7"
YELLOW = "#F0E442"
BLACK = "#000000"

#: A darker yellow-brown, used INSTEAD of :data:`YELLOW` for lines and markers.
#: Okabe-Ito's yellow is designed for large filled areas; on a 1 pt line against
#: white it measures 1.32:1 contrast, against a 3:1 floor for graphical objects
#: -- half the next-worst hue in the set.  This one measures 4.7:1 and keeps the
#: same position in colour-deficient vision.
YELLOW_DARK = "#9B7A01"

#: The eight safe hues, in the order they should be consumed.  Position 7 is
#: :data:`YELLOW_DARK`, not :data:`YELLOW`: this cycle is consumed by lines and
#: markers, where pure yellow is illegible on white (measured above).  Reach for
#: :data:`YELLOW` directly when filling a large area, where it works.
CYCLE = [BLUE, VERMILLION, GREEN, ORANGE, SKY, PURPLE, YELLOW_DARK, BLACK]

# Ink levels.  Journal figures are black-on-white; GREY is for de-emphasised
# annotation and MID for grid/reference lines only.
INK = "#000000"
GREY = "#4D4D4D"
MID = "#8C8C8C"
FAINT = "#D9D9D9"

# Sequential ramp for heatmaps: single hue, light -> dark, no rainbow.
SEQ = "Blues"
#: Diverging ramp, for quantities centred on zero (a row z-score, a signed
#: contrast).  Never use SEQ for those: a sequential ramp puts the neutral value
#: at an arbitrary lightness and invents an ordering the data does not have.
DIV = "RdBu_r"

# Nature Methods column widths, in inches (89 mm single, 183 mm double).
W1, W2 = 3.50, 7.20
FIG_1COL = (W1, 2.6)
FIG_2COL = (W2, 3.0)


def apply() -> None:
    """Install the theme.  Call once before any figure is created."""
    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 400,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # Helvetica is the Nature family face; Arial/DejaVu are the fallbacks
            # that actually exist on most machines.  Listed in that order so the
            # figure looks right where the font is present and still renders where
            # it is not.
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            # Titles left-aligned and unbold: a panel is identified by its letter,
            # not by a bold headline.
            "axes.titleweight": "normal",
            "axes.titlelocation": "left",
            "axes.titlepad": 3.0,
            "axes.labelpad": 2.0,
            "axes.linewidth": 0.6,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "xtick.direction": "out",
            "ytick.direction": "out",
            # No grid by default.  A grid is opt-in per panel, and when it appears
            # it is a hairline behind the data.
            "axes.grid": False,
            "grid.color": FAINT,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.2,
            "patch.linewidth": 0.5,
            "legend.frameon": False,
            "legend.handlelength": 1.1,
            "legend.handletextpad": 0.5,
            "legend.borderaxespad": 0.2,
            "legend.labelspacing": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=CYCLE),
        }
    )


def despine(ax, left: bool = True, bottom: bool = True) -> None:
    """Keep only the spines that carry a scale, and only their tick marks.

    Hiding a spine without hiding its ticks leaves the marks floating beside the
    labels, pointing at a frame that is not drawn -- visible on every categorical
    bar panel here, where ``left=False`` says the y axis is a list of names and
    not a scale.  A name does not get a tick mark; it gets a label.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if not left:
        ax.tick_params(axis="y", length=0)
    if not bottom:
        ax.tick_params(axis="x", length=0)


def panel_letter(ax, letter: str, dx: float = -0.085, dy: float = 1.06) -> None:
    """Lowercase bold panel letter, positioned outside the axes at the top-left."""
    ax.text(
        dx,
        dy,
        letter,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def strip_panel_prefix(title: str | None) -> str | None:
    """Drop a leading ``"(a) "`` / ``"a) "`` / ``"a "`` from a title.

    Panel identity is carried by :func:`panel_letter` -- a bold letter outside
    the axes -- never by characters inside the title string.  Callers still pass
    prefixed titles (the convention this package used previously), so the drawing
    functions launder them here rather than trusting every call site.  Without
    this the two conventions coexist and a panel renders as "a (a) ...".
    """
    if not title:
        return title
    # Require the bracket or the closing paren: a BARE letter followed by a word
    # is not a panel prefix, and matching it would turn "x vs y" into "vs y" and
    # "n cells above threshold" into "cells above threshold".
    return re.sub(r"^\s*(?:\([a-z]\)|[a-z]\))\s+(?=\S)", "", title, count=1)


def rho_label(rho: float, name: str = r"\rho") -> str:
    """``rho = +0.897`` as a typeset math label, e.g. for an in-axes annotation.

    Spelling a Greek letter out in Latin ("rho = ") is the tell of a figure
    drawn for a terminal rather than a caption.
    """
    return rf"${name}$ = {rho:+.3f}"


def annotate_stat(ax, text: str, loc: str = "lower right", pad: float = 0.03):
    """Put a statistic *inside* the axes instead of in the title.

    A title should name what a panel shows; a correlation is a result, and it
    belongs with the data.  ``loc`` accepts the four corners.
    """
    ha, va = (
        "right" if "right" in loc else "left",
        "bottom" if "lower" in loc else "top",
    )
    x = 1 - pad if ha == "right" else pad
    y = pad if va == "bottom" else 1 - pad
    return ax.text(
        x, y, text, transform=ax.transAxes, fontsize=7, ha=ha, va=va, color=INK
    )


def emptiest_corner(x, y, frac: float = 0.30) -> str:
    """Which corner of the data's bounding box holds the fewest points.

    Placement rules based on the *shape* of a relationship ("rising data leaves
    the lower right free") fail whenever the data has a floor or a ceiling: a
    zero-inflated readout keeps a dense band along the bottom, so the corner the
    trend vacates is the one the zeros occupy.  Counting is cheap and cannot be
    fooled by that.

    Parameters
    ----------
    frac
        Corner box size as a fraction of each axis range.

    Returns
    -------
    str
        One of ``"lower left"``, ``"lower right"``, ``"upper left"``,
        ``"upper right"`` -- the ``loc`` strings :func:`annotate_stat` accepts.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return "upper right"
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    dx = (x1 - x0) * frac or 1.0
    dy = (y1 - y0) * frac or 1.0
    counts = {
        "lower left": int(((x <= x0 + dx) & (y <= y0 + dy)).sum()),
        "lower right": int(((x >= x1 - dx) & (y <= y0 + dy)).sum()),
        "upper left": int(((x <= x0 + dx) & (y >= y1 - dy)).sum()),
        "upper right": int(((x >= x1 - dx) & (y >= y1 - dy)).sum()),
    }
    # Ties go to the reading-order-latest corner, which is the least likely to
    # sit over a y-axis label or a trend's start.
    order = ["upper right", "upper left", "lower right", "lower left"]
    return min(order, key=lambda c: (counts[c], order.index(c)))


def corner_occupancy(x, y, frac: float = 0.30, loc: str | None = None, allow=None):
    """``(loc, n_points)`` for the emptiest corner -- the count :func:`emptiest_corner`
    discards.

    Pass ``loc`` to ask about ONE named corner instead of the emptiest, which is
    what a caller needs after overriding the choice (e.g. rejecting the upper
    corners because a title occupies them): the returned count must describe the
    corner actually used, or the crowding test is applied to the wrong box.

    Pass ``allow`` (an iterable of corner names) to restrict the candidates.  Use
    this rather than picking the emptiest corner and then substituting another:
    on a rising diagonal that substitution walked from an empty corner into the
    band itself, and the fallback fired on a panel that had free space.

    The count is the part a caller needs to decide whether an in-axes annotation
    is safe AT ALL.  :func:`emptiest_corner` always names a winner, so a panel
    whose data fills all four corners still gets a corner back, and a legend
    placed there lands on the data.  That is not a hypothetical: a four-cluster
    trajectory embedding fills the plane by construction, and trusting the
    winning name alone put this package's legend on top of a cluster.

    Use it as ``loc, n = corner_occupancy(x, y)`` and fall back to a legend
    outside the frame when ``n`` exceeds what you are willing to cover.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return "upper right", 0
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    dx = (x1 - x0) * frac or 1.0
    dy = (y1 - y0) * frac or 1.0
    counts = {
        "lower left": int(((x <= x0 + dx) & (y <= y0 + dy)).sum()),
        "lower right": int(((x >= x1 - dx) & (y <= y0 + dy)).sum()),
        "upper left": int(((x <= x0 + dx) & (y >= y1 - dy)).sum()),
        "upper right": int(((x >= x1 - dx) & (y >= y1 - dy)).sum()),
    }
    if loc is not None:
        if loc not in counts:
            raise ValueError(f"loc must be one of {sorted(counts)}; got {loc!r}")
        return loc, counts[loc]
    order = ["upper right", "upper left", "lower right", "lower left"]
    if allow is not None:
        order = [c for c in order if c in set(allow)]
        if not order:
            raise ValueError("allow excluded every corner")
    best = min(order, key=lambda c: (counts[c], order.index(c)))
    return best, counts[best]


def legend_outside(ax, loc: str = "upper right", ncol: int = 1, **kw):
    """A legend placed clear of the data, above the axes.

    An in-axes legend on a dense panel covers exactly the region a reader wants
    (in this package it has covered the 0 h corner of a scatter and the rising
    limb of a curve).  Anchoring above the frame costs a little height and never
    hides a point.
    """
    anchor = (1.0, 1.02) if "right" in loc else (0.0, 1.02)
    return ax.legend(
        loc="lower right" if "right" in loc else "lower left",
        bbox_to_anchor=anchor,
        ncol=ncol,
        frameon=False,
        fontsize=7,
        handlelength=1.1,
        handletextpad=0.5,
        columnspacing=1.1,
        borderaxespad=0.0,
        **kw,
    )


def panel_letter_above(ax, letter: str, dx: float = -0.085, gap: float = 0.02):
    """Panel letter placed above whatever already sits on top of the axes.

    :func:`panel_letter` uses a fixed offset, which is correct for a bare panel
    but lands *on* the artists when a title and an out-of-frame legend stack above
    the frame.  Arithmetic on font sizes was tried and mispredicted the stack
    height (it put the letter below the legend); this measures the drawn extents
    instead, so it is right whatever the legend row count or the figure size.

    Call after the title and legend exist.
    """
    ax.figure.canvas.draw()
    top = 1.0
    inv = ax.transAxes.inverted()
    for art in (ax.get_legend(), ax.title):
        if art is None or not art.get_visible():
            continue
        try:
            bb = art.get_window_extent().transformed(inv)
        except Exception:
            continue
        if np.isfinite(bb.y1):
            top = max(top, float(bb.y1))
    return ax.text(
        dx,
        top + gap,
        letter,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def no_ticks(ax, axis: str = "both") -> None:
    """Drop tick marks but keep labels.

    Used on categorical axes where the tick adds nothing, because the label
    already names the category.
    """
    ax.tick_params(axis=axis, length=0)


def heatmap_grid(
    ax, nrow: int, ncol: int, color: str = "white", lw: float = 0.6
) -> None:
    """Hairline separators between heatmap cells, so adjacent fills never merge."""
    ax.set_xticks([x - 0.5 for x in range(1, ncol)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, nrow)], minor=True)
    ax.grid(which="minor", color=color, linewidth=lw)
    ax.tick_params(which="minor", length=0)


def get_cmap(name: str):
    """Fetch a *copy* of a colormap, safely across matplotlib versions.

    ``matplotlib.cm.get_cmap`` is deprecated (removed in 3.11) and
    ``plt.get_cmap`` returns a registry singleton -- mutating it with
    ``set_bad`` leaks that change into every other figure in the session.  So
    always copy.
    """
    return plt.get_cmap(name).copy()


def masked_cmap(name: str | None = None, bad: str = FAINT):
    """A colormap whose masked (NaN) cells render as a flat neutral grey.

    Missing data must read as *absent*, not as a low value: an unmasked NaN in a
    sequential ramp renders at the "zero" end and is indistinguishable from a
    real minimum.
    """
    cm = get_cmap(name or SEQ)
    cm.set_bad(bad)
    return cm


def categorical_palette(
    labels, cmap: str = "tab20", background=None, background_color: str = FAINT
) -> dict:
    """``{label: colour}`` for categorical labels, in descending-count order.

    Prefers the eight CVD-safe :data:`CYCLE` hues.  Above eight categories that
    set is exhausted, and this function falls back to ``tab20`` -- **CVD-safety
    is traded for n-way identity at that point, deliberately**.  Do not claim the
    result was validated for colour vision deficiency; mitigate it the way the
    tutorials do, with direct labels and a per-category ``n`` in every legend
    entry.

    In the fallback, colours come from ``tab20``'s **even** indices only.
    ``tab20`` alternates dark/light pairs, so consecutive indices are
    near-duplicates: measured minimum pairwise RGB distance is 0.265 across the
    even indices versus 0.148 across ``range(11)``.  That leaves ten usable hues.

    Parameters
    ----------
    labels
        Any iterable of category labels; counted to fix the assignment order, so
        the most common category gets the first hue and a legend built by
        iterating the result lists common categories first.
    background
        Label to paint a deliberate non-hue (default grey) instead of a colour,
        freeing a hue for the remaining categories.  Pass a progenitor or
        "unassigned" pool here: typically the largest group, and the one that is
        meaningfully *not yet* a category.

    Returns
    -------
    dict
        ``label -> colour``.
    """
    counts = pd.Series(np.asarray(labels).astype(str)).value_counts()
    order = [
        c
        for c in counts.index
        if c != (str(background) if background is not None else None)
    ]

    if len(order) <= len(CYCLE):
        hues = list(CYCLE[: len(order)])
    else:
        base = plt.get_cmap(cmap)
        hues = [base(i / max(1, base.N - 1)) for i in range(0, min(base.N, 20), 2)]
        if len(hues) < len(order):
            # More categories than distinguishable hues.  Cycling is the honest
            # failure mode -- it repeats visibly, whereas interpolating a 20-hue
            # map to n>20 produces neighbours no reader can separate.
            hues = [hues[i % len(hues)] for i in range(len(order))]

    pal = {lab: hues[i] for i, lab in enumerate(order)}
    if background is not None:
        pal[str(background)] = background_color
    return pal


def draw_order(labels) -> tuple[list, dict]:
    """Category draw order (largest first) and the counts behind it.

    Whichever group is plotted last sits on top, so a small group plotted first
    disappears under a large one.  Drawing largest-first keeps rare categories
    visible.

    Returns
    -------
    (order, sizes)
        ``order`` is descending by count; ``sizes`` maps label -> count.
    """
    counts = pd.Series(np.asarray(labels).astype(str)).value_counts()
    return list(counts.index), {k: int(v) for k, v in counts.items()}
