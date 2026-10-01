"""Single-axes panels for trajectory figures.

Every function here follows one contract, which is what makes them composable:

* **The axes comes first and is never created here.**  ``f(ax, data, ...)``.  The
  caller owns the figure, so panels drop into any ``subplots``/``gridspec``
  layout without fighting for the canvas.
* **No ``plt.show()``, no ``savefig``, no ``tight_layout``.**  Those are the
  caller's, and calling them from inside a panel breaks multi-panel figures.
* **A dict of what was actually drawn comes back.**  Row order, dropped rows,
  bin counts, the statistic in the title.  A figure that cannot be interrogated
  after the fact cannot be checked, and several of these panels *filter* their
  input -- a silent filter reads as biological absence.

The non-obvious choices below are load-bearing and each one is justified by a
measurement in its docstring, because most were made after a figure misled
someone.  Do not "simplify" them without re-checking the measurement.

Ported from ``notebooks/weinreb_tutorial_lib.py``, which these functions replace
as the consolidation point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import style as S

__all__ = [
    "axis_vs_groups",
    "embedding",
    "signed_edge_split",
    "view_share",
    "onset_vs_background",
    "axis_heatmap",
    "top_features",
    "profiles",
    "scatter_trend",
]


# ---------------------------------------------------------------------------
# 1. The axis against a grouping
# ---------------------------------------------------------------------------


def axis_vs_groups(
    ax,
    axis,
    groups,
    order=None,
    palette=None,
    title=None,
    xlabel="group",
    ylabel="pseudotime",
    strip=True,
    rotate=0,
    group_values=None,
    seed=0,
):
    """Box + jittered strip of ``axis`` across ordered groups.

    The box shows the WITHIN-group spread, which is the honest display for a
    small number of ordinal levels -- a regression line through three clouds
    would imply a resolution the design does not have.

    Groups are typically unequal, so ``n`` is carried in every tick label rather
    than letting equal box widths imply equal support.

    Parameters
    ----------
    group_values
        Numeric encoding of ``groups``, supplied **only** when the grouping is a
        genuine ordered quantity (sampling time).  When given, Spearman rho is
        appended to the title.  Leave as ``None`` for an unordered grouping such
        as cell type: rho against alphabetical category codes is meaningless, and
        rho against some *other* variable would mislabel the panel.

    Returns
    -------
    dict
        ``order``, ``n`` per group, and ``rho`` (nan when not requested).
    """
    groups = np.asarray(groups).astype(str)
    order = list(order) if order is not None else sorted(pd.unique(groups))
    parts = [np.asarray(axis, float)[groups == g] for g in order]
    pos = np.arange(len(order))

    # Narrower boxes, hairline outlines, and PALE fills.  Two things were wrong
    # before: the fill was a heavy flat blue at alpha 0.85, and it came from a
    # hardcoded "#7fa8c9" rather than the theme -- so this panel spoke a different
    # visual language from its neighbours in the same figure.  A box plot drawn
    # beside a strip is also encoding the distribution twice, so the box must
    # recede and let the points carry the data.
    bp = ax.boxplot(
        parts,
        positions=pos,
        widths=0.52,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color=S.INK, lw=1.0),
        boxprops=dict(lw=0.6, edgecolor=S.INK),
        whiskerprops=dict(lw=0.6, color=S.INK),
        capprops=dict(lw=0.6, color=S.INK),
    )
    for patch, g in zip(bp["boxes"], order):
        patch.set_facecolor((palette or {}).get(g, S.BLUE))
        patch.set_alpha(0.22)

    if strip:
        rng = np.random.default_rng(seed)
        for p, v, g in zip(pos, parts, order):
            if v.size:
                # Points in the GROUP's colour, not a single hardcoded navy, and
                # sized so a few hundred cells read as a cloud rather than dust.
                ax.scatter(
                    p + rng.normal(0, 0.062, size=v.size),
                    v,
                    s=3.2,
                    color=(palette or {}).get(g, S.BLUE),
                    alpha=0.40,
                    linewidths=0,
                    rasterized=True,
                    zorder=2,
                )

    ax.set_xticks(pos)
    # `n` goes on the SAME line when rotated.  A two-line label rotated about a
    # right-aligned anchor sends its second line diagonally down-right, landing
    # under the NEXT tick -- measured on a 13-category panel, every "n=" read as
    # belonging to the following category.  Only stack the lines when upright.
    labels = (
        [f"{g} (n={p.size:,})" for g, p in zip(order, parts)]
        if rotate
        else [f"{g}\nn={p.size:,}" for g, p in zip(order, parts)]
    )
    ax.set_xticklabels(
        labels, fontsize=7.5, rotation=rotate, ha="right" if rotate else "center"
    )
    # Rotated ticks already name the grouping; repeating it as an axis label
    # just pushes the panel taller.
    ax.set_xlabel("" if rotate else xlabel)
    ax.set_ylabel(ylabel)

    rho = float("nan")
    if group_values is not None:
        rho = float(
            spearmanr(
                np.asarray(axis, float), np.asarray(group_values, float)
            ).statistic
        )
    if title:
        ax.set_title(S.strip_panel_prefix(title))
    # rho is a RESULT, so it is annotated with the data rather than appended to
    # the title.  Lower-left is empty by construction here: the axis rises with
    # the grouping, so the first box sits at the bottom-LEFT and the corner above
    # it is free.
    if group_values is not None and np.isfinite(rho):
        S.annotate_stat(ax, S.rho_label(rho), loc="upper left")
    S.despine(ax)
    return {
        "order": order,
        "n": {g: int(p.size) for g, p in zip(order, parts)},
        "rho": rho,
    }


# ---------------------------------------------------------------------------
# 2. The embedding
# ---------------------------------------------------------------------------


def embedding(
    ax,
    x,
    y,
    labels,
    palette=None,
    order=None,
    title=None,
    xlabel="spectral axis 1",
    ylabel="spectral axis 2",
    draw_last=None,
    s=6,
    legend_loc=None,
    show_counts=True,
):
    """Scatter two axes coloured by a categorical, drawn largest-group-first.

    Draw order matters: whichever group is plotted last sits on top, so a small
    group plotted first disappears under a large one.  ``draw_last`` names the
    groups that must stay visible; everything else is drawn before them, biggest
    first.  Counts go in the legend because "which group is big" is otherwise
    invisible in an overplotted scatter -- but set ``show_counts=False`` when a
    NEIGHBOURING panel already prints n per group (as the axis-vs-groups panel
    does under each box).  Repeating them there is not just redundant: it roughly
    doubles the width of every entry, and a right-hand key that wide runs into
    the next panel's axis label.

    ``legend_loc`` defaults to ``None``, meaning "put it in whichever corner of
    the data's bounding box holds the fewest points", excluding the two upper
    corners when a ``title`` is set.  Pass an explicit ``loc`` string to override.

    Returns
    -------
    dict
        ``order`` (actual draw order) and ``n`` per group.
    """
    labels = np.asarray(labels).astype(str)
    uniq = list(order) if order is not None else sorted(pd.unique(labels))
    sizes = {u: int((labels == u).sum()) for u in uniq}
    last = list(draw_last or [])
    first = sorted([u for u in uniq if u not in last], key=lambda u: -sizes[u])
    drawn = []

    for u in first + [u for u in last if u in uniq]:
        m = labels == u
        if not m.any():
            continue
        # A thin WHITE stroke separates overlapping points, which is what stops a
        # dense cluster reading as one solid blob.  Alpha stays high enough that
        # the stroke is visible; heavy transparency plus no stroke was why these
        # clusters looked like paint.
        ax.scatter(
            np.asarray(x)[m],
            np.asarray(y)[m],
            s=s,
            linewidths=0.25,
            edgecolors="white",
            rasterized=True,
            color=(palette or {}).get(u, None),
            alpha=0.85 if sizes[u] > 0.25 * len(labels) else 0.95,
            label=(f"{u}  (n={sizes[u]:,})" if show_counts else str(u)),
        )
        drawn.append(u)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(S.strip_panel_prefix(title))

    # Where the legend goes is decided by MEASURED free space, not by the number
    # of groups.  The group count was the previous rule (>4 -> below the axes),
    # and it is the wrong variable: a four-cluster trajectory embedding fills the
    # plane by construction, so it took the in-axes branch and put the legend on
    # top of a cluster.  Ask how many points the emptiest corner holds instead,
    # and drop below the axes whenever no corner is actually free.
    n_lab = len(ax.get_legend_handles_labels()[0])
    # The title owns the top strip, so an upper corner collides with text even
    # when it is empty of data -- exclude those corners from the CANDIDATES
    # rather than choosing one and then substituting the corner beneath it.  That
    # substitution is what made a rising diagonal, whose upper-left is empty,
    # fall back to an outside legend: it moved the label into the band.
    allow = ["lower left", "lower right"] if title else None
    # Probe a box the size the LEGEND actually needs, not a fixed 30%.  A 4-entry
    # key on a short panel is far taller than 30% of the axes, so a corner that
    # measures empty at frac=0.30 can still be covered end to end -- which is how
    # the legend landed back on a cluster after the gate was already "fixed".
    # ~0.085 axes-fraction per row plus padding, capped so the probe cannot grow
    # to the whole panel and declare every corner crowded.
    frac = float(np.clip(0.20 + 0.085 * n_lab, 0.25, 0.55))
    loc, n_in_corner = S.corner_occupancy(
        np.asarray(x, float), np.asarray(y, float), frac=frac, allow=allow
    )
    # 2% of the cloud is the tolerance: a handful of stray points behind a
    # frameless legend is legible, a cluster is not.
    crowded = n_in_corner > max(3, 0.02 * len(labels))
    if legend_loc is not None:
        ax.legend(
            fontsize=7,
            frameon=False,
            loc=legend_loc,
            markerscale=2.2,
            handletextpad=0.3,
            labelspacing=0.3,
            framealpha=0.0,
            borderaxespad=0.15,
        )
    elif n_lab > 4 or crowded:
        # Beside the panel, not beneath it.  A row of entries under a narrow
        # (~W2/3) panel does not fit and spills sideways across the whole figure,
        # so a PANEL key ends up reading as a FIGURE key -- and it sits far from
        # the data it explains.  A single column on the right stays attached to
        # its own panel at any panel width, which is why journals use it for
        # embedding keys.
        ax.legend(
            fontsize=6.5,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            ncol=1,
            markerscale=2.0,
            handletextpad=0.3,
            labelspacing=0.35,
            borderpad=0.0,
            borderaxespad=0.0,
        )
    else:
        ax.legend(
            fontsize=7,
            frameon=False,
            loc=loc,
            markerscale=2.2,
            handletextpad=0.3,
            labelspacing=0.3,
            framealpha=0.0,
            borderaxespad=0.15,
        )
    S.despine(ax)
    return {"order": drawn, "n": sizes}


# ---------------------------------------------------------------------------
# 3. Signed edge split
# ---------------------------------------------------------------------------


def signed_edge_split(
    ax,
    edge_df,
    axis_name="tau",
    n=10,
    palette=None,
    name_col="edge_name",
    side_labels=("root", "mature"),
    title=None,
):
    """Diverging bars: edges pulling to the root end versus the mature end.

    ``edge_axis_contributions`` squares the projection, so ``contrib_*`` is
    sign-blind -- the direction lives only in ``signed_mean_{axis}``.  This panel
    takes the top ``n`` by contribution **within each sign**, so both ends of the
    axis are represented; ranking by contribution alone would fill the panel with
    whichever end happens to dominate.

    Parameters
    ----------
    side_labels
        ``(negative_end, positive_end)``.  Names the two ends in the ``side``
        column, the title and the palette lookup, so a caller whose axis is a
        real transition can say what it is -- e.g.
        ``("early activation", "late activation")``.  These were previously
        hardcoded to ``root``/``mature``, which mislabels any axis that is not a
        differentiation trajectory.

    Returns
    -------
    DataFrame
        The rows drawn, in draw order, with a ``side`` column naming the end.
    """
    sc_ = f"signed_mean_{axis_name}"
    cc = f"contrib_{axis_name}"
    cc = cc if cc in edge_df.columns else "contrib_total"
    if sc_ not in edge_df.columns:
        raise KeyError(f"{sc_!r} not in edge table (have: {list(edge_df.columns)})")

    neg = edge_df[edge_df[sc_] < 0].nlargest(n, cc)
    pos = edge_df[edge_df[sc_] > 0].nlargest(n, cc)
    # Root-end edges on top, mature-end below, so the vertical order matches the
    # left-to-right order the axis label states.  barh draws y=0 at the BOTTOM,
    # so the frame is built mature-first and the y ticks are not inverted.
    d = pd.concat(
        [pos.sort_values(sc_, ascending=False), neg.sort_values(sc_, ascending=False)]
    ).copy()
    if d.empty:
        ax.text(
            0.5,
            0.5,
            "no signed edges",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=S.MID,
        )
        S.no_ticks(ax)
        return d

    # Label the side explicitly.  The caller should not have to re-derive it
    # from the sign of a column whose name depends on `axis_name`.
    lo, hi = (str(side_labels[0]), str(side_labels[1]))
    d["side"] = np.where(d[sc_].values < 0, lo, hi)
    y = np.arange(len(d))
    cols = [
        (
            (palette or {}).get(lo, S.BLUE)
            if v < 0
            else (palette or {}).get(hi, S.VERMILLION)
        )
        for v in d[sc_].values
    ]
    ax.barh(y, d[sc_].values, color=cols, height=0.72, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(d[name_col].astype(str).values)
    ax.axvline(0, color=S.INK, lw=0.6)
    ax.set_xlabel(f"signed mean projection on {axis_name}")
    # Separator between the sign blocks: without it the two groups read as one
    # ranked list, which is what the contribution column is NOT.
    if len(neg) and len(pos):
        ax.axhline(len(pos) - 0.5, color=S.MID, lw=0.6, ls=(0, (3, 2)))
    # Title names the panel; the two sides are carried by a LEGEND, because the
    # blue/orange split is a colour encoding and an arrow glyph in a title is not
    # a key.  Previously this ran to 60+ characters and was clipped at the figure
    # edge.
    # pad clears the one-row key placed just above the frame below.
    ax.set_title(
        S.strip_panel_prefix(title) or f"top {n} edges per sign", loc="left", pad=12.5
    )
    from matplotlib.patches import Patch

    # Resolve the two swatches the SAME way `cols` above resolves them, so the
    # key cannot drift from the bars it explains.
    c_lo = (palette or {}).get(lo, S.BLUE)
    c_hi = (palette or {}).get(hi, S.VERMILLION)
    # Above the frame, not inside it: the bars span the full width at both ends,
    # so every in-axes corner is over data (it landed on the MAPK bar).
    #
    # LEFT-anchored, one row, and BELOW the title.  Right-anchoring put it at the
    # axes' right edge, which on a panel with long y-tick labels sits near the
    # figure's centre -- so a panel-level key read as a figure-level one.  The
    # title is short enough now to carry a row beneath it, and the key must be
    # unambiguously attached to THIS panel because the neighbouring panel reuses
    # the same two colours.
    ax.legend(
        handles=[
            Patch(facecolor=c_lo, label=lo, edgecolor="none"),
            Patch(facecolor=c_hi, label=hi, edgecolor="none"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        frameon=False,
        fontsize=7,
        handlelength=0.9,
        handletextpad=0.5,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    S.despine(ax, left=False)
    return d


# ---------------------------------------------------------------------------
# 4. View share
# ---------------------------------------------------------------------------


def view_share(
    axes, contrib, view_col="view", contrib_col="contrib_phi1", palette=None, scale=1e4
):
    """Two bars: a view's total axis contribution, and its contribution per edge.

    **Both panels are required, and the second is the honest one.**  A view's
    share of the axis depends on how many hyperedges it was allowed to build, so
    the total conflates "this view is informative" with "this view is large".
    Whether dividing by edge count reorders the views is a per-fit question, and
    the returned ``rank_changed`` flag is the only trustworthy answer: on the
    Richard 2018 *three*-view fit it is ``False`` (gene 666 edges / 71.6%,
    pathway 168 / 15.4%, tf 204 / 12.9% -- gene leads both ways), whereas on the
    retired four-view fit that included PCA the per-edge ranking *did* differ.
    Read the flag; do not carry a remembered ranking across fits.  Reporting the
    total alone is how a view gets credited for its size.

    Parameters
    ----------
    axes
        Two axes.  Left gets the total, right gets the per-edge mean.
    contrib
        Long-form edge contribution table, one row per hyperedge, carrying a view
        label and a contribution column (as returned by
        ``edge_axis_contributions``).

    Returns
    -------
    dict
        ``table`` (the per-view frame, including ``n_edges``, ``pct`` and
        ``per_edge``), ``order``, and ``rank_changed`` -- True when the per-edge
        ranking differs from the total ranking, which is the thing worth saying
        out loud in the caption.
    """
    axes = np.asarray(axes).ravel()
    if axes.size < 2:
        raise ValueError("view_share needs two axes (total, per-edge)")

    g = contrib.groupby(view_col)[contrib_col]
    total = g.sum().sort_values(ascending=False)
    n_edges = contrib.groupby(view_col).size()
    per_edge = (total / n_edges).rename("per_edge")
    pct = (total / total.sum() * 100).rename("pct")
    tbl = pd.concat(
        [n_edges.rename("n_edges"), total.rename("total"), pct, per_edge], axis=1
    ).loc[total.index]

    order = list(total.index)
    cols = [(palette or {}).get(v, S.BLUE) for v in order]

    ax = axes[0]
    vals = pct.loc[order].values
    # With few categories the SLOT height, not the bar fraction, decides how thick
    # a bar looks: `height` is a fraction of one slot, and three slots in a 1.9 in
    # panel makes each slot ~0.55 in, so even height=0.62 renders a 0.34 in slab.
    # The comment here used to claim the category axis was padded to a floor; it
    # was not (the ylim below is exactly the data range), and the panel showed
    # three slabs.  Cap the DRAWN bar thickness in axes units instead, so the bars
    # stay journal-thin at any category count while the slots keep their spacing.
    # Target ~0.20 in of drawn bar regardless of how many categories there are.
    # Slot height in inches is (panel height / n_categories), so the fraction that
    # yields a fixed thickness is (0.20 * n_categories / panel_height), clipped so
    # a many-category panel never exceeds the usual 0.62 of its slot.
    _ph = float(ax.get_figure().get_size_inches()[1]) or 1.0
    bar_h = float(np.clip(0.20 * max(len(order), 1) / _ph, 0.12, 0.62))
    ax.barh(range(len(order)), vals, color=cols, height=bar_h, edgecolor="none")
    for i, v in enumerate(vals):
        ax.text(v + 0.02 * max(vals), i, f"{v:.1f}%", va="center", fontsize=7)
    ax.set_yticks(range(len(order)))
    # Carry the edge count on the tick, so the confound is visible in the panel
    # it applies to rather than only in the one that corrects it.
    ax.set_yticklabels([f"{v}\n({int(n_edges[v])} edges)" for v in order], fontsize=7)
    ax.set_ylim(len(order) - 0.5, -0.5)  # also inverts: largest at top
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_xlabel(r"% of $\varphi_1$ contribution")
    ax.set_title("total share of the axis", loc="left")
    S.despine(ax, left=False)

    ax = axes[1]
    pe = per_edge.loc[order].values * scale
    ax.barh(range(len(order)), pe, color=cols, height=bar_h, edgecolor="none")
    for i, v in enumerate(pe):
        ax.text(v + 0.02 * max(pe), i, f"{v:.2f}", va="center", fontsize=7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(0, max(pe) * 1.22)
    exp = int(round(np.log10(scale)))
    ax.set_xlabel(rf"mean contribution per edge ($\times10^{{-{exp}}}$)")
    ax.set_title("contribution per edge", loc="left")
    S.despine(ax, left=False)

    per_edge_order = list(per_edge.sort_values(ascending=False).index)
    return {
        "table": tbl,
        "order": order,
        "per_edge_order": per_edge_order,
        "rank_changed": per_edge_order != order,
    }


# ---------------------------------------------------------------------------
# 5. Onset against a background band
# ---------------------------------------------------------------------------


def onset_vs_background(
    ax,
    rows,
    background,
    value_col="onset",
    label_col="label",
    group_col=None,
    palette=None,
    p_values=None,
    xlabel="onset (pseudotime)",
    title=None,
    band_label="background IQR",
):
    """Dot rows against the background distribution's IQR band.

    A panel of onsets with no reference invites the reader to treat any spread as
    signal.  The band is the IQR of ``background`` with its median as a line, so
    "earlier than the bulk" is a visual claim with a scale behind it.

    ``p_values`` are annotated per row when given, and are expected to be
    **already corrected** -- this panel does no multiple-testing correction and
    will not pretend to.  Rows are drawn in the order supplied, so sort before
    calling if rank matters.

    Returns
    -------
    dict
        ``labels`` in draw order, the ``band`` (q25, median, q75), and ``n_bg``.
    """
    bg = np.asarray(background, float)
    bg = bg[np.isfinite(bg)]
    if bg.size == 0:
        raise ValueError("background is empty or all-NaN")
    q25, med, q75 = (
        float(np.percentile(bg, 25)),
        float(np.median(bg)),
        float(np.percentile(bg, 75)),
    )

    ax.axvspan(q25, q75, color=S.FAINT, alpha=0.55, lw=0, zorder=0, label=band_label)
    ax.axvline(
        med, color=S.MID, lw=0.8, ls=(0, (3, 2)), zorder=1, label="background median"
    )

    labels = [str(v) for v in rows[label_col].values]
    y = np.arange(len(labels))
    if group_col is not None and group_col in rows.columns:
        cols = [(palette or {}).get(str(g), S.BLUE) for g in rows[group_col].values]
    else:
        cols = [S.BLUE] * len(labels)

    ax.scatter(rows[value_col].values, y, s=22, color=cols, zorder=3, edgecolors="none")
    if group_col is not None and group_col in rows.columns:
        from matplotlib.lines import Line2D

        seen = {}
        for g, c in zip(rows[group_col].values, cols):
            seen.setdefault(str(g), c)
        # Register handles ONLY.  The single ax.legend() below draws them --
        # calling legend() here as well would be replaced by that one, silently
        # dropping the group colours again.
        for g, c in seen.items():
            ax.plot([], [], "o", ms=4.5, color=c, ls="none", label=g)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # first row at top; must precede any set_ylim

    if p_values is not None:
        pv = np.asarray(p_values, float)
        xmax = float(np.nanmax(rows[value_col].values))
        for i, p in enumerate(pv):
            if np.isfinite(p):
                ax.annotate(
                    f"q={p:.3g}" if p >= 1e-3 else f"q={p:.1e}",
                    (xmax, i),
                    textcoords="offset points",
                    xytext=(6, 0),
                    va="center",
                    fontsize=6,
                    color=S.INK if p < 0.05 else S.MID,
                )

    ax.set_xlabel(xlabel)
    # Above the frame (the dots reach every corner), right-anchored so it sits
    # beside the left-aligned title.  Cap at 2 columns: four entries on one row
    # spans the full width and lands on the title.
    n_lab = len(ax.get_legend_handles_labels()[0])
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=min(n_lab, 2),
        frameon=False,
        fontsize=7,
        handlelength=0.9,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    # The title has to clear the legend rows now stacked above the frame.
    if title:
        ax.set_title(
            S.strip_panel_prefix(title),
            loc="left",
            pad=3.0 + 9.5 * int(np.ceil(n_lab / 2)),
        )
    S.despine(ax, left=False)
    return {"labels": labels, "band": (q25, med, q75), "n_bg": int(bg.size)}


# ---------------------------------------------------------------------------
# 6. Gene x pseudotime-bin heatmap
# ---------------------------------------------------------------------------


def axis_heatmap(
    ax,
    expression,
    axis,
    genes,
    n_bins=20,
    min_cells=20,
    min_detect=0.05,
    sign_of=None,
    curated=None,
    scale="minmax",
    signed_values=False,
    cbar_label=None,
    row_noun="gene edges",
    cmap=None,
    axis_name="pseudotime",
    colorbar=True,
    mode="smooth",
    n_grid=200,
    frac=0.08,
    min_local=8,
):
    """Gene x pseudotime cascade, as a smooth gradient or as masked bins.

    ``mode="smooth"`` (the default) evaluates every row with a Gaussian-kernel
    local mean on an evenly spaced grid of ``n_grid`` points, so the cascade reads
    as a continuous gradient instead of a row of discrete blocks.  ``frac`` is the
    kernel width as a fraction of the axis range.  Two properties of the binned
    mode are deliberately preserved rather than smoothed away:

    * the grid is **evenly spaced in the axis**, so the x-axis stays linear and
      equal pixel widths still mean equal axis widths;
    * support is still enforced, not interpolated over.  A grid point with fewer
      than ``min_local`` cells within one kernel width is masked and drawn grey,
      exactly as a thin bin is.  This is what stops the smoother inventing a
      smooth ramp across a genuine hole in the data -- the failure mode that makes
      a gap look like a gradient.

      The guard counts **nearby cells**, not Kish's effective sample size, and the
      difference is not academic: measured on a synthetic axis with a deliberate
      hole, effective-n inside the hole was 80 against 87 in dense regions -- the
      two are indistinguishable, so no effective-n threshold can detect the gap.
      Effective-n measures how concentrated the kernel weights are, and a kernel
      reaching symmetrically across a hole from both sides looks perfectly
      healthy by that measure.  Cells within one width separates 0 from 40 on the
      same data.  This is also why ``frac`` defaults to 0.08 rather than wider: a
      kernel wide enough to bridge a gap makes the gap undetectable at any
      threshold.

    ``mode="bins"`` keeps the original equal-width binned matrix, which is the
    honest choice when you want to *see* the support rather than a fitted surface.

    Binning is **equal width**, not quantile, and this is deliberately the
    opposite of what marker profiles do.  Measured reason: pseudotime here is
    heavily right-skewed, so 20 *quantile* bins put 18 bins below the midpoint
    and collapse every mature cell into the last one -- a fate fraction then
    reads 2-4% across 18 bins and 75% in bin 20, and the mature end, which is the
    entire point of a cascade figure, has no resolution.  With equal-width bins
    the same fraction climbs 2% -> 13% -> 44% -> 92% across seven bins and the
    gradient is visible.  Equal-width bins also make the x-axis linear in the
    axis, so equal cell widths mean equal axis widths.

    The cost is sparse tail bins, and it is handled honestly rather than hidden:
    bins under ``min_cells`` are masked and drawn grey, so a thin bin reads as
    absent rather than being interpolated over.

    The detection floor is **per bin, not global**.  A global floor is the wrong
    test here: measured, fate-specific genes sit at 2-4% detection overall yet
    38-63% in their own peak bin, because they are specific to a fate holding a
    few percent of cells.  A global floor therefore deletes the entire mature arm
    and leaves a one-sided figure that looks like an artefact.  A row is kept if
    **any** usable bin reaches ``min_detect``.  Dropped rows are **named** in the
    return value -- a silent filter reads as biological absence.

    ``scale`` controls the row transform and therefore the colour map:

    ``"minmax"``
        rows scaled to [0, 1]; a sequential ramp is correct.
    ``"zscore"``
        rows z-scored across bins; the quantity is now **diverging about 0**, so
        a sequential ramp would misrepresent it and a diverging one is used.

    Parameters
    ----------
    expression
        ``(n_cells, n_genes)`` dense or sparse matrix, column-aligned to ``genes``.
    sign_of
        Optional ``{gene: sign}``.  When given, rows are blocked by sign and
        ordered by peak bin within each block; ordering by contribution gives a
        visually random cascade, whereas peak position is derived rather than
        chosen.
    curated
        Genes to mark with a leading asterisk *if they also appear* in the
        supplied rows.  Annotation of an overlap, never selection by curation.
    row_noun
        What one row IS, used in the auto-title (default ``"gene edges"``).  Pass
        e.g. ``"TF regulons"`` for an activity matrix; the default mislabels every
        matrix that is not gene expression.
    cbar_label
        Override the colourbar label.  The default names the row transform, and
        for ``scale="minmax"`` it also says "expression" -- which is wrong for an
        activity matrix, so pass this whenever the values are not expression.
    signed_values
        Set True when the matrix holds **activity scores rather than expression**
        (TF footprints, PROGENy pathway scores), where negative means repressed
        rather than absent.  It switches the ``min_detect`` test from ``> 0`` to
        ``!= 0``, without which a uniformly repressed row measures 0% detected and
        is dropped.  Pair it with ``scale="zscore"``, since such a quantity is
        diverging about zero and a sequential ramp misstates it.

    Returns
    -------
    dict
        ``rows``, ``matrix``, ``bin_n``, ``centres``, ``mode``,
        ``dropped_low_detection``, ``signs``, ``curated_hits``.

        ``bin_n`` counts cells per bin in ``"bins"`` mode but holds the number of
        cells within one kernel width of each grid point in ``"smooth"`` mode
        (length ``n_grid``), so a threshold written against one mode does not
        carry over to the other unchanged.
    """
    X = expression
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    genes = [str(g) for g in genes]
    if X.shape[1] != len(genes):
        raise ValueError(
            f"expression has {X.shape[1]} columns but "
            f"{len(genes)} gene names were given"
        )

    t = np.asarray(axis, float)
    edges = np.linspace(t.min(), t.max(), n_bins + 1)
    binid = np.clip(np.digitize(t, edges[1:-1]), 0, n_bins - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_n = np.array([int((binid == b).sum()) for b in range(n_bins)])
    usable = [b for b in range(n_bins) if bin_n[b] >= min_cells]
    if not usable:
        # `min_cells` governs which BINS may be drawn, so in smooth mode it must
        # not be able to abort the call -- the smoother never draws those bins.
        # The bins are still needed for the per-bin detection floor below, so fall
        # back to the fullest ones rather than dropping the floor entirely.
        if mode == "smooth":
            usable = [int(np.argmax(bin_n))] if bin_n.max() > 0 else []
        if not usable:
            raise ValueError(
                f"no bin reaches min_cells={min_cells} "
                f"(largest holds {bin_n.max()})"
            )

    # Peak-bin detection, not global -- see the docstring.  A fate-specific gene
    # is rare globally by definition; that is the signal, not grounds to drop it.
    #
    # `signed_values` changes what "detected" MEANS.  For counts, `> 0` is right.
    # For an activity score (a TF footprint, a PROGENy pathway score) zero is the
    # middle of the scale, not its floor, so `> 0` would score a consistently
    # REPRESSED regulator as 0% detected and drop it -- deleting exactly the
    # rows whose down-regulation is the finding.  Test departure from zero instead.
    _obs = (lambda v: np.abs(v) > 0) if signed_values else (lambda v: v > 0)
    peak_detect = np.array(
        [
            max((_obs(X[binid == b, j]).mean() for b in usable), default=0.0)
            for j in range(len(genes))
        ]
    )
    keep = peak_detect >= min_detect
    dropped = [
        (g, round(float(d), 3)) for g, d, k in zip(genes, peak_detect, keep) if not k
    ]
    genes = [g for g, k in zip(genes, keep) if k]
    X = X[:, keep]
    if not genes:
        raise ValueError(
            f"no candidate gene reaches min_detect={min_detect} in any bin"
        )

    if mode == "smooth":
        # Gaussian-kernel local mean per row on an even grid.  Vectorised over
        # genes: K is (n_grid, n_cells), so K @ X is every row at once.
        span = float(t.max() - t.min()) or 1.0
        h = max(float(frac) * span, 1e-9)
        grid = np.linspace(t.min(), t.max(), int(n_grid))
        d = np.abs(grid[:, None] - t[None, :])
        K = np.exp(-0.5 * (d / h) ** 2)
        sw = K.sum(1)
        # Locality, not kernel concentration -- see the docstring.  A kernel
        # spanning a hole from both sides has a healthy effective-n but no cells
        # actually near the grid point.
        n_local = (d <= h).sum(1)
        M = (K @ X).T / np.where(sw > 0, sw, np.nan)
        M[:, n_local < int(min_local)] = np.nan
        n_col = int(n_grid)
        centres = grid
        bin_n = n_local
        x_note = "kernel-smoothed"
    elif mode == "bins":
        M = np.full((len(genes), n_bins), np.nan)
        for b in usable:
            M[:, b] = X[binid == b].mean(0)
        n_col = int(n_bins)
        x_note = "equal-width bins, centre shown"
    else:
        raise ValueError(f"mode must be 'smooth' or 'bins', got {mode!r}")

    if scale == "zscore":
        mu = np.nanmean(M, 1, keepdims=True)
        sd = np.nanstd(M, 1, keepdims=True)
        Z = (M - mu) / np.where(sd > 0, sd, np.nan)
        vmax = float(np.nanmax(np.abs(Z))) if np.isfinite(Z).any() else 1.0
        cm = S.masked_cmap(cmap or S.DIV)
        kw = {"vmin": -vmax, "vmax": vmax}
        cbar_label = cbar_label or "row z-score across bins"
    else:
        lo = np.nanmin(M, 1, keepdims=True)
        hi = np.nanmax(M, 1, keepdims=True)
        Z = (M - lo) / np.where(hi - lo > 0, hi - lo, np.nan)
        cm = S.masked_cmap(cmap or S.SEQ)
        kw = {"vmin": 0, "vmax": 1}
        cbar_label = cbar_label or "row min-max scaled mean expression"

    # Order by peak bin, blocked by sign when signs are supplied.
    peak = np.array([np.nanargmax(r) if np.isfinite(r).any() else n_col for r in Z])
    sgn = {g: int((sign_of or {}).get(g, 0)) for g in genes}
    idx = sorted(range(len(genes)), key=lambda i: (sgn[genes[i]], peak[i]))
    Z, genes = Z[idx], [genes[i] for i in idx]

    # "nearest" in smooth mode too: the grid is already fine, and bilinear
    # interpolation would additionally blur ACROSS rows, mixing one gene's
    # expression into its neighbour's.
    im = ax.imshow(
        np.ma.masked_invalid(Z), aspect="auto", cmap=cm, interpolation="nearest", **kw
    )
    cur = set(map(str, curated or []))
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels([("*" + g) if g in cur else g for g in genes])
    # An unexplained glyph is not an annotation.  The `*` marked curated rows on
    # every previous render with nothing on the figure saying so.
    if cur & set(map(str, genes)):
        ax.set_ylabel(
            "* named in the curated panels", fontsize=7, color=S.GREY, labelpad=4
        )
    n_tick = 6
    tick_at = np.linspace(0, n_col - 1, n_tick).round().astype(int)
    ax.set_xticks(tick_at)
    # Tick precision from the SPAN, not a hardcoded 2 dp.  A 0-1 pseudotime read
    # "0.00 0.20 0.40" -- trailing zeros that carry no information and read as
    # spurious precision -- while an axis spanning 0.01 needs more than 2 dp to
    # distinguish its ticks at all.  Enough digits to separate adjacent ticks,
    # capped so a degenerate axis cannot produce a wall of digits.
    _span = float(np.nanmax(centres) - np.nanmin(centres))
    _step = _span / max(1, n_tick - 1)
    _dp = 1 if _step >= 0.1 else min(6, int(np.ceil(-np.log10(_step))) + 1)
    ax.set_xticklabels([f"{centres[b]:.{_dp}f}" for b in tick_at])
    ax.set_xlabel(f"{axis_name} ({x_note})")
    # No cell grid in smooth mode -- 200 vertical gridlines would be a solid
    # block -- but ROW separators are still needed there: a min-max ramp starts at
    # white, so where two neighbouring rows are both low at the same x they merge
    # into one band and a row boundary is indistinguishable from a zero value.
    if mode == "bins":
        S.heatmap_grid(ax, len(genes), n_col)
    else:
        # FAINT GREY, not white: a white rule is invisible against the light end
        # of the ramp, which is exactly the case that needed separating.
        #
        # Gate on the DRAWN row height, not the row count.  The old gate was
        # `len(genes) <= 40`, which passed a 24-row panel 3.8 in tall -- rows there
        # are ~0.13 in, so 23 rules read as a texture laid over the gradient rather
        # than as separators.  Separators only help once a row is thick enough to
        # read as a band in its own right; below that the rows already separate by
        # value and the rules are pure ink.
        _hin = float(ax.get_figure().get_size_inches()[1]) or 1.0
        if (_hin / max(len(genes), 1)) >= 0.18:
            for k in range(1, len(genes)):
                ax.axhline(k - 0.5, color=S.FAINT, lw=0.4, zorder=4)
    if mode == "smooth":
        thin = int((bin_n < int(min_local)).sum())
        # Only mention the grey-gap rule when a gap ACTUALLY EXISTS.  Stating
        # "0% of the axis" describes a condition that never fired and spends the
        # title on it; the kernel width belongs in the caption, not here.
        gap = (
            (
                f"   (grey = <{min_local} cells per kernel width: "
                f"{100 * thin / n_col:.0f}% of the axis)"
            )
            if thin
            else ""
        )
        # "ordered by onset" was WRONG: the row sort above is by PEAK POSITION,
        # and it discards whatever order the caller passed in.  A title must not
        # assert an ordering the same function overrides.  `row_noun` replaces the
        # hardcoded "gene edges", which mislabelled every non-expression matrix.
        ax.set_title(f"{len(genes)} {row_noun}, ordered by peak{gap}", loc="left")
    else:
        thin = int((bin_n < min_cells).sum())
        gap = (f"   (grey = <{min_cells} cells: {thin} bin(s))") if thin else ""
        ax.set_title(f"{len(genes)} {row_noun} x {n_col} bins{gap}", loc="left")
    if colorbar:
        # `shrink` matters as much as `fraction`: without it the bar spans the
        # full axes height and, on a panel whose height is set by its row count,
        # ends up TALLER than the heatmap it describes -- a legend outsizing its
        # data.  0.72 keeps it clearly subordinate at any row count.
        ax.figure.colorbar(
            im, ax=ax, fraction=0.03, pad=0.02, shrink=0.72, label=cbar_label
        )
    return {
        "rows": genes,
        "matrix": Z,
        "bin_n": bin_n,
        "centres": centres,
        "mode": mode,
        "dropped_low_detection": dropped,
        "signs": [sgn[g] for g in genes],
        "curated_hits": [g for g in genes if g in cur],
    }


# ---------------------------------------------------------------------------
# 7. Top features
# ---------------------------------------------------------------------------


def top_features(
    ax,
    df,
    value_col,
    label_col,
    n=10,
    fdr_col=None,
    fdr=0.05,
    group_col=None,
    palette=None,
    color=None,
    xmax=None,
    xlabel=None,
    title=None,
):
    """Ranked horizontal bars for the strongest features, FDR-gated.

    Rows below the gate are dropped and the surviving count is annotated, so a
    sparse panel reads as "few survived" rather than "few tested".

    Pass ``xmax`` when drawing several of these side by side, and pass the
    **same** value to each.  Per-panel autoscaling makes equal bar lengths mean
    unequal values (measured across treatment arms: one tops at 2.2, another at
    3.2), which invites exactly the cross-panel comparison a free axis cannot
    support.

    Returns
    -------
    DataFrame
        The rows drawn, in draw order.
    """
    d = df
    n_tested = len(d)
    if fdr_col is not None and fdr_col in d.columns:
        d = d[d[fdr_col] < fdr]
    n_pass = len(d)
    d = d.sort_values(value_col, ascending=False).head(n).copy()

    if d.empty:
        msg = f"no feature at {fdr_col}<{fdr}" if fdr_col else "nothing to plot"
        ax.text(
            0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color=S.MID
        )
        S.no_ticks(ax)
        return d

    y = np.arange(len(d))[::-1]
    if group_col is not None and group_col in d.columns:
        cols = [(palette or {}).get(str(g), S.BLUE) for g in d[group_col]]
    else:
        cols = color or S.BLUE
    ax.barh(y, d[value_col].values, color=cols, height=0.72, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(d[label_col].astype(str).values)
    ax.axvline(0, color=S.INK, lw=0.6)
    ax.set_xlabel(xlabel or value_col)
    if xmax is not None:
        ax.set_xlim(0, xmax)
    if title is not None:
        gate = (
            f"  (n={n_pass:,d} of {n_tested:,d} at {fdr_col}<{fdr})"
            if fdr_col
            else f"  (top {len(d)} of {n_tested:,d})"
        )
        ax.set_title(f"{S.strip_panel_prefix(title)}{gate}", loc="left")
    S.despine(ax, left=False)
    return d


# ---------------------------------------------------------------------------
# Supporting: programme profiles along the axis
# ---------------------------------------------------------------------------


def profiles(
    ax,
    centres,
    curves,
    palette=None,
    title=None,
    xlabel="pseudotime (quantile-bin centres)",
    ylabel="programme (z, mean $\\pm$ SEM)",
    legend_ncol=2,
):
    """Trend lines with SEM ribbons along a binned axis.

    ``curves`` maps name -> ``(mean, sem)``, as returned by
    :func:`pseudoembed.plotting.composites.profile_along_axis`.
    """
    for nm, (mean, sem) in curves.items():
        c = (palette or {}).get(nm)
        (ln,) = ax.plot(centres, mean, "-o", ms=3, lw=1.4, color=c, label=nm)
        # Take the colour BACK off the line rather than reusing `c`: when no
        # palette is given `c` is None, and passing None to fill_between draws
        # every ribbon in the first cycle colour -- so an orange curve got a blue
        # ribbon and the two stopped corresponding.
        ax.fill_between(
            centres,
            np.asarray(mean) - np.asarray(sem),
            np.asarray(mean) + np.asarray(sem),
            color=ln.get_color(),
            alpha=0.25,
            lw=0,
        )
    # A dotted MID rule at zero was almost invisible at print size; a faint solid
    # hairline reads as a reference without competing with the series.
    ax.axhline(0, color=S.FAINT, lw=0.7, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # The legend now sits directly above the frame, so the title has to clear it.
    # A key ABOVE the frame costs a title-pad tall enough to clear it, and in a
    # multi-panel composite that pad pushes the title into the row the bold panel
    # letter occupies -- the two then sit side by side and the letter stops
    # reading as panel identity.  So only go outside when the key genuinely does
    # not fit inside: at 1-2 entries the headroom below is enough, and inside is
    # strictly better because it stays attached to its own curves.
    n_lab0 = len(ax.get_legend_handles_labels()[0])
    nrow0 = int(np.ceil(n_lab0 / max(int(legend_ncol), 1)))
    outside = n_lab0 > 2
    # `titlepad` in points: one legend row is ~9 pt at fontsize 7.
    if title:
        ax.set_title(
            S.strip_panel_prefix(title), pad=(3.0 + 9.5 * nrow0) if outside else 3.0
        )
    # `best` only optimises against the artists already drawn, so with many
    # z-scored crossing curves there is no empty region left and it collides
    # whatever it picks -- it landed on the 1 h peak once.  Make room instead:
    # add headroom above the data and put the legend in it.  This is the "empty
    # corner by construction" fix, not another placement guess.
    # Size the headroom from the DATA, not from the current ylim.  Under
    # `sharey` the panels share one axis, so reading `get_ylim()` here means the
    # second panel expands a range the first has already expanded, and the
    # headroom compounds per panel -- measured: a 2-panel z-scored figure ran to
    # y=5 for data topping out at 1.6, leaving over half the axes empty.
    # Reserving headroom INSIDE the axes still failed at 8 entries: four legend
    # rows either overlapped the 1 h peak or pushed the curves into the bottom
    # third of the panel.  Put the key outside the frame instead -- then no amount
    # of headroom arithmetic can collide with the data, and the y-axis shows only
    # the data's own range.
    ys = (
        np.concatenate([np.asarray(m, float).ravel() for m, _ in curves.values()])
        if curves
        else np.zeros(1)
    )
    ys = ys[np.isfinite(ys)]
    d_lo, d_hi = (float(ys.min()), float(ys.max())) if ys.size else (0.0, 1.0)
    span = (d_hi - d_lo) or 1.0
    lo, hi = ax.get_ylim()
    # Never shrink a shared axis below what a sibling panel already needs.
    ax.set_ylim(min(lo, d_lo - 0.06 * span), max(hi, d_hi + 0.06 * span))
    if outside:
        ax.legend(
            fontsize=7,
            frameon=False,
            ncol=legend_ncol,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            handlelength=1.3,
            columnspacing=1.1,
            borderaxespad=0.0,
            handletextpad=0.5,
        )
    else:
        # Reserve headroom for the in-axes key so it cannot land on a peak, then
        # place it in that reserved strip.  Two rows of headroom is ~0.16 of the
        # span at these font sizes.
        lo2, hi2 = ax.get_ylim()
        ax.set_ylim(lo2, max(hi2, d_hi + 0.30 * span))
        ax.legend(
            fontsize=7,
            frameon=False,
            ncol=legend_ncol,
            loc="upper left",
            handlelength=1.3,
            columnspacing=1.1,
            borderaxespad=0.2,
            handletextpad=0.5,
            labelspacing=0.3,
        )
    S.despine(ax)
    return {"names": list(curves)}


# ---------------------------------------------------------------------------
# 9. Scatter with a trend
# ---------------------------------------------------------------------------


def scatter_trend(
    ax,
    x,
    y,
    *,
    trend="smooth",
    n_bins=12,
    s=6,
    color=None,
    trend_color=None,
    xlabel=None,
    ylabel=None,
    title=None,
    annotate_rho=True,
    min_bin=3,
    frac=0.10,
    band=True,
):
    """Scatter of ``y`` against ``x`` with a trend line and Spearman rho.

    A held-out measurement plotted against a fitted axis is the commonest
    validation panel there is, and an overplotted cloud makes the reader guess at
    the relationship.  The trend states it.

    ``trend`` selects the line:

    ``"smooth"`` (the default)
        Local linear regression with a Gaussian kernel -- a continuous curve, so
        there are no visible bin steps, but it still **bends**.  That matters
        here: on the Richard 2018 proteins a straight fit is measurably wrong,
        with :math:`R^2` rising 0.155 -> 0.287 (CD62L) and 0.768 -> 0.833 (CD69)
        from a linear to a quadratic fit, because CD62L falls steeply then
        flattens.  A line that cannot bend reports those panels as weaker and
        more linear than they are.  ``band=True`` adds a pointwise SE ribbon.

        ``frac`` is the Gaussian kernel width as a fraction of the x-range.  The
        default 0.10 was calibrated against three synthetic curves (n=300, noise
        0.05): recovery error grows monotonically with ``frac`` for anything that
        bends -- on a plateau, peak absolute error goes 0.09 (frac 0.08) -> 0.13
        (0.12) -> 0.26 (0.25) -> 0.33 (0.50).  Widen it only for a noisy or small
        sample, and know that doing so straightens real curvature.
    ``"linear"``
        Least-squares straight line.  Correct only when the relationship really
        is linear -- check before using it.
    ``"binned"``
        Quantile-binned mean with an SEM ribbon.  Kept because it is the honest
        display when you want the *data* summarised rather than a model fitted,
        and it makes support explicit: bins under ``min_bin`` points are dropped
        rather than drawn as a noisy mean.  Quantile (not equal-width) bins match
        :func:`pseudoembed.plotting.composites.bin_by_axis`; the binning is
        duplicated here rather than imported because ``composites`` imports this
        module, so the reverse direction would be a circular import.
    ``None``
        Scatter only.

    LOWESS is deliberately absent: ``statsmodels`` is not a declared dependency.
    The kernel smoother here needs only numpy and is fitted on the same points.

    Returns
    -------
    dict
        ``rho``, ``p``, ``n``, the ``trend`` used, the ``centres``/``mean``/``sem``
        actually drawn (the curve's sample points for ``"smooth"``), and
        ``dropped_thin_bins``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    out = {
        "n": int(ok.sum()),
        "rho": float("nan"),
        "p": float("nan"),
        "trend": trend,
        "centres": np.zeros(0),
        "mean": np.zeros(0),
        "sem": np.zeros(0),
        "dropped_thin_bins": 0,
    }
    if len(x) < 3:
        ax.text(
            0.5,
            0.5,
            "too few finite points",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=S.MID,
        )
        S.no_ticks(ax)
        return out

    r = spearmanr(x, y)
    out["rho"], out["p"] = float(r.statistic), float(r.pvalue)

    ax.scatter(x, y, s=s, c=color or S.BLUE, alpha=0.7, linewidths=0, zorder=2)

    tcol = trend_color or S.VERMILLION
    if trend == "smooth":
        # Local linear (degree-1) kernel regression.  Degree 1 rather than a
        # kernel *mean*: a Nadaraya-Watson mean flattens the ends of a monotone
        # trend inward, which on CD62L reads as the decline stopping early.
        span = float(x.max() - x.min())
        if span <= 0:
            span = 1.0
        h = max(float(frac) * span, 1e-9)
        xs = np.linspace(x.min(), x.max(), 120)
        mu = np.full(xs.shape, np.nan)
        se = np.full(xs.shape, np.nan)
        for i, x0 in enumerate(xs):
            w = np.exp(-0.5 * ((x - x0) / h) ** 2)
            sw = w.sum()
            if sw <= 1e-12:
                continue
            dx = x - x0
            # Weighted least squares for [intercept, slope] about x0, so the
            # fitted value at x0 IS the intercept.
            s0, s1, s2 = sw, (w * dx).sum(), (w * dx * dx).sum()
            t0, t1 = (w * y).sum(), (w * y * dx).sum()
            det = s0 * s2 - s1 * s1
            if abs(det) < 1e-12 * max(1.0, s0 * s2):
                mu[i] = t0 / sw  # degenerate: fall back to the mean
            else:
                mu[i] = (s2 * t0 - s1 * t1) / det
            if band:
                # Pointwise SE of a weighted mean, using the local residual
                # spread.  Effective n is Kish's, so it shrinks where the kernel
                # has little data and the ribbon widens honestly at the ends.
                n_eff = sw * sw / max((w * w).sum(), 1e-12)
                resid = y - mu[i]
                var = (w * resid * resid).sum() / sw
                se[i] = np.sqrt(max(var, 0.0) / max(n_eff, 1.0))
        keep = np.isfinite(mu)
        out["centres"], out["mean"] = xs[keep], mu[keep]
        out["sem"] = se[keep] if band else np.zeros(int(keep.sum()))
        if keep.any():
            if band:
                ax.fill_between(
                    xs[keep],
                    mu[keep] - se[keep],
                    mu[keep] + se[keep],
                    color=tcol,
                    alpha=0.18,
                    linewidth=0,
                    zorder=3,
                )
            ax.plot(xs[keep], mu[keep], "-", color=tcol, lw=1.6, zorder=4)
    elif trend == "binned":
        qs = np.quantile(x, np.linspace(0, 1, int(n_bins) + 1))
        qs[-1] += 1e-9
        binid = np.clip(np.digitize(x, qs[1:-1]), 0, int(n_bins) - 1)
        cen, mu, se = [], [], []
        for b in range(int(n_bins)):
            m = binid == b
            k = int(m.sum())
            if k < int(min_bin):
                out["dropped_thin_bins"] += int(k > 0)
                continue
            cen.append(float(x[m].mean()))
            mu.append(float(y[m].mean()))
            se.append(float(y[m].std() / max(1.0, np.sqrt(k))))
        cen, mu, se = np.asarray(cen), np.asarray(mu), np.asarray(se)
        out["centres"], out["mean"], out["sem"] = cen, mu, se
        if len(cen):
            ax.fill_between(
                cen, mu - se, mu + se, color=tcol, alpha=0.18, linewidth=0, zorder=3
            )
            ax.plot(cen, mu, "-o", color=tcol, lw=1.2, ms=2.6, zorder=4)
    elif trend == "linear":
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b0 + b1 * xs, "-", color=tcol, lw=1.2, zorder=4)
        out["slope"], out["intercept"] = float(b1), float(b0)
    elif trend is not None:
        raise ValueError(
            f"trend must be 'smooth', 'binned', 'linear' or " f"None, got {trend!r}"
        )

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(S.strip_panel_prefix(title), loc="left")
    # Typeset rho, and put it with the data -- in the emptiest corner, MEASURED.
    # A sign rule is not enough here: these protein readouts are zero-inflated,
    # so the corner the trend vacates is exactly where the floor of zeros sits
    # (rho sat on top of the CD44 cloud when this was chosen from the sign).
    if annotate_rho and np.isfinite(out.get("rho", np.nan)):
        S.annotate_stat(ax, S.rho_label(out["rho"]), loc=S.emptiest_corner(x, y))
    S.despine(ax)
    return out
