"""Multi-panel figures and the axis helpers they need.

A composite owns its figure (unlike :mod:`pseudoembed.plotting.panels`, where the
caller does) and returns ``(fig, axes, info)``.  The point of a composite is that
panels built from the *same* stored embedding and axis cannot disagree with each
other -- the recurring failure mode when each panel is hand-rolled is two panels
in one figure computed from different fits.

Also here: the small axis-handling functions the composites need
(:func:`orient_axis`, :func:`bin_by_axis`, :func:`profile_along_axis`,
:func:`panel_score`).  They live beside the figures because that is where they
are used, and they are exported so a notebook can call them directly instead of
reimplementing them -- reimplementing the pipeline in a figure script is how two
confidently wrong figures got published internally.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import panels as P
from . import style as S

__all__ = [
    "orient_axis",
    "bin_by_axis",
    "profile_along_axis",
    "panel_score",
    "group_profiles",
    "profiles_grid",
    "scatter_grid",
    "standard_panels",
]


# ---------------------------------------------------------------------------
# Axis handling
# ---------------------------------------------------------------------------


def orient_axis(v, reference):
    """Flip a spectral axis so it increases with ``reference``.

    Eigenvector signs are arbitrary -- ``phi`` and ``-phi`` are the same axis --
    so any plot or correlation must fix the sign against something external
    before it means anything.

    Returns
    -------
    (oriented, abs_rho) : (ndarray, float)
    """
    r = spearmanr(v, reference).statistic
    if not np.isfinite(r):
        return np.asarray(v, float), float("nan")
    return np.asarray(v, float) * (-1.0 if r < 0 else 1.0), abs(float(r))


def bin_by_axis(axis, n_bins: int = 12):
    """Quantile-bin an axis.  Returns ``(bin_id, bin_centres)``.

    Quantile bins (not equal-width) so every bin holds a comparable number of
    cells; the centre is the observed mean inside the bin, so the last centre is
    typically below the axis maximum.  That is not axis truncation.

    Note the contrast with :func:`pseudoembed.plotting.panels.axis_heatmap`,
    which bins **equal-width** on purpose -- see its docstring.  Quantile bins are
    right for profiles, where each point needs comparable support; equal-width
    bins are right for a cascade, where the x-axis must be linear in the axis.
    """
    axis = np.asarray(axis, float)
    qs = np.quantile(axis, np.linspace(0, 1, n_bins + 1))
    qs[-1] += 1e-9
    binid = np.clip(np.digitize(axis, qs[1:-1]), 0, n_bins - 1)
    centres = np.array(
        [
            axis[binid == b].mean() if (binid == b).any() else np.nan
            for b in range(n_bins)
        ]
    )
    return binid, centres


def profile_along_axis(values, binid, n_bins: int | None = None, z_score: bool = True):
    """Mean and SEM of ``values`` in each axis bin.

    Returns ``(mean, sem)`` arrays of length ``n_bins``.  ``z_score=True``
    standardises first so panels with different dynamic ranges are comparable on
    one y-axis.
    """
    v = np.asarray(values, float)
    if z_score:
        v = (v - v.mean()) / (v.std() + 1e-12)
    nb = int(np.asarray(binid).max()) + 1 if n_bins is None else n_bins
    mean = np.full(nb, np.nan)
    sem = np.full(nb, np.nan)
    for b in range(nb):
        m = np.asarray(binid) == b
        k = int(m.sum())
        if k:
            mean[b] = v[m].mean()
            sem[b] = v[m].std() / max(1.0, np.sqrt(k))
    return mean, sem


def panel_score(adata, genes, name: str | None = None, use_raw: bool = True):
    """Mean expression of a marker panel per cell.

    Reads ``.raw`` by default so the panel is scored on all genes rather than
    only the HVGs kept for the embedding -- a marker filtered out as low-variance
    would otherwise look like a biological absence.

    Returns a Series indexed by ``obs_names``, or ``None`` if no gene is present.
    """
    src = adata.raw if (use_raw and adata.raw is not None) else adata
    gs = [g for g in genes if g in src.var_names]
    if not gs:
        return None
    X = src[:, gs].X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    return pd.Series(np.asarray(X).mean(1), index=adata.obs_names, name=name or "panel")


def group_profiles(
    adata, genes, group_key, *, use_raw: bool = True, z_score: bool = True, order=None
):
    """Per-gene mean expression in each group, ready to hand to ``panels.profiles``.

    For the common "how does each marker behave across sampling timepoints"
    figure, where the x-axis is a small ordered set of groups rather than a binned
    continuous axis.

    Reads ``.raw`` by default, for the same reason as :func:`panel_score`: a
    marker dropped by the HVG filter would otherwise read as a biological
    absence rather than a filtering artefact.

    ``z_score`` standardises each gene across groups so several markers with
    different dynamic ranges share one y-axis.

    Returns
    -------
    (curves, ticks, table)
        ``curves`` maps gene -> ``(mean, sem)`` for
        :func:`pseudoembed.plotting.panels.profiles`.  The SEM is **explicitly
        zero**: each point is already a group mean, so there is no within-series
        spread to draw, and fabricating one would imply a confidence interval
        that was never computed.  ``ticks`` are the group labels in plotted
        order; ``table`` is the un-z-scored group-by-gene frame, so a caller can
        quote a real expression value rather than a z.
    """
    src = adata.raw.to_adata() if (use_raw and adata.raw is not None) else adata
    gs = [g for g in genes if g in src.var_names]
    if not gs:
        return {}, [], pd.DataFrame()
    X = src[:, gs].X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    df = pd.DataFrame(X, columns=gs, index=adata.obs_names)
    df[group_key] = adata.obs[group_key].values
    table = df.groupby(group_key, observed=True).mean()
    if order is not None:
        table = table.reindex([g for g in order if g in table.index])
    z = ((table - table.mean()) / table.std(ddof=0)) if z_score else table
    curves = {g: (z[g].values, np.zeros(len(z))) for g in z.columns}
    return curves, list(table.index), table


# ---------------------------------------------------------------------------
# Grids of one panel per variable
# ---------------------------------------------------------------------------


def scatter_grid(
    x,
    ys,
    *,
    ncol=None,
    xlabel=None,
    ylabel_fmt="{name}",
    titles=None,
    letters=True,
    figsize=None,
    panel_h=2.5,
    sharex=True,
    **kw,
):
    """One :func:`~pseudoembed.plotting.panels.scatter_trend` panel per variable.

    Plots ``x`` against each column of ``ys`` on its own axes, so validating four
    held-out measurements is one call rather than a ``for`` loop in the caller.
    The loop is the thing worth removing: hand-rolled, it is where a panel gets a
    label from the wrong iteration, or one panel is drawn with different options
    from its neighbours.  Here every panel is guaranteed the same keywords.

    Parameters
    ----------
    x
        The shared x variable, e.g. a fitted pseudotime.
    ys
        ``DataFrame`` (one panel per column, in column order), or a mapping of
        name -> values.  A plain 2-D array is rejected: the panels would be
        unlabelled, and an unlabelled marker panel is not interpretable.
    ylabel_fmt
        Format string for each y-label, given ``name``.  E.g.
        ``"{name} protein (log10)"``.
    titles
        Optional mapping name -> title, or a sequence in column order.  Defaults
        to the name itself.  ``letters=True`` draws a bold ``a``, ``b``, ...
        outside each panel's top-left -- the letters come from the same
        enumeration that places the panel, so they cannot drift out of step the
        way a zipped literal can.  Any ``"(a) "`` already in a title is stripped,
        so a pre-lettered title cannot render as ``a (a) ...``.
    panel_h
        Height in inches; the width defaults to the package two-column width.
    **kw
        Forwarded unchanged to every ``scatter_trend`` call (``trend``, ``frac``,
        ``s``, ...), which is what keeps the panels comparable.

    Returns
    -------
    (fig, axes, info)
        ``info`` maps each name to that panel's ``scatter_trend`` dict, so the
        caller can tabulate rho without recomputing it.  Any surplus axes in the
        final grid row are removed, not left as empty boxes.
    """
    if isinstance(ys, pd.DataFrame):
        items = [(str(c), np.asarray(ys[c].values, float)) for c in ys.columns]
    elif isinstance(ys, dict):
        items = [(str(k), np.asarray(v, float)) for k, v in ys.items()]
    else:
        raise TypeError(
            "ys must be a DataFrame or a name->values mapping so every panel is "
            f"labelled; got {type(ys).__name__}"
        )
    if not items:
        raise ValueError("ys is empty -- nothing to plot")

    n = len(items)
    ncol = n if ncol is None else int(ncol)
    ncol = max(1, min(ncol, n))
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(
        nrow, ncol, sharex=bool(sharex), figsize=figsize or (S.W2, panel_h * nrow)
    )
    axes = np.atleast_1d(np.asarray(axes)).ravel()

    if titles is None:
        tmap = {nm: nm for nm, _ in items}
    elif isinstance(titles, dict):
        tmap = {nm: titles.get(nm, nm) for nm, _ in items}
    else:
        tmap = {nm: t for (nm, _), t in zip(items, titles)}

    info = {}
    for i, (nm, v) in enumerate(items):
        ax = axes[i]
        head = S.strip_panel_prefix(tmap[nm])
        # x-label only on the bottom row when the x-axis is shared, else every
        # panel repeats it and the row reads as four separate figures.
        bottom = (not sharex) or (i >= n - ncol)
        # Do NOT title a panel with the same string its own y-axis label already
        # carries.  `ylabel_fmt` defaults to "{name} ..." so the default grid drew
        # "CD69" above an axis labelled "CD69 (log10)" -- the title spent a whole
        # row restating the label, on every panel.  The y-label wins: it carries
        # the units.  An explicit, DIFFERENT title still shows.
        ylab = ylabel_fmt.format(name=nm)
        head_ok = head if (head and head not in ylab) else None
        info[nm] = P.scatter_trend(
            ax,
            x,
            v,
            xlabel=xlabel if bottom else None,
            ylabel=ylab,
            title=head_ok,
            **kw,
        )
        # Panel identity is a bold letter OUTSIDE the axes, never characters
        # inside the title.  Drawn after the panel so it is not clipped by a
        # later autoscale.
        if letters:
            S.panel_letter(ax, chr(ord("a") + i))
    for ax in axes[n:]:
        fig.delaxes(ax)
    fig.tight_layout()
    return fig, list(axes[:n]), info


def profiles_grid(
    panel_curves,
    *,
    ticks=None,
    centres=None,
    ncol=None,
    xlabel=None,
    ylabel=None,
    letters=True,
    figsize=None,
    panel_h=2.7,
    sharey=True,
    **kw,
):
    """One :func:`~pseudoembed.plotting.panels.profiles` panel per named group of
    curves.

    The companion to :func:`scatter_grid` for the other common grid: several sets
    of programme curves side by side, each with its own legend.  Removes the
    ``zip(axes, titles, letters)`` loop, which is where a panel picks up the
    wrong title -- a real defect in this package's history, where a reused helper
    labelled a cell-type panel with a sampling-day correlation.

    Parameters
    ----------
    panel_curves
        Mapping ``panel title -> curves``, where ``curves`` is itself the
        ``name -> (mean, sem)`` mapping :func:`~pseudoembed.plotting.panels.profiles`
        takes (as returned by :func:`group_profiles`).  Insertion order is panel
        order.
    ticks
        Categorical x tick labels shared by every panel, e.g. timepoint names.
        The x positions become ``0..len(ticks)-1``.  Mutually exclusive with
        ``centres``.
    centres
        Numeric x positions shared by every panel, e.g. from :func:`bin_by_axis`.
    letters
        Draw a bold ``a``, ``b``, ... outside each panel's top-left.  Any
        ``"(a) "`` already written into a title is stripped first, so passing
        pre-lettered titles cannot produce the doubled ``a (a)`` this package has
        drawn before.
    sharey
        Shares the y-axis, which is usually right for z-scored curves and is why
        it defaults on -- but note a shared y-grid can make two panels look
        comparable when their scales differ, so pass ``False`` for raw values.

    Returns
    -------
    (fig, axes, info)
        ``info`` maps panel title -> that panel's ``profiles`` dict.
    """
    if not isinstance(panel_curves, dict) or not panel_curves:
        raise ValueError("panel_curves must be a non-empty title -> curves mapping")
    if ticks is not None and centres is not None:
        raise ValueError("pass ticks or centres, not both")

    items = list(panel_curves.items())
    n = len(items)
    ncol = n if ncol is None else max(1, min(int(ncol), n))
    nrow = int(np.ceil(n / ncol))
    x = (
        np.arange(len(ticks), dtype=float)
        if ticks is not None
        else (np.asarray(centres, float) if centres is not None else None)
    )
    if x is None:
        raise ValueError("one of ticks or centres is required")

    fig, axes = plt.subplots(
        nrow, ncol, sharey=bool(sharey), figsize=figsize or (S.W2, panel_h * nrow)
    )
    axes = np.atleast_1d(np.asarray(axes)).ravel()

    info = {}
    for i, (title, curves) in enumerate(items):
        ax = axes[i]
        head = S.strip_panel_prefix(str(title))
        bottom = i >= n - ncol
        first_col = i % ncol == 0
        info[title] = P.profiles(
            ax,
            x,
            curves,
            title=head,
            xlabel=xlabel if bottom else None,
            ylabel=(ylabel if (first_col or not sharey) else None),
            **kw,
        )
        if sharey and not first_col:
            # `sharey` suppresses the tick LABELS on non-first columns but leaves
            # the tick marks and the left spine drawn, so the panel shows a scale
            # with its numbers missing -- it reads as clipped, not as shared.  Drop
            # the marks and the spine too; the shared scale is the first column's.
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
        if ticks is not None:
            ax.set_xticks(x)
            ax.set_xticklabels(list(ticks))
        if letters:
            # The legend and title stack ABOVE the frame in `profiles`, so the
            # letter must clear both.  Measure them rather than predicting the
            # stack height from font sizes -- that arithmetic put the letter
            # BELOW the legend.
            S.panel_letter_above(ax, chr(ord("a") + i))
    for ax in axes[n:]:
        fig.delaxes(ax)
    fig.tight_layout()
    return fig, list(axes[:n]), info


# ---------------------------------------------------------------------------
# The standard diagnostic figure
# ---------------------------------------------------------------------------


def standard_panels(
    emb,
    axis,
    groups,
    curves=None,
    centres=None,
    group_order=None,
    palette=None,
    group_name="group",
    axis_name="hypergraph pseudotime",
    space_name="hypergraph",
    reference=None,
    group_values=None,
    draw_last=None,
    rotate=0,
    panel_c_title=None,
    figsize=None,
    point_size=9,
):
    """The three standard diagnostic panels for a fitted axis.

    (a) the embedding coloured by ``groups``
    (b) the axis across ``groups``
    (c) programme profiles along the axis

    Intended to be called once per grouping -- sampling time, then cell type --
    which is the whole reason it is a function.  Panels (a) and (b) come from the
    SAME stored embedding and axis, so they cannot disagree.

    Parameters
    ----------
    emb
        The spectral embedding, e.g. ``obsm['multiview_pseudotime_spectral']``;
        columns 0 and 1 are plotted.  Column 0 is the first *retained* column,
        **not** necessarily the leading eigenvector -- do not label it as phi1.
    reference
        Numeric vector used to orient the two embedding axes (their sign is
        arbitrary).  Pass sampling time, or ``None`` to leave signs untouched.
        This orients panel (a) ONLY -- it is never used for a correlation.
    group_values
        Numeric encoding of ``groups``, supplied only when the grouping is a
        genuine ordered quantity.  When given, panel (b)'s title carries
        rho(axis, group_values).  Leave as ``None`` for an unordered grouping such
        as cell type: rho against alphabetical category codes is meaningless, and
        rho against some *other* variable would mislabel the panel.
    space_name
        Names the space actually fitted.  Derive it from the view set at the call
        site -- hardcoding "Fused" here once labelled a single-view fit as a
        fusion.
    curves, centres
        As produced by :func:`profile_along_axis` / :func:`bin_by_axis`.  Omit to
        skip (c) and get a two-panel figure.
    point_size
        Marker area for panel (a), forwarded to
        :func:`pseudoembed.plotting.panels.embedding`.  The default is 9, sized
        for a ~W2/3-wide panel holding a few hundred cells.  It was 22, which was
        chosen while this figure was drawn on a 14 in canvas; on a panel one third
        of the journal column width those markers touch and the clusters read as
        solid blobs.  Marker area must be re-chosen whenever the panel width
        changes -- it is not an absolute property of the dataset.

    Returns
    -------
    (fig, axes, info)
        ``info`` carries ``orient_rho`` (the two orientation correlations) and
        the per-panel return dicts.
    """
    emb = np.asarray(emb)
    if emb.shape[1] < 2:
        raise ValueError(f"embedding needs >=2 columns, got {emb.shape[1]}")

    ncol = 3 if curves else 2
    # Panel (a) is widest because its group key sits to the RIGHT of its axes
    # (see panels.embedding): without the extra share that key overlaps panel (b).
    widths = [1.22, 1.0, 1.18][:ncol]
    # Width is the JOURNAL COLUMN WIDTH (S.W2 = 183 mm), not a free parameter.
    # This used to be a hardcoded 14.0 in -- nearly double S.W2 -- and that one
    # number was the single biggest reason these figures did not read as
    # publishable: every rcParam in the theme (7 pt ticks, 0.6 pt spines, 1.0 pt
    # lines, 3.2 pt markers) is calibrated for a figure that is AT MOST W2 wide,
    # so on a 14 in canvas all of it rendered at roughly half its intended
    # relative weight.  The result was a huge figure with tiny labels and
    # hairline axes -- the classic signature of a plot built at the wrong size.
    # Do not reintroduce an absolute width here: a figure wider than the column
    # it prints in is always wrong, and the fix is never to enlarge the fonts.
    h = 2.45 if curves else 2.30
    fig = plt.figure(figsize=figsize or (S.W2, h))
    gs = fig.add_gridspec(1, ncol, width_ratios=widths, wspace=0.42 if curves else 0.34)

    if reference is not None:
        c0, r0 = orient_axis(emb[:, 0], reference)
        c1, r1 = orient_axis(emb[:, 1], reference)
        suffix = " (oriented)"
    else:
        c0, c1 = np.asarray(emb[:, 0], float), np.asarray(emb[:, 1], float)
        r0 = r1 = float("nan")
        suffix = ""

    axA = fig.add_subplot(gs[0, 0])
    infoA = P.embedding(
        axA,
        c0,
        c1,
        groups,
        palette=palette,
        order=group_order,
        draw_last=draw_last,
        s=point_size,
        xlabel=f"spectral axis 1{suffix}",
        ylabel=f"spectral axis 2{suffix}",
        title=f"{space_name} embedding",
        # Panel (b) prints n under every box, so repeating the
        # counts in this key only widens it into panel (b).
        show_counts=False,
    )
    # Eigenvector coordinates have NO interpretable scale: "-0.08" is not a
    # quantity a reader can use, and printing six such ticks per axis spends the
    # panel's ink implying otherwise.  Drop the numbers and keep the axis LABELS,
    # which is standard practice for embedding panels (UMAP/t-SNE included).  The
    # spines stay, so the panel still reads as a plot rather than a floating cloud.
    axA.set_xticks([])
    axA.set_yticks([])

    axB = fig.add_subplot(gs[0, 1])
    # rho belongs in the title ONLY when it describes the grouping actually
    # plotted.  `reference` orients the embedding and may be a different variable
    # (sampling time) from `groups` (cell type) -- quoting rho(axis, reference)
    # over a cell-type panel labels it with an unrelated number, which is worse
    # than omitting it.  So panels.axis_vs_groups requires an explicit ordered
    # numeric grouping before it will print one.
    infoB = P.axis_vs_groups(
        axB,
        axis,
        groups,
        order=group_order,
        palette=palette,
        xlabel=group_name,
        ylabel=axis_name,
        rotate=rotate,
        group_values=group_values,
        # NOT f"{axis_name} vs {group_name}" -- at true
        # column width each panel is ~2.4 in and a 38-char
        # title overruns into the next panel's legend.  The
        # y-axis already says what the axis is, so the title
        # only has to name the grouping.
        title=f"by {group_name}",
    )

    axes = [axA, axB]
    infoC = None
    if curves:
        axC = fig.add_subplot(gs[0, 2])
        infoC = P.profiles(
            axC,
            centres,
            curves,
            palette=None,
            # One legend entry per row: side by side, two programme names plus the
            # title exceed the panel width and the legend lands on the title.
            legend_ncol=1,
            title=S.strip_panel_prefix(panel_c_title) or "programmes along the axis",
            # The bin-centre caveat is caption material, not an axis label: it
            # made the label two lines tall on every render.
            xlabel=axis_name,
        )
        axes.append(axC)

    for ax, lt in zip(axes, "abc"):
        S.panel_letter(ax, lt)

    return (
        fig,
        axes,
        {
            "orient_rho": (r0, r1),
            "embedding": infoA,
            "axis_vs_groups": infoB,
            "profiles": infoC,
        },
    )
