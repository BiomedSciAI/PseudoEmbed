"""
Publication-quality figures for multi-view hypergraph perturbation analysis.

All functions share a consistent visual language suited to Nature / Science:
  - Helvetica-like sans-serif, 7–8 pt body, 9 pt titles
  - Minimal axes: no top/right spines, thin remaining spines
  - Colour palette: slate-blue (control) vs brick-red (drug), muted greys
  - 300 dpi TIFF output by default; PNG also accepted
  - All figures return (fig, axes) so callers can add panels or adjust further

Entry points
------------
plot_trajectory_divergence_pub          — Wasserstein + MMD over pseudotime, bar-coded significance
plot_top_tfs_pub                        — Ranked horizontal lollipop of top differential TFs
plot_top_genes_pub                      — Same layout for genes (feature_source='X')
plot_trajectory_drivers_heatmap         — Global (condition-agnostic) heatmap from identify_trajectory_drivers
plot_trajectory_drivers_heatmap_comparison — Control vs drug 3-panel heatmap from GAM fits
plot_gam_panel                          — n×2 grid of GAM trajectories for selected features
plot_summary_dashboard                  — Multi-panel summary figure combining all of the above
plot_activity_compare                   — Mean TF/pathway activity heatmap faceted by condition × timepoint
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

_CTRL_COLOR = "#3A6EA5"  # slate blue  – control
_DRUG_COLOR = "#B84040"  # brick red   – drug / treatment
_DELTA_COLOR = "#2A7A4B"  # forest green – difference
_SIG_COLOR = "#E8A838"  # amber       – significance highlight
_GREY_LIGHT = "#E8EAED"
_GREY_MED = "#B0B7C0"
_GREY_DARK = "#555F6D"
_BG = "white"

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": _BG,
    "axes.facecolor": _BG,
    "figure.facecolor": _BG,
}


def _apply_style() -> None:
    mpl.rcParams.update(_RC)


def _despine(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)


def _save(fig: mpl.figure.Figure, path: Optional[str]) -> None:
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=_BG)


# ---------------------------------------------------------------------------
# 1. Trajectory divergence
# ---------------------------------------------------------------------------


def plot_trajectory_divergence_pub(
    divergence_df: pd.DataFrame,
    control_label: str = "control",
    drug_label: str = "drug",
    fdr_threshold: float = 0.05,
    save_path: Optional[str] = None,
) -> Tuple[mpl.figure.Figure, np.ndarray]:
    """
    Two-panel figure: Wasserstein distance (top) and MMD² (bottom) over pseudotime.

    Significant bins (BH-corrected q < fdr_threshold) are marked with a filled
    circle and a semi-transparent shaded column spanning both panels.

    Parameters
    ----------
    divergence_df : pd.DataFrame
        Output of ``compute_trajectory_divergence``.
    control_label, drug_label : str
        Condition labels for the subtitle.
    fdr_threshold : float
        q-value cutoff for significance overlay.
    save_path : str, optional
        File path to save figure (PNG or TIFF). Omit to skip saving.

    Returns
    -------
    fig, axes : Figure and (2,) axes array.
    """
    _apply_style()
    df = divergence_df.copy()
    x = df["pseudotime_center"].values

    has_w = "wasserstein" in df.columns and df["wasserstein"].notna().any()
    has_mmd = "mmd" in df.columns and df["mmd"].notna().any()
    n_panels = int(has_w) + int(has_mmd)
    if n_panels == 0:
        raise ValueError("divergence_df contains no wasserstein or mmd values.")

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(3.5, 1.8 * n_panels),
        sharex=True,
    )
    if n_panels == 1:
        axes = np.array([axes])

    panel_specs = []
    if has_w:
        panel_specs.append(
            ("wasserstein", "wasserstein_qval", "Mean Wasserstein\ndistance")
        )
    if has_mmd:
        panel_specs.append(("mmd", "mmd_qval", "MMD²"))

    for ax, (metric, qcol, ylabel) in zip(axes, panel_specs):
        y = df[metric].values
        valid = ~np.isnan(y)

        # Bar chart of cell counts behind the trace
        ax2 = ax.twinx()
        ax2.bar(
            x,
            df["n_control"] + df["n_drug"],
            width=(x[1] - x[0]) * 0.85 if len(x) > 1 else 0.08,
            color=_GREY_LIGHT,
            alpha=0.5,
            zorder=0,
            linewidth=0,
        )
        ax2.set_ylabel("Cells per bin", fontsize=5, color=_GREY_MED)
        ax2.tick_params(axis="y", labelsize=5, colors=_GREY_MED)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_linewidth(0.4)
        ax2.spines["right"].set_color(_GREY_MED)
        ax2.yaxis.set_major_locator(mticker.MaxNLocator(3, integer=True))

        # Main trace
        ax.plot(
            x[valid],
            y[valid],
            "-o",
            color=_CTRL_COLOR,
            linewidth=1.2,
            markersize=3.5,
            zorder=3,
            clip_on=False,
        )

        # Significance overlay
        if qcol in df.columns:
            sig = df[qcol].values < fdr_threshold
            if sig.any():
                for xi, si, yi in zip(x, sig, y):
                    if si and not np.isnan(yi):
                        ax.axvspan(
                            xi - 0.04,
                            xi + 0.04,
                            color=_SIG_COLOR,
                            alpha=0.18,
                            linewidth=0,
                            zorder=1,
                        )
                        ax.scatter(
                            [xi], [yi], color=_SIG_COLOR, s=18, zorder=5, linewidths=0
                        )

        ax.set_ylabel(ylabel, labelpad=4)
        _despine(ax)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))

    axes[-1].set_xlabel("Pseudotime", labelpad=3)

    # Significance legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elems = [
        Line2D(
            [0],
            [0],
            color=_CTRL_COLOR,
            lw=1.2,
            label=f"{control_label} vs {drug_label}",
        ),
        Patch(facecolor=_SIG_COLOR, alpha=0.35, label=f"FDR < {fdr_threshold}"),
    ]
    axes[0].legend(
        handles=legend_elems, loc="upper right", handlelength=1.2, borderpad=0.4
    )

    fig.suptitle(
        "Trajectory-conditioned distributional divergence",
        fontsize=8,
        y=1.01,
        fontweight="bold",
    )
    plt.tight_layout(h_pad=0.4)
    _save(fig, save_path)
    return fig, axes


# ---------------------------------------------------------------------------
# 2. Top differential TFs / genes — lollipop
# ---------------------------------------------------------------------------


def _lollipop_plot(
    summary_df: pd.DataFrame,
    n_top: int,
    title: str,
    xlabel: str,
    fdr_threshold: float,
    save_path: Optional[str],
    value_col: str = "f_statistic",
    delta_col: str = "mean_abs_delta",
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    _apply_style()
    top = summary_df.head(n_top).copy()
    top = top.sort_values(value_col, ascending=True)  # ascending so top is at top

    sig = (
        top["significant"].values
        if "significant" in top.columns
        else np.ones(len(top), bool)
    )
    colors = [_DRUG_COLOR if s else _GREY_MED for s in sig]

    fig_h = max(2.0, 0.28 * n_top + 0.6)
    fig, ax = plt.subplots(figsize=(3.5, fig_h))

    y_pos = np.arange(len(top))
    vals = top[value_col].values

    # Stems
    ax.hlines(y_pos, 0, vals, colors=_GREY_LIGHT, linewidth=0.8, zorder=1)
    # Dots — coloured by direction of delta
    deltas = (
        top["mean_drug"].values - top["mean_ctrl"].values
        if ("mean_drug" in top.columns and "mean_ctrl" in top.columns)
        else np.zeros(len(top))
    )
    dot_colors = [_DRUG_COLOR if d > 0 else _CTRL_COLOR for d, s in zip(deltas, sig)]
    ax.scatter(vals, y_pos, c=dot_colors, s=22, zorder=3, linewidths=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["feature"].values, fontsize=6)
    ax.set_xlabel(xlabel, labelpad=3)
    ax.set_xlim(left=0)
    ax.set_title(title, fontweight="bold", pad=5)
    _despine(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))

    # Colour legend
    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_DRUG_COLOR,
            markersize=5,
            label="Drug > Control",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_CTRL_COLOR,
            markersize=5,
            label="Control > Drug",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_GREY_MED,
            markersize=5,
            label=f"FDR ≥ {fdr_threshold}",
        ),
    ]
    ax.legend(handles=legend_elems, loc="lower right", handlelength=0.8, borderpad=0.5)

    plt.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_top_tfs_pub(
    summary_df: pd.DataFrame,
    n_top: int = 25,
    fdr_threshold: float = 0.05,
    save_path: Optional[str] = None,
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Horizontal lollipop of the top differentially dynamic TFs ranked by F-statistic.

    Dot colour encodes direction: drug > control (brick red) or control > drug (slate blue).
    Non-significant features (FDR ≥ threshold) are rendered in grey.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of ``compute_differential_dynamics`` with ``feature_source='tf_activities'``.
    n_top : int
        Number of TFs to show.
    fdr_threshold : float
        BH q-value threshold for significance colouring.
    save_path : str, optional
        File path to save.
    """
    return _lollipop_plot(
        summary_df,
        n_top,
        title="Top differential TFs",
        xlabel="F-statistic (differential dynamics)",
        fdr_threshold=fdr_threshold,
        save_path=save_path,
    )


def plot_top_genes_pub(
    summary_df: pd.DataFrame,
    n_top: int = 25,
    fdr_threshold: float = 0.05,
    save_path: Optional[str] = None,
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Horizontal lollipop of the top differentially dynamic genes ranked by F-statistic.

    Same layout as ``plot_top_tfs_pub`` but labelled for gene expression.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of ``compute_differential_dynamics`` with ``feature_source='X'``.
    n_top : int
        Number of genes to show.
    fdr_threshold : float
        BH q-value threshold.
    save_path : str, optional
        File path to save.
    """
    return _lollipop_plot(
        summary_df,
        n_top,
        title="Top differential genes",
        xlabel="F-statistic (differential dynamics)",
        fdr_threshold=fdr_threshold,
        save_path=save_path,
    )


# ---------------------------------------------------------------------------
# 3a. Global trajectory heatmap  (condition-agnostic, from correlation drivers)
# ---------------------------------------------------------------------------


def plot_trajectory_drivers_heatmap(
    driver_df: pd.DataFrame,
    adata,
    pseudotime_key: str = "multiview_pseudotime",
    feature_source: str = "tf_activities",
    feature_col: str = "tf",
    feature_label: str = "TF",
    n_top: int = 30,
    smooth_window: int = 21,
    save_path: Optional[str] = None,
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Global (condition-agnostic) heatmap of feature activity along pseudotime.

    Takes the output of ``identify_trajectory_drivers`` and plots all cells
    sorted by pseudotime, one cell per column.  Each row is z-scored so
    up- and down-regulation are directly comparable across features.  A
    rolling mean along the cell axis is applied so the gradient of change
    over pseudotime is clearly visible even with tens of thousands of cells,
    without discarding single-cell resolution.

    Rows are ordered by the pseudotime position at which the smoothed
    z-scored value peaks, producing the characteristic cascade pattern.

    Parameters
    ----------
    driver_df : pd.DataFrame
        Output of ``identify_trajectory_drivers``.  Must contain a column
        named *feature_col* with feature names, already ranked by
        ``abs_correlation`` (top rows shown first).
    adata : AnnData
        Cells with pseudotime values.
    pseudotime_key : str
        Key in ``adata.obs`` with pseudotime values.
    feature_source : str
        ``'tf_activities'`` — reads ``adata.obsm['tf_activities']`` with names
        from ``adata.uns['tf_activities_info']['tf_names']``.
        ``'X'`` — reads ``adata.X`` with names from ``adata.var_names``.
    feature_col : str
        Column in *driver_df* that holds feature names.
        Use ``'tf'`` for TF drivers, ``'gene'`` for gene drivers.
    feature_label : str
        Y-axis label, e.g. ``'TF'`` or ``'Gene'``.
    n_top : int
        Number of features to show (rows).
    smooth_window : int
        Width of the uniform rolling-mean window applied along the pseudotime
        axis after z-scoring.  Larger values produce a smoother gradient.
        Set to 1 to disable.  Default 21 works well for ≥ 5,000 cells.
    save_path : str, optional
        File path to save.

    Returns
    -------
    fig, ax : Figure and heatmap Axes.
    """
    _apply_style()

    top_features = driver_df.head(n_top)[feature_col].tolist()

    # ------------------------------------------------------------------
    # Build feature matrix sorted by pseudotime (all cells)
    # ------------------------------------------------------------------
    pt_vals = adata.obs[pseudotime_key].values.astype(float)
    pt_order = np.argsort(pt_vals)

    if feature_source == "tf_activities":
        raw = adata.obsm["tf_activities"]
        if hasattr(raw, "values"):
            raw = raw.values
        all_names = (
            adata.uns["tf_activities_info"]["tf_names"]
            if "tf_activities_info" in adata.uns
            else list(range(raw.shape[1]))
        )
    else:  # 'X'
        raw = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        all_names = list(adata.var_names)

    name_to_idx = {n: i for i, n in enumerate(all_names)}
    valid = [f for f in top_features if f in name_to_idx]
    if not valid:
        raise ValueError("None of the top features found in the data.")

    feat_idx = [name_to_idx[f] for f in valid]
    # shape: (n_feats, n_cells) in pseudotime order
    plot_data = raw[pt_order][:, feat_idx].T.astype(float)

    # ------------------------------------------------------------------
    # Z-score each row (per-feature normalisation across all cells)
    # ------------------------------------------------------------------
    row_mean = plot_data.mean(axis=1, keepdims=True)
    row_std = plot_data.std(axis=1, keepdims=True)
    plot_data = (plot_data - row_mean) / (row_std + 1e-10)

    # ------------------------------------------------------------------
    # Rolling mean along the pseudotime axis to reveal gradient
    # ------------------------------------------------------------------
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        plot_data = np.apply_along_axis(
            lambda r: np.convolve(r, kernel, mode="same"), axis=1, arr=plot_data
        )

    # ------------------------------------------------------------------
    # Sort rows by pseudotime position of the smoothed peak
    # ------------------------------------------------------------------
    peak_col = np.argmax(plot_data, axis=1)
    row_order = np.argsort(peak_col)
    plot_data = plot_data[row_order]
    ordered_feats = [valid[r] for r in row_order]

    n_cells = plot_data.shape[1]
    xtick_pos = np.linspace(0, n_cells - 1, 5)
    xtick_lab = [
        f"{v:.2f}"
        for v in np.interp(
            xtick_pos,
            np.arange(n_cells),
            np.sort(pt_vals),
        )
    ]

    # Width capped so the figure stays manageable regardless of cell count
    fig_w = max(5.0, min(14.0, n_cells / 1000 * 3.0 + 2.5))
    # Height: each row gets the same pixels as a column is wide, capped sensibly
    cell_h = fig_w / max(n_cells / 200.0, 1.0)  # approx inches per row
    cell_h = min(max(cell_h, 0.18), 0.40)
    fig_h = max(3.0, cell_h * len(ordered_feats) + 1.2)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        plot_data,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-2.0,
        vmax=2.0,
        interpolation="nearest",
        rasterized=True,
    )

    ax.set_yticks(np.arange(len(ordered_feats)))
    ax.set_yticklabels(ordered_feats, fontsize=6)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, fontsize=6)
    ax.set_xlabel("Pseudotime", labelpad=3)
    ax.set_ylabel(feature_label, labelpad=3)
    ax.set_title(
        f"{feature_label} activity along pseudotime (all cells)",
        fontweight="bold",
        pad=6,
        fontsize=8,
    )

    # Horizontal grid lines between rows only (vertical lines are too busy)
    ax.set_yticks(np.arange(-0.5, len(ordered_feats), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4, axis="y")
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, aspect=30)
    cbar.set_label("Z-score", fontsize=6, labelpad=4)
    cbar.ax.tick_params(labelsize=5)
    cbar.outline.set_linewidth(0.4)

    _despine(ax)
    plt.tight_layout()
    _save(fig, save_path)
    return fig, ax


# ---------------------------------------------------------------------------
# 3b. Condition-comparison heatmap  (control vs drug, from GAM fits)
# ---------------------------------------------------------------------------


def plot_trajectory_drivers_heatmap_comparison(
    gam_fits: Dict[str, pd.DataFrame],
    summary_df: pd.DataFrame,
    n_top: int = 30,
    fdr_threshold: float = 0.05,
    feature_label: str = "TF",
    control_label: str = "control",
    drug_label: str = "drug",
    save_path: Optional[str] = None,
) -> Tuple[mpl.figure.Figure, np.ndarray]:
    """
    Three-panel heatmap comparing control vs drug trajectory drivers.

    Uses the GAM-smoothed fitted curves from ``fit_condition_gams`` /
    ``compute_differential_dynamics`` directly — no re-binning or re-mixing
    of raw data.  The GAM fits are the natural smooth representation of each
    condition's trajectory, so this is the cleanest way to compare them.

    **Panel layout**

    Left   — Control GAM mean, z-scored per row (shared scale with drug panel).
    Centre — Drug GAM mean, z-scored on the same scale.
    Right  — Δ (drug − control) raw GAM difference; diverging colormap
             centred at 0, slate-blue = control-higher, brick-red = drug-higher.

    Rows are ordered by the pseudotime position at which ``|Δ|`` is largest,
    so features with the strongest condition-specific divergence sit at the
    top of the cascade.

    Amber ticks on the left edge mark features significant at *fdr_threshold*
    (``mw_qval`` column in *summary_df*).

    Parameters
    ----------
    gam_fits : dict
        Keys are feature names; values are DataFrames with columns
        ``pseudotime``, ``control_mean``, ``drug_mean``, ``delta``,
        as returned by ``fit_condition_gams`` / ``compute_differential_dynamics``.
    summary_df : pd.DataFrame
        Summary from ``compute_differential_dynamics``; used to select top
        *n_top* features by F-statistic ranking.
    n_top : int
        Number of features (rows) to show.
    fdr_threshold : float
        Features with ``mw_qval < fdr_threshold`` are marked with an amber
        border tick on the left panel.
    feature_label : str
        Y-axis label ('TF' or 'Gene').
    control_label, drug_label : str
        Condition names used for panel titles.
    save_path : str, optional
        File path to save.

    Returns
    -------
    fig, axes : Figure and (3,) axes array [control, drug, delta].
    """
    _apply_style()
    import matplotlib.colors as mcolors

    top_features = summary_df.head(n_top)["feature"].tolist()
    top_features = [f for f in top_features if f in gam_fits]
    if not top_features:
        raise ValueError("None of the top features have GAM fits.")

    sig_set = (
        set(summary_df.loc[summary_df["mw_qval"] < fdr_threshold, "feature"].tolist())
        if "mw_qval" in summary_df.columns
        else set()
    )

    # ------------------------------------------------------------------
    # Build matrices from the GAM smooth curves
    # All gam_fits[feat]["pseudotime"] should be the same grid; interpolate
    # onto the first feature's grid to guard against any minor differences.
    # ------------------------------------------------------------------
    ref_t = gam_fits[top_features[0]]["pseudotime"].values
    n_pts = len(ref_t)

    ctrl_matrix = np.zeros((len(top_features), n_pts))
    drug_matrix = np.zeros((len(top_features), n_pts))
    delta_matrix = np.zeros((len(top_features), n_pts))

    for i, feat in enumerate(top_features):
        df = gam_fits[feat]
        t = df["pseudotime"].values
        ctrl_matrix[i] = np.interp(ref_t, t, df["control_mean"].values)
        drug_matrix[i] = np.interp(ref_t, t, df["drug_mean"].values)
        delta_matrix[i] = np.interp(ref_t, t, df["delta"].values)

    # ------------------------------------------------------------------
    # Sort rows by pseudotime of largest |Δ|
    # ------------------------------------------------------------------
    peak_col = np.argmax(np.abs(delta_matrix), axis=1)
    row_order = np.argsort(peak_col)
    ctrl_matrix = ctrl_matrix[row_order]
    drug_matrix = drug_matrix[row_order]
    delta_matrix = delta_matrix[row_order]
    ordered_feats = [top_features[r] for r in row_order]

    # ------------------------------------------------------------------
    # Z-score control and drug on a shared per-row scale so the two panels
    # are directly comparable (normalise using the pooled mean/std).
    # ------------------------------------------------------------------
    combined = np.concatenate([ctrl_matrix, drug_matrix], axis=1)
    row_mean = combined.mean(axis=1, keepdims=True)
    row_std = combined.std(axis=1, keepdims=True)
    ctrl_z = (ctrl_matrix - row_mean) / (row_std + 1e-10)
    drug_z = (drug_matrix - row_mean) / (row_std + 1e-10)

    abs_max_delta = max(float(np.nanpercentile(np.abs(delta_matrix), 95)), 1e-6)

    delta_cmap = mcolors.LinearSegmentedColormap.from_list(
        "ctrl_drug", [_CTRL_COLOR, "white", _DRUG_COLOR]
    )

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    n_feats = len(ordered_feats)
    # Each panel is n_pts columns × n_feats rows; aim for square cells.
    # cell_size: inches per cell (both directions), capped to keep figures manageable.
    raw_cell = min(max(8.0 / n_pts, 0.06), 0.35)  # width-driven
    panel_w = max(2.5, raw_cell * n_pts)
    cell_h = min(max(panel_w / max(n_pts / max(n_feats, 1), 1.0), 0.18), 0.40)
    fig_h = max(3.0, cell_h * n_feats + 1.2)
    fig_w = max(9.0, panel_w * 3 + 3.0)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.08, "width_ratios": [1, 1, 1]},
    )

    xtick_pos = np.linspace(0, n_pts - 1, 5)
    xtick_lab = [f"{v:.2f}" for v in np.linspace(ref_t[0], ref_t[-1], 5)]

    panel_data = [ctrl_z, drug_z, delta_matrix]
    panel_cmap = ["RdBu_r", "RdBu_r", delta_cmap]
    panel_vmin = [-2.0, -2.0, -abs_max_delta]
    panel_vmax = [2.0, 2.0, abs_max_delta]
    panel_titles = [
        f"{control_label}  (Z-score)",
        f"{drug_label}  (Z-score)",
        f"Δ  {drug_label} − {control_label}",
    ]
    panel_cbar = [
        "Z-score",
        "Z-score",
        f"Δ activity\n({drug_label}−{control_label})",
    ]

    for col, (ax, data, cm, vlo, vhi, ptitle, cbl) in enumerate(
        zip(
            axes,
            panel_data,
            panel_cmap,
            panel_vmin,
            panel_vmax,
            panel_titles,
            panel_cbar,
        )
    ):
        im = ax.imshow(
            data,
            aspect="auto",
            cmap=cm,
            vmin=vlo,
            vmax=vhi,
            interpolation="nearest",
            rasterized=True,
        )

        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_lab, fontsize=6)
        ax.set_xlabel("Pseudotime", labelpad=3)
        ax.set_title(ptitle, fontweight="bold", pad=6, fontsize=8)

        if col == 0:
            ax.set_yticks(np.arange(n_feats))
            ax.set_yticklabels(ordered_feats, fontsize=6)
            ax.set_ylabel(feature_label, labelpad=3)
            for row_i, feat in enumerate(ordered_feats):
                if feat in sig_set:
                    ax.plot(
                        [-0.5, -0.5],
                        [row_i - 0.4, row_i + 0.4],
                        color=_SIG_COLOR,
                        lw=2.0,
                        clip_on=False,
                        transform=ax.get_yaxis_transform(),
                    )
        else:
            ax.set_yticks([])

        ax.set_yticks(np.arange(-0.5, n_feats, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.4, axis="y")
        ax.tick_params(which="minor", length=0)

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, aspect=30)
        cbar.set_label(cbl, fontsize=6, labelpad=4)
        cbar.ax.tick_params(labelsize=5)
        cbar.outline.set_linewidth(0.4)

        _despine(ax)

    fig.suptitle(
        f"{feature_label} activity: {control_label} vs {drug_label} over pseudotime",
        fontsize=9,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    _save(fig, save_path)
    return fig, axes


# ---------------------------------------------------------------------------
# 4. GAM trajectory panel
# ---------------------------------------------------------------------------


def plot_gam_panel(
    gam_fits: Dict[str, pd.DataFrame],
    summary_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    n_top: int = 12,
    n_cols: int = 4,
    control_label: str = "control",
    drug_label: str = "drug",
    feature_label: str = "Activity",
    save_path: Optional[str] = None,
) -> Tuple[mpl.figure.Figure, np.ndarray]:
    """
    Grid of GAM trajectory plots for selected features.

    Each panel shows the GAM-fitted mean ± 95 % CI for control (slate blue)
    and drug (brick red), plus the Δ curve (forest green, dashed) referenced
    to a secondary y-axis.

    Parameters
    ----------
    gam_fits : dict
        Output of ``fit_condition_gams`` or ``compute_differential_dynamics``.
    summary_df : pd.DataFrame
        Used to look up F-statistic and q-value for panel subtitles.
    features : list of str, optional
        Specific features to plot. If None, uses top ``n_top`` from summary_df.
    n_top : int
        Number of features to plot when ``features`` is None.
    n_cols : int
        Number of columns in the grid.
    control_label, drug_label : str
        Condition labels.
    feature_label : str
        y-axis label per panel ('Activity' for TFs, 'Expression' for genes).
    save_path : str, optional
        File path to save.

    Returns
    -------
    fig, axes : Figure and 2D axes array (n_rows × n_cols).
    """
    _apply_style()

    if features is None:
        features = summary_df.head(n_top)["feature"].tolist()
    features = [f for f in features if f in gam_fits]
    if not features:
        raise ValueError("None of the requested features have GAM fits.")

    n = len(features)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.5 * n_cols, 2.2 * n_rows),
        squeeze=False,
    )

    # Build lookup for stats
    stats_lookup: Dict[str, dict] = {}
    if summary_df is not None and len(summary_df):
        for _, row in summary_df.iterrows():
            stats_lookup[row["feature"]] = row.to_dict()

    for idx, feat in enumerate(features):
        row_i, col_i = divmod(idx, n_cols)
        ax = axes[row_i][col_i]
        df = gam_fits[feat]
        t = df["pseudotime"].values

        # --- Control and drug GAM traces ---
        ax.fill_between(
            t,
            df["control_ci_lo"],
            df["control_ci_hi"],
            color=_CTRL_COLOR,
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(
            t, df["control_mean"], color=_CTRL_COLOR, linewidth=1.0, label=control_label
        )
        ax.fill_between(
            t,
            df["drug_ci_lo"],
            df["drug_ci_hi"],
            color=_DRUG_COLOR,
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(t, df["drug_mean"], color=_DRUG_COLOR, linewidth=1.0, label=drug_label)

        # --- Δ curve on secondary axis ---
        ax2 = ax.twinx()
        ax2.plot(
            t, df["delta"], color=_DELTA_COLOR, linewidth=0.8, linestyle="--", alpha=0.7
        )
        ax2.axhline(0, color=_GREY_MED, linewidth=0.4, linestyle=":")
        ax2.fill_between(
            t,
            df["delta_ci_lo"],
            df["delta_ci_hi"],
            color=_DELTA_COLOR,
            alpha=0.08,
            linewidth=0,
        )
        ax2.tick_params(axis="y", labelsize=5, colors=_DELTA_COLOR)
        ax2.set_ylabel("Δ", fontsize=5, color=_DELTA_COLOR, labelpad=2)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_linewidth(0.4)
        ax2.spines["right"].set_color(_DELTA_COLOR)
        ax2.yaxis.set_major_locator(mticker.MaxNLocator(3))

        # --- Title with stats ---
        stat_str = ""
        if feat in stats_lookup:
            s = stats_lookup[feat]
            f_val = s.get("f_statistic", float("nan"))
            q_val = s.get("mw_qval", float("nan"))
            stat_str = f"\nF={f_val:.1f}  q={q_val:.0e}"

        ax.set_title(
            f"{feat}{stat_str}",
            fontsize=6,
            pad=3,
            fontweight=(
                "bold"
                if feat in stats_lookup and stats_lookup[feat].get("significant", False)
                else "normal"
            ),
        )

        if col_i == 0:
            ax.set_ylabel(feature_label, fontsize=6, labelpad=2)
        if row_i == n_rows - 1:
            ax.set_xlabel("Pseudotime", fontsize=6, labelpad=2)

        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
        _despine(ax)
        ax.tick_params(axis="both", labelsize=5)

    # Hide unused panels
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    # Shared legend on first panel
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elems = [
        Line2D([0], [0], color=_CTRL_COLOR, lw=1.0, label=control_label),
        Line2D([0], [0], color=_DRUG_COLOR, lw=1.0, label=drug_label),
        Line2D([0], [0], color=_DELTA_COLOR, lw=0.8, ls="--", label="Δ (drug−ctrl)"),
    ]
    axes[0][0].legend(
        handles=legend_elems,
        fontsize=5,
        loc="upper left",
        handlelength=1.0,
        borderpad=0.4,
    )

    plt.tight_layout(h_pad=0.8, w_pad=0.6)
    _save(fig, save_path)
    return fig, axes


# ---------------------------------------------------------------------------
# 5. Summary dashboard
# ---------------------------------------------------------------------------


def plot_summary_dashboard(
    divergence_df: pd.DataFrame,
    tf_summary_df: pd.DataFrame,
    tf_gam_fits: Dict[str, pd.DataFrame],
    gene_summary_df: Optional[pd.DataFrame] = None,
    gene_gam_fits: Optional[Dict[str, pd.DataFrame]] = None,
    n_top_lollipop: int = 20,
    n_top_heatmap: int = 30,
    n_top_gam: int = 8,
    fdr_threshold: float = 0.05,
    control_label: str = "control",
    drug_label: str = "drug",
    adata=None,
    pseudotime_key: str = "multiview_pseudotime",
    save_path: Optional[str] = None,
) -> Tuple[mpl.figure.Figure, dict]:
    """
    Single-figure summary dashboard combining:
      A — Trajectory divergence (Wasserstein + MMD)
      B — Top TF lollipop
      C — TF drivers heatmap
      D — TF GAM panel (top 8)
      E — (optional) Gene lollipop if gene_summary_df is provided

    Parameters
    ----------
    divergence_df : pd.DataFrame
        Output of ``compute_trajectory_divergence``.
    tf_summary_df : pd.DataFrame
        Output of ``compute_differential_dynamics`` with feature_source='tf_activities'.
    tf_gam_fits : dict
        GAM fits for top TFs.
    gene_summary_df : pd.DataFrame, optional
        Output with feature_source='X'. Adds panel E when provided.
    gene_gam_fits : dict, optional
        GAM fits for top genes.
    n_top_lollipop : int
        Features in lollipop panels.
    n_top_heatmap : int
        Features in heatmap.
    n_top_gam : int
        Features in GAM grid.
    fdr_threshold : float
        BH q-value cutoff.
    control_label, drug_label : str
        Condition labels.
    save_path : str, optional
        File path to save.

    Returns
    -------
    fig : Figure
    axes_dict : dict mapping panel label ('A', 'B', …) to axes.
    """
    _apply_style()

    # Save individual panels to a temp directory alongside save_path, then
    # compose the dashboard by returning the separate figures. The dashboard
    # itself is a thin wrapper that lays them out in a 2-column grid.
    # (Composing dissimilar subplot types into one gridspec is fragile;
    # we keep the individual figures as the primary outputs and provide this
    # as a convenience overview.)

    base = Path(save_path).parent if save_path else Path(".")
    stem = Path(save_path).stem if save_path else "dashboard"
    ext = Path(save_path).suffix if save_path else ".png"

    figs: dict = {}

    figs["A"] = plot_trajectory_divergence_pub(
        divergence_df,
        control_label=control_label,
        drug_label=drug_label,
        fdr_threshold=fdr_threshold,
        save_path=str(base / f"{stem}_A_divergence{ext}"),
    )
    figs["B"] = plot_top_tfs_pub(
        tf_summary_df,
        n_top=n_top_lollipop,
        fdr_threshold=fdr_threshold,
        save_path=str(base / f"{stem}_B_top_tfs{ext}"),
    )
    figs["C"] = plot_trajectory_drivers_heatmap_comparison(
        tf_gam_fits,
        tf_summary_df,
        n_top=n_top_heatmap,
        fdr_threshold=fdr_threshold,
        feature_label="TF",
        control_label=control_label,
        drug_label=drug_label,
        save_path=str(base / f"{stem}_C_heatmap{ext}"),
    )
    figs["D"] = plot_gam_panel(
        tf_gam_fits,
        tf_summary_df,
        n_top=n_top_gam,
        control_label=control_label,
        drug_label=drug_label,
        feature_label="TF activity",
        save_path=str(base / f"{stem}_D_gam_tfs{ext}"),
    )
    if gene_summary_df is not None and gene_gam_fits is not None:
        figs["E"] = plot_top_genes_pub(
            gene_summary_df,
            n_top=n_top_lollipop,
            fdr_threshold=fdr_threshold,
            save_path=str(base / f"{stem}_E_top_genes{ext}"),
        )
        figs["F"] = plot_gam_panel(
            gene_gam_fits,
            gene_summary_df,
            n_top=n_top_gam,
            control_label=control_label,
            drug_label=drug_label,
            feature_label="Expression",
            save_path=str(base / f"{stem}_F_gam_genes{ext}"),
        )

    return figs


# ---------------------------------------------------------------------------
# 7. Pseudotime QC
# ---------------------------------------------------------------------------


def plot_pseudotime_qc(
    adata,
    pseudotime_key: str = "multiview_pseudotime",
    condition_key: str = "condition",
    control_label: str = "control",
    drug_label: str = "drug",
    time_key: Optional[str] = "timepoint",
    umap_key: str = "X_umap",
    cell_type_key: Optional[str] = "CellTypes",
    n_histogram_bins: int = 40,
    save_path: Optional[str] = None,
) -> Tuple[mpl.figure.Figure, list]:
    """
    Five-panel pseudotime QC figure.

    Panel A — Histogram + KDE of pseudotime values, overlaid per condition.
    Panel B — Violin of pseudotime by time point, split by condition.
    Panel C — UMAP scatter coloured by pseudotime (skipped if X_umap absent).
    Panel D — Spectral embedding PC1 vs PC2, coloured by pseudotime.
               Shows the actual computation space with the pseudotime gradient
               overlaid so root placement and trajectory topology are visible.
               Skipped if the spectral embedding is not stored in adata.obsm.
    Panel E — Strip plot of pseudotime vs experimental time, coloured by cell
               type (or condition if no cell-type key is present).  Directly
               tests whether pseudotime increases monotonically with known
               time labels.  Skipped if time_key is None.

    Parameters
    ----------
    adata : AnnData
    pseudotime_key : str
        Key in adata.obs with pseudotime values.
    condition_key : str
        Key in adata.obs with condition labels.
    control_label, drug_label : str
        Labels for the two conditions.
    time_key : str, optional
        Key in adata.obs with ordered time-point labels. Panels B and E are
        skipped if None or the key is not present in adata.obs.
    umap_key : str
        Key in adata.obsm for the 2-D embedding. Panel C is skipped if absent.
    cell_type_key : str, optional
        Key in adata.obs with cell-type labels for Panel E colouring.
        Falls back to condition if absent or None.
    n_histogram_bins : int
        Number of bins for the pseudotime histogram (default 40).
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    fig : Figure
    axes : list of Axes
    """
    import scipy.stats as scipy_stats

    _apply_style()

    pseudotime = adata.obs[pseudotime_key].values.astype(float)
    conditions = adata.obs[condition_key].values

    ctrl_mask = conditions == control_label
    drug_mask = conditions == drug_label

    has_timepoints = time_key is not None and time_key in adata.obs.columns
    has_umap = umap_key in adata.obsm

    spectral_key = f"{pseudotime_key}_spectral"
    has_spectral = spectral_key in adata.obsm and adata.obsm[spectral_key].shape[1] >= 2

    n_panels = (
        1
        + int(has_timepoints)
        + int(has_umap)
        + int(has_spectral)
        + int(has_timepoints)  # Panel E reuses has_timepoints
    )
    fig_width = 3.5 * n_panels
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 3.4))
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    # ------------------------------------------------------------------ #
    # Panel A: histogram + KDE per condition                              #
    # ------------------------------------------------------------------ #
    ax = axes[panel_idx]
    panel_idx += 1

    pt_range = (float(np.nanmin(pseudotime)), float(np.nanmax(pseudotime)))
    bin_edges = np.linspace(pt_range[0], pt_range[1], n_histogram_bins + 1)

    for mask, color, label in [
        (ctrl_mask, _CTRL_COLOR, control_label),
        (drug_mask, _DRUG_COLOR, drug_label),
    ]:
        subset = pseudotime[mask]
        ax.hist(
            subset,
            bins=bin_edges,
            color=color,
            alpha=0.45,
            label=label,
            density=True,
            linewidth=0,
        )
        # KDE overlay
        if len(subset) > 2:
            kde = scipy_stats.gaussian_kde(subset, bw_method="scott")
            x_grid = np.linspace(pt_range[0], pt_range[1], 300)
            ax.plot(x_grid, kde(x_grid), color=color, lw=1.4)

    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Density")
    ax.set_title("A  Pseudotime distribution")
    ax.legend(title="Condition", loc="upper right")
    _despine(ax)

    # ------------------------------------------------------------------ #
    # Panel B: violin by time point, split by condition                   #
    # ------------------------------------------------------------------ #
    if has_timepoints:
        ax = axes[panel_idx]
        panel_idx += 1

        time_labels = adata.obs[time_key].astype(str).values
        # Sort time points by 75th-percentile pseudotime so earliest pseudotime
        # is leftmost; Q3 is robust to sparse groups with a few low-value cells
        # that would otherwise drag a median-based sort to the wrong position.
        unique_times = sorted(
            set(time_labels),
            key=lambda t: np.percentile(pseudotime[time_labels == t], 75),
        )

        n_tp = len(unique_times)
        x_positions = np.arange(n_tp)
        half_w = 0.18  # half-width of each violin

        for tp_idx, tp in enumerate(unique_times):
            tp_mask = time_labels == tp
            for offset, mask, color in [
                (-half_w, ctrl_mask & tp_mask, _CTRL_COLOR),
                (+half_w, drug_mask & tp_mask, _DRUG_COLOR),
            ]:
                subset = pseudotime[mask]
                if len(subset) < 3:
                    # Too few cells — draw a dot instead
                    ax.scatter(
                        [tp_idx + offset] * len(subset),
                        subset,
                        color=color,
                        s=6,
                        zorder=3,
                        alpha=0.7,
                    )
                    continue
                parts = ax.violinplot(
                    subset,
                    positions=[tp_idx + offset],
                    widths=half_w * 1.8,
                    showmedians=True,
                    showextrema=False,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.55)
                    pc.set_edgecolor(color)
                    pc.set_linewidth(0.6)
                parts["cmedians"].set_color(color)
                parts["cmedians"].set_linewidth(1.2)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(unique_times, rotation=30, ha="right")
        ax.set_xlabel("Time point")
        ax.set_ylabel("Pseudotime")
        ax.set_title("B  Pseudotime by time point")

        # Manual legend
        import matplotlib.patches as mpatches

        legend_handles = [
            mpatches.Patch(facecolor=_CTRL_COLOR, alpha=0.7, label=control_label),
            mpatches.Patch(facecolor=_DRUG_COLOR, alpha=0.7, label=drug_label),
        ]
        ax.legend(handles=legend_handles, title="Condition", loc="upper left")
        _despine(ax)

    # ------------------------------------------------------------------ #
    # Panel C: UMAP coloured by pseudotime                                #
    # ------------------------------------------------------------------ #
    if has_umap:
        ax = axes[panel_idx]
        panel_idx += 1

        xy = adata.obsm[umap_key][:, :2]
        sc_plot = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=pseudotime,
            cmap="viridis",
            s=1.0,
            linewidths=0,
            rasterized=True,
        )
        cbar = fig.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pseudotime", size=6)
        cbar.ax.tick_params(labelsize=5)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("C  UMAP coloured by pseudotime")
        _despine(ax)

    # ------------------------------------------------------------------ #
    # Panel D: Spectral PC1 vs PC2, coloured by pseudotime                #
    # ------------------------------------------------------------------ #
    if has_spectral:
        ax = axes[panel_idx]
        panel_idx += 1

        emb = adata.obsm[spectral_key]
        sc_plot = ax.scatter(
            emb[:, 0],
            emb[:, 1],
            c=pseudotime,
            cmap="viridis",
            s=1.5,
            linewidths=0,
            rasterized=True,
            alpha=0.7,
        )
        cbar = fig.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pseudotime", size=6)
        cbar.ax.tick_params(labelsize=5)
        ax.set_xlabel("Hypergraph eigenvector 1")
        ax.set_ylabel("Hypergraph eigenvector 2")
        ax.set_title("D  Hypergraph eigenvector embedding")
        _despine(ax)

    # ------------------------------------------------------------------ #
    # Panel E: Strip plot — pseudotime vs experimental time               #
    # ------------------------------------------------------------------ #
    if has_timepoints:
        ax = axes[panel_idx]
        panel_idx += 1

        time_labels = adata.obs[time_key].astype(str).values

        # Determine colour grouping: cell type if available, else condition
        use_cell_type = cell_type_key is not None and cell_type_key in adata.obs.columns
        if use_cell_type:
            group_labels = adata.obs[cell_type_key].astype(str).values
            legend_title = "Cell type"
        else:
            group_labels = conditions
            legend_title = "Condition"

        unique_groups = list(dict.fromkeys(group_labels))  # preserve order
        # Build a colour palette: cycle through a qualitative set
        _palette_base = [
            "#3A6EA5",
            "#B84040",
            "#2A7A4B",
            "#E8A838",
            "#7B5EA7",
            "#4AAABF",
            "#C47A3A",
            "#888888",
        ]
        group_colors = {
            g: _palette_base[i % len(_palette_base)]
            for i, g in enumerate(unique_groups)
        }

        # Sort time labels by 75th-percentile pseudotime so earliest pseudotime
        # is leftmost; Q3 is robust to sparse groups with a few low-value cells
        # that would otherwise drag a median-based sort to the wrong position.
        unique_times = sorted(
            set(time_labels),
            key=lambda t: np.percentile(pseudotime[time_labels == t], 75),
        )
        tp_to_x = {tp: i for i, tp in enumerate(unique_times)}

        # Jitter x positions so overlapping dots are visible
        rng = np.random.default_rng(42)
        for g in unique_groups:
            g_mask = group_labels == g
            x_base = np.array([tp_to_x[t] for t in time_labels[g_mask]], dtype=float)
            x_jitter = x_base + rng.uniform(-0.25, 0.25, size=g_mask.sum())
            ax.scatter(
                x_jitter,
                pseudotime[g_mask],
                color=group_colors[g],
                s=1.5,
                alpha=0.5,
                linewidths=0,
                rasterized=True,
                label=g,
            )

        # Overlay Q3 line per time point (all cells) — matches the sort key
        q3_values = [
            np.percentile(pseudotime[time_labels == tp], 75) for tp in unique_times
        ]
        ax.plot(
            range(len(unique_times)),
            q3_values,
            color=_GREY_DARK,
            lw=1.2,
            zorder=5,
            marker="o",
            markersize=3,
            label="Q3 (75th pct)",
        )

        ax.set_xticks(range(len(unique_times)))
        ax.set_xticklabels(unique_times, rotation=30, ha="right")
        ax.set_xlabel("Experimental time point")
        ax.set_ylabel("Pseudotime")
        ax.set_title("E  Pseudotime vs experimental time")

        # Legend: limit to ≤12 entries to avoid overflow
        n_legend = min(len(unique_groups), 12)
        handles, labels_leg = ax.get_legend_handles_labels()
        # Put Median last
        median_idx = [i for i, l in enumerate(labels_leg) if l == "Median"]
        other_idx = [i for i, l in enumerate(labels_leg) if l != "Median"]
        ordered = other_idx[:n_legend] + median_idx
        ax.legend(
            [handles[i] for i in ordered],
            [labels_leg[i] for i in ordered],
            title=legend_title,
            loc="upper left",
            markerscale=2.5,
            ncol=1 if n_legend <= 6 else 2,
        )
        _despine(ax)

    fig.tight_layout()
    _save(fig, save_path)
    return fig, axes


# ---------------------------------------------------------------------------
# 8. Activity comparison across condition × timepoint
# ---------------------------------------------------------------------------


def plot_activity_compare(
    adata,
    condition_key: str = "condition",
    time_key: str = "timepoint",
    feature_source: str = "tf_activities",
    n_top_features: int = 30,
    control_label: str = "control",
    drug_label: str = "drug",
    save_path: Optional[str] = None,
) -> Tuple[mpl.figure.Figure, np.ndarray]:
    """
    Mean TF or pathway activity heatmap faceted by condition × timepoint.

    For each (condition, timepoint) group the mean activity across member cells
    is computed for the top *n_top_features* most variable features.  The result
    is a grid of heatmaps — one column per condition, one row per timepoint —
    so you can read off both how activity evolves over time and how the two
    conditions diverge at each time point.

    Rows (features) are z-scored across all groups so the colour scale reflects
    relative activation, not absolute magnitude.

    Parameters
    ----------
    adata : AnnData
        Must contain ``adata.obsm[feature_source]`` and the two obs columns.
    condition_key : str
        Column in ``adata.obs`` with condition labels.
    time_key : str
        Column in ``adata.obs`` with ordered time-point labels.
    feature_source : str
        ``'tf_activities'`` (default) or ``'pathway_activities'``.
        The key looked up in ``adata.obsm``.
    n_top_features : int
        Number of features to show, selected by highest variance across groups.
    control_label, drug_label : str
        The two condition labels (used to order columns: control left, drug right).
    save_path : str, optional
        File path to save.

    Returns
    -------
    fig : Figure
    axes : ndarray of Axes  (shape: n_timepoints × 2)
    """
    import matplotlib.colors as mcolors

    _apply_style()

    # ------------------------------------------------------------------ #
    # Extract feature matrix and names                                    #
    # ------------------------------------------------------------------ #
    raw = adata.obsm[feature_source]
    if hasattr(raw, "values"):
        raw = raw.values
    raw = np.asarray(raw, dtype=float)

    if feature_source == "tf_activities" and "tf_activities_info" in adata.uns:
        feat_names = adata.uns["tf_activities_info"]["tf_names"]
    elif (
        feature_source == "pathway_activities"
        and "pathway_activities_info" in adata.uns
    ):
        feat_names = adata.uns["pathway_activities_info"]["pathway_names"]
    elif feature_source == "pathway_activities":
        feat_names = [f"pathway_{i}" for i in range(raw.shape[1])]
    else:
        feat_names = [str(i) for i in range(raw.shape[1])]

    conditions = adata.obs[condition_key].astype(str).values
    timepoints = adata.obs[time_key].astype(str).values

    # Preserve natural time-point order as seen in obs
    seen_tp: list = []
    for t in timepoints:
        if t not in seen_tp:
            seen_tp.append(t)
    unique_times = seen_tp

    # Always put control left, drug right; any other conditions appended after
    ordered_conds = []
    for c in [control_label, drug_label]:
        if c in np.unique(conditions):
            ordered_conds.append(c)
    for c in np.unique(conditions):
        if c not in ordered_conds:
            ordered_conds.append(c)

    n_tp = len(unique_times)
    n_cond = len(ordered_conds)

    # ------------------------------------------------------------------ #
    # Build (condition × timepoint) mean activity matrix                  #
    # group_means: shape (n_cond * n_tp, n_features)                     #
    # ------------------------------------------------------------------ #
    group_means = []
    group_labels = []  # (condition, timepoint) pairs in column order
    group_sizes = []

    for cond in ordered_conds:
        for tp in unique_times:
            mask = (conditions == cond) & (timepoints == tp)
            n = mask.sum()
            group_sizes.append(n)
            if n > 0:
                group_means.append(raw[mask].mean(axis=0))
            else:
                group_means.append(np.full(raw.shape[1], np.nan))
            group_labels.append((cond, tp))

    group_means_arr = np.vstack(group_means)  # (n_groups, n_features)

    # Select top-N features by variance across groups (ignoring NaN groups)
    feat_var = np.nanvar(group_means_arr, axis=0)
    top_idx = np.argsort(feat_var)[::-1][:n_top_features]
    top_idx = top_idx[np.argsort(top_idx)]  # restore original order for now
    top_names = [feat_names[i] for i in top_idx]
    data = group_means_arr[:, top_idx]  # (n_groups, n_top)

    # Z-score each feature across all groups
    feat_mean = np.nanmean(data, axis=0, keepdims=True)
    feat_std = np.nanstd(data, axis=0, keepdims=True)
    data_z = (data - feat_mean) / (feat_std + 1e-10)

    # Reshape to (n_cond, n_tp, n_top) for easy panel indexing
    data_z_3d = data_z.reshape(n_cond, n_tp, len(top_idx))
    sizes_3d = np.array(group_sizes).reshape(n_cond, n_tp)

    # Sort features by peak activity position (across the control × time axis)
    ctrl_idx_in_conds = 0  # control is always first column
    peak_tp = np.argmax(np.abs(data_z_3d[ctrl_idx_in_conds]), axis=0)
    feat_order = np.argsort(peak_tp)
    data_z_3d = data_z_3d[:, :, feat_order]
    top_names = [top_names[i] for i in feat_order]

    # ------------------------------------------------------------------ #
    # Layout: n_tp rows × n_cond columns                                 #
    # ------------------------------------------------------------------ #
    cond_colors = {control_label: _CTRL_COLOR, drug_label: _DRUG_COLOR}

    # Compact Nature-panel cell sizing — square cells
    n_feats_act = len(top_names)
    cell_w = 0.18  # width per time-point column (inches)
    cell_h = 0.18  # height per feature row (inches) — same as width = square
    fig_w = max(3.5, cell_w * n_tp * n_cond + 2.8)
    fig_h = max(2.5, cell_h * n_feats_act + 1.2)

    fig, axes = plt.subplots(
        1,
        n_cond,
        figsize=(fig_w, fig_h),
        sharey=True,
    )
    if n_cond == 1:
        axes = np.array([axes])

    vmax = float(np.nanpercentile(np.abs(data_z_3d), 95))
    vmax = max(vmax, 1e-6)
    cmap = "RdBu_r"

    for ci, cond in enumerate(ordered_conds):
        ax = axes[ci]
        panel = data_z_3d[ci].T  # (n_features, n_tp)

        im = ax.imshow(
            panel,
            aspect="auto",
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
            rasterized=True,
        )

        # X axis — time points with cell counts
        xtick_labels = [
            f"{tp}\n(n={int(sizes_3d[ci, ti])})" for ti, tp in enumerate(unique_times)
        ]
        ax.set_xticks(np.arange(n_tp))
        ax.set_xticklabels(xtick_labels, fontsize=5, rotation=30, ha="right")

        # Y axis — feature names on both panels so each row is identifiable
        ax.set_yticks(np.arange(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=4.5)
        if ci == 0:
            ax.set_ylabel(
                "TF" if feature_source == "tf_activities" else "Pathway",
                labelpad=3,
            )

        # Minor grid between cells
        ax.set_xticks(np.arange(-0.5, n_tp, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(top_names), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.4)
        ax.tick_params(which="minor", length=0)

        # Column title coloured by condition
        title_color = cond_colors.get(cond, _GREY_DARK)
        ax.set_title(cond, fontsize=8, fontweight="bold", color=title_color, pad=4)

        # Colourbar on the rightmost panel only
        if ci == n_cond - 1:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, aspect=30)
            cbar.set_label("Z-score\n(mean activity)", fontsize=5, labelpad=3)
            cbar.ax.tick_params(labelsize=5)
            cbar.outline.set_linewidth(0.4)

        # Black border around the heatmap axes
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(0.6)

    feat_label = "TF" if feature_source == "tf_activities" else "Pathway"
    fig.suptitle(
        f"Mean {feat_label} activity by condition × time point  "
        f"(top {len(top_names)} by variance)",
        fontsize=8,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()
    _save(fig, save_path)
    return fig, axes


# ---------------------------------------------------------------------------
# 9. OL-differentiation multi-panel figure (A: cluster trajectories + heatmaps;
#    B: condition log2FC correlation scatter; C: GO enrichment bar charts)
# ---------------------------------------------------------------------------


def plot_ol_differentiation_panel(
    adata,
    gene_summary_df: pd.DataFrame,
    pseudotime_key: str = "multiview_pseudotime",
    condition_key: str = "condition",
    time_key: Optional[str] = "timepoint",
    control_label: str = "DMSO",
    drug_label: str = "O45",
    cell_type_key: Optional[str] = "cell_type",
    cell_type_groups: Optional[Dict[str, List[str]]] = None,
    n_top_de: int = 10,
    go_results_ctrl: Optional[pd.DataFrame] = None,
    go_results_drug: Optional[pd.DataFrame] = None,
    n_top_go: int = 10,
    fdr_threshold: float = 0.05,
    save_path: Optional[str] = None,
) -> Tuple[mpl.figure.Figure, dict]:
    """
    Multi-panel figure modelled after OL-differentiation publications.

    Panel A (top row)
        One sub-panel per cell-type group: mean z-scored expression per
        condition (control_label vs drug_label) binned over pseudotime.
        Each condition gets a solid line (average trend) plus two ribbons:
        a tight inner fill (±SE, average uncertainty) and a wide outer fill
        (±SD, individual-cell spread), matching the reference figure style.

    Panel A (bottom row)
        One heatmap per cell-type group: top *n_top_de* DE genes (ranked by
        |log₂FC| from gene_summary_df) shown as mean expression averaged
        per **experimental time point** (from time_key), z-scored per gene.
        Columns are the real time-point labels (e.g. day0, day1, day2).
        Rows are ordered by their peak time point.

    Panel B
        Scatter of per-gene log₂FC (drug/control) on the y-axis vs
        log₂FC on the x-axis for the same gene set, derived directly from
        gene_summary_df columns ``mean_ctrl`` and ``mean_drug``.
        Pearson R and p-value are annotated.

    Panel C
        Horizontal bar chart of GO enrichment split into three labelled
        sections: drug-specific terms (from go_results_drug genes that are
        NOT in the control-upregulated set), common terms, and
        control-specific terms — mirroring the Venn annotation in the
        reference figure.

    Parameters
    ----------
    adata : AnnData
        Must contain ``pseudotime_key`` in obs and gene expression in X.
    gene_summary_df : pd.DataFrame
        Full output of ``compute_differential_dynamics`` with
        feature_source='X'.  Must contain columns: 'feature', 'mean_ctrl',
        'mean_drug', 'mw_qval' (plus optionally 'f_statistic').
    pseudotime_key : str
        Key in adata.obs with pseudotime values.
    condition_key : str
        Key in adata.obs with condition labels.
    time_key : str, optional
        Key in adata.obs with experimental time-point labels
        (e.g. 'day0', 'day1', 'day2').  Used as heatmap columns.
        Falls back to pseudotime quintile bins when None or absent.
    control_label, drug_label : str
        Labels for the two conditions (default 'DMSO' and 'O45').
    cell_type_key : str, optional
        Key in adata.obs with cell-type labels.
    cell_type_groups : dict, optional
        Mapping of group name → list of cell-type labels, e.g.::

            {"OPC": ["OPC", "COP"], "NFOL": ["NFOL"], "MOL": ["MOL"]}

        Each group gets one trajectory column and one heatmap column in
        Panel A.  If None, each unique value in cell_type_key is its own
        group.
    n_top_de : int
        Number of top DE genes per cell-type group for the heatmaps.
    go_results_ctrl : pd.DataFrame, optional
        GO enrichment for control-upregulated genes.
        Expected columns: 'term_name', '-log10_pval'.
    go_results_drug : pd.DataFrame, optional
        GO enrichment for drug-upregulated genes.
    n_top_go : int
        GO terms shown per section in Panel C.
    fdr_threshold : float
        q-value cutoff used to classify a gene as significantly DE.
    save_path : str, optional
        File path to save.

    Returns
    -------
    fig : Figure
    axes_dict : dict  mapping 'A_traj_{group}', 'A_heat_{group}', 'B', 'C'
    """
    import matplotlib.gridspec as gridspec
    from scipy.ndimage import uniform_filter1d
    from scipy.stats import pearsonr

    _apply_style()

    # ------------------------------------------------------------------ #
    # Resolve cell-type groups                                            #
    # ------------------------------------------------------------------ #
    has_celltypes = cell_type_key is not None and cell_type_key in adata.obs.columns
    if cell_type_groups is not None:
        groups = cell_type_groups
    elif has_celltypes:
        unique_ct = list(dict.fromkeys(adata.obs[cell_type_key].astype(str).values))
        groups = {ct: [ct] for ct in unique_ct}
    else:
        groups = {"All cells": []}

    group_names = list(groups.keys())
    n_groups = len(group_names)

    # ------------------------------------------------------------------ #
    # Time-point labels for heatmap columns                               #
    # ------------------------------------------------------------------ #
    has_timepoints = time_key is not None and time_key in adata.obs.columns
    if has_timepoints:
        tp_vals = adata.obs[time_key].astype(str).values
        # Preserve first-appearance order
        unique_times: List[str] = list(dict.fromkeys(tp_vals))
    else:
        # Fall back to 5 pseudotime quintile labels
        pseudotime = adata.obs[pseudotime_key].values.astype(float)
        _pt_edges = np.nanpercentile(pseudotime, np.linspace(0, 100, 6))
        tp_vals = np.full(adata.n_obs, "Q5")
        for qi in range(5):
            mask = (pseudotime >= _pt_edges[qi]) & (pseudotime <= _pt_edges[qi + 1])
            tp_vals[mask] = f"Q{qi + 1}"
        unique_times = [f"Q{i}" for i in range(1, 6)]

    n_times = len(unique_times)

    # ------------------------------------------------------------------ #
    # Pseudotime bins for trajectory plots                                #
    # ------------------------------------------------------------------ #
    pseudotime = adata.obs[pseudotime_key].values.astype(float)
    n_bins = 20
    bin_edges = np.linspace(np.nanmin(pseudotime), np.nanmax(pseudotime), n_bins + 1)
    bin_ctrs = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.clip(
        np.digitize(pseudotime, bin_edges, right=False) - 1, 0, n_bins - 1
    )

    conditions = adata.obs[condition_key].astype(str).values

    # Dense expression matrix
    X_dense = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    gene_names = list(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # ------------------------------------------------------------------ #
    # Derive log2FC and significance from gene_summary_df                 #
    # ------------------------------------------------------------------ #
    has_summary = (
        gene_summary_df is not None
        and len(gene_summary_df) > 0
        and "mean_ctrl" in gene_summary_df.columns
        and "mean_drug" in gene_summary_df.columns
    )
    if has_summary:
        _gsdf = gene_summary_df.copy()
        # log2FC = log2(mean_drug / mean_ctrl), guard against zeros
        _eps = 1e-8
        _gsdf["log2fc"] = np.log2(
            (_gsdf["mean_drug"].values + _eps) / (_gsdf["mean_ctrl"].values + _eps)
        )
        _sig_col = "mw_qval" if "mw_qval" in _gsdf.columns else None
        if _sig_col is not None:
            _gsdf["significant"] = _gsdf[_sig_col] < fdr_threshold
        else:
            _gsdf["significant"] = True
    else:
        _gsdf = pd.DataFrame(
            columns=["feature", "log2fc", "significant", "mean_ctrl", "mean_drug"]
        )

    # ------------------------------------------------------------------ #
    # Helper: mean expression per timepoint (for heatmap)                 #
    # ------------------------------------------------------------------ #
    def _tp_heatmap(cell_mask: np.ndarray, top_features: List[str]):
        valid = [f for f in top_features if f in gene_to_idx]
        if not valid:
            return np.zeros((1, n_times)), ["(no DE genes)"]
        matrix = np.full((len(valid), n_times), np.nan)
        for fi, feat in enumerate(valid):
            fidx = gene_to_idx[feat]
            for ti, tp in enumerate(unique_times):
                tp_mask = (tp_vals == tp) & cell_mask
                sub = X_dense[tp_mask, fidx]
                if len(sub) > 0:
                    matrix[fi, ti] = sub.mean()
        # Z-score rows
        row_m = np.nanmean(matrix, axis=1, keepdims=True)
        row_s = np.nanstd(matrix, axis=1, keepdims=True)
        matrix = (matrix - row_m) / (row_s + 1e-10)
        # Order rows by peak time point
        peak = np.nanargmax(matrix, axis=1)
        order = np.argsort(peak)
        return matrix[order], [valid[i] for i in order]

    # ------------------------------------------------------------------ #
    # Figure layout                                                        #
    # ------------------------------------------------------------------ #
    heatmap_col_w = max(0.35, 0.22 * n_times)  # scale with number of time points
    traj_col_w = 1.8
    left_col_w = n_groups * (traj_col_w + heatmap_col_w + 0.15)
    right_col_w = 3.6

    fig_w = left_col_w + right_col_w + 0.5
    fig_h = 8.5

    fig = plt.figure(figsize=(fig_w, fig_h))

    outer_gs = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[left_col_w, right_col_w],
        wspace=0.07,
    )

    # Left: 2 rows × (2 * n_groups) cols — alternating traj / heatmap per group
    left_gs = gridspec.GridSpecFromSubplotSpec(
        2,
        2 * n_groups,
        subplot_spec=outer_gs[0],
        hspace=0.06,
        wspace=0.04,
        height_ratios=[1.0, 1.2],
        width_ratios=([traj_col_w, heatmap_col_w] * n_groups),
    )

    # Right: B on top, C on bottom
    right_gs = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=outer_gs[1],
        hspace=0.38,
        height_ratios=[1.0, 1.6],
    )

    axes_dict: dict = {}
    _cond_color = {control_label: _CTRL_COLOR, drug_label: _DRUG_COLOR}

    # ------------------------------------------------------------------ #
    # Panel A — per group: trajectory (left) + heatmap (right)           #
    # ------------------------------------------------------------------ #
    for gi, gname in enumerate(group_names):
        # ---- cell mask ----
        if has_celltypes and groups[gname]:
            ct_vals = adata.obs[cell_type_key].astype(str).values
            grp_mask = np.isin(ct_vals, groups[gname])
        else:
            grp_mask = np.ones(adata.n_obs, dtype=bool)

        # ================================================================ #
        # Trajectory plot (col 2*gi)                                       #
        # ================================================================ #
        ax_traj = fig.add_subplot(left_gs[0, 2 * gi])
        axes_dict[f"A_traj_{gname}"] = ax_traj

        # Summarise expression as mean of z-scored matrix across DE genes
        # (use top-n_top_de genes from gene_summary_df so the trajectory
        # reflects the same genes shown in the heatmap below)
        if has_summary and len(_gsdf):
            _rank = "f_statistic" if "f_statistic" in _gsdf.columns else "log2fc"
            _top_gene_list = (
                _gsdf.sort_values(_rank, key=np.abs, ascending=False)
                .head(n_top_de)["feature"]
                .tolist()
            )
            _top_idx = [gene_to_idx[g] for g in _top_gene_list if g in gene_to_idx]
        else:
            _top_idx = list(range(min(n_top_de, X_dense.shape[1])))

        if _top_idx:
            sub_X = X_dense[np.ix_(grp_mask, _top_idx)]
            col_m = sub_X.mean(axis=0, keepdims=True)
            col_s = sub_X.std(axis=0, keepdims=True) + 1e-10
            expr_grp = ((sub_X - col_m) / col_s).mean(axis=1)  # (n_grp,)
        else:
            expr_grp = np.zeros(grp_mask.sum())

        expr_full = np.full(adata.n_obs, np.nan)
        expr_full[grp_mask] = expr_grp

        for cond, ccolor in _cond_color.items():
            cond_mask = conditions == cond
            bin_means = np.full(n_bins, np.nan)
            bin_stds = np.full(n_bins, np.nan)
            for b in range(n_bins):
                sub = expr_full[cond_mask & (bin_idx == b)]
                sub = sub[~np.isnan(sub)]
                if len(sub) > 0:
                    bin_means[b] = sub.mean()
                    bin_stds[b] = sub.std()

            valid_b = ~np.isnan(bin_means)
            if valid_b.sum() < 2:
                continue
            x = bin_ctrs[valid_b]
            m = bin_means[valid_b]
            s = bin_stds[valid_b]
            ms = uniform_filter1d(m, size=3, mode="nearest")  # smooth trend

            # Individual spread (outer, light)
            ax_traj.fill_between(
                x, ms - s, ms + s, color=ccolor, alpha=0.12, linewidth=0
            )
            # Average trend uncertainty (inner, solid)
            n_cells_b = np.array(
                [(cond_mask & (bin_idx == b)).sum() for b in np.arange(n_bins)]
            )[valid_b].astype(float)
            se = s / np.sqrt(np.maximum(n_cells_b, 1))
            ax_traj.fill_between(
                x, ms - se, ms + se, color=ccolor, alpha=0.42, linewidth=0
            )
            ax_traj.plot(x, ms, color=ccolor, lw=1.2, label=cond if gi == 0 else None)

        ax_traj.axhline(0, color=_GREY_MED, lw=0.5, ls=":")
        ax_traj.set_title(gname, fontsize=7, fontweight="bold", pad=3)
        if gi == 0:
            ax_traj.set_ylabel("Expression\nchanges", fontsize=6, labelpad=2)
            ax_traj.legend(
                fontsize=5, loc="upper left", handlelength=1.0, borderpad=0.4
            )
        ax_traj.tick_params(axis="both", labelsize=5)
        ax_traj.set_xticks([])
        ax_traj.set_xlabel("Pseudotime →", fontsize=5, labelpad=1)
        _despine(ax_traj)

        # ================================================================ #
        # Heatmap — top DE genes × real time points (col 2*gi + 1)        #
        # ================================================================ #
        ax_heat = fig.add_subplot(left_gs[1, 2 * gi + 1])
        axes_dict[f"A_heat_{gname}"] = ax_heat

        # Select top DE genes: rank by |log2fc| among significant genes first,
        # then fill from non-significant if needed
        if has_summary and len(_gsdf):
            _sig_genes = (
                _gsdf[_gsdf["significant"]]
                .sort_values("log2fc", key=np.abs, ascending=False)
                .head(n_top_de)["feature"]
                .tolist()
            )
            if len(_sig_genes) < n_top_de:
                _extra = (
                    _gsdf[~_gsdf["significant"]]
                    .sort_values("log2fc", key=np.abs, ascending=False)
                    .head(n_top_de - len(_sig_genes))["feature"]
                    .tolist()
                )
                _sig_genes = _sig_genes + _extra
            top_feats = _sig_genes
        else:
            top_feats = gene_names[:n_top_de]

        heat_data, heat_labels = _tp_heatmap(grp_mask, top_feats)
        vmax_h = float(np.nanpercentile(np.abs(heat_data), 95))
        vmax_h = max(vmax_h, 0.5)

        im = ax_heat.imshow(
            heat_data,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax_h,
            vmax=vmax_h,
            interpolation="nearest",
            rasterized=True,
        )
        ax_heat.set_yticks(np.arange(len(heat_labels)))
        ax_heat.set_yticklabels(heat_labels, fontsize=4, va="center")
        ax_heat.set_xticks(np.arange(n_times))
        ax_heat.set_xticklabels(unique_times, fontsize=5, rotation=35, ha="right")

        # Minor grid lines
        ax_heat.set_xticks(np.arange(-0.5, n_times, 1), minor=True)
        ax_heat.set_yticks(np.arange(-0.5, len(heat_labels), 1), minor=True)
        ax_heat.grid(which="minor", color="white", linewidth=0.35)
        ax_heat.tick_params(which="minor", length=0)
        _despine(ax_heat)

        # Colourbar only on last group's heatmap
        if gi == n_groups - 1:
            cbar = fig.colorbar(im, ax=ax_heat, fraction=0.10, pad=0.06, aspect=22)
            cbar.set_label("Z-score", fontsize=5, labelpad=2)
            cbar.ax.tick_params(labelsize=4)
            cbar.outline.set_linewidth(0.4)

    # ------------------------------------------------------------------ #
    # Panel B — log2FC scatter (ctrl vs drug)                             #
    # ------------------------------------------------------------------ #
    ax_B = fig.add_subplot(right_gs[0])
    axes_dict["B"] = ax_B

    if has_summary and len(_gsdf) > 2:
        x_vals = _gsdf["log2fc"].values  # drug/ctrl for all genes
        # For the scatter: x-axis = mean expression in ctrl (as proxy for
        # "ctrl dataset"), y-axis = log2fc.  This matches the reference panel B
        # which plots one condition's log2FC against the other.
        # Here we plot: x = mean_ctrl (z-scored), y = mean_drug (z-scored),
        # so each dot is a gene and the correlation shows agreement.
        mc = _gsdf["mean_ctrl"].values
        md = _gsdf["mean_drug"].values
        # z-score both axes so they share a comparable scale
        eps = 1e-8
        xz = (mc - mc.mean()) / (mc.std() + eps)
        yz = (md - md.mean()) / (md.std() + eps)

        # Colour significant up-regulated genes per condition
        sig = _gsdf["significant"].values
        up_drug = sig & (x_vals > 0)
        up_ctrl = sig & (x_vals < 0)
        neither = ~(up_drug | up_ctrl)

        ax_B.scatter(
            xz[neither],
            yz[neither],
            s=1.5,
            color=_GREY_MED,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
        ax_B.scatter(
            xz[up_drug],
            yz[up_drug],
            s=2.5,
            color=_DRUG_COLOR,
            alpha=0.6,
            linewidths=0,
            rasterized=True,
            label=f"up in {drug_label}",
        )
        ax_B.scatter(
            xz[up_ctrl],
            yz[up_ctrl],
            s=2.5,
            color=_CTRL_COLOR,
            alpha=0.6,
            linewidths=0,
            rasterized=True,
            label=f"up in {control_label}",
        )

        if len(xz) > 2:
            r, p = pearsonr(xz, yz)
            coef = np.polyfit(xz, yz, 1)
            xl = np.array([xz.min(), xz.max()])
            ax_B.plot(xl, np.polyval(coef, xl), color=_SIG_COLOR, lw=1.0)
            pstr = "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}"
            ax_B.text(
                0.97,
                0.97,
                f"R = {r:.4f}\n{pstr}",
                transform=ax_B.transAxes,
                ha="right",
                va="top",
                fontsize=6,
            )

        ax_B.axhline(0, color=_GREY_MED, lw=0.5, ls=":")
        ax_B.axvline(0, color=_GREY_MED, lw=0.5, ls=":")
        ax_B.legend(fontsize=5, loc="lower right", handlelength=0.8, markerscale=2)
    else:
        ax_B.text(
            0.5,
            0.5,
            "gene_summary_df\nnot provided",
            ha="center",
            va="center",
            fontsize=7,
            transform=ax_B.transAxes,
            color=_GREY_DARK,
        )

    ax_B.set_xlabel(f"Mean expression — {control_label} (z)", fontsize=6, labelpad=2)
    ax_B.set_ylabel(f"Mean expression — {drug_label} (z)", fontsize=6, labelpad=2)
    ax_B.set_title("B", fontsize=8, fontweight="bold", pad=4, loc="left")
    ax_B.tick_params(axis="both", labelsize=5)
    _despine(ax_B)

    # ------------------------------------------------------------------ #
    # Panel C — GO enrichment (drug-specific / common / ctrl-specific)   #
    # ------------------------------------------------------------------ #
    ax_C = fig.add_subplot(right_gs[1])
    axes_dict["C"] = ax_C

    def _extract_go(
        df: Optional[pd.DataFrame], n: int
    ) -> Tuple[List[str], List[float]]:
        if df is None or len(df) == 0:
            return [], []
        val_col = next(
            (
                c
                for c in [
                    "-log10_pval",
                    "neg_log10_pval",
                    "score",
                    "-log10(p)",
                    "log10_p",
                ]
                if c in df.columns
            ),
            "p_value" if "p_value" in df.columns else None,
        )
        name_col = next(
            (
                c
                for c in ["term_name", "term", "description", "pathway"]
                if c in df.columns
            ),
            None,
        )
        if val_col is None or name_col is None:
            return [], []
        sub = df.copy()
        if val_col == "p_value":
            sub["-log10_pval"] = -np.log10(sub[val_col].clip(lower=1e-300))
            val_col = "-log10_pval"
        top = sub.nlargest(n, val_col)
        return top[name_col].tolist(), top[val_col].tolist()

    ctrl_terms, ctrl_vals = _extract_go(go_results_ctrl, n_top_go)
    drug_terms, drug_vals = _extract_go(go_results_drug, n_top_go)

    ctrl_set = set(ctrl_terms)
    drug_set = set(drug_terms)
    shared_set = ctrl_set & drug_set
    only_ctrl = [t for t in ctrl_terms if t not in shared_set]
    only_drug = [t for t in drug_terms if t not in shared_set]
    shared = [t for t in drug_terms if t in shared_set]  # preserve drug order

    # value lookup (prefer drug value for shared terms)
    _tv: dict = {}
    for src_t, src_v in [(ctrl_terms, ctrl_vals), (drug_terms, drug_vals)]:
        for t, v in zip(src_t, src_v):
            _tv[t] = v  # drug values overwrite ctrl for shared

    sections = [
        (f"{drug_label}-specific GO", only_drug, _DRUG_COLOR),
        ("Common GO", shared, _DELTA_COLOR),
        (f"{control_label}-specific GO", only_ctrl, _CTRL_COLOR),
    ]

    has_go_data = any(len(terms) > 0 for _, terms, _ in sections)

    if has_go_data:
        all_terms: List[str] = []
        all_vals: List[float] = []
        all_colors: List[str] = []
        sep_positions: List[int] = []

        for sec_label, terms, color in sections:
            if not terms:
                continue
            if all_terms:
                sep_positions.append(len(all_terms))
            for t in terms:
                all_terms.append(t)
                all_vals.append(_tv.get(t, 0.0))
                all_colors.append(color)

        y_pos = np.arange(len(all_terms))
        ax_C.barh(y_pos, all_vals, color=all_colors, height=0.65, linewidth=0)
        ax_C.set_yticks(y_pos)
        ax_C.set_yticklabels(all_terms, fontsize=4)
        ax_C.invert_yaxis()

        for sep in sep_positions:
            ax_C.axhline(sep - 0.5, color=_GREY_MED, lw=0.6, ls="--", zorder=3)

        # Section labels on right margin
        cursor = 0
        n_total = len(all_terms)
        for sec_label, terms, color in sections:
            if not terms:
                continue
            n_sec = len(terms)
            mid_y = cursor + (n_sec - 1) / 2.0
            frac = mid_y / max(n_total - 1, 1)
            ax_C.annotate(
                sec_label,
                xy=(1.02, frac),
                xycoords=("axes fraction", "axes fraction"),
                fontsize=5,
                color=color,
                ha="left",
                va="center",
            )
            cursor += n_sec

        ax_C.set_xlabel("−log₁₀(p-value)", fontsize=6, labelpad=2)
    else:
        ax_C.text(
            0.5,
            0.5,
            "GO enrichment not available.\n\n"
            "Install gprofiler-official and\nensure network access.",
            ha="center",
            va="center",
            fontsize=6,
            transform=ax_C.transAxes,
            color=_GREY_DARK,
            linespacing=1.6,
        )
        ax_C.set_axis_off()

    ax_C.set_title("C", fontsize=8, fontweight="bold", pad=4, loc="left")
    _despine(ax_C)

    # ------------------------------------------------------------------ #
    # Panel label "A"                                                     #
    # ------------------------------------------------------------------ #
    if group_names:
        first_traj = axes_dict.get(f"A_traj_{group_names[0]}")
        if first_traj is not None:
            first_traj.text(
                -0.22,
                1.1,
                "A",
                transform=first_traj.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
            )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle(
        f"OL differentiation dynamics: {control_label} vs {drug_label}",
        fontsize=9,
        fontweight="bold",
    )
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# 10. Trajectory branch detection and visualisation
# ---------------------------------------------------------------------------


def detect_trajectory_branches(
    adata,
    pseudotime_key: str = "multiview_pseudotime",
    min_branch_cells: int = 20,
    resolution: float = 0.8,
    anchor_quantile: float = 0.05,
    min_pt_coherence: float = 0.3,
) -> np.ndarray:
    """
    Detect trajectory branches from the hypergraph spectral embedding.

    Detection is performed entirely in the **hypergraph spectral embedding**
    (``adata.obsm[f'{pseudotime_key}_spectral']``), *not* in UMAP space.
    UMAP is used only for visualisation; its non-linear projection distorts
    inter-cluster distances and is unsuitable for trajectory partitioning.

    Three constraints are applied after community detection:

    1. **Anchor** — cells in the lowest *anchor_quantile* of pseudotime are
       forced into a single root community (Branch 0) before any splitting.
       This guarantees the trajectory always departs from a shared origin.

    2. **Size filter** — communities with fewer than *min_branch_cells* cells
       are absorbed into the large community whose median pseudotime is closest.

    3. **Pseudotime coherence** — each surviving community must have a
       within-cluster Spearman ρ between cell rank-in-cluster (by spectral
       PC1) and pseudotime ≥ *min_pt_coherence*.  Communities that fail this
       test are moving backwards or sideways in pseudotime and are merged into
       their nearest-pseudotime neighbour.

    Parameters
    ----------
    adata : AnnData
        Must contain ``adata.obs[pseudotime_key]`` and
        ``adata.obsm[f'{pseudotime_key}_spectral']``.
    pseudotime_key : str
        Key in ``adata.obs`` with pseudotime values.
    min_branch_cells : int
        Minimum cells per branch after merging.
    resolution : float
        Leiden resolution (higher → more branches).
    anchor_quantile : float
        Fraction of cells at the pseudotime origin that are locked into a
        single root branch before community detection (default 0.05 = bottom 5 %).
    min_pt_coherence : float
        Minimum Spearman ρ between within-cluster ordering and pseudotime for
        a branch to be kept as distinct (default 0.3).

    Returns
    -------
    branch_labels : np.ndarray of int, shape (n_cells,)
        Branch index for every cell.  Branch 0 always contains the anchor
        (root) cells and has the lowest median pseudotime.
    """
    from scipy.stats import spearmanr as _spearmanr
    from sklearn.neighbors import NearestNeighbors

    try:
        import igraph as ig
        import leidenalg

        _LEIDEN_OK = True
    except ImportError:
        _LEIDEN_OK = False

    spectral_key = f"{pseudotime_key}_spectral"
    if spectral_key not in adata.obsm:
        raise ValueError(
            f"Spectral embedding '{spectral_key}' not found in adata.obsm. "
            "Run hypergraph-analysis first."
        )

    emb = np.asarray(adata.obsm[spectral_key], dtype=float)
    pseudotime = np.asarray(adata.obs[pseudotime_key].values, dtype=float)
    n_cells = emb.shape[0]

    # ------------------------------------------------------------------
    # Step 1: Identify anchor (root) cells
    # Cells below the anchor_quantile pseudotime threshold are locked
    # into a single pre-assigned community so every branch departs from
    # the same shared origin.
    # ------------------------------------------------------------------
    anchor_threshold = float(np.quantile(pseudotime, anchor_quantile))
    anchor_mask = pseudotime <= anchor_threshold

    # ------------------------------------------------------------------
    # Step 2: Community detection on the spectral embedding
    # Build a symmetric kNN graph; self-edges are excluded.
    # ------------------------------------------------------------------
    k = min(15, n_cells - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(emb)
    distances, indices = nbrs.kneighbors(emb)  # col 0 = self

    if _LEIDEN_OK:
        edges = []
        weights = []
        for i in range(n_cells):
            for j_pos in range(1, k + 1):
                j = int(indices[i, j_pos])
                if i < j:
                    edges.append((i, j))
                    w = float(distances[i, j_pos])
                    weights.append(1.0 / (w + 1e-10))
        g = ig.Graph(n=n_cells, edges=edges)
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights,
            resolution_parameter=resolution,
            seed=42,
        )
        raw_labels = np.array(partition.membership, dtype=int)
    else:
        # Fallback: k-means
        from sklearn.cluster import KMeans

        n_clusters = max(2, int(np.sqrt(n_cells / max(min_branch_cells, 10))))
        n_clusters = min(n_clusters, 8)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        raw_labels = km.fit_predict(emb)

    # ------------------------------------------------------------------
    # Step 3: Enforce anchor — merge all anchor-cell communities together
    # Find the community label that contains the plurality of anchor cells
    # and reassign the rest of the anchor communities to it.
    # ------------------------------------------------------------------
    anchor_indices = np.where(anchor_mask)[0]
    if len(anchor_indices) > 0:
        anchor_comm_counts: dict = {}
        for idx in anchor_indices:
            c = int(raw_labels[idx])
            anchor_comm_counts[c] = anchor_comm_counts.get(c, 0) + 1
        # The dominant community among anchor cells becomes the root community
        root_comm = max(anchor_comm_counts, key=lambda c: anchor_comm_counts[c])
        # Reassign all other anchor communities to root_comm
        for c, _ in anchor_comm_counts.items():
            if c != root_comm:
                raw_labels[raw_labels == c] = root_comm

    # ------------------------------------------------------------------
    # Step 4: Merge small communities (< min_branch_cells)
    # ------------------------------------------------------------------
    def _merge_small(labels: np.ndarray) -> np.ndarray:
        labels = labels.copy()
        for _ in range(10):  # iterate until stable
            unique, counts = np.unique(labels, return_counts=True)
            small_set = set(unique[counts < min_branch_cells])
            if not small_set:
                break
            large_medians = {
                c: float(np.median(pseudotime[labels == c]))
                for c in unique
                if c not in small_set
            }
            if not large_medians:
                break
            large_arr = np.array(list(large_medians.keys()))
            med_arr = np.array(list(large_medians.values()))
            for c in small_set:
                c_med = float(np.median(pseudotime[labels == c]))
                nearest = large_arr[np.argmin(np.abs(med_arr - c_med))]
                labels[labels == c] = nearest
        return labels

    raw_labels = _merge_small(raw_labels)

    # ------------------------------------------------------------------
    # Step 5: Pseudotime coherence filter
    # A branch is coherent if cells within it increase monotonically along
    # pseudotime as we move in the direction of spectral PC1.  We measure
    # this as Spearman ρ between the within-cluster rank on spectral PC1
    # and the pseudotime values.  Incoherent branches are merged to the
    # large branch whose median pseudotime is closest.
    # ------------------------------------------------------------------
    unique_after_merge = np.unique(raw_labels)
    large_comms = [
        c for c in unique_after_merge if (raw_labels == c).sum() >= min_branch_cells
    ]

    incoherent = set()
    for c in large_comms:
        mask_c = raw_labels == c
        pt_c = pseudotime[mask_c]
        if len(pt_c) < 4:
            continue
        # Rank cells in this cluster by spectral PC1
        pc1_c = emb[mask_c, 0]
        rank_c = np.argsort(np.argsort(pc1_c)).astype(float)
        rho, _ = _spearmanr(rank_c, pt_c)
        if np.isnan(rho) or rho < min_pt_coherence:
            incoherent.add(c)

    if incoherent:
        # Preserve the root community even if incoherent (it's the anchor)
        if len(anchor_indices) > 0:
            incoherent.discard(root_comm)
        # Merge incoherent branches to nearest-pseudotime large branch
        coherent_medians = {
            c: float(np.median(pseudotime[raw_labels == c]))
            for c in unique_after_merge
            if c not in incoherent
        }
        if coherent_medians:
            coh_arr = np.array(list(coherent_medians.keys()))
            coh_med = np.array(list(coherent_medians.values()))
            for c in incoherent:
                c_med = float(np.median(pseudotime[raw_labels == c]))
                nearest = coh_arr[np.argmin(np.abs(coh_med - c_med))]
                raw_labels[raw_labels == c] = nearest

    # One final size-filter pass after coherence merging
    raw_labels = _merge_small(raw_labels)

    # ------------------------------------------------------------------
    # Step 6: Re-index contiguously, Branch 0 = lowest median pseudotime
    # ------------------------------------------------------------------
    unique_final = np.unique(raw_labels)
    branch_medians = {
        c: float(np.median(pseudotime[raw_labels == c])) for c in unique_final
    }
    sorted_clusters = sorted(unique_final, key=lambda c: branch_medians[c])
    remap = {old: new for new, old in enumerate(sorted_clusters)}
    branch_labels = np.array([remap[int(c)] for c in raw_labels], dtype=int)

    return branch_labels


def plot_trajectory_branches(
    adata,
    pseudotime_key: str = "multiview_pseudotime",
    cell_type_key: Optional[str] = None,
    min_branch_cells: int = 20,
    resolution: float = 0.8,
    anchor_quantile: float = 0.05,
    min_pt_coherence: float = 0.3,
    save_path: Optional[str] = None,
) -> Tuple[mpl.figure.Figure, list, np.ndarray]:
    """
    Detect and visualise trajectory branches in the hypergraph spectral embedding.

    All scatter panels use the **hypergraph spectral embedding** exclusively —
    UMAP is never shown.  The spectral embedding is the space in which branch
    detection is performed, so the geometry visible here is causal: branch
    separation corresponds to genuine divergence in the regulatory manifold,
    not a 2-D projection artefact.

    Layout
    ------
    Panel A — Spectral PC1 vs PC2, coloured by branch (always shown).
              Branch 0 is always the root/anchor branch.
    Panel B — Same spectral axes, coloured by cell type (only when
              *cell_type_key* is supplied and present in ``adata.obs``).
    Panel C — Horizontal stacked bar chart of cell-type fraction per branch
              (only when *cell_type_key* is supplied).

    Parameters
    ----------
    adata : AnnData
        Must contain ``adata.obs[pseudotime_key]`` and
        ``adata.obsm[f'{pseudotime_key}_spectral']``.
    pseudotime_key : str
        Key in ``adata.obs`` with pseudotime values.
    cell_type_key : str, optional
        Column in ``adata.obs`` with cell-type labels.  Enables Panels B and C.
    min_branch_cells : int
        Minimum cells per branch; smaller communities are merged.
    resolution : float
        Leiden resolution (higher → more branches).
    anchor_quantile : float
        Fraction of lowest-pseudotime cells locked into the root branch.
    min_pt_coherence : float
        Minimum Spearman ρ to keep a branch distinct.
    save_path : str, optional
        File path to save the figure (PNG at 300 dpi).

    Returns
    -------
    fig : Figure
    axes : list of Axes
    branch_labels : np.ndarray of int, shape (n_cells,)
        Branch assignment for every cell.  Branch 0 = root/anchor branch.
    """
    _apply_style()

    # ------------------------------------------------------------------ #
    # Branch detection                                                     #
    # ------------------------------------------------------------------ #
    branch_labels = detect_trajectory_branches(
        adata,
        pseudotime_key=pseudotime_key,
        min_branch_cells=min_branch_cells,
        resolution=resolution,
        anchor_quantile=anchor_quantile,
        min_pt_coherence=min_pt_coherence,
    )
    n_branches = int(branch_labels.max()) + 1
    pseudotime = np.asarray(adata.obs[pseudotime_key].values, dtype=float)

    # ------------------------------------------------------------------ #
    # Spectral coordinates (sole embedding used throughout)               #
    # ------------------------------------------------------------------ #
    spectral_key = f"{pseudotime_key}_spectral"
    xy = np.asarray(adata.obsm[spectral_key])[:, :2]

    # ------------------------------------------------------------------ #
    # Cell-type metadata                                                   #
    # ------------------------------------------------------------------ #
    has_ct = cell_type_key is not None and cell_type_key in adata.obs.columns
    if has_ct:
        ct_labels = adata.obs[cell_type_key].astype(str).values
        unique_cts = list(dict.fromkeys(ct_labels))  # preserve insertion order
    else:
        ct_labels = None
        unique_cts = []

    # ------------------------------------------------------------------ #
    # Colour palettes                                                      #
    # ------------------------------------------------------------------ #
    _BRANCH_PALETTE = [
        "#3A6EA5",
        "#B84040",
        "#2A7A4B",
        "#E8A838",
        "#7B5EA7",
        "#4AAABF",
        "#C47A3A",
        "#888888",
        "#1A6B5A",
        "#A0522D",
        "#4B4B9A",
        "#B07840",
    ]
    _CT_PALETTE = [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
        "#86BCB6",
        "#D37295",
        "#6B8E23",
        "#C17D4B",
        "#5580A0",
        "#9F5F80",
        "#4D8076",
        "#C4A35A",
        "#8E6045",
        "#6079A5",
    ]

    branch_colors = {
        b: _BRANCH_PALETTE[b % len(_BRANCH_PALETTE)] for b in range(n_branches)
    }
    ct_colors = {
        ct: _CT_PALETTE[i % len(_CT_PALETTE)] for i, ct in enumerate(unique_cts)
    }

    # ------------------------------------------------------------------ #
    # Layout: A (always) + B + C (both only when cell_type_key given)     #
    # ------------------------------------------------------------------ #
    n_panels = 1 + int(has_ct) + int(has_ct)
    fig_width = 4.0 * n_panels
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 4.0))
    if n_panels == 1:
        axes = [axes]
    axes = list(axes)

    panel_letters = "ABC"
    panel_idx = 0

    # ------------------------------------------------------------------ #
    # Panel A — spectral embedding coloured by branch                     #
    # ------------------------------------------------------------------ #
    ax_a = axes[panel_idx]
    panel_idx += 1
    for b in range(n_branches):
        mask = branch_labels == b
        if not mask.any():
            continue
        ax_a.scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=branch_colors[b],
            s=1.5,
            linewidths=0,
            alpha=0.65,
            rasterized=True,
            label=f"Branch {b}  (n={mask.sum():,})",
        )
    ax_a.set_xlabel("Hypergraph eigenvector 1")
    ax_a.set_ylabel("Hypergraph eigenvector 2")
    ax_a.set_title("A  Trajectory branches")
    _n = min(n_branches, 8)
    _h, _l = ax_a.get_legend_handles_labels()
    ax_a.legend(_h[:_n], _l[:_n], title="Branch", loc="best", markerscale=3, fontsize=5)
    _despine(ax_a)

    if has_ct:
        # -------------------------------------------------------------- #
        # Panel B — spectral embedding coloured by cell type              #
        # -------------------------------------------------------------- #
        ax_b = axes[panel_idx]
        panel_idx += 1
        for ct in unique_cts:
            mask = ct_labels == ct
            if not mask.any():
                continue
            ax_b.scatter(
                xy[mask, 0],
                xy[mask, 1],
                c=ct_colors[ct],
                s=1.5,
                linewidths=0,
                alpha=0.65,
                rasterized=True,
                label=ct,
            )
        ax_b.set_xlabel("Hypergraph eigenvector 1")
        ax_b.set_ylabel("Hypergraph eigenvector 2")
        ax_b.set_title("B  Cell types")
        _n_ct = min(len(unique_cts), 12)
        _h_ct, _l_ct = ax_b.get_legend_handles_labels()
        ax_b.legend(
            _h_ct[:_n_ct],
            _l_ct[:_n_ct],
            title="Cell type",
            loc="best",
            markerscale=3,
            fontsize=4.5,
            ncol=1 if _n_ct <= 8 else 2,
        )
        _despine(ax_b)

        # -------------------------------------------------------------- #
        # Panel C — stacked bar: cell-type composition per branch         #
        # -------------------------------------------------------------- #
        ax_c = axes[panel_idx]
        panel_idx += 1

        comp = np.zeros((n_branches, len(unique_cts)), dtype=float)
        for b in range(n_branches):
            b_mask = branch_labels == b
            total = b_mask.sum()
            if total == 0:
                continue
            for ci, ct in enumerate(unique_cts):
                comp[b, ci] = float((ct_labels[b_mask] == ct).sum()) / total

        y_pos = np.arange(n_branches)
        lefts = np.zeros(n_branches)
        for ci, ct in enumerate(unique_cts):
            ax_c.barh(
                y_pos,
                comp[:, ci],
                left=lefts,
                color=ct_colors[ct],
                label=ct,
                height=0.6,
                linewidth=0,
            )
            lefts += comp[:, ci]

        ytick_labels = []
        for b in range(n_branches):
            b_mask = branch_labels == b
            med_pt = float(np.median(pseudotime[b_mask])) if b_mask.sum() > 0 else 0.0
            ytick_labels.append(f"B{b}  pt={med_pt:.2f}  n={b_mask.sum():,}")
        ax_c.set_yticks(y_pos)
        ax_c.set_yticklabels(ytick_labels, fontsize=5.5)
        ax_c.set_xlabel("Fraction of cells")
        ax_c.set_xlim(0, 1)
        ax_c.set_title("C  Cell-type composition per branch")

        _n_leg_c = min(len(unique_cts), 12)
        _h_c, _l_c = ax_c.get_legend_handles_labels()
        ax_c.legend(
            _h_c[:_n_leg_c],
            _l_c[:_n_leg_c],
            title="Cell type",
            loc="lower right",
            fontsize=4.5,
            ncol=1 if _n_leg_c <= 6 else 2,
        )
        _despine(ax_c)

    # ------------------------------------------------------------------ #
    # Summary title                                                        #
    # ------------------------------------------------------------------ #
    fig.suptitle(
        f"Trajectory branch analysis  ·  {n_branches} branch{'es' if n_branches != 1 else ''} detected",
        fontsize=8,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, save_path)
    return fig, axes, branch_labels
