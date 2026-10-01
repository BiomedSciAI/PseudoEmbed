"""Figures for trajectory analysis: a theme, single-axes panels, composites.

Three modules, split by what owns the figure:

:mod:`~pseudoembed.plotting.style`
    The colourblind-safe journal theme -- palette, ``apply()``, and the small
    axis-furniture helpers (``despine``, ``panel_letter``, ``heatmap_grid``).
:mod:`~pseudoembed.plotting.panels`
    Single-axes panels.  Every one takes ``ax`` first, creates no figure, calls
    neither ``show`` nor ``savefig``, and returns a dict describing what it drew.
    They drop into any layout the caller builds.
:mod:`~pseudoembed.plotting.composites`
    Multi-panel figures that own their canvas, plus the axis helpers they need
    (``orient_axis``, ``bin_by_axis``, ``profile_along_axis``, ``panel_score``).

Typical use::

    from pseudoembed import plotting as pl

    pl.apply()                      # install the theme once
    fig, axes, info = pl.standard_panels(
        emb, tau, groups=adata.obs["timepoint"],
        group_values=adata.obs["hours"],       # ordered, so rho is printed
        reference=adata.obs["hours"],          # orients the embedding only
    )

This subpackage is deliberately separate from
:mod:`pseudoembed.analysis.publication_plots`, which is retained for backwards
compatibility.  Those functions each own a whole figure and write it through a
``save_path`` argument, so they cannot be composed into a multi-panel layout;
everything here can.  New figures should use this subpackage.

Imports are lazy (:pep:`562`) to match the parent package, so ``import
pseudoembed`` does not pull in matplotlib.
"""

from typing import TYPE_CHECKING

_LAZY: dict[str, str] = {
    # --- theme ---
    "apply": "pseudoembed.plotting.style",
    "despine": "pseudoembed.plotting.style",
    "panel_letter": "pseudoembed.plotting.style",
    "no_ticks": "pseudoembed.plotting.style",
    "heatmap_grid": "pseudoembed.plotting.style",
    "categorical_palette": "pseudoembed.plotting.style",
    "draw_order": "pseudoembed.plotting.style",
    "masked_cmap": "pseudoembed.plotting.style",
    # --- panels (ax-first) ---
    "axis_vs_groups": "pseudoembed.plotting.panels",
    "embedding": "pseudoembed.plotting.panels",
    "signed_edge_split": "pseudoembed.plotting.panels",
    "view_share": "pseudoembed.plotting.panels",
    "onset_vs_background": "pseudoembed.plotting.panels",
    "axis_heatmap": "pseudoembed.plotting.panels",
    "top_features": "pseudoembed.plotting.panels",
    "profiles": "pseudoembed.plotting.panels",
    "scatter_trend": "pseudoembed.plotting.panels",
    # --- composites and axis helpers ---
    "standard_panels": "pseudoembed.plotting.composites",
    "orient_axis": "pseudoembed.plotting.composites",
    "bin_by_axis": "pseudoembed.plotting.composites",
    "profile_along_axis": "pseudoembed.plotting.composites",
    "panel_score": "pseudoembed.plotting.composites",
    "group_profiles": "pseudoembed.plotting.composites",
    "profiles_grid": "pseudoembed.plotting.composites",
    "scatter_grid": "pseudoembed.plotting.composites",
}

if TYPE_CHECKING:  # pragma: no cover - for static analysers only
    from pseudoembed.plotting.composites import (
        bin_by_axis,
        orient_axis,
        panel_score,
        profile_along_axis,
        standard_panels,
    )
    from pseudoembed.plotting.panels import (
        axis_heatmap,
        axis_vs_groups,
        embedding,
        onset_vs_background,
        profiles,
        signed_edge_split,
        top_features,
        view_share,
    )
    from pseudoembed.plotting.style import (
        apply,
        categorical_palette,
        despine,
        draw_order,
        heatmap_grid,
        masked_cmap,
        no_ticks,
        panel_letter,
    )

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module

        value = getattr(import_module(_LAZY[name]), name)
        globals()[name] = value  # cache, so later lookups skip __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})
