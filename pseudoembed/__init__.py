"""
PseudoEmbed: multiview hypergraph trajectory inference and gene attribution
for single-cell transcriptomics.

Two analyses are supported:

- **Multiview hypergraph analysis** -- fuse TF-activity, pathway, and PCA views
  into a single hypergraph operator
  ``S = Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}`` and read a pseudotime off its
  spectral embedding (:mod:`pseudoembed.core.multiview_hypergraph`).
- **Gene attribution analysis** -- attribute trajectory structure back to genes,
  hyperedges, and driver programmes
  (:mod:`pseudoembed.core.hypergraph_gene_attribution`,
  :mod:`pseudoembed.core.hypergraph_drivers`,
  :mod:`pseudoembed.core.hypergraph_interpret`).

The command-line entrypoint for both is ``pseudoembed hypergraph-analysis``.

Figures live in :mod:`pseudoembed.plotting`: a colourblind-safe theme, ``ax``-first
single-panel functions, and multi-panel composites.  It is imported lazily like
everything else, so ``import pseudoembed`` does not pull in matplotlib.

Heavy third-party dependencies (scanpy, decoupler, matplotlib) are imported
lazily via :pep:`562`, so ``import pseudoembed`` stays cheap; the real import
happens on first attribute access.
"""

from typing import TYPE_CHECKING

__version__ = "0.1.0"
__author__ = "Matthew Madgwick"
__email__ = "mattmadgwick@ibm.com"

# Public name -> the submodule that defines it.  Access triggers the import.
_LAZY: dict[str, str] = {
    # --- Multiview hypergraph analysis ---
    "run_multiview_hypergraph_analysis": "pseudoembed.core.multiview_hypergraph",
    "compute_multiview_pseudotime": "pseudoembed.core.multiview_hypergraph",
    "build_view_operator": "pseudoembed.core.multiview_hypergraph",
    "fuse_view_operators": "pseudoembed.core.multiview_hypergraph",
    "fuse_hyperedges": "pseudoembed.core.multiview_hypergraph",
    "build_pathway_hypergraph": "pseudoembed.core.multiview_hypergraph",
    "build_pca_hypergraph": "pseudoembed.core.multiview_hypergraph",
    "build_gene_hypergraph": "pseudoembed.core.multiview_hypergraph",
    "select_gene_edge_pool": "pseudoembed.core.multiview_hypergraph",
    "sensitivity_analysis": "pseudoembed.core.multiview_hypergraph",
    "build_tf_hypergraph": "pseudoembed.analysis.tf_hypergraph_drug_analysis",
    "build_views": "pseudoembed.core.views",
    "pool_views": "pseudoembed.core.views",
    "fit_pseudotime": "pseudoembed.core.views",
    "compute_tf_activities": "pseudoembed.analysis.tf_hypergraph_drug_analysis",
    # --- Gene attribution analysis ---
    "run_driver_analysis": "pseudoembed.core.hypergraph_drivers",
    "rank_drivers": "pseudoembed.core.hypergraph_drivers",
    "expand_programmes_to_genes": "pseudoembed.core.hypergraph_drivers",
    "edge_onset": "pseudoembed.core.hypergraph_interpret",
    "onset_group_test": "pseudoembed.core.hypergraph_interpret",
    "edge_gene_attribution": "pseudoembed.core.hypergraph_gene_attribution",
    "aggregate_gene_attribution": "pseudoembed.core.hypergraph_gene_attribution",
    "build_multilevel_driver_report": "pseudoembed.core.hypergraph_gene_attribution",
    # --- Synthetic data (ground-truth benchmarking) ---
    "make_synthetic_th1": "pseudoembed.data.synthetic_th1",
    "SyntheticTh1Params": "pseudoembed.data.synthetic_th1",
    # --- OT delay/divergence decomposition (requires POT) ---
    "decompose_perturbation_ot": "pseudoembed.analysis.ot_delay_divergence",
    # --- Reference datasets (downloaded on first use) ---
    "richard2018": "pseudoembed.data.richard2018",
    # --- Figures (see pseudoembed.plotting for the full set) ---
    "standard_panels": "pseudoembed.plotting.composites",
    # --- Supporting ---
    "load_raw_data": "pseudoembed.core.io",
    "run_functional_analysis": "pseudoembed.analysis.functional",
    "run_go_enrichment": "pseudoembed.analysis.functional",
}

if TYPE_CHECKING:  # pragma: no cover - for static analysers only
    from pseudoembed.analysis.functional import (
        run_functional_analysis,
        run_go_enrichment,
    )
    from pseudoembed.analysis.tf_hypergraph_drug_analysis import (
        build_tf_hypergraph,
        compute_tf_activities,
    )
    from pseudoembed.core.hypergraph_drivers import (
        expand_programmes_to_genes,
        rank_drivers,
        run_driver_analysis,
    )
    from pseudoembed.core.hypergraph_gene_attribution import (
        aggregate_gene_attribution,
        build_multilevel_driver_report,
        edge_gene_attribution,
    )
    from pseudoembed.core.io import load_raw_data
    from pseudoembed.core.multiview_hypergraph import (
        build_gene_hypergraph,
        build_pathway_hypergraph,
        build_pca_hypergraph,
        build_view_operator,
        compute_multiview_pseudotime,
        fuse_hyperedges,
        fuse_view_operators,
        run_multiview_hypergraph_analysis,
        select_gene_edge_pool,
        sensitivity_analysis,
    )
    from pseudoembed.data.richard2018 import richard2018
    from pseudoembed.plotting.composites import standard_panels


def __getattr__(name: str):
    """Import the defining submodule on first attribute access (:pep:`562`)."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_path), name)
    globals()[name] = value  # cache, so later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = ["__version__", *sorted(_LAZY)]
