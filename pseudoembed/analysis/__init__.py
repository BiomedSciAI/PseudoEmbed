"""
Analysis-layer modules: TF/pathway hyperedge construction, differential
trajectory dynamics, functional enrichment, and publication figures.

Deliberately free of eager submodule imports.  ``publication_plots`` pulls in
matplotlib, ``trajectory_differential_dynamics`` pulls in pyGAM, and
``tf_hypergraph_drug_analysis`` pulls in decoupler.  Import the submodule you
need directly::

    from pseudoembed.analysis.tf_hypergraph_drug_analysis import build_tf_hypergraph
    from pseudoembed.analysis.functional import run_go_enrichment

The previous eager re-export of ``core.multiview_hypergraph`` from this package
was a layering violation -- ``analysis`` reaching up into ``core`` and
advertising it under the wrong namespace -- and has been removed.  Use
:mod:`pseudoembed.core.multiview_hypergraph`, or the lazy top-level
``pseudoembed.run_multiview_hypergraph_analysis``.
"""

__all__: list[str] = []
