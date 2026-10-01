"""
Core hypergraph construction, spectral readout, and gene attribution.

Deliberately free of eager submodule imports: every module here reaches for
scanpy, decoupler, or matplotlib, so importing them at package-init time would
make ``import pseudoembed.core`` cost seconds.  Import the submodule you need
directly::

    from pseudoembed.core.multiview_hypergraph import run_multiview_hypergraph_analysis
    from pseudoembed.core.hypergraph_drivers import run_driver_analysis

or use the lazy re-exports on the top-level :mod:`pseudoembed` package.
"""

__all__: list[str] = []
