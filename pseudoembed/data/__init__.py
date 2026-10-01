"""
Synthetic and reference datasets for PseudoEmbed.

Two entry points:

* :func:`make_synthetic_th1` -- a CD4+ T cell naive -> activated -> Th1
  trajectory under JAK inhibition, built from real HGNC symbols so PROGENy and
  CollecTRI score it exactly as they score real data.
* :func:`richard2018` -- the Richard et al. 2018 OT-I CD8+ T cell activation time
  course (E-MTAB-6051), **downloaded from EBI on first call** and cached.  Nothing
  is redistributed with this package.

Imports are lazy (:pep:`562`) to match the parent package -- anndata and pandas
are only loaded when a generator is actually called.
"""

from typing import TYPE_CHECKING

_LAZY: dict[str, str] = {
    "make_synthetic_th1": "pseudoembed.data.synthetic_th1",
    "SyntheticTh1Params": "pseudoembed.data.synthetic_th1",
    "GENE_PROGRAMMES": "pseudoembed.data.synthetic_th1",
    # Reference data, downloaded from EBI on first call and cached.
    "richard2018": "pseudoembed.data.richard2018",
    "marker_panels": "pseudoembed.data.richard2018",
}

if TYPE_CHECKING:  # pragma: no cover - for static analysers only
    from pseudoembed.data.richard2018 import marker_panels, richard2018
    from pseudoembed.data.synthetic_th1 import (
        GENE_PROGRAMMES,
        SyntheticTh1Params,
        make_synthetic_th1,
    )

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
