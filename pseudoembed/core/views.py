"""Building hyperedge views and fitting a pseudotime from them.

The two things every multiview analysis starts with -- construct one view per
prior, then fuse them and read an axis off the fusion -- expressed as two calls
so that a notebook does not have to sequence the eight underlying builders
itself.  Promoted from ``notebooks/weinreb_tutorial_lib.py``, which is where the
generic versions were developed and which had already donated
``orient_axis``/``bin_by_axis``/``panel_score`` to
:mod:`pseudoembed.plotting.composites`.

WHY THIS IS NOT ``run_multiview_hypergraph_analysis``
-----------------------------------------------------
:func:`pseudoembed.core.multiview_hypergraph.run_multiview_hypergraph_analysis`
is the batteries-included entry point for a *perturbation* experiment: it
requires ``control_label``/``drug_label``, runs drug-effect and state-discovery
stages, forces the PCA view whenever ``obsm['X_pca']`` exists, and omits the gene
view from its pooled hyperedge list.  A tutorial that just wants "build these
three views, fuse them, give me an axis" needs none of that, and the forced PCA
view is actively unwanted.  These functions are the small composable half.

WHY ``pool_views`` EXISTS
------------------------
Every interpretation function
(:func:`pseudoembed.core.hypergraph_interpret.edge_axis_contributions` and
friends) takes the *pooled* edge list, and the four per-edge arrays plus the
incidence matrix must stay index-aligned or a contribution is silently
attributed to the wrong edge.  Pooling was previously six lines of list
comprehension in each notebook.
"""

from __future__ import annotations

import contextlib
import io
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["build_views", "pool_views", "fit_pseudotime"]

#: Views built from a prior fetched over the network, and therefore skippable.
_PRIOR_VIEWS = ("tf", "pathway")


@contextlib.contextmanager
def _quiet(active: bool = True):
    """Swallow the builders' progress chatter on stdout, but never exceptions.

    Redirects **stdout only**.  Redirecting stderr as well would also swallow
    ``warnings.warn`` from the prior lookups, and a silenced warning there has
    already cost one debugging cycle (a broken root check survived several
    drafts because the warning never appeared).
    """
    if not active:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def build_views(
    adata,
    views: Sequence[str] = ("tf", "pathway", "gene"),
    *,
    n_cells: Optional[int] = None,
    pathway_bins: int = 50,
    tf_z: float = 1.5,
    gene_threshold: float = 1.0,
    n_genes: int = 1200,
    min_cells_per_edge: int = 10,
    gene_max_frac: float = 0.5,
    component_k: int = 15,
    pca_dims: int = 30,
    organism: str = "mouse",
    verbose: bool = True,
) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """Build one hyperedge view per named prior.

    A *view* is a set of hyperedges over the same cells, derived from one prior:
    ``tf`` groups cells by inferred regulon activity, ``pathway`` by pathway
    activity bin, ``gene`` by high expression of a single gene, ``pca`` by
    neighbourhood in PC space.  Each becomes a random-walk operator that
    :func:`fit_pseudotime` fuses.

    Parameters
    ----------
    adata
        Prepared object with log-normalised counts in ``X``.  The ``tf`` and
        ``pathway`` views need an activity matrix inferred from a prior first
        (CollecTRI / PROGENy via decoupler); this function runs that step itself
        and writes ``obsm['tf_activities']`` / ``obsm['pathway_activities']`` in
        place, so the caller does not have to sequence it.
    views
        Which views to build.  Default omits ``pca``: on several datasets the PCA
        view contributes nothing and dilutes the others, and a dead view is not
        weight-neutral.  Both prior-based views are fetched over the network and
        either can fail; an unavailable one is **skipped with a printed reason**
        rather than raising, so the analysis degrades instead of aborting.  A
        skipped view is reported unconditionally, because "view absent" is a
        coverage gap and must never be read as "that view does not help here".
    n_cells
        Defaults to ``adata.n_obs``.  Passed explicitly only to rebuild operators
        for a subset without re-deriving it.
    pathway_bins
        Quantile bins per pathway.  Deliberately high: with few bins each edge
        becomes a large clique that dominates the fused operator by size alone.
        A small dataset needs a *lower* value -- 50 bins over 250 cells leaves
        ~5 cells per bin.
    min_cells_per_edge, gene_max_frac
        The gene-view size window: an edge must cover at least
        ``min_cells_per_edge`` cells and at most ``gene_max_frac`` of them.  The
        ceiling is why a gene expressed in nearly every cell forms no edge --
        there is no selective subset to be a hyperedge.  Named here rather than
        left as a literal two levels down, because which genes *can* form an edge
        is load-bearing for interpreting which ones did.
    organism
        Passed to the prior lookup.  CollecTRI defaults to human and silently
        returns almost no regulons on mouse symbols, so this must match the data.

    Returns
    -------
    ops, E, W, T, M : dict
        Keyed by view name: the operator, the hyperedges (list of index arrays),
        the per-edge weights, the per-edge type labels and the per-edge metadata.
        Pass ``T[view]`` as ``hyperedge_types=`` to any interpretation function or
        edges come back named ``edge_298``.  Feed all four to :func:`pool_views`.
    """
    from pseudoembed.analysis.tf_hypergraph_drug_analysis import (
        build_tf_hypergraph,
        compute_tf_activities,
    )
    from pseudoembed.core.multiview_hypergraph import (
        build_gene_hypergraph,
        build_pathway_hypergraph,
        build_pca_hypergraph,
        build_view_operator,
        compute_pathway_activities,
    )

    n_cells = int(adata.n_obs) if n_cells is None else int(n_cells)
    ops: Dict = {}
    E: Dict = {}
    W: Dict = {}
    T: Dict = {}
    M: Dict = {}
    skipped: Dict[str, str] = {}

    def _add(name, edges, types, weights, meta=None):
        ops[name] = build_view_operator(edges, weights, n_cells)
        E[name], W[name], T[name] = edges, weights, list(types)
        M[name] = list(meta) if meta is not None else None

    with _quiet(not verbose):
        if "tf" in views:
            try:
                if "tf_activities" not in adata.obsm:
                    compute_tf_activities(adata, organism=organism, verbose=False)
                e, t, w = build_tf_hypergraph(
                    adata, z_score_threshold=tf_z, component_k=component_k
                )
                # The TF builder predates the metadata contract, so synthesise
                # its meta from the type label (`tf_programme_Spi1` -> `Spi1`).
                m = [
                    {
                        "view": "tf",
                        "name": str(x)
                        .replace("tf_programme_", "")
                        .replace("tf_coactivity_", ""),
                    }
                    for x in t
                ]
                _add("tf", e, t, w, m)
            except Exception as exc:  # noqa: BLE001
                skipped["tf"] = f"{type(exc).__name__}: {str(exc)[:110]}"

        if "pathway" in views:
            try:
                if "pathway_activities" not in adata.obsm:
                    compute_pathway_activities(adata, organism=organism, verbose=False)
                e, t, w, m = build_pathway_hypergraph(
                    adata, n_bins=pathway_bins, min_cells_per_edge=5, return_meta=True
                )
                _add("pathway", e, t, w, m)
            except Exception as exc:  # noqa: BLE001
                skipped["pathway"] = f"{type(exc).__name__}: {str(exc)[:110]}"

        if "pca" in views:
            # Not wrapped: PCA needs no network prior, so a failure here is a
            # real error in the caller's object, not an unavailable download.
            e, t, w, m = build_pca_hypergraph(
                adata, n_pca_dims=pca_dims, return_meta=True
            )
            _add("pca", e, t, w, m)

        if "gene" in views:
            e, t, w, m = build_gene_hypergraph(
                adata,
                n_genes=n_genes,
                threshold=gene_threshold,
                min_cells_per_edge=min_cells_per_edge,
                max_frac=gene_max_frac,
                return_meta=True,
            )
            _add("gene", e, t, w, m)

    for name, why in skipped.items():
        # Unconditional: a dropped view changes every number downstream, so it
        # is not suppressible by `verbose`.
        print(f"  {name} view UNAVAILABLE -> skipped ({why})")
    if verbose:
        print(f"{'view':10s} {'n_edges':>8s} {'mean size':>10s}")
        for k in ops:
            print(
                f"{k:10s} {len(E[k]):8,d} " f"{np.mean([len(e) for e in E[k]]):10.1f}"
            )
        print(f"\ncells N = {n_cells:,}")
    return ops, E, W, T, M


def pool_views(E, W, T, M, order: Optional[Sequence[str]] = None):
    """Concatenate the per-view bundles into one aligned edge list.

    Interpretation functions score the pooled edge list, so the four per-edge
    arrays and the incidence matrix must agree on what edge *i* is.  Doing this
    by hand is six lines of list comprehension whose failure mode is silent:
    attribution lands on the wrong edge and nothing raises.

    ``M`` may hold ``None`` for a view whose builder returned no metadata; a
    per-edge ``{"view": ..., "name": ...}`` dict is synthesised from the type
    label in that case, so ``edge_name`` never degrades to ``edge_298``.

    Parameters
    ----------
    order
        View names, fixing the pooled order.  Defaults to ``E``'s insertion
        order.  Pass it explicitly when the order must be reproducible across
        runs that built views conditionally.

    Returns
    -------
    edges, weights, types, meta, H
        ``weights`` is one concatenated float array; ``H`` is the sparse
        ``(n_cells, n_edges)`` incidence matrix.
    """
    from pseudoembed.analysis.tf_hypergraph_drug_analysis import (
        build_incidence_matrix,
    )

    order = list(E.keys()) if order is None else [k for k in order if k in E]
    if not order:
        raise ValueError("no views to pool (E is empty)")

    edges: List[np.ndarray] = []
    types: List[str] = []
    meta: List[Dict] = []
    weights: List[np.ndarray] = []

    for k in order:
        n = len(E[k])
        edges.extend(E[k])
        types.extend(list(T[k]))
        weights.append(np.asarray(W[k], dtype=float).reshape(-1))
        mk = M.get(k)
        if mk is None:
            meta.extend({"view": k, "name": str(t)} for t in T[k])
        else:
            meta.extend(dict(d) for d in mk)
        if len(types) != len(edges) or len(meta) != len(edges):
            raise ValueError(
                f"view {k!r} desynchronised the pool: {n} edges but "
                f"{len(types)} types / {len(meta)} meta rows"
            )

    weights = np.concatenate(weights) if weights else np.zeros(0)
    if len(weights) != len(edges):
        raise ValueError(
            f"pooled weights ({len(weights)}) != pooled edges ({len(edges)})"
        )

    n_cells = int(max(int(e.max()) for e in edges if len(e)) + 1) if edges else 0
    H = build_incidence_matrix(edges, n_cells)
    return edges, weights, types, meta, H


def fit_pseudotime(
    adata,
    ops,
    E,
    W,
    *,
    root_markers=None,
    time_key: str = "timepoint",
    views: Optional[Sequence[str]] = None,
    condition_key: Optional[str] = None,
    view_weights: Optional[Dict[str, float]] = None,
    n_comp: int = 10,
    tscale: int = 4,
    min_root_corr: float = 0.05,
    n_cells: Optional[int] = None,
    verbose: bool = True,
):
    """Fuse the views into one operator and read a pseudotime off the fusion.

    Returns the fitted object *and* the fused operator, so a later figure or
    attribution step never has to refit to recover them -- refitting for a
    figure is how a panel ends up disagreeing with the table above it.

    Parameters
    ----------
    root_markers
        Panel used to seed the trajectory, or ``None``.  **Not** an argmax: the
        library scores the panel, keeps cells above ``root_percentile`` (default
        90), and takes the *lowest-degree* member of that pool.  Use a panel with
        no overlap with any mature-fate panel -- a marker shared with a
        differentiated state has rooted a trajectory on a mature cell more than
        once.

        Note the filter is a percentile over **all** cells and ignores
        ``time_key``.  So when a dataset contains a genuine origin sample that does
        *not* score highly on the panel, the pool -- and hence the root -- is drawn
        from the later samples, and the pseudotime (a distance from the root) then
        places the true origin at the far end of the axis.  Pass ``None`` in that
        case: the fallback is *earliest timepoint + lowest degree*, which uses the
        clock directly instead of a marker proxy for it.
    time_key
        Column naming the sampling time.  Kept as the library default rather than
        ``None``: the downstream code indexes ``adata.obs[time_key]`` without a
        guard, so ``None`` raises ``KeyError`` instead of skipping the clock.
    views
        Subset of ``ops`` to fuse, for view-ablation.  Defaults to all of them.
    condition_key
        Optional condition column.  Note this is **not** merely cosmetic: it
        labels additional "perturbation" eigenvectors that are *retained* in the
        embedding, so passing it can change the number of embedding columns and
        hence every downstream per-edge contribution.  Leave it unset when the
        condition is collinear with time (there is no control arm to contrast).
    view_weights
        Defaults to equal weight per view.  A view that contributes nothing is
        not weight-neutral -- it dilutes the others.

    Returns
    -------
    (adata, P_fused, view_weights)
        ``adata`` is a fitted **copy** carrying ``obs['multiview_pseudotime']``
        and ``obsm['multiview_pseudotime_spectral']``.
    """
    from pseudoembed.core.multiview_hypergraph import (
        compute_multiview_pseudotime,
        fuse_view_operators,
    )

    keys = list(ops.keys()) if views is None else [k for k in views if k in ops]
    if not keys:
        raise ValueError(
            f"no views to fuse (asked for {views!r}, " f"have {list(ops)!r})"
        )

    n_cells = int(adata.n_obs) if n_cells is None else int(n_cells)
    o = {k: ops[k] for k in keys}
    vw = dict(view_weights) if view_weights else {k: 1.0 / len(o) for k in o}

    fitted = adata.copy()
    with _quiet(not verbose):
        # Named `P_fused`, never `P`: `P` is conventionally the marker-panel dict
        # in the calling notebook, and that shadowing has cost a debugging cycle.
        P_fused = fuse_view_operators(
            o,
            vw,
            fusion_method="operator",
            view_hyperedges={k: E[k] for k in o},
            view_hyperedge_weights={k: W[k] for k in o},
            n_cells=n_cells,
        )
        fitted = compute_multiview_pseudotime(
            fitted,
            P_fused=P_fused,
            view_weights=vw,
            time_key=time_key,
            condition_key=condition_key,
            # None is a documented, meaningful value: compute_multiview_pseudotime
            # falls back to earliest-timepoint + lowest-degree, which is the correct
            # root rule whenever the dataset contains a real origin sample whose
            # marker panel does not single it out. list(None) raised TypeError and
            # made that fallback unreachable through this wrapper.
            root_markers=None if root_markers is None else list(root_markers),
            n_spectral_components=n_comp,
            spectral_time_scale=tscale,
            min_root_corr=min_root_corr,
        )

    if verbose:
        emb = fitted.obsm.get("multiview_pseudotime_spectral")
        print(
            f"fused operator: {P_fused.shape}  view weights: "
            f"{ {k: round(v, 4) for k, v in vw.items()} }"
        )
        if emb is not None:
            print(f"spectral embedding: {emb.shape}")
    return fitted, P_fused, vw
