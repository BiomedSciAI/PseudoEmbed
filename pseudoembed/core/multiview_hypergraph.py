"""
Multi-View Hypergraph Pseudotime
=================================

Constructs a multi-view hypergraph from biologically annotated views:

  - TF regulon view      (CollecTRI network, decoupler ULM)
  - Pathway activity view (PROGENy network, decoupler ULM)
  - PCA kNN-star view    (cell-manifold geometry; used when ``X_pca`` exists)
  - Gene view            (one hyperedge per gene, He et al. 2026; opt-in via
                          ``use_gene_view=True``)

The gene view inverts the usual roles — cells are nodes and *genes* are
hyperedges — which makes every edge-native statistic a per-gene score without
a downstream GAM or enrichment step.  See :func:`build_gene_hypergraph`.

Each view produces its own incidence matrix H_v from which a
**symmetric normalised hypergraph operator** is built directly:

    S_v = Dv^{-1/2} H_v W_v De_v^{-1} H_v^T Dv^{-1/2}

The per-view S matrices are fused via a convex combination:

    S_fused = alpha * S_TF + beta * S_pathway     alpha + beta = 1

A convex combination of symmetric PSD matrices is symmetric PSD.

Pseudotime is computed via **hypergraph spectral pseudotime**: the leading
non-trivial eigenvectors of S_fused (found with symmetric ARPACK ``eigsh``,
which is guaranteed to converge for PSD matrices) form a low-dimensional
spectral embedding, and pseudotime is the L2 distance from the root cell in
that embedding.

This approach is immune to the mixing-collapse failure of repeated P^t
row-distance pseudotime: spectral distances depend on the algebraic
connectivity of the hypergraph, not on how quickly rows of P^t equalise.

Entry point
-----------
run_multiview_hypergraph_analysis()
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import LinearOperator
from scipy.stats import spearmanr

try:
    import decoupler as dc

    DECOUPLER_AVAILABLE = True
except ImportError:
    DECOUPLER_AVAILABLE = False
    warnings.warn(
        "decoupler not available. Install with: pip install decoupler",
        ImportWarning,
    )

# Re-use utilities from the existing hypergraph module (avoid duplication)
from pseudoembed.analysis.tf_hypergraph_drug_analysis import (
    analyze_drug_effects,
    build_incidence_matrix,
    build_tf_hypergraph,
    compute_degrees,
    compute_tf_activities,
    discover_states,
)

# ============================================================================
# STEP 1: Pathway Activity Scoring
# ============================================================================


def compute_pathway_activities(
    adata: sc.AnnData,
    organism: str = "human",
    net: Optional[pd.DataFrame] = None,
    method: str = "ulm",
    verbose: bool = True,
) -> sc.AnnData:
    """
    Compute pathway activities using decoupler ULM on PROGENy.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with gene expression.
    organism : str
        Organism for PROGENy network ('human' or 'mouse').
    net : pd.DataFrame, optional
        Custom pathway-gene network with columns: source, target, weight.
        If None, downloads PROGENy via ``dc.op.progeny()``.
    method : str
        Decoupler method ('ulm', 'mlm', 'wsum').
    verbose : bool
        Print progress.  Set False to silence when called in a loop
        (e.g. across benchmark datasets).

    Returns
    -------
    adata : AnnData
        Modified in-place.  Pathway activities in ``adata.obsm['pathway_activities']``
        and metadata in ``adata.uns['pathway_activities_info']``.
    """
    if not DECOUPLER_AVAILABLE:
        raise ImportError(
            "decoupler is required for pathway scoring. "
            "Install with: pip install decoupler"
        )

    _log = print if verbose else (lambda *a, **k: None)

    _log("=" * 70)
    _log("COMPUTING PATHWAY ACTIVITIES")
    _log("=" * 70)

    # --- Normalisation heuristic warning ---------------------------------
    X = adata.raw.X if adata.raw is not None else adata.X
    X_dense = X.toarray() if issparse(X) else np.asarray(X)
    if X_dense.max() > 30:
        warnings.warn(
            "adata.X max value > 30 — data may not be log-normalised. "
            "PROGENy ULM is designed for log-normalised expression. "
            "Run sc.pp.normalize_total() + sc.pp.log1p() before this step.",
            UserWarning,
        )

    # --- Load PROGENy network --------------------------------------------
    if net is None:
        _log(f"Loading PROGENy network for {organism}...")
        net = dc.op.progeny(organism=organism)
    else:
        _log("Using custom pathway-gene network...")

    _log(f"  Network: {len(net)} interactions")
    _log(f"  Pathways: {net['source'].nunique()}")
    _log(f"  Target genes: {net['target'].nunique()}")

    # --- Run decoupler ---------------------------------------------------
    _log(f"Running {method.upper()} method...")

    _acts_key_map = {"ulm": "score_ulm", "mlm": "score_mlm", "wsum": "score_wsum"}
    if method not in _acts_key_map:
        raise ValueError(f"Unknown method: {method}. Choose from: ulm, mlm, wsum")

    runner = getattr(dc.mt, method, None)
    if runner is None:
        raise AttributeError(
            f"decoupler.mt has no method '{method}'. "
            "Your installed version may not support it."
        )
    acts_key = _acts_key_map[method]
    runner(data=adata, net=net, verbose=verbose)

    if acts_key not in adata.obsm:
        raise ValueError(
            f"Expected key '{acts_key}' not found in adata.obsm after running {method}."
        )

    acts_data = adata.obsm[acts_key]
    if isinstance(acts_data, pd.DataFrame):
        pathway_names = list(acts_data.columns)
        acts_array = acts_data.values
    else:
        acts_array = np.asarray(acts_data)
        pathway_names = [f"pathway_{i}" for i in range(acts_array.shape[1])]

    adata.obsm["pathway_activities"] = acts_array

    adata.uns["pathway_activities_info"] = {
        "method": method,
        "organism": organism,
        "n_pathways": len(pathway_names),
        "pathway_names": pathway_names,
        "network_size": len(net),
    }

    _log(f"✓ Computed activities for {len(pathway_names)} pathways")
    _log(f"  Stored in: adata.obsm['pathway_activities']")

    return adata


# ============================================================================
# STEP 2: Pathway Hyperedge Construction
# ============================================================================


def build_pathway_hypergraph(
    adata: sc.AnnData,
    n_bins: int = 4,
    min_cells_per_edge: int = 5,
    max_cells_per_edge: Optional[int] = None,
    pathway_activity_key: str = "pathway_activities",
    random_seed: int = 42,
    return_meta: bool = False,
) -> Union[
    Tuple[List[np.ndarray], List[str], np.ndarray],
    Tuple[List[np.ndarray], List[str], np.ndarray, List[Dict]],
]:
    """
    Build hyperedges from pathway activity scores.

    Each pathway is binned into *n_bins* quantile groups.  Each non-empty bin
    with >= *min_cells_per_edge* cells becomes one hyperedge whose weight
    equals the mean pathway activity of its member cells.

    Parameters
    ----------
    adata : AnnData
        Annotated data with pathway activities in ``adata.obsm[pathway_activity_key]``.
    n_bins : int
        Number of quantile bins per pathway (default 4 = quartiles).
    min_cells_per_edge : int
        Minimum cells required for a valid hyperedge.
    max_cells_per_edge : int or None
        Maximum cells per hyperedge (randomly sub-sampled with fixed seed).
        ``None`` (default) keeps all cells in each bin — strongly recommended
        for large datasets where a hard cap of 200 would discard the majority
        of each bin and produce a disconnected operator.
    pathway_activity_key : str
        Key in adata.obsm for pathway activities.
    random_seed : int
        Seed for reproducible sub-sampling.
    return_meta : bool
        If True, additionally return a per-edge metadata list.  Default False
        so the historical 3-tuple contract is preserved for existing callers.

    Returns
    -------
    hyperedges : List[np.ndarray]
        List of hyperedges (each is an array of cell indices).
    hyperedge_types : List[str]
        Type label for each hyperedge, formatted ``pathway_{name}_bin{k}``.
    hyperedge_weights : np.ndarray
        Weight for each hyperedge.
    hyperedge_meta : List[Dict]
        Only when ``return_meta=True``.  One dict per edge with keys
        ``view``, ``name``, ``bin``, ``n_bins``, ``mean_activity``, ``n_cells``.
        Note that ``mean_activity`` is *signed*, whereas the edge weight is its
        absolute value — the sign says whether the bin is up- or down-regulated
        for that pathway, which the weight alone cannot express.
    """
    print("=" * 70)
    print("BUILDING PATHWAY-BASED HYPERGRAPH")
    print("=" * 70)

    if pathway_activity_key not in adata.obsm:
        raise ValueError(
            f"Pathway activities not found in adata.obsm['{pathway_activity_key}']. "
            "Run compute_pathway_activities() first."
        )

    acts = adata.obsm[pathway_activity_key]  # (n_cells, n_pathways)
    n_cells, n_pathways = acts.shape

    if "pathway_activities_info" in adata.uns:
        pathway_names = adata.uns["pathway_activities_info"]["pathway_names"]
    else:
        pathway_names = [f"pathway_{i}" for i in range(n_pathways)]

    rng = np.random.default_rng(random_seed)

    hyperedges: List[np.ndarray] = []
    hyperedge_types: List[str] = []
    hyperedge_weights: List[float] = []
    hyperedge_meta: List[Dict] = []

    print(f"  Pathways: {n_pathways}, bins per pathway: {n_bins}")

    for p_idx in range(n_pathways):
        pathway_name = pathway_names[p_idx]
        activity = acts[:, p_idx]

        try:
            bin_labels = pd.qcut(activity, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            # All identical values → skip
            continue

        for bin_id in range(n_bins):
            bin_mask = np.where(bin_labels == bin_id)[0]

            if len(bin_mask) < min_cells_per_edge:
                continue

            cells = bin_mask
            if max_cells_per_edge is not None and len(cells) > max_cells_per_edge:
                cells = rng.choice(cells, max_cells_per_edge, replace=False)

            # weight = float(np.mean(activity[cells]))
            mean_activity = float(np.mean(activity[cells]))
            weight = float(np.abs(mean_activity))

            hyperedges.append(cells)
            # The edge type carries the pathway name and bin index so that edge
            # identity survives into attribution (see hypergraph_interpret).  A
            # bare "pathway_dominant" label made every pathway edge
            # indistinguishable, which meant no downstream analysis could say
            # *which* pathway drove a signal.
            hyperedge_types.append(f"pathway_{pathway_name}_bin{bin_id}")
            hyperedge_meta.append(
                {
                    "view": "pathway",
                    "name": str(pathway_name),
                    "bin": int(bin_id),
                    "n_bins": int(n_bins),
                    "mean_activity": mean_activity,
                    "n_cells": int(len(cells)),
                }
            )
            hyperedge_weights.append(weight)

    hyperedge_weights_arr = np.array(hyperedge_weights, dtype=float)

    # Shift weights so all are positive (some pathway activities can be negative),
    # then rescale to [1e-3, 1] so the operator eigenvalues stay in [0, 1].
    # if len(hyperedge_weights_arr) > 0:
    #     min_w = hyperedge_weights_arr.min()
    #     if min_w <= 0:
    #         hyperedge_weights_arr = hyperedge_weights_arr - min_w + 1e-3
    #     max_w = hyperedge_weights_arr.max()
    #     if max_w > 0:
    #         hyperedge_weights_arr = hyperedge_weights_arr / max_w

    if len(hyperedge_weights_arr) > 0:
        max_w = hyperedge_weights_arr.max()
        if max_w > 0:
            hyperedge_weights_arr = hyperedge_weights_arr / max_w
        hyperedge_weights_arr = np.maximum(hyperedge_weights_arr, 1e-3)

    print(f"✓ Built pathway hypergraph:")
    print(f"  Total hyperedges: {len(hyperedges)}")
    if hyperedges:
        print(f"  Mean cells per edge: {np.mean([len(e) for e in hyperedges]):.1f}")

    if return_meta:
        return hyperedges, hyperedge_types, hyperedge_weights_arr, hyperedge_meta
    return hyperedges, hyperedge_types, hyperedge_weights_arr


# ============================================================================
# STEP 2b: PCA-based cell-manifold hypergraph view
# ============================================================================


def build_pca_hypergraph(
    adata: sc.AnnData,
    pca_key: str = "X_pca",
    n_pca_dims: int = 30,
    n_neighbours: int = 15,
    return_meta: bool = False,
) -> Union[
    Tuple[List[np.ndarray], List[str], np.ndarray],
    Tuple[List[np.ndarray], List[str], np.ndarray, List[Dict]],
]:
    """
    Build a kNN-star hypergraph from the PCA embedding of gene expression.

    Each cell contributes exactly one hyperedge consisting of itself and its
    *n_neighbours* nearest neighbours in PCA space.  This is the correct
    abstraction for a kNN graph in a hypergraph context: kNN relations are
    inherently local and asymmetric (cell A being near cell B does not imply
    cell B is near cell A), so each neighbourhood is a distinct higher-order
    relation among a specific group of vertices.

    Connected-component grouping is intentionally avoided.  A kNN graph on
    scRNA-seq data with k≥8 is almost always a single connected component,
    which would collapse the entire dataset into one hyperedge.  That edge's
    operator is the rank-one projector (1/n)·11ᵀ — it encodes no local
    geometry and wastes the entire PCA fusion weight on the trivial mode.

    Hyperedge weight = 1 / mean_distance_to_neighbours.  Tighter
    neighbourhoods (denser regions, well-defined cell-type clusters) receive
    higher weight; sparse transition regions receive lower weight.  Weights
    are rescaled to [1e-3, 1] before return.

    Parameters
    ----------
    adata : AnnData
        Must contain ``adata.obsm[pca_key]``.
    pca_key : str
        Key in ``adata.obsm`` for the PCA embedding (default ``'X_pca'``).
    n_pca_dims : int
        Number of PC dimensions to use (default 30).
    n_neighbours : int
        k for each local hyperedge (default 15).  Each hyperedge has size
        k+1 (the anchor cell plus its k neighbours).

    Returns
    -------
    hyperedges : List[np.ndarray]
        One hyperedge per cell, each of length k+1.
    hyperedge_types : List[str]
        All entries are ``"pca_knn_star"``.
    hyperedge_weights : np.ndarray
        Per-edge inverse-distance weights, rescaled to [1e-3, 1].
    """
    from sklearn.neighbors import NearestNeighbors

    print("=" * 70)
    print("BUILDING PCA-BASED HYPERGRAPH VIEW")
    print("=" * 70)

    if pca_key not in adata.obsm:
        raise ValueError(
            f"PCA embedding not found in adata.obsm['{pca_key}']. "
            "Run sc.pp.pca() or sc.tl.pca() first."
        )

    X = np.asarray(adata.obsm[pca_key])
    if X.shape[1] > n_pca_dims:
        X = X[:, :n_pca_dims]

    n_cells = adata.n_obs
    k_eff = min(n_neighbours, n_cells - 1)

    print(f"  PCA dims used: {X.shape[1]}  |  kNN-star k={k_eff}")

    # n_neighbors=k+1 because kneighbors returns the query cell itself at index 0
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)  # both shape (n_cells, k_eff+1)

    # distances[:, 0] == 0 (self); neighbours start at column 1
    neighbour_distances = distances[:, 1:]  # (n_cells, k_eff)
    neighbour_indices = indices[:, 1:]  # (n_cells, k_eff)

    hyperedges: List[np.ndarray] = []
    hyperedge_types: List[str] = []
    hyperedge_weights: List[float] = []
    hyperedge_meta: List[Dict] = []

    for i in range(n_cells):
        edge = np.concatenate([[i], neighbour_indices[i]])  # cell + its k neighbours
        mean_dist = float(neighbour_distances[i].mean())
        weight = 1.0 / (mean_dist + 1e-10)  # tighter cluster → higher weight
        hyperedges.append(edge)
        # Anchor cell index is recorded in both the type label and the metadata:
        # a kNN-star edge is only interpretable relative to the cell it is
        # centred on, and a shared "pca_knn_star" label loses that entirely.
        hyperedge_types.append(f"pca_knn_star_{i}")
        hyperedge_meta.append(
            {
                "view": "pca",
                "name": f"knn_star_{i}",
                "anchor": int(i),
                "mean_dist": mean_dist,
                "n_cells": int(len(edge)),
            }
        )
        hyperedge_weights.append(weight)

    hyperedge_weights_arr = np.array(hyperedge_weights, dtype=float)
    # Rescale to [1e-3, 1] so weights are on the same scale as TF/pathway views
    w_max = hyperedge_weights_arr.max()
    if w_max > 0:
        hyperedge_weights_arr = hyperedge_weights_arr / w_max
    hyperedge_weights_arr = np.maximum(hyperedge_weights_arr, 1e-3)

    print(f"✓ Built PCA kNN-star hypergraph:")
    print(f"  Hyperedges: {len(hyperedges)} (one per cell, size {k_eff + 1})")
    w = hyperedge_weights_arr
    print(f"  Weight range: [{w.min():.4f}, {w.max():.4f}], mean={w.mean():.4f}")

    if return_meta:
        return hyperedges, hyperedge_types, hyperedge_weights_arr, hyperedge_meta
    return hyperedges, hyperedge_types, hyperedge_weights_arr


# ============================================================================
# STEP 2c: Gene-as-hyperedge view (He et al. 2026)
# ============================================================================


def select_gene_edge_pool(
    adata: sc.AnnData,
    n_genes: int = 1200,
    force_include: Optional[List[str]] = None,
) -> List[str]:
    """
    Choose which genes become hyperedges.

    Ranks genes by ``adata.var['dispersions_norm']`` when scanpy's HVG step has
    already run, and falls back to per-gene variance otherwise — the core
    builders cannot assume ``sc.pp.highly_variable_genes`` was called, since
    only :mod:`pseudoembed.core.io` invokes it.

    Parameters
    ----------
    adata : AnnData
        Annotated data with expression in ``adata.X``.
    n_genes : int
        Number of top-ranked genes to keep.
    force_include : List[str], optional
        Genes appended regardless of rank (e.g. an anchor marker set).  Any
        that are absent from ``adata.var_names`` are dropped with a warning
        rather than silently ignored.

    Returns
    -------
    List[str]
        Gene symbols to use as hyperedges, ranked genes first.
    """
    if "dispersions_norm" in adata.var:
        score = np.asarray(adata.var["dispersions_norm"].values, dtype=float)
        score = np.nan_to_num(score, nan=-np.inf)
    else:
        X = adata.X
        if issparse(X):
            # E[x^2] - E[x]^2 without densifying the matrix.
            mean = np.asarray(X.mean(axis=0)).ravel()
            mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
            score = np.maximum(mean_sq - mean**2, 0.0)
        else:
            score = np.asarray(X, dtype=float).var(axis=0)
        warnings.warn(
            "'dispersions_norm' not found in adata.var — ranking gene edges by "
            "raw variance instead. Run sc.pp.highly_variable_genes() for the "
            "normalised-dispersion ranking used in the prototype.",
            UserWarning,
        )

    n_keep = int(min(n_genes, adata.n_vars))
    order = np.argsort(score)[::-1][:n_keep]
    pool = list(adata.var_names[order])

    if force_include:
        present = set(adata.var_names)
        missing = [g for g in force_include if g not in present]
        if missing:
            warnings.warn(
                f"{len(missing)} force_include genes are absent from adata.var_names "
                f"and were skipped: {missing[:10]}",
                UserWarning,
            )
        in_pool = set(pool)
        pool += [g for g in force_include if g in present and g not in in_pool]

    return pool


def build_gene_hypergraph(
    adata: sc.AnnData,
    genes: Optional[List[str]] = None,
    n_genes: int = 1200,
    force_include: Optional[List[str]] = None,
    threshold: float = 1.0,
    soft: bool = False,
    layer: Optional[str] = None,
    min_cells_per_edge: int = 10,
    max_frac: float = 0.5,
    return_meta: bool = False,
) -> Union[
    Tuple[List[np.ndarray], List[str], np.ndarray],
    Tuple[List[np.ndarray], List[str], np.ndarray, List[Dict]],
]:
    """
    Build one hyperedge per gene from expression level (He et al. 2026).

    Cells are nodes and *genes* are hyperedges: gene *g*'s edge is the set of
    cells where *g* is expressed above ``threshold`` standard deviations of its
    own mean.  Membership comes from a hinge on the z-scored expression,
    ``max(0, z - threshold)``.

    This inverts the usual roles, and the point is attribution rather than
    ordering: because a gene *is* an edge, every edge-native statistic
    (:func:`~pseudoembed.core.hypergraph_interpret.edge_axis_contributions`,
    edge onset, edge coupling) becomes a per-gene score directly, with no GAM
    on raw expression and no enrichment step downstream.

    Two knobs, both measured in the prototype sweep
    (``docs/notebooks/outputs/s3_gene_edge_sweep.csv``):

    * ``threshold`` — selectivity.  This is the load-bearing parameter.  An
      earlier soft-incidence attempt failed because median edge size approached
      half the dataset and the operator degenerated toward the rank-one
      projector; the hinge is what keeps edges local.
    * ``soft`` — whether the edge *weight* is the mean hinge value over its
      members (He et al.'s "incidence = expression") or a flat 1.0.  The
      default is ``False`` because binary beat soft at **every** threshold
      tested, which contradicts the graded-incidence premise of the paper: the
      win came from selectivity, not from softness.

    Edges covering more than ``max_frac`` of cells are dropped as near-global,
    and edges below ``min_cells_per_edge`` are dropped as too small.

    Parameters
    ----------
    adata : AnnData
        Annotated data with expression in ``adata.X`` (or ``adata.layers[layer]``).
    genes : List[str], optional
        Explicit gene list to use as edges.  When ``None`` (default), the pool
        is chosen by :func:`select_gene_edge_pool` using *n_genes* and
        *force_include*.
    n_genes : int
        Size of the automatically selected gene pool.  Ignored when *genes* is
        given.  Default 1200, matching the ~1,100-gene universe the GAM
        pipeline actually ranks.
    force_include : List[str], optional
        Genes to add to the automatic pool regardless of rank.  Ignored when
        *genes* is given.
    threshold : float
        Z-score hinge threshold for membership (default 1.0, the sweep winner).
    soft : bool
        Graded edge weights from mean membership (default False — see above).
    layer : str, optional
        Layer to read expression from.  ``None`` uses ``adata.X``.
    min_cells_per_edge : int
        Minimum cells for a valid gene edge (default 10).
    max_frac : float
        Drop any gene edge spanning more than this fraction of cells
        (default 0.5).
    return_meta : bool
        If True, additionally return per-edge metadata.  Default False to match
        the 3-tuple contract of the other view builders.

    Returns
    -------
    hyperedges : List[np.ndarray]
        One array of cell indices per surviving gene.
    hyperedge_types : List[str]
        Type label per edge, formatted ``gene_{symbol}``.
    hyperedge_weights : np.ndarray
        Weight per edge, rescaled to [1e-3, 1] like the other views.
    hyperedge_meta : List[Dict]
        Only when ``return_meta=True``.  Keys ``view``, ``name``, ``n_cells``,
        ``mean_membership``.

    Notes
    -----
    Per-edge axis contributions from this view correlate strongly with raw edge
    size (Spearman ρ ≈ 0.78 measured on the prototype), so a high-ranking gene
    edge is partly a large gene edge.  Size-match any enrichment test against
    this view rather than comparing to an unmatched background.
    """
    print("=" * 70)
    print("BUILDING GENE-AS-HYPEREDGE HYPERGRAPH")
    print("=" * 70)

    if genes is None:
        genes = select_gene_edge_pool(
            adata, n_genes=n_genes, force_include=force_include
        )
    else:
        present = set(adata.var_names)
        missing = [g for g in genes if g not in present]
        if missing:
            warnings.warn(
                f"{len(missing)} requested gene edges are absent from "
                f"adata.var_names and were skipped: {missing[:10]}",
                UserWarning,
            )
        genes = [g for g in genes if g in present]

    if len(genes) == 0:
        raise ValueError(
            "No genes available to build gene hyperedges. Check `genes` / "
            "`adata.var_names`."
        )

    X = adata[:, genes].layers[layer] if layer else adata[:, genes].X
    X = np.asarray(X.todense() if issparse(X) else X, dtype=float)

    mu, sd = X.mean(axis=0), X.std(axis=0)
    Z = (X - mu) / np.maximum(sd, 1e-8)
    M = np.maximum(0.0, Z - threshold)  # (n_cells, n_genes) hinge

    n_cells = adata.n_obs
    hyperedges: List[np.ndarray] = []
    hyperedge_types: List[str] = []
    hyperedge_weights: List[float] = []
    hyperedge_meta: List[Dict] = []
    dropped_global = 0
    dropped_small = 0

    print(f"  Candidate genes: {len(genes)}, z threshold: {threshold}, ")
    print(f"  weighting: {'soft (mean membership)' if soft else 'binary'}")

    for j, g in enumerate(genes):
        m = M[:, j]
        members = np.flatnonzero(m > 0)
        if len(members) < min_cells_per_edge:
            dropped_small += 1
            continue
        if len(members) > max_frac * n_cells:
            dropped_global += 1
            continue

        mean_membership = float(m[members].mean())
        hyperedges.append(members)
        hyperedge_weights.append(mean_membership if soft else 1.0)
        hyperedge_types.append(f"gene_{g}")
        hyperedge_meta.append(
            {
                "view": "gene",
                "name": str(g),
                "n_cells": int(len(members)),
                "mean_membership": mean_membership,
            }
        )

    hyperedge_weights_arr = np.array(hyperedge_weights, dtype=float)
    if hyperedge_weights_arr.size and hyperedge_weights_arr.max() > 0:
        hyperedge_weights_arr = hyperedge_weights_arr / hyperedge_weights_arr.max()
    hyperedge_weights_arr = np.maximum(hyperedge_weights_arr, 1e-3)

    print(f"✓ Built gene hypergraph:")
    print(f"  Total hyperedges: {len(hyperedges)}")
    print(f"  Dropped: {dropped_small} too small, {dropped_global} near-global")
    if hyperedges:
        sizes = np.array([len(e) for e in hyperedges])
        print(
            f"  Edge size: median {int(np.median(sizes))} "
            f"({np.median(sizes) / n_cells * 100:.1f}% of cells), max {sizes.max()}"
        )

    if return_meta:
        return hyperedges, hyperedge_types, hyperedge_weights_arr, hyperedge_meta
    return hyperedges, hyperedge_types, hyperedge_weights_arr


# ============================================================================
# STEP 3: Per-View Symmetric Normalised Operators
# ============================================================================


def _build_symmetric_operator(
    H: csr_matrix,
    weights: np.ndarray,
    node_degrees: np.ndarray,
    edge_degrees: np.ndarray,
    eps: float = 1e-10,
    zero_mask: Optional[np.ndarray] = None,
) -> LinearOperator:
    """
    Build the symmetric normalised hypergraph operator implicitly:

        S = Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}

    The operator is applied through matrix-vector products so that
    ``H @ H.T`` is never materialised.
    """
    node_deg_safe = np.maximum(node_degrees, eps)
    edge_deg_safe = np.maximum(edge_degrees, eps)
    dv_invsqrt = 1.0 / np.sqrt(node_deg_safe)
    w_de_inv = weights / edge_deg_safe
    n_cells = H.shape[0]
    zero_mask = np.zeros(n_cells, dtype=bool) if zero_mask is None else zero_mask

    # Zero-degree cells (in no hyperedge) are zeroed out, giving them no affinity
    # to anything -- including themselves.
    #
    # The previous handling assigned each such row the uniform value
    # ``sum(x)/n_cells``, which coupled a zero-degree cell to *every* other cell
    # in its row while giving it no reciprocal mass in the corresponding column.
    # That silently broke the symmetry this function is named for and promises:
    # ``rmatvec=_matvec`` asserts self-adjointness, and ``eigsh`` relies on it.
    # Measured on a 20-cell operator with 15 uncovered cells, S[0,5] was 0.05
    # against S[5,0] = 0, the adjoint identity <Sx,y> = <x,S^T y> failed
    # (0.221 vs -0.014), and eigsh returned a leading eigenvalue of 1.274 --
    # outside the documented [0, 1] bound for an SPSD operator.
    #
    # Zeroing rather than self-looping matters for the spectrum, not just for
    # symmetry.  A self-loop is also symmetric, but contributes an eigenvalue of
    # exactly 1.0 per isolated cell, and those degenerate unit eigenvalues crowd
    # out the real structure the readout is looking for: measured on a 200-cell
    # operator with two covered blocks and 6 isolated cells, self-loops put 4
    # junk eigenvectors -- supported *only* on the isolated cells -- into the top
    # 5, whereas zeroed rows leave the top eigenvectors on the two real blocks.
    has_zero = bool(zero_mask.any())

    def _matvec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        y = dv_invsqrt * x
        y = H.T @ y
        y = w_de_inv * y
        y = H @ y
        y = dv_invsqrt * y
        y = np.asarray(y, dtype=float)
        if has_zero:
            y[zero_mask] = 0.0
        return y

    def _matmat(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y = dv_invsqrt[:, None] * X  # Dv^{-1/2} X
        Y = H.T @ Y  # H^T (Dv^{-1/2} X)
        Y = w_de_inv[:, None] * Y  # W De^{-1} H^T Dv^{-1/2} X
        Y = H @ Y  # H W De^{-1} H^T Dv^{-1/2} X
        Y = dv_invsqrt[:, None] * Y  # Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2} X
        if has_zero:
            Y[zero_mask] = 0.0
        return Y

    return LinearOperator(
        shape=(n_cells, n_cells),
        matvec=_matvec,
        rmatvec=_matvec,
        matmat=_matmat,
        dtype=np.float64,
    )


def build_view_operator(
    hyperedges: List[np.ndarray],
    hyperedge_weights: np.ndarray,
    n_cells: int,
    eps: float = 1e-10,
) -> LinearOperator:
    """
    Build the symmetric normalised hypergraph operator for a single view.
    """
    if len(hyperedges) == 0:
        warnings.warn(
            "No hyperedges for this view — returning uniform operator.",
            UserWarning,
        )

        def _uniform_matvec(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float).reshape(-1)
            return np.full(n_cells, float(x.sum()) / n_cells, dtype=float)

        def _uniform_matmat(X: np.ndarray) -> np.ndarray:
            X = np.asarray(X, dtype=float)
            col_sums = X.sum(axis=0, keepdims=True) / n_cells
            return np.broadcast_to(col_sums, (n_cells, X.shape[1])).copy()

        return LinearOperator(
            shape=(n_cells, n_cells),
            matvec=_uniform_matvec,
            rmatvec=_uniform_matvec,
            matmat=_uniform_matmat,
            dtype=np.float64,
        )

    H = build_incidence_matrix(hyperedges, n_cells)
    node_degrees, edge_degrees = compute_degrees(H, hyperedge_weights)
    zero_mask = node_degrees <= 0
    node_degrees_safe = node_degrees.copy()
    node_degrees_safe[zero_mask] = 1.0
    return _build_symmetric_operator(
        H,
        hyperedge_weights,
        node_degrees_safe,
        edge_degrees,
        eps=eps,
        zero_mask=zero_mask,
    )


# ============================================================================
# STEP 4: Fusion — operator-level or edge-level
# ============================================================================


def _match_edges_by_jaccard(
    keys: List[tuple],
    key_views: List[set],
    threshold: float,
) -> List[List[int]]:
    """
    Group edge keys into clusters whose pairwise Jaccard overlap is >= threshold.

    Only edges from *different* views are ever merged: within a view, two similar
    edges are genuinely distinct structures (two overlapping kNN stars, two
    adjacent activity bins) and collapsing them would destroy resolution the view
    was built to provide.

    Uses an inverted index (member -> edges containing it) so only edges sharing
    at least one member are compared.  Exhaustive pairwise comparison would be
    O(n_edges^2), which is prohibitive at 10^4-10^5 edges.

    Greedy single-link agglomeration: each edge joins the first cluster it matches.
    Order-dependent, so `keys` should be in a deterministic order.  Single-link is
    chosen over complete-link because near-duplicate edges across views form
    chains (A~B, B~C, A somewhat~C) and requiring all-pairs agreement would leave
    most of them unmerged -- which is the current behaviour we are trying to fix.
    """
    from collections import defaultdict

    member_index: Dict[int, List[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        for m in k:
            member_index[m].append(i)

    key_sets = [set(k) for k in keys]
    cluster_of = [-1] * len(keys)
    clusters: List[List[int]] = []

    for i, ks in enumerate(key_sets):
        if cluster_of[i] != -1:
            continue
        cluster_of[i] = len(clusters)
        clusters.append([i])
        cid = cluster_of[i]
        cluster_views = set(key_views[i])

        # Candidates: edges sharing >=1 member, not yet clustered.
        seen = set()
        for m in ks:
            for j in member_index[m]:
                if j <= i or cluster_of[j] != -1 or j in seen:
                    continue
                seen.add(j)
                # Never merge two edges from the same view.
                if key_views[j] & cluster_views:
                    continue
                inter = len(ks & key_sets[j])
                union = len(ks) + len(key_sets[j]) - inter
                if union > 0 and inter / union >= threshold:
                    cluster_of[j] = cid
                    clusters[cid].append(j)
                    cluster_views |= key_views[j]

    return clusters


def fuse_hyperedges(
    view_hyperedges: Dict[str, List[np.ndarray]],
    view_hyperedge_weights: Dict[str, np.ndarray],
    view_weights: Dict[str, float],
    return_provenance: bool = False,
    match: str = "exact",
    jaccard_threshold: float = 0.5,
) -> Union[
    Tuple[List[np.ndarray], np.ndarray],
    Tuple[List[np.ndarray], np.ndarray, List[List[Tuple[str, int]]]],
]:
    """
    Merge per-view hyperedge sets into a single unified set for edge-level fusion.

    Edges that are *identical* across views (same sorted member set) are
    deduplicated: their weights are collapsed to a view-weight-averaged mean so
    that a well-represented biological signal shared by both views is not
    double-counted.  Edges unique to one view are included as-is, scaled by
    that view's normalised weight.

    Algorithm
    ---------
    1. Normalise ``view_weights`` to sum to 1.
    2. For each view, represent every hyperedge as a *frozen sorted tuple* of
       cell indices (the canonical key).
    3. Collect edges by key.  If a key appears in multiple views, average their
       weights weighted by the view's normalised importance weight.
    4. Return the deduplicated list with a single scalar weight per edge.

    Parameters
    ----------
    view_hyperedges : Dict[str, List[np.ndarray]]
        Mapping from view name to its list of hyperedges (each an array of
        cell indices).
    view_hyperedge_weights : Dict[str, np.ndarray]
        Mapping from view name to its per-edge weight array.
    view_weights : Dict[str, float]
        Relative importance of each view.  Need not sum to 1 — normalised
        internally.
    match : {'exact', 'jaccard'}
        How to decide two edges are "the same".  ``'exact'`` (default, historical
        behaviour) requires identical sorted member sets, which essentially never
        holds between differently constructed views — so it pools rather than
        merges.  ``'jaccard'`` merges edges whose overlap
        ``|A∩B| / |A∪B| >= jaccard_threshold``, comparing only edges from
        *different* views (two similar edges within one view are genuinely
        distinct structures).
    jaccard_threshold : float
        Overlap required to merge, in (0, 1].  Only used when
        ``match='jaccard'``.  Lower values merge more aggressively; note that
        merging picks the largest cluster member as representative, so a very low
        threshold grows edge sizes and risks the geodesic short-circuit that
        oversized hyperedges cause.

    Returns
    -------
    merged_edges : List[np.ndarray]
        Deduplicated hyperedge list.
    merged_weights : np.ndarray
        One scalar weight per merged edge.
    merged_provenance : List[List[Tuple[str, int]]]
        Only when ``return_provenance=True``.  For each merged edge, the list of
        ``(view_name, index_within_that_view)`` pairs that contributed to it.
        Required to map an attribution score on a fused edge back to the named
        source edge in its originating view.
    """
    if set(view_hyperedges.keys()) != set(view_hyperedge_weights.keys()):
        raise ValueError(
            "view_hyperedges and view_hyperedge_weights must have the same keys."
        )
    if set(view_hyperedges.keys()) != set(view_weights.keys()):
        raise ValueError("view_hyperedges and view_weights must have the same keys.")

    total = sum(view_weights.values())
    if total <= 0:
        raise ValueError("view_weights must have a positive sum.")
    norm_w = {k: v / total for k, v in view_weights.items()}

    # key → list of (view_weight, edge_weight) pairs
    edge_registry: Dict[tuple, List[Tuple[float, float]]] = {}
    # key → canonical member array (first time seen)
    edge_members: Dict[tuple, np.ndarray] = {}
    # key → provenance: (view_name, index within that view's edge list)
    edge_provenance: Dict[tuple, List[Tuple[str, int]]] = {}

    for view_name, edges in view_hyperedges.items():
        w_view = norm_w[view_name]
        weights = view_hyperedge_weights[view_name]
        for e_idx, (edge, ew) in enumerate(zip(edges, weights)):
            key = tuple(sorted(int(x) for x in edge))
            if key not in edge_registry:
                edge_registry[key] = []
                edge_members[key] = edge
                edge_provenance[key] = []
            edge_registry[key].append((w_view, float(ew)))
            edge_provenance[key].append((view_name, e_idx))

    if match == "jaccard":
        # Exact set equality essentially never holds between differently
        # constructed views (a kNN star and an activity bin agreeing on every
        # member is vanishingly unlikely), so exact matching pools rather than
        # merges.  Soft matching collapses edges that describe the *same*
        # neighbourhood in different views.
        if not 0.0 < jaccard_threshold <= 1.0:
            raise ValueError(
                f"jaccard_threshold must be in (0, 1], got {jaccard_threshold}."
            )
        keys = list(edge_registry.keys())
        key_views = [{v for v, _ in edge_provenance[k]} for k in keys]
        clusters = _match_edges_by_jaccard(keys, key_views, jaccard_threshold)

        merged_registry: Dict[tuple, List[Tuple[float, float]]] = {}
        merged_members: Dict[tuple, np.ndarray] = {}
        merged_prov: Dict[tuple, List[Tuple[str, int]]] = {}
        for members in clusters:
            # Representative = the largest edge in the cluster, so the merged
            # edge covers the union's dominant neighbourhood rather than an
            # arbitrary member.  Using the union itself would inflate edge size,
            # which is exactly the shortcut pathology large edges cause.
            rep = max(members, key=lambda i: (len(keys[i]), keys[i]))
            rep_key = keys[rep]
            merged_registry[rep_key] = [
                c for i in members for c in edge_registry[keys[i]]
            ]
            merged_members[rep_key] = edge_members[rep_key]
            merged_prov[rep_key] = [
                p for i in members for p in edge_provenance[keys[i]]
            ]
        n_before, n_after = len(edge_registry), len(merged_registry)
        edge_registry, edge_members, edge_provenance = (
            merged_registry,
            merged_members,
            merged_prov,
        )
        print(
            f"  Soft edge matching (Jaccard >= {jaccard_threshold}): "
            f"{n_before} → {n_after} edges "
            f"({n_before - n_after} merged across views)"
        )
    elif match != "exact":
        raise ValueError(f"match must be 'exact' or 'jaccard', got {match!r}.")

    merged_edges: List[np.ndarray] = []
    merged_weights_list: List[float] = []
    merged_provenance: List[List[Tuple[str, int]]] = []

    for key, contributions in edge_registry.items():
        # weighted average: sum(view_w * edge_w) / sum(view_w)
        # Skip edges whose contributing views all carry zero weight (e.g.
        # alpha=0 sweeps in sensitivity analysis — TF-only edges vanish).
        total_vw = sum(vw for vw, _ in contributions)
        if total_vw <= 0:
            continue
        avg_ew = sum(vw * ew for vw, ew in contributions) / total_vw
        merged_edges.append(edge_members[key])
        merged_weights_list.append(avg_ew)
        merged_provenance.append(edge_provenance[key])

    merged_weights = np.array(merged_weights_list, dtype=float)

    # ------------------------------------------------------------------
    # Make `view_weights` actually control influence.
    #
    # Measured (2026-08-07): without the correction below, `view_weights` is
    # very nearly INERT in edge fusion.  Sweeping w_A from 0.1 to 0.9 over two
    # views (200 small kNN-star edges vs 8 large bin edges), the fused
    # operator's correlation with view A alone stayed at 0.99 and with view B at
    # 0.32 across the *entire* range -- at w_A = 0.1 view A still dominated
    # completely.  Late (operator) fusion over the same inputs moved smoothly
    # (corr_B 1.00 -> 0.30), so the two fusion paths were not comparable.
    #
    # Two independent causes, both fixed here:
    #
    # 1. Within a single view, a uniform weight scale is an exact no-op.
    #    d_v = sum_e w_e H_ve, so w -> c*w sends d_v -> c*d_v, and
    #    Dv^{-1/2} W Dv^{-1/2} carries c/sqrt(c)/sqrt(c) = 1.  Verified
    #    numerically: ||Sx|| identical at w = 0.01, 1.0, 100.0.  So scaling a
    #    view's edge weights cannot change its influence -- what sets influence
    #    in the pooled operator is a view's share of total *incidence mass*
    #    (n_edges x mean_edge_size).  In the test above view B held 400 of 3600
    #    incidence entries (11.1%) while nominally weighted 0.5.
    #
    # 2. The old global `/= max_w` rescale then divided out the view-weighted
    #    averaging that had just been computed, discarding what little scaling
    #    survived (1).
    #
    # Fix: divide each view's edge weights by that view's incidence mass before
    # applying the view weight, so equal `view_weights` means equal aggregate
    # operator mass regardless of how many edges a view contributes or how large
    # they are.  Rescaling is then by the max only up to a *global* constant,
    # which is a genuine no-op for the spectrum (it cancels in Dv), so the
    # relative view balance is preserved.
    if len(merged_weights) > 0:
        # Incidence mass per view: sum of edge sizes (number of nonzeros H_ve).
        view_mass = {
            name: float(sum(len(e) for e in edges)) or 1.0
            for name, edges in view_hyperedges.items()
        }
        # Per merged edge, the mass-normalised view weight of its contributors.
        mass_scale = np.ones(len(merged_edges), dtype=float)
        for i, prov in enumerate(merged_provenance):
            contrib = [norm_w[v] / view_mass[v] for v, _ in prov]
            mass_scale[i] = float(np.mean(contrib)) if contrib else 1.0
        merged_weights = merged_weights * mass_scale

        min_w = merged_weights.min()
        if min_w <= 0:
            merged_weights = merged_weights - min_w + 1e-12
        max_w = merged_weights.max()
        if max_w > 0:
            # Global constant only -- preserves relative view balance.
            merged_weights = merged_weights / max_w

    n_shared = sum(1 for v in edge_registry.values() if len(v) > 1)
    n_total_in = sum(len(e) for e in view_hyperedges.values())
    print(
        f"  Edge fusion: {n_total_in} total edges "
        f"→ {len(merged_edges)} merged ({n_shared} deduplicated cross-view duplicates)"
    )
    if n_shared == 0 and len(view_hyperedges) > 1 and match == "exact":
        # Dedup keys on exact sorted-member equality.  Pathway quantile bins and
        # PCA kNN-stars essentially never produce byte-identical member sets, so
        # this branch is the common case rather than the exception: edge fusion
        # degenerates to "pool all edges and rescale".  Surfaced rather than
        # silent, because it means edge fusion is not doing the cross-view
        # merging its name implies.
        warnings.warn(
            "Edge-level fusion found zero cross-view duplicate hyperedges, so no "
            "merging occurred — the fused operator is the pooled union of all "
            "views' edges. Exact set equality rarely holds between differently "
            "constructed views; treat 'edge' fusion as pooling, not merging.",
            UserWarning,
        )

    if return_provenance:
        return merged_edges, merged_weights, merged_provenance
    return merged_edges, merged_weights


def fuse_view_operators(
    operators: Dict[str, LinearOperator],
    view_weights: Dict[str, float],
    eps: float = 1e-10,
    fusion_method: Literal["operator", "edge"] = "operator",
    # Required for edge-level fusion
    view_hyperedges: Optional[Dict[str, List[np.ndarray]]] = None,
    view_hyperedge_weights: Optional[Dict[str, np.ndarray]] = None,
    n_cells: Optional[int] = None,
    edge_match: Literal["exact", "jaccard"] = "exact",
    jaccard_threshold: float = 0.5,
) -> LinearOperator:
    """
    Fuse per-view symmetric normalised operators.

    Two fusion strategies are available via ``fusion_method``:

    **"operator"** (default — late fusion)
        Convex combination of the per-view operators:

            S_fused = sum_v(w_v * S_v)

        A convex combination of symmetric PSD matrices is symmetric PSD.
        Each view's structural information is preserved separately before
        being linearly blended.

    **"edge"** (early fusion)
        Hyperedges from all views are pooled into a single unified incidence
        matrix *before* the operator is built.  Exact-duplicate edges (same
        sorted member set shared across views) are deduplicated via
        view-weight-averaged weight collapsing.  The unified operator is then:

            S_fused = Dv^{-1/2} H_merged W_merged De_merged^{-1} H_merged^T Dv^{-1/2}

        This preserves higher-order multi-way relationships across views
        directly in the incidence structure rather than blending two separate
        operators.  Requires ``view_hyperedges``, ``view_hyperedge_weights``,
        and ``n_cells``.

    Parameters
    ----------
    operators : Dict[str, csr_matrix]
        Mapping from view name to its symmetric normalised operator S_v.
        Always required (used for operator fusion; shapes are validated for
        edge fusion).
    view_weights : Dict[str, float]
        Relative weight for each view.  Need not sum to 1 — normalised
        internally with a warning if the correction exceeds 1e-4.
    eps : float
        Regularisation floor passed to ``build_view_operator`` (edge fusion
        only; operator fusion ignores it).
    fusion_method : {"operator", "edge"}
        Fusion strategy.  Default ``"operator"``.
    view_hyperedges : Dict[str, List[np.ndarray]], optional
        Required when ``fusion_method="edge"``.  Per-view hyperedge lists.
    view_hyperedge_weights : Dict[str, np.ndarray], optional
        Required when ``fusion_method="edge"``.  Per-view edge weight arrays.
    n_cells : int, optional
        Required when ``fusion_method="edge"``.  Total number of cells.
    edge_match : {"exact", "jaccard"}
        Edge-fusion only.  How to decide two edges are the same; see
        :func:`fuse_hyperedges`.  Default ``"exact"`` preserves historical
        behaviour, which pools rather than merges.
    jaccard_threshold : float
        Edge-fusion only, used when ``edge_match="jaccard"``.

    Returns
    -------
    S_fused : csr_matrix
        Fused symmetric normalised operator, shape (n_cells, n_cells).

    Raises
    ------
    ValueError
        If keys are inconsistent or required edge-fusion arguments are absent.
    """
    if set(operators.keys()) != set(view_weights.keys()):
        raise ValueError(
            f"operator keys {set(operators.keys())} do not match "
            f"view_weight keys {set(view_weights.keys())}."
        )

    total = sum(view_weights.values())
    if abs(total - 1.0) > 1e-4:
        warnings.warn(
            f"view_weights sum to {total:.6f} (not 1.0). Normalising automatically.",
            UserWarning,
        )

    # ---------------------------------------------------------------
    # Edge-level (early) fusion
    # ---------------------------------------------------------------
    if fusion_method == "edge":
        if view_hyperedges is None or view_hyperedge_weights is None or n_cells is None:
            raise ValueError(
                "fusion_method='edge' requires view_hyperedges, "
                "view_hyperedge_weights, and n_cells."
            )
        if set(view_hyperedges.keys()) != set(operators.keys()):
            raise ValueError("view_hyperedges keys must match operators keys.")
        merged_edges, merged_weights = fuse_hyperedges(
            view_hyperedges,
            view_hyperedge_weights,
            view_weights,
            match=edge_match,
            jaccard_threshold=jaccard_threshold,
        )
        return build_view_operator(merged_edges, merged_weights, n_cells, eps=eps)

    # ---------------------------------------------------------------
    # Operator-level (late) fusion  — original behaviour
    # ---------------------------------------------------------------
    if fusion_method != "operator":
        raise ValueError(
            f"Unknown fusion_method '{fusion_method}'. Choose 'operator' or 'edge'."
        )

    normalised_weights = {k: v / total for k, v in view_weights.items()}
    first_operator = next(iter(operators.values()))
    shape = first_operator.shape

    def _matvec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.zeros(shape[0], dtype=float)
        for view_name, S_v in operators.items():
            y += normalised_weights[view_name] * (S_v @ x)
        return y

    def _matmat(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y = np.zeros((shape[0], X.shape[1]), dtype=float)
        for view_name, S_v in operators.items():
            # ``@`` rather than ``.matmat`` — views may be raw csr_matrix as well
            # as LinearOperator, and csr_matrix has no ``matmat``.  This path was
            # unreachable until the temporal bias began driving matmat directly.
            Y += normalised_weights[view_name] * np.asarray(S_v @ X, dtype=float)
        return Y

    return LinearOperator(
        shape=shape,
        matvec=_matvec,
        rmatvec=_matvec,
        matmat=_matmat,
        dtype=np.float64,
    )


# ============================================================================
# STEP 5: Fused Pseudotime Computation
# ============================================================================


def _apply_temporal_bias(
    S: csr_matrix,
    time_numeric: np.ndarray,
    same_weight: float,
    forward_weight: float,
    backward_penalty: float,
) -> csr_matrix:
    """
    Apply time-direction bias to a symmetric operator via element-wise scaling.

    Each non-zero entry S[i,j] is multiplied by the symmetrised scale factor:

        scale(i,j) = (scale(i→j) + scale(j→i)) / 2

    where scale(i→j) = forward_weight if t[j] > t[i], backward_penalty if
    t[j] < t[i], and same_weight otherwise.  Averaging the two directions
    preserves symmetry without any additional renormalisation: the scaled
    matrix is already a valid SPSD operator whose spectral properties are a
    direct function of the original hypergraph weights.
    """
    S_coo = S.tocoo().astype(float)
    # ``dt`` is nan wherever either endpoint is off the time axis.  Both
    # ``np.where`` predicates below are False for nan, so such a pair lands on
    # ``same_weight`` — deliberately no forward/backward claim.  Do not rewrite
    # this as an if/elif/else chain: a trailing ``else`` would capture nan and
    # call it forward.
    dt = time_numeric[S_coo.col] - time_numeric[S_coo.row]
    scale_fwd = np.where(
        dt > 0,
        forward_weight,
        np.where(dt < 0, backward_penalty, same_weight),
    )
    scale_bwd = np.where(
        dt < 0,
        forward_weight,
        np.where(dt > 0, backward_penalty, same_weight),
    )
    data = S_coo.data * ((scale_fwd + scale_bwd) / 2.0)

    return sparse.coo_matrix((data, (S_coo.row, S_coo.col)), shape=S.shape).tocsr()


def _time_transition_scale(
    src_time: np.ndarray,
    dst_time: np.ndarray,
    same_weight: float,
    forward_weight: float,
    backward_penalty: float,
) -> np.ndarray:
    dt = dst_time - src_time
    scale_fwd = np.where(
        dt > 0,
        forward_weight,
        np.where(dt < 0, backward_penalty, same_weight),
    )
    scale_bwd = np.where(
        dt < 0,
        forward_weight,
        np.where(dt > 0, backward_penalty, same_weight),
    )
    return (scale_fwd + scale_bwd) / 2.0


def _scaled_operator(
    S: Union[csr_matrix, LinearOperator], factor: float
) -> Union[csr_matrix, LinearOperator]:
    """Return ``factor * S`` without materialising a LinearOperator input."""
    if isinstance(S, csr_matrix):
        return S * factor

    def _matmat(X: np.ndarray) -> np.ndarray:
        return factor * np.asarray(S @ np.asarray(X, dtype=float), dtype=float)

    def _matvec(x: np.ndarray) -> np.ndarray:
        return _matmat(np.asarray(x, dtype=float).reshape(-1, 1)).ravel()

    return LinearOperator(
        shape=S.shape,
        matvec=_matvec,
        rmatvec=_matvec,
        matmat=_matmat,
        dtype=np.float64,
    )


def _apply_temporal_bias_operator(
    S: Union[csr_matrix, LinearOperator],
    time_numeric: np.ndarray,
    same_weight: float,
    forward_weight: float,
    backward_penalty: float,
) -> Union[csr_matrix, LinearOperator]:
    """
    Apply temporal bias to a symmetric operator.

    For ``csr_matrix`` inputs the per-entry scale is computed exactly from the
    time labels of each non-zero's row and column cells (see
    ``_apply_temporal_bias``).

    For implicit ``LinearOperator`` inputs the same per-pair scale is applied
    exactly, without materialising S, by exploiting the fact that the scale
    matrix is *block-constant*: the factor for pair (i, j) depends only on the
    time-points t_i and t_j, not on the cells themselves.  Writing P_a for the
    diagonal indicator of "cell is at time-point a"::

        S_biased  =  sum_{a,b} c(a,b) * P_a S P_b

    where ``c(a,b)`` is the symmetrised scale from ``_time_transition_scale``.
    This is algebraically identical to the ``csr_matrix`` path and costs one
    application of S per distinct time-point (typically 3-6), since the inner
    sum over b is gathered in a single pass.

    Notes
    -----
    The previous implementation used a rank-one surrogate,
    ``scale[i] * (S @ (scale * x))``, with each cell's scale taken against the
    *global mean* time.  That was not equivalent, and not merely loose:

    * It applies the pair factor **twice** — measured on a 12-cell symmetric
      operator with three time-points, a forward pair received 0.5625 = 0.75^2
      where the exact path gives 0.75, for a 26% relative Frobenius error
      against the exact result.
    * It was not directional at all.  ``_time_transition_scale`` symmetrises
      forward and backward, so every cell off the mean received the same
      factor regardless of whether its own time was early or late — contrary
      to the docstring's claim that early cells get ``forward_weight`` and
      late cells get ``backward_penalty``.

    Same-time pairs were correct under both implementations, which is why the
    discrepancy did not show up in the pseudotime summary statistics.
    """
    if isinstance(S, csr_matrix):
        return _apply_temporal_bias(
            S,
            time_numeric,
            same_weight=same_weight,
            forward_weight=forward_weight,
            backward_penalty=backward_penalty,
        )

    t = time_numeric.astype(float)
    # Cells excluded from the time axis carry nan.  They form their own group:
    # ``t == nan`` is False for every value, so building masks from
    # ``np.unique`` alone would give them an EMPTY mask and scatter zero into
    # their rows — silently deleting them from the operator.  Instead they get
    # an explicit mask, and ``_time_transition_scale`` returns ``same_weight``
    # for every pair involving them (nan compares false against both > 0 and
    # < 0), i.e. no direction claim is made about a cell with no time position.
    finite_times = np.unique(t[np.isfinite(t)])
    nan_mask = ~np.isfinite(t)
    has_nan = bool(nan_mask.any())
    if finite_times.size <= 1 and not has_nan:
        # No time structure — return original operator unchanged
        return S
    if finite_times.size <= 1:
        # One real timepoint plus off-axis cells: every pair scale is
        # ``same_weight``, so the bias is a global scalar multiple.
        return S if same_weight == 1.0 else _scaled_operator(S, same_weight)

    # Group masks, one per distinct time-point (plus one for the off-axis
    # cells), and the (n_groups, n_groups) matrix of symmetrised pair scales.
    unique_times = np.concatenate([finite_times, [np.nan]]) if has_nan else finite_times
    group_masks = [
        nan_mask if not np.isfinite(tv) else (t == tv) for tv in unique_times
    ]
    n_groups = len(unique_times)
    C = _time_transition_scale(
        unique_times[:, np.newaxis],
        unique_times[np.newaxis, :],
        same_weight,
        forward_weight,
        backward_penalty,
    )  # shape (n_groups, n_groups), symmetric by construction

    def _matmat(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Y = np.zeros((S.shape[0], X.shape[1]), dtype=float)
        # For each source group b, apply S once to the masked block, then
        # scatter into every destination group a with factor c(a, b).
        for b in range(n_groups):
            Xb = np.zeros_like(X)
            Xb[group_masks[b]] = X[group_masks[b]]
            SXb = np.asarray(S @ Xb, dtype=float)
            for a in range(n_groups):
                Y[group_masks[a]] += C[a, b] * SXb[group_masks[a]]
        return Y

    def _matvec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        return _matmat(x.reshape(-1, 1)).ravel()

    return LinearOperator(
        shape=S.shape,
        matvec=_matvec,
        rmatvec=_matvec,
        matmat=_matmat,
        dtype=np.float64,
    )


def _drop_trivial(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    mode: str = "detect",
    cv_threshold: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Drop the *trivial* (near-constant) spectral component, by test not position.

    The code this replaces did ``eigenvectors[:, 1:]`` with the comment "always
    index 0".  For the symmetric hypergraph operator
    ``S = Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}`` the trivial eigenvector is
    ``Dv^{1/2} 1`` — constant only once the ``Dv^{-1/2}`` back-transform is
    applied, which this module does not do.  On real data index 0 is therefore a
    genuine, strongly degree-aligned component (CV 0.202 / rho with sqrt(degree)
    -0.854 on Schiebinger 2019; CV 0.175 / -0.247 on Richard 2018) and dropping
    it unconditionally throws away signal.

    Constancy is judged by coefficient of variation of the *absolute* entries.
    A near-constant eigenvector may be all-negative (sign is arbitrary in an
    eigendecomposition), so ``std(v) / |mean(v)|`` on signed entries would be
    scale-correct but ``mean`` can sit near zero for a genuinely varying vector
    and inflate CV harmlessly — using ``|v|`` makes the statistic robust to the
    sign convention without changing the near-constant case.

    Parameters
    ----------
    eigenvalues, eigenvectors
        Full decomposition, already sorted by descending eigenvalue.
    mode : {"always", "detect", "never"}
        ``"always"`` (the shipped default) drops the leading component
        unconditionally.  ``"detect"`` drops it only if it tests as
        near-constant — theoretically the right test, but a measured
        regression here (see the caller's block comment), so it is opt-in.
        ``"never"`` keeps every component.  The CV is measured and printed
        under all three modes.
    cv_threshold : float
        CV below which the leading component counts as constant.

    Returns
    -------
    eigenvalues_nt, eigenvectors_nt
        The retained block.
    trivial_idx : int
        Index dropped from the full decomposition, or ``-1`` if none was.
    cv : float
        Measured CV of the candidate component (reported either way).
    """
    if mode not in {"detect", "always", "never"}:
        raise ValueError(
            f"drop_trivial_component must be 'detect', 'always' or 'never', "
            f"got {mode!r}."
        )

    v0 = np.asarray(eigenvectors[:, 0], dtype=float)
    a0 = np.abs(v0)
    mu = a0.mean()
    cv = float(a0.std() / mu) if mu > np.finfo(float).eps else np.inf

    if mode == "never":
        print(
            f"  Trivial-component test: CV(v0)={cv:.4f} — mode='never', "
            "keeping all components"
        )
        return eigenvalues, eigenvectors, -1, cv

    if mode == "detect" and cv > cv_threshold:
        print(
            f"  Trivial-component test: CV(v0)={cv:.4f} > {cv_threshold} — v0 is "
            "NOT constant, so it is a real component and is KEPT "
            "(legacy behaviour dropped it unconditionally)"
        )
        return eigenvalues, eigenvectors, -1, cv

    why = "mode='always'" if mode == "always" else f"CV(v0)={cv:.4f} <= {cv_threshold}"
    print(f"  Trivial-component test: {why} — dropping v0 as trivial")
    return eigenvalues[1:], eigenvectors[:, 1:], 0, cv


def _operator_row_mass(
    S: Union[csr_matrix, LinearOperator],
    n_cells: int,
) -> np.ndarray:
    """
    Compute row sums of an operator without materialising it.
    """
    ones = np.ones(n_cells, dtype=float)
    return np.asarray(S @ ones, dtype=float).ravel()


def _hypergraph_spectral_pseudotime(
    S: Union[csr_matrix, LinearOperator],
    root: int,
    n_spectral_components: int = 10,
    spectral_time_scale: int = 4,
    pseudotime_method: str = "geodesic",
    manifold_k: int = 15,
    embedding_scale: str = "dpt",
    root_guided_components: bool = True,
    min_root_corr: float = 0.15,
    time_numeric: Optional[np.ndarray] = None,
    drop_trivial_component: str = "always",
    trivial_cv_threshold: float = 0.05,
    condition_numeric: Optional[np.ndarray] = None,
    axis_labels_out: Optional[List[str]] = None,
) -> Tuple[np.ndarray, str, np.ndarray]:
    """
    Hypergraph spectral pseudotime from the symmetric normalised operator S.

    Parameters
    ----------
    S : csr_matrix
        Symmetric normalised hypergraph operator (SPSD), eigenvalues in [0,1].
    root : int
        Root cell index.  Its pseudotime is 0.
    n_spectral_components : int
        Number of non-trivial components to compute (before filtering).
    spectral_time_scale : int
        Exponent for ``"power"`` scaling only.  Ignored by ``"dpt"`` and ``"unit"``.
    pseudotime_method : {"geodesic", "chord"}
        Distance metric for pseudotime.  Default ``"geodesic"``.
    manifold_k : int
        Neighbours for the manifold kNN graph (geodesic only).  Default 15.
    embedding_scale : {"dpt", "power", "unit"}
        How to weight eigenvectors by their eigenvalue.

        * **"dpt"** (default) — ``w = λ/(1-λ)``.  The Haghverdi et al. DPT
          weighting.  Amplifies differences between near-1 eigenvalues so the
          slowest-relaxing (trajectory) mode dominates over branch-identity axes.
          Strongly recommended for bifurcating and convergent topologies.
        * **"power"** — ``w = λ^t`` (legacy behaviour, t=``spectral_time_scale``).
          Near-equal eigenvalues receive equal weight, causing branch-ID axes to
          pollute the embedding on Y-shaped trajectories.
        * **"unit"** — ``w = 1``.  Raw eigenvectors.  Debugging only.

    root_guided_components : bool
        If True, classify each non-trivial eigenvector as one of:
        ``"maturation"``, ``"perturbation"``, or ``"noise"``.
        Only maturation axes are used for pseudotime.  Both maturation and
        perturbation axes are retained in the returned spectral embedding so
        that downstream divergence and differential-dynamics steps have access
        to the full biologically relevant subspace.  Default True.
    min_root_corr : float
        Minimum |Spearman ρ| for an eigenvector to be classified as signal
        (maturation or perturbation) rather than noise.  Default 0.15.
    drop_trivial_component : {"always", "detect", "never"}
        Passed to :func:`_drop_trivial`, which reports the measured CV of the
        candidate component under every mode.  ``"detect"`` tests for constancy
        instead of assuming index 0 is trivial; it is not the default because
        it is a measured regression on two datasets (see that function).
    trivial_cv_threshold : float
        CV threshold for that test.
    time_numeric : ndarray, shape (n_cells,), optional
        Ordered integer time-point labels (0, 1, 2, …).  When provided,
        used as the primary external reference for maturation-axis detection,
        breaking the circularity of filtering eigenvectors against a reference
        derived from those same eigenvectors.  Falls back to L2 root distance
        in the embedding when ``None``.
    condition_numeric : ndarray, shape (n_cells,), optional
        Binary condition labels (0 = control, 1 = drug/treatment).  When
        provided alongside ``time_numeric``, used to identify perturbation axes:
        eigenvectors that do not track time but do separate conditions after
        time is partialled out.  These are preserved in the returned embedding
        but excluded from the pseudotime distance calculation.
    axis_labels_out : list, optional
        If supplied, cleared and filled in place with one label per *retained*
        embedding column (``"maturation"`` or ``"perturbation"``), aligned to
        axis 1 of the returned embedding.  Needed to select perturbation axes
        for hyperedge attribution; see ``hypergraph_interpret``.

    Returns
    -------
    tau : ndarray, shape (n_cells,), values in [0, 1]
    method_used : str
        ``"geodesic"`` or ``"chord"`` (may differ from ``pseudotime_method``
        if geodesic fell back due to disconnection).
    emb : ndarray, shape (n_cells, n_kept)
        Spectral embedding containing all signal axes (maturation +
        perturbation).  Pseudotime is computed on the maturation subset only;
        the full embedding is stored in ``adata.obsm`` for downstream use.
    """
    from scipy.sparse.linalg import ArpackNoConvergence, eigsh

    if isinstance(S, csr_matrix):
        S = S.astype(float)
    elif getattr(S, "dtype", None) != np.float64:
        S_base = S

        def _matvec(x):
            return np.asarray(S_base @ x, dtype=float)

        def _rmatvec(x):
            return np.asarray(S_base.T @ x, dtype=float)

        def _matmat(X):
            return np.asarray(S_base.matmat(X), dtype=float)

        S = LinearOperator(
            shape=S_base.shape,
            matvec=_matvec,
            rmatvec=_rmatvec,
            matmat=_matmat,
            dtype=np.float64,
        )
    n = S.shape[0]
    k = min(n_spectral_components + 1, n - 2)

    eigenvalues_all = None
    eigenvectors_all = None

    # ------------------------------------------------------------------
    # Primary: eigsh (symmetric ARPACK — always real, most reliable)
    # ------------------------------------------------------------------
    try:
        eigenvalues_all, eigenvectors_all = eigsh(
            S, k=k, which="LM", maxiter=n * 20, tol=1e-5
        )
        order = np.argsort(eigenvalues_all)[::-1]  # descending
        eigenvalues_all = eigenvalues_all[order]
        eigenvectors_all = eigenvectors_all[:, order]

        gap = (
            eigenvalues_all[0] - eigenvalues_all[1] if len(eigenvalues_all) > 1 else 0.0
        )
        print(
            f"  Spectral decomposition converged — "
            f"top eigenvalues: {eigenvalues_all[: min(6, len(eigenvalues_all))].round(4)}"
        )
        print(f"  Spectral gap (λ1−λ2): {gap:.4f}")

    except (ArpackNoConvergence, Exception) as exc:
        warnings.warn(
            f"eigsh did not converge ({exc}). Falling back to truncated SVD.",
            UserWarning,
        )

    # ------------------------------------------------------------------
    # Fallback: truncated SVD (always converges)
    # ------------------------------------------------------------------
    if eigenvectors_all is None:
        from scipy.sparse.linalg import LinearOperator, svds

        k_svd = min(n_spectral_components + 1, n - 2)
        ones = np.ones(n, dtype=float)
        col_means = np.asarray((S.T @ ones) / n, dtype=float).ravel()

        def _mv(x):
            return S @ x - col_means * float(ones @ x)

        def _rmv(x):
            return S.T @ x - ones * float(col_means @ x)

        op = LinearOperator(shape=(n, n), matvec=_mv, rmatvec=_rmv, dtype=float)
        U, Sv, _ = svds(op, k=k_svd, which="LM")
        order = np.argsort(Sv)[::-1]
        eigenvalues_all = Sv[order]
        eigenvectors_all = U[:, order]
        print(
            f"  SVD fallback — top singular values: "
            f"{eigenvalues_all[: min(5, len(eigenvalues_all))].round(4)}"
        )

    # ------------------------------------------------------------------
    # Drop the trivial component — TESTED, not assumed by position
    #
    # The leading eigenvector of S = Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2} is
    # Dv^{1/2} 1, which is constant only AFTER the Dv^{-1/2} back-transform
    # that this function does not apply.  So on real data index 0 is NOT
    # constant and is strongly degree-aligned: measured CV(u0) = 0.202 with
    # rho(u0, sqrt(degree)) = -0.854 on Schiebinger 2019, and CV = 0.175 /
    # rho = -0.247 on Richard 2018.  Dropping it unconditionally therefore
    # discards a real component.
    #
    # Do NOT 'fix' this by adding the Dv^{-1/2} back-transform.  That was
    # measured with a fixed root on two datasets and LOST: rho(tau, time)
    # 0.708 -> 0.590 at the shipping root, and within-day degree leakage moved
    # the wrong way, 0.223 -> 0.357.  It is a correct-theory / worse-empirics
    # change and was deliberately not shipped.  See
    # experiments/test_degree_back_transform.py.
    #
    # WHY THE DEFAULT IS STILL "always" DESPITE v0 NOT BEING CONSTANT
    # -------------------------------------------------------------------
    # ``mode="detect"`` was implemented and A/B'd with the root held fixed on
    # two datasets (experiments/test_time_order_and_trivial.py).  Keeping v0
    # LOSES on both:
    #
    #   Schiebinger 2019 16-day   rho_pooled 0.491 -> 0.325,
    #                             within-day degree leakage -0.088 -> +0.247
    #   Richard 2018              rho_pooled 0.901 -> 0.710,
    #                             within-day degree leakage -0.024 -> +0.387
    #
    # The reason is that v0 is degree, not biology.  Measured on the biased
    # fused operator: rho(v0, sqrt(degree)) = -0.837 while rho(v0, day) is only
    # +0.076 (Schiebinger); +0.247 vs +0.198 (Richard).  v0 also carries the
    # largest DPT weight by far (lambda 0.839 -> lambda/(1-lambda) = 5.2, against
    # 1.11 for v1), so retaining it lets degree dominate the embedding.
    #
    # In other words v0 IS the trivial vector Dv^{1/2} 1 — it is simply not
    # *constant*, because this module omits the Dv^{-1/2} back-transform that
    # would make it so.  The positional drop was therefore right for the wrong
    # stated reason.  ``mode="detect"`` is kept, and reports the CV either way,
    # so the assumption is visible and testable rather than asserted in a
    # comment; but it is NOT the default, because it is a measured regression.
    # ------------------------------------------------------------------
    eigenvalues_nt, eigenvectors_nt, _trivial_idx, _trivial_cv = _drop_trivial(
        eigenvalues_all,
        eigenvectors_all,
        mode=drop_trivial_component,
        cv_threshold=trivial_cv_threshold,
    )
    # Column j of the retained block is eigenvector ``_kept_index[j]`` of the
    # full decomposition.  Logged rather than inferred as j+2, which silently
    # assumed index 0 had been dropped.
    _kept_index = [i for i in range(eigenvectors_all.shape[1]) if i != _trivial_idx]

    # ------------------------------------------------------------------
    # Eigenvalue scaling → spectral embedding
    # ------------------------------------------------------------------
    lam = np.abs(eigenvalues_nt)

    if embedding_scale == "dpt":
        # DPT weighting: λ / (1 - λ).  Clips λ to [0, 1-eps] to avoid /0.
        eps_dpt = 1e-6
        lam_clipped = np.clip(lam, 0.0, 1.0 - eps_dpt)
        scale = lam_clipped / (1.0 - lam_clipped)
    elif embedding_scale == "power":
        scale = lam**spectral_time_scale
    elif embedding_scale == "unit":
        scale = np.ones_like(lam)
    else:
        raise ValueError(
            f"Unknown embedding_scale '{embedding_scale}'. "
            "Choose 'dpt', 'power', or 'unit'."
        )

    emb = eigenvectors_nt * scale[np.newaxis, :]

    rel_weights = scale / scale.sum() if scale.sum() > 0 else scale
    print(
        f"  Embedding scale={embedding_scale!r} — "
        f"top component weights: {rel_weights[: min(5, len(rel_weights))].round(3)}"
    )

    # ------------------------------------------------------------------
    # Eigenvector classification
    #
    # Each non-trivial eigenvector is assigned one of three categories:
    #
    #   "maturation"   — tracks experimental time; used for pseudotime
    #   "perturbation" — separates conditions after time is partialled out;
    #                    preserved in returned embedding but not used for
    #                    pseudotime distance
    #   "noise"        — neither; dropped entirely
    #
    # Using experimental time as the primary reference breaks the
    # circularity of filtering eigenvectors against a reference derived
    # from those same eigenvectors.  Experimental time is not treated as
    # a perfect proxy for differentiation state — it is used only to
    # identify which axes track the maturation direction at all.
    # ------------------------------------------------------------------
    # Build component labels array: one entry per non-trivial eigenvector
    component_labels = np.array(["noise"] * emb.shape[1], dtype=object)

    if root_guided_components and emb.shape[1] > 1:
        from scipy.stats import spearmanr as _spearmanr

        # --- Primary reference: experimental time (external, non-circular) ---
        if time_numeric is not None:
            time_ref = time_numeric.astype(float)
            ref_label = "experimental time"
            # Cells excluded from the time axis carry nan.  ``spearmanr``
            # propagates nan, so a single off-axis cell would make EVERY
            # rho_time nan and classify every axis as noise.  Correlate on the
            # cells that do have a time coordinate; the excluded cells keep
            # their embedding coordinates and their pseudotime, they just cast
            # no vote on which axes track maturation.
            _time_ok = np.isfinite(time_ref)
            if not _time_ok.all():
                print(
                    f"  {int((~_time_ok).sum())} cells are off the time axis — "
                    "excluded from the component-classification correlations"
                )
        else:
            # Fallback: L2 distance from root in the full unfiltered embedding.
            # Circular but acceptable when no time labels are available
            # (e.g. lineage sub-operator calls from compute_lineage_pseudotime).
            time_ref = np.linalg.norm(emb - emb[root], axis=1)
            ref_label = "L2 root distance (fallback)"
            _time_ok = np.ones(emb.shape[0], dtype=bool)
        print(f"  Component classification reference: {ref_label}")

        # --- Partial condition reference: condition residual after time ---
        # Regress time out of condition labels so we measure whether an
        # eigenvector separates conditions *at the same time point*.
        if condition_numeric is not None and time_numeric is not None:
            # Simple OLS: residualise condition on time
            # Fit the time-out regression on the on-axis cells only, for the
            # same nan-propagation reason as the time correlation above.
            t = time_numeric.astype(float)
            c_all = condition_numeric.astype(float)
            t_ok = t[_time_ok]
            t_c_ok = t_ok - t_ok.mean()
            denom = (t_c_ok**2).sum()
            cond_residual = np.full(len(c_all), np.nan, dtype=float)
            if denom > 0:
                beta = (t_c_ok * c_all[_time_ok]).sum() / denom
                cond_residual[_time_ok] = c_all[_time_ok] - (
                    beta * t_c_ok + c_all[_time_ok].mean()
                )
            else:
                cond_residual[_time_ok] = c_all[_time_ok]
            has_condition = True
        else:
            has_condition = False

        # --- Classify each eigenvector ---
        for j in range(emb.shape[1]):
            rho_time, _ = _spearmanr(emb[_time_ok, j], time_ref[_time_ok])
            if np.isfinite(rho_time) and abs(rho_time) >= min_root_corr:
                component_labels[j] = "maturation"
            elif has_condition:
                rho_cond, _ = _spearmanr(emb[_time_ok, j], cond_residual[_time_ok])
                if np.isfinite(rho_cond) and abs(rho_cond) >= min_root_corr:
                    component_labels[j] = "perturbation"
            # else: remains "noise"

        n_mat = (component_labels == "maturation").sum()
        n_pert = (component_labels == "perturbation").sum()
        n_drop = (component_labels == "noise").sum()
        print(
            f"  Component classification (|ρ| threshold={min_root_corr}): "
            f"{n_mat} maturation, {n_pert} perturbation, {n_drop} noise (dropped)"
        )

        # Detailed per-component log
        for j in range(emb.shape[1]):
            rho_t, _ = _spearmanr(emb[_time_ok, j], time_ref[_time_ok])
            rho_c_str = ""
            if has_condition:
                rho_c, _ = _spearmanr(emb[_time_ok, j], cond_residual[_time_ok])
                rho_c_str = f", ρ_cond={rho_c:+.3f}"
            print(
                f"    v{_kept_index[j]}: ρ_time={rho_t:+.3f}{rho_c_str} "
                f"→ {component_labels[j]}"
            )

        # Guard: if no maturation axes found, fall back to best time-correlated
        if n_mat == 0:
            warnings.warn(
                "No maturation axes found above threshold — falling back to the "
                "single eigenvector most correlated with the time reference.",
                UserWarning,
                stacklevel=3,
            )
            rhos = np.array(
                [
                    abs(_spearmanr(emb[_time_ok, j], time_ref[_time_ok])[0])
                    for j in range(emb.shape[1])
                ]
            )
            rhos = np.nan_to_num(rhos, nan=0.0)
            best = int(np.argmax(rhos))
            component_labels[best] = "maturation"
            print(
                f"  Fallback: v{_kept_index[best]} promoted to maturation "
                f"(|ρ|={rhos[best]:.3f})"
            )

    else:
        # Filtering disabled — treat all components as maturation
        component_labels[:] = "maturation"

    # Pseudotime uses maturation axes only
    mat_mask = component_labels == "maturation"
    pert_mask = component_labels == "perturbation"

    emb_pseudotime = emb[:, mat_mask]  # for distance computation
    emb_full = emb[:, mat_mask | pert_mask]  # stored in obsm

    if emb_pseudotime.shape[1] == 0:
        # Should not happen due to fallback above, but be safe
        emb_pseudotime = emb

    # ------------------------------------------------------------------
    # Pseudotime: geodesic or chord distance from root
    # ------------------------------------------------------------------
    method_used = pseudotime_method

    if pseudotime_method == "geodesic":
        tau, method_used = _geodesic_pseudotime(emb_pseudotime, root, manifold_k)
    else:
        tau = _chord_pseudotime(emb_pseudotime, root)

    # The labels of the *retained* columns, aligned to emb_full's axis 1.  Without
    # these, a caller holding the embedding cannot tell which columns are
    # perturbation axes, which is exactly what condition-vs-control attribution
    # needs to decompose.
    #
    # Surfaced through a caller-supplied list rather than a 4th return value:
    # several callers (experiments/, benchmark/, tests) unpack this as a 2- or
    # 3-tuple, and widening the tuple would break every one of them.
    if axis_labels_out is not None:
        axis_labels_out.clear()
        axis_labels_out.extend(component_labels[mat_mask | pert_mask].tolist())

    return tau, method_used, emb_full


def _lowest_degree_root(
    candidates: np.ndarray,
    operator_mass: np.ndarray,
    tiebreak_score: Optional[np.ndarray] = None,
    rtol: float = 1e-9,
) -> int:
    """Lowest-operator-degree candidate, with ties broken explicitly.

    ``np.argmin`` alone returns the lowest *array index* among tied cells, which
    is an arbitrary function of row order in the AnnData.  That is harmless when
    the minimum is unique, but a view built from few hyperedges has coarse,
    near-integer degrees and ties are then the norm rather than the exception.

    Measured on Weinreb 2020 LARRY (8,105 cells, 811 marker-positive candidates):
    the gene (875 edges), TF (404) and pathway (700) views each had exactly **one**
    cell at the pool minimum, so this helper is a no-op for them.  A 14-pathway
    hinge view with only **28** edges had **25** cells tied, and array order
    selected a mature ``Eos`` cell — rooting the trajectory on a terminal fate.

    When ``tiebreak_score`` is given (the marker score that defined the pool), the
    tie goes to the highest-scoring tied cell.  Be aware this is a weak
    discriminator, not a fix for a degenerate view: on random 25-cell subsets of
    that same pool the highest-scoring cell was the progenitor type 81.2% of the
    time against a 78.9% base rate.  Its merit is that it is deterministic and
    uses the quantity we actually care about, whereas array order carries no
    information whatsoever.  A view whose degrees cannot separate candidates
    should be rebuilt with more hyperedges, not rescued here.

    Parameters
    ----------
    candidates : ndarray of int
        Cell indices eligible to be the root.
    operator_mass : ndarray, shape (n_cells,)
        Per-cell operator row mass; the minimum over ``candidates`` is taken.
    tiebreak_score : ndarray, shape (n_cells,), optional
        Higher-is-better score used only among cells tied at the minimum degree.
        ``None`` falls back to the lowest candidate index.
    rtol : float
        Relative tolerance for treating degrees as tied.

    Returns
    -------
    int
        The chosen root cell index.
    """
    mass = operator_mass[candidates]
    tied = candidates[np.isclose(mass, mass.min(), rtol=rtol)]
    if len(tied) == 1:
        return int(tied[0])
    print(
        f"  {len(tied)} candidates tied at minimum operator degree "
        f"({'marker score' if tiebreak_score is not None else 'lowest index'} "
        f"breaks the tie); a view this coarse may not resolve a root reliably"
    )
    if tiebreak_score is None:
        return int(tied.min())
    return int(tied[np.argmax(tiebreak_score[tied])])


def _chord_pseudotime(emb: np.ndarray, root: int) -> np.ndarray:
    """L2 (chord) distance from root in spectral embedding, normalised [0,1]."""
    tau = np.linalg.norm(emb - emb[root], axis=1)
    denom = tau.max() - tau.min()
    return (tau - tau.min()) / denom if denom > 0 else np.zeros_like(tau)


def _geodesic_pseudotime(
    emb: np.ndarray,
    root: int,
    manifold_k: int,
) -> Tuple[np.ndarray, str]:
    """
    Geodesic pseudotime via Dijkstra on a kNN graph of the spectral embedding.

    Builds a weighted kNN graph (edge weights = Euclidean distance in spectral
    space) and runs Dijkstra from the root cell.  Falls back to chord distance
    with a warning if the graph is disconnected.

    Parameters
    ----------
    emb : ndarray, shape (n_cells, k)
        Spectral embedding (already scaled by λ^t).
    root : int
        Root cell index.
    manifold_k : int
        Number of nearest neighbours for the kNN graph.

    Returns
    -------
    tau : ndarray, shape (n_cells,), values in [0, 1]
    method_used : str
        ``"geodesic"`` or ``"chord"`` (if fallback was triggered).
    """
    from scipy.sparse.csgraph import connected_components, shortest_path
    from sklearn.neighbors import kneighbors_graph

    n = emb.shape[0]
    k_eff = min(manifold_k, n - 1)

    print(f"  Building manifold kNN graph (k={k_eff}, metric=euclidean)...")
    # mode='distance' gives edge weights = Euclidean distance between neighbours
    graph = kneighbors_graph(
        emb, n_neighbors=k_eff, mode="distance", include_self=False
    )

    ## TODO remove this later
    # # Symmetrise: ensure undirected graph so Dijkstra can reach all nodes
    # graph = graph + graph.T
    # # Where both directions exist, keep the smaller (true Euclidean) weight
    # graph = graph.tocsr()
    # cx = graph.tocoo()
    # cx.data = cx.data / 2  # average symmetric entries (both == same distance)
    # graph = cx.tocsr()

    # Check connectivity
    n_components, _ = connected_components(graph, directed=False)
    if n_components > 1:
        warnings.warn(
            f"Manifold graph has {n_components} connected components — "
            "not all cells are reachable from root.  "
            "Falling back to chord (L2) pseudotime.  "
            "Consider increasing manifold_k or fixing edge sparsity.",
            UserWarning,
        )
        tau = _chord_pseudotime(emb, root)
        return tau, "chord"

    print(f"  Running Dijkstra from root cell {root}...")
    # shortest_path returns (n_cells,) distances from the given indices
    dist = shortest_path(graph, method="D", indices=root, directed=False)

    # Unreachable cells (inf) fall back to max finite distance
    finite_mask = np.isfinite(dist)
    if not finite_mask.all():
        max_finite = dist[finite_mask].max() if finite_mask.any() else 1.0
        dist[~finite_mask] = max_finite
        warnings.warn(
            f"{(~finite_mask).sum()} cells had infinite geodesic distance "
            "and were clamped to the maximum finite distance.",
            UserWarning,
        )

    denom = dist.max() - dist.min()
    tau = (dist - dist.min()) / denom if denom > 0 else np.zeros_like(dist)
    print(f"  Geodesic pseudotime computed (range [{tau.min():.3f}, {tau.max():.3f}])")
    return tau, "geodesic"


def _extract_suboperator(
    S: Union[csr_matrix, LinearOperator],
    sub_idx: np.ndarray,
) -> csr_matrix:
    """
    Extract the symmetric sub-block ``S[sub_idx][:, sub_idx]`` as a csr_matrix.

    Accepts either a sparse matrix or a ``LinearOperator``.  The view/fusion
    builders in this module return matrix-free ``LinearOperator`` instances
    (``build_view_operator``, ``fuse_view_operators``), which support neither
    fancy indexing nor ``.tocsr()``.  For those, the sub-block is recovered by
    applying the operator to the indicator columns of ``sub_idx`` — one
    ``matmat`` call, materialising only ``(n_cells, n_sub)`` rather than the
    full ``(n_cells, n_cells)`` matrix.

    Parameters
    ----------
    S : csr_matrix or LinearOperator
        Symmetric operator over all cells.
    sub_idx : np.ndarray
        Integer indices of the cells forming the sub-block.

    Returns
    -------
    csr_matrix
        The ``(n_sub, n_sub)`` sub-block.
    """
    sub_idx = np.asarray(sub_idx, dtype=int)

    if sparse.issparse(S):
        return S[sub_idx][:, sub_idx].tocsr()

    if hasattr(S, "tocsr"):  # np.matrix / other array-likes exposing tocsr
        return S.tocsr()[sub_idx][:, sub_idx].tocsr()

    if isinstance(S, LinearOperator):
        n_cells, n_sub = S.shape[0], len(sub_idx)
        # Indicator columns: E[:, j] = e_{sub_idx[j]}.  S @ E selects the
        # sub-columns of S; restricting the rows then gives the sub-block.
        E = np.zeros((n_cells, n_sub), dtype=float)
        E[sub_idx, np.arange(n_sub)] = 1.0
        cols = np.asarray(S.matmat(E))  # (n_cells, n_sub)
        block = cols[sub_idx, :]  # (n_sub, n_sub)
        block = 0.5 * (block + block.T)  # enforce exact symmetry
        return sparse.csr_matrix(block)

    # Dense ndarray fallback
    dense = np.asarray(S)
    return sparse.csr_matrix(dense[np.ix_(sub_idx, sub_idx)])


def compute_lineage_pseudotime(
    S: Union[csr_matrix, LinearOperator],
    lineage_masks: Dict[str, np.ndarray],
    root: int,
    stem_mask: np.ndarray,
    n_spectral_components: int = 10,
    manifold_k: int = 50,
    min_lineage_cells: int = 20,
    pseudotime_key_prefix: str = "hypergraph_pt",
) -> Dict[str, np.ndarray]:
    """
    Compute per-lineage pseudotime from a fused hypergraph operator.

    For each lineage, extracts the sub-operator covering stem cells + that
    lineage's committed cells, computes spectral pseudotime on the sub-matrix,
    then maps results back to global cell indices.

    This mirrors Palantir's per-branch pseudotime strategy but uses the
    hypergraph operator throughout — branch identity is derived from the
    lineage_masks rather than the operator itself, keeping the method
    hypergraph-native while correctly handling the branching topology.

    Parameters
    ----------
    S : csr_matrix or LinearOperator
        Symmetric normalised fused hypergraph operator (n_cells × n_cells).
        A matrix-free ``LinearOperator`` — as returned by
        ``build_view_operator`` / ``fuse_view_operators`` — is accepted; each
        lineage sub-block is materialised on demand via
        :func:`_extract_suboperator`.
    lineage_masks : Dict[str, np.ndarray]
        Boolean masks of shape (n_cells,), one per lineage.
        Each True entry indicates a committed cell for that lineage.
    root : int
        Global index of the root cell (must be in stem_mask).
    stem_mask : np.ndarray
        Boolean mask (n_cells,) marking uncommitted / progenitor cells.
        These are always included in every lineage sub-operator.
    n_spectral_components : int
        Spectral components per sub-operator (default 10).
    manifold_k : int
        kNN for geodesic manifold graph (default 50).
    min_lineage_cells : int
        Skip lineages with fewer committed cells than this (default 20).
    pseudotime_key_prefix : str
        Prefix for returned dict keys (default "hypergraph_pt").

    Returns
    -------
    Dict[str, np.ndarray]
        Keys: ``f"{pseudotime_key_prefix}_{lineage}"``
        Values: float array (n_cells,) with NaN for cells not in that
        lineage's sub-operator.
    """
    n_cells = S.shape[0]
    results: Dict[str, np.ndarray] = {}

    for lineage_name, lin_mask in lineage_masks.items():
        n_committed = int(lin_mask.sum())
        if n_committed < min_lineage_cells:
            print(
                f"  [{lineage_name}] skipped — only {n_committed} committed cells "
                f"(< min_lineage_cells={min_lineage_cells})"
            )
            continue

        # Sub-matrix = stem cells + this lineage's committed cells
        sub_mask = stem_mask | lin_mask
        sub_idx = np.where(sub_mask)[0]
        n_sub = len(sub_idx)

        # Map global root index into the sub-matrix
        root_sub_hits = np.where(sub_idx == root)[0]
        if len(root_sub_hits) == 0:
            print(f"  [{lineage_name}] root cell {root} not in sub-mask — skipping")
            continue
        root_sub = int(root_sub_hits[0])

        print(
            f"  [{lineage_name}] sub-matrix: {n_sub} cells "
            f"({int(stem_mask.sum())} stem + {n_committed} committed)"
        )

        # Extract the sub-operator (already symmetric — row/col slicing preserves symmetry)
        S_sub = _extract_suboperator(S, sub_idx)

        # Spectral pseudotime on the sub-operator
        tau_sub, method_used, _ = _hypergraph_spectral_pseudotime(
            S_sub,
            root=root_sub,
            n_spectral_components=min(n_spectral_components, n_sub - 3),
            pseudotime_method="geodesic",
            embedding_scale="dpt",
            root_guided_components=True,
            min_root_corr=0.15,
            manifold_k=manifold_k,
            # time_numeric not available in lineage sub-operator context;
            # falls back to L2 root distance in the sub-embedding
        )

        # Map back to global array — NaN for cells outside this lineage's sub-operator
        tau_global = np.full(n_cells, np.nan)
        tau_global[sub_idx] = tau_sub

        key = f"{pseudotime_key_prefix}_{lineage_name}"
        results[key] = tau_global
        print(
            f"  [{lineage_name}] pseudotime range "
            f"[{np.nanmin(tau_global):.3f}, {np.nanmax(tau_global):.3f}] "
            f"via {method_used}"
        )

    return results


def compute_multiview_pseudotime(
    adata: sc.AnnData,
    P_fused: Union[csr_matrix, LinearOperator],
    view_weights: Dict[str, float],
    time_key: str = "timepoint",
    condition_key: Optional[str] = None,
    root_markers: Optional[List[str]] = None,
    root_percentile: float = 90.0,
    n_spectral_components: int = 10,
    spectral_time_scale: int = 4,
    pseudotime_method: str = "geodesic",
    manifold_k: int = 15,
    embedding_scale: str = "dpt",
    root_guided_components: bool = True,
    min_root_corr: float = 0.05,
    same_time_weight: float = 1.0,
    forward_time_weight: float = 1.2,
    backward_time_penalty: float = 0.3,
    pseudotime_key: str = "multiview_pseudotime",
    time_order: Optional[Union[Sequence[Any], Mapping[Any, Any]]] = None,
    validate_time_order: bool = True,
    drop_trivial_component: str = "always",
    trivial_cv_threshold: float = 0.05,
    # Deprecated aliases kept for one release cycle
    t_diffusion: Optional[int] = None,
    n_diffusion_components: Optional[int] = None,
) -> sc.AnnData:
    """
    Compute hypergraph spectral pseudotime from the fused symmetric operator.

    The fused operator ``P_fused`` (which should be the output of
    ``fuse_view_operators`` — a symmetric normalised hypergraph operator) is
    optionally biased by time-point information, then decomposed spectrally.
    Pseudotime is computed as either geodesic shortest-path distance or chord
    L2 distance from the root cell in the spectral embedding.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    P_fused : csr_matrix or LinearOperator
        Symmetric normalised fused operator (output of ``fuse_view_operators``).
    view_weights : Dict[str, float]
        View weights used to construct P_fused (stored in metadata).
    time_key : str
        Column in ``adata.obs`` with ordered time-point labels.
    condition_key : str, optional
        Column in ``adata.obs`` with condition labels (for summary reporting).
    root_markers : List[str], optional
        Gene markers to score root cells.  If None, uses earliest timepoint
        + lowest spectral degree.
    root_percentile : float
        Percentile threshold when using root_markers.
    n_spectral_components : int
        Number of non-trivial spectral components to retain (default 10).
    spectral_time_scale : int
        Eigenvalue scaling exponent lambda^t.  Use 1 for the raw spectral
        embedding; increase to emphasise the slowest-relaxing modes.
    pseudotime_method : {"geodesic", "chord"}
        Distance metric.  ``"geodesic"`` (default) uses shortest-path distance
        along a kNN graph on the spectral embedding.  ``"chord"`` uses L2.
    manifold_k : int
        k for the kNN manifold graph (geodesic method only, default 15).
    same_time_weight : float
        Multiplier for transitions between cells at the same time-point.
    forward_time_weight : float
        Multiplier for transitions to later time-points.
    backward_time_penalty : float
        Multiplier for transitions to earlier time-points.
    pseudotime_key : str
        Key to store pseudotime in ``adata.obs``.
    time_order : sequence or mapping, optional
        Explicit time ordering, earliest first (or ``label -> position``).
        Takes precedence over every inference, including an ordered
        ``Categorical``'s own categories.  Labels present in the data but
        absent here are excluded from the time axis.  Requires
        ``validate_time_order=True``.
    validate_time_order : bool
        ``True`` (default) resolves the ordering through
        :func:`pseudoembed.core.time_order.build_time_order`, which honours a
        declared ``Categorical`` order, numeric-parses labels such as ``"D0"``
        / ``"day 3"`` / ``"3h"``, excludes non-timepoint labels (an iPSC line,
        a NaN) from the axis instead of inventing a day for them, and raises
        rather than guessing when no ordering can be established.  ``False``
        restores the legacy lexicographic ``sorted(set(labels))``, which is a
        known bug (``"D10"`` sorts before ``"D2"``) and exists only so the fix
        can be A/B'd.  Cells excluded from the axis keep a pseudotime but
        contribute no time-direction evidence and are never root candidates.
    drop_trivial_component : {"always", "detect", "never"}
        How to handle the leading spectral component.  ``"always"`` (default)
        drops it unconditionally; ``"detect"`` drops it only if it tests as
        near-constant; ``"never"`` keeps everything.  The measured CV is
        printed under every mode, so the assumption is visible rather than
        asserted.

        The leading eigenvector of this operator is ``Dv^{1/2} 1``, which is
        constant only after the ``Dv^{-1/2}`` back-transform this module does
        not apply — so on real data it is genuinely non-constant (CV 0.236 on
        Schiebinger 2019, 0.175 on Richard 2018) and the legacy "always index
        0 is constant" comment was false.  It is nonetheless **degree, not
        biology**: rho(v0, sqrt(degree)) = -0.837 against rho(v0, day) = +0.076
        on Schiebinger.  ``"detect"`` therefore loses a fixed-root A/B on both
        datasets (rho_pooled 0.491 -> 0.325 and 0.901 -> 0.710; within-day
        degree leakage -0.088 -> +0.247 and -0.024 -> +0.387), which is why the
        default keeps dropping it.  See
        experiments/test_time_order_and_trivial.py.
    trivial_cv_threshold : float
        Coefficient of variation below which the leading component counts as
        constant under ``drop_trivial_component="detect"``.
    t_diffusion : int, optional
        Deprecated alias for ``spectral_time_scale``.
    n_diffusion_components : int, optional
        Deprecated alias for ``n_spectral_components``.

    Returns
    -------
    adata : AnnData
        Modified in-place.  Pseudotime in ``adata.obs[pseudotime_key]``;
        metadata in ``adata.uns[f'{pseudotime_key}_info']``.
    """
    # Handle deprecated aliases
    if t_diffusion is not None:
        warnings.warn(
            "t_diffusion is deprecated; use spectral_time_scale instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        spectral_time_scale = t_diffusion
    if n_diffusion_components is not None:
        warnings.warn(
            "n_diffusion_components is deprecated; use n_spectral_components instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        n_spectral_components = n_diffusion_components

    _cross_scale = (forward_time_weight + backward_time_penalty) / 2.0
    if _cross_scale > 1.0:
        warnings.warn(
            f"(forward_time_weight + backward_time_penalty) / 2 = {_cross_scale:.3f} > 1.0. "
            "Eigenvalues may exceed 1.0, causing DPT weights λ/(1-λ) to overflow. "
            "Rescaling both weights proportionally so their average = 1.0.",
            UserWarning,
            stacklevel=2,
        )
        _total = forward_time_weight + backward_time_penalty
        forward_time_weight = forward_time_weight / _total
        backward_time_penalty = backward_time_penalty / _total

    if same_time_weight > 1.0:
        warnings.warn(
            f"same_time_weight={same_time_weight:.3f} > 1.0 — clamping to 1.0.",
            UserWarning,
            stacklevel=2,
        )
        same_time_weight = 1.0

    print("=" * 70)
    print("COMPUTING MULTI-VIEW HYPERGRAPH SPECTRAL PSEUDOTIME")
    print("=" * 70)

    # --- Build time ordering -------------------------------------------
    # Ordering is *validated*, not string-sorted.  The previous
    # ``sorted(set(labels))`` was lexicographic, so "D10" < "D2" and a
    # non-timepoint level (an iPSC line, a control) silently received a time
    # index and became eligible as the earliest cell.  See
    # :mod:`pseudoembed.core.time_order` for the full contract, including what
    # happens to cells whose label is not a timepoint (they get nan and are
    # never root candidates).
    if validate_time_order:
        from pseudoembed.core.time_order import build_time_order

        _ordering = build_time_order(
            adata.obs[time_key], time_order=time_order, key_name=time_key
        )
        time_numeric = _ordering.time_numeric
        unique_times = _ordering.order
        _earliest_idx = _ordering.earliest_index
        _earliest_label = _ordering.earliest_label
        _time_order_source = _ordering.source
        _time_excluded = _ordering.excluded
    else:
        # Legacy path, kept only so the fix can be A/B'd against it.  This is
        # the bug; do not use it for published numbers.
        if time_order is not None:
            raise ValueError(
                "An explicit `time_order` cannot be honoured with "
                "validate_time_order=False (the legacy path string-sorts). "
                "Set validate_time_order=True."
            )
        time_labels = adata.obs[time_key].values
        unique_times = sorted(set(time_labels))
        _legacy_order = {t: i for i, t in enumerate(unique_times)}
        time_numeric = np.array([_legacy_order[t] for t in time_labels], dtype=float)
        _earliest_idx = min(_legacy_order.values())
        _earliest_label = unique_times[0]
        _time_order_source = "legacy lexicographic sort (KNOWN BUG)"
        _time_excluded = []
        print(f"  Time points (legacy string sort): {unique_times}")
    print(
        f"  Transition weights: same={same_time_weight}, "
        f"forward={forward_time_weight}, backward={backward_time_penalty}"
    )

    # --- Temporal bias on the symmetric operator -----------------------
    print("Applying temporal bias...")
    S_biased = _apply_temporal_bias_operator(
        P_fused,
        time_numeric,
        same_weight=same_time_weight,
        forward_weight=forward_time_weight,
        backward_penalty=backward_time_penalty,
    )

    # --- Identify root cell --------------------------------------------
    print("Identifying root cell...")
    operator_mass = _operator_row_mass(S_biased, adata.n_obs)

    if root_markers is not None:
        sc.tl.score_genes(adata, root_markers, score_name="_root_score_mv")
        root_score = adata.obs["_root_score_mv"].values
        threshold = np.percentile(root_score, root_percentile)
        root_candidates = np.where(root_score >= threshold)[0]
        if len(root_candidates) == 0:
            warnings.warn(
                "No root cells found with markers. Falling back to earliest timepoint.",
                UserWarning,
            )
            root_candidates = np.where(time_numeric == _earliest_idx)[0]
            root_score = None  # no valid marker scores; will use fallback below
        print(f"  Found {len(root_candidates)} root candidates via marker genes")
        # Among marker-positive candidates, pick the most peripheral cell in the
        # hypergraph (lowest operator degree = fewest shared regulatory connections).
        # score_genes is used only to *filter* — to confirm cells genuinely express
        # the progenitor programme.  Within that confirmed set, degree identifies the
        # true trajectory entry point: an uncommitted cell that has not yet entered a
        # well-defined TF/pathway programme appears in few hyperedges (low degree),
        # whereas the highest-scoring cell sits at the expression peak of the
        # progenitor cluster (high degree) — exactly the wrong root.
        root = _lowest_degree_root(
            root_candidates, operator_mass, tiebreak_score=root_score
        )
        adata.obs.drop("_root_score_mv", axis=1, inplace=True)
    else:
        # ``time_numeric`` is nan for cells whose label is not a timepoint, and
        # nan == _earliest_idx is False, so those cells are never root
        # candidates.  That is deliberate: an established line is a held-out
        # endpoint, not day 0.
        root_candidates = np.where(time_numeric == _earliest_idx)[0]
        if len(root_candidates) == 0:
            raise ValueError(
                f"No cells carry the earliest timepoint {_earliest_label!r}."
            )
        print(
            f"  Using earliest timepoint ({_earliest_label}), "
            f"lowest-degree cell of {len(root_candidates)} candidates"
        )
        # When no markers are given, prefer the most peripheral cell in the
        # earliest timepoint (smallest row-nnz in P_fused) rather than the hub,
        # so that pseudotime radiates outward from the true progenitor edge.
        root = _lowest_degree_root(root_candidates, operator_mass)

    print(f"  Root cell index: {root}")
    print(
        f"  Root cell type : {adata.obs['CellTypes'].iloc[root] if 'CellTypes' in adata.obs else 'n/a'}"
    )

    # --- Hypergraph spectral pseudotime --------------------------------
    print(
        f"Computing hypergraph spectral pseudotime "
        f"(n_spectral_components={n_spectral_components}, "
        f"spectral_time_scale={spectral_time_scale}, "
        f"method={pseudotime_method!r})..."
    )
    # Build condition_numeric if a condition key is available
    _condition_numeric: Optional[np.ndarray] = None
    if condition_key is not None and condition_key in adata.obs:
        from pandas import Categorical

        _cond_cat = Categorical(adata.obs[condition_key])
        _condition_numeric = np.array(_cond_cat.codes, dtype=float)

    _axis_labels: List[str] = []
    tau, method_used, spectral_emb = _hypergraph_spectral_pseudotime(
        S_biased,
        root=root,
        n_spectral_components=n_spectral_components,
        spectral_time_scale=spectral_time_scale,
        pseudotime_method=pseudotime_method,
        manifold_k=manifold_k,
        embedding_scale=embedding_scale,
        root_guided_components=root_guided_components,
        min_root_corr=min_root_corr,
        time_numeric=time_numeric,
        drop_trivial_component=drop_trivial_component,
        trivial_cv_threshold=trivial_cv_threshold,
        condition_numeric=_condition_numeric,
        axis_labels_out=_axis_labels,
    )

    adata.obs[pseudotime_key] = tau
    adata.obsm[f"{pseudotime_key}_spectral"] = spectral_emb
    # Column labels for the embedding, so downstream attribution can select the
    # perturbation axes without re-deriving the classification.
    adata.uns[f"{pseudotime_key}_axis_labels"] = list(_axis_labels)
    print(
        f"  Spectral embedding stored in adata.obsm['{pseudotime_key}_spectral'] "
        f"(shape {spectral_emb.shape})"
    )

    # --- Metadata ------------------------------------------------------
    adata.uns[f"{pseudotime_key}_info"] = {
        "method": "hypergraph_spectral_pseudotime",
        "pseudotime_method": method_used,
        "manifold_k": manifold_k,
        "embedding_scale": embedding_scale,
        "root_guided_components": root_guided_components,
        "min_root_corr": min_root_corr,
        "view_weights": dict(view_weights),
        "time_key": time_key,
        "root_cell": root,
        "root_markers": root_markers,
        "n_spectral_components": n_spectral_components,
        "spectral_time_scale": spectral_time_scale,
        "same_time_weight": same_time_weight,
        "forward_time_weight": forward_time_weight,
        "backward_time_penalty": backward_time_penalty,
        "time_points": unique_times,
    }

    # --- Summary statistics --------------------------------------------
    print(f"✓ Hypergraph spectral pseudotime computed:")
    print(f"  Range: [{tau.min():.3f}, {tau.max():.3f}]")
    print(f"  Mean: {tau.mean():.3f}, Std: {tau.std():.3f}")

    if condition_key is not None:
        for cond in adata.obs[condition_key].unique():
            mask = adata.obs[condition_key] == cond
            subset = tau[mask]
            print(f"  {cond}: mean={subset.mean():.3f}, std={subset.std():.3f}")

    return adata


# ============================================================================
# STEP 6: Sensitivity Analysis
# ============================================================================


def sensitivity_analysis(
    adata: sc.AnnData,
    P_TF: csr_matrix,
    P_pathway: csr_matrix,
    n_steps: int = 11,
    reference_key: str = "dpt_pseudotime",
    time_key: str = "timepoint",
    condition_key: Optional[str] = None,
    root_markers: Optional[List[str]] = None,
    root_percentile: float = 90.0,
    n_spectral_components: int = 10,
    spectral_time_scale: int = 4,
    pseudotime_key: str = "multiview_pseudotime",
    fusion_method: Literal["operator", "edge"] = "operator",
    tf_hyperedges: Optional[List[np.ndarray]] = None,
    tf_hyperedge_weights: Optional[np.ndarray] = None,
    pw_hyperedges: Optional[List[np.ndarray]] = None,
    pw_hyperedge_weights: Optional[np.ndarray] = None,
    plot: bool = False,
    ax: Optional[plt.Axes] = None,
) -> pd.DataFrame:
    """
    Sweep TF view weight ``alpha`` from 0 to 1 and measure Spearman correlation
    of the resulting hypergraph spectral pseudotime against a reference ordering.

    Parameters
    ----------
    adata : AnnData
        Annotated data.
    P_TF : csr_matrix
        Symmetric normalised TF view operator.
    P_pathway : csr_matrix
        Symmetric normalised pathway view operator.
    n_steps : int
        Number of evenly-spaced alpha values in [0, 1] (default 11).
    reference_key : str
        Column in ``adata.obs`` to use as Spearman reference.  If absent,
        the pseudotime at alpha=0.5 is used as reference.
    time_key : str
        Passed through to ``compute_multiview_pseudotime``.
    condition_key : str, optional
        Passed through to ``compute_multiview_pseudotime``.
    root_markers : List[str], optional
        Passed through to ``compute_multiview_pseudotime`` so every alpha step
        uses the same root selection strategy as the main run.
    root_percentile : float
        Passed through to ``compute_multiview_pseudotime``.
    n_spectral_components : int
        Passed through to ``compute_multiview_pseudotime``.
    spectral_time_scale : int
        Passed through to ``compute_multiview_pseudotime``.
    pseudotime_key : str
        Temporary key used internally (overwritten at each alpha step).
    fusion_method : {"operator", "edge"}
        Fusion strategy passed through to ``fuse_view_operators`` at every
        alpha step.  Use ``"edge"`` together with the four hyperedge
        arguments to sweep edge-level fusion.
    tf_hyperedges : List[np.ndarray], optional
        Required when ``fusion_method="edge"``.
    tf_hyperedge_weights : np.ndarray, optional
        Required when ``fusion_method="edge"``.
    pw_hyperedges : List[np.ndarray], optional
        Required when ``fusion_method="edge"``.
    pw_hyperedge_weights : np.ndarray, optional
        Required when ``fusion_method="edge"``.
    plot : bool
        Whether to produce an alpha vs. Spearman-ρ line plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.  A new figure is created if None.

    Returns
    -------
    results : pd.DataFrame
        Columns: ``alpha``, ``spearman_r``, ``spearman_p``,
        ``mean_pseudotime``, ``std_pseudotime``.
    """
    print("=" * 70)
    print("SENSITIVITY ANALYSIS — sweeping TF/pathway view weights")
    print(f"  fusion_method = {fusion_method!r}")
    print("=" * 70)

    if n_steps == 0:
        return pd.DataFrame(
            columns=[
                "alpha",
                "spearman_r",
                "spearman_p",
                "mean_pseudotime",
                "std_pseudotime",
            ]
        )

    alpha_values = np.linspace(0.0, 1.0, n_steps)
    _tmp_key = f"_sa_{pseudotime_key}_tmp"

    # Helper: build the fused operator for a given (alpha, beta) pair,
    # respecting the chosen fusion_method.
    def _fuse(alpha: float, beta: float) -> csr_matrix:
        vw = {"tf": alpha, "pathway": beta}
        ops = {"tf": P_TF, "pathway": P_pathway}
        if fusion_method == "edge":
            return fuse_view_operators(
                ops,
                vw,
                fusion_method="edge",
                view_hyperedges={"tf": tf_hyperedges, "pathway": pw_hyperedges},
                view_hyperedge_weights={
                    "tf": tf_hyperedge_weights,
                    "pathway": pw_hyperedge_weights,
                },
                n_cells=adata.n_obs,
            )
        return fuse_view_operators(ops, vw, fusion_method="operator")

    # Pre-compute pseudotime at alpha=0.5 in case reference_key is missing
    reference_pt: Optional[np.ndarray] = None
    if reference_key not in adata.obs:
        warnings.warn(
            f"'{reference_key}' not found in adata.obs. "
            "Will use pseudotime at alpha=0.5 as reference.",
            UserWarning,
        )
        P_mid = _fuse(0.5, 0.5)
        adata = compute_multiview_pseudotime(
            adata,
            P_fused=P_mid,
            view_weights={"tf": 0.5, "pathway": 0.5},
            time_key=time_key,
            condition_key=condition_key,
            root_markers=root_markers,
            root_percentile=root_percentile,
            n_spectral_components=n_spectral_components,
            spectral_time_scale=spectral_time_scale,
            pseudotime_key=_tmp_key,
        )
        reference_pt = adata.obs[_tmp_key].values.copy()
        adata.obs.drop(_tmp_key, axis=1, inplace=True, errors="ignore")
    else:
        reference_pt = adata.obs[reference_key].values.copy()

    # Stash any spectral embedding that exists from a prior real run so the
    # sweep does not clobber it.  It is restored after the loop.
    _stash_key = f"{pseudotime_key}_spectral"
    _stashed_spectral = adata.obsm.pop(_stash_key, None)

    records = []
    for alpha in alpha_values:
        beta = 1.0 - alpha
        print(f"  alpha={alpha:.2f}, beta={beta:.2f}", end=" ... ")

        P_fused = _fuse(alpha, beta)

        adata = compute_multiview_pseudotime(
            adata,
            P_fused=P_fused,
            view_weights={"tf": alpha, "pathway": beta},
            time_key=time_key,
            condition_key=condition_key,
            root_markers=root_markers,
            root_percentile=root_percentile,
            n_spectral_components=n_spectral_components,
            spectral_time_scale=spectral_time_scale,
            pseudotime_key=_tmp_key,
        )

        pt = adata.obs[_tmp_key].values
        rho, pval = spearmanr(pt, reference_pt)

        records.append(
            {
                "alpha": float(alpha),
                "spearman_r": float(rho),
                "spearman_p": float(pval),
                "mean_pseudotime": float(pt.mean()),
                "std_pseudotime": float(pt.std()),
            }
        )
        print(f"ρ={rho:.3f}")

    # Clean up the temporary sweep keys and restore the original spectral embedding
    adata.obs.drop(_tmp_key, axis=1, inplace=True, errors="ignore")
    adata.obsm.pop(f"{_tmp_key}_spectral", None)
    if _stashed_spectral is not None:
        adata.obsm[_stash_key] = _stashed_spectral

    results = pd.DataFrame(records)

    if plot:
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            results["alpha"], results["spearman_r"], marker="o", lw=2, color="#3b82d4"
        )
        ax.axhline(0.9, ls="--", lw=1, color="#7c5cd8", alpha=0.7, label="ρ = 0.9")
        ax.set_xlabel("TF view weight (α)")
        ax.set_ylabel("Spearman ρ with reference")
        ax.set_title("Multi-view pseudotime sensitivity analysis")
        ax.legend()
        ax.grid(True, alpha=0.3)

    return results


# ============================================================================
# Pseudotime Comparison Plot
# ============================================================================


def plot_pseudotime_comparison(
    adata: sc.AnnData,
    keys: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    color_by: Optional[str] = None,
    color_palette: Optional[str] = None,
    umap_key: str = "X_umap",
    figsize_umap: Tuple[int, int] = (5, 4),
    figsize_scatter: Tuple[int, int] = (4, 4),
    reference_key: Optional[str] = None,
    save: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side comparison of multiple pseudotime orderings.

    Produces a figure with two rows:

    * **Top row** – one UMAP panel per pseudotime, coloured by pseudotime value.
    * **Bottom row** – scatter plots of each pseudotime against the reference
      (defaults to the last key in *keys*), annotated with Pearson r and
      Spearman ρ.

    Parameters
    ----------
    adata : AnnData
        Must contain a 2-D embedding in ``adata.obsm[umap_key]`` and all
        pseudotime columns in ``adata.obs``.
    keys : list of str, optional
        ``adata.obs`` columns to compare, in order.  Defaults to
        ``['dpt_pseudotime', 'hypergraph_pseudotime', 'multiview_pseudotime']``
        — only columns that are actually present are kept.
    labels : list of str, optional
        Human-readable label for each key (same length as *keys*).
        Defaults to the column name with underscores replaced by spaces.
    color_by : str, optional
        ``adata.obs`` column used to colour the UMAP scatter (e.g. ``'celltype'``
        or ``'stage'``).  When given, an extra UMAP panel is prepended showing
        the cell-type / stage colouring so the reader has a reference.
    color_palette : str, optional
        Matplotlib / seaborn palette name used when *color_by* is categorical.
    umap_key : str
        Key in ``adata.obsm`` for the 2-D embedding (default ``'X_umap'``).
    figsize_umap : tuple
        Width × height in inches for each UMAP panel.
    figsize_scatter : tuple
        Width × height in inches for each scatter panel.
    reference_key : str, optional
        Which pseudotime to treat as the x-axis reference in scatter panels.
        Defaults to the *last* key in *keys*.
    save : str, optional
        File path to save the figure (e.g. ``'comparison.png'``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.colors as mcolors
    from scipy.stats import pearsonr
    from scipy.stats import spearmanr as _spearmanr

    # ------------------------------------------------------------------
    # Resolve keys
    # ------------------------------------------------------------------
    _default_keys = ["dpt_pseudotime", "hypergraph_pseudotime", "multiview_pseudotime"]
    if keys is None:
        keys = [k for k in _default_keys if k in adata.obs.columns]
    if not keys:
        raise ValueError(
            "No pseudotime columns found. Pass keys= explicitly or run the "
            "pseudotime methods first."
        )
    if labels is None:
        labels = [k.replace("_", " ") for k in keys]
    if len(labels) != len(keys):
        raise ValueError("labels must have the same length as keys.")
    if reference_key is None:
        reference_key = keys[-1]
    if reference_key not in adata.obs.columns:
        raise ValueError(f"reference_key '{reference_key}' not found in adata.obs.")

    ref_label = labels[keys.index(reference_key)]
    n_pt = len(keys)

    # ------------------------------------------------------------------
    # UMAP coordinates
    # ------------------------------------------------------------------
    if umap_key not in adata.obsm:
        raise ValueError(f"Embedding '{umap_key}' not found in adata.obsm.")
    umap = adata.obsm[umap_key]
    ux, uy = umap[:, 0], umap[:, 1]

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    # Top row: optional color_by panel + one panel per pseudotime
    # Bottom row: scatter of each pseudotime vs reference
    n_top_extra = 1 if color_by is not None else 0
    n_top = n_pt + n_top_extra
    n_bottom = n_pt  # scatter for every key including the reference itself

    fw_u, fh_u = figsize_umap
    fw_s, fh_s = figsize_scatter

    total_w = max(n_top * fw_u, n_bottom * fw_s)
    total_h = fh_u + fh_s + 0.6  # 0.6 for row gap

    fig = plt.figure(figsize=(total_w, total_h))

    # Build gridspec with two rows
    import matplotlib.gridspec as gridspec

    gs_top = gridspec.GridSpec(
        1,
        n_top,
        figure=fig,
        left=0.04,
        right=0.98,
        top=0.96,
        bottom=0.55,
        wspace=0.3,
    )
    gs_bot = gridspec.GridSpec(
        1,
        n_bottom,
        figure=fig,
        left=0.04,
        right=0.98,
        top=0.48,
        bottom=0.06,
        wspace=0.35,
    )

    # ------------------------------------------------------------------
    # Helper: categorical colour mapper
    # ------------------------------------------------------------------
    def _cat_colors(series, palette=None):
        cats = (
            series.cat.categories if hasattr(series, "cat") else sorted(series.unique())
        )
        if palette is None:
            cmap = plt.get_cmap("tab20", len(cats))
            col_map = {c: mcolors.to_hex(cmap(i)) for i, c in enumerate(cats)}
        else:
            import matplotlib.cm as cm

            cmap = plt.get_cmap(palette, len(cats))
            col_map = {c: mcolors.to_hex(cmap(i)) for i, c in enumerate(cats)}
        colors = [col_map[v] for v in series]
        return colors, col_map

    # ------------------------------------------------------------------
    # Top row — UMAP panels
    # ------------------------------------------------------------------
    col_offset = 0
    if color_by is not None and color_by in adata.obs.columns:
        ax = fig.add_subplot(gs_top[0, 0])
        series = adata.obs[color_by]
        colors, col_map = _cat_colors(series, color_palette)
        ax.scatter(ux, uy, c=colors, s=6, linewidths=0)
        ax.set_title(color_by.replace("_", " "), fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        # Compact legend
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=v,
                markersize=5,
                label=k,
            )
            for k, v in col_map.items()
        ]
        ax.legend(
            handles=handles,
            fontsize=5,
            loc="lower left",
            framealpha=0.6,
            ncol=max(1, len(col_map) // 10),
        )
        col_offset = 1

    pt_cmap = "viridis"
    for i, (key, lab) in enumerate(zip(keys, labels)):
        ax = fig.add_subplot(gs_top[0, i + col_offset])
        pt_vals = adata.obs[key].values.astype(float)
        sc_im = ax.scatter(
            ux, uy, c=pt_vals, cmap=pt_cmap, s=6, linewidths=0, vmin=0, vmax=1
        )
        ax.set_title(lab, fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(sc_im, ax=ax, fraction=0.04, pad=0.02)

    # ------------------------------------------------------------------
    # Bottom row — scatter vs reference
    # ------------------------------------------------------------------
    ref_pt = adata.obs[reference_key].values.astype(float)

    for i, (key, lab) in enumerate(zip(keys, labels)):
        ax = fig.add_subplot(gs_bot[0, i])
        pt_vals = adata.obs[key].values.astype(float)

        # Drop NaNs
        mask = ~(np.isnan(pt_vals) | np.isnan(ref_pt))
        x, y = ref_pt[mask], pt_vals[mask]

        if color_by is not None and color_by in adata.obs.columns:
            colors_all, _ = _cat_colors(adata.obs[color_by], color_palette)
            c = [colors_all[j] for j in np.where(mask)[0]]
        else:
            c = x  # colour by reference pseudotime

        ax.scatter(
            x,
            y,
            c=c,
            s=6,
            alpha=0.6,
            linewidths=0,
            cmap=None if (color_by and color_by in adata.obs.columns) else pt_cmap,
        )

        # Diagonal reference line
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, alpha=0.7)

        # Stats
        if key == reference_key:
            ax.set_title(f"{lab}\n(reference)", fontsize=8)
        else:
            r, _ = pearsonr(x, y)
            rho, _ = _spearmanr(x, y)
            ax.set_title(f"{lab}\nr={r:.3f}  ρ={rho:.3f}", fontsize=8)

        ax.set_xlabel(ref_label, fontsize=7)
        ax.set_ylabel(lab, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Pseudotime Method Comparison", fontsize=11, y=0.995)

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    return fig


# ============================================================================
# STEP 6b: Hypergraph Layout Plots
# ============================================================================


def plot_hypergraph_knn(
    adata: sc.AnnData,
    spectral_key: str = "multiview_pseudotime_spectral",
    pseudotime_key: str = "multiview_pseudotime",
    condition_key: Optional[str] = None,
    manifold_k: int = 15,
    max_cells: int = 3000,
    point_size: float = 8.0,
    save: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the kNN graph built on the hypergraph spectral embedding.

    Nodes are cells; edges are the nearest-neighbour connections used by the
    geodesic pseudotime step.  Node colour encodes pseudotime (viridis).
    When *condition_key* is given, node border colour encodes condition.

    Parameters
    ----------
    adata : AnnData
        Must contain ``adata.obsm[spectral_key]`` and ``adata.obs[pseudotime_key]``.
    spectral_key : str
        Key in ``adata.obsm`` for the spectral embedding (default
        ``'multiview_pseudotime_spectral'``).
    pseudotime_key : str
        Key in ``adata.obs`` for pseudotime values (default
        ``'multiview_pseudotime'``).
    condition_key : str, optional
        ``adata.obs`` column used to colour node edge rings by condition.
    manifold_k : int
        Number of nearest neighbours (must match the value used when computing
        pseudotime, default 15).
    max_cells : int
        Subsample to at most this many cells before building the graph so that
        edge drawing stays tractable (default 3 000).
    point_size : float
        Scatter point size (default 8).
    save : str, optional
        File path to save the figure.

    Returns
    -------
    fig : matplotlib.Figure
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from sklearn.neighbors import kneighbors_graph

    if spectral_key not in adata.obsm:
        raise ValueError(f"'{spectral_key}' not found in adata.obsm.")
    if pseudotime_key not in adata.obs.columns:
        raise ValueError(f"'{pseudotime_key}' not found in adata.obs.")

    emb_full = np.asarray(adata.obsm[spectral_key])
    pt_full = adata.obs[pseudotime_key].values.astype(float)
    n = emb_full.shape[0]

    # Subsample for rendering tractability
    rng = np.random.default_rng(0)
    if n > max_cells:
        idx = rng.choice(n, size=max_cells, replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(n)

    emb = emb_full[idx]
    pt = pt_full[idx]

    # Project to 2-D via PCA on the spectral embedding for plotting
    from sklearn.decomposition import PCA as _PCA

    n_components = min(2, emb.shape[1])
    if n_components < 2:
        proj = np.column_stack([emb[:, 0], np.zeros(len(emb))])
    else:
        proj = _PCA(n_components=2).fit_transform(emb)

    # Build kNN graph on the *full-dimensional* spectral embedding
    k_eff = min(manifold_k, len(idx) - 1)
    g = kneighbors_graph(
        emb, n_neighbors=k_eff, mode="connectivity", include_self=False
    )
    g = g + g.T  # symmetrise
    g = (g > 0).astype(float)
    cx = g.tocoo()

    # Condition colours for node borders
    cond_colors = None
    cond_map: dict = {}
    if condition_key is not None and condition_key in adata.obs.columns:
        series = adata.obs[condition_key].iloc[idx].astype(str)
        cats = sorted(series.unique())
        palette = plt.get_cmap("tab10", len(cats))
        cond_map = {c: mcolors.to_hex(palette(i)) for i, c in enumerate(cats)}
        cond_colors = [cond_map[v] for v in series]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Draw edges first (thin, light grey)
    rows, cols = cx.row, cx.col
    keep = rows < cols  # draw each edge once
    rows, cols = rows[keep], cols[keep]
    for r, c in zip(rows, cols):
        ax.plot(
            [proj[r, 0], proj[c, 0]],
            [proj[r, 1], proj[c, 1]],
            lw=0.3,
            color="#cccccc",
            alpha=0.4,
            zorder=1,
        )

    # Draw nodes
    sc_im = ax.scatter(
        proj[:, 0],
        proj[:, 1],
        c=pt,
        cmap="viridis",
        s=point_size,
        vmin=0,
        vmax=1,
        linewidths=0.6 if cond_colors is not None else 0,
        edgecolors=cond_colors if cond_colors is not None else "none",
        zorder=2,
    )
    cbar = fig.colorbar(sc_im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Pseudotime", fontsize=9)

    if cond_map:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor=v,
                markeredgewidth=1.2,
                markersize=6,
                label=k,
            )
            for k, v in cond_map.items()
        ]
        ax.legend(
            handles=handles,
            title=condition_key.replace("_", " "),
            fontsize=7,
            title_fontsize=7,
            loc="lower left",
        )

    n_shown = len(idx)
    title = f"Hypergraph kNN graph  (k={k_eff}, n={n_shown:,} cells)"
    if n > max_cells:
        title += f"  [subsample of {n:,}]"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Spectral PC 1", fontsize=8)
    ax.set_ylabel("Spectral PC 2", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    return fig


def plot_hypergraph_force_directed(
    adata: sc.AnnData,
    P_fused: Optional[csr_matrix] = None,
    spectral_key: str = "multiview_pseudotime_spectral",
    pseudotime_key: str = "multiview_pseudotime",
    condition_key: Optional[str] = None,
    knn_k: int = 10,
    max_cells: int = 1500,
    n_iter: int = 50,
    point_size: float = 10.0,
    seed: int = 42,
    save: Optional[str] = None,
) -> plt.Figure:
    """
    Force-directed (Fruchterman–Reingold) layout of the hypergraph.

    The layout is computed on a sparse kNN graph derived from the fused
    hypergraph operator (or, if *P_fused* is not supplied, from the spectral
    embedding stored in *adata*).  The Fruchterman–Reingold spring algorithm
    is run via NetworkX, then cells are plotted coloured by pseudotime.

    Parameters
    ----------
    adata : AnnData
        Must contain ``adata.obs[pseudotime_key]``.
    P_fused : csr_matrix, optional
        Fused hypergraph diffusion operator (shape n_cells × n_cells).
        When provided the top-*knn_k* neighbours per row are used as edges.
        When absent a kNN graph is built on ``adata.obsm[spectral_key]``
        instead.
    spectral_key : str
        Key in ``adata.obsm`` for the spectral embedding used as a fallback
        (default ``'multiview_pseudotime_spectral'``).
    pseudotime_key : str
        Key in ``adata.obs`` for pseudotime values (default
        ``'multiview_pseudotime'``).
    condition_key : str, optional
        ``adata.obs`` column used to colour node edge rings by condition.
    knn_k : int
        Edges per node in the sparse graph sent to the layout solver
        (default 10).
    max_cells : int
        Subsample to at most this many cells (default 1 500).
    n_iter : int
        Fruchterman–Reingold iterations (default 50).
    point_size : float
        Scatter point size (default 10).
    seed : int
        Random seed for reproducible layouts (default 42).
    save : str, optional
        File path to save the figure.

    Returns
    -------
    fig : matplotlib.Figure
    """
    import matplotlib.colors as mcolors

    if pseudotime_key not in adata.obs.columns:
        raise ValueError(f"'{pseudotime_key}' not found in adata.obs.")

    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "networkx is required for the force-directed layout. "
            "Install with: pip install networkx"
        ) from exc

    n = adata.n_obs
    rng = np.random.default_rng(seed)

    # Subsample
    if n > max_cells:
        idx = np.sort(rng.choice(n, size=max_cells, replace=False))
    else:
        idx = np.arange(n)

    n_sub = len(idx)
    pt = adata.obs[pseudotime_key].values.astype(float)[idx]

    # Build the sparse graph to lay out
    if P_fused is not None:
        # Extract the submatrix for the sampled cells and retain top-k entries.
        # P_fused may be a LinearOperator (not subscriptable), so materialise the
        # submatrix explicitly when needed.
        if hasattr(P_fused, "toarray"):
            sub_dense = P_fused.toarray()[np.ix_(idx, idx)]
        elif hasattr(P_fused, "__getitem__"):
            sub_dense = np.asarray(P_fused[np.ix_(idx, idx)].todense())
        else:
            # LinearOperator path: extract columns for idx then slice rows
            n_full = P_fused.shape[1]
            E = np.zeros((n_full, n_sub), dtype=float)
            E[idx, np.arange(n_sub)] = 1.0
            sub_dense = (P_fused @ E)[idx, :]
        # Keep only the knn_k largest entries per row → sparse graph
        k_eff = min(knn_k, n_sub - 1)
        rows_list, cols_list, data_list = [], [], []
        for r in range(n_sub):
            row_data = np.asarray(sub_dense[r]).ravel()
            top_k = np.argpartition(row_data, -k_eff)[-k_eff:]
            for c in top_k:
                if r != c and row_data[c] > 0:
                    rows_list.append(r)
                    cols_list.append(c)
                    data_list.append(float(row_data[c]))
        if len(rows_list) == 0:
            # Fallback: fully-connected subgraph with unit weights
            rows_list = [r for r in range(n_sub) for c in range(n_sub) if r != c]
            cols_list = [c for r in range(n_sub) for c in range(n_sub) if r != c]
            data_list = [1.0] * len(rows_list)
        adj = csr_matrix((data_list, (rows_list, cols_list)), shape=(n_sub, n_sub))
    else:
        # Fallback: kNN graph on spectral embedding
        if spectral_key not in adata.obsm:
            raise ValueError(
                f"'{spectral_key}' not found in adata.obsm and no P_fused supplied."
            )
        from sklearn.neighbors import kneighbors_graph

        emb_sub = np.asarray(adata.obsm[spectral_key])[idx]
        k_eff = min(knn_k, n_sub - 1)
        adj = kneighbors_graph(
            emb_sub, n_neighbors=k_eff, mode="distance", include_self=False
        )

    # Symmetrise
    adj = (adj + adj.T) / 2

    # Build NetworkX graph from sparse adjacency
    cx = adj.tocoo()
    G = nx.Graph()
    G.add_nodes_from(range(n_sub))
    for r, c, w in zip(cx.row, cx.col, cx.data):
        if r < c:
            G.add_edge(int(r), int(c), weight=float(w))

    print(
        f"  Running Fruchterman-Reingold layout ({n_sub:,} nodes, "
        f"{G.number_of_edges():,} edges, {n_iter} iterations)..."
    )
    pos = nx.spring_layout(G, iterations=n_iter, seed=seed, weight="weight")
    xy = np.array([pos[i] for i in range(n_sub)])

    # Condition colours for node borders
    cond_colors = None
    cond_map: dict = {}
    if condition_key is not None and condition_key in adata.obs.columns:
        series = adata.obs[condition_key].iloc[idx].astype(str)
        cats = sorted(series.unique())
        palette = plt.get_cmap("tab10", len(cats))
        cond_map = {c: mcolors.to_hex(palette(i)) for i, c in enumerate(cats)}
        cond_colors = [cond_map[v] for v in series]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Draw edges
    for u, v in G.edges():
        ax.plot(
            [xy[u, 0], xy[v, 0]],
            [xy[u, 1], xy[v, 1]],
            lw=0.3,
            color="#cccccc",
            alpha=0.35,
            zorder=1,
        )

    # Draw nodes
    sc_im = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=pt,
        cmap="viridis",
        s=point_size,
        vmin=0,
        vmax=1,
        linewidths=0.7 if cond_colors is not None else 0,
        edgecolors=cond_colors if cond_colors is not None else "none",
        zorder=2,
    )
    cbar = fig.colorbar(sc_im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Pseudotime", fontsize=9)

    if cond_map:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor=v,
                markeredgewidth=1.2,
                markersize=6,
                label=k,
            )
            for k, v in cond_map.items()
        ]
        ax.legend(
            handles=handles,
            title=condition_key.replace("_", " "),
            fontsize=7,
            title_fontsize=7,
            loc="lower left",
        )

    n_shown = len(idx)
    title = f"Hypergraph force-directed layout  (n={n_shown:,} cells)"
    if n > max_cells:
        title += f"  [subsample of {n:,}]"
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    return fig


# ============================================================================
# STEP 7: Top-Level Workflow
# ============================================================================


def run_multiview_hypergraph_analysis(
    adata: sc.AnnData,
    condition_key: str = "condition",
    control_label: str = "control",
    drug_label: str = "drug",
    time_key: str = "timepoint",
    organism: str = "human",
    root_markers: Optional[List[str]] = None,
    top_n_tfs: Optional[int] = None,
    z_score_threshold: float = 1.5,
    component_k: int = 15,
    n_states: int = 5,
    n_spectral_components: int = 10,
    spectral_time_scale: int = 4,
    pseudotime_method: Literal["geodesic", "chord"] = "geodesic",
    manifold_k: int = 15,
    embedding_scale: str = "dpt",
    min_root_corr: float = 0.05,
    view_weights: Optional[Dict[str, float]] = None,
    fusion_method: Literal["operator", "edge"] = "operator",
    n_sensitivity_steps: int = 11,
    pca_key: str = "X_pca",
    n_pca_dims: int = 30,
    use_gene_view: bool = False,
    gene_view_threshold: float = 1.0,
    gene_view_soft: bool = False,
    gene_view_n_genes: int = 1200,
    gene_view_genes: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Tuple[sc.AnnData, pd.DataFrame, Dict]:
    """
    Complete multi-view hypergraph drug perturbation analysis workflow.

    Orchestrates all steps in order:

    1. Compute TF activities (CollecTRI / decoupler ULM)
    2. Build TF hyperedges
    3. Compute pathway activities (PROGENy / decoupler ULM)
    4. Build pathway hyperedges
    4b. Build PCA kNN-star hyperedges (when ``X_pca`` is present) and gene
        hyperedges (when ``use_gene_view=True``)
    5. Build per-view symmetric normalised operators, one per active view
    6. Fuse operators (``fusion_method`` selects strategy):

       - ``"operator"`` (default): convex combination
         ``S_fused = alpha * S_TF + (1-alpha) * S_pathway``
       - ``"edge"``: pool hyperedges first, then build a single operator
         from the merged incidence matrix

    7. Compute hypergraph spectral pseudotime
    8. Analyze drug effects (reuses existing ``analyze_drug_effects``)
    9. Discover cell states (reuses existing ``discover_states``)
    10. Sensitivity analysis (sweep alpha, compute Spearman ρ)

    Parameters
    ----------
    adata : AnnData
        Annotated data with gene expression.
    condition_key : str
        Column in adata.obs with condition labels.
    control_label : str
        Label for control condition.
    drug_label : str
        Label for drug-treated condition.
    time_key : str
        Column in adata.obs with ordered time-point labels.
    organism : str
        Organism for TF / pathway networks ('human' or 'mouse').
    root_markers : List[str], optional
        Marker genes for progenitor root cells.
    top_n_tfs : int, optional
        Number of most variable TFs to use (``None`` = all TFs, default).
    z_score_threshold : float
        Z-score threshold for TF activity edges (default 2.0).
    component_k : int
        k for intra-TF connected-component splitting (default 10).
    n_states : int
        Number of cell states to discover.
    n_spectral_components : int
        Number of non-trivial spectral components for pseudotime (default 10).
    spectral_time_scale : int
        Eigenvalue scaling exponent for the spectral embedding (default 4).
    pseudotime_method : {"geodesic", "chord"}
        Final pseudotime metric.  ``"geodesic"`` (default) computes shortest-path
        distance along a kNN graph built on the spectral embedding — respects the
        curved regulatory manifold.  ``"chord"`` uses L2 distance directly in the
        spectral embedding (faster, less accurate for curved trajectories).
    manifold_k : int
        k for the kNN graph used in geodesic pseudotime (default 15).
    view_weights : Dict[str, float], optional
        Weights for each view, e.g. ``{'tf': 0.5, 'pathway': 0.5}``.
        Defaults to equal weighting.
    fusion_method : {"operator", "edge"}
        Fusion strategy.  ``"operator"`` (default) performs late fusion via
        a convex combination of per-view operators.  ``"edge"`` performs
        early fusion by pooling hyperedges before building the operator.
    n_sensitivity_steps : int
        Number of alpha values to sweep in sensitivity analysis.
    use_gene_view : bool
        Add the gene-as-hyperedge view (:func:`build_gene_hypergraph`) as a
        fourth view.  Default False — opt-in, so existing callers keep the
        TF/pathway(/PCA) composition they were validated on.  When enabled
        without an explicit *view_weights*, all active views are weighted
        equally.
    gene_view_threshold : float
        Z-score hinge threshold for gene edges (default 1.0, the sweep winner).
    gene_view_soft : bool
        Graded gene-edge weights.  Default False: binary beat soft at every
        threshold measured.
    gene_view_n_genes : int
        Size of the gene-edge pool (default 1200).
    gene_view_genes : List[str], optional
        Explicit gene list for the gene view, bypassing automatic selection.
    output_dir : str, optional
        Directory to write results, figures, and h5ad.

    Returns
    -------
    adata : AnnData
        Modified in-place with all results.
    drug_effects : pd.DataFrame
        Summary of drug effects on pseudotime progression.
    objects : Dict
        Intermediate objects: view operators, hyperedges per view,
        sensitivity analysis DataFrame, and ``fusion_method`` used.
    """
    has_pca = pca_key in adata.obsm

    if view_weights is None:
        active = ["tf", "pathway"]
        if has_pca:
            active.append("pca")
        if use_gene_view:
            active.append("gene")
        view_weights = {v: 1 / len(active) for v in active}

    print("\n" + "=" * 70)
    print("MULTI-VIEW HYPERGRAPH DRUG PERTURBATION ANALYSIS")
    print("=" * 70)
    print(f"Dataset: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"Conditions: {list(adata.obs[condition_key].unique())}")
    print(f"Timepoints: {list(adata.obs[time_key].unique())}")
    print(
        f"Views: TF + Pathway"
        + (" + PCA" if has_pca else " (PCA not found — 2-view mode)")
        + (" + Gene" if use_gene_view else "")
    )
    print(f"View weights: {view_weights}")
    print(f"Fusion method: {fusion_method!r}")
    print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # Step 1 & 2: TF view
    # ------------------------------------------------------------------
    adata = compute_tf_activities(adata, organism=organism)
    tf_hyperedges, tf_hyperedge_types, tf_hyperedge_weights = build_tf_hypergraph(
        adata,
        top_n_tfs=top_n_tfs,
        z_score_threshold=z_score_threshold,
        component_k=component_k,
    )

    # ------------------------------------------------------------------
    # Step 3 & 4: Pathway view
    # ------------------------------------------------------------------
    adata = compute_pathway_activities(adata, organism=organism)
    (
        pw_hyperedges,
        pw_hyperedge_types,
        pw_hyperedge_weights,
        pw_hyperedge_meta,
    ) = build_pathway_hypergraph(adata, return_meta=True)

    # ------------------------------------------------------------------
    # Step 3b: PCA view (optional — used when X_pca is present)
    # ------------------------------------------------------------------
    pca_hyperedges: List[np.ndarray] = []
    pca_hyperedge_types: List[str] = []
    pca_hyperedge_weights: np.ndarray = np.array([], dtype=float)
    pca_hyperedge_meta: List[Dict] = []
    if has_pca:
        (
            pca_hyperedges,
            pca_hyperedge_types,
            pca_hyperedge_weights,
            pca_hyperedge_meta,
        ) = build_pca_hypergraph(
            adata, pca_key=pca_key, n_pca_dims=n_pca_dims, return_meta=True
        )
    else:
        warnings.warn(
            f"'{pca_key}' not found in adata.obsm — skipping PCA view. "
            "Run sc.pp.pca() first to enable the three-view hypergraph.",
            UserWarning,
        )

    # ------------------------------------------------------------------
    # Step 3c: Gene-as-hyperedge view (opt-in)
    # ------------------------------------------------------------------
    gene_hyperedges: List[np.ndarray] = []
    gene_hyperedge_types: List[str] = []
    gene_hyperedge_weights: np.ndarray = np.array([], dtype=float)
    gene_hyperedge_meta: List[Dict] = []
    if use_gene_view:
        (
            gene_hyperedges,
            gene_hyperedge_types,
            gene_hyperedge_weights,
            gene_hyperedge_meta,
        ) = build_gene_hypergraph(
            adata,
            genes=gene_view_genes,
            n_genes=gene_view_n_genes,
            threshold=gene_view_threshold,
            soft=gene_view_soft,
            return_meta=True,
        )

    # ------------------------------------------------------------------
    # Step 5: Per-view diffusion operators
    # ------------------------------------------------------------------
    print("Building per-view diffusion operators...")
    P_TF = build_view_operator(tf_hyperedges, tf_hyperedge_weights, adata.n_obs)
    P_pathway = build_view_operator(pw_hyperedges, pw_hyperedge_weights, adata.n_obs)
    operators = {"tf": P_TF, "pathway": P_pathway}
    view_hyperedges_dict = {"tf": tf_hyperedges, "pathway": pw_hyperedges}
    view_hyperedge_weights_dict = {
        "tf": tf_hyperedge_weights,
        "pathway": pw_hyperedge_weights,
    }

    if has_pca and len(pca_hyperedges) > 0:
        P_pca = build_view_operator(pca_hyperedges, pca_hyperedge_weights, adata.n_obs)
        operators["pca"] = P_pca
        view_hyperedges_dict["pca"] = pca_hyperedges
        view_hyperedge_weights_dict["pca"] = pca_hyperedge_weights
        print("✓ P_TF, P_pathway, P_pca constructed")
    else:
        # Strip pca key from view_weights if no edges were built
        view_weights = {k: v for k, v in view_weights.items() if k != "pca"}
        print("✓ P_TF and P_pathway constructed")

    if use_gene_view and len(gene_hyperedges) > 0:
        P_gene = build_view_operator(
            gene_hyperedges, gene_hyperedge_weights, adata.n_obs
        )
        operators["gene"] = P_gene
        view_hyperedges_dict["gene"] = gene_hyperedges
        view_hyperedge_weights_dict["gene"] = gene_hyperedge_weights
        print("✓ P_gene constructed")
    else:
        # Same contract as the PCA view: a requested-but-empty view must not
        # leave a key in view_weights, or fuse_view_operators raises on the
        # operators/weights key mismatch.
        if use_gene_view:
            warnings.warn(
                "Gene view requested but no gene hyperedges survived the size "
                "filters — continuing without it.",
                UserWarning,
            )
        view_weights = {k: v for k, v in view_weights.items() if k != "gene"}

    # ------------------------------------------------------------------
    # Step 6: Fuse
    # ------------------------------------------------------------------
    print(f"Fusing view operators (method={fusion_method!r})...")
    P_fused = fuse_view_operators(
        operators,
        view_weights,
        fusion_method=fusion_method,
        view_hyperedges=view_hyperedges_dict,
        view_hyperedge_weights=view_hyperedge_weights_dict,
        n_cells=adata.n_obs,
    )
    print(f"✓ P_fused constructed (shape {P_fused.shape})")

    # ------------------------------------------------------------------
    # Step 7: Hypergraph spectral pseudotime
    # ------------------------------------------------------------------
    adata = compute_multiview_pseudotime(
        adata,
        P_fused=P_fused,
        view_weights=view_weights,
        time_key=time_key,
        condition_key=condition_key,
        root_markers=root_markers,
        n_spectral_components=n_spectral_components,
        spectral_time_scale=spectral_time_scale,
        pseudotime_method=pseudotime_method,
        manifold_k=manifold_k,
        embedding_scale=embedding_scale,
        min_root_corr=min_root_corr,
    )

    # ------------------------------------------------------------------
    # Step 8: Drug effect analysis (view-agnostic, reused from existing)
    # ------------------------------------------------------------------
    drug_effects = analyze_drug_effects(
        adata,
        condition_key=condition_key,
        control_label=control_label,
        drug_label=drug_label,
        time_key=time_key,
        pseudotime_key="multiview_pseudotime",
    )

    # ------------------------------------------------------------------
    # Step 9a: Soft incidence matrix (X_hyperedge)
    # H_soft[cell, edge] = edge_weight  for member cells, 0 elsewhere.
    # Encodes each cell as a weighted vector of hyperedge memberships —
    # binary if memberships are hard, soft if edge weights vary.
    # ------------------------------------------------------------------
    all_hyperedges = tf_hyperedges + pw_hyperedges + list(pca_hyperedges)
    all_hyperedge_weights = np.concatenate(
        [tf_hyperedge_weights, pw_hyperedge_weights]
        + ([pca_hyperedge_weights] if len(pca_hyperedge_weights) > 0 else [])
    )
    _H_bin = build_incidence_matrix(all_hyperedges, adata.n_obs)  # (n_cells, n_edges)
    # Scale each column by its edge weight → soft membership
    _H_soft = _H_bin.multiply(all_hyperedge_weights[np.newaxis, :]).tocsr()
    # L2-normalise each row so cells with many memberships aren't artificially inflated
    _row_norms = np.asarray(np.sqrt(_H_soft.multiply(_H_soft).sum(axis=1))).ravel()
    _row_norms = np.where(_row_norms == 0, 1.0, _row_norms)
    _H_soft = _H_soft.multiply(1.0 / _row_norms[:, np.newaxis]).tocsr()
    adata.obsm["X_hyperedge"] = np.asarray(_H_soft.todense())
    print(
        f"  Soft incidence matrix stored in adata.obsm['X_hyperedge'] "
        f"(shape {adata.obsm['X_hyperedge'].shape})"
    )

    # ------------------------------------------------------------------
    # Step 9b: State discovery — combine hyperedges from both views
    # ------------------------------------------------------------------
    adata = discover_states(
        adata,
        all_hyperedges,
        all_hyperedge_weights,
        n_states=n_states,
        state_key="multiview_state",
    )

    # ------------------------------------------------------------------
    # Step 10: Sensitivity analysis
    # ------------------------------------------------------------------
    print("Running sensitivity analysis...")
    sensitivity_df = sensitivity_analysis(
        adata,
        P_TF=P_TF,
        P_pathway=P_pathway,
        n_steps=n_sensitivity_steps,
        time_key=time_key,
        condition_key=condition_key,
        root_markers=root_markers,
        n_spectral_components=n_spectral_components,
        spectral_time_scale=spectral_time_scale,
        fusion_method=fusion_method,
        tf_hyperedges=tf_hyperedges,
        tf_hyperedge_weights=tf_hyperedge_weights,
        pw_hyperedges=pw_hyperedges,
        pw_hyperedge_weights=pw_hyperedge_weights,
        plot=output_dir is not None,
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        drug_effects.to_csv(out / "multiview_drug_effects.csv", index=False)
        print(f"  ✓ Saved: multiview_drug_effects.csv")

        sensitivity_df.to_csv(out / "multiview_sensitivity_analysis.csv", index=False)
        print(f"  ✓ Saved: multiview_sensitivity_analysis.csv")

        # Plot from the already-computed sensitivity_df — no rerun needed
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            sensitivity_df["alpha"],
            sensitivity_df["spearman_r"],
            marker="o",
            lw=2,
            color="#3b82d4",
        )
        ax.axhline(0.9, ls="--", lw=1, color="#7c5cd8", alpha=0.7, label="ρ = 0.9")
        ax.set_xlabel("TF view weight (α)")
        ax.set_ylabel("Spearman ρ with reference")
        ax.set_title("Multi-view pseudotime sensitivity analysis")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "multiview_sensitivity_plot.png", dpi=150)
        plt.close(fig)
        print(f"  ✓ Saved: multiview_sensitivity_plot.png")

        adata.write_h5ad(out / "adata_multiview_hypergraph.h5ad")
        print(f"  ✓ Saved: adata_multiview_hypergraph.h5ad")

    # ------------------------------------------------------------------
    # Assemble return objects
    # ------------------------------------------------------------------
    objects: Dict = {
        # TF view
        "tf_hyperedges": tf_hyperedges,
        "tf_hyperedge_types": tf_hyperedge_types,
        "tf_hyperedge_weights": tf_hyperedge_weights,
        "P_TF": P_TF,
        # Pathway view
        "pathway_hyperedges": pw_hyperedges,
        "pathway_hyperedge_types": pw_hyperedge_types,
        "pathway_hyperedge_weights": pw_hyperedge_weights,
        "pathway_hyperedge_meta": pw_hyperedge_meta,
        "P_pathway": P_pathway,
        # PCA view (may be empty if X_pca was absent)
        "pca_hyperedges": pca_hyperedges,
        "pca_hyperedge_types": pca_hyperedge_types,
        "pca_hyperedge_weights": pca_hyperedge_weights,
        "pca_hyperedge_meta": pca_hyperedge_meta,
        "P_pca": operators.get("pca"),
        # Gene view (may be empty if use_gene_view was False)
        "gene_hyperedges": gene_hyperedges,
        "gene_hyperedge_types": gene_hyperedge_types,
        "gene_hyperedge_weights": gene_hyperedge_weights,
        "gene_hyperedge_meta": gene_hyperedge_meta,
        "P_gene": operators.get("gene"),
        # Fused
        "P_fused": P_fused,
        "view_weights": view_weights,
        "fusion_method": fusion_method,
        # Sensitivity
        "sensitivity_df": sensitivity_df,
        # Counts
        "n_cells": adata.n_obs,
    }

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("Results stored in adata:")
    print("  - adata.obsm['tf_activities']                    : TF activity matrix")
    print(
        "  - adata.obsm['pathway_activities']               : Pathway activity matrix"
    )
    print(
        "  - adata.obs['multiview_pseudotime']              : Fused pseudotime [0, 1]"
    )
    print(
        "  - adata.obsm['multiview_pseudotime_spectral']    : Hypergraph spectral embedding"
    )
    print(
        "  - adata.obsm['X_hyperedge']                      : Soft incidence matrix (cell × edge)"
    )
    print("  - adata.obs['multiview_state']                   : Discovered cell states")
    print("=" * 70 + "\n")

    return adata, drug_effects, objects
