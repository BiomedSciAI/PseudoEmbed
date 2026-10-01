"""
TF-Based Hypergraph Drug Perturbation Analysis

This module implements a complete workflow for analyzing drug perturbation effects
on single-cell differentiation trajectories using transcription factor (TF) activity
to define hypergraph connectivity.

Key Features:
- TF activity inference using decoupler ULM
- TF-based hypergraph construction
- Time-aware hypergraph diffusion pseudotime
- Post-hoc condition comparison
- State discovery via hyperedge clustering or SBM
- Drug effect quantification and visualization

Author: Computational Biology Team
Date: 2026-06-22
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse
from scipy.sparse import csr_matrix, issparse
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

try:
    import decoupler as dc

    DECOUPLER_AVAILABLE = True
except ImportError:
    DECOUPLER_AVAILABLE = False
    warnings.warn(
        "decoupler not available. Install with: pip install decoupler", ImportWarning
    )


# ============================================================================
# STEP 1: TF Activity Inference
# ============================================================================


def compute_tf_activities(
    adata: sc.AnnData,
    organism: str = "human",
    net: Optional[pd.DataFrame] = None,
    method: str = "ulm",
    use_raw: bool = False,
    min_n_targets: int = 5,
    verbose: bool = True,
) -> sc.AnnData:
    """
    Compute transcription factor activities using decoupler.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with gene expression
    organism : str
        Organism for CollecTRI network ('human' or 'mouse')
    net : pd.DataFrame, optional
        Custom TF-target network with columns: source, target, weight
        If None, uses CollecTRI
    method : str
        Decoupler method ('ulm', 'mlm', 'wsum')
        ULM (Univariate Linear Model) is recommended for speed and robustness
    use_raw : bool
        Whether to use adata.raw.X for expression
    min_n_targets : int
        Minimum number of target genes required for a TF
    verbose : bool
        Print progress. Set False to silence when called in a loop
        (e.g. across benchmark datasets).

    Returns
    -------
    adata : AnnData
        Modified in place with TF activities in adata.obsm['tf_activities']
        and metadata in adata.uns['tf_activities_info']
    """
    if not DECOUPLER_AVAILABLE:
        raise ImportError("decoupler is required. Install with: pip install decoupler")

    _log = print if verbose else (lambda *a, **k: None)

    _log("=" * 70)
    _log("COMPUTING TF ACTIVITIES")
    _log("=" * 70)

    # Load TF-target network
    if net is None:
        _log(f"Loading CollecTRI network for {organism}...")
        # Use decoupler's omnipath module to get CollecTRI
        net = dc.op.collectri(organism=organism)
    else:
        _log(f"Using custom TF-target network...")

    # Filter network by genes present in data
    genes_in_data = set(adata.var_names)
    net_filtered = net[net["target"].isin(genes_in_data)].copy()

    # Filter TFs with too few targets
    tf_target_counts = net_filtered.groupby("source").size()
    valid_tfs = tf_target_counts[tf_target_counts >= min_n_targets].index
    net_filtered = net_filtered[net_filtered["source"].isin(valid_tfs)]

    _log(f"  Network: {len(net_filtered)} interactions")
    _log(f"  TFs: {net_filtered['source'].nunique()}")
    _log(f"  Target genes: {net_filtered['target'].nunique()}")

    # Run decoupler
    _log(f"Running {method.upper()} method...")

    if method == "ulm":
        dc.mt.ulm(data=adata, net=net_filtered, verbose=verbose)
        acts_key = "score_ulm"
    elif method == "mlm":
        dc.mt.mlm(data=adata, net=net_filtered, verbose=verbose)
        acts_key = "score_mlm"
    elif method == "wsum":
        dc.mt.wsum(data=adata, net=net_filtered, verbose=verbose)
        acts_key = "score_wsum"
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: ulm, mlm, wsum")

    # Extract activities from obsm
    # decoupler stores results directly in adata.obsm[acts_key]
    if acts_key not in adata.obsm:
        raise ValueError(
            f"Expected key '{acts_key}' not found in adata.obsm after running {method}"
        )

    acts_data = adata.obsm[acts_key]

    # Handle both DataFrame and ndarray cases
    import pandas as pd

    if isinstance(acts_data, pd.DataFrame):
        # If it's a DataFrame, extract values and column names
        adata.obsm["tf_activities"] = acts_data.values
        tf_names = list(acts_data.columns)
        n_tfs = acts_data.shape[1]
    else:
        # If it's already a numpy array
        adata.obsm["tf_activities"] = acts_data
        n_tfs = acts_data.shape[1]
        tf_names = [f"TF_{i}" for i in range(n_tfs)]

    # Store metadata

    adata.uns["tf_activities_info"] = {
        "method": method,
        "organism": organism,
        "n_tfs": n_tfs,
        "tf_names": tf_names,
        "network_size": len(net_filtered),
        "min_n_targets": min_n_targets,
    }

    _log(f"✓ Computed activities for {n_tfs} TFs")
    _log(f"  Stored in: adata.obsm['tf_activities']")

    return adata


# ============================================================================
# STEP 2: TF-Based Hypergraph Construction
# ============================================================================


def build_tf_hypergraph(
    adata: sc.AnnData,
    min_tf_corr: float = 0.6,
    top_n_tfs: Optional[int] = None,
    min_cells_per_edge: int = 10,
    max_cells_per_edge: int = 0,
    z_score_threshold: float = 1.5,
    component_k: int = 15,
    min_component_size: int = 10,
    include_coactivity_edges: bool = False,
    include_pair_edges: bool = False,
    include_hinge_edges: bool = False,
    hinge_z_threshold: float = 1.30,
    hinge_max_frac: float = 0.5,
    tf_activity_key: str = "tf_activities",
) -> Tuple[List[np.ndarray], List[str], np.ndarray]:
    """
    Build a biologically meaningful TF-programme hypergraph.

    Design principles
    -----------------
    The fundamental unit of a hyperedge here is a **TF regulatory programme**:
    a group of cells that share coherent, high-level activity for a given TF
    (or TF pair).  This is distinct from a pairwise k-NN graph — it encodes
    genuinely multi-way co-regulatory relationships.

    One-edge-per-cell k-NN edges are intentionally omitted.  They produce a
    near-regular operator (flat spectrum, spectral gap ≈ 0) that destroys all
    trajectory information.

    Three types of hyperedges are built:

    1. **TF co-activity edges** (one per TF, optional)
       All cells whose z-scored TF activity exceeds *z_score_threshold*.
       These are the largest edges and encode which cells are broadly under
       control of each TF.  Size varies strongly across TFs (some are
       ubiquitous, some cell-type-specific), which is what creates an
       informative, non-regular operator.

    2. **TF programme edges** (coherent connected components)
       High-activity cells for each TF are embedded in TF-activity space and
       their *k*-NN graph is computed.  Each connected component of that
       sub-graph becomes a separate hyperedge.  This splits a TF's cells into
       coherent regulatory contexts (e.g. MYC-active stem cells vs MYC-active
       TA cells) rather than lumping them all together.
       Weighted by mean activity × intra-component coherence (mean pairwise
       cosine similarity).

    3. **TF hinge edges** (optional, ``include_hinge_edges=True``)
       One edge per TF per *tail*: cells with z-scored activity above
       ``+hinge_z_threshold`` and, separately, cells below
       ``-hinge_z_threshold``.  Binary weight, no component splitting.  This is
       the construction ``build_gene_hypergraph`` uses, applied to TF activities.

       Measured on Weinreb 2020 LARRY (8,105 cells, 406 TFs), fused with the gene
       view at weight 0.1 and scored as ``rho(tau, day) - rho(tau, undiff)``
       against a gene-only baseline of 1.261, with the root pinned to one
       identical cell across configs so no config can win on root luck:

       ==========================================  ==============  ============
       TF edge construction                        mean edge size  delta
       ==========================================  ==============  ============
       programme edges (default, hi tail + split)           600.8       +0.032
       hinge, both tails, ``hinge_z_threshold``=1.30        600.5       +0.045
       hinge, hi tail only, matched size                   1433.9       +0.029
       programme edges with weights binarised               600.8       +0.030
       ==========================================  ==============  ============

       The comparison is size-matched deliberately — edge size is a confound in
       its own right, and ``hinge_z_threshold=1.30`` was chosen to reproduce the
       default construction's mean edge size, not to maximise the score.
       Binarising the default weights changes nothing (+0.032 -> +0.030), and
       dropping the low tail costs most of the gain, so the mechanism is the
       **low tail**: cells with unusually *low* activity for a TF are as
       informative as cells with unusually high activity, and the default
       one-sided threshold cannot represent them.

       Left off by default: it is one dataset, and the default construction's
       component splitting encodes context (a TF active in stem vs TA cells)
       that a single edge per tail deliberately discards.

    4. **TF-pair co-activity edges** (optional)
       For pairs of highly correlated TFs, cells that are simultaneously in the
       top quartile for *both* TFs form a hyperedge.  These capture combinatorial
       regulatory states that a single-TF analysis misses and are genuinely
       multi-way relationships that a pairwise graph cannot represent.

    Parameters
    ----------
    adata : AnnData
        Annotated data with TF activities in ``adata.obsm[tf_activity_key]``.
    min_tf_corr : float
        Minimum Pearson correlation between two TFs for them to be considered
        co-active (used for TF-pair edges).
    top_n_tfs : int, optional
        Restrict to the top *n* most variable TFs (``None`` = use all TFs,
        default).  Using all TFs at a strict z-score threshold produces
        ~1500–3500 edges on a 4000-cell dataset, which is necessary for a
        well-conditioned spectral operator.
    min_cells_per_edge : int
        Minimum cells in any hyperedge (smaller components are discarded).
    max_cells_per_edge : int
        Maximum cells per edge (randomly sub-sampled if exceeded).
        Default 0 (disabled) — the operator's ``D_e^{-1}`` term normalises
        every edge's contribution by its cardinality, so large edges do not
        dominate the spectrum.  Capping would silently exclude cells from
        their most important edges, corrupting node degrees and distorting
        the spectral embedding for biologically critical TFs (SOX10, OLIG2,
        NKX2-2) whose programmes are legitimately large.  Set to a positive
        integer only if memory is a hard constraint.
    z_score_threshold : float
        Z-score threshold for calling a cell "high activity" for a TF.
        Default 1.5 (~7% of cells per TF).  Since co-activity edges are
        disabled by default, this threshold only controls which cells enter
        the programme-edge connected-component step — a lower threshold
        gives more cells for better component discovery without creating the
        near-complete operator problem that broad co-activity edges cause.
    component_k : int
        k for the intra-TF kNN graph used to find connected components.
        Default 15 — higher k creates larger, better-connected components,
        reducing the chance of trivially small fragments.
    min_component_size : int
        Minimum cells in a connected component to form a hyperedge.
    include_coactivity_edges : bool
        Whether to add the broad TF co-activity edges (type 1).
        Default ``False`` — co-activity edges are too broad (all cells above
        a z-threshold) and make the operator near-complete.  Programme edges
        (type 2) already capture the same signal with better specificity.
        Only enable for debugging or if you have very few TFs.
    include_hinge_edges : bool
        Whether to add the both-tails hinge edges (type 3).  Default ``False``
        so existing callers are unaffected; see the design notes above for the
        size-matched LARRY comparison and the caveats.
    hinge_z_threshold : float
        |z| cutoff for hinge-edge membership.  Default 1.30, chosen on LARRY to
        match the default programme construction's mean edge size (600.5 vs
        600.8) so the two can be compared without an edge-size confound — not
        to maximise any score.  Larger values give smaller, sharper edges.
    hinge_max_frac : float
        Skip a hinge edge covering more than this fraction of cells.  Default
        0.5, matching ``build_gene_hypergraph``: a near-global edge contributes
        a near-complete clique and flattens the spectrum.
    tf_activity_key : str
        Key in ``adata.obsm`` for TF activities.

    Returns
    -------
    hyperedges : List[np.ndarray]
        List of hyperedges (each is an array of cell indices).
    hyperedge_types : List[str]
        Type label for each hyperedge.
    hyperedge_weights : np.ndarray
        Weight for each hyperedge.
    """
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import NearestNeighbors

    print("=" * 70)
    print("BUILDING TF-BASED HYPERGRAPH")
    print("=" * 70)

    if tf_activity_key not in adata.obsm:
        raise ValueError(
            f"TF activities not found in adata.obsm['{tf_activity_key}']. "
            "Run compute_tf_activities() first."
        )

    tf_acts = adata.obsm[tf_activity_key]
    n_cells = adata.n_obs
    n_tfs_total = tf_acts.shape[1]

    # --- Select most variable TFs -----------------------------------------
    if top_n_tfs is not None and top_n_tfs < n_tfs_total:
        print(f"Selecting top {top_n_tfs} most variable TFs...")
        tf_var = np.var(tf_acts, axis=0)
        top_tf_idx = np.argsort(tf_var)[-top_n_tfs:]
        tf_acts_subset = tf_acts[:, top_tf_idx]
        if "tf_activities_info" in adata.uns:
            tf_names = [
                adata.uns["tf_activities_info"]["tf_names"][i] for i in top_tf_idx
            ]
        else:
            tf_names = [f"TF_{i}" for i in top_tf_idx]
    else:
        tf_acts_subset = tf_acts
        top_tf_idx = np.arange(n_tfs_total)
        tf_names = (
            adata.uns["tf_activities_info"]["tf_names"]
            if "tf_activities_info" in adata.uns
            else [f"TF_{i}" for i in range(n_tfs_total)]
        )

    n_tfs = tf_acts_subset.shape[1]
    print(f"  Using {n_tfs} TFs for hypergraph construction")

    # Z-score each TF across cells so thresholds are on a common scale
    tf_mean = tf_acts_subset.mean(axis=0)
    tf_std = tf_acts_subset.std(axis=0)
    tf_std = np.where(tf_std == 0, 1.0, tf_std)  # avoid /0
    tf_z = (tf_acts_subset - tf_mean) / tf_std  # (n_cells, n_tfs)

    rng = np.random.default_rng(42)

    hyperedges: List[np.ndarray] = []
    hyperedge_types: List[str] = []
    hyperedge_weights: List[float] = []

    # -----------------------------------------------------------------------
    # Type 1: Broad TF co-activity edges
    # One hyperedge per TF = all cells with z-score > threshold.
    # Weight = activity strength × TF specificity (1 - fraction of cells above
    # threshold). A TF active in only 5% of cells gets higher weight than one
    # active in 40% — cell-type-restricted TFs carry more trajectory signal.
    # -----------------------------------------------------------------------
    n_coactivity = 0
    if include_coactivity_edges:
        print(f"Adding TF co-activity edges (z > {z_score_threshold})...")
        for tf_idx in range(n_tfs):
            tf_name = tf_names[tf_idx]
            high_mask = tf_z[:, tf_idx] > z_score_threshold
            cells = np.where(high_mask)[0]

            if len(cells) < min_cells_per_edge:
                continue
            if max_cells_per_edge > 0 and len(cells) > max_cells_per_edge:
                cells = rng.choice(cells, max_cells_per_edge, replace=False)

            # Weight = mean z-score × TF specificity
            # Specificity = 1 - (fraction of cells in this edge)
            # Ubiquitous TFs (large edges) get down-weighted; restricted TFs
            # (small edges, cell-type-specific) get up-weighted.
            mean_act = float(tf_z[cells, tf_idx].mean())
            specificity = 1.0 - len(cells) / n_cells
            weight = mean_act * specificity

            hyperedges.append(cells)
            hyperedge_types.append(f"tf_coactivity_{tf_name}")
            hyperedge_weights.append(max(weight, 1e-3))
            n_coactivity += 1

        print(f"  Added {n_coactivity} co-activity hyperedges")

    # -----------------------------------------------------------------------
    # Type 2: TF programme edges (connected components of high-activity cells)
    # High-activity cells for each TF are split into locally coherent groups
    # by running a kNN graph and finding connected components.
    # -----------------------------------------------------------------------
    n_programme = 0
    print(f"Adding TF programme edges (component_k={component_k})...")
    for tf_idx in range(n_tfs):
        tf_name = tf_names[tf_idx]
        high_mask = tf_z[:, tf_idx] > z_score_threshold
        cells = np.where(high_mask)[0]

        if len(cells) < min_component_size * 2:
            # Too few cells to find meaningful components
            continue

        # Build kNN graph among high-activity cells in TF-activity space
        k_eff = min(component_k, len(cells) - 1)
        nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine")
        nn.fit(tf_acts_subset[cells])
        adj = nn.kneighbors_graph(tf_acts_subset[cells], mode="connectivity")
        # Symmetrise
        adj = (adj + adj.T).astype(bool).astype(float)

        n_comp, labels = connected_components(adj, directed=False)

        for comp_id in range(n_comp):
            comp_mask = labels == comp_id
            comp_cells = cells[comp_mask]

            if len(comp_cells) < min_component_size:
                continue
            if max_cells_per_edge > 0 and len(comp_cells) > max_cells_per_edge:
                comp_cells = rng.choice(comp_cells, max_cells_per_edge, replace=False)

            # Weight = mean activity × intra-component cosine coherence
            profiles = tf_acts_subset[comp_cells]
            norms = np.linalg.norm(profiles, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            normed = profiles / norms
            # Mean pairwise cosine similarity (upper triangle only for speed)
            if len(comp_cells) > 1:
                cos_sim = float(np.triu(normed @ normed.T, k=1).sum())
                n_pairs = len(comp_cells) * (len(comp_cells) - 1) / 2
                coherence = cos_sim / n_pairs if n_pairs > 0 else 0.0
            else:
                coherence = 1.0

            # mean_act = float(tf_z[comp_cells, tf_idx].mean())
            mean_act = float((tf_z[comp_cells, tf_idx] - z_score_threshold).mean())
            weight = max(mean_act * max(coherence, 0.1), 1e-3)

            hyperedges.append(comp_cells)
            hyperedge_types.append(f"tf_programme_{tf_name}")
            hyperedge_weights.append(weight)
            n_programme += 1

    print(f"  Added {n_programme} TF programme hyperedges")

    # -----------------------------------------------------------------------
    # Type 3: TF hinge edges (optional) — one edge per TF per tail, binary.
    # The construction build_gene_hypergraph uses, applied to TF activity.
    # See the docstring for the size-matched LARRY comparison that motivates it.
    # -----------------------------------------------------------------------
    if include_hinge_edges:
        n_hinge = 0
        max_cells = int(hinge_max_frac * n_cells)
        print(
            f"Adding TF hinge edges (|z| > {hinge_z_threshold}, both tails, "
            f"max {max_cells} cells/edge)..."
        )
        for tf_idx in range(n_tfs):
            tf_name = tf_names[tf_idx]
            for sign, tail in ((1.0, "hi"), (-1.0, "lo")):
                cells = np.where(sign * tf_z[:, tf_idx] > hinge_z_threshold)[0]
                # Guards match build_gene_hypergraph: an edge smaller than
                # min_cells_per_edge is noise, and one covering most of the data
                # adds a near-complete clique that flattens the spectrum.
                if len(cells) < min_cells_per_edge or len(cells) > max_cells:
                    continue
                hyperedges.append(cells)
                hyperedge_types.append(f"tf_hinge_{tf_name}_{tail}")
                # Binary: measured inert on LARRY when applied to the default
                # construction, so the gain is the edge definition, not weighting.
                hyperedge_weights.append(1.0)
                n_hinge += 1
        print(f"  Added {n_hinge} TF hinge hyperedges")

    # -----------------------------------------------------------------------
    # Type 3: TF-pair co-activity edges
    # Cells simultaneously high for two correlated TFs form a hyperedge.
    # These capture combinatorial regulatory states — genuinely multi-way
    # relationships a pairwise graph cannot represent.
    # -----------------------------------------------------------------------
    n_pair = 0
    if include_pair_edges:
        print(f"Adding TF-pair co-activity edges (min_corr={min_tf_corr})...")
        # TF-TF correlation matrix
        tf_corr = np.corrcoef(tf_acts_subset.T)  # (n_tfs, n_tfs)
        for i in range(n_tfs):
            for j in range(i + 1, n_tfs):
                if tf_corr[i, j] < min_tf_corr:
                    continue

                # Cells high for both TFs simultaneously
                cells = np.where(
                    (tf_z[:, i] > z_score_threshold) & (tf_z[:, j] > z_score_threshold)
                )[0]

                if len(cells) < min_cells_per_edge:
                    continue
                if max_cells_per_edge > 0 and len(cells) > max_cells_per_edge:
                    cells = rng.choice(cells, max_cells_per_edge, replace=False)

                # Weight = geometric mean of the two activity levels × correlation
                mean_i = float(tf_z[cells, i].mean())
                mean_j = float(tf_z[cells, j].mean())
                weight = max(
                    np.sqrt(max(mean_i, 0) * max(mean_j, 0)) * tf_corr[i, j], 1e-3
                )

                hyperedges.append(cells)
                hyperedge_types.append(f"tf_pair_{tf_names[i]}_{tf_names[j]}")
                hyperedge_weights.append(weight)
                n_pair += 1

        print(f"  Added {n_pair} TF-pair co-activity hyperedges")
    else:
        print(f"  Skipping TF-pair edges (include_pair_edges=False)")

    hyperedge_weights_arr = np.array(hyperedge_weights, dtype=float)

    # Normalise the CONTINUOUS weights to [., 1] so co-activity, programme and
    # TF-pair edges (whose raw scales differ) are comparable.  Hinge edges are
    # excluded: they are binary by construction, and including them made their
    # weight depend on the largest programme weight in the run -- measured on
    # LARRY, hinge edges came back at 0.987 rather than 1.0 and sat at 8.65x the
    # mean programme weight, so a mixed call handed the hinge block an arbitrary
    # dominance that the standalone hinge view (all weights exactly 1.0, the
    # configuration the +0.045 delta was measured in) does not have.
    is_hinge = np.array(
        [t.startswith("tf_hinge_") for t in hyperedge_types], dtype=bool
    )
    cont = ~is_hinge
    if cont.any() and hyperedge_weights_arr[cont].max() > 0:
        hyperedge_weights_arr[cont] = (
            hyperedge_weights_arr[cont] / hyperedge_weights_arr[cont].max()
        )

    print(f"\n✓ Built hypergraph:")
    print(f"  Total hyperedges: {len(hyperedges)}")
    if hyperedges:
        sizes = [len(e) for e in hyperedges]
        print(
            f"  Edge sizes: min={min(sizes)}, median={int(np.median(sizes))}, max={max(sizes)}"
        )
        print(f"  Mean cells per edge: {np.mean(sizes):.1f}")
    print(f"  Co-activity: {n_coactivity}, Programme: {n_programme}, TF-pair: {n_pair}")

    return hyperedges, hyperedge_types, hyperedge_weights_arr


# ============================================================================
# STEP 3: Hypergraph Utilities
# ============================================================================


def build_incidence_matrix(hyperedges: List[np.ndarray], n_nodes: int) -> csr_matrix:
    """
    Build sparse incidence matrix H.

    H[v, e] = 1 if node v belongs to hyperedge e.

    Parameters
    ----------
    hyperedges : List[np.ndarray]
        List of hyperedges (each is array of node indices)
    n_nodes : int
        Total number of nodes

    Returns
    -------
    H : csr_matrix
        Sparse incidence matrix (n_nodes × n_edges)
    """
    rows = []
    cols = []

    for e_idx, edge in enumerate(hyperedges):
        for node in edge:
            rows.append(node)
            cols.append(e_idx)

    data = np.ones(len(rows), dtype=float)

    H = sparse.coo_matrix(
        (data, (rows, cols)), shape=(n_nodes, len(hyperedges))
    ).tocsr()

    return H


def compute_degrees(
    H: csr_matrix, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute node and hyperedge degrees.

    Parameters
    ----------
    H : csr_matrix
        Incidence matrix (n_nodes × n_edges)
    weights : np.ndarray
        Hyperedge weights (n_edges,)

    Returns
    -------
    node_degrees : np.ndarray
        Weighted degree of each node
    edge_degrees : np.ndarray
        Cardinality of each hyperedge
    """
    # Node degrees: sum of weights of hyperedges containing each node
    node_degrees = H @ weights

    # Edge degrees: number of nodes in each hyperedge
    edge_degrees = np.array(H.sum(axis=0)).flatten()

    return node_degrees, edge_degrees


def compute_diffusion_operator(
    H: csr_matrix,
    weights: np.ndarray,
    node_degrees: np.ndarray,
    edge_degrees: np.ndarray,
    eps: float = 1e-10,
) -> csr_matrix:
    """
    Compute hypergraph random-walk diffusion operator.

    P = Dv^{-1} H W De^{-1} H^T

    where:
    - Dv^{-1}: Inverse node degree matrix
    - W: Hyperedge weight matrix
    - De^{-1}: Inverse hyperedge degree matrix

    Parameters
    ----------
    H : csr_matrix
        Incidence matrix
    weights : np.ndarray
        Hyperedge weights
    node_degrees : np.ndarray
        Node degrees
    edge_degrees : np.ndarray
        Edge degrees
    eps : float
        Small constant to avoid division by zero

    Returns
    -------
    P : csr_matrix
        Row-stochastic diffusion operator
    """
    # Avoid division by zero
    node_degrees_safe = np.maximum(node_degrees, eps)
    edge_degrees_safe = np.maximum(edge_degrees, eps)

    # Compute P = Dv^{-1} H W De^{-1} H^T
    # Step by step for clarity and efficiency

    # H W
    HW = H.multiply(weights[np.newaxis, :])

    # H W De^{-1}
    HW_De_inv = HW.multiply(1.0 / edge_degrees_safe[np.newaxis, :])

    # H W De^{-1} H^T
    P = HW_De_inv @ H.T

    # Dv^{-1} (H W De^{-1} H^T)
    P = P.multiply(1.0 / node_degrees_safe[:, np.newaxis])

    return P.tocsr()


# ============================================================================
# STEP 4: Time-Aware Hypergraph Diffusion Pseudotime
# ============================================================================


def compute_time_aware_pseudotime(
    adata: sc.AnnData,
    hyperedges: List[np.ndarray],
    hyperedge_weights: np.ndarray,
    time_key: str = "timepoint",
    condition_key: Optional[str] = None,
    root_markers: Optional[List[str]] = None,
    root_percentile: float = 90,
    t_diffusion: int = 10,
    same_time_weight: float = 1.0,
    forward_time_weight: float = 1.2,
    backward_time_penalty: float = 0.3,
    pseudotime_key: str = "hypergraph_pseudotime",
    time_order: Optional[Union[Sequence[Any], Mapping[Any, Any]]] = None,
    validate_time_order: bool = True,
) -> sc.AnnData:
    """
    Compute time-aware hypergraph diffusion pseudotime.

    This function:
    1. Builds hypergraph diffusion operator
    2. Applies temporal constraints (penalize backward time transitions)
    3. Computes diffusion distance from root cells
    4. Normalizes to [0, 1] pseudotime

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    hyperedges : List[np.ndarray]
        List of hyperedges
    hyperedge_weights : np.ndarray
        Hyperedge weights
    time_key : str
        Column in adata.obs with time point labels
    condition_key : str, optional
        Column in adata.obs with condition labels (for reporting only)
    root_markers : List[str], optional
        Marker genes for root cells (e.g., progenitor markers)
        If None, uses earliest timepoint + highest degree cell
    root_percentile : float
        Percentile threshold for root marker score
    t_diffusion : int
        Number of diffusion steps
    same_time_weight : float
        Weight for same-time transitions
    forward_time_weight : float
        Weight for forward-time transitions
    backward_time_penalty : float
        Penalty for backward-time transitions
    pseudotime_key : str
        Key to store pseudotime in adata.obs

    Returns
    -------
    adata : AnnData
        Modified in place with pseudotime in adata.obs[pseudotime_key]
        and metadata in adata.uns[f'{pseudotime_key}_info']
    """
    print("=" * 70)
    print("COMPUTING TIME-AWARE HYPERGRAPH DIFFUSION PSEUDOTIME")
    print("=" * 70)

    n_cells = adata.n_obs

    # Build incidence matrix
    print("Building incidence matrix...")
    H = build_incidence_matrix(hyperedges, n_cells)
    print(f"  Shape: {H.shape}")
    print(f"  Sparsity: {H.nnz / (H.shape[0] * H.shape[1]):.2%}")

    # Compute degrees
    node_degrees, edge_degrees = compute_degrees(H, hyperedge_weights)
    print(f"  Mean node degree: {node_degrees.mean():.1f}")
    print(f"  Mean edge degree: {edge_degrees.mean():.1f}")

    # Compute standard diffusion operator
    print("Computing diffusion operator...")
    P = compute_diffusion_operator(H, hyperedge_weights, node_degrees, edge_degrees)

    # Apply time-aware modification
    print("Applying temporal constraints...")
    # Validated ordering — see pseudoembed.core.time_order.  The previous
    # ``sorted(set(labels))`` was a lexicographic sort ("D10" < "D2") that also
    # handed a time index to labels that are not timepoints at all.
    if validate_time_order:
        from pseudoembed.core.time_order import build_time_order

        _ordering = build_time_order(
            adata.obs[time_key], time_order=time_order, key_name=time_key
        )
        time_numeric = _ordering.time_numeric
        unique_times = _ordering.order
        earliest_time = _ordering.earliest_index
        earliest_label = _ordering.earliest_label
    else:
        if time_order is not None:
            raise ValueError(
                "An explicit `time_order` cannot be honoured with "
                "validate_time_order=False. Set validate_time_order=True."
            )
        time_labels = adata.obs[time_key].values
        unique_times = sorted(set(time_labels))
        _legacy_order = {t: i for i, t in enumerate(unique_times)}
        time_numeric = np.array([_legacy_order[t] for t in time_labels], dtype=float)
        earliest_time = min(_legacy_order.values())
        earliest_label = unique_times[0]
        print(f"  Time points (legacy string sort): {unique_times}")
    print(
        f"  Weights: same={same_time_weight}, forward={forward_time_weight}, backward={backward_time_penalty}"
    )

    # Create time penalty matrix
    P_dense = P.toarray()
    # dt[i, j] = t_j - t_i.  Cells excluded from the time axis carry nan, so
    # their dt is nan and falls through to ``same_time_weight``: no forward or
    # backward claim is made about a cell that has no position in time.  The
    # previous Python double loop put nan in the final ``else`` and called it
    # forward.
    dt = time_numeric[None, :] - time_numeric[:, None]
    T = np.where(
        dt < 0,
        backward_time_penalty,
        np.where(dt > 0, forward_time_weight, same_time_weight),
    ).astype(float)

    # Apply time weighting
    P_time = P_dense * T

    # Re-normalize rows to maintain stochasticity
    row_sums = P_time.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    P_time = P_time / row_sums

    # Diffuse
    print(f"Diffusing for {t_diffusion} steps...")
    Pt = np.eye(n_cells)
    for step in range(t_diffusion):
        Pt = Pt @ P_time
        if (step + 1) % 5 == 0:
            print(f"  Step {step + 1}/{t_diffusion}")

    # Identify root cells
    print("Identifying root cells...")
    if root_markers is not None:
        # Use marker genes
        sc.tl.score_genes(adata, root_markers, score_name="_root_score_temp")
        root_score = adata.obs["_root_score_temp"].values
        root_threshold = np.percentile(root_score, root_percentile)
        root_cells = np.where(root_score >= root_threshold)[0]

        if len(root_cells) == 0:
            warnings.warn("No root cells found with markers. Using earliest timepoint.")
            root_cells = np.where(time_numeric == earliest_time)[0]

        print(f"  Found {len(root_cells)} root cells using markers")
        print(f"  Root score threshold: {root_threshold:.3f}")

        # Clean up temporary column
        adata.obs.drop("_root_score_temp", axis=1, inplace=True)
    else:
        # Use earliest timepoint + highest degree.  nan == earliest_time is
        # False, so cells excluded from the time axis are never root candidates.
        early_cells = np.where(time_numeric == earliest_time)[0]
        if len(early_cells) == 0:
            raise ValueError(
                f"No cells carry the earliest timepoint {earliest_label!r}."
            )
        root_cells = [early_cells[np.argmax(node_degrees[early_cells])]]
        print(f"  Using earliest timepoint ({earliest_label}) + highest degree cell")

    # Select single root (highest degree among root cells)
    root = root_cells[np.argmax(node_degrees[root_cells])]
    print(f"  Root cell index: {root}")

    # Compute pseudotime as diffusion distance from root
    print("Computing pseudotime...")
    tau = np.linalg.norm(Pt - Pt[root], axis=1)

    # Normalize to [0, 1]
    tau = (tau - tau.min()) / (tau.max() - tau.min() + 1e-10)

    # Store in adata
    adata.obs[pseudotime_key] = tau

    # Store metadata
    adata.uns[f"{pseudotime_key}_info"] = {
        "time_key": time_key,
        "root_cell": int(root),
        "root_markers": root_markers,
        "t_diffusion": t_diffusion,
        "n_hyperedges": len(hyperedges),
        "same_time_weight": same_time_weight,
        "forward_time_weight": forward_time_weight,
        "backward_time_penalty": backward_time_penalty,
        "time_points": unique_times,
    }

    # Report statistics
    print(f"✓ Pseudotime computed:")
    print(f"  Range: [{tau.min():.3f}, {tau.max():.3f}]")
    print(f"  Mean: {tau.mean():.3f}")
    print(f"  Std: {tau.std():.3f}")

    if condition_key is not None:
        for condition in adata.obs[condition_key].unique():
            mask = adata.obs[condition_key] == condition
            print(
                f"  {condition}: mean={tau[mask].mean():.3f}, std={tau[mask].std():.3f}"
            )

    return adata


# ============================================================================
# STEP 5: Post-hoc Condition Analysis
# ============================================================================


def analyze_drug_effects(
    adata: sc.AnnData,
    condition_key: str = "condition",
    control_label: str = "control",
    drug_label: str = "drug",
    time_key: str = "timepoint",
    pseudotime_key: str = "hypergraph_pseudotime",
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Analyze drug effects on pseudotime progression.

    Compares control vs drug at each timepoint and across pseudotime bins.

    Parameters
    ----------
    adata : AnnData
        Annotated data with pseudotime
    condition_key : str
        Column in adata.obs with condition labels
    control_label : str
        Label for control condition
    drug_label : str
        Label for drug condition
    time_key : str
        Column in adata.obs with time point labels
    pseudotime_key : str
        Column in adata.obs with pseudotime values
    n_bins : int
        Number of pseudotime bins for analysis

    Returns
    -------
    results : pd.DataFrame
        Summary statistics of drug effects
    """
    print("=" * 70)
    print("ANALYZING DRUG EFFECTS")
    print("=" * 70)

    if pseudotime_key not in adata.obs:
        raise ValueError(
            f"Pseudotime not found in adata.obs['{pseudotime_key}']. Run compute_time_aware_pseudotime() first."
        )

    results = []

    # Analysis by timepoint
    print("\nPseudotime by timepoint:")
    for timepoint in sorted(adata.obs[time_key].unique()):
        mask_time = adata.obs[time_key] == timepoint

        control_mask = mask_time & (adata.obs[condition_key] == control_label)
        drug_mask = mask_time & (adata.obs[condition_key] == drug_label)

        if control_mask.sum() == 0 or drug_mask.sum() == 0:
            continue

        control_pt = adata.obs.loc[control_mask, pseudotime_key].values
        drug_pt = adata.obs.loc[drug_mask, pseudotime_key].values

        # Statistical test
        stat, pval = mannwhitneyu(control_pt, drug_pt, alternative="two-sided")

        results.append(
            {
                "analysis_type": "by_timepoint",
                "timepoint": timepoint,
                "pseudotime_bin": None,
                "control_n": len(control_pt),
                "drug_n": len(drug_pt),
                "control_mean_pt": control_pt.mean(),
                "control_std_pt": control_pt.std(),
                "drug_mean_pt": drug_pt.mean(),
                "drug_std_pt": drug_pt.std(),
                "delta_pt": drug_pt.mean() - control_pt.mean(),
                "fold_change": drug_pt.mean() / (control_pt.mean() + 1e-10),
                "mann_whitney_stat": stat,
                "mann_whitney_pval": pval,
            }
        )

        print(f"  {timepoint}:")
        print(
            f"    Control: {control_pt.mean():.3f} ± {control_pt.std():.3f} (n={len(control_pt)})"
        )
        print(
            f"    Drug: {drug_pt.mean():.3f} ± {drug_pt.std():.3f} (n={len(drug_pt)})"
        )
        print(f"    Delta: {drug_pt.mean() - control_pt.mean():.3f} (p={pval:.3e})")

    # Analysis by pseudotime bins
    print(f"\nCell distribution across {n_bins} pseudotime bins:")
    pt_values = adata.obs[pseudotime_key].values
    bins = np.linspace(pt_values.min(), pt_values.max(), n_bins + 1)
    bin_labels = [f"bin_{i}" for i in range(n_bins)]
    adata.obs["_pt_bin_temp"] = pd.cut(
        pt_values, bins=bins, labels=bin_labels, include_lowest=True
    )

    for bin_label in bin_labels:
        mask_bin = adata.obs["_pt_bin_temp"] == bin_label

        control_mask = mask_bin & (adata.obs[condition_key] == control_label)
        drug_mask = mask_bin & (adata.obs[condition_key] == drug_label)

        control_n = control_mask.sum()
        drug_n = drug_mask.sum()

        if control_n == 0 and drug_n == 0:
            continue

        total_n = control_n + drug_n
        control_frac = control_n / total_n if total_n > 0 else 0
        drug_frac = drug_n / total_n if total_n > 0 else 0

        results.append(
            {
                "analysis_type": "by_pseudotime_bin",
                "timepoint": None,
                "pseudotime_bin": bin_label,
                "control_n": int(control_n),
                "drug_n": int(drug_n),
                "control_fraction": control_frac,
                "drug_fraction": drug_frac,
                "enrichment": drug_frac / (control_frac + 1e-10),
                "control_mean_pt": None,
                "control_std_pt": None,
                "drug_mean_pt": None,
                "drug_std_pt": None,
                "delta_pt": None,
                "fold_change": None,
                "mann_whitney_stat": None,
                "mann_whitney_pval": None,
            }
        )

        print(f"  {bin_label}:")
        print(f"    Control: {control_n} ({control_frac:.1%})")
        print(f"    Drug: {drug_n} ({drug_frac:.1%})")
        print(f"    Enrichment: {drug_frac / (control_frac + 1e-10):.2f}x")

    # Clean up temporary column
    adata.obs.drop("_pt_bin_temp", axis=1, inplace=True)

    results_df = pd.DataFrame(results)

    print(f"\n✓ Analysis complete")
    print(f"  Results shape: {results_df.shape}")

    return results_df


# ============================================================================
# STEP 6: State Discovery
# ============================================================================


def discover_states(
    adata: sc.AnnData,
    hyperedges: List[np.ndarray],
    hyperedge_weights: np.ndarray,
    n_states: int = 5,
    method: str = "hyperedge",
    state_key: str = "hypergraph_state",
) -> sc.AnnData:
    """
    Discover cell states using hypergraph structure.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    hyperedges : List[np.ndarray]
        List of hyperedges
    hyperedge_weights : np.ndarray
        Hyperedge weights
    n_states : int
        Number of states to discover
    method : str
        Clustering method:
        - 'hyperedge': Cluster by hyperedge participation patterns (recommended)
        - 'kmeans': K-means on hypergraph Laplacian eigenvectors
    state_key : str
        Key to store states in adata.obs

    Returns
    -------
    adata : AnnData
        Modified in place with states in adata.obs[state_key]
    """
    print("=" * 70)
    print(f"DISCOVERING {n_states} CELL STATES")
    print("=" * 70)

    n_cells = adata.n_obs

    # Build incidence matrix
    H = build_incidence_matrix(hyperedges, n_cells)

    if method == "hyperedge":
        print("Using hyperedge participation patterns...")

        # Create feature matrix: weighted hyperedge participation
        features = H.multiply(hyperedge_weights[np.newaxis, :])

        # Normalize by cell degree
        cell_degrees = np.array(H.sum(axis=1)).flatten()
        cell_degrees[cell_degrees == 0] = 1.0
        features_normalized = features.multiply(1.0 / cell_degrees[:, np.newaxis])

        # Convert to dense if not too large
        if n_cells * len(hyperedges) < 1e8:
            features_dense = features_normalized.toarray()

            # K-means clustering
            kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=20)
            states = kmeans.fit_predict(features_dense)
        else:
            # Use mini-batch k-means for large datasets
            from sklearn.cluster import MiniBatchKMeans

            kmeans = MiniBatchKMeans(
                n_clusters=n_states, random_state=42, batch_size=1000, n_init=10
            )
            states = kmeans.fit_predict(features_normalized)

    elif method == "kmeans":
        print("Using Laplacian eigenvectors...")

        # Compute hypergraph Laplacian
        node_degrees, edge_degrees = compute_degrees(H, hyperedge_weights)

        # Compute normalized Laplacian
        # L = I - Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}
        Dv_sqrt_inv = np.sqrt(1.0 / (node_degrees + 1e-10))
        De_inv = 1.0 / (edge_degrees + 1e-10)

        HW = H.multiply(hyperedge_weights[np.newaxis, :])
        HW_De_inv = HW.multiply(De_inv[np.newaxis, :])

        L = (
            np.eye(n_cells)
            - Dv_sqrt_inv[:, np.newaxis]
            * (HW_De_inv @ H.T).toarray()
            * Dv_sqrt_inv[np.newaxis, :]
        )

        # Compute eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        idx = np.argsort(eigenvalues)
        features = eigenvectors[:, idx[:n_states]]

        # K-means on eigenvectors
        kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=20)
        states = kmeans.fit_predict(features)

    else:
        raise ValueError(
            f"Unknown method: {method}. Choose from: 'hyperedge', 'kmeans'"
        )

    # Store states
    adata.obs[state_key] = states.astype(str)

    # Report statistics
    print(f"✓ States discovered:")
    state_counts = pd.Series(states).value_counts().sort_index()
    for state_id, count in state_counts.items():
        print(f"  State {state_id}: {count} cells ({count/n_cells:.1%})")

    return adata


# ============================================================================
# STEP 7: Complete Workflow
# ============================================================================


def run_tf_hypergraph_drug_analysis(
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
    t_diffusion: int = 10,
    output_dir: Optional[str] = None,
) -> Tuple[sc.AnnData, pd.DataFrame, Dict]:
    """
    Complete TF-based hypergraph drug perturbation analysis workflow.

    This function runs the entire pipeline:
    1. Compute TF activities
    2. Build TF-based hypergraph
    3. Compute time-aware pseudotime
    4. Analyze drug effects
    5. Discover cell states
    6. Generate visualizations (if output_dir provided)

    Parameters
    ----------
    adata : AnnData
        Annotated data with gene expression
    condition_key : str
        Column in adata.obs with condition labels
    control_label : str
        Label for control condition
    drug_label : str
        Label for drug condition
    time_key : str
        Column in adata.obs with time point labels
    organism : str
        Organism for TF network ('human' or 'mouse')
    root_markers : List[str], optional
        Marker genes for root cells
    top_n_tfs : int, optional
        Number of top variable TFs to use (None = all TFs, default).
    z_score_threshold : float
        Z-score threshold for TF activity edges (default 2.0).
    component_k : int
        k for intra-TF connected-component splitting (default 10).
    n_states : int
        Number of cell states to discover
    t_diffusion : int
        Number of diffusion steps
    output_dir : str, optional
        Directory to save results and figures

    Returns
    -------
    adata : AnnData
        Modified AnnData with all results
    drug_effects : pd.DataFrame
        Summary of drug effects
    objects : Dict
        Dictionary with intermediate objects (hyperedges, etc.)
    """
    print("\n" + "=" * 70)
    print("TF-BASED HYPERGRAPH DRUG PERTURBATION ANALYSIS")
    print("=" * 70)
    print(f"Dataset: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"Conditions: {adata.obs[condition_key].unique()}")
    print(f"Timepoints: {adata.obs[time_key].unique()}")
    print("=" * 70 + "\n")

    # Step 1: Compute TF activities
    adata = compute_tf_activities(adata, organism=organism)

    # Step 2: Build TF-based hypergraph
    hyperedges, hyperedge_types, hyperedge_weights = build_tf_hypergraph(
        adata,
        top_n_tfs=top_n_tfs,
        z_score_threshold=z_score_threshold,
        component_k=component_k,
    )

    # Step 3: Compute time-aware pseudotime
    adata = compute_time_aware_pseudotime(
        adata,
        hyperedges,
        hyperedge_weights,
        time_key=time_key,
        condition_key=condition_key,
        root_markers=root_markers,
        t_diffusion=t_diffusion,
    )

    # Step 4: Analyze drug effects
    drug_effects = analyze_drug_effects(
        adata,
        condition_key=condition_key,
        control_label=control_label,
        drug_label=drug_label,
        time_key=time_key,
    )

    # Step 5: Discover states
    adata = discover_states(adata, hyperedges, hyperedge_weights, n_states=n_states)

    # Step 6: Save results if output directory provided
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving results to: {output_dir}")

        # Save drug effects
        drug_effects.to_csv(output_path / "drug_effects_summary.csv", index=False)
        print(f"  ✓ Saved: drug_effects_summary.csv")

        # Save AnnData
        adata.write_h5ad(output_path / "adata_with_tf_hypergraph_analysis.h5ad")
        print(f"  ✓ Saved: adata_with_tf_hypergraph_analysis.h5ad")

    # Collect objects
    objects = {
        "hyperedges": hyperedges,
        "hyperedge_types": hyperedge_types,
        "hyperedge_weights": hyperedge_weights,
        "n_hyperedges": len(hyperedges),
        "n_cells": adata.n_obs,
    }

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results stored in adata.obs:")
    print(f"  - tf_activities: TF activity matrix")
    print(f"  - hypergraph_pseudotime: Time-aware pseudotime")
    print(f"  - hypergraph_state: Discovered cell states")
    print("=" * 70 + "\n")

    return adata, drug_effects, objects


def plot_hypergraph_diagnostics(
    adata: sc.AnnData,
    hyperedges: List[np.ndarray],
    hyperedge_types: List[str],
    hyperedge_weights: np.ndarray,
    S: csr_matrix,
    pseudotime: Optional[np.ndarray] = None,
    celltype_key: str = "CellTypes",
    umap_key: str = "X_umap",
    figsize: Tuple[int, int] = (18, 14),
    save: Optional[str] = None,
) -> plt.Figure:
    """
    Diagnostic figure for a constructed TF hypergraph.

    Six panels covering edge structure, operator spectrum, node degree
    distribution, and UMAP projections — together these tell you whether
    the hypergraph is biologically sensible and spectrally useful.

    Panel layout
    ------------
    Row 1 (edge structure)
      A  Edge size distribution by edge type — check that co-activity edges
         are large and variable, programme edges are smaller and coherent.
      B  Edge weight distribution by edge type — weights should be spread,
         not all piled at one value.
      C  Number of edges per edge type (bar chart).

    Row 2 (operator spectrum & node degrees)
      D  Top 20 eigenvalues of S — the spectral gap (λ1 − λ2) should be
         clearly visible; a flat spectrum means no trajectory information.
      E  Node (cell) degree distribution — non-regular = good.  Should be
         right-skewed (most cells in few edges, some in many).
      F  Cell degree on UMAP — stem/progenitor cells should have higher
         degree than terminally differentiated cells.

    Parameters
    ----------
    adata : AnnData
        Must have ``adata.obsm[umap_key]``.
    hyperedges : list of ndarray
        Hyperedge member arrays from ``build_tf_hypergraph``.
    hyperedge_types : list of str
        Type label per hyperedge.
    hyperedge_weights : ndarray
        Weight per hyperedge.
    S : csr_matrix
        Symmetric normalised hypergraph operator (output of
        ``build_view_operator``).
    pseudotime : ndarray, optional
        Cell pseudotime values — if provided, replaces the degree UMAP
        panel with a side-by-side degree / pseudotime comparison.
    celltype_key : str
        ``adata.obs`` column for cell-type annotation (used to label
        high-degree cells on the UMAP).
    umap_key : str
        Key in ``adata.obsm`` for the 2-D embedding.
    figsize : tuple
        Figure dimensions in inches.
    save : str, optional
        If given, save the figure to this path.

    Returns
    -------
    fig : matplotlib.Figure
    """
    from scipy.sparse import issparse
    from scipy.sparse.linalg import eigsh

    # ------------------------------------------------------------------ #
    # Prepare data                                                         #
    # ------------------------------------------------------------------ #
    sizes = np.array([len(e) for e in hyperedges])
    weights = hyperedge_weights.copy()

    # Edge type families (co-activity / programme / pair / other)
    def _family(t: str) -> str:
        if t.startswith("tf_coactivity"):
            return "co-activity"
        if t.startswith("tf_programme"):
            return "programme"
        if t.startswith("tf_pair"):
            return "TF-pair"
        return "other"

    families = np.array([_family(t) for t in hyperedge_types])
    unique_families = ["co-activity", "programme", "TF-pair", "other"]
    palette = {
        "co-activity": "#3b82d4",
        "programme": "#16a34a",
        "TF-pair": "#7c5cd8",
        "other": "#57606a",
    }

    # Node degrees from S (row-wise nnz as degree proxy)
    node_deg = np.diff(S.indptr)

    # UMAP coords
    if umap_key in adata.obsm:
        xy = adata.obsm[umap_key]
    else:
        xy = None

    # Top eigenvalues of S
    k_eig = min(22, S.shape[0] - 2)
    try:
        evals, _ = eigsh(
            S.astype(float), k=k_eig, which="LM", maxiter=S.shape[0] * 10, tol=1e-4
        )
        evals = np.sort(evals)[::-1]
    except Exception:
        evals = np.array([])

    # ------------------------------------------------------------------ #
    # Build figure                                                         #
    # ------------------------------------------------------------------ #
    n_cols = 3
    n_rows = 3 if pseudotime is not None else 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle("Hypergraph Diagnostic", fontsize=15, fontweight="bold", y=1.01)

    # ---- Panel A: edge size distribution by family ---- #
    ax = axes[0, 0]
    for fam in unique_families:
        mask = families == fam
        if mask.sum() == 0:
            continue
        ax.hist(
            sizes[mask],
            bins=30,
            alpha=0.6,
            color=palette[fam],
            label=f"{fam} (n={mask.sum()})",
            edgecolor="none",
        )
    ax.set_xlabel("Hyperedge size (# cells)")
    ax.set_ylabel("Count")
    ax.set_title("A — Edge size distribution")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # ---- Panel B: edge weight distribution by family ---- #
    ax = axes[0, 1]
    for fam in unique_families:
        mask = families == fam
        if mask.sum() == 0:
            continue
        ax.hist(
            weights[mask],
            bins=30,
            alpha=0.6,
            color=palette[fam],
            label=fam,
            edgecolor="none",
        )
    ax.set_xlabel("Edge weight")
    ax.set_ylabel("Count")
    ax.set_title("B — Edge weight distribution")
    ax.legend(fontsize=8)

    # ---- Panel C: edge counts per type ---- #
    ax = axes[0, 2]
    fam_counts = {
        f: (families == f).sum() for f in unique_families if (families == f).sum() > 0
    }
    bars = ax.bar(
        list(fam_counts.keys()),
        list(fam_counts.values()),
        color=[palette[f] for f in fam_counts],
    )
    ax.bar_label(bars, fmt="%d", fontsize=9)
    ax.set_ylabel("Number of hyperedges")
    ax.set_title("C — Edge counts by type")
    ax.tick_params(axis="x", rotation=20)

    # ---- Panel D: eigenvalue spectrum ---- #
    ax = axes[1, 0]
    if len(evals) > 0:
        ax.scatter(range(len(evals)), evals, s=30, color="#3b82d4", zorder=3)
        ax.plot(range(len(evals)), evals, lw=1, color="#3b82d4", alpha=0.5)
        if len(evals) > 1:
            gap = evals[0] - evals[1]
            ax.axvline(0.5, ls="--", lw=1, color="#e05252", alpha=0.7)
            ax.annotate(
                f"gap = {gap:.3f}",
                xy=(1, evals[1]),
                xytext=(3, (evals[0] + evals[1]) / 2),
                fontsize=9,
                color="#e05252",
                arrowprops=dict(arrowstyle="->", color="#e05252", lw=0.8),
            )
        ax.set_xlabel("Eigenvalue rank")
        ax.set_ylabel("Eigenvalue λ")
        ax.set_title(
            "D — Operator eigenvalue spectrum\n(gap = λ₁ − λ₂, bigger = better)"
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Eigendecomposition failed",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("D — Operator eigenvalue spectrum")

    # ---- Panel E: node degree distribution ---- #
    ax = axes[1, 1]
    ax.hist(node_deg, bins=40, color="#3b82d4", edgecolor="none", alpha=0.8)
    ax.axvline(
        node_deg.mean(),
        ls="--",
        lw=1.5,
        color="#e05252",
        label=f"mean={node_deg.mean():.1f}",
    )
    ax.set_xlabel("Node degree (row nnz in S)")
    ax.set_ylabel("Count")
    ax.set_title("E — Node degree distribution\n(right-skew = non-regular = good)")
    ax.legend(fontsize=8)

    # Gini coefficient as regularity measure
    sorted_deg = np.sort(node_deg)
    n = len(sorted_deg)
    gini = (
        (
            2 * np.sum((np.arange(1, n + 1)) * sorted_deg) / (n * sorted_deg.sum())
            - (n + 1) / n
        )
        if sorted_deg.sum() > 0
        else 0
    )
    ax.text(
        0.97,
        0.97,
        f"Gini = {gini:.3f}\n(0=regular, 1=max skew)",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
        color="#57606a",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e5e7eb"),
    )

    # ---- Panel F: cell degree on UMAP ---- #
    ax = axes[1, 2]
    if xy is not None:
        sc_plot = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=node_deg,
            cmap="viridis",
            s=3,
            alpha=0.7,
            rasterized=True,
        )
        plt.colorbar(sc_plot, ax=ax, label="degree", pad=0.01, shrink=0.9)
        ax.set_title("F — Cell degree on UMAP\n(stem cells should have high degree)")
    else:
        ax.text(
            0.5,
            0.5,
            f"No UMAP found\n(key='{umap_key}')",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("F — Cell degree on UMAP")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])

    # ---- Row 3 (optional): pseudotime vs degree comparison ---- #
    if pseudotime is not None and xy is not None:
        # G: pseudotime on UMAP
        ax = axes[2, 0]
        sc2 = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=pseudotime,
            cmap="plasma",
            s=3,
            alpha=0.7,
            rasterized=True,
        )
        plt.colorbar(sc2, ax=ax, label="pseudotime", pad=0.01, shrink=0.9)
        ax.set_title("G — Pseudotime on UMAP")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])

        # H: scatter degree vs pseudotime (should be negatively correlated)
        ax = axes[2, 1]
        ax.scatter(
            pseudotime, node_deg, s=4, alpha=0.2, color="#3b82d4", rasterized=True
        )
        from scipy.stats import spearmanr as _spr

        rho, pval = _spr(pseudotime, node_deg)
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel("Node degree")
        ax.set_title(f"H — Degree vs pseudotime\n(ρ={rho:.2f}, p={pval:.1e})")

        # I: pseudotime distribution by cell type
        ax = axes[2, 2]
        if celltype_key in adata.obs:
            # Build a fast name→position lookup
            obs_pos = {name: i for i, name in enumerate(adata.obs_names)}
            ct_series = adata.obs[celltype_key]
            ct_order = (
                ct_series.groupby(ct_series, observed=True)
                .apply(
                    lambda g: float(
                        np.median(pseudotime[[obs_pos[n] for n in g.index]])
                    )
                )
                .sort_values()
                .index.tolist()
            )
            ct_data = [
                pseudotime[[obs_pos[n] for n in adata.obs_names[ct_series == ct]]]
                for ct in ct_order
            ]
            parts = ax.violinplot(
                ct_data,
                positions=range(len(ct_order)),
                showmedians=True,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor("#3b82d4")
                pc.set_alpha(0.6)
            ax.set_xticks(range(len(ct_order)))
            ax.set_xticklabels(ct_order, rotation=40, ha="right", fontsize=7)
            ax.set_ylabel("Pseudotime")
            ax.set_title("I — Pseudotime by cell type\n(SC should be low, TA/EC high)")
        else:
            ax.text(
                0.5,
                0.5,
                f"'{celltype_key}' not in adata.obs",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("I — Pseudotime by cell type")

    elif pseudotime is not None:
        # No UMAP — just show pseudotime histogram in the extra row
        for col in range(3):
            axes[2, col].set_visible(False)

    # ------------------------------------------------------------------ #
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  Saved diagnostic figure: {save}")
    return fig


# Made with Bob
