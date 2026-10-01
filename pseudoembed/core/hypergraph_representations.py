"""Alternative hypergraph representations: star expansion and the hyperedge line graph.

Why this module exists
----------------------
The operator the pipeline builds for pseudotime,

    S = Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2},

is a weighted-graph kernel plus a diagonal, so a spectral readout of it sees only
that graph; the higher-order content survives solely as the ``1/|e|`` size
discount.  More importantly, ``H H^T`` is a function *on cells*, so no operator in
the pipeline has the hyperedge set as its domain.  Named hyperedges (TF regulons,
pathways, gene programmes) are where this project's strongest results live, yet
they only ever appeared as post-hoc enrichment of a cell-space axis.

Both constructions here put hyperedges in the *vertex* position:

* :func:`build_star_expansion` — Schaub eq. 29.  A bipartite graph on
  ``V ∪ E``.  Unlike clique expansion it *determines* the incidence matrix, so it
  is the lossless option; cells and edges land in one shared coordinate space.
* :func:`build_line_graph` — Schaub §5.1.  ``L = H^T H``, the edge-edge
  co-occupancy graph, whose vertices are hyperedges outright.

Neither touches the trajectory axis.  Both are additive and off by default; see
:func:`add_hypergraph_representations` for the single flag-gated entry point.

Read before trusting the numbers in the design prompt
-----------------------------------------------------
The implementation prompt for this module quoted measured figures that do **not**
reproduce.  Verified on ``richard_t_cell.h5ad``, gene view, in one session
(``experiments/test_star_and_line_graph.py``):

===========================  ==============  ==============================
quantity                     prompt claim    measured (m=1167 / m=666)
===========================  ==============  ==============================
edges in gene view           1169            1167 (n_genes=2000) / 666 (1200)
cardinality median           22              17
cardinality range            10-40           10-64 / 10-58
density (|e ∩ f| > 0)        0.985           0.774 / 0.788
median Jaccard               0.176           0.0385 / 0.0417
max Jaccard                  0.893           0.5614
"spectral gap lam1-lam2"     0.824           0.145  (lam1 itself is 0.726)
===========================  ==============  ==============================

The last row is the load-bearing correction.  0.824 is close to lam1 (the gap
from *zero*), not to lam1-lam2, so the prompt's "normalised spectral gap
lam1-lam2 = 0.824, so real modular structure exists" reads a large lam1 as
evidence of modularity.  It is the opposite: for a normalised Laplacian, lam1 far
above 0 means the graph is *well connected with no clean bipartition*.  A
near-zero lam1 would indicate modules.  So the honest summary of this line graph
is "one dense blob, weak substructure", which is also what density 0.77 and
median Jaccard 0.04 say.  Do not re-derive a modularity claim from lam1.

Tensor methods (Schaub §5.2) are deliberately absent: the adjacency tensor has
order ``max_e |e|`` and pathway-view edges average ~4107 cells, so it is not a
computational object.  Ruled out on representability, not effort.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

__all__ = [
    "build_incidence",
    "StarExpansion",
    "build_star_expansion",
    "LineGraph",
    "build_line_graph",
    "dedupe_programmes",
    "effective_n_tests",
    "add_hypergraph_representations",
]


# ---------------------------------------------------------------------------
# Shared incidence matrix
# ---------------------------------------------------------------------------
def build_incidence(
    hyperedges: Sequence[np.ndarray],
    n_cells: int,
    hyperedge_weights: Optional[np.ndarray] = None,
) -> sp.csr_matrix:
    """Binary (or weighted) incidence matrix ``H`` of shape ``(n_cells, n_edges)``.

    ``H[v, e] = 1`` iff cell ``v`` is a member of hyperedge ``e``.  No package-level
    incidence builder existed before this module (verified by grep); the pipeline
    only ever built the *fused operator*, never ``H`` itself as a returned object.

    Parameters
    ----------
    hyperedges : sequence of integer arrays
        Cell indices per edge, exactly the format returned by
        ``build_tf_hypergraph`` / ``build_gene_hypergraph`` / etc.
    n_cells : int
        Number of cells (rows).  Passed explicitly because an edge list alone
        cannot distinguish "no cell 249" from "only 249 cells".
    hyperedge_weights : array, optional
        Per-edge weight placed on that edge's column.  ``None`` gives a binary H.

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    m = len(hyperedges)
    if m == 0:
        return sp.csr_matrix((n_cells, 0))
    sizes = [len(np.asarray(e)) for e in hyperedges]
    rows = (
        np.concatenate([np.asarray(e, dtype=int) for e in hyperedges])
        if sum(sizes)
        else np.array([], int)
    )
    cols = (
        np.concatenate([np.full(s, i, dtype=int) for i, s in enumerate(sizes)])
        if sum(sizes)
        else np.array([], int)
    )
    if hyperedge_weights is None:
        vals = np.ones(len(rows), dtype=float)
    else:
        w = np.asarray(hyperedge_weights, dtype=float)
        if w.shape[0] != m:
            raise ValueError(
                f"hyperedge_weights has {w.shape[0]} entries for {m} hyperedges"
            )
        vals = np.concatenate([np.full(s, w[i]) for i, s in enumerate(sizes)])
    if rows.size and rows.max() >= n_cells:
        raise ValueError(
            f"hyperedge references cell {rows.max()} but n_cells={n_cells}"
        )
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_cells, m))


# ---------------------------------------------------------------------------
# Part 1 -- star expansion (Schaub eq. 29)
# ---------------------------------------------------------------------------
@dataclass
class StarExpansion:
    """Result of :func:`build_star_expansion`.

    Attributes
    ----------
    cell_coords : ndarray, shape (n_cells, k)
        Upper block of each retained eigenvector: one coordinate per cell.
    edge_coords : ndarray, shape (n_edges, k)
        Lower block: one coordinate per HYPEREDGE.  This is the object the
        package previously had no way to produce.
    edge_names : list of str
        Row order of ``edge_coords``.  **Row order contract:** row ``i`` of
        ``edge_coords`` is ``edge_names[i]`` is ``hyperedges[i]`` as passed in.
        Cells occupy rows ``0..n_cells-1`` of the joint space and edges occupy
        ``n_cells..n_cells+n_edges-1``; ``cell_coords`` and ``edge_coords`` are
        that single space split at ``n_cells``, so the two are directly
        comparable and a cell-edge distance is meaningful.
    eigenvalues : ndarray, shape (k,)
        Eigenvalues of the normalised Laplacian, ascending, trivial one removed.
    eigenvalues_raw : ndarray
        All computed eigenvalues including the trivial component, ascending.
        Reported so the spectrum near zero is inspectable.
    trivial_index : int
        Index into ``eigenvalues_raw`` judged trivial and dropped.
    trivial_is_trivial : bool
        Whether that component passed the constancy diagnostic.  A False here
        means the drop is NOT justified and the caller should look at the
        spectrum rather than trust the coordinates.
    trivial_diagnostic : str
        Human-readable form of the check, with the measured numbers.
    n_components : int
        Number of connected components of the bipartite graph, from the count of
        eigenvalues at ~0.  More than one means coordinates are only comparable
        within a component.
    """

    cell_coords: np.ndarray
    edge_coords: np.ndarray
    edge_names: List[str]
    eigenvalues: np.ndarray
    eigenvalues_raw: np.ndarray
    trivial_index: int
    trivial_is_trivial: bool
    trivial_diagnostic: str
    n_components: int
    view_weights: Optional[Dict[str, float]] = None

    @property
    def n_cells(self) -> int:
        return int(self.cell_coords.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_coords.shape[0])

    def edge_frame(self):
        """Edge coordinates as a DataFrame indexed by edge name."""
        import pandas as pd

        return pd.DataFrame(
            self.edge_coords,
            index=pd.Index(self.edge_names, name="edge"),
            columns=[f"star{j + 1}" for j in range(self.edge_coords.shape[1])],
        )


def build_star_expansion(
    hyperedges: Sequence[np.ndarray],
    n_cells: int,
    *,
    hyperedge_weights: Optional[np.ndarray] = None,
    edge_names: Optional[Sequence[str]] = None,
    n_components: int = 8,
    use_weights: bool = True,
    verbose: bool = True,
) -> StarExpansion:
    """Star expansion of a hypergraph, spectrally embedded (Schaub eq. 29).

    Builds ``A_star = [[0, Z W], [W Z^T, 0]]`` on ``V ∪ E`` and embeds the
    **normalised Laplacian** ``L = I - D^{-1/2} A D^{-1/2}`` using its SMALLEST
    eigenvalues.

    Why smallest, and why the Laplacian at all
    ------------------------------------------
    ``A_star`` is bipartite, so its spectrum is symmetric about zero and its
    leading (largest-|.|) eigenvectors encode the *bipartition* — cells versus
    edges — not the geometry.  Taking largest eigenvalues here, the way
    ``multiview_hypergraph.py`` does for its own operator ``S`` (``eigsh(...,
    which="LM")`` around :1743), would return that split and nothing useful.  The
    normalised Laplacian's small eigenvalues give the smooth-on-the-graph
    coordinates instead, which is the object we want.  This is a real difference
    in construction, not a stylistic one; do not "make it consistent" with the
    pseudotime path.

    Parameters
    ----------
    hyperedges, n_cells, hyperedge_weights
        As :func:`build_incidence`.  Weights sit on the off-diagonal block.
    edge_names : sequence of str, optional
        Names for the edge rows; defaults to ``e0, e1, ...``.
    n_components : int
        Number of non-trivial coordinates to retain.
    use_weights : bool
        If False, ignore ``hyperedge_weights`` (binary incidence).  Exposed so
        the caller can A/B whether weights are inert here, which they measurably
        are in early fusion (they get renormalised inside a shared ``Dv``; see
        the incidence-mass correction at ``multiview_hypergraph.py:1119-1138``).
    verbose : bool
        Print the spectrum near zero and the triviality diagnostic.

    Returns
    -------
    StarExpansion
    """
    from scipy.sparse.linalg import eigsh

    m = len(hyperedges)
    if m == 0:
        raise ValueError("star expansion needs at least one hyperedge")
    if edge_names is None:
        edge_names = [f"e{i}" for i in range(m)]
    if len(edge_names) != m:
        raise ValueError(f"edge_names has {len(edge_names)} entries for {m} hyperedges")

    Z = build_incidence(
        hyperedges,
        n_cells,
        hyperedge_weights if use_weights else None,
    )
    N = n_cells + m
    # A_star = [[0, Z], [Z^T, 0]] -- weights already folded into Z's columns.
    A = sp.bmat([[None, Z], [Z.T, None]], format="csr")

    deg = np.asarray(A.sum(axis=1)).ravel()
    isolated = deg <= 0
    if isolated.any() and verbose:
        print(
            f"  [star] {int(isolated.sum())} isolated nodes "
            f"({int(isolated[:n_cells].sum())} cells, "
            f"{int(isolated[n_cells:].sum())} edges) -- given zero coordinates"
        )
    # Degree-0 rows would divide by zero; hold them out and reinsert as zeros.
    keep = ~isolated
    n_keep = int(keep.sum())
    if n_keep < 3:
        raise ValueError(
            f"star expansion has only {n_keep} non-isolated nodes; nothing to embed"
        )
    A_k = A[np.ix_(keep, keep)]
    dk = deg[keep]
    Dm = sp.diags(1.0 / np.sqrt(dk))
    L = sp.eye(n_keep, format="csr") - Dm @ A_k @ Dm
    L = ((L + L.T) * 0.5).tocsr()  # kill asymmetry from float round-off

    k = int(min(n_components + 1, n_keep - 1))
    if k < 2:
        raise ValueError("need at least 2 components for a star embedding")
    # SMALLEST eigenvalues.  'SA' on L directly is ill-conditioned for ARPACK, so
    # use the standard shift: largest of (2I - L) == smallest of L, 2 being an
    # upper bound on the normalised-Laplacian spectrum.
    try:
        shifted = sp.eye(n_keep, format="csr") * 2.0 - L
        ev_s, vec_k = eigsh(shifted, k=k, which="LM", tol=1e-8)
        ev = 2.0 - ev_s
    except Exception as exc:  # pragma: no cover - ARPACK fallback
        warnings.warn(
            f"eigsh(shift) failed ({exc}); retrying with which='SA'.", UserWarning
        )
        ev, vec_k = eigsh(L, k=k, which="SA", tol=1e-8)
    order = np.argsort(ev)
    ev, vec_k = ev[order], vec_k[:, order]

    # --- triviality diagnostic -----------------------------------------------
    # For a connected graph, L's first eigenvector is D^{1/2} 1 with eigenvalue 0.
    # Verify rather than assume: this module's sibling bug (multiview_hypergraph
    # dropping component 0 by POSITION on the assertion "always index 0") was
    # exactly this assumption going unchecked, and it turned out to be false
    # there -- CV(v0) = 0.236 on Schiebinger, i.e. a real component was being
    # discarded as trivial.  Here the trivial vector genuinely IS recoverable,
    # because we use the Laplacian, so the check should pass; if it does not, say
    # so loudly instead of silently dropping a real coordinate.
    expected = np.sqrt(dk)
    expected = expected / np.linalg.norm(expected)
    v0 = vec_k[:, 0]
    align = float(abs(np.dot(v0 / np.linalg.norm(v0), expected)))
    # CV of |v0| after undoing the D^{1/2} tilt: should be ~0 for the trivial one.
    untilted = np.abs(v0) / expected
    cv = float(np.std(untilted) / (np.mean(untilted) + 1e-12))
    n_zero = int((ev < 1e-8).sum())
    is_trivial = bool(ev[0] < 1e-6 and (align > 0.99 or cv < 0.05))
    diag = (
        f"lam0={ev[0]:.3e}, |<v0, D^1/2 1>|={align:.4f}, CV(v0/D^1/2)={cv:.4f} "
        f"-> {'TRIVIAL as expected' if is_trivial else 'NOT trivial (!!)'}"
    )
    if verbose:
        print(f"  [star] spectrum near zero: {np.round(ev[: min(6, k)], 6)}")
        print(f"  [star] triviality check: {diag}")
        print(
            f"  [star] connected components (lam < 1e-8): {n_zero}"
            + ("" if n_zero == 1 else "  -- coords only comparable WITHIN a component")
        )
    if not is_trivial:
        warnings.warn(
            "Star expansion: the leading component is not the expected trivial "
            f"vector ({diag}). Dropping it anyway would discard real signal; "
            "inspect eigenvalues_raw before using the coordinates.",
            UserWarning,
        )

    kept = np.array([j for j in range(k) if j != 0])
    vec_nt = vec_k[:, kept]
    ev_nt = ev[kept]

    # Reinsert isolated rows as zero coordinates, then split at n_cells.
    full = np.zeros((N, vec_nt.shape[1]), dtype=float)
    full[keep] = vec_nt
    return StarExpansion(
        cell_coords=full[:n_cells],
        edge_coords=full[n_cells:],
        edge_names=list(edge_names),
        eigenvalues=ev_nt,
        eigenvalues_raw=ev,
        trivial_index=0,
        trivial_is_trivial=is_trivial,
        trivial_diagnostic=diag,
        n_components=n_zero,
    )


# ---------------------------------------------------------------------------
# Part 2 -- hyperedge line graph (Schaub 5.1)
# ---------------------------------------------------------------------------
@dataclass
class LineGraph:
    """Result of :func:`build_line_graph`.

    Attributes
    ----------
    counts : scipy.sparse matrix, shape (m, m)
        Raw ``H^T H`` off-diagonal: ``counts[e, f] = |e ∩ f|``.
    jaccard : ndarray, shape (m, m)
        ``|e ∩ f| / |e ∪ f|``, zero diagonal.
    excess : ndarray, shape (m, m)
        Overlap in excess of what edge sizes alone predict, as a z-like score
        against a degree-preserving (hypergeometric) null.  See
        :func:`build_line_graph` for why this and not a size residual.
    sizes : ndarray, shape (m,)
        Edge cardinalities, ``|e|``.
    edge_names : list of str
        Row/column order of every matrix above.
    density : float
        Fraction of edge pairs with ``|e ∩ f| > 0``.
    stats : dict
        Measured summary: median/max Jaccard, spectral gaps before and after
        normalisation, and the event floor used.
    """

    counts: sp.spmatrix
    jaccard: np.ndarray
    excess: np.ndarray
    sizes: np.ndarray
    edge_names: List[str]
    density: float
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_edges(self) -> int:
        return len(self.edge_names)


def _normalised_laplacian_spectrum(A: np.ndarray, k: int = 6) -> np.ndarray:
    """Ascending eigenvalues of ``I - D^{-1/2} A D^{-1/2}`` on the non-isolated part."""
    A = np.asarray(A, dtype=float).copy()
    np.fill_diagonal(A, 0.0)
    d = A.sum(1)
    keep = d > 0
    if keep.sum() < 3:
        return np.array([np.nan])
    A2 = A[np.ix_(keep, keep)]
    d2 = d[keep]
    Dm = np.diag(1.0 / np.sqrt(d2))
    L = np.eye(A2.shape[0]) - Dm @ A2 @ Dm
    L = (L + L.T) * 0.5
    ev = np.linalg.eigvalsh(L)
    return ev[: max(k, 3)]


def build_line_graph(
    hyperedges: Sequence[np.ndarray],
    n_cells: int,
    *,
    edge_names: Optional[Sequence[str]] = None,
    event_floor: int = 3,
    verbose: bool = True,
) -> LineGraph:
    """Hyperedge line graph ``L = H^T H`` with a size-aware normalisation.

    ``counts[e, f]`` is the number of cells co-occupied by programmes ``e`` and
    ``f``, so the vertices of this graph are hyperedges — the dual object the
    package lacked entirely (no ``H^T`` construction existed in
    ``multiview_hypergraph.py``; verified by grep).

    Normalisation, and why not a size residual
    ------------------------------------------
    Raw counts are near-uninformative: with median-17 edges drawn from 250 cells
    most overlap is chance, and measured density is 0.77 (see module docstring).
    Two normalisations are returned:

    * ``jaccard`` — scale-free, the standard choice.
    * ``excess`` — observed minus expected overlap divided by the null SD, under
      a hypergeometric draw with the two edge sizes held fixed.  This is the
      degree-preserving null.

    We deliberately do **not** residualise on size.  For a pair of small edges
    the overlap is *undefined* rather than low — a 10-cell and a 12-cell edge
    cannot exceed Jaccard 0.55 no matter how related the programmes are — so a
    size residual overcorrects exactly where the data is thinnest.  Instead the
    ``excess`` score is only reported where the expected count clears
    ``event_floor``; below that the entry is set to 0 and counted in
    ``stats["below_floor_pairs"]``.  This is the stratify-with-a-floor approach.

    Validation note: do not check the normalisation with a rank correlation
    against size.  Jaccard is a monotone function of the counts at fixed sizes,
    so a Spearman against ``|e|`` can look unchanged while the *pair ranking*
    across different size strata has changed completely.  The relevant evidence
    is the change in the spectral gap and in which pairs top the ranking, both of
    which this function reports.

    Parameters
    ----------
    event_floor : int
        Minimum expected overlap for an ``excess`` entry to be trusted.
    """
    m = len(hyperedges)
    if m == 0:
        raise ValueError("line graph needs at least one hyperedge")
    if edge_names is None:
        edge_names = [f"e{i}" for i in range(m)]
    if len(edge_names) != m:
        raise ValueError(f"edge_names has {len(edge_names)} entries for {m} hyperedges")

    H = build_incidence(hyperedges, n_cells)
    C = (H.T @ H).tocsr()
    inter = np.asarray(C.todense(), dtype=float)
    sizes = inter.diagonal().copy()
    np.fill_diagonal(inter, 0.0)

    n_pairs = m * (m - 1) / 2.0
    density = float((inter > 0).sum() / 2.0 / n_pairs) if n_pairs else 0.0

    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(jac, 0.0)

    # Hypergeometric null with both margins fixed: E = |e||f|/n,
    # Var = |e||f|(n-|e|)(n-|f|) / (n^2 (n-1)).
    n = float(n_cells)
    ef = sizes[:, None] * sizes[None, :]
    exp = ef / n
    var = ef * (n - sizes[:, None]) * (n - sizes[None, :]) / (n * n * (n - 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        exc = np.where(var > 0, (inter - exp) / np.sqrt(var), 0.0)
    below = exp < event_floor
    np.fill_diagonal(below, False)
    exc = np.where(below, 0.0, exc)
    np.fill_diagonal(exc, 0.0)

    iu = np.triu_indices(m, 1)
    jv = jac[iu]
    ev_raw = _normalised_laplacian_spectrum(inter)
    ev_jac = _normalised_laplacian_spectrum(jac)
    ev_exc = _normalised_laplacian_spectrum(np.clip(exc, 0, None))

    stats = {
        "n_edges": m,
        "n_cells": n_cells,
        "card_median": float(np.median(sizes)),
        "card_min": float(sizes.min()),
        "card_max": float(sizes.max()),
        "density": density,
        "jaccard_median": float(np.median(jv)),
        "jaccard_max": float(jv.max()) if jv.size else 0.0,
        "jaccard_mean": float(jv.mean()) if jv.size else 0.0,
        # lam1 is the gap from ZERO (algebraic connectivity of the normalised
        # Laplacian); lam1_minus_lam2 is the actual spectral *gap* between the
        # first two non-trivial modes.  Keeping both because conflating them is
        # how "0.824" got read as evidence of modular structure -- a large lam1
        # means well-connected, i.e. the opposite.
        "lam1_raw": float(ev_raw[1]) if ev_raw.size > 1 else float("nan"),
        "lam1_minus_lam2_raw": (
            float(abs(ev_raw[1] - ev_raw[2])) if ev_raw.size > 2 else float("nan")
        ),
        "lam1_jaccard": float(ev_jac[1]) if ev_jac.size > 1 else float("nan"),
        "lam1_minus_lam2_jaccard": (
            float(abs(ev_jac[1] - ev_jac[2])) if ev_jac.size > 2 else float("nan")
        ),
        "lam1_excess": float(ev_exc[1]) if ev_exc.size > 1 else float("nan"),
        "lam1_minus_lam2_excess": (
            float(abs(ev_exc[1] - ev_exc[2])) if ev_exc.size > 2 else float("nan")
        ),
        "event_floor": event_floor,
        "below_floor_pairs": int(below[iu].sum()),
        "below_floor_frac": float(below[iu].mean()) if jv.size else 0.0,
        "spectrum_raw": np.round(ev_raw[:6], 6).tolist(),
        "spectrum_jaccard": np.round(ev_jac[:6], 6).tolist(),
    }

    if verbose:
        print(
            f"  [line] m={m} edges, card median={stats['card_median']:.0f} "
            f"({stats['card_min']:.0f}-{stats['card_max']:.0f}), "
            f"density={density:.4f}"
        )
        print(
            f"  [line] Jaccard median={stats['jaccard_median']:.4f} "
            f"max={stats['jaccard_max']:.4f}"
        )
        print(
            f"  [line] lam1 raw={stats['lam1_raw']:.4f} "
            f"jaccard={stats['lam1_jaccard']:.4f} "
            f"excess={stats['lam1_excess']:.4f}"
        )
        print(
            f"  [line] lam1-lam2 raw={stats['lam1_minus_lam2_raw']:.4f} "
            f"jaccard={stats['lam1_minus_lam2_jaccard']:.4f} "
            f"excess={stats['lam1_minus_lam2_excess']:.4f}"
        )
        print(
            f"  [line] pairs below event floor (E<{event_floor}): "
            f"{stats['below_floor_pairs']:,} ({stats['below_floor_frac']:.1%})"
        )

    return LineGraph(
        counts=C,
        jaccard=jac,
        excess=exc,
        sizes=sizes,
        edge_names=list(edge_names),
        density=density,
        stats=stats,
    )


def dedupe_programmes(
    lg: LineGraph,
    *,
    threshold: float = 0.5,
    metric: str = "jaccard",
    verbose: bool = True,
):
    """Group near-identical programmes so one finding is not reported twice.

    Single-linkage components of the graph ``metric >= threshold``.  A
    0.89-Jaccard pair is one programme under two names; a prior IPA convergence
    result in this project turned out to be 306 pairs that were one module
    counted repeatedly, so this is a live failure mode rather than hygiene.

    Returns
    -------
    pandas.DataFrame
        One row per edge: ``edge``, ``group`` (representative index), ``group_size``,
        ``is_representative``, ``size``.  The representative is the LARGEST edge in
        the group, since the smaller ones are typically its subsets.
    dict
        Summary counts, including ``n_merged`` — the number of edges that are not
        their group's representative, i.e. how many the step actually removes.
    """
    import pandas as pd
    from scipy.sparse.csgraph import connected_components

    if metric == "jaccard":
        M = lg.jaccard
    elif metric == "excess":
        M = lg.excess
    else:
        raise ValueError("metric must be 'jaccard' or 'excess'")
    A = sp.csr_matrix(M >= threshold)
    n_comp, labels = connected_components(A, directed=False)

    sizes = lg.sizes
    rep_of: Dict[int, int] = {}
    for g in range(n_comp):
        members = np.where(labels == g)[0]
        rep_of[g] = int(members[np.argmax(sizes[members])])
    group_size = np.array(
        [int((labels == labels[i]).sum()) for i in range(len(labels))]
    )
    is_rep = np.array([rep_of[labels[i]] == i for i in range(len(labels))])

    df = pd.DataFrame(
        {
            "edge": lg.edge_names,
            "group": labels,
            "group_size": group_size,
            "is_representative": is_rep,
            "size": sizes,
        }
    )
    n_merged = int((~is_rep).sum())

    # Single-linkage CHAINS: a 100-member group at a low threshold is usually a
    # transitive chain (a~b, b~c, but a and c unrelated), not 100 duplicates.
    # Reporting only n_merged would present chaining as successful dedup, so
    # measure the worst group's internal cohesion and expose it.
    big = int(group_size.max())
    if big > 2:
        members = np.where(labels == labels[int(np.argmax(group_size))])[0]
        sub = M[np.ix_(members, members)]
        iu_s = np.triu_indices(len(members), 1)
        pw = sub[iu_s]
        cohesion = float((pw >= threshold).mean()) if pw.size else 1.0
        min_pair = float(pw.min()) if pw.size else 1.0
    else:
        cohesion, min_pair = 1.0, 1.0

    summary = {
        "metric": metric,
        "threshold": threshold,
        "n_edges": len(labels),
        "n_groups": int(n_comp),
        "n_merged": n_merged,
        "n_multi_groups": int((df.groupby("group").size() > 1).sum()),
        "largest_group": big,
        # Fraction of pairs INSIDE the largest group that actually clear the
        # threshold.  1.0 = a genuine clique of duplicates; near 0 = a chain.
        "largest_group_cohesion": cohesion,
        "largest_group_min_pair": min_pair,
        "chaining_suspected": bool(big > 2 and cohesion < 0.5),
    }
    if verbose:
        print(
            f"  [dedupe] {metric} >= {threshold}: {n_comp} groups from "
            f"{len(labels)} edges -> merges {n_merged} "
            f"({n_merged / max(len(labels), 1):.1%}), "
            f"largest group={summary['largest_group']}"
        )
        if n_merged == 0:
            print(
                "  [dedupe] NOTHING MERGED -- report this as a negative rather "
                "than lowering the threshold until something merges."
            )
        if summary["chaining_suspected"]:
            print(
                f"  [dedupe] WARNING: largest group ({summary['largest_group']} "
                f"members) has only {cohesion:.1%} of its internal pairs above "
                f"{threshold} (min pair {min_pair:.3f}) -- this is single-linkage "
                "CHAINING, not a set of duplicates. Do not report it as dedup."
            )
    return df, summary


def effective_n_tests(
    lg: LineGraph,
    *,
    metric: str = "jaccard",
    verbose: bool = True,
) -> Dict[str, float]:
    """Effective number of independent programmes, for honest per-edge FDR.

    Every per-edge test in the package assumes independence.  It is not harmless:
    a prior conservative bound on a related statistic dropped z from 97.6 to
    14.6.  This gives a Cheverud/Nyholt-style effective count from the
    eigenvalue variance of the programme-programme correlation structure:

        m_eff = 1 + (m - 1) (1 - Var(lambda) / m)

    Reported alongside the naive ``m`` so a caller can see the size of the
    correction rather than assuming it away.  This is a summary statistic, not a
    substitute for a permutation null — it says how far from independent the
    programme set is, which is the thing currently unstated.
    """
    M = lg.jaccard if metric == "jaccard" else lg.excess
    A = np.asarray(M, dtype=float).copy()
    np.fill_diagonal(A, 1.0)
    m = A.shape[0]
    ev = np.linalg.eigvalsh((A + A.T) * 0.5)
    var = float(np.var(ev))
    m_eff = 1.0 + (m - 1.0) * (1.0 - var / m)
    m_eff = float(np.clip(m_eff, 1.0, m))
    out = {
        "m": float(m),
        "m_eff": m_eff,
        "ratio": m_eff / m,
        "eigenvalue_variance": var,
        "metric": metric,
    }
    if verbose:
        print(
            f"  [n_eff] naive m={m}  effective m_eff={m_eff:.1f} "
            f"({m_eff / m:.1%} of naive) from {metric} structure"
        )
    return out


# ---------------------------------------------------------------------------
# Flag-gated entry point
# ---------------------------------------------------------------------------
def add_hypergraph_representations(
    adata,
    hyperedges: Sequence[np.ndarray],
    edge_names: Optional[Sequence[str]] = None,
    *,
    hyperedge_weights: Optional[np.ndarray] = None,
    star: bool = False,
    line_graph: bool = False,
    key: str = "hg",
    n_star_components: int = 8,
    dedupe_threshold: float = 0.5,
    event_floor: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Attach star-expansion and/or line-graph outputs to ``adata``.

    Both are **off by default** and neither touches the trajectory: this function
    only writes new keys and never modifies ``obsm['X_*']`` used by pseudotime,
    the fused operator, or any existing default.

    Writes, when enabled
    --------------------
    ``star=True``
        ``adata.obsm[f"{key}_star_cells"]`` — cell coordinates (n_cells, k).
        ``adata.uns[f"{key}_star"]`` — dict with ``edge_coords`` (n_edges, k),
        ``edge_names`` (the row order of ``edge_coords``), ``eigenvalues``,
        ``eigenvalues_raw``, and the triviality diagnostic.  Edge coordinates go
        in ``uns`` rather than ``obsm`` because they are indexed by hyperedge, not
        by cell, and ``obsm`` rows must align with ``obs``.
    ``line_graph=True``
        ``adata.uns[f"{key}_line_graph"]`` — ``jaccard``, ``excess``, ``sizes``,
        ``edge_names``, ``stats``, plus ``dedupe`` (records) and ``n_eff``.

    Returns
    -------
    dict
        The in-memory objects (``StarExpansion`` / ``LineGraph`` and summaries),
        so a caller can use them without a round trip through ``uns``.
    """
    out: Dict[str, Any] = {}
    if not (star or line_graph):
        if verbose:
            print(
                "  [representations] both star and line_graph are False -- nothing "
                "to do (these features are opt-in)"
            )
        return out

    n_cells = int(adata.n_obs)
    if edge_names is not None and len(edge_names) != len(hyperedges):
        raise ValueError(
            f"edge_names has {len(edge_names)} entries for {len(hyperedges)} hyperedges"
        )

    if star:
        se = build_star_expansion(
            hyperedges,
            n_cells,
            hyperedge_weights=hyperedge_weights,
            edge_names=edge_names,
            n_components=n_star_components,
            verbose=verbose,
        )
        adata.obsm[f"{key}_star_cells"] = se.cell_coords
        adata.uns[f"{key}_star"] = {
            "edge_coords": se.edge_coords,
            # Row order contract, stored next to the data it describes.
            "edge_names": np.array(se.edge_names, dtype=object),
            "eigenvalues": se.eigenvalues,
            "eigenvalues_raw": se.eigenvalues_raw,
            "trivial_index": se.trivial_index,
            "trivial_is_trivial": se.trivial_is_trivial,
            "trivial_diagnostic": se.trivial_diagnostic,
            "n_graph_components": se.n_components,
            "row_order": (
                "edge_coords row i == edge_names[i] == hyperedges[i]; cells and "
                "edges share one space, split at n_cells"
            ),
        }
        out["star"] = se

    if line_graph:
        lg = build_line_graph(
            hyperedges,
            n_cells,
            edge_names=edge_names,
            event_floor=event_floor,
            verbose=verbose,
        )
        dd, dd_sum = dedupe_programmes(lg, threshold=dedupe_threshold, verbose=verbose)
        neff = effective_n_tests(lg, verbose=verbose)
        adata.uns[f"{key}_line_graph"] = {
            "jaccard": lg.jaccard,
            "excess": lg.excess,
            "sizes": lg.sizes,
            "edge_names": np.array(lg.edge_names, dtype=object),
            "stats": lg.stats,
            "dedupe": dd.to_dict("list"),
            "dedupe_summary": dd_sum,
            "n_eff": neff,
        }
        out["line_graph"] = lg
        out["dedupe"] = dd
        out["dedupe_summary"] = dd_sum
        out["n_eff"] = neff

    return out
