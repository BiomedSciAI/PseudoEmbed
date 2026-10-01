"""
Hyperedge Interpretability
==========================

Extracts *which named hyperedges* carry the control-vs-condition signal and
which shape the spectral axes used for pseudotime.

Two independent measures are provided, and their intersection is what should be
trusted:

1. **Condition composition** (:func:`edge_condition_enrichment`)
   Per-edge enrichment of treated cells against background, with Fisher's exact
   test and BH-FDR.  Answers "is this cell state over-represented in the drug
   arm?"  Requires no spectral machinery.

2. **Spectral attribution** (:func:`edge_axis_contributions`)
   Exact closed-form decomposition of a spectral axis over hyperedges.  Answers
   "which edges shape the axis that separates conditions?"

These are different claims.  (1) is about *composition* — who is in the edge.
(2) is about *geometry* — how the edge bends the embedding.  An edge can score
on one and not the other, and a hit on both is far stronger evidence than either
alone, which is why :func:`combine_evidence` reports them side by side rather
than blending them into a single score.

Neither measure establishes causality, and neither is immune to batch
confounding: see :func:`edge_condition_enrichment` for the ``group`` argument
that lets you run the identical test against patient/batch labels as a negative
control.

The spectral decomposition
--------------------------
The symmetric normalised operator is a *sum over hyperedges* of rank-one terms::

    S = Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}
      = sum_e (w_e / d_e) * u_e u_e^T ,    u_e(v) = 1[v in e] / sqrt(d_v)

so for any unit vector phi the quadratic form splits exactly::

    phi^T S phi = sum_e (w_e / d_e) * ( sum_{v in e} phi(v)/sqrt(d_v) )^2

Each summand is one edge's contribution.  It is non-negative, requires a single
pass over the incidence matrix, and is exact — not a perturbation estimate, not
a permutation score, and it needs no refitting.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import fisher_exact

__all__ = [
    "edge_condition_enrichment",
    "edge_axis_contributions",
    "edge_onset",
    "onset_group_test",
    "aggregate_to_programme",
    "combine_evidence",
]


# ============================================================================
# Shared helpers
# ============================================================================


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg step-up adjusted p-values.

    Implemented directly rather than via statsmodels to avoid adding a hard
    dependency for twelve lines of arithmetic.  NaN p-values (from degenerate
    edges) are carried through as NaN rather than silently treated as 1.0, so
    they cannot be mistaken for tested-and-not-significant.
    """
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out

    p_f = p[finite]
    n = p_f.size
    order = np.argsort(p_f)
    ranked = p_f[order]
    adj = ranked * n / np.arange(1, n + 1)
    # Step-up: enforce monotonicity from the largest p downwards.
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)

    restored = np.empty(n, dtype=float)
    restored[order] = adj
    out[finite] = restored
    return out


def _edge_names(
    hyperedge_types: Sequence[str],
    hyperedge_meta: Optional[Sequence[Dict]],
    n_edges: int,
) -> Tuple[List[str], List[str]]:
    """
    Resolve a display name and a programme key for every edge.

    The programme key is what per-edge scores aggregate over: the pathway name
    (collapsing its quantile bins), the TF name (collapsing its programme
    components), or the kNN anchor.  Metadata is preferred when present; parsing
    the type label is the fallback, since the TF view already encodes its names
    there (``tf_programme_MYC``) and predates the metadata contract.
    """
    have_meta = hyperedge_meta is not None and len(hyperedge_meta) == n_edges

    names, keys = [], []
    for i in range(n_edges):
        # Prefer metadata, but only per edge and only when it actually carries a
        # name.  Views that predate the metadata contract (the TF view) are
        # pooled alongside ones that don't, so an all-or-nothing choice would
        # discard real names from whichever side loses.
        if have_meta:
            m = hyperedge_meta[i]
            if isinstance(m, dict) and m.get("name") is not None:
                names.append(str(m["name"]))
                keys.append(f"{m.get('view', 'unknown')}:{m['name']}")
                continue

        label = str(hyperedge_types[i]) if i < len(hyperedge_types) else f"edge_{i}"
        if label.startswith("pathway_") and "_bin" in label:
            # pathway_{name}_bin{k} -> collapse bins into one programme
            core = label[len("pathway_") : label.rindex("_bin")]
            names.append(label)
            keys.append(f"pathway:{core}")
        elif label.startswith("tf_programme_"):
            core = label[len("tf_programme_") :]
            names.append(label)
            keys.append(f"tf:{core}")
        elif label.startswith("tf_coactivity_"):
            core = label[len("tf_coactivity_") :]
            names.append(label)
            keys.append(f"tf:{core}")
        elif label.startswith("tf_pair_"):
            names.append(label)
            keys.append(f"tf_pair:{label[len('tf_pair_'):]}")
        elif label.startswith("pca_knn_star"):
            names.append(label)
            keys.append("pca:knn_star")
        elif label.startswith("gene_"):
            # gene_{symbol} -> one edge per gene, so the programme *is* the gene.
            # Unlike the pathway view there are no bins to collapse; the point of
            # this view is that the per-edge score is already a per-gene score.
            core = label[len("gene_") :]
            names.append(core)
            keys.append(f"gene:{core}")
        else:
            names.append(label)
            keys.append(f"other:{label}")
    return names, keys


# ============================================================================
# Change 2 — condition composition per edge
# ============================================================================


def edge_condition_enrichment(
    hyperedges: Sequence[np.ndarray],
    condition: np.ndarray,
    hyperedge_types: Optional[Sequence[str]] = None,
    hyperedge_meta: Optional[Sequence[Dict]] = None,
    group: Optional[np.ndarray] = None,
    min_cells: int = 10,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Test each hyperedge for over-representation of treated cells.

    For every edge, compares the fraction of condition-positive cells inside the
    edge against the fraction outside it (Fisher's exact, two-sided), then
    applies BH-FDR across edges.

    Parameters
    ----------
    hyperedges : sequence of ndarray
        Hyperedges as arrays of cell indices.
    condition : ndarray, shape (n_cells,)
        Binary condition labels.  ``0`` = control, ``1`` = treated.  Boolean
        arrays are accepted.  Must not be constant.
    hyperedge_types : sequence of str, optional
        Per-edge type labels, used for naming when ``hyperedge_meta`` is absent.
    hyperedge_meta : sequence of dict, optional
        Per-edge metadata from the builders' ``return_meta=True`` path.
        Preferred source of edge names and programme keys.
    group : ndarray, shape (n_cells,), optional
        A *confounder* label (e.g. patient or batch).  For each edge, reports
        its dominant level (``group_dominant``), the fraction of the edge made
        up of that level (``group_purity``), and that purity's log2 enrichment
        over background (``group_log2_enrich``).  An edge with high
        ``group_purity`` is largely one patient, so its condition enrichment
        cannot be read as a treatment effect.  Worth checking because
        pseudotime on at least one dataset in this project was overwhelmingly
        patient batch, which would otherwise present as a drug effect.
    min_cells : int
        Edges smaller than this are reported with NaN statistics rather than
        tested; Fisher's exact on a handful of cells has no power and only
        inflates the multiple-testing burden.
    eps : float
        Floor for the log2 ratio.

    Returns
    -------
    DataFrame
        One row per edge, sorted by ``fdr`` ascending, with columns:
        ``edge_index``, ``edge_name``, ``programme``, ``n_cells``,
        ``n_treated``, ``frac_treated``, ``background_frac_treated``,
        ``log2_enrich``, ``p_value``, ``fdr``, and — when ``group`` is given —
        ``group_dominant``, ``group_purity``, ``group_log2_enrich``.

    Notes
    -----
    Edges within a view are strongly dependent (quantile bins partition the
    cells; kNN stars overlap heavily), so BH-FDR is anti-conservative here.
    Treat the ranking as a screen and the FDR as indicative.  Aggregate to
    programme level with :func:`aggregate_to_programme` before drawing
    biological conclusions.
    """
    cond = np.asarray(condition)
    if cond.dtype == bool:
        cond = cond.astype(int)
    cond = cond.astype(float)

    uniq = np.unique(cond[np.isfinite(cond)])
    if uniq.size < 2:
        raise ValueError(
            f"`condition` must contain both classes; found only {uniq.tolist()}."
        )
    if not np.all(np.isin(uniq, (0.0, 1.0))):
        raise ValueError(
            f"`condition` must be binary 0/1 (or boolean); found values {uniq.tolist()}."
        )

    n_cells = cond.size
    n_edges = len(hyperedges)
    names, keys = _edge_names(hyperedge_types or [], hyperedge_meta, n_edges)

    total_treated = float(cond.sum())
    total_control = float(n_cells - total_treated)

    g_arr = None
    if group is not None:
        g_arr = np.asarray(group)
        if g_arr.size != n_cells:
            raise ValueError(
                f"`group` has length {g_arr.size} but `condition` has {n_cells}."
            )
        if np.unique(g_arr).size < 2:
            warnings.warn(
                "`group` has a single level — skipping the confounder control column.",
                UserWarning,
            )
            g_arr = None

    rows: List[Dict] = []
    for e_idx in range(n_edges):
        members = np.asarray(hyperedges[e_idx], dtype=int)
        members = np.unique(members)  # an edge is a set; duplicates would double-count
        k = members.size

        row: Dict = {
            "edge_index": e_idx,
            "edge_name": names[e_idx],
            "programme": keys[e_idx],
            "n_cells": int(k),
        }

        if k < min_cells:
            row.update(
                n_treated=int(cond[members].sum()) if k else 0,
                frac_treated=float(cond[members].mean()) if k else np.nan,
                background_frac_treated=np.nan,
                log2_enrich=np.nan,
                p_value=np.nan,
            )
            if g_arr is not None:
                row["group_dominant"] = np.nan
                row["group_purity"] = np.nan
                row["group_log2_enrich"] = np.nan
            rows.append(row)
            continue

        in_treated = float(cond[members].sum())
        in_control = float(k - in_treated)
        out_treated = total_treated - in_treated
        out_control = total_control - in_control

        frac_in = in_treated / k
        denom_out = out_treated + out_control
        frac_out = out_treated / denom_out if denom_out > 0 else np.nan

        # Compare against cells *outside* the edge, not the global mean: with
        # large edges (a quartile bin is n/4 cells) the edge contributes enough
        # of the global mean to dilute its own apparent enrichment.
        log2_enrich = np.log2((frac_in + eps) / (frac_out + eps))

        try:
            _, p = fisher_exact(
                [[in_treated, in_control], [out_treated, out_control]],
                alternative="two-sided",
            )
        except ValueError:
            p = np.nan

        row.update(
            n_treated=int(in_treated),
            frac_treated=float(frac_in),
            background_frac_treated=float(frac_out),
            log2_enrich=float(log2_enrich),
            p_value=float(p),
        )

        if g_arr is not None:
            # Score the edge's *own* dominant group level, not a globally chosen
            # one: the question is "is this edge just one patient?", which a
            # fixed reference level cannot answer for every edge.  Reported as
            # purity so it is directly readable — 1.0 means the edge is entirely
            # one batch and its condition enrichment is uninterpretable.
            g_edge = g_arr[members]
            lv, ct = np.unique(g_edge, return_counts=True)
            dom_i = int(np.argmax(ct))
            dom_level = lv[dom_i]
            purity_in = float(ct[dom_i]) / k
            purity_bg = float((g_arr == dom_level).sum()) / n_cells
            row["group_dominant"] = dom_level
            row["group_purity"] = purity_in
            row["group_log2_enrich"] = float(
                np.log2((purity_in + eps) / (purity_bg + eps))
            )

        rows.append(row)

    df = pd.DataFrame(rows)
    df["fdr"] = _bh_fdr(df["p_value"].to_numpy())

    n_tested = int(np.isfinite(df["p_value"]).sum())
    n_skipped = n_edges - n_tested
    if n_skipped:
        print(
            f"  Condition enrichment: {n_tested} edges tested, "
            f"{n_skipped} skipped (< {min_cells} cells)"
        )

    return df.sort_values("fdr", na_position="last").reset_index(drop=True)


# ============================================================================
# Change 3 — exact spectral attribution
# ============================================================================


def edge_axis_contributions(
    H: csr_matrix,
    hyperedge_weights: np.ndarray,
    phi: np.ndarray,
    node_degrees: Optional[np.ndarray] = None,
    edge_degrees: Optional[np.ndarray] = None,
    hyperedge_types: Optional[Sequence[str]] = None,
    hyperedge_meta: Optional[Sequence[Dict]] = None,
    axis_names: Optional[Sequence[str]] = None,
    eps: float = 1e-10,
) -> pd.DataFrame:
    """
    Decompose one or more spectral axes over hyperedges, exactly.

    Computes, for each edge ``e`` and axis ``phi_k``::

        contribution(e, k) = (w_e / d_e) * ( sum_{v in e} phi_k(v)/sqrt(d_v) )^2

    which sums over edges to exactly ``phi_k^T S phi_k`` (up to the zero-degree
    rows the operator masks out).  Contributions are non-negative.

    The **signed** inner sum ``proj(e, k) = sum_{v in e} phi_k(v)/sqrt(d_v)`` is
    also returned, as ``signed_{axis}`` and (size-normalised)
    ``signed_mean_{axis}``.  Squaring discards the sign, so magnitude alone
    cannot say whether an edge sits at the early or late end of a maturation
    axis, or on the treated or control side of a perturbation axis.  Since
    ``contrib = (w_e/d_e) * proj^2`` exactly, the signed column carries *only*
    direction and adds no independent magnitude information -- but that direction
    comes from the operator itself rather than from a construction convention
    such as ``bin_source``.

    Parameters
    ----------
    H : csr_matrix, shape (n_cells, n_edges)
        Incidence matrix, as built by ``build_incidence_matrix``.
    hyperedge_weights : ndarray, shape (n_edges,)
        Per-edge weights ``w_e``.
    phi : ndarray, shape (n_cells,) or (n_cells, n_axes)
        Spectral axis/axes to decompose — e.g. the perturbation axes for
        drug structure, or the maturation axes for trajectory structure.
    node_degrees, edge_degrees : ndarray, optional
        Precomputed degrees.  Recomputed from ``H`` and ``hyperedge_weights``
        when omitted; pass them to guarantee they match the operator that
        produced ``phi``.
    hyperedge_types, hyperedge_meta : optional
        Naming sources, as in :func:`edge_condition_enrichment`.
    axis_names : sequence of str, optional
        Column labels for the axes.  Defaults to ``axis0, axis1, ...``.
    eps : float
        Degree floor, matching ``_build_symmetric_operator``.

    Returns
    -------
    DataFrame
        One row per edge with ``edge_index``, ``edge_name``, ``programme``,
        ``n_cells``, one ``contrib_{axis}`` column per axis, one
        ``frac_{axis}`` column giving that edge's share of the axis total, and
        ``contrib_total`` summing the raw contributions across axes.

        Also **two signed columns per axis**, which the ``contrib_*`` columns
        cannot substitute for: the contribution squares the projection, so it is
        sign-blind and says only *how much* an edge moves the axis, never
        *which end*.  Direction lives in ``signed_{axis}`` (the raw summed
        projection, which grows with edge size and is therefore not comparable
        between a 21-cell and a 57-cell edge) and ``signed_mean_{axis}`` (the
        per-member mean, which is).  Any claim about early versus late must rest
        on ``signed_mean_{axis}``.

        Rows are returned **sorted by ``contrib_total`` descending**, so joining
        this frame to another per-edge table positionally misattributes every
        row -- merge on ``edge_index`` (see :func:`edge_onset`).

    Notes
    -----
    Contributions depend on ``d_v`` and ``d_e``, which change with the operator.
    Raw values are therefore **not comparable across runs** with different views
    or fusion weights — compare the ``frac_*`` columns instead, which are
    normalised within-run.
    """
    H = csr_matrix(H)
    w = np.asarray(hyperedge_weights, dtype=float).reshape(-1)
    n_cells, n_edges = H.shape

    if w.size != n_edges:
        raise ValueError(
            f"hyperedge_weights has length {w.size} but H has {n_edges} columns."
        )

    phi = np.asarray(phi, dtype=float)
    if phi.ndim == 1:
        phi = phi[:, None]
    if phi.shape[0] != n_cells:
        raise ValueError(
            f"phi has {phi.shape[0]} rows but H has {n_cells} rows (cells)."
        )
    n_axes = phi.shape[1]

    if node_degrees is None or edge_degrees is None:
        # Mirrors compute_degrees: d_v = sum_e w_e H[v,e], d_e = |e|.
        node_degrees = np.asarray(H @ w, dtype=float).reshape(-1)
        edge_degrees = np.asarray(H.sum(axis=0), dtype=float).reshape(-1)
    node_degrees = np.asarray(node_degrees, dtype=float).reshape(-1)
    edge_degrees = np.asarray(edge_degrees, dtype=float).reshape(-1)

    dv_safe = np.maximum(node_degrees, eps)
    de_safe = np.maximum(edge_degrees, eps)

    # proj[e, k] = sum_{v in e} phi_k(v) / sqrt(d_v)
    proj = H.T @ (phi / np.sqrt(dv_safe)[:, None])  # (n_edges, n_axes)
    contrib = (w / de_safe)[:, None] * np.square(proj)  # (n_edges, n_axes)

    names, keys = _edge_names(hyperedge_types or [], hyperedge_meta, n_edges)
    if axis_names is None:
        axis_names = [f"axis{k}" for k in range(n_axes)]
    elif len(axis_names) != n_axes:
        raise ValueError(
            f"axis_names has {len(axis_names)} entries but phi has {n_axes} axes."
        )

    out = pd.DataFrame(
        {
            "edge_index": np.arange(n_edges),
            "edge_name": names,
            "programme": keys,
            "n_cells": np.asarray(H.sum(axis=0)).reshape(-1).astype(int),
        }
    )
    for k, ax in enumerate(axis_names):
        col = contrib[:, k]
        out[f"contrib_{ax}"] = col
        total = col.sum()
        out[f"frac_{ax}"] = col / total if total > 0 else np.nan
        # Signed companion: the inner sum before squaring.  The quadratic form
        # discards this sign, which is why edge direction previously had to come
        # from the `bin_source` construction convention rather than from the
        # operator.  Sign says which END of the axis the edge sits at: positive
        # means its members carry above-average phi_k, negative below.  Magnitude
        # is not independent information -- contrib = (w_e/d_e) * proj^2 exactly,
        # so proj adds direction only.
        out[f"signed_{ax}"] = proj[:, k]
        # Per-member mean rather than the raw sum, so edges of different sizes
        # are comparable: proj grows with |e|, so a 200-cell edge otherwise
        # dominates an 8-cell one on membership count alone even when its cells
        # are less displaced.  Divide by d_e (= |e|), not sqrt(d_e) -- the latter
        # leaves a residual sqrt(|e|) size dependence.
        out[f"signed_mean_{ax}"] = proj[:, k] / de_safe

    out["contrib_total"] = contrib.sum(axis=1)
    return out.sort_values("contrib_total", ascending=False).reset_index(drop=True)


# ============================================================================
# Edge onset: turning named hyperedges into a timeline
# ============================================================================


def edge_onset(
    hyperedges: Sequence[np.ndarray],
    axis: np.ndarray,
    real_time: Optional[np.ndarray] = None,
    *,
    agg: str = "mean",
    edge_index: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """
    Position of each hyperedge's member cells along an axis -- its *onset*.

    Onset is what turns a set of named hyperedges into a timeline: an edge whose
    members sit early on the axis switches on early.  Ordering edges by onset
    gives a sequence of named programmes, which is the thing a pairwise graph
    cannot produce.

    When ``real_time`` is supplied the same statistic is computed against it, so
    the ordering the model produced can be read against a real clock rather than
    only against its own axis.

    Parameters
    ----------
    hyperedges
        The **pooled** edge list, in the same order the incidence matrix was
        built from.
    axis
        Per-cell axis, e.g. the fitted pseudotime, length ``n_cells``.
    real_time
        Optional per-cell external clock (sampling hours).
    agg
        ``"mean"`` (default) or ``"median"``.  The median is more robust for an
        edge with a few far-flung members, but the mean is what makes onsets
        comparable to a bin centre.
    edge_index
        Explicit edge ids.  Defaults to ``range(len(hyperedges))``.

    Returns
    -------
    DataFrame
        Columns ``edge_index``, ``n_cells``, ``onset`` and -- when ``real_time``
        is given -- ``onset_real``.

        The ``edge_index`` column is emitted **explicitly and deliberately**:
        :func:`edge_axis_contributions` returns its rows sorted by
        ``contrib_total`` descending, so joining these two frames positionally
        misattributes every onset.  Merge on ``edge_index``.
    """
    if agg not in ("mean", "median"):
        raise ValueError(f"agg must be 'mean' or 'median', got {agg!r}")
    reduce = np.mean if agg == "mean" else np.median

    axis = np.asarray(axis, dtype=float)
    rt = None if real_time is None else np.asarray(real_time, dtype=float)
    if rt is not None and len(rt) != len(axis):
        raise ValueError(f"real_time has length {len(rt)} but axis has {len(axis)}")

    idx = list(range(len(hyperedges))) if edge_index is None else list(edge_index)
    if len(idx) != len(hyperedges):
        raise ValueError(
            f"edge_index has {len(idx)} entries for {len(hyperedges)} edges"
        )

    rows: List[Dict] = []
    for i, e in zip(idx, hyperedges):
        m = np.asarray(e, dtype=int)
        # An empty edge has no onset.  NaN, never 0.0: a zero would sort to the
        # early end and read as "switches on first".
        row = {
            "edge_index": int(i),
            "n_cells": int(m.size),
            "onset": float(reduce(axis[m])) if m.size else np.nan,
        }
        if rt is not None:
            row["onset_real"] = float(reduce(rt[m])) if m.size else np.nan
        rows.append(row)

    cols = ["edge_index", "n_cells", "onset"] + (
        ["onset_real"] if rt is not None else []
    )
    return pd.DataFrame(rows, columns=cols)


def onset_group_test(
    onset_df: pd.DataFrame,
    groups,
    *,
    value_col: str = "onset",
    reference: Optional[str] = None,
    alternative: str = "less",
) -> pd.DataFrame:
    """
    Mann-Whitney over edge onsets, comparing named groups of edges.

    Tests whether one labelled set of edges switches on earlier than another --
    for instance whether a curated early-response panel precedes a curated
    effector panel on an axis that was fitted without either.

    **The groups are for labelling a ranking, never for selecting it.**  Scoring
    a curated set against a ranking that the same curated set defined is a prior
    recovering itself, and has produced a spuriously tiny p-value here before.

    Parameters
    ----------
    onset_df
        As returned by :func:`edge_onset` (or that merged onto a contribution
        table).  Must carry ``value_col``.
    groups
        Per-row group label, aligned to ``onset_df``.  Rows labelled ``None`` or
        NaN are excluded, so an "other" pool can be left unlabelled.
    reference
        Group every other group is tested against.  Defaults to the group with
        the **largest** median onset, so the default reading is
        "does group X precede the latest group".
    alternative
        Passed to :func:`scipy.stats.mannwhitneyu`; ``"less"`` asks whether the
        group's onsets are earlier than the reference's.

    Returns
    -------
    DataFrame
        One row per non-reference group: ``group``, ``n``, ``median``,
        ``reference``, ``n_reference``, ``median_reference``, ``U``, ``p``.

        A ``q`` column (Benjamini-Hochberg) is added **only when more than one
        comparison was made** -- reporting a corrected q for a single test would
        imply a correction that did no work.
    """
    from scipy.stats import mannwhitneyu

    if value_col not in onset_df.columns:
        raise KeyError(
            f"{value_col!r} not in onset table (have: {list(onset_df.columns)})"
        )

    g = pd.Series(list(groups), index=onset_df.index)
    keep = g.notna() & (g.astype(str) != "None")
    sub = onset_df.loc[keep]
    g = g.loc[keep].astype(str)
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "group",
                "n",
                "median",
                "reference",
                "n_reference",
                "median_reference",
                "U",
                "p",
            ]
        )

    med = sub.groupby(g.values)[value_col].median()
    ref = str(med.idxmax()) if reference is None else str(reference)
    if ref not in set(g.values):
        raise ValueError(f"reference {ref!r} not among groups {sorted(set(g))}")

    b = sub.loc[g.values == ref, value_col].dropna().values
    rows: List[Dict] = []
    for name in [x for x in sorted(set(g.values)) if x != ref]:
        a = sub.loc[g.values == name, value_col].dropna().values
        if len(a) == 0 or len(b) == 0:
            u = p = np.nan
        else:
            r = mannwhitneyu(a, b, alternative=alternative)
            u, p = float(r.statistic), float(r.pvalue)
        rows.append(
            {
                "group": name,
                "n": int(len(a)),
                "median": float(np.median(a)) if len(a) else np.nan,
                "reference": ref,
                "n_reference": int(len(b)),
                "median_reference": float(np.median(b)) if len(b) else np.nan,
                "U": u,
                "p": p,
            }
        )

    out = pd.DataFrame(rows)
    if len(out) > 1:
        out["q"] = _bh_fdr(out["p"].values)
    return out


# ============================================================================
# Aggregation and evidence combination
# ============================================================================


def aggregate_to_programme(
    df: pd.DataFrame,
    value_cols: Optional[Sequence[str]] = None,
    programme_col: str = "programme",
) -> pd.DataFrame:
    """
    Collapse per-edge scores to per-programme scores.

    A pathway contributes ``n_bins`` edges that partition all cells, and a TF
    contributes one edge per coherent component, so a per-edge table is not
    directly interpretable as biology.  Contribution-like columns are summed
    (contributions are additive by construction); enrichment-like columns are
    reduced to the value of the single most extreme edge, since a drug effect
    confined to one quantile bin is real signal that averaging would wash out.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`edge_axis_contributions` or
        :func:`edge_condition_enrichment`.
    value_cols : sequence of str, optional
        Columns to aggregate.  Defaults to every ``contrib_*``/``frac_*``
        column plus ``log2_enrich`` and ``fdr`` when present.
    programme_col : str
        Grouping column.

    Returns
    -------
    DataFrame
        One row per programme, with ``n_edges`` and ``n_cells_max`` alongside
        the aggregated columns.  ``fdr`` is reduced by minimum, which is
        optimistic across dependent bins — read it as a screen.
    """
    if programme_col not in df.columns:
        raise ValueError(f"`df` has no column '{programme_col}'.")

    if value_cols is None:
        value_cols = [
            c for c in df.columns if c.startswith("contrib_") or c.startswith("frac_")
        ]
        for extra in ("log2_enrich", "fdr", "group_log2_enrich", "group_purity"):
            if extra in df.columns:
                value_cols.append(extra)

    agg: Dict[str, object] = {}
    for c in value_cols:
        if c.startswith("contrib_") or c.startswith("frac_"):
            agg[c] = "sum"
        elif c == "fdr":
            agg[c] = "min"
        elif c in ("log2_enrich", "group_log2_enrich"):
            # Most extreme bin, sign preserved.
            #
            # This is lossy for a bidirectional programme: a pathway whose top
            # AND bottom bin are both drug-enriched (up in one cell population,
            # down in another) collapses to whichever bin is larger, and the
            # other direction disappears. That is why direction is NOT read off
            # this table -- hypergraph_drivers.rank_drivers works per edge, so
            # bins stay separate and both directions survive. Use this
            # aggregation for a programme-level overview, not for directionality.
            agg[c] = lambda s: s.loc[s.abs().idxmax()] if s.notna().any() else np.nan
        elif c == "group_purity":
            # Worst case across bins: if any constituent edge is one batch, the
            # programme's condition signal is suspect.
            agg[c] = "max"
        else:
            agg[c] = "mean"

    grouped = df.groupby(programme_col, dropna=False)
    out = grouped.agg(agg)
    out["n_edges"] = grouped.size()
    if "n_cells" in df.columns:
        out["n_cells_max"] = grouped["n_cells"].max()

    sort_key = (
        "contrib_total"
        if "contrib_total" in out.columns
        else ("log2_enrich" if "log2_enrich" in out.columns else out.columns[0])
    )
    ascending = sort_key == "fdr"
    return out.sort_values(sort_key, ascending=ascending).reset_index()


def combine_evidence(
    enrichment: pd.DataFrame,
    contributions: pd.DataFrame,
    axis_col: Optional[str] = None,
    fdr_threshold: float = 0.05,
    top_frac: float = 0.5,
) -> pd.DataFrame:
    """
    Join composition and geometry evidence at programme level.

    The two measures are deliberately *not* blended into one score: they answer
    different questions, and a weighted sum would hide which one fired.  The
    join instead flags programmes supported by both.

    Parameters
    ----------
    enrichment : DataFrame
        Programme-level output from :func:`aggregate_to_programme` applied to
        :func:`edge_condition_enrichment`.
    contributions : DataFrame
        Programme-level output from :func:`aggregate_to_programme` applied to
        :func:`edge_axis_contributions`.
    axis_col : str, optional
        Which ``frac_*`` column to use as the geometry score.  Defaults to the
        first ``frac_*`` column found.
    fdr_threshold : float
        FDR cutoff for the composition criterion.
    top_frac : float
        Cumulative-share cutoff for the geometry criterion: a programme
        qualifies if it falls in the set of highest-contributing programmes
        together accounting for this fraction of the axis.

    Returns
    -------
    DataFrame
        Programmes with ``log2_enrich``, ``fdr``, the geometry score,
        ``both_support``, and ``evidence`` in
        {``both``, ``composition_only``, ``geometry_only``, ``neither``}.
    """
    if axis_col is None:
        candidates = [c for c in contributions.columns if c.startswith("frac_")]
        if not candidates:
            raise ValueError(
                "`contributions` has no 'frac_*' column; pass axis_col explicitly."
            )
        axis_col = candidates[0]

    keep_enr = [
        c
        for c in (
            "programme",
            "log2_enrich",
            "fdr",
            "group_purity",
            "group_log2_enrich",
            "n_edges",
        )
        if c in enrichment.columns
    ]
    keep_con = ["programme", axis_col]

    merged = enrichment[keep_enr].merge(
        contributions[keep_con], on="programme", how="outer"
    )

    comp_ok = merged.get("fdr", pd.Series(np.nan, index=merged.index)) <= fdr_threshold

    ranked = merged.sort_values(axis_col, ascending=False)
    cum = ranked[axis_col].fillna(0.0).cumsum()
    geom_programmes = set(ranked.loc[cum <= top_frac, "programme"])
    if not geom_programmes and len(ranked):
        geom_programmes = {ranked.iloc[0]["programme"]}
    geom_ok = merged["programme"].isin(geom_programmes)

    comp_ok = comp_ok.fillna(False).to_numpy(dtype=bool)
    geom_ok = geom_ok.to_numpy(dtype=bool)

    merged["both_support"] = comp_ok & geom_ok
    merged["evidence"] = np.select(
        [comp_ok & geom_ok, comp_ok & ~geom_ok, ~comp_ok & geom_ok],
        ["both", "composition_only", "geometry_only"],
        default="neither",
    )

    return merged.sort_values(
        ["both_support", axis_col], ascending=[False, False]
    ).reset_index(drop=True)
