"""
Driver Identification and Export
================================

Turns per-hyperedge evidence into a ranked, directional, confidence-scored
driver table — and exports it in forms IPA can consume.

Why this is not differential expression
---------------------------------------
The point of the hypergraph is to say something DE and WGCNA cannot.  A DE test
asks, per gene independently, whether its mean differs between arms.  WGCNA asks
which genes co-vary into modules.  Neither uses the *cell-set* structure: that a
specific group of cells jointly occupying a high-activity state for a programme
is over-represented in one arm, and that this same cell-set is what bends the
spectral axis separating the arms.

Two gene-level paths exist here, and only one is a result
--------------------------------------------------------
:func:`expand_programmes_to_genes` (**deprecated**) apportions the
programme-level effect to member genes by their PROGENy/CollecTRI footprint
weight.  That paragraph used to argue this was a deliberate choice rather than a
fold change.  It was not defensible: within a programme the score is a monotone
function of the prior weight — measured at ``pearson == 1.0`` exactly on real
output — so every gene shares one ``log2_enrich``/``fdr``/``driver_score`` and
the ranking would be byte-identical on shuffled data.  It carries no information
from the experiment.  Its rows now self-label with
``score_source="prior_footprint"``.

:mod:`pseudoembed.core.hypergraph_gene_attribution` is the replacement, and it
does use the cell-set structure: for each hyperedge it measures the treatment
contrast *among that edge's cells only*, times how specific the gene is to that
edge versus the other states the hypergraph found.  A gene must both move inside
the state and characterise it.  Global DE cannot produce that number, and the
regression tests prove it — a gene raised in all treated cells is demoted despite
an identical fold change.

Directionality
--------------
Quantile bins are ordered: ``bin 0`` is the *lowest*-activity quartile,
``bin n_bins-1`` the highest.  A driver's direction therefore comes from the
interaction of two facts:

===================  =====================  ==========================
Bin                  Enriched in            Interpretation
===================  =====================  ==========================
high (top bin)       treated                programme **up** in drug
low (bottom bin)     treated                programme **down** in drug
high (top bin)       control                programme **down** in drug
low (bottom bin)     control                programme **up** in drug
===================  =====================  ==========================

This is why bins must not be collapsed by taking the single most extreme one: a
pathway can be bidirectional (high bin drug-enriched *and* low bin drug-enriched
in different cell populations), which is real biology that averaging destroys.
:func:`rank_drivers` keeps both and flags them ``bidirectional``.

Confidence
----------
``fdr`` from the composition test is anti-conservative — quantile bins partition
the same cells, so the tests are strongly dependent.  The spectral contribution
has no uncertainty at all on its own.  :func:`bootstrap_driver_stability`
therefore resamples cells, rebuilds the operator, and reports how often each
programme survives in the top-k.  Selection frequency is the confidence measure
for a *ranking*, which is what a driver list is.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from pseudoembed.core.hypergraph_interpret import (
    _bh_fdr,
    edge_axis_contributions,
    edge_condition_enrichment,
)

__all__ = [
    "rank_drivers",
    "bootstrap_driver_stability",
    "expand_programmes_to_genes",
    "export_ipa_gene_table",
    "export_driver_tables",
    "plot_driver_summary",
    "run_driver_analysis",
]


# ============================================================================
# Directional driver ranking
# ============================================================================


def rank_drivers(
    enrichment: pd.DataFrame,
    contributions: pd.DataFrame,
    axis_col: Optional[str] = None,
    fdr_threshold: float = 0.05,
    min_abs_log2: float = 0.25,
    max_group_enrichment: Optional[float] = 0.5,
    group_background: Optional[float] = None,
    hyperedge_meta: Optional[Sequence[Dict]] = None,
    hyperedge_types: Optional[Sequence[str]] = None,
    tf_direction_from_construction: bool = True,
    max_group_purity: Optional[float] = None,
) -> pd.DataFrame:
    """
    Rank drivers at the *edge* level, preserving bin direction.

    Unlike ``combine_evidence``, which works on programme-level aggregates, this
    operates on the per-edge tables so that bin identity — and therefore
    direction — survives.  Programmes whose high and low bins are both enriched
    for the same arm are flagged ``bidirectional`` rather than silently reduced
    to one number.

    Parameters
    ----------
    enrichment : DataFrame
        Per-edge output of
        :func:`~pseudoembed.core.hypergraph_interpret.edge_condition_enrichment`.
        Must retain the ``edge_index`` column.
    contributions : DataFrame
        Per-edge output of
        :func:`~pseudoembed.core.hypergraph_interpret.edge_axis_contributions`.
    axis_col : str, optional
        Which ``frac_*`` column is the geometry score.  Defaults to the first
        found; prefer a perturbation axis for condition drivers.
    fdr_threshold : float
        Composition FDR cutoff.
    min_abs_log2 : float
        Minimum |log2_enrich| for a driver call.  Guards against edges that are
        statistically significant on large n but biologically trivial.
    max_group_enrichment : float or None
        Confound threshold on **excess purity over background**, not on absolute
        purity.  An edge is ``confounded=True`` when

        ``(group_purity - group_background) / (1 - group_background)``

        exceeds this: 0 means the edge mixes donors exactly as the dataset does,
        1 means it is a single donor.  The default 0.5 flags an edge that closes
        more than half the remaining gap to complete purity.  ``None`` disables
        the gate.  Only active when ``edge_condition_enrichment`` was given a
        ``group``.

        This replaces an absolute ``group_purity > 0.8`` test, which was unusable
        on donor-imbalanced data: with one donor supplying 89% of cells the
        threshold sat *below* the null and rejected unconfounded edges.  A
        fold-change scale was considered and rejected for the mirror-image
        reason — purity is capped at 1.0, so at background 0.893 no edge can
        exceed 1.12x and a 1.15x threshold would never fire at all.
    group_background : float, optional
        Dataset-wide fraction of cells from the dominant donor — the null above.
        When omitted, the median edge purity is used and a warning is issued;
        pass the true fraction for an exact null.
    hyperedge_meta : sequence of dict, optional
        Per-edge metadata, indexed by ``edge_index``.  This is the *authoritative*
        source for ``bin``/``n_bins``: when metadata is present the edge's display
        name is the bare programme name (``"TGFb"``) with no ``_bin3`` suffix to
        parse, so without this the bin — and therefore the direction — cannot be
        recovered.
    hyperedge_types : sequence of str, optional
        Per-edge type labels, used to parse the bin when ``hyperedge_meta`` is
        absent or lacks it.
    tf_direction_from_construction : bool
        TF programme edges have no activity bin: ``build_tf_hypergraph`` selects
        connected components of cells with ``tf_z > z_score_threshold``, so there
        is no low-activity companion edge to compare against and ``bin`` is NaN
        for every TF edge.  Under the bin-based rule that made all of them
        ``direction="unclear"``, which excluded the entire TF view from the
        driver table by construction.

        But the construction itself supplies the missing fact: **every TF edge is
        a high-activity set by definition.**  With this True, TF edges are
        assigned ``bin_position="high"`` from that, and ``bin_source`` records
        which edges were resolved this way rather than from a real bin.

        This is weaker evidence than a pathway bin, because the two-sided
        consistency check (top bin and bottom bin independently agreeing) is not
        available — it rests on ``log2_enrich`` sign alone.  Set False to restore
        the previous bin-only behaviour.
    max_group_purity : float, optional
        Removed.  Retained only so existing callers raise a directed error rather
        than silently getting a different filter; see ``max_group_enrichment``.

    Returns
    -------
    DataFrame
        One row per edge, ranked by ``driver_score`` descending, with:

        ``programme``, ``edge_name``, ``bin``, ``n_bins``, ``bin_position``
        (``"high"``/``"low"``/``"mid"``/``"na"``), ``bin_source``
        (``"activity_bin"``/``"tf_construction"``/``"none"``), ``log2_enrich``,
        ``fdr``, ``frac_*`` geometry score, ``direction``
        (``"up_in_treated"``/``"down_in_treated"``), ``driver_score``,
        ``group_background``, ``group_enrichment``, ``confounded``,
        ``is_driver``, and ``bidirectional``.

    Notes
    -----
    ``driver_score`` is ``|log2_enrich| * geometry_fraction`` — the product of
    composition and geometry evidence, so an edge must score on both to rank
    highly.  It is a ranking heuristic, not a calibrated statistic; use
    :func:`bootstrap_driver_stability` for confidence.
    """
    # This knob's meaning changed from an absolute dominant-donor fraction to
    # excess purity over background, and the two scales share the 0-1 range --
    # so a stale 0.8 would be silently accepted with a completely different
    # effect.  There is no safe way to guess which was meant; refuse and say so.
    if max_group_purity is not None:
        raise ValueError(
            "`max_group_purity` was removed: it tested the dominant-donor "
            "fraction against an absolute constant, which is unusable on "
            "donor-imbalanced data (with one donor at 89% of cells, the 0.8 "
            "default sat below the null and rejected unconfounded edges). "
            "Use `max_group_enrichment` instead -- excess purity over "
            "background, (purity - background) / (1 - background), 0 at the "
            f"null and 1 at complete purity; default 0.5. Received "
            f"{max_group_purity!r}, which cannot be translated automatically "
            "because both scales span 0-1."
        )

    if axis_col is None:
        cands = [c for c in contributions.columns if c.startswith("frac_")]
        if not cands:
            raise ValueError(
                "`contributions` has no 'frac_*' column; pass axis_col explicitly."
            )
        axis_col = cands[0]

    if (
        "edge_index" not in enrichment.columns
        or "edge_index" not in contributions.columns
    ):
        raise ValueError(
            "Both tables must retain 'edge_index' to join at edge level. "
            "Pass the per-edge outputs, not programme-level aggregates."
        )

    keep_con = ["edge_index", axis_col]
    df = enrichment.merge(contributions[keep_con], on="edge_index", how="left")

    # --- direction ------------------------------------------------------
    # Bin identity, in order of reliability:
    #   1. an explicit 'bin' column
    #   2. hyperedge_meta[edge_index]['bin'] -- authoritative
    #   3. parsing a '_bin{k}' suffix off the type label, then the display name
    # Step 3 is a fallback only: once metadata exists the display name is the
    # bare programme name and carries no bin to parse.
    def _from_source(src, field):
        if src is None:
            return None
        vals = np.full(len(df), np.nan)
        idx = df["edge_index"].to_numpy()
        for r, e_i in enumerate(idx):
            if 0 <= e_i < len(src):
                item = src[e_i]
                if isinstance(item, dict):
                    v = item.get(field)
                    if v is not None:
                        vals[r] = float(v)
                elif field == "bin" and isinstance(item, str):
                    m = pd.Series([item]).str.extract(r"_bin(\d+)$")[0].iloc[0]
                    if pd.notna(m):
                        vals[r] = float(m)
        return vals if np.isfinite(vals).any() else None

    if "bin" not in df.columns:
        got = _from_source(hyperedge_meta, "bin")
        if got is None:
            got = _from_source(hyperedge_types, "bin")
        if got is None:
            got = df["edge_name"].str.extract(r"_bin(\d+)$")[0].astype(float).to_numpy()
        df["bin"] = got

    if "n_bins" not in df.columns:
        got_nb = _from_source(hyperedge_meta, "n_bins")
        if got_nb is not None:
            df["n_bins"] = got_nb
    if "n_bins" not in df.columns:
        # Infer GLOBALLY, not per programme: the binning scheme is shared across
        # pathways, and a programme that only produced its bottom bin (because
        # its other bins fell below min_cells) would otherwise be inferred as
        # n_bins=1 and lose its direction entirely.
        global_n_bins = df["bin"].max() + 1
        if not np.isfinite(global_n_bins) or global_n_bins < 2:
            global_n_bins = np.nan
        df["n_bins"] = global_n_bins

    def _position(row) -> str:
        b, nb = row["bin"], row["n_bins"]
        if not np.isfinite(b) or not np.isfinite(nb) or nb < 2:
            return "na"
        if b >= nb - 1:
            return "high"
        if b <= 0:
            return "low"
        return "mid"

    df["bin_position"] = df.apply(_position, axis=1)
    df["bin_source"] = np.where(df["bin_position"] == "na", "none", "activity_bin")

    # TF edges carry no bin, because build_tf_hypergraph emits only components of
    # cells ABOVE the z-threshold -- there is no low-activity companion edge.
    # That is itself the missing information: such an edge is a high-activity set
    # by construction.  Recording it in `bin_source` keeps the weaker provenance
    # visible, since the two-sided (top bin vs bottom bin) cross-check that backs
    # a pathway call is not available here.
    if tf_direction_from_construction:
        is_tf = df["programme"].astype(str).str.startswith("tf")
        needs = is_tf & (df["bin_position"] == "na")
        if needs.any():
            df.loc[needs, "bin_position"] = "high"
            df.loc[needs, "bin_source"] = "tf_construction"
            print(
                f"  {int(needs.sum())} TF edge(s) assigned bin_position='high' "
                "from construction (high-activity components; no activity bin "
                "exists). See bin_source."
            )

    # A treated-enriched HIGH bin means the programme is up in treated; a
    # treated-enriched LOW bin means it is down.  Enrichment for control
    # inverts both.  See the module docstring table.
    treated_enriched = df["log2_enrich"] > 0
    high = df["bin_position"] == "high"
    low = df["bin_position"] == "low"

    direction = np.full(len(df), "unclear", dtype=object)
    direction[(treated_enriched & high).to_numpy()] = "up_in_treated"
    direction[(treated_enriched & low).to_numpy()] = "down_in_treated"
    direction[(~treated_enriched & high).to_numpy()] = "down_in_treated"
    direction[(~treated_enriched & low).to_numpy()] = "up_in_treated"
    df["direction"] = direction

    # --- score ----------------------------------------------------------
    df["driver_score"] = df["log2_enrich"].abs() * df[axis_col].fillna(0.0)

    # --- confound gate --------------------------------------------------
    # `group_purity` is the fraction of an edge's cells from its single most
    # common donor.  Testing it against a fixed constant is wrong whenever the
    # dataset is itself donor-imbalanced: if one donor supplies 89% of all cells,
    # a perfectly *unconfounded* edge scores 0.89 and a 0.8 threshold rejects it.
    # The threshold sat below the null, so the filter was discarding edges that
    # were at or below background donor mixing.
    #
    # The fix is relative: enrichment of the dominant donor over how often that
    # donor appears in the data at large.  Null is 0; only genuine concentration
    # is flagged.
    #
    # The scale is *excess purity*, not a fold-change:
    #
    #     excess = (purity - background) / (1 - background)
    #
    # A fold-change threshold cannot work here, because purity is capped at 1.0:
    # with background 0.893 the largest attainable ratio is 1/0.893 = 1.12, so a
    # 1.15x threshold could never fire on any edge -- silently disabling the
    # filter, which is the same class of bug in the other direction.  Excess
    # purity is 0 at background and 1 at complete purity whatever the background
    # is, so one threshold means the same thing on a donor-balanced dataset and a
    # 90%-one-donor dataset alike.
    if max_group_enrichment is not None and "group_purity" in df.columns:
        if group_background is None:
            if "group_background" in df.columns:
                bg = df["group_background"].astype(float).to_numpy()
            else:
                # Without the true dataset-wide fraction, the best available
                # reference is the typical edge.  Using the median rather than
                # the mean keeps a handful of genuinely confounded edges from
                # dragging the reference up and masking themselves.
                bg = np.full(len(df), float(df["group_purity"].median()))
                warnings.warn(
                    "No `group_background` supplied — using the median edge "
                    "purity as the reference for donor enrichment. Pass the "
                    "dataset-wide dominant-donor fraction for an exact null.",
                    UserWarning,
                )
        else:
            bg = np.full(len(df), float(group_background))

        bg = np.clip(bg, 0.0, 1.0 - 1e-9)
        df["group_background"] = bg
        purity = df["group_purity"].astype(float).to_numpy()
        df["group_enrichment"] = (purity - bg) / (1.0 - bg)
        df["confounded"] = (
            df["group_enrichment"] > float(max_group_enrichment)
        ).fillna(False)
    else:
        df["confounded"] = False
        if max_group_enrichment is not None:
            warnings.warn(
                "No 'group_purity' column — batch confounding was not screened. "
                "Pass `group=` to edge_condition_enrichment to enable it.",
                UserWarning,
            )

    df["is_driver"] = (
        (df["fdr"] <= fdr_threshold)
        & (df["log2_enrich"].abs() >= min_abs_log2)
        & (~df["confounded"])
        & df["direction"].isin(["up_in_treated", "down_in_treated"])
    ).fillna(False)

    # --- bidirectional programmes --------------------------------------
    drv = df[df["is_driver"]]
    bidir = {p for p, sub in drv.groupby("programme") if sub["direction"].nunique() > 1}
    df["bidirectional"] = df["programme"].isin(bidir)
    if bidir:
        print(
            f"  {len(bidir)} programme(s) are bidirectional (both directions "
            f"significant in different cell populations): {sorted(bidir)}"
        )

    return df.sort_values("driver_score", ascending=False).reset_index(drop=True)


# ============================================================================
# Confidence via bootstrap
# ============================================================================


def bootstrap_driver_stability(
    hyperedges: Sequence[np.ndarray],
    hyperedge_weights: np.ndarray,
    condition: np.ndarray,
    phi: np.ndarray,
    n_cells: int,
    hyperedge_types: Optional[Sequence[str]] = None,
    hyperedge_meta: Optional[Sequence[Dict]] = None,
    n_boot: int = 100,
    top_k: int = 10,
    frac: float = 0.8,
    random_seed: int = 0,
    axis_name: str = "axis0",
) -> pd.DataFrame:
    """
    Bootstrap the driver ranking to get a selection frequency per programme.

    Repeatedly subsamples cells without replacement, restricts the hypergraph to
    the retained cells, recomputes both evidence measures, and records which
    programmes land in the top-``top_k`` by ``driver_score``.  The resulting
    ``selection_freq`` is the confidence measure for the *ranking* — the thing a
    driver list actually asserts — which neither the composition FDR nor the
    point-estimate contribution provides.

    Subsampling is *without* replacement (a 0.8 fraction) rather than a classic
    with-replacement bootstrap: duplicated cell indices would inflate hyperedge
    degrees and distort the operator's normalisation.

    Parameters
    ----------
    hyperedges : sequence of ndarray
        Hyperedges over the full cell set.
    hyperedge_weights : ndarray
        Per-edge weights.
    condition : ndarray, shape (n_cells,)
        Binary condition labels.
    phi : ndarray, shape (n_cells,) or (n_cells, n_axes)
        Spectral axis/axes.  Restricted to the subsample each round rather than
        recomputed: re-running ``eigsh`` per bootstrap would be far slower and
        would conflate axis instability with ranking instability.
    n_cells : int
        Total cells.
    hyperedge_types, hyperedge_meta : optional
        Naming sources.
    n_boot : int
        Bootstrap rounds.  100 is usually enough to separate >0.9 from <0.5.
    top_k : int
        A programme is "selected" in a round if it ranks in the top-k.
    frac : float
        Fraction of cells retained per round.
    random_seed : int
        Seed.
    axis_name : str
        Label for the axis column.

    Returns
    -------
    DataFrame
        ``programme``, ``selection_freq``, ``mean_rank``, ``mean_driver_score``,
        ``n_rounds_present``, sorted by ``selection_freq`` descending.

    Notes
    -----
    An edge is kept in a round only if at least 3 of its members survive the
    subsample; smaller remnants make the contribution term meaningless.
    """
    from pseudoembed.analysis.tf_hypergraph_drug_analysis import (
        build_incidence_matrix,
        compute_degrees,
    )

    rng = np.random.default_rng(random_seed)
    w_full = np.asarray(hyperedge_weights, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if phi.ndim == 1:
        phi = phi[:, None]
    cond = np.asarray(condition)
    if cond.dtype == bool:
        cond = cond.astype(int)

    n_sub = max(int(round(frac * n_cells)), 10)
    counts: Dict[str, int] = {}
    rank_sums: Dict[str, float] = {}
    score_sums: Dict[str, float] = {}
    present: Dict[str, int] = {}

    n_skipped = 0
    for b in range(n_boot):
        idx = rng.choice(n_cells, n_sub, replace=False)
        idx_sorted = np.sort(idx)
        remap = -np.ones(n_cells, dtype=int)
        remap[idx_sorted] = np.arange(n_sub)

        sub_edges: List[np.ndarray] = []
        sub_w: List[float] = []
        sub_types: List[str] = []
        sub_meta: List[Dict] = []
        for e_i, edge in enumerate(hyperedges):
            mapped = remap[np.asarray(edge, dtype=int)]
            mapped = mapped[mapped >= 0]
            if mapped.size < 3:
                continue
            sub_edges.append(mapped)
            sub_w.append(w_full[e_i])
            if hyperedge_types is not None and e_i < len(hyperedge_types):
                sub_types.append(hyperedge_types[e_i])
            if hyperedge_meta is not None and e_i < len(hyperedge_meta):
                sub_meta.append(hyperedge_meta[e_i])

        if len(sub_edges) < 2:
            n_skipped += 1
            continue

        sub_w_arr = np.asarray(sub_w, dtype=float)
        cond_sub = cond[idx_sorted]
        if np.unique(cond_sub).size < 2:
            n_skipped += 1
            continue

        try:
            H_sub = build_incidence_matrix(sub_edges, n_sub)
            dv, de = compute_degrees(H_sub, sub_w_arr)
            dv = np.maximum(dv, 1e-10)

            con = edge_axis_contributions(
                H_sub,
                sub_w_arr,
                phi[idx_sorted],
                node_degrees=dv,
                edge_degrees=de,
                hyperedge_types=sub_types or None,
                hyperedge_meta=sub_meta or None,
                axis_names=(
                    [axis_name] * phi.shape[1]
                    if phi.shape[1] == 1
                    else [f"{axis_name}{k}" for k in range(phi.shape[1])]
                ),
            )
            enr = edge_condition_enrichment(
                sub_edges,
                cond_sub,
                hyperedge_types=sub_types or None,
                hyperedge_meta=sub_meta or None,
                min_cells=5,
            )
        except Exception:
            n_skipped += 1
            continue

        frac_cols = [c for c in con.columns if c.startswith("frac_")]
        merged = enr.merge(con[["edge_index"] + frac_cols], on="edge_index", how="left")
        merged["score"] = merged["log2_enrich"].abs().fillna(0.0) * merged[
            frac_cols
        ].fillna(0.0).sum(axis=1)

        prog = merged.groupby("programme")["score"].sum().sort_values(ascending=False)
        for r, (p, s) in enumerate(prog.items()):
            present[p] = present.get(p, 0) + 1
            rank_sums[p] = rank_sums.get(p, 0.0) + (r + 1)
            score_sums[p] = score_sums.get(p, 0.0) + float(s)
            if r < top_k:
                counts[p] = counts.get(p, 0) + 1

    n_valid = n_boot - n_skipped
    if n_valid == 0:
        raise RuntimeError(
            "Every bootstrap round failed — check that `condition` has both "
            "classes and that hyperedges are large enough to survive subsampling."
        )
    if n_skipped:
        print(f"  Bootstrap: {n_valid}/{n_boot} rounds usable ({n_skipped} skipped)")

    rows = []
    for p, n_present in present.items():
        rows.append(
            {
                "programme": p,
                "selection_freq": counts.get(p, 0) / n_valid,
                "mean_rank": rank_sums[p] / n_present,
                "mean_driver_score": score_sums[p] / n_present,
                "n_rounds_present": n_present,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["selection_freq", "mean_driver_score"], ascending=[False, False]
    ).reset_index(drop=True)


# ============================================================================
# Programme -> gene expansion
# ============================================================================


def expand_programmes_to_genes(
    drivers: pd.DataFrame,
    net: pd.DataFrame,
    top_n_genes: Optional[int] = 100,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str = "weight",
    restrict_to: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Expand pathway programmes to member genes via a prior network.

    .. deprecated::
        The per-gene score this produces is **determined by the prior**, not by
        the data: within a programme it is a monotone function of the footprint
        weight (``pearson == 1.0`` on real output), so all genes share one
        ``log2_enrich``/``fdr``/``driver_score`` and the ranking is unchanged by
        shuffling the condition labels.  Use
        :func:`~pseudoembed.core.hypergraph_gene_attribution.edge_gene_attribution`
        for gene-level claims.

        Kept because it defines the IPA export contract and existing outputs
        reference it.  Its rows carry ``score_source="prior_footprint"`` so files
        produced either way are distinguishable after the fact.

    Only ``pathway:*`` programmes are expanded.  ``tf:*`` programmes are
    returned as-is with their TF symbol as the gene, because IPA treats a TF as
    an upstream regulator in its own right and a regulon dump would both bury
    that and double-count targets shared with pathway footprints.

    Parameters
    ----------
    drivers : DataFrame
        Output of :func:`rank_drivers`.  Uses ``programme``, ``direction``,
        ``log2_enrich``, ``driver_score``, ``fdr``, and any
        ``selection_freq`` merged in from :func:`bootstrap_driver_stability`.
    net : DataFrame
        Prior network, e.g. ``decoupler.op.progeny(organism='human')`` with
        columns ``source`` (pathway), ``target`` (gene symbol), ``weight``.
    top_n_genes : int or None
        Keep this many genes per programme by |weight|.  ``None`` keeps all.
        PROGENy is unfiltered — MAPK carries ~9.5k members against WNT's ~270 —
        so an unfiltered export is dominated by the largest pathways.
    source_col, target_col, weight_col : str
        Column names in ``net``.
    restrict_to : sequence of str, optional
        Only expand these programmes (e.g. the significant ones).

    Returns
    -------
    DataFrame
        One row per (programme, gene): ``programme``, ``programme_type``,
        ``symbol``, ``prior_weight``, ``gene_direction``, ``direction``,
        ``log2_enrich``, ``driver_score``, ``fdr``, ``edge_score``, plus
        ``selection_freq`` when available.

    Notes
    -----
    ``gene_direction`` combines the programme's direction with the sign of the
    gene's prior weight: a negatively-weighted footprint gene moves *opposite*
    to its pathway's activity, so a pathway up in treated implies that gene is
    down.  Collapsing this to the programme direction alone would give IPA the
    wrong sign for a meaningful minority of genes.
    """
    for c in (source_col, target_col, weight_col):
        if c not in net.columns:
            raise ValueError(f"`net` has no column '{c}'.")

    warnings.warn(
        "expand_programmes_to_genes assigns gene scores from the prior "
        "footprint, not from expression: within a programme the ranking is a "
        "monotone function of the prior weight and is invariant to the condition "
        "labels. Rows are tagged score_source='prior_footprint'. Use "
        "hypergraph_gene_attribution.edge_gene_attribution for gene-level "
        "claims.",
        DeprecationWarning,
        stacklevel=2,
    )

    d = drivers.copy()
    if restrict_to is not None:
        d = d[d["programme"].isin(set(restrict_to))]

    # One row per (programme, direction): a bidirectional programme keeps both.
    keep_cols = [
        c
        for c in (
            "programme",
            "direction",
            "log2_enrich",
            "driver_score",
            "fdr",
            "selection_freq",
            "bin_position",
            "bidirectional",
        )
        if c in d.columns
    ]
    d = (
        d[keep_cols]
        .sort_values("driver_score", ascending=False)
        .drop_duplicates(subset=["programme", "direction"])
    )

    rows: List[pd.DataFrame] = []
    for _, r in d.iterrows():
        prog = str(r["programme"])
        ptype, _, pname = prog.partition(":")

        if ptype != "pathway":
            # TFs (and anything else) stay as a single named regulator.
            rows.append(
                pd.DataFrame(
                    {
                        "programme": [prog],
                        "programme_type": [ptype],
                        "symbol": [pname],
                        "prior_weight": [np.nan],
                        **{k: [r[k]] for k in keep_cols if k not in ("programme",)},
                    }
                )
            )
            continue

        members = net[net[source_col] == pname]
        if members.empty:
            warnings.warn(
                f"Programme '{prog}' has no members in `net` — pathway name may "
                f"not match the network's '{source_col}' values. Skipping.",
                UserWarning,
            )
            continue

        members = members.reindex(
            members[weight_col].abs().sort_values(ascending=False).index
        )
        if top_n_genes is not None:
            members = members.head(top_n_genes)

        sub = pd.DataFrame(
            {
                "programme": prog,
                "programme_type": ptype,
                "symbol": members[target_col].astype(str).to_numpy(),
                "prior_weight": members[weight_col].to_numpy(dtype=float),
            }
        )
        for k in keep_cols:
            if k != "programme":
                sub[k] = r[k]
        rows.append(sub)

    if not rows:
        return pd.DataFrame(
            columns=["programme", "programme_type", "symbol", "prior_weight"]
        )

    out = pd.concat(rows, ignore_index=True)

    # A negative prior weight inverts the programme's direction for that gene.
    prog_up = out["direction"].eq("up_in_treated") if "direction" in out else True
    w_neg = out["prior_weight"].fillna(1.0) < 0
    gene_up = np.where(w_neg, ~np.asarray(prog_up), np.asarray(prog_up))
    out["gene_direction"] = np.where(gene_up, "up_in_treated", "down_in_treated")

    # Programme-level magnitude apportioned by normalised prior |weight|.  Since
    # log2_enrich is constant within a programme, this is the prior weight
    # rescaled -- which is why it is prior-determined and self-labelled below.
    if "log2_enrich" in out.columns:
        wabs = out["prior_weight"].abs()
        denom = out.groupby("programme")["prior_weight"].transform(
            lambda s: s.abs().max()
        )
        wnorm = (wabs / denom).fillna(1.0)
        sign = np.where(out["gene_direction"] == "up_in_treated", 1.0, -1.0)
        out["edge_score"] = sign * out["log2_enrich"].abs() * wnorm

    # Literal, so a file read months from now states its own provenance without
    # needing to know which version of the code wrote it.
    out["score_source"] = "prior_footprint"

    return out


# ============================================================================
# Exports
# ============================================================================


def export_ipa_gene_table(
    gene_table: pd.DataFrame,
    path: Union[str, Path],
    score_col: Optional[str] = None,
    include_all: bool = True,
) -> pd.DataFrame:
    """
    Write an IPA-ready gene table.

    IPA's Core Analysis expects a gene identifier column plus optional
    observation columns.  This writes:

    ``Symbol``, ``Expr_Value``, ``Expr_p_value``, ``Programme``,
    ``Programme_Direction``, ``Gene_Direction``, ``Prior_Weight``,
    ``Driver_Score``, ``Score_Source``, ``Prior_Corroboration``,
    ``Selection_Freq``.

    What ``Expr_Value`` means depends on which table was passed, so it is
    labelled per row in ``Score_Source`` rather than left to the reader:

    - ``"attribution"`` — from
      :func:`~pseudoembed.core.hypergraph_gene_attribution.edge_gene_attribution`.
      A within-edge effect size (a difference of log1p means, times a
      specificity factor), so it *is* a directional expression change, measured
      among that edge's cells.
    - ``"prior_footprint"`` — from :func:`expand_programmes_to_genes`.  A
      programme-level enrichment apportioned by prior weight; **not** a fold
      change, and not data-driven within the programme.  Any methods description
      must say so.

    ``Prior_Weight`` and ``Programme`` stay in the same file so provenance is
    never separated from the number.

    Parameters
    ----------
    gene_table : DataFrame
        Output of :func:`expand_programmes_to_genes` or of
        :func:`~pseudoembed.core.hypergraph_gene_attribution.aggregate_gene_attribution`.
    path : str or Path
        Destination ``.csv`` (or ``.txt`` — IPA accepts tab or comma).
        Written with ``.tsv``/``.txt`` as tab-delimited, otherwise comma.
    score_col : str, optional
        Which column to place in ``Expr_Value``.  By default ``gene_score``
        (attribution) when present, else ``edge_score`` (prior footprint), so
        the expression-derived number wins whenever it exists.
    include_all : bool
        Also write a companion ``*_all_genes`` file with no gene cap applied
        upstream, so the cutoff can be varied without re-running.  Only has an
        effect if ``gene_table`` itself is uncapped.

    Returns
    -------
    DataFrame
        The table as written.
    """
    path = Path(path)
    if gene_table.empty:
        warnings.warn("`gene_table` is empty — writing header only.", UserWarning)

    if score_col is None:
        score_col = "gene_score" if "gene_score" in gene_table.columns else "edge_score"
    from_attribution = score_col == "gene_score"

    # Prefer the within-edge FDR when this came from the attribution: the
    # programme-level `fdr` is one number shared by every gene in the programme
    # and would misrepresent per-gene evidence.
    p_col = "fdr_within_edge" if "fdr_within_edge" in gene_table.columns else "fdr"

    out = pd.DataFrame(
        {
            "Symbol": gene_table.get("symbol", pd.Series(dtype=str)),
            "Expr_Value": gene_table.get(score_col, np.nan),
            "Expr_p_value": gene_table.get(p_col, np.nan),
            "Programme": gene_table.get("programme", ""),
            "Programme_Direction": gene_table.get("direction", ""),
            "Gene_Direction": gene_table.get("gene_direction", ""),
            "Prior_Weight": gene_table.get("prior_weight", np.nan),
            "Driver_Score": gene_table.get("driver_score", np.nan),
        }
    )
    if "score_source" in gene_table.columns:
        out["Score_Source"] = gene_table["score_source"].to_numpy()
    else:
        out["Score_Source"] = "attribution" if from_attribution else "prior_footprint"

    # Whether the prior independently lists this gene for this programme.  For an
    # attribution table this is a genuine corroboration flag; for a footprint
    # table it is trivially True and says nothing.
    if "in_prior_regulon" in gene_table.columns:
        out["Prior_Corroboration"] = np.where(
            gene_table["in_prior_regulon"].to_numpy(dtype=bool),
            "in_prior",
            "novel_vs_prior",
        )
    elif not from_attribution:
        out["Prior_Corroboration"] = "in_prior_by_construction"

    if "selection_freq" in gene_table.columns:
        out["Selection_Freq"] = gene_table["selection_freq"].to_numpy()

    # A gene can belong to several programmes; IPA wants one row per identifier.
    # Keep the strongest-|Expr_Value| instance and report the collision count so
    # the collapse is visible rather than silent.
    n_before = len(out)
    out = out.reindex(out["Expr_Value"].abs().sort_values(ascending=False).index)
    dup_mask = out.duplicated(subset=["Symbol"], keep="first")
    n_dup = int(dup_mask.sum())
    out_unique = out[~dup_mask].reset_index(drop=True)
    if n_dup:
        print(
            f"  {n_dup} of {n_before} rows were genes shared across programmes; "
            f"kept the strongest per gene ({len(out_unique)} unique symbols)."
        )

    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    path.parent.mkdir(parents=True, exist_ok=True)
    out_unique.to_csv(path, sep=sep, index=False)
    print(f"✓ IPA gene table written: {path}  ({len(out_unique)} genes)")

    if include_all and n_dup:
        all_path = path.with_name(f"{path.stem}_with_duplicates{path.suffix}")
        out.reset_index(drop=True).to_csv(all_path, sep=sep, index=False)
        print(f"✓ Full table (gene-programme pairs): {all_path}  ({len(out)} rows)")

    return out_unique


def export_driver_tables(
    outdir: Union[str, Path],
    drivers: pd.DataFrame,
    gene_table: Optional[pd.DataFrame] = None,
    stability: Optional[pd.DataFrame] = None,
    enrichment: Optional[pd.DataFrame] = None,
    contributions: Optional[pd.DataFrame] = None,
    prefix: str = "hypergraph",
    attribution: Optional[pd.DataFrame] = None,
    gene_level: Optional[pd.DataFrame] = None,
    tf_level: Optional[pd.DataFrame] = None,
    pathway_level: Optional[pd.DataFrame] = None,
) -> Dict[str, Path]:
    """
    Write the full set of driver tables to ``outdir``.

    Writes both filtered and complete versions so a cutoff can be changed
    without re-running anything:

    ===============================  =======================================
    File                             Contents
    ===============================  =======================================
    ``*_drivers_significant.csv``    edges passing ``is_driver``
    ``*_drivers_all_edges.csv``      every edge, unfiltered
    ``*_genes_ipa.txt``              IPA-ready, tab-delimited
    ``*_genes_all.csv``              every programme-gene pair (prior footprint)
    ``*_stability.csv``              bootstrap selection frequencies
    ``*_edge_enrichment_raw.csv``    raw composition table
    ``*_edge_contributions_raw.csv`` raw spectral table
    ===============================  =======================================

    With ``gene_attribution`` enabled, four more — these are the
    expression-derived tables, and the ones to use for gene-level claims:

    ======================================  ================================
    ``*_gene_attribution_edge_level.csv``   one row per (edge, gene)
    ``*_genes_by_programme.csv``            aggregated to (programme, gene)
    ``*_tf_drivers.csv``                    TF programme summary
    ``*_pathway_drivers.csv``               pathway programme summary
    ======================================  ================================

    Returns
    -------
    dict
        Mapping from label to written path.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    merged = drivers
    if stability is not None and "programme" in stability.columns:
        merged = drivers.merge(
            stability[["programme", "selection_freq", "mean_rank"]],
            on="programme",
            how="left",
        )

    if "is_driver" in merged.columns:
        sig = merged[merged["is_driver"]]
        p = outdir / f"{prefix}_drivers_significant.csv"
        sig.to_csv(p, index=False)
        written["drivers_significant"] = p
        print(f"✓ Significant drivers: {p}  ({len(sig)} edges)")

    p = outdir / f"{prefix}_drivers_all_edges.csv"
    merged.to_csv(p, index=False)
    written["drivers_all"] = p
    print(f"✓ All edges (unfiltered): {p}  ({len(merged)} edges)")

    # The IPA file is the one people actually load, so it gets the
    # expression-derived scores whenever they exist.  The prior-footprint table
    # is still written alongside, under its own name, for comparison.
    ipa_src = (
        gene_level if gene_level is not None and not gene_level.empty else gene_table
    )
    if ipa_src is not None and not ipa_src.empty:
        p = outdir / f"{prefix}_genes_ipa.txt"
        export_ipa_gene_table(ipa_src, p)
        written["genes_ipa"] = p

    if gene_table is not None and not gene_table.empty:
        p = outdir / f"{prefix}_genes_all.csv"
        gene_table.to_csv(p, index=False)
        written["genes_all"] = p
        print(f"✓ All programme-gene pairs: {p}  ({len(gene_table)} rows)")

    for label, table, fname, desc in (
        (
            "gene_attribution_edge_level",
            attribution,
            f"{prefix}_gene_attribution_edge_level.csv",
            "Gene attribution (edge level)",
        ),
        (
            "genes_by_programme",
            gene_level,
            f"{prefix}_genes_by_programme.csv",
            "Genes by programme (expression-derived)",
        ),
        ("tf_drivers", tf_level, f"{prefix}_tf_drivers.csv", "TF drivers"),
        (
            "pathway_drivers",
            pathway_level,
            f"{prefix}_pathway_drivers.csv",
            "Pathway drivers",
        ),
    ):
        if table is not None and not table.empty:
            p = outdir / fname
            table.to_csv(p, index=False)
            written[label] = p
            print(f"✓ {desc}: {p}  ({len(table)} rows)")

    if stability is not None:
        p = outdir / f"{prefix}_stability.csv"
        stability.to_csv(p, index=False)
        written["stability"] = p
        print(f"✓ Bootstrap stability: {p}")

    if enrichment is not None:
        p = outdir / f"{prefix}_edge_enrichment_raw.csv"
        enrichment.to_csv(p, index=False)
        written["enrichment_raw"] = p

    if contributions is not None:
        p = outdir / f"{prefix}_edge_contributions_raw.csv"
        contributions.to_csv(p, index=False)
        written["contributions_raw"] = p

    return written


# ============================================================================
# Plot
# ============================================================================


def plot_driver_summary(
    drivers: pd.DataFrame,
    stability: Optional[pd.DataFrame] = None,
    top_n: int = 20,
    figsize: Tuple[float, float] = (13.0, 6.0),
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Two-panel driver summary: directional effect sizes and ranking confidence.

    Left panel is a diverging bar chart of ``log2_enrich`` for the top
    programmes, coloured by direction, so up- and down-in-treated read at a
    glance.  Right panel plots bootstrap ``selection_freq`` against
    ``driver_score``, which is where an unstable-but-high-scoring programme
    becomes visible — the failure mode a bar chart alone hides.

    Colours are colour-blind safe (blue/orange, not red/green) and direction is
    additionally encoded by which side of zero the bar falls on, so the figure
    does not rely on hue alone.

    Parameters
    ----------
    drivers : DataFrame
        Output of :func:`rank_drivers`.
    stability : DataFrame, optional
        Output of :func:`bootstrap_driver_stability`.  The right panel is
        omitted when absent.
    top_n : int
        Programmes to show.
    figsize : tuple
        Figure size.
    save_path : str or Path, optional
        If given, save here at 300 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    UP, DOWN, MUTED, GRID = "#c2662a", "#3b6fb5", "#9aa3ad", "#d8dde3"

    d = drivers.copy()
    if "is_driver" in d.columns and d["is_driver"].any():
        d = d[d["is_driver"]]
    if d.empty:
        warnings.warn("No drivers to plot.", UserWarning)
        d = drivers.copy()

    # Strongest edge per (programme, direction) so bidirectional pathways show
    # as two bars rather than being collapsed to one.
    d = (
        d.sort_values("driver_score", ascending=False)
        .drop_duplicates(subset=["programme", "direction"])
        .head(top_n)
        .sort_values("log2_enrich")
    )

    n_panels = 2 if stability is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    axes = np.atleast_1d(axes)
    ax = axes[0]

    labels = [
        f"{p.split(':', 1)[-1]}"
        + (f" [{bp}]" if isinstance(bp, str) and bp in ("high", "low") else "")
        for p, bp in zip(d["programme"], d.get("bin_position", [""] * len(d)))
    ]
    colours = [UP if dirn == "up_in_treated" else DOWN for dirn in d["direction"]]
    y = np.arange(len(d))
    ax.barh(y, d["log2_enrich"], color=colours, edgecolor="none", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="#4a5259", lw=1.0)
    ax.set_xlabel("log2 enrichment (treated vs control)", fontsize=10)
    ax.set_title("Directional driver effect size", fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=UP),
        plt.Rectangle((0, 0), 1, 1, color=DOWN),
    ]
    ax.legend(
        handles,
        ["up in treated", "down in treated"],
        fontsize=9,
        frameon=False,
        loc="lower right",
    )

    if stability is not None:
        ax2 = axes[1]
        m = d.merge(stability, on="programme", how="left")
        freq = m["selection_freq"].fillna(0.0)
        ax2.scatter(
            m["driver_score"],
            freq,
            s=46,
            c=[UP if x == "up_in_treated" else DOWN for x in m["direction"]],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax2.axhline(0.8, ls="--", lw=1.0, color=MUTED, zorder=2, label="0.8 stability")
        for _, r in m.iterrows():
            if np.isfinite(r.get("driver_score", np.nan)):
                ax2.annotate(
                    str(r["programme"]).split(":", 1)[-1],
                    (r["driver_score"], r.get("selection_freq", 0.0) or 0.0),
                    fontsize=7.5,
                    color="#3d444b",
                    xytext=(4, 3),
                    textcoords="offset points",
                )
        ax2.set_xlabel("driver score  (|log2 enrich| × geometry)", fontsize=10)
        ax2.set_ylabel("bootstrap selection frequency", fontsize=10)
        ax2.set_ylim(-0.04, 1.04)
        ax2.set_title("Ranking confidence", fontsize=11, loc="left")
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.grid(True, color=GRID, lw=0.7)
        ax2.set_axisbelow(True)
        ax2.legend(fontsize=9, frameon=False, loc="lower right")

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Driver summary figure: {save_path}")
    return fig


# ============================================================================
# End-to-end orchestration
# ============================================================================


def run_driver_analysis(
    adata,
    objects: Dict,
    condition_key: str = "condition",
    control_label: str = "control",
    drug_label: str = "drug",
    output_dir: Optional[Union[str, Path]] = None,
    pseudotime_key: str = "multiview_pseudotime",
    batch_key: Optional[str] = None,
    organism: str = "human",
    views: Sequence[str] = ("pathway", "tf"),
    fdr_threshold: float = 0.05,
    min_abs_log2: float = 0.25,
    max_group_enrichment: Optional[float] = 1.15,
    top_n_genes: Optional[int] = 100,
    n_boot: int = 100,
    top_k: int = 10,
    min_cells: int = 10,
    random_seed: int = 0,
    prefix: str = "hypergraph",
    gene_attribution: bool = True,
    attribution_top_n_genes: Optional[int] = 200,
    attribution_n_perm: int = 0,
    attribution_layer: Optional[str] = None,
    tf_direction_from_construction: bool = True,
) -> Dict:
    """
    Run the full edge-driven driver analysis and write IPA-ready exports.

    Chains the pieces: pool the requested views' hyperedges, score each edge on
    composition (condition enrichment) and geometry (spectral contribution),
    rank directional drivers, bootstrap the ranking for confidence, expand
    significant pathway programmes to genes, and write every table.

    The perturbation axis is chosen empirically as the spectral axis whose cell
    scores best separate the two conditions (largest |point-biserial r|).  This
    matters: attributing condition drivers to a maturation axis would rank edges
    by how much they shape the *differentiation* trajectory, which is a different
    question.

    Parameters
    ----------
    adata : AnnData
        Must carry ``adata.obs[condition_key]`` and the spectral embedding at
        ``adata.obsm[f"{pseudotime_key}_spectral"]``.
    objects : dict
        Third return value of ``run_multiview_hypergraph_analysis``.
    condition_key, control_label, drug_label : str
        Condition column and its two levels.  Cells with other labels are
        dropped from the composition test.
    output_dir : str or Path, optional
        Where to write tables and the figure.  Nothing is written when omitted.
    pseudotime_key : str
        Used to locate ``adata.obsm[f"{pseudotime_key}_spectral"]`` and
        ``adata.uns[f"{pseudotime_key}_axis_labels"]``.
    batch_key : str, optional
        ``adata.obs`` column for batch/patient.  Strongly recommended — without
        it a driver that is really one patient's cells cannot be screened out.
    organism : str
        Passed to ``decoupler`` when fetching PROGENy for gene expansion.
    views : sequence of str
        Which views to pool: any of ``"pathway"``, ``"tf"``, ``"pca"``.  The PCA
        view is excluded by default — its kNN-star edges have no programme
        identity to interpret.
    fdr_threshold, min_abs_log2, max_group_enrichment, tf_direction_from_construction
        Passed to :func:`rank_drivers`.  ``group_background`` is computed here
        from ``batch_key`` rather than passed in, so the confound null is the
        real dataset-wide dominant-donor fraction.
    top_n_genes : int or None
        Genes per pathway in the (prior-based, deprecated) IPA export.
    gene_attribution : bool
        Compute expression-derived gene scores inside each driver edge via
        :func:`~pseudoembed.core.hypergraph_gene_attribution.edge_gene_attribution`.
        This is what makes the gene level a result rather than a prior lookup;
        see that module's docstring.
    attribution_top_n_genes : int or None
        Genes retained per edge in the attribution table.  ``None`` keeps all,
        which is ``n_driver_edges * n_genes`` rows.
    attribution_n_perm : int
        Within-donor permutations for an empirical p-value on the retained
        attribution rows.  0 disables (the normal-approximation FDR is still
        reported).
    attribution_layer : str or None
        Expression layer for the attribution.  ``None`` uses ``adata.X``, which
        this package stores log1p-normalised.
    n_boot, top_k : int
        Bootstrap settings.  ``n_boot=0`` skips the confidence step.
    min_cells : int
        Minimum edge size for the composition test.
    random_seed : int
        Seed for the bootstrap.
    prefix : str
        Filename prefix for outputs.

    Returns
    -------
    dict
        ``drivers``, ``gene_table``, ``stability``, ``enrichment``,
        ``contributions``, ``axis_name``, ``paths``, ``figure``.
    """
    from pseudoembed.analysis.tf_hypergraph_drug_analysis import (
        build_incidence_matrix,
        compute_degrees,
    )

    print("\n" + "=" * 70)
    print("HYPERGRAPH DRIVER ANALYSIS")
    print("=" * 70)

    # ---------------------------------------------------------------- pool
    hyperedges: List[np.ndarray] = []
    weights: List[float] = []
    types: List[str] = []
    meta: List[Dict] = []
    for v in views:
        e = objects.get(f"{v}_hyperedges")
        if not e:
            print(f"  View '{v}' absent or empty — skipping.")
            continue
        w = np.asarray(objects[f"{v}_hyperedge_weights"], dtype=float)
        t = list(objects.get(f"{v}_hyperedge_types") or [])
        m = list(objects.get(f"{v}_hyperedge_meta") or [])
        hyperedges.extend(e)
        weights.extend(w.tolist())
        # Fall back to a synthetic label so index alignment never breaks.
        types.extend(
            t if len(t) == len(e) else [f"{v}_edge_{i}" for i in range(len(e))]
        )
        if len(m) == len(e):
            meta.extend(m)
        else:
            # No metadata for this view (the TF view predates the contract).
            # Emit None rather than a {"view": v} stub: a stub has no 'name', and
            # _edge_names would otherwise be pushed onto its synthetic-label path
            # and lose the real TF names encoded in the type strings.
            meta.extend([None] * len(e))
        print(f"  View '{v}': {len(e)} hyperedges")

    if not hyperedges:
        raise ValueError(
            f"No hyperedges found for views {list(views)} in `objects`. "
            "Check that run_multiview_hypergraph_analysis built those views."
        )
    weights_arr = np.asarray(weights, dtype=float)
    n_cells = int(adata.n_obs)
    print(f"  Pooled: {len(hyperedges)} hyperedges over {n_cells} cells")

    # ----------------------------------------------------------- condition
    if condition_key not in adata.obs:
        raise ValueError(f"adata.obs has no column '{condition_key}'.")
    labels = adata.obs[condition_key].astype(str).to_numpy()
    is_drug = labels == str(drug_label)
    is_ctrl = labels == str(control_label)
    if not is_drug.any() or not is_ctrl.any():
        raise ValueError(
            f"'{condition_key}' must contain both '{control_label}' and "
            f"'{drug_label}'. Found: {sorted(set(labels))[:10]}"
        )
    n_other = int((~is_drug & ~is_ctrl).sum())
    if n_other:
        print(
            f"  {n_other} cells are neither '{control_label}' nor "
            f"'{drug_label}' — excluded from the composition test."
        )
    # Cells outside the two arms are masked to NaN and dropped per edge by
    # restricting the analysis to the two-arm subset.
    keep = is_drug | is_ctrl
    cond_full = np.where(is_drug, 1, 0)

    # -------------------------------------------------------------- axis
    spectral_key = f"{pseudotime_key}_spectral"
    if spectral_key not in adata.obsm:
        raise ValueError(
            f"adata.obsm has no '{spectral_key}'. Run the multiview pseudotime "
            "step first, or pass the matching `pseudotime_key`."
        )
    emb = np.asarray(adata.obsm[spectral_key], dtype=float)
    if emb.ndim == 1:
        emb = emb[:, None]
    axis_labels = list(adata.uns.get(f"{pseudotime_key}_axis_labels", []))
    if len(axis_labels) != emb.shape[1]:
        axis_labels = [f"axis{k}" for k in range(emb.shape[1])]

    # Point-biserial r between each axis and the condition indicator, on the
    # two-arm subset only.
    c = cond_full[keep].astype(float)
    corrs = np.zeros(emb.shape[1])
    for k in range(emb.shape[1]):
        x = emb[keep, k]
        if np.std(x) < 1e-12:
            continue
        corrs[k] = float(np.corrcoef(x, c)[0, 1])
    best = int(np.argmax(np.abs(corrs)))
    axis_name = f"{axis_labels[best]}_ax{best}"
    print(
        f"  Perturbation axis: component {best} "
        f"({axis_labels[best]}), |r| with condition = {abs(corrs[best]):.3f}"
    )
    if abs(corrs[best]) < 0.05:
        warnings.warn(
            f"The best condition-separating axis has |r| = {abs(corrs[best]):.3f}. "
            "No spectral axis meaningfully separates the arms, so the geometry "
            "evidence is near-noise and driver_score is effectively composition "
            "only. Interpret the geometry column with caution.",
            UserWarning,
        )
    phi = emb[:, best]

    # ------------------------------------------------------------ evidence
    print("\n[1/5] Spectral edge contributions...")
    H = build_incidence_matrix(hyperedges, n_cells)
    dv, de = compute_degrees(H, weights_arr)
    dv = np.maximum(dv, 1e-10)
    contributions = edge_axis_contributions(
        H,
        weights_arr,
        phi,
        node_degrees=dv,
        edge_degrees=de,
        hyperedge_types=types,
        hyperedge_meta=meta,
        axis_names=[axis_name],
    )

    print("[2/5] Condition enrichment per edge...")
    group = (
        adata.obs[batch_key].astype(str).to_numpy()
        if batch_key is not None and batch_key in adata.obs
        else None
    )
    if batch_key is not None and group is None:
        warnings.warn(
            f"batch_key='{batch_key}' not found in adata.obs — batch confounding "
            "will not be screened.",
            UserWarning,
        )
    # Restrict to the two arms so cells in other conditions don't dilute the
    # background fraction.
    idx_keep = np.where(keep)[0]
    remap = -np.ones(n_cells, dtype=int)
    remap[idx_keep] = np.arange(idx_keep.size)
    sub_edges, sub_types, sub_meta, sub_w, kept_orig = [], [], [], [], []
    for e_i, edge in enumerate(hyperedges):
        mapped = remap[np.asarray(edge, dtype=int)]
        mapped = mapped[mapped >= 0]
        if mapped.size == 0:
            continue
        sub_edges.append(mapped)
        sub_types.append(types[e_i])
        sub_meta.append(meta[e_i])
        sub_w.append(weights_arr[e_i])
        kept_orig.append(e_i)

    enrichment = edge_condition_enrichment(
        sub_edges,
        cond_full[idx_keep],
        hyperedge_types=sub_types,
        hyperedge_meta=sub_meta,
        group=group[idx_keep] if group is not None else None,
        min_cells=min_cells,
    )
    # Re-key to the pooled edge indices so the join with `contributions` is
    # against the original edge numbering, not the two-arm subset's.
    # MAP the returned indices -- do not assign positionally.
    # edge_condition_enrichment sorts its output by significance, so its row
    # order is not the input edge order; a positional assignment silently pairs
    # each edge's composition with a different edge's geometry.
    _kept = np.asarray(kept_orig, dtype=int)
    enrichment["edge_index"] = _kept[enrichment["edge_index"].to_numpy()]

    print("[3/5] Ranking directional drivers...")
    # The confound null is the dominant donor's share of the *analysed* cells.
    # An absolute purity threshold is meaningless without it: on a dataset where
    # one donor is 89% of cells, every edge looks "impure" by construction.
    group_background = None
    if group is not None:
        gk = group[idx_keep]
        if gk.size:
            _, counts = np.unique(gk, return_counts=True)
            group_background = float(counts.max() / gk.size)
            print(
                f"  Dominant donor is {group_background:.1%} of the two-arm "
                f"cells — confound null set there, not at an absolute purity"
            )

    drivers = rank_drivers(
        enrichment,
        contributions,
        axis_col=f"frac_{axis_name}",
        fdr_threshold=fdr_threshold,
        min_abs_log2=min_abs_log2,
        max_group_enrichment=max_group_enrichment,
        group_background=group_background,
        tf_direction_from_construction=tf_direction_from_construction,
        # enrichment's edge_index is re-keyed to the pooled numbering, so the
        # pooled meta/types line up positionally.
        hyperedge_meta=meta,
        hyperedge_types=types,
    )
    n_sig = int(drivers["is_driver"].sum())
    print(f"  {n_sig} / {len(drivers)} edges pass as drivers")
    if n_sig:
        up = int(
            (drivers.loc[drivers["is_driver"], "direction"] == "up_in_treated").sum()
        )
        print(f"    {up} up in {drug_label}, {n_sig - up} down in {drug_label}")

    # ---------------------------------------------------------- confidence
    stability = None
    if n_boot > 0:
        print(f"[4/5] Bootstrapping driver ranking ({n_boot} rounds)...")
        try:
            stability = bootstrap_driver_stability(
                sub_edges,
                np.asarray(sub_w, dtype=float),
                cond_full[idx_keep],
                phi[idx_keep],
                idx_keep.size,
                hyperedge_types=sub_types,
                hyperedge_meta=sub_meta,
                n_boot=n_boot,
                top_k=top_k,
                random_seed=random_seed,
                axis_name=axis_name,
            )
            n_stable = int((stability["selection_freq"] >= 0.8).sum())
            print(
                f"  {n_stable} programme(s) selected in >=80% of resamples "
                f"(top-{top_k})"
            )
        except Exception as exc:  # keep the rest of the analysis usable
            warnings.warn(
                f"Bootstrap stability failed ({type(exc).__name__}: {exc}). "
                "Driver tables are still written, but without confidence scores.",
                UserWarning,
            )
    else:
        print("[4/5] Bootstrap skipped (n_boot=0) — no confidence scores.")

    # --------------------------------------------------- gene attribution
    attribution = None
    gene_level = None
    tf_level = None
    pathway_level = None
    sig_edges = drivers.loc[drivers["is_driver"], "edge_index"].astype(int).unique()

    if gene_attribution and len(sig_edges):
        print(f"[5/6] Attributing genes within {len(sig_edges)} driver edge(s)...")
        try:
            from pseudoembed.core.hypergraph_gene_attribution import (
                aggregate_gene_attribution,
                build_multilevel_driver_report,
                edge_gene_attribution,
            )

            # `sub_edges` is indexed by position in the two-arm subset, while
            # driver `edge_index` values are in POOLED numbering (they were
            # re-keyed through `kept_orig` above).  Invert that mapping rather
            # than indexing sub_edges with a pooled id, which would silently
            # attribute the wrong cells to the edge.
            pooled_to_sub = {int(o): s for s, o in enumerate(kept_orig)}
            sel_sub, sel_pooled = [], []
            for e_i in sig_edges:
                s = pooled_to_sub.get(int(e_i))
                if s is not None:
                    sel_sub.append(s)
                    sel_pooled.append(int(e_i))

            if not sel_sub:
                warnings.warn(
                    "No driver edge survived the two-arm remapping, so gene "
                    "attribution was skipped.",
                    UserWarning,
                )
            else:
                adata_sub = adata[idx_keep]
                attribution = edge_gene_attribution(
                    adata_sub,
                    [sub_edges[s] for s in sel_sub],
                    cond_full[idx_keep],
                    edge_index=sel_pooled,
                    hyperedge_types=[sub_types[s] for s in sel_sub],
                    hyperedge_meta=[sub_meta[s] for s in sel_sub],
                    group=group[idx_keep] if group is not None else None,
                    layer=attribution_layer,
                    top_n_genes_per_edge=attribution_top_n_genes,
                    n_perm=attribution_n_perm,
                    random_seed=random_seed,
                )

                # Both priors are needed, concatenated.  Programme names are
                # matched against the prior's `source` column, so a PROGENy-only
                # net leaves every `tf:*` programme unmatched and reports a
                # uniform frac_in_prior_regulon == 0 -- which reads as "the TF
                # genes are incoherent" when it only means "wrong prior".
                # CollecTRI has no weight magnitude (just +/-1 mode of
                # regulation), so `weight` is filled to keep one schema.
                net_annot = None
                try:
                    import decoupler as dc

                    nets = []
                    try:
                        nets.append(dc.op.progeny(organism=organism))
                    except Exception:
                        pass
                    try:
                        tf_net = dc.op.collectri(organism=organism)
                        if "weight" not in tf_net.columns:
                            tf_net = tf_net.assign(weight=1.0)
                        nets.append(tf_net)
                    except Exception:
                        pass
                    cols = ["source", "target", "weight"]
                    nets = [n[cols] for n in nets if set(cols).issubset(n.columns)]
                    if nets:
                        net_annot = pd.concat(nets, ignore_index=True)
                except Exception:
                    # Annotation only -- absence costs the in_prior_regulon
                    # column, nothing else.
                    pass

                report = build_multilevel_driver_report(
                    attribution, drivers=drivers, net=net_annot
                )
                gene_level = report["gene"]
                tf_level = report["tf"]
                pathway_level = report["pathway"]
                print(
                    f"  {len(gene_level)} (programme, gene) rows; "
                    f"{len(tf_level)} TF and {len(pathway_level)} pathway "
                    "programmes with attributed genes"
                )
        except Exception as exc:
            warnings.warn(
                f"Gene attribution failed ({type(exc).__name__}: {exc}). "
                "Programme-level tables are still written.",
                UserWarning,
            )
    elif gene_attribution:
        print("[5/6] Gene attribution skipped — no driver edges.")

    # ------------------------------------------------------- gene expansion
    print("[6/6] Expanding pathway programmes to genes (prior footprint)...")
    gene_table = None
    sig = drivers[drivers["is_driver"]]
    if sig.empty:
        warnings.warn(
            "No edges passed the driver thresholds, so no gene table was built. "
            "The unfiltered edge table is still written — lower --fdr-threshold "
            "or --min-abs-log2 to inspect near-misses.",
            UserWarning,
        )
    else:
        with_freq = sig
        if stability is not None:
            with_freq = sig.merge(
                stability[["programme", "selection_freq"]], on="programme", how="left"
            )
        try:
            import decoupler as dc

            net = dc.op.progeny(organism=organism)
            gene_table = expand_programmes_to_genes(
                with_freq, net, top_n_genes=top_n_genes
            )
            n_path = int((gene_table["programme_type"] == "pathway").sum())
            n_tf = int((gene_table["programme_type"] == "tf").sum())
            print(f"  {n_path} pathway genes + {n_tf} TF regulators")
        except Exception as exc:
            warnings.warn(
                f"Could not fetch PROGENy for gene expansion "
                f"({type(exc).__name__}: {exc}). Programme-level tables are "
                "still written; the IPA gene table is not.",
                UserWarning,
            )

    # ------------------------------------------------------------- outputs
    paths: Dict[str, Path] = {}
    figure = None
    if output_dir is not None:
        print()
        paths = export_driver_tables(
            output_dir,
            drivers,
            gene_table=gene_table,
            stability=stability,
            enrichment=enrichment,
            contributions=contributions,
            prefix=prefix,
            attribution=attribution,
            gene_level=gene_level,
            tf_level=tf_level,
            pathway_level=pathway_level,
        )
        try:
            import matplotlib

            matplotlib.use("Agg", force=False)
            figure = plot_driver_summary(
                drivers,
                stability=stability,
                save_path=Path(output_dir) / f"{prefix}_driver_summary.png",
            )
        except Exception as exc:
            warnings.warn(
                f"Driver figure failed ({type(exc).__name__}: {exc}); "
                "tables were written.",
                UserWarning,
            )

    print("\n" + "=" * 70)
    print("DRIVER ANALYSIS COMPLETE")
    print("=" * 70 + "\n")

    return {
        "drivers": drivers,
        "gene_table": gene_table,
        "attribution": attribution,
        "gene_level": gene_level,
        "tf_level": tf_level,
        "pathway_level": pathway_level,
        "group_background": group_background,
        "stability": stability,
        "enrichment": enrichment,
        "contributions": contributions,
        "axis_name": axis_name,
        "axis_index": best,
        "axis_condition_r": float(corrs[best]),
        "paths": paths,
        "figure": figure,
    }
