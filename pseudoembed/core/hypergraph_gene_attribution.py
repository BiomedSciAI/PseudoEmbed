"""
Edge-driven gene attribution
============================

Scores individual genes against the *cells in each hyperedge*, so that a driver
programme can be opened up into the genes that actually move inside it.

Why this module exists
----------------------
:func:`pseudoembed.core.hypergraph_drivers.expand_programmes_to_genes` maps a
programme to genes through the PROGENy/CollecTRI prior alone.  Within a
programme its score is a monotone function of the prior weight — measured on
real output, ``pearson(prior_weight, edge_score) == 1.0`` exactly.  The gene
ranking therefore carries no information from the experiment and would be
identical on shuffled data.  That is a footprint lookup, not a result.

What is computed instead
------------------------
Once you condition on a hyperedge ``e`` (a *cell set*), three distinct
quantities exist, and only the first is available to a plain DE test:

===========================  ==================================================
Global DE                    ``mean(X[treated]) - mean(X[control])``
**Within-edge contrast**     ``mean(X[e & treated]) - mean(X[e & control])``
**Edge specificity**         ``mean(X[e]) - mean(X[cells not in e])``
===========================  ==================================================

The within-edge contrast is the treatment effect *localised to one cell state*:
cells outside the state can neither dilute it nor manufacture it.  Specificity
asks which genes mark this state versus the other states the hypergraph found;
it is direction-free with respect to treatment.

Neither alone is enough.  Within-edge contrast alone is stratified DE, and a
reviewer would rightly say so.  Specificity alone has no direction — a stable
marker of the state scores highly while being unaffected by the drug.  The
product is what requires the hypergraph:

    gene_score = sign(delta) * |z_within| * sqrt(frac_specificity)

so a gene must both move within the state *and* be characteristic of it.

Donor structure
---------------
Cells are not independent replicates, and in the datasets this was built for one
donor can dominate.  ``group=`` enables inverse-variance pooling of *within-donor*
deltas, which removes between-donor offsets from the effect.  The number of
donors that actually contribute is reported as ``n_donors_effective`` rather than
assumed: a donor with cells in only one arm of an edge gets zero weight and drops
out.  Below two contributing donors ``delta_adj`` is NaN and ``testable`` is
False, so a single-donor contrast is never silently presented as a pooled one.

Multiple testing
----------------
``fdr_within_edge`` is Benjamini-Hochberg across genes *within a single edge*,
computed only for edges that were already selected as drivers.  It is therefore
conditional on that selection and is not a family-wise claim over all
(edge, gene) pairs — with ~780 edges and ~16.5k genes that would be 13M tests
whose dependence structure is not remotely exchangeable.  Cells within an edge
are also not independent, so the normal-approximation p-value is
anti-conservative in the same way the edge-level Fisher FDR already is.  Use
``n_perm`` for a within-donor permutation p-value on the reported shortlist when
a calibrated number is needed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import ndtr

from pseudoembed.core.hypergraph_interpret import _bh_fdr, _edge_names

__all__ = [
    "edge_gene_attribution",
    "aggregate_gene_attribution",
    "build_multilevel_driver_report",
]


# ============================================================================
# Expression access
# ============================================================================


def _get_expression(adata, layer: Optional[str] = None) -> sparse.csr_matrix:
    """
    Fetch the expression matrix as CSR float64.

    ``layer=None`` gives ``adata.X``, which by this package's convention
    (``pseudoembed/core/io.py``) is library-size normalised and log1p
    transformed.  ``layer="counts"`` gives raw counts.  The moment arithmetic
    below assumes an additive scale, so the log-normalised default is the
    correct one; raw counts are offered for callers who have already
    variance-stabilised some other way.

    float64 rather than float32: ``Q/n - m**2`` is a catastrophic-cancellation
    shape, and in float32 it goes negative often enough to matter on genes whose
    variance is small relative to their mean.
    """
    mat = adata.layers[layer] if layer is not None else adata.X
    if mat is None:
        raise ValueError(
            f"No expression matrix found ({'layer ' + layer if layer else 'adata.X'})."
        )
    if not sparse.issparse(mat):
        mat = sparse.csr_matrix(mat)
    return mat.tocsr().astype(np.float64)


def _variance_from_moments(
    sum_x: np.ndarray, sum_x2: np.ndarray, n: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean and (population) variance from raw moments, clamped at zero.

    ``E[x^2] - E[x]^2`` can land slightly below zero through floating-point
    cancellation on near-constant genes.  Clamping is safe here because the true
    variance is non-negative by definition; the alternative (propagating a tiny
    negative into a sqrt) produces NaNs that then silently drop genes.
    """
    n_safe = np.where(n > 0, n, np.nan)
    mean = sum_x / n_safe
    var = sum_x2 / n_safe - mean**2
    return mean, np.maximum(var, 0.0)


# ============================================================================
# Core attribution
# ============================================================================


def edge_gene_attribution(
    adata,
    hyperedges: Sequence[np.ndarray],
    condition: np.ndarray,
    edge_index: Optional[Sequence[int]] = None,
    hyperedge_types: Optional[Sequence[str]] = None,
    hyperedge_meta: Optional[Sequence[Dict]] = None,
    group: Optional[np.ndarray] = None,
    layer: Optional[str] = None,
    min_cells_per_arm: int = 10,
    min_mean_expression: float = 0.05,
    min_cells_expressing: int = 3,
    top_n_genes_per_edge: Optional[int] = 200,
    specificity_background: str = "other_edges",
    kappa: Optional[float] = None,
    n_perm: int = 0,
    mem_budget_bytes: int = 4 * 1024**3,
    random_seed: int = 0,
    eps: float = 1e-9,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Score every gene inside every hyperedge, from expression.

    Parameters
    ----------
    adata : AnnData
        Cells must be in the same order as the indices in ``hyperedges``.
    hyperedges : sequence of ndarray
        Cell indices per edge, as built by the view builders.
    condition : ndarray
        Binary indicator, length ``adata.n_obs``: 1 = treated, 0 = control.
    edge_index : sequence of int, optional
        Pooled edge indices for the supplied edges, so the result joins back to
        ``rank_drivers`` output.  Defaults to ``range(len(hyperedges))``.
    hyperedge_types, hyperedge_meta : optional
        Naming sources, as in :func:`edge_condition_enrichment`.  Must be
        indexed by *position within* ``hyperedges``, not by pooled index.
    group : ndarray, optional
        Donor/batch label per cell.  Enables ``delta_adj``.
    layer : str, optional
        Expression layer; ``None`` uses ``adata.X`` (log-normalised).
    min_cells_per_arm : int
        Edges with fewer treated or control cells than this are skipped —
        a Welch statistic on 3 cells is noise.
    min_mean_expression, min_cells_expressing : float, int
        Gene filters applied *within the edge*.
    top_n_genes_per_edge : int, optional
        Keep only this many genes per edge by ``|gene_score|``.  ``None`` keeps
        all, which is ``n_edges * n_genes`` rows.
    specificity_background : {"other_edges", "all_cells"}
        Comparison set for specificity.  ``"other_edges"`` compares against
        cells in *other* edges, which is the hypergraph-native contrast;
        ``"all_cells"`` compares against every non-member cell.
    kappa : float, optional
        Soft-gate scale for specificity.  Defaults to the median ``|spec_z|``,
        which keeps the gate scale-free across datasets.
    n_perm : int
        Within-donor permutations for an empirical p-value on the retained
        shortlist.  0 disables.
    mem_budget_bytes : int
        Edges are processed in chunks so the dense (edges x genes) blocks stay
        under this.
    random_seed : int
        Seeds the permutation null.
    verbose : bool
        Print progress and the cost projection.

    Returns
    -------
    DataFrame
        One row per (edge, gene) surviving the filters, sorted by
        ``|gene_score|`` descending.  See the module docstring for the
        interpretation of ``fdr_within_edge``.
    """
    if specificity_background not in ("other_edges", "all_cells"):
        raise ValueError(
            "specificity_background must be 'other_edges' or 'all_cells', "
            f"got {specificity_background!r}."
        )

    n_cells = int(adata.n_obs)
    cond = np.asarray(condition).reshape(-1)
    if cond.size != n_cells:
        raise ValueError(
            f"condition has {cond.size} entries but adata has {n_cells} cells."
        )
    cond = (cond > 0).astype(np.float64)

    gene_names = np.asarray(adata.var_names, dtype=object)
    X = _get_expression(adata, layer)
    n_genes = X.shape[1]

    if group is not None:
        group_arr = np.asarray(group).reshape(-1)
        if group_arr.size != n_cells:
            raise ValueError(
                f"group has {group_arr.size} entries but adata has {n_cells} cells."
            )
    else:
        group_arr = None

    edge_ids = (
        np.arange(len(hyperedges), dtype=int)
        if edge_index is None
        else np.asarray(edge_index, dtype=int)
    )
    if edge_ids.size != len(hyperedges):
        raise ValueError(
            f"edge_index has {edge_ids.size} entries but {len(hyperedges)} edges "
            "were supplied."
        )

    names, programmes = _edge_names(
        list(hyperedge_types or []), hyperedge_meta, len(hyperedges)
    )

    # ---- incidence over the supplied edges ----------------------------------
    # Imported lazily to match how hypergraph_drivers does it, keeping the
    # analysis->core import direction one-way.
    from pseudoembed.analysis.tf_hypergraph_drug_analysis import build_incidence_matrix

    H = build_incidence_matrix([np.asarray(e, dtype=int) for e in hyperedges], n_cells)
    H = H.astype(np.float64).tocsc()
    n_edges = H.shape[1]

    X2 = X.multiply(X).tocsr()

    # ---- global (edge-independent) totals for the specificity background ----
    if specificity_background == "other_edges":
        in_any = np.asarray(H.sum(axis=1)).ravel() > 0
    else:
        in_any = np.ones(n_cells, dtype=bool)
    bg_rows = np.where(in_any)[0]
    n_bg = int(bg_rows.size)
    tot_S = np.asarray(X[bg_rows].sum(axis=0)).ravel()
    tot_Q = np.asarray(X2[bg_rows].sum(axis=0)).ravel()

    # Global per-gene detection, for the expression filter.
    global_mean = np.asarray(X.mean(axis=0)).ravel()

    # ---- chunking ----------------------------------------------------------
    # Each chunk materialises a handful of dense (chunk x genes) float64 blocks.
    # Budget for ~8 of them so the peak stays inside mem_budget_bytes.
    per_edge_bytes = n_genes * 8 * 8
    chunk = max(1, min(n_edges, int(mem_budget_bytes // max(per_edge_bytes, 1))))
    n_chunks = int(np.ceil(n_edges / chunk))

    if verbose:
        print(
            f"  Gene attribution: {n_edges} edges x {n_genes} genes "
            f"in {n_chunks} chunk(s) of <={chunk}"
        )
        if group_arr is not None:
            print(f"  Donor stratification over {len(np.unique(group_arr))} group(s)")

    donors = np.unique(group_arr) if group_arr is not None else None

    frames: List[pd.DataFrame] = []
    n_skipped_arm = 0

    for c0 in range(0, n_edges, chunk):
        c1 = min(c0 + chunk, n_edges)
        Hc = H[:, c0:c1].tocsr()

        n_e = np.asarray(Hc.sum(axis=0)).ravel()
        H1 = Hc.multiply(cond[:, None]).tocsr()
        H0 = Hc.multiply((1.0 - cond)[:, None]).tocsr()
        n1 = np.asarray(H1.sum(axis=0)).ravel()
        n0 = np.asarray(H0.sum(axis=0)).ravel()

        ok_arm = (n1 >= min_cells_per_arm) & (n0 >= min_cells_per_arm)
        n_skipped_arm += int((~ok_arm).sum())
        if not ok_arm.any():
            continue

        # Four moment blocks -> Welch within edge.  H1.T @ X is (chunk, genes).
        S1 = np.asarray((H1.T @ X).todense(), dtype=np.float64)
        S0 = np.asarray((H0.T @ X).todense(), dtype=np.float64)
        Q1 = np.asarray((H1.T @ X2).todense(), dtype=np.float64)
        Q0 = np.asarray((H0.T @ X2).todense(), dtype=np.float64)

        m1, v1 = _variance_from_moments(S1, Q1, n1[:, None])
        m0, v0 = _variance_from_moments(S0, Q0, n0[:, None])

        delta = m1 - m0
        se = np.sqrt(v1 / n1[:, None] + v0 / n0[:, None])
        z_within = delta / np.maximum(se, eps)

        # Specificity: member vs background, complement by subtraction so no
        # second pass over X is needed.  A cell in several edges is counted once
        # in tot_S, so S_out really is "cells not in this edge".
        S_e = np.asarray((Hc.T @ X).todense(), dtype=np.float64)
        Q_e = np.asarray((Hc.T @ X2).todense(), dtype=np.float64)
        mu_e, var_e = _variance_from_moments(S_e, Q_e, n_e[:, None])
        n_out = np.maximum(n_bg - n_e, 0)
        mu_out, var_out = _variance_from_moments(
            tot_S[None, :] - S_e, tot_Q[None, :] - Q_e, n_out[:, None]
        )
        se_spec = np.sqrt(
            var_e / np.maximum(n_e[:, None], 1)
            + var_out / np.maximum(n_out[:, None], 1)
        )
        spec_delta = mu_e - mu_out
        spec_z = spec_delta / np.maximum(se_spec, eps)

        # Detection filter, within the edge.
        n_expressing = np.asarray((Hc.T @ (X > 0)).todense(), dtype=np.float64)

        # ---- donor-stratified pooling -------------------------------------
        if donors is not None and donors.size >= 2:
            num = np.zeros_like(delta)
            den = np.zeros_like(delta)
            n_don_eff = np.zeros_like(delta)
            for d in donors:
                dm = (group_arr == d).astype(np.float64)
                Hd1 = Hc.multiply((cond * dm)[:, None]).tocsr()
                Hd0 = Hc.multiply(((1.0 - cond) * dm)[:, None]).tocsr()
                nd1 = np.asarray(Hd1.sum(axis=0)).ravel()
                nd0 = np.asarray(Hd0.sum(axis=0)).ravel()
                # A donor present in only one arm of this edge carries no
                # within-donor contrast; zero weight drops it out entirely.
                usable = (nd1 >= min_cells_per_arm) & (nd0 >= min_cells_per_arm)
                if not usable.any():
                    continue
                Sd1 = np.asarray((Hd1.T @ X).todense(), dtype=np.float64)
                Sd0 = np.asarray((Hd0.T @ X).todense(), dtype=np.float64)
                Qd1 = np.asarray((Hd1.T @ X2).todense(), dtype=np.float64)
                Qd0 = np.asarray((Hd0.T @ X2).todense(), dtype=np.float64)
                md1, vd1 = _variance_from_moments(Sd1, Qd1, nd1[:, None])
                md0, vd0 = _variance_from_moments(Sd0, Qd0, nd0[:, None])
                dd = md1 - md0
                vv = vd1 / np.maximum(nd1[:, None], 1) + vd0 / np.maximum(
                    nd0[:, None], 1
                )
                w = np.where(usable[:, None] & (vv > 0), 1.0 / np.maximum(vv, eps), 0.0)
                w = np.where(np.isfinite(dd), w, 0.0)
                num += w * np.nan_to_num(dd)
                den += w
                n_don_eff += (w > 0).astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                delta_adj = np.where(den > 0, num / den, np.nan)
                se_adj = np.where(den > 0, np.sqrt(1.0 / np.maximum(den, eps)), np.nan)
            z_adj = delta_adj / np.maximum(se_adj, eps)
            # Fewer than two contributing donors is not a pooled estimate.
            too_few = n_don_eff < 2
            delta_adj = np.where(too_few, np.nan, delta_adj)
            z_adj = np.where(too_few, np.nan, z_adj)
            se_adj = np.where(too_few, np.nan, se_adj)
        else:
            delta_adj = np.full_like(delta, np.nan)
            se_adj = np.full_like(delta, np.nan)
            z_adj = np.full_like(delta, np.nan)
            n_don_eff = np.zeros_like(delta)

        # ---- assemble long rows -------------------------------------------
        for j in np.where(ok_arm)[0]:
            pos = c0 + j
            keep = (
                (global_mean >= min_mean_expression)
                & (n_expressing[j] >= min_cells_expressing)
                & np.isfinite(z_within[j])
            )
            if not keep.any():
                continue
            gi = np.where(keep)[0]

            use_adj = np.isfinite(z_adj[j, gi])
            z_used = np.where(use_adj, z_adj[j, gi], z_within[j, gi])
            d_used = np.where(use_adj, delta_adj[j, gi], delta[j, gi])

            frames.append(
                pd.DataFrame(
                    {
                        "edge_index": edge_ids[pos],
                        "edge_name": names[pos],
                        "programme": programmes[pos],
                        "symbol": gene_names[gi],
                        "n_cells_edge": n_e[j],
                        "n_treated_edge": n1[j],
                        "n_control_edge": n0[j],
                        "mean_treated": m1[j, gi],
                        "mean_control": m0[j, gi],
                        "delta": delta[j, gi],
                        "se_delta": se[j, gi],
                        "z_within": z_within[j, gi],
                        "delta_adj": delta_adj[j, gi],
                        "se_delta_adj": se_adj[j, gi],
                        "z_adj": z_adj[j, gi],
                        "n_donors_effective": n_don_eff[j, gi],
                        "spec_delta": spec_delta[j, gi],
                        "spec_z": spec_z[j, gi],
                        "delta_used": d_used,
                        "z_used": z_used,
                        "effect_source": np.where(
                            use_adj, "donor_adjusted", "raw_within_edge"
                        ),
                    }
                )
            )

    if not frames:
        if verbose:
            print(
                f"  No edges yielded gene attribution "
                f"({n_skipped_arm} skipped for arm size < {min_cells_per_arm})."
            )
        return _empty_attribution(n_edges, n_genes, layer, specificity_background)

    out = pd.concat(frames, ignore_index=True)

    # ---- specificity soft gate ------------------------------------------
    # Scaled by the median |spec_z| so the gate is dimensionless and does not
    # swamp the effect term (|spec_z| grows with the background size, which is
    # ~n_cells, and would otherwise dominate the product).
    abs_spec = out["spec_z"].abs().to_numpy()
    if kappa is None:
        finite = abs_spec[np.isfinite(abs_spec)]
        kappa_used = float(np.median(finite)) if finite.size else 1.0
        if not np.isfinite(kappa_used) or kappa_used <= 0:
            kappa_used = 1.0
    else:
        kappa_used = float(kappa)
    out["frac_specificity"] = abs_spec / (abs_spec + kappa_used)

    out["gene_score"] = (
        np.sign(out["delta_used"].to_numpy())
        * out["z_used"].abs().to_numpy()
        * np.sqrt(out["frac_specificity"].to_numpy())
    )
    out["gene_direction"] = np.where(
        out["delta_used"].to_numpy() > 0, "up_in_treated", "down_in_treated"
    )

    # ---- p-values, BH within each edge ----------------------------------
    z = out["z_used"].abs().to_numpy()
    out["p_within"] = 2.0 * (1.0 - ndtr(z))
    out["fdr_within_edge"] = out.groupby("edge_index")["p_within"].transform(
        lambda s: pd.Series(_bh_fdr(s.to_numpy()), index=s.index)
    )
    out["testable"] = (
        (out["n_donors_effective"] >= 2) if group_arr is not None else True
    )

    # ---- per-edge truncation --------------------------------------------
    out = out.reindex(
        out["gene_score"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
    if top_n_genes_per_edge is not None:
        out = (
            out.groupby("edge_index", group_keys=False)
            .head(top_n_genes_per_edge)
            .reset_index(drop=True)
        )

    if n_perm > 0:
        out = _add_permutation_pvalues(
            out,
            adata,
            hyperedges,
            edge_ids,
            cond,
            group_arr,
            X,
            layer,
            n_perm=n_perm,
            random_seed=random_seed,
            verbose=verbose,
        )

    out.attrs.update(
        {
            "n_edges_tested": int(n_edges - n_skipped_arm),
            "n_edges_skipped_arm_size": int(n_skipped_arm),
            "n_genes_tested": int(n_genes),
            "layer": layer if layer is not None else "X",
            "specificity_background": specificity_background,
            "kappa": kappa_used,
            "min_cells_per_arm": int(min_cells_per_arm),
            "n_perm": int(n_perm),
        }
    )

    if verbose:
        n_adj = int((out["effect_source"] == "donor_adjusted").sum())
        print(
            f"  {len(out)} (edge, gene) rows over "
            f"{out['edge_index'].nunique()} edge(s); "
            f"{n_adj} donor-adjusted, {len(out) - n_adj} raw"
        )
        if n_skipped_arm:
            print(
                f"  {n_skipped_arm} edge(s) skipped: fewer than "
                f"{min_cells_per_arm} cells in an arm"
            )

    return out


def _empty_attribution(
    n_edges: int, n_genes: int, layer: Optional[str], background: str
) -> pd.DataFrame:
    """Empty frame with the full column contract, so callers need no None check."""
    cols = [
        "edge_index",
        "edge_name",
        "programme",
        "symbol",
        "n_cells_edge",
        "n_treated_edge",
        "n_control_edge",
        "mean_treated",
        "mean_control",
        "delta",
        "se_delta",
        "z_within",
        "delta_adj",
        "se_delta_adj",
        "z_adj",
        "n_donors_effective",
        "spec_delta",
        "spec_z",
        "delta_used",
        "z_used",
        "effect_source",
        "frac_specificity",
        "gene_score",
        "gene_direction",
        "p_within",
        "fdr_within_edge",
        "testable",
    ]
    df = pd.DataFrame({c: pd.Series(dtype=float) for c in cols})
    for c in ("edge_name", "programme", "symbol", "effect_source", "gene_direction"):
        df[c] = pd.Series(dtype=object)
    df.attrs.update(
        {
            "n_edges_tested": 0,
            "n_edges_skipped_arm_size": int(n_edges),
            "n_genes_tested": int(n_genes),
            "layer": layer if layer is not None else "X",
            "specificity_background": background,
            "kappa": np.nan,
            "n_perm": 0,
        }
    )
    return df


def _add_permutation_pvalues(
    out: pd.DataFrame,
    adata,
    hyperedges: Sequence[np.ndarray],
    edge_ids: np.ndarray,
    cond: np.ndarray,
    group_arr: Optional[np.ndarray],
    X: sparse.csr_matrix,
    layer: Optional[str],
    n_perm: int,
    random_seed: int,
    verbose: bool,
) -> pd.DataFrame:
    """
    Empirical p-value on the retained shortlist, permuting condition.

    Permutation is *within donor* when ``group`` was supplied, so the donor
    structure that ``delta_adj`` corrects for is preserved under the null rather
    than being destroyed by it — otherwise the null would be a mixture of
    treatment and donor effects and the p-value would be optimistic.

    Only the rows already retained are refined, which keeps this affordable; it
    is not a screen over all genes.
    """
    rng = np.random.default_rng(random_seed)
    pos_of = {int(e): i for i, e in enumerate(edge_ids)}

    counts = np.zeros(len(out), dtype=float)
    obs_abs = out["delta_used"].abs().to_numpy()
    row_edge = out["edge_index"].to_numpy()
    row_gene = {s: i for i, s in enumerate(np.asarray(adata.var_names, dtype=object))}
    gene_pos = np.array([row_gene[s] for s in out["symbol"]], dtype=int)

    if verbose:
        print(f"  Permutation null: {n_perm} rounds over {len(out)} rows")

    for _ in range(n_perm):
        if group_arr is not None:
            c_perm = cond.copy()
            for d in np.unique(group_arr):
                m = np.where(group_arr == d)[0]
                c_perm[m] = rng.permutation(cond[m])
        else:
            c_perm = rng.permutation(cond)

        for e_id in np.unique(row_edge):
            cells = np.asarray(hyperedges[pos_of[int(e_id)]], dtype=int)
            sel = row_edge == e_id
            gi = gene_pos[sel]
            ct = c_perm[cells] > 0
            if ct.sum() < 1 or (~ct).sum() < 1:
                continue
            sub = X[cells][:, gi]
            m1 = np.asarray(sub[ct].mean(axis=0)).ravel()
            m0 = np.asarray(sub[~ct].mean(axis=0)).ravel()
            counts[sel] += (np.abs(m1 - m0) >= obs_abs[sel]).astype(float)

    out = out.copy()
    out["p_perm"] = (counts + 1.0) / (n_perm + 1.0)
    return out


# ============================================================================
# Aggregation
# ============================================================================


def aggregate_gene_attribution(
    attribution: pd.DataFrame,
    level: str = "programme",
    weight_col: str = "n_cells_edge",
    min_edges: int = 1,
) -> pd.DataFrame:
    """
    Roll (edge, gene) attribution up to (programme, gene).

    Deliberately different from
    :func:`pseudoembed.core.hypergraph_interpret.aggregate_to_programme`, which
    keeps the single most extreme *edge*.  That rule exists to protect
    bidirectional pathway biology at edge level.  At gene level an extreme value
    from one small TF component is more likely noise than signal, so effects are
    combined as cell-count-weighted means with the sign preserved.

    Disagreement is reported rather than averaged away: ``sign_consistency`` is
    the weighted fraction of supporting edges agreeing with the pooled sign, and
    ``direction_conflict`` flags genes below 0.6.  A gene up in one component and
    down in another averages toward zero, and without the flag that is
    indistinguishable from a gene with no effect.

    Parameters
    ----------
    attribution : DataFrame
        Output of :func:`edge_gene_attribution`.
    level : {"programme", "edge"}
        ``"edge"`` returns the input unchanged (with a stable column order).
    weight_col : str
        Column supplying the weights; cell count by default.
    min_edges : int
        Drop (programme, gene) pairs supported by fewer edges than this.
    """
    if level not in ("programme", "edge"):
        raise ValueError(f"level must be 'programme' or 'edge', got {level!r}.")
    if level == "edge" or attribution.empty:
        return attribution.reset_index(drop=True)

    df = attribution.copy()
    w = df[weight_col].astype(float).to_numpy()
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    df["_w"] = w

    def _agg(sub: pd.DataFrame) -> pd.Series:
        ww = sub["_w"].to_numpy()
        tot = ww.sum()
        if tot <= 0:
            ww = np.ones(len(sub))
            tot = float(len(sub))

        def wmean(col: str) -> float:
            v = sub[col].to_numpy(dtype=float)
            m = np.isfinite(v)
            return float(np.sum(v[m] * ww[m]) / np.sum(ww[m])) if m.any() else np.nan

        d = wmean("delta_used")
        score = wmean("gene_score")
        sgn = np.sign(sub["delta_used"].to_numpy(dtype=float))
        agree = np.isfinite(sgn) & (sgn == np.sign(d)) & (sgn != 0)
        consistency = (
            float(np.sum(ww[agree]) / tot) if np.isfinite(d) and d != 0 else np.nan
        )
        return pd.Series(
            {
                "delta_used": d,
                "z_used": wmean("z_used"),
                "gene_score": score,
                "spec_z": wmean("spec_z"),
                "frac_specificity": wmean("frac_specificity"),
                "n_edges_supporting": int(len(sub)),
                "n_edges_concordant": int(agree.sum()),
                "sign_consistency": consistency,
                "fdr_within_edge": (
                    float(np.nanmin(sub["fdr_within_edge"].to_numpy(dtype=float)))
                    if sub["fdr_within_edge"].notna().any()
                    else np.nan
                ),
                "n_cells_total": float(sub["n_cells_edge"].sum()),
                "n_donors_effective": (
                    float(np.nanmax(sub["n_donors_effective"].to_numpy(dtype=float)))
                    if sub["n_donors_effective"].notna().any()
                    else np.nan
                ),
                "any_testable": (
                    bool(sub["testable"].any()) if "testable" in sub.columns else True
                ),
            }
        )

    out = (
        df.groupby(["programme", "symbol"], sort=False)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    out = out[out["n_edges_supporting"] >= min_edges].copy()
    out["gene_direction"] = np.where(
        out["delta_used"] > 0, "up_in_treated", "down_in_treated"
    )
    out["direction_conflict"] = out["sign_consistency"] < 0.6
    out["programme_type"] = out["programme"].astype(str).str.split(":").str[0]

    out = out.reindex(
        out["gene_score"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
    out.attrs.update(dict(attribution.attrs))
    return out


# ============================================================================
# Three-level report
# ============================================================================


def build_multilevel_driver_report(
    attribution: pd.DataFrame,
    drivers: Optional[pd.DataFrame] = None,
    net: Optional[pd.DataFrame] = None,
    prior_source_col: str = "source",
    prior_target_col: str = "target",
    prior_weight_col: str = "weight",
    top_genes_per_programme: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Gene-, TF- and pathway-level driver tables from one attribution.

    The prior network is used **only as annotation**, never as a filter — the
    point of the attribution is to find genes the prior does not list.  The
    resulting ``frac_in_prior_regulon`` per programme is a useful diagnostic:
    it should be *elevated but well below 1*.  Near 1 means the prior has leaked
    back into the ranking; at genome background it means the programme's gene
    assignment is incoherent.

    Returns
    -------
    dict
        ``{"gene": ..., "tf": ..., "pathway": ...}``.
    """
    gene = aggregate_gene_attribution(attribution, level="programme")

    if net is not None and len(net) and not gene.empty:
        prior = net[[prior_source_col, prior_target_col, prior_weight_col]].copy()
        prior.columns = ["_src", "symbol", "prior_weight"]
        prior["_key"] = prior["_src"].astype(str)
        gene["_key"] = gene["programme"].astype(str).str.split(":").str[1]
        gene = gene.merge(
            prior[["_key", "symbol", "prior_weight"]].drop_duplicates(
                subset=["_key", "symbol"]
            ),
            on=["_key", "symbol"],
            how="left",
        ).drop(columns=["_key"])
        gene["in_prior_regulon"] = gene["prior_weight"].notna()
    else:
        gene["prior_weight"] = np.nan
        gene["in_prior_regulon"] = False

    if gene.empty:
        empty = pd.DataFrame()
        return {"gene": gene, "tf": empty, "pathway": empty}

    def _programme_summary(sub: pd.DataFrame) -> pd.Series:
        top = sub.reindex(sub["gene_score"].abs().sort_values(ascending=False).index)
        return pd.Series(
            {
                "n_genes_attributed": int(len(sub)),
                "n_up": int((sub["gene_direction"] == "up_in_treated").sum()),
                "n_down": int((sub["gene_direction"] == "down_in_treated").sum()),
                "mean_abs_z": float(sub["z_used"].abs().mean()),
                "max_abs_gene_score": float(sub["gene_score"].abs().max()),
                "frac_in_prior_regulon": float(sub["in_prior_regulon"].mean()),
                "n_direction_conflict": int(sub["direction_conflict"].sum()),
                "top_genes": ", ".join(
                    top["symbol"].head(top_genes_per_programme).astype(str)
                ),
            }
        )

    summary = (
        gene.groupby("programme", sort=False)
        .apply(_programme_summary, include_groups=False)
        .reset_index()
    )
    summary["programme_type"] = summary["programme"].astype(str).str.split(":").str[0]

    if drivers is not None and len(drivers):
        cols = [
            c
            for c in (
                "programme",
                "direction",
                "log2_enrich",
                "driver_score",
                "fdr",
                "group_purity",
                "group_enrichment",
                "bin_position",
                "bin_source",
                "selection_freq",
                "bidirectional",
                "view_weight",
            )
            if c in drivers.columns
        ]
        # One row per programme: a programme spans several edges, so collapse on
        # the strongest before merging or the join would fan out.
        d = drivers[cols].copy()
        if "driver_score" in d.columns:
            d = d.reindex(
                d["driver_score"].abs().sort_values(ascending=False).index
            ).drop_duplicates(subset=["programme"])
        else:
            d = d.drop_duplicates(subset=["programme"])
        summary = summary.merge(d, on="programme", how="left")

    tf = summary[summary["programme_type"].str.startswith("tf")].reset_index(drop=True)
    pathway = summary[summary["programme_type"] == "pathway"].reset_index(drop=True)

    return {"gene": gene, "tf": tf, "pathway": pathway}


def plot_programme_drivers(
    summary: pd.DataFrame,
    attribution: Optional[pd.DataFrame] = None,
    top_n: int = 20,
    n_genes: int = 8,
    figsize: Tuple[float, float] = (15.0, 8.5),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Three-panel view of a programme-level driver table from the hypergraph.

    Deliberately plots the quantities that carry information and omits the ones
    that do not.  On a real TF table three columns are constant or near-constant
    and would make misleading panels:

    - ``n_genes_attributed`` equals ``top_n_genes_per_edge`` for every row (it is
      the export cap, not a property of the programme),
    - ``fdr`` underflows to 0.0 for every row,
    - ``driver_score`` is a share over all pooled edges, so it is mechanically
      ~1/n_edges and its spread reflects pool size rather than effect strength.

    Panels
    ------
    left
        Diverging bars of ``log2_enrich`` per programme -- condition enrichment
        of the programme's *cells*, i.e. the edge-level direction.  Coloured by
        direction and placed by side-of-zero, so hue is never load-bearing.
    middle
        Up/down split of each programme's attributed genes as a stacked
        proportion.  This is the *gene*-level direction, which is a different
        quantity from the left panel: an edge enriched in treated cells can
        still contain mostly down-shifted genes.  A programme near 50/50 is
        internally mixed and its single ``direction`` label should be distrusted.
    right
        Strongest genes by ``|gene_score|`` for the leading programme, signed.
        Shows that per-gene scores vary within a programme -- the property the
        prior-footprint expansion lacked entirely.

    Parameters
    ----------
    summary : DataFrame
        ``tf`` or ``pathway`` frame from :func:`build_multilevel_driver_report`
        (equivalently ``hypergraph_tf_drivers.csv``).
    attribution : DataFrame, optional
        Edge-level attribution. Needed for the right panel only; when absent that
        panel is replaced by ``mean_abs_z`` per programme.
    top_n, n_genes : int
        Programmes to show, and genes in the right panel.
    figsize, title, save_path
        Passed through to matplotlib; ``save_path`` saves at 300 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    UP, DOWN, MUTED, GRID = "#c2662a", "#3b6fb5", "#9aa3ad", "#d8dde3"

    if summary is None or summary.empty:
        raise ValueError("`summary` is empty — nothing to plot.")

    d = summary.copy()
    # Rank on evidence strength, not driver_score (see docstring).
    rank_col = "mean_abs_z" if "mean_abs_z" in d.columns else "log2_enrich"
    d = d.reindex(d[rank_col].abs().sort_values(ascending=False).index).head(top_n)
    label = d["programme"].astype(str).str.replace("^(tf|pathway):", "", regex=True)
    d = d.assign(_label=label).sort_values("log2_enrich")

    fig, axes = plt.subplots(1, 3, figsize=figsize, width_ratios=[1.15, 1.0, 1.0])

    # ---------------------------------------------------------------- left
    ax = axes[0]
    colours = [UP if v >= 0 else DOWN for v in d["log2_enrich"]]
    ax.barh(
        d["_label"], d["log2_enrich"], color=colours, edgecolor="white", height=0.75
    )
    ax.axvline(0, color="#444", lw=1.0)
    ax.set_xlabel("log2 enrichment of treated cells in edge")
    ax.set_title("Edge-level direction\n(which cells the programme marks)", fontsize=10)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if "bin_source" in d.columns:
        n_con = int((d["bin_source"] == "tf_construction").sum())
        if n_con:
            ax.text(
                0.98,
                0.02,
                f"{n_con}/{len(d)} direction from edge construction",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color=MUTED,
                style="italic",
            )

    # -------------------------------------------------------------- middle
    ax = axes[1]
    if {"n_up", "n_down"}.issubset(d.columns):
        tot = (d["n_up"] + d["n_down"]).replace(0, np.nan)
        f_up = d["n_up"] / tot
        ax.barh(
            d["_label"],
            f_up,
            color=UP,
            edgecolor="white",
            height=0.75,
            label="genes up in treated",
        )
        ax.barh(
            d["_label"],
            1 - f_up,
            left=f_up,
            color=DOWN,
            edgecolor="white",
            height=0.75,
            label="genes down in treated",
        )
        ax.axvline(0.5, color="#444", lw=1.0, ls="--")
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of attributed genes")
        ax.set_title(
            "Gene-level direction\n(mixed programmes sit near 0.5)", fontsize=10
        )
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
        ax.set_yticklabels([])
    else:
        ax.axis("off")

    # --------------------------------------------------------------- right
    ax = axes[2]
    lead = str(d.iloc[-1]["programme"]) if len(d) else None
    sub = None
    if attribution is not None and not attribution.empty and lead is not None:
        sub = attribution[attribution["programme"].astype(str) == lead]
    if sub is not None and not sub.empty:
        g = (
            sub.reindex(sub["gene_score"].abs().sort_values(ascending=False).index)
            .head(n_genes)
            .sort_values("gene_score")
        )
        cols = [UP if v >= 0 else DOWN for v in g["gene_score"]]
        ax.barh(
            g["symbol"].astype(str),
            g["gene_score"],
            color=cols,
            edgecolor="white",
            height=0.72,
        )
        ax.axvline(0, color="#444", lw=1.0)
        ax.set_xlabel("gene_score  (signed |z| x sqrt(specificity))")
        ax.set_title(
            f"Top genes in {lead}\n(scores vary within a programme)", fontsize=10
        )
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    elif "mean_abs_z" in d.columns:
        ax.barh(
            d["_label"], d["mean_abs_z"], color=MUTED, edgecolor="white", height=0.75
        )
        ax.set_xlabel("mean |z| of attributed genes")
        ax.set_title("Within-edge evidence strength", fontsize=10)
        ax.set_yticklabels([])
    else:
        ax.axis("off")

    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12.5, y=0.99)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
