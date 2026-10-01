"""Functional enrichment analysis using decoupler."""

from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

try:
    import decoupler as dc
except ImportError:
    dc = None


def run_functional_analysis(
    adata: AnnData,
    cluster_key: str,
    n_top_tfs: Union[int, str, List[str]] = 10,
    max_score: Optional[float] = None,
    organism: str = "human",
    show_plots: bool = True,
) -> dict:
    """
    Run functional enrichment analysis using transcription factors and pathways.

    Performs enrichment analysis using:
    1. CollecTRI - Transcription factor activities
    2. PROGENy - Pathway activities
    3. MSigDB Hallmark - Hallmark gene sets

    Parameters
    ----------
    adata : AnnData
        Annotated data object with dpt_pseudotime computed
    cluster_key : str
        Key in adata.obs for cluster labels
    n_top_tfs : int, str, or list of str, default=10
        Number of top TFs to plot, or specific TF name(s)
    max_score : float, optional
        Maximum score for color scale. If None, computed from data
    organism : str, default='human'
        Organism for gene sets ('human' or 'mouse')
    show_plots : bool, default=True
        Whether to display plots

    Returns
    -------
    dict
        Dictionary containing:
        - 'tf_scores': TF activity scores (AnnData)
        - 'tf_ranking': Ranked TFs by correlation with pseudotime (DataFrame)
        - 'pathway_scores': Pathway activity scores (AnnData)
        - 'pathway_ranking': Ranked pathways (DataFrame)
        - 'hallmark_scores': Hallmark gene set scores (AnnData)
        - 'hallmark_ranking': Ranked hallmark sets (DataFrame)

    Raises
    ------
    ImportError
        If decoupler is not installed
    ValueError
        If n_top_tfs is not int, str, or list
    """
    if dc is None:
        raise ImportError(
            "decoupler is required for functional analysis. "
            "Install with: pip install 'decoupler[full]'"
        )

    if "dpt_pseudotime" not in adata.obs.columns:
        raise ValueError("dpt_pseudotime not found. Run compute_dpt_pseudotime first.")

    results = {}

    # 1. Transcription Factor Analysis
    print("Running transcription factor analysis...")
    collectri = dc.op.collectri(organism=organism)
    dc.mt.ulm(data=adata, net=collectri)
    tf_scores = dc.pp.get_obsm(adata=adata, key="score_ulm")

    results["tf_scores"] = tf_scores

    # Rank TFs by correlation with pseudotime using distance correlation
    tf_ranking = dc.tl.rankby_order(
        adata=tf_scores,
        order="dpt_pseudotime",
        stat="dcor",
    )
    results["tf_ranking"] = tf_ranking

    # Get top TFs
    if isinstance(n_top_tfs, int):
        top_tfs = tf_ranking.head(n_top_tfs)["name"].tolist()
    elif isinstance(n_top_tfs, str):
        top_tfs = [n_top_tfs]
    elif isinstance(n_top_tfs, list):
        top_tfs = n_top_tfs
    else:
        raise ValueError("n_top_tfs must be int, str, or list of str")

    # Determine color scale
    if max_score is None:
        tf_values = tf_scores.X
        if hasattr(tf_values, "toarray"):
            tf_values = tf_values.toarray()  # type: ignore
        max_score = float(np.max(np.abs(tf_values)))  # type: ignore

    if show_plots:
        # Plot TF activities on UMAP
        sc.pl.umap(
            tf_scores,
            color=top_tfs,
            cmap="coolwarm",
            ncols=2,
            title=[f"{t} score" for t in top_tfs],
            vmin=-max_score,
            vmax=max_score,
            size=20,
        )

        # Plot binned TF activities along pseudotime
        bin_tfs = dc.pp.bin_order(
            adata=tf_scores,
            order="dpt_pseudotime",
            names=top_tfs,
            label=cluster_key,
        )

        dc.pl.order(
            df=bin_tfs,
            mode="line",
            figsize=(6, 3),
        )
        plt.ylim(-max_score, max_score)
        plt.show()

        dc.pl.order(
            df=bin_tfs,
            mode="mat",
            kw_order={"vmin": -max_score, "vmax": max_score, "cmap": "RdBu_r"},
            figsize=(6, 4),
        )
        plt.show()

    # 2. Pathway Analysis (PROGENy)
    print("\nRunning pathway analysis...")
    progeny = dc.op.progeny(organism=organism)
    dc.mt.ulm(data=adata, net=progeny)
    pathway_scores = dc.pp.get_obsm(adata=adata, key="score_ulm")

    results["pathway_scores"] = pathway_scores

    pathway_ranking = dc.tl.rankby_order(
        adata=pathway_scores,
        order="dpt_pseudotime",
        stat="dcor",
    )
    results["pathway_ranking"] = pathway_ranking

    if show_plots:
        top_pathways = pathway_ranking.head(5)["name"].tolist()

        bin_pws = dc.pp.bin_order(
            adata=pathway_scores,
            order="dpt_pseudotime",
            names=top_pathways,
            label=cluster_key,
        )

        dc.pl.order(
            df=bin_pws,
            mode="mat",
            kw_order={"vmin": -10, "vmax": +10, "cmap": "RdBu_r"},
            figsize=(6, 3),
        )
        plt.show()

    # 3. Hallmark Gene Sets
    print("\nRunning hallmark gene set analysis...")
    hallmark = dc.op.hallmark(organism=organism)
    dc.mt.ulm(data=adata, net=hallmark)
    hallmark_scores = dc.pp.get_obsm(adata=adata, key="score_ulm")

    results["hallmark_scores"] = hallmark_scores

    hallmark_ranking = dc.tl.rankby_order(
        adata=hallmark_scores,
        order="dpt_pseudotime",
        stat="dcor",
    )
    results["hallmark_ranking"] = hallmark_ranking

    if show_plots:
        top_hallmarks = hallmark_ranking.head(5)["name"].tolist()

        bin_hlm = dc.pp.bin_order(
            adata=hallmark_scores,
            order="dpt_pseudotime",
            names=top_hallmarks,
            label=cluster_key,
        )

        dc.pl.order(
            df=bin_hlm,
            mode="mat",
            kw_order={"vmin": -5, "vmax": +5, "cmap": "RdBu_r"},
            figsize=(6, 3),
        )
        plt.show()

    print("\nFunctional analysis complete!")
    return results


# ---------------------------------------------------------------------------
# GO enrichment via g:Profiler (gprofiler-official)
# ---------------------------------------------------------------------------


def run_go_enrichment(
    gene_list: List[str],
    organism: str = "hsapiens",
    sources: Optional[List[str]] = None,
    n_top: int = 50,
) -> pd.DataFrame:
    """
    Run GO enrichment on *gene_list* using the g:Profiler REST API.

    Returns a tidy DataFrame with columns:
        term_name, source, term_id, -log10_pval, gene_count, query_size

    If ``gprofiler-official`` is not installed the function returns an empty
    DataFrame with the same columns so callers can proceed without crashing.

    Parameters
    ----------
    gene_list : list of str
        Gene symbols to test.
    organism : str
        g:Profiler organism code, e.g. ``'hsapiens'`` or ``'mmusculus'``.
    sources : list of str, optional
        g:Profiler annotation sources, e.g. ``['GO:BP', 'GO:MF', 'GO:CC']``.
        Defaults to ``['GO:BP', 'GO:MF', 'GO:CC', 'KEGG', 'REAC']``.
    n_top : int
        Maximum number of terms to return (ranked by p-value).

    Returns
    -------
    pd.DataFrame
    """
    _empty = pd.DataFrame(
        columns=[
            "term_name",
            "source",
            "term_id",
            "-log10_pval",
            "p_value",
            "intersection_size",
            "query_size",
        ]
    )

    if not gene_list:
        return _empty

    try:
        from gprofiler import GProfiler  # type: ignore[import]
    except ImportError:
        import warnings

        warnings.warn(
            "gprofiler-official is not installed; GO enrichment skipped. "
            "Install with: pip install gprofiler-official",
            ImportWarning,
            stacklevel=2,
        )
        return _empty

    if sources is None:
        sources = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"]

    gp = GProfiler(return_dataframe=True)
    try:
        result = gp.profile(
            organism=organism,
            query=gene_list,
            sources=sources,
            significance_threshold_method="fdr",
        )
    except Exception as exc:  # network errors, empty results, etc.
        import warnings

        warnings.warn(f"g:Profiler query failed: {exc}", RuntimeWarning, stacklevel=2)
        return _empty

    if result is None or len(result) == 0:
        return _empty

    # Rename g:Profiler columns to standard names used throughout the codebase
    col_map = {
        "name": "term_name",
        "native": "term_id",
        "p_value": "p_value",
        "significant": "significant",
        "intersection_size": "intersection_size",
        "query_size": "query_size",
    }
    result = result.rename(
        columns={k: v for k, v in col_map.items() if k in result.columns}
    )

    # Compute -log10(p-value) for plotting
    result["-log10_pval"] = -np.log10(result["p_value"].clip(lower=1e-300))

    # Keep only significant terms and return top-n
    if "significant" in result.columns:
        result = result[result["significant"]]
    result = result.sort_values("-log10_pval", ascending=False).head(n_top)

    keep = [
        c
        for c in [
            "term_name",
            "source",
            "term_id",
            "-log10_pval",
            "p_value",
            "intersection_size",
            "query_size",
        ]
        if c in result.columns
    ]
    return result[keep].reset_index(drop=True)
