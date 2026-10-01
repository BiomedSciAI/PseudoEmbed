"""
Trajectory-Conditioned Latent-Space Divergence & GAM-Based Differential Dynamics.

This module implements post-hoc analysis for comparing control vs drug conditions
over pseudotime using:
1. Trajectory-conditioned distributional divergence (Wasserstein distance, MMD)
2. GAM-based differential dynamics for TF activities and gene expression

Designed for cell-line experiments where pseudobulking is inappropriate
(cells are not from distinct biological replicates/donors). Instead, this
module operates at single-cell resolution and uses resampling/permutation
for uncertainty estimation.

Author: Computational Biology Team
"""

import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.spatial.distance import cdist
from scipy.stats import false_discovery_control, mannwhitneyu, pearsonr, spearmanr

try:
    from pygam import LinearGAM, s

    GAM_AVAILABLE = True
except ImportError:
    GAM_AVAILABLE = False
    warnings.warn(
        "pygam not available. Install with: pip install pygam",
        ImportWarning,
    )


# np.trapezoid was introduced in NumPy 2.0; np.trapz was removed in NumPy 2.0.
_trapz: Callable = getattr(np, "trapezoid", getattr(np, "trapz", None))  # type: ignore[attr-defined]
if _trapz is None:
    raise RuntimeError(
        "Neither np.trapezoid nor np.trapz found — unsupported NumPy version."
    )

# ============================================================================
# 1. TRAJECTORY-CONDITIONED DISTRIBUTIONAL DIVERGENCE
# ============================================================================


def _wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
    """Compute 1D Wasserstein-1 (Earth Mover's) distance between two samples."""
    u_sorted = np.sort(u)
    v_sorted = np.sort(v)
    all_values = np.sort(np.concatenate([u_sorted, v_sorted]))
    u_cdf = np.searchsorted(u_sorted, all_values, side="right") / len(u_sorted)
    v_cdf = np.searchsorted(v_sorted, all_values, side="right") / len(v_sorted)
    return float(_trapz(np.abs(u_cdf - v_cdf), all_values))


def _wasserstein_mean_nd(x: np.ndarray, y: np.ndarray) -> float:
    """
    Mean *marginal* 1D Wasserstein-1 distance across all columns of x and y.

    Computes W1(x[:,d], y[:,d]) for each dimension d independently and returns
    the mean. This is **not** the true multi-dimensional Wasserstein distance
    (which requires solving an optimal transport problem in R^d); it is the
    average of d univariate Earth-Mover distances. The interpretation is:
    "on average across TF dimensions, how far apart are the two distributions?"

    Handles unequal sample sizes correctly via CDF interpolation onto the
    combined support. ~100× faster than calling _wasserstein_1d in a Python loop
    because all d sorts are done in a single np.sort call.
    """
    n, d = x.shape
    m = y.shape[0]
    x_s = np.sort(x, axis=0)  # (n, d)
    y_s = np.sort(y, axis=0)  # (m, d)
    # combined support per dimension: (n+m, d)
    combined = np.sort(np.concatenate([x_s, y_s], axis=0), axis=0)
    # searchsorted doesn't support axis=, so we loop over d only;
    # d << n*n_permutations so this stays fast.
    total = 0.0
    for dim in range(d):
        u_cdf = np.searchsorted(x_s[:, dim], combined[:, dim], side="right") / n
        v_cdf = np.searchsorted(y_s[:, dim], combined[:, dim], side="right") / m
        total += float(_trapz(np.abs(u_cdf - v_cdf), combined[:, dim]))
    return total / d


def _mmd_from_kernel(
    K: np.ndarray,
    ctrl_idx: np.ndarray,
    drug_idx: np.ndarray,
) -> float:
    """
    Compute unbiased MMD² from a precomputed kernel matrix K using index arrays.

    Avoids recomputing cdist during permutations — just re-slices K.
    """
    n = len(ctrl_idx)
    m = len(drug_idx)
    kxx = K[np.ix_(ctrl_idx, ctrl_idx)]
    kyy = K[np.ix_(drug_idx, drug_idx)]
    kxy = K[np.ix_(ctrl_idx, drug_idx)]
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    term_xx = kxx.sum() / (n * (n - 1)) if n > 1 else 0.0
    term_yy = kyy.sum() / (m * (m - 1)) if m > 1 else 0.0
    term_xy = kxy.sum() / (n * m)
    return float(term_xx + term_yy - 2.0 * term_xy)


def _gaussian_kernel(x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute Gaussian (RBF) kernel matrix between two sets of samples."""
    sq_dists = cdist(x, y, metric="sqeuclidean")
    return np.exp(-sq_dists / (2.0 * bandwidth**2))


def _mmd_squared(
    x: np.ndarray, y: np.ndarray, bandwidth: Optional[float] = None
) -> float:
    """
    Compute unbiased squared Maximum Mean Discrepancy (MMD²) with Gaussian kernel.

    Parameters
    ----------
    x : np.ndarray, shape (n, d)
        Samples from distribution P.
    y : np.ndarray, shape (m, d)
        Samples from distribution Q.
    bandwidth : float, optional
        Kernel bandwidth. If None, uses median heuristic.

    Returns
    -------
    float
        Unbiased MMD² estimate.
    """
    if bandwidth is None:
        # Median heuristic for bandwidth selection
        combined = np.vstack([x, y])
        pairwise_dists = cdist(combined, combined, metric="euclidean")
        bandwidth = float(np.median(pairwise_dists[pairwise_dists > 0]))
        if bandwidth == 0:
            bandwidth = 1.0

    n = len(x)
    m = len(y)

    kxx = _gaussian_kernel(x, x, bandwidth)
    kyy = _gaussian_kernel(y, y, bandwidth)
    kxy = _gaussian_kernel(x, y, bandwidth)

    # Unbiased estimate: exclude diagonal for kxx and kyy
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)

    term_xx = kxx.sum() / (n * (n - 1)) if n > 1 else 0.0
    term_yy = kyy.sum() / (m * (m - 1)) if m > 1 else 0.0
    term_xy = kxy.sum() / (n * m)

    return float(term_xx + term_yy - 2.0 * term_xy)


def compute_trajectory_divergence(
    adata: AnnData,
    pseudotime_key: str = "hypergraph_pseudotime",
    condition_key: str = "condition",
    control_label: str = "control",
    drug_label: str = "drug",
    latent_key: str = "X_pca",
    n_bins: int = 10,
    n_latent_dims: Optional[int] = 30,
    metrics: Optional[List[str]] = None,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute trajectory-conditioned latent-space divergence between conditions.

    At each pseudotime window (bin), measures how different the control and drug
    cell distributions are in latent space using Wasserstein distance and/or MMD.

    This avoids pseudobulking — every cell contributes individually. Statistical
    significance is assessed via permutation testing (shuffling condition labels
    within each pseudotime bin).

    Parameters
    ----------
    adata : AnnData
        Annotated data with pseudotime and latent embedding computed.
    pseudotime_key : str
        Column in adata.obs with pseudotime values.
    condition_key : str
        Column in adata.obs with condition labels.
    control_label : str
        Label for control condition.
    drug_label : str
        Label for drug/treatment condition.
    latent_key : str
        Key in adata.obsm for latent embedding (e.g. 'X_pca', 'tf_activities').
    n_bins : int
        Number of pseudotime bins for trajectory conditioning.
    n_latent_dims : int, optional
        Number of latent dimensions to use. None = use all.
    metrics : list of str, optional
        Which divergence metrics to compute. Default: ['wasserstein', 'mmd'].
    n_permutations : int
        Number of permutations for p-value estimation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Per-bin divergence results with columns:
        - bin_idx, pseudotime_start, pseudotime_end, pseudotime_center
        - n_control, n_drug
        - wasserstein (mean over latent dims), mmd
        - wasserstein_pval, mmd_pval (permutation p-values)
    """
    if pseudotime_key not in adata.obs:
        raise ValueError(f"Pseudotime key '{pseudotime_key}' not in adata.obs")
    if condition_key not in adata.obs:
        raise ValueError(f"Condition key '{condition_key}' not in adata.obs")
    if latent_key not in adata.obsm:
        raise ValueError(f"Latent key '{latent_key}' not in adata.obsm")

    if metrics is None:
        metrics = ["wasserstein", "mmd"]

    rng = np.random.default_rng(random_state)

    # Extract latent embedding
    latent = adata.obsm[latent_key]
    if isinstance(latent, pd.DataFrame):
        latent = latent.values
    if n_latent_dims is not None:
        latent = latent[:, :n_latent_dims]

    pseudotime = adata.obs[pseudotime_key].values
    conditions = adata.obs[condition_key].values

    n_cells, n_dims = latent.shape
    print(f"\n{'='*70}")
    print("COMPUTING TRAJECTORY DIVERGENCE")
    print(f"{'='*70}")
    print(f"  Cells: {n_cells}  |  Latent dims: {n_dims}")
    print(f"  Pseudotime key : {pseudotime_key!r}")
    print(f"  Latent key     : {latent_key!r}")
    print(f"  Conditions     : {control_label!r} vs {drug_label!r}")
    print(f"  Bins           : {n_bins}  |  Permutations: {n_permutations}")
    print(f"  Metrics        : {metrics}")
    print(f"{'='*70}\n")

    # Define pseudotime bins
    pt_min, pt_max = np.nanmin(pseudotime), np.nanmax(pseudotime)
    bin_edges = np.linspace(pt_min, pt_max, n_bins + 1)

    results = []
    MMD_MAX_CELLS = 3000  # K_bin is O(n²) float64; cap to keep memory ≤ ~72 MB

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            bin_mask = (pseudotime >= lo) & (pseudotime < hi)
        else:
            bin_mask = (pseudotime >= lo) & (pseudotime <= hi)

        ctrl_mask = bin_mask & (conditions == control_label)
        drug_mask = bin_mask & (conditions == drug_label)

        n_ctrl = ctrl_mask.sum()
        n_drug = drug_mask.sum()

        print(
            f"  Bin {i+1:2d}/{n_bins}  pt=[{lo:.3f}, {hi:.3f}]  "
            f"ctrl={n_ctrl}  drug={n_drug}",
            end="",
        )

        row = {
            "bin_idx": i,
            "pseudotime_start": lo,
            "pseudotime_end": hi,
            "pseudotime_center": (lo + hi) / 2.0,
            "n_control": int(n_ctrl),
            "n_drug": int(n_drug),
        }

        # Need at least 5 cells in each condition for meaningful divergence
        if n_ctrl < 5 or n_drug < 5:
            for m in metrics:
                row[m] = np.nan
                row[f"{m}_pval"] = np.nan
            results.append(row)
            print("  → skipped (too few cells)")
            continue

        x_ctrl = latent[ctrl_mask]
        x_drug = latent[drug_mask]

        # --- Permutation test setup ---
        # Only include cells belonging to one of the two conditions so that
        # the permutation null matches the observed group sizes exactly.
        bin_pos_all = np.where(np.asarray(bin_mask))[0]
        bin_conds_all = np.asarray(conditions)[bin_pos_all]
        in_either = (bin_conds_all == control_label) | (bin_conds_all == drug_label)
        bin_pos = bin_pos_all[in_either]
        bin_latent = latent[bin_pos]
        bin_conditions = bin_conds_all[in_either]
        n_bin = len(bin_pos)  # cells from the two conditions only

        # Observed ctrl/drug positions *within* bin_latent
        obs_ctrl_idx = np.where(bin_conditions == control_label)[0]
        obs_drug_idx = np.where(bin_conditions == drug_label)[0]

        # --- Compute observed metrics ---
        if "wasserstein" in metrics:
            row["wasserstein"] = _wasserstein_mean_nd(x_ctrl, x_drug)

        # Precompute kernel matrix for the whole bin once (reused across all perms).
        # For large bins, subsample down to MMD_MAX_CELLS so K_bin stays ≤ ~72 MB
        # while keeping observed stat and null distribution on the same scale.
        K_bin: Optional[np.ndarray] = None
        K_latent = bin_latent  # rows used to build K_bin (may be subsampled)
        K_ctrl_idx = obs_ctrl_idx  # ctrl indices into K_latent
        K_drug_idx = obs_drug_idx  # drug indices into K_latent
        K_n_ctrl = n_ctrl  # ctrl group size for permutation split
        K_n_bin = n_bin  # total pool size for permutation split
        mmd_mode = ""
        if "mmd" in metrics:
            if n_bin > MMD_MAX_CELLS:
                # Subsample proportionally, preserving ctrl/drug ratio
                sub_idx = rng.choice(n_bin, size=MMD_MAX_CELLS, replace=False)
                sub_idx.sort()
                K_latent = bin_latent[sub_idx]
                sub_conds = bin_conditions[sub_idx]
                K_ctrl_idx = np.where(sub_conds == control_label)[0]
                K_drug_idx = np.where(sub_conds == drug_label)[0]
                K_n_ctrl = len(K_ctrl_idx)
                K_n_bin = MMD_MAX_CELLS
                mmd_mode = f"subsample({MMD_MAX_CELLS})"
            else:
                mmd_mode = "kernel"
            pairwise = cdist(K_latent, K_latent, metric="sqeuclidean")
            bw = (
                float(np.median(pairwise[pairwise > 0])) ** 0.5
                if np.any(pairwise > 0)
                else 1.0
            )
            K_bin = np.exp(-pairwise / (2.0 * bw**2))
            row["mmd"] = _mmd_from_kernel(K_bin, K_ctrl_idx, K_drug_idx)

        # Print observed stats before the slow permutation loop
        stat_parts = []
        if "wasserstein" in metrics:
            stat_parts.append(f"W={row['wasserstein']:.4f}")
        if "mmd" in metrics:
            stat_parts.append(f"MMD²={row['mmd']:.4f}({mmd_mode})")
        print(
            f"  [{', '.join(stat_parts)}]  running {n_permutations} permutations...",
            end="",
            flush=True,
        )

        perm_w = np.zeros(n_permutations)
        perm_mmd = np.zeros(n_permutations)

        for p in range(n_permutations):
            # Shuffle all n_bin positions; assign first n_ctrl to control.
            perm_idx = rng.permutation(n_bin)
            perm_ctrl_idx = perm_idx[:n_ctrl]
            perm_drug_idx = perm_idx[n_ctrl:]

            if "wasserstein" in metrics:
                perm_w[p] = _wasserstein_mean_nd(
                    bin_latent[perm_ctrl_idx], bin_latent[perm_drug_idx]
                )

            if "mmd" in metrics and K_bin is not None:
                # Permute within K_latent (full bin or subsample — same pool used
                # for the observed stat, so null distribution is on the same scale).
                perm_K_idx = rng.permutation(K_n_bin)
                perm_mmd[p] = _mmd_from_kernel(
                    K_bin, perm_K_idx[:K_n_ctrl], perm_K_idx[K_n_ctrl:]
                )

        if "wasserstein" in metrics:
            row["wasserstein_pval"] = float(
                (np.sum(perm_w >= row["wasserstein"]) + 1) / (n_permutations + 1)
            )
        if "mmd" in metrics:
            row["mmd_pval"] = float(
                (np.sum(perm_mmd >= row["mmd"]) + 1) / (n_permutations + 1)
            )

        pval_parts = []
        if "wasserstein" in metrics:
            pval_parts.append(f"p={row['wasserstein_pval']:.3f}")
        if "mmd" in metrics:
            pval_parts.append(f"p={row['mmd_pval']:.3f}")
        print(f"  done  ({', '.join(pval_parts)})")

        results.append(row)

    print(f"\n✓ Trajectory divergence complete ({len(results)} bins)\n")
    df = pd.DataFrame(results)

    # FDR correction across bins
    for m in metrics:
        pval_col = f"{m}_pval"
        valid = df[pval_col].notna()
        if valid.sum() > 0:
            fdr = false_discovery_control(df.loc[valid, pval_col].values, method="bh")
            df.loc[valid, f"{m}_qval"] = fdr

    return df


# ============================================================================
# 2. GAM-BASED DIFFERENTIAL DYNAMICS
# ============================================================================


def identify_trajectory_drivers(
    adata: AnnData,
    pseudotime_key: str = "hypergraph_pseudotime",
    n_top: int = 20,
    correlation_method: str = "spearman",
    min_expression: float = 0.01,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify TFs and genes that drive the pseudotime trajectory.

    Ranks features by their absolute Spearman (or Pearson) correlation with
    pseudotime — condition-agnostic, no GAM required.  Run this before
    condition-specific analysis to find the globally strongest trajectory
    drivers regardless of treatment.

    Parameters
    ----------
    adata : AnnData
        Annotated data with TF activities and pseudotime.
    pseudotime_key : str
        Column in ``adata.obs`` with pseudotime values.
    n_top : int
        Number of top drivers to return per category (TFs and genes).
    correlation_method : str
        ``'spearman'`` (default) or ``'pearson'``.
    min_expression : float
        Minimum mean absolute value for a feature to be considered.
        Features below this threshold are skipped.

    Returns
    -------
    tf_drivers : pd.DataFrame
        Top TF drivers with columns ``tf``, ``correlation``,
        ``abs_correlation``, ``p_value``, ``mean_activity``, ``std_activity``.
        Empty DataFrame if TF activities are absent.
    gene_drivers : pd.DataFrame
        Top gene drivers with columns ``gene``, ``correlation``,
        ``abs_correlation``, ``p_value``, ``mean_expression``,
        ``std_expression``.
    """
    _corr = spearmanr if correlation_method == "spearman" else pearsonr

    pseudotime = adata.obs[pseudotime_key].values

    # ------------------------------------------------------------------
    # TF drivers
    # ------------------------------------------------------------------
    if "tf_activities" not in adata.obsm:
        tf_drivers = pd.DataFrame()
    else:
        tf_acts = adata.obsm["tf_activities"]
        if hasattr(tf_acts, "values"):
            tf_acts = tf_acts.values
        tf_names = (
            adata.uns["tf_activities_info"]["tf_names"]
            if "tf_activities_info" in adata.uns
            else [f"TF_{i}" for i in range(tf_acts.shape[1])]
        )

        rows = []
        for i, tf_name in enumerate(tf_names):
            corr, pval = _corr(tf_acts[:, i], pseudotime)
            rows.append(
                {
                    "tf": tf_name,
                    "correlation": float(corr),
                    "abs_correlation": abs(float(corr)),
                    "p_value": float(pval),
                    "mean_activity": float(np.mean(tf_acts[:, i])),
                    "std_activity": float(np.std(tf_acts[:, i])),
                }
            )

        tf_drivers = (
            pd.DataFrame(rows)
            .sort_values("abs_correlation", ascending=False)
            .head(n_top)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Gene drivers
    # ------------------------------------------------------------------
    raw = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    if raw.ndim != 2:
        raw = raw.reshape(-1, len(adata.var_names))
    gene_names = list(adata.var_names)

    mean_expr = np.asarray(np.mean(raw, axis=0)).ravel()
    expressed = mean_expr > min_expression
    if not expressed.any():
        # Fall back to top expressed genes rather than returning nothing
        fallback_n = min(len(gene_names), 2000)
        top_idx = np.argsort(mean_expr)[-fallback_n:]
        expressed = np.zeros(len(gene_names), dtype=bool)
        expressed[top_idx] = True

    rows = []
    for i, gene_name in enumerate(gene_names):
        if not expressed[i]:
            continue
        gene_vec = np.asarray(raw[:, i]).ravel()
        corr, pval = _corr(gene_vec, pseudotime)
        rows.append(
            {
                "gene": gene_name,
                "correlation": float(corr),
                "abs_correlation": abs(float(corr)),
                "p_value": float(pval),
                "mean_expression": float(mean_expr[i]),
                "std_expression": float(np.std(gene_vec)),
            }
        )

    gene_drivers = (
        pd.DataFrame(rows)
        .sort_values("abs_correlation", ascending=False)
        .head(n_top)
        .reset_index(drop=True)
    )

    return tf_drivers, gene_drivers


def fit_condition_gams(
    adata: AnnData,
    features: Union[str, List[str]],
    pseudotime_key: str = "hypergraph_pseudotime",
    condition_key: str = "condition",
    control_label: str = "control",
    drug_label: str = "drug",
    feature_source: str = "X",
    n_splines: int = 6,
    spline_order: int = 3,
    n_grid: int = 100,
) -> Dict[str, pd.DataFrame]:
    """
    Fit separate GAMs for each condition to model feature dynamics over pseudotime.

    For each feature (gene / TF activity), fits:
        E[feature | pseudotime, condition=c] = f_c(pseudotime)

    using a penalised spline GAM per condition, then computes the difference
    curve Δ(t) = f_drug(t) - f_control(t) with pointwise confidence intervals.

    Parameters
    ----------
    adata : AnnData
        Annotated data.
    features : str or list of str
        Feature names (gene names or TF names).
    pseudotime_key : str
        Column in adata.obs with pseudotime.
    condition_key : str
        Column in adata.obs with condition labels.
    control_label : str
        Control condition label.
    drug_label : str
        Drug condition label.
    feature_source : str
        Where to get feature values:
        - 'X': from adata.X (gene expression)
        - 'tf_activities': from adata.obsm['tf_activities']
    n_splines : int
        Number of spline basis functions for GAM.
    spline_order : int
        Spline order (3 = cubic).
    n_grid : int
        Number of pseudotime grid points for prediction.

    Returns
    -------
    dict
        Keys are feature names. Values are DataFrames with columns:
        - pseudotime: grid points
        - control_mean, control_ci_lo, control_ci_hi: control GAM fit ± 95% CI
        - drug_mean, drug_ci_lo, drug_ci_hi: drug GAM fit ± 95% CI
        - delta: drug_mean - control_mean
        - delta_ci_lo, delta_ci_hi: approximate CI on delta
        - significant: whether CI on delta excludes 0
    """
    if not GAM_AVAILABLE:
        raise ImportError(
            "pygam is required for GAM fitting. Install with: pip install pygam"
        )

    if isinstance(features, str):
        features = [features]

    if pseudotime_key not in adata.obs:
        raise ValueError(f"Pseudotime key '{pseudotime_key}' not in adata.obs")
    if condition_key not in adata.obs:
        raise ValueError(f"Condition key '{condition_key}' not in adata.obs")

    pseudotime = adata.obs[pseudotime_key].values
    conditions = adata.obs[condition_key].values

    ctrl_mask = conditions == control_label
    drug_mask = conditions == drug_label

    pt_ctrl = pseudotime[ctrl_mask]
    pt_drug = pseudotime[drug_mask]

    # Grid for prediction
    pt_min = np.nanmin(pseudotime)
    pt_max = np.nanmax(pseudotime)
    grid = np.linspace(pt_min, pt_max, n_grid)

    # Get feature values
    if feature_source == "X":
        if hasattr(adata.X, "toarray"):
            X = adata.X.toarray()
        else:
            X = np.asarray(adata.X)
        gene_names = list(adata.var_names)
    elif feature_source == "tf_activities":
        X = adata.obsm["tf_activities"]
        if isinstance(X, pd.DataFrame):
            gene_names = list(X.columns)
            X = X.values
        elif "tf_activities_info" in adata.uns:
            gene_names = adata.uns["tf_activities_info"]["tf_names"]
        else:
            gene_names = [f"TF_{i}" for i in range(X.shape[1])]
    else:
        raise ValueError(
            f"Unknown feature_source '{feature_source}'. Use 'X' or 'tf_activities'."
        )

    results = {}

    for feat_name in features:
        if feat_name not in gene_names:
            warnings.warn(f"Feature '{feat_name}' not found, skipping.")
            continue

        feat_idx = list(gene_names).index(feat_name)
        y_all = X[:, feat_idx]

        y_ctrl = y_all[ctrl_mask]
        y_drug = y_all[drug_mask]

        # Fit GAM for control
        gam_ctrl = LinearGAM(s(0, n_splines=n_splines, spline_order=spline_order))
        gam_ctrl.fit(pt_ctrl.reshape(-1, 1), y_ctrl)

        # Fit GAM for drug
        gam_drug = LinearGAM(s(0, n_splines=n_splines, spline_order=spline_order))
        gam_drug.fit(pt_drug.reshape(-1, 1), y_drug)

        # Predictions on grid
        grid_2d = grid.reshape(-1, 1)

        ctrl_pred = gam_ctrl.predict(grid_2d)
        ctrl_ci = gam_ctrl.confidence_intervals(grid_2d, width=0.95)

        drug_pred = gam_drug.predict(grid_2d)
        drug_ci = gam_drug.confidence_intervals(grid_2d, width=0.95)

        # Delta and approximate CI (propagate CIs as independent)
        delta = drug_pred - ctrl_pred
        # Conservative CI on difference: combine half-widths in quadrature
        ctrl_hw = (ctrl_ci[:, 1] - ctrl_ci[:, 0]) / 2.0
        drug_hw = (drug_ci[:, 1] - drug_ci[:, 0]) / 2.0
        delta_hw = np.sqrt(ctrl_hw**2 + drug_hw**2)

        delta_ci_lo = delta - delta_hw
        delta_ci_hi = delta + delta_hw

        # Significant where CI excludes 0
        significant = (delta_ci_lo > 0) | (delta_ci_hi < 0)

        df = pd.DataFrame(
            {
                "pseudotime": grid,
                "control_mean": ctrl_pred,
                "control_ci_lo": ctrl_ci[:, 0],
                "control_ci_hi": ctrl_ci[:, 1],
                "drug_mean": drug_pred,
                "drug_ci_lo": drug_ci[:, 0],
                "drug_ci_hi": drug_ci[:, 1],
                "delta": delta,
                "delta_ci_lo": delta_ci_lo,
                "delta_ci_hi": delta_ci_hi,
                "significant": significant,
            }
        )

        results[feat_name] = df

    return results


def compute_differential_dynamics(
    adata: AnnData,
    pseudotime_key: str = "hypergraph_pseudotime",
    condition_key: str = "condition",
    control_label: str = "control",
    drug_label: str = "drug",
    feature_source: str = "X",
    n_top_features: int = 50,
    n_splines: int = 6,
    spline_order: int = 3,
    n_grid: int = 100,
    min_expression: float = 0.1,
    fdr_threshold: float = 0.05,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Screen all features for differential dynamics between conditions.

    Identifies features whose temporal profiles differ significantly between
    control and drug across pseudotime. Uses GAM fits + a likelihood-ratio-style
    test comparing the joint model (shared GAM) vs separate-condition GAMs.

    Parameters
    ----------
    adata : AnnData
        Annotated data.
    pseudotime_key : str
        Column in adata.obs with pseudotime.
    condition_key : str
        Column in adata.obs with condition labels.
    control_label : str
        Control condition label.
    drug_label : str
        Drug condition label.
    feature_source : str
        'X' for gene expression or 'tf_activities'.
    n_top_features : int
        Max number of top differentially dynamic features to return.
    n_splines : int
        Number of spline basis functions.
    spline_order : int
        Spline order.
    n_grid : int
        Grid points for GAM prediction.
    min_expression : float
        Minimum mean expression to consider a feature.
    fdr_threshold : float
        FDR threshold for significance.

    Returns
    -------
    summary_df : pd.DataFrame
        Summary of differentially dynamic features (sorted by significance).
    gam_fits : dict
        GAM fit DataFrames for the top features (same format as fit_condition_gams).
    """
    if not GAM_AVAILABLE:
        raise ImportError(
            "pygam is required for GAM fitting. Install with: pip install pygam"
        )

    if pseudotime_key not in adata.obs:
        raise ValueError(f"Pseudotime key '{pseudotime_key}' not in adata.obs")
    if condition_key not in adata.obs:
        raise ValueError(f"Condition key '{condition_key}' not in adata.obs")

    pseudotime = adata.obs[pseudotime_key].values
    conditions = adata.obs[condition_key].values

    ctrl_mask = conditions == control_label
    drug_mask = conditions == drug_label

    pt_ctrl = pseudotime[ctrl_mask]
    pt_drug = pseudotime[drug_mask]
    pt_all = pseudotime

    # Get feature matrix
    if feature_source == "X":
        if hasattr(adata.X, "toarray"):
            X = adata.X.toarray()
        else:
            X = np.asarray(adata.X)
        feature_names = list(adata.var_names)
    elif feature_source == "tf_activities":
        X = adata.obsm["tf_activities"]
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X = X.values
        elif "tf_activities_info" in adata.uns:
            feature_names = adata.uns["tf_activities_info"]["tf_names"]
        else:
            feature_names = [f"TF_{i}" for i in range(X.shape[1])]
    else:
        raise ValueError(f"Unknown feature_source '{feature_source}'")

    n_features = X.shape[1]
    grid = np.linspace(np.nanmin(pseudotime), np.nanmax(pseudotime), n_grid)
    grid_2d = grid.reshape(-1, 1)

    screening_results = []

    for feat_idx in range(n_features):
        y_all = X[:, feat_idx]

        # Filter low-expression features
        if np.mean(np.abs(y_all)) < min_expression:
            continue

        y_ctrl = y_all[ctrl_mask]
        y_drug = y_all[drug_mask]

        try:
            # Fit shared GAM (null: no condition effect)
            gam_shared = LinearGAM(s(0, n_splines=n_splines, spline_order=spline_order))
            gam_shared.fit(pt_all.reshape(-1, 1), y_all)

            # Fit separate GAMs (alternative: condition-specific dynamics)
            gam_ctrl = LinearGAM(s(0, n_splines=n_splines, spline_order=spline_order))
            gam_ctrl.fit(pt_ctrl.reshape(-1, 1), y_ctrl)

            gam_drug = LinearGAM(s(0, n_splines=n_splines, spline_order=spline_order))
            gam_drug.fit(pt_drug.reshape(-1, 1), y_drug)

            # Compute residual sum of squares for model comparison
            # Shared model RSS
            pred_shared = gam_shared.predict(pt_all.reshape(-1, 1))
            rss_shared = np.sum((y_all - pred_shared) ** 2)

            # Separate model RSS
            pred_ctrl = gam_ctrl.predict(pt_ctrl.reshape(-1, 1))
            pred_drug = gam_drug.predict(pt_drug.reshape(-1, 1))
            rss_separate = np.sum((y_ctrl - pred_ctrl) ** 2) + np.sum(
                (y_drug - pred_drug) ** 2
            )

            # F-statistic-like score for differential dynamics
            n_total = len(y_all)
            df_shared = n_splines
            df_separate = 2 * n_splines
            df_diff = df_separate - df_shared

            if rss_separate > 0 and df_diff > 0:
                f_stat = ((rss_shared - rss_separate) / df_diff) / (
                    rss_separate / (n_total - df_separate)
                )
            else:
                f_stat = 0.0

            # Mean absolute delta on grid
            ctrl_grid_pred = gam_ctrl.predict(grid_2d)
            drug_grid_pred = gam_drug.predict(grid_2d)
            mean_abs_delta = float(np.mean(np.abs(drug_grid_pred - ctrl_grid_pred)))
            max_abs_delta = float(np.max(np.abs(drug_grid_pred - ctrl_grid_pred)))

            # Also compute a simple Mann-Whitney on residuals as backup
            ctrl_resid = y_ctrl - gam_shared.predict(pt_ctrl.reshape(-1, 1))
            drug_resid = y_drug - gam_shared.predict(pt_drug.reshape(-1, 1))
            _, mw_pval = mannwhitneyu(ctrl_resid, drug_resid, alternative="two-sided")

            screening_results.append(
                {
                    "feature": feature_names[feat_idx],
                    "f_statistic": f_stat,
                    "rss_shared": rss_shared,
                    "rss_separate": rss_separate,
                    "rss_improvement": (
                        (rss_shared - rss_separate) / rss_shared
                        if rss_shared > 0
                        else 0.0
                    ),
                    "mean_abs_delta": mean_abs_delta,
                    "max_abs_delta": max_abs_delta,
                    "mw_pval": mw_pval,
                    "mean_ctrl": float(np.mean(y_ctrl)),
                    "mean_drug": float(np.mean(y_drug)),
                }
            )

        except Exception:
            # Skip features where GAM fitting fails
            continue

    if not screening_results:
        return pd.DataFrame(), {}

    summary_df = pd.DataFrame(screening_results)

    # FDR correction on MW p-values
    summary_df["mw_qval"] = false_discovery_control(
        summary_df["mw_pval"].values, method="bh"
    )

    # Sort by F-statistic (larger = more evidence of differential dynamics)
    summary_df = summary_df.sort_values("f_statistic", ascending=False).reset_index(
        drop=True
    )

    # Mark significance
    summary_df["significant"] = summary_df["mw_qval"] < fdr_threshold

    # Get top features and fit detailed GAMs
    top_features = summary_df.head(n_top_features)["feature"].tolist()
    gam_fits = fit_condition_gams(
        adata,
        features=top_features,
        pseudotime_key=pseudotime_key,
        condition_key=condition_key,
        control_label=control_label,
        drug_label=drug_label,
        feature_source=feature_source,
        n_splines=n_splines,
        spline_order=spline_order,
        n_grid=n_grid,
    )

    return summary_df, gam_fits


# ============================================================================
# 3. VISUALIZATION
# ============================================================================


def plot_trajectory_divergence(
    divergence_df: pd.DataFrame,
    metric: str = "wasserstein",
    title: Optional[str] = None,
    ax=None,
):
    """
    Plot trajectory-conditioned divergence over pseudotime.

    Parameters
    ----------
    divergence_df : pd.DataFrame
        Output of compute_trajectory_divergence.
    metric : str
        Which metric to plot ('wasserstein' or 'mmd').
    title : str, optional
        Plot title.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    x = divergence_df["pseudotime_center"].values
    y = divergence_df[metric].values
    qval_col = f"{metric}_qval"

    ax.plot(x, y, "o-", color="steelblue", linewidth=2, markersize=6)

    # Highlight significant bins
    if qval_col in divergence_df.columns:
        sig_mask = divergence_df[qval_col] < 0.05
        if sig_mask.any():
            ax.scatter(
                x[sig_mask],
                y[sig_mask],
                color="red",
                s=80,
                zorder=5,
                label="FDR < 0.05",
            )
            ax.legend()

    ax.set_xlabel("Pseudotime", fontsize=12)
    ylabel = "Wasserstein Distance" if metric == "wasserstein" else "MMD²"
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        title or f"Trajectory-Conditioned {ylabel}: Control vs Drug", fontsize=13
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return ax


def plot_differential_gam(
    gam_result: pd.DataFrame,
    feature_name: str,
    control_label: str = "control",
    drug_label: str = "drug",
    ax=None,
):
    """
    Plot GAM fits for control vs drug with difference curve.

    Parameters
    ----------
    gam_result : pd.DataFrame
        One entry from the dict returned by fit_condition_gams.
    feature_name : str
        Feature name for title.
    control_label : str
        Label for control.
    drug_label : str
        Label for drug.
    ax : matplotlib Axes, optional
        If None, creates a 2-panel figure.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    else:
        axes = [ax, ax.twinx()]  # fallback

    t = gam_result["pseudotime"].values

    # Top panel: GAM fits
    ax1 = axes[0]
    ax1.plot(
        t,
        gam_result["control_mean"],
        color="steelblue",
        linewidth=2,
        label=control_label,
    )
    ax1.fill_between(
        t,
        gam_result["control_ci_lo"],
        gam_result["control_ci_hi"],
        alpha=0.2,
        color="steelblue",
    )
    ax1.plot(
        t, gam_result["drug_mean"], color="firebrick", linewidth=2, label=drug_label
    )
    ax1.fill_between(
        t,
        gam_result["drug_ci_lo"],
        gam_result["drug_ci_hi"],
        alpha=0.2,
        color="firebrick",
    )
    ax1.set_ylabel("Expression / Activity", fontsize=11)
    ax1.set_title(f"{feature_name}: GAM Temporal Dynamics", fontsize=13)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Bottom panel: Difference curve
    ax2 = axes[1]
    delta = gam_result["delta"].values
    delta_lo = gam_result["delta_ci_lo"].values
    delta_hi = gam_result["delta_ci_hi"].values
    sig = gam_result["significant"].values

    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.plot(t, delta, color="darkgreen", linewidth=2, label="Δ (drug − control)")
    ax2.fill_between(t, delta_lo, delta_hi, alpha=0.2, color="darkgreen")

    # Shade significant regions
    sig_starts = []
    in_sig = False
    for i in range(len(sig)):
        if sig[i] and not in_sig:
            sig_starts.append(i)
            in_sig = True
        elif not sig[i] and in_sig:
            ax2.axvspan(t[sig_starts[-1]], t[i - 1], alpha=0.1, color="red")
            in_sig = False
    if in_sig:
        ax2.axvspan(t[sig_starts[-1]], t[-1], alpha=0.1, color="red")

    ax2.set_xlabel("Pseudotime", fontsize=11)
    ax2.set_ylabel("Δ (drug − control)", fontsize=11)
    ax2.set_title(f"{feature_name}: Differential Dynamics", fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return axes


def plot_top_differential_features(
    summary_df: pd.DataFrame,
    gam_fits: Dict[str, pd.DataFrame],
    n_top: int = 8,
    control_label: str = "control",
    drug_label: str = "drug",
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    Plot a grid of the top differentially dynamic features.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of compute_differential_dynamics.
    gam_fits : dict
        GAM fit results dict.
    n_top : int
        Number of top features to plot.
    control_label : str
        Control label.
    drug_label : str
        Drug label.
    figsize : tuple, optional
        Figure size.
    """
    import matplotlib.pyplot as plt

    top = summary_df.head(n_top)["feature"].tolist()
    # Only plot features that have GAM fits
    top = [f for f in top if f in gam_fits]
    n = len(top)

    if n == 0:
        warnings.warn("No features to plot.")
        return

    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, feat in enumerate(top):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        df = gam_fits[feat]
        t = df["pseudotime"].values

        ax.plot(
            t, df["control_mean"], color="steelblue", linewidth=1.5, label=control_label
        )
        ax.fill_between(
            t, df["control_ci_lo"], df["control_ci_hi"], alpha=0.15, color="steelblue"
        )
        ax.plot(t, df["drug_mean"], color="firebrick", linewidth=1.5, label=drug_label)
        ax.fill_between(
            t, df["drug_ci_lo"], df["drug_ci_hi"], alpha=0.15, color="firebrick"
        )

        ax.set_title(feat, fontsize=10)
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Top Differentially Dynamic Features", fontsize=13, y=1.01)
    plt.tight_layout()
    return fig
