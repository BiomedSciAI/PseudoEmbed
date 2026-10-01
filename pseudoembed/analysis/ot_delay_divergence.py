"""
Optimal-transport decomposition of a perturbation into delay and divergence.

Motivation
----------
A treatment can move cells in two qualitatively different ways:

* **Delay** — the cells follow the *same* trajectory but sit further back along
  it.  Nothing new is expressed; the programme is simply behind schedule.
* **Divergence** — the cells leave the trajectory, entering a state the control
  arm never occupies.

These demand different interpretations (a delayed cell will catch up; a diverged
one will not), yet the existing readouts cannot tell them apart:

* :func:`~pseudoembed.analysis.trajectory_differential_dynamics.compute_trajectory_divergence`
  measures *marginal* distances (mean per-dimension 1-D Wasserstein, MMD).  Those
  say the two distributions differ; they do not say in which direction, so a pure
  delay and a pure divergence of equal magnitude score alike.
* The eigenvector classifier in the multiview pipeline is *structurally* blind to
  delay: a delay reparametrises the existing maturation axis rather than creating
  a new one, so it yields zero perturbation axes no matter how large it is
  (measured: pure delay gives 0 perturbation axes but a −0.194 ± 0.009 pseudotime
  gap, ~10σ from the null spread of 0.019).

What is missing in both is a *coupling* — which control cell each treated cell
corresponds to.  Without it there is no displacement vector to decompose.  This
module supplies the coupling with entropic optimal transport and then splits each
displacement into a component along the trajectory (delay) and a component
orthogonal to it (divergence).

Method
------
Within each timepoint (so that the comparison is never confounded by
developmental stage):

1. Solve an entropic OT problem between treated and control cells in latent
   space, giving a soft coupling ``P`` where ``P[i, j]`` is the mass moved from
   treated cell *i* to control cell *j*.  Treated and control cells are matched
   by *proximity*, not by index, so unequal group sizes are handled natively.
2. For each treated cell, form its **barycentric projection** — the coupling-
   weighted mean of its matched control cells.  This is the counterfactual
   "where would this cell be, had it not been treated".
3. Take the displacement ``d_i = x_i − x̂_i`` and project it onto the local
   trajectory tangent ``t_i``:

   * ``delay_i = −⟨d_i, t_i⟩`` — signed. Positive = behind its counterfactual.
   * ``divergence_i = ‖d_i − ⟨d_i, t_i⟩ t_i‖`` — unsigned magnitude, since
     "off-trajectory" has no privileged direction.

The tangent is estimated from the *control* cells only, by local linear
regression of latent coordinates on pseudotime.  Using control cells matters: on
treated data the trajectory itself may be distorted, which would make the
reference frame depend on the effect being measured.

Because both components come from one coupling, they are directly comparable and
sum (in quadrature) to the total displacement — so a perturbation can be reported
as a delay/divergence *ratio* rather than two incomparable statistics.

Significance is assessed by permuting condition labels within timepoint, which
preserves the trajectory structure and tests only the condition assignment.

Requires POT (``pip install pot``).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData

try:  # pragma: no cover - exercised by the import-guard test
    import ot as pot

    POT_AVAILABLE = True
except ImportError:  # pragma: no cover
    POT_AVAILABLE = False


__all__ = [
    "decompose_perturbation_ot",
    "estimate_trajectory_tangent",
    "ot_barycentric_projection",
]


# ============================================================================
# Trajectory tangent
# ============================================================================


def estimate_trajectory_tangent(
    latent: np.ndarray,
    pseudotime: np.ndarray,
    query_pseudotime: np.ndarray,
    bandwidth: Optional[float] = None,
    min_neighbours: int = 20,
) -> np.ndarray:
    """
    Local trajectory direction d(latent)/d(pseudotime) at each query point.

    Fits a weighted linear regression of each latent coordinate on pseudotime
    inside a Gaussian window, and returns the unit-normalised slope vector.  The
    slope *is* the tangent: it is the direction in latent space that a cell moves
    as pseudotime advances.

    Parameters
    ----------
    latent : ndarray, shape (n_ref, n_dims)
        Reference cells defining the trajectory.  Pass **control cells only** —
        see the module docstring for why.
    pseudotime : ndarray, shape (n_ref,)
        Pseudotime of the reference cells.
    query_pseudotime : ndarray, shape (n_query,)
        Pseudotime values at which the tangent is wanted.
    bandwidth : float, optional
        Gaussian window width in pseudotime units.  Defaults to 10% of the
        reference pseudotime range, which trades resolution against stability;
        too narrow and the slope is fit to noise, too wide and it averages away
        real curvature.
    min_neighbours : int
        If fewer than this many reference cells fall within ~2 bandwidths of a
        query point, the window is widened for that point rather than returning a
        noise-dominated slope.

    Returns
    -------
    tangents : ndarray, shape (n_query, n_dims)
        Unit vectors.  Rows are all-zero where the tangent is indeterminate
        (degenerate pseudotime spread); callers must treat a zero row as
        "no tangent here" rather than as a direction.
    """
    latent = np.asarray(latent, dtype=float)
    pseudotime = np.asarray(pseudotime, dtype=float).ravel()
    query_pseudotime = np.asarray(query_pseudotime, dtype=float).ravel()

    if latent.shape[0] != pseudotime.shape[0]:
        raise ValueError(
            f"latent has {latent.shape[0]} rows but pseudotime has "
            f"{pseudotime.shape[0]} entries."
        )
    if latent.shape[0] < 2:
        raise ValueError("Need at least 2 reference cells to estimate a tangent.")

    pt_range = float(np.ptp(pseudotime))
    if bandwidth is None:
        bandwidth = 0.1 * pt_range if pt_range > 0 else 1.0
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}.")

    n_query = query_pseudotime.shape[0]
    n_dims = latent.shape[1]
    tangents = np.zeros((n_query, n_dims), dtype=float)

    for i, t0 in enumerate(query_pseudotime):
        h = bandwidth
        # Widen until the window holds enough cells.  Capped so a query point
        # outside the reference range cannot loop forever.
        for _ in range(6):
            w = np.exp(-0.5 * ((pseudotime - t0) / h) ** 2)
            if int(np.sum(w > 0.05)) >= min_neighbours:
                break
            h *= 2.0

        sw = w.sum()
        if sw <= 0:
            continue
        # Weighted least squares slope, computed in closed form per dimension:
        # beta = cov_w(t, x) / var_w(t).  Cheaper and better conditioned than
        # assembling a design matrix for a single predictor.
        t_bar = float(np.dot(w, pseudotime) / sw)
        dt = pseudotime - t_bar
        var_t = float(np.dot(w, dt * dt) / sw)
        if var_t <= 1e-12:
            # All reference cells in this window share one pseudotime, so the
            # slope is undefined.  Leave the row at zero.
            continue
        x_bar = (w[:, None] * latent).sum(axis=0) / sw
        cov_tx = (w[:, None] * dt[:, None] * (latent - x_bar)).sum(axis=0) / sw
        beta = cov_tx / var_t

        norm = float(np.linalg.norm(beta))
        if norm > 1e-12:
            tangents[i] = beta / norm

    return tangents


# ============================================================================
# OT coupling
# ============================================================================


def ot_barycentric_projection(
    x_treated: np.ndarray,
    x_control: np.ndarray,
    reg: float = 0.05,
    max_iter: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Entropic-OT counterfactual position for each treated cell.

    Solves an entropic OT problem with uniform marginals and squared-Euclidean
    cost, then returns the barycentric projection: the coupling-weighted mean
    control cell for each treated cell.

    Parameters
    ----------
    x_treated, x_control : ndarray
        Latent coordinates, shapes (n_t, d) and (n_c, d).  ``d`` must match.
    reg : float
        Entropic regularisation, as a **fraction of the mean transport cost** —
        not an absolute value.  Scaling by the cost matrix makes the setting
        transferable between latent spaces of different magnitude (raw PCA
        coordinates and unit-variance embeddings differ by orders of magnitude,
        so a fixed epsilon that works on one diverges or over-smooths on the
        other).  Larger = blurrier coupling.
    max_iter : int
        Sinkhorn iteration cap.

    Returns
    -------
    projection : ndarray, shape (n_t, d)
        Counterfactual position of each treated cell.
    coupling : ndarray, shape (n_t, n_c)
        The transport plan, row-normalised to sum to 1 per treated cell.
    """
    if not POT_AVAILABLE:
        raise ImportError(
            "Optimal transport requires POT. Install with: pip install pot"
        )

    x_treated = np.asarray(x_treated, dtype=float)
    x_control = np.asarray(x_control, dtype=float)
    if x_treated.ndim != 2 or x_control.ndim != 2:
        raise ValueError("x_treated and x_control must both be 2-D.")
    if x_treated.shape[1] != x_control.shape[1]:
        raise ValueError(
            f"Dimension mismatch: treated has {x_treated.shape[1]} dims, "
            f"control has {x_control.shape[1]}."
        )
    if x_treated.shape[0] == 0 or x_control.shape[0] == 0:
        raise ValueError("Both groups must be non-empty.")
    if reg <= 0:
        raise ValueError(f"reg must be positive, got {reg}.")

    n_t, n_c = x_treated.shape[0], x_control.shape[0]
    cost = pot.dist(x_treated, x_control, metric="sqeuclidean")

    mean_cost = float(cost.mean())
    if mean_cost <= 0:
        # Every treated cell coincides with every control cell: displacement is
        # exactly zero, and Sinkhorn would divide by zero. Return the uniform
        # coupling, whose projection is the control mean — the right answer here.
        coupling = np.full((n_t, n_c), 1.0 / n_c)
        return coupling @ x_control, coupling

    a = np.full(n_t, 1.0 / n_t)
    b = np.full(n_c, 1.0 / n_c)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coupling = pot.sinkhorn(a, b, cost / mean_cost, reg=reg, numItermax=max_iter)

    row_sums = coupling.sum(axis=1, keepdims=True)
    # A row can underflow to zero for an extreme outlier at small reg.  Fall
    # back to the uniform coupling for those rows so the projection stays
    # defined instead of producing NaN and silently poisoning the mean.
    bad = (row_sums[:, 0] <= 1e-300) | ~np.isfinite(row_sums[:, 0])
    if bad.any():
        coupling[bad] = 1.0 / n_c
        row_sums = coupling.sum(axis=1, keepdims=True)
    coupling = coupling / row_sums

    return coupling @ x_control, coupling


# ============================================================================
# Main entry point
# ============================================================================


def _decompose_one_group(
    x_treated: np.ndarray,
    x_control: np.ndarray,
    pt_treated: np.ndarray,
    pt_control: np.ndarray,
    reg: float,
    bandwidth: Optional[float],
) -> Dict[str, np.ndarray]:
    """Per-cell delay / divergence for one timepoint. See module docstring."""
    projection, _ = ot_barycentric_projection(x_treated, x_control, reg=reg)
    displacement = x_treated - projection

    # Tangent evaluated at the *counterfactual* pseudotime is not available (the
    # projection has no pseudotime of its own), so use the treated cell's own
    # pseudotime.  The tangent field is smooth on the scale of the displacement,
    # so this is a second-order difference.
    tangents = estimate_trajectory_tangent(
        x_control, pt_control, pt_treated, bandwidth=bandwidth
    )

    along = np.einsum("ij,ij->i", displacement, tangents)
    parallel = along[:, None] * tangents
    orthogonal = displacement - parallel

    # Sign convention: positive delay = treated cell sits BEHIND its
    # counterfactual, i.e. displaced against the direction of maturation.
    delay = -along
    divergence = np.linalg.norm(orthogonal, axis=1)

    # Rows with an indeterminate tangent get no decomposition. All the
    # displacement is unattributable, so mark them rather than crediting it to
    # divergence (which is what an all-zero tangent would silently do).
    valid = np.linalg.norm(tangents, axis=1) > 0.5  # unit vectors or zero
    delay[~valid] = np.nan
    divergence[~valid] = np.nan

    return {
        "delay": delay,
        "divergence": divergence,
        "displacement_norm": np.linalg.norm(displacement, axis=1),
        "valid": valid,
    }


def decompose_perturbation_ot(
    adata: AnnData,
    pseudotime_key: str = "hypergraph_pseudotime",
    condition_key: str = "condition",
    control_label: str = "control",
    drug_label: str = "drug",
    timepoint_key: Optional[str] = None,
    latent_key: str = "X_pca",
    n_latent_dims: Optional[int] = 30,
    reg: float = 0.05,
    tangent_bandwidth: Optional[float] = None,
    n_permutations: int = 200,
    random_state: int = 42,
    key_added: str = "ot_decomposition",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Split a perturbation's effect into delay along and divergence off the trajectory.

    Couples treated to control cells with entropic OT within each timepoint,
    then decomposes every treated cell's displacement from its counterfactual
    into a component along the local trajectory tangent (**delay**) and a
    component orthogonal to it (**divergence**).

    Unlike the marginal distances in
    :func:`~pseudoembed.analysis.trajectory_differential_dynamics.compute_trajectory_divergence`,
    the two components here come from a single coupling and are therefore
    directly comparable — a perturbation can be summarised as a delay/divergence
    ratio.  Unlike the eigenvector classifier, this detects pure delay, which
    creates no new eigenvector.

    Parameters
    ----------
    adata : AnnData
        Must carry pseudotime in ``obs`` and a latent embedding in ``obsm``.
    pseudotime_key, condition_key : str
        Columns in ``adata.obs``.
    control_label, drug_label : str
        Values within ``condition_key``.
    timepoint_key : str, optional
        Column in ``adata.obs`` giving the experimental timepoint.  Coupling is
        solved **within** each timepoint, so that a treated cell is never matched
        to a control cell harvested days apart — which would confound the
        treatment effect with ordinary development.  ``None`` pools all cells
        into one group; only appropriate for a single-timepoint experiment.
    latent_key : str
        Key in ``adata.obsm``.
    n_latent_dims : int, optional
        Truncate the embedding to this many dimensions.  ``None`` uses all.
        Truncation matters here: OT cost in high dimensions is dominated by noise
        dimensions, which inflates apparent divergence.
    reg : float
        Entropic regularisation as a fraction of mean transport cost.
    tangent_bandwidth : float, optional
        Pseudotime window for the tangent fit.  Default 10% of the range.
    n_permutations : int
        Condition-label permutations, shuffled **within timepoint**, for the null
        distribution.  0 skips the test and leaves p-values as NaN.
    random_state : int
        Seed.
    key_added : str
        Per-cell results are written to ``adata.obs[f'{key_added}_delay']`` and
        ``..._divergence`` (NaN for control cells, which have no counterfactual),
        and the summary to ``adata.uns[key_added]``.

    Returns
    -------
    pd.DataFrame
        One row per timepoint plus an ``'all'`` row, with columns:
        ``timepoint``, ``n_control``, ``n_drug``, ``mean_delay``,
        ``mean_divergence`` (raw orthogonal norm — see the warning below),
        ``excess_divergence``, ``divergence_z``, ``median_delay``,
        ``median_divergence``, ``delay_fraction``, ``mean_displacement``,
        ``delay_pval``, ``divergence_pval``, ``n_valid``.

    Notes
    -----
    **Use ``excess_divergence``, not ``mean_divergence``.**  The raw orthogonal
    norm has a noise floor that grows with latent dimension (measured on a
    zero-effect null contrast: 0.18 at 2 PCs, 2.36 at 10, 6.05 at 30), because it
    is a magnitude over ``d − 1`` dimensions that noise can only inflate.  At 15
    PCs a genuinely diverging perturbation scored *below* its own null floor, so
    the raw number is not merely offset — it is inverted.  ``excess_divergence``
    subtracts the permutation null and ``divergence_z`` expresses that excess in
    null standard deviations.  ``mean_divergence`` is retained only for
    transparency.  ``mean_delay`` needs no such correction: it is a signed
    projection onto one direction, so noise cancels in the mean.

    ``delay_fraction`` is the interpretive headline:
    ``|delay| / (|delay| + max(excess_divergence, 0))``.  Near 1 means an
    essentially pure schedule shift, near 0 means the cells left the trajectory.
    Both terms are null-calibrated effect sizes, so the ratio is comparable
    across datasets and latent dimensionalities.  It is NaN when
    ``n_permutations=0``, since without a null there is nothing to calibrate
    against.
    """
    if not POT_AVAILABLE:
        raise ImportError(
            "decompose_perturbation_ot requires POT. Install with: pip install pot"
        )
    for key in (pseudotime_key, condition_key):
        if key not in adata.obs:
            raise ValueError(f"'{key}' not found in adata.obs.")
    if timepoint_key is not None and timepoint_key not in adata.obs:
        raise ValueError(f"'{timepoint_key}' not found in adata.obs.")
    if latent_key not in adata.obsm:
        raise ValueError(f"'{latent_key}' not found in adata.obsm.")

    latent = adata.obsm[latent_key]
    if isinstance(latent, pd.DataFrame):
        latent = latent.values
    latent = np.asarray(latent, dtype=float)
    if n_latent_dims is not None:
        latent = latent[:, :n_latent_dims]

    pseudotime = np.asarray(adata.obs[pseudotime_key].values, dtype=float)
    conditions = np.asarray(adata.obs[condition_key].values).astype(str)

    for label in (control_label, drug_label):
        if not np.any(conditions == label):
            raise ValueError(
                f"No cells with {condition_key}=={label!r}. "
                f"Present: {sorted(set(conditions))}"
            )

    if timepoint_key is None:
        groups = np.zeros(adata.n_obs, dtype=object)
        groups[:] = "all"
    else:
        groups = np.asarray(adata.obs[timepoint_key].values).astype(str)

    rng = np.random.default_rng(random_state)

    if verbose:
        print(f"\n{'=' * 70}")
        print("OT DELAY / DIVERGENCE DECOMPOSITION")
        print(f"{'=' * 70}")
        print(f"  Cells        : {adata.n_obs}  |  latent dims: {latent.shape[1]}")
        print(f"  Pseudotime   : {pseudotime_key!r}")
        print(f"  Conditions   : {control_label!r} vs {drug_label!r}")
        print(f"  Timepoints   : {timepoint_key!r}")
        print(f"  reg          : {reg}  |  permutations: {n_permutations}")

    delay_all = np.full(adata.n_obs, np.nan)
    diverge_all = np.full(adata.n_obs, np.nan)
    disp_all = np.full(adata.n_obs, np.nan)

    rows: List[dict] = []
    for grp in pd.unique(groups):
        in_grp = groups == grp
        t_mask = in_grp & (conditions == drug_label)
        c_mask = in_grp & (conditions == control_label)
        n_t, n_c = int(t_mask.sum()), int(c_mask.sum())
        if n_t == 0 or n_c == 0:
            if verbose:
                print(
                    f"  [skip] {grp}: n_drug={n_t} n_control={n_c} "
                    "— needs both arms."
                )
            continue

        res = _decompose_one_group(
            latent[t_mask],
            latent[c_mask],
            pseudotime[t_mask],
            pseudotime[c_mask],
            reg=reg,
            bandwidth=tangent_bandwidth,
        )
        t_idx = np.flatnonzero(t_mask)
        delay_all[t_idx] = res["delay"]
        diverge_all[t_idx] = res["divergence"]
        disp_all[t_idx] = res["displacement_norm"]

        obs_delay = float(np.nanmean(res["delay"]))
        obs_diverge = float(np.nanmean(res["divergence"]))

        # ---- permutation null: shuffle condition WITHIN this group ----------
        p_delay = p_diverge = np.nan
        if n_permutations > 0:
            x_grp = latent[in_grp]
            pt_grp = pseudotime[in_grp]
            n_grp = x_grp.shape[0]
            null_delay = np.empty(n_permutations)
            null_diverge = np.empty(n_permutations)
            for p in range(n_permutations):
                perm = rng.permutation(n_grp)
                fake_t = perm[:n_t]
                fake_c = perm[n_t:]
                r = _decompose_one_group(
                    x_grp[fake_t],
                    x_grp[fake_c],
                    pt_grp[fake_t],
                    pt_grp[fake_c],
                    reg=reg,
                    bandwidth=tangent_bandwidth,
                )
                null_delay[p] = np.nanmean(r["delay"])
                null_diverge[p] = np.nanmean(r["divergence"])
            # Two-sided for delay (sign is meaningful — a treatment can push
            # cells forward as well as back); one-sided for divergence, whose
            # null is a positive magnitude and can only be exceeded.
            p_delay = float(
                (np.sum(np.abs(null_delay) >= abs(obs_delay)) + 1)
                / (n_permutations + 1)
            )
            p_diverge = float(
                (np.sum(null_diverge >= obs_diverge) + 1) / (n_permutations + 1)
            )

        # Null-calibrate divergence.  MEASURED (2026-08-07, synthetic Th1): the
        # raw orthogonal norm is unusable on its own.  It is a magnitude over
        # d-1 dimensions, so latent noise can only ever inflate it, and the floor
        # grows with dimension -- mean divergence on a zero-effect null contrast
        # was 0.18 at 2 PCs, 2.36 at 10, and 6.05 at 30.  At 15 PCs the true
        # divergence regime scored 3.14 against a null floor of 3.57, i.e. the
        # raw statistic is not merely offset but *inverted* by noise.  Delay does
        # not need this: it is a signed projection onto ONE direction, so noise
        # cancels in the mean (null delay stayed ~0.00 at every dimension while
        # the real contrast held +0.28 to +0.36).
        #
        # So report divergence as EXCESS over the permutation null, in units of
        # the null's own spread.  That makes it comparable across latent
        # dimensionalities and gives it the signed, zero-centred behaviour delay
        # already has.
        excess_div = np.nan
        z_div = np.nan
        if n_permutations > 0:
            null_mean = float(np.mean(null_diverge))
            null_sd = float(np.std(null_diverge))
            excess_div = obs_diverge - null_mean
            if null_sd > 1e-12:
                z_div = excess_div / null_sd

        # delay_fraction uses the NULL-CORRECTED divergence, not the raw norm.
        # Dividing by the raw total displacement would put the whole latent noise
        # budget in the denominator, driving the fraction toward 0 in every
        # regime (measured: 0.057-0.089 across all four regimes at 15 PCs,
        # including pure delay, where the truth is ~1). Using excess divergence
        # makes the ratio a comparison of two calibrated effect sizes.
        excess_div_pos = max(excess_div, 0.0) if np.isfinite(excess_div) else np.nan
        denom = abs(obs_delay) + excess_div_pos
        delay_fraction = (
            abs(obs_delay) / denom if np.isfinite(denom) and denom > 0 else np.nan
        )
        rows.append(
            {
                "timepoint": grp,
                "n_control": n_c,
                "n_drug": n_t,
                "mean_delay": obs_delay,
                "mean_divergence": obs_diverge,
                "excess_divergence": excess_div,
                "divergence_z": z_div,
                "median_delay": float(np.nanmedian(res["delay"])),
                "median_divergence": float(np.nanmedian(res["divergence"])),
                "delay_fraction": delay_fraction,
                "mean_displacement": float(np.nanmean(res["displacement_norm"])),
                "delay_pval": p_delay,
                "divergence_pval": p_diverge,
                "n_valid": int(np.sum(res["valid"])),
            }
        )

        if verbose:
            print(
                f"  {grp!s:>10}: delay={obs_delay:+.4f} (p={p_delay:.3g})  "
                f"excess_div={excess_div:+.4f} (z={z_div:+.2f}, "
                f"p={p_diverge:.3g})"
            )

    if not rows:
        raise ValueError(
            "No timepoint contained both a control and a treated arm — "
            "nothing to decompose."
        )

    df = pd.DataFrame(rows)

    # Pooled row.  Per-cell means where the quantity is per-cell (delay,
    # divergence, displacement); the null-calibrated quantities are averaged over
    # timepoints instead, because each timepoint has its OWN null floor -- the
    # floor depends on that timepoint's within-group spread, so pooling raw
    # divergences and subtracting a single null would mix incompatible baselines.
    def _nanmean(values: np.ndarray) -> float:
        """nanmean that returns NaN for an all-NaN input without warning.

        The calibrated columns are entirely NaN when n_permutations=0, which is a
        supported call, not an anomaly — so it should not emit a RuntimeWarning.
        """
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else float("nan")

    pooled_delay = _nanmean(delay_all)
    pooled_excess = _nanmean(df["excess_divergence"].to_numpy(dtype=float))
    pooled_excess_pos = (
        max(pooled_excess, 0.0) if np.isfinite(pooled_excess) else np.nan
    )
    pooled_denom = abs(pooled_delay) + pooled_excess_pos
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "timepoint": "all",
                        "n_control": int(np.sum(conditions == control_label)),
                        "n_drug": int(np.sum(conditions == drug_label)),
                        "mean_delay": pooled_delay,
                        "mean_divergence": float(np.nanmean(diverge_all)),
                        "excess_divergence": pooled_excess,
                        "divergence_z": _nanmean(
                            df["divergence_z"].to_numpy(dtype=float)
                        ),
                        "median_delay": float(np.nanmedian(delay_all)),
                        "median_divergence": float(np.nanmedian(diverge_all)),
                        "delay_fraction": (
                            abs(pooled_delay) / pooled_denom
                            if np.isfinite(pooled_denom) and pooled_denom > 0
                            else np.nan
                        ),
                        "mean_displacement": float(np.nanmean(disp_all)),
                        "delay_pval": np.nan,  # pooling p-values is not meaningful; combine
                        "divergence_pval": np.nan,  # the per-timepoint tests instead.
                        "n_valid": int(np.sum(np.isfinite(delay_all))),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    adata.obs[f"{key_added}_delay"] = delay_all
    adata.obs[f"{key_added}_divergence"] = diverge_all
    adata.obs[f"{key_added}_displacement"] = disp_all
    adata.uns[key_added] = {
        "summary": df,
        "params": {
            "pseudotime_key": pseudotime_key,
            "condition_key": condition_key,
            "control_label": control_label,
            "drug_label": drug_label,
            "timepoint_key": timepoint_key,
            "latent_key": latent_key,
            "n_latent_dims": n_latent_dims,
            "reg": reg,
            "tangent_bandwidth": tangent_bandwidth,
            "n_permutations": n_permutations,
            "random_state": random_state,
        },
    }

    if verbose:
        pooled = df.iloc[-1]
        print(f"\n  POOLED delay_fraction = {pooled['delay_fraction']:.3f}")
        print(
            "    -> "
            + (
                "predominantly a SCHEDULE SHIFT (delay)"
                if pooled["delay_fraction"] > 0.5
                else "predominantly OFF-TRAJECTORY (divergence)"
            )
        )

    return df
