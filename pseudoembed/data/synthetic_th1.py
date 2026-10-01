"""
Synthetic CD4+ T cell activation → Th1 differentiation under JAK inhibition.

Why this system
---------------
The dataset PseudoEmbed actually needs does not exist publicly: a longitudinal
control-vs-condition design where the *true* per-cell trajectory position and the
*true* drug effect are both known.  This module generates one.  Every gene is a
real HGNC symbol, so PROGENy and CollecTRI score it exactly as they would score
real data — the pathway and TF views are not mock-ups, they run on real priors.

The biology is chosen because the perturbation is unambiguous in the literature.
JAK1/JAK3 inhibition (tofacitinib and relatives, licensed in RA and UC) blocks
signalling downstream of the γc cytokine receptors:

* IL-2 → JAK1/JAK3 → STAT5, and IL-12 → JAK2/TYK2 → STAT4, are both
  interrupted, so the STAT1/STAT4-driven arm of Th1 commitment is impaired.
* TBX21 (T-bet) induction depends on STAT1/STAT4, so IFNG and the cytotoxic
  module (GZMB, PRF1, CCL5) fail to reach control levels.
* The immediate-early activation module (CD69, TNFRSF9, IL2RA, MKI67) is driven
  by TCR signalling *upstream* of JAK, so it is largely spared — this asymmetry
  is what makes the perturbation a *partial* block rather than a global shutdown,
  and it is the reason the effect is interesting rather than trivial.
* SOCS1/SOCS3/CISH are direct STAT targets and negative-feedback genes, so they
  drop with the pathway rather than compensating.

Two effect types, deliberately separated
----------------------------------------
The central methodological problem is that *delay along* a trajectory and
*divergence off* it are not distinguishable from snapshot data without a
coupling assumption.  A benchmark that mixes them cannot test whether a method
separates them.  So this generator encodes both, independently and with separate
ground-truth columns:

* ``true_delay`` — treated cells advance more slowly along the *same* latent
  axis.  A perfect pseudotime should place them at lower values at matched
  sampling time, with no new expression state.
* ``true_divergence`` — treated cells acquire an *orthogonal* expression
  component (a distinct off-manifold state), at matched latent position.

Setting ``delay_strength=0`` gives pure divergence; ``divergence_strength=0``
gives pure delay.  Both non-zero gives the realistic, hard case.  A method that
reports a single "condition effect" scalar cannot tell these apart, and this
dataset will show that.

Negative controls are built in
------------------------------
``control_only_split=True`` relabels the control arm into two fake conditions
with no real difference.  Any divergence detected there is a false positive.
This is the cheapest and most important sanity floor for a condition-effect
metric, and it costs one flag.

What is *not* claimed
---------------------
This is a generative model with a hand-built covariance structure, not a
simulation of transcriptional dynamics.  It does not model bursting, RNA
velocity, cell-cycle phase, ambient contamination, or doublets.  Absolute
effect sizes are set by the caller, not derived from data, so "method X recovers
80% of the effect" is a statement about this generator's parameters and does not
transfer to a real drug.  Its purpose is to test *whether a method can recover a
known answer*, not to predict real effect magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "GENE_PROGRAMMES",
    "SyntheticTh1Params",
    "make_synthetic_th1",
]


# ---------------------------------------------------------------------------
# Gene programmes — real HGNC symbols, grouped by how they behave along the
# trajectory and under JAK inhibition.
#
# `trajectory` is the shape along the naive → activated → Th1 axis:
#   "down"      monotone decreasing (naive identity lost)
#   "transient" peaks mid-trajectory (activation burst)
#   "up"        monotone increasing (effector programme acquired)
#
# `jak_dependent` marks genes whose induction requires JAK/STAT signalling and
# which therefore fail under inhibition.  Genes driven by TCR signalling
# upstream of JAK are marked False and are *spared*, which is the biologically
# meaningful asymmetry.
# ---------------------------------------------------------------------------
GENE_PROGRAMMES: Dict[str, Dict] = {
    # Naive / quiescent identity — lost on activation, JAK-independent.
    "naive": {
        "genes": [
            "SELL",
            "CCR7",
            "TCF7",
            "LEF1",
            "IL7R",
            "FOXP1",
            "BACH2",
            "KLF2",
            "S1PR1",
            "TXK",
        ],
        "trajectory": "down",
        "jak_dependent": False,
    },
    # Immediate-early TCR activation — JAK-INDEPENDENT, so spared by the drug.
    # This is the asymmetry that makes the perturbation partial.
    "activation_early": {
        "genes": [
            "CD69",
            "TNFRSF9",
            "NR4A1",
            "EGR1",
            "EGR2",
            "FOS",
            "JUN",
            "IL2RA",
            "CD40LG",
            "TNFRSF4",
        ],
        "trajectory": "transient",
        "jak_dependent": False,
    },
    # Proliferation — partially JAK-dependent (IL-2 drives clonal expansion).
    "proliferation": {
        "genes": [
            "MKI67",
            "TOP2A",
            "CCNB1",
            "CDK1",
            "PCNA",
            "TYMS",
            "RRM2",
            "UBE2C",
            "BIRC5",
            "AURKB",
        ],
        "trajectory": "transient",
        "jak_dependent": True,
    },
    # JAK/STAT signalling machinery and direct negative-feedback targets.
    # These are the most directly drug-sensitive genes in the design.
    "jak_stat_core": {
        "genes": [
            "STAT1",
            "STAT4",
            "SOCS1",
            "SOCS3",
            "CISH",
            "JAK1",
            "JAK3",
            "IL2RB",
            "IL12RB2",
            "IFNGR1",
            "IL2RG",
            "TYK2",
        ],
        "trajectory": "up",
        "jak_dependent": True,
    },
    # Th1 effector identity — depends on STAT1/STAT4 → TBX21, so JAK-dependent.
    "th1_effector": {
        "genes": ["TBX21", "IFNG", "CXCR3", "IL12RB1", "CCL4", "CCL5", "KLRG1", "TNF"],
        "trajectory": "up",
        "jak_dependent": True,
    },
    # Cytotoxic module — downstream of Th1 commitment, JAK-dependent.
    "cytotoxic": {
        "genes": ["GZMB", "GZMA", "GZMK", "PRF1", "NKG7", "GNLY", "CST7", "CTSW"],
        "trajectory": "up",
        "jak_dependent": True,
    },
    # Interferon-stimulated genes — strongly JAK-dependent (canonical readout
    # of the pathway; these are the PROGENy JAK-STAT top targets).
    "isg": {
        "genes": [
            "ISG15",
            "IFIT1",
            "IFIT3",
            "MX1",
            "OAS1",
            "STAT2",
            "IRF7",
            "RSAD2",
            "IFI6",
            "IFI44L",
        ],
        "trajectory": "up",
        "jak_dependent": True,
    },
    # Alternative-fate / regulatory genes — the OFF-MANIFOLD divergence
    # programme.  Under JAK inhibition, blocked Th1 cells drift toward a
    # regulatory/exhausted-like state rather than simply lagging.  This is what
    # `true_divergence` drives, and it is what makes the drug effect more than
    # a delay.
    "divergence": {
        "genes": [
            "FOXP3",
            "CTLA4",
            "IKZF2",
            "TIGIT",
            "LAG3",
            "PDCD1",
            "HAVCR2",
            "TOX",
            "ENTPD1",
            "IL10",
            "GATA3",
            "RORC",
        ],
        "trajectory": "divergence",
        "jak_dependent": False,
    },
}

# Housekeeping / structural genes: expressed, trajectory-independent.  Present so
# that HVG selection and normalisation behave like they do on real data, and so
# a method cannot succeed by assuming every gene is informative.
_HOUSEKEEPING: List[str] = [
    "ACTB",
    "GAPDH",
    "B2M",
    "TMSB4X",
    "RPL13A",
    "RPS18",
    "RPLP0",
    "EEF1A1",
    "PTMA",
    "HSP90AB1",
    "UBC",
    "PPIA",
    "YWHAZ",
    "SDHA",
    "TBP",
    "PGK1",
    "RPL10",
    "RPS6",
    "RPL41",
    "TPT1",
    "OAZ1",
    "NACA",
    "BTF3",
    "SRP14",
    "MYL6",
    "CFL1",
    "ARPC2",
    "COTL1",
    "GABARAP",
    "SH3BGRL3",
]


@dataclass
class SyntheticTh1Params:
    """
    Parameters for :func:`make_synthetic_th1`.

    Effect sizes are on the latent scale and are *set*, not fitted, so they are
    the ground truth by construction rather than an estimate.
    """

    n_cells_per_group: int = 400
    timepoints: Tuple[float, ...] = (0.0, 24.0, 48.0, 72.0)
    #: Fraction of the latent axis a treated cell fails to advance.  0.35 means
    #: treated cells reach ~65% as far along the trajectory at matched time.
    delay_strength: float = 0.35
    #: Magnitude of the orthogonal (off-manifold) treated-only component.
    divergence_strength: float = 0.6
    #: Divergence only appears once cells are far enough along to commit; before
    #: this latent position there is nothing to divert.  Mirrors the biology:
    #: the drug cannot redirect a fate that has not begun.
    divergence_onset: float = 0.35
    n_hvg_noise_genes: int = 250
    #: Per-cell, per-gene multiplicative expression jitter (CV).  Applies to the
    #: structured genes so cells at the same latent position are not identical.
    #: Does not perturb the recorded ground truth.
    biological_noise: float = 0.30
    dropout_rate: float = 0.35
    library_size: float = 8000.0
    #: Negative control: split the control arm into two fake conditions with no
    #: real difference.  Any effect found is a false positive.
    control_only_split: bool = False
    #: Batch/donor effect magnitude.  Non-zero exercises the confound screening
    #: that the driver analysis relies on.
    n_donors: int = 3
    donor_effect: float = 0.25
    seed: int = 0
    #: Per-programme effect overrides, e.g. {"isg": 1.5}.
    programme_scale: Dict[str, float] = field(default_factory=dict)


#: Resting expression level of the divergence-programme genes.  Set to the same
#: level as housekeeping genes (0.5) so they are unremarkable at baseline and only
#: the treated-cell increase distinguishes them.
_DIVERGENCE_BASELINE = 0.5


def _trajectory_shape(kind: str, s: np.ndarray) -> np.ndarray:
    """Latent expression profile along normalised trajectory position s∈[0,1]."""
    if kind == "down":
        return 1.0 - s
    if kind == "up":
        return s
    if kind == "transient":
        # Peaks at s=0.4; activation burst that resolves.
        return np.exp(-((s - 0.4) ** 2) / (2 * 0.18**2))
    if kind == "divergence":
        # Handled separately via the divergence axis, not the maturation axis.
        return np.zeros_like(s)
    raise ValueError(f"unknown trajectory kind {kind!r}")


def make_synthetic_th1(
    params: Optional[SyntheticTh1Params] = None,
    **overrides,
):
    """
    Generate the synthetic Th1 / JAK-inhibition AnnData.

    Returns
    -------
    adata : AnnData
        ``adata.X`` is log1p-normalised counts, ready for PROGENy/CollecTRI.
        Raw counts are kept in ``adata.layers["counts"]``.

        Ground truth in ``adata.obs``:

        ``true_latent``
            The real trajectory position s∈[0,1] **after** any treatment delay.
            This is what a perfect pseudotime should recover (up to a monotone
            transform), so it is the target for rank correlation.
        ``true_latent_undelayed``
            Position the cell *would* have had absent treatment — the
            counterfactual.  Unobservable in real data; the reason OT coupling
            is needed.  ``true_latent_undelayed - true_latent`` is the delay.
        ``true_delay``
            Per-cell delay magnitude (0 for controls).
        ``true_divergence``
            Per-cell off-manifold displacement (0 for controls).
        ``condition``, ``timepoint``, ``donor``, ``cell_state``

        ``adata.var`` carries ``programme``, ``trajectory``, ``jak_dependent``,
        and ``is_ground_truth_driver``.

        ``adata.uns["synthetic_th1"]`` records the full parameter set and the
        expected driver programmes, so a scoring function needs no external
        key file.
    """
    try:
        import anndata as ad
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("anndata and pandas are required") from exc

    p = params or SyntheticTh1Params()
    if overrides:
        p = SyntheticTh1Params(**{**p.__dict__, **overrides})
    rng = np.random.default_rng(p.seed)

    conditions = ["control", "treated"]
    n_per = p.n_cells_per_group
    n_tp = len(p.timepoints)

    # ---------------- assemble per-cell latent structure ----------------
    cond_list, tp_list, s_true, s_undelayed, div_list = [], [], [], [], []

    for cond in conditions:
        for tp in p.timepoints:
            n = n_per // n_tp
            # Latent position is driven by sampling time but deliberately broad:
            # a real timepoint contains a spread of differentiation states, so a
            # perfect pseudotime correlates with time only loosely.  This is the
            # property that stops `rho vs time == 1` being the target.
            t_frac = tp / max(p.timepoints)
            s0 = np.clip(
                rng.beta(2.0 + 4.0 * t_frac, 2.0 + 4.0 * (1 - t_frac), size=n),
                0.0,
                1.0,
            )

            if cond == "treated" and not p.control_only_split:
                # Delay: compress progress toward the root, growing with time
                # (the drug's cumulative effect).  Recorded both ways so the
                # counterfactual is available as ground truth.
                delay = p.delay_strength * t_frac
                s_obs = s0 * (1.0 - delay)
                # Divergence: only past the commitment onset, scaled by how far
                # along the cell is.
                past = np.clip(
                    (s_obs - p.divergence_onset) / max(1e-9, 1.0 - p.divergence_onset),
                    0.0,
                    1.0,
                )
                div = p.divergence_strength * past * t_frac
            else:
                s_obs = s0
                div = np.zeros(n)

            cond_list.append(np.full(n, cond, dtype=object))
            tp_list.append(np.full(n, tp))
            s_true.append(s_obs)
            s_undelayed.append(s0)
            div_list.append(div)

    condition = np.concatenate(cond_list)
    timepoint = np.concatenate(tp_list)
    s = np.concatenate(s_true)
    s_undel = np.concatenate(s_undelayed)
    divergence = np.concatenate(div_list)
    n_cells = s.size

    donor = rng.choice([f"donor{i + 1}" for i in range(p.n_donors)], size=n_cells)

    # Negative control: relabel controls into two arms with no real difference.
    if p.control_only_split:
        ctrl = condition == "control"
        fake = rng.choice(["fake_A", "fake_B"], size=int(ctrl.sum()))
        condition = condition.copy()
        condition[ctrl] = fake
        keep = ctrl
        condition, timepoint = condition[keep], timepoint[keep]
        s, s_undel, divergence = s[keep], s_undel[keep], divergence[keep]
        donor = donor[keep]
        n_cells = s.size

    # ---------------- build the expression matrix ----------------
    gene_names: List[str] = []
    gene_programme: List[str] = []
    gene_traj: List[str] = []
    gene_jak: List[bool] = []
    latent_cols: List[np.ndarray] = []

    is_treated = np.isin(condition, ["treated"])

    for prog_name, spec in GENE_PROGRAMMES.items():
        scale = p.programme_scale.get(prog_name, 1.0)
        for g in spec["genes"]:
            if spec["trajectory"] == "divergence":
                # Off-manifold: driven by the divergence axis only, so these
                # genes carry no maturation signal at all.
                #
                # They keep a non-zero BASELINE, because FOXP3/CTLA4/GATA3 etc.
                # are genuinely expressed in resting CD4+ T cells.  Without it
                # they would be identically zero in every control cell, and a
                # method could then flag the divergence programme purely from
                # "detected vs not detected" rather than from any structure --
                # an artifact that would inflate driver recall.
                base = _DIVERGENCE_BASELINE + divergence * 2.0 * scale
            else:
                base = _trajectory_shape(spec["trajectory"], s) * scale
                if spec["jak_dependent"]:
                    # JAK-dependent induction is attenuated in treated cells,
                    # beyond the delay already applied to s.  This is what makes
                    # the effect more than a shift along the axis.
                    atten = np.where(is_treated, 1.0 - 0.55 * (divergence > 0), 1.0)
                    base = base * atten
            gene_names.append(g)
            gene_programme.append(prog_name)
            gene_traj.append(spec["trajectory"])
            gene_jak.append(bool(spec["jak_dependent"]))
            latent_cols.append(base)

    for g in _HOUSEKEEPING:
        gene_names.append(g)
        gene_programme.append("housekeeping")
        gene_traj.append("flat")
        gene_jak.append(False)
        latent_cols.append(np.full(n_cells, 0.5))

    # Unstructured genes so HVG selection has something to reject.  Named with
    # a clear synthetic prefix so they are never mistaken for real symbols.
    for i in range(p.n_hvg_noise_genes):
        gene_names.append(f"SYNTHNOISE{i:04d}")
        gene_programme.append("noise")
        gene_traj.append("none")
        gene_jak.append(False)
        latent_cols.append(rng.normal(0.3, 0.12, size=n_cells).clip(0))

    L = np.vstack(latent_cols).T  # (cells, genes)

    # Per-cell biological variability on the structured genes.  Without this the
    # only randomness is Poisson sampling, so every cell at a given latent
    # position is transcriptionally identical up to shot noise -- kNN graphs then
    # look far cleaner than on real data and the benchmark is optimistic.  This
    # is a gene-level jitter, so it does not move the cell's latent position and
    # therefore does not corrupt the ground truth in obs.
    if p.biological_noise > 0:
        L = L * rng.normal(1.0, p.biological_noise, size=L.shape).clip(0.0)

    # Donor effect: a multiplicative per-donor, per-gene offset.  Present so the
    # confound screening in the driver analysis has something real to screen.
    if p.donor_effect > 0:
        for d in np.unique(donor):
            m = donor == d
            shift = rng.normal(1.0, p.donor_effect, size=L.shape[1]).clip(0.2)
            L[m] = L[m] * shift[None, :]

    # ---------------- latent → counts ----------------
    # Log-normal mean structure, Poisson sampling, then multiplicative dropout.
    mu = np.exp(1.2 * L) - 1.0
    mu = np.clip(mu, 1e-6, None)
    mu = mu / mu.sum(axis=1, keepdims=True) * p.library_size
    counts = rng.poisson(mu).astype(np.float32)
    if p.dropout_rate > 0:
        counts *= rng.random(counts.shape) > p.dropout_rate

    var = pd.DataFrame(
        {
            "programme": gene_programme,
            "trajectory": gene_traj,
            "jak_dependent": gene_jak,
        },
        index=pd.Index(gene_names, name=None),
    )
    # A gene is a ground-truth driver if the perturbation actually changes it:
    # JAK-dependent genes (attenuated) and the divergence programme.
    var["is_ground_truth_driver"] = var["jak_dependent"] | (
        var["programme"] == "divergence"
    )

    state = np.where(
        s < 0.25,
        "naive",
        np.where(
            s < 0.55, "activated", np.where(s < 0.8, "early_effector", "th1_effector")
        ),
    )

    obs = pd.DataFrame(
        {
            "condition": pd.Categorical(condition),
            "timepoint": timepoint,
            "donor": pd.Categorical(donor),
            "cell_state": pd.Categorical(state),
            "true_latent": s,
            "true_latent_undelayed": s_undel,
            "true_delay": s_undel - s,
            "true_divergence": divergence,
        },
        index=pd.Index([f"cell{i:05d}" for i in range(n_cells)]),
    )

    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.layers["counts"] = counts.copy()

    # log1p-normalise in place so PROGENy/CollecTRI receive what they expect.
    lib = adata.X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    adata.X = np.log1p(adata.X / lib * 1e4).astype(np.float32)

    adata.uns["synthetic_th1"] = {
        "params": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in p.__dict__.items()
        },
        # What a correct driver analysis should recover, for automated scoring.
        "expected_driver_programmes": [
            "jak_stat_core",
            "th1_effector",
            "cytotoxic",
            "isg",
            "proliferation",
            "divergence",
        ],
        "expected_spared_programmes": [
            "naive",
            "activation_early",
            "housekeeping",
            "noise",
        ],
        # Measured on seed=0 with decoupler ULM: JAK-STAT is the single largest
        # drop (delta -0.97, p~5e-24), then NFkB and TNFa.  Listed in the order
        # a correct analysis should rank them.
        "expected_pathways_down": ["JAK-STAT", "NFkB", "TNFa"],
        # Direction matters, so these are split.  Blocked STAT signalling pulls
        # the STAT/IRF module down; the divergence programme pushes the
        # regulatory TFs up.  The IRF3/IRF5/IRF9/STAT2 interferon module also
        # drops (it shares the ISG target set) -- expected, not a confound.
        "expected_tfs_down": [
            "STAT1",
            "STAT2",
            "STAT4",
            "TBX21",
            "IRF1",
            "IRF3",
            "IRF5",
            "IRF9",
        ],
        # Only FOXP3 survives as a robust up-call (rank 5 of 107 at seed=0).
        # GATA3/RBPJ shift in the right direction but are near-zero once
        # biological_noise is applied -- the divergence programme is only 12
        # genes, so its TF-level signal is intrinsically weaker than the
        # STAT/IRF drop.  Listing them as expected would make a correct
        # analysis look like it had missed something.
        "expected_tfs_up": ["FOXP3"],
        "is_negative_control": bool(p.control_only_split),
        "root_rule": "argmin(true_latent) — exact, since the latent is known",
    }
    return adata
