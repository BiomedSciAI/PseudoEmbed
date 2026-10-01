"""Main CLI application for PseudoEmbed."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import typer
from typing_extensions import Annotated

from pseudoembed.core.io import load_raw_data

app = typer.Typer(
    name="pseudoembed",
    help="PseudoEmbed: Pseudotime analysis with embedding integration",
    add_completion=False,
)


@app.command()
def preprocess(
    input_path: Annotated[str, typer.Option(help="Path to raw h5ad file")],
    output_path: Annotated[
        str, typer.Option(help="Path to save preprocessed h5ad file")
    ],
    experiment_filter: Annotated[
        Optional[str], typer.Option(help="Experiment name to exclude")
    ] = None,
    min_genes: Annotated[int, typer.Option(help="Minimum genes per cell")] = 100,
    min_cells: Annotated[int, typer.Option(help="Minimum cells per gene")] = 3,
    n_top_genes: Annotated[
        int, typer.Option(help="Number of highly variable genes")
    ] = 5000,
    batch_key: Annotated[
        str, typer.Option(help="Key in obs for batch correction")
    ] = "experiment",
) -> None:
    """
    Preprocess raw single-cell data.

    Example:
        pseudoembed preprocess --input-path raw.h5ad --output-path processed.h5ad
    """
    typer.echo(f"Loading raw data from {input_path}...")

    adata = load_raw_data(
        adata_file=input_path,
        experiment_filter=experiment_filter,
        min_genes=min_genes,
        min_cells=min_cells,
        n_top_genes=n_top_genes,
        batch_key=batch_key,
    )

    adata.write_h5ad(output_path)
    typer.echo(f"✓ Saved preprocessed data to {output_path}")
    typer.echo(f"  Shape: {adata.shape}")
    typer.echo(f"  Highly variable genes: {adata.var['highly_variable'].sum()}")


@app.command()
def hypergraph_analysis(
    data_path: Annotated[str, typer.Option(help="Path to input h5ad file")],
    output_dir: Annotated[
        str, typer.Option(help="Directory to write results, figures, and h5ad")
    ],
    condition_key: Annotated[
        str, typer.Option(help="Column in adata.obs with condition labels")
    ] = "condition",
    control_label: Annotated[
        str, typer.Option(help="Label for the control condition")
    ] = "control",
    drug_label: Annotated[
        str, typer.Option(help="Label for the drug/treatment condition")
    ] = "drug",
    time_key: Annotated[
        str, typer.Option(help="Column in adata.obs with ordered time-point labels")
    ] = "timepoint",
    organism: Annotated[
        str, typer.Option(help="Organism for TF/pathway networks: 'human' or 'mouse'")
    ] = "human",
    root_markers: Annotated[
        Optional[str],
        typer.Option(
            help="Marker genes for progenitor root cells (comma-separated, e.g. 'PDGFRA,CSPG4,OLIG1')"
        ),
    ] = None,
    top_n_tfs: Annotated[
        Optional[int],
        typer.Option(help="Number of most variable TFs to use (None = all)"),
    ] = None,
    z_score_threshold: Annotated[
        float, typer.Option(help="Z-score threshold for TF activity edges")
    ] = 1.5,
    n_states: Annotated[
        int, typer.Option(help="Number of cell states to discover")
    ] = 5,
    n_spectral_components: Annotated[
        int, typer.Option(help="Number of spectral components for pseudotime embedding")
    ] = 10,
    spectral_time_scale: Annotated[
        int,
        typer.Option(
            help="Eigenvalue scaling exponent for spectral embedding (used with --embedding-scale power)"
        ),
    ] = 4,
    manifold_k: Annotated[
        int,
        typer.Option(help="k for the kNN manifold graph used in geodesic pseudotime"),
    ] = 15,
    embedding_scale: Annotated[
        str,
        typer.Option(
            help="Eigenvalue weighting for spectral embedding: 'dpt' (lambda/(1-lambda)), 'power' (lambda^t), or 'unit' (raw)"
        ),
    ] = "dpt",
    min_root_corr: Annotated[
        float,
        typer.Option(
            help="Minimum |Spearman rho| for an eigenvector to be kept as a maturation axis (default 0.05)"
        ),
    ] = 0.05,
    pseudotime_method: Annotated[
        str, typer.Option(help="Pseudotime metric: 'geodesic' or 'chord'")
    ] = "geodesic",
    fusion_method: Annotated[
        str,
        typer.Option(help="View fusion strategy: 'operator' (late) or 'edge' (early)"),
    ] = "operator",
    tf_weight: Annotated[float, typer.Option(help="Weight for the TF view")] = 0.0,
    pathway_weight: Annotated[
        float, typer.Option(help="Weight for the pathway view")
    ] = 0.0,
    pca_weight: Annotated[
        float, typer.Option(help="Weight for the PCA expression-manifold view")
    ] = 0.0,
    n_pca_dims: Annotated[
        int, typer.Option(help="Number of PCA dimensions to use for the PCA view")
    ] = 30,
    cell_type_key: Annotated[
        Optional[str],
        typer.Option(
            help="adata.obs column with cell-type labels for pseudotime QC strip plot (Panel E)"
        ),
    ] = None,
    pseudotime_key: Annotated[
        str, typer.Option(help="adata.obs key where pseudotime is stored after step 1")
    ] = "multiview_pseudotime",
    latent_key: Annotated[
        str,
        typer.Option(
            help="adata.obsm key for latent embedding used by divergence step (default: hypergraph spectral embedding; use 'X_pca' to revert to PCA)"
        ),
    ] = "multiview_pseudotime_spectral",
    n_bins: Annotated[
        int, typer.Option(help="Number of pseudotime bins for trajectory divergence")
    ] = 10,
    n_permutations: Annotated[
        int, typer.Option(help="Permutations for divergence significance testing")
    ] = 1000,
    feature_source: Annotated[
        str,
        typer.Option(
            help="Feature source for differential dynamics: 'X' or 'tf_activities'"
        ),
    ] = "tf_activities",
    n_top_features: Annotated[
        int,
        typer.Option(
            help="Max top differentially dynamic features to return from GAM screen"
        ),
    ] = 50,
    n_sensitivity_steps: Annotated[
        int,
        typer.Option(
            help="Alpha steps for sensitivity sweep (0 = skip, saves ~10× the pseudotime compute time)"
        ),
    ] = 0,
    fdr_threshold: Annotated[
        float, typer.Option(help="FDR threshold for GAM likelihood-ratio test")
    ] = 0.05,
    detect_branches: Annotated[
        bool,
        typer.Option(
            help="Run branch detection on the spectral embedding and produce a trajectory-branches figure"
        ),
    ] = True,
    branch_min_cells: Annotated[
        int,
        typer.Option(help="Minimum cells per branch; smaller communities are merged"),
    ] = 20,
    branch_resolution: Annotated[
        float,
        typer.Option(
            help="Leiden resolution for branch detection (higher → more branches)"
        ),
    ] = 0.8,
    branch_anchor_quantile: Annotated[
        float,
        typer.Option(
            help="Fraction of lowest-pseudotime cells locked into the root branch (default 0.05 = bottom 5%%)"
        ),
    ] = 0.05,
    branch_min_pt_coherence: Annotated[
        float,
        typer.Option(
            help="Minimum Spearman rho between within-branch ordering and pseudotime; branches below this are merged (default 0.3)"
        ),
    ] = 0.3,
    driver_analysis: Annotated[
        bool,
        typer.Option(
            help="Run edge-driven driver identification and write IPA-ready gene tables"
        ),
    ] = True,
    driver_batch_key: Annotated[
        Optional[str],
        typer.Option(
            help="adata.obs column with batch/patient labels; used to flag batch-confounded hyperedges (strongly recommended)"
        ),
    ] = None,
    driver_min_abs_log2: Annotated[
        float,
        typer.Option(
            help="Minimum |log2 enrichment| for an edge to be called a driver"
        ),
    ] = 0.25,
    driver_max_group_enrichment: Annotated[
        float,
        typer.Option(
            help="Confound cutoff on EXCESS batch purity: (purity - background) / (1 - background). 0 = mixes batches like the dataset does, 1 = a single batch. Not an absolute purity — an absolute threshold is meaningless when one batch already dominates"
        ),
    ] = 0.5,
    driver_top_n_genes: Annotated[
        int,
        typer.Option(
            help="Genes per pathway in the prior-footprint export (0 = all members, which the largest PROGENy pathways will dominate)"
        ),
    ] = 100,
    driver_n_boot: Annotated[
        int,
        typer.Option(help="Bootstrap rounds for driver ranking confidence (0 = skip)"),
    ] = 100,
    driver_top_k: Annotated[
        int,
        typer.Option(
            help="A programme counts as selected in a bootstrap round if it ranks in this top-k"
        ),
    ] = 10,
    gene_attribution: Annotated[
        bool,
        typer.Option(
            help="Score genes from expression inside each driver hyperedge (within-edge contrast x edge specificity). This is what makes the gene level a result rather than a prior-footprint lookup"
        ),
    ] = True,
    attribution_top_n_genes: Annotated[
        int,
        typer.Option(help="Genes retained per edge in the attribution table (0 = all)"),
    ] = 200,
    attribution_n_perm: Annotated[
        int,
        typer.Option(
            help="Within-donor permutations for an empirical gene p-value on the retained rows (0 = skip; the normal-approximation FDR is still reported)"
        ),
    ] = 0,
    tf_direction_from_construction: Annotated[
        bool,
        typer.Option(
            help="Give TF hyperedges a direction from the fact that they are high-activity sets by construction. Without this the whole TF view is 'unclear' and never reaches the driver table"
        ),
    ] = True,
) -> None:
    """
    Run the full multi-view hypergraph perturbation analysis pipeline.

    Step 1 — Multi-view hypergraph pseudotime
        Builds TF-activity and pathway-activity hypergraphs, fuses them,
        and derives spectral pseudotime for each cell.

    Step 2 — Trajectory divergence
        Bins pseudotime and measures Wasserstein / MMD distance between
        control and drug distributions at each bin, with permutation p-values.

    Step 3 — Differential dynamics
        Fits GAMs per feature × condition and runs a likelihood-ratio test
        to identify features whose temporal profiles differ between conditions.

    Step 4 — Edge-driven drivers (--driver-analysis)
        Scores each hyperedge on condition composition and spectral
        contribution, ranks directional drivers, and bootstraps the ranking for
        confidence.

        With --gene-attribution (default on) the gene, TF and pathway levels are
        all measured from expression *inside* each driver hyperedge: the
        treatment contrast among that edge's cells, times how specific the gene
        is to that edge versus the other states.  A gene raised uniformly in all
        treated cells is demoted despite an identical fold change, which is the
        thing global DE cannot do.  Written to *_genes_by_programme.csv,
        *_tf_drivers.csv and *_pathway_drivers.csv.

        *_genes_all.csv is the older prior-footprint expansion, kept for
        comparison and tagged score_source="prior_footprint"; its within-programme
        ranking is determined by the PROGENy weight, not by the data.

    All outputs (h5ad, CSVs, figures) are written to OUTPUT_DIR.

    Example:
        pseudoembed hypergraph-analysis \\
            --data-path data.h5ad \\
            --output-dir results/ \\
            --condition-key condition \\
            --control-label control \\
            --drug-label drug
    """
    import scanpy as sc

    from pseudoembed.analysis.functional import run_go_enrichment
    from pseudoembed.analysis.publication_plots import (
        plot_activity_compare,
        plot_ol_differentiation_panel,
        plot_pseudotime_qc,
        plot_summary_dashboard,
        plot_trajectory_branches,
        plot_trajectory_drivers_heatmap,
    )
    from pseudoembed.analysis.trajectory_differential_dynamics import (
        compute_differential_dynamics,
        compute_trajectory_divergence,
        identify_trajectory_drivers,
    )
    from pseudoembed.core.multiview_hypergraph import (
        plot_hypergraph_force_directed,
        plot_hypergraph_knn,
        run_multiview_hypergraph_analysis,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load                                                                 #
    # ------------------------------------------------------------------ #
    typer.echo(f"Loading data from {data_path}...")
    adata = sc.read_h5ad(data_path)
    typer.echo(f"  {adata.n_obs} cells × {adata.n_vars} genes")

    # ------------------------------------------------------------------ #
    # Step 1: Multi-view hypergraph pseudotime                            #
    # ------------------------------------------------------------------ #
    typer.echo("\n[1/3] Running multi-view hypergraph analysis...")
    parsed_root_markers = (
        [g.strip() for g in root_markers.split(",") if g.strip()]
        if root_markers is not None
        else None
    )
    # Build explicit view_weights only when the user provided non-zero values;
    # otherwise pass None so run_multiview_hypergraph_analysis auto-selects
    # equal weights across whichever views are available.
    _total_w = tf_weight + pathway_weight + pca_weight
    parsed_view_weights = (
        {"tf": tf_weight, "pathway": pathway_weight, "pca": pca_weight}
        if _total_w > 0
        else None
    )
    adata, drug_effects, hypergraph_objects = run_multiview_hypergraph_analysis(
        adata=adata,
        condition_key=condition_key,
        control_label=control_label,
        drug_label=drug_label,
        time_key=time_key,
        organism=organism,
        root_markers=parsed_root_markers,
        top_n_tfs=top_n_tfs,
        z_score_threshold=z_score_threshold,
        n_states=n_states,
        n_spectral_components=n_spectral_components,
        spectral_time_scale=spectral_time_scale,
        manifold_k=manifold_k,
        embedding_scale=embedding_scale,
        min_root_corr=min_root_corr,
        pseudotime_method=pseudotime_method,  # type: ignore[arg-type]
        fusion_method=fusion_method,  # type: ignore[arg-type]
        view_weights=parsed_view_weights,
        n_sensitivity_steps=n_sensitivity_steps,
        n_pca_dims=n_pca_dims,
        output_dir=output_dir,
    )

    # drug_effects already written to multiview_drug_effects.csv inside
    # run_multiview_hypergraph_analysis(); keep a reference for the echo.
    drug_effects_path = output_path / "multiview_drug_effects.csv"

    # ------------------------------------------------------------------ #
    # Pseudotime QC figure                                                 #
    # ------------------------------------------------------------------ #
    import matplotlib.pyplot as plt

    typer.echo("\nGenerating pseudotime QC figure...")
    plot_pseudotime_qc(
        adata,
        pseudotime_key=pseudotime_key,
        condition_key=condition_key,
        control_label=control_label,
        drug_label=drug_label,
        time_key=time_key,
        cell_type_key=cell_type_key,
        save_path=str(output_path / "fig_pseudotime_qc.png"),
    )
    plt.close("all")
    typer.echo("  ✓ fig_pseudotime_qc.png")

    # ------------------------------------------------------------------ #
    # Branch detection                                                     #
    # ------------------------------------------------------------------ #
    if detect_branches and f"{pseudotime_key}_spectral" in adata.obsm:
        typer.echo("\nDetecting trajectory branches...")
        try:
            _branch_fig, _branch_axes, _branch_labels = plot_trajectory_branches(
                adata,
                pseudotime_key=pseudotime_key,
                cell_type_key=cell_type_key,
                min_branch_cells=branch_min_cells,
                resolution=branch_resolution,
                anchor_quantile=branch_anchor_quantile,
                min_pt_coherence=branch_min_pt_coherence,
                save_path=str(output_path / "fig_trajectory_branches.png"),
            )
            plt.close("all")
            n_branches_found = int(_branch_labels.max()) + 1
            typer.echo(
                f"  ✓ fig_trajectory_branches.png  ({n_branches_found} branches)"
            )

            # Write branch assignments to CSV
            _branch_csv = pd.DataFrame(
                {"cell": adata.obs_names, "branch": _branch_labels}
            )
            _branch_csv_path = output_path / "trajectory_branches.csv"
            _branch_csv.to_csv(_branch_csv_path, index=False)
            typer.echo(f"  ✓ trajectory_branches.csv")
        except Exception as _branch_exc:
            typer.echo(f"  ⚠  Branch detection skipped: {_branch_exc}")
    elif detect_branches:
        typer.echo(
            "\n  ⚠  Branch detection skipped: "
            f"spectral embedding '{pseudotime_key}_spectral' not found in adata.obsm."
        )

    # TF activity comparison across condition × timepoint
    if "tf_activities" in adata.obsm:
        plot_activity_compare(
            adata,
            condition_key=condition_key,
            time_key=time_key,
            feature_source="tf_activities",
            control_label=control_label,
            drug_label=drug_label,
            save_path=str(output_path / "fig_activity_compare_tf.png"),
        )
        plt.close("all")
        typer.echo("  ✓ fig_activity_compare_tf.png")

    # Pathway activity comparison across condition × timepoint
    if "pathway_activities" in adata.obsm:
        plot_activity_compare(
            adata,
            condition_key=condition_key,
            time_key=time_key,
            feature_source="pathway_activities",
            control_label=control_label,
            drug_label=drug_label,
            save_path=str(output_path / "fig_activity_compare_pathway.png"),
        )
        plt.close("all")
        typer.echo("  ✓ fig_activity_compare_pathway.png")

    # ------------------------------------------------------------------ #
    # Step 2: Trajectory divergence                                        #
    # ------------------------------------------------------------------ #
    typer.echo("\n[2/3] Computing trajectory divergence...")
    divergence_df = compute_trajectory_divergence(
        adata=adata,
        pseudotime_key=pseudotime_key,
        condition_key=condition_key,
        control_label=control_label,
        drug_label=drug_label,
        latent_key=latent_key,
        n_bins=n_bins,
        n_permutations=n_permutations,
    )

    divergence_path = output_path / "trajectory_divergence.csv"
    divergence_df.to_csv(divergence_path, index=False)
    typer.echo(f"  Divergence results saved to {divergence_path}")

    # ------------------------------------------------------------------ #
    # Step 3a: TF differential dynamics                                    #
    # ------------------------------------------------------------------ #
    typer.echo("\n[3/3] Computing differential dynamics — TF activities...")
    tf_summary_df, tf_gam_fits = compute_differential_dynamics(
        adata=adata,
        pseudotime_key=pseudotime_key,
        condition_key=condition_key,
        control_label=control_label,
        drug_label=drug_label,
        feature_source="tf_activities",
        n_top_features=n_top_features,
        fdr_threshold=fdr_threshold,
    )
    tf_summary_path = output_path / "differential_dynamics_tfs.csv"
    tf_summary_df.to_csv(tf_summary_path, index=False)
    typer.echo(f"  TF dynamics saved to {tf_summary_path}")

    # ------------------------------------------------------------------ #
    # Step 3b: Gene differential dynamics                                  #
    # ------------------------------------------------------------------ #
    typer.echo("        Computing differential dynamics — gene expression...")
    gene_summary_df, gene_gam_fits = compute_differential_dynamics(
        adata=adata,
        pseudotime_key=pseudotime_key,
        condition_key=condition_key,
        control_label=control_label,
        drug_label=drug_label,
        feature_source="X",
        n_top_features=n_top_features,
        fdr_threshold=fdr_threshold,
    )
    gene_summary_path = output_path / "differential_dynamics_genes.csv"
    gene_summary_df.to_csv(gene_summary_path, index=False)
    typer.echo(f"  Gene dynamics saved to {gene_summary_path}")

    # ------------------------------------------------------------------ #
    # Publication plots                                                    #
    # ------------------------------------------------------------------ #
    typer.echo("\nGenerating publication figures...")

    # Global TF drivers: correlation-based, condition-agnostic
    tf_drivers, gene_drivers = identify_trajectory_drivers(
        adata,
        pseudotime_key=pseudotime_key,
        n_top=30,
    )

    # Save driver tables so users can inspect the ranked lists
    if len(tf_drivers):
        tf_drivers_path = output_path / "trajectory_drivers_tfs.csv"
        tf_drivers.to_csv(tf_drivers_path, index=False)
        typer.echo(f"  TF drivers saved to {tf_drivers_path}")
    if len(gene_drivers):
        gene_drivers_path = output_path / "trajectory_drivers_genes.csv"
        gene_drivers.to_csv(gene_drivers_path, index=False)
        typer.echo(f"  Gene drivers saved to {gene_drivers_path}")

    # Global TF heatmap (all cells, sorted by pseudotime)
    if len(tf_drivers):
        plot_trajectory_drivers_heatmap(
            tf_drivers,
            adata,
            pseudotime_key=pseudotime_key,
            feature_source="tf_activities",
            feature_col="tf",
            feature_label="TF",
            n_top=30,
            save_path=str(output_path / "fig_heatmap_tfs_global.png"),
        )
        plt.close("all")
        typer.echo("  ✓ fig_heatmap_tfs_global.png")

    # Global gene heatmap (all cells, sorted by pseudotime)
    if len(gene_drivers):
        plot_trajectory_drivers_heatmap(
            gene_drivers,
            adata,
            pseudotime_key=pseudotime_key,
            feature_source="X",
            feature_col="gene",
            feature_label="Gene",
            n_top=30,
            save_path=str(output_path / "fig_heatmap_genes_global.png"),
        )
        plt.close("all")
        typer.echo("  ✓ fig_heatmap_genes_global.png")

    # Hypergraph kNN graph
    typer.echo("\nGenerating hypergraph kNN graph...")
    plot_hypergraph_knn(
        adata,
        pseudotime_key=pseudotime_key,
        condition_key=condition_key,
        manifold_k=manifold_k,
        save=str(output_path / "fig_hypergraph_knn.png"),
    )
    plt.close("all")
    typer.echo("  ✓ fig_hypergraph_knn.png")

    # Hypergraph force-directed layout
    typer.echo("Generating hypergraph force-directed layout...")
    plot_hypergraph_force_directed(
        adata,
        P_fused=hypergraph_objects.get("P_fused"),
        pseudotime_key=pseudotime_key,
        condition_key=condition_key,
        save=str(output_path / "fig_hypergraph_force_directed.png"),
    )
    plt.close("all")
    typer.echo("  ✓ fig_hypergraph_force_directed.png")

    # Full summary dashboard — saves all panels individually as
    # fig_dashboard_{A-F}_*.png; standalone duplicates removed above.
    plot_summary_dashboard(
        divergence_df=divergence_df,
        tf_summary_df=tf_summary_df,
        tf_gam_fits=tf_gam_fits,
        gene_summary_df=gene_summary_df if len(gene_gam_fits) else None,
        gene_gam_fits=gene_gam_fits if len(gene_gam_fits) else None,
        n_top_lollipop=20,
        n_top_heatmap=30,
        n_top_gam=8,
        fdr_threshold=fdr_threshold,
        control_label=control_label,
        drug_label=drug_label,
        adata=adata,
        pseudotime_key=pseudotime_key,
        save_path=str(output_path / "fig_dashboard.png"),
    )
    plt.close("all")
    typer.echo("  ✓ fig_dashboard_*.png")

    # ------------------------------------------------------------------ #
    # OL-differentiation multi-panel figure                               #
    # ------------------------------------------------------------------ #
    typer.echo("\nGenerating OL-differentiation panel (A/B/C)...")

    # ---- GO enrichment on condition-specific DE genes ----
    # From gene_summary_df: genes significantly up in drug (mean_drug > mean_ctrl
    # + significant) form the drug-specific list; the reverse for ctrl.
    _go_ctrl: Optional[pd.DataFrame] = None
    _go_drug: Optional[pd.DataFrame] = None
    _go_organism = "hsapiens" if organism == "human" else "mmusculus"
    _go_n_genes = 200

    if len(gene_summary_df) and "mean_ctrl" in gene_summary_df.columns:
        _sig_mask = (
            gene_summary_df["mw_qval"] < fdr_threshold
            if "mw_qval" in gene_summary_df.columns
            else pd.Series(True, index=gene_summary_df.index)
        )
        _feat_col = "feature"
        _eps = 1e-8
        _md = np.asarray(gene_summary_df["mean_drug"].values, dtype=float)
        _mc = np.asarray(gene_summary_df["mean_ctrl"].values, dtype=float)
        _lfc = np.log2((_md + _eps) / (_mc + _eps))
        gene_summary_df = gene_summary_df.copy()
        gene_summary_df["_lfc"] = _lfc

        # Drug-upregulated: positive log2FC and significant
        _drug_up = gene_summary_df[_sig_mask & (gene_summary_df["_lfc"] > 0)]
        _drug_genes: List[str] = (
            _drug_up.sort_values(by="_lfc", ascending=False)
            .head(_go_n_genes)[_feat_col]
            .tolist()
        )
        # Ctrl-upregulated: negative log2FC and significant
        _ctrl_up = gene_summary_df[_sig_mask & (gene_summary_df["_lfc"] < 0)]
        _ctrl_genes: List[str] = (
            _ctrl_up.sort_values(by="_lfc", ascending=True)
            .head(_go_n_genes)[_feat_col]
            .tolist()
        )

        typer.echo(
            f"  Running GO enrichment — {control_label}: {len(_ctrl_genes)} genes "
            f"(up in ctrl), {drug_label}: {len(_drug_genes)} genes (up in drug)"
        )
        _go_ctrl = run_go_enrichment(_ctrl_genes, organism=_go_organism)
        _go_drug = run_go_enrichment(_drug_genes, organism=_go_organism)
        typer.echo(
            f"  GO terms — {control_label}: {len(_go_ctrl)}, "
            f"{drug_label}: {len(_go_drug)}"
        )
        if len(_go_ctrl):
            _go_ctrl.to_csv(
                output_path / f"go_enrichment_{control_label}.csv", index=False
            )
        if len(_go_drug):
            _go_drug.to_csv(
                output_path / f"go_enrichment_{drug_label}.csv", index=False
            )

    plot_ol_differentiation_panel(
        adata,
        gene_summary_df=gene_summary_df,
        pseudotime_key=pseudotime_key,
        condition_key=condition_key,
        time_key=time_key,
        control_label=control_label,
        drug_label=drug_label,
        cell_type_key=cell_type_key,
        fdr_threshold=fdr_threshold,
        go_results_ctrl=_go_ctrl,
        go_results_drug=_go_drug,
        save_path=str(output_path / "fig_ol_differentiation_panel.png"),
    )
    plt.close("all")
    typer.echo("  ✓ fig_ol_differentiation_panel.png")

    # ------------------------------------------------------------------ #
    # Step 4: Edge-driven driver identification + IPA export               #
    # ------------------------------------------------------------------ #
    driver_paths: dict = {}
    if driver_analysis:
        typer.echo("\n[4/4] Identifying drivers from hyperedges...")
        from pseudoembed.core.hypergraph_drivers import run_driver_analysis

        try:
            _driver_res = run_driver_analysis(
                adata,
                hypergraph_objects,
                condition_key=condition_key,
                control_label=control_label,
                drug_label=drug_label,
                output_dir=output_dir,
                pseudotime_key=pseudotime_key,
                batch_key=driver_batch_key,
                organism=organism,
                min_abs_log2=driver_min_abs_log2,
                max_group_enrichment=driver_max_group_enrichment,
                # 0 means "no cap"; the function takes None for that.
                top_n_genes=driver_top_n_genes if driver_top_n_genes > 0 else None,
                n_boot=driver_n_boot,
                top_k=driver_top_k,
                fdr_threshold=fdr_threshold,
                gene_attribution=gene_attribution,
                attribution_top_n_genes=(
                    attribution_top_n_genes if attribution_top_n_genes > 0 else None
                ),
                attribution_n_perm=attribution_n_perm,
                tf_direction_from_construction=tf_direction_from_construction,
            )
            driver_paths = _driver_res["paths"]
            plt.close("all")
        except Exception as exc:
            # The upstream pipeline's results are already on disk; a failure here
            # must not discard them.
            typer.echo(
                f"  ⚠ Driver analysis failed ({type(exc).__name__}: {exc})", err=True
            )

    # adata already written to adata_multiview_hypergraph.h5ad inside
    # run_multiview_hypergraph_analysis(); no second write needed.
    adata_out = output_path / "adata_multiview_hypergraph.h5ad"
    typer.echo(f"\n✓ Analysis complete. All results written to {output_dir}/")
    typer.echo(f"  AnnData        : {adata_out}")
    typer.echo(f"  Drug Δ         : {drug_effects_path}")
    typer.echo(f"  Divergence     : {divergence_path}")
    typer.echo(f"  TF dynamics    : {tf_summary_path}")
    typer.echo(f"  Gene dynamics  : {gene_summary_path}")
    typer.echo(f"  kNN graph      : {output_path / 'fig_hypergraph_knn.png'}")
    typer.echo(
        f"  Force layout   : {output_path / 'fig_hypergraph_force_directed.png'}"
    )
    if detect_branches:
        typer.echo(f"  Branches fig   : {output_path / 'fig_trajectory_branches.png'}")
        typer.echo(f"  Branches CSV   : {output_path / 'trajectory_branches.csv'}")
    if driver_paths:
        typer.echo("  Drivers:")
        for _label, _p in driver_paths.items():
            typer.echo(f"    {_label:<22}: {_p}")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
