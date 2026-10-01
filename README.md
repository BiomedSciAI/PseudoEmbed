# PseudoEmbed

Multiview hypergraph trajectory inference and gene attribution for single-cell
transcriptomics.

PseudoEmbed builds a hypergraph over cells from several complementary "views" of
the same data — transcription-factor activity, pathway activity, and expression
PCA — fuses them into one operator, and reads a pseudotime off that operator's
spectral embedding. It then attributes the resulting trajectory structure back to
hyperedges, driver programmes, and individual genes.

Two analyses are supported, and they are the whole of the public surface:

- **(A) Multiview hypergraph analysis** — construct the TF / pathway / PCA views,
  fuse them, and compute pseudotime.
- **(B) Gene attribution analysis** — rank the programmes and genes responsible
  for the trajectory and for a condition contrast along it.

## Start here

An example tutorial notebooks in [`examples/`](examples/):
- [`richard2018_tcell_tutorial.ipynb`](examples/richard2018_tcell_tutorial.ipynb) —
  250 real CD8⁺ T cells sampled at 0/1/3/6 h ([Richard et al.
  2018](https://doi.org/10.1038/s41590-018-0160-9)), **downloaded on first run**.
  Four surface proteins measured per cell never enter the model, so they are a
  held-out validation axis. Read this to see the workflow end to end on real
  data: build three views, fuse them, validate the axis against the held-out
  proteins, then decompose it into the named hyperedges that built it.

The data is fetched, not redistributed:

```python
import pseudoembed
adata = pseudoembed.richard2018()   # downloads E-MTAB-6051 once, then caches
```

See [`examples/README.md`](examples/README.md) for how to run them, and what to do
if the kernel dies with no traceback (a `libomp` clash inside decoupler).

## Method

Each view contributes hyperedges (sets of cells) with weights. From the incidence
matrix `H`, edge weights `W`, and the vertex/edge degree matrices `Dv`/`De`, the
per-view operator is

```
S = Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}
```

which is symmetric and positive semi-definite with eigenvalues in `[0, 1]`. Views
are combined either by averaging their operators (`--fusion-method operator`, the
default, "late" fusion) or by pooling their hyperedges before building a single
operator (`--fusion-method edge`, "early" fusion).

Pseudotime is the distance from a root cell in the scaled spectral embedding
`emb[cell, j] = f(lambda_j) * phi_j[cell]`, where `f` is set by
`--embedding-scale`: `dpt` gives `lambda/(1 - lambda)` (default), `power` gives
`lambda^t` with `t = --spectral-time-scale`, and `unit` leaves eigenvalues raw.
The distance itself is either a graph geodesic on a kNN manifold
(`--pseudotime-method geodesic`, default) or a straight chord in the embedding
(`chord`).

## Installation

`decoupler` is a required dependency, not an optional one: both the TF and the
pathway view are built from its univariate linear model (ULM) activity scores, so
neither analysis runs without it.

## From git repo

```bash
uv venv pseudoembed_env --python 3.12
source pseudoembed_env/bin/activate
uv pip install git+ssh://git@github.ibm.com/scDynamics/PseudoEmbed.git
```

## From clone

```bash
git clone git@github.ibm.com:scDynamics/PseudoEmbed.git
cd pseudoembed
uv pip install -e ".[dev]"
```

Extras:

| Extra | Contents | When you need it |
|---|---|---|
| `dev` | pytest, pytest-cov, black, ruff, mypy | Running the test suite or linting |
| `full` | `decoupler[full]`, dcor, igraph, gprofiler-official | GO enrichment, distance correlation |
| `benchmark` | palantir, pyyaml | The `experiments/benchmark` evaluation suite |
| `hypergraph` | xgi | Optional hypergraph utilities |
| `notebook` | jupyterlab, ipywidgets | Interactive work |

## Command-line interface

The CLI has exactly two commands.

### `preprocess`

Filtering, normalisation, and highly-variable-gene selection. The PCA and pathway
views both consume its output, so this is a prerequisite rather than a
convenience.

```bash
pseudoembed preprocess \
    --input-path raw.h5ad \
    --output-path processed.h5ad \
    --n-top-genes 5000 \
    --batch-key experiment
```

Options: `--input-path` and `--output-path` (both required),
`--experiment-filter`, `--min-genes` (100), `--min-cells` (3),
`--n-top-genes` (5000), `--batch-key` (`experiment`).

### `hypergraph-analysis`

The single entrypoint for both (A) and (B). Only `--data-path` and
`--output-dir` are required; the other 45 options have defaults.

```bash
pseudoembed hypergraph-analysis \
    --data-path processed.h5ad \
    --output-dir results/run1 \
    --condition-key condition \
    --control-label control \
    --drug-label drug \
    --cell-type-key cell_type
```

Note that `--driver-analysis`, `--gene-attribution`, `--detect-branches`, and
`--tf-direction-from-construction` all **default to on**. To get (A) alone, turn
the attribution stages off explicitly:

```bash
pseudoembed hypergraph-analysis \
    --data-path processed.h5ad \
    --output-dir results/pseudotime_only \
    --no-driver-analysis --no-gene-attribution
```

Frequently-used option groups:

- **Views and fusion** — `--fusion-method` (`operator` | `edge`),
  `--tf-weight`, `--pathway-weight`, `--pca-weight`, `--n-pca-dims`,
  `--top-n-tfs`, `--z-score-threshold`, `--organism`
- **Readout** — `--n-spectral-components`, `--embedding-scale`,
  `--spectral-time-scale`, `--pseudotime-method`, `--manifold-k`,
  `--root-markers`, `--min-root-corr`, `--n-states`
- **Drivers** — `--driver-batch-key`, `--driver-min-abs-log2`,
  `--driver-max-group-enrichment`, `--driver-n-boot`, `--driver-top-k`,
  `--fdr-threshold`
- **Gene attribution** — `--attribution-top-n-genes`, `--attribution-n-perm`

**Set `--driver-batch-key`, and check `--driver-max-group-enrichment`.** Together
they guard against a programme being called a driver when its signal comes from a
single batch, but both need attention:

- Without `--driver-batch-key` the confound columns are never computed and *every*
  edge passes the screen. Pass a genuine batch variable — the experimental design
  variable is usually right. A label that merely looks like a batch (an artifact of
  cell annotation, say) yields confound statistics that appear real and mean
  nothing.
- `--driver-max-group-enrichment` defaults to `1.15`, but the quantity it thresholds
  is *excess purity*, `(purity - background) / (1 - background)`, which is bounded
  above by `1.0`. **Nothing can exceed 1.15, so the screen is inert at the
  default.** Set a value in `(0, 1)` chosen from the observed distribution. On the
  SOW48 tutorial dataset, moving from the default to `0.3` cuts driver edges from
  45 to 16.

Run `pseudoembed hypergraph-analysis --help` for the full annotated list.

## Python API

Heavy dependencies are imported lazily ([PEP 562][pep562]), so `import
pseudoembed` stays cheap and the real import happens on first attribute access.

```python
import scanpy as sc
from pseudoembed import run_multiview_hypergraph_analysis, run_driver_analysis

adata = sc.read_h5ad("processed.h5ad")

# (A) Build the views, fuse them, compute pseudotime.
adata, drug_effects, objects = run_multiview_hypergraph_analysis(
    adata,
    condition_key="condition",
    control_label="control",
    drug_label="drug",
    fusion_method="operator",
    view_weights={"tf": 1.0, "pathway": 1.0, "pca": 1.0},
    output_dir="results/run1",
)

# (B) Attribute the trajectory to programmes and genes.
#     `objects` carries the hyperedges and incidence built in (A).
driver_results = run_driver_analysis(
    adata,
    objects,
    condition_key="condition",
    control_label="control",
    drug_label="drug",
    pseudotime_key="multiview_pseudotime",
    views=("pathway", "tf"),
    gene_attribution=True,
    output_dir="results/run1",
)
```

`run_multiview_hypergraph_analysis` returns `(adata, drug_effects_df,
objects_dict)`; pseudotime lands in `adata.obs["multiview_pseudotime"]`. Passing
that same `objects` dict into `run_driver_analysis` is what couples (B) to (A) —
it avoids rebuilding the hypergraph and guarantees the two stages describe the
same edges.

### Public exports

Available directly from the `pseudoembed` namespace:

*Multiview hypergraph* — `run_multiview_hypergraph_analysis`,
`compute_multiview_pseudotime`, `build_view_operator`, `fuse_view_operators`,
`fuse_hyperedges`, `build_pathway_hypergraph`, `build_pca_hypergraph`,
`build_tf_hypergraph`, `compute_tf_activities`, `sensitivity_analysis`

*Gene attribution* — `run_driver_analysis`, `rank_drivers`,
`expand_programmes_to_genes`, `edge_gene_attribution`,
`aggregate_gene_attribution`, `build_multilevel_driver_report`

*Supporting* — `load_raw_data`, `run_functional_analysis`, `run_go_enrichment`

## Package layout

```
pseudoembed/
  core/
    multiview_hypergraph.py        # (A): views, operators, fusion, spectral readout
    hypergraph_drivers.py          # (B): run_driver_analysis orchestrator
    hypergraph_gene_attribution.py # (B): edge_gene_attribution
    hypergraph_interpret.py        # shared: edge_condition_enrichment, BH-FDR
    io.py                          # load_raw_data
  analysis/
    tf_hypergraph_drug_analysis.py # incidence + TF activities (shared leaf)
    trajectory_differential_dynamics.py
    publication_plots.py
    functional.py                  # run_go_enrichment
  cli/main.py                      # preprocess, hypergraph-analysis
```

`core/__init__.py` and `analysis/__init__.py` deliberately perform no eager
submodule imports — every module there reaches for scanpy, decoupler, or
matplotlib. Import submodules directly, or use the lazy top-level re-exports.

## Outputs

Written to `--output-dir`:

**(A)** `adata_multiview_hypergraph.h5ad`, `multiview_drug_effects.csv`,
`multiview_sensitivity_analysis.csv`, `multiview_sensitivity_plot.png`

**(B)** (prefixed `hypergraph_` by default) `..._drivers_significant.csv`,
`..._drivers_all_edges.csv`, `..._pathway_drivers.csv`, `..._tf_drivers.csv`,
`..._gene_attribution_edge_level.csv`, `..._genes_by_programme.csv`,
`..._genes_all.csv`, `..._stability.csv`, `..._edge_enrichment_raw.csv`,
`..._edge_contributions_raw.csv`, `..._driver_summary.png`

## Benchmark suite

`experiments/benchmark/` evaluates (A) against DPT and Palantir:

```bash
cd experiments
python -m benchmark.cli --config benchmark/configs/quick.yaml
```

The suite imports `experiments/hypergraph_readout.py` via `sys.path` rather than
the package. That module is a **deliberate independent reimplementation**, not a
prototype awaiting promotion — see its docstring. The benchmark's `*_package`
methods delegate to the shipped `_hypergraph_spectral_pseudotime`, while the
unsuffixed ones use the research library's candidate readouts; keeping the two
paths separate is what makes that comparison an ablation rather than a
package-vs-package tautology.

Expect float drift on the order of `1e-6` between runs of the `*_package`
methods, from unseeded ARPACK starting vectors. Per `PACKAGE_REPRO_FLOOR = 1e-5`,
only **rank** changes count as regressions.

`benchmark/cache/` is not version-controlled and regenerates on first run.

## Development

```bash
pytest tests/ -q
ruff check pseudoembed experiments/benchmark
```

Coverage is intentionally not in the default `addopts`, since it roughly doubles
the run time. Request it explicitly:

```bash
pytest --cov=pseudoembed --cov-report=term-missing --cov-report=html
```

Long-running CLI jobs are where `libomp` can segfault inside decoupler's ULM,
with no Python traceback. Run them in a conda environment where `libomp` is
installed once, rather than one mixing conda and pip OpenMP runtimes. The
tutorials in `examples/` are small enough to run in a plain venv.

### A note on `.gitignore`

The bare `results/` pattern matches at *any* depth, so
`experiments/benchmark/results/` is ignored — including the `raw_metrics.json`
that `--report-only` reads. The blanket `*.csv` / `*.png` / `*.pdf` rules
likewise hide benchmark CSVs and the PDFs under `docs/` and `slides/`. If you
want those artefacts versioned, root-anchor the pattern (`/results/`) and add
negations.

## Status and known limitations

Read this before quoting a number out of this package.

**The spectral embedding omits the `D_v^{-1/2}` back-transform.** The operator is
built as `S = Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}`, but the eigenvectors are used
directly rather than mapped back through `Dv^{-1/2}`. The consequence is concrete:
`sqrt(cell hyperedge degree)` leaks into every pseudotime this package produces.
This is **not fixed in this release**, and pseudotime values will move when it is.

**On raw pseudotime accuracy, DPT is usually competitive or better.** Across an
internal five-dataset sweep, diffusion pseudotime ranked higher than every
PseudoEmbed variant on four of the five. Richard 2018 is the exception, and it is
the dataset shipped here — so treat the tutorial's win as one measurement, not as a
general claim.

What the method contributes is not a better pseudotime. It is that every hyperedge
carries a name, so the axis can be **attributed** to named programmes and those
programmes **ordered by onset**. That decomposition has no equivalent in a
pairwise k-NN graph, and it is what the tutorials are built to demonstrate.

**Views are not equally useful, and edge count is a confound.** A view's share of
the axis scales with how many hyperedges it was allowed to build, so always read
the per-edge normalisation alongside the total (`pseudoembed.plotting.view_share`
plots both, and reports whether the ranking changes). Hyperedge size is the single
most important parameter: edges spanning a large fraction of the dataset flatten
the structure the trajectory is meant to find.

**Tests.** `pytest -m "not slow"` gives 438 passed, 11 failed, 1 skipped. None of
the 11 are defects in shipped code: 9 are stale assertions in
`tests/test_benchmark_figures.py` (the writer calls `plt.close(fig)` before the
test inspects `plt.gcf()`, so the test examines an empty figure), and 2 are a
pandas 3.0 / anndata `ArrowStringArray` incompatibility in one environment's test
fixture. They are documented rather than silenced.

## Requirements

Python >= 3.10, with numpy >= 1.21, pandas >= 1.3, scipy >= 1.7,
scikit-learn >= 1.0, matplotlib >= 3.4, seaborn >= 0.11, scanpy >= 1.10,
anndata >= 0.8, typer >= 0.9, and decoupler >= 2.0.

## Contact

Matthew Madgwick — mattmadgwick@ibm.com

[pep562]: https://peps.python.org/pep-0562/
