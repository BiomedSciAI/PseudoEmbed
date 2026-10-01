"""Richard et al. 2018 OT-I CD8+ T cell activation time course (E-MTAB-6051).

Downloaded on first use from EBI BioStudies and cached; nothing is redistributed
with this package.

THE DATASET
-----------
Richard et al., *Nature Immunology* 19:849-858 (2018), "T cell cytolytic capacity
is independent of initial stimulation strength".  Plate-based Smart-seq2 of naive
OT-I CD8+ T cells stimulated with peptides of varying affinity and harvested at
0, 1, 3 and 6 h.

:func:`richard2018` returns the **high-affinity (N4) arm plus the unstimulated
baseline**: 250 cells over four timepoints.  That is one arm of a four-arm study,
and the choice is not neutral -- read :func:`richard2018`'s ``arm`` parameter and
the note below before using it as a trajectory benchmark.

WHY THIS ARM
------------
Only the N4 arm has a time course.  The reduced-affinity (T4, G4) and non-binding
(NP68) arms were all harvested at **6 h only** (measured from the SDRF: 278
QC-passing cells, every one at 6 h), so they cannot supply a second trajectory and
cannot act as a time-matched control.  Consequently:

* ``stimulus`` is perfectly collinear with ``time`` in the returned object --
  unstimulated cells exist only at 0 h.  There is no control arm at 1/3/6 h, so
  any "condition effect" here is a time effect wearing a different name.
* Pass ``arm="all"`` to get all 528 QC-passing cells across four affinities if you
  want the affinity contrast instead.  It is a *dose* axis at fixed time, not a
  trajectory.

WHAT IT IS GOOD FOR
-------------------
Four independently measured surface proteins (CD69, CD25, CD44, CD62L) were
recorded per cell by index sorting, and land in ``obs`` as ``prot_*``.  These are
not derived from the transcriptome, so they are a genuinely held-out axis to score
a pseudotime against -- rare in a public trajectory dataset, and the reason this
one is used for the tutorial.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# BioStudies serves ArrayExpress studies.
_BASE = "https://www.ebi.ac.uk/biostudies/files/E-MTAB-6051"
#: FIRE is where `_BASE` redirects to, so it is tried FIRST: the redirect target
#: is the real file, and the redirecting endpoint sometimes serves 200-with-no-body.
_MIRROR = "https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/051/E-MTAB-6051/Files"
_COUNTS = ("counts_table_sortdate20160727.txt", "counts_table_sortdate20161026.txt")
_SDRF = "E-MTAB-6051.sdrf.txt"

#: Ensembl BioMart mart/dataset used to map gene IDs to MGI symbols.
_BIOMART_HOST = "http://www.ensembl.org"
_BIOMART_DATASET = "mmusculus_gene_ensembl"

# --- QC columns in the SDRF, all three of which must pass ------------------
_Q_WELL = "Characteristics[single cell well quality]"
_Q_POST = "Characteristics[post-analysis well quality]"
_Q_CELL = "Characteristics[single cell quality]"

#: Index-sort protein measurements, ``obs`` name -> SDRF column stem.
_PROTEINS = ("CD69", "CD25", "CD44", "CD62L")

# --- Marker panels from the paper, for scoring and for figures -------------
#: Fig 1e/1f early-response transcriptional regulators.
EARLY_TF = ["Nr4a1", "Nr4a2", "Nr4a3", "Egr1", "Egr2", "Egr3", "Fosb", "Atf3"]
#: The full Fig 1e gene list, in the paper's order.
FIG1E = [
    "Als2cl",
    "S1pr1",
    "Klf3",
    "Kif23",
    "Vps37b",
    "Tcp11l2",
    "Btg2",
    "Saal1",
    "Fosb",
    "Dusp10",
    "Plk3",
    "Gch1",
    "Atf3",
    "Nr4a2",
    "Nfkbid",
    "Nr4a1",
    "Nr4a3",
    "Egr1",
    "Egr2",
]
#: Late effector markers.
LATE = ["Myc", "Gzmb", "Ifng", "Il2", "Tnf"]
#: Surface protein -> the gene encoding it, for transcript-vs-protein checks.
SURFACE_GENES = {"CD62L": "Sell", "CD25": "Il2ra", "CD69": "Cd69", "CD44": "Cd44"}

#: Naive/quiescence markers used to seed the trajectory root.
#:
#: Deliberately **not** in :func:`marker_panels`: that dict holds panels *from
#: the paper*, and §2 of the tutorial iterates it to report which paper panels
#: lost genes to filtering.  Root markers are a modelling choice, and listing
#: them there would report them as a paper panel.
#:
#: Chosen to have no overlap with :data:`LATE` or :data:`EARLY_TF`.  A root panel
#: sharing a marker with a differentiated state has rooted a trajectory on a
#: mature cell more than once -- and the rule is not an argmax, it scores the
#: panel and takes the *lowest-degree* cell above the percentile, so a single
#: contaminating marker is enough to move the root.
ROOT_MARKERS = ["Sell", "Klf2", "Tcf7", "Lef1", "Ccr7"]


def _tilde(path) -> str:
    """Render a path with the user's home directory shown as ``~``.

    Cosmetic, but it is what keeps an absolute home path out of the committed
    tutorial notebook's outputs.  Falls back to the plain path when it is not
    under ``$HOME``.
    """
    path = Path(path)
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def _cache_dir(cache_dir: str | os.PathLike | None) -> Path:
    """Resolve the cache directory, preferring an explicit absolute location.

    scanpy's ``settings.datasetdir`` defaults to the *relative* ``./data``, so a
    notebook run from a different working directory silently re-downloads into a
    new tree.  We therefore resolve to an absolute path and honour
    ``PSEUDOEMBED_DATA`` before falling back to a user-level cache.
    """
    if cache_dir is not None:
        p = Path(cache_dir)
    elif os.environ.get("PSEUDOEMBED_DATA"):
        p = Path(os.environ["PSEUDOEMBED_DATA"])
    else:
        p = (
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            / "pseudoembed"
        )
    p = p.expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _download(name: str, dest: Path, verbose: bool = True) -> Path:
    """Fetch one study file into ``dest`` unless it is already there."""
    out = dest / name
    if out.exists() and out.stat().st_size > 0:
        return out
    import shutil
    from urllib.request import urlopen

    tmp = out.with_suffix(out.suffix + ".part")
    errors = []
    for url in (f"{_MIRROR}/{name}", f"{_BASE}/{name}"):
        if verbose:
            print(f"  downloading {name} ...", flush=True)
        try:
            with urlopen(url, timeout=180) as r, open(tmp, "wb") as fh:
                shutil.copyfileobj(r, fh)
            # `/biostudies/files/` 302-redirects to the FIRE mirror and can hand
            # back a 200 with an EMPTY body, so a zero exit status is not proof
            # of a download.  Check the size, not the status.
            if tmp.stat().st_size == 0:
                raise RuntimeError("server returned an empty body")
            tmp.replace(out)  # atomic: an interrupted download never looks done
            return out
        except Exception as e:  # noqa: BLE001 - try the next mirror
            errors.append(f"{url}: {e}")
        finally:
            tmp.unlink(missing_ok=True)
    raise RuntimeError(
        f"Could not download {name} from EBI. Tried:\n  "
        + "\n  ".join(errors)
        + f"\nYou can place the file manually at {out}."
    )


def _ensembl_to_symbol(dest: Path, verbose: bool = True) -> dict[str, str]:
    """``{ensembl_id: mgi_symbol}``, cached as CSV beside the counts.

    CollecTRI(mouse) and PROGENy(mouse) are keyed on **MGI symbols**, so the
    Ensembl IDs the ArrayExpress matrix ships with cannot reach either prior.
    """
    cache = dest / "mouse_ensembl_to_symbol.csv"
    if not cache.exists():
        try:
            from pybiomart import Dataset
        except ImportError as e:  # pragma: no cover - dependency message
            raise ImportError(
                "pybiomart is needed to build the Ensembl->MGI symbol map. "
                "Install it (`pip install pybiomart`), or place a two-column "
                f"CSV (ensembl_id,symbol) at {cache}."
            ) from e
        if verbose:
            print("  querying Ensembl BioMart for gene symbols ...", flush=True)
        ds = Dataset(name=_BIOMART_DATASET, host=_BIOMART_HOST)
        m = ds.query(attributes=["ensembl_gene_id", "external_gene_name"])
        m.columns = ["ensembl_id", "symbol"]
        m.to_csv(cache, index=False)

    m = pd.read_csv(cache)
    m.columns = ["ensembl_id", "symbol"][: m.shape[1]]
    m = m.dropna(subset=["symbol"])
    m = m[m["symbol"].astype(str).str.len() > 0]
    return dict(zip(m["ensembl_id"].astype(str), m["symbol"].astype(str)))


def _load_sdrf(path: Path) -> pd.DataFrame:
    """Per-well annotation, one row per sequencing well.

    The SDRF has two rows per well (paired-end read files), so it is
    de-duplicated on the well key extracted from ``Assay Name``:
    ``SLX-12611.N701_S502.HGFMMBBXX.s_8`` -> ``SLX-12611.N701_S502.``, which is
    exactly the column name used in the count tables.
    """
    s = pd.read_csv(path, sep="\t", low_memory=False)
    s["well"] = s["Assay Name"].astype(str).str.extract(r"^([^.]+\.[^.]+\.)")[0]
    s = s.dropna(subset=["well"]).drop_duplicates(subset=["well"])
    return s.set_index("well")


def _expect(got, want, what: str) -> None:
    """Warn loudly when a reconstructed count departs from the verified value."""
    if got != want:
        import warnings

        warnings.warn(
            f"Richard 2018: expected {want} {what}, got {got}. The study files at "
            "EBI may have been revised; results will not match the published "
            "reference numbers.",
            RuntimeWarning,
            stacklevel=3,
        )


def _select_wells(sdrf: pd.DataFrame, arm: str) -> list[str]:
    """Wells passing all three QC flags, restricted to ``arm``."""
    qc = (sdrf[_Q_WELL] == "OK") & (sdrf[_Q_POST] == "pass") & (sdrf[_Q_CELL] == "OK")
    stim = sdrf["Factor Value[stimulus]"].astype(str)
    if arm == "N4":
        # The high-affinity arm plus its unstimulated baseline: the only cells
        # with a time course.  Yields exactly 250 wells.
        keep = qc & (
            stim.str.startswith("OT-I high affinity") | (stim == "unstimulated")
        )
    elif arm == "all":
        keep = qc
    else:
        raise ValueError(f"arm must be 'N4' or 'all', got {arm!r}")
    return list(sdrf.index[keep])


def richard2018(
    cache_dir: str | os.PathLike | None = None,
    *,
    arm: str = "N4",
    n_top_genes: int = 2000,
    seed: int = 0,
    preprocess: bool = True,
    verbose: bool = True,
):
    """Richard 2018 OT-I CD8+ T cell time course, ready for trajectory analysis.

    Downloads E-MTAB-6051 on first call (~30 MB of text; the cached result is
    reused afterwards), maps Ensembl IDs to MGI symbols, and optionally runs the
    standard scanpy preprocessing.

    Parameters
    ----------
    cache_dir
        Where to keep the downloads.  Defaults to ``$PSEUDOEMBED_DATA``, else
        ``~/.cache/pseudoembed``.  Always resolved to an absolute path, so the
        cache does not depend on the working directory.
    arm
        ``"N4"`` (default) returns the 250 high-affinity + unstimulated cells,
        the only ones forming a time course.  ``"all"`` returns all 528
        QC-passing cells across four peptide affinities -- an affinity contrast at
        6 h, **not** a trajectory.  See the module docstring.
    n_top_genes, seed
        Highly-variable-gene count and the PCA random seed.
    preprocess
        When False, return the raw count object with annotation only (no
        normalisation, HVG, scaling or PCA).

    Returns
    -------
    AnnData
        ``X`` log-normalised (``layers['counts']`` raw, ``layers['lognorm']`` the
        same as ``X``), ``obsm['X_pca']`` from scaled HVGs, and ``.raw`` set to
        the full log-normalised matrix so marker panels can be scored on genes
        the HVG filter dropped.

        Key ``obs`` columns: ``hours`` (float), ``timepoint`` (ordered
        categorical), ``condition`` (``unstim``/``N4``), ``affinity``, ``donor``,
        ``plate``, and ``prot_CD69`` / ``prot_CD25`` / ``prot_CD44`` /
        ``prot_CD62L`` -- the index-sort protein measurements, which are
        **independent of the transcriptome** and so usable as held-out validation.

    Notes
    -----
    ERCC spike-in rows are removed (92 of 46,695) along with unmapped Ensembl
    IDs.  Duplicate symbols are collapsed by **summing** counts: 21 symbols carry
    more than one Ensembl ID, and keeping both would double-count them in the TF
    and pathway priors and create two gene hyperedges for the same gene.
    """
    import anndata as ad
    import scanpy as sc

    dest = _cache_dir(cache_dir)
    if verbose:
        # Print the cache path with $HOME collapsed to "~".  The absolute path
        # is the user's home directory, and this line lands in the outputs of a
        # committed tutorial notebook -- shipping "/Users/<name>/..." there leaks
        # a real path into a public artefact for no gain to the reader.
        print(f"Richard 2018 (E-MTAB-6051), cache: {_tilde(dest)}")

    sdrf = _load_sdrf(_download(_SDRF, dest, verbose))
    wells = _select_wells(sdrf, arm)
    if verbose:
        print(f"  {len(wells)} QC-passing cells (arm={arm!r})")

    frames = [
        pd.read_csv(_download(f, dest, verbose), sep="\t", index_col=0) for f in _COUNTS
    ]
    counts = pd.concat(frames, axis=1)
    missing = [w for w in wells if w not in counts.columns]
    if missing:
        raise RuntimeError(
            f"{len(missing)} selected wells absent from the count tables "
            f"(first: {missing[:3]}). The study files may have changed."
        )
    counts = counts[wells]

    # Guard the acquisition, not just the code path.  These were reconstructed
    # from the study files and checked cell-for-cell against an independently
    # prepared copy of this dataset; if EBI revises the study, the numbers move
    # and a silent difference here would propagate into every downstream result.
    if arm == "N4":
        _expect(counts.shape[1], 250, "cells (N4 arm)")
        _expect(counts.shape[0], 46695, "gene rows before ERCC removal")

    # ERCC spike-ins are not genes; they inflate library size and cannot map to
    # a symbol, so they go before normalisation rather than after.
    spike = counts.index.astype(str).str.startswith("ERCC-")
    if verbose:
        print(f"  dropping {int(spike.sum())} ERCC spike-in rows")
    counts = counts[~spike]

    adata = ad.AnnData(
        X=counts.T.to_numpy(dtype=np.float32),
        obs=pd.DataFrame(index=pd.Index(counts.columns.astype(str), name=None)),
        var=pd.DataFrame(index=pd.Index(counts.index.astype(str), name=None)),
    )

    # ---- annotation ------------------------------------------------------
    ann = sdrf.loc[adata.obs_names]
    stim = ann["Factor Value[stimulus]"].astype(str)
    # `time` is blank for unstimulated wells in the SDRF; those are the 0 h
    # baseline, so they are coerced to 0.0 rather than dropped as missing.
    hours = pd.to_numeric(ann["Factor Value[time]"], errors="coerce").fillna(0.0)
    adata.obs["hours"] = hours.to_numpy(dtype=float)
    adata.obs["timepoint"] = pd.Categorical(
        [f"{h:g}h" for h in adata.obs["hours"]],
        categories=[f"{h:g}h" for h in sorted(adata.obs["hours"].unique())],
        ordered=True,
    )
    adata.obs["condition"] = pd.Categorical(
        np.where(
            stim.str.startswith("OT-I high affinity"),
            "N4",
            np.where(stim == "unstimulated", "unstim", "other"),
        ),
        categories=["unstim", "N4", "other"],
        ordered=False,
    )
    adata.obs["affinity"] = pd.Categorical(stim.to_numpy())
    adata.obs["donor"] = ann["Characteristics[individual]"].astype(str).to_numpy()
    adata.obs["plate"] = [w.split(".")[0] for w in adata.obs_names]
    # These are index-sort FACS readings, and they live under `Comment[...]`,
    # not `Characteristics[...]`.  A tolerant `if col in ann.columns` here would
    # have silently dropped all four -- i.e. dropped the only transcriptome-
    # independent validation channel this dataset has -- so a missing column is
    # a hard error instead.
    for prot in _PROTEINS:
        col = f"Comment[{prot} measurement (log10)]"
        if col not in ann.columns:
            raise RuntimeError(
                f"SDRF is missing {col!r}; the index-sort protein measurements "
                "are the held-out validation channel and cannot be skipped."
            )
        adata.obs[f"prot_{prot}"] = pd.to_numeric(ann[col], errors="coerce").to_numpy(
            dtype=float
        )

    if arm == "N4":
        vc = pd.Series(adata.obs["hours"]).value_counts().to_dict()
        _expect(
            {float(k): int(v) for k, v in vc.items()},
            {0.0: 44, 1.0: 51, 3.0: 64, 6.0: 91},
            "cells per timepoint",
        )

    # ---- Ensembl -> MGI symbol -------------------------------------------
    mapping = _ensembl_to_symbol(dest, verbose)
    adata.var["ensembl_id"] = adata.var_names.astype(str)
    adata.var["symbol"] = [mapping.get(g) for g in adata.var["ensembl_id"]]
    n_unmapped = int(adata.var["symbol"].isna().sum())
    adata = adata[:, adata.var["symbol"].notna()].copy()
    if verbose:
        print(
            f"  symbol-mapped: kept {adata.n_vars}, " f"dropped {n_unmapped} unmapped"
        )

    # Collapse duplicate symbols by summing, so a symbol means exactly one row.
    sym = adata.var["symbol"].astype(str).to_numpy()
    if len(set(sym)) != len(sym):
        summed = pd.DataFrame(adata.X, columns=sym).T.groupby(level=0).sum().T
        adata = ad.AnnData(
            X=summed.to_numpy(dtype=np.float32),
            obs=adata.obs.copy(),
            var=pd.DataFrame(index=pd.Index(summed.columns.astype(str))),
        )
        if verbose:
            print(f"  collapsed duplicate symbols -> {adata.n_vars} genes")
    else:
        adata.var_names = sym
    adata.var_names_make_unique()

    if not preprocess:
        return adata

    # ---- QC, normalisation, HVG, PCA -------------------------------------
    # The matrix is already the authors' post-QC set (every retained well is
    # flagged OK/pass), so only undetected genes are dropped -- cells are not
    # re-filtered.
    adata.var["mt"] = adata.var_names.astype(str).str.startswith("mt-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    n_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=3)
    if verbose:
        print(f"  gene filter (min_cells=3): {n_before} -> {adata.n_vars}")

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

    # PCA on scaled HVGs, but `.raw` keeps the UNSCALED lognorm matrix: the TF,
    # pathway and gene views all want lognorm rather than z-scored input, and a
    # marker dropped by the HVG filter must still be scoreable.
    adata.raw = adata
    hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(hvg, max_value=10)
    sc.tl.pca(
        hvg, n_comps=min(30, hvg.n_obs - 1), svd_solver="arpack", random_state=seed
    )
    adata.obsm["X_pca"] = hvg.obsm["X_pca"].copy()
    adata.uns["pca"] = dict(hvg.uns["pca"])
    if verbose:
        vr = np.round(adata.uns["pca"]["variance_ratio"][:5], 3)
        print(
            f"  HVG: {int(adata.var['highly_variable'].sum())}, "
            f"PCA {adata.obsm['X_pca'].shape}, var(PC1-5)={vr}"
        )
    return adata


def marker_panels() -> dict[str, list[str]]:
    """The paper's marker panels, as ``{name: genes}``."""
    return {
        "early_TF": list(EARLY_TF),
        "fig1e": list(FIG1E),
        "late_effector": list(LATE),
        "surface": list(SURFACE_GENES.values()),
    }


def present(adata, genes: Iterable[str]) -> list[str]:
    """Subset ``genes`` to those actually in ``adata`` (checking ``.raw``).

    Scoring a panel silently over its intersection is how a missing marker turns
    into an apparent biological absence, so callers should report what this drops.
    """
    src = adata.raw if adata.raw is not None else adata
    have = set(map(str, src.var_names))
    return [g for g in genes if g in have]
