"""Input/Output utilities for loading and saving data."""

from pathlib import Path
from typing import Union

import scanpy as sc
from anndata import AnnData


def load_raw_data(
    adata_file: Union[str, Path],
    experiment_filter: str = None,
    min_genes: int = 100,
    min_cells: int = 3,
    n_top_genes: int = 5000,
    batch_key: str = "experiment",
) -> AnnData:
    """
    Load and preprocess raw single-cell data.

    Parameters
    ----------
    adata_file : str or Path
        Path to h5ad file containing raw data
    experiment_filter : str, optional
        Experiment name to exclude from analysis
    min_genes : int, default=100
        Minimum number of genes per cell
    min_cells : int, default=3
        Minimum number of cells per gene
    n_top_genes : int, default=5000
        Number of highly variable genes to identify
    batch_key : str, default='experiment'
        Key in obs for batch correction

    Returns
    -------
    AnnData
        Preprocessed AnnData object with:
        - Filtered cells and genes
        - Normalized and log-transformed data
        - Highly variable genes identified
        - Raw counts stored in layers['counts']
    """
    adata = sc.read_h5ad(adata_file)

    # Filter experiment if specified
    if experiment_filter is not None and batch_key in adata.obs.columns:
        initial_shape = adata.shape
        adata = adata[adata.obs[batch_key] != experiment_filter]
        print(f"Filtered {experiment_filter}: {initial_shape} -> {adata.shape}")

    # Quality control filtering
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Store raw counts
    adata.layers["counts"] = adata.X.copy()

    # Normalize and log-transform
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # Identify highly variable genes
    if batch_key in adata.obs.columns:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key=batch_key)
    else:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

    return adata
