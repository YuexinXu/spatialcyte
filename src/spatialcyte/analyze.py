"""Stage 4 — analyze.

Panel-agnostic summaries over the single-cell object: marker positivity rates,
phenotype composition, and permutation-based neighbourhood enrichment over the
spatial graph.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd


def positivity_summary(
    adata: ad.AnnData, markers: list[str] | None = None, *, groupby: str | None = None
) -> pd.DataFrame:
    """Fraction of cells positive per marker, optionally per group."""
    cols = (
        [f"{m}_positive" for m in markers]
        if markers
        else [c for c in adata.obs.columns if c.endswith("_positive")]
    )
    if not cols:
        raise ValueError("no '<marker>_positive' columns; run phenotype.mark_positive first")
    df = adata.obs[cols].astype(float).copy()
    df.columns = [c[: -len("_positive")] for c in cols]
    if groupby:
        df[groupby] = adata.obs[groupby].to_numpy()
        return df.groupby(groupby, observed=True).mean()
    return df.mean().to_frame("positive_fraction")


def composition(
    adata: ad.AnnData, key: str = "phenotype", *, groupby: str | None = None
) -> pd.DataFrame:
    """Phenotype frequencies, optionally per group."""
    if groupby:
        return (
            adata.obs.groupby(groupby, observed=True)[key]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
        )
    return adata.obs[key].value_counts(normalize=True).to_frame("fraction")


def neighborhood_enrichment(
    adata: ad.AnnData,
    key: str = "phenotype",
    *,
    graph: str = "spatial_connectivities",
    n_perms: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    """Permutation Z-scores of phenotype-pair adjacency over the spatial graph.

    Positive Z ⇒ the two phenotypes neighbour each other more than chance;
    negative ⇒ they are spatially segregated.
    """
    if graph not in adata.obsp:
        raise KeyError(f"no obsp[{graph!r}]; build the spatial graph in quantify first")
    A = adata.obsp[graph]
    A = ((A + A.T) > 0).astype(np.float64)  # undirected, binary
    cats = adata.obs[key].astype("category")
    codes = cats.cat.codes.to_numpy()
    K = len(cats.cat.categories)

    def pair_counts(c: np.ndarray) -> np.ndarray:
        oh = np.zeros((c.size, K))
        oh[np.arange(c.size), c] = 1.0
        return oh.T @ (A @ oh)

    observed = pair_counts(codes)
    rng = np.random.default_rng(seed)
    null = np.empty((n_perms, K, K))
    for p in range(n_perms):
        null[p] = pair_counts(rng.permutation(codes))
    mu, sd = null.mean(0), null.std(0)
    sd[sd == 0] = 1.0
    z = (observed - mu) / sd
    return pd.DataFrame(z, index=cats.cat.categories, columns=cats.cat.categories)
