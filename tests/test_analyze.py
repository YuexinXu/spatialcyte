import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from spatialcyte.analyze import composition, neighborhood_enrichment, positivity_summary


def _segregated():
    """10 cells; A's all neighbour A's, B's all neighbour B's, no A–B edges."""
    n = 10
    A = np.zeros((n, n))
    A[:5, :5] = 1
    A[5:, 5:] = 1
    np.fill_diagonal(A, 0)
    a = ad.AnnData(
        X=np.zeros((n, 1), "float32"),
        var=pd.DataFrame(index=["x"]),
        obs=pd.DataFrame({"phenotype": ["A"] * 5 + ["B"] * 5}, index=[f"c{i}" for i in range(n)]),
    )
    a.obsp["spatial_connectivities"] = csr_matrix(A)
    return a


def test_neighborhood_enrichment_detects_segregation():
    z = neighborhood_enrichment(_segregated(), n_perms=200, seed=0)
    assert z.loc["A", "A"] > 0  # A clusters with A
    assert z.loc["A", "B"] < 0  # A and B are segregated
    assert z.loc["A", "A"] > z.loc["A", "B"]


def test_neighborhood_enrichment_needs_graph():
    a = ad.AnnData(
        X=np.zeros((2, 1), "float32"),
        obs=pd.DataFrame({"phenotype": ["A", "B"]}, index=["c0", "c1"]),
    )
    with pytest.raises(KeyError, match="spatial graph"):
        neighborhood_enrichment(a)


def test_positivity_summary():
    a = ad.AnnData(
        X=np.zeros((4, 1), "float32"),
        var=pd.DataFrame(index=["x"]),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
    )
    a.obs["CD3_positive"] = [True, True, False, False]
    a.obs["CD8_positive"] = [True, False, False, False]
    s = positivity_summary(a)
    assert s.loc["CD3", "positive_fraction"] == 0.5
    assert s.loc["CD8", "positive_fraction"] == 0.25


def test_composition():
    a = ad.AnnData(
        X=np.zeros((4, 1), "float32"),
        obs=pd.DataFrame(
            {"phenotype": ["A", "A", "B", "Other"]}, index=[f"c{i}" for i in range(4)]
        ),
    )
    assert composition(a).loc["A", "fraction"] == 0.5
