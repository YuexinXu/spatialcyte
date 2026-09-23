import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spatialcyte.config import PanelConfig
from spatialcyte.phenotype import gate, mark_positive, phenotype


def _bimodal_adata():
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.normal(1.0, 0.2, 150), rng.normal(6.0, 0.2, 150)])
    a = ad.AnnData(X=vals.reshape(-1, 1).astype("float32"), var=pd.DataFrame(index=["STAB1"]))
    a.layers["arcsinh"] = a.X.copy()
    return a, 150


@pytest.mark.parametrize("method", ["gmm", "otsu"])
def test_mark_positive_splits_bimodal(method):
    a, n = _bimodal_adata()
    mark_positive(a, method=method)
    p = a.obs["STAB1_positive"].to_numpy()
    assert p[n:].mean() > 0.95  # high mode -> positive
    assert p[:n].mean() < 0.05  # low mode -> negative


def test_gate_first_match_wins():
    a = ad.AnnData(
        X=np.zeros((4, 1), "float32"),
        var=pd.DataFrame(index=["x"]),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
    )
    a.obs["CD3_positive"] = [True, True, False, False]
    a.obs["CD4_positive"] = [True, False, False, False]
    a.obs["FOXP3_positive"] = [True, False, False, False]
    a.obs["CD8_positive"] = [False, True, False, False]
    a.obs["CD68_positive"] = [False, False, True, False]
    gates = [
        {"label": "Treg", "all": ["CD3", "CD4", "FOXP3"]},
        {"label": "CD8 T", "all": ["CD3", "CD8"]},
        {"label": "CD4 T", "all": ["CD3", "CD4"]},
        {"label": "TAM", "all": ["CD68"]},
        {"label": "Other"},
    ]
    gate(a, gates)
    assert list(a.obs["phenotype"]) == ["Treg", "CD8 T", "TAM", "Other"]


def test_mark_positive_unknown_method_raises():
    a, _ = _bimodal_adata()
    with pytest.raises(ValueError, match="unknown positivity method"):
        mark_positive(a, method="bogus")


def test_gate_missing_marker_raises():
    a = ad.AnnData(X=np.zeros((2, 1), "float32"), var=pd.DataFrame(index=["x"]))
    with pytest.raises(KeyError, match="mark_positive"):
        gate(a, [{"label": "T", "all": ["CD3"]}])


def test_phenotype_from_config():
    a, _ = _bimodal_adata()
    cfg = PanelConfig.from_dict(
        {
            "pixel_size_um": 0.5,
            "channels": [
                {"index": 0, "marker": "DAPI", "role": "nuclear"},
                {"index": 1, "marker": "STAB1", "role": "functional"},
            ],
            "phenotype": {
                "positivity": {"method": "otsu"},
                "gates": [{"label": "STAB1hi", "all": ["STAB1"]}, {"label": "Other"}],
            },
        }
    )
    phenotype(a, cfg)
    assert set(a.obs["phenotype"].unique()) <= {"STAB1hi", "Other"}
    assert (a.obs["phenotype"] == "STAB1hi").sum() > 100
