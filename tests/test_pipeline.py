import numpy as np
import pytest

from spatialcyte.config import PanelConfig
from spatialcyte.pipeline import build, main


class ArrayReader:
    def __init__(self, arr):
        self.arr = np.asarray(arr)

    @property
    def n_channels(self):
        return self.arr.shape[0]

    @property
    def shape(self):
        return self.arr.shape[1:]

    def read_channel(self, i):
        return self.arr[i]


def _cfg():
    return PanelConfig.from_dict(
        {
            "pixel_size_um": 0.5,
            "channels": [
                {"index": 0, "marker": "DAPI", "role": "nuclear"},
                {"index": 1, "marker": "CD31", "role": "region", "region": "vessel"},
                {"index": 2, "marker": "STAB1", "role": "functional"},
            ],
            "phenotype": {
                "positivity": {"method": "otsu"},
                "gates": [{"label": "STAB1hi", "all": ["STAB1"]}, {"label": "Other"}],
            },
        }
    )


def _image():
    H = W = 40
    dapi = np.zeros((H, W), np.float32)
    for y, x in [(4, 4), (4, 24), (24, 4), (24, 24)]:
        dapi[y : y + 8, x : x + 8] = 800
    cd31 = np.zeros((H, W), np.float32)
    cd31[0:3, 0:20] = 600
    stab1 = np.zeros((H, W), np.float32)
    stab1[4:12, 4:12] = 900  # one strongly positive cell
    return np.stack([dapi, cd31, stab1])


def test_build_end_to_end():
    a = build(ArrayReader(_image()), _cfg(), backend="threshold")
    assert a.n_obs == 4
    assert list(a.var_names) == ["DAPI", "CD31", "STAB1"]
    assert "phenotype" in a.obs
    assert "dist_to_vessel_um" in a.obs
    assert "spatial_connectivities" in a.obsp
    assert np.isfinite(np.asarray(a.layers["membrane"])).all()


def test_run_cli_writes_h5ad(tmp_path):
    import anndata as ad
    import tifffile

    img = tmp_path / "img.tif"
    tifffile.imwrite(str(img), _image().astype("uint16"), photometric="minisblack")
    panel = tmp_path / "panel.yaml"
    panel.write_text(
        "pixel_size_um: 0.5\n"
        "channels:\n"
        "  - {index: 0, marker: DAPI, role: nuclear}\n"
        "  - {index: 1, marker: CD31, role: region, region: vessel}\n"
        "  - {index: 2, marker: STAB1, role: functional}\n"
        "phenotype:\n"
        "  positivity: {method: otsu}\n"
        "  gates: [{label: STAB1hi, all: [STAB1]}, {label: Other}]\n"
    )
    out = tmp_path / "out.h5ad"
    rc = main(["run", "--config", str(panel), "--image", str(img), "--out", str(out), "--force"])
    assert rc == 0
    a = ad.read_h5ad(str(out))
    assert a.n_obs == 4
    assert "phenotype" in a.obs


def test_build_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        build(ArrayReader(_image()), _cfg(), backend="nope")
