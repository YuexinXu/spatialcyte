import numpy as np
import pytest

from spatialcyte.config import PanelConfig


@pytest.fixture
def panel():
    """A small 3-channel panel matching the synthetic image."""
    return PanelConfig.from_dict(
        {
            "pixel_size_um": 0.5,
            "channels": [
                {"index": 0, "marker": "DAPI", "role": "nuclear"},
                {"index": 1, "marker": "CD31", "role": "region", "region": "vessel"},
                {"index": 2, "marker": "STAB1", "role": "functional"},
            ],
        }
    )


@pytest.fixture
def synthetic_image(tmp_path):
    """A tiny (C=3, H=32, W=32) uint16 TIFF. Channel 2 is left blank on purpose
    so QC has a problem to flag."""
    tifffile = pytest.importorskip("tifffile")
    rng = np.random.default_rng(0)
    arr = (rng.random((3, 32, 32)) * 1000).astype(np.uint16)
    arr[2] = 0  # blank STAB1 channel
    path = tmp_path / "img.tif"
    tifffile.imwrite(str(path), arr, photometric="minisblack")
    return path
