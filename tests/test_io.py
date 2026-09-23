import pytest

from spatialcyte.io import TiffReader


def test_reads_channels(synthetic_image):
    r = TiffReader(synthetic_image)
    assert r.n_channels == 3
    assert r.shape == (32, 32)
    ch0 = r.read_channel(0)
    assert ch0.shape == (32, 32)


def test_out_of_range_channel_raises(synthetic_image):
    r = TiffReader(synthetic_image)
    with pytest.raises(IndexError):
        r.read_channel(9)


def test_negative_level_raises(synthetic_image):
    with pytest.raises(ValueError, match="level must be >= 0"):
        TiffReader(synthetic_image, level=-1)


def test_reads_correct_channel_content(tmp_path):
    import numpy as np

    tifffile = pytest.importorskip("tifffile")
    arr = np.zeros((3, 8, 8), np.uint16)
    arr[0], arr[1], arr[2] = 10, 20, 30
    p = tmp_path / "x.tif"
    tifffile.imwrite(str(p), arr, photometric="minisblack")
    r = TiffReader(p)
    assert r.read_channel(0).mean() == 10
    assert r.read_channel(2).mean() == 30
    assert r.read_channel(1).shape == (8, 8)
