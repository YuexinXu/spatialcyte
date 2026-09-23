import numpy as np
import pytest

from spatialcyte.config import PanelConfig
from spatialcyte.quantify import quantify


class ArrayReader:
    """Minimal ImageReader over an in-memory (C, H, W) array (test double)."""

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


@pytest.fixture
def scene():
    """Two 8x8 cells with analytically-known intensities."""
    H = W = 40
    cell_mask = np.zeros((H, W), np.int32)
    cell_mask[4:12, 4:12] = 1
    cell_mask[24:32, 24:32] = 2

    dapi = np.zeros((H, W), np.float32)
    dapi[cell_mask == 1] = 100
    dapi[cell_mask == 2] = 200

    stab1 = np.zeros((H, W), np.float32)
    stab1[cell_mask == 1] = 100  # cell1 rim default
    stab1[6:10, 6:10] = 10  # cell1 core (= eroded-by-2 interior) = 10
    stab1[cell_mask == 2] = 70
    # 1-px signal frame just outside cell1 -> its extracellular ring
    stab1[3, 3:13] = 500
    stab1[12, 3:13] = 500
    stab1[3:13, 3] = 500
    stab1[3:13, 12] = 500

    cfg = PanelConfig.from_dict(
        {
            "pixel_size_um": 0.5,
            "channels": [
                {"index": 0, "marker": "DAPI", "role": "nuclear"},
                {"index": 1, "marker": "STAB1", "role": "functional"},
            ],
            "quantify": {
                "compartments": ["membrane", "cytoplasm", "extracellular"],
                "arcsinh_cofactor": 5,
                "spatial_graph": {"k": 1},
            },
        }
    )
    return ArrayReader(np.stack([dapi, stab1])), cell_mask, cfg


def test_whole_cell_means_exact(scene):
    reader, mask, cfg = scene
    a = quantify(reader, mask, cfg)
    assert a.shape == (2, 2)
    assert list(a.var_names) == ["DAPI", "STAB1"]
    dapi = a[:, "DAPI"].X.ravel()
    assert dapi[0] == 100
    assert dapi[1] == 200


def test_compartments_exact(scene):
    reader, mask, cfg = scene
    a = quantify(reader, mask, cfg, erosion_px=2)
    cyto = a[:, "STAB1"].layers["cytoplasm"].ravel()
    mem = a[:, "STAB1"].layers["membrane"].ravel()
    assert cyto[0] == pytest.approx(10)
    assert mem[0] == pytest.approx(100)
    assert "arcsinh" in a.layers


def test_extracellular_ring(scene):
    reader, mask, cfg = scene
    a = quantify(reader, mask, cfg, ring_px=3)
    extra = a[:, "STAB1"].layers["extracellular"].ravel()
    assert extra[0] > 0  # cell1 surrounded by signal
    assert extra[1] == 0  # cell2 surroundings are empty


def test_region_distances_and_graph(scene):
    reader, mask, cfg = scene
    H, W = mask.shape
    vessel = np.zeros((H, W), bool)
    vessel[0:3, 0:3] = True  # near cell1
    a = quantify(reader, mask, cfg, region_masks={"vessel": vessel})
    assert "dist_to_vessel_um" in a.obs
    d = a.obs["dist_to_vessel_um"].to_numpy()
    assert d[0] < d[1]  # cell1 is closer to the vessel
    assert a.obsp["spatial_distances"].shape == (2, 2)
    assert a.obsp["spatial_connectivities"].shape == (2, 2)
    # spatial coords are µm (centroid 7.5 px * 0.5 µm/px)
    assert a.obsm["spatial"][0][0] == pytest.approx(3.75, abs=0.01)


def test_uns_provenance(scene):
    reader, mask, cfg = scene
    a = quantify(reader, mask, cfg)
    assert a.uns["spatialcyte"]["pixel_size_um"] == 0.5
    assert set(a.uns["spatialcyte"]["compartments"]) == {"membrane", "cytoplasm", "extracellular"}


@pytest.fixture
def dense_scene():
    """A 3x3 tiling of touching cells (no background between them) — the dense
    TME case where naive union-erosion produced NaN compartment means."""
    cell = 6
    n_side = 3
    H = W = cell * n_side
    cell_mask = np.zeros((H, W), np.int32)
    lab = 1
    for i in range(n_side):
        for j in range(n_side):
            cell_mask[i * cell : (i + 1) * cell, j * cell : (j + 1) * cell] = lab
            lab += 1
    img = np.stack([(cell_mask > 0).astype(np.float32) * 100, np.full((H, W), 50, np.float32)])
    cfg = PanelConfig.from_dict(
        {
            "pixel_size_um": 0.5,
            "channels": [
                {"index": 0, "marker": "DAPI", "role": "nuclear"},
                {"index": 1, "marker": "STAB1", "role": "functional"},
            ],
            "quantify": {"compartments": ["membrane", "cytoplasm", "extracellular"]},
        }
    )
    return ArrayReader(img), cell_mask, cfg


def test_no_nan_compartments_in_dense_tissue(dense_scene):
    reader, mask, cfg = dense_scene
    a = quantify(reader, mask, cfg, erosion_px=1, ring_px=3)
    for layer in ("membrane", "cytoplasm", "extracellular"):
        vals = a.layers[layer]
        assert np.isfinite(vals).all(), f"{layer} has non-finite values"
    # the fully-enclosed centre cell (label 5) must still have a membrane
    centre = list(a.obs_names).index("cell_5")
    mem = a[:, "STAB1"].layers["membrane"].ravel()
    assert mem[centre] > 0  # enclosed cell has a membrane (shared borders)


def test_touching_cells_membrane_not_leaked(dense_scene):
    """Shared cell-cell borders are membrane, not cytoplasm; per-cell interiors
    do not merge across the union."""
    reader, mask, cfg = dense_scene
    a = quantify(reader, mask, cfg, erosion_px=1)
    # every 6x6 touching cell has a genuine (non-empty) membrane and interior
    mem = a[:, "STAB1"].layers["membrane"].ravel()
    cyto = a[:, "STAB1"].layers["cytoplasm"].ravel()
    assert (mem > 0).all()
    assert (cyto > 0).all()


def test_absent_region_distance_is_inf(scene):
    reader, mask, cfg = scene
    H, W = mask.shape
    empty = np.zeros((H, W), bool)  # region declared but not present
    a = quantify(reader, mask, cfg, region_masks={"vessel": empty})
    d = a.obs["dist_to_vessel_um"].to_numpy()
    assert np.isinf(d).all()  # not a fabricated finite distance


def test_unknown_compartment_raises(scene):
    reader, cell_mask, cfg = scene
    with pytest.raises(ValueError, match="unknown compartment"):
        quantify(reader, cell_mask, cfg, compartments=["membrain"])
