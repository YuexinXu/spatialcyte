"""Regression against a reference stage-2 implementation.

Validated 2026-06-12 against an independently written reference feature-extraction
stage on one whole-slide RCC 8-plex sample (33,345 cells x 8 markers):
``spatialcyte.quantify`` reproduced the whole-cell means to within the tolerance
asserted below.

That run uses local patient data, so it is not committed here. To re-run, set
``SPATIALCYTE_REGRESSION_DIR`` to a directory holding a prepared, deidentified
fixture: ``cell_mask.tif`` (label image), ``image.tif`` (channels at the same
resolution as the mask), ``reference.h5ad`` (the reference single-cell object),
and ``panel.yaml``.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("SPATIALCYTE_REGRESSION_DIR"),
    reason="set SPATIALCYTE_REGRESSION_DIR to a prepared reference fixture dir",
)


def test_quantify_matches_reference():
    from pathlib import Path

    import anndata
    import numpy as np
    import tifffile

    from spatialcyte.config import PanelConfig
    from spatialcyte.io import TiffReader
    from spatialcyte.quantify import quantify

    d = Path(os.environ["SPATIALCYTE_REGRESSION_DIR"])
    mask = tifffile.imread(str(d / "cell_mask.tif"))
    cfg = PanelConfig.from_yaml(d / "panel.yaml")
    new = quantify(TiffReader(d / "image.tif"), mask, cfg, compartments=[])
    ref = anndata.read_h5ad(str(d / "reference.h5ad"))

    ref_markers = list(ref.var_names)
    shared = [m for m in new.var_names if m in ref_markers]
    assert shared, "no shared markers between quantify output and reference"
    for m in shared:
        a = np.asarray(new[:, m].X).ravel()
        b = np.asarray(ref[:, m].X).ravel()
        assert a.shape == b.shape
        assert np.allclose(a, b, atol=1e-3), f"{m}: whole-cell means differ"
