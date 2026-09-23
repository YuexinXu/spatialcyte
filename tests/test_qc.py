from spatialcyte.io import TiffReader
from spatialcyte.qc import run_qc


def test_qc_covers_measured_channels(synthetic_image, panel):
    report = run_qc(TiffReader(synthetic_image), panel)
    assert [c.marker for c in report.channels] == panel.markers
    assert len(report.channels) == 3


def test_qc_flags_blank_channel(synthetic_image, panel):
    report = run_qc(TiffReader(synthetic_image), panel)
    stab1 = next(c for c in report.channels if c.marker == "STAB1")
    assert stab1.blank_frac == 1.0
    assert any("blank" in w for w in stab1.warnings)
    assert not report.ok  # the blank channel makes the report not-ok


def test_qc_warns_on_panel_image_mismatch(synthetic_image):
    # panel references channel 5, image only has 3 channels
    from spatialcyte.config import PanelConfig

    cfg = PanelConfig.from_dict(
        {
            "pixel_size_um": 0.5,
            "channels": [
                {"index": 0, "marker": "DAPI", "role": "nuclear"},
                {"index": 5, "marker": "STAB1", "role": "functional"},
            ],
        }
    )
    report = run_qc(TiffReader(synthetic_image), cfg)
    assert any("only" in w and "channels" in w for w in report.warnings)
