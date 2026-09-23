import pytest

from spatialcyte.config import Channel, PanelConfig


def _panel_dict():
    return {
        "pixel_size_um": 0.5,
        "channels": [
            {"index": 0, "marker": "DAPI", "role": "nuclear"},
            {"index": 1, "marker": "CD31", "role": "region", "region": "vessel"},
            {"index": 2, "marker": "CD8", "role": "lineage"},
            {"index": 5, "marker": "STAB1", "role": "functional"},
            {"index": 8, "marker": "AF", "role": "exclude"},
        ],
    }


def test_from_dict_views():
    p = PanelConfig.from_dict(_panel_dict())
    assert p.pixel_size_um == 0.5
    assert [c.marker for c in p.nuclear_channels] == ["DAPI"]
    assert [c.region for c in p.region_channels] == ["vessel"]
    # 'exclude' channel (AF) is dropped from the quantified markers
    assert p.markers == ["DAPI", "CD31", "CD8", "STAB1"]
    assert p.channel_for("STAB1").index == 5


def test_region_without_name_raises():
    with pytest.raises(ValueError, match="region"):
        Channel(index=1, marker="CD31", role="region")


def test_bad_role_raises():
    with pytest.raises(ValueError, match="role"):
        Channel(index=0, marker="x", role="bogus")


def test_needs_a_nuclear_channel():
    d = _panel_dict()
    d["channels"] = [c for c in d["channels"] if c["role"] != "nuclear"]
    with pytest.raises(ValueError, match="nuclear"):
        PanelConfig.from_dict(d)


def test_duplicate_index_raises():
    d = _panel_dict()
    d["channels"][1]["index"] = 0
    with pytest.raises(ValueError, match="duplicate"):
        PanelConfig.from_dict(d)


def test_bad_pixel_size_raises():
    d = _panel_dict()
    d["pixel_size_um"] = 0
    with pytest.raises(ValueError, match="pixel_size_um"):
        PanelConfig.from_dict(d)


def test_from_yaml_roundtrip(tmp_path):
    yaml = pytest.importorskip("yaml")
    f = tmp_path / "panel.yaml"
    f.write_text(yaml.safe_dump(_panel_dict()))
    p = PanelConfig.from_yaml(f)
    assert p.markers == ["DAPI", "CD31", "CD8", "STAB1"]
