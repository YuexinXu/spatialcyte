"""Panel configuration.

Declares what each image channel is (marker + role), so the rest of the
pipeline is panel-agnostic. A panel is normally written once as YAML and loaded
with :meth:`PanelConfig.from_yaml`.

Roles:
    nuclear     nuclear stain used for cell segmentation (e.g. DAPI)
    region      structural marker segmented into regions (e.g. CD31 -> vessel)
    lineage     cell-type marker used downstream for phenotyping
    functional  marker of interest quantified per cell (e.g. STAB1)
    exclude     not analysed (e.g. autofluorescence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROLES = frozenset({"nuclear", "region", "lineage", "functional", "exclude"})


@dataclass(frozen=True)
class Channel:
    """One image channel and its biological role."""

    index: int
    marker: str
    role: str
    region: str | None = None  # required when role == "region"

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(
                f"channel {self.index} ({self.marker}): role {self.role!r} not in {sorted(ROLES)}"
            )
        if self.role == "region" and not self.region:
            raise ValueError(
                f"channel {self.index} ({self.marker}): role 'region' needs a 'region' "
                f"name (e.g. region: vessel)"
            )


@dataclass(frozen=True)
class PanelConfig:
    """A validated marker panel plus segmentation/quantification settings."""

    pixel_size_um: float
    channels: tuple[Channel, ...]
    segmentation: dict[str, Any] = field(default_factory=dict)
    quantify: dict[str, Any] = field(default_factory=dict)
    phenotype: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pixel_size_um <= 0:
            raise ValueError(f"pixel_size_um must be > 0, got {self.pixel_size_um}")
        if not self.channels:
            raise ValueError("panel has no channels")
        indices = [c.index for c in self.channels]
        if len(set(indices)) != len(indices):
            raise ValueError(f"duplicate channel indices: {sorted(indices)}")
        if not self.nuclear_channels:
            raise ValueError("panel needs at least one 'nuclear' channel for segmentation")

    # --- convenience views -------------------------------------------------
    @property
    def nuclear_channels(self) -> tuple[Channel, ...]:
        return tuple(c for c in self.channels if c.role == "nuclear")

    @property
    def region_channels(self) -> tuple[Channel, ...]:
        return tuple(c for c in self.channels if c.role == "region")

    @property
    def measured_channels(self) -> tuple[Channel, ...]:
        """Channels whose intensity is quantified (everything except 'exclude')."""
        return tuple(c for c in self.channels if c.role != "exclude")

    @property
    def markers(self) -> list[str]:
        """Marker names, in channel order, for the quantified channels."""
        return [c.marker for c in self.measured_channels]

    def channel_for(self, marker: str) -> Channel:
        for c in self.channels:
            if c.marker == marker:
                return c
        raise KeyError(f"no channel for marker {marker!r}")

    # --- constructors ------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PanelConfig:
        try:
            channels = tuple(
                Channel(
                    index=int(c["index"]),
                    marker=str(c["marker"]),
                    role=str(c["role"]),
                    region=c.get("region"),
                )
                for c in d["channels"]
            )
        except KeyError as e:
            raise ValueError(f"panel/channel entry missing required key: {e}") from e
        return cls(
            pixel_size_um=float(d["pixel_size_um"]),
            channels=channels,
            segmentation=dict(d.get("segmentation", {})),
            quantify=dict(d.get("quantify", {})),
            phenotype=dict(d.get("phenotype", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> PanelConfig:
        import yaml

        with open(path) as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a YAML mapping at the top level")
        return cls.from_dict(data)
