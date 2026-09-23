"""Stage 0 — QC.

Validate the image against the panel and summarise each channel's intensity so
problems (dead channels, saturation, panel/image mismatch) surface before
segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spatialcyte.config import Channel, PanelConfig
from spatialcyte.io import ImageReader


@dataclass
class ChannelQC:
    index: int
    marker: str
    role: str
    min: float
    max: float
    mean: float
    p99: float
    saturated_frac: float  # fraction at the dtype/observed ceiling
    blank_frac: float  # fraction exactly zero
    warnings: list[str] = field(default_factory=list)


@dataclass
class QCReport:
    channels: list[ChannelQC] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings and all(not c.warnings for c in self.channels)


def run_qc(reader: ImageReader, config: PanelConfig) -> QCReport:
    """QC every measured channel of ``reader`` against ``config``."""
    report = QCReport()
    max_idx = max(c.index for c in config.channels)
    if reader.n_channels <= max_idx:
        report.warnings.append(
            f"panel references channel index up to {max_idx} but the image has only "
            f"{reader.n_channels} channels"
        )
    for ch in config.measured_channels:
        if ch.index >= reader.n_channels:
            report.warnings.append(f"{ch.marker}: channel {ch.index} missing from image")
            continue
        report.channels.append(_channel_qc(reader.read_channel(ch.index), ch))
    return report


def _channel_qc(arr: np.ndarray, ch: Channel) -> ChannelQC:
    a = np.asarray(arr).ravel()
    n = a.size
    if n == 0:
        return ChannelQC(
            ch.index, ch.marker, ch.role, 0, 0, 0, 0, 0.0, 1.0, warnings=["empty channel"]
        )
    amin, amax, mean = float(a.min()), float(a.max()), float(a.mean())
    p99 = float(np.percentile(a, 99))
    if np.issubdtype(a.dtype, np.integer):
        ceil = np.iinfo(a.dtype).max
        sat = float(np.count_nonzero(a >= ceil) / n) if ceil > 0 else 0.0
    else:
        # float has no fixed ceiling; use the observed max, but a flat channel
        # (max == min) is not "saturated" — avoid a 100%-saturation false alarm.
        sat = float(np.count_nonzero(a >= amax) / n) if amax > amin else 0.0
    blank = float(np.count_nonzero(a == 0) / n)

    warnings: list[str] = []
    if blank >= 0.99:
        warnings.append("near-blank channel (>=99% zero)")
    if sat >= 0.10:
        warnings.append(f"possible saturation ({sat:.0%} at ceiling)")
    return ChannelQC(
        ch.index, ch.marker, ch.role, amin, amax, mean, p99, sat, blank, warnings=warnings
    )
