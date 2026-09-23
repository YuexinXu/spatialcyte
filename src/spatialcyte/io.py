"""Image readers.

A reader exposes the channels of one multiplexed image so the rest of the
pipeline never cares about the on-disk format. New formats (QPTIFF, HALO object
data, OME-Zarr) are added by implementing :class:`ImageReader`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class ImageReader(ABC):
    """Read-only view of a multiplexed image as indexed 2D channels."""

    @property
    @abstractmethod
    def n_channels(self) -> int: ...

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """(height, width) at full resolution."""

    @abstractmethod
    def read_channel(self, index: int) -> np.ndarray:
        """Return channel ``index`` as a 2D array."""


class TiffReader(ImageReader):
    """Multi-channel TIFF reader (channel-first ``CYX``, ``YXC``, or 2D).

    Reads one channel at a time. When the channels are stored as separate pages
    (the common multiplexed-IHC / HALO layout) reads are lazy — only the requested
    page is decoded — so large images don't load whole into RAM. Otherwise the
    image is loaded once and cached. ``level`` selects a resolution level of a
    pyramidal TIFF (0 = full resolution).
    """

    def __init__(self, path: str | Path, *, level: int = 0):
        import tifffile

        self.path = Path(path)
        self._tif = tifffile.TiffFile(str(self.path))
        series = self._tif.series[0]
        if level < 0:
            raise ValueError(f"{self.path}: pyramid level must be >= 0, got {level}")
        if level:
            levels = getattr(series, "levels", None) or [series]
            if level >= len(levels):
                raise ValueError(
                    f"{self.path}: pyramid level {level} unavailable ({len(levels)} level(s))"
                )
            series = levels[level]
        self.level = level
        self._series = series
        shape = series.shape
        if len(shape) == 2:
            self._n, self._hw, self._caxis = 1, shape, 0
        elif len(shape) == 3:
            # prefer the labelled channel axis (OME 'C'/'S') when tifffile knows
            # it; fall back to "smallest axis" (markers are few, H/W large).
            # The fallback is only wrong when a spatial dim <= channel count
            # (e.g. a highly-multiplexed panel on a tiny crop) — rare for mIHC.
            axes = (getattr(series, "axes", "") or "").upper()
            caxis = next(
                (axes.index(a) for a in ("C", "S") if len(axes) == 3 and axes.count(a) == 1),
                None,
            )
            self._caxis = int(np.argmin(shape)) if caxis is None else caxis
            self._n = shape[self._caxis]
            self._hw = tuple(s for k, s in enumerate(shape) if k != self._caxis)
        else:
            raise ValueError(f"{self.path}: expected a 2D or 3D image, got shape {shape}")
        # lazy per-page reads when channels are channel-first separate pages
        self._lazy = self._caxis == 0 and len(series.pages) == self._n
        self._full: np.ndarray | None = None

    @property
    def n_channels(self) -> int:
        return int(self._n)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self._hw[0]), int(self._hw[1]))

    def read_channel(self, index: int) -> np.ndarray:
        if not 0 <= index < self._n:
            raise IndexError(f"channel {index} out of range (n={self._n})")
        if self._lazy:
            return np.asarray(self._series.pages[index].asarray())
        if self._full is None:
            arr = np.asarray(self._series.asarray())
            if arr.ndim == 2:
                arr = arr[None, ...]
            elif self._caxis != 0:
                arr = np.moveaxis(arr, self._caxis, 0)
            self._full = arr
        return self._full[index]
