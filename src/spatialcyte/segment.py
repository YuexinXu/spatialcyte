"""Stage 1 — segmentation.

Cells are segmented from the nuclear marker; regions (e.g. vessels) from a
structural marker. Cell segmentation is pluggable via :class:`Segmenter`:
Cellpose is the production backend (optional ``[cellpose]`` extra), and
:class:`ThresholdSegmenter` is a dependency-free deterministic fallback used as
the default and in tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import ndimage as ndi


class Segmenter(ABC):
    """Turn a single-channel image into an integer label mask (0 = background)."""

    @abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray: ...


class ThresholdSegmenter(Segmenter):
    """Otsu threshold → connected components → size filter.

    Deterministic and dependency-free. Adequate for well-separated nuclei and
    for testing; use :class:`CellposeSegmenter` for crowded real tissue.
    """

    def __init__(self, min_size_px: int = 4, sigma: float = 1.0):
        self.min_size_px = min_size_px
        self.sigma = sigma

    def segment(self, image: np.ndarray) -> np.ndarray:
        from skimage.filters import gaussian, threshold_otsu

        a = np.asarray(image, dtype=np.float32)
        if a.max() <= a.min():
            return np.zeros(a.shape, dtype=np.int32)
        sm = gaussian(a, sigma=self.sigma, preserve_range=True)
        fg = sm > threshold_otsu(sm)
        labels, _ = ndi.label(fg)
        if self.min_size_px > 1:
            counts = np.bincount(labels.ravel())
            too_small = np.where(counts < self.min_size_px)[0]
            too_small = too_small[too_small != 0]
            if too_small.size:
                labels[np.isin(labels, too_small)] = 0
                # renumber to keep labels contiguous (1..N)
                labels, _ = ndi.label(labels > 0)
        return labels.astype(np.int32)


class CellposeSegmenter(Segmenter):
    """Cellpose backend. Requires the ``[cellpose]`` extra (pulls in torch)."""

    def __init__(self, model: str = "cpsam", diameter_px: float | None = None, gpu: bool = False):
        self.model = model
        self.diameter_px = diameter_px
        self.gpu = gpu

    def segment(self, image: np.ndarray) -> np.ndarray:
        try:
            from cellpose.models import CellposeModel
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "CellposeSegmenter needs the optional dependency: "
                "pip install 'spatialcyte[cellpose]'"
            ) from e
        m = CellposeModel(gpu=self.gpu, pretrained_model=self.model)
        masks, _, _ = m.eval(np.asarray(image), diameter=self.diameter_px)
        return np.asarray(masks, dtype=np.int32)


def segment_regions(image: np.ndarray, *, min_size_px: int = 50, closing_px: int = 2) -> np.ndarray:
    """Region (e.g. vessel) label mask from a structural marker via
    Otsu threshold → morphological closing → size filter → label."""
    from skimage.filters import threshold_otsu
    from skimage.measure import label
    from skimage.morphology import disk

    a = np.asarray(image, dtype=np.float32)
    if a.max() <= a.min():
        return np.zeros(a.shape, dtype=np.int32)
    fg = a > threshold_otsu(a)
    if closing_px > 0:
        fg = ndi.binary_closing(fg, structure=disk(closing_px))
    labels = label(fg)
    if min_size_px > 1:
        counts = np.bincount(labels.ravel())
        small = np.where(counts < min_size_px)[0]
        small = small[small != 0]
        if small.size:
            labels[np.isin(labels, small)] = 0
            labels = label(labels > 0)
    return labels.astype(np.int32)
