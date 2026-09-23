"""Stage 2 — quantify.

Build the spatial single-cell object: an AnnData where, for every marker, each
cell carries whole-cell intensity (``X``), intracellular sub-compartments
(membrane / cytoplasm / nuclear) and the extracellular pericellular ring (as
``layers``), morphology and region distances (``obs``), spatial coordinates
(``obsm['spatial']``, in µm) and a sparse spatial neighbour graph (``obsp`` — the
"distance matrix"). Generalises the reference RCC 8-plex feature-extraction stage to
all markers.

The reader's channels and ``cell_mask`` must be at the same resolution
(``config.pixel_size_um`` µm/px); stage 1 / the caller handles any rescaling.

This operates on a single in-memory field of view: every channel it measures and
the full-image distance transforms are held in RAM. For whole-slide images larger
than memory, tile upstream and run ``quantify`` per tile (as the reference
whole-slide pipeline does), then concatenate the resulting AnnData objects.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table

from spatialcyte.config import PanelConfig
from spatialcyte.io import ImageReader

_COMPARTMENTS = ("membrane", "cytoplasm", "extracellular", "nuclear")


def quantify(
    reader: ImageReader,
    cell_mask: np.ndarray,
    config: PanelConfig,
    *,
    region_masks: dict[str, np.ndarray] | None = None,
    nuclear_mask: np.ndarray | None = None,
    compartments: list[str] | None = None,
    erosion_px: int = 2,
    ring_px: int = 3,
    knn: int = 15,
) -> ad.AnnData:
    cell_mask = np.asarray(cell_mask)
    if cell_mask.ndim != 2:
        raise ValueError(f"cell_mask must be 2D, got shape {cell_mask.shape}")
    H, W = cell_mask.shape
    region_masks = region_masks or {}
    px_um = config.pixel_size_um

    q = config.quantify
    comps = compartments if compartments is not None else q.get("compartments", list(_COMPARTMENTS))
    unknown = [c for c in comps if c != "whole" and c not in _COMPARTMENTS]
    if unknown:
        raise ValueError(
            f"unknown compartment(s) {unknown}; choose from {('whole', *_COMPARTMENTS)}"
        )
    comps = [c for c in comps if c in _COMPARTMENTS]  # 'whole' lives in X
    cofactor = float(q.get("arcsinh_cofactor", 5))
    knn = int(q.get("spatial_graph", {}).get("k", knn))

    # --- cells + morphology (label-ordered) ---
    props = pd.DataFrame(
        regionprops_table(
            cell_mask,
            properties=("label", "centroid", "area", "eccentricity", "solidity", "perimeter"),
        )
    )
    if props.empty:
        raise ValueError("cell_mask has no cells")
    labels = props["label"].to_numpy()
    n = labels.size
    cy = props["centroid-0"].to_numpy()
    cx = props["centroid-1"].to_numpy()

    markers = config.markers
    chan_idx = [c.index for c in config.measured_channels]

    # cache each channel once (cropped to mask when larger; a smaller channel is
    # a resolution mismatch and cannot be measured against this mask)
    channels: dict[int, np.ndarray] = {}
    for ci in chan_idx:
        a = np.asarray(reader.read_channel(ci))
        if a.shape == (H, W):
            channels[ci] = a
        elif a.shape[0] >= H and a.shape[1] >= W:
            channels[ci] = a[:H, :W]
        else:
            raise ValueError(
                f"channel {ci} shape {a.shape} is smaller than the cell mask {(H, W)}; "
                "reader and mask must be at the same resolution"
            )

    def _means(label_img: np.ndarray) -> np.ndarray:
        out = np.empty((n, len(markers)), np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):  # empty label -> 0/0, handled below
            for j, ci in enumerate(chan_idx):
                out[:, j] = ndi.mean(channels[ci], labels=label_img, index=labels)
        # a label with no pixels in this compartment yields 0/0 -> NaN; the
        # sensible value is 0 (no signal measured in that compartment). Common
        # in dense tissue: enclosed cells have no membrane/extracellular ring,
        # thin cells no cytoplasm after erosion.
        return np.nan_to_num(out, nan=0.0)

    # --- whole-cell intensity -> X ---
    X = _means(cell_mask)

    # --- compartment label images ---
    label_imgs: dict[str, np.ndarray] = {}
    if "membrane" in comps or "cytoplasm" in comps:
        interior = _interior_mask(cell_mask, erosion_px)
        if "cytoplasm" in comps:
            label_imgs["cytoplasm"] = np.where(interior, cell_mask, 0)
        if "membrane" in comps:
            label_imgs["membrane"] = np.where((cell_mask > 0) & ~interior, cell_mask, 0)
    if "extracellular" in comps:
        label_imgs["extracellular"] = _pericellular_ring(cell_mask, ring_px)
    if "nuclear" in comps:
        # with a nuclear sub-mask, measure within it; with nuclear-only
        # segmentation the cell mask *is* the nucleus, so nuclear == whole.
        label_imgs["nuclear"] = (
            np.where(np.asarray(nuclear_mask) > 0, cell_mask, 0)
            if nuclear_mask is not None
            else cell_mask
        )
    layers = {"arcsinh": np.arcsinh(X / cofactor).astype(np.float32)}
    for name, img in label_imgs.items():
        layers[name] = _means(img)

    # --- obs: morphology + region distances ---
    cyi = np.clip(cy, 0, H - 1).astype(int)
    cxi = np.clip(cx, 0, W - 1).astype(int)
    obs = pd.DataFrame(
        {
            "centroid_x": cx.astype(np.float32),
            "centroid_y": cy.astype(np.float32),
            "area_um2": (props["area"].to_numpy() * px_um**2).astype(np.float32),
            "eccentricity": props["eccentricity"].to_numpy().astype(np.float32),
            "solidity": props["solidity"].to_numpy().astype(np.float32),
            "perimeter_um": (props["perimeter"].to_numpy() * px_um).astype(np.float32),
        },
        index=[f"cell_{int(lbl)}" for lbl in labels],
    )
    for rname, rmask in region_masks.items():
        rm = np.asarray(rmask) > 0
        if not rm.any():
            # region absent in this sample: distance is undefined, not a
            # fabricated finite value from an empty distance transform.
            obs[f"dist_to_{rname}_um"] = np.float32(np.inf)
            continue
        dist_px = ndi.distance_transform_edt(~rm).astype(np.float32)
        obs[f"dist_to_{rname}_um"] = (dist_px[cyi, cxi] * px_um).astype(np.float32)

    # --- spatial: coords (µm) + neighbour graph ---
    xy_um = np.column_stack([cx, cy]).astype(np.float64) * px_um
    distances, connectivities = _knn_graph(xy_um, knn)

    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=markers),
        layers=layers,
        obsm={"spatial": xy_um.astype(np.float32)},
    )
    adata.obsp["spatial_distances"] = distances
    adata.obsp["spatial_connectivities"] = connectivities
    adata.uns["spatialcyte"] = {
        "pixel_size_um": px_um,
        "compartments": list(label_imgs),
        "erosion_px": erosion_px,
        "ring_px": ring_px,
        "knn": knn,
        "regions": list(region_masks),
    }
    return adata


def _interior_mask(cell_mask: np.ndarray, erosion_px: int) -> np.ndarray:
    """Per-cell interior (cytoplasm) mask: pixels whose whole ``erosion_px``
    neighbourhood is a single, same-labelled cell.

    Unlike eroding the binary union of all cells, this keeps touching cells
    apart — a shared cell–cell border spans two labels, so it is excluded from
    both interiors and correctly falls to the membrane compartment. A fully
    enclosed cell therefore still has a membrane (its outer border with
    neighbours), and border pixels never leak between adjacent cells.
    """
    if erosion_px <= 0:
        return cell_mask > 0
    from skimage.morphology import disk

    fp = disk(erosion_px)
    # constant/0 outside the image -> pixels near the image edge count as
    # boundary (membrane), matching a background border.
    mn = ndi.minimum_filter(cell_mask, footprint=fp, mode="constant", cval=0)
    mx = ndi.maximum_filter(cell_mask, footprint=fp, mode="constant", cval=0)
    return (cell_mask > 0) & (mn == cell_mask) & (mx == cell_mask)


def _pericellular_ring(cell_mask: np.ndarray, ring_px: int) -> np.ndarray:
    """Label each background pixel within ``ring_px`` of a cell by that nearest
    cell, giving a per-cell extracellular ring (nearest-cell partition)."""
    bg = cell_mask == 0
    dist, (iy, ix) = ndi.distance_transform_edt(bg, return_indices=True)
    nearest = cell_mask[iy, ix]
    return np.where(bg & (dist > 0) & (dist <= ring_px), nearest, 0)


def _knn_graph(xy: np.ndarray, k: int) -> tuple[csr_matrix, csr_matrix]:
    """Symmetric-by-construction-free kNN graph: sparse neighbour distances +
    binary connectivities (self excluded)."""
    n = len(xy)
    k = min(k, n - 1)
    if k <= 0:
        empty = csr_matrix((n, n), dtype=np.float32)
        return empty, empty.copy()
    dist, idx = cKDTree(xy).query(xy, k=k + 1)  # +1 for self at column 0
    dist, idx = dist[:, 1:], idx[:, 1:]
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    distances = csr_matrix((dist.ravel().astype(np.float32), (rows, cols)), shape=(n, n))
    connectivities = csr_matrix((np.ones(rows.size, np.float32), (rows, cols)), shape=(n, n))
    return distances, connectivities
