# spatialcyte

**Multiplexed IHC images → spatial single-cell objects (AnnData), config-driven and
panel-agnostic.**

`spatialcyte` turns a multiplexed immunohistochemistry (mIHC) image plus a one-file
**panel config** into an [AnnData](https://anndata.readthedocs.io) object in which every
segmented cell is a row and, for every marker, carries **whole-cell**, **intracellular**
(nuclear / membrane / cytoplasm) and **extracellular** (pericellular ring) signal,
morphology, distances to segmented tissue regions, and a spatial neighbour graph. That is
the object the single-cell ecosystem (scanpy, squidpy, scimap, scVI) already knows how to
analyse. The biology of each channel is declared once in YAML, never in code, so the same
package works on any tissue and any marker panel.

**v0.1.0** · Python 3.10–3.12 · MIT · CI runs ruff + 39 unit tests on synthetic images
with analytically known answers · no data in the repository.

- [Installation](#installation)
- [Input requirements](#input-requirements)
- [The panel config](#the-panel-config)
- [Usage](#usage) — CLI · Python API
- [What you get: the output object](#what-you-get-the-output-object)
- [How the compartments are defined](#how-the-compartments-are-defined)
- [Whole-slide images and HPC](#whole-slide-images-and-hpc)
- [Validation](#validation)
- [Scope and limitations](#scope-and-limitations)
- [Related tools](#related-tools)
- [Development and tests](#development-and-tests)
- [Citation](#citation) · [License](#license)

---

## Installation

Requires Python ≥ 3.10. Core dependencies are numpy, pandas, scipy, scikit-image,
scikit-learn, anndata, tifffile and pyyaml; all are permissively licensed (BSD/MIT).

**From GitHub (recommended, pinned to a release):**
```bash
pip install "spatialcyte @ git+https://github.com/YuexinXu/spatialcyte.git@v0.1.0"
```

**With Cellpose cell segmentation** (pulls in PyTorch; needed for real tissue):
```bash
pip install "spatialcyte[cellpose] @ git+https://github.com/YuexinXu/spatialcyte.git@v0.1.0"
```
For GPU segmentation install a CUDA build of PyTorch first, following the
[PyTorch instructions](https://pytorch.org/get-started/locally/) for your CUDA version,
then install spatialcyte with the `[cellpose]` extra and pass `--gpu`.

**From a clone (development):**
```bash
git clone https://github.com/YuexinXu/spatialcyte.git
cd spatialcyte
pip install -e ".[dev]"            # core + pytest + ruff
pip install -e ".[dev,cellpose]"   # ... plus Cellpose
```

**On an HPC cluster:** load a Python ≥ 3.10 module (and an HDF5 module if your site
builds h5py against it), create a virtual environment or install with `--user`, then run
the commands above. Cellpose needs a GPU node only at segmentation time; every other stage
is CPU-only.

**Check the install:**
```bash
spatialcyte --help
python -c "import spatialcyte; print(spatialcyte.__version__)"
```

## Input requirements

| Requirement | Detail |
|---|---|
| **Image** | A multi-channel TIFF. Channel-first (`CYX`) pages, `YXC`, or a 2-D single channel are recognised. If the file carries an OME/ImageJ channel axis label it is used; otherwise the smallest axis is taken as the channel axis. Pyramidal TIFFs: pick a resolution level with `--level N` (0 = full resolution) and set `pixel_size_um` to that level's pixel size, since it is not rescaled automatically. Channels stored as separate pages are read lazily, one at a time. |
| **Pixel size** | Declared in the panel config (`pixel_size_um`); it is not read from the file. All distances, areas and graph edge lengths are reported in micrometres from this value. |
| **Nuclear channel** | At least one channel with `role: nuclear` (for example DAPI); cells are segmented from it. |
| **Region channels** | Optional. Any channel with `role: region` (for example CD31 → `vessel`) is segmented into a region mask, and every cell receives `dist_to_<region>_um`. |
| **Other formats** | Add a reader by implementing `spatialcyte.io.ImageReader` (three members: `n_channels`, `shape`, `read_channel`). Everything downstream is format-agnostic. |

## The panel config

One YAML file describes the panel and the analysis settings. The full worked example is
[`examples/panel_rcc_8plex.yaml`](examples/panel_rcc_8plex.yaml):

```yaml
pixel_size_um: 0.5
channels:
  - { index: 0, marker: DAPI,  role: nuclear }                  # segmentation stain
  - { index: 1, marker: CD31,  role: region, region: vessel }   # structural -> region mask
  - { index: 2, marker: CD8,   role: lineage }                  # cell-type marker
  - { index: 5, marker: STAB1, role: functional }               # marker of interest
  - { index: 8, marker: AF,    role: exclude }                  # autofluorescence, ignored

segmentation:
  cell: { backend: cellpose, model: cpsam, diameter_um: 10 }

quantify:
  compartments: [whole, nuclear, membrane, cytoplasm, extracellular]
  arcsinh_cofactor: 5
  spatial_graph: { k: 15 }

phenotype:
  positivity: { method: gmm }          # or otsu
  gates:                               # hierarchical, first match wins
    - { label: Treg,        all: [CD3, CD4, FOXP3] }
    - { label: CD8 T cell,  all: [CD3, CD8] }
    - { label: Endothelial, all: [CD31] }
    - { label: Other }                 # no condition = default
```

**Channel roles**

| `role` | Meaning | Quantified? |
|---|---|---|
| `nuclear` | Nuclear stain used for cell segmentation | yes |
| `region` | Structural marker segmented into a named region (`region:` is required) | yes |
| `lineage` | Cell-type marker used by the gates | yes |
| `functional` | Marker of interest measured per cell | yes |
| `exclude` | Not analysed (autofluorescence, empty channels) | no |

**Settings**

| Key | Effect | Default |
|---|---|---|
| `segmentation.cell.backend` | `cellpose` or `threshold` (Otsu + connected components, dependency-free) | `threshold` |
| `segmentation.cell.model` | Cellpose pretrained model name or path | `cpsam` |
| `segmentation.cell.diameter_um` | Expected cell diameter, converted to pixels for Cellpose | Cellpose auto |
| `quantify.compartments` | Any of `whole`, `nuclear`, `membrane`, `cytoplasm`, `extracellular` | all |
| `quantify.arcsinh_cofactor` | Cofactor for the `arcsinh` layer, `arcsinh(X / cofactor)` | 5 |
| `quantify.spatial_graph.k` | Neighbours in the kNN spatial graph | 15 |
| `phenotype.positivity.method` | `gmm` (2-component Gaussian mixture on the `arcsinh` layer) or `otsu` | `gmm` |
| `phenotype.markers` | Restrict positivity calls to these markers | all |
| `phenotype.gates` | Ordered list of `{label, all?, any?, not?}` over marker positivity | none |

Positivity calling and gating run only when `phenotype.gates` or `phenotype.markers` is
present; a config without either yields no `<marker>_positive` columns. Region channels
are segmented automatically (Otsu threshold → morphological closing → size filter). Membrane width and extracellular-ring width are pixel parameters of the
Python API (`erosion_px`, `ring_px`, see below), not YAML keys.

Configuration errors fail fast: duplicate channel indices, a missing nuclear channel, a
region channel without a region name, an unknown role, a non-positive pixel size, an
unknown compartment name and an unknown positivity method all raise `ValueError`.

## Usage

### Command line

**Stage 0, QC an image against a panel** (blank or saturated channels, panel ↔ image
mismatch). Exit status 1 when anything is flagged:
```bash
spatialcyte qc --config panel.yaml --image image.tif
```

**Full pipeline, image → `.h5ad`** (QC → segment cells and regions → quantify →
phenotype):
```bash
spatialcyte run --config panel.yaml --image image.tif --out cells.h5ad
```

| Option | Meaning |
|---|---|
| `--backend {cellpose,threshold}` | Override the cell-segmentation backend from the config |
| `--gpu` | Run Cellpose on the GPU |
| `--level N` | Read pyramid level N of a pyramidal TIFF (default 0); set `pixel_size_um` in the config to match that level |
| `--force` | Proceed even if QC flags problems |

The run prints the QC table, then the cell × marker count and, when gates are configured,
the phenotype counts.

### Python API

Each stage is a plain function; the object is AnnData throughout.

```python
from spatialcyte import PanelConfig
from spatialcyte.io import TiffReader
from spatialcyte.qc import run_qc
from spatialcyte.pipeline import build
from spatialcyte.analyze import positivity_summary, composition, neighborhood_enrichment

cfg = PanelConfig.from_yaml("panel.yaml")
reader = TiffReader("image.tif")  # TiffReader("image.tif", level=1) for a pyramid level

report = run_qc(reader, cfg)  # stage 0; report.ok, report.channels, report.warnings
adata = build(reader, cfg)  # stages 1-3: image -> spatial single-cell AnnData
adata.write_h5ad("cells.h5ad")

composition(adata)  # phenotype frequencies
positivity_summary(adata)  # fraction positive per marker
neighborhood_enrichment(adata, n_perms=1000)  # Z-scores of phenotype-pair adjacency
```

Stages can also be called individually, for example to bring your own masks:

```python
from spatialcyte.segment import ThresholdSegmenter, CellposeSegmenter, segment_regions
from spatialcyte.quantify import quantify
from spatialcyte.phenotype import mark_positive, gate

cell_mask = CellposeSegmenter(model="cpsam", gpu=True).segment(reader.read_channel(0))
vessel = segment_regions(reader.read_channel(1))
adata = quantify(
    reader, cell_mask, cfg, region_masks={"vessel": vessel}, erosion_px=2, ring_px=3
)  # membrane width / ring width in pixels
mark_positive(adata, method="gmm")  # adds obs['<marker>_positive']
gate(adata, cfg.phenotype["gates"])  # adds obs['phenotype']
```

`quantify` accepts any label image as `cell_mask` (0 = background, integer labels =
cells), so masks from HALO, QuPath, Mesmer or any other tool can be used, provided they
are at the same resolution as the image channels. Pass `nuclear_mask=` to measure the
`nuclear` compartment inside a nuclear sub-mask; without it, `nuclear` equals the whole
cell.

## What you get: the output object

| Slot | Content | Units |
|---|---|---|
| `X` | Whole-cell mean intensity, cells × markers | image intensity |
| `var` | One row per quantified marker, in channel order | |
| `layers["arcsinh"]` | `arcsinh(X / cofactor)` | |
| `layers["membrane"]`, `["cytoplasm"]`, `["nuclear"]` | Mean intensity per intracellular compartment | image intensity |
| `layers["extracellular"]` | Mean intensity in the pericellular ring assigned to each cell | image intensity |
| `obs["centroid_x"]`, `["centroid_y"]` | Cell centroid | pixels |
| `obs["area_um2"]`, `["perimeter_um"]`, `["eccentricity"]`, `["solidity"]` | Morphology | µm², µm, unitless |
| `obs["dist_to_<region>_um"]` | Euclidean distance from the centroid to the nearest pixel of each region (`inf` when the region is absent from the sample) | µm |
| `obs["<marker>_positive"]` | Boolean positivity call per marker (after `phenotype`) | |
| `obs["phenotype"]` | Gated cell type (after `phenotype`) | |
| `obsm["spatial"]` | Centroid coordinates `(x, y)` | µm |
| `obsp["spatial_distances"]`, `["spatial_connectivities"]` | Sparse kNN graph (distances in µm; binary connectivities); self excluded | |
| `uns["spatialcyte"]` | Provenance: `pixel_size_um`, `compartments` (the layers computed, so without `whole`), `erosion_px`, `ring_px`, `knn`, `regions` | |

Cell names are `cell_<label>` from the segmentation mask, so every row can be traced back
to the mask.

## How the compartments are defined

All compartments are geometric partitions of the cell mask and its surroundings, computed
per cell so that touching cells never share pixels:

- **whole** — every pixel of the cell label.
- **cytoplasm** — pixels whose entire `erosion_px` neighbourhood (default 2 px) lies inside
  the same cell.
- **membrane** — the remaining cell pixels: the outer rim, including shared borders with
  neighbouring cells.
- **nuclear** — the cell pixels inside `nuclear_mask` when one is supplied; otherwise the
  whole cell (nuclear-only segmentation).
- **extracellular** — background pixels within `ring_px` (default 3 px) of a cell, each
  assigned to its nearest cell (a Voronoi-style partition of the near-cell background). This
  is where secreted, matrix-bound and neighbour-contact signal is measured.

A compartment that has no pixels for a given cell (an enclosed cell has no extracellular
ring; a thin cell may have no cytoplasm after erosion) reports 0, not NaN. Widths are in
pixels: at 0.5 µm/px the defaults give a 1 µm membrane rim and a 1.5 µm ring; adjust
`erosion_px` and `ring_px` to your resolution.

## Whole-slide images and HPC

`quantify` works on one in-memory field of view: every measured channel and the region
distance transforms are held in RAM. For whole-slide images larger than memory, tile the
slide upstream, run `quantify` per tile and concatenate. A tile is presented to
`quantify` through any `ImageReader`; an in-memory array reader is enough:

```python
import numpy as np, anndata as ad
from spatialcyte.io import ImageReader
from spatialcyte.quantify import quantify


class ArrayReader(ImageReader):
    def __init__(self, arr):  # arr: (channels, height, width)
        self.arr = arr

    @property
    def n_channels(self):
        return self.arr.shape[0]

    @property
    def shape(self):
        return self.arr.shape[1:]

    def read_channel(self, i):
        return self.arr[i]


tiles = []
for (y0, x0), img_tile, mask_tile, region_tiles in my_tiler(...):  # your tiling
    a = quantify(ArrayReader(img_tile), mask_tile, cfg, region_masks=region_tiles)
    a.obsm["spatial"] += np.array([x0, y0], np.float32) * cfg.pixel_size_um  # to slide µm
    a.obs["centroid_x"] += x0
    a.obs["centroid_y"] += y0  # to slide px
    tiles.append(a)
adata = ad.concat(tiles, index_unique="-")
```

Three caveats are the caller's responsibility: cells cut by a tile border should be handled
with overlapping tiles and de-duplication; region distances are computed within the tile,
so tiles need enough margin for the largest distance you intend to interpret; and
`ad.concat` keeps `X`, layers, `obs` and `obsm` but drops the per-tile graph (`obsp`) and
`uns`, so rebuild the neighbour graph on the merged `obsm["spatial"]` (for example with
`squidpy.gr.spatial_neighbors`). On a
SLURM cluster, one tile (or one slide) per array task followed by a concatenation step is
the natural layout; only the Cellpose step needs a GPU.

## Validation

On one whole-slide 8-plex renal-cell-carcinoma mIHC sample (33,345 cells × 8 markers),
`quantify` reproduced the whole-cell mean intensities of an independently written
reference implementation to within 1e-3, the tolerance asserted by the shipped regression
test, while additionally producing the compartment layers, region distances and spatial
graph in one pass. The same design underlies an analysis of 2.29 million cells across
12 whole-slide samples. Because the reference run used patient data that cannot be
committed, the regression test (`tests/test_regression_reference.py`) is skipped unless
`SPATIALCYTE_REGRESSION_DIR` points to a prepared fixture directory containing
`image.tif`, `cell_mask.tif`, `reference.h5ad` and `panel.yaml`.

Every stage is additionally unit-tested on synthetic scenes with analytically known
answers: whole-cell and compartment means, extracellular-ring localisation,
region-distance ordering, absent-region handling, dense-tissue membrane assignment,
positivity splitting of bimodal data, first-match-wins gating, and neighbourhood
enrichment detecting a segregated layout.

## Scope and limitations

- **Single field of view in memory.** Tile very large images upstream (see above).
- **Segmentation quality bounds everything.** Segmentation is delegated to Cellpose (or
  the simple threshold segmenter) and is the dominant source of biological error.
  Compartment assignment is a geometric approximation that is most faithful for
  round-to-ovoid cells at adequate resolution.
- **Positivity calling assumes a bimodal marker.** Inspect the fitted threshold for
  continuous or unimodal markers; `otsu` is available as an alternative.
- **Region segmentation is a fixed Otsu-plus-morphology rule.** Supply your own region
  masks through the Python API when a marker needs a custom procedure.
- **Spatial graph is kNN only.** Radius or Delaunay graphs can be built downstream from
  `obsm["spatial"]` (for example with squidpy).
- **The reusable core, not a study pipeline.** spatialcyte was generalised from a
  whole-slide renal-cell-carcinoma study; study-specific downstream analyses (vessel
  classification, perivascular zoning, statistics, figures) are outside its scope.

## Related tools

Upstream, end-to-end tools such as MCMICRO, steinbock, SPACEc, spatialproteomics and Sopa
also produce per-cell tables from multiplexed images, typically whole-cell mean
intensity plus morphology, and several emit AnnData or SpatialData. Digital-pathology
workstations (HALO, inForm, QuPath) offer interactive cell detection and per-study
configuration. spatialcyte differs in three ways: it measures five compartments per
marker, including a first-class extracellular ring in the spirit of REDSEA's boundary
geometry but reported rather than subtracted; it is panel-agnostic by declaration (a
validated YAML `role` per channel) rather than by technology; and it is small (about a
thousand lines of numpy/scipy/scikit-image) with a dependency-free default path.
Downstream packages such as scimap and squidpy begin where spatialcyte ends and consume
the AnnData it produces.

## Development and tests

```bash
git clone https://github.com/YuexinXu/spatialcyte.git && cd spatialcyte
pip install -e ".[dev]"
ruff check .
pytest                     # 39 tests on synthetic fixtures; under 10 s
SPATIALCYTE_REGRESSION_DIR=/path/to/fixture pytest tests/test_regression_reference.py
```

CI (GitHub Actions) runs ruff and the test suite on Python 3.10, 3.11 and 3.12. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the extension points (new image formats,
segmenters and stages). No images, `.h5ad` files or patient information are ever committed.

**Roadmap:** additional readers (QPTIFF, OME-Zarr) via the `ImageReader` interface;
built-in tiling for whole slides; further spatial statistics.

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23402931.svg)](https://doi.org/10.5281/zenodo.23402931)

If you use spatialcyte, please cite the software (metadata in
[`CITATION.cff`](CITATION.cff)):

> Xu Y. spatialcyte: multiplexed IHC images to spatial single-cell objects. Version 0.1.0.
> Zenodo; 2026. https://doi.org/10.5281/zenodo.23402932

The badge carries the concept DOI, which always resolves to the latest version; the
version DOI above identifies v0.1.0 specifically.

## License

MIT, see [`LICENSE`](LICENSE). Optional Cellpose segmentation depends on Cellpose (BSD-3)
and PyTorch (BSD-3), installed only with the `[cellpose]` extra.
