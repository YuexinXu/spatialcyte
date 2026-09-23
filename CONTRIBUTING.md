# Contributing to spatialcyte

## Dev setup
```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

## Extension points (how to add features)
- **New image format** → implement `spatialcyte.io.ImageReader`
  (`n_channels`, `shape`, `read_channel`). Everything downstream is format-agnostic.
- **New segmenter** → implement the `Segmenter` interface (`spatialcyte.segment`).
  Cellpose is the production backend, behind the `[cellpose]` extra; the
  dependency-free `ThresholdSegmenter` is the default.
- **New stage** → add a module under `src/spatialcyte/` and wire it into
  `pipeline.py`. Keep stages independent and AnnData-in / AnnData-out where it fits.

## Conventions
- One feature branch per change; CI (ruff + pytest) must pass before merge.
- Add a test with every behaviour change.
- **No data ever in the repo** — use synthetic fixtures (`tests/conftest.py`).
  Never commit images, `.h5ad`, or anything with patient information.
