"""Pipeline orchestration + CLI.

Two entry points:

* ``spatialcyte qc`` — Stage 0 only: QC an image against a panel config.
* ``spatialcyte run`` — the full image → spatial single-cell object pipeline
  (segment → quantify → phenotype), writing an ``.h5ad``.

``build`` is the importable orchestration function behind ``run``.
"""

from __future__ import annotations

import argparse
import sys

from spatialcyte.config import PanelConfig


def _cmd_qc(args: argparse.Namespace) -> int:
    from spatialcyte.io import TiffReader
    from spatialcyte.qc import run_qc

    cfg = PanelConfig.from_yaml(args.config)
    report = run_qc(TiffReader(args.image, level=args.level), cfg)
    _print_qc(report)
    return 0 if report.ok else 1


def _print_qc(report) -> None:
    for c in report.channels:
        note = ("  " + "; ".join(c.warnings)) if c.warnings else ""
        print(
            f"ch{c.index:>2}  {c.marker:<8} {c.role:<10} "
            f"mean={c.mean:>9.2f}  p99={c.p99:>9.2f}  "
            f"sat={c.saturated_frac:>5.1%}  blank={c.blank_frac:>5.1%}{note}"
        )
    for w in report.warnings:
        print(f"WARN: {w}")


def _make_cell_segmenter(config: PanelConfig, *, gpu: bool, backend: str | None):
    """Pick a cell segmenter from the config (or an explicit override)."""
    from spatialcyte.segment import CellposeSegmenter, ThresholdSegmenter

    cell = config.segmentation.get("cell", {})
    backend = backend or cell.get("backend", "threshold")
    if backend == "cellpose":
        diameter_px = None
        if cell.get("diameter_um"):
            diameter_px = float(cell["diameter_um"]) / config.pixel_size_um
        return CellposeSegmenter(model=cell.get("model", "cpsam"), diameter_px=diameter_px, gpu=gpu)
    if backend == "threshold":
        return ThresholdSegmenter()
    raise ValueError(
        f"unknown cell segmentation backend {backend!r} (use 'cellpose' or 'threshold')"
    )


def build(reader, config: PanelConfig, *, gpu: bool = False, backend: str | None = None):
    """Run the full pipeline on one opened image, returning the AnnData.

    segment (cells from the nuclear channel, regions from structural channels)
    → quantify → phenotype (when the config declares gates/markers).
    """
    from spatialcyte.quantify import quantify
    from spatialcyte.segment import segment_regions

    nuc_idx = config.nuclear_channels[0].index
    segmenter = _make_cell_segmenter(config, gpu=gpu, backend=backend)
    cell_mask = segmenter.segment(reader.read_channel(nuc_idx))

    region_masks = {
        ch.region: segment_regions(reader.read_channel(ch.index)) for ch in config.region_channels
    }

    adata = quantify(reader, cell_mask, config, region_masks=region_masks or None)

    if config.phenotype.get("gates") or config.phenotype.get("markers"):
        from spatialcyte.phenotype import phenotype

        phenotype(adata, config)
    return adata


def _cmd_run(args: argparse.Namespace) -> int:
    from spatialcyte.io import TiffReader
    from spatialcyte.qc import run_qc

    cfg = PanelConfig.from_yaml(args.config)
    reader = TiffReader(args.image, level=args.level)

    report = run_qc(reader, cfg)
    _print_qc(report)
    if not report.ok and not args.force:
        print("QC found problems; re-run with --force to proceed anyway.", file=sys.stderr)
        return 1

    adata = build(reader, cfg, gpu=args.gpu, backend=args.backend)
    adata.write_h5ad(args.out)

    print(f"\nwrote {args.out}: {adata.n_obs} cells x {adata.n_vars} markers")
    if "phenotype" in adata.obs:
        counts = adata.obs["phenotype"].value_counts()
        print("phenotypes: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="spatialcyte",
        description="Multiplexed IHC images -> spatial single-cell objects.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("qc", help="Stage 0 — QC an image against a panel config")
    q.add_argument("--config", required=True, help="panel YAML")
    q.add_argument("--image", required=True, help="multiplexed image (TIFF)")
    q.add_argument("--level", type=int, default=0, help="pyramid level to read (default 0)")
    q.set_defaults(func=_cmd_qc)

    r = sub.add_parser("run", help="Full pipeline: image -> spatial single-cell AnnData (.h5ad)")
    r.add_argument("--config", required=True, help="panel YAML")
    r.add_argument("--image", required=True, help="multiplexed image (TIFF)")
    r.add_argument("--out", required=True, help="output .h5ad path")
    r.add_argument("--level", type=int, default=0, help="pyramid level to read (default 0)")
    r.add_argument(
        "--backend",
        choices=["cellpose", "threshold"],
        default=None,
        help="cell segmentation backend (default: from config, else 'threshold')",
    )
    r.add_argument("--gpu", action="store_true", help="use GPU for Cellpose")
    r.add_argument("--force", action="store_true", help="proceed even if QC flags problems")
    r.set_defaults(func=_cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
