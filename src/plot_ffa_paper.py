#!/usr/bin/env python3
"""Render the paper's FFA figures from saved total-light FFA products.

The sample order comes from ``DRMH0/tables/sample_observations.csv``.  The
first source is the main-text example; the remainder use two source pairs per
row and up to three rows per appendix page. PDF composition retains vectors.
No FFA fitting, posterior sampling, or MICA processing is performed.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

import fitz

from results_diagnostics import (
    _configured_dir,
    _ffa_products,
    make_ffa_host_figure,
    read_yaml,
    source_path,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = ROOT / "DRMH0/tables/sample_observations.csv"
DEFAULT_OUTPUT = ROOT / "DRMH0/fig/paper/ffa"
SED_LEGEND_LOC = {"NGC4151_a": "upper right", "NGC6814": "upper right"}
SED_UPPER_MARGIN = {"NGC4151_a": 4.0, "NGC6814": 3.0}


def source_label(source: str) -> str:
    """Format the object name; selected intervals are defined in the sample table."""
    base, separator, segment = source.partition("_")
    if base.startswith("Mrk"):
        label = "Mrk " + base[3:]
    elif base.startswith("NGC"):
        label = "NGC " + base[3:]
    elif base == "F9":
        label = "Fairall 9"
    else:
        label = base
    return label


def ffa_inputs(source_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the exact saved products used by the existing figure renderer."""
    ffa, _, components, posterior, reconstructed_x = _ffa_products(source_dir, config)
    source = str(ffa.get("source") or source_dir.name)
    component = str(ffa["component"]["name"])
    root = source_path(source_dir, ffa.get("out", "fflux"))
    input_dir = _configured_dir(source_dir, ffa, "fit", "input_dir", root / "dered")
    template = str(ffa.get("sed", {}).get("input_template", "{source}_{band}_{component}.txt"))
    light_curves = {
        str(band): str(input_dir / template.format(source=source, band=band, component=component))
        for band in ffa["bands"]
    }
    paths = [components, posterior, reconstructed_x, *map(Path, light_curves.values())]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing saved FFA products for {source}: {missing}")
    return {
        "config": str(source_dir / "config/source_config.yaml"),
        "components": str(components),
        "joint_flux_posterior": str(posterior),
        "reconstructed_driver": str(reconstructed_x),
        "light_curves": light_curves,
        "bands": list(ffa["bands"]),
    }


def stack_ffa_page(pdf_paths: list[Path], output_stem: Path) -> dict[str, str]:
    """Stack one or two source panels as vector-preserving PDF form objects."""
    if not 1 <= len(pdf_paths) <= 2:
        raise ValueError("An FFA appendix page must contain one or two sources")
    width = 7.2 * 72.0
    gap = 0.16 * 72.0
    with fitz.open() as output:
        inputs = [fitz.open(path) for path in pdf_paths]
        try:
            heights = [width * document[0].rect.height / document[0].rect.width for document in inputs]
            page = output.new_page(width=width, height=sum(heights) + gap * (len(inputs) - 1))
            y = 0.0
            for document, height in zip(inputs, heights):
                if len(document) != 1:
                    raise ValueError("Expected a single-page source FFA figure")
                page.show_pdf_page(fitz.Rect(0, y, width, y + height), document, 0)
                y += height + gap
            pdf = output_stem.with_suffix(".pdf")
            png = output_stem.with_suffix(".png")
            output.save(pdf, garbage=4, deflate=True)
            page.get_pixmap(dpi=300, alpha=False).save(png)
        finally:
            for document in inputs:
                document.close()
    return {"pdf": str(pdf), "png": str(png)}


def build_ffa_paper(
    sample_table: Path = DEFAULT_SAMPLE,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    """Render or reuse full-size sources, then redraw the compact appendix grid."""
    sample_table = Path(sample_table).resolve()
    workspace = sample_table.parents[2]
    output_dir = Path(output_dir).resolve()
    with sample_table.open(newline="", encoding="utf-8") as handle:
        sources = [row["source"] for row in csv.DictReader(handle)]
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("Expected a nonempty sample of unique sources")
    if sources[0] != "Mrk817_a":
        raise ValueError("The approved main-text FFA example must be Mrk817_a")
    output_dir.mkdir(parents=True, exist_ok=True)
    previous = {}
    if reuse_existing:
        previous = {
            entry["source"]: entry
            for entry in json.loads((output_dir / "manifest.json").read_text())["single_sources"]
        }
    manifest: dict[str, Any] = {
        "sample_table": str(sample_table),
        "source_order": sources,
        "method": "Render existing total-light FFA products; no fitting or resampling",
        "reused_single_source_figures": reuse_existing,
        "right_legend_fontsize": 6.0,
        "single_sources": [],
        "appendix_pages": [],
    }
    for source in sources:
        if reuse_existing:
            entry = previous[source]
            for path in entry["outputs"].values():
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
            manifest["single_sources"].append(entry)
            print(f"Reuse FFA {source}: {entry['outputs']['pdf']}", flush=True)
            continue
        source_dir = workspace / source
        config = read_yaml(source_dir / "config/source_config.yaml")
        inputs = ffa_inputs(source_dir, config)
        outputs = make_ffa_host_figure(
            source_dir,
            config,
            output_dir=output_dir,
            source_label=source_label(source),
            sed_legend_loc=SED_LEGEND_LOC.get(source, "best"),
            sed_upper_margin=SED_UPPER_MARGIN.get(source, 1.0),
        )
        entry = {
            "source": source,
            "label": source_label(source),
            "sed_legend_loc": SED_LEGEND_LOC.get(source, "best"),
            "sed_upper_margin": SED_UPPER_MARGIN.get(source, 1.0),
            "inputs": inputs,
            "outputs": {kind: str(path) for kind, path in outputs.items()},
        }
        manifest["single_sources"].append(entry)
        print(f"FFA {source}: {outputs['pdf']}", flush=True)
    manifest["main_text"] = manifest["single_sources"][0]["outputs"]
    compact = reflow_ffa_appendix(sample_table, output_dir)
    manifest["appendix_pages"] = compact["appendix_pages"]
    manifest["compact_sources"] = compact["compact_sources"]
    manifest["appendix_layout"] = compact["layout"]
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def reflow_ffa_appendix(
    sample_table: Path = DEFAULT_SAMPLE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Redraw saved appendix products at final size and place two pairs per row.

    The main-text example and the existing full-size source PDFs are left
    untouched.  This callable returns metadata without updating a manifest.
    """
    sample_table, output_dir = Path(sample_table).resolve(), Path(output_dir).resolve()
    workspace = sample_table.parents[2]
    with sample_table.open(newline="", encoding="utf-8") as handle:
        sources = [row["source"] for row in csv.DictReader(handle)]
    if not sources or sources[0] != "Mrk817_a" or len(sources) != len(set(sources)):
        raise ValueError("Expected unique sources with Mrk817_a as the main-text example")
    compact_dir = output_dir / "compact"
    entries = []
    sed_legend_items = {}
    for source in sources[1:]:
        source_dir = workspace / source
        config = read_yaml(source_dir / "config/source_config.yaml")
        inputs = ffa_inputs(source_dir, config)
        outputs = make_ffa_host_figure(
            source_dir, config, output_dir=compact_dir,
            source_label=source_label(source),
            sed_legend_loc=SED_LEGEND_LOC.get(source, "best"),
            sed_upper_margin=SED_UPPER_MARGIN.get(source, 1.0),
            compact_appendix=True,
            sed_legend_items=sed_legend_items,
        )
        entries.append(dict(source=source, label=source_label(source), inputs=inputs,
                            outputs={kind: str(path) for kind, path in outputs.items()}))
        print(f"Compact FFA {source}: {outputs['pdf']}", flush=True)

    # One shared key preserves every original FFA entry without covering data.
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    bands = entries[0]["inputs"]["bands"]
    if any(entry["inputs"]["bands"] != bands for entry in entries):
        raise ValueError("A shared FFA key requires the same band order in all sources")
    reference_handles = [
        Line2D([], [], color="k", alpha=0.5, lw=0.9, ls="--", label=r"$X=0$"),
        Line2D([], [], color="k", alpha=0.5, lw=0.9, ls=":", label=r"$X_F$"),
        Line2D([], [], color="k", alpha=0.5, lw=0.9, ls=":", label=r"$X_B$"),
        Line2D([], [], color="k", alpha=0.8, lw=0.9, ls="-.", label=r"$X_0$"),
    ]
    band_handles = [Line2D([], [], color=color, lw=1.15, label=band)
                    for color, band in zip(plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(bands))), bands)]
    header_height = 0.70 * 72
    header_figure = plt.figure(figsize=(7.2, 0.70))
    header_figure.legend(handles=reference_handles, loc="upper center", ncol=4,
                         fontsize=7.0, frameon=False, handlelength=1.7, borderaxespad=0.0,
                         bbox_to_anchor=(0.5, 1.0), columnspacing=2.0)
    header_figure.legend(handles=band_handles, loc="upper center", ncol=len(bands),
                         fontsize=7.0, frameon=False, handlelength=1.7, borderaxespad=0.0,
                         bbox_to_anchor=(0.5, 0.66), columnspacing=2.0)
    for x, label in (
        (0.26, r"(a) $f_\lambda$ [$10^{-15}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$]"),
        (0.75, r"(b) $\lambda F_\lambda$ [$10^{-15}\,\mathrm{erg\,s^{-1}\,cm^{-2}}$]"),
    ):
        header_figure.text(x, 0.025, label, ha="center", va="bottom", fontsize=7.5)
    header_buffer = io.BytesIO()
    header_figure.savefig(header_buffer, format="pdf")
    plt.close(header_figure)
    header_document = fitz.open(stream=header_buffer.getvalue(), filetype="pdf")
    width, pair_width, gap = 7.2 * 72, 3.55 * 72, 0.10 * 72
    footer_height = 0.66 * 72
    pages = []
    for start in range(0, len(entries), 6):
        chunk = entries[start:start + 6]
        documents = [fitz.open(entry["outputs"]["pdf"]) for entry in chunk]
        try:
            heights = [doc[0].rect.height for doc in documents]
            if any(len(doc) != 1 or abs(doc[0].rect.width - pair_width) > 0.02
                   for doc in documents):
                raise ValueError("Compact FFA inputs must be single pages of width 3.55 inches")
            row_height = max(heights)
            rows = (len(chunk) + 1) // 2
            grid_height = rows * row_height + (rows - 1) * gap
            page_height = header_height + grid_height + footer_height
            # The header supplies common y-axis definitions. The footer uses
            # actual SED artists to retain conditional open-square keys.
            annotation_figure = plt.figure(figsize=(7.2, page_height / 72))
            annotation_figure.legend(handles=list(sed_legend_items.values()),
                                     labels=list(sed_legend_items), loc="lower center",
                                     bbox_to_anchor=(0.52, 0.006), ncol=2,
                                     fontsize=7, frameon=False, handlelength=1.7,
                                     labelspacing=0.25, columnspacing=1.5)
            annotation_buffer = io.BytesIO()
            annotation_figure.savefig(annotation_buffer, format="pdf", transparent=True)
            plt.close(annotation_figure)
            annotation_document = fitz.open(stream=annotation_buffer.getvalue(), filetype="pdf")
            with fitz.open() as combined:
                page = combined.new_page(width=width, height=page_height)
                page.show_pdf_page(fitz.Rect(0, 0, width, header_height), header_document, 0)
                for index, (doc, height) in enumerate(zip(documents, heights)):
                    x = (index % 2) * (pair_width + gap)
                    y = header_height + (index // 2) * (row_height + gap)
                    page.show_pdf_page(fitz.Rect(x, y, x + pair_width, y + height), doc, 0)
                page.show_pdf_page(page.rect, annotation_document, 0)
                stem = output_dir / f"FigA_ffa_{start // 6 + 1:02d}"
                pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
                combined.save(pdf, garbage=4, deflate=True)
                page.get_pixmap(dpi=300, alpha=False).save(png)
            annotation_document.close()
            pages.append(dict(sources=[entry["source"] for entry in chunk],
                              outputs=dict(pdf=str(pdf), png=str(png))))
        finally:
            for doc in documents:
                doc.close()
    header_document.close()
    return dict(source_order=sources, compact_sources=entries, appendix_pages=pages,
                layout=dict(source_columns=2, panels_per_row=4, sources_per_page=6,
                            pair_width_inches=3.55, pair_height_inches=2.03,
                            minimum_text_points=7.0,
                            shared_ffa_legend=True, shared_sed_legend=True,
                            shared_y_labels=True, independent_y_scales=True,
                            shared_legend_fontsize=7.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-table", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compact-appendix", action="store_true",
                        help="Redraw saved appendix products and update only appendix manifest entries; preserve the main example")
    parser.add_argument("--reuse-existing", action="store_true",
                        help="Reuse existing full-size source figures in the default paper workflow")
    args = parser.parse_args()
    if args.compact_appendix:
        result = reflow_ffa_appendix(args.sample_table, args.output_dir)
        manifest_path = args.output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["source_order"] != result["source_order"]:
            raise ValueError("Existing FFA manifest source order differs from the selected sample")
        manifest["appendix_pages"] = result["appendix_pages"]
        manifest["compact_sources"] = result["compact_sources"]
        manifest["appendix_layout"] = result["layout"]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    else:
        manifest = build_ffa_paper(args.sample_table, args.output_dir, reuse_existing=args.reuse_existing)
        print(json.dumps({"sources": len(manifest["single_sources"]), "appendix_pages": len(manifest["appendix_pages"])}))


if __name__ == "__main__":
    main()
