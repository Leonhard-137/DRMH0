#!/usr/bin/env python3
"""Plot the Mrk817 multi-band light curves with error bars."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "Mrk817"


def main() -> None:
    src_dir = ROOT / SOURCE
    config = yaml.safe_load(
        (src_dir / "config" / "source_config.yaml").read_text(encoding="utf-8")
    )
    bands = list(config["band_list"])
    wavelengths = dict(config.get("bands", {}))

    data = pd.read_csv(src_dir / "raw" / f"{SOURCE}.csv")
    data["Filter"] = pd.Categorical(data["Filter"], categories=bands, ordered=True)
    data = data.sort_values(["Filter", "MJD"]).reset_index(drop=True)

    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(bands)))

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(17.5, 7.4),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    axes = axes.ravel()

    for ax, band, color in zip(axes, bands, colors):
        band_data = data[data["Filter"] == band]
        ax.errorbar(
            band_data["MJD"],
            band_data["Flux"],
            yerr=band_data["Error"],
            fmt="o",
            markersize=2.4,
            linewidth=0,
            elinewidth=0.8,
            capsize=1.4,
            color=color,
            alpha=0.85,
            rasterized=True,
        )
        wave = wavelengths.get(band)
        title = f"{band}  ({wave} Å)" if wave else band
        ax.set_title(title, loc="left", fontsize=11)
        ax.text(
            0.98,
            0.96,
            f"N = {len(band_data)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="0.25",
        )
        ax.grid(True, color="#E2E6EA", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.minorticks_on()
        ax.xaxis.set_major_locator(MultipleLocator(200))
        ax.xaxis.set_minor_locator(MultipleLocator(50))

    for ax in axes[::5]:
        ax.set_ylabel("Flux (mJy)")
    for ax in axes[-5:]:
        ax.set_xlabel("MJD")

    fig.suptitle("Mrk 817 light curves", fontsize=15, y=1.005)

    png_path = src_dir / f"{SOURCE}_lightcurves.png"
    pdf_path = src_dir / f"{SOURCE}_lightcurves.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
