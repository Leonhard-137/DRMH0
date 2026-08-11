#!/usr/bin/env python3
"""Draw the geometry of the shifted k=2 Gamma transfer function."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUTPUT = Path("Mrk817/results/tf_comparison/gamma_k2_geometry")

# Solutions of x exp(1-x) = 1/2 on the rising and falling sides.
HALF_LEFT = 0.231960952986534
HALF_RIGHT = 2.67834699001666
FWHM_PER_W = HALF_RIGHT - HALF_LEFT


def plot(output: Path) -> None:
    ink = "#25282D"
    muted = "#5C6470"
    grid = "#E2E6EA"
    blue = "#3B6F91"
    blue_fill = "#DCE9F1"
    orange = "#C46B3B"
    gold = "#B4872B"

    # Dimensionless example: c=0 and w=1. The labels retain the generic form.
    t = np.linspace(-0.65, 7.0, 1600)
    u = t
    response = np.where(u >= 0.0, u * np.exp(1.0 - u), 0.0)

    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    ax.axvspan(t.min(), 0.0, color="#F0F2F4", alpha=0.9, zorder=0)
    ax.fill_between(t, response, color=blue_fill, alpha=0.9, zorder=1)
    ax.plot(t, response, color=blue, lw=2.4, zorder=3)

    # Half maximum and FWHM.
    ax.hlines(0.5, HALF_LEFT, HALF_RIGHT, color=ink, lw=1.2, ls="--", zorder=4)
    ax.scatter(
        [HALF_LEFT, HALF_RIGHT],
        [0.5, 0.5],
        s=36,
        facecolor="white",
        edgecolor=ink,
        linewidth=1.2,
        zorder=5,
    )
    ax.annotate(
        "",
        xy=(HALF_RIGHT, 0.55),
        xytext=(HALF_LEFT, 0.55),
        arrowprops={"arrowstyle": "<->", "color": ink, "lw": 1.2},
    )
    ax.text(
        (HALF_LEFT + HALF_RIGHT) / 2.0,
        0.585,
        rf"FWHM $=({HALF_RIGHT:.3f}-{HALF_LEFT:.3f})w"
        rf"={FWHM_PER_W:.6f}w$",
        ha="center",
        va="bottom",
        fontsize=10.2,
        color=ink,
    )

    # Onset, mode, and centroid.
    ax.axvline(0.0, color=muted, lw=1.25, ls=":", zorder=2)
    ax.axvline(1.0, color=orange, lw=1.5, ls="--", zorder=2)
    ax.axvline(2.0, color=gold, lw=1.5, ls="-.", zorder=2)
    ax.scatter([1.0], [1.0], s=52, color=orange, edgecolor="white",
               linewidth=0.8, zorder=6)
    centroid_y = 2.0 * np.exp(-1.0)
    ax.scatter([2.0], [centroid_y], s=52, color=gold, edgecolor="white",
               linewidth=0.8, zorder=6)

    ax.annotate(
        r"onset $c$" + "\n" + r"$\Psi(c)=0$",
        xy=(0.0, 0.0),
        xytext=(0.28, 0.22),
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=muted,
        arrowprops={"arrowstyle": "->", "color": muted, "lw": 1.0},
    )
    ax.annotate(
        r"mode $c+w$" + "\n" + "maximum response",
        xy=(1.0, 1.0),
        xytext=(0.55, 1.12),
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=orange,
        arrowprops={"arrowstyle": "->", "color": orange, "lw": 1.0},
    )
    ax.annotate(
        r"centroid $c+2w$" + "\n" + r"(mean lag, not the peak)",
        xy=(2.0, centroid_y),
        xytext=(2.55, 0.96),
        ha="left",
        va="center",
        fontsize=10.5,
        color=gold,
        arrowprops={"arrowstyle": "->", "color": gold, "lw": 1.0},
    )

    # Leading-side extent discussed in the preceding analysis.
    ax.annotate(
        "",
        xy=(2.0, -0.105),
        xytext=(0.0, -0.105),
        arrowprops={"arrowstyle": "<->", "color": blue, "lw": 1.5},
        annotation_clip=False,
    )
    ax.text(
        1.0,
        -0.145,
        r"$L_-=(c+2w)-c=2w=0.81753\,\mathrm{FWHM}$",
        ha="center",
        va="top",
        fontsize=10.5,
        color=blue,
        clip_on=False,
    )

    ax.text(
        -0.48,
        0.53,
        r"zero support for $t<c$",
        ha="center",
        va="bottom",
        rotation=90,
        fontsize=9.0,
        color=muted,
    )
    ax.text(
        5.0,
        0.20,
        "long falling tail",
        ha="center",
        va="center",
        fontsize=10,
        color=muted,
    )

    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(-0.24, 1.28)
    ax.set_xticks(
        [0.0, HALF_LEFT, 1.0, 2.0, HALF_RIGHT, 4.0, 6.0],
        [
            r"$c$",
            r"$c+0.232w$",
            r"$c+w$",
            r"$c+2w$",
            r"$c+2.678w$",
            r"$c+4w$",
            r"$c+6w$",
        ],
    )
    ax.set_xlabel("Lag, $t$")
    ax.set_ylabel(r"Response normalized to its peak, $\Psi(t)/\Psi_{\max}$")
    ax.grid(True, color=grid, linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.minorticks_on()

    fig.suptitle(
        "Geometry of the shifted Gamma transfer function (k = 2)",
        x=0.085,
        y=0.975,
        ha="left",
        fontsize=15,
        color=ink,
    )
    fig.text(
        0.085,
        0.925,
        r"$\Psi(t)=A(t-c)e^{-(t-c)/w}/w^2$ for $t\geq c$; "
        r"the illustration uses $c=0$, $w=1$, and peak normalization.",
        ha="left",
        fontsize=10,
        color=muted,
    )
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.18, top=0.86)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.output)
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
