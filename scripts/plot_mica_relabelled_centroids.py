#!/usr/bin/env python3
"""Plot weighted MICA centroid posteriors after per-draw component relabeling."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parents[1]
BLUE = "#1f77b4"
ORANGE = "#d55e00"
INK = "#222222"
GRID = "#d9d9d9"
LINEAR_SCALE_DAYS = 0.25


def source_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def ncomp_from_config(fit: dict) -> int:
    if "ncomp" in fit:
        return int(fit["ncomp"])
    value = fit.get("number_component", [2, 2])
    return int(value[0] if isinstance(value, list) else value)


def weighted_quantile(values, weights, probabilities=(0.15865, 0.5, 0.84135)):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return np.interp(probabilities, cumulative, values)


def display_coordinate(tau):
    """Symmetric log-like coordinate that remains linear around zero."""
    return np.arcsinh(np.asarray(tau, float) / LINEAR_SCALE_DAYS)


def weighted_density(tau, weights, bin_edges):
    transformed = display_coordinate(tau)
    counts, _ = np.histogram(transformed, bins=bin_edges, weights=weights)
    counts = gaussian_filter1d(counts.astype(float), sigma=1.15, mode="nearest")
    if np.max(counts) > 0.0:
        counts /= np.max(counts)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centers, counts


def validate_relabeling(table: pd.DataFrame, bands: list[str]) -> None:
    paired = (
        table[table["band"].isin(bands) & table["component"].isin(["tf0", "tf1"])]
        .pivot(index=["band", "sample"], columns="component", values="tau")
        .dropna()
    )
    violations = int(np.sum(paired["tf0"].to_numpy() > paired["tf1"].to_numpy()))
    if violations:
        raise ValueError(
            f"posterior is not fully relabeled by centroid: {violations} draws have tf0 > tf1"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Source directory or source name, e.g. Mrk335")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    source_dir = source_path(args.source)
    with (source_dir / "config" / "source_config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    fit = dict(config.get("mica_round1", {}) or {})
    ncomp = ncomp_from_config(fit)
    output_root = Path(fit.get("output_root", "runs/mica"))
    if not output_root.is_absolute():
        output_root = source_dir / output_root
    result_dir = output_root / f"{ncomp}comp" / "result"
    posterior_path = result_dir / "mica_component_posterior.txt"
    table = pd.read_csv(posterior_path, sep=r"\s+")

    bands = [
        str(band)
        for band in config.get("response_bands", fit.get("responses", []))
    ]
    if not bands:
        raise ValueError("no response bands configured")
    validate_relabeling(table, bands)

    tick_days = np.array([-10, -3, -1, -0.3, 0, 0.3, 1, 3, 10, 30, 100], float)
    tick_positions = display_coordinate(tick_days)
    lo = float(display_coordinate(-10.5))
    hi = float(display_coordinate(120.0))
    bin_edges = np.linspace(lo, hi, 220)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.9,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
    })
    fig, axes = plt.subplots(
        len(bands), 1, figsize=(11.4, 9.0), sharex=True,
        gridspec_kw={"hspace": 0.10},
    )
    if len(bands) == 1:
        axes = [axes]

    summary_rows = []
    for ax, band in zip(axes, bands):
        for component, color, linestyle, fill_alpha, interval_y in (
            ("tf0", BLUE, "-", 0.24, 1.02),
            ("tf1", ORANGE, "--", 0.10, 0.91),
        ):
            data = table[(table["band"].astype(str) == band) & (table["component"] == component)]
            tau = pd.to_numeric(data["tau"], errors="coerce").to_numpy(float)
            weights = pd.to_numeric(data["weight"], errors="coerce").to_numpy(float)
            valid = np.isfinite(tau) & np.isfinite(weights) & (weights > 0.0)
            tau = tau[valid]
            weights = weights[valid]
            weights /= np.sum(weights)

            x, density = weighted_density(tau, weights, bin_edges)
            density *= 0.78
            ax.fill_between(x, 0.0, density, color=color, alpha=fill_alpha, linewidth=0)
            ax.plot(x, density, color=color, lw=1.9, ls=linestyle)

            q16, q50, q84 = weighted_quantile(tau, weights)
            qx = display_coordinate([q16, q50, q84])
            ax.plot(qx[[0, 2]], [interval_y, interval_y], color=color, lw=3.0,
                    solid_capstyle="round")
            ax.plot(qx[1], interval_y, marker="o", ms=5.2, color=color,
                    markeredgecolor="white", markeredgewidth=0.8)
            summary_rows.append({
                "band": band,
                "component": component,
                "centroid_p16_days": q16,
                "centroid_p50_days": q50,
                "centroid_p84_days": q84,
                "probability_centroid_le_zero": float(np.sum(weights[tau <= 0.0])),
                "effective_sample_size": float(1.0 / np.sum(weights**2)),
            })

        ax.set_ylim(0.0, 1.10)
        ax.set_yticks([])
        ax.text(0.012, 0.83, band, transform=ax.transAxes, fontsize=12,
                fontweight="bold", ha="left", va="top")
        ax.grid(axis="x", color=GRID, lw=0.7, alpha=0.75)
        ax.axvline(display_coordinate(0.0), color="#666666", lw=0.9, ls=":")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if ax is not axes[-1]:
            ax.spines["bottom"].set_color("#aaaaaa")

    axes[0].plot([], [], color=BLUE, lw=2.2, label="comp0 / tf0 (shorter centroid)")
    axes[0].plot([], [], color=ORANGE, lw=2.2, ls="--", label="comp1 / tf1 (longer centroid)")
    axes[0].plot([], [], color="#555555", lw=3.0, marker="o", ms=5,
                 label="weighted 16–50–84% interval")
    axes[0].legend(loc="upper right", frameon=False, ncol=3, fontsize=9.5,
                   bbox_to_anchor=(1.0, 1.30))

    axes[-1].set_xlim(lo, hi)
    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels(["−10", "−3", "−1", "−0.3", "0", "0.3", "1", "3", "10", "30", "100"])
    axes[-1].set_xlabel(
        r"Gamma centroid $\tau=c+2w$ (days; symmetric log-like display)",
        fontsize=11.5,
    )
    fig.suptitle(
        f"{config.get('source', source_dir.name)}: relabelled component-centroid posteriors",
        x=0.08, y=0.982, ha="left", fontsize=17, fontweight="bold",
    )
    fig.text(
        0.08, 0.947,
        "Each posterior draw is relabelled so comp0 has the shorter centroid; original nested-sampling weights are retained.",
        ha="left", va="top", fontsize=10.5, color="#555555",
    )
    fig.text(
        0.08, 0.012,
        "Horizontal display is linear within about ±0.25 d and logarithmic-like farther out; no posterior tail is clipped.",
        ha="left", va="bottom", fontsize=9.2, color="#666666",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.895, bottom=0.10)

    output_dir = args.output_dir or (result_dir / "diagnostics")
    if not output_dir.is_absolute():
        output_dir = source_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{config.get('source', source_dir.name)}_relabelled_centroid_distributions"
    fig.savefig(stem.with_suffix(".png"), dpi=240, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    pd.DataFrame(summary_rows).to_csv(stem.with_name(stem.name + "_summary.csv"), index=False)
    plt.close(fig)

    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    print(stem.with_name(stem.name + "_summary.csv"))


if __name__ == "__main__":
    main()
