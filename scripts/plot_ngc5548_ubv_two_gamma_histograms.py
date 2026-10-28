#!/usr/bin/env python3
"""Plot weighted UBV parameter histograms for the NGC 5548 two-Gamma fit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSTERIOR = ROOT / (
    "NGC5548/runs/mica/gamma2/2comp/result/comp0/"
    "mica_component_posterior.txt"
)
DEFAULT_OUTPUT = ROOT / "output/pdf/NGC5548_UBV_two_gamma_parameter_histograms"

BANDS = ("U", "B", "V")
COMPONENTS = (
    ("tf0", "short", "#326B95", "#1E435E"),
    ("tf1", "delayed", "#C8753D", "#7E4525"),
)
QUANTITIES = (
    ("tau", r"Centroid $\tau=c+2w$ (days)", False),
    ("center", r"Location $c$ (days)", False),
    ("width", r"Scale $w$ (days; log x)", True),
)
DISPLAY_TAIL = 0.001

INK = "#25282D"
MUTED = "#5C626B"
GRID = "#E1E5E9"
ZERO = "#737A82"


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: tuple[float, ...] | list[float] | np.ndarray,
) -> np.ndarray:
    """Return linearly interpolated weighted quantiles."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0 or np.sum(weights) <= 0.0:
        raise ValueError("weighted quantile received no positive-weight samples")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return np.interp(probabilities, cumulative, values)


def kish_ess(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights) & (weights > 0.0)]
    weights /= np.sum(weights)
    return float(1.0 / np.sum(weights**2))


def load_and_validate(path: Path) -> pd.DataFrame:
    required = {
        "band",
        "sample",
        "component",
        "center",
        "width",
        "tau",
        "weight",
    }
    table = pd.read_csv(path, sep=r"\s+", usecols=list(required))
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"missing posterior columns: {sorted(missing)}")

    table = table[
        table["band"].astype(str).isin(BANDS)
        & table["component"].astype(str).isin({item[0] for item in COMPONENTS})
    ].copy()
    for name in ("sample", "center", "width", "tau", "weight"):
        table[name] = pd.to_numeric(table[name], errors="raise")

    expected_groups = {(band, component[0]) for band in BANDS for component in COMPONENTS}
    observed_groups = set(zip(table["band"], table["component"]))
    if observed_groups != expected_groups:
        raise ValueError(
            f"posterior groups differ from expected UBV x tf0/tf1: {observed_groups}"
        )

    counts = table.groupby(["band", "component"], sort=False).size()
    if counts.nunique() != 1 or int(counts.iloc[0]) < 100:
        raise ValueError(f"unexpected or inconsistent group sizes: {counts.to_dict()}")
    if not np.all(np.isfinite(table[["center", "width", "tau", "weight"]])):
        raise ValueError("posterior contains NaN or infinite values")
    if np.any(table["width"].to_numpy() <= 0.0):
        raise ValueError("Gamma scale w must be strictly positive")
    if np.any(table["weight"].to_numpy() < 0.0):
        raise ValueError("posterior weights must be non-negative")

    tau_recomputed = table["center"].to_numpy() + 2.0 * table["width"].to_numpy()
    max_tau_error = float(np.max(np.abs(table["tau"].to_numpy() - tau_recomputed)))
    if max_tau_error > 1.0e-7:
        raise ValueError(f"tau != c + 2w; max absolute error is {max_tau_error:.3g}")

    paired = table.pivot(index=["band", "sample"], columns="component", values="tau")
    if paired.isna().any().any():
        raise ValueError("component pairing failed for one or more posterior draws")
    violations = int(np.sum(paired["tf0"].to_numpy() > paired["tf1"].to_numpy()))
    if violations:
        raise ValueError(f"relabeling check failed: {violations} draws have tf0 > tf1")

    paired_weights = table.pivot(
        index=["band", "sample"], columns="component", values="weight"
    )
    if not np.allclose(
        paired_weights["tf0"].to_numpy(),
        paired_weights["tf1"].to_numpy(),
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise ValueError("tf0 and tf1 do not carry identical per-draw weights")
    return table


def panel_range(
    values: np.ndarray, weights: np.ndarray, logarithmic: bool
) -> tuple[float, float]:
    lo, hi = weighted_quantile(
        values, weights, (DISPLAY_TAIL, 1.0 - DISPLAY_TAIL)
    )
    if logarithmic:
        if lo <= 0.0:
            raise ValueError("logarithmic panel includes a non-positive value")
        factor = np.exp(0.045 * np.log(hi / lo)) if hi > lo else 1.08
        return float(lo / factor), float(hi * factor)
    span = float(hi - lo)
    pad = 0.045 * span if span > 0.0 else max(0.05 * abs(float(lo)), 0.05)
    return float(lo - pad), float(hi + pad)


def histogram_edges(
    values: np.ndarray,
    weights: np.ndarray,
    lo: float,
    hi: float,
    logarithmic: bool,
) -> np.ndarray:
    ess = kish_ess(weights)
    transformed = np.log10(values) if logarithmic else values
    q25, q75 = weighted_quantile(transformed, weights, (0.25, 0.75))
    display_lo = np.log10(lo) if logarithmic else lo
    display_hi = np.log10(hi) if logarithmic else hi
    width = 2.0 * float(q75 - q25) / np.cbrt(max(ess, 1.0))
    if width <= 0.0 or not np.isfinite(width):
        bins = 45
    else:
        bins = int(np.clip(np.ceil((display_hi - display_lo) / width), 32, 90))
    if logarithmic:
        return np.geomspace(lo, hi, bins + 1)
    return np.linspace(lo, hi, bins + 1)


def format_days(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 100.0:
        return f"{value:.1f}"
    if magnitude >= 10.0:
        return f"{value:.2f}"
    if magnitude >= 1.0:
        return f"{value:.3f}"
    if magnitude >= 0.01:
        return f"{value:.4f}"
    return f"{value:.3g}"


def write_summary(rows: list[dict[str, object]], output: Path) -> Path:
    summary_path = output.with_name(output.name + "_summary.csv")
    fields = list(rows[0])
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def plot(table: pd.DataFrame, output: Path) -> tuple[Path, Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.4,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(6, 3, figsize=(14.2, 17.2), squeeze=False)
    summary_rows: list[dict[str, object]] = []

    row_index = 0
    for band in BANDS:
        for component, role, fill, edge in COMPONENTS:
            group = table[
                (table["band"].astype(str) == band)
                & (table["component"].astype(str) == component)
            ]
            weights = group["weight"].to_numpy(dtype=float, copy=True)
            weights /= np.sum(weights)
            ess = kish_ess(weights)

            for column, (quantity, title, logarithmic) in enumerate(QUANTITIES):
                ax = axes[row_index, column]
                values = group[quantity].to_numpy(float)
                q001, q16, q50, q84, q999 = weighted_quantile(
                    values,
                    weights,
                    (DISPLAY_TAIL, 0.15865, 0.5, 0.84135, 1.0 - DISPLAY_TAIL),
                )
                lo, hi = panel_range(values, weights, logarithmic)
                edges = histogram_edges(values, weights, lo, hi, logarithmic)

                ax.hist(
                    values,
                    bins=edges,
                    weights=weights,
                    color=fill,
                    edgecolor=edge,
                    linewidth=0.45,
                    alpha=0.82,
                )
                ax.axvline(q50, color=INK, lw=1.25, zorder=4)
                ax.axvline(q16, color=INK, lw=0.9, ls=":", alpha=0.88, zorder=4)
                ax.axvline(q84, color=INK, lw=0.9, ls=":", alpha=0.88, zorder=4)
                if not logarithmic and lo < 0.0 < hi:
                    ax.axvline(0.0, color=ZERO, lw=0.8, ls="--", alpha=0.65, zorder=2)
                if logarithmic:
                    ax.set_xscale("log")
                ax.set_xlim(lo, hi)
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
                ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.9)
                ax.set_axisbelow(True)
                ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=8.4)
                ax.text(
                    0.975,
                    0.94,
                    f"median {format_days(q50)} d\n"
                    f"68% [{format_days(q16)}, {format_days(q84)}] d",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8.0,
                    bbox={
                        "boxstyle": "round,pad=0.24",
                        "facecolor": "white",
                        "edgecolor": GRID,
                        "alpha": 0.92,
                    },
                )
                if row_index == 0:
                    ax.set_title(title, fontsize=11.2, weight="bold", pad=10)
                if column == 0:
                    ax.set_ylabel(
                        f"{band} - {component} ({role})\n"
                        f"posterior mass / bin\nKish ESS = {ess:,.0f}",
                        fontsize=9.0,
                    )

                summary_rows.append(
                    {
                        "band": band,
                        "component": component,
                        "role": role,
                        "quantity": quantity,
                        "n_raw": int(group.shape[0]),
                        "n_positive_weight": int(np.sum(weights > 0.0)),
                        "kish_ess": f"{ess:.10g}",
                        "display_q001": f"{q001:.10g}",
                        "q16": f"{q16:.10g}",
                        "median": f"{q50:.10g}",
                        "q84": f"{q84:.10g}",
                        "display_q999": f"{q999:.10g}",
                    }
                )
            row_index += 1

    legend_handles = [
        Patch(facecolor=COMPONENTS[0][2], edgecolor=COMPONENTS[0][3], label="tf0 (shorter centroid)"),
        Patch(facecolor=COMPONENTS[1][2], edgecolor=COMPONENTS[1][3], label="tf1 (longer centroid)"),
        Line2D([0], [0], color=INK, lw=1.25, label="median"),
        Line2D([0], [0], color=INK, lw=0.9, ls=":", label="16th / 84th percentiles"),
        Line2D([0], [0], color=ZERO, lw=0.8, ls="--", label="zero reference"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.075, 0.934),
        ncol=5,
        frameon=False,
        fontsize=9.0,
        handlelength=2.2,
        columnspacing=1.3,
    )
    fig.suptitle(
        "NGC 5548: UBV two-Gamma posterior parameter distributions",
        x=0.075,
        y=0.987,
        ha="left",
        fontsize=17.0,
        weight="bold",
    )
    fig.text(
        0.075,
        0.962,
        "Full weighted comp0 branch; components are relabelled within every draw by increasing centroid, "
        r"with $\tau=c+2w$.",
        ha="left",
        va="top",
        fontsize=10.0,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.947,
        "Each panel displays the weighted 0.1-99.9 percentile range (at most 0.2% tail mass omitted); "
        "w panels use logarithmic x-axes.",
        ha="left",
        va="top",
        fontsize=9.4,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.018,
        "Input: collected raw nested-sampling posterior and original weights; no equal-weight resampling and no extra posterior cut.",
        ha="left",
        va="bottom",
        fontsize=8.8,
        color=MUTED,
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.055,
        top=0.895,
        wspace=0.24,
        hspace=0.50,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    png_path = output.with_suffix(".png")
    pdf_path = output.with_suffix(".pdf")
    fig.savefig(png_path, dpi=240, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    summary_path = write_summary(summary_rows, output)
    return png_path, pdf_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", type=Path, default=DEFAULT_POSTERIOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    posterior = args.posterior.expanduser().resolve()
    output = args.output.expanduser()
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    table = load_and_validate(posterior)
    for path in plot(table, output):
        print(path)


if __name__ == "__main__":
    main()
