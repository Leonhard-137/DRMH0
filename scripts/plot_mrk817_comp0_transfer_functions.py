#!/usr/bin/env python3
"""Reconstruct and plot the component-0 MICA transfer functions for Mrk817.

The component transfer functions are reconstructed directly from
``posterior_sample1d.txt_2``.  This is necessary because MICA's
``tranfunc_0_1.txt_2`` is indexed by driver/response light curves (0 -> 1),
not by the Gaussian/Gamma component number; it contains the total response.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


BANDS = ("UVM2", "UVW1", "U", "B", "V")
WAVELENGTHS = {"UVM2": 2246, "UVW1": 2600, "U": 3465, "B": 4392, "V": 5468}
COLORS = {"Gaussian": "#3B6F91", "Gamma": "#C46B3B"}
LINESTYLES = {"Gaussian": "-", "Gamma": "--"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gaussian-root",
        type=Path,
        default=root / "Mrk817/runs/mica/gaussian2_uvw2_lag_m10_100/2comp",
    )
    parser.add_argument(
        "--gamma-root",
        type=Path,
        default=root / "Mrk817/runs/mica/gamma2_uvw2_lag_m10_100/2comp",
    )
    parser.add_argument(
        "--gamma-b-root",
        type=Path,
        default=root / "Mrk817/runs/mica/gamma2_uvw2_B_lag_m10_200/2comp",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "Mrk817/results/tf_comparison/Mrk817_comp0_transfer_functions_linear",
    )
    return parser.parse_args()


def run_dir(root: Path, band: str, model: str) -> Path:
    return root / f"run_UVW2_to_{band}_2comp_{model.lower()}"


def read_component0_samples(run: Path) -> tuple[np.ndarray, dict[str, int]]:
    data = run / "data"
    sample = np.loadtxt(data / "posterior_sample1d.txt_2", comments="#")
    sample = np.atleast_2d(sample)
    names: dict[str, int] = {}
    for line in (data / "para_names_line.txt_2").read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            names[parts[1]] = int(parts[0])
    required = (
        "0-th_component_amplitude",
        "0-th_component_center",
        "0-th_component_sigma",
    )
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"Missing component-0 parameters in {run}: {missing}")
    return sample, names


def component0_parameters(sample: np.ndarray, names: dict[str, int]):
    amplitude = np.exp(sample[:, names["0-th_component_amplitude"]])
    center = sample[:, names["0-th_component_center"]]
    sigma = np.exp(sample[:, names["0-th_component_sigma"]])
    return amplitude, center, sigma


def component0_grid_bounds(center: np.ndarray, sigma: np.ndarray, model: str) -> tuple[float, float]:
    if model == "Gaussian":
        lower = np.quantile(center - 3.0 * sigma, 0.05)
        upper = np.quantile(center + 3.0 * sigma, 0.95)
    else:
        lower = np.quantile(center - 0.2 * sigma, 0.05)
        upper = np.quantile(center + 6.0 * sigma, 0.95)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError(f"Invalid lag range for {model}: {lower}, {upper}")
    return float(lower), float(upper)


def reconstruct_component0(
    sample: np.ndarray,
    names: dict[str, int],
    model: str,
    lag: np.ndarray,
    chunk_size: int = 512,
) -> tuple[np.ndarray, ...]:
    amplitude, center, sigma = component0_parameters(sample, names)
    transfer = np.empty((sample.shape[0], lag.size), dtype=float)
    for start in range(0, sample.shape[0], chunk_size):
        stop = min(start + chunk_size, sample.shape[0])
        amp = amplitude[start:stop, None]
        cen = center[start:stop, None]
        sig = sigma[start:stop, None]
        delta = lag[None, :] - cen
        if model == "Gaussian":
            transfer[start:stop] = (
                amp / (sig * np.sqrt(2.0 * np.pi))
                * np.exp(-0.5 * (delta / sig) ** 2)
            )
        elif model == "Gamma":
            positive_delta = np.maximum(delta, 0.0)
            transfer[start:stop] = np.where(
                delta >= 0.0,
                amp / sig**2 * positive_delta * np.exp(-positive_delta / sig),
                0.0,
            )
        else:
            raise ValueError(f"Unsupported model: {model}")
    lower, median, upper = np.quantile(transfer, [0.1585, 0.5, 0.8415], axis=0)
    if model == "Gaussian":
        component_centroid = center
    else:
        component_centroid = center + 2.0 * sigma
    centroid_median = float(np.quantile(component_centroid, 0.5))
    return lag, median, lower, upper, centroid_median


def load_curves(args: argparse.Namespace) -> dict[str, dict[str, tuple[np.ndarray, ...]]]:
    curves: dict[str, dict[str, tuple[np.ndarray, ...]]] = {}
    for band in BANDS:
        inputs: dict[str, tuple[np.ndarray, dict[str, int]]] = {}
        bounds = []
        for model in ("Gaussian", "Gamma"):
            root = args.gamma_b_root if model == "Gamma" and band == "B" else (
                args.gamma_root if model == "Gamma" else args.gaussian_root
            )
            sample, names = read_component0_samples(run_dir(root, band, model))
            inputs[model] = (sample, names)
            _, center, sigma = component0_parameters(sample, names)
            bounds.append(component0_grid_bounds(center, sigma, model))

        lag = np.linspace(min(bound[0] for bound in bounds), max(bound[1] for bound in bounds), 1200)
        curves[band] = {
            model: reconstruct_component0(sample, names, model, lag)
            for model, (sample, names) in inputs.items()
        }
    return curves


def positive_mask(*values: np.ndarray) -> np.ndarray:
    return np.logical_and.reduce([np.isfinite(value) & (value > 0.0) for value in values])


def plot(curves: dict[str, dict[str, tuple[np.ndarray, ...]]], output: Path) -> None:
    ink = "#25282D"
    muted = "#59616B"
    grid = "#DDE2E6"
    zero = "#7A828A"

    fig, axes = plt.subplots(
        len(BANDS),
        1,
        figsize=(10.5, 14.5),
        gridspec_kw={"height_ratios": [1.45, 1.0, 1.0, 1.0, 1.0]},
        sharex=False,
        squeeze=False,
    )

    for row, band in enumerate(BANDS):
        band_curves = curves[band]
        xmin = min(values[0][0] for values in band_curves.values())
        xmax = max(values[0][-1] for values in band_curves.values())
        if band == "UVM2":
            # The M2 component-0 core is concentrated near zero; zoom in so
            # its shape and the centroid markers are readable.
            xmin, xmax = -2.0, 8.0
        all_upper = np.concatenate([values[3] for values in band_curves.values()])
        ymax = float(np.nanmax(all_upper)) * 1.10

        linear_ax = axes[row, 0]
        linear_ax.axvline(0.0, color=zero, linestyle=":", linewidth=0.85, zorder=0)
        linear_ax.grid(True, color=grid, linewidth=0.65, alpha=0.85)
        linear_ax.set_axisbelow(True)
        linear_ax.set_xlim(xmin, xmax)
        linear_ax.set_ylim(0.0, ymax)
        linear_ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=8.5)
        linear_ax.minorticks_on()

        for model in ("Gaussian", "Gamma"):
            lag, median, lower, upper, centroid_median = band_curves[model]
            color = COLORS[model]
            linestyle = LINESTYLES[model]

            linear_ax.fill_between(
                lag,
                lower,
                upper,
                color=color,
                alpha=0.14,
                linewidth=0.0,
                zorder=1,
            )
            linear_ax.plot(
                lag,
                median,
                color=color,
                linestyle=linestyle,
                linewidth=1.45,
                zorder=2,
            )
            linear_ax.axvline(
                centroid_median,
                color=color,
                linestyle=":",
                linewidth=1.15,
                alpha=0.9,
                zorder=3,
            )

        linear_ax.set_ylabel(f"{band} ({WAVELENGTHS[band]} Å)\n$\Psi_0(\tau)$", fontsize=9.5)
        if band == "UVM2":
            linear_ax.text(
                0.98,
                0.94,
                "zoomed in",
                transform=linear_ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color=muted,
            )
        if row == 0:
            linear_ax.set_title("Component-0 transfer function (linear y-scale)", fontsize=11.5, color=ink)
        if row == len(BANDS) - 1:
            linear_ax.set_xlabel("Lag $\tau$ (day)", fontsize=10)

    legend_handles = [
        Line2D([0], [0], color=COLORS["Gaussian"], linestyle="-", linewidth=1.6, label="Double Gaussian TF$_0$"),
        Line2D([0], [0], color=COLORS["Gamma"], linestyle="--", linewidth=1.6, label="Double Gamma TF$_0$"),
        Line2D([0], [0], color=ink, linestyle=":", linewidth=1.2, label="TF$_0$ centroid"),
        Line2D([0], [0], color=ink, linewidth=7.0, alpha=0.14, label="68.3% posterior band"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        frameon=False,
        fontsize=9.2,
    )
    fig.suptitle(
        "Mrk817 two-component MICA fits: component-0 transfer functions",
        x=0.065,
        y=0.998,
        ha="left",
        fontsize=15,
        color=ink,
    )
    fig.text(
        0.065,
        0.968,
        "Direct component-0 posterior reconstruction; vertical lines mark the posterior-median centroid; "
        "M2 is shown with a focused lag range.",
        ha="left",
        fontsize=9.3,
        color=muted,
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.055, top=0.885, hspace=0.34)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    curves = load_curves(args)
    plot(curves, args.output)
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
