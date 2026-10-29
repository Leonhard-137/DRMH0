#!/usr/bin/env python3
"""Compare MICA Gamma center/width posteriors before and after tau0 >= 0."""

# 比较既有 comp0 与 comp0_tau_nonnegative 分支的 c、w 后验。
# 先验证筛后样本恰为 tau0>=0 的条件子集及正确重归一权重；没有运行新的分支拟合。
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
BEFORE = "#8A929B"
BEFORE_DARK = "#555D66"
AFTER = "#2F6F9F"
AFTER_FILL = "#A9C8DE"
INK = "#20252B"
MUTED = "#59636E"
GRID = "#DEE3E8"
CENTER_LINEAR_SCALE_DAYS = 0.25


def resolve_source(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: tuple[float, ...] = (0.16, 0.50, 0.84),
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0 or float(np.sum(weights)) <= 0.0:
        raise ValueError("weighted quantile received no positive finite weight")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return np.interp(probabilities, cumulative, values)


def normalized_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = pd.to_numeric(frame["weight"], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("posterior contains invalid weights")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("posterior has zero total weight")
    return weights / total


def component_frame(
    table: pd.DataFrame,
    band: str,
    component: str,
) -> pd.DataFrame:
    frame = table.loc[
        (table["band"].astype(str) == band)
        & (table["component"].astype(str) == component)
    ].copy()
    if frame.empty:
        raise ValueError(f"missing posterior rows for {band}/{component}")
    return frame.sort_values("sample").reset_index(drop=True)


# 先核对分量 sample 一一配对，再核对筛后 ID 恰为 tau0>=0，最后核对条件化权重和 ESS。
def validate_conditioning(
    before: pd.DataFrame,
    after: pd.DataFrame,
    bands: list[str],
) -> dict[str, dict[str, float]]:
    quality: dict[str, dict[str, float]] = {}
    for band in bands:
        b0 = component_frame(before, band, "tf0")
        b1 = component_frame(before, band, "tf1")
        a0 = component_frame(after, band, "tf0")
        a1 = component_frame(after, band, "tf1")

        if not np.array_equal(b0["sample"].to_numpy(), b1["sample"].to_numpy()):
            raise ValueError(f"before posterior is not paired for {band}")
        if not np.array_equal(a0["sample"].to_numpy(), a1["sample"].to_numpy()):
            raise ValueError(f"after posterior is not paired for {band}")

        tau0 = pd.to_numeric(b0["tau"], errors="coerce").to_numpy(float)
        tau1 = pd.to_numeric(b1["tau"], errors="coerce").to_numpy(float)
        if np.any(tau0 > tau1 + 1e-10):
            raise ValueError(f"before posterior is not relabelled by centroid for {band}")
        expected = b0.loc[tau0 >= 0.0, "sample"].to_numpy()
        actual = a0["sample"].to_numpy()
        if not np.array_equal(expected, actual):
            raise ValueError(f"after posterior is not exactly tau0 >= 0 for {band}")

        base_weights = normalized_weights(b0)
        keep = tau0 >= 0.0
        retained_mass = float(np.sum(base_weights[keep]))
        conditional = base_weights[keep] / retained_mass
        after_weights = normalized_weights(a0)
        if not np.allclose(conditional, after_weights, rtol=5e-8, atol=5e-12):
            raise ValueError(f"conditional weights do not match the base posterior for {band}")

        quality[band] = {
            "n_before": float(len(b0)),
            "n_after": float(len(a0)),
            "retained_mass": retained_mass,
            "ess_before": float(1.0 / np.sum(base_weights**2)),
            "ess_after": float(1.0 / np.sum(after_weights**2)),
        }
    return quality


# 按每个 bin 累积概率；不同宽度/坐标变换的 bin 高度不能直接当单位天的密度比较。
def probability_per_bin(
    values: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges, weights=weights)
    total = float(np.sum(counts))
    return counts / total if total > 0.0 else counts


def center_coordinate(values: np.ndarray | float) -> np.ndarray:
    """Symmetric log-like display coordinate that is linear near c=0."""
    return np.arcsinh(np.asarray(values, dtype=float) / CENTER_LINEAR_SCALE_DAYS)


def interval_label(values: np.ndarray, weights: np.ndarray) -> str:
    q16, q50, q84 = weighted_quantile(values, weights)
    return f"{q50:.3g} [{q16:.3g}, {q84:.3g}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="source directory or name, e.g. Mrk335")
    parser.add_argument("--before-branch", default="comp0")
    parser.add_argument("--after-branch", default="comp0_tau_nonnegative")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    source_dir = resolve_source(args.source)
    config_path = source_dir / "config" / "source_config.yaml"
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    fit = dict(config.get("mica_round1", {}) or {})
    ncomp = int(fit.get("ncomp", 2))
    output_root = Path(str(fit.get("output_root", "runs/mica")))
    if not output_root.is_absolute():
        output_root = source_dir / output_root
    result_dir = output_root / f"{ncomp}comp" / "result"

    before_path = result_dir / args.before_branch / "mica_component_posterior.txt"
    after_path = result_dir / args.after_branch / "mica_component_posterior.txt"
    before = pd.read_csv(before_path, sep=r"\s+")
    after = pd.read_csv(after_path, sep=r"\s+")
    bands = [str(value) for value in config.get("response_bands", [])]
    if not bands:
        raise ValueError("source config has no response_bands")
    quality = validate_conditioning(before, after, bands)

    center_edges = np.linspace(
        float(center_coordinate(-10.5)), float(center_coordinate(50.5)), 64
    )
    width_edges = np.geomspace(0.04, 34.0, 58)
    panels = (
        ("tf0", "center", center_edges, r"comp0 location $c_0$ [d]", "center"),
        ("tf0", "width", width_edges, r"comp0 width $w_0$ [d]", "log"),
        ("tf1", "center", center_edges, r"comp1 location $c_1$ [d]", "center"),
        ("tf1", "width", width_edges, r"comp1 width $w_1$ [d]", "log"),
    )
    center_tick_days = np.array(
        [-10.0, -3.0, -1.0, -0.3, 0.0, 0.3, 1.0, 3.0, 10.0, 30.0],
        dtype=float,
    )
    center_tick_labels = ["−10", "−3", "−1", "−0.3", "0", "0.3", "1", "3", "10", "30"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.2,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
    })
    fig, axes = plt.subplots(
        len(bands), len(panels), figsize=(16.4, 11.8),
        sharex="col", sharey=False,
        gridspec_kw={"hspace": 0.34, "wspace": 0.20},
    )
    summary_rows: list[dict[str, object]] = []

    for row_index, band in enumerate(bands):
        info = quality[band]
        for column_index, (component, quantity, edges, xlabel, scale) in enumerate(panels):
            ax = axes[row_index, column_index]
            b = component_frame(before, band, component)
            a = component_frame(after, band, component)
            b_values = pd.to_numeric(b[quantity], errors="coerce").to_numpy(float)
            a_values = pd.to_numeric(a[quantity], errors="coerce").to_numpy(float)
            b_weights = normalized_weights(b)
            a_weights = normalized_weights(a)
            if scale == "center":
                b_display = center_coordinate(b_values)
                a_display = center_coordinate(a_values)
            else:
                b_display = b_values
                a_display = a_values
            b_hist = probability_per_bin(b_display, b_weights, edges)
            a_hist = probability_per_bin(a_display, a_weights, edges)

            ax.stairs(
                b_hist, edges, fill=True, color=BEFORE, alpha=0.28,
                linewidth=0.9, edgecolor=BEFORE_DARK,
            )
            ax.stairs(
                a_hist, edges, fill=True, color=AFTER_FILL, alpha=0.42,
                linewidth=0.0,
            )
            ax.stairs(a_hist, edges, color=AFTER, linewidth=1.55)
            b_q16, b_q50, b_q84 = weighted_quantile(b_values, b_weights)
            a_q16, a_q50, a_q84 = weighted_quantile(a_values, a_weights)
            b_median_display = (
                float(center_coordinate(b_q50)) if scale == "center" else b_q50
            )
            a_median_display = (
                float(center_coordinate(a_q50)) if scale == "center" else a_q50
            )
            ax.axvline(b_median_display, color=BEFORE_DARK, lw=1.0, ls="--")
            ax.axvline(a_median_display, color=AFTER, lw=1.25)
            if scale == "log":
                ax.set_xscale("log")
            elif scale == "center":
                ax.set_xticks(center_coordinate(center_tick_days))
                ax.set_xticklabels(center_tick_labels)
            ax.set_xlim(edges[0], edges[-1])
            ax.grid(axis="y", color=GRID, linewidth=0.65, alpha=0.9)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction="out", length=3.2, width=0.7, labelsize=8.2)
            ax.text(
                0.98, 0.94,
                f"median {b_q50:.3g} → {a_q50:.3g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.7,
                color=INK,
                bbox={
                    "boxstyle": "round,pad=0.20", "facecolor": "white",
                    "edgecolor": GRID, "alpha": 0.92,
                },
            )
            if row_index == len(bands) - 1:
                ax.set_xlabel(xlabel, fontsize=9.2)
            if column_index == 0:
                ax.set_ylabel(
                    f"{band}\nweighted probability / bin\n"
                    f"mass={info['retained_mass']:.1%}",
                    fontsize=8.5,
                )

            for state, values, weights, q16, q50, q84 in (
                ("before", b_values, b_weights, b_q16, b_q50, b_q84),
                ("after", a_values, a_weights, a_q16, a_q50, a_q84),
            ):
                summary_rows.append({
                    "band": band,
                    "component": component,
                    "quantity": quantity,
                    "state": state,
                    "n_rows": len(values),
                    "retained_parent_mass": info["retained_mass"] if state == "after" else 1.0,
                    "effective_sample_size": info["ess_after"] if state == "after" else info["ess_before"],
                    "p16": q16,
                    "p50": q50,
                    "p84": q84,
                })

    handles = [
        Line2D([0], [0], color=BEFORE_DARK, lw=5.0, alpha=0.55,
               label=r"Before: full relabelled posterior"),
        Line2D([0], [0], color=AFTER, lw=2.2,
               label=r"After: conditional posterior, $\tau_0=c_0+2w_0\geq0$"),
        Line2D([0], [0], color=BEFORE_DARK, lw=1.0, ls="--",
               label="Before weighted median"),
        Line2D([0], [0], color=AFTER, lw=1.25,
               label="After weighted median"),
    ]
    fig.legend(
        handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.965),
        frameon=False, ncol=2, fontsize=9.2,
    )
    source_name = str(config.get("source", source_dir.name))
    fig.suptitle(
        f"{source_name}: Gamma center and width before/after the positive-centroid filter",
        x=0.065, y=0.990, ha="left", fontsize=16.0, fontweight="bold",
    )
    fig.text(
        0.065, 0.963,
        r"Samples are first relabelled so comp0 has the shorter Gamma centroid. "
        r"The filter removes paired draws with $\tau_0=c_0+2w_0<0$; original nested-sampling weights are used.",
        ha="left", va="top", fontsize=9.5, color=MUTED,
    )
    fig.text(
        0.065, 0.018,
        r"Histograms show weighted probability per common bin (no smoothing). "
        r"The $c$ panels use a symmetric log-like display that is linear near zero; width panels use logarithmic x axes. "
        r"Here $c$ is the Gamma location/onset parameter, not its centroid.",
        ha="left", va="bottom", fontsize=8.8, color=MUTED,
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.075, top=0.925)

    output_dir = args.output_dir or (result_dir / "diagnostics")
    if not output_dir.is_absolute():
        output_dir = source_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{source_name}_tau0_filter_center_width_histograms"
    fig.savefig(stem.with_suffix(".png"), dpi=260, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    pd.DataFrame(summary_rows).to_csv(
        stem.with_name(stem.name + "_summary.csv"), index=False
    )
    plt.close(fig)

    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    print(stem.with_name(stem.name + "_summary.csv"))


if __name__ == "__main__":
    main()
