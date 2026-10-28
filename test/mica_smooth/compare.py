#!/usr/bin/env python3
"""Compare collected smoothing cases with the original two-component MICA fits.

This reads existing results only; it never starts MICA or collects unfinished runs.
Widths are the native scale parameter, not a common Gaussian/Gamma RMS width.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import gammainc, ndtr

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from collect_mica_results import (  # noqa: E402
    Q68, mica_run_root, ncomp_from_config, read_yaml, weighted_quantile,
)

CASES = ("baseline", "responses_s20", "responses_s40", "all_s20", "all_s40")
BANDS = ("UVM2", "UVW1", "U", "B", "V")
STAGES = {"gaussian": "mica_gaussian2_free", "gamma": "mica_gamma2_free"}
COMPONENTS = ("tf0", "tf1", "total")
METRICS = ("tau", "width", "negative_area_fraction")
QUANTITIES = [f"{m}{s}" for m in METRICS for s in ("", "_err_low", "_err_high")]
COLUMNS = ["case", "model", "band", "component", "lambda_obs", *QUANTITIES,
           "summary_path", "posterior_path"]
STATUS_COLUMNS = ["case", "model", "band", "status", "detail"]
POST_COLUMNS = ["band", "sample", "component", "tf_kind", "center", "width", "amp", "weight"]
# Equal-status experimental categories use the project's category palette.
COLORS = ("#1F4E9E", "#C43C39", "#E69F00", "#009E73", "#7A5195")
MARKERS = ("o", "s", "^", "D", "v")
NOTES = """Existing collected results only; missing bands/cases are listed in comparison_status.csv.
Component labels and tau/width summaries are retained from the main collector.
Tau is the component centroid in observer-frame days; no redshift conversion is applied here.
Width is the native MICA scale parameter in days: Gaussian standard deviation, Gamma k=2 scale.
The Gamma RMS width is sqrt(2) times its scale; the plotted scales are not a common RMS width.
Negative-lag area fraction integrates the response over tau<0, not negative response amplitude.
Per-draw fractions: Gaussian Phi(-center/width); Gamma P(2,max(-onset/width,0)).
Total fractions are amplitude-weighted across components within each matched posterior draw.
Negative-area summaries use the collector's weighted quantiles at 0.15865, 0.5, 0.84135;
error bars are the distances from the median to these posterior quantiles.
Nonfinite or nonpositive posterior weights do not contribute, as in weighted_quantile.
Total width is undefined and remains empty. Only tf0 and tf1 are plotted.
"""


def negative_area_statistics(posterior: pd.DataFrame, model: str) -> dict:
    """Return weighted component/total negative-area summaries for one band.

    Pair components by sample ID before forming totals. Ignore zero-weight rows
    before checking physical parameters, matching the collector's quantile rule.
    """
    frames = []
    for component in ("tf0", "tf1"):
        frame = posterior.loc[posterior["component"].eq(component)].copy()
        weights = frame["weight"].to_numpy(float)
        frame = frame.loc[np.isfinite(weights) & (weights > 0)].set_index("sample").sort_index()
        if frame.empty or not frame.index.is_unique:
            raise ValueError(f"{component}: no positive-weight draws or duplicate sample IDs")
        if not frame["tf_kind"].eq(model).all():
            raise ValueError(f"{component}: tf_kind does not match {model}")
        values = frame[["center", "width", "amp"]].to_numpy(float)
        if (not np.isfinite(values).all() or np.any(values[:, 1] <= 0)
                or np.any(values[:, 2] < 0)):
            raise ValueError(f"{component}: invalid positive-weight center/width/amplitude")
        frames.append(frame)
    if not frames[0].index.equals(frames[1].index):
        raise ValueError("tf0/tf1 posterior sample IDs do not match")
    weights = frames[0]["weight"].to_numpy(float)
    if not np.allclose(weights, frames[1]["weight"].to_numpy(float), rtol=1e-10, atol=0):
        raise ValueError("tf0/tf1 posterior weights do not match")

    fractions, amplitudes = [], []
    for frame in frames:
        z = -frame["center"].to_numpy(float) / frame["width"].to_numpy(float)
        if model == "gaussian":
            fractions.append(ndtr(z))
        elif model == "gamma":
            fractions.append(gammainc(2.0, np.maximum(z, 0.0)))
        else:
            raise ValueError(f"unsupported model: {model}")
        amplitudes.append(frame["amp"].to_numpy(float))
    amplitudes = np.asarray(amplitudes)
    amplitude_sum = amplitudes.sum(axis=0)
    if np.any(amplitude_sum <= 0) or not np.isfinite(amplitude_sum).all():
        raise ValueError("total response amplitude must be finite and positive")
    total = (amplitudes * np.asarray(fractions)).sum(axis=0) / amplitude_sum
    result = {}
    for component, values in zip(COMPONENTS, [*fractions, total]):
        lo, median, hi = weighted_quantile(values, weights, Q68)
        result[component] = {
            "negative_area_fraction": float(median),
            "negative_area_fraction_err_low": float(median - lo),
            "negative_area_fraction_err_high": float(hi - median),
        }
    return result


def collect_comparison(root: Path, baseline_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, statuses = [], []
    for case in CASES:
        source_dir = baseline_dir if case == "baseline" else root / "cases" / case
        config_path = source_dir / "config/source_config.yaml"
        cfg = read_yaml(config_path) if config_path.is_file() else None
        for model, stage in STAGES.items():
            def status(bands, state, detail):
                statuses.extend({"case": case, "model": model, "band": band,
                                 "status": state, "detail": str(detail)} for band in bands)
            if cfg is None:
                status(BANDS, "missing_config", config_path)
                continue
            fit = cfg.get(stage)
            if not fit:
                status(BANDS, "missing_stage", f"{config_path}: {stage}")
                continue
            if ncomp_from_config(fit) != 2 or fit.get("type_tf") != model:
                raise ValueError(f"{config_path}: {stage} must be two-component {model}")
            result_dir = mica_run_root(source_dir, fit) / "result"
            summary_path = result_dir / "mica_component_summary.csv"
            posterior_path = result_dir / "mica_component_posterior.txt"
            missing = [str(p) for p in (summary_path, posterior_path) if not p.is_file()]
            if missing:
                status(BANDS, "missing_results", "; ".join(missing))
                continue
            summary = pd.read_csv(summary_path)
            posterior = pd.read_csv(posterior_path, sep=r"\s+", usecols=POST_COLUMNS)
            if summary.duplicated(["band", "component"]).any():
                raise ValueError(f"duplicate band/component summaries: {summary_path}")
            for band in BANDS:
                band_summary = summary.loc[summary["band"].eq(band)].set_index("component")
                band_post = posterior.loc[posterior["band"].eq(band)]
                if not set(COMPONENTS).issubset(band_summary.index) or band_post.empty:
                    status([band], "missing_band", result_dir)
                    continue
                stats = negative_area_statistics(band_post, model)
                for component in COMPONENTS:
                    source = band_summary.loc[component]
                    row = {"case": case, "model": model, "band": band,
                           "component": component, "lambda_obs": source["lambda_obs"],
                           "summary_path": str(summary_path), "posterior_path": str(posterior_path)}
                    for metric in ("tau", "width"):
                        for suffix in ("", "_err_low", "_err_high"):
                            key = metric + suffix
                            row[key] = source[key]
                    row.update(stats[component])
                    rows.append(row)
                status([band], "available", result_dir)
    return pd.DataFrame(rows, columns=COLUMNS), pd.DataFrame(statuses, columns=STATUS_COLUMNS)


def plot_comparison(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, NullLocator

    paths = []
    for model in STAGES:
        data = frame.loc[frame["model"].eq(model)]
        if data.empty:
            continue
        with plt.style.context(PROJECT / "src/image-creator.mplstyle"):
            fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.6), sharex=True)
            handles, labels = [], []
            for row, component in enumerate(("tf0", "tf1")):
                for col, metric in enumerate(METRICS):
                    ax = axes[row, col]
                    for case, color, marker in zip(CASES, COLORS, MARKERS):
                        selected = data.loc[data["case"].eq(case) & data["component"].eq(component)]
                        if selected.empty:
                            continue
                        selected = selected.set_index("band").reindex(BANDS)
                        x = np.arange(len(BANDS))
                        y = selected[metric].to_numpy(float)
                        yerr = selected[[metric + "_err_low", metric + "_err_high"]].to_numpy(float).T
                        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#B8B8B8",
                                    elinewidth=0.55, alpha=0.55, capsize=0, zorder=1)
                        line, = ax.plot(x, y, color=color, marker=marker, ls="-", lw=1.2,
                                        ms=4.2, label=case, zorder=2)
                        if row == 0 and col == 0:
                            handles.append(line)
                            labels.append(case)
                    ax.set_xticks(np.arange(len(BANDS)), BANDS)
                    ax.xaxis.set_minor_locator(NullLocator())
                    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
                    ax.set_xlim(-0.25, len(BANDS) - 0.75)
                    if metric == "negative_area_fraction":
                        ax.set_ylim(-0.02, 1.02)
                    ax.set_ylabel((f"{component}\n" if col == 0 else "") + {
                        "tau": r"Centroid $\tau$ (d)", "width": "Scale width (d)",
                        "negative_area_fraction": "Negative-lag area fraction",
                    }[metric])
                    if row == 1:
                        ax.set_xlabel("Band")
            fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)),
                       bbox_to_anchor=(0.5, 1.0))
            fig.subplots_adjust(left=0.085, right=0.985, bottom=0.09, top=0.91,
                                wspace=0.32, hspace=0.16)
            for extension in ("png", "pdf"):
                path = output_dir / f"comparison_{model}.{extension}"
                fig.savefig(path, dpi=300)
                paths.append(path)
            plt.close(fig)
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--baseline-dir", type=Path, default=PROJECT / "Mrk817_a")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    output_dir = (args.output_dir or root / "results/comparison").expanduser().resolve()
    frame, statuses = collect_comparison(root, args.baseline_dir.expanduser().resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "comparison.csv", index=False, na_rep="")
    statuses.to_csv(output_dir / "comparison_status.csv", index=False)
    (output_dir / "comparison_notes.txt").write_text(NOTES, encoding="utf-8")
    for (case, model, state), group in statuses.groupby(["case", "model", "status"], sort=False):
        print(f"{case}/{model}: {state} [{', '.join(group['band'])}]")
    if not args.no_plots:
        for path in plot_comparison(frame, output_dir):
            print(path)
    print(f"{len(frame)} summary rows written to {output_dir / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
