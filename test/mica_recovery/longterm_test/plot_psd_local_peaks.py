#!/usr/bin/env python3
"""Plot existing local-PSD-peak checks; never simulate, select, or fit a model.

Consumes local_peak_check/summary.json, examples/{candidate_id}.npz, and
candidate_rates.csv. Only the raw spectra are drawn. Captions document the
deliberate selection of example realizations separately from ordinary draws.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import COLORS, SHORT_DASH, configure_style, style_axis

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "psd_match_817/local_peak_check"
BANDS = ("W2", "U")


def save(fig, stem):
    outputs = []
    for suffix in ("png", "pdf"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def read_examples(directory, candidate_ids=None):
    summary = json.loads((directory / "summary.json").read_text())
    if candidate_ids:
        available = {item['candidate_id']: item for item in summary['confirmation']}
        selected = [available[candidate] for candidate in candidate_ids]
    else:
        selected = summary.get("selected_examples", [])[:3]
    if not selected:
        raise ValueError("summary.json has no selected_examples; nothing to display")
    examples = []
    for item in selected:
        candidate = str(item["candidate_id"])
        if Path(candidate).name != candidate:
            raise ValueError(f"Invalid candidate identifier: {candidate}")
        with np.load(directory / "examples" / f"{candidate}.npz") as saved:
            payload = {name: np.asarray(saved[name], dtype=float) for name in
                       ("frequency", "observed", "example", "median", "first_draws", "params")}
        for name in ("frequency", "observed", "example", "median"):
            values = payload[name]
            if values.shape != (2, 23) or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f"{candidate}/{name}: expected finite positive (2,23) array")
        background = payload["first_draws"]
        if background.ndim != 3 or background.shape[1:] != (2, 23) or not np.isfinite(background).all() or np.any(background <= 0):
            raise ValueError(f"{candidate}: invalid first_draws")
        if payload["params"].shape != (6,):
            raise ValueError(f"{candidate}: invalid params")
        if not np.all(np.diff(payload["frequency"], axis=1) > 0):
            raise ValueError(f"{candidate}: frequencies must increase")
        np.testing.assert_allclose(payload["params"][:2], [item["tauI"], item["tauS"]], rtol=1e-10)
        examples.append((item, payload))
    return summary, examples


def plot_examples(directory, examples, output_stem="local_peak_examples"):
    nrows = len(examples)
    fig, axes = plt.subplots(nrows, 2, figsize=(11.4, 3.2*nrows+.45),
                             sharex=True, sharey=True, squeeze=False)
    all_values = np.concatenate([payload[name].ravel() for _, payload in examples
                                for name in ("observed", "example", "median", "first_draws")])
    ylims = (all_values.min()/1.4, all_values.max()*1.5)
    for row, (item, payload) in enumerate(examples):
        tau_i, tau_s = payload["params"][:2]
        for col, band in enumerate(BANDS):
            ax, freq = axes[row, col], payload["frequency"][col]
            for index, draw in enumerate(payload["first_draws"][:8, col, :]):
                ax.loglog(freq, draw, color=COLORS["GUIDE"], alpha=.52, linewidth=.65,
                          label="First 8 draws" if index == 0 else None, zorder=1)
            ax.loglog(freq, payload["median"][col], color=COLORS["ALTERNATIVE"],
                      linestyle=SHORT_DASH, linewidth=1.55, label="Ensemble median", zorder=3)
            ax.loglog(freq, payload["example"][col], color=COLORS["PRIMARY"],
                      linewidth=1.55, label="Selected draw", zorder=4)
            ax.loglog(freq, payload["observed"][col], color=COLORS["INK"],
                      linewidth=1.55, marker="o", markersize=2.6,
                      markeredgewidth=0, label="Observed", zorder=5)
            panel = chr(ord("a")+2*row+col)
            ax.set_title(rf"({panel}) {item['candidate_id']} / {band}: $\tau_I={tau_i:.3g}$ d, $\tau_S={tau_s:.3g}$ d",
                         loc="left", fontsize=12)
            ax.set_xlim(freq[0]*.93, freq[-1]*1.06)
            ax.set_ylim(*ylims)
            if col == 0:
                ax.set_ylabel(r"$P(f)$ [fractional flux$^2$ day]")
            if row == nrows-1:
                ax.set_xlabel(r"Frequency [day$^{-1}$]")
            style_axis(ax, y_on_right=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [labels.index(label) for label in ("Observed", "Selected draw", "Ensemble median", "First 8 draws")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.52, 1.01), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .975))
    return save(fig, directory / output_stem)


def read_rates(directory):
    path = directory / "candidate_rates.csv"
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"tauI", "tauS", "joint_similar_rate"}
    if rows and not required <= rows[0].keys():
        raise ValueError(f"candidate_rates.csv requires {sorted(required)}")
    for row in rows:
        tau_i, tau_s, rate = (float(row[key]) for key in ("tauI", "tauS", "joint_similar_rate"))
        if not np.isfinite([tau_i, tau_s, rate]).all() or min(tau_i, tau_s) <= 0 or not 0 <= rate <= 1:
            raise ValueError("Candidate time scales/rates are invalid")
    return rows


def plot_rates(directory, rows):
    if not rows:
        return []
    x = np.array([float(row["tauI"]) for row in rows])
    y = np.array([float(row["tauS"]) for row in rows])
    percentages = 100*np.array([float(row["joint_similar_rate"]) for row in rows])
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    points = ax.scatter(x, y, c=percentages, cmap="viridis", vmin=0,
                        vmax=max(float(percentages.max()), .1),
                        s=45, edgecolors=COLORS["INK"], linewidths=.35, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau_I$ [day]")
    ax.set_ylabel(r"$\tau_S$ [day]")
    style_axis(ax, y_on_right=False)
    bar = fig.colorbar(points, ax=ax, pad=.035)
    bar.set_label("Joint similar-realization rate [%]")
    bar.ax.tick_params(direction="in")
    fig.tight_layout()
    return save(fig, directory / "local_peak_rates")


def write_caption(directory, summary, examples, rates, output_stem="local_peak_examples"):
    index = {str(row.get("candidate_id", "")): row for row in rates}
    lines = ["# Local PSD peak check: figure captions", "",
        f"## {output_stem}.png / .pdf", "",
        "Each row compares one saved candidate in W2 (left) and U (right). Black: observed "
        "binned PSD. Blue: a deliberately selected single Monte Carlo realization. Red short "
        "dashes: the pointwise ensemble median. Pale grey: the first eight ordinary saved draws "
        "in their original order. All panels show raw, unfiltered spectra on the same PSD scale.", "",
        "**Selection disclosure:** The blue curves were intentionally chosen from the simulated "
        "ensemble by the upstream local-peak check. They are illustrative matches, not random "
        "or necessarily representative realizations. Their visual agreement must not be used "
        "to infer a match probability. Candidate frequencies are reported separately in "
        "candidate_rates.csv and the rate figure; the exact local-peak rule is recorded in "
        "definition.json. Screening uses 500 realizations per candidate. The displayed blue "
        "examples, red medians and first-eight grey draws come from a separate 2000-realization "
        "confirmation ensemble with a new seed. The example was chosen for low whole-spectrum "
        "mismatch among confirmation draws passing the joint local-peak rule. This plotter "
        "performs no new selection.", "",
        "PSD estimator and normalization are inherited unchanged from the supplied arrays: "
        "unweighted floating-mean Lomb-Scargle × 2T/N, with arithmetic averaging within 23 "
        "logarithmic frequency bins, in fractional-flux units. No extra smoothing or normalization "
        "is applied by this plotting script.", "", "### Rows", ""]
    for rownum, (item, payload) in enumerate(examples, 1):
        candidate = str(item["candidate_id"])
        rate = index.get(candidate, {})
        nreal = item.get("nreal", "see summary.json")
        lines.append(f"{rownum}. `{candidate}`; role `{item.get('role', '')}`; "
                     f"tauI={payload['params'][0]:.8g} d, tauS={payload['params'][1]:.8g} d; "
                     f"selected zero-based realization index={item['example_index0']}; "
                     f"ensemble size={nreal}.")
    lines.extend(["", "## local_peak_rates.png / .pdf", "",
        "Each point is an evaluated candidate, placed by its central-driver and independent-slow "
        "time scales. Color gives the saved joint similar-realization rate as a percentage. "
        "Rates and their statistical definition are read from the upstream check, not recomputed "
        "from the selected example curves. These are 500-draw screening rates, distinct from "
        "the 2000-draw confirmation rates. Coincident parameter locations may overlap. The "
        "feature and tolerance were chosen after inspecting the observations: rates are "
        "model-conditional replication fractions, not p-values or posterior probabilities.", ""])
    caption_name = "figure_captions.md" if output_stem == "local_peak_examples" else f"{output_stem}_caption.md"
    path = directory / caption_name
    path.write_text("\n".join(lines))
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--candidates", nargs="+", help="Previously confirmed candidate IDs, in row order")
    parser.add_argument("--output-stem", help="Filename stem for this example figure")
    args = parser.parse_args()
    directory = args.input_dir.resolve()
    summary, examples = read_examples(directory, args.candidates)
    rates = read_rates(directory)
    output_stem = args.output_stem or ("local_peak_examples_" + "_".join(args.candidates)
                                      if args.candidates else "local_peak_examples")
    configure_style()
    files = plot_examples(directory, examples, output_stem)
    if not args.candidates:
        files += plot_rates(directory, rates)
    files.append(write_caption(directory, summary, examples, rates, output_stem))
    print(json.dumps(dict(selected_examples=len(examples), candidates=len(rates), files=files), indent=2))


if __name__ == "__main__":
    main()
