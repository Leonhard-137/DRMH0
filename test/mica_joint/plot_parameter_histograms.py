#!/usr/bin/env python3
"""Plot the saved NGC2617_a joint/pairwise MICA posteriors; never run a fit.

Chart contract: compare the marginal posterior distributions of every free
parameter using all 10,000 saved rows and their normalized native weights.
The 30 response parameters are shown both by native label and after per-draw
centroid sorting; the two free shared DRW parameters are compared separately.
Six systematic-error slots are fixed at zero and are documented, not estimated.
Use common bins and posterior probability per bin, with no smoothing or cuts.
Blue solid/open fill denotes joint fitting; orange dashed denotes pairwise.
Export standalone PNG/PDF figures and numerical/source summaries to the sibling
figures_20260909 directory; inspect the exported images before delivery.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ngc2617_joint_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "NGC2617_a/runs/mica/gamma/broad_np4/2comp"
OUT = HERE / "figures_20260909"
BANDS = ["M2", "W1", "U", "B", "V"]
COLORS = ["#275D8A", "#C47728"]
KINDS = ["centroid", "log10_width", "log10_amplitude"]
LABELS = [r"Centroid $\tau$ [day]", r"$\log_{10}(w/\mathrm{day})$",
          r"$\log_{10} A$ [MICA normalized]"]
PARAM_ROWS, BIN_ROWS = [], []


def load_run(name):
    rd = BASE / name
    data = rd / "data"
    state = json.loads((rd / "run_mica_state.json").read_text())
    if state["status"] != "complete":
        raise ValueError(f"Run is not complete: {name}")
    sample = np.loadtxt(data / "sample1d.txt_2")
    raw_weight = np.loadtxt(data / "weights.txt")
    info = np.loadtxt(data / "sample_info1d.txt_2")
    posterior = np.loadtxt(data / "posterior_sample1d.txt_2")
    params = {}
    for line in (data / "para_names_line.txt_2").read_text().splitlines():
        fields = line.split()
        params[int(fields[0])] = {
            "name": fields[1], "coordinate": fields[2],
            "bounds": list(map(float, fields[3:5])), "fixed": int(fields[5]),
        }
    dimension = 38 if name.startswith("joint_") else 10
    assert sample.shape == (10000, dimension) and raw_weight.shape == (10000, 3)
    assert info.shape == (10000, 4) and posterior.shape[1] == dimension
    assert np.isfinite(sample).all() and np.isfinite(raw_weight).all()
    assert np.allclose(raw_weight[:, 1], info[:, 1], rtol=1e-6, atol=1e-6)
    weight = raw_weight[:, -1]
    assert np.all(weight >= 0) and weight.sum() > 0
    weight = weight / weight.sum()
    fixed = [index for index, p in params.items() if p["fixed"]]
    assert fixed == ([0, 3, 10, 17, 24, 31] if dimension == 38 else [0, 3])
    assert np.all(sample[:, fixed] == 0)
    metadata = {
        "run": str(rd), "raw_rows": len(sample), "parameter_slots": dimension,
        "free_parameters": dimension - len(fixed), "posterior_rows": len(posterior),
        "kish_ess": float(1 / np.sum(weight ** 2)),
        "entropy_ess": float(np.exp(-np.sum(weight[weight > 0] * np.log(weight[weight > 0])))),
        "fixed_zero_slots": fixed,
        "elapsed_seconds": state.get("last_attempt_seconds"),
    }
    return {"sample": sample, "weight": weight, "params": params, "meta": metadata}


def components(run, band, *, joint, ordered):
    start = 3 + (7 * BANDS.index(band) if joint else 0)
    p = run["params"]
    indices = [start + 1, start + 2, start + 3, start + 4, start + 5, start + 6]
    assert "centroid" in p[start + 2]["name"] and "centroid" in p[start + 5]["name"]
    triples = run["sample"][:, indices].reshape(-1, 2, 3).copy()
    swapped_mass = float(run["weight"][triples[:, 0, 1] > triples[:, 1, 1]].sum())
    if ordered:
        order = np.argsort(triples[:, :, 1], axis=1, kind="stable")
        triples = np.take_along_axis(triples, order[:, :, None], axis=1)
    # Stored centroid columns already are centroids: do not add 2*w.
    values = np.stack([triples[:, :, 1], triples[:, :, 2] / np.log(10),
                       triples[:, :, 0] / np.log(10)], axis=2)
    return values, swapped_mass


def quantiles(values, weights):
    keep = weights > 0
    values, weights = values[keep], weights[keep]
    order = np.argsort(values, kind="stable")
    return np.interp([0.16, 0.5, 0.84], np.cumsum(weights[order]), values[order])


def legend(fig, y):
    handles = [Line2D([0], [0], color=COLORS[0], lw=1.8, label="Joint: all six bands"),
               Line2D([0], [0], color=COLORS[1], lw=1.8, ls="--", label="Pairwise: W2 + this band")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=2, frameon=False, fontsize=10)


def histogram(ax, values, weights, edges, *, save_key=None):
    for method, value, weight, color, style in zip(
            ["joint", "pairwise"], values, weights, COLORS, ["-", "--"]):
        mass = np.histogram(value, bins=edges, weights=weight)[0]
        ax.stairs(100 * mass, edges, color=color, lw=1.6, ls=style)
        if method == "joint":
            ax.stairs(100 * mass, edges, color=color, lw=0, fill=True, alpha=0.09)
        if save_key is not None:
            band, component, parameter = save_key
            if not np.isclose(mass.sum(), 1, atol=1e-9):
                raise ValueError(f"Histogram omitted mass: {save_key} {mass.sum()}")
            q16, q50, q84 = quantiles(value, weight)
            PARAM_ROWS.append(dict(band=band, component=component, parameter=parameter,
                                   method=method, q16=q16, median=q50, q84=q84))
            for lo, hi, prob in zip(edges[:-1], edges[1:], mass):
                BIN_ROWS.append(dict(band=band, component=component, parameter=parameter,
                                     method=method, bin_low=lo, bin_high=hi, posterior_mass=prob))
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis="y", color="#E5E5E5", lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def response_edges(joint):
    bounds = joint["params"]
    return [np.linspace(-10.00001, 100.00001, 221),
            np.linspace(bounds[6]["bounds"][0] / np.log(10) - 1e-6,
                        bounds[6]["bounds"][1] / np.log(10) + 1e-6, 61),
            np.linspace(-5.000001, 1.000001, 61)]


def overview(joint, pairs, edges, *, ordered):
    fig, axes = plt.subplots(5, 6, figsize=(20, 13))
    for row, band in enumerate(BANDS):
        j, _ = components(joint, band, joint=True, ordered=ordered)
        s, _ = components(pairs[band], band, joint=False, ordered=ordered)
        for component in range(2):
            for kind in range(3):
                ax = axes[row, component * 3 + kind]
                histogram(ax, [j[:, component, kind], s[:, component, kind]],
                          [joint["weight"], pairs[band]["weight"]], edges[kind])
                if row == 0:
                    label = ["Smaller centroid", "Larger centroid"][component] if ordered else f"Native comp{component}"
                    ax.set_title(label, fontsize=11)
                ax.set_xlabel(LABELS[kind], fontsize=9)
        axes[row, 0].set_ylabel(f"{band}\nProbability / bin", fontsize=10)
    ordering = "components sorted by centroid in each draw" if ordered else "native component indices; label switching retained"
    fig.suptitle("NGC2617_a | Joint and pairwise parameter posteriors", fontsize=18, y=0.985)
    fig.text(0.5, 0.955, f"Broad two-Gamma priors | {ordering}", ha="center", fontsize=11)
    legend(fig, 0.937)
    fig.text(0.5, 0.016, "All 10,000 raw rows per run with native posterior weights; common bins; no cuts or smoothing. "
             "Widths/amplitudes use log10 coordinates. DRW parameters are on a separate page.", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.89, bottom=0.075, wspace=0.32, hspace=0.55)
    stem = "parameters_overview" if ordered else "parameters_native_labels"
    fig.savefig(OUT / f"{stem}.png", dpi=180)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 0.7,
                         "text.color": "#242424", "axes.labelcolor": "#242424", "pdf.fonttype": 42})
    joint = load_run("joint_W2_to_M2_W1_U_B_V_2comp_gamma")
    pairs = {band: load_run(f"run_W2_to_{band}_2comp_gamma") for band in BANDS}
    edges = response_edges(joint)
    metadata = {"joint": joint["meta"], "pairwise": {b: pairs[b]["meta"] for b in BANDS},
                "label_rule": "Sort the whole (ln amplitude, centroid, ln width) triple by centroid within each draw and response band.",
                "weights": "Normalize weights.txt final column; its logL column matches sample_info1d.txt_2 row by row.",
                "source_family": str(BASE), "native_label_swap_mass": {},
                "fixed_parameters": "Continuum and five response systematic-error slots are fixed at zero; no histogram is inferred for them."}
    with PdfPages(OUT / "parameter_histograms.pdf") as pdf:
        for band in BANDS:
            j, jswap = components(joint, band, joint=True, ordered=True)
            s, sswap = components(pairs[band], band, joint=False, ordered=True)
            metadata["native_label_swap_mass"][band] = {"joint": jswap, "pairwise": sswap}
            fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
            for component in range(2):
                for kind in range(3):
                    ax = axes[component, kind]
                    histogram(ax, [j[:, component, kind], s[:, component, kind]],
                              [joint["weight"], pairs[band]["weight"]], edges[kind],
                              save_key=(band, component, KINDS[kind]))
                    ax.set_xlabel(LABELS[kind])
                    if kind == 0:
                        ax.set_ylabel(["Smaller-centroid component", "Larger-centroid component"][component] + "\nProbability / bin")
            fig.suptitle(f"NGC2617_a | {band} response parameters", y=0.98, fontsize=17)
            fig.text(0.5, 0.925, "Broad two-Gamma priors | components sorted by centroid within each posterior draw", ha="center", fontsize=10)
            legend(fig, 0.905)
            fig.text(0.5, 0.035, f"10,000 raw rows/run; weighted Kish ESS: joint {joint['meta']['kish_ess']:.0f}, pairwise {pairs[band]['meta']['kish_ess']:.0f}. "
                     "Full prior ranges; no cuts or smoothing.", ha="center", fontsize=9)
            fig.subplots_adjust(left=0.09, right=0.985, top=0.82, bottom=0.12, hspace=0.4, wspace=0.3)
            fig.savefig(OUT / f"parameters_{band}.png", dpi=180)
            pdf.savefig(fig)
            plt.close(fig)

        fig, axes = plt.subplots(5, 2, figsize=(10, 13))
        for k, index in enumerate([1, 2]):
            lo = min(r["params"][index]["bounds"][0] for r in [joint, *pairs.values()]) - 1e-6
            hi = max(r["params"][index]["bounds"][1] for r in [joint, *pairs.values()]) + 1e-6
            bins = np.linspace(lo, hi, 71)
            for row, band in enumerate(BANDS):
                ax = axes[row, k]
                histogram(ax, [joint["sample"][:, index], pairs[band]["sample"][:, index]],
                          [joint["weight"], pairs[band]["weight"]], bins,
                          save_key=(band, "shared_driver", ["ln_sigmad", "ln_taud"][k]))
                ax.set_xlabel([r"$\ln(\mathrm{sigmad})$ [MICA coordinate]", r"$\ln(\tau_d/\mathrm{day})$"][k])
                if k == 0:
                    ax.set_ylabel(f"{band} pair\nProbability / bin")
        fig.suptitle("NGC2617_a | Shared-driver DRW parameters", y=0.985, fontsize=17)
        fig.text(0.5, 0.954, "The same joint posterior is repeated against each pairwise fit", ha="center")
        legend(fig, 0.938)
        fig.text(0.5, 0.022, "All weighted raw samples; common bins. Each fit used its own W2 prefit.\n"
                 "Continuum and response systematic-error parameters are fixed at zero.", ha="center", fontsize=10)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.89, bottom=0.09, hspace=0.6, wspace=0.28)
        fig.savefig(OUT / "drw_parameters.png", dpi=180)
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
        bins = np.linspace(-2, 4, 121)
        for band, ax in zip(BANDS, axes.flat):
            j, _ = components(joint, band, joint=True, ordered=True)
            s, _ = components(pairs[band], band, joint=False, ordered=True)
            histogram(ax, [j[:, 0, 0], s[:, 0, 0]], [joint["weight"], pairs[band]["weight"]], bins)
            jm = joint["weight"][(j[:, 0, 0] >= -2) & (j[:, 0, 0] <= 4)].sum()
            sm = pairs[band]["weight"][(s[:, 0, 0] >= -2) & (s[:, 0, 0] <= 4)].sum()
            ax.set_title(f"{band} | visible mass J {jm:.1%}, P {sm:.1%}", fontsize=10)
            ax.set_xlabel(r"Smaller centroid $\tau$ [day]")
            ax.set_ylabel("Probability / bin")
        axes.flat[-1].axis("off")
        fig.suptitle("NGC2617_a | Smaller-centroid detail", y=0.985, fontsize=17)
        fig.text(0.5, 0.931, "Zoom to -2 to 4 days; probabilities retain the full posterior normalization", ha="center")
        legend(fig, 0.91)
        fig.text(0.5, 0.033, "The full -10 to 100 day prior is shown on each band page. J = joint; P = pairwise. No posterior filtering.", ha="center", fontsize=9)
        fig.subplots_adjust(left=0.075, right=0.98, top=0.79, bottom=0.12, hspace=0.52, wspace=0.32)
        fig.savefig(OUT / "centroid_detail.png", dpi=180)
        pdf.savefig(fig)
        plt.close(fig)

    overview(joint, pairs, edges, ordered=True)
    overview(joint, pairs, edges, ordered=False)
    for filename, rows in [("parameter_summary.csv", PARAM_ROWS), ("histogram_bins.csv", BIN_ROWS)]:
        with (OUT / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (OUT / "posterior_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Saved all response and DRW posterior histograms:", OUT)
    print("Joint weight ESS:", metadata["joint"]["kish_ess"])


if __name__ == "__main__":
    main()
