#!/usr/bin/env python3
"""Summarize the eight F9 injection fits without discarding posterior modes.

Chart contract: compare the two signal origins at fixed cadence, baseline,
noise and slow/disk RMS. Points and bars are weighted posterior medians and
central 68.27% intervals; diamonds are injected moment/gain references.
Grey whiskers show native prior bounds, not credible intervals. All time
quantities are observer-frame days. Component labels mean smaller/larger
centroid per joint draw; they are not a demonstrated physical identification.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from collect_mica_results import (  # noqa: E402
    component_parameter_arrays,
    mica_flux_scales,
    normalized_weights,
    param_names,
    reorder_component_columns,
    weighted_quantile,
)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BANDS = ("M2", "B")
CASE_ORDER = ("response_within", "random_within", "response_beyond", "random_beyond")
COLORS = {"response": "#2457A6", "random": "#C67C18"}
EDGE_FRACTION = .05
Q = (.15865, .5, .84135)


def interval(values, weights):
    low, mid, high = weighted_quantile(values, weights, Q)
    return {"q15865": float(low), "median": float(mid), "q84135": float(high)}


def native_parameter_metadata(path):
    result = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 5 or not fields[0].isdigit():
            continue
        result.append({"index": int(fields[0]), "name": fields[1],
                       "native_prior_type": fields[2],
                       "native_low": float(fields[3]), "native_high": float(fields[4]),
                       "fixed": bool(int(fields[5])) if len(fields) > 5 else False})
    if [row["index"] for row in result] != list(range(len(result))):
        raise ValueError(f"Nonconsecutive native parameter indices: {path}")
    return result


def edge_screen(values, weights, lower, upper):
    """Screen in the sampled coordinate (centroid or logarithm), without cuts."""
    if not upper > lower:
        return None
    scaled = (np.asarray(values) - lower) / (upper - lower)
    return {"fraction_of_prior_span": EDGE_FRACTION,
            "lower_mass": float(weights[scaled <= EDGE_FRACTION].sum()),
            "upper_mass": float(weights[scaled >= 1-EDGE_FRACTION].sum()),
            # Small tolerance accommodates rounded native parameter exports.
            "outside_mass": float(weights[(scaled < -1e-6) | (scaled > 1+1e-6)].sum())}


def mixture_moments(centroids, widths, amplitudes):
    """Exact moments of full-support, shape-2 Gamma mixtures per joint draw."""
    total_area = amplitudes.sum(axis=0)
    fraction = amplitudes / total_area
    centroid = np.sum(fraction * centroids, axis=0)
    variance = np.sum(fraction * (2*widths**2 + (centroids-centroid)**2), axis=0)
    return centroid, variance, total_area, fraction


def file_metadata(paths):
    return {str(path): {"bytes": path.stat().st_size,
                       "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
            for path in paths}


def summarize_fit(case, band):
    case_root = Path(case["path"])
    if not case_root.resolve().is_relative_to(HERE):
        raise ValueError(f"Case outside this experiment: {case_root}")
    run = case_root / f"runs/mica/2comp/run_W2_to_{band}_2comp_gamma"
    data = run / "data"
    paths = {"sample": data / "sample1d.txt_2", "weights": data / "weights.txt",
             "names": data / "para_names_line.txt_2",
             "posterior": data / "posterior_sample1d.txt_2",
             "driver": run / "input/cont.txt", "target": run / "input/line.txt"}
    result = {"case": case["name"], "origin": case["origin"],
              "timescale": case["timescale"], "band": band,
              "run": str(run), "truth": case["bands"][band]}
    missing = [str(path) for path in paths.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        return {**result, "status": "missing_or_incomplete", "missing": missing}
    try:
        sample = np.loadtxt(paths["sample"], ndmin=2)
        table = np.loadtxt(paths["weights"], ndmin=2)
        if len(sample) != len(table):
            raise ValueError(f"Raw sample/weight row mismatch: {len(sample)} != {len(table)}")
        weights = normalized_weights(table[:, -1])
        if not np.isfinite(sample[weights > 0]).all():
            raise ValueError("Nonfinite parameter in a positive-weight raw sample")
        names = param_names(paths["names"])
        native = native_parameter_metadata(paths["names"])
        if sample.shape[1] != len(names) or len(native) != len(names):
            raise ValueError("Native parameter names and sample columns do not match")
        meta = {row["name"]: row for row in native}
        # The prepared experiment has identical priors for the exchangeable
        # components. This makes bounds meaningful after whole-tuple relabeling.
        for quantity in ("amplitude", "centroid", "sigma"):
            a = meta[f"0-th_component_{quantity}"]
            b = meta[f"1-th_component_{quantity}"]
            if not np.allclose([a["native_low"], a["native_high"]],
                               [b["native_low"], b["native_high"]], rtol=1e-7, atol=1e-6):
                raise ValueError("Different component priors require row-specific relabeled bounds")
        native_edges = []
        for item in native:
            if "_component_" not in item["name"] or item["fixed"]:
                continue
            coordinate = "linear_centroid_days" if item["name"].endswith("centroid") else "natural_log"
            native_edges.append({"parameter": item["name"], "coordinate": coordinate,
                                 "native_low": item["native_low"], "native_high": item["native_high"],
                                 **edge_screen(sample[:, item["index"]], weights,
                                               item["native_low"], item["native_high"])})
        ordered, swapped = reorder_component_columns(sample, names, 2, "gamma", {"component_order": "moment1"})
        centers, widths, amplitudes, centroids = component_parameter_arrays(ordered, names, 2, "gamma")
        if not all(np.isfinite(x[weights > 0]).all() for x in (*centers, *widths, *amplitudes, *centroids)):
            raise ValueError("Nonfinite transformed posterior parameter")
        driver_scale, target_scale, normalization_mode = mica_flux_scales(run)
        gain_scale = target_scale / driver_scale
        total_centroid, total_variance, total_area, fraction = mixture_moments(centroids, widths, amplitudes)
        components = []
        for k in range(2):
            parameters = {"centroid_days": centroids[k], "onset_days": centers[k],
                          "width_days": widths[k], "native_amplitude": amplitudes[k],
                          "input_flux_gain": amplitudes[k]*gain_scale,
                          "area_fraction": fraction[k]}
            prior = {}
            sorted_edges = {}
            for key, suffix, coordinate_values in (
                ("centroid_days", "centroid", centroids[k]),
                ("width_days", "sigma", np.log(widths[k])),
                ("native_amplitude", "amplitude", np.log(amplitudes[k])),
            ):
                bounds = meta[f"{k}-th_component_{suffix}"]
                lo, hi = bounds["native_low"], bounds["native_high"]
                sorted_edges[key] = edge_screen(coordinate_values, weights, lo, hi)
                prior[key] = [float(lo), float(hi)] if suffix == "centroid" else [float(np.exp(lo)), float(np.exp(hi))]
            prior["input_flux_gain"] = [v*gain_scale for v in prior["native_amplitude"]]
            components.append({"label": "smaller_centroid" if k == 0 else "larger_centroid",
                               "parameters": {key: interval(value, weights) for key, value in parameters.items()},
                               "prior_bounds": prior, "near_prior_edges": sorted_edges})
        positive = weights > 0
        diagnostics = {"raw_samples": len(sample), "positive_weight_samples": int(positive.sum()),
                       "importance_weight_ess": float(1/np.sum(weights**2)),
                       "weight_entropy_ess": float(np.exp(-np.sum(weights[positive]*np.log(weights[positive])))),
                       "resampled_posterior_rows": int(np.loadtxt(paths["posterior"], ndmin=2).shape[0]),
                       "label_swap_posterior_mass": float(weights[swapped].sum()),
                       "near_prior_edges_native_labels": native_edges,
                       "retained_posterior_mass": 1.0}
        info_path = data / "sample_info1d.txt_2"
        if info_path.is_file():
            info = np.loadtxt(info_path, ndmin=2)
            if len(info) != len(sample) or info.shape[1] < 4:
                raise ValueError("Raw sample_info and sample rows do not align")
            if table.shape[1] >= 3 and not np.allclose(table[:, 1], info[:, 1], rtol=1e-6, atol=1e-5):
                raise ValueError("weights/sample_info log likelihood columns do not align")
            diagnostics["log_likelihood"] = interval(info[:, 1], weights)
            diagnostics["maximum_log_likelihood"] = float(np.max(info[:, 1]))
            diagnostics["posterior_mass_by_native_particle_id"] = {
                str(int(pid)): float(weights[info[:, 3] == pid].sum()) for pid in np.unique(info[:, 3])}
            paths["sample_info"] = info_path
        truth = result["truth"]
        comparison = {"smaller_centroid_minus_injected_short_moment_days": interval(
            centroids[0] - truth["short_relative_centroid_days"], weights)}
        if case["origin"] == "response":
            comparison.update({
                "larger_centroid_minus_injected_slow_centroid_days": interval(
                    centroids[1] - truth["injected_slow_centroid_days"], weights),
                "larger_width_minus_injected_slow_width_days": interval(
                    widths[1] - truth["injected_slow_width_days"], weights),
                "total_centroid_minus_injected_total_moment_days": interval(
                    total_centroid - truth["injected_total_relative_centroid_days"], weights),
            })
        warnings = []
        if diagnostics["importance_weight_ess"] < 100:
            warnings.append("Importance-weight ESS below 100; this is a sampling-resolution screen, not a convergence test.")
        if any(max(row["lower_mass"], row["upper_mass"]) > .1 for row in native_edges):
            warnings.append("At least one native parameter has >10% posterior mass in an outer 5% prior strip; inspect prior sensitivity.")
        if any(row["outside_mass"] > 1e-8 for row in native_edges):
            warnings.append("Positive posterior mass lies outside printed native bounds beyond rounding tolerance.")
        return {**result, "status": "analyzed", "files": file_metadata(paths.values()),
                "normalization": {"mode": normalization_mode, "driver_scale": driver_scale,
                                  "target_scale": target_scale, "native_amplitude_to_input_gain": gain_scale},
                "native_parameters": native, "components": components,
                "total": {"centroid_days": interval(total_centroid, weights),
                          "variance_days2": interval(total_variance, weights),
                          "standard_deviation_days": interval(np.sqrt(total_variance), weights),
                          "native_amplitude": interval(total_area, weights),
                          "input_flux_gain": interval(total_area*gain_scale, weights)},
                "comparison": comparison, "diagnostics": diagnostics, "warnings": warnings}
    except (OSError, ValueError, KeyError, IndexError, FloatingPointError) as exc:
        return {**result, "status": "invalid_or_incomplete", "error": f"{type(exc).__name__}: {exc}"}


def write_csv(path, rows):
    if not rows:
        path.write_text("case,band,status\n")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def tabular_rows(results):
    parameters, edges = [], []
    for result in results:
        identity = {key: result[key] for key in ("case", "origin", "timescale", "band", "status")}
        if result["status"] != "analyzed":
            parameters.append({**identity, "error": result.get("error", "; ".join(result.get("missing", [])))})
            continue
        for k, component in enumerate(result["components"]):
            for parameter, value in component["parameters"].items():
                prior = component["prior_bounds"].get(parameter)
                parameters.append({**identity, "component": component["label"], "parameter": parameter,
                                   **value, "prior_low": prior[0] if prior else None,
                                   "prior_high": prior[1] if prior else None,
                                   "importance_weight_ess": result["diagnostics"]["importance_weight_ess"]})
            for parameter, value in component["near_prior_edges"].items():
                edges.append({**identity, "component": component["label"], "parameter": parameter,
                              "coordinate": "linear" if parameter == "centroid_days" else "natural_log", **value})
        for parameter, value in result["total"].items():
            parameters.append({**identity, "component": "total", "parameter": parameter, **value,
                               "importance_weight_ess": result["diagnostics"]["importance_weight_ess"]})
    return parameters, edges


def injected_reference(result, component_index, parameter):
    truth = result["truth"]
    if component_index == 0:
        if parameter == "centroid_days":
            return truth["short_relative_centroid_days"]
        if parameter == "input_flux_gain":
            return 1.0
    if result["origin"] != "response":
        return None
    if component_index == 1:
        return {"centroid_days": truth["injected_slow_centroid_days"],
                "width_days": truth["injected_slow_width_days"],
                "input_flux_gain": truth["slow_scale"]}.get(parameter)
    if component_index is None:
        if parameter == "centroid_days":
            return truth["injected_total_relative_centroid_days"]
        if parameter == "standard_deviation_days":
            variance = truth["injected_total_relative_second_cumulant_days2"]
            return np.sqrt(variance) if variance is not None and variance >= 0 else None
    return None


def plot_results(results, truth, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    panels = ((0, "centroid_days", "Smaller centroid (day)", "symlog"),
              (1, "centroid_days", "Larger centroid (day)", "symlog"),
              (0, "width_days", "Smaller-centroid component width (day)", "log"),
              (1, "width_days", "Larger-centroid component width (day)", "log"),
              (0, "input_flux_gain", "Smaller-centroid component area (input flux)", "log"),
              (1, "input_flux_gain", "Larger-centroid component area (input flux)", "log"),
              (None, "centroid_days", "Total area-weighted centroid (day)", "symlog"),
              (None, "standard_deviation_days", "Total kernel standard deviation (day)", "log"))
    for band in BANDS:
        available = {r["case"]: r for r in results if r["band"] == band and r["status"] == "analyzed"}
        fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharex=True, layout="constrained")
        for ax, (index, parameter, label, scale) in zip(axes.flat, panels):
            if scale == "symlog":
                ax.set_yscale(scale, linthresh=1)
            else:
                ax.set_yscale(scale)
            for x, case in enumerate(CASE_ORDER):
                if case not in available:
                    ax.text(x, .5, "pending", transform=ax.get_xaxis_transform(), ha="center", color=".5", fontsize=8)
                    continue
                result = available[case]
                if index is None:
                    value = result["total"][parameter]
                    prior = None
                else:
                    component = result["components"][index]
                    value = component["parameters"][parameter]
                    prior = component["prior_bounds"].get(parameter)
                if prior:
                    ax.plot([x, x], prior, color=".80", lw=1, zorder=1)
                    ax.plot([x, x], prior, linestyle="none", marker="_", color=".65", markersize=9, zorder=1)
                mid, low, high = value["median"], value["q15865"], value["q84135"]
                ax.errorbar(x-.06, mid, yerr=[[max(0, mid-low)], [max(0, high-mid)]],
                            fmt="o", ms=5, color=COLORS[result["origin"]], capsize=4, zorder=3)
                reference = injected_reference(result, index, parameter)
                if reference is not None:
                    ax.scatter(x+.08, reference, marker="D", s=24, facecolor="none", edgecolor="black", zorder=4)
            ax.set_title(label, loc="left", fontsize=10)
            ax.grid(axis="y", color=".92", lw=.6)
            ax.set_xlim(-.45, 3.45)
            ax.set_xticks(range(4), ["within\nresponse", "within\nindependent", "beyond\nresponse", "beyond\nindependent"])
        legend = [Line2D([0], [0], marker="o", color=COLORS["response"], lw=1, label="Response posterior: median + 68.27%"),
                  Line2D([0], [0], marker="o", color=COLORS["random"], lw=1, label="Independent posterior: median + 68.27%"),
                  Line2D([0], [0], marker="D", markerfacecolor="none", color="black", lw=0, label="Injected moment / area reference"),
                  Line2D([0], [0], marker="_", color=".65", lw=1, label="Native prior range")]
        fig.legend(handles=legend, loc="outside lower center", ncol=2, fontsize=8)
        fig.suptitle(f"F9 {band}: slow-signal recovery, W2 driver — {len(available)}/4 fits analyzed\n"
                     f"Observer frame; actual F9 sampling, {truth['baseline_days']:.1f} d baseline; one realization\n"
                     "Components ordered by centroid, without posterior cuts; short reference is a disk moment, not a Gamma-fit truth",
                     fontsize=11)
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"F9_{band}_recovery_parameters.{suffix}", dpi=180)
        plt.close(fig)


def format_interval(value):
    return f"{value['median']:.4g} [{value['q15865']:.4g}, {value['q84135']:.4g}]"


def write_markdown(summary, out):
    complete = summary["n_analyzed"]
    lines = ["# F9 M2/B slow-signal injection recovery", "",
             f"Status: **{complete}/8 fits analyzed**. Generated {summary['generated_utc']}.", "",
             "Values below are weighted medians [15.865%, 84.135%]. Time quantities are observer-frame days. "
             "All joint raw samples retain their posterior weight; no branch selection or prior-based rejection was applied.", "",
             "| Band | Case | Smaller centroid | Larger centroid | Larger width | Total centroid | Weight ESS |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for result in summary["results"]:
        if result["status"] == "analyzed":
            small, large = result["components"]
            values = [format_interval(small["parameters"]["centroid_days"]),
                      format_interval(large["parameters"]["centroid_days"]),
                      format_interval(large["parameters"]["width_days"]),
                      format_interval(result["total"]["centroid_days"]),
                      f"{result['diagnostics']['importance_weight_ess']:.1f}"]
            lines.append(f"| {result['band']} | {result['case']} | " + " | ".join(values) + " |")
        else:
            lines.append(f"| {result['band']} | {result['case']} | {result['status']} | — | — | — | — |")
    lines += ["", "## Interpretation and limits", "",
              "- The two components are ordered by centroid within each complete joint draw. Calling the smaller one a recovered disk is an identification hypothesis, not a consequence of this relabeling.",
              "- The injected short-reference centroid is the absolute band-kernel centroid minus the W2-kernel centroid. The finite-difference disk transfer ratio is not assumed to be an exact positive two-Gamma kernel. There is no exact injected Gamma width for the disk component.",
              "- True-response cases have a known slow Gamma component and formal total relative moments. Independent-signal cases have no injected total transfer function or total lag. A fitted slow component there demonstrates possible misattribution in this realization, not a measured false-positive probability.",
              "- Broad configured priors are centroid −10 to 2000 d and width 0.01 to 1200 d for both components; the tables and charts use each run's native exported bounds. Width/amplitude edge fractions are measured in log coordinates, centroid edges in linear coordinates. An outer-5%-strip mass is a screen, not evidence by itself of truncation or a nonphysical fit.",
              "- Importance-weight ESS describes concentration of saved-sample weights. Neither high ESS, multiple particles, nor 10000 saved samples establishes convergence or coverage of all posterior modes. Native particle IDs are not independent chains.",
              "- Native amplitudes describe MICA-normalized curves. Input-flux gain is native amplitude times target normalization divided by driver normalization; only this restored gain is compared with injected area. Total centroid and variance do not change under this common gain factor.",
              "- Total moments are calculated analytically for each full-support Gamma mixture and then summarized. They never come from integrating the standard −100 to 500 d exported plot, which truncates the long-response support.",
              "- Both slow-timescale tiers retain the same F9 cadence and baseline. The distinction concerns the amount of baseline covering the slow variability; it is not a strict Nyquist test for irregular observations.",
              "- This is one paired stochastic realization at one slow/disk RMS. Quantities outside the data-supported frequency range remain model/prior dependent.", "",
              "## Injected references", "",
              "| Band | Case | Short relative centroid | Slow centroid | Slow width | Total relative centroid |",
              "|---|---|---:|---:|---:|---:|"]
    for result in summary["results"]:
        truth = result["truth"]
        values = [truth["short_relative_centroid_days"], truth["injected_slow_centroid_days"],
                  truth["injected_slow_width_days"], truth["injected_total_relative_centroid_days"]]
        lines.append(f"| {result['band']} | {result['case']} | " + " | ".join("not defined" if v is None else f"{v:.6g}" for v in values) + " |")
    lines += ["", "## Screens and incomplete outputs", ""]
    alerts = []
    for result in summary["results"]:
        prefix = f"{result['case']}/{result['band']}"
        if result["status"] != "analyzed":
            alerts.append(f"- {prefix}: {result.get('error', result.get('missing'))}")
        for warning in result.get("warnings", []):
            alerts.append(f"- {prefix}: {warning}")
    lines += alerts or ["No defined screening flag was triggered. This does not establish convergence or a physical interpretation."]
    lines += ["", "## Files", "", "- `recovery_summary.json`: complete provenance, native priors, weighted intervals, truth and diagnostics.",
              "- `recovery_parameters.csv`: component/total intervals and physical prior bounds.",
              "- `recovery_prior_edges.csv`: sorted-component prior-edge mass in the native sampling coordinate.",
              "- `F9_M2_recovery_parameters.png/pdf` and `F9_B_recovery_parameters.png/pdf`: paired comparisons, with injected references and native prior ranges.", ""]
    (out / "recovery_summary.md").write_text("\n".join(lines))


def self_check():
    # The second sample is a pure label exchange of the first one.
    names = [f"{k}-th_component_{name}" for k in range(2) for name in ("amplitude", "centroid", "sigma")]
    sample = np.array([[0, 2, 0, np.log(3), 10, np.log(2)],
                       [np.log(3), 10, np.log(2), 0, 2, 0]])
    ordered, swapped = reorder_component_columns(sample, names, 2, "gamma", {"component_order": "moment1"})
    assert np.array_equal(swapped, [False, True])
    assert np.allclose(ordered[0], ordered[1])
    _, width, amp, centroid = component_parameter_arrays(ordered, names, 2, "gamma")
    mean, variance, area, fraction = mixture_moments(centroid, width, amp)
    assert np.allclose(mean, 8) and np.allclose(variance, 18.5) and np.allclose(area, 4)
    assert np.allclose(fraction[:, 0], [.25, .75])
    assert np.isclose(weighted_quantile([0, 10], [.25, .75], [.5])[0], 10/3)
    edge = edge_screen(np.array([0, .99, .5]), np.array([.2, .3, .5]), 0, 1)
    assert np.isclose(edge["lower_mass"], .2) and np.isclose(edge["upper_mass"], .3)
    print("PASS: joint-tuple relabeling, analytic Gamma mixture moments, weighted quantile and prior-edge coordinates")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="Run small analytic checks without reading fits or writing outputs")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    out = HERE / "outputs"
    truth = json.loads((out / "truth.json").read_text())
    cases = {case["name"]: case for case in truth["cases"]}
    if set(cases) != set(CASE_ORDER) or tuple(truth["bands"]) != BANDS:
        raise ValueError("Expected exactly four declared cases and M2/B")
    results = [summarize_fit(cases[case], band) for band in BANDS for case in CASE_ORDER]
    n_analyzed = sum(result["status"] == "analyzed" for result in results)
    summary = {"status": "complete" if n_analyzed == 8 else "partial", "n_expected": 8,
               "n_analyzed": n_analyzed, "generated_utc": datetime.now(timezone.utc).isoformat(),
               "truth_file": str(out / "truth.json"), "time_frame": "observer_days",
               "posterior_source": "aligned raw sample1d.txt_2 and final column of weights.txt; all rows, no branch selection",
               "component_labels": "each joint draw sorted by increasing physical centroid; the full tuple moves together",
               "interval_probabilities": list(Q), "edge_strip_fraction_of_native_prior_span": EDGE_FRACTION,
               "results": results}
    (out / "recovery_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    parameter_rows, edge_rows = tabular_rows(results)
    write_csv(out / "recovery_parameters.csv", parameter_rows)
    write_csv(out / "recovery_prior_edges.csv", edge_rows)
    plot_results(results, truth, out)
    write_markdown(summary, out)
    print(json.dumps({"status": summary["status"], "analyzed": n_analyzed,
                      "summary": str(out / "recovery_summary.md")}, indent=2))
    if n_analyzed != 8:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
