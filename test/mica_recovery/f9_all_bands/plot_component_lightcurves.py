#!/usr/bin/env python3
"""Compare all 25 injection truths with centroid-ordered MICA GP means.

Chart contract: time-series small multiples answer whether the short/long
recovered variations track injected disk/slow variations. Each band is one
page, with five simulation conditions and disk/slow/total columns. All curves
are in the original normalized input-flux units; only constant offsets are
removed, never amplitudes or time shifts. Scientific PNG/PDF, not a dashboard.

Reconstruction uses every existing joint posterior row, relabelled by centroid,
and native conditional GP covariances in isolated directories. No sampling,
posterior cuts, new resampling, or changes to the original fits. The mean-only
calculation is validated against original native total reconstruction curves.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = HERE / "outputs" / "component_lightcurves"
BANDS = ["M2", "W1", "U", "B", "V"]
CASES = ["disk_only", "response_within", "random_within", "response_beyond", "random_beyond"]
LABELS = {
    "disk_only": "Disk only",
    "response_within": "Disk + slow response\nGamma centroid = 60 d",
    "random_within": "Disk + independent slow\nwithin-window scale",
    "response_beyond": "Disk + slow response\nGamma centroid = 800 d",
    "random_beyond": "Disk + independent slow\nbeyond-window scale",
}


def records():
    return json.loads((HERE / "manifest.json").read_text())["results"]


def target(row):
    return OUT / "reconstruction" / row["case"] / row["band"]


def stage(row, nd_rec):
    from collect_mica_results import param_names, reorder_component_columns

    original = Path(row["run_dir"])
    dest = target(row)
    marker = dest / "complete.json"
    if marker.exists():
        info = json.loads(marker.read_text())
        if info["nd_rec"] != nd_rec:
            raise ValueError("Existing reconstruction has another nd_rec; preserve it and use a new output directory.")
        return info
    (dest / "data").mkdir(parents=True, exist_ok=True)
    (dest / "input").mkdir(exist_ok=True)
    for name in ("cont.txt", "line.txt"):
        shutil.copy2(original / "input" / name, dest / "input" / name)
    posterior_path = original / "data" / "posterior_sample1d.txt_2"
    sample = np.loadtxt(posterior_path, ndmin=2)
    names = param_names(original / "data" / "para_names_line.txt_2")
    ordered, swapped = reorder_component_columns(sample, names, 2, "gamma", {"component_order": "moment1"})
    np.savetxt(dest / "data" / "posterior_sample1d.txt_2", ordered, fmt="%.17g", header=str(len(ordered)))
    weights = np.loadtxt(original / "data" / "weights.txt", ndmin=2)[:, -1]
    weights /= weights.sum()
    info = {**row, "case_dir": str(original.parents[3]), "stage_dir": str(dest),
            "posterior_source": str(posterior_path), "posterior_rows": len(sample),
            "swapped_rows": int(swapped.sum()), "kish_ess": float(1 / np.dot(weights, weights)),
            "nd_rec": nd_rec, "sample_columns": sample.shape[1]}
    (dest / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def native_worker(stage_dir):
    from mpi4py import MPI
    import pymica
    from collect_mica_results import read_yaml
    from run_mica import fit_config, setup_kwargs
    from decompose_mica_branches import decompose_conditional_posterior

    stage_dir = Path(stage_dir).resolve()
    info = json.loads((stage_dir / "provenance.json").read_text())
    config = read_yaml(Path(info["case_dir"]) / "config" / "source_config.yaml")
    fit = fit_config(config, stage="mica_recovery")
    fit["nd_rec"] = info["nd_rec"]
    os.chdir(stage_dir)
    comm = MPI.COMM_WORLD
    data = None
    if comm.rank == 0:
        data = {"set1": [np.loadtxt("input/cont.txt"), np.loadtxt("input/line.txt")]}
    data = comm.bcast(data, root=0)
    model = pymica.gmodel()
    kwargs = setup_kwargs(fit, data, info["band"])
    model.setup(**kwargs)
    decompose_conditional_posterior(pymica, 2, int(kwargs.get("type_lag_prior", 0)),
                                    "./data/posterior_sample1d.txt_2", info["sample_columns"])
    comm.Barrier()


def worker(stage_dir):
    from reconstruct_component_means import run_mean_reconstruction
    run_mean_reconstruction(stage_dir)


def block(path, index=1):
    lines = path.read_text().splitlines()
    sizes = [int(v) for v in lines[1].lstrip("# ").split(":")]
    values = np.loadtxt(path, ndmin=2)
    if sum(sizes) != len(values):
        raise ValueError(f"Incomplete native output: {path}")
    return values[sum(sizes[:index]):sum(sizes[:index + 1])]


def reconstruct(row, nd_rec):
    info = stage(row, nd_rec)
    dest = Path(info["stage_dir"])
    if ((dest / "complete.json").exists() and (dest / "means.npz").exists()
            and "native_total_max_abs_diff" in info):
        return info
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(OUT / ".mplconfig"))
    mpiexec = Path(sys.executable).parent / "mpiexec"
    cmd = [str(mpiexec), "-n", "1", sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(dest)]
    start = time.monotonic()
    print(f"START {row['case']} {row['band']}, N={info['posterior_rows']}", flush=True)
    if not (dest / "means.npz").exists():
        with (dest / "mean_reconstruction.log").open("w") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    means = np.load(dest / "means.npz")
    if not all(np.isfinite(means[k]).all() for k in ("t", "short", "long", "total")):
        raise ValueError(f"Nonfinite reconstructed means: {dest}")
    validation = json.loads((dest / "mean_validation.json").read_text())
    err = validation["native_total_max_abs_diff"]
    if err > 1e-5:
        raise ValueError(f"Mean-only reconstruction differs from original native total: {dest}, max diff {err}")
    staged = np.loadtxt(dest / "data" / "posterior_sample1d.txt_2", ndmin=2)
    if len(staged) != info["posterior_rows"] or not np.all(staged[:, 5] <= staged[:, 8]):
        raise ValueError(f"Staged posterior changed: {dest}")
    info.update(reconstruction_rows=len(means["t"]), native_total_max_abs_diff=err,
                runtime_seconds=round(validation["runtime_seconds"], 2))
    (dest / "complete.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"DONE {row['case']} {row['band']}, {info['runtime_seconds']} s, native total diff={err:.2g}", flush=True)
    return info


def load_comparison(row):
    dest = target(row)
    info = json.loads((dest / "complete.json").read_text())
    source = Path(info["case_dir"])
    truth = np.genfromtxt(source / f"noiseless_{row['band']}.csv", delimiter=",", names=True)
    at_obs = np.genfromtxt(source / f"truth_at_{row['band']}_observations.csv", delimiter=",", names=True)
    obs = np.loadtxt(dest / "input" / "line.txt")
    if not np.allclose(obs[:, 0], at_obs["observer_day"], atol=1e-7, rtol=0):
        raise ValueError("Truth and fitted observation timestamps do not match")
    if not np.allclose(obs[:, 1], at_obs["observed_flux"], atol=1e-8, rtol=0):
        raise ValueError("Truth and fitted observation fluxes do not match")
    means = np.load(dest / "means.npz")
    grid = means["t"]
    native = np.column_stack([means[k] for k in ("short", "long", "total")])
    # Each curve's arithmetic mean evaluated on the same target observation
    # times is removed. This leaves shape and amplitude fixed. Both the truth
    # and conditional means are evaluated at the actual observation times.
    if np.max(np.abs(grid[np.searchsorted(grid, obs[:, 0])] - obs[:, 0])) > 1e-7:
        raise ValueError("Observation times absent from mean reconstruction grid")
    offsets = np.array([np.interp(obs[:, 0], grid, native[:, i]).mean() for i in range(3)])
    recovered = native - offsets
    truth_offsets = np.array([at_obs["disk_variation"].mean(), at_obs["slow_variation"].mean(),
                              (at_obs["disk_variation"] + at_obs["slow_variation"] + 1).mean()])
    truth_y = np.column_stack([truth["disk_variation"], truth["slow_variation"], truth["total_noiseless_flux"]]) - truth_offsets
    good = (grid >= max(truth["observer_day"][0], obs[0, 0])) & (grid <= min(truth["observer_day"][-1], obs[-1, 0]))
    truth_grid = np.column_stack([np.interp(grid, truth["observer_day"], truth_y[:, i]) for i in range(3)])
    metrics = []
    truth_obs = np.column_stack([at_obs["disk_variation"], at_obs["slow_variation"],
                                 at_obs["disk_variation"] + at_obs["slow_variation"] + 1]) - truth_offsets
    for i, comp in enumerate(["short_vs_disk", "long_vs_slow", "total"]):
        a, b = truth_obs[:, i], np.interp(obs[:, 0], grid, recovered[:, i])
        std = float(np.std(a))
        rmse = float(np.sqrt(np.mean((b - a) ** 2)))
        metrics.append({"case": row["case"], "band": row["band"], "comparison": comp,
                        "rmse": rmse, "truth_std": std, "rmse_over_truth_std": rmse / std if std > 1e-10 else None,
                        "pearson_r": float(np.corrcoef(a, b)[0, 1]) if std > 1e-10 and np.std(b) > 1e-10 else None,
                        "posterior_rows": info["posterior_rows"], "kish_ess": info["kish_ess"],
                        "n_observations": len(obs),
                        "removed_reconstruction_constant": offsets[i], "removed_truth_constant": truth_offsets[i]})
    export = np.column_stack([grid[good], recovered[good], truth_grid[good]])
    csvpath = OUT / "curves" / f"{row['case']}_{row['band']}.csv"
    csvpath.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(csvpath, export, delimiter=",", fmt="%.10g", comments="",
               header="observer_day,recovered_short,recovered_long,recovered_total,true_disk,true_slow,true_total")
    return dict(info=info, grid=grid[good], rec=recovered[good], truth_t=truth["observer_day"],
                truth=truth_y, obs=obs, observed_variation=obs[:, 1] - obs[:, 1].mean(), metrics=metrics)


def render():
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D

    all_data = {(r["case"], r["band"]): load_comparison(r) for r in records()}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
                         "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42})
    palette = ["#2864B0", "#C37522", "#2864B0"]
    with PdfPages(OUT / "all_bands_component_comparison.pdf") as pdf:
        for band in BANDS:
            fig, axes = plt.subplots(5, 3, figsize=(17.5, 14.2), sharex=True)
            lims = []
            for col in range(3):
                values = np.concatenate([np.r_[all_data[c, band]["rec"][:, col], all_data[c, band]["truth"][:, col]] for c in CASES])
                extent = max(float(np.max(np.abs(values))) * 1.09, 0.03)
                lims.append((-extent, extent))
            for row_idx, case in enumerate(CASES):
                data = all_data[case, band]
                for col in range(3):
                    ax = axes[row_idx, col]
                    ax.axhline(0, color="0.83", lw=0.6, zorder=0)
                    ax.plot(data["truth_t"], data["truth"][:, col], color="0.2", ls=(0, (3, 2)), lw=1.0, alpha=0.9, zorder=2)
                    ax.plot(data["grid"], data["rec"][:, col], color=palette[col], lw=1.35, zorder=3)
                    if col == 2:
                        ax.errorbar(data["obs"][:, 0], data["observed_variation"], yerr=data["obs"][:, 2],
                                    color="0.62", fmt=".", ms=2, lw=0.45, alpha=0.48, zorder=1)
                    ax.set_ylim(*lims[col])
                    ax.set_xlim(0, 710)
                    ax.grid(axis="x", color="0.92", lw=0.5)
                    if row_idx == 0:
                        ax.set_title(["Short-centroid component vs. true disk", "Long-centroid component vs. true slow", "Total reconstruction vs. noiseless truth"][col], pad=10)
                    if row_idx == 4:
                        ax.set_xlabel("Observer-frame days since MJD 58550.52092")
                    if col == 0:
                        ax.set_ylabel(LABELS[case] + "\n\nNormalized flux variation", fontsize=10)
                    if data["info"]["kish_ess"] < 50:
                        ax.text(0.02, 0.96, f"LOW ESS = {data['info']['kish_ess']:.1f}; Npost = {data['info']['posterior_rows']}",
                                transform=ax.transAxes, va="top", fontsize=9, color="#A34B13",
                                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2))
            fig.suptitle(f"F9 sampling injection / recovery — {band} band", fontsize=21, x=0.08, ha="left", y=0.99)
            fig.text(0.08, 0.958, "Every joint posterior draw ordered by centroid · Conditional GP means · Constants removed; no amplitude scaling or time shifts", fontsize=11)
            handles = [Line2D([0], [0], color="0.2", ls=(0, (3, 2)), lw=1.2, label="Injected truth"),
                       Line2D([0], [0], color=palette[0], lw=1.6, label="MICA short / total mean"),
                       Line2D([0], [0], color=palette[1], lw=1.6, label="MICA long mean"),
                       Line2D([0], [0], color="0.62", marker=".", ls="none", label="Noisy target data (total panels)")]
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.026), ncol=4, frameon=False)
            fig.text(0.08, 0.015, "Reconstruction includes every observation time and a grid of 1 d or finer. Posterior means only; no credible intervals shown.\nSame vertical scale across the five cases within each column; slow column is enlarged. Independent slow truth has no response kernel.", fontsize=9, color="0.35")
            fig.subplots_adjust(left=0.115, right=0.985, bottom=0.09, top=0.917, hspace=0.25, wspace=0.20)
            fig.savefig(OUT / f"component_comparison_{band}.png", dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
    metrics = [m for v in all_data.values() for m in v["metrics"]]
    with (OUT / "comparison_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    write_notes(all_data)
    render_paired_b(all_data)
    print(f"PLOTS COMPLETE: {OUT}", flush=True)


def render_paired_b(all_data=None):
    """Readable companion to the full atlas: one band, the controlled pair."""
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    subset = CASES[:3]
    if all_data is None:
        all_data = {(r["case"], "B"): load_comparison(r) for r in records()
                    if r["band"] == "B" and r["case"] in subset}
    colors = ["#2864B0", "#C37522"]
    fig, axes = plt.subplots(3, 2, figsize=(12.4, 8.6), sharex=True)
    for col in range(2):
        values = np.concatenate([np.r_[all_data[c, "B"]["rec"][:, col],
                                        all_data[c, "B"]["truth"][:, col]] for c in subset])
        extent = max(float(np.max(np.abs(values))) * 1.1, 0.03)
        for row_idx, case in enumerate(subset):
            d, ax = all_data[case, "B"], axes[row_idx, col]
            ax.axhline(0, color="0.85", lw=0.6)
            ax.plot(d["truth_t"], d["truth"][:, col], color="0.2", ls=(0, (3, 2)), lw=1.15)
            ax.plot(d["grid"], d["rec"][:, col], color=colors[col], lw=1.3)
            ax.set(xlim=(0, 710), ylim=(-extent, extent))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x", color="0.92", lw=0.5)
            if row_idx == 0:
                ax.set_title(["Short-centroid component vs. true disk", "Long-centroid component vs. true slow"][col], fontsize=13, pad=10)
            if col == 0:
                ax.set_ylabel(LABELS[case] + "\n\nFlux variation", fontsize=10)
            if row_idx == 2:
                ax.set_xlabel("Observer-frame days since MJD 58550.52092", fontsize=10)
    fig.suptitle("B band · paired injection / recovery", fontsize=21, x=0.12, ha="left", y=0.985)
    fig.text(0.12, 0.94, "Original normalized flux units · Constants removed · No amplitude adjustment or time shift", fontsize=10)
    fig.legend(handles=[Line2D([0], [0], color="0.2", ls="--", label="Injected truth"),
                        Line2D([0], [0], color=colors[0], label="MICA short mean"),
                        Line2D([0], [0], color=colors[1], label="MICA long mean")],
               loc="lower center", bbox_to_anchor=(0.55, 0.035), ncol=3, frameon=False)
    fig.text(0.12, 0.012, "Means only; no credible intervals. Same scale down each column; slow column enlarged.\nMICA conditions on the observed W2 and target curves: resemblance alone does not establish physical response.", fontsize=9, color="0.35")
    fig.subplots_adjust(left=0.145, right=0.985, top=0.88, bottom=0.13, hspace=0.28, wspace=0.22)
    fig.savefig(OUT / "component_comparison_B_paired.png", dpi=170)
    plt.close(fig)


def write_notes(all_data):
    ess_rows = "\n".join(f"| {case} | {band} | {d['info']['posterior_rows']} | {d['info']['kish_ess']:.1f} | {d['info']['swapped_rows']} |" for (case, band), d in all_data.items())
    text = f"""# Injected and recovered component light curves

All 25 completed simulations from `../../manifest.json` are included. Original M2/B fits live under f9_m2_b; W1/U/V under f9_all_bands. Each PDF page is one response band; rows are five injection conditions; columns compare short/disk, long/slow, and total/noiseless target. PNGs contain the same pages.

## Exact comparison definition

For every existing joint row in `posterior_sample1d.txt_2`, exchange the entire amplitude/centroid/log-width triplet if necessary so component 0 has the shorter centroid. Keep all rows and all other parameters. The centroid column is already the Gamma centroid; do not add twice the width. Initialize the native covariance routines using the same sequence as `src/decompose_mica_branches.py::decompose_conditional_posterior` in a new isolated directory. The helper `reconstruct_component_means.py` uses native data and component cross-covariance matrices and solves for conditional GP means, skipping prediction variances. For each joint draw, q=(L^T C^-1 L)^-1 L^T C^-1 y, alpha=C^-1(y-Lq), and mean_k=U_k alpha + [A_k/sum(A)]q_line. Multiply back by the native target flux scale. No sampling or posterior regeneration occurs. Each reconstruction checks agreement with the existing original native total at its exact exported time grid, tolerance 1e-5 input-flux units. Additional full native sorted-component reconstruction validates response_within/U on 600 time points.

Native light curves include constant baselines, allocated to components by amplitude fraction. Remove each reconstructed curve's arithmetic mean at the exact target observation times, all included in the reconstruction grid. Remove each truth curve's exact mean at those same observation times. Keep amplitudes and times unchanged. The plots thus compare variable flux, not absolute component DC levels. The known injected noiseless target is 1 + disk_variation + slow_variation. Truth files and input flux/time columns are checked against each other. Time is observer-frame days; units are the original synthetic normalized flux units, mean approximately 1. These experiments did not inject an interband physical SED or a special U excess.

## Reading limits

- Native means condition on the observed driver AND target. A recovered long curve following an independent injected slow curve does not establish physical response or causality.
- The injected driver is simulated using F9 sampling/noise properties. These are recovery experiments, not decompositions of the real F9_a measured flux.
- In independent-slow cases there is a true slow light curve but no true response transfer function for that component.
- Solid curves are posterior averages of conditional GP means, not a single median-parameter prediction. No credible interval is drawn. Native component error columns are max(across-draw SD of conditional means, maximum conditional GP SD); they are neither quantiles nor complete uncertainty of the centered curve. The mean-only helper does not compute or export them.
- response_within/W1 and response_within/U have Kish ESS approximately 6.9 and 11.6 (12 and 19 exported posterior rows). Their plots are tentative and marked LOW ESS. Completion of sampling alone did not guarantee stable posteriors.
- The reconstruction includes every actual target observation time and a grid of 1 day or finer; truth is displayed on its saved 0.2-day grid. Observation gaps and noise limit interpretation of fine-scale details; do not attribute all discrepancies to transfer-function failure.
- `comparison_metrics.csv` is descriptive only: RMSE and correlation compare centered means with exact truth at actual target observation times. They are not calibrated tests and do not use uncertainty weights. A zero slow truth has no normalized RMSE or correlation.

## Posterior provenance

| Case | Band | Exported posterior rows | Kish ESS from raw weights | Rows relabelled |
| --- | --- | ---: | ---: | ---: |
{ess_rows}

Reproduction: `/home/leonhard/miniforge3/envs/astro/bin/python -B test/mica_recovery/f9_all_bands/plot_component_lightcurves.py --jobs 4`. Completed isolated reconstructions are reused. `--plot-only` regenerates plots and CSV from them. Native stdout is retained per reconstruction directory. Original fit artifacts are never overwritten.
"""
    (OUT / "README.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--native-worker")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--nd-rec", type=int, default=600)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--band", choices=BANDS)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.native_worker:
        native_worker(args.native_worker)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        selected = [r for r in records() if (not args.case or r["case"] == args.case) and (not args.band or r["band"] == args.band)]
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(reconstruct, r, args.nd_rec) for r in selected]
            for future in as_completed(futures):
                future.result()
    if all((target(r) / "complete.json").exists() and (target(r) / "means.npz").exists() for r in records()):
        render()


if __name__ == "__main__":
    main()
