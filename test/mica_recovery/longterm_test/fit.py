#!/usr/bin/env python3
"""Fit saved inputs using the project's complete native MICA workflow.

python fit.py                         # validate current shared-slow cases only
python fit.py --prepare-only          # also write per-case MICA configuration
python fit.py --run                   # two Gaussians, then two Gammas
python fit.py --case disk_only --run  # select another saved case
python fit.py --suite all --run       # all configured cases, sequentially
python fit.py --suite all --batch-size 5 --run  # batches of 5 concurrent fits
python fit.py --config experiment_broad.yaml --suite known_response --run
python fit.py --config experiment_spectral.yaml --suite spectral_shared --model gaussian
python fit.py --model gamma --run     # one kernel family only

All native MICA samples, reconstruction, decomposition, plots and post_process
products are retained. Add --plot-results for the three case-local figure sets.
Input/component TXT files are never regenerated or rescaled. Repeat --run with
unchanged settings to use src's checked resume/skip behavior; no outer mpiexec.
"""
from __future__ import annotations

import argparse
import fcntl
from math import sqrt
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import yaml

from prepare import DEFAULT_CONFIG, case_name, load_config, load_observed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if "PMI_RANK" in os.environ or "PMIX_RANK" in os.environ:
    raise RuntimeError("Run fit.py with Python directly, without outer mpiexec")
sys.path.insert(0, str(ROOT / "src"))
import run_mica


def bounds(value: object, name: str, *, positive: bool = False) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be [low, high]")
    low, high = map(float, value)
    if not np.isfinite([low, high]).all() or low >= high or (positive and low <= 0):
        raise ValueError(f"Invalid {name}: {value}")
    return [low, high]


def source_config(config: dict) -> dict:
    """Translate physical priors into each native kernel's parameters."""
    spec = config["mica_fit"]
    lag = bounds(spec["centroid_lag_days"], "mica_fit.centroid_lag_days")
    sigma = bounds(spec["sigma_days"], "mica_fit.sigma_days", positive=True)
    gamma_sigma = bounds(
        spec.get("gamma_sigma_days", sigma),
        "mica_fit.gamma_sigma_days",
        positive=True,
    )
    for key in ("max_num_saves", "mpi_per_fit", "num_particles"):
        if isinstance(spec[key], bool) or not isinstance(spec[key], int) or spec[key] < 1:
            raise ValueError(f"mica_fit.{key} must be a positive integer")
    if not isinstance(spec["flag_negative_resp"], bool):
        raise ValueError("mica_fit.flag_negative_resp must be true or false")
    gaussian_order = spec.get("gaussian_component_order", "free")
    if gaussian_order not in ("native", "free"):
        raise ValueError("mica_fit.gaussian_component_order must be native or free")
    ptol = float(spec["ptol"])
    if not np.isfinite(ptol) or not 0 < ptol < 1:
        raise ValueError("mica_fit.ptol must be between zero and one")
    driver = str(config["observations"]["driver_band"])
    response = str(config["observations"]["response_band"])
    result = {
        "source": str(config["experiment"]["simulation_source"]),
        "reference_band": driver, "response_bands": [response],
        "mica_stages": ["mica_gaussian2", "mica_gamma2"],
    }
    for family in ("gaussian", "gamma"):
        # Local MICA gamma has shape k=2, centroid mu and scale theta:
        # onset=mu-2*theta, physical std=sqrt(2)*theta. Gaussian width is std.
        physical_width = gamma_sigma if family == "gamma" else sigma
        width = [
            value / (sqrt(2) if family == "gamma" else 1)
            for value in physical_width
        ]
        result[f"mica_{family}2"] = {
            "output_root": "runs/mica", "prepared_dir": "input",
            "driver": driver, "responses": [response],
            "run_template": "run_{driver}_to_{band}_{ncomp}comp_{type_tf}",
            "ncomp": 2, "type_tf": family,
            # Explicit component priors bypass native +/-Tspan/2 lag caps.
            # Type 4 permits label exchange; component index is not identity.
            "type_lag_prior": 4, "lag_limit": lag.copy(),
            "lag_prior": [lag.copy(), lag.copy()],
            "width_prior": [width.copy(), width.copy()],
            "flag_lag_posivity": False,
            "flag_negative_resp": spec["flag_negative_resp"],
            "flag_con_sys_err": False, "flag_line_sys_err": False,
            "flag_trend": 0,
            "max_num_saves": spec["max_num_saves"],
            "mpi_per_band": spec["mpi_per_fit"], "max_parallel": 1,
            "num_particles": spec["num_particles"], "ptol": ptol,
            "fit_only": False, "plot_decomp": True,
        }
        if family == "gaussian" and gaussian_order == "native":
            # Reuse src's native set_priors hook; passing lag_prior straight to
            # pymica.setup would silently switch ordered type 0 back to type 4.
            result[f"mica_{family}2"].update(type_lag_prior=0, component_order="native")
    return result


def build_plan(config_path: Path, *, suite: str | None = None,
               cases: list[str] | None = None, model: str = "both") -> list[dict]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    catalog = {case_name(float(tau), name): "gaussian"
               for tau in config.get("drw", {}).get("tau_days", [])
               for name in config.get("response", {}).get("gaussian_cases", {})}
    catalog.update(dict.fromkeys(config.get("independent_response", {}).get("cases", {}), "independent"))
    catalog.update(dict.fromkeys(config.get("shared_slow", {}).get("cases", {}), "shared_slow"))
    catalog.update(dict.fromkeys(config.get("known_response", {}).get("cases", {}), "known_response"))
    catalog.update(dict.fromkeys(config.get("spectral_shared", {}).get("cases", {}), "spectral_shared"))
    catalog.update(dict.fromkeys(config.get("real_disk_mismatch", {}).get("cases", {}), "real_disk_mismatch"))
    # Completed reused_cases are references, not additional fit jobs.
    catalog.update(dict.fromkeys(config.get("paired_ablation", {}).get("cases", {}), "paired_ablation"))
    if cases:
        selected = list(dict.fromkeys(cases))
        unknown = [name for name in selected if name not in catalog]
        if unknown:
            raise ValueError(f"Unknown cases: {unknown}")
        if suite not in (None, "all") and any(catalog[name] != suite for name in selected):
            raise ValueError("Selected --case does not belong to --suite")
    else:
        suite = suite or "shared_slow"
        selected = [name for name, kind in catalog.items() if suite == "all" or kind == suite]
    if not selected:
        raise ValueError("No configured cases selected")
    if model not in ("both", "gaussian", "gamma"):
        raise ValueError(f"Unknown kernel family: {model}")
    cfg = source_config(config)
    stages = cfg["mica_stages"] if model == "both" else [f"mica_{model}2"]
    output_dir = (config_path.parent / config["experiment"]["output_dir"]).resolve()
    driver, response = cfg["reference_band"], cfg["response_bands"][0]
    plan = []
    for name in selected:
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Invalid case directory name: {name!r}")
        case = output_dir / name
        data = {"set1": [load_observed(case / "input" / f"{cfg['source']}_{band}.txt")
                         for band in (driver, response)]}
        if any(np.any(curve[:, 1] <= 0) for curve in data["set1"]):
            raise ValueError(f"Nonpositive input flux in {case}")
        # Never consult truth/component tables or prescribe injected DRW tau.
        for stage in stages:
            fit = run_mica.fit_config(cfg, stage)
            run_mica.setup_kwargs(fit, data, response)
            plan.append({"case": case, "config": cfg, "stage": stage,
                         "fit": fit, "response": response,
                         "run_dir": run_mica.run_dir(case, fit, response)})
    return plan


def check_config_target(case: Path, cfg: dict) -> None:
    path = case / "config/source_config.yaml"
    if path.exists() and run_mica.read_yaml(path) != cfg:
        runs = case / "runs"
        if runs.exists() and any(runs.iterdir()):
            raise ValueError(f"Existing MICA runs have a different configuration: {case}; "
                             "keep the original settings for resume, or use a separate experiment for a new fit")


def write_config(case: Path, cfg: dict) -> None:
    check_config_target(case, cfg)
    path = case / "config/source_config.yaml"
    if path.exists() and run_mica.read_yaml(path) == cfg:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Generated by longterm_test/fit.py from the selected experiment configuration.\n"
                    + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def run_batches(plan: list[dict], config_path: Path, batch_size: int) -> None:
    """Run whole batches; each child uses src's existing MPI/run cleanup."""
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1"})
    # A second batch scheduler must not multiply the requested concurrency.
    lock_path = plan[0]["case"].parent / ".mica_batch.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another MICA batch is already running: {lock_path}") from exc
        total_batches = (len(plan) + batch_size - 1) // batch_size
        for offset in range(0, len(plan), batch_size):
            group = plan[offset:offset + batch_size]
            print(f"Batch {offset // batch_size + 1}/{total_batches}: {len(group)} concurrent fits", flush=True)
            running = []
            try:
                for job in group:
                    # Do not forward --batch-size: each child runs one fit.
                    command = [sys.executable, "-u", str(HERE / "fit.py"),
                               "--config", str(config_path.resolve()),
                               "--case", job["case"].name,
                               "--model", job["fit"]["type_tf"], "--run"]
                    process = subprocess.Popen(command, cwd=ROOT, env=env)
                    running.append((process, job))
                    print(f"Started scheduler PID {process.pid}: {job['case'].name}/{job['fit']['type_tf']}", flush=True)
                pending = list(running)
                while pending:
                    for process, job in list(pending):
                        code = process.poll()
                        if code is None:
                            continue
                        pending.remove((process, job))
                        name = f"{job['case'].name}/{job['fit']['type_tf']}"
                        if code != 0:
                            raise RuntimeError(f"MICA fit failed: {name}, exit={code}; later batches will not start")
                        print(f"Completed: {name}", flush=True)
                    if pending:
                        time.sleep(0.2)
            finally:
                for process, _ in running:
                    if process.poll() is None:
                        process.terminate()
                for process, _ in running:
                    try:
                        # src.launch needs up to 5 s to clean up its MPI groups.
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        print(f"All {len(plan)} MICA fits completed.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--suite",
        choices=("all", "gaussian", "independent", "shared_slow",
                 "known_response", "spectral_shared", "real_disk_mismatch", "paired_ablation"),
    )
    parser.add_argument("--case", action="append", help="Select a saved case; repeat for several cases.")
    parser.add_argument("--model", choices=("both", "gaussian", "gamma"), default="both")
    parser.add_argument("--plot-results", action="store_true",
                        help="After --run completes, write transfer/light-curve/corner figures to each case's results directory.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Concurrent fits per batch; wait for the entire batch before starting the next.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Write configuration and start native MICA fits.")
    mode.add_argument("--prepare-only", action="store_true", help="Write configuration without starting MICA.")
    args = parser.parse_args()
    if args.plot_results and not args.run:
        parser.error("--plot-results requires --run; use plot_results.py for already completed fits")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    plan = build_plan(args.config, suite=args.suite, cases=args.case, model=args.model)
    for job in plan:
        check_config_target(job["case"], job["config"])
        fit = job["fit"]
        print(f"{job['case'].name}/{fit['type_tf']}: {fit['ncomp']} components; "
              f"centroid={fit['lag_prior'][0]} d; native width={fit['width_prior'][0]} d; "
              f"positive amplitude={not fit['flag_negative_resp']}; "
              f"saves={fit['max_num_saves']}; MPI ranks={fit['mpi_per_band']}; "
              f"particles per rank={fit['num_particles']}; PTol={fit['ptol']}\n  -> {job['run_dir']}", flush=True)
    sizes = [len(plan[i:i + args.batch_size]) for i in range(0, len(plan), args.batch_size)]
    print(f"Batch sizes: {sizes}; up to {min(args.batch_size, len(plan))} concurrent fits", flush=True)
    if not (args.run or args.prepare_only):
        print(f"Validated {len(plan)} fits; no files written or fits started. Use --run to start.")
        return
    for case in dict.fromkeys(job["case"] for job in plan):
        write_config(case, plan[0]["config"])
    if args.prepare_only:
        print("MICA configurations written; no fits started.")
        return

    # Preserve the src scheduler's cleanup on terminal interrupts/termination.
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        if args.batch_size > 1:
            run_batches(plan, args.config, args.batch_size)
        else:
            for job in plan:
                run_mica.launch(job["case"], job["fit"], [job["response"]], "run", job["stage"])
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if args.plot_results:
        from plot_results import plot_jobs
        plot_jobs(plan, args.config)


if __name__ == "__main__":
    main()
