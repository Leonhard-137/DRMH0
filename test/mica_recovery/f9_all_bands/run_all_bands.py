#!/usr/bin/env python3
"""Run only the 15 new F9 W1/U/V recovery fits; preserve the 10 M2/B fits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STATUS_PATH = HERE / "run_status.json"
REQUIRED_OUTPUTS = (
    "posterior_sample1d.txt_2", "sample1d.txt_2", "weights.txt",
    "tranfunc_0_1.txt_2", "pline.txt_2_comp0", "pline.txt_2_comp1",
)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_status(status):
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(STATUS_PATH)


def missing_outputs(run_dir):
    files = [Path(run_dir) / "data" / name for name in REQUIRED_OUTPUTS]
    return [str(path) for path in files if not path.is_file() or path.stat().st_size == 0]


def validate_inputs():
    """Read and validate prepared artifacts without creating logs or run folders."""
    sys.path.insert(0, str(ROOT / "src"))
    import run_mica

    batch = yaml.safe_load((HERE / "batch_config.yaml").read_text())
    manifest = json.loads((HERE / "manifest.json").read_text())
    expected_cases = ["disk_only", "response_within", "random_within",
                      "response_beyond", "random_beyond"]
    if batch["cases"] != expected_cases:
        raise ValueError("Expected the five paired disk/slow-signal conditions.")
    if (batch["driver"] != "W2" or batch["new_response_bands"] != ["W1", "U", "V"]
            or batch["reused_response_bands"] != ["M2", "B"]
            or batch["all_response_bands"] != ["M2", "W1", "U", "B", "V"]
            or batch["stage"] != "mica_recovery"
            or batch["fit_batch_sizes"] != [8, 7]):
        raise ValueError("Batch bands, stage or concurrency differ from the prepared extension.")
    if {key: manifest[key] for key in ("n_total_fits", "n_new_fits", "n_reused_fits")} != {
            "n_total_fits": 25, "n_new_fits": 15, "n_reused_fits": 10}:
        raise ValueError("Expected 25 results: 15 new fits and 10 reused fits.")
    case_paths = [(HERE / "cases" / name).resolve() for name in expected_cases]
    if [Path(path).resolve() for path in manifest["cases"]] != case_paths:
        raise ValueError("Manifest case paths do not match this extension directory.")
    parent = (HERE / batch["parent_experiment"]).resolve()
    if parent != (HERE.parent / "f9_m2_b").resolve():
        raise ValueError("The parent experiment must be the existing f9_m2_b experiment.")
    if not Path(manifest["truth_file"]).is_file():
        raise FileNotFoundError(manifest["truth_file"])
    records = {(record["case"], record["band"]): record for record in manifest["results"]}
    if len(records) != 25 or len(manifest["results"]) != 25:
        raise ValueError("Manifest must contain 25 unique case/band results.")
    jobs = []
    for name, case_path in zip(expected_cases, case_paths):
        cfg = run_mica.read_yaml(case_path / "config/source_config.yaml")
        fit = run_mica.fit_config(cfg, batch["stage"])
        old_case = parent / "cases" / name
        old_cfg = run_mica.read_yaml(old_case / "config/source_config.yaml")
        old_fit = run_mica.fit_config(old_cfg, batch["stage"])
        differences = [key for key in set(fit) | set(old_fit)
                       if key not in ("prepared_dir", "responses")
                       and fit.get(key) != old_fit.get(key)]
        if differences:
            raise ValueError(f"{name}: inference settings differ from parent: {differences}")
        if cfg["source"] != "F9SIM" or fit["responses"] != batch["new_response_bands"]:
            raise ValueError(f"{name}: only the new W1/U/V responses may be scheduled.")
        if (fit["driver"], fit["ncomp"], fit["type_tf"], fit["mpi_per_band"],
                fit["num_particles"], fit["max_parallel"]) != ("W2", 2, "gamma", 2, 4, 1):
            raise ValueError(f"{name}: unexpected fit or parallelism settings.")
        input_dir = run_mica.rel(case_path, fit["prepared_dir"]).resolve()
        if input_dir != case_path / "input":
            raise ValueError(f"{name}: prepared inputs must remain in the new case directory.")
        for band in [batch["driver"], *batch["new_response_bands"]]:
            path = input_dir / f"F9SIM_{band}.txt"
            values = np.loadtxt(path, ndmin=2)
            if (values.ndim != 2 or values.shape[1] != 3 or len(values) < 2
                    or not np.isfinite(values).all()
                    or np.any(np.diff(values[:, 0]) <= 0)
                    or np.any(values[:, 1:] <= 0)):
                raise ValueError(f"Invalid timestamps, fluxes or positive errors: {path}")
        if ((input_dir / "F9SIM_W2.txt").read_bytes()
                != (old_case / "input/F9SIM_W2.txt").read_bytes()):
            raise ValueError(f"{name}: W2 input differs from the paired parent input.")
        new_runs = []
        for band in batch["all_response_bands"]:
            reused = band in batch["reused_response_bands"]
            expected_run = run_mica.run_dir(old_case if reused else case_path,
                                           old_fit if reused else fit, band).resolve()
            record = records.get((name, band))
            if (record is None or record["reused"] != reused
                    or Path(record["run_dir"]).resolve() != expected_run):
                raise ValueError(f"Manifest result path/provenance mismatch: {name}/{band}")
            if reused:
                missing = missing_outputs(expected_run)
                if missing:
                    raise FileNotFoundError(f"Incomplete reused fit {name}/{band}: {missing}")
            else:
                kwargs = run_mica.setup_kwargs(fit, {"set1": [None, None]}, band)
                if (kwargs["lag_prior"] != [[-10.0, 2000.0], [-10.0, 2000.0]]
                        or kwargs["width_prior"] != [[0.01, 1200.0], [0.01, 1200.0]]):
                    raise ValueError(f"{name}/{band}: unexpected native component priors.")
                new_runs.append(expected_run)
        jobs.append({"name": name, "path": str(case_path),
                     "run_dirs": [str(path) for path in new_runs]})
    return batch, jobs


def expand_batches(batch, case_jobs):
    """Expand the five cases into exactly 15 individual fits, in two waves."""
    jobs = []
    for case in case_jobs:
        for band, run_dir in zip(batch["new_response_bands"], case["run_dirs"]):
            jobs.append({"key": f"{case['name']}/{band}", "case": case["name"],
                         "band": band, "path": case["path"], "run_dir": run_dir})
    if len(jobs) != 15 or len({job["key"] for job in jobs}) != 15:
        raise ValueError("Expected exactly 15 unique new case/band fits.")
    if len({job["run_dir"] for job in jobs}) != 15:
        raise ValueError("New fits must have 15 distinct run directories.")
    waves, offset = [], 0
    for size in batch["fit_batch_sizes"]:
        waves.append(jobs[offset:offset + size])
        offset += size
    if offset != len(jobs) or [len(wave) for wave in waves] != [8, 7]:
        raise ValueError("The complete run plan must consist of waves of 8 and 7 fits.")
    return jobs, waves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true",
                        help="Read/validate configuration and input files; start no fits.")
    args = parser.parse_args()
    if "PMI_RANK" in os.environ or "PMIX_RANK" in os.environ:
        raise RuntimeError("Run this scheduler with Python directly, without outer mpiexec.")
    batch, case_jobs = validate_inputs()
    jobs, waves = expand_batches(batch, case_jobs)
    for number, wave in enumerate(waves, 1):
        print(f"Wave {number} ({len(wave)} concurrent fits): "
              + ", ".join(job["key"] for job in wave), flush=True)
    if args.validate_only:
        print("PASS: 15 new fits in waves of 8/7; 10 completed M2/B fits reused. No fits started.")
        return
    log_dir = HERE / "logs"
    if STATUS_PATH.exists() or log_dir.exists():
        raise RuntimeError("Batch status/logs already exist. Inspect them before restarting; no automatic overwrite or resume.")
    for case in case_jobs:
        case_path = Path(case["path"])
        if (case_path / "runs").exists() or any(Path(path).exists() for path in case["run_dirs"]):
            raise RuntimeError(f"Existing new-case run outputs: {case_path}; refusing to overwrite.")
    status = {"status": "running", "started_utc": timestamp(),
              "n_new_fits": 15, "n_reused_fits": 10, "fit_batch_sizes": [8, 7],
              "max_concurrent_fits": 8, "ranks_per_fit": 2,
              "max_mpi_ranks": 16, "particles_per_rank": 4,
              "batches": [{"batch": number, "status": "pending",
                           "fits": [job["key"] for job in wave]}
                          for number, wave in enumerate(waves, 1)],
              "fits": {job["key"]: {**job, "status": "pending"} for job in jobs}}
    # Exclusive creation protects a second invocation from starting the same batch.
    with STATUS_PATH.open("x") as handle:
        handle.write(json.dumps(status, indent=2) + "\n")
    log_dir.mkdir()
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": str(HERE / ".mplconfig"), "TMPDIR": "/tmp"})
    running = {}
    try:
        for batch_state, wave in zip(status["batches"], waves):
            batch_state.update(status="running", started_utc=timestamp())
            write_status(status)
            for job in wave:
                key = job["key"]
                logfile = log_dir / f"{job['case']}_{job['band']}.log"
                handle = logfile.open("x")
                command = [sys.executable, "-u", str(ROOT / "src/run_mica.py"),
                           job["path"], job["band"], "--stage", batch["stage"]]
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    handle.close()
                    raise
                running[key] = (process, handle)
                status["fits"][key].update(status="running", batch=batch_state["batch"],
                    pid=process.pid, started_utc=timestamp(), log=str(logfile), command=command)
                print(f"Started {key}; scheduler PID {process.pid}", flush=True)
                write_status(status)
            # The next wave cannot start until every fit in this wave has exited.
            while running:
                for key, (process, handle) in list(running.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    handle.close()
                    missing = missing_outputs(status["fits"][key]["run_dir"])
                    state = "completed" if code == 0 and not missing else "failed"
                    status["fits"][key].update(status=state, exit_code=code,
                        missing_outputs=missing, finished_utc=timestamp())
                    print(f"{state}: {key}; exit {code}", flush=True)
                    del running[key]
                    write_status(status)
                if running:
                    time.sleep(5)
            batch_state.update(status=("completed" if all(status["fits"][key]["status"] == "completed"
                               for key in batch_state["fits"]) else "failed"), finished_utc=timestamp())
            write_status(status)
            print(f"Wave {batch_state['batch']} {batch_state['status']}", flush=True)
    except BaseException as exc:
        # Stop each scheduler and its MPI descendants when this foreground batch is interrupted.
        for key, (process, handle) in running.items():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            handle.close()
            status["fits"][key].update(status="interrupted", exit_code=process.returncode,
                                      finished_utc=timestamp())
        for batch_state in status["batches"]:
            if batch_state["status"] == "running":
                batch_state.update(status="interrupted", finished_utc=timestamp())
        status.update(status="interrupted", error=str(exc), finished_utc=timestamp())
        write_status(status)
        raise
    status.update(status=("completed" if all(fit["status"] == "completed"
                  for fit in status["fits"].values()) else "failed"), finished_utc=timestamp())
    write_status(status)
    print(f"Batch {status['status']}. Status: {STATUS_PATH}", flush=True)
    if status["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
