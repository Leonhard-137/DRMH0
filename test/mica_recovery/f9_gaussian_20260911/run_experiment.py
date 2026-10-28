#!/usr/bin/env python3
"""Queue Gaussian recovery of the saved 25 F9 Gamma inputs after Mrk817."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.parent / "f9_all_bands"
DEPENDENCY = HERE.parent / "mrk817_w2_thermal_20260910"
BANDS = ["M2", "W1", "U", "B", "V"]
CASES = ["random_within", "random_beyond", "disk_only",
         "response_within", "response_beyond"]
STAGE = "mica_recovery"
ASTRO = Path("/home/leonhard/miniforge3/envs/astro/bin/python")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def engine():
    # Reuse queue functions in a new module instance; never edit the old queue.
    queue = load_module("_f9_queue_engine", DEPENDENCY / "run_experiment.py")
    queue.HERE = HERE
    queue.STATE = HERE / "queue_state.json"
    queue.LOCK = HERE / "queue.lock"
    queue.BANDS = BANDS
    queue.STAGES = [STAGE]
    return queue


def exclusive_content(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite different existing artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def expected_artifacts():
    previous = json.loads((OLD / "manifest.json").read_text())
    records = {(r["case"], r["band"]): r for r in previous["results"]}
    artifacts, results = {}, []
    for name in CASES:
        case = HERE / "cases" / name
        config = yaml.safe_load((OLD / "cases" / name / "config/source_config.yaml").read_text())
        fit = deepcopy(config[STAGE])
        fit.update(type_tf="gaussian", prepared_dir=str(case / "input"),
                   responses=BANDS, num_particles=1,
                   width_prior=[[math.sqrt(2)*v for v in pair]
                                for pair in fit["width_prior"]])
        config[STAGE] = fit
        config["simulation"] = {
            "case": name, "parent_manifest": str(OLD / "manifest.json"),
            "construction": "Exact saved Gamma input bytes; no new simulation/noise.",
            "comparison": "Equal centroid and physical-standard-deviation priors; "
                          "particles per rank changed from 4 to 1 as recently requested.",
        }
        artifacts[case / "config/source_config.yaml"] = yaml.safe_dump(
            config, sort_keys=False, allow_unicode=True).encode()
        driver_bytes = None
        for band in BANDS:
            old_run = Path(records[name, band]["run_dir"])
            cont = (old_run / "input/cont.txt").read_bytes()
            if driver_bytes is not None and cont != driver_bytes:
                raise ValueError(f"Old {name} pair fits used different drivers.")
            driver_bytes = cont
            artifacts[case / "input/F9SIM_W2.txt"] = cont
            artifacts[case / "input" / f"F9SIM_{band}.txt"] = (
                old_run / "input/line.txt").read_bytes()
            results.append({
                "case": name, "case_dir": str(case), "band": band, "stage": STAGE,
                "run_dir": str(case / "runs/mica/2comp" /
                               f"run_W2_to_{band}_2comp_gaussian"),
                "gamma_run_dir": str(old_run),
                "original_config": str(old_run.parents[3] / "config/source_config.yaml"),
            })
    manifest = {
        "n_total_fits": 25, "responses": BANDS, "stages": [STAGE],
        "cases": [str(HERE / "cases" / name) for name in CASES],
        "results": results, "truth_file": previous["truth_file"],
        "parent_manifest": str(OLD / "manifest.json"),
        "wait_for": str(DEPENDENCY / "queue_state.json"),
        "particles_per_rank": 1, "gamma_particles_per_rank": 4,
    }
    artifacts[HERE / "manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    return artifacts


def validate(queue):
    for path, content in expected_artifacts().items():
        if path.read_bytes() != content:
            raise ValueError(f"Prepared artifact differs from paired design: {path}")
    jobs = queue.validate_manifest(HERE / "manifest.json")
    runner = load_module("_f9_runner_check", ROOT / "src/run_mica.py")
    for job in jobs:
        case = Path(job["case_dir"])
        fit = runner.fit_config(runner.read_yaml(case / "config/source_config.yaml"), STAGE)
        old_fit = runner.fit_config(runner.read_yaml(Path(job["original_config"])), STAGE)
        old_kw = runner.setup_kwargs(old_fit, {"set1": [None, None]}, job["band"])
        new_kw = runner.setup_kwargs(fit, {"set1": [None, None]}, job["band"])
        for key in old_kw.keys() | new_kw.keys():
            if key not in {"data", "type_tf", "width_prior", "num_particles"}:
                if old_kw.get(key) != new_kw.get(key):
                    raise ValueError(f"Unexpected inference change: {job['key']} {key}")
        if not np.allclose(new_kw["width_prior"],
                           math.sqrt(2)*np.asarray(old_kw["width_prior"]), rtol=1e-12):
            raise ValueError("Physical width prior is not matched.")
        for band in ["W2", job["band"]]:
            values = np.loadtxt(case / "input" / f"F9SIM_{band}.txt")
            if (values.shape[1] != 3 or not np.isfinite(values).all()
                    or np.any(np.diff(values[:, 0]) <= 0) or np.any(values[:, 2] <= 0)):
                raise ValueError(f"Invalid light curve: {job['key']} {band}")
    return jobs


def dependency_ready(queue):
    with (DEPENDENCY / "queue.lock").open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return False, "Mrk817 controller is still running."
        state = queue.read_json(DEPENDENCY / "queue_state.json")
        if state["status"] != "complete":
            return False, f"Mrk817 status is {state['status']}; waiting for successful completion."
        if any(queue.is_alive(job) for job in state["jobs"]):
            return False, "Mrk817 still has live fit processes."
        if not all(queue.completion(job)[0] for job in state["jobs"]):
            return False, "Mrk817 completion products are incomplete."
        return True, "Mrk817 completed and released its queue."


def run(queue, jobs, args):
    # Covers both waiting and fitting; the engine also holds its own queue lock.
    with (HERE / "scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {
            "controller_pid": os.getpid(),
            "controller_start_ticks": queue.process_identity(os.getpid())["start_ticks"],
            "status": "waiting_for_mrk817", "started_at": queue.now(),
            "max_parallel": 10, "dependency": str(DEPENDENCY),
            "jobs": queue.reconcile(jobs, args.retry_failed),
        }
        def stop(signum, frame):
            state.update(status="interrupted", message="Waiting controller was stopped.")
            queue.save_state(state)
            raise SystemExit(128 + signum)
        previous = {s: signal.signal(s, stop)
                    for s in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]}
        try:
            while True:
                ready, message = dependency_ready(queue)
                state["message"] = message
                queue.save_state(state)
                if ready:
                    break
                time.sleep(15)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
        return queue.run_queue(args, jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--detach", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    queue = engine()
    if args.prepare:
        for path, content in expected_artifacts().items():
            exclusive_content(path, content)
    jobs = validate(queue)
    if args.prepare or args.validate_only:
        print(json.dumps({"valid": True, "fits": len(jobs), "bands": BANDS,
                          "cases": CASES, "identical_saved_inputs": True,
                          "particles_per_rank": 1, "no_fits_started": True}))
        return 0
    args.manifest = HERE / "manifest.json"
    args.max_parallel = 10
    args.poll_seconds = 5
    if args.detach:
        command = [str(ASTRO), "-B", "-u", str(Path(__file__).resolve()), "--run"]
        if args.retry_failed:
            command.append("--retry-failed")
        with (HERE / "controller.log").open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Controller exited {process.returncode}; inspect controller.log.")
            state = queue.read_json(queue.STATE) if queue.STATE.exists() else {}
            if state.get("controller_pid") == process.pid:
                print(json.dumps({"controller_pid": process.pid, "status": state["status"],
                    "counts": dict(Counter(j["status"] for j in state["jobs"])),
                    "state_file": str(queue.STATE)}))
                return 0
            time.sleep(0.1)
        raise RuntimeError("Controller startup not confirmed; inspect before retrying.")
    return run(queue, jobs, args)


if __name__ == "__main__":
    raise SystemExit(main())
