#!/usr/bin/env python3
"""Run up to three prepared Mrk 817 cases concurrently.

Each case completes the Gamma fit and its full post-MICA pipeline, then
its Gaussian pipeline. A completed case releases a slot for the next case;
each active case retains five concurrent response bands. Postprocessing
shares a file lock to bound plotting memory. A failure stops new submissions
while other active cases finish.
Inspect monitoring/status.json; invoking this script without arguments only
shows the saved status. Explicit ``run`` starts the experiment.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback

from run import (CASES, HERE, ROOT, STAGES, make_sed_config, read_config,
                 sed_complete, stage_config, step_outputs)

MONITOR = HERE / "monitoring"
MAX_CASES = 3
BANDS = 5


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Controller:
    def __init__(self, cases_dir: Path, plan: dict[str, list[str]]) -> None:
        self.cases_dir = cases_dir
        self.state_lock = threading.RLock()
        self.state = {
            "started_at": now(), "updated_at": now(),
            "controller_pid": os.getpid(), "status": "running",
            "cases_dir": str(cases_dir),
            "model_order": ["gamma", "gaussian"],
            "max_parallel_cases": MAX_CASES, "max_parallel_bands_per_case": BANDS,
            "cases": {name: {"phase": "queued" if name in plan else "complete",
                             "updated_at": now()} for name in CASES},
        }
        self.save()

    def save(self) -> None:
        with self.state_lock:
            MONITOR.mkdir(parents=True, exist_ok=True)
            target = MONITOR / "status.json"
            temporary = target.with_name(f"status.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, target)

    def update(self, name: str | None = None, **changes: object) -> None:
        with self.state_lock:
            if name is None:
                self.state.update(changes)
            else:
                self.state["cases"][name].update(changes)
                self.state["cases"][name]["updated_at"] = now()
            self.state["updated_at"] = now()
            self.save()

    def command(self, name: str, label: str, argv: list[str]) -> None:
        path = MONITOR / f"{name}.log"
        env = os.environ.copy()
        env.update({
            "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "MPLBACKEND": "Agg", "TMPDIR": "/tmp",
            "MPLCONFIGDIR": str(MONITOR / "mplconfig"),
            "MICA_SMOOTH_POST_LOCK": str(MONITOR / "postprocess.lock"),
        })
        for key in ("TEMP", "TMP"):
            if str(env.get(key, "")).startswith("/mnt/"):
                env.pop(key, None)
        with path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{now()}] {label}: {' '.join(argv)}\n")
            log.flush()
            subprocess.run(argv, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        print(f"[{now()}] {name}: {label} complete", flush=True)

    def case(self, name: str, models: list[str]) -> bool:
        try:
            self.update(name, phase="pipeline", started_at=now())
            self.command(name, "run remaining pipelines",
                         [sys.executable, "-B", str(HERE / "run.py"), "all",
                          "--case", name, "--model", *models,
                          "--cases-dir", str(self.cases_dir)])
            self.update(name, phase="complete", completed_at=now())
            return True
        except Exception as exc:
            with (MONITOR / f"{name}.log").open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
            self.update(name, phase="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"[{now()}] {name}: failed: {exc}", flush=True)
            return False


def validate(cases_dir: Path) -> None:
    for name in CASES:
        cfg = read_config(cases_dir / name)
        for model in STAGES:
            fit = stage_config(cfg, model)
            if len(fit["responses"]) != BANDS or int(fit["max_parallel"]) != BANDS:
                raise ValueError(f"{name}/{model} must launch exactly five bands concurrently")


def model_complete(case: Path, model: str) -> bool:
    cfg = read_config(case)
    fit = stage_config(cfg, model)
    run_root = case / fit["output_root"] / "2comp"
    template = fit.get("run_template", "run_{driver}_to_{band}_{ncomp}comp_{type_tf}")
    for band in fit["responses"]:
        state = run_root / template.format(
            driver=fit["driver"], band=band, ncomp=2, type_tf=model,
        ) / "run_mica_state.json"
        if not state.is_file() or json.loads(state.read_text()).get("status") != "complete":
            return False
    result = run_root / "result"
    if not (result / "mica_component_summary.csv").is_file():
        return False
    if not sed_complete(case, make_sed_config(cfg, model)):
        return False
    return all(path.is_file() and path.stat().st_size > 0
               for step in ("corner", "transfer", "lagfit")
               for path in step_outputs(case, cfg, model, step))


def run(cases_dir: Path | None = None, resume: bool = False) -> int:
    cases_dir = (cases_dir if cases_dir is not None else HERE / "cases").resolve()
    validate(cases_dir)
    plan = {name: [model for model in STAGES
                   if not resume or not model_complete(cases_dir / name, model)]
            for name in CASES}
    plan = {name: models for name, models in plan.items() if models}
    MONITOR.mkdir(parents=True, exist_ok=True)
    with (MONITOR / "controller.lock").open("w", encoding="utf-8") as lockfile:
        try:
            fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another mica_smooth controller is already running") from exc
        controller = Controller(cases_dir, plan)
        queued = iter(plan)
        failed_cases = []
        with ThreadPoolExecutor(max_workers=MAX_CASES) as executor:
            active = {}

            def fill_slots() -> None:
                while len(active) < MAX_CASES:
                    name = next(queued, None)
                    if name is None:
                        break
                    active[executor.submit(controller.case, name, plan[name])] = name

            fill_slots()
            while active:
                completed, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in completed:
                    name = active.pop(future)
                    if not future.result():
                        failed_cases.append(name)
                # Check the whole completed batch before submitting more work.
                if failed_cases:
                    controller.update(status="failed", failed_case=failed_cases[0],
                                      failed_cases=list(failed_cases))
                else:
                    fill_slots()
        controller.update(status="failed" if failed_cases else "complete", completed_at=now())
        return 1 if failed_cases else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status"), nargs="?", default="status")
    parser.add_argument("--cases-dir", type=Path, default=HERE / "cases",
                        help="Prepared case directory (prepare.py --output-dir)")
    parser.add_argument("--resume", action="store_true",
                        help="Run only models missing required fit and postprocessing outputs")
    args = parser.parse_args()
    if args.action == "status":
        path = MONITOR / "status.json"
        print(path.read_text(encoding="utf-8") if path.is_file() else "No controller status yet.")
    else:
        raise SystemExit(run(args.cases_dir, resume=args.resume))


if __name__ == "__main__":
    main()
