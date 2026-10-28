"""Run or replot the frozen shared-SED, one-distance-per-source analysis."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


@contextmanager
def output_lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"An analysis is already using {path.parent}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_worker(settings, source):
    from .inference import source_run
    folder = settings.output_dir / "sources" / source
    with output_lock(folder / ".source.lock"):
        try:
            source_run(source, settings)
        except Exception as error:
            # Preserve a completed baseline if its compatibility check failed.
            status = folder / "status.json"
            previous = json.loads(status.read_text()) if status.exists() else {}
            if previous.get("state") != "complete":
                write_json(status, dict(state="failed", source=source, error=str(error)))
            raise


def run_sources(settings):
    from .config import PROJECT_ROOT
    from .inference import validate_cached_source
    from .kernel import build_kernel
    pending, running, done = [], {}, {}
    out = settings.output_dir
    for source in settings.sources:
        status = out / "sources" / source / "status.json"
        if status.exists() and json.loads(status.read_text()).get("state") == "complete":
            validate_cached_source(source, settings)
            done[source] = 0
            print(f"reuse {source}", flush=True)
        else:
            pending.append(source)
    if pending:
        build_kernel(out / "build")
    run_config = out / "run_config.json"
    write_json(run_config, settings.to_dict())
    while pending or running:
        while pending and len(running) < settings.parallel:
            source = pending.pop(0)
            log = (out / f"{source}.log").open("a")
            command = [sys.executable, "-m", "src.shared_sed_hubble",
                       "--worker", source, "--run-config", str(run_config)]
            proc = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, env=os.environ.copy())
            running[source] = (proc, log)
            print(f"started {source} {proc.pid}", flush=True)
        for source, (proc, log) in list(running.items()):
            code = proc.poll()
            if code is not None:
                log.close()
                done[source] = code
                del running[source]
                print(f"finished {source} {code}", flush=True)
        write_json(out / "controller_status.json", dict(
            pending=pending, running={s: p.pid for s, (p, _) in running.items()},
            exit_codes=done))
        if running:
            time.sleep(2)
    # Also record the no-work, all-cached case.
    write_json(out / "controller_status.json", dict(pending=[], running={}, exit_codes=done))
    if any(done.values()):
        raise RuntimeError("Source fit failed; inspect its log and rerun to resume")


def execute(settings, step, source=None):
    from . import MODEL_VERSION
    out = settings.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out / ".controller.lock"):
        started = time.time()
        write_json(out / "controller.json", dict(
            pid=os.getpid(), state="running", started=started, step=step,
            source=source, model_version=MODEL_VERSION))
        write_json(out / "run_config.json", settings.to_dict())
        try:
            if source:
                status = out / "sources" / source / "status.json"
                complete = status.exists() and json.loads(status.read_text()).get("state") == "complete"
                if not complete:
                    from .kernel import build_kernel
                    build_kernel(out / "build")
                run_worker(settings, source)
            else:
                if step in ("all", "sources"):
                    run_sources(settings)
                if step in ("all", "h0"):
                    from .inference import finalize
                    finalize(settings)
                if step in ("all", "plot"):
                    from .plotting import make_plots
                    make_plots(out, list(settings.sources), om0=.3)
                if step in ("all", "h0", "validate"):
                    from .inference import validate_results
                    report = validate_results(settings)
                    print(json.dumps(report, indent=2), flush=True)
                    validation = out / "validation.json"
                    data = json.loads(validation.read_text()) if validation.exists() else {}
                    data.update(state="numerically_complete", model_version=MODEL_VERSION,
                                numerical_review="passed", numerical_validation_file="numerical_validation.json")
                    write_json(validation, data)
        except Exception as error:
            write_json(out / "controller.json", dict(
                pid=os.getpid(), state="failed", started=started,
                finished=time.time(), step=step, source=source, error=str(error)))
            raise
        write_json(out / "controller.json", dict(
            pid=os.getpid(), state="complete", started=started, finished=time.time(),
            step=step, source=source, model_version=MODEL_VERSION))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML paths/source selection; default is package defaults.yaml")
    parser.add_argument("--root", type=Path, help="Project/data root; defaults to the repository")
    parser.add_argument("--output", type=Path, help="Result directory, relative to project root or absolute")
    parser.add_argument("--step", choices=("all", "sources", "h0", "plot", "validate"), default="all")
    parser.add_argument("--parallel", type=int, help="Maximum concurrent source fits; default 3")
    parser.add_argument("--source", help="Fit or resume only this configured source")
    parser.add_argument("--background", action="store_true", help="Detach and write controller.log in the result directory")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--run-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    # Set before importing numpy/FFA/emcee in this process or its workers.
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    from .config import PROJECT_ROOT, RunConfig, load_config
    if args.worker:
        if args.run_config is None:
            parser.error("worker requires run-config")
        settings = RunConfig.from_dict(json.loads(args.run_config.read_text()))
        if args.worker not in settings.sources:
            parser.error("unknown worker source")
        run_worker(settings, args.worker)
        return
    settings = load_config(args.config, root=args.root, output=args.output, parallel=args.parallel)
    if args.source and (args.source not in settings.sources or args.step not in ("all", "sources")):
        parser.error("source must be configured and used with step all or sources")
    os.environ["MPLCONFIGDIR"] = str(settings.output_dir / ".mplconfig")
    if args.background:
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "src.shared_sed_hubble",
                   "--root", str(settings.project_root), "--output", str(settings.output_dir),
                   "--step", args.step, "--parallel", str(settings.parallel)]
        if args.config is not None:
            command.extend(["--config", str(args.config.resolve())])
        if args.source:
            command.extend(["--source", args.source])
        with (settings.output_dir / "controller.log").open("a") as log:
            proc = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True,
                                    env=os.environ.copy())
        write_json(settings.output_dir / "launch.json", dict(pid=proc.pid, command=command))
        print(f"Started PID {proc.pid}; results: {settings.output_dir}")
        return
    execute(settings, args.step, args.source)


if __name__ == "__main__":
    main()
