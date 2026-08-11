#!/usr/bin/env python3
"""One-command driver for the analysis scripts under src/.

Examples:
  python main.py Mrk817 --mpi 2
  python main.py Mrk817 --pipeline width_constrained --dry-run
  python main.py Mrk817 --steps mica_r1,mica_r2 --mpi 2
  python main.py Mrk817 --dry-run
"""
from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

DEFAULT_STEPS = ("prep", "mica", "lag", "tau", "ffa", "distance")
VALID_STEPS = DEFAULT_STEPS + ("iccf",)

STEP_ALIASES = {
    "all": list(DEFAULT_STEPS),
    "data": ["prep"],
    "prepare": ["prep"],
    "run_mica": ["mica"],
    "mica_post": ["mica"],
    "summary": ["lag"],
    "sum": ["lag"],
    "sum_mica": ["lag"],
    "lambda": ["tau"],
    "lambda_tau": ["tau"],
    "fit_tau": ["tau"],
    "flux": ["ffa"],
    "fvg": ["ffa"],
    "nuc": ["ffa"],
    "sed": ["ffa"],
    "dist": ["distance"],
    "hubble": ["distance"],
}

# Legacy default pipeline steps (when config has no "pipelines:" key).
LEGACY_DEFAULT_PIPELINE = [
    {"name": "prep", "script": "prep_data.py"},
    {"name": "mica", "script": "run_mica.py", "config_section": "mica_fit", "mpi": True},
    {"name": "lag", "script": "sum_mica_runs.py", "config_section": "mica_lag_summary"},
    {"name": "tau", "script": "fit_tau.py", "config_section": "lambda_tau"},
    {"name": "ffa", "script": "run_ffa.py", "config_section": "fflux", "mpi": True},
    {"name": "distance", "script": "disk_distance.py", "config_section": "hubble_distance"},
]

# Mapping from legacy --steps filter names to the pipeline step names.
# This maps old hardcoded step names (e.g. "mica") to the default pipeline step names.
LEGACY_STEP_TO_PIPELINE = {
    "prep": "prep",
    "mica": "mica",
    "lag": "lag",
    "tau": "tau",
    "ffa": "ffa",
    "distance": "distance",
    "iccf": "iccf",
}


class PipelineError(RuntimeError):
    pass


def resolve_source(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def resolve_config(source_dir: Path, cfg: str | Path | None) -> Path:
    if cfg is None:
        return source_dir / "config" / "source_config.yaml"
    p = Path(cfg).expanduser()
    return p.resolve() if p.is_absolute() else (source_dir / p).resolve()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def split_items(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None:
        return list(default)
    return [x for chunk in value.split(",") for x in chunk.split() if x]


def quote_cmd(cmd: list[str]) -> str:
    return shlex.join(str(x) for x in cmd)


def mpi_wrap(cmd: list[str], mpi: int, mpiexec: str) -> list[str]:
    if int(mpi) <= 1:
        return cmd
    return [mpiexec, "-n", str(int(mpi)), *cmd]


def script(name: str) -> str:
    path = SRC / name
    if not path.exists():
        raise PipelineError(f"missing script: {path}")
    return str(path)


# ── step command builders ──────────────────────────────────────────────


def _build_prep_data(args, source_dir, cfg_path, step_cfg):
    return [args.python, script("prep_data.py"), str(source_dir)]


def _build_run_mica(args, source_dir, cfg_path, step_cfg):
    section = step_cfg.get("config_section", "mica_fit")
    cmd = [
        args.python, script("run_mica.py"), str(source_dir),
        "--config-section", section,
    ]
    if args.post_only:
        cmd.append("--post-only")
    return cmd


def _build_sum_mica(args, source_dir, cfg_path, step_cfg):
    section = step_cfg.get("config_section", "mica_lag_summary")
    cmd = [
        args.python, script("sum_mica_runs.py"), str(source_dir),
        "--config-section", section,
    ]
    if args.config:
        cmd.extend(["--config", str(cfg_path)])
    return cmd


def _build_fit_tau(args, source_dir, cfg_path, step_cfg):
    section = step_cfg.get("config_section", "lambda_tau")
    cmd = [
        args.python, script("fit_tau.py"), str(source_dir),
        "--config-section", section,
    ]
    if args.config:
        cmd.extend(["--config", str(cfg_path)])
    if args.tau_nwalkers is not None:
        cmd.extend(["--nwalkers", str(args.tau_nwalkers)])
    if args.tau_nsteps is not None:
        cmd.extend(["--nsteps", str(args.tau_nsteps)])
    if args.tau_burn_in is not None:
        cmd.extend(["--burn-in", str(args.tau_burn_in)])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    return cmd


def _build_ffa(args, source_dir, cfg_path, step_cfg):
    section = step_cfg.get("config_section", "fflux")
    cmd = [
        args.python, script("run_ffa.py"), str(source_dir),
        "--config-section", section,
    ]
    if args.config:
        cmd.extend(["--config", str(cfg_path)])
    if args.ffa_steps:
        cmd.extend(["--steps", args.ffa_steps])
    return cmd


def _build_ffa_raw(args, source_dir, cfg_path, step_cfg):
    section = step_cfg.get("config_section", "ffa_raw")
    cmd = [
        args.python, script("ffa_raw.py"), str(source_dir),
        "--config-section", section,
    ]
    if args.config:
        cmd.extend(["--config", str(cfg_path)])
    return cmd


def _build_distance(args, source_dir, cfg_path, step_cfg):
    section = step_cfg.get("config_section", "hubble_distance")
    cmd = [
        args.python, script("disk_distance.py"), str(source_dir),
        "--config-section", section,
    ]
    if args.distance_ncores is not None:
        cmd.extend(["--ncores", str(args.distance_ncores)])
    if args.distance_steps is not None:
        cmd.extend(["--steps", str(args.distance_steps)])
    if args.no_distance_mcmc:
        cmd.append("--no-mcmc")
    return cmd


def _build_iccf(args, source_dir, cfg_path, step_cfg):
    return [args.python, script("run_iccf.py"), str(source_dir)]


STEP_BUILDERS: dict[str, Callable] = {
    "prep_data.py": _build_prep_data,
    "run_mica.py": _build_run_mica,
    "sum_mica_runs.py": _build_sum_mica,
    "fit_tau.py": _build_fit_tau,
    "run_ffa.py": _build_ffa,
    "ffa_raw.py": _build_ffa_raw,
    "disk_distance.py": _build_distance,
    "run_iccf.py": _build_iccf,
}


# ── pipeline resolution ────────────────────────────────────────────────


def _read_pipeline_config(source_dir: Path, cfg_path: Path) -> dict[str, list[dict]]:
    """Return ``pipelines:`` block from config, or {} if absent."""
    return read_yaml(cfg_path).get("pipelines") or {}


def _filter_step_list(pipeline_steps: list[dict], wanted: list[str]) -> list[dict]:
    """Keep only steps whose name is in *wanted*, preserving pipeline order."""
    wanted_set = set(wanted)
    return [s for s in pipeline_steps if s["name"] in wanted_set]


def _legacy_step_names_to_pipeline(wanted: list[str]) -> list[str]:
    """Map legacy step names (from --steps) to pipeline step names."""
    mapped = []
    for name in wanted:
        if name in LEGACY_STEP_TO_PIPELINE:
            mapped.append(LEGACY_STEP_TO_PIPELINE[name])
        else:
            mapped.append(name)
    return mapped


def build_step_list(args, source_dir: Path, cfg_path: Path) -> list[dict]:
    """Resolve the ordered list of step descriptors to execute.

    1. If the config has a ``pipelines:`` block, use the named pipeline.
    2. Otherwise, synthesise the legacy default pipeline.
    3. Apply ``--steps`` as a name filter on top.
    """
    all_pipelines = _read_pipeline_config(source_dir, cfg_path)

    if all_pipelines:
        pipeline_name = args.pipeline or "default"
        if pipeline_name not in all_pipelines:
            available = ", ".join(sorted(all_pipelines))
            raise PipelineError(
                f"unknown pipeline {pipeline_name!r}; available: {available}"
            )
        pipeline_steps = list(all_pipelines[pipeline_name])
    else:
        pipeline_steps = list(LEGACY_DEFAULT_PIPELINE)

    # Build valid step name set from the resolved pipeline
    valid_names = {s["name"] for s in pipeline_steps}

    # Parse --steps filter
    if args.steps is None:
        wanted = [s["name"] for s in pipeline_steps]
    else:
        raw = [x for chunk in args.steps.split(",") for x in chunk.split() if x]
        wanted = []
        for r in raw:
            key = r.strip().lower().replace("-", "_")
            expanded = STEP_ALIASES.get(key, [key])
            mapped = _legacy_step_names_to_pipeline(expanded)
            for name in mapped:
                if name not in valid_names:
                    available = ", ".join(sorted(valid_names))
                    raise PipelineError(
                        f"unknown step {r!r}; available in pipeline: {available}"
                    )
                if name not in wanted:
                    wanted.append(name)

    if not wanted:
        raise PipelineError("no pipeline steps selected")
    return _filter_step_list(pipeline_steps, wanted)


def build_commands(args, source_dir: Path, cfg_path: Path, steps: list[dict]) -> list[tuple[str, list[str]]]:
    py = args.python
    commands: list[tuple[str, list[str]]] = []

    for step_cfg in steps:
        name = step_cfg["name"]
        script_name = step_cfg["script"]
        builder = STEP_BUILDERS.get(script_name)
        if builder is None:
            raise PipelineError(f"no builder for script {script_name!r} (step {name!r})")

        cmd = builder(args, source_dir, cfg_path, step_cfg)
        if step_cfg.get("mpi", False):
            cmd = mpi_wrap(cmd, args.mpi, args.mpiexec)

        # Append extra_args from pipeline step definition as CLI flags.
        for key, val in (step_cfg.get("extra_args") or {}).items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(val, bool):
                if val:
                    cmd.append(flag)
            elif isinstance(val, list):
                for v in val:
                    cmd.extend([flag, str(v)])
            else:
                cmd.extend([flag, str(val)])

        commands.append((name, cmd))

    return commands


# ── execution helpers ──────────────────────────────────────────────────


def make_log_path(source_dir: Path, log_dir_arg: str | Path | None, dry_run: bool) -> Path | None:
    if dry_run:
        return None
    if log_dir_arg:
        log_dir = Path(log_dir_arg).expanduser()
        log_dir = log_dir.resolve() if log_dir.is_absolute() else (ROOT / log_dir).resolve()
    else:
        log_dir = source_dir / "results" / "pipeline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"pipeline_{stamp}.log"


def write_line(log_fh, text: str = "") -> None:
    print(text, flush=True)
    if log_fh is not None:
        log_fh.write(text + "\n")
        log_fh.flush()


def run_command(step: str, cmd: list[str], log_fh, dry_run: bool) -> int:
    write_line(log_fh)
    write_line(log_fh, "=" * 80)
    write_line(log_fh, f"STEP: {step}")
    write_line(log_fh, f"CMD : {quote_cmd(cmd)}")
    write_line(log_fh, "=" * 80)

    if dry_run:
        return 0

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        write_line(log_fh, line)

    rc = proc.wait()
    write_line(log_fh, f"STEP {step} finished with exit code {rc}")
    return rc


def preflight(source_dir: Path, cfg_path: Path) -> None:
    if not source_dir.exists():
        raise PipelineError(f"source directory not found: {source_dir}")
    if not cfg_path.exists():
        raise PipelineError(f"config not found: {cfg_path}")
    if not SRC.exists():
        raise PipelineError(f"src directory not found: {SRC}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the src/ analysis pipeline in one command.")
    parser.add_argument("source", help="source directory/name, e.g. Mrk817")
    parser.add_argument("--config", help="YAML config path; relative paths are resolved from the source directory")
    parser.add_argument(
        "--pipeline",
        default=None,
        help=(
            "Named pipeline from config['pipelines']. "
            "Default: 'default' if pipelines block exists, else legacy hardcoded pipeline."
        ),
    )
    parser.add_argument(
        "--steps",
        help=(
            "comma/space-separated step names to filter the pipeline. "
            "Default: all steps in the selected pipeline."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failed step")
    parser.add_argument("--log-dir", help="log directory; default: <source>/results/pipeline_logs")

    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    parser.add_argument("--mpi", type=int, default=1, help="MPI ranks for steps with mpi:true; default: 1")
    parser.add_argument("--mpiexec", default="mpiexec", help="MPI launcher; default: mpiexec")
    parser.add_argument("--post-only", action="store_true", help="pass --post-only to src/run_mica.py")

    parser.add_argument("--ffa-steps", help="steps passed to src/run_ffa.py, e.g. prep,dered,fit,summ,fvg,nuc")
    parser.add_argument("--tau-nwalkers", type=int, help="override fit_tau.py --nwalkers")
    parser.add_argument("--tau-nsteps", type=int, help="override fit_tau.py --nsteps")
    parser.add_argument("--tau-burn-in", type=int, help="override fit_tau.py --burn-in")
    parser.add_argument("--seed", type=int, help="override fit_tau.py --seed")
    parser.add_argument("--distance-ncores", type=int, help="override disk_distance.py --ncores")
    parser.add_argument("--distance-steps", type=int, help="override disk_distance.py --steps")
    parser.add_argument("--no-distance-mcmc", action="store_true", help="pass --no-mcmc to disk_distance.py")

    args = parser.parse_args()

    try:
        source_dir = resolve_source(args.source)
        cfg_path = resolve_config(source_dir, args.config)
        preflight(source_dir, cfg_path)
        steps = build_step_list(args, source_dir, cfg_path)
        commands = build_commands(args, source_dir, cfg_path, steps)
        log_path = make_log_path(source_dir, args.log_dir, args.dry_run)

        log_fh = None if log_path is None else log_path.open("w", encoding="utf-8")
        try:
            pipeline_name = args.pipeline or "default (legacy)"
            write_line(log_fh, "analysis pipeline")
            write_line(log_fh, f"root     : {ROOT}")
            write_line(log_fh, f"source   : {source_dir}")
            write_line(log_fh, f"config   : {cfg_path}")
            write_line(log_fh, f"pipeline : {pipeline_name}")
            write_line(log_fh, f"steps    : {', '.join(s['name'] for s in steps)}")
            if log_path is not None:
                write_line(log_fh, f"log      : {log_path}")
            if args.dry_run:
                write_line(log_fh, "mode     : dry-run")

            failed: list[tuple[str, int]] = []
            for step_name, cmd in commands:
                rc = run_command(step_name, cmd, log_fh, args.dry_run)
                if rc != 0:
                    failed.append((step_name, rc))
                    if not args.keep_going:
                        break

            if failed:
                msg = "; ".join(f"{s}={rc}" for s, rc in failed)
                write_line(log_fh, f"FAILED: {msg}")
                return failed[0][1] or 1

            write_line(log_fh)
            write_line(log_fh, "pipeline finished successfully")
            return 0
        finally:
            if log_fh is not None:
                log_fh.close()

    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
