#!/usr/bin/env python3
"""Run prepared fast-residual cases through the existing MICA pipeline, serially.

Examples (use the project's mica2 Python environment):
    python test/mica_smooth/run.py all --case responses_s20 --model gaussian --dry-run
    python test/mica_smooth/run.py post --case all --model all
    python test/mica_smooth/run.py sed --case all_s20 --model gamma

Initial total-light FFA is skipped. Post-MICA comp0 reconstruction and FFA/SED
are part of every all/post run; sed runs just that downstream step.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CASES = ("responses_s20", "responses_s40", "all_s20", "all_s40")
STAGES = {"gamma": "mica_gamma2_free", "gaussian": "mica_gaussian2_free"}
POST = ("collect", "corner", "transfer", "lagfit")
SED_STEPS = "prep,dered,fit,summ,fvg,sedfit"


@dataclass(frozen=True)
class Command:
    label: str
    argv: tuple[str, ...]


def selected(values: list[str], available: tuple[str, ...]) -> list[str]:
    return list(available) if "all" in values else list(dict.fromkeys(values))


def read_config(case: Path) -> dict:
    path = case / "config/source_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing prepared case config: {path}; run prepare.py first")
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if (cfg.get("smoothing") or {}).get("output") != "highpass_residual_plus_original_mean":
        raise ValueError(
            f"{path}: expected original - Gaussian smooth + original mean. "
            "Old low-pass inputs cannot be fitted by this runner; regenerate "
            "inputs with the corrected prepare.py in a new --output-dir."
        )
    return cfg


def stage_config(cfg: dict, model: str) -> dict:
    stage = STAGES[model]
    fit = cfg.get(stage)
    if not isinstance(fit, dict) or not fit:
        raise ValueError(f"Missing MICA stage {stage}")
    if str(fit.get("type_tf")) != model or int(fit.get("ncomp", 2)) not in (1, 2):
        raise ValueError(f"{stage} must use one or two {model} components")
    return fit


def pipeline_commands(case: Path, model: str, action: str, python: str,
                      cfg: dict | None = None) -> list[Command]:
    cfg = read_config(case) if cfg is None else cfg
    if action == "sed":
        return sed_commands(case, model, cfg, python)
    actions = ("mica", *POST) if action == "all" else POST if action == "post" else ()
    commands = []
    for step in actions:
        if step == "lagfit":
            commands.extend(sed_commands(case, model, cfg, python))
        # main.py currently omits collect from its --stage allowlist.
        if step == "collect":
            argv = (python, "-B", str(ROOT / "src/collect_mica_results.py"), str(case), STAGES[model])
        else:
            argv = (python, "-B", str(ROOT / "main.py"), str(case), step, "--stage", STAGES[model])
        commands.append(Command(step, argv))
    return commands


def make_sed_config(cfg: dict, model: str) -> dict:
    """Adapt the selected stage to the existing full-posterior branch interface."""
    out = deepcopy(cfg)
    fit = deepcopy(stage_config(cfg, model))
    ff = deepcopy(cfg.get("fflux_comp0") or {})
    if not ff:
        raise ValueError("Post-MICA comp0 FFA/SED requires fflux_comp0 in the prepared config")
    source = str(cfg["source"])
    driver = str(fit["driver"])
    prepared = Path(str(fit.get("prepared_dir", "prepared")))
    result = Path(str(fit["output_root"])) / f"{int(fit.get('ncomp', 2))}comp/result"
    fit["posterior_branches"] = {
        "enabled": True, "seed": 12345, "decompose_nsamp": 3000,
        "mpi_per_band": 1, "max_parallel": 1,
        "branches": {"full": {
            "label": f"Full posterior, {fit.get('component_order', 'native')} component order",
            "ranges": {"tau0": [None, None]},
        }},
    }
    out["mica_round1"] = fit
    out["mica_stages"] = ["mica_round1"]
    ff.update({
        "source": source, "component": {"name": "comp0"},
        "bands": [driver, *fit["responses"]],
        "wavelengths": deepcopy(cfg["bands"]),
        "steps": SED_STEPS.split(","),
        "posterior_branch_out_template": str(result / "comp0/fflux"),
    })
    # The case configuration records already dereddened/host-subtracted input.
    # Preserve that configuration, including its explicit zero extra extinction.
    ff["mica"] = {
        "driver": driver, "driver_file": str(prepared / f"{source}_{driver}.txt"),
        "component_index": 0, "response_time_support": "driver",
        "run_template": fit.get("run_template", "run_{driver}_to_{band}_{ncomp}comp_{type_tf}"),
    }
    out["fflux_comp0"] = ff
    return out


def sed_commands(case: Path, model: str, cfg: dict, python: str) -> list[Command]:
    path = case / "config" / f"{model}_sed.yaml"
    mpi = str(Path(python).with_name("mpiexec"))
    ranks = str(int((cfg.get("pipeline") or {}).get("ffa_mpi", 2)))
    return [
        Command("decompose_full", (python, "-B", str(ROOT / "src/decompose_mica_branches.py"),
                str(case), "--config", str(path), "--branch", "full", "--skip-complete")),
        Command("sed", (mpi, "-n", ranks, python, "-B", str(ROOT / "src/run_ffa.py"),
                str(case), "--config", str(path), "--config-section", "fflux_comp0",
                "--posterior-branch", "full", "--steps", SED_STEPS)),
    ]


def sed_root(case: Path, cfg: dict) -> Path:
    return case / cfg["fflux_comp0"]["posterior_branch_out_template"]


def sed_outputs(case: Path, cfg: dict) -> list[Path]:
    ff = cfg["fflux_comp0"]
    source, nbands = ff["source"], len(ff["bands"])
    analysis = sed_root(case, cfg) / "analysis"
    return [analysis / name for name in (
        f"{source}_comp0_{nbands}band_components.csv",
        f"{source}_comp0_{nbands}band_flux_posterior.txt",
        f"{source}_comp0_sed_fit_summary.csv", "sed_model_posterior.txt",
        f"{source}_comp0_sed_points.csv", f"{source}_comp0_sed_models.png",
        f"{source}_comp0_sed_models.pdf",
    )]


def sed_complete(case: Path, cfg: dict) -> bool:
    return all(p.is_file() and p.stat().st_size > 0 for p in sed_outputs(case, cfg))


def check_sed_start(case: Path, cfg: dict) -> None:
    sample = sed_root(case, cfg) / "fit/sample.txt"
    if sample.is_file() and sample.stat().st_size > 0:
        raise RuntimeError(f"Incomplete SED has existing FFA samples: {sample}. "
                           "Inspect/recover that fit using run_ffa.py before retrying; "
                           "this runner will not overwrite or guess an FFA restart.")


def step_outputs(case: Path, cfg: dict, model: str, step: str) -> list[Path]:
    """Figure products required by the existing main.py plotting commands."""
    fit = stage_config(cfg, model)
    result = case / fit["output_root"] / f"{int(fit.get('ncomp', 2))}comp/result"
    source = cfg["source"]
    if step == "corner":
        return [result / "corner" / f"{source}_{band}_{int(fit.get('ncomp', 2))}comp_{model}_corner.{ext}"
                for band in fit["responses"] for ext in ("png", "pdf")]
    if step == "transfer":
        stems = [f"{source}_{band}_{component}_transfer"
                 for band in fit["responses"] for component in ("comp0", "total")]
        stems.append(f"{source}_transfer_functions_overview")
        return [result / "transfer_functions" / f"{stem}{suffix}.{ext}"
                for stem in stems for suffix in ("", "_median_parameters")
                for ext in ("png", "pdf")]
    if step == "lagfit":
        configured = (cfg.get("lambda_tau") or {}).get("result_dir")
        lag = case / configured if configured else result / "comp0/lag"
        return [lag / f"tau_tf0.{ext}" for ext in ("png", "pdf")]
    return []


def check_step_outputs(case: Path, cfg: dict, model: str, step: str) -> None:
    missing = [str(p) for p in step_outputs(case, cfg, model, step)
               if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing outputs after {step}: " + ", ".join(missing))


@contextmanager
def postprocess_slot(command: Command):
    """Serialize post-processing across cases when the controller supplies a lock."""
    configured = os.environ.get("MICA_SMOOTH_POST_LOCK")
    if command.label == "mica" or not configured:
        yield
        return
    path = Path(configured)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        print(f"Waiting for post-processing slot: {command.label}", flush=True)
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            print(f"Acquired post-processing slot: {command.label}", flush=True)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def execute(command: Command, case: Path, model: str, dry_run: bool) -> None:
    print("$ " + shlex.join(command.argv), flush=True)
    if dry_run:
        return
    logs = case / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{model}_{command.label}.log"
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(logs / "mplconfig"),
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1"})
    print(f"log: {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] "
                     + shlex.join(command.argv) + "\n")
        handle.flush()
        with postprocess_slot(command):
            subprocess.run(command.argv, cwd=ROOT, env=env, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)


def run_case(case: Path, model: str, action: str, dry_run: bool) -> None:
    cfg = read_config(case)
    stage_config(cfg, model)
    python = sys.executable
    sed_cfg = make_sed_config(cfg, model)
    completed_sed = not dry_run and sed_complete(case, sed_cfg)
    if not dry_run and not completed_sed:
        check_sed_start(case, sed_cfg)
        mpi = Path(python).with_name("mpiexec")
        if not mpi.is_file():
            raise FileNotFoundError(f"Missing MPI launcher beside active Python: {mpi}")
    print(f"\n[{case.name} / {model}]", flush=True)
    for command in pipeline_commands(case, model, action, python, cfg):
        if command.label in {"decompose_full", "sed"} and completed_sed:
            if command.label == "decompose_full":
                print("Skip comp0 FFA/SED: all core SED outputs already exist.", flush=True)
            continue
        if command.label == "decompose_full":
            config_path = case / "config" / f"{model}_sed.yaml"
            if dry_run:
                print(f"Would write full-posterior SED config: {config_path}", flush=True)
            else:
                with config_path.open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(sed_cfg, handle, sort_keys=False)
        execute(command, case, model, dry_run)
        if not dry_run:
            check_step_outputs(case, cfg, model, command.label)
        if command.label == "sed" and not dry_run and not sed_complete(case, sed_cfg):
            missing = [str(p) for p in sed_outputs(case, sed_cfg)
                       if not p.is_file() or p.stat().st_size == 0]
            raise RuntimeError("Missing SED outputs after FFA: " + ", ".join(missing))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all", "post", "sed"), default="all")
    parser.add_argument("--cases-dir", type=Path, default=HERE / "cases",
                        help="prepared cases directory (matches prepare.py --output-dir)")
    parser.add_argument("--case", nargs="+", choices=(*CASES, "all"), default=["all"])
    parser.add_argument("--model", nargs="+", choices=(*STAGES, "all"), default=["all"])
    # Compatibility with the earlier documented command; SED is always included.
    parser.add_argument("--with-sed", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="print commands without writes or fitting")
    parser.add_argument("--ncomp", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    STAGES.update({model: f"mica_{model}{args.ncomp}_free" for model in STAGES})
    # This runner is serial; orchestrate.py manages concurrent case runners.
    for name in selected(args.case, CASES):
        for model in selected(args.model, tuple(STAGES)):
            run_case(args.cases_dir.expanduser().resolve() / name, model, args.action, args.dry_run)


if __name__ == "__main__":
    main()
