#!/usr/bin/env python3
"""Single-source FFA -> MICA -> branch products -> lag -> distance pipeline."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
FLOW = (
    "prep", "ffa", "mica", "collect", "decompose", "ffa_branch", "lagfit", "distance"
)
OPTIONAL_ACTIONS = ("iccf",)
ACTIONS = {"all", "fresh", "clean", *FLOW, *OPTIONAL_ACTIONS}


def source_dir(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_config(src: Path) -> dict:
    with (src / "config" / "source_config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def rel(src: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else src / path


def run(command: list[str | Path]) -> None:
    command = [str(item) for item in command]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def python(script: str, src: Path, *args: str) -> None:
    run([sys.executable, SRC / script, src, *args])


def compatible_mpiexec() -> Path | str:
    candidate = Path(sys.executable).with_name("mpiexec")
    return candidate if candidate.exists() else "mpiexec"


def posterior_branch_ids(cfg: dict) -> list[str]:
    mica = dict(cfg.get("mica_round1", {}) or {})
    branch_cfg = dict(mica.get("posterior_branches", {}) or {})
    definitions = dict(branch_cfg.get("branches", {}) or {})
    return [
        str(branch_id)
        for branch_id, definition in definitions.items()
        if bool((definition or {}).get("enabled", True))
    ]


def ncomp_from_config(fit: dict) -> int:
    if "ncomp" in fit:
        return int(fit["ncomp"])
    nc = fit.get("number_component", [2, 2])
    return int(nc[0] if isinstance(nc, list) else nc)


def clean(src: Path, purge_mica_runs: bool = False) -> None:
    cfg = read_config(src)
    ff = cfg.get("fflux", {}) or {}
    mica = cfg.get("mica_round1", {}) or {}
    iccf = cfg.get("iccf", {}) or {}
    lag = cfg.get("lambda_tau", {}) or {}
    hubble = cfg.get("hubble_distance", {}) or {}

    mica_root = rel(src, mica.get("output_root", "runs/mica"))
    targets = [rel(src, ff.get("out", "fflux"))]
    if purge_mica_runs:
        targets.append(mica_root)
    else:
        # Preserve expensive completed MICA runs. Only remove regenerated
        # collection/lag products, including the legacy result location.
        targets.extend([
            mica_root / f"{ncomp_from_config(mica)}comp" / "result",
            mica_root / "result",
        ])

    # Optional or legacy standalone outputs. The current lag/distance outputs
    # live inside the MICA run root and are removed with output_root above.
    for section in (iccf, lag, hubble):
        if section.get("output_dir"):
            targets.append(rel(src, section["output_dir"]))

    for path in dict.fromkeys(targets):
        if path.exists():
            print("remove", path)
            shutil.rmtree(path)

    invalid = src / "results" / "invalid_flux_points.txt"
    if invalid.exists():
        invalid.unlink()


def execute(src: Path, action: str) -> None:
    if action == "prep":
        python("prep_data.py", src)

    elif action == "ffa":
        cfg = read_config(src)
        mpi_n = int((cfg.get("pipeline", {}) or {}).get("ffa_mpi", 2))
        ff = dict(cfg.get("fflux", {}) or {})
        configured_steps = ff.get("steps", "all")
        step_arg = (
            configured_steps
            if isinstance(configured_steps, str)
            else ",".join(str(step) for step in configured_steps)
        )
        run([compatible_mpiexec(), "-n", mpi_n, sys.executable, SRC / "run_ffa.py", src, "--steps", step_arg])

    elif action == "mica":
        python("run_mica.py", src)

    elif action == "collect":
        python("collect_mica_results.py", src, "r1")

    elif action == "decompose":
        python(
            "decompose_mica_branches.py",
            src,
            "--all-branches",
            "--skip-complete",
        )

    elif action == "ffa_branch":
        cfg = read_config(src)
        mpi_n = int((cfg.get("pipeline", {}) or {}).get("ffa_mpi", 2))
        branches = posterior_branch_ids(cfg)
        ff_branch = dict(cfg.get("fflux_comp0", {}) or {})
        branch_steps = ff_branch.get(
            "posterior_branch_steps",
            ["prep", "dered", "fit", "summ", "fvg", "sedfit"],
        )
        step_arg = (
            branch_steps
            if isinstance(branch_steps, str)
            else ",".join(str(step) for step in branch_steps)
        )
        if not branches:
            raise ValueError("no enabled posterior branches for component-0 FFA")
        for branch_id in branches:
            run([
                compatible_mpiexec(), "-n", mpi_n, sys.executable,
                SRC / "run_ffa.py", src,
                "--config-section", "fflux_comp0",
                "--posterior-branch", branch_id,
                "--steps", step_arg,
            ])

    elif action == "iccf":
        python("run_iccf.py", src)

    elif action == "lagfit":
        cfg = read_config(src)
        if posterior_branch_ids(cfg):
            python("fit_tau.py", src, "--all-branches")
        else:
            python("fit_tau.py", src)

    elif action == "distance":
        python("disk_distance.py", src)

    else:
        raise ValueError(action)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("action", nargs="?", default="all", choices=sorted(ACTIONS))
    parser.add_argument(
        "--purge-mica-runs",
        action="store_true",
        help="allow clean/fresh to delete completed MICA run directories",
    )
    args = parser.parse_args()

    src = source_dir(args.source)

    if args.action == "clean":
        clean(src, purge_mica_runs=args.purge_mica_runs)
        return

    if args.action == "fresh":
        if not args.purge_mica_runs:
            parser.error("fresh reruns MICA and requires --purge-mica-runs")
        clean(src, purge_mica_runs=True)
        actions = FLOW
    elif args.action == "all":
        actions = FLOW
    else:
        actions = (args.action,)

    for action in actions:
        print(f"\n== {action} ==", flush=True)
        execute(src, action)


if __name__ == "__main__":
    main()
