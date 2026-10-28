#!/usr/bin/env python3
"""Single-source FFA -> MICA -> native-comp0 products -> lag -> distance pipeline.

The optional ``diagnostics`` action builds publication-facing post-processing
products from completed sampler outputs; it does not rerun FFA or MICA. ICCF is
run only through the explicit ``iccf`` action. When ``all`` is combined with
``--stage``, the shared prep/FFA inputs are reused and that MICA stage plus all
of its downstream products (including diagnostics) are completed in one call.

The ``corner`` action reads completed posterior products and writes one corner
figure per response band directly into each selected stage's ``result`` folder.
It also runs automatically after collection in both default pipeline flows.

The ``transfer`` action writes both transfer-function representations for each
selected MICA stage: pointwise posterior quantiles and analytic kernels built
from the posterior-median parameters. Project-wide Hubble diagrams remain a
separate cross-source calculation and are dispatched with ``python main.py
hubble`` using ``joint_hubble_config.yaml`` (or an explicit ``--config``).
The frozen shared-SED, fixed-f0 analysis is ``python main.py shared-hubble``.
"""
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
    "prep", "ffa", "mica", "collect", "corner", "decompose", "transfer",
    "ffa_comp0", "lagfit", "distance",
)
STAGE_FLOW = (
    "mica", "collect", "corner", "decompose", "ffa_comp0",
    "lagfit", "distance", "diagnostics",
)
OPTIONAL_ACTIONS = ("iccf", "diagnostics")
ACTIONS = {"all", "fresh", "clean", "ffa_branch", *FLOW, *OPTIONAL_ACTIONS}
STAGE_ACTIONS = {
    "all", "mica", "collect", "corner", "transfer", "ffa_comp0", "ffa_branch",
    "iccf", "diagnostics", "lagfit", "distance",
}


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


def mica_stage_names(cfg: dict) -> list[str]:
    stages = cfg.get("mica_stages")
    if stages is None:
        return ["mica_round1"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("mica_stages must be a non-empty list")
    names = [str(stage) for stage in stages]
    missing = [stage for stage in names if not (cfg.get(stage) or {})]
    if missing:
        raise ValueError("mica_stages references missing sections: " + ", ".join(missing))
    return names


def analysis_stages(cfg: dict, selected_stage: str | None = None) -> list[str]:
    """MICA stages whose native comp0 products are analysed downstream."""
    if selected_stage is None:
        return mica_stage_names(cfg)
    selected_stage = str(selected_stage).strip()
    if not selected_stage or not (cfg.get(selected_stage) or {}):
        raise ValueError(f"unknown MICA config section: {selected_stage!r}")
    return [selected_stage]


def ncomp_from_config(fit: dict) -> int:
    if "ncomp" in fit:
        return int(fit["ncomp"])
    nc = fit.get("number_component", [2, 2])
    return int(nc[0] if isinstance(nc, list) else nc)


def mica_run_root(src: Path, fit: dict) -> Path:
    root = rel(src, fit.get("output_root", "runs/mica"))
    model = str(fit.get("type_tf", "gaussian")).strip().lower()
    return root if model.isdigit() else root / f"{ncomp_from_config(fit)}comp"


def clean(src: Path, purge_mica_runs: bool = False) -> None:
    cfg = read_config(src)
    ff = cfg.get("fflux", {}) or {}
    iccf = cfg.get("iccf", {}) or {}
    lag = cfg.get("lambda_tau", {}) or {}
    hubble = cfg.get("hubble_distance", {}) or {}

    targets = [rel(src, ff.get("out", "fflux"))]
    for stage in mica_stage_names(cfg):
        mica = cfg.get(stage, {}) or {}
        run_root = mica_run_root(src, mica)
        output_root = rel(src, mica.get("output_root", "runs/mica"))
        if purge_mica_runs:
            targets.append(run_root)
        else:
            # Preserve expensive completed runs; remove only derived tables.
            targets.extend([run_root / "result", output_root / "result"])

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


def plot_transfer_figures(src: Path, stage: str) -> None:
    """Render the posterior-band and median-kernel transfer-function products."""
    python("plot_mica_transfer.py", src, "--stage", stage)
    python(
        "plot_mica_transfer.py", src,
        "--stage", stage,
        "--median-parameters",
    )


def execute(src: Path, action: str, selected_stage: str | None = None) -> None:
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
        args = () if selected_stage is None else ("--stage", selected_stage)
        python("run_mica.py", src, *args)

    elif action == "collect":
        cfg = read_config(src)
        for stage in analysis_stages(cfg, selected_stage):
            python("collect_mica_results.py", src, stage)

    elif action == "corner":
        cfg = read_config(src)
        for stage in analysis_stages(cfg, selected_stage):
            python("plot_mica_corner.py", src, "--stage", stage)

    elif action == "decompose":
        print(
            "skip conditional decomposition: MICA already wrote the full-posterior "
            "native component curves used downstream"
        )

    elif action == "transfer":
        cfg = read_config(src)
        for stage in analysis_stages(cfg, selected_stage):
            plot_transfer_figures(src, stage)

    elif action in {"ffa_comp0", "ffa_branch"}:
        cfg = read_config(src)
        mpi_n = int((cfg.get("pipeline", {}) or {}).get("ffa_mpi", 2))
        ff_comp0 = dict(cfg.get("fflux_comp0", {}) or {})
        component_steps = ff_comp0.get(
            "steps",
            ff_comp0.get(
                "posterior_branch_steps",
                ["prep", "dered", "fit", "summ", "fvg", "distance_flux", "sedfit"],
            ),
        )
        step_arg = (
            component_steps
            if isinstance(component_steps, str)
            else ",".join(str(step) for step in component_steps)
        )
        for stage in analysis_stages(cfg, selected_stage):
            run([
                compatible_mpiexec(), "-n", mpi_n, sys.executable,
                SRC / "run_ffa.py", src,
                "--config-section", "fflux_comp0",
                "--mica-stage", stage,
                "--mica-component", "comp0",
                "--steps", step_arg,
            ])

    elif action == "iccf":
        cfg = read_config(src)
        for stage in analysis_stages(cfg, selected_stage):
            python("run_iccf.py", src, "--stage", stage)

    elif action == "diagnostics":
        cfg = read_config(src)
        for stage in analysis_stages(cfg, selected_stage):
            plot_transfer_figures(src, stage)
            python("results_diagnostics.py", src, "--stage", stage)
            python("sed_u_diagnostics.py", src, "--stage", stage)

    elif action == "lagfit":
        cfg = read_config(src)
        for stage in analysis_stages(cfg, selected_stage):
            python("fit_tau.py", src, "--stage", stage)

    elif action == "distance":
        cfg = read_config(src)
        for stage in analysis_stages(cfg, selected_stage):
            python("disk_distance.py", src, "--stage", stage)

    else:
        raise ValueError(action)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "shared-hubble":
        # Exact shared-SED FFA -> one distance per source -> fixed-f0 H0.
        run([sys.executable, "-m", "src.shared_sed_hubble", *argv[1:]])
        return
    if argv and argv[0] == "hubble":
        # This is intentionally project-level: the dedicated script reads the
        # configured set of independent AGN only after their distance products
        # exist, instead of rerunning the joint fit after every single source.
        run([sys.executable, SRC / "joint_hubble.py", *argv[1:]])
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("action", nargs="?", default="all", choices=sorted(ACTIONS))
    parser.add_argument(
        "--purge-mica-runs",
        action="store_true",
        help="allow clean/fresh to delete completed MICA run directories",
    )
    parser.add_argument(
        "--stage",
        help=(
            "run only one MICA config section for stage-aware actions; "
            "for example mica_mix220"
        ),
    )
    args = parser.parse_args(argv)

    src = source_dir(args.source)

    if args.stage and args.action not in STAGE_ACTIONS:
        parser.error(
            "--stage is valid only with: " + ", ".join(sorted(STAGE_ACTIONS))
        )

    if args.action == "clean":
        clean(src, purge_mica_runs=args.purge_mica_runs)
        return

    if args.action == "fresh":
        if not args.purge_mica_runs:
            parser.error("fresh reruns MICA and requires --purge-mica-runs")
        clean(src, purge_mica_runs=True)
        actions = FLOW
    elif args.action == "all":
        actions = STAGE_FLOW if args.stage else FLOW
    else:
        actions = (args.action,)

    for action in actions:
        print(f"\n== {action} ==", flush=True)
        execute(src, action, args.stage)


if __name__ == "__main__":
    main()
