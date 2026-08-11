#!/usr/bin/env python3
"""Run MICA on nuclear-corrected total light curves.

Usage
-----
python src/run_mica.py SOURCE                  # all response bands
python src/run_mica.py SOURCE B                # one response band
python src/run_mica.py SOURCE post             # redo post-processing
python src/run_mica.py SOURCE B plot           # redraw decomposition plot

type_tf and ncomp are read from mica_round1 in source_config.yaml.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pymica
import yaml
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "output_root": "runs/mica",
    "prepared_dir": "fflux/nuclear_corrected",
    "ncomp": 2,
    "type_tf": "gaussian",
    "lag_limit": [-10.0, 80.0],
    "type_lag_prior": 0,
    "max_num_saves": 10000,
    "mpi_per_band": 2,
    "max_parallel": 5,
    "plot_decomp": True,
}

# map type_tf name to subfolder label used in run directory names
_TF_LABEL = {
    "gaussian": "gaussian",
    "tophat": "tophat",
    "gamma": "gamma",
    "exponential": "exp",
    "exp": "exp",
}

SETUP_KEYS = [
    "type_lag_prior",
    "lag_prior",
    "width_limit",
    "width_prior",
    "flag_lag_posivity",
    "flag_negative_resp",
    "flag_con_sys_err",
    "flag_line_sys_err",
    "flag_trend",
    "flag_gap",
    "gap_prior",
    "ptol",
    "num_particles",
    "thread_steps_factor",
    "new_level_interval_factor",
    "save_interval_factor",
    "lam",
    "beta",
    "max_num_levels",
    "nd_rec",
    "trec_ext",
]

ACTIONS = {"run", "post", "plot", "plotdecomp"}


def log(msg=""):
    if RANK == 0:
        print(msg, flush=True)


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def source_dir(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def rel(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def fit_config(cfg: dict) -> dict:
    fit = dict(DEFAULTS)
    fit.update(cfg.get("mica_round1", {}) or {})
    fit.setdefault("driver", cfg.get("reference_band", cfg.get("driver_band")))
    fit.setdefault("responses", cfg.get("response_bands", []))

    ncomp = int(fit["ncomp"])
    type_tf = str(fit["type_tf"]).lower()
    if ncomp < 1:
        raise ValueError(f"mica_round1.ncomp must be >= 1, got {ncomp}")
    if type_tf not in _TF_LABEL:
        raise ValueError(
            f"mica_round1.type_tf '{type_tf}' not recognised; "
            f"valid values: {list(_TF_LABEL)}"
        )
    fit["ncomp"] = ncomp
    fit["type_tf"] = type_tf
    return fit


def run_dir(src: Path, fit: dict, response: str) -> Path:
    ncomp = fit["ncomp"]
    tf_label = _TF_LABEL[fit["type_tf"]]
    name = f"run_{fit['driver']}_to_{response}_{ncomp}comp_{tf_label}"
    return src / fit["output_root"] / f"{ncomp}comp" / name


def copy_inputs(src: Path, source: str, fit: dict, response: str, rd: Path) -> None:
    pdir = rel(src, fit["prepared_dir"])
    inp = rd / "input"
    inp.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdir / f"{source}_{fit['driver']}.txt", inp / "cont.txt")
    shutil.copy2(pdir / f"{source}_{response}.txt", inp / "line.txt")


def load_data() -> dict:
    data = None
    if RANK == 0:
        data = {"set1": [np.loadtxt("input/cont.txt"), np.loadtxt("input/line.txt")]}
    return COMM.bcast(data, root=0)


def setup_kwargs(fit: dict, data: dict) -> dict:
    ncomp = fit["ncomp"]
    out = {
        "data": data,
        "type_tf": fit["type_tf"],
        "number_component": [ncomp, ncomp],
        "lag_limit": fit["lag_limit"],
        "max_num_saves": int(fit["max_num_saves"]),
    }
    for key in SETUP_KEYS:
        if fit.get(key) is not None:
            out[key] = fit[key]
    return out


def have_decomp(ncomp: int) -> bool:
    return bool(list(Path("data").glob(f"pline.txt_{ncomp}_comp*")))


def finish(model, fit: dict, post_run: bool) -> None:
    if post_run:
        log("MICA post_run()")
        model.post_run()
        COMM.Barrier()

    log("MICA decompose()")
    model.decompose()
    COMM.Barrier()

    if RANK == 0:
        log("MICA plot_results()")
        model.plot_results(
            doshow=False,
            tf_lag_range=fit["lag_limit"],
            hist_lag_range=fit["lag_limit"],
        )
        if fit.get("plot_decomp", True):
            log("MICA plot_decomp()")
            model.plot_decomp(doshow=False)
        log("MICA post_process()")
        model.post_process()
    COMM.Barrier()


def run_band(src: Path, cfg: dict, fit: dict, response: str, action: str) -> None:
    source = str(cfg.get("source", src.name))
    rd = run_dir(src, fit, response)

    if RANK == 0:
        copy_inputs(src, source, fit, response, rd)
    COMM.Barrier()

    os.chdir(rd)
    ncomp = fit["ncomp"]
    tf_label = _TF_LABEL[fit["type_tf"]]
    log("\n" + "=" * 70)
    log(f"{source}  {fit['driver']} -> {response}  {ncomp} {tf_label} component(s)  {action}")
    log(str(rd))
    log("=" * 70)

    model = pymica.gmodel()
    model.setup(**setup_kwargs(fit, load_data()))

    if action == "run":
        log("MICA run()")
        model.run()
        COMM.Barrier()
        finish(model, fit, post_run=False)
    elif action == "post":
        finish(model, fit, post_run=True)
    else:
        if not have_decomp(fit["ncomp"]):
            model.decompose()
            COMM.Barrier()
        if RANK == 0:
            log("MICA plot_decomp()")
            model.plot_decomp(doshow=False)
        COMM.Barrier()


def batches(values: list[str], size: int):
    for i in range(0, len(values), size):
        yield values[i : i + size]


def launch(src: Path, fit: dict, responses: list[str], action: str) -> None:
    script = Path(__file__).resolve()
    log_dir = src / fit["output_root"] / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )

    for group in batches(responses, int(fit["max_parallel"])):
        jobs = []
        for response in group:
            cmd = [
                "mpiexec",
                "-n",
                str(int(fit["mpi_per_band"])),
                sys.executable,
                "-u",
                str(script),
                str(src),
                response,
                action,
            ]
            log_path = log_dir / f"{response}_{action}.log"
            log("start: " + " ".join(cmd) + f" -> {log_path}")
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
            jobs.append((response, process, handle, log_path))

        for response, process, handle, log_path in jobs:
            code = process.wait()
            handle.close()
            log(("done: " if code == 0 else "FAIL: ") + f"{response} -> {log_path}")
            if code:
                raise SystemExit(code)


def parse_args(words: list[str]) -> tuple[str, str | None, str]:
    source = words[0]
    rest = words[1:]
    action = "run"
    if rest and rest[-1] in ACTIONS:
        action = rest.pop()
    if action == "plotdecomp":
        action = "plot"
    band = rest[0] if rest else None
    return source, band, action


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)

    source, band, action = parse_args(sys.argv[1:])
    src = source_dir(source)
    cfg = read_yaml(src / "config" / "source_config.yaml")
    fit = fit_config(cfg)
    responses = [band] if band else [str(x) for x in fit["responses"]]

    if COMM.Get_size() == 1:
        launch(src, fit, responses, action)
    else:
        run_band(src, cfg, fit, responses[0], action)


if __name__ == "__main__":
    main()
