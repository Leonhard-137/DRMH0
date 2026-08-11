#!/usr/bin/env python3
"""Reconstruct MICA component light curves for conditional posterior branches.

This script never reruns nested sampling. It resamples the original weighted
``sample1d`` posterior inside each configured branch and calls MICA's
``decompose()`` in an isolated branch run directory.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from collect_mica_results import (
    arr,
    col,
    mica_result_dir,
    model_kinds,
    model_label,
    normalized_weights,
    param_names,
    parse_run_name,
    read_yaml,
    reorder_component_columns,
    run_dirs,
    stage_config,
)
from posterior_branches import (
    branch_config,
    branch_definitions,
    branch_mask,
    branch_result_relative_dir,
    effective_sample_size,
    posterior_features,
    stable_seed_offset,
)


ROOT = Path(__file__).resolve().parents[1]
MICA_MAX_STR_LENGTH = 256
DNEST_STR_MAX_LENGTH = 156
DECOMPOSITION_ALGORITHM_VERSION = 2
DECOMPOSITION_SETTING_KEYS = (
    "type_tf",
    "ncomp",
    "lag_limit",
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
    "nd_rec",
    "trec_ext",
)


class DNestOptions(ctypes.Structure):
    """ctypes mirror of MICA2/cdnest/dnest.h::DNestOptions."""

    _fields_ = [
        ("num_particles", ctypes.c_uint),
        ("new_level_interval", ctypes.c_uint),
        ("save_interval", ctypes.c_uint),
        ("thread_steps", ctypes.c_uint),
        ("max_num_levels", ctypes.c_uint),
        ("lam", ctypes.c_double),
        ("beta", ctypes.c_double),
        ("max_ptol", ctypes.c_double),
        ("max_num_saves", ctypes.c_uint),
        ("thread_steps_factor", ctypes.c_double),
        ("new_level_interval_factor", ctypes.c_double),
        ("save_interval_factor", ctypes.c_double),
        ("sample_file", ctypes.c_char * DNEST_STR_MAX_LENGTH),
        ("sample_info_file", ctypes.c_char * DNEST_STR_MAX_LENGTH),
        ("levels_file", ctypes.c_char * DNEST_STR_MAX_LENGTH),
        ("sampler_state_file", ctypes.c_char * DNEST_STR_MAX_LENGTH),
        ("posterior_sample_file", ctypes.c_char * DNEST_STR_MAX_LENGTH),
        ("posterior_sample_info_file", ctypes.c_char * DNEST_STR_MAX_LENGTH),
        ("limits_file", ctypes.c_char * DNEST_STR_MAX_LENGTH),
    ]


def _native_mica_library(pymica_module):
    """Load the already-imported pymica extension and validate required symbols."""
    native_module = sys.modules.get("pymica.pymica")
    if native_module is None:
        native_module = __import__("pymica.pymica", fromlist=["pymica"])
    library = ctypes.CDLL(native_module.__file__)
    required = (
        "set_argv",
        "read_data",
        "init",
        "mc_line_init",
        "dnest_line",
        "output_reconstruction_parallel",
        "output_decompose_line_parallel",
        "mc_line_end",
        "end_run",
    )
    missing = [name for name in required if not hasattr(library, name)]
    if missing:
        raise RuntimeError(
            "installed pymica does not expose conditional-decomposition symbols: "
            + ", ".join(missing)
        )
    return library


def _set_native_string(library, symbol: str, value: str, capacity: int) -> None:
    encoded = os.fsencode(value)
    if len(encoded) >= capacity:
        raise ValueError(f"{symbol} is too long for MICA ({len(encoded)} >= {capacity})")
    native = (ctypes.c_char * capacity).in_dll(library, symbol)
    native.value = encoded


def decompose_conditional_posterior(
    pymica_module,
    ncomp: int,
    type_lag_prior: int,
    posterior_file: str,
    expected_columns: int,
) -> None:
    """Run MICA's decomposition directly from a staged posterior sample.

    MICA's public ``decompose()`` first invokes CDNest postprocessing and thus
    requires the original sampler state.  The sequence below follows the same
    native setup used by ``mc_line()``, but puts ``dnest_line()`` in
    parameter-name-only mode.  This initializes the model's parameter layout
    without sampling or postprocessing, after which the native decomposition
    consumes the explicitly selected conditional posterior.
    """
    library = _native_mica_library(pymica_module)
    library.set_argv.argtypes = [ctypes.c_int] * 5
    library.dnest_line.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    library.dnest_line.restype = ctypes.c_double

    initialized = False
    line_initialized = False
    try:
        library.set_argv(0, 0, 0, 1, 0)
        if library.read_data() != 0:
            raise RuntimeError("MICA read_data() failed")
        library.init()
        initialized = True
        if library.mc_line_init() != 0:
            raise RuntimeError("MICA mc_line_init() failed")
        line_initialized = True

        ctypes.c_int.in_dll(library, "num_gaussian").value = int(ncomp)
        ctypes.c_int.in_dll(library, "type_lag_prior_pr").value = (
            0 if int(type_lag_prior) == 0 and int(ncomp) > 1 else 1
        )
        _set_native_string(library, "postfix", f"_{ncomp}", MICA_MAX_STR_LENGTH)

        # flag_para_name=1 makes this call initialize num_params and index maps,
        # while skipping the actual CDNest run.
        library.dnest_line(0, None)
        native_columns = ctypes.c_int.in_dll(library, "num_params").value
        if native_columns != int(expected_columns):
            raise ValueError(
                "conditional posterior column mismatch: "
                f"MICA expects {native_columns}, file has {expected_columns}"
            )

        options = DNestOptions.in_dll(library, "options")
        encoded_path = os.fsencode(posterior_file)
        if len(encoded_path) >= DNEST_STR_MAX_LENGTH:
            raise ValueError(
                "conditional posterior path is too long for CDNest "
                f"({len(encoded_path)} >= {DNEST_STR_MAX_LENGTH})"
            )
        options.posterior_sample_file = encoded_path
        library.output_reconstruction_parallel()
        library.output_decompose_line_parallel()
    finally:
        if line_initialized:
            library.mc_line_end()
        if initialized:
            library.end_run()


def systematic_resample(weights, size: int, rng: np.random.Generator) -> np.ndarray:
    """Return low-variance resampling indices for normalized weights."""
    weights = normalized_weights(weights)
    size = int(size)
    if size < 1:
        raise ValueError("resampling size must be positive")
    positions = (rng.random() + np.arange(size, dtype=float)) / size
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="left")


def file_digest(path: Path) -> str | None:
    """Return a stable SHA256 digest, or None when the file does not exist."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decomposition_settings(fit: dict) -> str:
    """Canonical settings fingerprint for safe output reuse."""
    settings = {
        "algorithm_version": DECOMPOSITION_ALGORITHM_VERSION,
        **{key: fit.get(key) for key in DECOMPOSITION_SETTING_KEYS},
    }
    return json.dumps(settings, ensure_ascii=True, sort_keys=True, indent=2) + "\n"


def run_name(fit: dict, band: str) -> str:
    ncomp = int(fit.get("ncomp", 2))
    model = model_label(fit.get("type_tf", "gaussian"))
    return f"run_{fit['driver']}_to_{band}_{ncomp}comp_{model}"


def branch_root(source_dir: Path, fit: dict, branch_id: str) -> Path:
    return mica_result_dir(source_dir, fit) / branch_result_relative_dir(fit, branch_id)


def branch_run_dir(source_dir: Path, fit: dict, branch_id: str, band: str) -> Path:
    return branch_root(source_dir, fit, branch_id) / "decomposition" / run_name(fit, band)


def raw_component_arrays(sample: np.ndarray, names: list[str], ncomp: int):
    centers = np.vstack([
        sample[:, col(names, f"{index}-th_component_center")]
        for index in range(ncomp)
    ])
    widths = np.vstack([
        np.exp(sample[:, col(names, f"{index}-th_component_sigma")])
        for index in range(ncomp)
    ])
    amplitudes = np.vstack([
        np.exp(sample[:, col(names, f"{index}-th_component_amplitude")])
        for index in range(ncomp)
    ])
    return centers, widths, amplitudes


def stage_run(
    source_dir: Path,
    fit: dict,
    branch_id: str,
    definition: dict,
    original_run: Path,
    config: dict,
) -> dict:
    drive, band, ncomp, raw_model = parse_run_name(original_run.name)
    model = model_label(raw_model)
    model_kinds(model, ncomp)  # validate mixed labels before writing anything
    data_dir = original_run / "data"
    sample = arr(data_dir / f"sample1d.txt_{ncomp}")
    raw_weights = arr(data_dir / "weights.txt")[:, -1]
    if len(sample) != len(raw_weights):
        raise ValueError(f"sample/weight length mismatch in {original_run}")
    raw_weights = normalized_weights(raw_weights)
    names = param_names(data_dir / f"para_names_line.txt_{ncomp}")
    sample, label_swapped = reorder_component_columns(
        sample, names, ncomp, model, fit
    )
    centers, widths, amplitudes = raw_component_arrays(sample, names, ncomp)
    features = posterior_features(centers, widths, amplitudes)
    keep = branch_mask(branch_id, definition, features)
    parent_mass = float(np.sum(raw_weights[keep]))
    if parent_mass <= 0.0:
        raise ValueError(f"posterior branch {branch_id!r} has zero mass for {band}")

    selected_ids = np.flatnonzero(keep)
    conditional_weights = normalized_weights(raw_weights[keep])
    nsamp = int(config["decompose_nsamp"])
    seed = (
        int(config["seed"])
        + stable_seed_offset(branch_id)
        + stable_seed_offset(str(band))
    ) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    selected_positions = systematic_resample(conditional_weights, nsamp, rng)
    raw_ids = selected_ids[selected_positions]
    resampled = sample[raw_ids]

    target = branch_run_dir(source_dir, fit, branch_id, band)
    input_dir = target / "input"
    target_data = target / "data"
    posterior_path = target_data / f"posterior_sample1d.txt_{ncomp}"
    settings_path = target_data / "decomposition_settings.json"
    tracked_paths = {
        "posterior": posterior_path,
        "cont": input_dir / "cont.txt",
        "line": input_dir / "line.txt",
    }
    old_digests = {name: file_digest(path) for name, path in tracked_paths.items()}
    old_settings = (
        settings_path.read_text(encoding="utf-8")
        if settings_path.is_file()
        else None
    )
    input_dir.mkdir(parents=True, exist_ok=True)
    target_data.mkdir(parents=True, exist_ok=True)
    for name in ("cont.txt", "line.txt"):
        shutil.copy2(original_run / "input" / name, input_dir / name)

    np.savetxt(
        posterior_path,
        resampled,
        fmt="%.10e",
        header=str(nsamp),
        comments="# ",
    )
    sample_map = pd.DataFrame({
        "draw": np.arange(nsamp),
        "raw_sample": raw_ids,
        "raw_weight": raw_weights[raw_ids],
        "conditional_weight": conditional_weights[selected_positions],
        "label_swapped": label_swapped[raw_ids].astype(int),
    })
    sample_map.to_csv(
        target_data / "decomposition_sample_map.txt",
        sep=" ",
        index=False,
        float_format="%.10e",
    )
    current_settings = decomposition_settings(fit)
    settings_path.write_text(current_settings, encoding="utf-8")
    new_digests = {name: file_digest(path) for name, path in tracked_paths.items()}
    staged_inputs_unchanged = (
        old_settings == current_settings
        and all(old_digests[name] == new_digests[name] for name in tracked_paths)
    )

    ess = effective_sample_size(conditional_weights)
    status = []
    if parent_mass < float(config["low_mass_warning"]):
        status.append("low_mass")
    if ess < float(config["low_ess_warning"]):
        status.append("low_ess")
    return {
        "branch_id": branch_id,
        "label": definition["label"],
        "driver_band": drive,
        "band": band,
        "model": model,
        "ncomp": ncomp,
        "parent_posterior_mass": parent_mass,
        "n_raw": len(sample),
        "n_selected": len(selected_ids),
        "conditional_ess": ess,
        "label_order": str(fit.get("component_order", "native")),
        "label_swap_fraction": float(np.mean(label_swapped[keep])),
        "label_swap_mass": float(np.sum(conditional_weights[label_swapped[keep]])),
        "decompose_draws": nsamp,
        "unique_raw_draws": len(np.unique(raw_ids)),
        "seed": seed,
        "status": "+".join(status) if status else "ok",
        "staged_inputs_unchanged": staged_inputs_unchanged,
        "run_dir": str(target),
    }


def validate_decomposition_file(path: Path) -> tuple[int, int, int]:
    """Return driver, response, and total row counts after strict validation."""
    if not path.is_file():
        raise FileNotFoundError(f"missing MICA decomposition file: {path}")
    counts = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") and ":" in line:
                words = line[1:].strip().split(":")
                if len(words) == 2:
                    counts = (int(words[0]), int(words[1]))
                    break
            elif not line.startswith("#"):
                break
    if counts is None:
        raise ValueError(f"missing '# n_driver:n_response' header in {path}")
    data = np.loadtxt(path, usecols=(0, 1, 2))
    data = data.reshape(1, -1) if data.ndim == 1 else data
    expected = counts[0] + counts[1]
    if data.shape != (expected, 3):
        raise ValueError(
            f"decomposition shape mismatch in {path}: expected {(expected, 3)}, got {data.shape}"
        )
    if not np.isfinite(data).all() or np.any(data[:, 2] < 0.0):
        raise ValueError(f"non-finite values or negative errors in {path}")
    return counts[0], counts[1], len(data)


def validate_run_outputs(target: Path, ncomp: int, require_plot: bool) -> dict:
    """Validate all machine- and human-readable outputs for one run."""
    result = {}
    n_driver, n_response, n_total = validate_decomposition_file(
        target / "data" / f"pall.txt_{ncomp}"
    )
    result.update({
        "total_rows": n_total,
        "total_n_driver": n_driver,
        "total_n_response": n_response,
    })
    for component in range(ncomp):
        path = target / "data" / f"pline.txt_{ncomp}_comp{component}"
        n_driver, n_response, n_total = validate_decomposition_file(path)
        result.update({
            f"comp{component}_rows": n_total,
            f"comp{component}_n_driver": n_driver,
            f"comp{component}_n_response": n_response,
        })
    if require_plot:
        plot = target / "data" / f"fig_line_decomp_{ncomp}.pdf"
        if not plot.is_file() or plot.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty decomposition plot: {plot}")
        result["plot_pdf"] = str(plot)
    return result


def posterior_column_count(path: Path) -> int:
    """Count columns in the first posterior draw without loading the sample."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return len(stripped.split())
    raise ValueError(f"posterior file contains no draws: {path}")


def write_decomposition_summaries(path: Path, rows: list[dict]) -> None:
    """Write a compact human CSV and a complete machine-readable manifest."""
    frame = pd.DataFrame(rows)
    frame.to_csv(path.with_name("decomposition_manifest.txt"), sep="\t", index=False)
    human_columns = [
        "branch_id",
        "label",
        "band",
        "parent_posterior_mass",
        "n_selected",
        "conditional_ess",
        "decompose_draws",
        "unique_raw_draws",
        "status",
        "decomposition_action",
        "outputs_valid",
    ]
    for column in human_columns:
        if column not in frame:
            frame[column] = ""
    frame[human_columns].to_csv(path, index=False)


def worker(source_dir: Path, branch_id: str, band: str) -> None:
    """MPI worker: decompose one staged branch posterior without CDNest."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    from mpi4py import MPI
    import pymica
    from run_mica import fit_config, setup_kwargs

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cfg = read_yaml(source_dir / "config" / "source_config.yaml")
    fit = fit_config(cfg)
    target = branch_run_dir(source_dir, fit, branch_id, band)
    if not target.is_dir():
        raise FileNotFoundError(f"branch run has not been staged: {target}")
    os.chdir(target)

    data = None
    if rank == 0:
        data = {
            "set1": [
                np.loadtxt("input/cont.txt"),
                np.loadtxt("input/line.txt"),
            ]
        }
    data = comm.bcast(data, root=0)
    model = pymica.gmodel()
    kwargs = setup_kwargs(fit, data)
    model.setup(**kwargs)
    ncomp = int(fit.get("ncomp", 2))
    posterior_file = f"./data/posterior_sample1d.txt_{ncomp}"
    expected_columns = posterior_column_count(Path(posterior_file))
    decompose_conditional_posterior(
        pymica,
        ncomp=ncomp,
        type_lag_prior=int(kwargs.get("type_lag_prior", 0)),
        posterior_file=posterior_file,
        expected_columns=expected_columns,
    )
    comm.Barrier()
    plot_error = None
    if rank == 0 and bool(fit.get("plot_decomp", True)):
        try:
            model.plot_decomp(doshow=False)
        except Exception as exc:  # broadcast before raising to avoid an MPI deadlock
            plot_error = f"conditional decomposition plot failed: {exc}"
    plot_error = comm.bcast(plot_error, root=0)
    if plot_error is not None:
        raise RuntimeError(plot_error)


def launch_group(
    source_dir: Path,
    fit: dict,
    branch_id: str,
    bands: list[str],
    config: dict,
) -> None:
    log_dir = branch_root(source_dir, fit, branch_id) / "decomposition" / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(log_dir / "mplconfig"),
        "TMPDIR": "/tmp",
        "HYDRA_TMPDIR": "/tmp",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    mpi_per_band = int(config.get("mpi_per_band", fit.get("mpi_per_band", 2)))
    configured_mpiexec = config.get("mpiexec")
    if configured_mpiexec:
        mpiexec = Path(str(configured_mpiexec)).expanduser()
    else:
        mpiexec = Path(sys.executable).with_name("mpiexec")
    if not mpiexec.exists():
        discovered = shutil.which("mpiexec")
        if discovered is None:
            raise FileNotFoundError("cannot find an mpiexec compatible with the active Python")
        mpiexec = Path(discovered)
    jobs = []
    for band in bands:
        log_path = log_dir / f"{band}_decompose.log"
        command = [
            str(mpiexec),
            "-n",
            str(mpi_per_band),
            sys.executable,
            str(Path(__file__).resolve()),
            str(source_dir),
            "--worker",
            "--branch",
            branch_id,
            "--band",
            band,
        ]
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        jobs.append((band, process, handle, log_path))

    failed = []
    for band, process, handle, log_path in jobs:
        code = process.wait()
        handle.close()
        print(("done" if code == 0 else "FAIL") + f": {branch_id}/{band} -> {log_path}")
        if code:
            failed.append((band, code, log_path))
    if failed:
        details = "; ".join(f"{band}: exit {code}, {path}" for band, code, path in failed)
        raise RuntimeError("MICA branch decomposition failed: " + details)


def batches(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def controller(
    source_dir: Path,
    branch_ids: list[str],
    requested_band: str | None,
    stage_only: bool,
    skip_complete: bool,
) -> None:
    cfg = read_yaml(source_dir / "config" / "source_config.yaml")
    fit = stage_config(cfg, "r1")
    definitions = branch_definitions(fit)
    config = branch_config(fit)
    original_runs = run_dirs(source_dir, fit)
    by_band = {parse_run_name(path.name)[1]: path for path in original_runs}
    bands = [requested_band] if requested_band else list(by_band)
    missing = [band for band in bands if band not in by_band]
    if missing:
        raise ValueError("unknown response band(s): " + ", ".join(missing))

    for branch_id in branch_ids:
        if branch_id not in definitions:
            raise ValueError(f"unknown or disabled posterior branch: {branch_id}")
        rows = [
            stage_run(
                source_dir,
                fit,
                branch_id,
                definitions[branch_id],
                by_band[band],
                config,
            )
            for band in bands
        ]
        summary_path = branch_root(source_dir, fit, branch_id) / "decomposition_summary.csv"
        write_decomposition_summaries(summary_path, rows)
        print(f"staged: {branch_id} -> {summary_path}")
        if stage_only:
            continue

        pending_bands = []
        for row in rows:
            if not skip_complete:
                pending_bands.append(str(row["band"]))
                continue
            if not bool(row["staged_inputs_unchanged"]):
                pending_bands.append(str(row["band"]))
                continue
            try:
                row.update(validate_run_outputs(
                    Path(row["run_dir"]),
                    int(row["ncomp"]),
                    bool(fit.get("plot_decomp", True)),
                ))
                row["decomposition_action"] = "reused"
                print(f"reuse: {branch_id}/{row['band']}")
            except (FileNotFoundError, ValueError):
                pending_bands.append(str(row["band"]))

        max_parallel = max(1, int(config.get("max_parallel", fit.get("max_parallel", 1))))
        for group in batches(pending_bands, max_parallel):
            launch_group(source_dir, fit, branch_id, group, config)

        for row in rows:
            target = Path(row["run_dir"])
            row.update(validate_run_outputs(
                target,
                int(row["ncomp"]),
                bool(fit.get("plot_decomp", True)),
            ))
            row.setdefault("decomposition_action", "computed")
            row["outputs_valid"] = True
        write_decomposition_summaries(summary_path, rows)
        print(f"validated: {branch_id} -> {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--branch", help="one configured posterior branch")
    parser.add_argument("--all-branches", action="store_true")
    parser.add_argument("--band", help="one response band")
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        help="reuse staged runs whose total, component, and plot outputs validate",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    source_dir = args.source_dir.expanduser()
    source_dir = (
        source_dir.resolve()
        if source_dir.is_absolute()
        else (ROOT / source_dir).resolve()
    )
    if args.worker:
        if not args.branch or not args.band:
            parser.error("--worker requires --branch and --band")
        worker(source_dir, args.branch, args.band)
        return

    cfg = read_yaml(source_dir / "config" / "source_config.yaml")
    definitions = branch_definitions(stage_config(cfg, "r1"))
    if args.all_branches:
        branch_ids = list(definitions)
    elif args.branch:
        branch_ids = [args.branch]
    else:
        parser.error("choose --branch BRANCH_ID or --all-branches")
    controller(
        source_dir,
        branch_ids,
        args.band,
        args.stage_only,
        args.skip_complete,
    )


if __name__ == "__main__":
    main()
