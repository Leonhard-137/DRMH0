#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run total or MICA-component light-curve FFA/FVG analyses."""
from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from mpi4py import MPI

from collect_mica_results import (
    configured_run_name,
    mica_analysis_dir,
    mica_run_root,
    model_label,
    ncomp_from_config,
    stage_config,
)
from posterior_branches import branch_result_relative_dir


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
STEPS = (
    "prep",
    "dered",
    "fit",
    "summ",
    "fvg",
    "host",
    "distance_flux",
    "nuc",
    "sedfit",
)


def log(message: str) -> None:
    if RANK == 0:
        print(message, flush=True)


def source_dir(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parents[1] / path).resolve()


def config_path(src: Path, path: Path | None) -> Path:
    if path is None:
        return src / "config" / "source_config.yaml"
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (src / path).resolve()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def rel(src: Path, path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else src / path


def fflux_config(config: dict[str, Any], section: str) -> dict[str, Any]:
    ff = dict(config.get(section, {}) or {})
    if not ff and section == "fflux":
        ff = dict(config.get("fflux_analysis", {}) or {})
    if not ff:
        raise KeyError(f"missing YAML section: {section}")

    component = dict(ff.get("component", {}) or {})
    component.setdefault("name", "total")
    ff["component"] = component
    ff.setdefault("source", config.get("source"))
    ff.setdefault("bands", list((config.get("bands", {}) or {}).keys()))
    ff.setdefault("wavelengths", config.get("bands", {}) or {})
    ff.setdefault("prepared_dir", "prepared")
    ff.setdefault("out", "fflux")
    return ff


def posterior_branch_fflux_config(
    src: Path,
    config: dict[str, Any],
    section: str,
    branch_id: str,
) -> dict[str, Any]:
    """Build one component-0 FFA config from a collected MICA branch."""
    ff = fflux_config(config, section)
    mica_fit = dict(config.get("mica_round1", {}) or {})
    branch_cfg = dict(mica_fit.get("posterior_branches", {}) or {})
    definitions = dict(branch_cfg.get("branches", {}) or {})
    definition = dict(definitions.get(branch_id, {}) or {})
    if not definition or not bool(definition.get("enabled", True)):
        raise ValueError(f"unknown or disabled posterior branch: {branch_id}")

    ncomp = int(mica_fit.get("ncomp", 2))
    output_root = Path(str(mica_fit.get("output_root", "runs/mica")))
    branch_base = (
        output_root
        / f"{ncomp}comp"
        / "result"
        / branch_result_relative_dir(mica_fit, branch_id)
    )
    out_template = ff.get("posterior_branch_out_template")
    if out_template:
        out = str(out_template).format(branch_id=branch_id, ncomp=ncomp)
    else:
        out = str(branch_base / "ffa" / "comp0")

    mica = dict(ff.get("mica", {}) or {})
    mica.update({
        "driver": mica_fit.get("driver", mica.get("driver", "W2")),
        "ncomp": ncomp,
        "type_tf": mica_fit.get("type_tf", mica.get("type_tf", "gaussian")),
        "root": str(branch_base / "decomposition"),
    })
    ff["out"] = out
    ff["mica"] = mica
    decomposition_summary = src / branch_base / "decomposition_summary.csv"
    upstream_status = "not_checked"
    warning_bands = ""
    warning_details = ""
    if decomposition_summary.exists():
        quality = pd.read_csv(decomposition_summary)
        bad = quality.loc[quality["status"].astype(str) != "ok"]
        upstream_status = "ok" if bad.empty else "warning"
        if not bad.empty:
            warning_bands = ";".join(bad["band"].astype(str))
            warning_details = ";".join(
                f"{row.band}:{row.status}" for row in bad.itertuples(index=False)
            )
    ff["posterior_branch"] = {
        "id": branch_id,
        "label": str(definition.get("label", branch_id)),
        "source_dir": str(src),
        "upstream_status": upstream_status,
        "warning_bands": warning_bands,
        "warning_details": warning_details,
        "decomposition_summary": str(decomposition_summary),
        "mica_prepared_dir": str(mica_fit.get("prepared_dir", "prepared")),
    }
    return ff


def mica_component_fflux_config(
    src: Path,
    config: dict[str, Any],
    section: str,
    stage: str,
    component: str = "comp0",
) -> dict[str, Any]:
    """Build FFA config for a native component of one full MICA posterior."""
    ff = fflux_config(config, section)
    fit = stage_config(config, stage)
    ncomp = ncomp_from_config(fit)
    component_order = str(fit.get("component_order", "native")).strip().lower()
    if component_order not in {"", "native", "none"}:
        raise ValueError(
            f"{stage} uses component_order={component_order!r}; native MICA pline "
            "curves cannot be relabelled draw by draw. Direct component FFA is "
            "only valid for stages whose non-overlapping priors fix native identity."
        )
    match = re.fullmatch(r"comp(\d+)", str(component).strip().lower())
    if match is None or int(match.group(1)) >= ncomp:
        raise ValueError(
            f"{stage} component must be comp0..comp{ncomp - 1}, got {component!r}"
        )
    component = f"comp{int(match.group(1))}"
    model = model_label(fit.get("type_tf", "gaussian"))
    run_root = mica_run_root(src, fit)
    analysis_root = mica_analysis_dir(src, fit, component)
    source = str(config.get("source") or src.name)
    driver = str(fit.get("driver") or config.get("reference_band"))
    responses = list(dict.fromkeys(str(value) for value in (fit.get("responses") or [])))
    if not driver or not responses:
        raise ValueError(f"{stage} must define a driver and response bands")

    ff["out"] = str(analysis_root / "fflux")
    ff["bands"] = list(dict.fromkeys([driver, *responses]))
    ff["component"] = {"name": component}
    ff["mica"] = {
        "stage": stage,
        "driver": driver,
        "driver_file": str(
            rel(src, fit.get("prepared_dir", "fflux/host_sub"))
            / f"{source}_{driver}.txt"
        ),
        "ncomp": ncomp,
        "component_index": int(match.group(1)),
        "root": str(run_root),
        "run_template": str(
            fit.get(
                "run_template",
                "run_{driver}_to_{band}_{ncomp}comp_{type_tf}",
            )
        ),
        "type_tf": model,
        "response_time_support": str(
            (ff.get("mica", {}) or {}).get("response_time_support", "driver")
        ),
    }
    ff["mica_component"] = {
        "stage": stage,
        "component": component,
        "model": model,
        "ncomp": ncomp,
        "run_root": str(run_root),
        "collection_root": str(run_root / "result"),
        "analysis_root": str(analysis_root),
        "selection": "full_native_posterior",
    }

    # Validate every expected run name before FFA creates output directories.
    for response in responses:
        expected = run_root / configured_run_name(fit, response)
        if not expected.is_dir():
            raise FileNotFoundError(f"missing MICA run for {stage}/{response}: {expected}")
    return ff


def parse_steps(value: str | None, ff: dict[str, Any]) -> list[str]:
    value = value or ff.get("steps", STEPS)
    if isinstance(value, str):
        selected = list(STEPS) if value.strip().lower() == "all" else [
            item for chunk in value.split(",") for item in chunk.split()
        ]
    else:
        selected = list(value)

    selected = [str(step).strip().lower() for step in selected if str(step).strip()]
    unknown = [step for step in selected if step not in STEPS]
    if unknown:
        raise ValueError(f"unknown step(s): {', '.join(unknown)}")
    return list(dict.fromkeys(selected))


def root_dir(src: Path, ff: dict[str, Any]) -> Path:
    return rel(src, ff["out"])


def run_config(
    src: Path,
    original: dict[str, Any],
    ff: dict[str, Any],
    steps: list[str],
    *,
    fit_restart: Path | None = None,
) -> Path:
    payload = dict(original)
    payload["fflux"] = ff
    payload["_run_ffa"] = {"source_dir": str(src), "steps": steps}
    if fit_restart is not None:
        payload["_run_ffa"]["fit_restart"] = str(fit_restart)

    path = root_dir(src, ff) / "run.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return path


def validate_light_curve(
    data: np.ndarray,
    path: Path,
    *,
    label: str = "light curve",
) -> np.ndarray:
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != 3:
        raise ValueError(f"expected three columns in {path}, got shape {data.shape}")
    if len(data) < 2:
        raise ValueError(f"{label} has fewer than two rows: {path}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{label} contains non-finite values: {path}")
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError(f"{label} times are not strictly increasing: {path}")
    if np.any(data[:, 2] < 0):
        raise ValueError(f"{label} contains negative uncertainties: {path}")
    return data


def load_light_curve(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"light-curve file not found: {path}")

    data = np.loadtxt(path, usecols=(0, 1, 2))
    return validate_light_curve(data, path)


def save_light_curve(path: Path, data: np.ndarray) -> None:
    np.savetxt(
        path,
        data,
        fmt="%.10e %.10e %.10e",
        header="time flux flux_error",
        comments="# ",
    )


def prepare_total(src: Path, ff: dict[str, Any], output_dir: Path) -> None:
    source = str(ff.get("source") or src.name)
    component = str(ff["component"]["name"])
    bands = [str(band) for band in ff["bands"]]
    input_dir = rel(src, ff["prepared_dir"])
    template = str(ff.get("prepared_template", "{source}_{band}.txt"))

    for band in bands:
        source_path = input_dir / template.format(source=source, band=band)
        target_path = output_dir / f"{source}_{band}_{component}.txt"
        data = load_light_curve(source_path)
        save_light_curve(target_path, data)
        log(
            f"        {band}: {source_path} "
            f"[total, n={len(data)}, ptp={np.ptp(data[:, 1]):.6g}]"
        )


def mica_component_index(component: str, mica: dict[str, Any]) -> int:
    match = re.fullmatch(r"comp(\d+)", component.lower())
    if match is None:
        raise ValueError(
            f"MICA component name must have the form compN, got {component!r}"
        )

    index = int(match.group(1))
    configured = mica.get("component_index")
    if configured is not None and int(configured) != index:
        raise ValueError(
            f"component name {component!r} conflicts with "
            f"mica.component_index={configured}"
        )

    ncomp = int(mica.get("ncomp", 2))
    if not 0 <= index < ncomp:
        raise ValueError(
            f"component index {index} is outside the valid range 0..{ncomp - 1}"
        )
    return index


def mica_run_dir(
    src: Path,
    mica: dict[str, Any],
    response: str,
    component_index: int,
) -> Path:
    root = rel(src, mica.get("root", "runs/mica/2comp"))
    driver = str(mica.get("driver", "W2"))
    ncomp = int(mica.get("ncomp", 2))
    type_tf = str(mica.get("type_tf", "gaussian")).lower()
    template = str(
        mica.get(
            "run_template",
            "run_{driver}_to_{band}_{ncomp}comp_{type_tf}",
        )
    )
    name = template.format(
        driver=driver,
        band=response,
        response=response,
        ncomp=ncomp,
        type_tf=type_tf,
        component=f"comp{component_index}",
        component_index=component_index,
    )
    return root / name


def load_mica_decomposition(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"MICA decomposition file not found: {path}")

    counts: tuple[int, int] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = re.fullmatch(r"#\s*(\d+)\s*:\s*(\d+)\s*", line)
            if match is not None:
                counts = (int(match.group(1)), int(match.group(2)))
                break
            if not line.startswith("#"):
                break

    if counts is None:
        raise ValueError(
            f"cannot find '# n_driver:n_response' header in {path}"
        )

    data = np.loadtxt(path, usecols=(0, 1, 2))
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"expected three columns in {path}, got shape {data.shape}")

    n_driver, n_response = counts
    expected = n_driver + n_response
    if len(data) != expected:
        raise ValueError(
            f"MICA decomposition row-count mismatch in {path}: "
            f"header expects {expected} rows "
            f"({n_driver}+{n_response}), found {len(data)}"
        )

    driver = validate_light_curve(
        data[:n_driver].copy(),
        path,
        label="MICA driver block",
    )
    response = validate_light_curve(
        data[n_driver:expected].copy(),
        path,
        label="MICA response block",
    )
    return driver, response


def clip_light_curve_to_support(
    data: np.ndarray,
    support_min: float,
    support_max: float,
    path: Path,
) -> np.ndarray:
    """Clip a reconstructed light curve to an observed time support.

    MICA ``pline`` products are evaluated on a padded reconstruction grid.
    Those padding intervals are useful for displaying the reconstruction but
    must not define the bright/faint states used by the downstream FFA.  Keep
    only the configured driver support and interpolate the smooth MICA
    reconstruction to the exact support boundaries.
    """
    support_min = float(support_min)
    support_max = float(support_max)
    if (
        not np.isfinite(support_min)
        or not np.isfinite(support_max)
        or support_max <= support_min
    ):
        raise ValueError("light-curve support must be finite and increasing")
    if data[0, 0] > support_min or data[-1, 0] < support_max:
        raise ValueError(
            f"MICA reconstruction in {path} does not span the driver support "
            f"[{support_min:.10g}, {support_max:.10g}]"
        )

    time = data[:, 0]
    interior = data[(time > support_min) & (time < support_max)]
    boundaries = []
    for boundary in (support_min, support_max):
        exact = np.flatnonzero(np.isclose(time, boundary, rtol=0.0, atol=1.0e-10))
        if len(exact):
            boundaries.append(data[int(exact[0])].copy())
        else:
            boundaries.append(np.array([
                boundary,
                np.interp(boundary, time, data[:, 1]),
                np.interp(boundary, time, data[:, 2]),
            ]))
    clipped = np.vstack((boundaries[0], interior, boundaries[1]))
    return validate_light_curve(
        clipped,
        path,
        label="MICA response clipped to driver support",
    )


def prepare_mica_component(
    src: Path,
    ff: dict[str, Any],
    output_dir: Path,
) -> None:
    source = str(ff.get("source") or src.name)
    component = str(ff["component"]["name"]).lower()
    bands = [str(band) for band in ff["bands"]]
    mica = dict(ff.get("mica", {}) or {})
    if not mica:
        raise KeyError(
            f"fflux component {component!r} requires a 'mica' configuration block"
        )

    driver_band = str(mica.get("driver", "W2"))
    if driver_band not in bands:
        raise ValueError(
            f"MICA driver {driver_band!r} is not present in fflux bands: {bands}"
        )

    component_index = mica_component_index(component, mica)
    ncomp = int(mica.get("ncomp", 2))
    decomp_name = f"pline.txt_{ncomp}_comp{component_index}"
    manifest = []

    driver_file = mica.get("driver_file")
    if driver_file is not None:
        driver_path = rel(src, driver_file)
        driver_data = load_light_curve(driver_path)
        driver_label = f"{driver_path} [configured driver_file]"
    else:
        response_bands = [band for band in bands if band != driver_band]
        if not response_bands:
            raise ValueError("at least one MICA response band is required")
        first_run = mica_run_dir(
            src,
            mica,
            response_bands[0],
            component_index,
        )
        driver_path = first_run / "data" / decomp_name
        driver_data, _ = load_mica_decomposition(driver_path)
        driver_label = f"{driver_path} [driver block]"

    target_path = output_dir / f"{source}_{driver_band}_{component}.txt"
    save_light_curve(target_path, driver_data)
    support_mode = str(mica.get("response_time_support", "driver")).lower()
    if support_mode not in {"driver", "mica_grid"}:
        raise ValueError(
            "fflux_comp0.mica.response_time_support must be 'driver' or "
            f"'mica_grid', got {support_mode!r}"
        )
    support_min = float(driver_data[0, 0])
    support_max = float(driver_data[-1, 0])
    manifest.append({
        "band": driver_band,
        "role": "driver",
        "n": len(driver_data),
        "n_input": len(driver_data),
        "n_trimmed": 0,
        "time_min": float(driver_data[0, 0]),
        "time_max": float(driver_data[-1, 0]),
        "input_time_min": float(driver_data[0, 0]),
        "input_time_max": float(driver_data[-1, 0]),
        "support_mode": "observed_driver",
        "support_min": support_min,
        "support_max": support_max,
        "flux_ptp": float(np.ptp(driver_data[:, 1])),
        "input": str(driver_path),
        "output": str(target_path),
    })
    log(
        f"        {driver_band}: {driver_label} "
        f"[n={len(driver_data)}, ptp={np.ptp(driver_data[:, 1]):.6g}]"
    )

    for band in bands:
        if band == driver_band:
            continue

        run_dir = mica_run_dir(src, mica, band, component_index)
        source_path = run_dir / "data" / decomp_name
        _, response_input = load_mica_decomposition(source_path)
        if support_mode == "driver":
            response_data = clip_light_curve_to_support(
                response_input,
                support_min,
                support_max,
                source_path,
            )
        else:
            response_data = response_input
        target_path = output_dir / f"{source}_{band}_{component}.txt"
        save_light_curve(target_path, response_data)
        manifest.append({
            "band": band,
            "role": "response_comp0" if component_index == 0 else f"response_comp{component_index}",
            "n": len(response_data),
            "n_input": len(response_input),
            "n_trimmed": len(response_input) - len(response_data),
            "time_min": float(response_data[0, 0]),
            "time_max": float(response_data[-1, 0]),
            "input_time_min": float(response_input[0, 0]),
            "input_time_max": float(response_input[-1, 0]),
            "support_mode": support_mode,
            "support_min": support_min,
            "support_max": support_max,
            "flux_ptp": float(np.ptp(response_data[:, 1])),
            "input": str(source_path),
            "output": str(target_path),
        })
        log(
            f"        {band}: {source_path} [response block, "
            f"n={len(response_data)}, ptp={np.ptp(response_data[:, 1]):.6g}]"
        )
    pd.DataFrame(manifest).to_csv(output_dir / "prep_manifest.csv", index=False)


def prepare(src: Path, ff: dict[str, Any]) -> Path:
    component = str(ff["component"]["name"]).lower()
    output_dir = root_dir(src, ff) / "input"
    output_dir.mkdir(parents=True, exist_ok=True)

    if component == "total":
        prepare_total(src, ff, output_dir)
    elif re.fullmatch(r"comp\d+", component):
        prepare_mica_component(src, ff, output_dir)
    else:
        raise ValueError(
            f"unsupported fflux component {component!r}; "
            "expected 'total' or 'compN'"
        )

    return output_dir


def run_dered(src: Path, ff: dict[str, Any]) -> Path:
    from ffa.ext import dered

    rows = dered(src, ff)
    return Path(rows[0][1]["out"]).parent


def run_fit(src: Path, ff: dict[str, Any]) -> Path:
    from ffa.disk import fit

    result = fit(src, ff)
    return Path(result["sample_dir"])


def _data_row_count(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"CDNest continuation file not found: {path}")
    with path.open("rb") as handle:
        return sum(
            1
            for line in handle
            if line.strip() and not line.lstrip().startswith(b"#")
        )


def validate_fit_restart(
    sample_dir: Path,
    restart_file: str | Path,
    max_num_saves: int,
) -> tuple[Path, int]:
    """Validate a CDNest checkpoint and its append-only sample files."""
    sample_dir = sample_dir.expanduser().resolve()
    path = Path(restart_file).expanduser()
    path = path.resolve() if path.is_absolute() else (sample_dir / path).resolve()
    if path.parent != sample_dir:
        raise ValueError(
            f"FFA restart file must be directly under {sample_dir}, got {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"FFA restart file not found: {path}")
    if int(max_num_saves) <= 0:
        raise ValueError("fflux.fit.max_saves must be positive for restart")

    int_size = struct.calcsize("@i")
    with path.open("rb") as handle:
        raw_count = handle.read(int_size)
    if len(raw_count) != int_size:
        raise ValueError(f"truncated CDNest restart file: {path}")
    count_saves = int(struct.unpack("@i", raw_count)[0])
    if count_saves <= 0:
        raise ValueError(
            f"invalid saved-sample count {count_saves} in CDNest restart file: {path}"
        )
    if count_saves >= int(max_num_saves):
        raise ValueError(
            f"restart already contains {count_saves} saves, which is not below "
            f"configured max_saves={int(max_num_saves)}"
        )

    for name in ("sample.txt", "sample_info.txt"):
        continuation = sample_dir / name
        rows = _data_row_count(continuation)
        if rows != count_saves:
            raise ValueError(
                f"unsafe CDNest restart: {continuation} has {rows} data rows, "
                f"but {path.name} records {count_saves}; restore both files to "
                "the same checkpoint before restarting"
            )
    return path, count_saves


def run_fit_restart(src: Path, ff: dict[str, Any], restart_file: Path) -> Path:
    """Resume the configured FFA model from an in-place CDNest checkpoint."""
    import cydnest
    from ffa.ffa import build_model

    source = str(ff.get("source") or src.name)
    component = str(ff["component"]["name"])
    bands = [str(band) for band in ff["bands"]]
    root = root_dir(src, ff)
    input_dir = root / "dered" if (root / "dered").exists() else root / "input"
    sample_dir = root / "fit"
    filenames = [f"{source}_{band}_{component}.txt" for band in bands]
    missing = [name for name in filenames if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing FFA restart input(s) under {input_dir}: {', '.join(missing)}"
        )

    fit_cfg = dict(ff.get("fit", {}) or {})
    shifts = fit_cfg.get("time_shifts")
    if shifts:
        shifts = {int(key): float(value) for key, value in shifts.items()}
    max_num_saves = int(fit_cfg.get("max_saves", 1000))
    restart_path, count_saves = validate_fit_restart(
        sample_dir, restart_file, max_num_saves
    )
    log(
        f"        restart: {restart_path} "
        f"({count_saves}/{max_num_saves} saves)"
    )

    model = build_model(
        filenames=filenames,
        data_dir=input_dir,
        time_shifts=shifts,
        tau_min=float(fit_cfg.get("tau_min", 100.0)),
        R_min=float(fit_cfg.get("r_min", 0.01)),
        R_max=float(fit_cfg.get("r_max", 10.0)),
    )
    sampler = cydnest.sampler(
        model,
        sample_dir=str(sample_dir),
        thread_steps_factor=int(fit_cfg.get("thread_steps_factor", 2)),
        max_num_saves=max_num_saves,
    )
    sampler.restart(
        restart_file=str(restart_path),
        max_num_saves=max_num_saves,
    )
    if RANK == 0:
        model.postprocess(
            sample_path=sample_dir / "posterior_sample.txt",
            output_dir=sample_dir,
            make_plots=True,
            save_plots=True,
            show_plots=False,
        )
    return sample_dir


def run_summ(src: Path, ff: dict[str, Any]) -> Path:
    from ffa.disk import summ

    path, _ = summ(src, ff)
    return Path(path)


def run_fvg(src: Path, cfg: Path) -> tuple[Path, Path]:
    from ffa import sed

    table, _ = sed.comp(src, cfg)
    c = sed.config(src, cfg)
    posterior = Path(c["out_dir"]) / c["posterior_name"]
    return Path(table), posterior


def run_nuc(src: Path, cfg: Path) -> dict[str, Path]:
    from ffa import nuc

    result = nuc.run(src, cfg)
    return {name: Path(path) for name, path in result["paths"].items()}


def band_key(band: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(band)).strip("_")


def ffa_flux_posterior(src: Path, ff: dict[str, Any]) -> Path:
    source = str(ff.get("source") or src.name)
    component = str(ff["component"]["name"])
    bands = [str(band) for band in ff["bands"]]
    analysis = root_dir(src, ff) / "analysis"
    expected = analysis / f"{source}_{component}_{len(bands)}band_flux_posterior.txt"
    if expected.is_file():
        return expected
    candidates = sorted(analysis.glob("*_flux_posterior.txt"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one FFA flux posterior under {analysis}, found {len(candidates)}"
        )
    return candidates[0]


def run_host_sub(src: Path, ff: dict[str, Any]) -> dict[str, Path]:
    """Subtract the FFA host estimate without applying nuclear extinction."""
    source = str(ff.get("source") or src.name)
    component = str(ff["component"]["name"])
    if component.lower() != "total":
        raise ValueError("host subtraction is defined only for the total-light FFA")
    bands = [str(band) for band in ff["bands"]]
    posterior_path = ffa_flux_posterior(src, ff)
    posterior = pd.read_csv(posterior_path, sep=r"\s+")
    host_cfg = dict(ff.get("host_sub", {}) or {})
    output_dir = rel(src, host_cfg.get("output_dir", root_dir(src, ff) / "host_sub"))
    invalid_path = rel(
        src,
        host_cfg.get("invalid_flux_file", "results/invalid_host_sub_flux_points.txt"),
    )

    staged = []
    invalid = []
    for band in bands:
        key = band_key(band)
        host_column = f"F_Galaxy_{key}"
        if host_column not in posterior:
            raise KeyError(f"missing FFA posterior column: {host_column}")
        host_samples = pd.to_numeric(posterior[host_column], errors="coerce").to_numpy(float)
        host = float(np.nanmedian(host_samples))
        input_path = root_dir(src, ff) / "dered" / f"{source}_{band}_{component}.txt"
        data = load_light_curve(input_path)
        corrected = data.copy()
        corrected[:, 1] -= host
        bad = corrected[:, 1] <= 0.0
        for row in corrected[bad]:
            invalid.append({
                "band": band,
                "time": row[0],
                "flux_host_sub": row[1],
                "flux_error": row[2],
            })
        q16, q50, q84 = np.nanpercentile(host_samples, [16.0, 50.0, 84.0])
        staged.append((band, corrected, {
            "band": band,
            "wavelength": float(ff["wavelengths"][band]),
            "F_Galaxy_p16": q16,
            "F_Galaxy_p50": q50,
            "F_Galaxy_p84": q84,
            "n": len(corrected),
            "minimum_host_sub_flux": float(np.min(corrected[:, 1])),
        }))

    if invalid:
        invalid_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(invalid).to_csv(
            invalid_path, sep=" ", index=False, float_format="%.10e"
        )
        raise ValueError(
            f"non-positive host-subtracted flux; see {invalid_path}"
        )
    if invalid_path.exists():
        invalid_path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for band, corrected, summary in staged:
        output = output_dir / f"{source}_{band}.txt"
        save_light_curve(output, corrected)
        summary["file"] = str(output)
        summary_rows.append(summary)
    summary_path = output_dir / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return {"output_dir": output_dir, "summary": summary_path}


def run_distance_flux(src: Path, ff: dict[str, Any]) -> dict[str, Path]:
    """Build comp0 bright/faint/epsilon samples with no nuclear correction."""
    source = str(ff.get("source") or src.name)
    component = str(ff["component"]["name"])
    if component.lower() != "comp0":
        raise ValueError("distance_flux is defined only for component comp0")
    bands = [str(band) for band in ff["bands"]]
    posterior_path = ffa_flux_posterior(src, ff)
    raw = pd.read_csv(posterior_path, sep=r"\s+")
    output = {"sample": np.arange(len(raw), dtype=int)}
    summary_rows = []

    for band in bands:
        key = band_key(band)
        required = [
            f"F_Galaxy_{key}",
            f"F_Faint_{key}",
            f"F_Bright_{key}",
            f"Delta_F_{key}",
        ]
        missing = [column for column in required if column not in raw]
        if missing:
            raise KeyError(f"missing FFA posterior columns for {band}: {missing}")
        assigned_constant = pd.to_numeric(raw[required[0]], errors="coerce").to_numpy(float)
        faint = pd.to_numeric(raw[required[1]], errors="coerce").to_numpy(float) - assigned_constant
        bright = pd.to_numeric(raw[required[2]], errors="coerce").to_numpy(float) - assigned_constant
        delta = pd.to_numeric(raw[required[3]], errors="coerce").to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            epsilon = faint / bright

        output[f"F_Constant_comp0_{key}"] = assigned_constant
        output[f"F_Faint_comp0_{key}"] = faint
        output[f"F_Bright_comp0_{key}"] = bright
        output[f"Delta_F_comp0_{key}"] = delta
        output[f"Epsilon_comp0_{key}"] = epsilon
        valid = (
            np.isfinite(faint)
            & np.isfinite(bright)
            & np.isfinite(epsilon)
            & (faint > 0.0)
            & (bright > faint)
            & (epsilon > 0.0)
            & (epsilon < 1.0)
        )
        values = {
            "F_Constant_comp0": assigned_constant,
            "F_Faint_comp0": faint,
            "F_Bright_comp0": bright,
            "Delta_F_comp0": delta,
            "Epsilon_comp0": epsilon,
        }
        row = {
            "band": band,
            "wavelength": float(ff["wavelengths"][band]),
            "valid_fraction": float(np.mean(valid)),
        }
        for name, array in values.items():
            q16, q50, q84 = np.nanpercentile(array, [16.0, 50.0, 84.0])
            row[f"{name}_p16"] = q16
            row[f"{name}_p50"] = q50
            row[f"{name}_p84"] = q84
        summary_rows.append(row)

    analysis = root_dir(src, ff) / "analysis"
    posterior_out = analysis / "distance_flux_posterior.txt"
    summary_out = analysis / "distance_flux_summary.csv"
    pd.DataFrame(output).to_csv(
        posterior_out, sep=" ", index=False, float_format="%.10e"
    )
    pd.DataFrame(summary_rows).to_csv(summary_out, index=False)
    return {"posterior": posterior_out, "summary": summary_out}


def run_sedfit(src: Path, ff: dict[str, Any]) -> dict[str, Path]:
    """Create SED-model products from an existing joint FFA posterior."""
    source = str(ff.get("source") or src.name)
    component = str(ff["component"]["name"])
    bands = [str(band) for band in ff["bands"]]
    out_dir = root_dir(src, ff) / "analysis"
    expected = out_dir / f"{source}_{component}_{len(bands)}band_components.csv"
    if expected.exists():
        components_csv = expected
    else:
        candidates = sorted(out_dir.glob("*_components.csv"))
        if not candidates:
            raise FileNotFoundError(f"No *_components.csv found in {out_dir}")
        if len(candidates) != 1:
            raise RuntimeError(
                f"ambiguous FFA component tables in {out_dir}: "
                f"{[path.name for path in candidates]}"
            )
        components_csv = candidates[0]
    return run_sed_models(src, ff, components_csv)


def _weighted_log_powerlaw(
    log_y: np.ndarray,
    x: np.ndarray,
    sigma_log: np.ndarray,
    alpha: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit log(N) and optionally alpha for every joint FFA posterior draw."""
    if log_y.ndim != 2 or log_y.shape[1] != len(x):
        raise ValueError("log_y must have shape (n_sample, n_band)")
    if len(x) != len(sigma_log):
        raise ValueError("x and sigma_log must have the same length")
    if not np.all(np.isfinite(log_y)):
        raise ValueError("log_y contains non-finite values")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(sigma_log)):
        raise ValueError("x or sigma_log contains non-finite values")
    if np.any(sigma_log <= 0.0):
        raise ValueError("sigma_log must be strictly positive")
    weight = 1.0 / np.square(sigma_log)
    weight_sum = float(np.sum(weight))
    xbar = float(np.sum(weight * x) / weight_sum)
    if alpha is None:
        ybar = np.sum(log_y * weight[None, :], axis=1) / weight_sum
        denom = float(np.sum(weight * np.square(x - xbar)))
        if denom <= 0.0:
            raise ValueError("free power-law fit requires distinct wavelengths")
        slopes = np.sum(
            weight[None, :] * (x - xbar)[None, :] * (log_y - ybar[:, None]),
            axis=1,
        ) / denom
        intercept = ybar - slopes * xbar
    else:
        slopes = np.full(log_y.shape[0], float(alpha))
        intercept = np.sum(
            weight[None, :] * (log_y - float(alpha) * x[None, :]),
            axis=1,
        ) / weight_sum
    residual = (log_y - intercept[:, None] - slopes[:, None] * x[None, :])
    chi2 = np.sum(np.square(residual / sigma_log[None, :]), axis=1)
    return np.exp(intercept), slopes, chi2


def _percentiles(values: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.percentile(values, [16.0, 50.0, 84.0]))


def _joint_rms_sed_samples(
    posterior: pd.DataFrame,
    bands: list[str],
    wavelengths: np.ndarray,
) -> np.ndarray:
    """Return all-positive joint samples of ``lambda * R_lambda``."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.ndim != 1 or len(wavelengths) != len(bands):
        raise ValueError("wavelengths must be one-dimensional and match bands")
    if not np.isfinite(wavelengths).all() or np.any(wavelengths <= 0.0):
        raise ValueError("wavelengths must be finite and positive")

    columns = [f"R_{band_key(band)}" for band in bands]
    missing = [column for column in columns if column not in posterior]
    if missing:
        raise KeyError(f"missing FFA RMS posterior columns: {missing}")
    rms_amplitude = posterior[columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    samples = rms_amplitude * wavelengths[None, :]
    good = np.all(np.isfinite(samples) & (samples > 0.0), axis=1)
    return samples[good]


def _powerlaw_posterior_envelope(
    normalizations: np.ndarray,
    slopes: np.ndarray,
    wavelengths: np.ndarray,
    pivot: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the full fitted power-law posterior envelope.

    The normalization is a nuisance parameter, but its posterior uncertainty
    remains part of the physical model prediction.  It is therefore retained
    here instead of being projected out draw by draw for visualization.
    """
    normalizations = np.asarray(normalizations, dtype=float)
    slopes = np.asarray(slopes, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    if normalizations.ndim != 1 or slopes.shape != normalizations.shape:
        raise ValueError("normalizations and slopes must be matching 1D arrays")
    if wavelengths.ndim != 1:
        raise ValueError("wavelengths must be one-dimensional")
    if (
        not np.isfinite(normalizations).all()
        or not np.isfinite(slopes).all()
        or not np.isfinite(wavelengths).all()
        or not np.isfinite(pivot)
    ):
        raise ValueError("power-law envelope inputs must be finite")
    if np.any(normalizations <= 0.0) or np.any(wavelengths <= 0.0) or pivot <= 0.0:
        raise ValueError("normalizations, wavelengths, and pivot must be positive")

    curves = (
        normalizations[:, None]
        * (wavelengths[None, :] / float(pivot)) ** slopes[:, None]
    )
    lo, med, hi = np.percentile(curves, [16.0, 50.0, 84.0], axis=0)
    return lo, med, hi


def _component_display_name(component: str) -> str:
    """Return a human-readable label without changing the product identifier."""
    match = re.fullmatch(r"comp(\d+)", str(component).lower())
    if match is not None:
        return f"component {match.group(1)}"
    return str(component)


def _sed_product_metadata(ff: dict[str, Any]) -> dict[str, str]:
    """Describe total and posterior-branch SED products consistently."""
    component = str(ff["component"]["name"])
    branch = dict(ff.get("posterior_branch", {}) or {})
    product_id = str(branch.get("id") or component)
    default_upstream = "not_checked" if branch else "not_applicable"
    return {
        "component": component,
        "product_id": product_id,
        "upstream_status": str(branch.get("upstream_status", default_upstream)),
        "warning_bands": str(branch.get("warning_bands", "")),
        "warning_details": str(branch.get("warning_details", "")),
        "mica_prepared_dir": str(branch.get("mica_prepared_dir", "")),
    }


def _sed_plot_text(source: str, component: str) -> tuple[str, str]:
    """Return a blank panel title and a component-aware data label."""
    return "", f"FFA {_component_display_name(component)} RMS amplitude"


def run_sed_models(src: Path, ff: dict[str, Any], components_csv: Path) -> dict[str, Path]:
    """Fit disk SED shapes to the joint FFA RMS-amplitude posterior."""
    import matplotlib.pyplot as plt

    source = str(ff.get("source") or src.name)
    component = str(ff["component"]["name"])
    bands = [str(band) for band in ff["bands"]]
    waves = np.asarray([float(ff["wavelengths"][band]) for band in bands])
    pivot = float(np.exp(np.mean(np.log(waves))))
    x = np.log(waves / pivot)

    expected_posterior = (
        components_csv.parent
        / f"{source}_{component}_{len(bands)}band_flux_posterior.txt"
    )
    posterior_candidates = sorted(components_csv.parent.glob("*_flux_posterior.txt"))
    if expected_posterior.exists():
        posterior_path = expected_posterior
    elif len(posterior_candidates) == 1:
        posterior_path = posterior_candidates[0]
    elif not posterior_candidates:
        raise FileNotFoundError(f"no FFA flux posterior in {components_csv.parent}")
    else:
        raise RuntimeError(
            f"ambiguous FFA flux posteriors in {components_csv.parent}: "
            f"{[path.name for path in posterior_candidates]}"
        )
    posterior = pd.read_csv(posterior_path, sep=r"\s+")
    y_samples = _joint_rms_sed_samples(posterior, bands, waves)
    if len(y_samples) < 20:
        raise ValueError(
            f"too few positive joint RMS SED posterior draws: {len(y_samples)}"
        )

    y16, y50, y84 = np.percentile(y_samples, [16.0, 50.0, 84.0], axis=0)
    sigma_log = np.clip(0.5 * (y84 - y16) / y50, 1.0e-4, None)
    log_y = np.log(y_samples)

    configured = dict(ff.get("sed_models", {}) or {})
    hypotheses = configured.get("hypotheses") or {
        "slim": {"sed_alpha": 0.0},
        "standard": {"sed_alpha": -4.0 / 3.0},
        "free": {"sed_alpha": None},
    }
    metadata = _sed_product_metadata(ff)
    product_id = metadata["product_id"]
    upstream_status = metadata["upstream_status"]
    result_status = (
        "ok" if upstream_status in {"ok", "not_applicable"} else "upstream_warning"
    )

    summaries = []
    posterior_frames = []
    fit_cache = {}
    for model_name, model_cfg in hypotheses.items():
        model_cfg = dict(model_cfg or {})
        alpha_fixed = model_cfg.get("sed_alpha")
        norm, alpha_samples, chi2 = _weighted_log_powerlaw(
            log_y,
            x,
            sigma_log,
            None if alpha_fixed is None else float(alpha_fixed),
        )
        lag_beta = 0.5 * (alpha_samples + 4.0)
        moment2_gamma = alpha_samples + 4.0
        a16, a50, a84 = _percentiles(alpha_samples)
        n16, n50, n84 = _percentiles(norm)
        b16, b50, b84 = _percentiles(lag_beta)
        g16, g50, g84 = _percentiles(moment2_gamma)
        c16, c50, c84 = _percentiles(chi2)
        dof = len(bands) - (2 if alpha_fixed is None else 1)
        summaries.append({
            "component": component,
            "product_id": product_id,
            "model": str(model_name),
            "sed_quantity": "lambda_R_lambda",
            "credible_interval": "marginal_p16_p84_from_joint_posterior",
            "n_band": len(bands),
            "n_posterior": len(norm),
            "pivot_angstrom": pivot,
            "sed_alpha": a50,
            "sed_alpha_err_minus": a50 - a16,
            "sed_alpha_err_plus": a84 - a50,
            "lag_beta_pred": b50,
            "lag_beta_pred_err_minus": b50 - b16,
            "lag_beta_pred_err_plus": b84 - b50,
            "moment2_gamma_pred": g50,
            "moment2_gamma_pred_err_minus": g50 - g16,
            "moment2_gamma_pred_err_plus": g84 - g50,
            "normalization": n50,
            "normalization_err_minus": n50 - n16,
            "normalization_err_plus": n84 - n50,
            "chi2": c50,
            "chi2_err_minus": c50 - c16,
            "chi2_err_plus": c84 - c50,
            "dof": dof,
            "chi2_dof": c50 / dof,
            "status": result_status,
            "upstream_status": upstream_status,
            "warning_bands": metadata["warning_bands"],
            "warning_details": metadata["warning_details"],
            "mica_prepared_dir": metadata["mica_prepared_dir"],
        })
        posterior_frames.append(pd.DataFrame({
            "model": str(model_name),
            "component": component,
            "product_id": product_id,
            "sed_quantity": "lambda_R_lambda",
            "sample": np.arange(len(norm)),
            "normalization": norm,
            "sed_alpha": alpha_samples,
            "lag_beta_pred": lag_beta,
            "moment2_gamma_pred": moment2_gamma,
            "chi2": chi2,
        }))
        fit_cache[str(model_name)] = (norm, alpha_samples)

    out_dir = components_csv.parent
    summary_path = out_dir / f"{source}_{component}_sed_fit_summary.csv"
    posterior_out = out_dir / "sed_model_posterior.txt"
    points_path = out_dir / f"{source}_{component}_sed_points.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    pd.concat(posterior_frames, ignore_index=True).to_csv(
        posterior_out, sep=" ", index=False, float_format="%.10e"
    )
    point_rows = []
    for index, band in enumerate(bands):
        q16, q50, q84 = _percentiles(y_samples[:, index])
        point_rows.append({
            "band": band,
            "wavelength_angstrom": waves[index],
            "quantity": "lambda_R_lambda",
            "lambda_rms_p16": q16,
            "lambda_rms_p50": q50,
            "lambda_rms_p84": q84,
        })
    pd.DataFrame(point_rows).to_csv(points_path, index=False)

    # Reuse the unit-variance latent-process response posterior without a
    # draw-by-draw normalization transform.  These are the same p16/p50/p84
    # intervals shown in the host/RMS diagnostic figure.
    plot_yerr = np.vstack((y50 - y16, y84 - y50))

    xx = np.logspace(np.log10(waves.min() * 0.9), np.log10(waves.max() * 1.1), 400)
    colors = {"free": "#CC79A7", "standard": "#D55E00", "slim": "0.35"}
    styles = {"free": "-", "standard": "--", "slim": ":"}
    title, data_label = _sed_plot_text(source, component)
    with plt.rc_context({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.linewidth": 1.0,
    }):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.errorbar(
            waves, y50, yerr=plot_yerr, fmt="D", color="#0072B2", markersize=5,
            capsize=2, label=data_label, zorder=5,
        )
        for model_name in hypotheses:
            name = str(model_name)
            norm, alpha_samples = fit_cache[name]
            use = np.linspace(0, len(norm) - 1, min(1000, len(norm)), dtype=int)
            _, med, _ = _powerlaw_posterior_envelope(
                norm[use], alpha_samples[use], xx, pivot
            )
            color = colors.get(name, "0.2")
            ax.plot(xx, med, color=color, linestyle=styles.get(name, "-"),
                    linewidth=2.0, label=name.capitalize())
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Wavelength [$\AA$]")
        ax.set_ylabel(
            r"$\lambda\,R_\lambda$ "
            r"[$10^{-15}\,\mathrm{erg\,s^{-1}\,cm^{-2}}$]"
        )
        if title:
            ax.set_title(title, loc="left")
        ax.legend(frameon=False, ncol=2)
        ax.grid(alpha=0.18, which="major")
        ax.tick_params(which="both", direction="in", top=True, right=True)
        fig.tight_layout()
        png_path = out_dir / f"{source}_{component}_sed_models.png"
        pdf_path = out_dir / f"{source}_{component}_sed_models.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

    return {
        "sed_summary": summary_path,
        "sed_posterior": posterior_out,
        "sed_points": points_path,
        "sed_png": png_path,
        "sed_pdf": pdf_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--steps", help="comma-separated steps or all")
    parser.add_argument("--config-section", default="fflux")
    parser.add_argument("--posterior-branch")
    parser.add_argument("--mica-stage")
    parser.add_argument("--mica-component", default="comp0")
    parser.add_argument(
        "--fit-restart",
        type=Path,
        metavar="FILE",
        help=(
            "resume the fit step from a CDNest checkpoint under the configured "
            "fit directory (for example restart_dnest.txt_2000)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src = source_dir(args.source)
    cfg_path = config_path(src, args.config)
    config = read_yaml(cfg_path)
    if args.posterior_branch and args.mica_stage:
        parser.error("--posterior-branch and --mica-stage are mutually exclusive")
    if args.posterior_branch:
        ff = posterior_branch_fflux_config(
            src, config, args.config_section, args.posterior_branch
        )
    elif args.mica_stage:
        ff = mica_component_fflux_config(
            src,
            config,
            args.config_section,
            args.mica_stage,
            args.mica_component,
        )
    else:
        ff = fflux_config(config, args.config_section)
    steps = parse_steps(args.steps, ff)
    if args.fit_restart is not None and "fit" not in steps:
        parser.error("--fit-restart requires the fit step")

    if RANK == 0:
        normalized = run_config(
            src, config, ff, steps, fit_restart=args.fit_restart
        )
        log(f"source: {src}")
        log(f"config: {cfg_path}")
        log(f"steps: {', '.join(steps)}")
        log(f"run config: {normalized}")
    else:
        normalized = root_dir(src, ff) / "run.yaml"
    COMM.Barrier()

    if args.dry_run:
        return

    for step in steps:
        if step == "prep":
            if RANK == 0:
                log(f"prep  -> {prepare(src, ff)}")
            COMM.Barrier()

        elif step == "dered":
            if RANK == 0:
                log(f"dered -> {run_dered(src, ff)}")
            COMM.Barrier()

        elif step == "fit":
            if args.fit_restart is None:
                output = run_fit(src, ff)
            else:
                output = run_fit_restart(src, ff, args.fit_restart)
            if RANK == 0:
                log(f"fit   -> {output}")
                log(f"        {output / 'flux_flux_posterior.txt'}")
            COMM.Barrier()

        elif step == "summ":
            if RANK == 0:
                log(f"summ  -> {run_summ(src, ff)}")
            COMM.Barrier()

        elif step == "fvg":
            if RANK == 0:
                table, posterior = run_fvg(src, normalized)
                log(f"fvg   -> {table}")
                log(f"        {posterior}")
            COMM.Barrier()

        elif step == "host":
            if RANK == 0:
                paths = run_host_sub(src, ff)
                log(f"host  -> {paths['output_dir']}")
                log(f"        {paths['summary']}")
            COMM.Barrier()

        elif step == "distance_flux":
            if RANK == 0:
                paths = run_distance_flux(src, ff)
                log(f"flux  -> {paths['posterior']}")
                log(f"        {paths['summary']}")
            COMM.Barrier()

        elif step == "nuc":
            if RANK == 0:
                paths = run_nuc(src, normalized)
                log(f"nuc   -> {paths['int_sed']}")
                log(f"        posterior: {paths['posterior']}")
                log(f"        MICA input: {paths['mica_input_dir']}")
            COMM.Barrier()

        elif step == "sedfit":
            if RANK == 0:
                paths = run_sedfit(src, ff)
                log(f"sedfit -> {paths['sed_summary']}")
                log(f"        posterior: {paths['sed_posterior']}")
                log(f"        figure: {paths['sed_pdf']}")
            COMM.Barrier()

    log("done")


if __name__ == "__main__":
    main()
