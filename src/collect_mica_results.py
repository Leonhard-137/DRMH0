#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from posterior_branches import (
    branch_config,
    branch_definitions,
    branch_mask,
    branch_result_relative_dir,
    effective_sample_size,
    posterior_features,
)


TF_CODES = {"0": "gaussian", "1": "tophat", "2": "gamma", "3": "exp"}
TF_LABELS = {
    "gaussian": "gaussian",
    "tophat": "tophat",
    "gamma": "gamma",
    "exponential": "exp",
    "exp": "exp",
}
Q68 = np.array([0.15865, 0.5, 0.84135])


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def arr(path: Path) -> np.ndarray:
    x = np.loadtxt(path, comments="#")
    return x.reshape(1, -1) if x.ndim == 1 else x


def normalized_weights(weights) -> np.ndarray:
    weights = np.asarray(weights, float)
    if weights.ndim != 1:
        raise ValueError("posterior weights must be one-dimensional")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("posterior weights must be finite and non-negative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("posterior weights have zero total mass")
    return weights / total


def weighted_quantile(values, weights, quantiles=Q68) -> np.ndarray:
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    if len(values) == 0:
        raise ValueError("no finite, positive-weight posterior samples")
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    cdf = np.cumsum(weights[order])
    cdf /= cdf[-1]
    return np.interp(np.asarray(quantiles, float), cdf, values)


def q68(values, weights):
    p16, p50, p84 = weighted_quantile(values, weights)
    return float(p50), float(p50 - p16), float(p84 - p50)


def parse_run_name(name: str):
    m = re.match(r"^run_(.+?)_to_(.+?)_(\d+)comp_(.+)$", name)
    return None if m is None else (m.group(1), m.group(2), int(m.group(3)), m.group(4))


def model_label(model: str) -> str:
    model = str(model).lower()
    return TF_LABELS.get(model, model)


def component_order(name: str) -> int:
    if name == "total":
        return 99
    m = re.match(r"tf(\d+)$", str(name))
    return int(m.group(1)) if m else 98


def model_kinds(model: str, ncomp: int) -> list[str]:
    model = model.lower()
    if model.startswith("mix"):
        codes = model[3:]
        if len(codes) != ncomp or any(code not in TF_CODES for code in codes):
            raise ValueError(f"invalid mixed transfer-function label: {model}")
        return [TF_CODES[c] for c in codes]
    return [model_label(model)] * ncomp


def tau_from_kind(center, width, kind: str):
    if kind in {"gaussian", "tophat"}:
        return center
    if kind == "gamma":
        return center + 2.0 * width
    if kind == "exp":
        return center + width
    return center


def reorder_component_columns(
    sample: np.ndarray,
    names: list[str],
    ncomp: int,
    model: str,
    fit: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Relabel identical components sample-by-sample by increasing first moment.

    This changes labels only; it never drops a posterior sample. Native MICA
    labels are retained unless component_order: moment1 is explicit.
    """
    mode = str(fit.get("component_order", "native")).strip().lower()
    if mode in {"", "native", "none"}:
        return sample, np.zeros(len(sample), dtype=bool)
    if mode != "moment1":
        raise ValueError(f"unknown MICA component_order: {mode!r}")
    if ncomp != 2:
        raise ValueError("component_order=moment1 currently requires ncomp=2")

    kinds = model_kinds(model, ncomp)
    if kinds[0] != kinds[1]:
        raise ValueError(
            "component_order=moment1 cannot exchange components with different "
            "transfer-function families"
        )
    centers = [
        sample[:, col(names, f"{index}-th_component_center")]
        for index in range(2)
    ]
    widths = [
        np.exp(sample[:, col(names, f"{index}-th_component_sigma")])
        for index in range(2)
    ]
    moments = [
        tau_from_kind(centers[index], widths[index], kinds[index])
        for index in range(2)
    ]
    swapped = np.asarray(moments[0] > moments[1], dtype=bool)
    if not np.any(swapped):
        return sample, swapped

    ordered = sample.copy()
    for quantity in ("amplitude", "center", "sigma"):
        left = col(names, f"0-th_component_{quantity}")
        right = col(names, f"1-th_component_{quantity}")
        temporary = ordered[swapped, left].copy()
        ordered[swapped, left] = ordered[swapped, right]
        ordered[swapped, right] = temporary
    return ordered, swapped


def param_names(path: Path) -> list[str]:
    names: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) >= 2 and p[0].isdigit():
            i = int(p[0])
            while len(names) <= i:
                names.append("")
            names[i] = p[1]
    return names


def col(names: list[str], name: str) -> int:
    return names.index(name)


def stage_config(cfg: dict, stage: str) -> dict:
    # r1 -> mica_round1; r2 -> mica_round2
    m = re.fullmatch(r"r(\d+)", stage)
    section = f"mica_round{m.group(1)}" if m else stage
    fit = dict(cfg.get(section, {}) or {})
    if not fit:
        raise SystemExit(f"missing config section: {section}")
    return fit


def ncomp_from_config(fit: dict) -> int:
    if "ncomp" in fit:
        return int(fit["ncomp"])
    nc = fit.get("number_component", [2, 2])
    return int(nc[0] if isinstance(nc, list) else nc)


def mica_run_root(source_dir: Path, fit: dict) -> Path:
    ncomp = ncomp_from_config(fit)
    output_root = Path(fit.get("output_root", "runs/mica")).expanduser()
    if not output_root.is_absolute():
        output_root = source_dir / output_root
    return output_root / f"{ncomp}comp"


def mica_result_dir(source_dir: Path, fit: dict) -> Path:
    out = mica_run_root(source_dir, fit) / "result"
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_dirs(source_dir: Path, fit: dict) -> list[Path]:
    root = mica_run_root(source_dir, fit)
    if not root.is_dir():
        raise FileNotFoundError(f"missing MICA run directory: {root}")
    driver = str(fit.get("driver", ""))
    responses = list(dict.fromkeys(str(band) for band in (fit.get("responses") or [])))
    ncomp = ncomp_from_config(fit)
    model = model_label(fit.get("type_tf", "gaussian"))
    if not driver:
        raise ValueError("MICA config is missing driver")
    if not responses:
        raise ValueError("MICA config is missing responses")

    misplaced = []
    for path in sorted(root.iterdir()):
        info = parse_run_name(path.name) if path.is_dir() else None
        if info is None:
            continue
        run_driver, _, run_ncomp, run_model = info
        if (
            run_driver != driver
            or run_ncomp != ncomp
            or model_label(run_model) != model
        ):
            misplaced.append(path.name)
    if misplaced:
        raise ValueError(
            f"MICA root {root} contains runs inconsistent with the active "
            f"configuration ({driver=}, {ncomp=}, {model=}): "
            + ", ".join(misplaced)
        )

    expected = [
        root / f"run_{driver}_to_{band}_{ncomp}comp_{model}"
        for band in responses
    ]
    missing = [path for path in expected if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "missing expected MICA run directories:\n"
            + "\n".join(str(path) for path in missing)
        )
    return expected


def add_q(row: dict, prefix: str, values, weights) -> None:
    mid, lo, hi = q68(values, weights)
    row[prefix] = mid
    row[f"{prefix}_err_low"] = lo
    row[f"{prefix}_err_high"] = hi


def mica_flux_scales(run_dir: Path) -> tuple[float, float, str]:
    """Reproduce MICA2's per-run flux normalization for physical TF areas."""
    curves = []
    for name in ("cont.txt", "line.txt"):
        path = run_dir / "input" / name
        data = np.loadtxt(path, usecols=(1,))
        values = np.asarray(data, float).reshape(-1)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(f"invalid MICA input light curve: {path}")
        mean = float(np.mean(values))
        half_range = 0.5 * float(np.max(values) - np.min(values))
        if mean <= 0.0 or half_range <= 0.0:
            raise ValueError(
                f"MICA normalization requires positive mean and non-zero range: {path}"
            )
        curves.append((mean, half_range))

    use_half_range = any(mean < half_range / 100.0 for mean, half_range in curves)
    mode = "half_range" if use_half_range else "mean"
    scales = [
        half_range if use_half_range else mean
        for mean, half_range in curves
    ]
    return float(scales[0]), float(scales[1]), mode


def role(k: int, ncomp: int) -> str:
    if ncomp == 2:
        return "disk" if k == 0 else "delayed"
    return "disk" if k == 0 else f"comp{k}"


def summarize_run(
    run_dir: Path,
    cfg: dict,
    fit: dict,
    *,
    branch_id: str | None = None,
    branch_definition: dict | None = None,
    force_all: bool = False,
):
    drive, band, ncomp, raw_model = parse_run_name(run_dir.name)
    model = model_label(raw_model)
    kinds = model_kinds(model, ncomp)
    data = run_dir / "data"
    sample = arr(data / f"sample1d.txt_{ncomp}")
    weight_table = arr(data / "weights.txt")
    weights = weight_table[:, -1]
    if len(sample) != len(weights):
        raise ValueError(
            f"sample/weight length mismatch in {run_dir}: "
            f"{len(sample)} != {len(weights)}"
        )
    weights = normalized_weights(weights)
    names = param_names(data / f"para_names_line.txt_{ncomp}")
    sample, label_swapped = reorder_component_columns(
        sample, names, ncomp, model, fit
    )
    driver_scale, response_scale, normalization_mode = mica_flux_scales(run_dir)
    scale_ratio = response_scale / driver_scale
    wave = float(cfg["bands"][band])

    sample_ids = np.arange(len(sample))
    q_max = None if force_all or branch_definition is not None else fit.get("posterior_q_max")
    retained_weight = 1.0
    selection = "all"
    keep = np.ones(len(sample), dtype=bool)
    if branch_definition is not None:
        raw_centers = np.vstack([
            sample[:, col(names, f"{k}-th_component_center")]
            for k in range(ncomp)
        ])
        raw_widths = np.vstack([
            np.exp(sample[:, col(names, f"{k}-th_component_sigma")])
            for k in range(ncomp)
        ])
        raw_amplitudes = np.vstack([
            np.exp(sample[:, col(names, f"{k}-th_component_amplitude")])
            for k in range(ncomp)
        ])
        features = posterior_features(raw_centers, raw_widths, raw_amplitudes)
        keep = branch_mask(str(branch_id), branch_definition, features)
        retained_weight = float(np.sum(weights[keep]))
        if retained_weight <= 0.0:
            raise ValueError(
                f"posterior branch {branch_id!r} has zero posterior mass in {run_dir}"
            )
        sample = sample[keep]
        sample_ids = sample_ids[keep]
        label_swapped = label_swapped[keep]
        weights = normalized_weights(weights[keep])
        selection = str(branch_id)
    elif q_max is not None:
        if model != "gaussian":
            raise ValueError("posterior_q_max is only defined for Gaussian runs")
        q_max = float(q_max)
        tf0_center = sample[:, col(names, "0-th_component_center")]
        tf0_width = np.exp(sample[:, col(names, "0-th_component_sigma")])
        q = tf0_center / tf0_width
        keep = np.isfinite(q) & (q < q_max)
        retained_weight = float(np.sum(weights[keep]))
        if retained_weight <= 0.0:
            raise ValueError(f"q<{q_max:g} removed all posterior mass in {run_dir}")
        sample = sample[keep]
        sample_ids = sample_ids[keep]
        label_swapped = label_swapped[keep]
        weights = normalized_weights(weights[keep])
        selection = f"q0<{q_max:g}"

    conditional_ess = effective_sample_size(weights)
    label_order = str(fit.get("component_order", "native"))
    label_swap_fraction = float(np.mean(label_swapped)) if len(label_swapped) else 0.0
    label_swap_mass = float(np.sum(weights[label_swapped]))

    metadata = {
        "run_name": run_dir.name,
        "driver_band": drive,
        "model": model,
        "ncomp": ncomp,
        "selection": selection,
        "retained_weight": retained_weight,
        "conditional_ess": conditional_ess,
        "label_order": label_order,
        "label_swap_fraction": label_swap_fraction,
        "label_swap_mass": label_swap_mass,
        "normalization_mode": normalization_mode,
        "driver_scale": driver_scale,
        "response_scale": response_scale,
        "response_to_driver_scale": scale_ratio,
    }

    amps, centers, taus, widths = [], [], [], []
    for k in range(ncomp):
        amp = np.exp(sample[:, col(names, f"{k}-th_component_amplitude")])
        center = sample[:, col(names, f"{k}-th_component_center")]
        width = np.exp(sample[:, col(names, f"{k}-th_component_sigma")])
        amps.append(amp)
        centers.append(center)
        widths.append(width)
        taus.append(tau_from_kind(center, width, kinds[k]))

    amp_sum = np.sum(np.vstack(amps), axis=0)
    gains = [amp * scale_ratio for amp in amps]
    gain_sum = amp_sum * scale_ratio
    rows = []
    for k in range(ncomp):
        row = {
            **metadata,
            "band": band,
            "lambda_obs": wave,
            "component": f"tf{k}",
            "role": role(k, ncomp),
        }
        add_q(row, "tau", taus[k], weights)
        add_q(row, "width", widths[k], weights)
        add_q(row, "amp", amps[k], weights)
        add_q(row, "gain", gains[k], weights)
        add_q(row, "amp_frac", amps[k] / amp_sum, weights)
        rows.append(row)

    total_tau = np.sum(np.vstack(amps) * np.vstack(taus), axis=0) / amp_sum
    row = {
        **metadata,
        "band": band,
        "lambda_obs": wave,
        "component": "total",
        "role": "total",
    }
    add_q(row, "tau", total_tau, weights)
    row.update({"width": np.nan, "width_err_low": np.nan, "width_err_high": np.nan})
    add_q(row, "amp", amp_sum, weights)
    add_q(row, "gain", gain_sum, weights)
    row.update({"amp_frac": 1.0, "amp_frac_err_low": 0.0, "amp_frac_err_high": 0.0})
    rows.append(row)

    tf0_center = centers[0]
    tf0_width = widths[0]
    tf0_tau = taus[0]
    posterior = [
        {
            **metadata,
            "band": band,
            "lambda_obs": wave,
            "sample": int(sample_ids[i]),
            "center": float(tf0_center[i]),
            "width": float(tf0_width[i]),
            "tau": float(tf0_tau[i]),
            "amp": float(amps[0][i]),
            "gain": float(gains[0][i]),
            "weight": float(weights[i]),
        }
        for i in range(len(sample))
    ]

    component_posterior = []
    amp_frac = np.vstack(amps) / amp_sum
    for k in range(ncomp):
        component_posterior.extend([
            {
                **metadata,
                "band": band,
                "lambda_obs": wave,
                "sample": int(sample_ids[i]),
                "component": f"tf{k}",
                "role": role(k, ncomp),
                "center": float(centers[k][i]),
                "width": float(widths[k][i]),
                "tau": float(taus[k][i]),
                "amp": float(amps[k][i]),
                "gain": float(gains[k][i]),
                "amp_frac": float(amp_frac[k, i]),
                "weight": float(weights[i]),
            }
            for i in range(len(sample))
        ])

    priors = read_applied_priors(data / "table_priors_applied.csv", band)
    for prior in priors:
        prior.update({key: metadata[key] for key in ("run_name", "driver_band", "model", "ncomp")})
    selection_row = {
        **metadata,
        "band": band,
        "lambda_obs": wave,
        "n_raw": len(weight_table),
        "n_selected": len(sample),
        "posterior_q_max": q_max,
        "conditional_ess": conditional_ess,
        "label_order": label_order,
        "label_swap_fraction": label_swap_fraction,
        "label_swap_mass": label_swap_mass,
    }
    return rows, priors, posterior, component_posterior, selection_row


def read_applied_priors(path: Path, band: str) -> list[dict]:
    if not path.exists():
        return []
    out = []
    df = pd.read_csv(path)
    for r in df.itertuples(index=False):
        name = str(r.param_name)
        m = re.match(r"(\d+)-th_component_(amplitude|center|sigma)$", name)
        if not m:
            continue
        k, raw = int(m.group(1)), m.group(2)
        quantity = {"amplitude": "amp", "center": "tau", "sigma": "width"}[raw]
        lo, hi = float(r.prior_min), float(r.prior_max)
        if quantity in {"amp", "width"}:
            lo, hi = np.exp(lo), np.exp(hi)
        out.append({"band": band, "component": f"tf{k}", "quantity": quantity, "prior_low": lo, "prior_high": hi})
    return out


COMPONENT_SUMMARY_COLUMNS = [
    "run_name", "driver_band", "band", "model", "ncomp",
    "selection", "retained_weight", "lambda_obs", "component", "role",
    "normalization_mode", "driver_scale", "response_scale",
    "response_to_driver_scale",
    "tau", "tau_err_low", "tau_err_high",
    "width", "width_err_low", "width_err_high",
    "amp", "amp_err_low", "amp_err_high",
    "gain", "gain_err_low", "gain_err_high",
    "amp_frac", "amp_frac_err_low", "amp_frac_err_high",
]
TF0_POSTERIOR_COLUMNS = [
    "run_name", "driver_band", "band", "model", "ncomp",
    "selection", "retained_weight", "lambda_obs", "sample",
    "normalization_mode", "driver_scale", "response_scale",
    "response_to_driver_scale",
    "center", "width", "tau", "amp", "gain", "weight",
]
COMPONENT_POSTERIOR_COLUMNS = [
    "run_name", "driver_band", "band", "model", "ncomp",
    "selection", "retained_weight", "lambda_obs", "sample",
    "normalization_mode", "driver_scale", "response_scale",
    "response_to_driver_scale",
    "component", "role", "center", "width", "tau", "amp", "gain",
    "amp_frac", "weight",
]


def component_summary_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COMPONENT_SUMMARY_COLUMNS)
    frame["_order"] = frame["component"].map(component_order)
    return frame.sort_values(["lambda_obs", "_order"]).drop(columns="_order")


def tf0_posterior_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=TF0_POSTERIOR_COLUMNS).sort_values(
        ["lambda_obs", "sample"]
    )


def component_posterior_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COMPONENT_POSTERIOR_COLUMNS)
    frame["_order"] = frame["component"].map(component_order)
    return frame.sort_values(
        ["lambda_obs", "sample", "_order"]
    ).drop(columns="_order")


def save_collection(
    out_dir: Path,
    comp_rows: list[dict],
    posterior_rows: list[dict],
    component_posterior_rows: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    comp = component_summary_frame(comp_rows)
    posterior = tf0_posterior_frame(posterior_rows)
    component_posterior = component_posterior_frame(component_posterior_rows)
    comp.to_csv(out_dir / "mica_component_summary.csv", index=False, na_rep="")
    posterior.to_csv(
        out_dir / "mica_tf0_posterior.txt",
        sep=" ",
        index=False,
        float_format="%.10e",
    )
    component_posterior.to_csv(
        out_dir / "mica_component_posterior.txt",
        sep=" ",
        index=False,
        float_format="%.10e",
    )
    return comp, posterior, component_posterior


def selection_status(parent_mass: float, ess: float, config: dict) -> str:
    flags = []
    if parent_mass < float(config["low_mass_warning"]):
        flags.append("low_mass")
    if ess < float(config["low_ess_warning"]):
        flags.append("low_ess")
    return "+".join(flags) if flags else "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect compact MICA component and prior tables.")
    ap.add_argument("source_dir", type=Path)
    ap.add_argument("stage", nargs="?", default="r1", help="r1, r2, or a config section name")
    args = ap.parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    cfg = read_yaml(source_dir / "config/source_config.yaml")
    fit = stage_config(cfg, args.stage)
    out_dir = mica_result_dir(source_dir, fit)

    runs = run_dirs(source_dir, fit)
    comp_rows, prior_rows, posterior_rows, component_posterior_rows, selection_rows = [], [], [], [], []
    for rd in runs:
        comp, pri, post, component_post, selection = summarize_run(rd, cfg, fit)
        comp_rows.extend(comp)
        prior_rows.extend(pri)
        posterior_rows.extend(post)
        component_posterior_rows.extend(component_post)
        selection_rows.append(selection)

    save_collection(out_dir, comp_rows, posterior_rows, component_posterior_rows)

    prior_cols = [
        "run_name", "driver_band", "band", "model", "ncomp",
        "component", "quantity", "prior_low", "prior_high",
    ]
    pri = pd.DataFrame(prior_rows, columns=prior_cols)
    if len(pri):
        pri["_order"] = pri["component"].map(component_order)
        pri = pri.sort_values(["band", "_order", "quantity"]).drop(columns="_order")
    pri.to_csv(out_dir / "mica_prior_summary.csv", index=False, na_rep="")

    legacy_selection_columns = [
        "run_name", "driver_band", "model", "ncomp", "selection",
        "retained_weight", "band", "lambda_obs", "n_raw", "n_selected",
        "posterior_q_max", "label_order", "label_swap_fraction", "label_swap_mass",
    ]
    selection = pd.DataFrame(selection_rows)[legacy_selection_columns].sort_values("lambda_obs")
    selection.to_csv(out_dir / "mica_posterior_selection.csv", index=False)

    all_comp_rows, all_posterior_rows, all_component_rows = [], [], []
    for rd in runs:
        comp, _, post, component_post, _ = summarize_run(rd, cfg, fit, force_all=True)
        all_comp_rows.extend(comp)
        all_posterior_rows.extend(post)
        all_component_rows.extend(component_post)
    all_posterior = tf0_posterior_frame(all_posterior_rows)
    all_component_posterior = component_posterior_frame(all_component_rows)
    all_posterior.to_csv(
        out_dir / "mica_tf0_posterior_all.txt",
        sep=" ",
        index=False,
        float_format="%.10e",
    )
    all_component_posterior.to_csv(
        out_dir / "mica_component_posterior_all.txt",
        sep=" ",
        index=False,
        float_format="%.10e",
    )

    definitions = branch_definitions(fit)
    if definitions:
        branches_root = out_dir / "branches"
        branches_root.mkdir(parents=True, exist_ok=True)
        config = branch_config(fit)
        branch_summaries: dict[str, pd.DataFrame] = {}
        branch_tf0: dict[str, pd.DataFrame] = {}

        for branch_id, definition in definitions.items():
            branch_comp_rows, branch_post_rows, branch_component_rows = [], [], []
            branch_selection_rows = []
            for rd in runs:
                comp, _, post, component_post, selected = summarize_run(
                    rd,
                    cfg,
                    fit,
                    branch_id=branch_id,
                    branch_definition=definition,
                )
                branch_comp_rows.extend(comp)
                branch_post_rows.extend(post)
                branch_component_rows.extend(component_post)
                branch_selection_rows.append(selected)

            branch_dir = out_dir / branch_result_relative_dir(fit, branch_id)
            _, branch_posterior, _ = save_collection(
                branch_dir,
                branch_comp_rows,
                branch_post_rows,
                branch_component_rows,
            )
            branch_tf0[branch_id] = branch_posterior

            summary_rows = []
            for row in branch_selection_rows:
                parent_mass = float(row["retained_weight"])
                ess = float(row["conditional_ess"])
                summary_rows.append({
                    "branch_id": branch_id,
                    "label": definition["label"],
                    "band": row["band"],
                    "lambda_obs_A": row["lambda_obs"],
                    "parent_posterior_mass": parent_mass,
                    "n_raw": int(row["n_raw"]),
                    "n_selected": int(row["n_selected"]),
                    "conditional_ess": ess,
                    "label_order": row["label_order"],
                    "label_swap_fraction": row["label_swap_fraction"],
                    "label_swap_mass": row["label_swap_mass"],
                    "status": selection_status(parent_mass, ess, config),
                })
            branch_summary = pd.DataFrame(summary_rows).sort_values("lambda_obs_A")
            branch_summary.to_csv(branch_dir / "branch_summary.csv", index=False)
            branch_summaries[branch_id] = branch_summary
            with (branch_dir / "branch_definition.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    {
                        "branch_id": branch_id,
                        "label": definition["label"],
                        "ranges": definition["ranges"],
                    },
                    handle,
                    sort_keys=False,
                    allow_unicode=True,
                )

        catalog_rows = []
        for branch_id, summary in branch_summaries.items():
            masses = summary["parent_posterior_mass"].to_numpy(float)
            ess = summary["conditional_ess"].to_numpy(float)
            warnings_found = sorted(set(summary.loc[summary["status"].ne("ok"), "status"]))
            catalog_rows.append({
                "branch_id": branch_id,
                "label": definitions[branch_id]["label"],
                "n_band": len(summary),
                "min_parent_mass": np.min(masses),
                "median_parent_mass": np.median(masses),
                "max_parent_mass": np.max(masses),
                "min_conditional_ess": np.min(ess),
                "status": "+".join(warnings_found) if warnings_found else "ok",
            })
        pd.DataFrame(catalog_rows).to_csv(branches_root / "branch_catalog.csv", index=False)

        partition_rows = []
        for band, all_band in all_posterior.groupby("band", sort=False):
            all_band = all_band.sort_values("sample")
            sample_ids = all_band["sample"].to_numpy(int)
            weights = all_band["weight"].to_numpy(float)
            membership = np.zeros(len(all_band), dtype=int)
            for branch_id, posterior in branch_tf0.items():
                selected_ids = set(
                    posterior.loc[posterior["band"].astype(str).eq(str(band)), "sample"].astype(int)
                )
                membership += np.fromiter(
                    (sample_id in selected_ids for sample_id in sample_ids),
                    dtype=bool,
                    count=len(sample_ids),
                )
            assigned = float(np.sum(weights[membership > 0]))
            overlap = float(np.sum(weights[membership > 1]))
            partition_rows.append({
                "band": band,
                "lambda_obs_A": float(all_band["lambda_obs"].iloc[0]),
                "assigned_union_mass": assigned,
                "overlapping_mass": overlap,
                "unassigned_mass": float(max(0.0, 1.0 - assigned)),
                "max_membership": int(np.max(membership)),
            })
        pd.DataFrame(partition_rows).sort_values("lambda_obs_A").to_csv(
            branches_root / "branch_partition_summary.csv",
            index=False,
        )

    print(out_dir / "mica_component_summary.csv")
    print(out_dir / "mica_prior_summary.csv")
    print(out_dir / "mica_tf0_posterior.txt")
    print(out_dir / "mica_component_posterior.txt")
    print(out_dir / "mica_posterior_selection.csv")
    print(out_dir / "mica_tf0_posterior_all.txt")
    print(out_dir / "mica_component_posterior_all.txt")
    if definitions:
        print(out_dir / "branches" / "branch_catalog.csv")
        print(out_dir / "branches" / "branch_partition_summary.csv")


if __name__ == "__main__":
    main()
