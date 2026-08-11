#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


TF_CODES = {
    "0": "gaussian",
    "1": "tophat",
    "2": "gamma",
    "3": "exp",
}

ALIASES = {
    "gauss": "gaussian",
    "g": "gaussian",
    "top-hat": "tophat",
    "top_hat": "tophat",
    "hat": "tophat",
    "exponential": "exp",
}

OUT_COLS = [
    "drive_band",
    "resp_band",
    "lambda_obs",
    "ncomp",
    "model",
    "component",
    "tf_index",
    "tf_kind",
    "tau",
    "tau_err_low",
    "tau_err_high",
    "center",
    "center_err_low",
    "center_err_high",
    "width",
    "width_err_low",
    "width_err_high",
    "amplitude",
    "amplitude_err_low",
    "amplitude_err_high",
    "amplitude_frac",
]


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def rel(base: Path, path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else base / p


def read_array(path: Path) -> np.ndarray:
    x = np.loadtxt(path, comments="#")
    return x.reshape(1, -1) if x.ndim == 1 else x


def is_int_token(x: str) -> bool:
    return bool(re.fullmatch(r"\d+", x))


def is_number_token(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def q68(x) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan

    p16, p50, p84 = np.percentile(x, [15.865, 50.0, 84.135])
    return float(p50), float(p50 - p16), float(p84 - p50)


def add_q(row: dict, prefix: str, x, scale: float = 1.0) -> None:
    mid, lo, hi = q68(np.asarray(x, dtype=float) * scale)
    row[prefix] = mid
    row[f"{prefix}_err_low"] = lo
    row[f"{prefix}_err_high"] = hi


def parse_run_name(name: str):
    m = re.match(r"^run_(.+?)_to_(.+?)_(\d+)comp_(.+)$", name)
    if not m:
        return None

    drive, resp, ncomp, model = m.groups()
    return drive, resp, int(ncomp), model.lower()


def norm_model(model: str) -> str:
    model = model.strip().lower()
    return ALIASES.get(model, model)


def model_kinds(model: str, ncomp: int) -> list[str]:
    model = norm_model(model)

    if model.startswith("mix"):
        codes = model[3:]
        if len(codes) != ncomp:
            raise ValueError(f"{model}: code length {len(codes)} != ncomp {ncomp}")

        bad = [c for c in codes if c not in TF_CODES]
        if bad:
            raise ValueError(f"{model}: unknown TF code(s): {bad}")

        return [TF_CODES[c] for c in codes]

    if model not in set(TF_CODES.values()):
        raise ValueError(f"unknown model {model!r}")

    return [model] * ncomp


def find_run_dirs(input_dir: Path) -> list[Path]:
    if parse_run_name(input_dir.name):
        return [input_dir]

    runs = [
        p for p in input_dir.iterdir()
        if p.is_dir() and parse_run_name(p.name)
    ]
    return sorted(runs, key=lambda p: p.name)


def read_param_names(data_dir: Path, ncomp: int) -> list[str]:
    path = data_dir / f"para_names_line.txt_{ncomp}"
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"#.*", " ", text)
    tokens = text.split()

    names = []
    for i, tok in enumerate(tokens[:-1]):
        next_tok = tokens[i + 1]
        if (
            is_int_token(tok)
            and int(tok) == len(names)
            and not is_number_token(next_tok)
        ):
            names.append(next_tok)

    if names:
        return names

    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) >= 2 and parts[0].isdigit():
            names.append(parts[1])
        else:
            names.append(parts[-1])

    if not names:
        raise ValueError(f"no parameter names found in {path}")

    return names


def find_col(names: list[str], name: str) -> int:
    for i, x in enumerate(names):
        if x == name:
            return i
    raise ValueError(f"missing parameter column: {name}")


def component_cols(names: list[str], ncomp: int) -> list[tuple[int, int, int]]:
    cols = []
    for k in range(ncomp):
        cols.append(
            (
                find_col(names, f"{k}-th_component_amplitude"),
                find_col(names, f"{k}-th_component_center"),
                find_col(names, f"{k}-th_component_sigma"),
            )
        )
    return cols


def tau_from_kind(center, width, kind: str):
    if kind in {"gaussian", "tophat"}:
        return center
    if kind == "gamma":
        return center + 2.0 * width
    if kind == "exp":
        return center + width

    raise ValueError(f"unknown TF kind: {kind!r}")


def lag_scale(cfg: dict) -> float:
    if not cfg.get("use_rest_frame", False):
        return 1.0

    if cfg.get("redshift") is None:
        raise ValueError("use_rest_frame=true but redshift is missing")

    return 1.0 / (1.0 + float(cfg["redshift"]))


def summarize_run(run_dir: Path, cfg: dict) -> list[dict]:
    info = parse_run_name(run_dir.name)
    if info is None:
        return []

    drive, resp, ncomp, model = info
    kinds = model_kinds(model, ncomp)
    data_dir = run_dir / "data"

    sample_path = data_dir / f"posterior_sample1d.txt_{ncomp}"
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)

    sample = read_array(sample_path)
    names = read_param_names(data_dir, ncomp)
    cols = component_cols(names, ncomp)

    if sample.shape[1] < len(names):
        raise ValueError(
            f"{sample_path}: sample has {sample.shape[1]} columns, "
            f"but {len(names)} parameter names were found"
        )

    bands = cfg.get("bands", {})
    if resp not in bands:
        raise KeyError(f"response band {resp!r} not found in config['bands']")

    scale = lag_scale(cfg)
    lam = float(bands[resp])

    amps = []
    taus = []
    rows = []

    for k, (camp, ccen, csig) in enumerate(cols):
        amp = np.exp(sample[:, camp])
        center = sample[:, ccen]
        width = np.exp(sample[:, csig])
        tau = tau_from_kind(center, width, kinds[k])

        amps.append(amp)
        taus.append(tau)

    amp_sum = np.sum(np.vstack(amps), axis=0)

    for k, (amp, tau, (_, ccen, csig)) in enumerate(zip(amps, taus, cols)):
        center = sample[:, ccen]
        width = np.exp(sample[:, csig])
        frac = np.divide(
            amp,
            amp_sum,
            out=np.full_like(amp, np.nan),
            where=amp_sum != 0,
        )

        row = {
            "drive_band": drive,
            "resp_band": resp,
            "lambda_obs": lam,
            "ncomp": ncomp,
            "model": model,
            "component": f"tf{k}",
            "tf_index": k,
            "tf_kind": kinds[k],
            "amplitude_frac": q68(frac)[0],
        }

        add_q(row, "tau", tau, scale)
        add_q(row, "center", center, scale)
        add_q(row, "width", width, scale)
        add_q(row, "amplitude", amp)

        rows.append(row)

    opt = cfg.get("mica_lag_summary", {})
    if opt.get("include_total", True):
        total_tau = np.divide(
            np.sum(np.vstack(amps) * np.vstack(taus), axis=0),
            amp_sum,
            out=np.full_like(amp_sum, np.nan),
            where=amp_sum != 0,
        )

        row = {
            "drive_band": drive,
            "resp_band": resp,
            "lambda_obs": lam,
            "ncomp": ncomp,
            "model": model,
            "component": "total",
            "tf_index": ncomp,
            "tf_kind": "mixture",
            "center": np.nan,
            "center_err_low": np.nan,
            "center_err_high": np.nan,
            "width": np.nan,
            "width_err_low": np.nan,
            "width_err_high": np.nan,
            "amplitude_frac": 1.0,
        }

        add_q(row, "tau", total_tau, scale)
        add_q(row, "amplitude", amp_sum)

        rows.append(row)

    return rows


def build_table(
    source_dir: Path,
    cfg: dict,
    input_dir: Optional[Path] = None,
    config_section: str = "mica_lag_summary",
) -> pd.DataFrame:
    opt = cfg.get(config_section, {})
    in_dir = input_dir or rel(source_dir, opt.get("input_dir", "runs/gaussian/2comp"))

    rows = []
    for run_dir in find_run_dirs(in_dir):
        rows.extend(summarize_run(run_dir, cfg))

    if not rows:
        raise RuntimeError(f"no run directories found in {in_dir}")

    return (
        pd.DataFrame(rows)[OUT_COLS]
        .sort_values(["lambda_obs", "tf_index", "resp_band"])
        .reset_index(drop=True)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source_dir", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--input-dir", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--config-section", default="mica_lag_summary",
                    help="YAML config section name (default: mica_lag_summary).")
    args = ap.parse_args()

    source_dir = args.source_dir.expanduser().resolve()

    cfg_path = args.config or source_dir / "config/source_config.yaml"
    if not cfg_path.is_absolute():
        cfg_path = source_dir / cfg_path

    cfg = read_yaml(cfg_path)
    section = args.config_section

    input_dir = rel(source_dir, args.input_dir) if args.input_dir else None
    df = build_table(source_dir, cfg, input_dir, section)

    opt = cfg.get(section, {})
    out = args.output or Path(opt.get("output", "results/mica_lag_summary.csv"))
    out = rel(source_dir, out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out, index=False, float_format="%.6g", na_rep="")
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()