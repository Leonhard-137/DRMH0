#!/usr/bin/env python3
"""Apply total-light FVG nuclear extinction to an existing comp0 H0 posterior.

This is deliberately a post-processing program.  It never calls MICA or the
FFA sampler.  The numerical transformation matches ``ffa.nuc.ebv`` and
``ffa.nuc.corr`` but accepts a separate total-light FVG posterior and comp0
distance-flux posterior, which is the data flow needed by the current pipeline.
"""
from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from astropy import units as u
from dust_extinction.parameter_averages import F99

try:
    from dust_extinction.averages import G03_SMCBar
except ImportError:  # pragma: no cover - compatibility with older dust_extinction
    from dust_extinction.parameter_averages import G03_SMCBar

import disk_distance
from collect_mica_results import configured_stages, mica_analysis_dir, stage_config
from posterior_branches import branch_result_relative_dir


DEFAULT_PAIR = ("W2", "V")
DEFAULT_SLOPE = 7.0 / 3.0
DEFAULT_LAW = "SMC"
DEFAULT_RV = 2.71
DEFAULT_SEED = 12345


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def source_dir(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parents[1] / path).resolve()


def rel(root: Path, path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def within(path: Path, parent: Path) -> bool:
    """Return whether *path* is *parent* or one of its descendants."""
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def validate_output_root(
    output_root: Path,
    *,
    input_root: Path,
    total_path: Path,
    target_path: Path,
) -> None:
    """Require a new, isolated result directory and protect all inputs."""
    output_root = output_root.resolve()
    protected_dirs = [
        input_root / name
        for name in ("fflux", "lag", "distance", "h0", "decomposition")
    ]
    protected_dirs.extend([total_path.parent, target_path.parent])
    if output_root == input_root.resolve():
        raise ValueError("output root must not be the input branch root")
    for protected in protected_dirs:
        if within(output_root, protected) or within(protected, output_root):
            raise ValueError(
                f"output root {output_root} overlaps protected input/results "
                f"directory {protected.resolve()}"
            )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_root}. Choose a new labelled "
            "output directory; existing results are never overwritten."
        )


def key(band: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(band)).strip("_")


def q68(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    return tuple(float(value) for value in np.percentile(values, [16.0, 50.0, 84.0]))


def extinction_k(
    wavelengths: np.ndarray,
    *,
    law: str = DEFAULT_LAW,
    rv: float = DEFAULT_RV,
) -> tuple[np.ndarray, float]:
    """Return k(lambda)=A(lambda)/E(B-V) and the law's effective R_V."""
    wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError("extinction wavelengths must be finite and positive")
    name = str(law).strip().lower()
    if name in {"smc", "smcbar", "g03", "g03_smcbar"}:
        model = G03_SMCBar()
        effective_rv = float(model.Rv)
    elif name in {"mw", "f99", "fitzpatrick99"}:
        model = F99(Rv=float(rv))
        effective_rv = float(rv)
    else:
        raise ValueError(f"unknown extinction law: {law}")
    values = np.asarray(model(wavelengths * u.Angstrom), dtype=float) * effective_rv
    return values, effective_rv


def infer_fvg_ebv(
    posterior: pd.DataFrame,
    wavelengths: dict[str, float],
    *,
    pair: tuple[str, str] = DEFAULT_PAIR,
    slope: float = DEFAULT_SLOPE,
    law: str = DEFAULT_LAW,
    rv: float = DEFAULT_RV,
    truncate: bool = True,
    redshift: float = 0.0,
    rest_wavelengths: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Infer the FVG colour-excess posterior from a blue/red slope pair."""
    blue, red = map(str, pair)
    required = [f"R_{key(blue)}", f"R_{key(red)}"]
    missing = [column for column in required if column not in posterior]
    if missing:
        raise KeyError(f"FVG posterior is missing {missing}")

    rb = pd.to_numeric(posterior[required[0]], errors="coerce").to_numpy(float)
    rr = pd.to_numeric(posterior[required[1]], errors="coerce").to_numpy(float)
    wave_blue = float(wavelengths[blue])
    wave_red = float(wavelengths[red])
    curve_waves = np.array([wave_blue, wave_red], dtype=float)
    if rest_wavelengths:
        curve_waves /= 1.0 + float(redshift)
    k_pair, effective_rv = extinction_k(curve_waves, law=law, rv=rv)
    ratio_intrinsic = (wave_red / wave_blue) ** float(slope)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_observed = rb / rr
        delta_a = -2.5 * np.log10(ratio_observed / ratio_intrinsic)
        ebv_raw = delta_a / float(k_pair[0] - k_pair[1])
    valid = np.isfinite(ebv_raw) & (ratio_observed > 0.0)
    if not np.any(valid):
        raise ValueError("FVG posterior has no valid positive blue/red ratios")
    ebv_raw = ebv_raw[valid]
    ebv_used = np.maximum(ebv_raw, 0.0) if truncate else ebv_raw.copy()

    ratio_q = q68(ratio_observed[valid])
    raw_q = q68(ebv_raw)
    used_q = q68(ebv_used)
    summary = {
        "blue": blue,
        "red": red,
        "wave_blue_observed": wave_blue,
        "wave_red_observed": wave_red,
        "curve_frame": "rest" if rest_wavelengths else "observed_legacy",
        "wave_blue_curve": float(curve_waves[0]),
        "wave_red_curve": float(curve_waves[1]),
        "law": law,
        "rv_effective": effective_rv,
        "slope_Flambda_exponent": -float(slope),
        "ratio_intrinsic": ratio_intrinsic,
        "k_blue": float(k_pair[0]),
        "k_red": float(k_pair[1]),
        "truncate": bool(truncate),
        "truncated_fraction": float(np.mean(ebv_raw < 0.0)),
        "n_source": int(len(posterior)),
        "n_valid": int(len(ebv_used)),
        "ratio_observed_p16": ratio_q[0],
        "ratio_observed_p50": ratio_q[1],
        "ratio_observed_p84": ratio_q[2],
        "ebv_raw_p16": raw_q[0],
        "ebv_raw_p50": raw_q[1],
        "ebv_raw_p84": raw_q[2],
        "ebv_p16": used_q[0],
        "ebv_p50": used_q[1],
        "ebv_p84": used_q[2],
    }
    return ebv_raw, ebv_used, summary


def distance_flux_columns(cfg: dict[str, Any], band: str) -> dict[str, str]:
    """Resolve current comp0 distance-flux columns without changing epsilon."""
    bright, epsilon = disk_distance.distance_flux_columns(cfg, band)
    if "F_Bright" not in bright:
        raise ValueError(f"cannot derive faint/delta columns from {bright!r}")
    return {
        "bright": bright,
        "faint": bright.replace("F_Bright", "F_Faint", 1),
        "delta": bright.replace("F_Bright", "Delta_F", 1),
        "epsilon": epsilon,
    }


def apply_nuclear_extinction(
    target: pd.DataFrame,
    ebv_raw: np.ndarray,
    ebv_used: np.ndarray,
    cfg: dict[str, Any],
    bands: list[str],
    wavelengths: dict[str, float],
    *,
    law: str = DEFAULT_LAW,
    rv: float = DEFAULT_RV,
    seed: int = DEFAULT_SEED,
    redshift: float = 0.0,
    rest_wavelengths: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scale an existing comp0 joint posterior by independent total-FVG draws."""
    if len(ebv_raw) != len(ebv_used):
        raise ValueError("raw and used E(B-V) posteriors must have equal length")
    if not len(target) or not len(ebv_used):
        raise ValueError("empty target or E(B-V) posterior")

    rng = np.random.default_rng(int(seed))
    source_rows = rng.choice(
        len(ebv_used),
        size=len(target),
        replace=len(target) > len(ebv_used),
    )
    selected_raw = np.asarray(ebv_raw, float)[source_rows]
    selected_used = np.asarray(ebv_used, float)[source_rows]
    corrected = target.copy()
    corrected.insert(1, "ebv_source_row", source_rows.astype(int))
    corrected.insert(2, "ebv_raw_nuc", selected_raw)
    corrected.insert(3, "ebv_nuc", selected_used)

    curve_waves = np.array([float(wavelengths[band]) for band in bands], float)
    if rest_wavelengths:
        curve_waves /= 1.0 + float(redshift)
    kvals, effective_rv = extinction_k(curve_waves, law=law, rv=rv)
    summary_rows = []
    for band, kval in zip(bands, kvals):
        cols = distance_flux_columns(cfg, band)
        missing = [column for column in cols.values() if column not in corrected]
        if missing:
            raise KeyError(f"target distance-flux posterior is missing {band}: {missing}")
        factor = 10.0 ** (0.4 * float(kval) * selected_used)
        before = {
            name: pd.to_numeric(corrected[column], errors="coerce").to_numpy(float)
            for name, column in cols.items()
        }
        corrected[f"corr_nuc_{key(band)}"] = factor
        for name in ("faint", "bright", "delta"):
            corrected[cols[name]] = before[name] * factor
        with np.errstate(divide="ignore", invalid="ignore"):
            epsilon_check = corrected[cols["faint"]].to_numpy(float) / corrected[
                cols["bright"]
            ].to_numpy(float)
        finite = np.isfinite(epsilon_check) & np.isfinite(before["epsilon"])
        max_epsilon_difference = (
            float(np.max(np.abs(epsilon_check[finite] - before["epsilon"][finite])))
            if np.any(finite)
            else np.nan
        )
        factor_q = q68(factor)
        before_q = q68(before["bright"])
        after_q = q68(corrected[cols["bright"]].to_numpy(float))
        summary_rows.append({
            "band": band,
            "wavelength_observed": float(wavelengths[band]),
            "wavelength_curve": float(curve_waves[bands.index(band)]),
            "law": law,
            "rv_effective": effective_rv,
            "k_nuc": float(kval),
            "corr_nuc_p16": factor_q[0],
            "corr_nuc_p50": factor_q[1],
            "corr_nuc_p84": factor_q[2],
            "h0_scale_p50": float(np.sqrt(factor_q[1])),
            "F_Bright_before_p16": before_q[0],
            "F_Bright_before_p50": before_q[1],
            "F_Bright_before_p84": before_q[2],
            "F_Bright_after_p16": after_q[0],
            "F_Bright_after_p50": after_q[1],
            "F_Bright_after_p84": after_q[2],
            "epsilon_max_abs_difference": max_epsilon_difference,
        })
    return corrected, pd.DataFrame(summary_rows)


def branch_root(
    source: Path,
    cfg: dict[str, Any],
    branch_id: str,
    stage: str | None = None,
) -> Path:
    if stage is not None:
        return mica_analysis_dir(source, stage_config(cfg, stage), "comp0")
    mica = dict(cfg.get("mica_round1", {}) or {})
    ncomp = int(mica.get("ncomp", 2))
    root = rel(source, mica.get("output_root", "runs/mica"))
    return root / f"{ncomp}comp" / "result" / branch_result_relative_dir(mica, branch_id)


def find_total_fvg_posterior(source: Path, cfg: dict[str, Any]) -> Path:
    ff = dict(cfg.get("fflux", {}) or {})
    analysis = rel(source, ff.get("out", "fflux")) / "analysis"
    source_name = str(cfg.get("source", source.name))
    bands = [str(band) for band in (ff.get("bands") or cfg.get("band_list") or [])]
    expected = analysis / f"{source_name}_total_{len(bands)}band_flux_posterior.txt"
    if expected.is_file():
        return expected
    hits = [
        path for path in sorted(analysis.glob("*_flux_posterior.txt"))
        if "_nuc_" not in path.name
    ]
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected one total FVG posterior under {analysis}, found {len(hits)}"
        )
    return hits[0]


def h0_config(
    cfg: dict[str, Any],
    *,
    input_branch_root: Path,
    output_root: Path,
    corrected_flux: Path,
) -> dict[str, Any]:
    """Route each H0 model to corrected flux and the original lag posterior."""
    result = copy.deepcopy(cfg)
    hd = dict(result.get("hubble_distance", {}) or {})
    models = dict(hd.get("models", {}) or {})
    if not models:
        raise ValueError("hubble_distance.models is required")
    suffixes = {
        "4/3": "b4_3",
        "b4_3": "b4_3",
        "standard": "b4_3",
        "2": "b2",
        "b2": "b2",
        "slim": "b2",
        "free": "free",
    }
    routed_models = {}
    for name, raw in models.items():
        model = dict(raw or {})
        lag_name = str(model.get("lag_fit_model", name)).lower()
        if lag_name not in suffixes:
            raise ValueError(f"unknown lag_fit_model {lag_name!r} for {name}")
        lag_path = input_branch_root / "lag" / (
            f"absolute_lag_posterior_{suffixes[lag_name]}.txt"
        )
        if not lag_path.is_file():
            raise FileNotFoundError(f"lag posterior not found: {lag_path}")
        # A model-specific path in the source configuration must never be able
        # to route this post-processing run back to the uncorrected posterior.
        model["flux_posterior"] = str(corrected_flux)
        model["lag_posterior"] = str(lag_path)
        routed_models[str(name)] = model
    hd["models"] = routed_models
    hd["flux_posterior"] = str(corrected_flux)
    hd["result_root"] = str(output_root)
    result["hubble_distance"] = hd
    return result


def write_h0_comparison(input_root: Path, output_root: Path) -> Path | None:
    baseline_path = input_root / "h0" / "hubble_model_summary.csv"
    corrected_path = output_root / "h0" / "hubble_model_summary.csv"
    if not baseline_path.is_file() or not corrected_path.is_file():
        return None
    baseline = pd.read_csv(baseline_path)
    corrected = pd.read_csv(corrected_path)
    keep = ["distance_model", "H0", "H0_err_low", "H0_err_high", "f0"]
    before = baseline[keep].rename(columns={column: f"{column}_before" for column in keep[1:]})
    after = corrected[keep].rename(columns={column: f"{column}_after" for column in keep[1:]})
    comparison = before.merge(after, on="distance_model", validate="one_to_one")
    comparison["H0_shift"] = comparison["H0_after"] - comparison["H0_before"]
    comparison["H0_ratio"] = comparison["H0_after"] / comparison["H0_before"]
    path = output_root / "h0" / "h0_before_after.csv"
    comparison.to_csv(path, index=False)
    return path


def run(args: argparse.Namespace) -> dict[str, Path]:
    source = source_dir(args.source)
    config_path = rel(source, args.config or "config/source_config.yaml")
    cfg = read_yaml(config_path)
    hd = dict(cfg.get("hubble_distance", {}) or {})
    branch_id = str(args.posterior_branch or hd.get("posterior_branch", "comp0"))
    stage = str(args.stage) if args.stage else None
    input_root = branch_root(source, cfg, branch_id, stage)
    output_root = (
        rel(source, args.output_root)
        if args.output_root
        else input_root / "nuclear_extinction_total_fvg"
    )

    total_path = (
        rel(source, args.ebv_flux_posterior)
        if args.ebv_flux_posterior
        else find_total_fvg_posterior(source, cfg)
    )
    target_path = (
        rel(source, args.target_flux_posterior)
        if args.target_flux_posterior
        else input_root / "fflux" / "analysis" / "distance_flux_posterior.txt"
    )
    if not target_path.is_file():
        raise FileNotFoundError(
            f"completed comp0 distance-flux posterior not found: {target_path}. "
            "Do not start FFA here; finish it separately, or explicitly select a "
            "completed archived posterior for a labelled sensitivity run."
        )
    validate_output_root(
        output_root,
        input_root=input_root,
        total_path=total_path,
        target_path=target_path,
    )

    bands, wavelengths = disk_distance.bands_and_waves(cfg)
    total = pd.read_csv(total_path, sep=r"\s+")
    target = pd.read_csv(target_path, sep=r"\s+")
    pair = tuple(args.pair or DEFAULT_PAIR)
    redshift = disk_distance.redshift(cfg)
    ebv_raw, ebv_used, ebv_summary = infer_fvg_ebv(
        total,
        wavelengths,
        pair=pair,
        slope=float(args.slope),
        law=args.law,
        rv=float(args.rv),
        truncate=not args.no_truncate,
        redshift=redshift,
        rest_wavelengths=args.rest_wavelengths,
    )
    corrected, correction_summary = apply_nuclear_extinction(
        target,
        ebv_raw,
        ebv_used,
        cfg,
        bands,
        wavelengths,
        law=args.law,
        rv=float(args.rv),
        seed=int(args.seed),
        redshift=redshift,
        rest_wavelengths=args.rest_wavelengths,
    )

    analysis = output_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    posterior_path = analysis / "distance_flux_posterior.txt"
    ebv_path = analysis / "fvg_nuclear_ebv.csv"
    correction_path = analysis / "nuclear_flux_summary.csv"
    corrected.to_csv(posterior_path, sep=" ", index=False, float_format="%.10e")
    pd.DataFrame([{
        "source": cfg.get("source", source.name),
        "ebv_flux_posterior": str(total_path),
        "target_flux_posterior": str(target_path),
        "seed": int(args.seed),
        **ebv_summary,
    }]).to_csv(ebv_path, index=False)
    correction_summary.to_csv(correction_path, index=False)

    routed = h0_config(
        cfg,
        input_branch_root=input_root,
        output_root=output_root,
        corrected_flux=posterior_path,
    )
    if stage is not None:
        routed["_mica_stage"] = stage
    run_path = output_root / "run.yaml"
    with run_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({
            "method": "total-light FVG nuclear extinction applied post hoc to comp0",
            "does_not_run": ["MICA", "FFA"],
            "source_config": str(config_path),
            "posterior_branch": branch_id,
            "mica_stage": stage,
            "ebv_flux_posterior": str(total_path),
            "target_flux_posterior": str(target_path),
            "corrected_flux_posterior": str(posterior_path),
            "output_root": str(output_root),
            "pair": list(pair),
            "law": args.law,
            "rv_requested": float(args.rv),
            "slope": float(args.slope),
            "truncate": not args.no_truncate,
            "curve_frame": "rest" if args.rest_wavelengths else "observed_legacy",
            "seed": int(args.seed),
        }, handle, sort_keys=False, allow_unicode=True)

    # emcee uses NumPy's legacy global random state internally.  Seeding it
    # here makes the complete lightweight H0 run reproducible, not merely the
    # initial walker locations built by disk_distance.py.
    np.random.seed(int(args.seed))
    disk_distance.run(
        source,
        no_mcmc=args.no_h0_mcmc,
        _cfg=routed,
    )
    comparison_path = None
    if not args.no_h0_mcmc:
        default_target = input_root / "fflux" / "analysis" / "distance_flux_posterior.txt"
        if args.baseline_result_root:
            baseline_root = rel(source, args.baseline_result_root)
            comparison_path = write_h0_comparison(baseline_root, output_root)
            if comparison_path is None:
                raise FileNotFoundError(
                    f"baseline H0 summary not found under {baseline_root}"
                )
        elif target_path.resolve() == default_target.resolve():
            comparison_path = write_h0_comparison(input_root, output_root)
        else:
            print(
                "H0 before/after comparison skipped: an alternate target flux "
                "posterior requires explicit --baseline-result-root provenance."
            )
    print(f"FVG E(B-V): {ebv_summary['ebv_p50']:.5f} "
          f"[-{ebv_summary['ebv_p50'] - ebv_summary['ebv_p16']:.5f}, "
          f"+{ebv_summary['ebv_p84'] - ebv_summary['ebv_p50']:.5f}]")
    print(f"corrected flux posterior: {posterior_path}")
    if comparison_path is not None:
        print(f"H0 before/after: {comparison_path}")
    return {
        "ebv": ebv_path,
        "correction": correction_path,
        "posterior": posterior_path,
        "run": run_path,
        **({"comparison": comparison_path} if comparison_path else {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FVG nuclear-extinction and H0 post-processing; never runs MICA/FFA"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--posterior-branch")
    parser.add_argument("--stage", help="MICA config section; default runs every active stage")
    parser.add_argument("--ebv-flux-posterior", type=Path)
    parser.add_argument("--target-flux-posterior", type=Path)
    parser.add_argument(
        "--baseline-result-root",
        type=Path,
        help="explicit matching raw distance/H0 root for a before/after table",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--pair", nargs=2, metavar=("BLUE", "RED"), default=DEFAULT_PAIR)
    parser.add_argument("--law", default=DEFAULT_LAW)
    parser.add_argument("--rv", type=float, default=DEFAULT_RV)
    parser.add_argument("--slope", type=float, default=DEFAULT_SLOPE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-truncate", action="store_true")
    parser.add_argument(
        "--rest-wavelengths",
        action="store_true",
        help="evaluate the extinction curve in the source rest frame; default reproduces legacy code",
    )
    parser.add_argument("--no-h0-mcmc", action="store_true")
    args = parser.parse_args()
    if args.stage:
        run(args)
        return
    source = source_dir(args.source)
    cfg = read_yaml(rel(source, args.config or "config/source_config.yaml"))
    for stage in configured_stages(cfg):
        stage_args = argparse.Namespace(**vars(args))
        stage_args.stage = stage
        run(stage_args)


if __name__ == "__main__":
    main()
