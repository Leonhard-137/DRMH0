#!/usr/bin/env python3
"""Independently verify native CDNest quadrature; never replace its weights.

This reader reconstructs the deterministic within-level prior-mass nodes and
trapezoidal weights from levels.txt and sample_info.txt, without calling either
CDNest postprocessor. Singleton levels use the well-defined midpoint node.
Native singleton NaNs are accepted only in the diagnostic logX column; every
posterior weight must already be finite and agree at native print precision.
This is an arithmetic/output check, not a nested-sampling convergence test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


def verify_run(folder: Path) -> dict:
    folder = Path(folder)
    result = {"path": str(folder.resolve()), "status": "failed", "errors": []}
    try:
        info = np.loadtxt(folder / "sample_info.txt", ndmin=2)
        levels = np.loadtxt(folder / "levels.txt", ndmin=2)
        native = np.loadtxt(folder / "weights.txt", ndmin=2)
        sample = np.loadtxt(folder / "sample.txt", ndmin=2)
        if info.shape[1] < 3 or levels.shape[1] < 2 or native.shape[1] != 3:
            raise ValueError("unexpected native column schema")
        if not (len(info) == len(native) == len(sample)) or len(info) == 0:
            raise ValueError("native sample/info/weight rows do not align")
        if not np.isfinite(info[:, :3]).all() or not np.isfinite(levels[:, :2]).all():
            raise ValueError("non-finite sample info or levels")
        if not np.isfinite(sample).all() or not np.isfinite(native[:, 1:]).all():
            raise ValueError("non-finite physical sample, likelihood, or weight")
        if np.any(np.diff(levels[:, 0]) > 0) or np.any(np.diff(levels[:, 1]) < 0):
            raise ValueError("levels must have decreasing logX and increasing logL")
        assigned = info[:, 0].astype(int)
        if np.any(assigned != info[:, 0]) or np.any(assigned < 0) or np.any(assigned >= len(levels)):
            raise ValueError("invalid sample level assignments")
        if np.any(native[:, 2] < 0) or native[:, 2].sum() <= 0:
            raise ValueError("native posterior weights must be nonnegative with positive sum")
        sandwich = np.maximum(assigned, np.searchsorted(levels[:, 1], info[:, 1], side="left") - 1)
        logx = np.empty(len(info))
        log_prior_weight = np.empty(len(info))
        singleton = []
        for level in range(len(levels)):
            rows = np.flatnonzero(sandwich == level)
            if len(rows) == 0:
                continue
            # Independent, three-way lexicographic sorting rather than native
            # C cmp_sample (which returns only 0/1, a portability concern).
            rows = rows[np.lexsort((info[rows, 2], info[rows, 1]))]
            n = len(rows)
            upper = levels[level, 0]
            lower = levels[level + 1, 0] if level + 1 < len(levels) else -np.inf
            minimum_u = np.exp(lower - upper)
            u = minimum_u + (1.0 - minimum_u) * np.linspace(1/(n+1), 1-1/(n+1), n)[::-1]
            nodes = upper + np.log(u)
            left = np.r_[nodes[1:], lower]
            right = np.r_[upper, nodes[:-1]]
            logx[rows] = nodes
            # Stable log(0.5 * (X_right-X_left)). Normalization below makes
            # the common 0.5 irrelevant, but it matches native quadrature.
            with np.errstate(divide="ignore", invalid="raise"):
                log_prior_weight[rows] = np.log(0.5) + right + np.log(-np.expm1(left-right))
            if n == 1:
                singleton.append(int(rows[0]))
        normalization = logsumexp(log_prior_weight)
        if not np.isfinite(normalization):
            raise ValueError("reconstructed prior quadrature has no finite mass")
        log_prior_weight -= normalization
        logz = float(logsumexp(log_prior_weight + info[:, 1]))
        if not np.isfinite(logz):
            raise ValueError("reconstructed log evidence is non-finite")
        weights = np.exp(log_prior_weight + info[:, 1] - logz)
        native_weights = native[:, 2] / native[:, 2].sum()
        bad_logx = np.flatnonzero(~np.isfinite(native[:, 0]))
        finite_logx = np.isfinite(native[:, 0])
        checks = {
            "sample_loglikelihood_alignment": bool(np.allclose(native[:, 1], info[:, 1], rtol=2e-6, atol=1e-10)),
            "native_weight_agreement": bool(np.allclose(weights, native_weights, rtol=3e-6, atol=1e-12)),
            "native_weight_L1_agreement": bool(np.abs(weights-native_weights).sum() <= 3e-6),
            "nonfinite_logx_only_singleton": set(bad_logx).issubset(singleton),
            "finite_logx_agreement": bool(np.allclose(logx[finite_logx], native[finite_logx, 0], rtol=2e-6, atol=1e-10)),
        }
        diagnostics_path = folder / "diagnostics.json"
        raw_logz = None
        if diagnostics_path.exists():
            raw_logz = json.loads(diagnostics_path.read_text()).get("raw_log_evidence")
            if raw_logz is not None:
                checks["raw_logevidence_agreement"] = bool(np.isclose(logz, raw_logz, rtol=1e-9, atol=1e-8))
        result.update({
            "checks": checks,
            "sample_count": len(info),
            "level_count": len(levels),
            "singleton_rows_zero_based": singleton,
            "native_nonfinite_logx_rows_zero_based": bad_logx.tolist(),
            "weight_max_abs_difference": float(np.max(np.abs(weights-native_weights))),
            "weight_L1_difference": float(np.abs(weights-native_weights).sum()),
            "finite_logx_max_abs_difference": float(np.max(np.abs(logx[finite_logx]-native[finite_logx, 0]))) if finite_logx.any() else 0.0,
            "recomputed_log_evidence": logz,
            "native_raw_log_evidence": raw_logz,
            "native_weight_sum_before_normalization": float(native[:, 2].sum()),
            "status": "passed" if all(checks.values()) else "failed",
            "interpretation": "Printed native quadrature checked; convergence, level calibration and evidence reliability are not established.",
            "native_files_modified": False,
        })
        result["errors"] = [name for name, passed in checks.items() if not passed]
    except (OSError, ValueError, IndexError, FloatingPointError) as error:
        result["errors"].append(str(error))
    return result


def verify_seed_pair(first: Path, second: Path) -> dict:
    """Check the seeded physical trajectories/weights, excluding native resampling."""
    try:
        first, second = Path(first), Path(second)
        configs = [json.loads((p / "sampler_config.json").read_text()) for p in (first, second)]
        samples = [np.loadtxt(p / "sample.txt", ndmin=2) for p in (first, second)]
        weights = [np.loadtxt(p / "weights.txt", ndmin=2) for p in (first, second)]
        info = [np.loadtxt(p / "sample_info.txt", ndmin=2) for p in (first, second)]
        checks = {
            "same_declared_seed": configs[0]["seed"] == configs[1]["seed"],
            "physical_samples_identical": bool(np.array_equal(*samples, equal_nan=True)),
            "native_weight_logx_loglikelihood_identical": bool(np.array_equal(*weights, equal_nan=True)),
            "sample_loglikelihood_identical": bool(np.array_equal(info[0][:, 1], info[1][:, 1])),
        }
        return {"paths": [str(first.resolve()), str(second.resolve())], "checks": checks,
                "status": "passed" if all(checks.values()) else "failed",
                "excluded": "Native equal-weight resampling and initial floor-level random tiebreaker remain time seeded."}
    except (OSError, ValueError, KeyError, IndexError) as error:
        return {"status": "failed", "errors": [str(error)]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--root", type=Path, help="Bounded output tree to search for weights.txt")
    parser.add_argument("--seed-pair", type=Path, nargs=2, metavar=("FIRST", "SECOND"))
    parser.add_argument("--report", type=Path, help="Explicit report path; native run files are never changed")
    args = parser.parse_args(argv)
    folders = list(args.runs)
    if args.root:
        folders.extend(path.parent for path in args.root.rglob("weights.txt"))
    folders = sorted(set(folder.resolve() for folder in folders))
    if not folders and not args.seed_pair:
        parser.error("provide at least one run, --root, or --seed-pair")
    reports = [verify_run(folder) for folder in folders]
    payload = {"runs": reports, "all_passed": all(r["status"] == "passed" for r in reports)}
    if args.seed_pair:
        payload["seed_pair"] = verify_seed_pair(*args.seed_pair)
        payload["all_passed"] &= payload["seed_pair"]["status"] == "passed"
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if args.report:
        args.report.write_text(text)
    print(text, end="")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
