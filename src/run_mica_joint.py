#!/usr/bin/env python3
"""Fit all configured response bands jointly, without downstream processing.

Standalone: python src/run_mica_joint.py SOURCE --stage mica_gamma2_broad
Integrated: python src/run_mica.py SOURCE --stage mica_gamma2_broad --joint
The existing launcher supplies MPI, checkpoints, and fit-only completion checks.
"""
from __future__ import annotations

from pathlib import Path
import math
import sys

import numpy as np


PRIOR_PATH = Path("param/joint_parameter_priors.txt")


def input_files(src: Path, source: str, fit: dict) -> dict[str, Path]:
    bands = list(fit["responses"])
    if len(bands) < 2 or len(set(bands)) != len(bands) or fit["driver"] in bands:
        raise ValueError("joint fitting needs at least two distinct response bands, excluding the driver")
    for band in [fit["driver"], *bands]:
        if not isinstance(band, str) or Path(band).name != band or "/" in band or "\\" in band:
            raise ValueError("band names must be simple file-name components")
    folder = Path(fit["prepared_dir"])
    if not folder.is_absolute():
        folder = src / folder
    return {"cont.txt": folder / f"{source}_{fit['driver']}.txt", **{
        f"line_{band}.txt": folder / f"{source}_{band}.txt" for band in bands
    }}


def _native_times(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve)
    if (curve.ndim != 2 or curve.shape[1] != 3 or len(curve) < 2
            or not np.isfinite(curve).all() or np.any(curve[:, 2] <= 0)):
        raise ValueError("MICA inputs must have at least two finite time/flux/positive-error rows")
    # Match pymica's existing data serialization, keeping the single-band inputs unchanged.
    times = np.array([float(f"{t:.6e}") for t in curve[:, 0]])
    if np.any(np.diff(times) <= 0):
        raise ValueError("MICA input times must remain strictly increasing at native precision")
    return times


def setup(fit: dict, data: dict) -> tuple[dict, str]:
    """Keep each band's pairwise effective Gamma priors in the joint parameter table."""
    import run_mica

    if fit["_native_type_tf"] != "gamma" or fit["ncomp"] != 2 or fit.get("type_lag_prior") != 4:
        raise ValueError("the joint test supports two Gamma components with explicit type-4 centroid priors")
    input_files(Path("."), "source", fit)  # validate band identity/order
    bands = list(fit["responses"])
    if len(data) != 1 or len(next(iter(data.values()))) != len(bands) + 1:
        raise ValueError("joint data must contain one driver followed by all configured responses in one dataset")
    curves = next(iter(data.values()))
    times = [_native_times(curve) for curve in curves]
    kwargs = run_mica.setup_kwargs(fit, data, bands[0])
    rows = []
    for j, band in enumerate(bands):
        lags = run_mica.component_prior_for_band(fit, band, "lag_prior")
        if lags is None:
            raise ValueError("joint type-4 Gamma fitting requires explicit centroid bounds for every band")
        # pymica serializes explicit prior endpoints with %f for pairwise fits.
        lags = [[float(f"{value:f}") for value in bounds] for bounds in lags]
        widths = run_mica.component_prior_for_band(fit, band, "width_prior", positive=True)
        if widths is not None:
            widths = [[float(f"{value:f}") for value in bounds] for bounds in widths]
        else:
            pair = [times[0], times[j + 1]]
            span = max(float(np.ptp(t)) for t in pair)
            cadence = min(float(np.ptp(t)) / (len(t) - 1) for t in pair)
            lo, hi = cadence / 10.0, span / 3.0
            if fit.get("width_limit") is not None:
                limits = fit["width_limit"]
                if limits[0] is not None:
                    lo = float(limits[0])
                if limits[1] is not None:
                    hi = min(float(limits[1]), hi)
            if not 0 < lo < hi:
                raise ValueError(f"invalid effective width bounds for {band}: {lo}, {hi}")
            widths = [[lo, hi], [lo, hi]]
        if any(lo >= hi for lo, hi in lags) or any(not 0 < lo < hi for lo, hi in widths):
            raise ValueError(f"invalid prior bounds at native precision for {band}")
        start = 3 + j * 7
        for k in range(2):
            lo, hi = lags[k]
            rows.append(f"{start + 2 + 3*k} {band}_{k}_centroid UNI {lo:.17g} {hi:.17g} 0 0\n")
            lo, hi = map(math.log, widths[k])
            rows.append(f"{start + 3 + 3*k} {band}_{k}_log_width UNI {lo:.17g} {hi:.17g} 0 0\n")
    return kwargs, "".join(rows)


def register_priors(model, text: str, comm, rank: int) -> None:
    error = None
    if rank == 0:
        try:
            PRIOR_PATH.parent.mkdir(parents=True, exist_ok=True)
            PRIOR_PATH.write_text(text, encoding="utf-8")
        except Exception as exc:
            error = str(exc)
    error = comm.bcast(error, root=0)
    if error:
        raise RuntimeError(error)
    comm.Barrier()
    model.set_priors(str(PRIOR_PATH))


def main() -> None:
    import run_mica

    if "--joint" not in sys.argv:
        sys.argv.append("--joint")
    run_mica.main()


if __name__ == "__main__":
    main()
