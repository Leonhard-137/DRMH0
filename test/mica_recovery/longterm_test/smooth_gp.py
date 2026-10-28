"""Sample the original smooth Gaussian process at requested times.

C(h) = sigma**2 * (1 + |h| / tau) * exp(-|h| / tau).
For a shared realization across bands, pass all requested times in one call.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def sample_smooth_gp(
    times_days: np.ndarray, tau_days: float, sigma: float = 1.0, seed: int | None = None
) -> np.ndarray:
    """Draw S(t) with zero ensemble mean and stationary standard deviation sigma.

    The finite realization is neither demeaned nor rescaled. Repeated times
    receive the same value up to floating-point precision.
    """
    times = np.asarray(times_days, dtype=float)
    if times.ndim != 1 or times.size == 0 or not np.isfinite(times).all():
        raise ValueError("times_days must be a nonempty, finite 1-D array")
    if not np.isfinite(tau_days) or tau_days <= 0:
        raise ValueError("tau_days must be finite and positive")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")

    lag = np.abs(times[:, None] - times[None, :]) / tau_days
    covariance = sigma**2 * (1 + lag) * np.exp(-lag)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = 256 * np.finfo(float).eps * len(times) * max(
        float(eigenvalues[-1]), np.finfo(float).tiny
    )
    if eigenvalues[0] < -tolerance:
        raise ValueError("GP covariance is not positive semidefinite")

    rng = np.random.default_rng(seed)
    return eigenvectors @ (
        np.sqrt(np.maximum(eigenvalues, 0)) * rng.standard_normal(len(times))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    times = parser.add_mutually_exclusive_group(required=True)
    times.add_argument("--times-file", type=Path, help="Text file; read times from column 1")
    times.add_argument("--cadence-days", type=float, help="Regular sampling interval in days")
    parser.add_argument("--n-points", type=int, help="Number of points for regular sampling")
    parser.add_argument("--start-day", type=float, default=0.0)
    parser.add_argument("--tau-days", type=float, required=True)
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Stationary standard deviation, not realized sample RMS")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.times_file is not None:
        if args.n_points is not None:
            parser.error("--n-points is only used with --cadence-days")
        times_days = np.loadtxt(args.times_file, ndmin=2)[:, 0]
    else:
        if args.n_points is None or args.n_points < 1:
            parser.error("--cadence-days requires --n-points >= 1")
        if not np.isfinite(args.cadence_days) or args.cadence_days <= 0:
            parser.error("--cadence-days must be finite and positive")
        times_days = args.start_day + args.cadence_days * np.arange(args.n_points)

    signal = sample_smooth_gp(times_days, args.tau_days, args.sigma, args.seed)
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, np.column_stack((times_days, signal)),
               fmt="%.12g", header="time_days S")
    print(args.output)


if __name__ == "__main__":
    main()
