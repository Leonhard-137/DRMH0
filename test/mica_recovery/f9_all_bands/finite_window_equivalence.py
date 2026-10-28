#!/usr/bin/env python3
"""Analytic finite-window equivalence: independent GP vs displaced response.

This is a specified analytic toy on the actual B/W2 sampling and measurement
errors. It is NOT an additional MICA fit or the exact original disk injection.
The true disk is a unit-gain delta response at 1 d, the driver a DRW with
timescale 20 d and stationary sigma 0.2. Independent target covariance is
chosen to equal that of a shape-2 Gamma filtered independent copy of the DRW,
with scale 20 d and stationary RMS 0.3 times the disk. A competing same-driver
Gamma component has the same covariance but variable onset c. Compare the
Gaussian likelihoods after removing both unknown constant mean modes.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import block_diag, helmert, cholesky, solve_triangular, eigvalsh

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "f9_m2_b/cases/disk_only/runs/mica/2comp/run_W2_to_B_2comp_gamma/input"
OUT = HERE / "outputs/independent_influence/finite_window_equivalence.json"


def main():
    x, y = [np.loadtxt(SOURCE / name) for name in ("cont.txt", "line.txt")]
    tx, ty = x[:, 0], y[:, 0]
    d, w, sigma, disk_lag, rms_ratio = 20., 20., .2, 1., .3
    area = rms_ratio / np.sqrt(3 / 8)
    k = lambda h: sigma**2 * np.exp(-np.abs(h) / d)
    hyy = ty[None, :] - ty[:, None]
    hxy = ty[None, :] - tx[:, None]
    r = np.abs(hyy) / d
    kuu = area**2 * sigma**2 * np.exp(-r) * (3 + 3*r + r*r) / 8
    cxx = k(tx[None, :] - tx[:, None]) + np.diag(x[:, 2]**2)
    cxy = k(hxy - disk_lag)
    cyy = k(hyy) + kuu + np.diag(y[:, 2]**2)
    true_cov = np.block([[cxx, cxy], [cxy.T, cyy]])
    # Orthogonal contrasts eliminate the same two unknown constant modes used
    # in MICA's flat-mean marginalized Gaussian likelihood.
    g = block_diag(helmert(len(tx), full=False), helmert(len(ty), full=False))
    c0 = g @ true_cov @ g.T
    l0 = cholesky(c0, lower=True)

    def cross(h, onset):
        z = (h - onset) / d
        ans = np.empty_like(z)
        negative = z <= 0
        ans[negative] = np.exp(z[negative]) / 4
        p = z[~negative]
        ans[~negative] = np.exp(-p) * (p*p/2 + p/2 + 1/4)
        return area * sigma**2 * ans

    rows = []
    for onset in [0., 100., 400., 700., 710., 750., 800., 850., 900., 1000., 1100.]:
        b = cross(hxy, onset)
        sl = cross(hyy + disk_lag, onset)
        delta = np.block([[np.zeros_like(cxx), b], [b.T, sl + sl.T]])
        delta = g @ delta @ g.T
        v = solve_triangular(l0, delta, lower=True)
        whitened = solve_triangular(l0, v.T, lower=True).T
        lam = eigvalsh((whitened + whitened.T)/2)
        if lam.min() <= -1:
            raise ValueError("Competing covariance is not positive definite")
        term = np.log1p(lam) - lam / (1 + lam)
        near = np.abs(lam) < 1e-4
        z = lam[near]
        term[near] = sum(((-1)**n) * (n-1)/n * z**n for n in range(2, 8))
        rows.append({"onset_days": onset, "centroid_days": onset + 2*w,
                     "KL_true_to_response_nats": float(term.sum()/2),
                     "max_absolute_driver_long_correlation": float(np.max(np.abs(b)) / (sigma**2 * rms_ratio)),
                     "whitened_difference_frobenius": float(np.linalg.norm(whitened))})
    result = {"description": __doc__, "source_inputs": str(SOURCE),
              "n_driver": len(tx), "n_target": len(ty), "contrast_dimension": len(c0),
              "span_days": float(max(tx[-1], ty[-1])-min(tx[0], ty[0])),
              "drw_timescale_days": d, "drw_stationary_sigma": sigma,
              "disk_delta_lag_days": disk_lag, "gamma_scale_days": w,
              "gamma_area": area, "independent_to_disk_stationary_rms_ratio": rms_ratio,
              "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
