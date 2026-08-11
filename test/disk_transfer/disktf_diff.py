#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ============================================================
# Physical constants, SI
# ============================================================

h = 6.62607015e-34          # J s
c_light = 2.99792458e8      # m s^-1
kB = 1.380649e-23           # J K^-1


@dataclass
class DiskConfig:
    # Radii are in light-days
    Rin: float = 0.1
    Rout: float = 30.0
    R0: float = 1.0

    # Temperatures at R0
    TB: float = 29000.0
    TF: float = 26500.0

    # T(R) = T0 * (R/R0)^(-b)
    b: float = 0.75

    # Inclination in degrees
    inc_deg: float = 30.0

    # Wavelength in Angstrom
    lam_A: float = 4392.0

    # Numerical resolution
    NR: int = 800
    Ntau: int = 2000

    # "finite_difference" or "linearized"
    response_mode: str = "finite_difference"


def _check_positive(name: str, x: float) -> None:
    if not np.isfinite(x) or x <= 0:
        raise ValueError(f"{name} must be positive and finite; got {x}")


def planck_nu(nu, T):
    """
    Planck function B_nu(T) in SI units.

    Parameters
    ----------
    nu : float or ndarray
        Frequency in Hz.
    T : float or ndarray
        Temperature in K.
    """
    nu = np.asarray(nu, dtype=float)
    T = np.asarray(T, dtype=float)

    x = h * nu / (kB * T)
    x = np.clip(x, 1e-12, 700.0)

    return (2.0 * h * nu**3 / c_light**2) / np.expm1(x)


def dplanck_nu_dT(nu, T):
    """
    Analytic derivative dB_nu/dT.
    """
    nu = np.asarray(nu, dtype=float)
    T = np.asarray(T, dtype=float)

    x = h * nu / (kB * T)
    x = np.clip(x, 1e-12, 700.0)

    ex = np.exp(x)
    A = 2.0 * h * nu**3 / c_light**2

    return A * ex * x / (T * (ex - 1.0) ** 2)


def annulus_weight(nu: float, R: float, disk: DiskConfig) -> float:
    """
    Response weight of one annulus.

    finite_difference:
        B_nu(TB) - B_nu(TF)

    linearized:
        dB_nu/dT at Tmid times Delta T
    """
    T_bright = disk.TB * (R / disk.R0) ** (-disk.b)
    T_faint = disk.TF * (R / disk.R0) ** (-disk.b)

    if disk.response_mode == "finite_difference":
        return float(planck_nu(nu, T_bright) - planck_nu(nu, T_faint))

    if disk.response_mode == "linearized":
        T_mid = 0.5 * (T_bright + T_faint)
        dT = T_bright - T_faint
        return float(dplanck_nu_dT(nu, T_mid) * dT)

    raise ValueError("response_mode must be 'finite_difference' or 'linearized'")


def normalize_tf(tau: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """
    Normalize transfer function so that integral Psi(tau) d tau = 1.
    """
    dtau = float(np.median(np.diff(tau)))
    area = float(np.sum(psi) * dtau)

    if not np.isfinite(area) or area <= 0:
        raise RuntimeError("Transfer function has non-positive or invalid area.")

    return psi / area


def disk_transfer_function(disk: DiskConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute theoretical thin-disk transfer function.

    Parameters
    ----------
    disk : DiskConfig
        Disk and numerical parameters.

    Returns
    -------
    tau : ndarray
        Delay grid in days.

    psi : ndarray
        Normalized transfer function density.
        Normalization: sum(psi) * dtau = 1.

    Notes
    -----
    Radius is measured in light-days, so the delay is also in days.

    For an annulus at radius R and azimuth angle theta, the delay is

        tau = R * (1 + sin(i) * sin(theta))

    giving a delay range

        R * (1 - sin(i)) <= tau <= R * (1 + sin(i)).
    """
    for name in ["Rin", "Rout", "R0", "TB", "TF", "b", "lam_A"]:
        _check_positive(name, getattr(disk, name))

    if disk.Rout <= disk.Rin:
        raise ValueError("Rout must be greater than Rin.")

    if disk.NR < 2:
        raise ValueError("NR must be >= 2.")

    if disk.Ntau < 20:
        raise ValueError("Ntau must be >= 20.")

    inc = np.radians(disk.inc_deg)
    sin_i = float(np.sin(inc))
    cos_i = float(np.cos(inc))

    if cos_i < 0:
        raise ValueError("inc_deg should be between 0 and 90 degrees.")

    # Wavelength and frequency
    lam_m = disk.lam_A * 1e-10
    nu = c_light / lam_m

    # Radial grid in light-days
    R_edges = np.logspace(np.log10(disk.Rin), np.log10(disk.Rout), disk.NR + 1)
    R_mid = np.sqrt(R_edges[:-1] * R_edges[1:])
    dR = R_edges[1:] - R_edges[:-1]

    # Delay grid
    tau_max = disk.Rout * (1.0 + sin_i) * 1.02
    tau_edges = np.linspace(0.0, tau_max, disk.Ntau + 1)
    tau = 0.5 * (tau_edges[:-1] + tau_edges[1:])
    dtau = tau_edges[1] - tau_edges[0]

    psi = np.zeros_like(tau)

    for R, dRi in zip(R_mid, dR):
        W = annulus_weight(nu, float(R), disk)

        # projected annulus area response
        ring_prefactor = W * R * dRi * cos_i

        if sin_i < 1e-12:
            # face-on disk: every azimuth has tau = R
            idx = np.searchsorted(tau_edges, R) - 1
            idx = int(np.clip(idx, 0, len(psi) - 1))

            ring_mass = 2.0 * np.pi * ring_prefactor
            psi[idx] += ring_mass / dtau
            continue

        # tau = R * (1 + sin_i * sin(theta))
        # u = sin(theta) = (tau/R - 1) / sin_i
        u_left = (tau_edges[:-1] / R - 1.0) / sin_i
        u_right = (tau_edges[1:] / R - 1.0) / sin_i

        overlap = (u_right > -1.0) & (u_left < 1.0)
        if not np.any(overlap):
            continue

        u_left_clip = np.clip(u_left, -1.0, 1.0)
        u_right_clip = np.clip(u_right, -1.0, 1.0)

        # For each bin, total azimuth measure satisfying delay in that bin.
        # Factor 2 accounts for the two theta branches for a given sin(theta).
        dtheta = np.zeros_like(psi)
        dtheta[overlap] = 2.0 * (
            np.arcsin(u_right_clip[overlap])
            - np.arcsin(u_left_clip[overlap])
        )

        psi += ring_prefactor * dtheta / dtau

    psi = normalize_tf(tau, psi)

    return tau, psi


def tf_moments(tau: np.ndarray, psi: np.ndarray):
    """
    Return peak lag, centroid lag, and standard-deviation width.
    """
    dtau = float(np.median(np.diff(tau)))
    area = float(np.sum(psi) * dtau)

    if area <= 0 or not np.isfinite(area):
        return np.nan, np.nan, np.nan

    peak = float(tau[np.argmax(psi)])
    centroid = float(np.sum(tau * psi) * dtau / area)
    var = float(np.sum((tau - centroid) ** 2 * psi) * dtau / area)
    width = float(np.sqrt(max(var, 0.0)))

    return peak, centroid, width

def plot_disk_transfer_functions(
    bands=None,
    disk_kwargs=None,
    outname="disk_transfer_functions_linear.png",
    xlim=None,
):
    """
    Plot theoretical disk transfer functions in linear scale.

    Parameters
    ----------
    bands : dict or None
        Example:
        {"UVW2": 1928.0, "U": 3465.0, ...}

    disk_kwargs : dict or None
        Extra keyword arguments passed to DiskConfig.

    outname : str
        Output figure name.

    xlim : tuple or None
        Example: (0, 30). If None, use full tau range.
    """
    import matplotlib.pyplot as plt

    if bands is None:
        bands = {
            "UVW2": 1928.0,
            "UVM2": 2246.0,
            "UVW1": 2600.0,
            "U": 3465.0,
            "B": 4392.0,
            "V": 5468.0,
        }

    if disk_kwargs is None:
        disk_kwargs = {}

    plt.figure(figsize=(8, 5))

    for name, lam in bands.items():
        disk = DiskConfig(lam_A=lam, **disk_kwargs)
        tau, psi = disk_transfer_function(disk)
        peak, centroid, width = tf_moments(tau, psi)

        label = (
            f"{name} {lam:.0f} Å "
            f"(peak={peak:.2f} d, cent={centroid:.2f} d)"
        )

        plt.plot(tau, psi, lw=1.8, label=label)

    plt.xlabel("Time lag (day)")
    plt.ylabel(r"$\Psi(\tau)$")
    plt.title("Theoretical disk transfer functions")
    if xlim is not None:
        plt.xlim(*xlim)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outname, dpi=220)
    plt.close()

if __name__ == "__main__":
    bands = {
        "UVW2": 1928.0,
        "UVM2": 2246.0,
        "UVW1": 2600.0,
        "U": 3465.0,
        "B": 4392.0,
        "V": 5468.0,
    }

    print("band  lambda_A  peak_d  centroid_d  width_d")

    for name, lam in bands.items():
        disk = DiskConfig(lam_A=lam)
        tau, psi = disk_transfer_function(disk)
        peak, centroid, width = tf_moments(tau, psi)

        print(
            f"{name:5s} {lam:8.1f} "
            f"{peak:8.3f} {centroid:10.3f} {width:8.3f}"
        )
    
    plot_disk_transfer_functions(
        outname="disk_transfer_functions_linear.png",
        xlim=(0, 30),
    )