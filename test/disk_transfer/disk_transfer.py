"""Pure forward model for a Jaiswal-style irradiated cold accretion disk.

The module is deliberately free of plotting, file I/O, and optimizer code.
It is therefore suitable for repeated calls from a future likelihood function.

Main interface
--------------
model = DiskTransferModel(...)
prediction = model.predict(
    m_bh_msun=...,
    mdot_g_s=...,
    lx_erg_s=...,
    h_rg=...,
    inclination_deg=...,
    rin_rg=...,
)

The returned transfer-function arrays always have shape
(n_wavelength, n_time), including the single-wavelength case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# CGS constants
G = 6.67430e-8
C = 2.99792458e10
H_PLANCK = 6.62607015e-27
K_BOLTZMANN = 1.380649e-16
SIGMA_SB = 5.670374419e-5
M_SUN = 1.98847e33
DAY = 86400.0


@dataclass(frozen=True)
class DiskParameters:
    """Physical and geometric parameters of one model evaluation."""

    m_bh_msun: float
    mdot_g_s: float
    lx_erg_s: float
    h_rg: float
    inclination_deg: float
    rin_rg: float


def gravitational_radius_cm(m_bh_msun: float) -> float:
    """Return r_g = GM/c^2 in cm."""
    return G * m_bh_msun * M_SUN / C**2


def mdot_from_eddington_ratio(
    m_bh_msun: float,
    eddington_ratio: float,
    radiative_efficiency: float = 0.1,
) -> float:
    """Convert Eddington ratio into mass accretion rate in g s^-1."""
    l_edd = 1.26e38 * m_bh_msun
    return eddington_ratio * l_edd / (
        radiative_efficiency * C**2
    )


def planck_lambda(
    temperature_k: np.ndarray,
    wavelength_angstrom: float,
) -> np.ndarray:
    """Planck B_lambda in erg s^-1 cm^-2 sr^-1 Angstrom^-1."""
    wavelength_cm = wavelength_angstrom * 1.0e-8
    x = H_PLANCK * C / (
        wavelength_cm * K_BOLTZMANN * temperature_k
    )

    prefactor = (
        2.0 * H_PLANCK * C**2 / wavelength_cm**5
    )
    return prefactor / np.expm1(x) * 1.0e-8


def _normalize_density(
    time_days: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    """Normalize one time-density array to unit integral."""
    integral = np.trapezoid(density, time_days)
    return density / integral


def _quantile_from_density(
    time_days: np.ndarray,
    normalized_density: np.ndarray,
    quantile: float,
) -> float:
    """Return a response-time quantile."""
    dt_days = time_days[1] - time_days[0]
    cumulative = np.cumsum(normalized_density) * dt_days
    cumulative /= cumulative[-1]
    return float(
        np.interp(quantile, cumulative, time_days)
    )


def _density_statistics(
    time_days: np.ndarray,
    normalized_density: np.ndarray,
) -> dict[str, float]:
    """Return standard lag diagnostics."""
    mean_lag = float(
        np.trapezoid(
            time_days * normalized_density,
            time_days,
        )
    )

    variance = float(
        np.trapezoid(
            (time_days - mean_lag) ** 2
            * normalized_density,
            time_days,
        )
    )

    return {
        "peak_lag_days": float(
            time_days[np.argmax(normalized_density)]
        ),
        "mean_lag_days": mean_lag,
        "std_lag_days": float(np.sqrt(variance)),
        "q05_lag_days": _quantile_from_density(
            time_days,
            normalized_density,
            0.05,
        ),
        "q16_lag_days": _quantile_from_density(
            time_days,
            normalized_density,
            0.16,
        ),
        "median_lag_days": _quantile_from_density(
            time_days,
            normalized_density,
            0.50,
        ),
        "q84_lag_days": _quantile_from_density(
            time_days,
            normalized_density,
            0.84,
        ),
        "q95_lag_days": _quantile_from_density(
            time_days,
            normalized_density,
            0.95,
        ),
        "q999_lag_days": _quantile_from_density(
            time_days,
            normalized_density,
            0.999,
        ),
    }


class DiskTransferModel:
    """Fixed-grid forward model for single- or multiwavelength responses.

    The radial, azimuthal, wavelength, and time grids are fixed at
    initialization. Model evaluations therefore always return arrays with
    identical shapes, which is important for likelihood optimization.
    """

    def __init__(
        self,
        *,
        wavelengths_angstrom: np.ndarray | list[float],
        time_days: np.ndarray,
        global_rmin_rg: float = 6.0,
        rout_rg: float = 1.0e4,
        nr: int = 900,
        nphi: int = 720,
        pulse_width_days: float = 0.05,
    ) -> None:
        self.wavelengths_angstrom = np.asarray(
            wavelengths_angstrom,
            dtype=float,
        )
        self.time_days = np.asarray(time_days, dtype=float)

        if self.wavelengths_angstrom.ndim != 1:
            raise ValueError(
                "wavelengths_angstrom must be one-dimensional."
            )
        if self.time_days.ndim != 1:
            raise ValueError(
                "time_days must be one-dimensional."
            )

        time_steps = np.diff(self.time_days)
        if not np.allclose(
            time_steps,
            time_steps[0],
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "time_days must be uniformly spaced."
            )

        self.dt_days = float(time_steps[0])
        self.global_rmin_rg = float(global_rmin_rg)
        self.rout_rg = float(rout_rg)
        self.nr = int(nr)
        self.nphi = int(nphi)
        self.pulse_width_days = float(
            pulse_width_days
        )

        self.r_edges_rg = np.geomspace(
            self.global_rmin_rg,
            self.rout_rg,
            self.nr + 1,
        )
        self.r_inner_rg = self.r_edges_rg[:-1]
        self.r_outer_rg = self.r_edges_rg[1:]
        self.r_center_rg = np.sqrt(
            self.r_inner_rg * self.r_outer_rg
        )

        self.dphi = 2.0 * np.pi / self.nphi
        self.phi_rad = (
            np.arange(self.nphi, dtype=float) + 0.5
        ) * self.dphi
        self.cos_phi = np.cos(self.phi_rad)

        self.pulse_bins = max(
            1,
            int(
                round(
                    self.pulse_width_days
                    / self.dt_days
                )
            ),
        )
        self.pulse_profile = (
            np.ones(self.pulse_bins, dtype=float)
            / self.pulse_bins
        )
        self.pulse_centroid_days = (
            0.5
            * (self.pulse_bins - 1)
            * self.dt_days
        )

    def _active_annuli(
        self,
        rin_rg: float,
    ) -> dict[str, np.ndarray]:
        """Return exact active annulus areas on the fixed radial grid."""
        active_inner_rg = np.maximum(
            self.r_inner_rg,
            rin_rg,
        )
        active = active_inner_rg < self.r_outer_rg

        area_factor_rg2 = np.zeros(self.nr)
        area_factor_rg2[active] = 0.5 * (
            self.r_outer_rg[active] ** 2
            - active_inner_rg[active] ** 2
        )

        effective_radius_rg = self.r_center_rg.copy()

        a = active_inner_rg[active]
        b = self.r_outer_rg[active]
        effective_radius_rg[active] = (
            (2.0 / 3.0)
            * (b**3 - a**3)
            / (b**2 - a**2)
        )

        dlnr = np.zeros(self.nr)
        dlnr[active] = np.log(b / a)

        return {
            "active": active,
            "active_inner_rg": active_inner_rg,
            "area_factor_rg2": area_factor_rg2,
            "effective_radius_rg": (
                effective_radius_rg
            ),
            "dlnr": dlnr,
        }

    def _thermal_structure(
        self,
        parameters: DiskParameters,
        annuli: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Compute viscous, irradiation, and total disk heating."""
        rg_cm = gravitational_radius_cm(
            parameters.m_bh_msun
        )
        radius_cm = (
            annuli["effective_radius_rg"] * rg_cm
        )
        rin_cm = parameters.rin_rg * rg_cm
        height_cm = parameters.h_rg * rg_cm
        mass_g = parameters.m_bh_msun * M_SUN

        boundary_factor = np.maximum(
            0.0,
            1.0 - np.sqrt(rin_cm / radius_cm),
        )

        viscous_flux = (
            3.0
            * G
            * mass_g
            * parameters.mdot_g_s
            / (8.0 * np.pi * radius_cm**3)
            * boundary_factor
        )

        irradiation_flux = (
            parameters.lx_erg_s
            * height_cm
            / (4.0 * np.pi * radius_cm**3)
        )

        active = annuli["active"]
        viscous_flux = np.where(
            active,
            viscous_flux,
            0.0,
        )
        irradiation_flux = np.where(
            active,
            irradiation_flux,
            0.0,
        )

        total_flux = viscous_flux + irradiation_flux

        total_temperature_k = np.zeros(self.nr)
        viscous_temperature_k = np.zeros(self.nr)

        total_temperature_k[active] = (
            total_flux[active] / SIGMA_SB
        ) ** 0.25
        viscous_temperature_k[active] = (
            viscous_flux[active] / SIGMA_SB
        ) ** 0.25

        return {
            "radius_cm": radius_cm,
            "boundary_factor": boundary_factor,
            "viscous_flux": viscous_flux,
            "irradiation_flux": irradiation_flux,
            "total_flux": total_flux,
            "total_temperature_k": (
                total_temperature_k
            ),
            "viscous_temperature_k": (
                viscous_temperature_k
            ),
        }

    def _delay_surface_days(
        self,
        parameters: DiskParameters,
        radius_cm: np.ndarray,
    ) -> np.ndarray:
        """Return the two-dimensional disk delay surface."""
        rg_cm = gravitational_radius_cm(
            parameters.m_bh_msun
        )
        height_cm = parameters.h_rg * rg_cm
        inclination = np.deg2rad(
            parameters.inclination_deg
        )

        radius_2d = radius_cm[:, None]

        path_cm = (
            np.sqrt(radius_2d**2 + height_cm**2)
            + height_cm * np.cos(inclination)
            - radius_2d
            * np.sin(inclination)
            * self.cos_phi[None, :]
        )

        return path_cm / C / DAY

    def _delay_envelopes_days(
        self,
        parameters: DiskParameters,
        radius_cm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return near- and far-side ring-delay envelopes."""
        rg_cm = gravitational_radius_cm(
            parameters.m_bh_msun
        )
        height_cm = parameters.h_rg * rg_cm
        inclination = np.deg2rad(
            parameters.inclination_deg
        )

        common = (
            np.sqrt(radius_cm**2 + height_cm**2)
            + height_cm * np.cos(inclination)
        )
        spread = radius_cm * np.sin(inclination)

        near = (common - spread) / C / DAY
        far = (common + spread) / C / DAY
        return near, far

    def _deposit_to_time_grid(
        self,
        delays_days: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Cloud-in-cell deposition onto the fixed time grid."""
        t0 = self.time_days[0]
        delay_index = (
            delays_days.ravel() - t0
        ) / self.dt_days

        left_index = np.floor(
            delay_index
        ).astype(np.int64)
        fraction = delay_index - left_index
        flat_weights = weights.ravel()

        valid = (
            (left_index >= 0)
            & (
                left_index
                < self.time_days.size - 1
            )
        )

        left = left_index[valid]
        frac = fraction[valid]
        weight = flat_weights[valid]

        deposited = (
            np.bincount(
                left,
                weights=weight * (1.0 - frac),
                minlength=self.time_days.size,
            )
            + np.bincount(
                left + 1,
                weights=weight * frac,
                minlength=self.time_days.size,
            )
        )[: self.time_days.size]

        captured_fraction = float(
            deposited.sum() / flat_weights.sum()
        )
        return deposited / self.dt_days, captured_fraction

    def _apply_pulse(
        self,
        instantaneous_density: np.ndarray,
    ) -> np.ndarray:
        """Apply the configured causal rectangular pulse."""
        pulsed = np.convolve(
            instantaneous_density,
            self.pulse_profile,
            mode="full",
        )
        return pulsed[: self.time_days.size]

    def predict(
        self,
        *,
        m_bh_msun: float,
        mdot_g_s: float,
        lx_erg_s: float,
        h_rg: float,
        inclination_deg: float,
        rin_rg: float,
        reference_index: int = 0,
    ) -> dict:
        """Evaluate the model for one physical parameter set."""
        parameters = DiskParameters(
            m_bh_msun=float(m_bh_msun),
            mdot_g_s=float(mdot_g_s),
            lx_erg_s=float(lx_erg_s),
            h_rg=float(h_rg),
            inclination_deg=float(inclination_deg),
            rin_rg=float(rin_rg),
        )

        annuli = self._active_annuli(
            parameters.rin_rg
        )
        thermal = self._thermal_structure(
            parameters,
            annuli,
        )

        delay_days = self._delay_surface_days(
            parameters,
            thermal["radius_cm"],
        )
        near_delay_days, far_delay_days = (
            self._delay_envelopes_days(
                parameters,
                thermal["radius_cm"],
            )
        )

        rg_cm = gravitational_radius_cm(
            parameters.m_bh_msun
        )

        cell_area_cm2 = (
            annuli["area_factor_rg2"][:, None]
            * rg_cm**2
            * self.dphi
        )

        ring_area_cm2 = (
            2.0
            * np.pi
            * annuli["area_factor_rg2"]
            * rg_cm**2
        )

        n_wave = self.wavelengths_angstrom.size
        n_time = self.time_days.size

        instant_response = np.zeros(
            (n_wave, n_time)
        )
        pulsed_response = np.zeros(
            (n_wave, n_time)
        )
        radial_response_per_dlnr = np.zeros(
            (n_wave, self.nr)
        )
        radial_response_fraction = np.zeros(
            (n_wave, self.nr)
        )

        captured_fraction = np.zeros(n_wave)
        instant_stats = []
        pulsed_stats = []
        peak_response_radius_rg = np.zeros(n_wave)
        mean_response_radius_rg = np.zeros(n_wave)

        active = annuli["active"]

        for index, wavelength in enumerate(
            self.wavelengths_angstrom
        ):
            b_lambda = np.zeros(self.nr)
            b_lambda[active] = planck_lambda(
                thermal["total_temperature_k"][
                    active
                ],
                wavelength,
            )

            cell_weights = np.broadcast_to(
                b_lambda[:, None] * cell_area_cm2,
                delay_days.shape,
            )

            instantaneous_raw, captured = (
                self._deposit_to_time_grid(
                    delay_days,
                    cell_weights,
                )
            )
            pulsed_raw = self._apply_pulse(
                instantaneous_raw
            )

            instant_response[index] = (
                _normalize_density(
                    self.time_days,
                    instantaneous_raw,
                )
            )
            pulsed_response[index] = (
                _normalize_density(
                    self.time_days,
                    pulsed_raw,
                )
            )

            captured_fraction[index] = captured

            instant_stats.append(
                _density_statistics(
                    self.time_days,
                    instant_response[index],
                )
            )
            pulsed_stats.append(
                _density_statistics(
                    self.time_days,
                    pulsed_response[index],
                )
            )

            ring_weight = b_lambda * ring_area_cm2
            radial_response_fraction[index] = (
                ring_weight / ring_weight.sum()
            )

            per_dlnr = np.zeros(self.nr)
            per_dlnr[active] = (
                ring_weight[active]
                / annuli["dlnr"][active]
            )

            normalization = np.trapezoid(
                per_dlnr[active],
                np.log(
                    annuli[
                        "effective_radius_rg"
                    ][active]
                ),
            )
            radial_response_per_dlnr[index] = (
                per_dlnr / normalization
            )

            peak_response_radius_rg[index] = (
                annuli["effective_radius_rg"][
                    np.argmax(per_dlnr)
                ]
            )
            mean_response_radius_rg[index] = (
                np.sum(
                    annuli[
                        "effective_radius_rg"
                    ]
                    * radial_response_fraction[
                        index
                    ]
                )
            )

        instant_mean_lag_days = np.array(
            [
                stats["mean_lag_days"]
                for stats in instant_stats
            ]
        )
        pulsed_mean_lag_days = np.array(
            [
                stats["mean_lag_days"]
                for stats in pulsed_stats
            ]
        )
        pulsed_peak_lag_days = np.array(
            [
                stats["peak_lag_days"]
                for stats in pulsed_stats
            ]
        )
        pulsed_median_lag_days = np.array(
            [
                stats["median_lag_days"]
                for stats in pulsed_stats
            ]
        )
        pulsed_q16_lag_days = np.array(
            [
                stats["q16_lag_days"]
                for stats in pulsed_stats
            ]
        )
        pulsed_q84_lag_days = np.array(
            [
                stats["q84_lag_days"]
                for stats in pulsed_stats
            ]
        )
        pulsed_q999_lag_days = np.array(
            [
                stats["q999_lag_days"]
                for stats in pulsed_stats
            ]
        )

        relative_mean_lag_days = (
            pulsed_mean_lag_days
            - pulsed_mean_lag_days[
                reference_index
            ]
        )

        stationary_radius_rg = float(
            parameters.h_rg
            * np.tan(
                np.deg2rad(
                    parameters.inclination_deg
                )
            )
        )

        active_delay = delay_days[active]
        area_numerical_rg2 = float(
            np.sum(
                2.0
                * np.pi
                * annuli["area_factor_rg2"]
            )
        )
        area_analytic_rg2 = float(
            np.pi
            * (
                self.rout_rg**2
                - parameters.rin_rg**2
            )
        )

        diagnostics = {
            "rg_cm": float(rg_cm),
            "stationary_radius_rg": (
                stationary_radius_rg
            ),
            "stationary_point_in_disk": bool(
                parameters.rin_rg
                <= stationary_radius_rg
                <= self.rout_rg
            ),
            "minimum_delay_days": float(
                active_delay.min()
            ),
            "maximum_delay_days": float(
                active_delay.max()
            ),
            "pulse_centroid_days": float(
                self.pulse_centroid_days
            ),
            "captured_fraction_min": float(
                captured_fraction.min()
            ),
            "captured_fraction_by_wavelength": (
                captured_fraction
            ),
            "disk_area_relative_error": float(
                area_numerical_rg2
                / area_analytic_rg2
                - 1.0
            ),
        }

        return {
            "parameters": parameters,
            "wavelengths_angstrom": (
                self.wavelengths_angstrom.copy()
            ),
            "time_days": self.time_days.copy(),
            "reference_index": int(reference_index),
            "reference_wavelength_angstrom": float(
                self.wavelengths_angstrom[
                    reference_index
                ]
            ),
            "radius_rg": annuli[
                "effective_radius_rg"
            ].copy(),
            "active_radial_cells": active.copy(),
            "near_delay_days": (
                near_delay_days.copy()
            ),
            "far_delay_days": (
                far_delay_days.copy()
            ),
            "viscous_flux": (
                thermal["viscous_flux"].copy()
            ),
            "irradiation_flux": (
                thermal[
                    "irradiation_flux"
                ].copy()
            ),
            "total_temperature_k": (
                thermal[
                    "total_temperature_k"
                ].copy()
            ),
            "viscous_temperature_k": (
                thermal[
                    "viscous_temperature_k"
                ].copy()
            ),
            "instant_response": instant_response,
            "pulsed_response": pulsed_response,
            "radial_response_per_dlnr": (
                radial_response_per_dlnr
            ),
            "radial_response_fraction": (
                radial_response_fraction
            ),
            "instant_stats": instant_stats,
            "pulsed_stats": pulsed_stats,
            "instant_mean_lag_days": (
                instant_mean_lag_days
            ),
            "mean_lag_days": (
                pulsed_mean_lag_days
            ),
            "peak_lag_days": (
                pulsed_peak_lag_days
            ),
            "median_lag_days": (
                pulsed_median_lag_days
            ),
            "q16_lag_days": (
                pulsed_q16_lag_days
            ),
            "q84_lag_days": (
                pulsed_q84_lag_days
            ),
            "q999_lag_days": (
                pulsed_q999_lag_days
            ),
            "relative_mean_lag_days": (
                relative_mean_lag_days
            ),
            "peak_response_radius_rg": (
                peak_response_radius_rg
            ),
            "mean_response_radius_rg": (
                mean_response_radius_rg
            ),
            "diagnostics": diagnostics,
        }
