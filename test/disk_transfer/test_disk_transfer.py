"""Executable scientific test and diagnostic script.

Edit the parameter block below, then run:

    python test_disk_transfer.py

The script performs numerical/physical checks, prints a lag table, and creates
research-style figures plus CSV/JSON diagnostics in ``test_outputs``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np

from disk_transfer import (
    DiskTransferModel,
    mdot_from_eddington_ratio,
)


# ---------------------------------------------------------------------------
# User-editable parameter block
# ---------------------------------------------------------------------------

M_BH_MSUN = 5.0e7
EDDINGTON_RATIO = 0.015
RADIATIVE_EFFICIENCY = 0.1
L_X_ERG_S = 9.68e43

H_RG = 48.29
INCLINATION_DEG = 40.0
RIN_RG = 94.87
ROUT_RG = 1.0e4

WAVELENGTHS_ANGSTROM = np.array([
    1230.0,
    1548.0,
    1949.0,
    2454.0,
    3090.0,
    3890.0,
    4897.0,
    6165.0,
    7762.0,
])

REFERENCE_INDEX = 0

TIME_DAYS = np.arange(
    0.0,
    50.0 + 0.002,
    0.002,
)

NR = 900
NPHI = 720
PULSE_WIDTH_DAYS = 0.05


# ---------------------------------------------------------------------------
# Scientific checks
# ---------------------------------------------------------------------------

def run_scientific_checks(
    model: DiskTransferModel,
    result: dict,
) -> list[str]:
    """Run interface, mathematical, physical, and convergence checks."""
    messages: list[str] = []

    n_wave = WAVELENGTHS_ANGSTROM.size
    n_time = TIME_DAYS.size

    assert result["instant_response"].shape == (
        n_wave,
        n_time,
    )
    assert result["pulsed_response"].shape == (
        n_wave,
        n_time,
    )
    messages.append("PASS  fixed output shapes")

    assert np.all(
        np.isfinite(result["instant_response"])
    )
    assert np.all(
        np.isfinite(result["pulsed_response"])
    )
    assert np.all(result["instant_response"] >= 0.0)
    assert np.all(result["pulsed_response"] >= 0.0)
    messages.append(
        "PASS  finite and non-negative responses"
    )

    instant_integrals = np.trapezoid(
        result["instant_response"],
        TIME_DAYS,
        axis=1,
    )
    pulsed_integrals = np.trapezoid(
        result["pulsed_response"],
        TIME_DAYS,
        axis=1,
    )

    assert np.allclose(
        instant_integrals,
        1.0,
        atol=1.0e-9,
    )
    assert np.allclose(
        pulsed_integrals,
        1.0,
        atol=1.0e-9,
    )
    messages.append("PASS  unit-area normalization")

    mean_direct = np.trapezoid(
        result["pulsed_response"]
        * TIME_DAYS[None, :],
        TIME_DAYS,
        axis=1,
    )
    assert np.allclose(
        mean_direct,
        result["mean_lag_days"],
        atol=1.0e-10,
    )
    messages.append("PASS  mean-lag consistency")

    assert np.all(
        result["near_delay_days"][
            result["active_radial_cells"]
        ]
        <= result["far_delay_days"][
            result["active_radial_cells"]
        ]
    )
    messages.append(
        "PASS  near-side delays precede far-side delays"
    )

    assert np.all(
        np.diff(result["mean_lag_days"])
        >= -1.0e-10
    )
    assert np.all(
        np.diff(
            result["mean_response_radius_rg"]
        )
        >= -1.0e-10
    )
    messages.append(
        "PASS  wavelength-lag and wavelength-radius trends"
    )

    assert np.isclose(
        result["relative_mean_lag_days"][
            REFERENCE_INDEX
        ],
        0.0,
        atol=1.0e-14,
    )
    messages.append(
        "PASS  zero lag at the reference wavelength"
    )

    pulse_shift = (
        result["mean_lag_days"]
        - result["instant_mean_lag_days"]
    )
    assert np.allclose(
        pulse_shift,
        model.pulse_centroid_days,
        atol=2.0e-6,
    )
    messages.append(
        "PASS  finite-pulse centroid shift"
    )

    assert (
        result["diagnostics"][
            "captured_fraction_min"
        ]
        > 0.9999
    )
    messages.append(
        "PASS  fixed time window captures >99.99% response"
    )

    expected_stationary = (
        RIN_RG
        <= H_RG
        * np.tan(
            np.deg2rad(INCLINATION_DEG)
        )
        <= ROUT_RG
    )
    assert (
        result["diagnostics"][
            "stationary_point_in_disk"
        ]
        == expected_stationary
    )
    messages.append(
        "PASS  geometric stationary-point classification"
    )

    assert abs(
        result["diagnostics"][
            "disk_area_relative_error"
        ]
    ) < 1.0e-12
    messages.append(
        "PASS  exact active-disk area accounting"
    )

    coarse_model = DiskTransferModel(
        wavelengths_angstrom=(
            WAVELENGTHS_ANGSTROM
        ),
        time_days=TIME_DAYS,
        global_rmin_rg=6.0,
        rout_rg=ROUT_RG,
        nr=NR,
        nphi=NPHI // 2,
        pulse_width_days=PULSE_WIDTH_DAYS,
    )
    coarse_result = coarse_model.predict(
        m_bh_msun=M_BH_MSUN,
        mdot_g_s=mdot_from_eddington_ratio(
            M_BH_MSUN,
            EDDINGTON_RATIO,
            RADIATIVE_EFFICIENCY,
        ),
        lx_erg_s=L_X_ERG_S,
        h_rg=H_RG,
        inclination_deg=INCLINATION_DEG,
        rin_rg=RIN_RG,
        reference_index=REFERENCE_INDEX,
    )

    convergence_error = np.max(
        np.abs(
            result["mean_lag_days"]
            - coarse_result["mean_lag_days"]
        )
        / result["mean_lag_days"]
    )
    assert convergence_error < 2.0e-3
    messages.append(
        "PASS  azimuthal convergence "
        f"(max relative mean-lag change "
        f"{convergence_error:.2e})"
    )

    perturbed_result = model.predict(
        m_bh_msun=M_BH_MSUN,
        mdot_g_s=mdot_from_eddington_ratio(
            M_BH_MSUN,
            EDDINGTON_RATIO,
            RADIATIVE_EFFICIENCY,
        ),
        lx_erg_s=L_X_ERG_S,
        h_rg=H_RG,
        inclination_deg=INCLINATION_DEG,
        rin_rg=RIN_RG * 1.001,
        reference_index=REFERENCE_INDEX,
    )

    continuity_error = (
        np.linalg.norm(
            perturbed_result["pulsed_response"]
            - result["pulsed_response"]
        )
        / np.linalg.norm(
            result["pulsed_response"]
        )
    )
    assert continuity_error < 0.02
    messages.append(
        "PASS  inner-radius continuity "
        f"(relative response change "
        f"{continuity_error:.2e})"
    )

    return messages


# ---------------------------------------------------------------------------
# Output tables and plots
# ---------------------------------------------------------------------------

def save_tables(
    result: dict,
    output_dir: Path,
) -> None:
    """Save lag-spectrum and diagnostic tables."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        output_dir / "lag_spectrum.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "wavelength_angstrom",
            "peak_lag_day",
            "mean_lag_day",
            "median_lag_day",
            "q16_lag_day",
            "q84_lag_day",
            "relative_mean_lag_day",
            "peak_response_radius_rg",
            "mean_response_radius_rg",
        ])

        for index, wavelength in enumerate(
            result["wavelengths_angstrom"]
        ):
            writer.writerow([
                wavelength,
                result["peak_lag_days"][index],
                result["mean_lag_days"][index],
                result["median_lag_days"][index],
                result["q16_lag_days"][index],
                result["q84_lag_days"][index],
                result[
                    "relative_mean_lag_days"
                ][index],
                result[
                    "peak_response_radius_rg"
                ][index],
                result[
                    "mean_response_radius_rg"
                ][index],
            ])

    parameters = result["parameters"]
    diagnostics = {
        "parameters": {
            "m_bh_msun": parameters.m_bh_msun,
            "mdot_g_s": parameters.mdot_g_s,
            "lx_erg_s": parameters.lx_erg_s,
            "h_rg": parameters.h_rg,
            "inclination_deg": (
                parameters.inclination_deg
            ),
            "rin_rg": parameters.rin_rg,
            "rout_rg": ROUT_RG,
        },
        "numerical_grid": {
            "nr": NR,
            "nphi": NPHI,
            "dt_days": float(
                TIME_DAYS[1] - TIME_DAYS[0]
            ),
            "time_max_days": float(
                TIME_DAYS[-1]
            ),
            "pulse_width_days": (
                PULSE_WIDTH_DAYS
            ),
        },
        "diagnostics": {
            key: (
                value.tolist()
                if isinstance(value, np.ndarray)
                else value
            )
            for key, value in result[
                "diagnostics"
            ].items()
        },
    }

    (
        output_dir / "diagnostics.json"
    ).write_text(
        json.dumps(
            diagnostics,
            indent=2,
        ),
        encoding="utf-8",
    )


def scientific_style() -> None:
    """Apply a restrained publication-style Matplotlib configuration."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12.0,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.3,
        "ytick.labelsize": 9.3,
        "legend.fontsize": 8.6,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 220,
    })


def plot_scientific_overview(
    result: dict,
    output_dir: Path,
) -> None:
    """Create a four-panel physical and numerical diagnostic figure."""
    scientific_style()

    palette = {
        "navy": "#173F5F",
        "blue": "#3E78A8",
        "teal": "#269E8A",
        "orange": "#D97722",
        "red": "#B33A3A",
        "purple": "#725A9A",
        "gray": "#65727A",
        "grid": "#D8E0E5",
        "ink": "#172126",
    }

    wavelengths = result[
        "wavelengths_angstrom"
    ]
    time_days = result["time_days"]
    active = result["active_radial_cells"]
    radius = result["radius_rg"][active]

    cmap = plt.get_cmap("viridis")
    wavelength_norm = Normalize(
        wavelengths.min(),
        wavelengths.max(),
    )

    figure = plt.figure(
        figsize=(13.6, 9.2)
    )
    grid = figure.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.95,
        bottom=0.08,
        top=0.90,
        wspace=0.27,
        hspace=0.30,
    )

    ax_temperature = figure.add_subplot(
        grid[0, 0]
    )
    ax_radius = figure.add_subplot(
        grid[0, 1]
    )
    ax_response = figure.add_subplot(
        grid[1, 0]
    )
    ax_lag = figure.add_subplot(
        grid[1, 1]
    )

    ax_temperature.loglog(
        radius,
        result["total_temperature_k"][
            active
        ],
        linewidth=2.2,
        color=palette["navy"],
        label=r"$T_{\rm eff}$",
    )
    ax_temperature.loglog(
        radius,
        result["viscous_temperature_k"][
            active
        ],
        linewidth=1.6,
        linestyle="--",
        color=palette["orange"],
        label=r"$T_{\rm visc}$",
    )
    ax_temperature.axvline(
        RIN_RG,
        linewidth=1.2,
        linestyle=":",
        color=palette["red"],
        label=r"$R_{\rm in}$",
    )
    ax_temperature.set_xlabel(
        r"Radius $R/r_g$"
    )
    ax_temperature.set_ylabel(
        "Temperature [K]"
    )
    ax_temperature.set_title(
        "A. Thermal structure"
    )
    ax_temperature.grid(
        which="major",
        color=palette["grid"],
        linewidth=0.55,
        alpha=0.70,
    )
    ax_temperature.legend(
        frameon=False,
        loc="lower left",
    )

    selected_indices = np.unique(
        np.linspace(
            0,
            wavelengths.size - 1,
            min(5, wavelengths.size),
        ).astype(int)
    )

    for index in selected_indices:
        ax_radius.semilogx(
            radius,
            result[
                "radial_response_per_dlnr"
            ][index, active],
            linewidth=1.9,
            color=cmap(
                wavelength_norm(
                    wavelengths[index]
                )
            ),
            label=(
                rf"{wavelengths[index]:.0f} "
                r"$\AA$"
            ),
        )

    ax_radius.set_xlabel(
        r"Radius $R/r_g$"
    )
    ax_radius.set_ylabel(
        r"Normalized response per $d\ln R$"
    )
    ax_radius.set_title(
        "B. Wavelength-dependent emitting radii"
    )
    ax_radius.grid(
        which="major",
        color=palette["grid"],
        linewidth=0.55,
        alpha=0.70,
    )
    ax_radius.legend(
        frameon=False,
        ncol=2,
    )

    response_limit = (
        result["q999_lag_days"].max()
        * 1.12
    )
    time_mask = (
        (time_days > 0.0)
        & (time_days <= response_limit)
    )

    for index, wavelength in enumerate(
        wavelengths
    ):
        ax_response.plot(
            time_days[time_mask],
            result["pulsed_response"][
                index,
                time_mask,
            ],
            linewidth=1.6,
            color=cmap(
                wavelength_norm(wavelength)
            ),
        )

    ax_response.set_xlabel(
        "Delay [day]"
    )
    ax_response.set_ylabel(
        "Unit-area response [day$^{-1}$]"
    )
    ax_response.set_title(
        "C. Multiwavelength transfer functions"
    )
    ax_response.grid(
        which="major",
        color=palette["grid"],
        linewidth=0.55,
        alpha=0.70,
    )

    scalar_mappable = plt.cm.ScalarMappable(
        norm=wavelength_norm,
        cmap=cmap,
    )
    colorbar = figure.colorbar(
        scalar_mappable,
        ax=ax_response,
        pad=0.018,
        fraction=0.050,
    )
    colorbar.set_label(
        r"Wavelength [$\AA$]"
    )

    ax_lag.fill_between(
        wavelengths,
        result["q16_lag_days"],
        result["q84_lag_days"],
        color=palette["blue"],
        alpha=0.15,
        label="central 68%",
    )
    ax_lag.plot(
        wavelengths,
        result["mean_lag_days"],
        marker="o",
        markersize=4.8,
        linewidth=1.9,
        color=palette["navy"],
        label="mean lag",
    )
    ax_lag.plot(
        wavelengths,
        result["median_lag_days"],
        marker="s",
        markersize=4.2,
        linewidth=1.4,
        linestyle="--",
        color=palette["orange"],
        label="median lag",
    )
    ax_lag.set_xscale("log")
    ax_lag.set_xlabel(
        r"Wavelength [$\AA$]"
    )
    ax_lag.set_ylabel(
        "Absolute lag [day]"
    )
    ax_lag.set_title(
        "D. Lag spectrum"
    )
    ax_lag.grid(
        which="major",
        color=palette["grid"],
        linewidth=0.55,
        alpha=0.70,
    )
    ax_lag.legend(
        frameon=False,
        loc="upper left",
    )

    diagnostic_text = (
        rf"$\tau_{{\min}}$ = "
        f"{result['diagnostics']['minimum_delay_days']:.3f} d\n"
        rf"$R_*=h\tan i$ = "
        f"{result['diagnostics']['stationary_radius_rg']:.1f} "
        r"$r_g$"
        "\n"
        rf"capture = "
        f"{100.0 * result['diagnostics']['captured_fraction_min']:.4f}%\n"
        rf"$N_R\times N_\phi$ = "
        f"{NR} × {NPHI}"
    )
    ax_lag.text(
        0.97,
        0.05,
        diagnostic_text,
        transform=ax_lag.transAxes,
        ha="right",
        va="bottom",
        color=palette["ink"],
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": palette["grid"],
            "alpha": 0.94,
        },
    )

    figure.suptitle(
        (
            "Jaiswal-style cold-disk transfer model  |  "
            rf"$M_\bullet={M_BH_MSUN:.2e}\,M_\odot$, "
            rf"$h={H_RG:.2f}\,r_g$, "
            rf"$R_{{\rm in}}={RIN_RG:.2f}\,r_g$, "
            rf"$i={INCLINATION_DEG:.1f}^\circ$"
        ),
        fontsize=14,
        color=palette["ink"],
    )

    figure.savefig(
        output_dir
        / "scientific_overview.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_wavelength_delay_map(
    result: dict,
    output_dir: Path,
) -> None:
    """Create a normalized wavelength-delay response map."""
    scientific_style()

    time_days = result["time_days"]
    wavelengths = result[
        "wavelengths_angstrom"
    ]
    response_limit = (
        result["q999_lag_days"].max()
        * 1.12
    )
    time_mask = time_days <= response_limit
    response_map = result[
        "pulsed_response"
    ][:, time_mask]

    positive = response_map[
        response_map > 0.0
    ]
    vmax = positive.max()
    vmin = vmax * 1.0e-5

    figure, axis = plt.subplots(
        figsize=(8.6, 5.8)
    )

    mesh = axis.pcolormesh(
        time_days[time_mask],
        wavelengths,
        response_map,
        shading="auto",
        cmap="magma",
        norm=LogNorm(
            vmin=vmin,
            vmax=vmax,
        ),
    )

    axis.plot(
        result["mean_lag_days"],
        wavelengths,
        color="white",
        linewidth=1.8,
        label=r"$\langle\tau\rangle$",
    )
    axis.plot(
        result["median_lag_days"],
        wavelengths,
        color="#73D2DE",
        linewidth=1.4,
        linestyle="--",
        label=r"$\tau_{50}$",
    )
    axis.plot(
        result["q16_lag_days"],
        wavelengths,
        color="#F4D35E",
        linewidth=1.0,
        linestyle=":",
        label="16–84% envelope",
    )
    axis.plot(
        result["q84_lag_days"],
        wavelengths,
        color="#F4D35E",
        linewidth=1.0,
        linestyle=":",
    )

    axis.set_xlabel("Delay [day]")
    axis.set_ylabel(
        r"Wavelength [$\AA$]"
    )
    axis.set_title(
        "Normalized wavelength-delay response"
    )
    axis.legend(
        frameon=False,
        loc="lower right",
        labelcolor="white",
    )

    colorbar = figure.colorbar(
        mesh,
        ax=axis,
        pad=0.018,
    )
    colorbar.set_label(
        "Unit-area response [day$^{-1}$]"
    )

    figure.tight_layout()
    figure.savefig(
        output_dir
        / "wavelength_delay_map.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def print_summary(
    result: dict,
    messages: list[str],
) -> str:
    """Print and return a compact scientific run report."""
    lines = [
        "Scientific test report",
        "=" * 76,
        *messages,
        "",
        "Model parameters",
        "-" * 76,
        f"M_BH                 = {M_BH_MSUN:.6e} M_sun",
        f"Mdot                 = {result['parameters'].mdot_g_s:.6e} g/s",
        f"L_X                  = {L_X_ERG_S:.6e} erg/s",
        f"h                    = {H_RG:.4f} r_g",
        f"inclination          = {INCLINATION_DEG:.4f} deg",
        f"R_in                 = {RIN_RG:.4f} r_g",
        f"R_out                = {ROUT_RG:.4f} r_g",
        "",
        (
            " wavelength[A]  peak[d]  mean[d]  median[d]"
            "  rel.mean[d]  Rmean[r_g]"
        ),
        "-" * 76,
    ]

    for index, wavelength in enumerate(
        result["wavelengths_angstrom"]
    ):
        lines.append(
            f"{wavelength:13.1f}"
            f"{result['peak_lag_days'][index]:9.4f}"
            f"{result['mean_lag_days'][index]:9.4f}"
            f"{result['median_lag_days'][index]:11.4f}"
            f"{result['relative_mean_lag_days'][index]:13.4f}"
            f"{result['mean_response_radius_rg'][index]:12.1f}"
        )

    lines.extend([
        "",
        "Diagnostics",
        "-" * 76,
        (
            "minimum delay         = "
            f"{result['diagnostics']['minimum_delay_days']:.6f} day"
        ),
        (
            "maximum delay         = "
            f"{result['diagnostics']['maximum_delay_days']:.6f} day"
        ),
        (
            "stationary radius     = "
            f"{result['diagnostics']['stationary_radius_rg']:.6f} r_g"
        ),
        (
            "stationary point disk = "
            f"{result['diagnostics']['stationary_point_in_disk']}"
        ),
        (
            "minimum captured frac = "
            f"{result['diagnostics']['captured_fraction_min']:.10f}"
        ),
        (
            "disk area rel. error  = "
            f"{result['diagnostics']['disk_area_relative_error']:.3e}"
        ),
    ])

    report = "\n".join(lines)
    print(report)
    return report


def main(
    output_dir: Path | None = None,
) -> None:
    """Run the reference calculation, checks, and plots."""
    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent
            / "test_outputs"
        )

    mdot_g_s = mdot_from_eddington_ratio(
        M_BH_MSUN,
        EDDINGTON_RATIO,
        RADIATIVE_EFFICIENCY,
    )

    model = DiskTransferModel(
        wavelengths_angstrom=(
            WAVELENGTHS_ANGSTROM
        ),
        time_days=TIME_DAYS,
        global_rmin_rg=6.0,
        rout_rg=ROUT_RG,
        nr=NR,
        nphi=NPHI,
        pulse_width_days=PULSE_WIDTH_DAYS,
    )

    result = model.predict(
        m_bh_msun=M_BH_MSUN,
        mdot_g_s=mdot_g_s,
        lx_erg_s=L_X_ERG_S,
        h_rg=H_RG,
        inclination_deg=INCLINATION_DEG,
        rin_rg=RIN_RG,
        reference_index=REFERENCE_INDEX,
    )

    messages = run_scientific_checks(
        model,
        result,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = print_summary(
        result,
        messages,
    )
    (
        output_dir / "test_report.txt"
    ).write_text(
        report,
        encoding="utf-8",
    )

    save_tables(
        result,
        output_dir,
    )
    plot_scientific_overview(
        result,
        output_dir,
    )
    plot_wavelength_delay_map(
        result,
        output_dir,
    )

    print()
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
