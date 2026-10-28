#!/usr/bin/env python3
"""Plot the generated UVW2 and U input light curves for every case."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from prepare import (case_name, disk_bin_masses, gaussian_bin_masses, kernel_grid,
                     known_response_kernels, real_disk_mismatch_cases,
                     real_disk_mismatch_input_kernels, spectral_shared_cases)
from plot_style import (COLORS, SHORT_DASH, REFERENCE_DASH, configure_style,
                        observations, save_figure, style_axis)


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "experiment.yaml"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration is not a mapping: {path}")
    return config


def load_curve(path: Path) -> np.ndarray:
    values = np.loadtxt(path)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Expected three columns in {path}")
    if not np.all(np.isfinite(values)) or not np.all(values[:, 2] > 0):
        raise ValueError(f"Invalid values in {path}")
    return values


def load_components(case_dir: Path, band: str, curve: np.ndarray,
                    shared_case: dict | None = None,
                    spectral_case: dict | None = None,
                    real_case: dict | None = None) -> np.ndarray:
    component_file = case_dir / "components" / f"{band}_components.txt"
    components = np.loadtxt(component_file)
    is_ablation = real_case is not None and "shared_driver" in real_case
    expected_columns = 7 if is_ablation else 5 if real_case is not None else 4
    if components.shape != (len(curve), expected_columns) or not np.isfinite(components).all():
        raise ValueError(f"Invalid component table: {component_file}")
    np.testing.assert_allclose(components[:, 0], curve[:, 0], atol=1e-9, rtol=0)
    noise_index = 4 if real_case is not None else 3
    np.testing.assert_allclose(1 + components[:, 1:noise_index + 1].sum(axis=1),
                               curve[:, 1], atol=1e-10, rtol=0)
    if is_ablation:
        np.testing.assert_allclose(components[:, 5] + components[:, 6], components[:, 1],
                                   atol=1e-10, rtol=0)
    if shared_case is not None or spectral_case is not None or real_case is not None:
        # Do not silently relabel old data after changing the YAML parameters.
        metadata = {}
        with component_file.open() as handle:
            first_line = handle.readline().strip()
            if real_case is not None:
                expected = ("# time illumination_variation shared_long_variation "
                            "shared_short_variation measurement_noise")
                if is_ablation:
                    expected += " disk_illumination_variation blr_illumination_variation"
            elif spectral_case is not None:
                expected = "# time primary_variation shared_slow_variation measurement_noise"
            else:
                expected = "# time disk_variation shared_slow_variation measurement_noise"
            if first_line != expected:
                raise ValueError(f"Unexpected component schema: {component_file}")
            for line in handle:
                if not line.startswith("# "):
                    break
                for item in line[2:].strip().split("; "):
                    if "=" in item:
                        key, value = item.split("=", 1)
                        metadata[key] = value
        fields = {"slow_tau_days": "tau_days", "lag_days": "lag_days",
                  "aU_over_aW": "response_amplitude_ratio",
                  "driver_slow_to_clean_rms": "driver_rms_ratio"}
        settings = shared_case
        if spectral_case is not None:
            fields = {key: key for key in spectral_case}
            settings = spectral_case
        if real_case is not None:
            fields = {key: key for key in real_case}
            settings = real_case
        for field, setting in fields.items():
            if field not in metadata or not np.isclose(float(metadata[field]), float(settings[setting]), rtol=1e-10, atol=1e-10):
                raise ValueError(f"Configuration does not match {field} in {component_file}; regenerate data or use its original config")
    return components


def plot_case(case_dir: Path, source: str, bands: tuple[str, str], independent_case: dict | None = None,
              shared_case: dict | None = None, known_case: dict | None = None,
              spectral_case: dict | None = None,
              real_case: dict | None = None) -> None:
    if sum(item is not None for item in (
        independent_case, shared_case, known_case, spectral_case, real_case
    )) > 1:
        raise ValueError("A case must belong to only one input suite")
    curves = {
        band: load_curve(case_dir / "input" / f"{source}_{band}.txt")
        for band in bands
    }
    components = {}
    if independent_case is not None:
        components[bands[1]] = load_components(case_dir, bands[1], curves[bands[1]])
    if shared_case is not None:
        components = {band: load_components(case_dir, band, curves[band], shared_case) for band in bands}
    if known_case is not None:
        components[bands[1]] = load_components(case_dir, bands[1], curves[bands[1]])
    if spectral_case is not None:
        components = {band: load_components(case_dir, band, curves[band], spectral_case=spectral_case)
                      for band in bands}
    if real_case is not None:
        components = {
            band: load_components(case_dir, band, curves[band], real_case=real_case)
            for band in bands
        }
    has_shared = shared_case is not None or spectral_case is not None
    has_mixed = has_shared or real_case is not None
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.4 if has_mixed else 5.8), sharex=True)

    for ax, band in zip(axes, bands):
        time, flux, error = curves[band].T
        is_response = independent_case is not None and band == bands[1]
        known_response = known_case is not None and band == bands[1]
        has_independent = is_response and independent_case["tau_days"] is not None
        total_label = None
        if real_case is not None:
            total_label = "Total input"
        elif spectral_case is not None:
            total_label = ("Driver (fast + shared slow)" if band == bands[0]
                           else "Response (narrow + shared slow)")
        elif shared_case is not None:
            total_label = "Total input (disk + shared slow)"
        elif known_response:
            total_label = "Response (disk + broad echo)"
        elif is_response:
            total_label = "Response (disk + independent)" if has_independent else "Response (disk only)"
        ax.plot(time, flux, color=COLORS["INK"], linewidth=.8, zorder=1)
        observations(ax, time, flux, error, label=total_label)
        lower = float(np.min(flux - error))
        upper = float(np.max(flux + error))
        if has_independent:
            # Independent contributions are zero-mean variations. Offset by 1
            # for display on the same normalized-flux axis; add no extra noise.
            independent_display = 1 + components[band][:, 2]
            tau = float(independent_case["tau_days"])
            ax.plot(time, independent_display, linestyle="none", marker="o",
                    markersize=2.5, markeredgewidth=0, color=COLORS["ALTERNATIVE"], zorder=4,
                    label=rf"Independent (+1), $\tau={tau:g}$ d")
            lower = min(lower, float(independent_display.min()))
            upper = max(upper, float(independent_display.max()))
        if has_shared:
            # Both components are noise-free variations; +1 is display-only.
            # Preserve the saved amplitudes and the original observing epochs.
            disk_display = 1 + components[band][:, 1]
            slow_display = 1 + components[band][:, 2]
            ax.plot(time, disk_display, color=COLORS["REFERENCE"], linestyle=REFERENCE_DASH, linewidth=1.25,
                    label=("Clean disk (+1)" if spectral_case is None else
                           "Fast component (+1)" if band == bands[0] else "Narrow response (+1)"), zorder=2)
            ax.plot(time, slow_display, linestyle="none", marker="s", markersize=2.4,
                    markeredgewidth=0, color=COLORS["ALTERNATIVE"], zorder=4,
                    label=("Delayed shared slow (+1)" if spectral_case is not None and band == bands[1]
                           else "Shared slow (+1)"))
            lower = min(lower, float(disk_display.min()), float(slow_display.min()))
            upper = max(upper, float(disk_display.max()), float(slow_display.max()))
        if real_case is not None:
            illumination_display = 1 + components[band][:, 1]
            long_display = 1 + components[band][:, 2]
            short_display = 1 + components[band][:, 3]
            illumination_label = (
                r"$D_{W2}*I_0$ (+1)" if band == bands[0]
                else r"$(aD_U+bB)*I_0$ (+1)" if real_case["response_blr_gain"] > 0
                else r"$aD_U*I_0$ (+1)"
            )
            ax.plot(
                time, illumination_display, color=COLORS["REFERENCE"], linestyle=REFERENCE_DASH,
                linewidth=1.25, label=illumination_label, zorder=2,
            )
            displays = [illumination_display]
            shared_enabled = real_case.get("shared_driver" if band == bands[0] else "shared_response", True)
            if real_case["long_q"] > 0 and shared_enabled:
                independent_u = band == bands[1] and not real_case.get("shared_driver", True)
                ax.plot(
                    time, long_display, linestyle="none", marker="s", markersize=2.4,
                    markeredgewidth=0, color=COLORS["ALTERNATIVE"], zorder=4,
                    label="Independent U process (+1)" if independent_u else "Shared process (+1)",
                )
                displays.append(long_display)
            if real_case["short_q"] > 0:
                ax.plot(
                    time, short_display, linestyle="none", marker="^", markersize=2.7,
                    markeredgewidth=0, color=COLORS["SECONDARY"], zorder=4,
                    label="Shifted short shared process (+1)",
                )
                displays.append(short_display)
            lower = min(lower, *(float(value.min()) for value in displays))
            upper = max(upper, *(float(value.max()) for value in displays))
        padding = 0.06 * (upper - lower)
        if known_response:
            disk_display = 1 + components[band][:, 1]
            broad_display = 1 + components[band][:, 2]
            ax.plot(time, disk_display, color=COLORS["REFERENCE"], linestyle=REFERENCE_DASH, linewidth=1.25,
                    label="Weighted disk (+1)", zorder=2)
            ax.plot(time, broad_display, linestyle="none", marker="s", markersize=2.4,
                    markeredgewidth=0, color=COLORS["ALTERNATIVE"], zorder=4,
                    label="Weighted broad echo (+1)")
            lower = min(lower, float(disk_display.min()), float(broad_display.min()))
            upper = max(upper, float(disk_display.max()), float(broad_display.max()))
            padding = 0.06 * (upper - lower)
        ax.set_ylim(lower - padding, upper + (7 * padding if has_mixed or known_response else
                                             3 * padding if is_response else padding))
        if is_response or has_mixed or known_response:
            ax.legend(loc="upper right", frameon=False, fontsize=10, markerscale=1.6)
        ax.set_ylabel("Normalized Flux", fontsize=13, labelpad=8)
        ax.text(
            0.025,
            0.95 if has_mixed or known_response else 0.89,
            band,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
        )
        style_axis(ax)

    axes[-1].set_xlabel("Time (days)", fontsize=14)
    axes[-1].set_xlim(
        min(curve[:, 0].min() for curve in curves.values()),
        max(curve[:, 0].max() for curve in curves.values()),
    )
    plt.setp(axes[0].get_xticklabels(), visible=False)
    fig.subplots_adjust(left=0.055, right=0.86, bottom=0.12, top=0.98, hspace=0.035)

    result_dir = case_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    output = result_dir / "input_light_curves"
    save_figure(fig, output)
    print(f"wrote {output}.jpg and {output}.pdf")


def response_limits(lag: np.ndarray, mass: np.ndarray, dt: float) -> tuple[float, float]:
    """Central 99% of response area plus 4% display margins."""
    edges = np.concatenate(([lag[0] - dt / 2], lag + dt / 2))
    cumulative = np.concatenate(([0.0], np.cumsum(mass) / mass.sum()))
    lower, upper = np.interp([0.005, 0.995], cumulative, edges)
    padding = 0.04 * (upper - lower)
    return max(edges[0], lower - padding), min(edges[-1], upper + padding)


def spectral_transfer_functions(config: dict) -> tuple[np.ndarray, dict[str, dict]]:
    """Display H_eff only; the eta=1 process also has nonzero residual power."""
    curves = {}
    lag = np.array([])
    for name, p in spectral_shared_cases(config).items():
        dt = min(p["narrow_sigma_days"] / 8, p["slow_sigma_days"] / 80)
        lower = min(p["narrow_mean_days"] - 8 * p["narrow_sigma_days"],
                    p["lag_days"] - 8 * p["slow_sigma_days"])
        upper = max(p["narrow_mean_days"] + 8 * p["narrow_sigma_days"],
                    p["lag_days"] + 8 * p["slow_sigma_days"])
        lower = np.floor(lower / dt) * dt
        count = int(np.ceil((upper - lower) / dt))
        lag = lower + (np.arange(count) + 0.5) * dt
        narrow, _, _ = gaussian_bin_masses(lag, dt, p["narrow_mean_days"], p["narrow_sigma_days"])
        broad, _, _ = gaussian_bin_masses(lag, dt, p["lag_days"], p["slow_sigma_days"])
        narrow *= p["narrow_gain"]
        broad *= p["slow_gain"]
        total = narrow + broad
        curves[name] = {
            "lag": lag, "total": total / dt, "narrow": narrow / dt, "gaussian": broad / dt,
            "limits": response_limits(lag, total, dt),
            "labels": {"total": "Effective linear predictor", "narrow": "Narrow component",
                       "gaussian": "Broad effective component"},
            "title": "Effective kernel, not a deterministic total response",
            "note": "Shared latent process: unpredictable residual power is not represented by this kernel.",
            "ylabel": r"Effective Kernel (day$^{-1}$)",
        }
    return lag, curves


def build_transfer_functions(config: dict, config_path: Path | None = None) -> tuple[np.ndarray, dict[str, dict]]:
    lag, curves = spectral_transfer_functions(config)
    real_cases = real_disk_mismatch_cases(config)
    if "paired_ablation" in config:
        if config_path is None:
            raise ValueError("paired_ablation kernels require the experiment config_path")
        from prepare_ablation import ablation_cases
        real_cases.update(ablation_cases(config_path, config))
    for name, parameters in real_cases.items():
        real_lag, kernels = real_disk_mismatch_input_kernels(parameters)
        dt = real_lag[1] - real_lag[0]
        curves[name] = {
            "lag": real_lag,
            "total": kernels["total"],
            "disk": kernels["disk"],
            "blr": kernels["blr"],
            "limits": response_limits(real_lag, kernels["total"] * dt, dt),
            "labels": {
                "total": r"Physical $I_0\rightarrow U$ response",
                "disk": rf"${parameters['response_disk_gain']:.3g}\,\Psi_{{U,\rm disk}}$",
                "blr": rf"${parameters['response_blr_gain']:.3g}\,\Psi_{{U,\rm BLR}}$",
            },
            "title": "Causal illumination response, not a total W2-to-U kernel",
            "note": ("Only the causal illumination pathway is shown. Additional process "
                     "pathways are omitted; mixed inputs have no single deterministic "
                     "W2-to-U transfer function represented by this curve."),
        }
        if parameters["response_blr_gain"] == 0:
            curves[name].pop("blr")
            curves[name]["labels"].pop("blr")
        lag = real_lag
    if "disk" not in config and curves:
        return lag, curves
    grid_config = config["grid"]
    disk_config = config["disk"]
    response_config = config.get("response", {})
    response_band = str(config["observations"]["response_band"])
    lag, dt, oversample = kernel_grid(grid_config)
    redshift = float(disk_config["redshift"])
    wavelength_rest = (
        float(disk_config["wavelength_observed_angstrom"][response_band])
        / (1.0 + redshift)
    )
    disk_mass, _ = disk_bin_masses(
        lag,
        dt,
        oversample,
        wavelength_rest_angstrom=wavelength_rest,
        t1_kelvin=float(disk_config["t1_kelvin"]),
        inclination_deg=float(disk_config["inclination_deg"]),
        redshift=redshift,
    )
    disk_fraction = float(response_config.get("disk_fraction", 1.0))
    gaussian_fraction = float(response_config.get("gaussian_fraction", 0.0))
    for name, parameters in response_config.get("gaussian_cases", {}).items():
        mean = float(parameters["mean_days"])
        sigma = float(parameters["sigma_days"])
        gaussian_mass, _, _ = gaussian_bin_masses(lag, dt, mean, sigma)
        disk_component_mass = disk_fraction * disk_mass
        gaussian_component_mass = gaussian_fraction * gaussian_mass
        total_mass = disk_component_mass + gaussian_component_mass
        curves[str(name)] = {
            "total": total_mass / dt,
            "disk": disk_component_mass / dt,
            "gaussian": gaussian_component_mass / dt,
            "limits": response_limits(lag, total_mass, dt),
            "labels": {
                "total": "Total response",
                "disk": rf"${disk_fraction:g}\,\Psi_{{{response_band},\rm disk}}$",
                "gaussian": rf"${gaussian_fraction:g}\,\Psi_{{\rm Gaussian}}$",
            },
        }
    # In the independent suite the known W2-to-U transfer is this single disk
    # kernel. The independent OU has no transfer function relative to W2.
    for name in config.get("independent_response", {}).get("cases", {}):
        curves[name] = {
            "total": disk_mass / dt,
            "limits": response_limits(lag, disk_mass, dt),
            "labels": {"total": f"Injected {config['observations']['driver_band']}-to-{response_band} disk"},
        }
    for name, parameters in known_response_kernels(config).items():
        wide_lag = parameters["lag"]
        # Embed the already integrated disk bins in the wider display grid.
        offset = int(round((lag[0] - wide_lag[0]) / dt))
        wide_disk = np.zeros_like(wide_lag)
        wide_disk[offset:offset + len(disk_mass)] = disk_mass
        disk_part = parameters["disk_fraction"] * wide_disk
        broad_part = parameters["gaussian_fraction"] * parameters["mass"]
        total = disk_part + broad_part
        curves[name] = {
            "lag": wide_lag, "total": total / dt,
            "disk": disk_part / dt, "gaussian": broad_part / dt,
            "limits": response_limits(wide_lag, total, dt),
            "labels": {"total": "Injected W2-to-U response",
                       "disk": "$" + rf"{parameters['disk_fraction']:g}\,\Psi_{{\rm disk}}" + "$",
                       "gaussian": "$" + rf"{parameters['gaussian_fraction']:g}\,\Psi_{{\rm Gaussian}}" + "$"},
        }
    shared = config.get("shared_slow", {})
    for name, parameters in shared.get("cases", {}).items():
        # Only the clean-disk pathway has this kernel. There is deliberately
        # no "total" entry: do not add the shared latent signal to the disk.
        curves[name] = {
            "disk": disk_mass / dt,
            "limits": response_limits(lag, disk_mass, dt),
            "labels": {"disk": rf"Clean $W_0\rightarrow {response_band}_{{\rm disk}}$ kernel"},
            "title": "Clean-disk kernel, not the total mixed-driver transfer",
            "note": (rf"Shared slow: $\tau_s={float(shared['tau_days']):g}$ d; "
                     rf"$\Delta=+{float(parameters['lag_days']):g}$ d; "
                     rf"$a_U/a_W={float(shared['response_amplitude_ratio']):g}$ (not included in this kernel)."),
        }
    return lag, curves


def plot_transfer_function(
    case_dir: Path,
    gaussian_name: str,
    lag: np.ndarray,
    curves: dict,
) -> None:
    lag = curves.get("lag", lag)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    # Project convention: black total; colored short-dashed components.
    colors = {"total": "black", "disk": COLORS["PRIMARY"], "narrow": COLORS["PRIMARY"],
              "gaussian": COLORS["ALTERNATIVE"], "blr": COLORS["ALTERNATIVE"]}
    labels = curves["labels"]
    main_component = "total" if "total" in curves else "disk"
    for ax in axes:
        ax.plot(
            lag, curves[main_component], color=colors["total"],
            linewidth=1.8, linestyle="-", label=labels[main_component], zorder=2,
        )
        for component in ("disk", "narrow", "gaussian", "blr"):
            if component not in curves or component == main_component:
                continue
            ax.plot(
                lag, curves[component], color=colors[component],
                linewidth=1.35, linestyle=SHORT_DASH, label=labels[component], zorder=3,
            )
        ax.set_xlim(*curves["limits"])
        ax.set_xlabel(r"$\tau$ (days)", fontsize=14)

    maximum = float(np.max(curves[main_component]))
    axes[0].set_ylim(0.0, 1.10 * maximum)
    axes[0].set_ylabel(curves.get("ylabel", r"Transfer Function (day$^{-1}$)"), fontsize=13)
    axes[0].legend(loc="upper right", frameon=False, fontsize=10.5)

    axes[1].set_yscale("log")
    axes[1].set_ylim(maximum * 1e-6, maximum * 1.25)
    axes[1].set_ylabel(curves.get("ylabel", r"Transfer Function (day$^{-1}$)"), fontsize=13)
    style_axis(axes[0], y_on_right=False)
    style_axis(axes[1], y_on_right=True)
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.18, top=0.97, wspace=0.18)
    # Kernel definitions remain in the legend and README; explanatory
    # metadata in curves["note"] is not rendered without an explicit request.

    result_dir = case_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    output = result_dir / "input_transfer_function"
    save_figure(fig, output)
    print(f"wrote {output}.jpg and {output}.pdf ({gaussian_name})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--suite",
        choices=("all", "gaussian", "independent", "shared_slow",
                 "known_response", "spectral_shared", "real_disk_mismatch", "paired_ablation"),
        default="all",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Plot only the named case; repeat to select multiple cases.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    output_dir = (config_path.parent / config["experiment"]["output_dir"]).resolve()
    source = str(config["experiment"]["simulation_source"])
    observations = config["observations"]
    bands = (str(observations["driver_band"]), str(observations["response_band"]))
    case_dirs = (
        [output_dir / name for name in args.case]
        if args.case
        else sorted(path for path in output_dir.iterdir() if path.is_dir())
    )
    independent_cases = config.get("independent_response", {}).get("cases", {})
    shared_suite = config.get("shared_slow", {})
    shared_cases = {name: {**shared_suite, **parameters}
                    for name, parameters in shared_suite.get("cases", {}).items()}
    known_cases = config.get("known_response", {}).get("cases", {})
    spectral_cases = spectral_shared_cases(config)
    real_cases = real_disk_mismatch_cases(config)
    ablation_parameters = {}
    if "paired_ablation" in config:
        from prepare_ablation import ablation_cases
        ablation_parameters = ablation_cases(config_path, config)
    gaussian_cases = {case_name(float(tau), name): name for tau in config.get("drw", {}).get("tau_days", [])
                      for name in config.get("response", {}).get("gaussian_cases", {})}
    case_suites = {**dict.fromkeys(gaussian_cases, "gaussian"),
                   **dict.fromkeys(independent_cases, "independent"),
                   **dict.fromkeys(shared_cases, "shared_slow"),
                   **dict.fromkeys(known_cases, "known_response"),
                   **dict.fromkeys(spectral_cases, "spectral_shared"),
                   **dict.fromkeys(real_cases, "real_disk_mismatch"),
                   **dict.fromkeys(ablation_parameters, "paired_ablation")}
    real_cases.update(ablation_parameters)
    if not args.case:
        # Several experiment configurations can share the same cases root.
        case_dirs = [path for path in case_dirs if path.name in case_suites]
    if args.suite != "all":
        case_dirs = [path for path in case_dirs if case_suites.get(path.name) == args.suite]
    if not case_dirs:
        raise FileNotFoundError(f"No cases found in {output_dir}")

    configure_style()
    lag, transfer_functions = build_transfer_functions(config, config_path)
    for case_dir in case_dirs:
        if not case_dir.is_dir():
            raise FileNotFoundError(case_dir)
        independent_case = independent_cases.get(case_dir.name)
        shared_case = shared_cases.get(case_dir.name)
        if case_dir.name not in case_suites:
            raise ValueError(f"Case is not defined in the configuration: {case_dir.name!r}")
        gaussian_name = gaussian_cases.get(case_dir.name, case_dir.name)
        plot_case(case_dir, source, bands, independent_case, shared_case, known_cases.get(case_dir.name),
                  spectral_cases.get(case_dir.name), real_cases.get(case_dir.name))
        plot_transfer_function(
            case_dir,
            gaussian_name,
            lag,
            transfer_functions[gaussian_name],
        )


if __name__ == "__main__":
    main()
