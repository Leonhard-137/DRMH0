#!/usr/bin/env python3
"""Fit Mrk817 Gamma comp0 centroids with a lambda^(4/3) lag law.

U is excluded from the fit and predicted from UVM2, UVW1, B, and V.  The
primary model is anchored to zero relative lag at UVW2:

    tau(lambda) = A * ((lambda / lambda_UVW2)**(4/3) - 1).

The saved Gamma posterior is used directly.  Results are also recomputed after
importance reweighting from flat(log w) to flat(w0, w1), and after the w0-only
diagnostic reweighting used elsewhere in this project.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "Mrk817/runs/mica/legacy_archives/gamma0_100_2comp.zip"
OUTPUT = ROOT / "Mrk817/results/tf_comparison"
WAVELENGTH = {
    "UVW2": 1928.0,
    "UVM2": 2246.0,
    "UVW1": 2600.0,
    "U": 3465.0,
    "B": 4392.0,
    "V": 5468.0,
}
FIT_BANDS = ("UVM2", "UVW1", "B", "V")
ALL_BANDS = ("UVM2", "UVW1", "U", "B", "V")
SCHEMES = ("original", "flat_w0_w1", "flat_w0_only")
SCHEME_LABEL = {
    "original": "Original",
    "flat_w0_w1": r"flat $w_0,w_1$",
    "flat_w0_only": r"flat $w_0$ only",
}
SCHEME_COLOR = {
    "original": "#333333",
    "flat_w0_w1": "#D97732",
    "flat_w0_only": "#3976A8",
}


def member(band: str, filename: str) -> str:
    return f"2comp/run_UVW2_to_{band}_2comp_gamma/data/{filename}"


def normalize(log_weight: np.ndarray) -> np.ndarray:
    shifted = log_weight - np.max(log_weight)
    weight = np.exp(shifted)
    return weight / np.sum(weight)


def load_band(archive: zipfile.ZipFile, band: str) -> dict[str, np.ndarray]:
    sample = np.loadtxt(io.BytesIO(archive.read(member(band, "posterior_sample1d.txt_2"))))
    w0 = np.exp(sample[:, 6])
    w1 = np.exp(sample[:, 9])
    tau0 = sample[:, 5] + 2.0 * w0
    tau1 = sample[:, 8] + 2.0 * w1
    keep = tau0 < tau1
    tau0 = tau0[keep]
    w0 = w0[keep]
    w1 = w1[keep]
    n = tau0.size
    return {
        "tau0": tau0,
        "original": np.full(n, 1.0 / n),
        "flat_w0_w1": normalize(np.log(w0) + np.log(w1)),
        "flat_w0_only": normalize(np.log(w0)),
    }


def weighted_quantile(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)
    return np.interp([0.16, 0.50, 0.84], cdf, values, left=values[0], right=values[-1])


def x_value(wavelength: float) -> float:
    return (wavelength / WAVELENGTH["UVW2"]) ** (4.0 / 3.0) - 1.0


def fit_scheme(
    data: dict[str, dict[str, np.ndarray]], scheme: str, rng: np.random.Generator
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    x = np.array([x_value(WAVELENGTH[band]) for band in FIT_BANDS])
    summaries = {
        band: weighted_quantile(data[band]["tau0"], data[band][scheme])
        for band in ALL_BANDS
    }
    y = np.array([summaries[band][1] for band in FIT_BANDS])
    sigma = np.array(
        [0.5 * (summaries[band][2] - summaries[band][0]) for band in FIT_BANDS]
    )
    inverse_variance = 1.0 / np.square(sigma)

    denominator = np.sum(inverse_variance * np.square(x))
    amplitude = np.sum(inverse_variance * x * y) / denominator
    anchored_residual = y - amplitude * x
    anchored_chi2 = np.sum(np.square(anchored_residual / sigma))

    design = np.column_stack((np.ones_like(x), x))
    normal_inverse = np.linalg.inv(design.T @ (inverse_variance[:, None] * design))
    beta = normal_inverse @ (design.T @ (inverse_variance * y))
    free_residual = y - design @ beta
    free_chi2 = np.sum(np.square(free_residual / sigma))

    draw_count = 150_000
    draws = np.empty((draw_count, len(FIT_BANDS)))
    for column, band in enumerate(FIT_BANDS):
        values = data[band]["tau0"]
        weights = data[band][scheme]
        indices = rng.choice(values.size, size=draw_count, replace=True, p=weights)
        draws[:, column] = values[indices]

    anchored_amplitude_draws = (
        draws @ (inverse_variance * x) / denominator
    )
    free_beta_draws = draws @ (inverse_variance[:, None] * design) @ normal_inverse
    x_u = x_value(WAVELENGTH["U"])
    anchored_u_draws = anchored_amplitude_draws * x_u
    free_u_draws = free_beta_draws[:, 0] + free_beta_draws[:, 1] * x_u
    u_values = data["U"]["tau0"]
    u_weights = data["U"][scheme]
    u_indices = rng.choice(u_values.size, size=draw_count, replace=True, p=u_weights)
    observed_u_draws = u_values[u_indices]
    anchored_u_difference_draws = anchored_u_draws - observed_u_draws

    result = {
        "amplitude": amplitude,
        "anchored_chi2": anchored_chi2,
        "anchored_dof": len(FIT_BANDS) - 1,
        "free_intercept": beta[0],
        "free_slope": beta[1],
        "free_chi2": free_chi2,
        "free_dof": len(FIT_BANDS) - 2,
        "observed_u_q16": summaries["U"][0],
        "observed_u_median": summaries["U"][1],
        "observed_u_q84": summaries["U"][2],
    }
    arrays = {
        "anchored_amplitude": anchored_amplitude_draws,
        "anchored_u": anchored_u_draws,
        "free_u": free_u_draws,
        "observed_u": observed_u_draws,
        "anchored_u_difference": anchored_u_difference_draws,
        "summaries": summaries,
    }
    return result, arrays


def q(values: np.ndarray) -> np.ndarray:
    return np.quantile(values, [0.16, 0.50, 0.84])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        data = {band: load_band(archive, band) for band in ALL_BANDS}

    rng = np.random.default_rng(817)
    results: dict[str, dict[str, float]] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    rows = []
    band_rows = []
    for scheme in SCHEMES:
        results[scheme], arrays[scheme] = fit_scheme(data, scheme, rng)
        amplitude_q = q(arrays[scheme]["anchored_amplitude"])
        anchored_u_q = q(arrays[scheme]["anchored_u"])
        free_u_q = q(arrays[scheme]["free_u"])
        u_difference_q = q(arrays[scheme]["anchored_u_difference"])
        result = results[scheme]
        rows.append(
            {
                "scheme": scheme,
                "fit_bands": ";".join(FIT_BANDS),
                "law": "A*((lambda/1928A)^(4/3)-1)",
                "A_point_estimate_days": result["amplitude"],
                "A_q16_days": amplitude_q[0],
                "A_median_days": amplitude_q[1],
                "A_q84_days": amplitude_q[2],
                "chi2": result["anchored_chi2"],
                "dof": result["anchored_dof"],
                "predicted_U_q16_days": anchored_u_q[0],
                "predicted_U_median_days": anchored_u_q[1],
                "predicted_U_q84_days": anchored_u_q[2],
                "predicted_minus_observed_U_q16_days": u_difference_q[0],
                "predicted_minus_observed_U_median_days": u_difference_q[1],
                "predicted_minus_observed_U_q84_days": u_difference_q[2],
                "probability_predicted_U_exceeds_observed": float(
                    np.mean(arrays[scheme]["anchored_u_difference"] > 0.0)
                ),
                "free_intercept_point_days": result["free_intercept"],
                "free_slope_point_days": result["free_slope"],
                "free_chi2": result["free_chi2"],
                "free_dof": result["free_dof"],
                "free_predicted_U_q16_days": free_u_q[0],
                "free_predicted_U_median_days": free_u_q[1],
                "free_predicted_U_q84_days": free_u_q[2],
                "observed_U_q16_days": result["observed_u_q16"],
                "observed_U_median_days": result["observed_u_median"],
                "observed_U_q84_days": result["observed_u_q84"],
            }
        )
        for band in ALL_BANDS:
            band_q = arrays[scheme]["summaries"][band]
            band_rows.append(
                {
                    "scheme": scheme,
                    "band": band,
                    "wavelength_angstrom": WAVELENGTH[band],
                    "used_in_fit": band in FIT_BANDS,
                    "tau0_q16_days": band_q[0],
                    "tau0_median_days": band_q[1],
                    "tau0_q84_days": band_q[2],
                }
            )

    pd.DataFrame(rows).to_csv(
        OUTPUT / "Mrk817_gamma_comp0_powerlaw_exclude_U_fit_summary.csv", index=False
    )
    pd.DataFrame(band_rows).to_csv(
        OUTPUT / "Mrk817_gamma_comp0_powerlaw_exclude_U_band_values.csv", index=False
    )

    figure, axis = plt.subplots(figsize=(9.3, 6.3), constrained_layout=True)
    original_summary = arrays["original"]["summaries"]
    for band in ALL_BANDS:
        band_q = original_summary[band]
        excluded = band == "U"
        color = "#B34A3C" if excluded else "#333333"
        marker = "s" if excluded else "o"
        axis.errorbar(
            WAVELENGTH[band],
            band_q[1],
            yerr=[[band_q[1] - band_q[0]], [band_q[2] - band_q[1]]],
            fmt=marker,
            markersize=7.0,
            markerfacecolor="white" if excluded else color,
            markeredgecolor=color,
            color=color,
            capsize=3.0,
            linewidth=1.2,
            zorder=5,
        )
        axis.annotate(
            band,
            (WAVELENGTH[band], band_q[1]),
            xytext=(5, -15 if band == "UVM2" else 7),
            textcoords="offset points",
            fontsize=9,
            color=color,
        )
    axis.scatter(
        [WAVELENGTH["UVW2"]],
        [0.0],
        marker="D",
        s=38,
        color="#333333",
        zorder=5,
    )
    axis.annotate("UVW2 anchor", (WAVELENGTH["UVW2"], 0.0), xytext=(6, 7), textcoords="offset points", fontsize=9)

    wavelength_grid = np.linspace(WAVELENGTH["UVW2"], 5700.0, 350)
    x_grid = np.array([x_value(value) for value in wavelength_grid])
    line_styles = {"original": "-", "flat_w0_w1": "-.", "flat_w0_only": "--"}
    for scheme in SCHEMES:
        amplitude_q = q(arrays[scheme]["anchored_amplitude"])
        axis.plot(
            wavelength_grid,
            amplitude_q[1] * x_grid,
            color=SCHEME_COLOR[scheme],
            linestyle=line_styles[scheme],
            linewidth=2.0 if scheme == "original" else 1.55,
            label=SCHEME_LABEL[scheme],
        )
        if scheme == "original":
            axis.fill_between(
                wavelength_grid,
                amplitude_q[0] * x_grid,
                amplitude_q[2] * x_grid,
                color=SCHEME_COLOR[scheme],
                alpha=0.12,
                linewidth=0.0,
            )
        u_q = q(arrays[scheme]["anchored_u"])
        offset = {"original": -35.0, "flat_w0_w1": 0.0, "flat_w0_only": 35.0}[scheme]
        axis.errorbar(
            WAVELENGTH["U"] + offset,
            u_q[1],
            yerr=[[u_q[1] - u_q[0]], [u_q[2] - u_q[1]]],
            fmt="*",
            markersize=10,
            color=SCHEME_COLOR[scheme],
            capsize=2.5,
            zorder=6,
        )

    original_free = results["original"]
    axis.plot(
        wavelength_grid,
        original_free["free_intercept"] + original_free["free_slope"] * x_grid,
        color="#777777",
        linestyle=":",
        linewidth=1.3,
        label="Original, free intercept",
    )
    axis.axhline(0.0, color="#888888", linewidth=0.75, alpha=0.55)
    axis.set_xlabel(r"Observed wavelength $\lambda$ ($\AA$)")
    axis.set_ylabel(r"Gamma comp0 centroid $\tau_0$ (d)")
    axis.set_title(r"Mrk817 Gamma comp0 lag--wavelength fit ($\lambda^{4/3}$)")
    axis.text(
        0.01,
        0.98,
        "U excluded; primary fit anchored to zero relative lag at UVW2\n"
        "circles: fitted Original bands; square: observed U; stars: predicted U",
        transform=axis.transAxes,
        va="top",
        fontsize=9.5,
    )
    axis.grid(alpha=0.18, linewidth=0.55)
    axis.legend(frameon=False, loc="lower right", fontsize=9)
    axis.set_xlim(1800.0, 5700.0)
    axis.set_ylim(-0.22, 1.35)
    figure.savefig(OUTPUT / "Mrk817_gamma_comp0_powerlaw_exclude_U.png", dpi=220)
    figure.savefig(OUTPUT / "Mrk817_gamma_comp0_powerlaw_exclude_U.pdf")
    plt.close(figure)

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
