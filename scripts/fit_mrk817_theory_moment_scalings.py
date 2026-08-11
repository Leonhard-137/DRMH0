#!/usr/bin/env python3
"""Check wavelength scalings of the finite-disk transfer-function moments.

The input table is produced by ``plot_mrk817_theory_raw_moment_gamma.py``.
It stores the raw B_nu finite-difference response.  This script also derives
the equivalent B_lambda zeroth and raw moments, fits log-log power laws, and
compares them with the infinite self-similar b=3/4 disk predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "Mrk817/results/tf_comparison/theory_forward_gamma"
    / "Mrk817_theory_raw_moment_gamma_overlay_moments.csv"
)
DEFAULT_OUTPUT_BASE = (
    ROOT
    / "Mrk817/results/tf_comparison/theory_forward_gamma"
    / "Mrk817_theory_moment_powerlaw_check"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--relative-output",
        type=Path,
        default=None,
        help="Optional output base for the W2-relative diagnostic.",
    )
    args = parser.parse_args()
    if args.output is None:
        if args.input == DEFAULT_INPUT:
            args.output = DEFAULT_OUTPUT_BASE
        else:
            suffix = "_raw_moment_gamma_overlay_moments"
            stem = args.input.stem
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)] + "_moment_powerlaw_check"
            else:
                stem += "_moment_powerlaw_check"
            args.output = args.input.with_name(stem)
    return args


def build_band_table(source: pd.DataFrame) -> pd.DataFrame:
    first = source.iloc[0]
    driver = pd.DataFrame(
        [
            {
                "band": "UVW2",
                "wavelength_angstrom": 1928.0,
                "m0_nu": first["driver_M0_raw"],
                "centroid_days": first["driver_centroid_days"],
                "variance_days2": first["driver_variance_days2"],
            }
        ]
    )
    targets = source.rename(
        columns={
            "target_M0_raw": "m0_nu",
            "target_centroid_days": "centroid_days",
            "target_variance_days2": "variance_days2",
        }
    )[
        [
            "band",
            "wavelength_angstrom",
            "m0_nu",
            "centroid_days",
            "variance_days2",
        ]
    ]
    table = pd.concat([driver, targets], ignore_index=True)
    table = table.sort_values("wavelength_angstrom").reset_index(drop=True)

    lambda0 = float(table.loc[0, "wavelength_angstrom"])
    m0nu0 = float(table.loc[0, "m0_nu"])
    table["lambda_over_uvw2"] = table["wavelength_angstrom"] / lambda0
    table["m0_nu_relative"] = table["m0_nu"] / m0nu0
    # The omitted dimensional constant c cancels in this W2-relative form.
    table["m0_lambda_relative"] = table["m0_nu_relative"] * (
        lambda0 / table["wavelength_angstrom"]
    ) ** 2
    table["std_days"] = np.sqrt(table["variance_days2"])
    table["normalized_raw_second_days2"] = (
        table["variance_days2"] + table["centroid_days"] ** 2
    )
    table["raw_m1_nu_relative"] = (
        table["m0_nu"] * table["centroid_days"]
    ) / (table.loc[0, "m0_nu"] * table.loc[0, "centroid_days"])
    table["raw_m2_nu_relative"] = (
        table["m0_nu"] * table["normalized_raw_second_days2"]
    ) / (
        table.loc[0, "m0_nu"]
        * table.loc[0, "normalized_raw_second_days2"]
    )
    table["raw_m1_lambda_relative"] = table["raw_m1_nu_relative"] * (
        lambda0 / table["wavelength_angstrom"]
    ) ** 2
    table["raw_m2_lambda_relative"] = table["raw_m2_nu_relative"] * (
        lambda0 / table["wavelength_angstrom"]
    ) ** 2
    table["relative_centroid_days"] = (
        table["centroid_days"] - table.loc[0, "centroid_days"]
    )
    table["relative_variance_days2"] = (
        table["variance_days2"] - table.loc[0, "variance_days2"]
    )
    return table


def powerlaw_fit(
    table: pd.DataFrame,
    column: str,
    theory_slope: float,
    subset: str,
    selection: np.ndarray,
) -> dict[str, float | str | int]:
    selected = table.loc[selection]
    x = np.log(selected["lambda_over_uvw2"].to_numpy(dtype=float))
    y = np.log(selected[column].to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residual = y - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan
    if len(x) > 2:
        slope_se = float(
            np.sqrt(
                (ss_res / (len(x) - 2))
                / np.sum((x - np.mean(x)) ** 2)
            )
        )
    else:
        slope_se = np.nan

    fixed_intercept = float(np.mean(y - theory_slope * x))
    fixed_fraction = np.exp(y - (fixed_intercept + theory_slope * x)) - 1.0

    all_x = table["lambda_over_uvw2"].to_numpy(dtype=float)
    w2_anchored = float(table.loc[0, column]) * all_x**theory_slope
    w2_fraction = table[column].to_numpy(dtype=float) / w2_anchored - 1.0

    return {
        "quantity": column,
        "subset": subset,
        "n_bands": len(selected),
        "fit_slope": float(slope),
        "residual_slope_se": slope_se,
        "theory_slope": theory_slope,
        "slope_minus_theory": float(slope - theory_slope),
        "fit_intercept_ln_at_uvw2": float(intercept),
        "r2_log": r2,
        "fixed_slope_bestnorm_rms_fraction": float(
            np.sqrt(np.mean(fixed_fraction**2))
        ),
        "fixed_slope_bestnorm_max_abs_fraction": float(
            np.max(np.abs(fixed_fraction))
        ),
        "w2_anchored_theory_rms_fraction_all6": float(
            np.sqrt(np.mean(w2_fraction**2))
        ),
        "w2_anchored_theory_max_abs_fraction_all6": float(
            np.max(np.abs(w2_fraction))
        ),
    }


def fit_all(table: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "m0_nu_relative": -1.0 / 3.0,
        "m0_lambda_relative": -7.0 / 3.0,
        "centroid_days": 4.0 / 3.0,
        "variance_days2": 8.0 / 3.0,
        "std_days": 4.0 / 3.0,
        "normalized_raw_second_days2": 8.0 / 3.0,
        "raw_m1_nu_relative": 1.0,
        "raw_m2_nu_relative": 7.0 / 3.0,
        "raw_m1_lambda_relative": -1.0,
        "raw_m2_lambda_relative": 1.0 / 3.0,
    }
    subsets = {
        "all_6_bands": np.ones(len(table), dtype=bool),
        "U_B_V_only": table["band"].isin(["U", "B", "V"]).to_numpy(),
    }
    rows = []
    for column, theory_slope in definitions.items():
        for subset, selection in subsets.items():
            rows.append(
                powerlaw_fit(table, column, theory_slope, subset, selection)
            )
    return pd.DataFrame(rows)


def fit_relative_difference(
    table: pd.DataFrame,
    column: str,
    theory_exponent: float,
) -> dict[str, float | str | int]:
    selected = table[table["lambda_over_uvw2"] > 1.0]
    x = selected["lambda_over_uvw2"].to_numpy(dtype=float)
    y = selected[column].to_numpy(dtype=float)

    fixed_basis = x**theory_exponent - 1.0
    fixed_scale = float(np.sum(y * fixed_basis) / np.sum(fixed_basis**2))
    fixed_prediction = fixed_scale * fixed_basis
    fixed_fraction = y / fixed_prediction - 1.0

    solution = least_squares(
        lambda parameters: y
        - parameters[0] * (x ** parameters[1] - 1.0),
        x0=np.array([fixed_scale, theory_exponent]),
        bounds=(np.array([0.0, 0.01]), np.array([np.inf, 10.0])),
    )
    free_scale, free_exponent = solution.x
    free_prediction = free_scale * (x**free_exponent - 1.0)
    free_fraction = y / free_prediction - 1.0

    return {
        "quantity": column,
        "n_target_bands": len(selected),
        "theory_exponent": theory_exponent,
        "fixed_exponent_scale": fixed_scale,
        "fixed_exponent_rms_fraction": float(
            np.sqrt(np.mean(fixed_fraction**2))
        ),
        "fixed_exponent_max_abs_fraction": float(
            np.max(np.abs(fixed_fraction))
        ),
        "free_exponent": float(free_exponent),
        "free_scale": float(free_scale),
        "free_rms_fraction": float(np.sqrt(np.mean(free_fraction**2))),
        "free_max_abs_fraction": float(np.max(np.abs(free_fraction))),
        "optimizer_success": bool(solution.success),
    }


def fit_relative_relations(table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            fit_relative_difference(
                table, "relative_centroid_days", 4.0 / 3.0
            ),
            fit_relative_difference(
                table, "relative_variance_days2", 8.0 / 3.0
            ),
        ]
    )


def plot_check(
    table: pd.DataFrame,
    fits: pd.DataFrame,
    output_base: Path,
    rin: float,
) -> None:
    panels = [
        ("m0_lambda_relative", r"$M_{0,\lambda}/M_{0,\lambda}(\mathrm{W2})$", -7 / 3),
        ("centroid_days", r"Centroid $\bar{\tau}$ [day]", 4 / 3),
        ("variance_days2", r"Central variance $V_\tau$ [day$^2$]", 8 / 3),
        (
            "normalized_raw_second_days2",
            r"Normalized raw second moment $\langle\tau^2\rangle$ [day$^2$]",
            8 / 3,
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.8), constrained_layout=False)
    fig.subplots_adjust(
        left=0.085,
        right=0.98,
        bottom=0.105,
        top=0.92,
        wspace=0.22,
        hspace=0.27,
    )
    x = table["lambda_over_uvw2"].to_numpy(dtype=float)
    grid = np.geomspace(x.min(), x.max(), 300)

    for ax, (column, ylabel, theory_slope) in zip(axes.ravel(), panels):
        y = table[column].to_numpy(dtype=float)
        record = fits[
            (fits["quantity"] == column)
            & (fits["subset"] == "all_6_bands")
        ].iloc[0]
        free_curve = np.exp(record["fit_intercept_ln_at_uvw2"]) * grid ** record[
            "fit_slope"
        ]
        theory_curve = y[0] * grid**theory_slope

        ax.plot(
            grid,
            theory_curve,
            color="#343A40",
            linewidth=1.6,
            linestyle="--",
            label=rf"self-similar, $p={theory_slope:.3f}$ (W2 anchored)",
        )
        ax.plot(
            grid,
            free_curve,
            color="#2468A2",
            linewidth=2.0,
            label=rf"free fit, $p={record['fit_slope']:.3f}$",
        )
        ax.scatter(
            x,
            y,
            s=48,
            facecolor="#D8A31A",
            edgecolor="#343A40",
            linewidth=0.8,
            zorder=3,
        )
        for xx, yy, band in zip(x, y, table["band"]):
            ax.annotate(
                band,
                (xx, yy),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
                color="#343A40",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.97, 2.95)
        ax.set_xticks([1.0, 1.25, 1.5, 2.0, 2.5])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2g"))
        ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
        ax.set_xlabel(r"$\lambda/\lambda_{\mathrm{W2}}$")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", color="#D9DEE3", linewidth=0.65)
        ax.legend(frameon=False, fontsize=8.5, loc="best")

    fig.suptitle(
        rf"Mrk 817 finite-disk moment scaling ($R_{{\rm in}}={rin:g}$ light-day)",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.018,
        (
            r"Finite-difference $B_\nu$ model: $b=3/4$, "
            rf"$R_{{\rm in}}={rin:g}$ light-day, $R_{{\rm out}}=30$ light-day. "
            r"Only $M_0$ is converted to the $f_\lambda$ convention."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4B535B",
    )
    fig.savefig(output_base.with_suffix(".png"), dpi=250, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_relative_check(
    table: pd.DataFrame,
    fits: pd.DataFrame,
    relative_fits: pd.DataFrame,
    relative_base: Path,
    rin: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.18, top=0.84, wspace=0.29)
    x = table["lambda_over_uvw2"].to_numpy(dtype=float)
    grid = np.linspace(1.0, x.max(), 400)
    colors = {"data": "#D8A31A", "free": "#2468A2", "theory": "#343A40"}

    area = table["m0_lambda_relative"].to_numpy(dtype=float)
    area_record = fits[
        (fits["quantity"] == "m0_lambda_relative")
        & (fits["subset"] == "all_6_bands")
    ].iloc[0]
    axes[0].plot(
        grid,
        grid ** (-7.0 / 3.0),
        linestyle="--",
        linewidth=1.6,
        color=colors["theory"],
        label=r"theory $r^{-7/3}$",
    )
    axes[0].plot(
        grid,
        np.exp(area_record["fit_intercept_ln_at_uvw2"])
        * grid ** area_record["fit_slope"],
        linewidth=2.0,
        color=colors["free"],
        label=rf"free $r^p$, $p={area_record['fit_slope']:.3f}$",
    )
    axes[0].scatter(
        x,
        area,
        s=48,
        facecolor=colors["data"],
        edgecolor=colors["theory"],
        linewidth=0.8,
        zorder=3,
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Relative area $A_\lambda$")

    difference_panels = [
        ("relative_centroid_days", r"Relative centroid $\Delta\bar{\tau}$ [day]"),
        ("relative_variance_days2", r"Relative variance $\Delta V_\tau$ [day$^2$]"),
    ]
    for ax, (column, ylabel) in zip(axes[1:], difference_panels):
        record = relative_fits[relative_fits["quantity"] == column].iloc[0]
        theory_curve = record["fixed_exponent_scale"] * (
            grid ** record["theory_exponent"] - 1.0
        )
        free_curve = record["free_scale"] * (
            grid ** record["free_exponent"] - 1.0
        )
        ax.plot(
            grid,
            theory_curve,
            linestyle="--",
            linewidth=1.6,
            color=colors["theory"],
            label=rf"theory form, $p={record['theory_exponent']:.3f}$",
        )
        ax.plot(
            grid,
            free_curve,
            linewidth=2.0,
            color=colors["free"],
            label=rf"free form, $p={record['free_exponent']:.3f}$",
        )
        ax.scatter(
            x,
            table[column],
            s=48,
            facecolor=colors["data"],
            edgecolor=colors["theory"],
            linewidth=0.8,
            zorder=3,
        )
        ax.set_ylabel(ylabel)

    for ax in axes:
        for xx, band in zip(x, table["band"]):
            yy = (
                area[list(table["band"]).index(band)]
                if ax is axes[0]
                else table[
                    "relative_centroid_days"
                    if ax is axes[1]
                    else "relative_variance_days2"
                ].iloc[list(table["band"]).index(band)]
            )
            ax.annotate(
                band,
                (xx, yy),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
                color=colors["theory"],
            )
        ax.set_xlim(0.97, 2.95)
        ax.set_xticks([1.0, 1.5, 2.0, 2.5])
        ax.set_xlabel(r"$r=\lambda/\lambda_{\mathrm{W2}}$")
        ax.grid(True, which="both", color="#D9DEE3", linewidth=0.65)
        ax.legend(frameon=False, fontsize=8.3, loc="best")

    fig.suptitle(
        rf"Mrk 817 W2-relative moment relations ($R_{{\rm in}}={rin:g}$ light-day)",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.035,
        (
            r"$A_\lambda=M_{0,\lambda}/M_{0,\lambda,\mathrm{W2}}$; "
            r"relative centroid and variance are cumulant differences."
        ),
        ha="center",
        fontsize=9,
        color="#4B535B",
    )
    fig.savefig(relative_base.with_suffix(".png"), dpi=250, bbox_inches="tight")
    fig.savefig(relative_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.input)
    output_base = args.output
    relative_base = args.relative_output
    if relative_base is None:
        relative_name = output_base.name.replace(
            "moment_powerlaw_check", "relative_moment_check"
        )
        if relative_name == output_base.name:
            relative_name = output_base.name + "_relative"
        relative_base = output_base.with_name(relative_name)
    rin = (
        float(source["Rin_light_days"].iloc[0])
        if "Rin_light_days" in source.columns
        else 0.1
    )
    table = build_band_table(source)
    fits = fit_all(table)
    relative_fits = fit_relative_relations(table)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_base.with_name(output_base.name + "_band_values.csv"), index=False)
    fits.to_csv(output_base.with_name(output_base.name + "_fits.csv"), index=False)
    relative_fits.to_csv(
        output_base.with_name(output_base.name + "_relative_fits.csv"),
        index=False,
    )
    plot_check(table, fits, output_base, rin)
    plot_relative_check(table, fits, relative_fits, relative_base, rin)

    main_rows = fits[fits["subset"] == "all_6_bands"]
    for quantity in [
        "m0_lambda_relative",
        "centroid_days",
        "variance_days2",
        "normalized_raw_second_days2",
        "raw_m2_lambda_relative",
    ]:
        row = main_rows[main_rows["quantity"] == quantity].iloc[0]
        print(
            f"{quantity:36s} p={row['fit_slope']:+.6f} "
            f"theory={row['theory_slope']:+.6f} "
            f"R2={row['r2_log']:.8f}"
        )


if __name__ == "__main__":
    main()
