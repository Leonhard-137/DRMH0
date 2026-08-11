#!/usr/bin/env python3
"""Plot two shifted k=2 Gamma responses and their exact relative kernel."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "gamma_test_disk_outputs"


def gamma_k2(t: np.ndarray, start: float, theta: float) -> np.ndarray:
    """Unit-area shifted Gamma density with shape k=2."""
    u = t - start
    response = np.zeros_like(t)
    mask = u >= 0.0
    response[mask] = u[mask] * np.exp(-u[mask] / theta) / theta**2
    return response


def relative_continuous_k2(
    t: np.ndarray,
    delta_start: float,
    theta_a: float,
    theta_b: float,
) -> tuple[np.ndarray, float]:
    """Continuous part and delta weight of psi_B / psi_A for k_A=k_B=2."""
    u = t - delta_start
    mask = u >= 0.0
    continuous = np.zeros_like(t)
    ratio = theta_a / theta_b
    continuous[mask] = (
        2.0
        * ratio
        * (1.0 - ratio)
        * np.exp(-u[mask] / theta_b)
        / theta_b
        + (1.0 - ratio) ** 2
        * u[mask]
        * np.exp(-u[mask] / theta_b)
        / theta_b**2
    )
    return continuous, ratio**2


def main() -> None:
    t = np.linspace(0.0, 9.0, 4501)
    start_a = 0.5
    start_b = 1.0
    delta_start = start_b - start_a
    theta_b = 1.0
    cases = [
        (0.5, r"$\theta_A < \theta_B$", "positive mixture"),
        (1.0, r"$\theta_A = \theta_B$", "pure shift"),
        (1.5, r"$\theta_A > \theta_B$", "signed inverse kernel"),
    ]

    colors = {
        "a": "#2457A6",
        "b": "#D17A00",
        "relative": "#6D4C7D",
        "delta": "#222222",
        "zero": "#777777",
    }

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(13.2, 8.2),
        sharex=True,
        gridspec_kw={"hspace": 0.16, "wspace": 0.16},
    )

    for column, (theta_a, relation, interpretation) in enumerate(cases):
        psi_a = gamma_k2(t, start_a, theta_a)
        psi_b = gamma_k2(t, start_b, theta_b)
        relative, delta_weight = relative_continuous_k2(
            t, delta_start, theta_a, theta_b
        )

        axes[0, column].plot(t, psi_a, color=colors["a"], lw=2.2)
        axes[1, column].plot(t, psi_b, color=colors["b"], lw=2.2)
        axes[2, column].plot(t, relative, color=colors["relative"], lw=2.2)
        axes[2, column].axhline(0.0, color=colors["zero"], lw=0.8, ls="--")

        # A Dirac delta has an area but no finite height. Draw an arrow only as
        # a location marker, and label its exact coefficient separately.
        axes[2, column].annotate(
            "",
            xy=(delta_start, 0.82),
            xytext=(delta_start, 0.0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": colors["delta"],
                "lw": 1.8,
            },
        )
        axes[2, column].text(
            delta_start + 0.12,
            0.72,
            rf"$\delta$ weight $={delta_weight:.2f}$",
            color=colors["delta"],
            fontsize=9.5,
            va="center",
        )

        axes[0, column].set_title(
            relation + "\n" + interpretation,
            fontsize=12,
            pad=9,
        )
        axes[0, column].text(
            0.98,
            0.88,
            rf"$\theta_A={theta_a:.1f}$ d",
            transform=axes[0, column].transAxes,
            ha="right",
            va="top",
            color=colors["a"],
            fontsize=10,
        )
        axes[1, column].text(
            0.98,
            0.88,
            rf"$\theta_B={theta_b:.1f}$ d",
            transform=axes[1, column].transAxes,
            ha="right",
            va="top",
            color=colors["b"],
            fontsize=10,
        )

    row_labels = [
        r"$\psi_A(t)$  [day$^{-1}$]",
        r"$\psi_B(t)$  [day$^{-1}$]",
        r"$h_{B\leftarrow A}(t)$ continuous part  [day$^{-1}$]",
    ]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=11)

    for ax in axes.flat:
        ax.set_xlim(0.0, 9.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#D9D9D9", lw=0.7, alpha=0.75)
        ax.tick_params(direction="out", length=4, width=0.8)

    for ax in axes[0, :]:
        ax.set_ylim(-0.03, 0.82)
    for ax in axes[1, :]:
        ax.set_ylim(-0.02, 0.43)
    for ax in axes[2, :]:
        ax.set_ylim(-1.65, 0.95)
        ax.set_xlabel("Delay [day]", fontsize=11)

    fig.suptitle(
        r"Relative transfer kernel for two shifted $k=2$ Gamma responses",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        r"$t_A=0.5$ d, $t_B=1.0$ d, unit areas. "
        r"Arrows mark the Dirac $\delta(t-[t_B-t_A])$ term; arrow height is schematic.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.095)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "gamma_k2_relative_transfer_three_cases.png"
    pdf_path = OUTPUT_DIR / "gamma_k2_relative_transfer_three_cases.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
