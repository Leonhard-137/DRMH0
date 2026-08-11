#!/usr/bin/env python3
"""Plot Delta_F SED from fflux_comp0 components CSV (no extinction correction)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_csv(comp0_dir: Path) -> Path:
    candidates = sorted((comp0_dir / "analysis").glob("*_components.csv"))
    if not candidates:
        raise FileNotFoundError(f"No *_components.csv found in {comp0_dir / 'analysis'}")
    return candidates[0]


def plot_sed(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df = df.sort_values("Wave_Rest")

    wave = df["Wave_Rest"].to_numpy()
    dF   = df["Delta_F_p50"].to_numpy()
    dF_lo = df["Delta_F_p50"].to_numpy() - df["Delta_F_p16"].to_numpy()
    dF_hi = df["Delta_F_p84"].to_numpy() - df["Delta_F_p50"].to_numpy()

    # lambda * Delta_F_lambda (arbitrary units, proportional to nu * F_nu)
    ldF    = wave * dF
    ldF_lo = wave * dF_lo
    ldF_hi = wave * dF_hi

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- left: Delta_F vs wavelength ---
    ax = axes[0]
    ax.errorbar(wave, dF, yerr=[dF_lo, dF_hi],
                fmt="o", color="steelblue", capsize=4, ms=5)
    for i, row in df.iterrows():
        ax.annotate(row["Filter"], (row["Wave_Rest"], row["Delta_F_p50"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel(r"$\lambda_{\rm rest}$ (Å)")
    ax.set_ylabel(r"$\Delta F_\lambda$ (arb.)")
    ax.set_title("Variable component (comp0)")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # --- right: lambda * Delta_F vs wavelength (nu * F_nu shape) ---
    ax = axes[1]
    ax.errorbar(wave, ldF, yerr=[ldF_lo, ldF_hi],
                fmt="o", color="tomato", capsize=4, ms=5)
    for i, row in df.iterrows():
        ax.annotate(row["Filter"], (row["Wave_Rest"], row["Wave_Rest"] * row["Delta_F_p50"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)

    # thin-disk reference slope: lambda * F_lambda ~ lambda^{-1/3}
    w_ref = np.logspace(np.log10(wave.min()), np.log10(wave.max()), 100)
    norm = ldF[len(ldF) // 2] / (wave[len(wave) // 2] ** (-1/3))
    ax.plot(w_ref, norm * w_ref ** (-1/3), "k--", lw=1, label=r"$\lambda^{-1/3}$ (thin disk)")
    ax.set_xlabel(r"$\lambda_{\rm rest}$ (Å)")
    ax.set_ylabel(r"$\lambda \Delta F_\lambda$ (arb.)")
    ax.set_title(r"$\nu F_\nu$ shape")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    source = csv_path.stem.split("_")[0]
    fig.suptitle(f"{source}  comp0  —  no extinction correction", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path, help="e.g. Mrk142")
    parser.add_argument("--section", default="fflux_comp0")
    args = parser.parse_args()

    src = Path(args.source_dir).expanduser().resolve()
    comp0_dir = src / args.section
    csv_path = find_csv(comp0_dir)
    out_path = comp0_dir / "analysis" / (csv_path.stem + "_sed.png")
    plot_sed(csv_path, out_path)


if __name__ == "__main__":
    main()
