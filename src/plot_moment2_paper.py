#!/usr/bin/env python3
# 三个固定示例源的二阶矩图；数据来自已保存的 tf0 二阶矩点和拟合摘要。
# 此处的 gamma=8/3、4、free 描述波长标度，与传递函数族名称要区分。
"""Build the three-source short-component second-moment paper figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCES = ("Mrk817", "Mrk142", "Mrk335")
MODELS = {
    "8/3": {"label": r"$\gamma=8/3$", "ls": "-", "color": "#0072B2"},
    "4": {"label": r"$\gamma=4$", "ls": "--", "color": "#D55E00"},
    "free": {"label": r"$\gamma$ free", "ls": "-.", "color": "#009E73"},
}


# 相对二阶矩幂律；lambda 和 lambda0 必须使用同一波长单位。
def moment2_model(wavelength, moment20, gamma, lambda0):
    wavelength = np.asarray(wavelength, dtype=float)
    return moment20 * ((wavelength / float(lambda0)) ** gamma - 1.0)


# 当前实现固定访问 gamma2 目录，不自动跟随 main.py 的 --stage。
def source_paths(workspace: Path, source: str) -> tuple[Path, Path]:
    root = (
        workspace
        / source
        / "runs"
        / "mica"
        / "gamma2"
        / "2comp"
        / "result"
        / "comp0"
        / "moment2"
    )
    return (
        root / f"{source}_lambda_moment2_points.csv",
        root / f"{source}_lambda_moment2_fit_summary.csv",
    )


# 把已有点、误差和三类拟合曲线绘出，不重采样或重新拟合。
def plot(workspace: Path, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.75))
    model_handles = []
    model_labels = []

    for ax, source in zip(axes, SOURCES):
        points_path, summary_path = source_paths(workspace, source)
        points = pd.read_csv(points_path)
        summary = pd.read_csv(summary_path)
        points = points[points["component"].eq("tf0")]
        response = points[~points["is_ref"].astype(bool)]
        reference = points[points["is_ref"].astype(bool)]
        rows = summary[summary["component"].eq("tf0")]
        lambda0 = float(rows["lambda0"].iloc[0])

        ax.errorbar(
            response["wavelength"],
            response["moment2"],
            yerr=np.vstack(
                [response["moment2_err_low"], response["moment2_err_high"]]
            ),
            fmt="o",
            ms=4.8,
            color="black",
            markerfacecolor="white",
            markeredgewidth=1.0,
            ecolor="0.35",
            elinewidth=1.0,
            capsize=2.5,
            zorder=4,
            label="short component",
        )
        ax.scatter(
            reference["wavelength"],
            reference["moment2"],
            marker="s",
            s=21,
            color="black",
            zorder=5,
        )

        grid = np.linspace(points["wavelength"].min() * 0.94, points["wavelength"].max() * 1.06, 600)
        for row in rows.itertuples(index=False):
            key = str(row.fit_model)
            style = MODELS[key]
            line, = ax.plot(
                grid,
                moment2_model(grid, row.moment20, row.gamma, lambda0),
                color=style["color"],
                ls=style["ls"],
                lw=1.45,
                label=style["label"],
                zorder=2,
            )
            if ax is axes[0]:
                model_handles.append(line)
                model_labels.append(style["label"])

        positive = response.loc[response["moment2"] > 0.0, "moment2"].to_numpy(float)
        if len(positive):
            ax.set_yscale("symlog", linthresh=max(1.0e-3, 0.5 * float(np.min(positive))))
        ax.axhline(0.0, color="0.55", lw=0.7, ls=":", zorder=1)
        ax.axvline(lambda0, color="0.55", lw=0.7, ls=":", zorder=1)
        ax.set_title(source.replace("Mrk", "Mrk "), fontsize=11)
        ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.minorticks_on()
        ax.grid(which="major", lw=0.5, alpha=0.12)

    axes[0].set_ylabel(r"Relative second central moment (day$^2$)")
    fig.legend(
        model_handles,
        model_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.18, top=0.78, wspace=0.28)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=350)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "DRMH0" / "fig" / "second_moment_short_three_sources.pdf",
    )
    args = parser.parse_args()
    plot(args.workspace.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
