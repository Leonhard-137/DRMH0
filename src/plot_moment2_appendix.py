#!/usr/bin/env python3
# 保存的二阶矩拟合结果的附录绘图器，不执行二阶矩参数推断。
# SOURCES 和 gamma2 路径在本文件中固定；比较新分支时需先核对输入路径。
"""Build compact appendix pages for short-component second-moment fits."""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCES = (
    ("F9_a", "Fairall 9--A"),
    ("MCG+08-11-011_b", "MCG +08-11-011--B"),
    ("Mrk142", "Mrk 142"),
    ("Mrk279_a", "Mrk 279--A"),
    ("Mrk279_b", "Mrk 279--B"),
    ("Mrk335", "Mrk 335"),
    ("Mrk509", "Mrk 509"),
    ("Mrk817", "Mrk 817"),
    ("NGC2617_a", "NGC 2617--A"),
    ("NGC2617", "NGC 2617--B"),
    ("NGC4151_a", "NGC 4151--A"),
    ("NGC4593", "NGC 4593"),
    ("NGC5548", "NGC 5548"),
    ("NGC6814", "NGC 6814"),
)

MODELS = {
    "8/3": {"label": r"$\gamma=8/3$", "ls": "-", "color": "#0072B2"},
    "4": {"label": r"$\gamma=4$", "ls": "--", "color": "#D55E00"},
    "free": {"label": r"$\gamma$ free", "ls": "-.", "color": "#009E73"},
}


# 绘制 m2(lambda)=m20*((lambda/lambda0)**gamma-1)；m20 的单位随输入二阶矩为 day^2。
# 这里的 gamma 是波长幂指数，不是 MICA Gamma 核的形状参数。
def model(wavelength, moment20, gamma, lambda0):
    wavelength = np.asarray(wavelength, dtype=float)
    return moment20 * ((wavelength / float(lambda0)) ** gamma - 1.0)


# 固定读取 gamma2/2comp/result/comp0/moment2 的已有表。
def paths(workspace: Path, source: str) -> tuple[Path, Path]:
    root = workspace / source / "runs/mica/gamma2/2comp/result/comp0/moment2"
    return (
        root / f"{source}_lambda_moment2_points.csv",
        root / f"{source}_lambda_moment2_fit_summary.csv",
    )


# 读取 tf0、参考点和已拟合参数；参考点是相对二阶矩的零点。
def draw_panel(ax, workspace: Path, source: str, title: str):
    points_path, summary_path = paths(workspace, source)
    points = pd.read_csv(points_path)
    summary = pd.read_csv(summary_path)
    points = points[points["component"].eq("tf0")]
    response = points[~points["is_ref"].astype(bool)]
    reference = points[points["is_ref"].astype(bool)]
    rows = summary[summary["component"].eq("tf0")]
    lambda0 = float(rows["lambda0"].iloc[0])

    ax.errorbar(
        response["wavelength"], response["moment2"],
        yerr=np.vstack([response["moment2_err_low"], response["moment2_err_high"]]),
        fmt="o", ms=4.2, color="black", markerfacecolor="white",
        markeredgewidth=0.9, ecolor="0.35", elinewidth=0.9, capsize=2.2,
        zorder=4,
    )
    ax.scatter(reference["wavelength"], reference["moment2"], marker="s", s=18,
               color="black", zorder=5)

    grid = np.linspace(points["wavelength"].min() * 0.94,
                       points["wavelength"].max() * 1.06, 600)
    handles = []
    for row in rows.itertuples(index=False):
        style = MODELS[str(row.fit_model)]
        line, = ax.plot(grid, model(grid, row.moment20, row.gamma, lambda0),
                       color=style["color"], ls=style["ls"], lw=1.35,
                       label=style["label"], zorder=2)
        handles.append(line)

    positive = response.loc[response["moment2"] > 0.0, "moment2"].to_numpy(float)
    if len(positive):
        ax.set_yscale("symlog", linthresh=max(1.0e-3, 0.5 * float(np.min(positive))))
    ax.axhline(0.0, color="0.55", lw=0.65, ls=":", zorder=1)
    ax.axvline(lambda0, color="0.55", lw=0.65, ls=":", zorder=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
    ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=8)
    ax.minorticks_on()
    ax.grid(which="major", lw=0.45, alpha=0.12)
    return handles


# 只组织面板布局和公共图例，不参与拟合。
def draw_page(workspace: Path, targets, output: Path):
    ncols = min(3, len(targets))
    nrows = ceil(len(targets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.55 * ncols, 3.15 * nrows),
                             squeeze=False)
    handles = None
    for ax, (source, title) in zip(axes.flat, targets):
        panel_handles = draw_panel(ax, workspace, source, title)
        handles = handles or panel_handles
    for ax in axes.flat[len(targets):]:
        ax.remove()
    for ax in axes[:, 0]:
        if ax in fig.axes:
            ax.set_ylabel(r"Relative second central moment (day$^2$)")
    fig.legend(handles, [MODELS[key]["label"] for key in ("8/3", "4", "free")],
               loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3,
               frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.105, top=0.89,
                        hspace=0.40, wspace=0.27)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main():
    workspace = Path(__file__).resolve().parents[1]
    output_dir = workspace / "DRMH0/fig/Appendix/moment2"
    for page, start in enumerate(range(0, len(SOURCES), 6), start=1):
        draw_page(workspace, SOURCES[start:start + 6],
                  output_dir / f"short_moment2_page{page}.pdf")


if __name__ == "__main__":
    main()
