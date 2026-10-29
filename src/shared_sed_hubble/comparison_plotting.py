# 四情景图只复用已保存密度和冻结距离；不拟合、不 KDE 平滑，也不重归一化保存的密度。
# 纵轴误差条来自距离分位数的单调变换，横轴来自外部红移分布，不是条件于 H0 的红移后验。
"""Comparison figures for the frozen-distance, four-case Hubble analysis.

Only completed comparison outputs are read.  Observed luminosity distances
are kept fixed, redshift intervals summarize the external velocity/redshift
model, and extra distance scatter is never folded into observational bars.
The existing project style takes precedence over generic plotting defaults;
the requested gray capless errors, three minor ticks, and 300 dpi exports are
local overrides.  No posterior smoothing or inference is performed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, trapezoid

from .h0 import distance_scale
from .style import (
    INK, BLUE, ORANGE, CYAN, HOLLOW_SOURCES, apply_style, apply_frame,
    REDSHIFT_TICKS, SOURCE_LABEL_OFFSETS, DEFAULT_LABEL_OFFSET, source_label,
)

CASES = ("A", "B", "C", "D")
# Four equal-status analysis cases; these colors do not encode reference H0s.
CASE_COLORS = {"A": INK, "B": BLUE, "C": ORANGE, "D": CYAN}
CASE_LABELS = {
    "A": r"A: no PV, $f_D=0$",
    "B": r"B: no PV, free $f_D$",
    "C": r"C: PV, $f_D=0$",
    "D": r"D: PV, free $f_D$",
}
ERROR_GRAY = "#B8B8B8"
SAVE_DPI = 300
H0_LABEL = r"$H_0$ (km s$^{-1}$ Mpc$^{-1}$)"


# 统一刻度和边框；分类轴不使用数值次刻度。
def _frame(ax, *, categorical_y=False):
    apply_frame(ax)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(NullLocator() if categorical_y else AutoMinorLocator(4))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


# 保存 PNG/PDF，并返回路径及说明文本。
def _save(fig, folder: Path, stem: str, caption: str) -> dict:
    paths = {extension: folder / f"{stem}.{extension}" for extension in ("pdf", "png")}
    for path in paths.values():
        fig.savefig(path, dpi=SAVE_DPI, facecolor="white", bbox_inches="tight", pad_inches=.08)
    plt.close(fig)
    return {"name": stem, **{key: str(path) for key, path in paths.items()}, "caption": caption}


# 检查绘图所需列，避免静默画错变量。
def _require(frame: pd.DataFrame, columns: tuple[str, ...], name: str):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} lacks columns: {sorted(missing)}")


# 读取有限、有序的中央分位数。
def _quantiles(row, prefix: str) -> tuple[float, float, float]:
    q = tuple(float(row[f"{prefix}_{suffix}"]) for suffix in ("q16", "q50", "q84"))
    if not np.all(np.isfinite(q)) or not 0 <= q[0] <= q[1] <= q[2]:
        raise ValueError(f"Invalid {prefix} quantiles: {q}")
    return q


# 校验已存网格/密度后原样返回，显示不修改推断产品。
def _posterior(path: Path, column: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    _require(frame, (column, "pdf"), str(path))
    grid, pdf = frame[column].to_numpy(float), frame.pdf.to_numpy(float)
    if (len(grid) < 2 or not np.all(np.isfinite(grid)) or
            not np.all(np.diff(grid) > 0) or not np.all(np.isfinite(pdf) & (pdf >= 0)) or
            not np.any(pdf > 0)):
        raise ValueError(f"Invalid saved posterior: {path}")
    # Draw the saved density verbatim, without a new KDE or renormalization.
    return grid, pdf


# 中央 99.8% 仅决定可视范围，不截断或重新归一化数据。
def _display_window(grid, pdf):
    """Central 99.8% display bounds only; never alter the saved density."""
    cdf = cumulative_trapezoid(pdf, grid, initial=0.)
    if cdf[-1] <= 0:
        raise ValueError("Posterior has zero integrated mass")
    return tuple(np.interp([.001, .999], cdf / cdf[-1], grid))


# 合并各情景可视范围并留边距。
def _padded_window(windows):
    low = min(window[0] for window in windows)
    high = max(window[1] for window in windows)
    pad = .08 * max(high - low, 1e-6)
    return max(0., low - pad), high + pad


# 核对 A–D 情景标签、相同源集和红移区间；绘图比拟合额外需要 D_q16/q50/q84 列。
def _load(folder: Path):
    common = pd.read_csv(folder / "common_inputs.csv")
    comparison = pd.read_csv(folder / "comparison.csv")
    run = json.loads((folder / "run_summary.json").read_text(encoding="utf-8"))
    _require(common, ("source", "D_mean", "D_std", "D_q16", "D_q50", "D_q84",
                      "z_saved", "sigma_v_kms"), "common_inputs.csv")
    _require(comparison, ("case", "pv_enabled", "scatter_mode", "h0_q16", "h0_q50", "h0_q84"),
             "comparison.csv")
    common["source"] = common.source.astype(str)
    if common.empty or common.source.duplicated().any():
        raise ValueError("Expected one fixed distance per distinct source")
    obs, err = common.D_mean.to_numpy(float), common.D_std.to_numpy(float)
    if not np.all(np.isfinite(obs) & (obs > 0) & np.isfinite(err) & (err >= 0)):
        raise ValueError("Saved luminosity distance means/errors must be finite and valid")
    comparison = comparison.loc[comparison.case.isin(CASES)].set_index("case")
    if len(comparison) != 4 or comparison.index.duplicated().any() or set(comparison.index) != set(CASES):
        raise ValueError("Comparison figures require exactly cases A, B, C, and D")
    om0 = float(run["settings"]["om0"])
    if not np.isfinite(om0) or not 0 <= om0 <= 1:
        raise ValueError("Invalid shared flat-LCDM matter density")
    fits = {}
    for case in CASES:
        row = comparison.loc[case]
        is_pv = str(row.pv_enabled).strip().lower() in {"true", "1", "1.0"}
        is_free = str(row.scatter_mode) != "fixed_zero"
        if is_pv != (case in {"C", "D"}) or is_free != (case in {"B", "D"}):
            raise ValueError(f"Case {case} configuration does not match its plotted label")
        if _quantiles(row, "h0")[0] <= 0:
            raise ValueError("H0 quantiles must be positive")
        fit = pd.read_csv(folder / "cases" / case / "fit_quality.csv")
        _require(fit, ("source", "z_q16", "z_q50", "z_q84"), f"{case}/fit_quality.csv")
        fit["source"] = fit.source.astype(str)
        if fit.source.duplicated().any() or set(fit.source) != set(common.source):
            raise ValueError(f"Case {case} source list differs from the fixed distance input")
        fit = fit.set_index("source").loc[common.source].reset_index()
        z = fit[["z_q16", "z_q50", "z_q84"]].to_numpy(float)
        if (not np.all(np.isfinite(z)) or np.any(z[:, 0] <= 0) or
                np.any(np.diff(z, axis=1) < 0)):
            raise ValueError(f"Invalid external redshift intervals for {case}")
        fits[case] = fit
    return common, comparison, fits, om0


# 从实际输入 sigma_v 生成图注，不把固定分析假设写成观测测量。
def _velocity_caption(common):
    """Describe the residual dispersions actually used by the saved run."""
    values = common.sigma_v_kms.to_numpy(float)
    if not np.all(np.isfinite(values) & (values >= 0)):
        raise ValueError("Residual velocity dispersions must be finite and nonnegative")
    distinct = np.unique(values)
    if len(distinct) == 1:
        return f"a residual velocity dispersion of sigma_v={distinct[0]:g} km/s in C/D. "
    return ("source-dependent residual velocity dispersions spanning "
            f"sigma_v={distinct.min():g}–{distinct.max():g} km/s in C/D. ")


# 逐个将 D 分位点变为 μ；不改变拟合中使用的线性距离 mean/std。
def _distance_modulus_quantiles(common):
    """Use the legacy plot's monotone transform of saved distance quantiles.

    This is a display transformation only. The fitted distance likelihood
    continues to use the saved linear-distance mean and standard deviation.
    """
    distances = common[["D_q16", "D_q50", "D_q84"]].to_numpy(float)
    if (not np.all(np.isfinite(distances) & (distances > 0)) or
            np.any(np.diff(distances, axis=1) < 0)):
        raise ValueError("Saved distance quantiles must be finite, positive and ordered")
    return 5 * np.log10(distances) + 25


# 四情景共用坐标范围；纵向残差是展示量，不当作边缘化后的 χ²；另存实际画出的坐标。
def _hubble(folder, common, comparison, fits, om0):
    # Each case inherits the original modulus/redshift + residual layout.
    moduli = _distance_modulus_quantiles(common)
    obs = moduli[:, 1]
    yerr = np.vstack((obs - moduli[:, 0], moduli[:, 2] - obs))
    zmin = min(float(fits[case].z_q16.min()) for case in CASES) * .80
    zmax = max(float(fits[case].z_q84.max()) for case in CASES) * 1.18
    if zmin <= 0:
        raise ValueError("The modulus diagram requires positive redshift intervals")
    zz = np.geomspace(zmin, zmax, 400)
    scale = distance_scale(zz, om0)
    models = {
        case: 5 * np.log10(distance_scale(fits[case].z_q50.to_numpy(float), om0)
                          / float(comparison.loc[case, "h0_q50"])) + 25
        for case in CASES
    }
    residual_low = min(float(np.min(moduli[:, 0] - models[case])) for case in CASES)
    residual_high = max(float(np.max(moduli[:, 2] - models[case])) for case in CASES)
    ylim = (float(moduli[:, 0].min()) - .55, float(moduli[:, 2].max()) + .65)
    rlim = (min(residual_low, 0.) - .3, max(residual_high, 0.) + .3)

    fig = plt.figure(figsize=(13.5, 10.8))
    outer = fig.add_gridspec(2, 2, left=.07, right=.985, bottom=.065, top=.96,
                            wspace=.14, hspace=.30)
    axes = []
    for index, case in enumerate(CASES):
        inner = outer[index // 2, index % 2].subgridspec(
            2, 1, height_ratios=[3.05, 1.10], hspace=.055)
        ax = fig.add_subplot(inner[0])
        res = fig.add_subplot(inner[1], sharex=ax)
        axes.append((ax, res))
        fit = fits[case]
        hlow, hmed, hhigh = _quantiles(comparison.loc[case], "h0")
        color = CASE_COLORS[case]
        curve = 5 * np.log10(scale / hmed) + 25
        lower = 5 * np.log10(scale / hhigh) + 25
        upper = 5 * np.log10(scale / hlow) + 25
        ax.fill_between(zz, lower, upper, color=ERROR_GRAY, alpha=.32, linewidth=0, zorder=0)
        ax.plot(zz, curve, color=color, linewidth=1.7, zorder=1)
        res.fill_between(zz, lower - curve, upper - curve,
                         color=ERROR_GRAY, alpha=.32, linewidth=0, zorder=0)
        res.axhline(0., color=color, linewidth=1.4, zorder=1)

        z = fit.z_q50.to_numpy(float)
        xerr = np.vstack((z - fit.z_q16.to_numpy(float), fit.z_q84.to_numpy(float) - z))
        residual = obs - models[case]
        for panel, y in ((ax, obs), (res, residual)):
            panel.errorbar(z, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=ERROR_GRAY,
                           elinewidth=.70, capsize=0, zorder=2)
            for i, source in enumerate(common.source):
                panel.plot(z[i], y[i], "o", ms=4.8, mec=INK, mew=.85,
                           mfc="white" if source in HOLLOW_SOURCES else INK, zorder=3)
            panel.set_xscale("log")
            panel.set_xlim(zmin, zmax)
            panel.xaxis.set_major_locator(FixedLocator(REDSHIFT_TICKS))
            panel.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
            panel.xaxis.set_minor_locator(LogLocator(base=10., subs=np.arange(2, 10)))
            panel.xaxis.set_minor_formatter(NullFormatter())
            panel.yaxis.set_minor_locator(AutoMinorLocator(4))
            apply_frame(panel)
        for i, source in enumerate(common.source):
            dx, dy = SOURCE_LABEL_OFFSETS.get(source, DEFAULT_LABEL_OFFSET)
            # The free-scatter median relation is higher than the old f_D=0
            # relation; move these two labels clear of its curve and band.
            if case in {"B", "D"} and source == "Mrk817_a":
                dx, dy = -7, 20
            elif case in {"B", "D"} and source == "Mrk509":
                dx, dy = 7, 16
            ax.annotate(source_label(source), (z[i], obs[i]), xytext=(dx, dy),
                        textcoords="offset points", ha="right" if dx < 0 else "left",
                        fontsize=9.4, color=INK, zorder=5)
        ax.set_ylim(*ylim)
        res.set_ylim(*rlim)
        ax.set_title(CASE_LABELS[case], fontsize=13, pad=8)
        ax.tick_params(labelbottom=False)
        res.set_xlabel(r"Corrected redshift $z_{\rm cos}$" if case in {"C", "D"}
                       else r"CMB-frame redshift $z_{\rm CMB}$", fontsize=12)
        if index % 2 == 0:
            ax.set_ylabel(r"Distance modulus $\mu$ (mag)", fontsize=13)
            res.set_ylabel(r"$\Delta\mu$ (mag)", fontsize=12)
        else:
            ax.tick_params(labelleft=False)
            res.tick_params(labelleft=False)

        # Save exactly the plotted coordinates for inspection, without
        # replacing inference or source-distance tables.
        coordinates = pd.DataFrame({
            "source": common.source, "z_q16": fit.z_q16, "z_q50": fit.z_q50,
            "z_q84": fit.z_q84, "mu_q16": moduli[:, 0], "mu_q50": obs,
            "mu_q84": moduli[:, 2], "mu_model_at_h0_median": models[case],
            "delta_mu": residual,
        })
        coordinates.to_csv(folder / "cases" / case / "hubble_modulus_plot_data.csv", index=False)

    axes[0][0].legend(handles=[
        Line2D([], [], color=CASE_COLORS["A"], lw=1.7, label=r"Median $H_0$ relation"),
        Patch(facecolor=ERROR_GRAY, alpha=.32, label=r"68% $H_0$ interval"),
    ], frameon=False, fontsize=9.5, loc="lower right", handlelength=2.)
    caption = (
        "Distance-modulus versus logarithmic-redshift diagrams, following the original "
        "shared-SED Hubble plot. All four cases share axis limits. Points and asymmetric vertical "
        "bars are the saved luminosity-distance posterior q50 and q16/q84 transformed separately "
        "by mu=5*log10(D_L/Mpc)+25; no Gaussian error propagation or added f_D scatter is used "
        "for these bars. The inference itself still uses the unchanged linear-distance mean/std. "
        "Each lower panel shows delta_mu=mu_q50-mu_model(z_q50,H0_median), with the same "
        "vertical quantile bars; it is a display residual, not a marginalized chi-square statistic. "
        "Horizontal bars show external redshift q16/q84: CMB uncertainty in A/B and the "
        "flow-corrected distribution including " + _velocity_caption(common) +
        "These are not redshift posteriors conditioned on H0. The residual panels do not fold "
        "horizontal velocity uncertainty into an additional vertical error. Each model curve "
        "uses the same flat-LCDM distance_scale and its case's marginal H0 median. Gray bands "
        "show only the H0 68% interval, not predictive scatter. Source labels and the hollow "
        "Mrk 509 ESS-warning marker follow the original plotting conventions. "
        "Exact displayed coordinates are saved as cases/<case>/hubble_modulus_plot_data.csv."
    )
    return _save(fig, folder, "hubble_four_cases", caption)


# 直接画四组保存的 H0 边缘密度及中央区间。
def _h0_comparison(folder, comparison):
    fig, (density, intervals) = plt.subplots(1, 2, figsize=(10.8, 4.4),
                                          gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=.075, right=.985, bottom=.17, top=.97, wspace=.24)
    ymax = 0.
    windows = []
    for index, case in enumerate(CASES):
        grid, pdf = _posterior(folder / "cases" / case / "h0_posterior.csv", "H0")
        low, median, high = _quantiles(comparison.loc[case], "h0")
        color = CASE_COLORS[case]
        density.plot(grid, pdf, color=color, ls="-", lw=1.65, label=CASE_LABELS[case])
        ymax = max(ymax, float(pdf.max()))
        windows.append(_display_window(grid, pdf))
        intervals.errorbar(median, index, xerr=[[median - low], [high - median]],
                           fmt="none", ecolor=ERROR_GRAY, elinewidth=1.5, capsize=0, zorder=1)
        intervals.plot(median, index, marker="D", ls="none", ms=5.0,
                       mfc=color, mec=INK, mew=.65, zorder=2)
    density.set(xlabel=H0_LABEL, ylabel="Posterior density", ylim=(0., ymax * 1.12),
                xlim=_padded_window(windows))
    density.legend(frameon=False, fontsize=10, loc="best", handlelength=2.0)
    lows = [float(comparison.loc[case, "h0_q16"]) for case in CASES]
    highs = [float(comparison.loc[case, "h0_q84"]) for case in CASES]
    pad = .14 * max(max(highs) - min(lows), 1.)
    intervals.set(xlabel=H0_LABEL, xlim=(max(0., min(lows) - pad), max(highs) + pad),
                  ylim=(3.55, -.55), yticks=np.arange(4), yticklabels=list(CASES))
    intervals.set_ylabel("Case")
    _frame(density)
    _frame(intervals, categorical_y=True)
    return _save(fig, folder, "h0_four_case_comparison", (
        "Left: saved marginal H0 probability densities for A–D, drawn without KDE, reweighting, "
        "or smoothing. Right: saved marginal medians (diamonds) and central 68% credible "
        "intervals (16th–84th percentiles). The probability-density panel uses a common "
        "display window containing the central 99.8% of every saved marginal, with padding; "
        "density values retain their full-grid normalization. The interval panel uses a "
        "tighter scale to resolve the reported intervals."
    ))


# 仅 B/D 存在连续 f_D 后验；A/C 是零点质量，不伪造曲线。
def _scatter_comparison(folder, comparison):
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    fig.subplots_adjust(left=.12, right=.985, bottom=.16, top=.97)
    ymax = 0.
    windows = []
    for case in ("B", "D"):
        grid, pdf = _posterior(folder / "cases" / case / "scatter_posterior.csv", "f_D")
        color = CASE_COLORS[case]
        ax.plot(grid, pdf, color=color, ls="-", lw=1.7, label=CASE_LABELS[case])
        ymax = max(ymax, float(pdf.max()))
        windows.append(_display_window(grid, pdf))
    ax.set(xlim=(0, _padded_window(windows)[1]), ylim=(0, ymax * 1.12),
           xlabel=r"Extra fractional distance scatter $f_D$", ylabel="Posterior density")
    ax.legend(frameon=False, fontsize=11, loc="best", handlelength=2.1)
    _frame(ax)
    return _save(fig, folder, "scatter_B_D_comparison", (
        "Saved marginal f_D probability densities for the two cases in which f_D is free: "
        "B excludes the flow correction/residual-PV model and D includes it. Both retain "
        "the same measured-redshift uncertainty. The common display window contains each "
        "central 99.8% interval and extends to zero; no smoothing or renormalization is applied. "
        "Cases A and C fix f_D=0 and are not represented as continuous densities."
    ))


# B/D 使用统一颜色标度画保存的二维 H0–f_D 密度。
def _joint_comparison(folder):
    loaded = {}
    maximum = 0.
    h_windows, f_windows = [], []
    for case in ("B", "D"):
        with np.load(folder / "cases" / case / "joint_posterior.npz") as saved:
            h, f, pdf = (np.asarray(saved[key], float) for key in ("H0", "f_D", "pdf"))
        if (h.ndim != 1 or f.ndim != 1 or pdf.shape != (len(h), len(f)) or
                not np.all(np.isfinite(pdf) & (pdf >= 0))):
            raise ValueError(f"Invalid saved joint posterior for {case}")
        loaded[case] = h, f, pdf
        h_windows.append(_display_window(h, trapezoid(pdf, f, axis=1)))
        f_windows.append(_display_window(f, trapezoid(pdf, h, axis=0)))
        maximum = max(maximum, float(pdf.max()))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.86, bottom=.17, top=.91, wspace=.08)
    for case, ax in zip(("B", "D"), axes):
        h, f, pdf = loaded[case]
        mesh = ax.pcolormesh(h, f, pdf.T, shading="auto", cmap="viridis", vmin=0.,
                             vmax=maximum, rasterized=True)
        ax.set_title(CASE_LABELS[case], fontsize=12)
        ax.set_xlabel(H0_LABEL)
        ax.set(xlim=_padded_window(h_windows), ylim=(0, _padded_window(f_windows)[1]))
        _frame(ax)
    axes[0].set_ylabel(r"Extra fractional distance scatter $f_D$")
    color_axis = fig.add_axes([.89, .17, .022, .74])
    fig.colorbar(mesh, cax=color_axis, label="Joint posterior density")
    return _save(fig, folder, "joint_B_D_comparison", (
        "Saved joint H0–f_D posterior densities for B and D. Both panels use the same "
        "axes and the same probability-density color normalization. The density is "
        "displayed directly on the saved grid without resampling or smoothing. Common axis "
        "windows contain the central 99.8% intervals of both corresponding marginals, "
        "with padding; the color normalization remains that of the full saved grid."
    ))


# 生成三组标准图及可选联合图，并保存文件清单和图注，便于复核。
def make_comparison_plots(output_dir: Path, *, include_joint: bool = False) -> dict:
    """Render three comparison figures (optionally a fourth joint-density figure).

    Return and save a JSON-serializable manifest of actual exported paths and
    captions.  PDF and 300 dpi PNG are always exported together.
    """
    folder = Path(output_dir).expanduser().resolve()
    common, comparison, fits, om0 = _load(folder)
    with plt.rc_context():
        apply_style()
        figures = [_hubble(folder, common, comparison, fits, om0),
                   _h0_comparison(folder, comparison),
                   _scatter_comparison(folder, comparison)]
        if include_joint:
            figures.append(_joint_comparison(folder))
    manifest = {
        "figure_count": len(figures),
        "image_file_count": 2 * len(figures),
        "dpi": SAVE_DPI,
        "cases": list(CASES),
        "source_count": len(common),
        "omega_m": om0,
        "style": "existing shared_sed_hubble style with requested local error/tick/export overrides",
        "figures": figures,
    }
    (folder / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


# 只重绘已完成四情景输出。
def main(argv=None):
    parser = argparse.ArgumentParser(description="Render completed four-case Hubble comparison outputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-joint", action="store_true")
    args = parser.parse_args(argv)
    manifest = make_comparison_plots(args.output, include_joint=args.include_joint)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

