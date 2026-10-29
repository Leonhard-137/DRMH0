# 只读已完成 shared-SED 的 CSV/后验画图，不导入源拟合或运行采样。
# 距离必须为 Mpc 的 D_L；μ=5log10(D_L/Mpc)+25。SciPy KDE 仅用于画旧 H0 样本密度。
"""Reusable plotting for existing shared-SED source distances and H0 samples.

The module only consumes result CSV files; it does not run inference or import
analysis code from output directories. Distances must be luminosity distances
in Mpc. The shown H0 quantiles are read verbatim from the inference summary.
"""
from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, NullFormatter

from .style import (
    INK, GRAY, ORANGE, ERROR_COLOR, H0_REFERENCES,
    HUBBLE_FIGSIZE, HUBBLE_LAYOUT, HUBBLE_MARKER_SIZE,
    SED_MAX_COLUMNS, SED_PANEL_SIZE, SED_LAYOUT, SED_MARKER_SIZE,
    SAVE_DPI, H0_KDE_GRID_SIZE, H0_KDE_BANDWIDTH,
    REDSHIFT_TICKS, WAVELENGTH_TICKS, SOURCE_LABEL_OFFSETS,
    DEFAULT_LABEL_OFFSET, apply_style, apply_frame, marker_face, source_label,
)


# 显示曲线采用相同平直 ΛCDM 的 H0 D_L；除 H0 得到 Mpc。
def dl_scale(z: float, om0: float) -> float:
    """Return H0 times luminosity distance in flat LCDM, in km/s."""
    return 299792.458 * (1 + z) * quad(
        lambda x: 1 / np.sqrt(om0 * (1 + x)**3 + 1 - om0),
        0, z, epsabs=1e-11,
    )[0]


# 同步保存 PNG/PDF，关闭图对象释放内存。
def save(fig, output_dir: Path, stem: str) -> tuple[Path, Path]:
    paths = tuple(output_dir / f"{stem}.{extension}" for extension in ("png", "pdf"))
    for path in paths:
        fig.savefig(path, dpi=SAVE_DPI, facecolor="white")
    plt.close(fig)
    return paths


# 距离分位数逐个单调变换到 μ；灰带仅含 H0 参数区间，参考 H0 带不参与拟合。
def plot_hubble(distances, summary, h_samples, output_dir: Path, om0: float):
    d = distances.sort_values("z").reset_index(drop=True).copy()
    for suffix in ("q16", "q50", "q84"):
        values = d[f"D_{suffix}"].to_numpy(float)
        if not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError("Luminosity-distance quantiles must be positive and finite")
        d[f"mu_{suffix}"] = 5 * np.log10(values) + 25
    hlo, hmed, hhi = [float(summary[f"h0_{suffix}"]) for suffix in ("q16", "q50", "q84")]
    if not 0 < hlo <= hmed <= hhi:
        raise ValueError("Invalid H0 quantiles")
    h_samples = np.asarray(h_samples, float)
    if not np.all(np.isfinite(h_samples) & (h_samples > 0)):
        raise ValueError("Invalid H0 samples")
    d["mu_model"] = 5 * np.log10(np.array([dl_scale(z, om0) for z in d.z]) / hmed) + 25
    d["residual"] = d.mu_q50 - d.mu_model

    fig = plt.figure(figsize=HUBBLE_FIGSIZE)
    gs = fig.add_gridspec(2, 2, **HUBBLE_LAYOUT)
    ax = fig.add_subplot(gs[0, 0])
    res = fig.add_subplot(gs[1, 0], sharex=ax)
    post = fig.add_subplot(gs[:, 1])
    zmin, zmax = float(d.z.min()) * .80, float(d.z.max()) * 1.18
    zz = np.geomspace(zmin, zmax, 400)
    scales = np.array([dl_scale(z, om0) for z in zz])
    mu = 5 * np.log10(scales / hmed) + 25
    mu_lo, mu_hi = 5 * np.log10(scales / hhi) + 25, 5 * np.log10(scales / hlo) + 25
    ax.fill_between(zz, mu_lo, mu_hi, color=GRAY, alpha=.65, lw=0, zorder=1)
    ax.plot(zz, mu, color=ORANGE, lw=1.7, zorder=2)
    res.fill_between(zz, mu_lo - mu, mu_hi - mu, color=GRAY, alpha=.65, lw=0, zorder=1)
    res.axhline(0, color=ORANGE, lw=1.4, zorder=2)
    for row in d.itertuples():
        low, high = row.mu_q50 - row.mu_q16, row.mu_q84 - row.mu_q50
        face = marker_face(row.source)
        for panel, ordinate in ((ax, row.mu_q50), (res, row.residual)):
            panel.errorbar(row.z, ordinate, yerr=[[low], [high]], fmt="o", ms=HUBBLE_MARKER_SIZE,
                           mfc=face, mec=INK, mew=1, ecolor=ERROR_COLOR, elinewidth=.85,
                           capsize=2, zorder=4)
        dx, dy = SOURCE_LABEL_OFFSETS.get(row.source, DEFAULT_LABEL_OFFSET)
        ax.annotate(source_label(row.source), (row.z, row.mu_q50), xytext=(dx, dy),
                    textcoords="offset points", ha="right" if dx < 0 else "left",
                    fontsize=9.7, color=INK, zorder=6)
    # Limits include every source error bar; no discrepant source is cropped.
    ax.set_ylim(float(d.mu_q16.min()) - .55, float(d.mu_q84.max()) + .65)
    rlo = (d.mu_q16 - d.mu_model).min()
    rhi = (d.mu_q84 - d.mu_model).max()
    res.set_ylim(min(float(rlo), 0) - .3, max(float(rhi), 0) + .3)
    for panel in (ax, res):
        panel.set_xscale("log")
        panel.set_xlim(zmin, zmax)
        panel.xaxis.set_major_locator(FixedLocator(REDSHIFT_TICKS))
        panel.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
        panel.xaxis.set_minor_formatter(NullFormatter())
        panel.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(labelbottom=False)
    ax.set_ylabel(r"Distance modulus $\mu$ (mag)")
    res.set_xlabel(r"Redshift $z$")
    res.set_ylabel(r"$\Delta\mu$ (mag)")
    ax.legend(handles=[Line2D([], [], color=ORANGE, lw=1.7, label="Joint median"),
                       Patch(facecolor=GRAY, alpha=.65, edgecolor="none", label=r"68% $H_0$ interval")],
              loc="lower right", frameon=False, fontsize=10, handlelength=2)

    # References are display-only: no likelihood or prior is evaluated here.
    references = H0_REFERENCES
    xmin = min(float(h_samples.min()), min(ref.mean - 3 * ref.sigma for ref in references))
    xmax = max(float(h_samples.max()), max(ref.mean + 3 * ref.sigma for ref in references))
    pad = max(.5, .04 * (xmax - xmin))
    xlim = (max(0, xmin - pad), xmax + pad)
    # Scott-bandwidth KDE represents all saved MCMC samples for display only;
    # quoted quantiles remain those in h0_summary.csv, without smoothing.
    centers = np.linspace(*xlim, H0_KDE_GRID_SIZE)
    density = gaussian_kde(h_samples, bw_method=H0_KDE_BANDWIDTH)(centers)
    for ref in references:
        post.axvspan(ref.mean - ref.sigma, ref.mean + ref.sigma,
                     facecolor=ref.color, alpha=.22, lw=0, zorder=0)
        post.axvline(ref.mean, color=ref.color, ls=ref.linestyle, lw=1.2, zorder=3)
    post.plot(centers, density, color=INK, lw=1.6, zorder=4)
    post.fill_between(centers, 0, density, where=(centers >= hlo) & (centers <= hhi),
                      interpolate=True, color=GRAY, lw=0, zorder=2)
    post.plot([hmed, hmed], [0, np.interp(hmed, centers, density)], color=ORANGE, lw=1.5, zorder=5)
    post.set(xlim=xlim, ylim=(0, float(density.max()) * 1.53),
             xlabel=r"$H_0$ (km s$^{-1}$ Mpc$^{-1}$)", ylabel="Posterior density")
    post.text(.96, .965, rf"$H_0={hmed:.1f}^{{+{hhi-hmed:.1f}}}_{{-{hmed-hlo:.1f}}}$",
              transform=post.transAxes, va="top", ha="right", fontsize=19)
    post.legend(handles=[Line2D([], [], color=INK, lw=1.6, label="Joint posterior"),
                         *[Line2D([], [], color=ref.color, ls=ref.linestyle,
                                  lw=1.2, label=ref.label) for ref in reversed(references)]],
                frameon=False, loc="upper right", bbox_to_anchor=(.99, .86), fontsize=10,
                handlelength=2.3, labelspacing=.55)
    post.xaxis.set_minor_locator(AutoMinorLocator(2))
    post.yaxis.set_minor_locator(AutoMinorLocator(2))
    for panel in (ax, res, post):
        apply_frame(panel)
    assert not fig.texts and all(not panel.get_title() for panel in (ax, res, post))
    return save(fig, output_dir, "joint_hubble_modulus")


# 画归一化 λRλ，自由 FFA 点与强制 −7/3 的共享 Fλ SED 比较；乘 λ 后斜率变为 −4/3。
def plot_seds(bands, sources: list[str], output_dir: Path):
    ncols = min(SED_MAX_COLUMNS, len(sources))
    nrows = math.ceil(len(sources) / ncols)
    figsize = (SED_PANEL_SIZE[0] * ncols, SED_PANEL_SIZE[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             gridspec_kw=SED_LAYOUT)
    for i, (source, ax) in enumerate(zip(sources, axes.flat)):
        b = bands.loc[bands.source == source].sort_values("wavelength").copy()
        if len(b) < 2 or b.band.duplicated().any():
            raise ValueError(f"Expected at least two distinct SED bands for {source}")
        wave = b.wavelength.to_numpy(float)
        reference = b.iloc[0]
        normalization = float(reference.wavelength * reference.R_shared_q50)
        if not np.isfinite(normalization) or normalization <= 0:
            raise ValueError(f"Invalid shared reference normalization for {source}")
        free = {s: wave * b[f"R_free_{s}"].to_numpy(float) / normalization for s in ("q16", "q50", "q84")}
        shared = {s: wave * b[f"R_shared_{s}"].to_numpy(float) / normalization for s in ("q16", "q50", "q84")}
        values = np.concatenate([*free.values(), *shared.values()])
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Nonfinite SED data for {source}")
        xgrid = np.geomspace(wave.min() * .95, wave.max() * 1.055, 200)
        shape = (xgrid / float(reference.wavelength)) ** (-4 / 3)
        low = shape * float(reference.R_shared_q16 / reference.R_shared_q50)
        high = shape * float(reference.R_shared_q84 / reference.R_shared_q50)
        ax.fill_between(xgrid, low, high, color=ORANGE, alpha=.25, lw=0, zorder=1)
        ax.plot(xgrid, shape, color=ORANGE, lw=1.7, zorder=2)
        ax.errorbar(wave, free["q50"], yerr=[free["q50"] - free["q16"], free["q84"] - free["q50"]],
                    fmt="o", ms=SED_MARKER_SIZE, mfc=marker_face(source), mec=INK,
                    ecolor=INK, mew=.85, elinewidth=.85, capsize=2, zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(float(xgrid.min()), float(xgrid.max()))
        if values.min() > 0:
            ax.set_yscale("log")
            ymin, ymax = min(float(values.min()), float(low.min())), max(float(values.max()), float(high.max()))
            span = max(np.log(ymax / ymin), .4)
            ax.set_ylim(ymin * np.exp(-.14 * span), ymax * np.exp(.22 * span))
        else:
            ymin, ymax = min(float(values.min()), float(low.min())), max(float(values.max()), float(high.max()))
            span = max(ymax - ymin, .1)
            ax.set_ylim(ymin - .14 * span, ymax + .22 * span)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_major_locator(FixedLocator(WAVELENGTH_TICKS))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.0f}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=10)
        ax.text(.045, .965, source_label(source), transform=ax.transAxes, ha="left", va="top", fontsize=12)
        if i // ncols == nrows - 1:
            ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$)", fontsize=12)
        else:
            ax.tick_params(labelbottom=False)
        if i % ncols == 0:
            ax.set_ylabel(r"$\lambda R_\lambda\,/\,(\lambda_{\rm ref} R_{\rm ref,shared})$", fontsize=12)
        apply_frame(ax)
    axes.flat[0].legend(handles=[Line2D([], [], ls="none", marker="o", color=INK, ms=4,
                                      label="Free FFA"),
                               Line2D([], [], color=ORANGE, lw=1.7, label="Shared standard SED")],
                         loc="lower left", frameon=False, fontsize=9, handlelength=1.5, labelspacing=.3)
    for ax in axes.flat[len(sources):]:
        ax.set_visible(False)
    assert not fig.texts
    return save(fig, output_dir, "shared_sed_comparison")


# 要求距离摘要和 SED 表包含完全相同的源集，防止子样本图搭配全样本 H0。
def make_plots(
    output_dir: Path,
    sources: list[str] | None = None,
    om0: float = .3,
) -> dict[str, tuple[Path, Path]]:
    """Render the two accepted PNG/PDF figures from completed result CSVs.

    ``sources`` controls SED panel order and must cover precisely the sources
    in the saved distance summary. Omitting it preserves the CSV order. This
    prevents displaying a subset alongside an H0 posterior fitted to more AGN.
    The function never changes posterior samples or numerical summaries.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    distances = pd.read_csv(output_dir / "source_distance_summary.csv")
    summary = pd.read_csv(output_dir / "h0_summary.csv").iloc[0]
    samples = pd.read_csv(output_dir / "h0_posterior.csv")["H0"].to_numpy(float)
    bands = pd.read_csv(output_dir / "band_comparison.csv")
    actual_sources = distances.source.astype(str).tolist()
    sources = actual_sources if sources is None else list(sources)
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("Sources must be nonempty and distinct")
    if len(actual_sources) != len(sources) or set(actual_sources) != set(sources):
        raise ValueError("Source selection must match the saved distance summary")
    if set(bands.source.astype(str)) != set(sources):
        raise ValueError("SED comparison and distance summary source sets differ")
    if not np.isfinite(om0) or not 0 <= om0 <= 1:
        raise ValueError("Flat LCDM plotting requires 0 <= om0 <= 1")
    if not np.all(np.isfinite(distances.z) & (distances.z > 0)):
        raise ValueError("Redshifts must be positive and finite")
    with plt.rc_context():
        apply_style()
        hubble = plot_hubble(distances, summary, samples, output_dir, om0)
        sed = plot_seds(bands, sources, output_dir)
    return {"joint_hubble_modulus": hubble, "shared_sed_comparison": sed}


# 独立重绘入口，读取已完成结果并输出两组 PNG/PDF。
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Redraw existing shared-SED distance and H0 figures")
    parser.add_argument("--output", type=Path, required=True, help="Directory containing the four result CSV files")
    parser.add_argument("--sources", nargs="+", help="Source order; must match the saved source set")
    parser.add_argument("--om0", type=float, default=.3, help="Flat LCDM matter density used for display (default: 0.3)")
    args = parser.parse_args(argv)
    paths = make_plots(args.output, sources=args.sources, om0=args.om0)
    for name, files in paths.items():
        print(name + ": " + ", ".join(str(path) for path in files))


if __name__ == "__main__":
    main()
