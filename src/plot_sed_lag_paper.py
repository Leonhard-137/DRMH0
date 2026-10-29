#!/usr/bin/env python3
# 并排比较总光/短分量的 RMS SED 与短分量延迟谱。
# 左图使用 lambda*R_lambda；两套 SED 各用自身 UVW2 中位数做固定展示缩放。
# 左侧标准盘虚线是锚定后的理论形状，右侧虚线及区间则来自已保存的延迟拟合。
"""Render the paper's SED/lag comparison from saved FFA and MICA products.

Chart contract: one source, two compact paper-width panels in the serif,
inward-tick style of Guo et al. 2022 (arXiv:2201.08533). Left: the total-light
and short-component FFA RMS SED scaled by their own fixed UVW2 medians,
preserving the original point uncertainties. Right: relative short-component
centroid lags and the UVW2 reference. The left dashed curve is the SSD shape
anchored at UVW2, not a fit to either dataset. The right dashed curve remains
the saved SSD-slope fit to short-component lags, including U.

The default branch is mica_gamma2_physw_relaxed, as in Fig. 2. No new
inference is performed. This figure displays only short-component lags.
For each SED, C_W2 is the saved median of lambda_W2*R_W2. Divide all three
saved point quantiles by this fixed number, rather than forming a random
ratio in each posterior draw. W2 has central value one and retains its
uncertainty. These are the FFA R_lambda marginal errors; neither an extra
FFA intercept error nor the SED power-law slope error is added. The SSD
reference has fixed alpha=-4/3 and a fixed anchor, so has no uncertainty
band. The saved lag-fit posterior still supplies the lag confidence band.

Usage and data definitions: src/README_paper_figures.md.
"""
from pathlib import Path
import argparse
import json
import re
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
BLUE = "#00008B"  # Guo et al. 2022 dark blue (PDF drawing RGB: 0, 0, 139/255).


# 从联合 FFA 后验重新取 R 的边际分位数，避免旧 sed_points 表混入差分光谱定义。
def sed_points(path):
    frame = pd.read_csv(path).sort_values("wavelength_angstrom")
    # Several older summaries contain a difference spectrum. Always use the
    # saved FFA R posterior so the whole sample has the same RMS definition.
    posterior_path = path.with_name(path.name.replace("_sed_points.csv", f"_{len(frame)}band_flux_posterior.txt"))
    posterior = pd.read_csv(posterior_path, sep=r"\s+")
    columns = [f"R_{band}" for band in frame["band"]]
    if any(column not in posterior for column in columns):
        aliases = {"W2": "UVW2", "M2": "UVM2", "W1": "UVW1"}
        columns = [f"R_{aliases.get(band, band)}" for band in frame["band"]]
    values = posterior[columns].to_numpy(float) * frame["wavelength_angstrom"].to_numpy()[None, :]
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"Invalid saved RMS posterior: {posterior_path}")
    for quantile, values_q in zip((16, 50, 84), np.percentile(values, [16, 50, 84], axis=0)):
        frame[f"lambda_rms_p{quantile}"] = values_q
    frame["band"] = frame["band"].replace({"W2": "UVW2", "M2": "UVM2", "W1": "UVW1"})
    frame["quantity"] = "lambda_R_lambda"
    frame.attrs.update(flux_posterior=str(posterior_path), posterior_draws=len(posterior))
    vals = frame[["wavelength_angstrom", "lambda_rms_p16", "lambda_rms_p50", "lambda_rms_p84"]].to_numpy()
    if not np.isfinite(vals).all() or np.any(vals <= 0):
        raise ValueError(f"Invalid logarithmic SED points: {path}")
    if np.any(vals[:, 1] > vals[:, 2]) or np.any(vals[:, 2] > vals[:, 3]):
        raise ValueError(f"Reversed SED quantiles: {path}")
    return frame


# 所有分位数除以同一个固定 UVW2 中位数。
# 这不是逐样本求波段比值的后验，因此参考点中心为1但保留原有误差。
def w2_normalized_sed(points):
    """Apply a fixed display scale to the original FFA R_lambda quantiles."""
    reference = points.loc[points["band"].eq("UVW2")]
    if len(reference) != 1:
        raise ValueError("Expected one UVW2 SED anchor")
    anchor = reference.iloc[0]
    scale = float(anchor["lambda_rms_p50"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Expected a positive finite UVW2 median")
    display = points[["band", "wavelength_angstrom"]].copy()
    for quantile in (16, 50, 84):
        display[f"scaled_p{quantile}"] = points[f"lambda_rms_p{quantile}"] / scale
    information = dict(
        **points.attrs,
        method="fixed UVW2 median scaling", anchor_band="UVW2",
        anchor_wavelength_angstrom=float(anchor["wavelength_angstrom"]),
        anchor_value=1, scale_lambda_rms=scale,
        scaled_points=display.to_dict(orient="records"),
    )
    return display, information


def source_title(source_name):
    base, separator, segment = source_name.partition("_")
    base = re.sub(r"^(Mrk|NGC)(?=\d)", r"\1 ", base)
    base = re.sub(r"^F(?=\d)", "Fairall ", base)
    return base


# 要求两分量 Gamma 方案及唯一 UVW2 参考；输出图和来源 JSON。
# 先检查 source/stage/result，再解释图中任何科学差异。
def plot(source, stage="mica_gamma2_physw_relaxed", output=None, title=None, *, compact=False, show_x_labels=True, half_width=False):
    """Write PDF, PNG and provenance JSON; return their paths.

    ``source`` is a source directory and ``output`` is an optional filename
    stem (or a PDF/PNG/JSON path). Only existing analysis products are read.
    """
    source = Path(source).resolve()
    cfg = yaml.safe_load((source / "config/source_config.yaml").read_text())
    source_name = cfg["source"]
    stage_cfg = cfg[stage]
    if stage_cfg.get("type_tf") != "gamma" or stage_cfg.get("ncomp") != 2:
        raise ValueError("This figure requires a two-component Gamma result branch")
    result = source / stage_cfg["output_root"] / "2comp/result/comp0"
    total_dir = source / cfg.get("fflux", {}).get("out", "fflux") / "analysis"
    short_dir = result / "fflux/analysis"
    lag_dir = result / "lag"
    total_path = total_dir / f"{source_name}_total_sed_points.csv"
    short_path = short_dir / f"{source_name}_comp0_sed_points.csv"
    lag_path = lag_dir / f"{source_name}_lambda_tau_points.csv"
    total = sed_points(total_path)
    short = sed_points(short_path)
    if not total[["band", "wavelength_angstrom"]].reset_index(drop=True).equals(
        short[["band", "wavelength_angstrom"]].reset_index(drop=True)
    ):
        raise ValueError("Total and short SED bands differ")
    total, total_shape = w2_normalized_sed(total)
    short, short_shape = w2_normalized_sed(short)
    lags = pd.read_csv(lag_path).sort_values("lambda_obs")
    lags["band"] = lags["band"].replace({"W2": "UVW2", "M2": "UVM2", "W1": "UVW1"})
    response = lags[~lags["is_ref"]]
    reference = lags[lags["is_ref"]]
    if len(reference) != 1 or reference.iloc[0]["band"] != "UVW2":
        raise ValueError("Expected exactly one UVW2 lag reference")
    lag_values = response[["tau", "tau_err_low", "tau_err_high"]].to_numpy()
    if not np.isfinite(lag_values).all() or np.any(lag_values[:, 1:] < 0):
        raise ValueError("Invalid lag point/uncertainty")

    lag_fit = pd.read_csv(lag_dir / f"{source_name}_lambda_tau_fit_summary.csv")
    lag_fit = lag_fit[lag_fit["fit_model"].eq("4/3")].iloc[0]
    lag_post = pd.read_csv(lag_dir / "tau0_posterior.txt", sep=r"\s+")
    if not np.isclose(lag_fit["beta"], 4/3):
        raise ValueError("The saved standard lag fit does not have beta=4/3")
    if not np.isclose(lag_fit["lambda0"], short_shape["anchor_wavelength_angstrom"]):
        raise ValueError("The saved lag fit and SED must share the UVW2 reference wavelength")

    wavelengths = np.concatenate((total["wavelength_angstrom"], lags["lambda_obs"]))
    x_limits = (min(1800, wavelengths.min() * 0.94), max(5750, wavelengths.max() * 1.05))
    wave = np.linspace(*x_limits, 450)
    sed_anchor = short_shape["anchor_wavelength_angstrom"]
    sed_curve = (wave / sed_anchor) ** (-4/3)
    lag_q = np.percentile(lag_post["tau0_b4_3"], [15.865, 50, 84.135])
    lag_factor = (wave / lag_fit["lambda0"]) ** (4/3) - 1
    # 参考波长左侧的 lag_factor 可为负；重新排序避免误差带上下界翻转。
    lag_curves = np.sort(lag_q[:, None] * lag_factor[None, :], axis=0)
    source_label = title or source_title(source_name)
    if output is None:
        output = source.parent / "DRMH0/fig/previews" / f"Fig3_{source_name}_sed_lag"
    output = Path(output).resolve()

    style = {
        "font.family": "serif", "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 8.5,
        "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.75, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "axes.grid": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if half_width:
        style.update({"font.size": 8, "axes.labelsize": 8,
                      "xtick.labelsize": 7, "ytick.labelsize": 7})
    with plt.rc_context(style):
        # Shared figure labels leave more room for data. The last row adds
        # space for numeric x ticks only; every row has the same axes height.
        pair_height = 1.53 if show_x_labels else 1.38
        figsize = ((3.55, pair_height) if half_width
                   else (7.2, 1.88 if compact else 3.3))
        fig, (ax_sed, ax_lag) = plt.subplots(1, 2, figsize=figsize)
        fig.subplots_adjust(left=0.085 if half_width else 0.10,
                            right=0.99 if half_width else 0.982,
                            top=(1 - 0.18 / pair_height) if half_width else (0.83 if compact else 0.86),
                            bottom=((0.20 if show_x_labels else 0.05) / pair_height) if half_width
                            else ((0.23 if show_x_labels else 0.07) if compact else 0.185),
                            wspace=0.28 if half_width else 0.31)
        fig.text(0.085 if half_width else 0.10, 0.98 if compact else 0.965,
                 source_label, ha="left", va="top", fontsize=8.5 if half_width else (9 if compact else 11))
        for ax, title in ((ax_sed, "(a) Variable SED"), (ax_lag, "(b) Lag spectrum")):
            if not compact:
                ax.set_title(title, loc="left", fontsize=9.5, pad=6)
            if show_x_labels and not half_width:
                ax.set_xlabel(r"Observed wavelength ($\mathrm{\AA}$)", labelpad=3 if compact else 5)
            elif not show_x_labels:
                ax.tick_params(labelbottom=False)
            ax.set_xlim(*x_limits)
            ax.tick_params(which="major", length=3.4, width=0.7)
            ax.tick_params(which="minor", length=1.8, width=0.6)
            ax.axvline(3465, color="0.75", lw=0.65, ls=":", zorder=0)
            ax.text(3465, 0.96 if ax is ax_sed else 0.73, "U", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color="0.35")

        ax_sed.set_xscale("log")
        ax_sed.set_yscale("log")
        model_line, = ax_sed.plot(wave, sed_curve, color="black", lw=1.0,
                                  ls="--", label=r"SSD shape ($\alpha=-4/3$)")
        data_handles = []
        for frame, marker, color, label, filled in (
            (total, "s", "0.35", "Before decomposition", False),
            (short, "o", BLUE, "Disk component", True),
        ):
            y = frame["scaled_p50"].to_numpy()
            errors = np.vstack((frame["scaled_p50"] - frame["scaled_p16"],
                                frame["scaled_p84"] - frame["scaled_p50"]))
            item = ax_sed.errorbar(frame["wavelength_angstrom"], y, yerr=errors,
                                  fmt=marker, ms=4, color=color,
                                  mfc=color if filled else "white", mec=color, mew=0.8,
                                  elinewidth=0.85, capsize=2, zorder=4 if filled else 3, label=label)
            data_handles.append(item)
        ax_sed.set_ylim(
            min(0.15, total["scaled_p16"].min() * 0.85, short["scaled_p16"].min() * 0.85),
            max(1.25, total["scaled_p84"].max() * 1.25, short["scaled_p84"].max() * 1.25),
        )
        wave_ticks = [2000, 3500, 5000] if half_width else [2000, 3000, 4000, 5000]
        ax_sed.xaxis.set_major_locator(FixedLocator(wave_ticks))
        ax_sed.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
        ax_sed.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
        ax_sed.xaxis.set_minor_formatter(NullFormatter())
        ax_sed.yaxis.set_major_locator(FixedLocator([0.2, 0.5, 1]))
        ax_sed.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:g}"))
        ax_sed.yaxis.set_minor_formatter(NullFormatter())
        if not half_width:
            ax_sed.set_ylabel(r"$\lambda R_\lambda\,/\,C_{\rm W2}$", labelpad=5)
        if not compact:
            ax_sed.legend(handles=[*data_handles, model_line], loc="lower left",
                          frameon=False, fontsize=7, handlelength=1.8, labelspacing=0.35)
            ax_sed.text(0.98, 0.96, "W2 median = 1", transform=ax_sed.transAxes,
                        ha="right", va="top", fontsize=7, color="0.35")
            ax_sed.annotate("W2", xy=(sed_anchor, 1), xytext=(7, 6),
                            textcoords="offset points", fontsize=7, color="0.35")

        ax_lag.fill_between(wave, lag_curves[0], lag_curves[2], color="0.6", alpha=0.18, lw=0)
        lag_line, = ax_lag.plot(wave, lag_curves[1], color="black", lw=1.0,
                                ls="--", label=r"$\beta=4/3$ fit")
        lag_data = ax_lag.errorbar(response["lambda_obs"], response["tau"],
                                  yerr=np.vstack((response["tau_err_low"], response["tau_err_high"])),
                                  fmt="o", ms=4, color=BLUE, mew=0.8,
                                  elinewidth=0.85, capsize=2, zorder=4, label="Disk component")
        ref_data = ax_lag.plot(reference["lambda_obs"], reference["tau"],
                              marker="s", ms=3.5, mfc="white", mec="0.35", mew=0.8,
                              ls="none", zorder=4, label="UVW2 reference")[0]
        ax_lag.axhline(0, color="0.7", lw=0.65, ls=":", zorder=0)
        lag_lower = (response["tau"] - response["tau_err_low"]).min()
        lag_upper = max((response["tau"] + response["tau_err_high"]).max(), lag_curves.max())
        ax_lag.set_ylim(min(-0.06, lag_lower - 0.06), max(0.80, lag_upper * 1.15))
        ax_lag.set_xticks(wave_ticks)
        if ax_lag.get_ylim() == (-0.06, 0.80):
            ax_lag.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax_lag.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax_lag.yaxis.set_minor_locator(AutoMinorLocator(2))
        if not half_width:
            ax_lag.set_ylabel(r"Lag relative to UVW2 (day)", labelpad=5)
        if not compact:
            ax_lag.legend(handles=[lag_data, lag_line, ref_data], loc="upper left",
                          frameon=False, fontsize=7, handlelength=1.8, labelspacing=0.35)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output.with_suffix(".pdf"), metadata={
            "Title": f"{source_label}: variable SED and short-component lag spectrum",
            "Subject": f"{stage}; UVW2-normalized RMS SED with SSD reference shape; relative centroid lags",
        })
        fig.savefig(output.with_suffix(".png"), dpi=300)
        plt.close(fig)
    provenance = dict(
        source=source_name, stage=stage, source_config=str(source / "config/source_config.yaml"),
        result_directory=str(result), title=source_label,
        style_reference="https://arxiv.org/abs/2201.08533", short_component_color=BLUE,
        total_sed=str(total_path), short_sed=str(short_path), short_lags=str(lag_path),
        sed_quantity="(lambda R_lambda)/C_W2; C_W2 is the fixed saved UVW2 median for each SED",
        displayed_sed_unit="dimensionless", wavelength_frame="observer",
        lag_quantity="short-component Gamma centroid relative to UVW2; observer-frame days",
        original_light_curve_lags="not displayed; this figure uses saved short-component lags only",
        sed_reference_curve="(lambda/lambda_W2)^(-4/3); fixed common anchor; not fitted to either SED",
        lag_model_curve="beta=4/3 fitted to short-component lags, including U",
        models_are_independent_physical_predictions=False,
        sed_point_intervals="original lambda_R_lambda p16/p50/p84 divided by fixed C_W2",
        sed_uncertainty_scope="FFA R_lambda marginal uncertainties, including W2; no added A or alpha errors",
        sed_display_scaling=dict(total=total_shape, short=short_shape),
        sed_model_band="none: SSD reference slope and anchor are fixed; point uncertainties are retained",
        lag_model_band="unchanged: saved tau0 posterior propagated through beta=4/3",
    )
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    return {extension: output.with_suffix(f".{extension}") for extension in ("pdf", "png", "json")}


# 按给定样本顺序在最终大小重绘多源面板，并保留来源记录。
def render_sample(workspace, sources, output_dir, stage="mica_gamma2_physw_relaxed", *, layout_only=False):
    """Render two source pairs per row, preserving readable final-size text."""
    import fitz
    from matplotlib.lines import Line2D

    workspace, output_dir = Path(workspace), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compact_dir = output_dir / "compact"
    compact_dir.mkdir(exist_ok=True)
    manifest = dict(workspace=str(workspace), stage=stage, sources=list(sources), outputs={}, pages=[])
    previous = json.loads((output_dir / "manifest.json").read_text()) if layout_only else None
    for source in sources:
        if layout_only:
            manifest["outputs"][source] = previous["outputs"][source]
        else:
            paths = plot(workspace / source, stage, output_dir / f"Fig3_{source}_sed_lag")
            manifest["outputs"][source] = {key: str(value) for key, value in paths.items()}
        print(f"SED/lag {source}", flush=True)
    with plt.rc_context({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "pdf.fonttype": 42}):
        header = plt.figure(figsize=(7.2, 0.42))
        header.text(0.5, 0.97,
                    r"Left in each pair: $\lambda R_\lambda/C_{\rm W2}$ (dimensionless)"
                    r"     Right: lag relative to UVW2 (day)",
                    fontsize=8.5, ha="center", va="top")
        handles = [Line2D([], [], marker="s", mfc="white", mec="0.35", ls="none", ms=4, label="Before decomposition"),
                   Line2D([], [], marker="o", color=BLUE, ls="none", ms=4, label="Disk component"),
                   Line2D([], [], color="black", ls="--", label="SSD reference / lag fit")]
        header.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
                      bbox_to_anchor=(0.5, -0.10), handlelength=1.7, columnspacing=1.4)
        header_path = compact_dir / "header.pdf"
        header.savefig(header_path)
        plt.close(header)
        footer = plt.figure(figsize=(7.2, 0.16))
        footer.text(0.5, 0.50, r"Observed wavelength ($\mathrm{\AA}$)",
                    fontsize=8, ha="center", va="center")
        footer_path = compact_dir / "footer.pdf"
        footer.savefig(footer_path)
        plt.close(footer)
    for page_index, start in enumerate(range(0, len(sources), 10), 1):
        chunk = sources[start:start+10]
        row_paths = []
        for i, source in enumerate(chunk):
            paths = plot(workspace / source, stage, compact_dir / source,
                         compact=True, half_width=True,
                         show_x_labels=i // 2 == (len(chunk)-1) // 2)
            row_paths.append(paths["pdf"])
        page_path = output_dir / f"Fig3_sed_lag_page{page_index}.pdf"
        documents = [fitz.open(path) for path in [header_path, *row_paths, footer_path]]
        try:
            width = 7.2 * 72
            pair_width = 3.55 * 72
            gap = 0.10 * 72
            header_height = documents[0][0].rect.height
            footer_height = documents[-1][0].rect.height
            pairs = documents[1:-1]
            row_heights = [max(doc[0].rect.height for doc in pairs[i:i+2])
                           for i in range(0, len(chunk), 2)]
            with fitz.open() as combined:
                page = combined.new_page(width=width, height=header_height + sum(row_heights) + footer_height)
                page.show_pdf_page(fitz.Rect(0, 0, width, header_height), documents[0], 0)
                y = header_height
                for row, height in enumerate(row_heights):
                    for col, doc in enumerate(pairs[2*row:2*row+2]):
                        x = col * (pair_width + gap)
                        page.show_pdf_page(fitz.Rect(x, y, x+pair_width, y+height), doc, 0)
                    y += height
                page.show_pdf_page(fitz.Rect(0, y, width, y+footer_height), documents[-1], 0)
                combined.save(page_path, deflate=True)
                page.get_pixmap(dpi=300).save(page_path.with_suffix(".png"))
        finally:
            for doc in documents:
                doc.close()
        entry = dict(stage=stage, sources=chunk, source_columns=2, panel_columns=4,
                     source_order="left to right, then top to bottom",
                     pdf=str(page_path), png=str(page_path.with_suffix(".png")))
        page_path.with_suffix(".json").write_text(json.dumps(entry, indent=2) + "\n")
        manifest["pages"].append(entry)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT, help="Directory containing source folders")
    parser.add_argument("--source", default="Mrk817_a", help="Source folder name or absolute directory")
    parser.add_argument("--stage", default="mica_gamma2_physw_relaxed", help="Two-Gamma result branch in source_config.yaml")
    parser.add_argument("--output", type=Path, help="Output stem or PDF path; also writes PNG and JSON")
    parser.add_argument("--title", help="Override the source title")
    parser.add_argument("--all", action="store_true", help="Render all sources and four-column paper pages")
    parser.add_argument("--layout-only", action="store_true", help="Keep standalone figures and rebuild compact paper panels")
    parser.add_argument("--output-dir", type=Path, help="All-source output directory")
    args = parser.parse_args(argv)
    if args.all:
        with (args.workspace / "DRMH0/tables/sample_observations.csv").open() as handle:
            sources = [row["source"] for row in csv.DictReader(handle)]
        render_sample(args.workspace, sources, args.output_dir or args.workspace / "DRMH0/fig/paper/sed_lag", args.stage,
                      layout_only=args.layout_only)
        return
    paths = plot(args.workspace / args.source, args.stage, args.output, args.title)
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
