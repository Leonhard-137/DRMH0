#!/usr/bin/env python3
# 论文的原生 MICA 分量展示：观测光变、总响应、各分量和传递函数同图对照。
# 光变阴影来自原生重建误差；TF 阴影来自保存的分位数，不能把两者视为同一统计定义。
"""Render paper MICA decomposition figures from existing native products.

Contract: one source, a shared UVW2 driver and five response rows; observed
light curves, native total/short/long reconstructions, and native total TFs.
This is a static, double-column scientific figure, inspired by the compact
panels, serif labels, inward ticks and restrained marks of Guo et al. (2022),
arXiv:2201.08533, Figure 1. Blue / orange solid lines identify components.

Light-curve bands are the native reconstruction mean +/- native error, not
posterior quantile intervals. TF bands use the stored median and asymmetric
15.85/84.15 percentile errors. Native TF normalization and samples are kept;
nonpositive ordinates are outside the displayed logarithmic axis.
No fitting or resampling occurs.
"""
from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FixedLocator, LogFormatterMathtext, MaxNLocator
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
BLUE = "#00008B"
ORANGE = "#D55E00"
SAMPLE_TABLE = ROOT / "DRMH0/tables/sample_observations.csv"
PAPER_OUTPUT = ROOT / "DRMH0/fig/paper/mica"


def source_label(source_name):
    base, _, segment = source_name.partition("_")
    if base == "F9":
        label = "Fairall 9"
    elif base.startswith("Mrk"):
        label = f"Mrk {base[3:]}"
    elif base.startswith("NGC"):
        label = f"NGC {base[3:]}"
    else:
        label = base
    return label


def band_label(band):
    return {"W2": "UVW2", "M2": "UVM2", "W1": "UVW1"}.get(band, band)


# 按原生文件头声明的 driver/response 行数切成两个三列数组。
# 这里允许打印精度导致的相同时间戳，但拒绝时间倒序；与严格递增读取器有区别。
def blocks(path):
    with path.open() as handle:
        count = int(handle.readline().lstrip("# "))
        if count != 1:
            raise ValueError(f"Expected one driver/response pair: {path}")
        sizes = [int(x) for x in handle.readline().lstrip("# ").split(":")]
    values = np.loadtxt(path, ndmin=2)
    if len(sizes) != 2 or sum(sizes) != len(values) or values.shape[1] != 3:
        raise ValueError(f"Invalid native block dimensions: {path}")
    if not np.isfinite(values).all() or np.any(values[:, 2] < 0):
        raise ValueError(f"Invalid native values: {path}")
    result = np.split(values, [sizes[0]])
    # Native text output rounds MJD values: nearby observations may share a
    # printed timestamp, so accept ties while rejecting reversed time order.
    if any(len(block) < 2 or np.any(np.diff(block[:, 0]) < 0) for block in result):
        raise ValueError(f"Invalid native time grid: {path}")
    return result


# 从指定源/方案定位原生文件并组织各响应波段；本图有固定的数据结构约定。
def load_rows(source, config, stage):
    fit = config[stage]
    if fit["ncomp"] != 2 or fit.get("component_order") != "native":
        raise ValueError("This figure expects two natively ordered components")
    if len(fit["responses"]) != 5 or len(set(fit["responses"])) != 5:
        raise ValueError("This figure expects five distinct response bands")
    run_root = source / fit["output_root"] / "2comp"
    rows = []
    for band in fit["responses"]:
        name = fit["run_template"].format(driver=fit["driver"], band=band,
                                           ncomp=2, type_tf=fit["type_tf"])
        folder = run_root / name / "data"
        driver, observed = blocks(folder / "data_input.txt")
        driver_model, total = blocks(folder / "pall.txt_2")
        short = blocks(folder / "pline.txt_2_comp0")[1]
        long = blocks(folder / "pline.txt_2_comp1")[1]
        transfer = np.loadtxt(folder / "tranfunc_0_1.txt_2")
        if transfer.shape[1] != 4 or not np.isfinite(transfer).all():
            raise ValueError(f"Invalid native TF: {folder}")
        if np.any(transfer[:, 1:] < 0) or np.any(np.diff(transfer[:, 0]) <= 0):
            raise ValueError(f"Invalid native TF values/grid: {folder}")
        for component in (short, long):
            if not np.array_equal(total[:, 0], component[:, 0]):
                raise ValueError(f"Misaligned reconstruction grid: {band}")
        if not np.allclose(total[:, 1], short[:, 1] + long[:, 1], atol=3e-5, rtol=3e-6):
            raise ValueError(f"Native component means do not sum to total: {band}")
        if rows and not np.array_equal(driver, rows[0]["driver"]):
            raise ValueError("Driver observations differ between runs")
        rows.append(dict(band=band, folder=folder, driver=driver,
                         driver_model=driver_model, observed=observed,
                         total=total, short=short, long=long, transfer=transfer))
    return rows


# 读取指定原生重建列，保持原始归一化，不再按峰值或面积重新缩放。
def native_curve(ax, values, color, time_origin, linestyle="-", alpha=0.20, lw=0.85):
    x, mean, error = values.T
    ax.fill_between(x - time_origin, mean - error, mean + error,
                    color=color, alpha=alpha, linewidth=0, zorder=1)
    ax.plot(x - time_origin, mean, color=color, lw=lw, ls=linestyle, zorder=3)


# 取观测时刻、通量及误差，用于与连续重建曲线分开显示。
def observed_points(ax, values, time_origin):
    ax.errorbar(values[:, 0] - time_origin, values[:, 1], yerr=values[:, 2],
                fmt="o", ms=1.35, color="black", ecolor="0.45",
                elinewidth=0.4, markeredgewidth=0, capsize=0, zorder=5)


# 在同一源内对齐驱动和响应面板；对数坐标看不到的非正响应不等于不存在。
def draw(source, config, stage, output):
    source, output = Path(source), Path(output)
    rows = load_rows(source, config, stage)
    title = source_label(source.name)
    time_origin = int(np.floor(min(row["driver"][0, 0] for row in rows) / 1000) * 1000)
    style = {
        "font.family": "serif", "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 8.5,
        "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.75, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "axes.grid": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(7.2, 8.0))
        grid = fig.add_gridspec(6, 2, left=0.105, right=0.915, bottom=0.065,
                               top=0.945, width_ratios=[3, 1],
                               height_ratios=[0.72, 1, 1, 1, 1, 1],
                               hspace=0.075, wspace=0.055)
        fig.text(0.105, 0.973, title, fontsize=11, va="top")
        lc_axes = []
        driver_ax = fig.add_subplot(grid[0, 0])
        lc_axes.append(driver_ax)
        # The driver is reconstructed separately in each paired run; use the
        # first response run for this display, without mixing reconstructed means.
        native_curve(driver_ax, rows[0]["driver_model"], "0.2", time_origin, alpha=0.18)
        observed_points(driver_ax, rows[0]["driver"], time_origin)
        driver_ax.text(0.975, 0.84, f"{band_label(config[stage]['driver'])} (driver)", ha="right", va="top",
                       transform=driver_ax.transAxes, fontsize=8)
        legend_ax = fig.add_subplot(grid[0, 1])
        legend_ax.set_axis_off()
        handles = [Line2D([], [], marker="o", color="black", ls="none", ms=2, label="Observed"),
                   Line2D([], [], color="black", lw=0.85, label="Total"),
                   Line2D([], [], color=BLUE, lw=0.85, label="Disk component"),
                   Line2D([], [], color=ORANGE, lw=0.85, label="Longer-lag component")]
        legend_ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.04, 0.5),
                         frameon=False, fontsize=7.5, handlelength=1.8,
                         handletextpad=0.6, labelspacing=0.38, borderaxespad=0)

        time_min = min(row["total"][0, 0] for row in rows) - time_origin
        time_max = max(row["total"][-1, 0] for row in rows) - time_origin
        tf_max = max(np.max(row["transfer"][:, 1] + row["transfer"][:, 3]) for row in rows)
        tf_upper = 10 ** np.ceil(np.log10(tf_max * 1.1))
        tf_lower = 1e-5
        tf_tmin = min(row["transfer"][0, 0] for row in rows)
        tf_tmax = max(row["transfer"][-1, 0] for row in rows)
        tf_ticks = 10.0 ** np.arange(int(np.log10(tf_lower)) + 1,
                                    int(np.log10(tf_upper)) + 1, 2)
        for index, row in enumerate(rows, start=1):
            ax = fig.add_subplot(grid[index, 0], sharex=driver_ax)
            lc_axes.append(ax)
            native_curve(ax, row["total"], "0.4", time_origin, alpha=0.18, lw=0.8)
            # Keep the total central curve black, distinct from its grey band.
            ax.lines[-1].set_color("black")
            native_curve(ax, row["short"], BLUE, time_origin)
            native_curve(ax, row["long"], ORANGE, time_origin)
            observed_points(ax, row["observed"], time_origin)
            high = max(np.max(row[k][:, 1] + row[k][:, 2])
                       for k in ("observed", "total", "short", "long"))
            low = min(np.min(row[k][:, 1] - row[k][:, 2])
                      for k in ("observed", "total", "short", "long"))
            ax.set_ylim(min(0, low * 1.06), high * 1.12)
            ax.text(0.975, 0.86, band_label(row["band"]), ha="right", va="top",
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.4))

            tf_ax = fig.add_subplot(grid[index, 1])
            tau, median, err_low, err_high = row["transfer"].T
            lower, upper = median - err_low, median + err_high
            tf_ax.set_yscale("log")
            tf_ax.fill_between(tau, np.maximum(lower, tf_lower), np.maximum(upper, tf_lower),
                               color="0.65", alpha=0.6, linewidth=0)
            tf_ax.plot(tau, np.where(median > 0, median, np.nan), color="black", lw=0.8)
            tf_ax.set_xlim(tf_tmin, tf_tmax)
            tf_ax.set_ylim(tf_lower, tf_upper)
            tf_ax.xaxis.set_major_locator(MaxNLocator(3, steps=[1, 2, 2.5, 5, 10]))
            tf_ax.yaxis.set_major_locator(FixedLocator(tf_ticks))
            tf_ax.yaxis.set_major_formatter(LogFormatterMathtext())
            tf_ax.yaxis.tick_right()
            tf_ax.yaxis.set_minor_locator(FixedLocator([]))
            tf_ax.xaxis.set_minor_locator(AutoMinorLocator(4))
            tf_ax.tick_params(which="both", left=True, right=True, top=True, labelsize=7.5)
            tf_ax.tick_params(labelbottom=index == 5)
            if index == 5:
                tf_ax.set_xlabel(r"Lag $\tau$ (day)", labelpad=5)

        for index, ax in enumerate(lc_axes):
            ax.set_xlim(time_min, time_max)
            ax.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(3, min_n_ticks=3))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(labelbottom=index == 5, which="both", top=True, right=True)
            ax.tick_params(which="major", length=3.3, width=0.7)
            ax.tick_params(which="minor", length=1.8, width=0.6)
        lc_axes[-1].set_xlabel(f"MJD − {time_origin}", labelpad=5)
        fig.text(0.028, 0.505,
                 r"$f_\lambda\;(10^{-15}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}})$",
                 rotation=90, ha="center", va="center", fontsize=10)
        fig.text(0.984, 0.465, r"$\Psi_\lambda(\tau)$", rotation=270,
                 ha="center", va="center", fontsize=10)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output.with_suffix(".pdf"), metadata={
            "Title": f"{title}: MICA light-curve decomposition and total transfer functions",
            "Subject": f"Native MICA products; {stage}; Guo et al. 2022-inspired Fig. 2",
        })
        fig.savefig(output.with_suffix(".png"), dpi=250)
        plt.close(fig)

    provenance = dict(
        source=source.name, stage=stage, reference="https://arxiv.org/abs/2201.08533",
        driver_display_run=str(rows[0]["folder"]),
        config_file=str(source / "config/source_config.yaml"),
        time_axis=f"MJD - {time_origin}; observer frame", time_origin_mjd=time_origin,
        flux_unit="1e-15 erg s^-1 cm^-2 Angstrom^-1",
        light_curve_bands="native reconstruction mean +/- native reconstruction error",
        transfer_bands="native pointwise posterior median with 15.85/84.15 percentile bounds",
        transfer_normalization="unaltered MICA output; logarithmic y",
        transfer_log_floor=tf_lower,
        transfer_display_limits_days=[float(tf_tmin), float(tf_tmax)],
        short_component_color=BLUE, long_component_color=ORANGE,
        products=[dict(
            band=row["band"], native_data_directory=str(row["folder"]),
            resolved_native_data_directory=str(row["folder"].resolve()),
            reuses_existing_run=row["folder"] != row["folder"].resolve(),
            input_files={name: str(row["folder"] / name) for name in (
                "data_input.txt", "pall.txt_2", "pline.txt_2_comp0",
                "pline.txt_2_comp1", "tranfunc_0_1.txt_2")},
            n_observed=len(row["observed"]), n_reconstruction=len(row["total"]),
            n_transfer=len(row["transfer"]),
            transfer_native_limits_days=[float(row["transfer"][0, 0]),
                                         float(row["transfer"][-1, 0])],
            max_component_sum_residual=float(np.max(np.abs(
                row["total"][:, 1] - row["short"][:, 1] - row["long"][:, 1]))),
        ) for row in rows],
    )
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(output.with_suffix(".pdf"))
    print(output.with_suffix(".png"))
    return {extension: str(output.with_suffix(f".{extension}"))
            for extension in ("pdf", "png", "json")}


# 逐源出图并登记失败源；完成部分图不等于整套样本成功。
def render_sample(workspace, sources, output_dir, stage="mica_gamma2_physw_relaxed"):
    """Render supplied sources in order; raise after recording any failures."""
    output_dir, root = Path(output_dir), Path(workspace)
    sources = list(sources)
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("Empty or duplicated sample")
    results, errors = {}, {}
    for name in sources:
        source = root / name
        try:
            config = yaml.safe_load((source / "config/source_config.yaml").read_text())
            results[name] = draw(source, config, stage,
                                 output_dir / f"Fig2_{name}_mica_decomposition")
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
            print(f"ERROR {name}: {errors[name]}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(workspace=str(root), stage=stage, sources=sources,
                    outputs=results, errors=errors)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if errors:
        raise RuntimeError("MICA figure generation failed: " + "; ".join(
            f"{name}: {error}" for name, error in errors.items()))
    return manifest


# 论文样本表控制批量范围，不从根目录任意发现源。
def run_batch(sample_table=SAMPLE_TABLE, stage="mica_gamma2_physw_relaxed", output_dir=PAPER_OUTPUT,
              root=ROOT):
    """Read the manuscript sample table and render every listed source."""
    with Path(sample_table).open(newline="") as handle:
        sources = [row["source"] for row in csv.DictReader(handle)]
    manifest = render_sample(root, sources, output_dir, stage)
    manifest["sample_table"] = str(sample_table)
    (Path(output_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="Mrk817_a")
    parser.add_argument("--stage", default="mica_gamma2_physw_relaxed")
    parser.add_argument("--all", action="store_true", help="Render the full manuscript sample")
    parser.add_argument("--sample-table", type=Path, default=SAMPLE_TABLE)
    parser.add_argument("--output-dir", type=Path,
                        help="Default: paper/mica for --all, otherwise fig/previews")
    args = parser.parse_args(argv)
    if args.all:
        return run_batch(args.sample_table, args.stage, args.output_dir or PAPER_OUTPUT)
    source = ROOT / args.source
    config = yaml.safe_load((source / "config/source_config.yaml").read_text())
    output_dir = args.output_dir or ROOT / "DRMH0/fig/previews"
    return draw(source, config, args.stage,
                output_dir / f"Fig2_{source.name}_mica_decomposition")


if __name__ == "__main__":
    main()
