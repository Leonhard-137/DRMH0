#!/usr/bin/env python3
"""Plot comp0 and native total MICA transfer functions, without running fits.

Examples (from the project root):
    python src/plot_mica_transfer.py Mrk817 --stage mica_gamma2_physw
    python src/plot_mica_transfer.py Mrk817 --stage mica_gamma2_legacy
    python src/plot_mica_transfer.py --all --stage mica_mix220
    python src/plot_mica_transfer.py Mrk142 --stage mica_gaussian2_free --median-parameters

Each band produces separate comp0/total monochrome PNG/PDF figures, named
SOURCE_BAND_COMPONENT_transfer, in result/transfer_functions under its run
root. The same directory also receives SOURCE_transfer_functions_overview as
a five-row, two-column comp0/total summary. Single panels are 89 x 67 mm and
the overview is 180 x 235 mm (PNG: 300 dpi). X is linear observer-frame lag
relative to the driver; Y is logarithmic in MICA's original normalization.
The paired figures share a Y window; X windows omit only tails below that
window's floor. The combined CSV retains the original, uncropped curves.
The total comes ONLY from tranfunc_0_1.txt_N. Comp0 uses every equal-weight
posterior_sample1d.txt_N draw, with pointwise 15.85/50/84.15 percentiles.
Gamma accepts both historical center columns and current centroid columns;
the common MICA reader converts either representation to the same physical
kernel parameters. No posterior cuts or reweighting are applied; component
ordering follows the existing source configuration.

With ``--median-parameters``, no old file is replaced. The script independently
takes the posterior median of each component's integrated response ``f``,
physical centroid ``tau``, and width ``w``, inserts those medians into the
configured analytic kernel, and sums the component curves for ``total``. These
line-only comparison products end in ``_median_parameters``; the accompanying
parameter CSV records the exact medians and the Gamma/exp onset derived from
them.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import warnings

import numpy as np
import yaml

from collect_mica_results import (
    TF_CODES,
    center_from_kind,
    component_parameter_arrays,
    configured_run_name,
    configured_stages,
    mica_run_root,
    model_kinds,
    model_label,
    ncomp_from_config,
    param_names,
    read_yaml,
    reorder_component_columns,
    stage_config,
)


ROOT = Path(__file__).resolve().parents[1]
QUANTILES = (0.1585, 0.5, 0.8415)  # Match native plotfig.py, including rounding.
GRID_POINTS = 2000
MAX_GRID_POINTS = 12000
ONSET_GRID_POINTS = 96
COLORS = {"comp0": "black", "total": "black"}
LEGACY_GAMMA_STAGE = "mica_gamma2_legacy"


@dataclass
class BandTransfer:
    band: str
    wavelength: float
    kind: str
    nposterior: int
    nswapped: int
    posterior_file: Path
    total_file: Path
    comp0: np.ndarray  # Columns: lag, q15.85, q50, q84.15.
    total: np.ndarray
    median_parameters: tuple["MedianKernel", ...] = ()


@dataclass(frozen=True)
class MedianKernel:
    component: int
    kind: str
    amplitude: float
    centroid: float
    width: float
    center: float


# 计算面积归一的正响应核：积分为 A，纵轴单位为 A/day；w 的意义随核族而变。
# Gaussian 的 w 是标准差；tophat 的 w 是半宽；Gamma(shape=2)/exp 的 w 是尺度。
# 这里 center 为 Gamma/exp 起点，后验统计中的 tau 则是质心。
def kernel(lag, amplitude, center, width, kind: str) -> np.ndarray:
    """Native positive-response kernels; center is the onset for Gamma/exp."""
    delta = np.atleast_1d(lag)[None, :] - np.atleast_1d(center)[:, None]
    w = np.atleast_1d(width)[:, None]
    # A 除以宽度提供 1/day 的归一化；后面的无量纲核形状决定面积仍等于 A。
    scale = np.atleast_1d(amplitude)[:, None] / w
    u = delta / w
    if kind == "gaussian":
        return scale / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * u**2)
    if kind == "tophat":
        return 0.5 * scale * (np.abs(delta) <= w)
    positive_u = np.maximum(u, 0.0)
    if kind == "gamma":
        return scale * positive_u * np.exp(-positive_u)
    if kind in {"exp", "exponential"}:
        return scale * np.exp(-positive_u) * (delta >= 0.0)
    raise ValueError(f"unsupported transfer-function family: {kind}")


# 这里 joint 指同一行的联合参数，不是 run_mica_joint 的多响应联合模型。
# 该 reader 明确要求 4+3*ncomp 列，即一个驱动、一个响应；保存参数需与配置核族一致。
# posterior_sample1d 已经等权重采样，不能再次乘 raw sample 的 weights.txt。
def read_joint_parameters(run: Path, fit: dict):
    """Return validated native joint rows, with whole-component relabelling.

    Read all equal-weight draws; never apply raw-sample weights to this file.
    A malformed draw invalidates the run instead of silently removing it.
    """
    run = Path(run)
    ncomp = ncomp_from_config(fit)
    if ncomp < 1:
        raise ValueError("component count must be positive")
    model = model_label(fit.get("type_tf", "gaussian"))
    kinds = model_kinds(model, ncomp)
    settings = {}
    for line in (run / "param/param_input").read_text().splitlines():
        fields = line.split("#", 1)[0].split()
        if len(fields) >= 2:
            settings[fields[0]] = fields[1]
    type_model = int(settings["TypeModel"])
    if type_model not in (0, 3):
        raise ValueError("only general/mmap runs are supported (not pmap/vmap)")
    if int(settings.get("FlagNegativeResp", "0")) != 0 or fit.get("flag_negative_resp", False):
        raise ValueError("signed responses cannot be plotted on Log y")
    native_kinds = (
        [TF_CODES[code] for code in settings["StrTypeTFMix"]]
        if type_model == 3 else [TF_CODES[settings["TypeTF"]]] * ncomp
    )
    if kinds != native_kinds:
        raise ValueError("configured kernel types differ from the saved MICA run")
    if not int(settings["NumCompLow"]) <= ncomp <= int(settings["NumCompUpp"]):
        raise ValueError("configured component count differs from the saved MICA run")

    names = param_names(run / "data" / f"para_names_line.txt_{ncomp}")
    sample = np.loadtxt(run / "data" / f"posterior_sample1d.txt_{ncomp}", ndmin=2)
    if (
        not sample.size or sample.shape[1] != len(names)
        or len(names) != 4 + 3 * ncomp or len(set(names)) != len(names)
        or not all(names)
    ):
        raise ValueError("expected one driver/response dataset with complete, unique parameter names")
    if not np.isfinite(sample).all():
        raise ValueError("posterior contains nonfinite values; no draws were removed")
    with np.errstate(over="raise", invalid="raise"):
        sample, swapped = reorder_component_columns(sample, names, ncomp, model, fit)
    return sample, names, kinds, int(np.count_nonzero(swapped))


# 共用后验语义转换得到物理面积、起点/中心、尺度和质心，拒绝非有限或非正参数。
def read_component_parameters(run: Path, fit: dict):
    """Return physical kernel arrays from the shared validated joint rows."""
    sample, names, kinds, nswapped = read_joint_parameters(run, fit)
    ncomp = len(kinds)
    model = model_label(fit.get("type_tf", "gaussian"))
    with np.errstate(over="raise", invalid="raise"):
        centers, widths, amplitudes, centroids = component_parameter_arrays(sample, names, ncomp, model)
    if (
        not all(np.isfinite(values).all() for values in (centers, widths, amplitudes, centroids))
        or np.any(widths <= 0.0) or np.any(amplitudes <= 0.0)
    ):
        raise ValueError("posterior kernel parameters must be finite with positive widths/amplitudes")
    return amplitudes, centers, widths, centroids, kinds, nswapped


# 取统计标签 comp0；函数名 disk_grid 等历史命名不构成 comp0 的盘成分认定。
def read_parameters(run: Path, fit: dict):
    """Return comp0 arrays for the pointwise-posterior-quantile figures."""
    amplitudes, centers, widths, _, kinds, nswapped = read_component_parameters(run, fit)
    return amplitudes[0], centers[0], widths[0], kinds[0], nswapped


# legacy Gamma 是只读绘图别名，只重定向历史输出目录，不把历史拟合加入默认执行列表。
def transfer_stage_config(cfg: dict, stage: str) -> dict:
    """Resolve one plotting stage, including the read-only legacy Gamma alias.

    Historical free two-Gamma runs live in ``runs/mica/gamma2/2comp``. Their
    response/ordering metadata is the same as ``mica_gamma2_free``; only the
    saved output root and location-column convention differ. Keeping this as
    an explicit plotting alias avoids adding a runnable legacy fit to every
    source configuration.
    """
    if stage != LEGACY_GAMMA_STAGE:
        return stage_config(cfg, stage)
    fit = dict(stage_config(cfg, "mica_gamma2_free"))
    if model_label(fit.get("type_tf", "gaussian")) != "gamma" or ncomp_from_config(fit) != 2:
        raise ValueError("mica_gamma2_free must describe a two-Gamma model")
    fit["output_root"] = "runs/mica/gamma2"
    return fit


# 原生 total 文件四列是 lag、median、下误差、上误差；转换成下界/中位数/上界。
# 不对核再归一化；文本舍入导致略负的下界仍保存在数据中，仅绘图时受 Log 轴限制。
def read_total(path: Path) -> np.ndarray:
    """Read native lag/median/error columns without resampling or normalizing."""
    values = np.loadtxt(path, ndmin=2)
    if values.shape[1] != 4 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"expected at least two finite rows of four columns: {path}")
    lag, median, lower_error, upper_error = values.T
    if np.any(np.diff(lag) <= 0.0):
        raise ValueError(f"native transfer lag grid is not strictly increasing: {path}")
    if np.any(median < 0.0) or np.any(lower_error < 0.0) or np.any(upper_error < 0.0):
        raise ValueError(f"negative native median or error size: {path}")
    # Rounded native text can give a slightly negative lower bound. Keep it.
    return np.column_stack((lag, median - lower_error, median, median + upper_error))


# 仅为 comp0 构造绘图网格：覆盖每条后验核的有限尾部窗口并包含零/负延迟。
# 宽度分位数控制分辨率而不是筛选样本；点数超上限时用 asinh 网格并明确警告尾部可能欠分辨。
# Gamma 起点处增加局部点，避免对数纵轴把突变附近的形状画错。
def disk_grid(center, width, kind: str) -> np.ndarray:
    """Fine comp0 grid including negative lag and zero, independent of total.

    Bounds include every draw's kernel region. Width quantiles set only grid
    spacing: all posterior draws still enter every pointwise quantile. When a
    uniform grid would exceed the cap, an asinh grid resolves the typical peak
    while retaining the full bounds; the warning makes this compromise explicit.
    Gamma reserves local points around each quantile's zero-to-positive onset,
    resolving its steep rise on Log y without changing the linear X axis.
    """
    center, width = np.atleast_1d(center), np.atleast_1d(width)
    if center.shape != width.shape or not center.size:
        raise ValueError("center/width must be nonempty matching arrays")
    if not np.isfinite(center).all() or not np.isfinite(width).all() or np.any(width <= 0):
        raise ValueError("grid parameters must be finite, with positive widths")
    if kind == "gaussian":
        lower, upper, peak = center - 6 * width, center + 6 * width, center
    elif kind == "tophat":
        lower, upper, peak = center - 1.2 * width, center + 1.2 * width, center
    elif kind == "gamma":
        lower, upper, peak = center - 0.2 * width, center + 12 * width, center + width
    elif kind in {"exp", "exponential"}:
        lower, upper, peak = center - 0.2 * width, center + 12 * width, center
    else:
        raise ValueError(f"unsupported transfer-function family: {kind}")
    scale = float(np.quantile(width, 0.1))
    low = min(float(np.min(lower)), -scale)
    high = max(float(np.max(upper)), scale)
    span = high - low
    if not np.isfinite(span) or span <= 0:
        raise ValueError("comp0 grid has invalid bounds")
    # Interpolated quantiles can involve two onset order statistics each.
    # 点数上限需要预留 Gamma 分位数起点附近的额外网格，不能只限制均匀基础网格。
    reserve = 2 * len(QUANTILES) * (ONSET_GRID_POINTS + 1) if kind == "gamma" else 0
    base_cap = MAX_GRID_POINTS - reserve
    required = span / scale * 25.0
    if required > base_cap - 1:
        warnings.warn(
            f"comp0 uniform grid would exceed {base_cap} base points; using an adaptive "
            f"asinh grid within the {MAX_GRID_POINTS}-point cap "
            "(extreme tails or separated narrow modes may be under-resolved)",
            RuntimeWarning, stacklevel=2,
        )
        pivot = float(np.median(peak))
        transformed = np.linspace(np.arcsinh((low - pivot) / scale),
                                  np.arcsinh((high - pivot) / scale), base_cap)
        grid = pivot + scale * np.sinh(transformed)
        grid[0], grid[-1] = low, high
    else:
        grid = np.linspace(low, high, max(GRID_POINTS, int(np.ceil(required)) + 1))
    # Replace, rather than append, one point so the cap includes exact zero.
    zero_index = int(np.clip(np.argmin(np.abs(grid)), 1, len(grid) - 2))
    grid[zero_index] = 0.0
    grid = np.sort(grid)
    if kind == "gamma":
        # Before an onset the draw contributes an exact zero. Reverse the
        # response-quantile ranks to locate the relevant sorted onset times;
        # interpolating the onset times themselves would miss even-N cases.
        ranks = (len(center) - 1) * np.asarray(QUANTILES)
        indices = (len(center) - 1 - np.r_[np.floor(ranks), np.ceil(ranks)]).astype(int)
        onsets = np.unique(np.sort(center)[indices])
        right = np.minimum(np.searchsorted(grid, onsets, side="right"), len(grid) - 1)
        offsets = (grid[right] - onsets)[:, None] * np.geomspace(1e-12, 1.0, ONSET_GRID_POINTS)
        grid = np.unique(np.concatenate((grid, onsets, (onsets[:, None] + offsets).ravel())))
    return grid


# 在每个 lag 上先求每条后验的核值，再取点态分位数；分块限制内存，不减少样本。
# 这是逐点可信带，不是整条函数同时被包住的概率带；曲线中位数通常也不属于任一真实样本。
def pointwise_quantiles(lag, amp, center, width, kind: str) -> np.ndarray:
    """Return q15.85/q50/q84.15 at each lag using all equal-weight draws."""
    lag = np.asarray(lag, dtype=float)
    amp, center, width = (np.atleast_1d(value) for value in (amp, center, width))
    if not amp.size or amp.shape != center.shape or amp.shape != width.shape:
        raise ValueError("posterior parameter arrays must be nonempty and have matching shapes")
    quantiles = np.empty((len(lag), 3))
    # Limit each temporary sample x lag matrix to about one million doubles.
    # 按 lag 分块，每块仍使用所有后验样本；大致把临时矩阵限制在一百万个浮点数。
    chunk = max(1, min(256, 1_000_000 // len(amp)))
    for start in range(0, len(lag), chunk):
        values = kernel(lag[start:start + chunk], amp, center, width, kind)
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("nonfinite or negative kernel values; no samples were removed")
        quantiles[start:start + chunk] = np.quantile(values, QUANTILES, axis=0).T
    return quantiles


# 另一种展示：分别取 A、tau、w 的边际中位数，再由 tau 与 w 反算核起点。
# 分别取中位数会忽略参数间相关性；这条确定性曲线不是点态后验中位曲线，也不携带可信带。
def median_kernels(amplitudes, centroids, widths, kinds: list[str]) -> tuple[MedianKernel, ...]:
    """Take one independent posterior median for each physical kernel parameter.

    ``centroid`` is the first moment called ``tau`` in the scientific summaries.
    The analytic Gamma/exp kernel is shifted by its onset, so that shift is
    derived only after taking the medians: Gamma onset = tau - 2w and
    exponential onset = tau - w.
    """
    amplitudes = np.asarray(amplitudes, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    widths = np.asarray(widths, dtype=float)
    if (
        amplitudes.ndim != 2 or centroids.shape != amplitudes.shape
        or widths.shape != amplitudes.shape or amplitudes.shape[0] != len(kinds)
        or amplitudes.shape[1] == 0
    ):
        raise ValueError("component posterior arrays must have shape ncomp x nposterior")
    if (
        not np.isfinite(amplitudes).all() or not np.isfinite(centroids).all()
        or not np.isfinite(widths).all() or np.any(amplitudes <= 0.0)
        or np.any(widths <= 0.0)
    ):
        raise ValueError("component posterior parameters must be finite and positive")
    result = []
    for index, kind in enumerate(kinds):
        amplitude = float(np.median(amplitudes[index]))
        centroid = float(np.median(centroids[index]))
        width = float(np.median(widths[index]))
        center = float(center_from_kind(centroid, width, kind))
        result.append(MedianKernel(index, kind, amplitude, centroid, width, center))
    return tuple(result)


# 把每个中位参数核的局部网格与全局网格合并，保留起点、质心和零点。
def median_parameter_grid(parameters: tuple[MedianKernel, ...]) -> np.ndarray:
    """Resolve every median kernel from its onset through its visible tail."""
    if not parameters:
        raise ValueError("at least one median kernel is required")
    local_grids = []
    bounds = []
    special = [0.0]
    for parameter in parameters:
        center, width, kind = parameter.center, parameter.width, parameter.kind
        if kind == "gaussian":
            low, high = center - 6.0 * width, center + 6.0 * width
        elif kind == "tophat":
            low, high = center - 1.2 * width, center + 1.2 * width
        elif kind == "gamma":
            low, high = center - 0.2 * width, center + 20.0 * width
            special.extend(center + width * np.geomspace(1e-8, 1.0, 128))
        elif kind in {"exp", "exponential"}:
            low, high = center - 0.2 * width, center + 16.0 * width
            special.extend(center + width * np.geomspace(1e-8, 1.0, 128))
        else:
            raise ValueError(f"unsupported transfer-function family: {kind}")
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError("median kernel has invalid plotting support")
        bounds.append((low, high))
        local_grids.append(np.linspace(low, high, GRID_POINTS))
        special.extend((center, parameter.centroid))
    low = min(value[0] for value in bounds)
    high = max(value[1] for value in bounds)
    global_grid = np.linspace(low, high, GRID_POINTS)
    grid = np.unique(np.concatenate((*local_grids, global_grid, np.asarray(special))))
    if len(grid) > MAX_GRID_POINTS:
        raise ValueError("median-parameter grid unexpectedly exceeds its fixed cap")
    return grid


# 先用每个分量的边际中位参数生成核，再求和；不是对总核逐点取后验中位数。
def median_parameter_curves(
    parameters: tuple[MedianKernel, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate comp0 and the sum of all median-parameter analytic kernels."""
    lag = median_parameter_grid(parameters)
    components = [
        kernel(
            lag,
            [parameter.amplitude],
            [parameter.center],
            [parameter.width],
            parameter.kind,
        )[0]
        for parameter in parameters
    ]
    if not all(np.isfinite(values).all() and np.all(values >= 0.0) for values in components):
        raise ValueError("median-parameter kernel produced nonfinite or negative values")
    comp0 = components[0]
    total = np.sum(components, axis=0)
    # Reuse the plotting data contract. A single deterministic curve has no
    # lower/upper interval, so all three ordinate columns are identical.
    return (
        np.column_stack((lag, comp0, comp0, comp0)),
        np.column_stack((lag, total, total, total)),
    )


# comp0 从等权后验重新计算点态区间；total 保留原生 tranfunc 文件。
# 两者拥有独立网格，比较前要留意原生 total 网格分辨率和来源是否与后验匹配。
def stage_transfers(source_dir: Path, cfg: dict, fit: dict) -> list[BandTransfer]:
    """Validate and calculate every response before creating any output folder."""
    driver = str(fit.get("driver", ""))
    responses = fit.get("responses")
    if not driver or not isinstance(responses, (list, tuple)) or not responses:
        raise ValueError("stage must define a driver and nonempty responses list")
    if len(set(responses)) != len(responses) or driver in responses:
        raise ValueError("response bands must be unique and exclude the driver")
    wavelengths = {str(band): float(cfg["bands"][band]) for band in responses}
    if any(not np.isfinite(value) or value <= 0.0 for value in wavelengths.values()):
        raise ValueError("response wavelengths must be finite and positive")
    ncomp = ncomp_from_config(fit)
    root = mica_run_root(source_dir, fit)
    result = []
    for band in sorted(wavelengths, key=wavelengths.get):
        run = root / configured_run_name(fit, band)
        amp, center, width, kind, nswapped = read_parameters(run, fit)
        total_file = run / "data" / f"tranfunc_0_1.txt_{ncomp}"
        total = read_total(total_file)
        lag = disk_grid(center, width, kind)
        comp0 = np.column_stack((lag, pointwise_quantiles(lag, amp, center, width, kind)))
        result.append(BandTransfer(
            band, wavelengths[band], kind, len(amp), nswapped,
            run / "data" / f"posterior_sample1d.txt_{ncomp}", total_file, comp0, total,
        ))
    return result


# 中位参数分支只依赖后验与核定义，total 也重新求和，不读取旧原生 total 曲线。
def median_stage_transfers(source_dir: Path, cfg: dict, fit: dict) -> list[BandTransfer]:
    """Build analytic curves from independent posterior-median f, tau, and w."""
    driver = str(fit.get("driver", ""))
    responses = fit.get("responses")
    if not driver or not isinstance(responses, (list, tuple)) or not responses:
        raise ValueError("stage must define a driver and nonempty responses list")
    if len(set(responses)) != len(responses) or driver in responses:
        raise ValueError("response bands must be unique and exclude the driver")
    wavelengths = {str(band): float(cfg["bands"][band]) for band in responses}
    if any(not np.isfinite(value) or value <= 0.0 for value in wavelengths.values()):
        raise ValueError("response wavelengths must be finite and positive")
    ncomp = ncomp_from_config(fit)
    root = mica_run_root(source_dir, fit)
    result = []
    for band in sorted(wavelengths, key=wavelengths.get):
        run = root / configured_run_name(fit, band)
        amplitudes, _, widths, centroids, kinds, nswapped = read_component_parameters(run, fit)
        parameters = median_kernels(amplitudes, centroids, widths, kinds)
        comp0, total = median_parameter_curves(parameters)
        posterior_file = run / "data" / f"posterior_sample1d.txt_{ncomp}"
        result.append(BandTransfer(
            band, wavelengths[band], kinds[0], amplitudes.shape[1], nswapped,
            posterior_file, posterior_file, comp0, total, parameters,
        ))
    return result


# 为同一波段的 comp0/total 设统一对数纵轴，显示峰值以下约六个数量级。
def amplitude_limits(band: BandTransfer) -> tuple[float, float]:
    """Shared per-band log-Y window, about six decades below the peak."""
    maximum = max(float(np.max(curve[:, 3])) for curve in (band.comp0, band.total))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(f"{band.band}: no positive transfer-function quantiles to display")
    lower = 10.0 ** (np.floor(np.log10(maximum)) - 6)
    target = maximum * 1.5
    if lower <= 0.0 or not np.isfinite(target):
        raise ValueError("amplitude range cannot be represented on a log axis")
    scale = 10.0 ** np.floor(np.log10(target))
    upper = next(value * scale for value in (1, 2, 3, 5, 10) if value * scale >= target)
    return float(lower), float(upper)


# 根据当前纵轴可见尾部选择显示窗口；不裁剪 CSV，不把窗外面积当成零。
def lag_limits(curve: np.ndarray, floor: float) -> tuple[float, float]:
    """Frame every visible envelope point, including negative lag and other peaks.

    This selects a display window only. Native grids and saved values are never
    cropped, resampled, smoothed, or normalized. Include a neighbouring grid
    point at either edge so a sharp onset remains visible.
    """
    lag = curve[:, 0]
    visible = np.flatnonzero(curve[:, 3] > floor)
    if not len(visible):
        return float(lag[0]), float(lag[-1])
    low = float(lag[max(0, visible[0] - 1)])
    high = float(lag[min(len(lag) - 1, visible[-1] + 1)])
    if lag[0] <= 0 <= lag[-1]:
        low, high = min(low, 0.0), max(high, 0.0)
    span = high - low
    if span <= 0.0 or not np.isfinite(span):
        return float(lag[0]), float(lag[-1])
    step = 10.0 ** np.floor(np.log10(span))
    low = np.floor((low - 0.04 * span) / step) * step
    high = np.ceil((high + 0.04 * span) / step) * step
    return max(float(lag[0]), float(low)), min(float(lag[-1]), float(high))


# 独立单面板与概览共用同一数据；横轴为观测者系相对 driver 的天数，未除以 (1+z)。
# 对数轴不显示非正中位值，填色下界只为显示截到纵轴下限；导出表仍保留原始数值。
def plot_stage(
    source: str,
    stage: str,
    fit: dict,
    bands: list[BandTransfer],
    output: Path,
    *,
    show_interval: bool = True,
    filename_suffix: str = "",
):
    """Write independent paper-sized comp0/total figures for every band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator, LogLocator, NullFormatter

    limits_by_band = {band.band: amplitude_limits(band) for band in bands}
    output.mkdir(parents=True, exist_ok=True)
    style = {
        "font.family": "serif", "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 9.5,
        "axes.labelsize": 10.5, "axes.titlesize": 10.5,
        "axes.linewidth": 0.7, "axes.edgecolor": "0.20",
        "axes.labelcolor": "0.12", "text.color": "0.12",
        "xtick.color": "0.20", "ytick.color": "0.20",
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    title = re.sub(r"^(Mrk|NGC)(?=\d)", r"\1 ", source)
    with plt.rc_context(style):
        for band in bands:
            limits = limits_by_band[band.band]
            for component in ("comp0", "total"):
                fig, ax = plt.subplots(figsize=(89 / 25.4, 67 / 25.4))
                try:
                    fig.subplots_adjust(left=0.19, right=0.965, bottom=0.205, top=0.885)
                    curve = getattr(band, component)
                    lag, lower, median, upper = curve.T
                    color = COLORS[component]
                    ax.set(xscale="linear", yscale="log", xlim=lag_limits(curve, limits[0]), ylim=limits)
                    if show_interval:
                        ax.fill_between(lag, np.maximum(lower, limits[0]), upper,
                                        where=upper > limits[0], color=color, alpha=0.22,
                                        linewidth=0, interpolate=True, zorder=1)
                    ax.plot(lag, np.where(median > 0.0, median, np.nan), color=color,
                            linewidth=1.25, zorder=3)
                    ax.set_title(f"{title}  |  {band.band}", loc="left", pad=5)
                    ax.set_title(component, loc="right", pad=5, fontsize=9.5)
                    ax.set_xlabel(r"$\tau$ (day)", labelpad=4)
                    ax.set_ylabel(r"$\Psi_0(\tau)$" if component == "comp0" else
                                  r"$\Psi_{\rm tot}(\tau)$", labelpad=3)
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
                    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
                    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[2, 5], numticks=30))
                    ax.yaxis.set_minor_formatter(NullFormatter())
                    ax.tick_params(which="major", direction="in", top=True, right=True,
                                   width=0.7, length=3.2, pad=3)
                    ax.tick_params(which="minor", direction="in", top=True, right=True,
                                   width=0.55, length=1.8)
                    if component == "comp0" and lag[0] < 0 < lag[-1]:
                        ax.axvline(0, color="0.65", linestyle=":", linewidth=0.6, zorder=0)
                    for extension in ("png", "pdf"):
                        fig.savefig(output / f"{source}_{band.band}_{component}_transfer"
                                    f"{filename_suffix}.{extension}",
                                    dpi=300, metadata={"Title": f"{source} {band.band} {component}",
                                                      "Creator": (f"plot_mica_transfer.py; {stage}; "
                                                                  f"interval={show_interval}")})
                finally:
                    plt.close(fig)

        fig, axes = plt.subplots(
            len(bands), 2, squeeze=False, figsize=(180 / 25.4, 235 / 25.4),
        )
        try:
            fig.subplots_adjust(
                left=0.105, right=0.98, bottom=0.07, top=0.95,
                wspace=0.34, hspace=0.42,
            )
            fig.suptitle(title, x=0.105, y=0.985, ha="left", fontsize=12)
            for row, band in enumerate(bands):
                limits = limits_by_band[band.band]
                for column, component in enumerate(("comp0", "total")):
                    ax = axes[row, column]
                    curve = getattr(band, component)
                    lag, lower, median, upper = curve.T
                    color = COLORS[component]
                    ax.set(
                        xscale="linear", yscale="log",
                        xlim=lag_limits(curve, limits[0]), ylim=limits,
                    )
                    if show_interval:
                        ax.fill_between(
                            lag, np.maximum(lower, limits[0]), upper,
                            where=upper > limits[0], color=color, alpha=0.22,
                            linewidth=0, interpolate=True, zorder=1,
                        )
                    ax.plot(
                        lag, np.where(median > 0.0, median, np.nan),
                        color=color, linewidth=1.15, zorder=3,
                    )
                    ax.set_title(band.band, loc="left", pad=3, fontsize=9)
                    ax.set_title(component, loc="right", pad=3, fontsize=9)
                    if row == len(bands) - 1:
                        ax.set_xlabel(r"$\tau$ (day)", labelpad=2)
                    ax.set_ylabel(
                        r"$\Psi_0(\tau)$" if component == "comp0" else
                        r"$\Psi_{\rm tot}(\tau)$", labelpad=2,
                    )
                    ax.xaxis.set_major_locator(
                        MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]),
                    )
                    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
                    ax.yaxis.set_minor_locator(
                        LogLocator(base=10, subs=[2, 5], numticks=24),
                    )
                    ax.yaxis.set_minor_formatter(NullFormatter())
                    ax.tick_params(
                        which="major", direction="in", top=True, right=True,
                        width=0.7, length=3.0, pad=2,
                    )
                    ax.tick_params(
                        which="minor", direction="in", top=True, right=True,
                        width=0.55, length=1.6,
                    )
                    if component == "comp0" and lag[0] < 0 < lag[-1]:
                        ax.axvline(
                            0, color="0.65", linestyle=":",
                            linewidth=0.6, zorder=0,
                        )
            for extension in ("png", "pdf"):
                fig.savefig(
                    output / f"{source}_transfer_functions_overview"
                    f"{filename_suffix}.{extension}",
                    dpi=300,
                    metadata={
                        "Title": f"{source} transfer functions overview",
                        "Creator": (f"plot_mica_transfer.py; {stage}; "
                                    f"interval={show_interval}"),
                    },
                )
        finally:
            plt.close(fig)


# 导出全部网格及来源路径、标签规则、换标号数量；curve_method 区分两种不同曲线定义。
def write_csv(
    source: str,
    stage: str,
    fit: dict,
    bands: list[BandTransfer],
    path: Path,
    *,
    curve_method: str = "pointwise_posterior_quantiles",
):
    fields = [
        "source", "stage", "curve_method", "driver_band", "band", "wavelength_angstrom",
        "component", "kernel",
        "lag_days", "q1585", "q5000", "q8415", "nposterior", "component_order", "nswapped",
        "source_file", "posterior_file", "parameter_names_file", "native_parameter_file",
    ]
    ncomp = ncomp_from_config(fit)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for band in bands:
            run = band.posterior_file.parent.parent
            for component in ("comp0", "total"):
                metadata = dict(
                    source=source, stage=stage, curve_method=curve_method,
                    driver_band=fit["driver"], band=band.band,
                    wavelength_angstrom=band.wavelength, component=component,
                    kernel=band.kind if component == "comp0" else model_label(fit.get("type_tf", "gaussian")),
                    nposterior=band.nposterior, component_order=fit.get("component_order", "native"),
                    nswapped=band.nswapped,
                    source_file=str(band.posterior_file if component == "comp0" else band.total_file),
                    posterior_file=str(band.posterior_file),
                    parameter_names_file=str(run / "data" / f"para_names_line.txt_{ncomp}"),
                    native_parameter_file=str(run / "param" / "param_input"),
                )
                for lag, lower, median, upper in getattr(band, component):
                    writer.writerow(dict(metadata, lag_days=lag, q1585=lower, q5000=median, q8415=upper))


# 记录实际代入解析核的边际中位数，便于从参数重建并核查图线。
def write_median_parameter_summary(
    source: str,
    stage: str,
    fit: dict,
    bands: list[BandTransfer],
    path: Path,
):
    """Write the exact parameter medians used to construct the new curves."""
    fields = [
        "source", "stage", "driver_band", "band", "wavelength_angstrom",
        "component", "kernel", "f_median", "tau_centroid_median_days",
        "width_median_days", "kernel_center_or_onset_days", "nposterior",
        "component_order", "nswapped", "posterior_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for band in bands:
            if not band.median_parameters:
                raise ValueError(f"{band.band}: median parameters are unavailable")
            for parameter in band.median_parameters:
                writer.writerow({
                    "source": source,
                    "stage": stage,
                    "driver_band": fit["driver"],
                    "band": band.band,
                    "wavelength_angstrom": band.wavelength,
                    "component": f"comp{parameter.component}",
                    "kernel": parameter.kind,
                    "f_median": parameter.amplitude,
                    "tau_centroid_median_days": parameter.centroid,
                    "width_median_days": parameter.width,
                    "kernel_center_or_onset_days": parameter.center,
                    "nposterior": band.nposterior,
                    "component_order": fit.get("component_order", "native"),
                    "nswapped": band.nswapped,
                    "posterior_file": str(band.posterior_file),
                })


# 读取已保存结果后批量绘图；不拟合，但会新建/更新图与 CSV。
# 至少一个 source/stage 成功就返回 0，因此批量任务还需查看 [skip]，不能把 0 当作全部成功。
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="*", help="source directory name(s) or paths")
    parser.add_argument("--all", action="store_true", help="all top-level sources with config/source_config.yaml")
    parser.add_argument(
        "--stage", action="append",
        help=("MICA stage; repeat to select several (default: mica_stages). "
              f"Use {LEGACY_GAMMA_STAGE} for saved runs/mica/gamma2 results."),
    )
    parser.add_argument(
        "--median-parameters",
        action="store_true",
        help=("plot a second representation made by inserting each component's "
              "independent posterior-median f, centroid, and width into its analytic kernel; "
              "existing pointwise-quantile figures are not changed"),
    )
    args = parser.parse_args(argv)
    if args.all == bool(args.sources):
        parser.error("choose positional source(s) or --all, but not both")
    sources = (sorted(path for path in ROOT.iterdir() if path.is_dir()
                      and (path / "config/source_config.yaml").is_file()) if args.all else
               list(dict.fromkeys((ROOT / Path(name).expanduser()).resolve() for name in args.sources)))
    successes = 0
    for source_dir in sources:
        try:
            cfg = read_yaml(source_dir / "config/source_config.yaml")
            source = str(cfg.get("source", source_dir.name))
            if Path(source).name != source or source in {"", ".", ".."}:
                raise ValueError("source label must be a safe filename")
            stages = list(dict.fromkeys(args.stage)) if args.stage else configured_stages(cfg)
        except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError) as error:
            print(f"[skip] {source_dir.name}: {error}", file=sys.stderr)
            continue
        for stage in stages:
            try:
                fit = transfer_stage_config(cfg, stage)
                bands = (median_stage_transfers(source_dir, cfg, fit)
                         if args.median_parameters else stage_transfers(source_dir, cfg, fit))
                output = mica_run_root(source_dir, fit) / "result" / "transfer_functions"
                if args.median_parameters:
                    suffix = "_median_parameters"
                    plot_stage(
                        source, stage, fit, bands, output,
                        show_interval=False, filename_suffix=suffix,
                    )
                    write_csv(
                        source, stage, fit, bands,
                        output / f"{source}_transfer_functions{suffix}.csv",
                        curve_method="independent_posterior_median_parameters",
                    )
                    write_median_parameter_summary(
                        source, stage, fit, bands,
                        output / f"{source}_transfer_function_median_parameters.csv",
                    )
                else:
                    plot_stage(source, stage, fit, bands, output)
                    write_csv(source, stage, fit, bands,
                              output / f"{source}_transfer_functions.csv")
            except (OSError, ValueError, TypeError, KeyError, FloatingPointError, SystemExit) as error:
                print(f"[skip] {source}/{stage}: {error}", file=sys.stderr)
                continue
            successes += 1
            method = "median parameters" if args.median_parameters else "posterior quantiles"
            print(f"[ok] {source}/{stage}: {len(bands)} bands ({method}) -> {output}")
    if successes == 0:
        print("No source/stage produced transfer-function plots.", file=sys.stderr)
    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
