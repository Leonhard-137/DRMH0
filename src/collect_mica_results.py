#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from posterior_branches import (
    branch_config,
    branch_definitions,
    branch_mask,
    branch_result_relative_dir,
    effective_sample_size,
    posterior_features,
)


TF_CODES = {"0": "gaussian", "1": "tophat", "2": "gamma", "3": "exp"}
TF_LABELS = {
    "gaussian": "gaussian",
    "tophat": "tophat",
    "gamma": "gamma",
    "exponential": "exp",
    "exp": "exp",
}
Q68 = np.array([0.15865, 0.5, 0.84135])


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def arr(path: Path) -> np.ndarray:
    x = np.loadtxt(path, comments="#")
    return x.reshape(1, -1) if x.ndim == 1 else x


# 原始 DNest 样本并非等权后验；先校验权重为有限非负值，再归一化到总质量 1。
def normalized_weights(weights) -> np.ndarray:
    weights = np.asarray(weights, float)
    if weights.ndim != 1:
        raise ValueError("posterior weights must be one-dimensional")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("posterior weights must be finite and non-negative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("posterior weights have zero total mass")
    return weights / total


# 按数值稳定排序，用累计正权重构造经验 CDF，再线性插值得分位数。
# 本实现会忽略非有限数值和非正权重；若有效样本全空则报错，不把空集写成一个延迟。
def weighted_quantile(values, weights, quantiles=Q68) -> np.ndarray:
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    if len(values) == 0:
        raise ValueError("no finite, positive-weight posterior samples")
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    cdf = np.cumsum(weights[order])
    cdf /= cdf[-1]
    return np.interp(np.asarray(quantiles, float), cdf, values)


# 输出 median、median-p15.865、p84.135-median，是等尾区间，不是拟合高斯后的标准差。
def q68(values, weights):
    p16, p50, p84 = weighted_quantile(values, weights)
    return float(p50), float(p50 - p16), float(p84 - p50)


def parse_run_name(name: str):
    m = re.match(r"^run_(.+?)_to_(.+?)_(\d+)comp_(.+)$", name)
    return None if m is None else (m.group(1), m.group(2), int(m.group(3)), m.group(4))


def model_label(model: str) -> str:
    model = str(model).strip().lower()
    if model.startswith("mix"):
        codes = model[3:]
        if not codes or any(code not in TF_CODES for code in codes):
            raise ValueError(f"invalid mixed transfer-function label: {model!r}")
        return f"mix{codes}"
    if model.isdigit():
        if any(code not in TF_CODES for code in model):
            raise ValueError(f"invalid mmap transfer-function code: {model!r}")
        return f"mix{model}"
    return TF_LABELS.get(model, model)


def component_order(name: str) -> int:
    if name == "total":
        return 99
    m = re.match(r"tf(\d+)$", str(name))
    return int(m.group(1)) if m else 98


# 把单核族扩展到各分量，或把 mix220 一类标签逐位解码成 Gamma/Gamma/Gaussian。
def model_kinds(model: str, ncomp: int) -> list[str]:
    model = model.lower()
    if model.startswith("mix"):
        codes = model[3:]
        if len(codes) != ncomp or any(code not in TF_CODES for code in codes):
            raise ValueError(f"invalid mixed transfer-function label: {model}")
        return [TF_CODES[c] for c in codes]
    return [model_label(model)] * ncomp


# tau 表示核的一阶矩质心；Gaussian/tophat 的 center 就是质心，
# Gamma(shape=2) 的 tau=center+2w，指数核的 tau=center+w；这里的 w 是尺度参数。
def tau_from_kind(center, width, kind: str):
    if kind in {"gaussian", "tophat"}:
        return center
    if kind == "gamma":
        return center + 2.0 * width
    if kind == "exp":
        return center + width
    return center


# 质心回到解析核的平移参数；Gamma/指数的 center 是起点，不能与延迟质心互换使用。
def center_from_kind(centroid, width, kind: str):
    """Convert a transfer-function centroid back to its onset/shift parameter."""
    if kind in {"gaussian", "tophat"}:
        return centroid
    if kind == "gamma":
        return centroid - 2.0 * width
    if kind == "exp":
        return centroid - width
    return centroid


# 用原生列名区分历史 component_center 和当前 component_centroid，拒绝缺失/重复语义。
# 同一个数值若来自不同列名，物理解释不同，不能只凭列位置猜测。
def component_location_name(names: list[str], index: int) -> str:
    """Return the native lag-column name used by this MICA output.

    Historical MICA outputs sampled ``component_center``.  Current MICA
    samples and writes ``component_centroid``.  The distinct column names let
    one reader preserve the physical meanings of both ``center`` and ``tau``.
    """
    candidates = [
        f"{index}-th_component_centroid",
        f"{index}-th_component_center",
    ]
    found = [name for name in candidates if name in names]
    if len(found) != 1:
        raise ValueError(
            f"component {index} must have exactly one center/centroid column; "
            f"found {found or 'none'}"
        )
    return found[0]


# 输入是每行一个完整联合参数样本；输出四个 (ncomp, nsamples) 矩阵。
# 原生 amplitude 与 sigma 存的是 ln(A)、ln(w)，须先 exp 才得到面积与物理宽度。
# 此读取路径按正响应参数化解释 amplitude；不应直接用于 signed-response 的原生输出。
def component_parameter_arrays(
    sample: np.ndarray,
    names: list[str],
    ncomp: int,
    model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return physical center, width, amplitude, and centroid arrays."""
    kinds = model_kinds(model, ncomp)
    centers, widths, amplitudes, centroids = [], [], [], []
    for index, kind in enumerate(kinds):
        width = np.exp(sample[:, col(names, f"{index}-th_component_sigma")])
        amplitude = np.exp(
            sample[:, col(names, f"{index}-th_component_amplitude")]
        )
        location_name = component_location_name(names, index)
        location = sample[:, col(names, location_name)]
        if location_name.endswith("_centroid"):
            centroid = location
            center = center_from_kind(centroid, width, kind)
        else:
            center = location
            centroid = tau_from_kind(center, width, kind)
        centers.append(center)
        widths.append(width)
        amplitudes.append(amplitude)
        centroids.append(centroid)
    return tuple(np.vstack(values) for values in (
        centers,
        widths,
        amplitudes,
        centroids,
    ))


# 只有显式 component_order=moment1 才逐样本按质心重标号，且限于两个同核族分量。
# 幅度、位置、宽度三列必须成组交换，保留同一样本内相关性；不删样本、不改变权重。
# 编号顺序是统计标签约定，不是盘/BLR 物理分类。
def reorder_component_columns(
    sample: np.ndarray,
    names: list[str],
    ncomp: int,
    model: str,
    fit: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Relabel identical components sample-by-sample by increasing first moment.

    This changes labels only; it never drops a posterior sample. Native MICA
    labels are retained unless component_order: moment1 is explicit.
    """
    mode = str(fit.get("component_order", "native")).strip().lower()
    if mode in {"", "native", "none"}:
        return sample, np.zeros(len(sample), dtype=bool)
    if mode != "moment1":
        raise ValueError(f"unknown MICA component_order: {mode!r}")
    if ncomp != 2:
        raise ValueError("component_order=moment1 currently requires ncomp=2")

    kinds = model_kinds(model, ncomp)
    if kinds[0] != kinds[1]:
        raise ValueError(
            "component_order=moment1 cannot exchange components with different "
            "transfer-function families"
        )
    _, _, _, moments = component_parameter_arrays(sample, names, ncomp, model)
    swapped = np.asarray(moments[0] > moments[1], dtype=bool)
    if not np.any(swapped):
        return sample, swapped

    ordered = sample.copy()
    column_pairs = [
        (
            col(names, "0-th_component_amplitude"),
            col(names, "1-th_component_amplitude"),
        ),
        (
            col(names, component_location_name(names, 0)),
            col(names, component_location_name(names, 1)),
        ),
        (
            col(names, "0-th_component_sigma"),
            col(names, "1-th_component_sigma"),
        ),
    ]
    for left, right in column_pairs:
        temporary = ordered[swapped, left].copy()
        ordered[swapped, left] = ordered[swapped, right]
        ordered[swapped, right] = temporary
    return ordered, swapped


# 按原生整数列索引建立名称表；后续按名称找列，避免硬编码历史 center/新 centroid 的差异。
def param_names(path: Path) -> list[str]:
    names: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) >= 2 and p[0].isdigit():
            i = int(p[0])
            while len(names) <= i:
                names.append("")
            names[i] = p[1]
    return names


def col(names: list[str], name: str) -> int:
    return names.index(name)


# 将 r1/r2 简写映射为 mica_round1/mica_round2，再读命名 stage 的配置。
def stage_config(cfg: dict, stage: str) -> dict:
    # r1 -> mica_round1; r2 -> mica_round2
    m = re.fullmatch(r"r(\d+)", stage)
    section = f"mica_round{m.group(1)}" if m else stage
    fit = dict(cfg.get(section, {}) or {})
    if not fit:
        raise SystemExit(f"missing config section: {section}")
    return fit


def configured_stages(cfg: dict) -> list[str]:
    """Return enabled MICA stages, preserving the order in the YAML file."""
    raw = cfg.get("mica_stages")
    if raw is None:
        return ["mica_round1"]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("mica_stages must be a non-empty list")
    stages = [str(value).strip() for value in raw]
    if any(not value for value in stages) or len(stages) != len(set(stages)):
        raise ValueError("mica_stages must contain unique, non-empty section names")
    missing = [stage for stage in stages if not (cfg.get(stage) or {})]
    if missing:
        raise ValueError("mica_stages references missing sections: " + ", ".join(missing))
    return stages


def ncomp_from_config(fit: dict) -> int:
    if "ncomp" in fit:
        ncomp = int(fit["ncomp"])
    else:
        nc = fit.get("number_component", [2, 2])
        if isinstance(nc, (list, tuple)):
            if len(nc) != 2 or int(nc[0]) != int(nc[1]):
                raise ValueError("number_component must specify one fixed count [n, n]")
            ncomp = int(nc[0])
        else:
            ncomp = int(nc)
    model = model_label(fit.get("type_tf", "gaussian"))
    if model.startswith("mix") and len(model[3:]) != ncomp:
        raise ValueError(
            f"{model} encodes {len(model[3:])} components, but config specifies {ncomp}"
        )
    return ncomp


def mica_run_root(source_dir: Path, fit: dict) -> Path:
    ncomp = ncomp_from_config(fit)
    output_root = Path(fit.get("output_root", "runs/mica")).expanduser()
    if not output_root.is_absolute():
        output_root = source_dir / output_root
    model = model_label(fit.get("type_tf", "gaussian"))
    return output_root if model.startswith("mix") else output_root / f"{ncomp}comp"


def mica_result_dir(source_dir: Path, fit: dict) -> Path:
    out = mica_run_root(source_dir, fit) / "result"
    out.mkdir(parents=True, exist_ok=True)
    return out


def mica_analysis_dir(
    source_dir: Path,
    fit: dict,
    component: str = "comp0",
) -> Path:
    """Return the model-local downstream directory for one native component."""
    component = str(component).strip().lower()
    if not re.fullmatch(r"comp\d+", component):
        raise ValueError(f"invalid MICA component label: {component!r}")
    return mica_run_root(source_dir, fit) / "result" / component


def configured_run_name(fit: dict, response: str) -> str:
    """Render the exact run directory name used by ``run_mica.py``."""
    ncomp = ncomp_from_config(fit)
    model = model_label(fit.get("type_tf", "gaussian"))
    template = str(
        fit.get(
            "run_template",
            "run_{driver}_to_{band}_{ncomp}comp_{type_tf}",
        )
    )
    name = template.format(
        driver=str(fit.get("driver", "")),
        band=str(response),
        response=str(response),
        ncomp=ncomp,
        type_tf=model,
        model=model,
        tf_label=model,
    )
    path = Path(name)
    if (
        not name.strip()
        or name != name.strip()
        or path.is_absolute()
        or len(path.parts) != 1
        or name in {".", ".."}
    ):
        raise ValueError(f"run_template rendered an unsafe directory name: {name!r}")
    return name


# 只收集当前 driver、模型、分量数与响应列表对应的运行目录。
# 根目录若混入其他模型会拒绝，防止静默把旧模型结果与新配置拼表。
def run_dirs(source_dir: Path, fit: dict) -> list[Path]:
    root = mica_run_root(source_dir, fit)
    if not root.is_dir():
        raise FileNotFoundError(f"missing MICA run directory: {root}")
    driver = str(fit.get("driver", ""))
    responses = list(dict.fromkeys(str(band) for band in (fit.get("responses") or [])))
    ncomp = ncomp_from_config(fit)
    model = model_label(fit.get("type_tf", "gaussian"))
    if not driver:
        raise ValueError("MICA config is missing driver")
    if not responses:
        raise ValueError("MICA config is missing responses")

    misplaced = []
    for path in sorted(root.iterdir()):
        info = parse_run_name(path.name) if path.is_dir() else None
        if info is None:
            continue
        run_driver, _, run_ncomp, run_model = info
        if (
            run_driver != driver
            or run_ncomp != ncomp
            or model_label(run_model) != model
        ):
            misplaced.append(path.name)
    if misplaced:
        raise ValueError(
            f"MICA root {root} contains runs inconsistent with the active "
            f"configuration ({driver=}, {ncomp=}, {model=}): "
            + ", ".join(misplaced)
        )

    expected = [root / configured_run_name(fit, band) for band in responses]
    missing = [path for path in expected if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "missing expected MICA run directories:\n"
            + "\n".join(str(path) for path in missing)
        )
    return expected


def add_q(row: dict, prefix: str, values, weights) -> None:
    mid, lo, hi = q68(values, weights)
    row[prefix] = mid
    row[f"{prefix}_err_low"] = lo
    row[f"{prefix}_err_high"] = hi


# 复现 MICA 的每条曲线归一化尺度：通常用均值；任一曲线均值过小时双方用半振幅。
# 面积 A 在归一化光变单位下；gain=A×(response_scale/driver_scale) 才回到两曲线原单位比。
def mica_flux_scales(run_dir: Path) -> tuple[float, float, str]:
    """Reproduce MICA2's per-run flux normalization for physical TF areas."""
    curves = []
    for name in ("cont.txt", "line.txt"):
        path = run_dir / "input" / name
        data = np.loadtxt(path, usecols=(1,))
        values = np.asarray(data, float).reshape(-1)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(f"invalid MICA input light curve: {path}")
        mean = float(np.mean(values))
        half_range = 0.5 * float(np.max(values) - np.min(values))
        if mean <= 0.0 or half_range <= 0.0:
            raise ValueError(
                f"MICA normalization requires positive mean and non-zero range: {path}"
            )
        curves.append((mean, half_range))

    use_half_range = any(mean < half_range / 100.0 for mean, half_range in curves)
    mode = "half_range" if use_half_range else "mean"
    scales = [
        half_range if use_half_range else mean
        for mean, half_range in curves
    ]
    return float(scales[0]), float(scales[1]), mode


# short/delayed 是按分量编号生成的展示标签；native 标签未必已经按质心有序。
# 即使按质心排序，short 也不自动等于真实吸积盘响应。
def role(k: int, ncomp: int) -> str:
    if ncomp == 2:
        return "short" if k == 0 else "delayed"
    return "short" if k == 0 else f"comp{k}"


# 核心汇总：sample1d 原始行 + weights 最后一列 → 标签转换 → 可选条件筛选 → 加权统计。
# 这里读的不是已等权重采样的 posterior_sample1d；两种文件不可混用权重。
# 输出分量统计、先验、tf0 后验、全部分量后验，以及保留质量/权重 ESS 等选择元数据。
def summarize_run(
    run_dir: Path,
    cfg: dict,
    fit: dict,
    *,
    branch_id: str | None = None,
    branch_definition: dict | None = None,
    force_all: bool = False,
):
    drive, band, ncomp, raw_model = parse_run_name(run_dir.name)
    model = model_label(raw_model)
    kinds = model_kinds(model, ncomp)
    data = run_dir / "data"
    sample = arr(data / f"sample1d.txt_{ncomp}")
    weight_table = arr(data / "weights.txt")
    weights = weight_table[:, -1]
    if len(sample) != len(weights):
        raise ValueError(
            f"sample/weight length mismatch in {run_dir}: "
            f"{len(sample)} != {len(weights)}"
        )
    weights = normalized_weights(weights)
    names = param_names(data / f"para_names_line.txt_{ncomp}")
    sample, label_swapped = reorder_component_columns(
        sample, names, ncomp, model, fit
    )
    centers_matrix, widths_matrix, amplitudes_matrix, centroids_matrix = (
        component_parameter_arrays(sample, names, ncomp, model)
    )
    driver_scale, response_scale, normalization_mode = mica_flux_scales(run_dir)
    scale_ratio = response_scale / driver_scale
    wave = float(cfg["bands"][band])

    sample_ids = np.arange(len(sample))
    q_max = None if force_all or branch_definition is not None else fit.get("posterior_q_max")
    # 保留质量使用筛选前的全后验权重；筛选后 weights 是条件分布，二者含义不同。
    retained_weight = 1.0
    selection = "all"
    keep = np.ones(len(sample), dtype=bool)
    if branch_definition is not None:
        features = posterior_features(
            centers_matrix,
            widths_matrix,
            amplitudes_matrix,
            centroids=centroids_matrix,
        )
        keep = branch_mask(str(branch_id), branch_definition, features)
        retained_weight = float(np.sum(weights[keep]))
        if retained_weight <= 0.0:
            raise ValueError(
                f"posterior branch {branch_id!r} has zero posterior mass in {run_dir}"
            )
        sample = sample[keep]
        sample_ids = sample_ids[keep]
        label_swapped = label_swapped[keep]
        weights = normalized_weights(weights[keep])
        selection = str(branch_id)
    elif q_max is not None:
        if model != "gaussian":
            raise ValueError("posterior_q_max is only defined for Gaussian runs")
        q_max = float(q_max)
        # 旧 Gaussian q0 筛选是 center/width 的形状条件，不是独立的盘成分证据。
        q = centers_matrix[0] / widths_matrix[0]
        keep = np.isfinite(q) & (q < q_max)
        retained_weight = float(np.sum(weights[keep]))
        if retained_weight <= 0.0:
            raise ValueError(f"q<{q_max:g} removed all posterior mass in {run_dir}")
        sample = sample[keep]
        sample_ids = sample_ids[keep]
        label_swapped = label_swapped[keep]
        weights = normalized_weights(weights[keep])
        selection = f"q0<{q_max:g}"

    centers_matrix = centers_matrix[:, keep]
    widths_matrix = widths_matrix[:, keep]
    amplitudes_matrix = amplitudes_matrix[:, keep]
    centroids_matrix = centroids_matrix[:, keep]

    conditional_ess = effective_sample_size(weights)
    label_order = str(fit.get("component_order", "native"))
    label_swap_fraction = float(np.mean(label_swapped)) if len(label_swapped) else 0.0
    label_swap_mass = float(np.sum(weights[label_swapped]))

    metadata = {
        "run_name": run_dir.name,
        "driver_band": drive,
        "model": model,
        "ncomp": ncomp,
        "selection": selection,
        "retained_weight": retained_weight,
        "conditional_ess": conditional_ess,
        "label_order": label_order,
        "label_swap_fraction": label_swap_fraction,
        "label_swap_mass": label_swap_mass,
        "normalization_mode": normalization_mode,
        "driver_scale": driver_scale,
        "response_scale": response_scale,
        "response_to_driver_scale": scale_ratio,
    }

    amps = [amplitudes_matrix[k] for k in range(ncomp)]
    centers = [centers_matrix[k] for k in range(ncomp)]
    widths = [widths_matrix[k] for k in range(ncomp)]
    taus = [centroids_matrix[k] for k in range(ncomp)]

    amp_sum = np.sum(np.vstack(amps), axis=0)
    # 把原生归一化面积换回 response/driver 的物理单位比；并未做红移或盘模型换算。
    gains = [amp * scale_ratio for amp in amps]
    gain_sum = amp_sum * scale_ratio
    rows = []
    for k in range(ncomp):
        row = {
            **metadata,
            "band": band,
            "lambda_obs": wave,
            "component": f"tf{k}",
            "role": role(k, ncomp),
            "tf_kind": kinds[k],
        }
        add_q(row, "tau", taus[k], weights)
        add_q(row, "width", widths[k], weights)
        add_q(row, "amp", amps[k], weights)
        add_q(row, "gain", gains[k], weights)
        add_q(row, "amp_frac", amps[k] / amp_sum, weights)
        rows.append(row)

    # 混合核总质心须逐样本先求 sum(A_i*tau_i)/sum(A_i)，再统计分位数；不能混合各列中位数。
    total_tau = np.sum(np.vstack(amps) * np.vstack(taus), axis=0) / amp_sum
    row = {
        **metadata,
        "band": band,
        "lambda_obs": wave,
        "component": "total",
        "role": "total",
        "tf_kind": "mixture" if ncomp > 1 else kinds[0],
    }
    add_q(row, "tau", total_tau, weights)
    row.update({"width": np.nan, "width_err_low": np.nan, "width_err_high": np.nan})
    add_q(row, "amp", amp_sum, weights)
    add_q(row, "gain", gain_sum, weights)
    row.update({"amp_frac": 1.0, "amp_frac_err_low": 0.0, "amp_frac_err_high": 0.0})
    rows.append(row)

    tf0_center = centers[0]
    tf0_width = widths[0]
    tf0_tau = taus[0]
    posterior = [
        {
            **metadata,
            "band": band,
            "lambda_obs": wave,
            "sample": int(sample_ids[i]),
            "tf_kind": kinds[0],
            "center": float(tf0_center[i]),
            "width": float(tf0_width[i]),
            "tau": float(tf0_tau[i]),
            "amp": float(amps[0][i]),
            "gain": float(gains[0][i]),
            "weight": float(weights[i]),
        }
        for i in range(len(sample))
    ]

    component_posterior = []
    amp_frac = np.vstack(amps) / amp_sum
    for k in range(ncomp):
        component_posterior.extend([
            {
                **metadata,
                "band": band,
                "lambda_obs": wave,
                "sample": int(sample_ids[i]),
                "component": f"tf{k}",
                "role": role(k, ncomp),
                "tf_kind": kinds[k],
                "center": float(centers[k][i]),
                "width": float(widths[k][i]),
                "tau": float(taus[k][i]),
                "amp": float(amps[k][i]),
                "gain": float(gains[k][i]),
                "amp_frac": float(amp_frac[k, i]),
                "weight": float(weights[i]),
            }
            for i in range(len(sample))
        ])

    priors = read_applied_priors(
        data / "table_priors_applied.csv",
        band,
        native_path=data / f"para_names_line.txt_{ncomp}",
    )
    for prior in priors:
        match = re.fullmatch(r"tf(\d+)", str(prior["component"]))
        if match:
            prior["tf_kind"] = kinds[int(match.group(1))]
        prior.update({key: metadata[key] for key in ("run_name", "driver_band", "model", "ncomp")})
    selection_row = {
        **metadata,
        "band": band,
        "lambda_obs": wave,
        "n_raw": len(weight_table),
        "n_selected": len(sample),
        "posterior_q_max": q_max,
        "conditional_ess": conditional_ess,
        "label_order": label_order,
        "label_swap_fraction": label_swap_fraction,
        "label_swap_mass": label_swap_mass,
    }
    return rows, priors, posterior, component_posterior, selection_row


# 优先读实际先验表，缺失时从原生参数名文件恢复边界；并将 ln(A)、ln(w) 边界 exp。
# center 与 tau 分开记录，避免把旧 Gamma 起点先验误标成质心先验。
def read_applied_priors(
    path: Path,
    band: str,
    *,
    native_path: Path | None = None,
) -> list[dict]:
    if not path.exists():
        if native_path is None or not native_path.exists():
            return []
        out = []
        for line in native_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            fields = line.split()
            if len(fields) < 5 or not fields[0].isdigit():
                continue
            name = fields[1]
            match = re.fullmatch(
                r"(\d+)-th_component_(amplitude|center|centroid|sigma)",
                name,
            )
            if not match:
                continue
            index, raw = int(match.group(1)), match.group(2)
            low, high = float(fields[3]), float(fields[4])
            quantity = {
                "amplitude": "amp",
                "center": "center",
                "centroid": "tau",
                "sigma": "width",
            }[raw]
            if quantity in {"amp", "width"}:
                low, high = np.exp(low), np.exp(high)
            out.append({
                "band": band,
                "component": f"tf{index}",
                "quantity": quantity,
                "prior_low": low,
                "prior_high": high,
            })
        return out
    out = []
    df = pd.read_csv(path)
    for r in df.itertuples(index=False):
        name = str(r.param_name)
        m = re.match(
            r"(\d+)-th_component_(amplitude|center|centroid|sigma)$",
            name,
        )
        if not m:
            continue
        k, raw = int(m.group(1)), m.group(2)
        quantity = {
            "amplitude": "amp",
            "center": "center",
            "centroid": "tau",
            "sigma": "width",
        }[raw]
        lo, hi = float(r.prior_min), float(r.prior_max)
        if quantity in {"amp", "width"}:
            lo, hi = np.exp(lo), np.exp(hi)
        out.append({"band": band, "component": f"tf{k}", "quantity": quantity, "prior_low": lo, "prior_high": hi})
    return out


COMPONENT_SUMMARY_COLUMNS = [
    "run_name", "driver_band", "band", "model", "ncomp",
    "selection", "retained_weight", "conditional_ess", "lambda_obs",
    "component", "role", "tf_kind",
    "normalization_mode", "driver_scale", "response_scale",
    "response_to_driver_scale",
    "tau", "tau_err_low", "tau_err_high",
    "width", "width_err_low", "width_err_high",
    "amp", "amp_err_low", "amp_err_high",
    "gain", "gain_err_low", "gain_err_high",
    "amp_frac", "amp_frac_err_low", "amp_frac_err_high",
]
TF0_POSTERIOR_COLUMNS = [
    "run_name", "driver_band", "band", "model", "ncomp",
    "selection", "retained_weight", "conditional_ess", "lambda_obs", "sample",
    "normalization_mode", "driver_scale", "response_scale",
    "response_to_driver_scale",
    "tf_kind", "center", "width", "tau", "amp", "gain", "weight",
]
COMPONENT_POSTERIOR_COLUMNS = [
    "run_name", "driver_band", "band", "model", "ncomp",
    "selection", "retained_weight", "conditional_ess", "lambda_obs", "sample",
    "normalization_mode", "driver_scale", "response_scale",
    "response_to_driver_scale",
    "component", "role", "tf_kind", "center", "width", "tau", "amp", "gain",
    "amp_frac", "weight",
]


def component_summary_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COMPONENT_SUMMARY_COLUMNS)
    frame["_order"] = frame["component"].map(component_order)
    return frame.sort_values(["lambda_obs", "_order"]).drop(columns="_order")


def tf0_posterior_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=TF0_POSTERIOR_COLUMNS).sort_values(
        ["lambda_obs", "sample"]
    )


def component_posterior_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COMPONENT_POSTERIOR_COLUMNS)
    frame["_order"] = frame["component"].map(component_order)
    return frame.sort_values(
        ["lambda_obs", "sample", "_order"]
    ).drop(columns="_order")


# 结果契约：mica_component_summary.csv 给每波段/分量统计，
# mica_tf0_posterior.txt 给下游短分量样本；mica_component_posterior.txt 保留全分量逐样本关联。
def save_collection(
    out_dir: Path,
    comp_rows: list[dict],
    posterior_rows: list[dict],
    component_posterior_rows: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    comp = component_summary_frame(comp_rows)
    posterior = tf0_posterior_frame(posterior_rows)
    component_posterior = component_posterior_frame(component_posterior_rows)
    comp.to_csv(out_dir / "mica_component_summary.csv", index=False, na_rep="")
    posterior.to_csv(
        out_dir / "mica_tf0_posterior.txt",
        sep=" ",
        index=False,
        float_format="%.10e",
    )
    component_posterior.to_csv(
        out_dir / "mica_component_posterior.txt",
        sep=" ",
        index=False,
        float_format="%.10e",
    )
    return comp, posterior, component_posterior


# 低分支质量与低权重 ESS 只生成诊断标记，不会自动拒绝或提高分支可信度。
def selection_status(parent_mass: float, ess: float, config: dict) -> str:
    flags = []
    if parent_mass < float(config["low_mass_warning"]):
        flags.append("low_mass")
    if ess < float(config["low_ess_warning"]):
        flags.append("low_ess")
    return "+".join(flags) if flags else "ok"


# 按指定参数区域构造条件后验，并把分支产物放进独立子目录。
# 分支内权重重新归一化；parent_posterior_mass 仍记录它在完整后验中的原始质量。
def collect_branch(
    out_dir: Path,
    runs: list[Path],
    cfg: dict,
    fit: dict,
    branch_id: str,
    definition: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect one conditional branch without rewriting the full-posterior products."""
    branch_comp_rows, branch_post_rows, branch_component_rows = [], [], []
    branch_selection_rows = []
    for rd in runs:
        comp, _, post, component_post, selected = summarize_run(
            rd,
            cfg,
            fit,
            branch_id=branch_id,
            branch_definition=definition,
        )
        branch_comp_rows.extend(comp)
        branch_post_rows.extend(post)
        branch_component_rows.extend(component_post)
        branch_selection_rows.append(selected)

    branch_dir = out_dir / branch_result_relative_dir(fit, branch_id)
    _, branch_posterior, _ = save_collection(
        branch_dir,
        branch_comp_rows,
        branch_post_rows,
        branch_component_rows,
    )
    config = branch_config(fit)
    summary_rows = []
    for row in branch_selection_rows:
        parent_mass = float(row["retained_weight"])
        ess = float(row["conditional_ess"])
        summary_rows.append({
            "branch_id": branch_id,
            "label": definition["label"],
            "band": row["band"],
            "lambda_obs_A": row["lambda_obs"],
            "parent_posterior_mass": parent_mass,
            "n_raw": int(row["n_raw"]),
            "n_selected": int(row["n_selected"]),
            "conditional_ess": ess,
            "label_order": row["label_order"],
            "label_swap_fraction": row["label_swap_fraction"],
            "label_swap_mass": row["label_swap_mass"],
            "status": selection_status(parent_mass, ess, config),
        })
    branch_summary = pd.DataFrame(summary_rows).sort_values("lambda_obs_A")
    branch_summary.to_csv(branch_dir / "branch_summary.csv", index=False)
    with (branch_dir / "branch_definition.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "branch_id": branch_id,
                "label": definition["label"],
                "ranges": definition["ranges"],
            },
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
    return branch_summary, branch_posterior


# 入口默认同时保存当前选择与 *_all 完整后验；启用分支时再逐分支汇总。
# 额外统计未分配质量和重叠质量，提醒配置的多个区域未必组成互斥且完备的划分。
def main() -> None:
    ap = argparse.ArgumentParser(description="Collect compact MICA component and prior tables.")
    ap.add_argument("source_dir", type=Path)
    ap.add_argument("stage", nargs="?", default="r1", help="r1, r2, or a config section name")
    ap.add_argument(
        "--branch",
        help="collect one configured posterior branch without rewriting full-posterior products",
    )
    args = ap.parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    cfg = read_yaml(source_dir / "config/source_config.yaml")
    fit = stage_config(cfg, args.stage)
    out_dir = mica_result_dir(source_dir, fit)

    runs = run_dirs(source_dir, fit)
    definitions = branch_definitions(fit)
    if args.branch:
        if args.branch not in definitions:
            available = ", ".join(definitions) if definitions else "none"
            raise ValueError(
                f"unknown or disabled posterior branch {args.branch!r}; "
                f"available: {available}"
            )
        branch_summary, _ = collect_branch(
            out_dir,
            runs,
            cfg,
            fit,
            args.branch,
            definitions[args.branch],
        )
        branch_dir = out_dir / branch_result_relative_dir(fit, args.branch)
        print(branch_dir / "mica_component_summary.csv")
        print(branch_dir / "mica_tf0_posterior.txt")
        print(branch_dir / "mica_component_posterior.txt")
        print(branch_dir / "branch_summary.csv")
        return

    comp_rows, prior_rows, posterior_rows, component_posterior_rows, selection_rows = [], [], [], [], []
    for rd in runs:
        comp, pri, post, component_post, selection = summarize_run(rd, cfg, fit)
        comp_rows.extend(comp)
        prior_rows.extend(pri)
        posterior_rows.extend(post)
        component_posterior_rows.extend(component_post)
        selection_rows.append(selection)

    save_collection(out_dir, comp_rows, posterior_rows, component_posterior_rows)

    prior_cols = [
        "run_name", "driver_band", "band", "model", "ncomp",
        "component", "tf_kind", "quantity", "prior_low", "prior_high",
    ]
    pri = pd.DataFrame(prior_rows, columns=prior_cols)
    if len(pri):
        pri["_order"] = pri["component"].map(component_order)
        pri = pri.sort_values(["band", "_order", "quantity"]).drop(columns="_order")
    pri.to_csv(out_dir / "mica_prior_summary.csv", index=False, na_rep="")

    legacy_selection_columns = [
        "run_name", "driver_band", "model", "ncomp", "selection",
        "retained_weight", "band", "lambda_obs", "n_raw", "n_selected",
        "posterior_q_max", "conditional_ess", "label_order",
        "label_swap_fraction", "label_swap_mass",
    ]
    selection = pd.DataFrame(selection_rows)[legacy_selection_columns].sort_values("lambda_obs")
    selection.to_csv(out_dir / "mica_posterior_selection.csv", index=False)

    all_comp_rows, all_posterior_rows, all_component_rows = [], [], []
    for rd in runs:
        comp, _, post, component_post, _ = summarize_run(rd, cfg, fit, force_all=True)
        all_comp_rows.extend(comp)
        all_posterior_rows.extend(post)
        all_component_rows.extend(component_post)
    all_posterior = tf0_posterior_frame(all_posterior_rows)
    all_component_posterior = component_posterior_frame(all_component_rows)
    all_posterior.to_csv(
        out_dir / "mica_tf0_posterior_all.txt",
        sep=" ",
        index=False,
        float_format="%.10e",
    )
    all_component_posterior.to_csv(
        out_dir / "mica_component_posterior_all.txt",
        sep=" ",
        index=False,
        float_format="%.10e",
    )

    if definitions:
        branches_root = out_dir / "branches"
        branches_root.mkdir(parents=True, exist_ok=True)
        config = branch_config(fit)
        branch_summaries: dict[str, pd.DataFrame] = {}
        branch_tf0: dict[str, pd.DataFrame] = {}

        for branch_id, definition in definitions.items():
            branch_summary, branch_posterior = collect_branch(
                out_dir,
                runs,
                cfg,
                fit,
                branch_id,
                definition,
            )
            branch_tf0[branch_id] = branch_posterior
            branch_summaries[branch_id] = branch_summary

        catalog_rows = []
        for branch_id, summary in branch_summaries.items():
            masses = summary["parent_posterior_mass"].to_numpy(float)
            ess = summary["conditional_ess"].to_numpy(float)
            warnings_found = sorted(set(summary.loc[summary["status"].ne("ok"), "status"]))
            catalog_rows.append({
                "branch_id": branch_id,
                "label": definitions[branch_id]["label"],
                "n_band": len(summary),
                "min_parent_mass": np.min(masses),
                "median_parent_mass": np.median(masses),
                "max_parent_mass": np.max(masses),
                "min_conditional_ess": np.min(ess),
                "status": "+".join(warnings_found) if warnings_found else "ok",
            })
        pd.DataFrame(catalog_rows).to_csv(branches_root / "branch_catalog.csv", index=False)

        partition_rows = []
        for band, all_band in all_posterior.groupby("band", sort=False):
            all_band = all_band.sort_values("sample")
            sample_ids = all_band["sample"].to_numpy(int)
            weights = all_band["weight"].to_numpy(float)
            # 逐原始 sample ID 追踪分支归属次数：0=遗漏，1=唯一归属，>1=分支重叠。
            membership = np.zeros(len(all_band), dtype=int)
            for branch_id, posterior in branch_tf0.items():
                selected_ids = set(
                    posterior.loc[posterior["band"].astype(str).eq(str(band)), "sample"].astype(int)
                )
                membership += np.fromiter(
                    (sample_id in selected_ids for sample_id in sample_ids),
                    dtype=bool,
                    count=len(sample_ids),
                )
            assigned = float(np.sum(weights[membership > 0]))
            overlap = float(np.sum(weights[membership > 1]))
            partition_rows.append({
                "band": band,
                "lambda_obs_A": float(all_band["lambda_obs"].iloc[0]),
                "assigned_union_mass": assigned,
                "overlapping_mass": overlap,
                "unassigned_mass": float(max(0.0, 1.0 - assigned)),
                "max_membership": int(np.max(membership)),
            })
        pd.DataFrame(partition_rows).sort_values("lambda_obs_A").to_csv(
            branches_root / "branch_partition_summary.csv",
            index=False,
        )

    print(out_dir / "mica_component_summary.csv")
    print(out_dir / "mica_prior_summary.csv")
    print(out_dir / "mica_tf0_posterior.txt")
    print(out_dir / "mica_component_posterior.txt")
    print(out_dir / "mica_posterior_selection.csv")
    print(out_dir / "mica_tf0_posterior_all.txt")
    print(out_dir / "mica_component_posterior_all.txt")
    if definitions:
        print(out_dir / "branches" / "branch_catalog.csv")
        print(out_dir / "branches" / "branch_partition_summary.csv")


if __name__ == "__main__":
    main()
