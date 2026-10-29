#!/usr/bin/env python3
"""Utilities for configurable conditional branches of a MICA posterior."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np


BRANCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


# 为条件后验分析提供默认开关、重采样数、随机种子和低质量提示阈值。
# 这些参数控制后处理，不改变原始 MICA 拟合或先验。
def branch_config(fit: Mapping) -> dict:
    """Return a normalized posterior-branch configuration."""
    config = dict(fit.get("posterior_branches", {}) or {})
    config.setdefault("enabled", False)
    config.setdefault("seed", 12345)
    config.setdefault("decompose_nsamp", 3000)
    config.setdefault("low_mass_warning", 1.0e-4)
    config.setdefault("low_ess_warning", 20.0)
    config.setdefault("branches", {})
    return config


# 只接纳启用且带非空 ranges 的命名区域；安全 ID 同时用于输出目录名。
def branch_definitions(fit: Mapping) -> dict[str, dict]:
    """Validate and return enabled branch definitions in YAML order."""
    config = branch_config(fit)
    if not bool(config["enabled"]):
        return {}

    raw = config.get("branches", {}) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("posterior_branches.branches must be a mapping")

    definitions: dict[str, dict] = {}
    for raw_id, raw_definition in raw.items():
        branch_id = str(raw_id)
        if BRANCH_ID_PATTERN.fullmatch(branch_id) is None:
            raise ValueError(
                f"invalid posterior branch id {branch_id!r}; use letters, numbers, '.', '_' or '-'"
            )
        definition = dict(raw_definition or {})
        if not bool(definition.get("enabled", True)):
            continue
        ranges = definition.get("ranges", {}) or {}
        if not isinstance(ranges, Mapping) or not ranges:
            raise ValueError(
                f"posterior branch {branch_id!r} must define a non-empty ranges mapping"
            )
        definition["label"] = str(definition.get("label", branch_id))
        definition["ranges"] = dict(ranges)
        definitions[branch_id] = definition
    return definitions


# 生成受限于结果目录内部的相对路径，拒绝绝对路径和父目录跳转。
def branch_result_relative_dir(fit: Mapping, branch_id: str) -> Path:
    """Return a safe path relative to the active MICA result directory."""
    config = branch_config(fit)
    template = str(config.get("result_dir_template", "branches/{branch_id}"))
    ncomp_value = fit.get("ncomp", fit.get("number_component", 2))
    if isinstance(ncomp_value, (list, tuple)):
        ncomp_value = ncomp_value[0]
    rendered = template.format(
        branch_id=str(branch_id),
        ncomp=int(ncomp_value),
        model=str(fit.get("type_tf", "gaussian")).lower(),
    )
    path = Path(rendered)
    if not rendered.strip() or path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "posterior_branches.result_dir_template must render to a non-empty "
            "relative path without '..'"
        )
    return path


# 把 (ncomp, nsamples) 参数矩阵展开为可筛选变量：center、width、A、A/sum(A)、q=center/width。
# 可选 tau 是真实核质心；双分量还提供起点差/中心差及绝对值，不能把它们默认解释为质心差。
def posterior_features(
    centers,
    widths,
    amplitudes,
    centroids=None,
) -> dict[str, np.ndarray]:
    """Build named scalar features used by branch range selections."""
    centers = np.asarray(centers, float)
    widths = np.asarray(widths, float)
    amplitudes = np.asarray(amplitudes, float)
    if centers.ndim != 2 or widths.shape != centers.shape or amplitudes.shape != centers.shape:
        raise ValueError("centers, widths, and amplitudes must have matching (ncomp, nsample) shapes")
    if centers.shape[0] < 1:
        raise ValueError("at least one transfer-function component is required")
    if centroids is not None:
        centroids = np.asarray(centroids, float)
        if centroids.shape != centers.shape:
            raise ValueError("centroids must match the component-array shape")

    amp_sum = np.sum(amplitudes, axis=0)
    features: dict[str, np.ndarray] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for index in range(centers.shape[0]):
            features[f"center{index}"] = centers[index]
            features[f"width{index}"] = widths[index]
            features[f"amp{index}"] = amplitudes[index]
            features[f"amp_frac{index}"] = amplitudes[index] / amp_sum
            features[f"q{index}"] = centers[index] / widths[index]
            if centroids is not None:
                features[f"tau{index}"] = centroids[index]

    if centers.shape[0] >= 2:
        separation = centers[1] - centers[0]
        features["center_separation"] = separation
        features["abs_center_separation"] = np.abs(separation)
    return features


# 允许 [low, high] 或命名边界形式；缺一侧代表半无限区间，边界仍须是有限数。
def _range_bounds(branch_id: str, feature: str, value) -> tuple[float | None, float | None]:
    if isinstance(value, Mapping):
        unexpected = sorted(set(value).difference({"min", "max"}))
        if unexpected:
            raise ValueError(
                f"posterior branch {branch_id!r}, feature {feature!r} has unknown keys: "
                + ", ".join(unexpected)
            )
        low, high = value.get("min"), value.get("max")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        low, high = value
    else:
        raise ValueError(
            f"posterior branch {branch_id!r}, feature {feature!r} must use [min, max] "
            "or {min: ..., max: ...}"
        )

    low = None if low is None else float(low)
    high = None if high is None else float(high)
    if low is not None and not np.isfinite(low):
        raise ValueError(f"non-finite lower bound for {branch_id}.{feature}")
    if high is not None and not np.isfinite(high):
        raise ValueError(f"non-finite upper bound for {branch_id}.{feature}")
    if low is not None and high is not None and low >= high:
        raise ValueError(f"posterior branch {branch_id!r} requires min < max for {feature!r}")
    return low, high


# 一个分支内各条件取逻辑 AND；每项使用 low<=x<high，且要求被筛选变量有限。
# 不同分支之间没有互斥约束，所以总质量可能重叠；未被 ranges 使用的变量不会在此做有限性筛除。
def branch_mask(
    branch_id: str,
    definition: Mapping,
    features: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Apply inclusive lower and exclusive upper range cuts."""
    lengths = {len(np.asarray(values)) for values in features.values()}
    if len(lengths) != 1:
        raise ValueError("posterior feature arrays do not have a common length")
    size = lengths.pop()
    mask = np.ones(size, dtype=bool)
    for feature, bounds in definition["ranges"].items():
        feature = str(feature)
        if feature not in features:
            raise ValueError(
                f"posterior branch {branch_id!r} uses unknown feature {feature!r}; "
                f"available features: {', '.join(sorted(features))}"
            )
        values = np.asarray(features[feature], float)
        low, high = _range_bounds(branch_id, feature, bounds)
        selected = np.isfinite(values)
        if low is not None:
            selected &= values >= low
        if high is not None:
            selected &= values < high
        mask &= selected
    return mask


# 条件区域内重新把保留权重归一化；拒绝负权重、非有限权重及零总质量。
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


# 权重集中度 ESS=1/sum(w_i^2)；函数先把输入权重归一化，再代入公式。
# 它不是考虑链自相关后的 MCMC ESS，也不能用来单独证明收敛。
def effective_sample_size(weights) -> float:
    weights = normalized_weights(weights)
    return float(1.0 / np.sum(weights**2))


# 用 SHA256 而非 Python hash 生成稳定偏移，使同一分支/波段跨进程运行仍可复现重采样。
def stable_seed_offset(branch_id: str, modulus: int = 1_000_000_007) -> int:
    """Return a reproducible seed offset independent of Python hash randomization."""
    digest = hashlib.sha256(str(branch_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % int(modulus)
