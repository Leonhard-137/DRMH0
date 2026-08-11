#!/usr/bin/env python3
"""Utilities for configurable conditional branches of a MICA posterior."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np


BRANCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


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


def posterior_features(centers, widths, amplitudes) -> dict[str, np.ndarray]:
    """Build named scalar features used by branch range selections."""
    centers = np.asarray(centers, float)
    widths = np.asarray(widths, float)
    amplitudes = np.asarray(amplitudes, float)
    if centers.ndim != 2 or widths.shape != centers.shape or amplitudes.shape != centers.shape:
        raise ValueError("centers, widths, and amplitudes must have matching (ncomp, nsample) shapes")
    if centers.shape[0] < 1:
        raise ValueError("at least one transfer-function component is required")

    amp_sum = np.sum(amplitudes, axis=0)
    features: dict[str, np.ndarray] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for index in range(centers.shape[0]):
            features[f"center{index}"] = centers[index]
            features[f"width{index}"] = widths[index]
            features[f"amp{index}"] = amplitudes[index]
            features[f"amp_frac{index}"] = amplitudes[index] / amp_sum
            features[f"q{index}"] = centers[index] / widths[index]

    if centers.shape[0] >= 2:
        separation = centers[1] - centers[0]
        features["center_separation"] = separation
        features["abs_center_separation"] = np.abs(separation)
    return features


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


def effective_sample_size(weights) -> float:
    weights = normalized_weights(weights)
    return float(1.0 / np.sum(weights**2))


def stable_seed_offset(branch_id: str, modulus: int = 1_000_000_007) -> int:
    """Return a reproducible seed offset independent of Python hash randomization."""
    digest = hashlib.sha256(str(branch_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % int(modulus)
