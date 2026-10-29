#!/usr/bin/env python3
# 留出U波段的差分SED诊断：每个FFA联合样本内拟合其余波段并预测U。
# 总光和短分量FFA是独立推断，不按相同样本编号相减；过量是描述性模型比较量。
# 本入口读 Delta_F；论文表另有基于R的 u_rms_diagnostic，比较时须核对量和权重。
"""Build leave-U-out SED diagnostics from existing joint FFA posteriors.

For each of the total-light and official short-component FFA products, this
postprocessor fits the non-U values of ``S = lambda * Delta_F`` in every joint
posterior draw.  The U-band value is then predicted under a standard
``alpha=-4/3`` power law and a free-alpha power law.

The two FFA fits have independent posterior samples.  Sample identifiers are
therefore retained only within their own stage, and this script deliberately
does not construct a draw-by-draw total-minus-short comparison.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from collect_mica_results import (
    configured_stages,
    mica_analysis_dir,
    stage_config,
)


MODELS = {
    "standard": -4.0 / 3.0,
    "free": None,
}


# 记录一套FFA后验的身份、波段、波长和输入路径，防止跨stage串用。
@dataclass(frozen=True)
class StageInput:
    stage: str
    label: str
    component: str
    bands: tuple[str, ...]
    wavelengths: dict[str, float]
    posterior_path: Path
    components_path: Path


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def rel(source_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else source_dir / path


def ncomp_from_config(mica: dict) -> int:
    if "ncomp" in mica:
        return int(mica["ncomp"])
    value = mica.get("number_component", 2)
    if isinstance(value, (list, tuple)):
        value = value[0]
    return int(value)


def component_name(section: dict, default: str) -> str:
    return str((section.get("component", {}) or {}).get("name", default))


def configured_bands(section: dict, cfg: dict) -> tuple[str, ...]:
    values = section.get("bands")
    if values is None:
        values = cfg.get("band_list")
    if values is None:
        values = list((cfg.get("bands", {}) or {}).keys())
    bands = tuple(dict.fromkeys(str(value) for value in (values or [])))
    if not bands:
        raise ValueError("FFA configuration contains no bands")
    return bands


def configured_wavelengths(section: dict, cfg: dict, bands: tuple[str, ...]) -> dict[str, float]:
    values = dict(section.get("wavelengths", {}) or cfg.get("bands", {}) or {})
    missing = [band for band in bands if band not in values]
    if missing:
        raise KeyError("missing configured wavelength(s): " + ", ".join(missing))
    wavelengths = {band: float(values[band]) for band in bands}
    if any(not np.isfinite(value) or value <= 0.0 for value in wavelengths.values()):
        raise ValueError("configured wavelengths must be finite and positive")
    return wavelengths


def expected_flux_posterior(
    analysis_dir: Path,
    source: str,
    component: str,
    bands: tuple[str, ...],
) -> Path:
    path = analysis_dir / f"{source}_{component}_{len(bands)}band_flux_posterior.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing official joint FFA posterior: {path}; "
            "rerun or restore the corresponding FFA/FVG product"
        )
    return path


def expected_components_table(
    analysis_dir: Path,
    source: str,
    component: str,
    bands: tuple[str, ...],
) -> Path:
    path = analysis_dir / f"{source}_{component}_{len(bands)}band_components.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing official FFA component table: {path}; "
            "rerun or restore the corresponding FFA/FVG product"
        )
    return path


def official_branch_id(cfg: dict) -> str:
    hubble = dict(cfg.get("hubble_distance", {}) or {})
    return str(hubble.get("posterior_branch", "comp0"))


def branch_root(source_dir: Path, cfg: dict, stage: str) -> Path:
    return mica_analysis_dir(source_dir, stage_config(cfg, stage), "comp0")


def short_ffa_root(source_dir: Path, cfg: dict, stage: str) -> Path:
    section = dict(cfg.get("fflux_comp0", {}) or {})
    if not section:
        raise KeyError("missing YAML section: fflux_comp0")
    return branch_root(source_dir, cfg, stage) / "fflux"


# 分别组装总光和所选MICA短分量的FFA输入；这里的stage标签也区分两套后验。
def stage_inputs(source_dir: Path, cfg: dict, mica_stage: str) -> tuple[list[StageInput], Path]:
    source = str(cfg.get("source") or source_dir.name)
    total = dict(cfg.get("fflux", {}) or {})
    short = dict(cfg.get("fflux_comp0", {}) or {})
    if not total:
        raise KeyError("missing YAML section: fflux")
    if not short:
        raise KeyError("missing YAML section: fflux_comp0")

    output_root = branch_root(source_dir, cfg, mica_stage) / "diagnostics"

    total_bands = configured_bands(total, cfg)
    short_bands = configured_bands(short, cfg)
    total_component = component_name(total, "total")
    short_component = component_name(short, "comp0")
    total_analysis = rel(source_dir, total.get("out", "fflux")) / "analysis"
    short_analysis = short_ffa_root(source_dir, cfg, mica_stage) / "analysis"

    inputs = [
        StageInput(
            stage="total_light",
            label="Total light",
            component=total_component,
            bands=total_bands,
            wavelengths=configured_wavelengths(total, cfg, total_bands),
            posterior_path=expected_flux_posterior(
                total_analysis,
                source,
                total_component,
                total_bands,
            ),
            components_path=expected_components_table(
                total_analysis,
                source,
                total_component,
                total_bands,
            ),
        ),
        StageInput(
            stage="short_component",
            label="Short component",
            component=short_component,
            bands=short_bands,
            wavelengths=configured_wavelengths(short, cfg, short_bands),
            posterior_path=expected_flux_posterior(
                short_analysis,
                source,
                short_component,
                short_bands,
            ),
            components_path=expected_components_table(
                short_analysis,
                source,
                short_component,
                short_bands,
            ),
        ),
    ]
    return inputs, output_root


def normalized_band(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(value)).upper()


def resolve_excluded_band(bands: tuple[str, ...], requested: str) -> str:
    matches = [band for band in bands if normalized_band(band) == normalized_band(requested)]
    if len(matches) != 1:
        raise ValueError(
            f"excluded band {requested!r} must match exactly one configured band; "
            f"matches={matches}, configured={list(bands)}"
        )
    return matches[0]


# 兼容已保存差分通量后验列名；不自动把RMS振幅R当成Delta_F。
def delta_flux_column(columns: pd.Index, band: str) -> str:
    exact = f"Delta_F_{band}"
    if exact in columns:
        return exact
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", band).strip("_")
    fallback = f"Delta_F_{safe}"
    if fallback in columns:
        return fallback
    raise KeyError(f"missing FFA posterior column for {band}: expected {exact!r}")


# 对所有样本向量化求解 log(S)=log_norm+alpha*log(lambda/pivot)。
# 固定alpha只估计截距；自由alpha使用加权线性回归，权重由输入边际误差固定。
def fit_log_powerlaw_draws(
    log_s: np.ndarray,
    x: np.ndarray,
    sigma_log: np.ndarray,
    alpha_fixed: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log normalization and alpha for weighted log-space fits."""
    if log_s.ndim != 2 or log_s.shape[1] != len(x):
        raise ValueError("log_s must have shape (n_draw, n_fit_band)")
    if len(x) != len(sigma_log):
        raise ValueError("x and sigma_log must have the same length")
    if len(x) < (2 if alpha_fixed is None else 1):
        raise ValueError("too few non-excluded bands for the requested power-law fit")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(sigma_log)):
        raise ValueError("x or sigma_log contains non-finite values")
    if np.any(sigma_log <= 0.0):
        raise ValueError("sigma_log must be strictly positive")

    weight = 1.0 / np.square(sigma_log)
    weight_sum = float(np.sum(weight))
    xbar = float(np.sum(weight * x) / weight_sum)
    if alpha_fixed is None:
        mean_log_s = np.sum(log_s * weight[None, :], axis=1) / weight_sum
        denominator = float(np.sum(weight * np.square(x - xbar)))
        if denominator <= 0.0:
            raise ValueError("free-alpha fit requires distinct wavelengths")
        alpha = np.sum(
            weight[None, :]
            * (x - xbar)[None, :]
            * (log_s - mean_log_s[:, None]),
            axis=1,
        ) / denominator
        log_norm = mean_log_s - alpha * xbar
    else:
        alpha = np.full(log_s.shape[0], float(alpha_fixed))
        log_norm = np.sum(
            weight[None, :] * (log_s - alpha[:, None] * x[None, :]),
            axis=1,
        ) / weight_sum
    return log_norm, alpha


def percentiles(values: np.ndarray) -> tuple[float, float, float]:
    q16, q50, q84 = np.percentile(np.asarray(values, float), [16.0, 50.0, 84.0])
    return float(q16), float(q50), float(q84)


def add_quantiles(row: dict, name: str, values: np.ndarray) -> None:
    q16, q50, q84 = percentiles(values)
    row[f"{name}_p16"] = q16
    row[f"{name}_p50"] = q50
    row[f"{name}_p84"] = q84


# 取同一联合样本的所有波段，筛选全正有限样本，再做留一波段预测。
# valid_fraction描述这一筛选保留率；输出区间是筛选后的条件分布。
def analyse_stage(
    source: str,
    stage: StageInput,
    excluded_request: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    posterior = pd.read_csv(stage.posterior_path, sep=r"\s+")
    components = pd.read_csv(stage.components_path)
    if "Filter" not in components:
        raise KeyError(f"missing Filter column in {stage.components_path}")
    components["Filter"] = components["Filter"].astype(str)
    if components["Filter"].duplicated().any():
        raise ValueError(f"duplicate filters in {stage.components_path}")
    components = components.set_index("Filter")
    missing_component_bands = [band for band in stage.bands if band not in components.index]
    if missing_component_bands:
        raise ValueError(
            f"FFA component table is missing bands: {', '.join(missing_component_bands)}"
        )
    components = components.loc[list(stage.bands)]
    for column in ("Delta_F_p50", "Delta_F_err"):
        if column not in components:
            raise KeyError(f"missing {column} in {stage.components_path}")

    excluded = resolve_excluded_band(stage.bands, excluded_request)
    fit_bands = tuple(band for band in stage.bands if band != excluded)
    if len(fit_bands) < 2:
        raise ValueError(f"{stage.stage} has fewer than two non-{excluded} bands")

    flux_columns = [delta_flux_column(posterior.columns, band) for band in stage.bands]
    wavelengths = np.asarray([stage.wavelengths[band] for band in stage.bands], float)
    delta_flux = posterior[flux_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    s_values = delta_flux * wavelengths[None, :]
    valid = np.all(np.isfinite(s_values) & (s_values > 0.0), axis=1)
    if np.count_nonzero(valid) < 20:
        raise ValueError(
            f"{stage.stage} has too few all-positive joint SED draws: "
            f"{np.count_nonzero(valid)}"
        )
    s_values = s_values[valid]

    if "sample" in posterior:
        stage_sample = posterior.loc[valid, "sample"].to_numpy()
    else:
        stage_sample = np.flatnonzero(valid)
    if pd.Series(stage_sample).duplicated().any():
        raise ValueError(f"{stage.stage} joint FFA posterior has duplicate sample identifiers")

    excluded_index = stage.bands.index(excluded)
    fit_indices = np.asarray([stage.bands.index(band) for band in fit_bands], int)
    pivot = float(np.exp(np.mean(np.log(wavelengths))))
    x = np.log(wavelengths / pivot)
    log_s_fit = np.log(s_values[:, fit_indices])
    y50 = wavelengths * pd.to_numeric(
        components["Delta_F_p50"], errors="coerce"
    ).to_numpy(float)
    yerr = wavelengths * pd.to_numeric(
        components["Delta_F_err"], errors="coerce"
    ).to_numpy(float)
    if not np.all(np.isfinite(y50) & np.isfinite(yerr) & (y50 > 0.0) & (yerr > 0.0)):
        raise ValueError(f"invalid FFA SED uncertainties in {stage.components_path}")
    # 相对通量误差近似为对数误差，设置下限防止某一波段取得无限权重。
    # 这里未把完整跨波段协方差矩阵代入广义最小二乘。
    sigma_log = np.clip(yerr / y50, 1.0e-4, None)
    observed = s_values[:, excluded_index]

    posterior_frames = []
    summary_rows = []
    fit_band_text = ";".join(fit_bands)
    for model, alpha_fixed in MODELS.items():
        log_norm, alpha = fit_log_powerlaw_draws(
            log_s_fit,
            x[fit_indices],
            sigma_log[fit_indices],
            alpha_fixed,
        )
        predicted = np.exp(log_norm + alpha * x[excluded_index])
        ratio = observed / predicted
        # 这是无量纲 fractional_excess；需要百分比时再乘100。
        excess = ratio - 1.0
        if not np.all(np.isfinite(predicted) & (predicted > 0.0)):
            raise ValueError(f"{stage.stage}/{model} produced invalid U-band predictions")

        posterior_frames.append(pd.DataFrame({
            "source": source,
            "stage": stage.stage,
            "component": stage.component,
            "fit_model": model,
            "stage_sample_id": stage_sample,
            "excluded_band": excluded,
            "fit_bands": fit_band_text,
            "fit_weighting": "inverse_variance_sigma_log",
            "alpha": alpha,
            "observed_s_u": observed,
            "predicted_s_u": predicted,
            "observed_over_predicted": ratio,
            "fractional_excess": excess,
        }))

        summary = {
            "source": source,
            "stage": stage.stage,
            "component": stage.component,
            "fit_model": model,
            "excluded_band": excluded,
            "fit_bands": fit_band_text,
            "fit_weighting": "inverse_variance_sigma_log",
            "n_input": len(posterior),
            "n_posterior": len(observed),
            "valid_fraction": float(np.mean(valid)),
            "probability_excess_gt_zero": float(np.mean(excess > 0.0)),
            "pairing_scope": "within_stage_joint_ffa_only",
            "input_posterior": str(stage.posterior_path),
            "input_components": str(stage.components_path),
        }
        add_quantiles(summary, "alpha", alpha)
        add_quantiles(summary, "observed_s_u", observed)
        add_quantiles(summary, "predicted_s_u", predicted)
        add_quantiles(summary, "observed_over_predicted", ratio)
        add_quantiles(summary, "fractional_excess", excess)
        summary_rows.append(summary)

    return (
        pd.concat(posterior_frames, ignore_index=True),
        pd.DataFrame(summary_rows),
    )


# 只汇总已算出的观测/预测比与过量分布；总光和短分量的样本不人为配对。
def plot_summary(summary: pd.DataFrame, source: str, output_dir: Path) -> tuple[Path, Path]:
    stage_order = ("total_light", "short_component")
    stage_labels = {"total_light": "Total light", "short_component": "Short component"}
    model_order = ("standard", "free")
    colors = {"standard": "#D55E00", "free": "#0072B2"}
    markers = {"standard": "o", "free": "s"}
    offsets = {"standard": -0.11, "free": 0.11}

    with plt.rc_context({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.linewidth": 1.0,
    }):
        fig, ax = plt.subplots(figsize=(6.2, 4.3), constrained_layout=True)
        for model in model_order:
            medians = []
            lower = []
            upper = []
            x_values = []
            for index, stage in enumerate(stage_order):
                selected = summary[
                    summary["stage"].eq(stage) & summary["fit_model"].eq(model)
                ]
                if len(selected) != 1:
                    raise ValueError(f"expected one summary row for {stage}/{model}")
                row = selected.iloc[0]
                median = float(row["fractional_excess_p50"])
                medians.append(median)
                lower.append(median - float(row["fractional_excess_p16"]))
                upper.append(float(row["fractional_excess_p84"]) - median)
                x_values.append(index + offsets[model])
            ax.errorbar(
                x_values,
                medians,
                yerr=np.vstack([lower, upper]),
                fmt=markers[model],
                color=colors[model],
                markerfacecolor="white" if model == "free" else colors[model],
                markersize=6.5,
                capsize=3,
                linewidth=1.3,
                label=r"Standard ($\alpha=-4/3$)" if model == "standard" else r"Free $\alpha$",
            )

        ax.axhline(0.0, color="0.45", linewidth=0.9, linestyle="--")
        ax.set_xticks(range(len(stage_order)), [stage_labels[stage] for stage in stage_order])
        ax.set_ylabel(r"U-band fractional excess  $S_U/\widehat{S}_{U,-U}-1$")
        ax.set_title(source, loc="left")
        ax.grid(axis="y", alpha=0.18)
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(frameon=False)

        png = output_dir / f"{source}_sed_u_excess.png"
        pdf = output_dir / f"{source}_sed_u_excess.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
    return png, pdf


# 遍历两套输入、写样本和汇总表、生成诊断图；不是纯读取操作。
def run(
    source_dir: Path,
    excluded_band: str = "U",
    mica_stage: str = "mica_round1",
) -> dict[str, Path]:
    source_dir = source_dir.expanduser().resolve()
    cfg_path = source_dir / "config" / "source_config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing source configuration: {cfg_path}")
    cfg = read_yaml(cfg_path)
    source = str(cfg.get("source") or source_dir.name)
    inputs, output_dir = stage_inputs(source_dir, cfg, mica_stage)

    posterior_frames = []
    summary_frames = []
    for stage in inputs:
        posterior, summary = analyse_stage(source, stage, excluded_band)
        posterior_frames.append(posterior)
        summary_frames.append(summary)

    posterior = pd.concat(posterior_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    posterior_path = output_dir / f"{source}_sed_u_excess_posterior.txt"
    summary_path = output_dir / f"{source}_sed_u_excess_summary.csv"
    posterior.to_csv(posterior_path, sep=" ", index=False, float_format="%.10e")
    summary.to_csv(summary_path, index=False)
    png, pdf = plot_summary(summary, source, output_dir)

    return {
        "posterior": posterior_path,
        "summary": summary_path,
        "figure_png": png,
        "figure_pdf": pdf,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--excluded-band", default="U")
    parser.add_argument("--stage", help="one MICA config section; default: all mica_stages")
    args = parser.parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    cfg = read_yaml(source_dir / "config" / "source_config.yaml")
    stages = [args.stage] if args.stage else configured_stages(cfg)
    for mica_stage in stages:
        outputs = run(source_dir, args.excluded_band, mica_stage)
        for path in outputs.values():
            print(path)


if __name__ == "__main__":
    main()
