#!/usr/bin/env python3
"""Compare post-hoc FVG nuclear-extinction H0 sensitivities.

This script only reads completed posteriors and the lightweight nuclear-
extinction outputs made by ``src/fvg_nuclear_h0.py``.  It never calls MICA or
FFA.  For an apples-to-apples standard-disc comparison it marginalizes the
saved two-parameter H0/f0 likelihood on a deterministic grid, avoiding the
small between-chain Monte Carlo shifts visible when E(B-V)=0.
"""
# 读取已完成的核消光/测距产物，比较指定源的 H0 和 SED 斜率对事后修正的敏感性。
# SOURCE_SPECS 固定了历史数据版本；部分 NGC2617 数据互相包含，不能当独立样本合并。
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.integrate import cumulative_trapezoid


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import disk_distance  # noqa: E402


SOURCE_SPECS = (
    {
        "directory": "Mrk335",
        "label": "Mrk335",
        "result_dir": "nuclear_extinction_total_fvg_pre_support_fix",
        "dataset_role": "active total FVG + archived pre-support comp0 flux",
        "independent": True,
    },
    {
        "directory": "F9_a",
        "label": "F9A",
        "result_dir": "nuclear_extinction_total_fvg",
        "dataset_role": "active six-band campaign",
        "independent": True,
    },
    {
        "directory": "Mrk279_a",
        "label": "Mrk279-A",
        "result_dir": "nuclear_extinction_total_fvg",
        "dataset_role": "active six-band campaign",
        "independent": True,
    },
    {
        "directory": "Mrk509",
        "label": "Mrk509",
        "result_dir": "nuclear_extinction_total_fvg",
        "dataset_role": "active six-band campaign",
        "independent": True,
    },
    {
        "directory": "NGC2617_a",
        "label": "NGC2617-A",
        "result_dir": "nuclear_extinction_total_fvg",
        "dataset_role": "first campaign only",
        "independent": False,
    },
    {
        "directory": "NGC2617",
        "label": "NGC2617 (A+B combined)",
        "result_dir": "nuclear_extinction_total_fvg",
        "dataset_role": "contains all A rows plus later observations; not B-only",
        "independent": False,
    },
    {
        "directory": "NGC5548",
        "label": "NGC5548",
        "result_dir": "nuclear_extinction_total_fvg",
        "dataset_role": "active six-band campaign",
        "independent": True,
    },
)


def branch_root(source: Path, cfg: dict) -> Path:
    mica = dict(cfg.get("mica_round1", {}) or {})
    ncomp = int(mica.get("ncomp", 2))
    output = source / mica.get("output_root", "runs/mica")
    return output / f"{ncomp}comp" / "result" / "comp0"


# 从保存的 log-distance 均值/协方差构造 C=C_saved+f0^2 I，D(H0)=D70*70/H0。
# 在有限矩形先验网格边际化 f0：logaddexp 累加稳定权重，再对 H0 梯形积分反演分位数。
def marginal_h0_grid(
    distance_dir: Path,
    *,
    redshift: float,
    om0: float,
    h0_prior: tuple[float, float],
    f0_prior: tuple[float, float],
    n_h0: int = 10001,
    n_f0: int = 1001,
) -> dict[str, float]:
    """Integrate the saved H0/f0 likelihood on a dense rectangular grid."""
    mean_log = pd.read_csv(distance_dir / "log_distance_mean.csv")[
        "mean_log_D"
    ].to_numpy(float)
    cov_log = pd.read_csv(
        distance_dir / "log_distance_covariance.csv", index_col=0
    ).to_numpy(float)
    h0 = np.linspace(h0_prior[0] + 1.0e-6, h0_prior[1] - 1.0e-6, n_h0)
    f0 = np.linspace(f0_prior[0], f0_prior[1] - 1.0e-6, n_f0)
    log_weights = np.zeros(len(f0))
    log_weights[[0, -1]] = np.log(0.5)

    d70 = disk_distance.model_d70(float(redshift), float(om0))
    predicted_log_d = np.log(d70 * 70.0 / h0)
    ones = np.ones(len(mean_log))
    marginal_logp = np.full(len(h0), -np.inf)

    for index, f0_value in enumerate(f0):
        covariance = cov_log + (f0_value**2 + 1.0e-12) * np.eye(len(mean_log))
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            continue
        inverse = np.linalg.inv(covariance)
        quadratic_a = float(ones @ inverse @ ones)
        quadratic_b = float(ones @ inverse @ mean_log)
        quadratic_c = float(mean_log @ inverse @ mean_log)
        quadratic = (
            quadratic_c
            - 2.0 * quadratic_b * predicted_log_d
            + quadratic_a * predicted_log_d**2
        )
        loglike = -0.5 * (
            quadratic + logdet + len(mean_log) * np.log(2.0 * np.pi)
        )
        marginal_logp = np.logaddexp(
            marginal_logp, loglike + log_weights[index]
        )

    density = np.exp(marginal_logp - np.nanmax(marginal_logp))
    cdf = np.concatenate([[0.0], cumulative_trapezoid(density, h0)])
    cdf /= cdf[-1]
    p16, p50, p84 = np.interp([0.16, 0.50, 0.84], cdf, h0)
    p_lt25 = float(np.interp(25.0, h0, cdf))
    p_gt115 = float(1.0 - np.interp(115.0, h0, cdf))
    return {
        "p16": float(p16),
        "p50": float(p50),
        "p84": float(p84),
        "p_lt25": p_lt25,
        "p_gt115": p_gt115,
    }


# 对每个联合样本拟合 ln(lambda*Delta_F) 的幂律；固定权重来自边际相对半宽，保留原行配对。
def _alpha_quantiles(
    delta_flux: np.ndarray,
    wavelengths: np.ndarray,
) -> dict[str, float]:
    """Reproduce the current per-draw free SED-alpha estimator.

    The weights follow ``run_ffa.run_sed_models``: each band uses the
    posterior half-width divided by its median flux, while every joint draw
    is fitted analytically in log space.
    """
    delta_flux = np.asarray(delta_flux, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    if delta_flux.ndim != 2 or delta_flux.shape[1] != len(wavelengths):
        raise ValueError("delta_flux must have shape (n_draw, n_band)")
    good = np.all(np.isfinite(delta_flux) & (delta_flux > 0.0), axis=1)
    delta_flux = delta_flux[good]
    if len(delta_flux) < 20:
        raise ValueError("too few positive joint SED draws")

    p16, p50, p84 = np.percentile(delta_flux, [16.0, 50.0, 84.0], axis=0)
    sigma_log = np.clip(0.5 * (p84 - p16) / p50, 1.0e-4, None)
    weight = 1.0 / np.square(sigma_log)
    pivot = float(np.exp(np.mean(np.log(wavelengths))))
    x = np.log(wavelengths / pivot)
    xbar = float(np.sum(weight * x) / np.sum(weight))
    log_y = np.log(delta_flux * wavelengths[None, :])
    ybar = np.sum(log_y * weight[None, :], axis=1) / np.sum(weight)
    alpha = np.sum(
        weight[None, :]
        * (x - xbar)[None, :]
        * (log_y - ybar[:, None]),
        axis=1,
    ) / np.sum(weight * np.square(x - xbar))
    p16_alpha, p50_alpha, p84_alpha = np.percentile(
        alpha, [16.0, 50.0, 84.0]
    )
    return {
        "p16": float(p16_alpha),
        "p50": float(p50_alpha),
        "p84": float(p84_alpha),
        "n": int(len(alpha)),
    }


# 用已保存的逐波段 corr_nuc 反推修正前样本，再与既有 free SED 摘要核对。
def sed_alpha_before_after(
    source: Path,
    cfg: dict,
    result: Path,
) -> dict[str, float | str]:
    """Measure comp0 SED alpha before and after the saved FVG correction."""
    ff = dict(cfg.get("fflux_comp0", cfg.get("fflux", {})) or {})
    bands = [str(band) for band in ff["bands"]]
    wavelengths_cfg = dict(ff.get("wavelengths", cfg.get("bands", {})) or {})
    wavelengths = {band: float(wavelengths_cfg[band]) for band in bands}
    source_name = str(cfg.get("source", source.name))
    posterior_path = result / "analysis" / "distance_flux_posterior.txt"
    posterior = pd.read_csv(posterior_path, sep=r"\s+")
    corrected_columns = [f"Delta_F_comp0_{band}" for band in bands]
    correction_columns = [f"corr_nuc_{band}" for band in bands]
    missing = [
        column
        for column in corrected_columns + correction_columns
        if column not in posterior
    ]
    if missing:
        raise KeyError(f"{source_name}: corrected comp0 posterior is missing {missing}")
    corrected = posterior[corrected_columns].to_numpy(float)
    correction = posterior[correction_columns].to_numpy(float)
    if np.any(~np.isfinite(correction)) or np.any(correction <= 0.0):
        raise ValueError(f"{source_name}: invalid nuclear-extinction correction factor")
    uncorrected = corrected / correction
    wave_array = np.asarray([wavelengths[band] for band in bands], dtype=float)
    before = _alpha_quantiles(uncorrected, wave_array)
    after = _alpha_quantiles(corrected, wave_array)

    summary_path = result.parent / "fflux" / "analysis" / (
        f"{source_name}_comp0_sed_fit_summary.csv"
    )
    saved = pd.read_csv(summary_path)
    saved_free = saved.loc[saved["model"] == "free"].iloc[0]
    if not np.isclose(before["p50"], float(saved_free["sed_alpha"]), atol=1.0e-9):
        raise ValueError(f"{source_name}: reconstructed alpha does not match saved SED fit")
    return {
        "alpha_before_p16": before["p16"],
        "alpha_before_p50": before["p50"],
        "alpha_before_p84": before["p84"],
        "alpha_after_p16": after["p16"],
        "alpha_after_p50": after["p50"],
        "alpha_after_p84": after["p84"],
        "alpha_n_posterior": before["n"],
        "alpha_source": str(posterior_path),
        "alpha_component": "comp0",
    }


# 部分质量标签是对特定历史源名的硬编码判断，不会随新数据自动重新推断。
def quality_label(row: dict[str, float | str | bool]) -> str:
    if row["label"] == "F9A":
        return "unstable: active EBV entirely negative before truncation"
    if row["label"] == "NGC2617-A":
        return "weak EBV: 31% of raw draws truncated at zero"
    if row["label"] == "NGC2617 (A+B combined)":
        return "overlaps NGC2617-A; not an independent B-only result"
    if float(row["h0_before_p_lt25"]) > 0.5:
        return "H0 posterior strongly affected by lower prior boundary"
    if float(row["h0_before_p_lt25"]) > 0.1:
        return "H0 lower tail moderately affected by prior boundary"
    return "usable only as pre-support sensitivity"


def build_summary(root: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    for spec in SOURCE_SPECS:
        source = root / spec["directory"]
        config_path = source / "config" / "source_config.yaml"
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        hd = dict(cfg.get("hubble_distance", {}) or {})
        branch = branch_root(source, cfg)
        result = branch / spec["result_dir"]
        ebv_path = result / "analysis" / "fvg_nuclear_ebv.csv"
        correction_path = result / "analysis" / "nuclear_flux_summary.csv"
        comparison_path = result / "h0" / "h0_before_after.csv"
        required = [ebv_path, correction_path, comparison_path]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing completed post-processing outputs: {missing}")

        ebv = pd.read_csv(ebv_path).iloc[0]
        correction = pd.read_csv(correction_path).set_index("band")
        comparison = pd.read_csv(comparison_path)
        standard = comparison.loc[comparison["distance_model"] == "standard"].iloc[0]
        h0_prior = tuple(map(float, hd.get("h0_prior", [20.0, 120.0])))
        f0_prior = tuple(map(float, hd.get("f0_prior", [0.0, 1.0])))
        om0 = float(hd.get("om0", 0.3))
        redshift = float(hd.get("redshift", cfg.get("redshift", 0.0)))
        before = marginal_h0_grid(
            branch / "distance" / "standard",
            redshift=redshift,
            om0=om0,
            h0_prior=h0_prior,
            f0_prior=f0_prior,
        )
        after = marginal_h0_grid(
            result / "distance" / "standard",
            redshift=redshift,
            om0=om0,
            h0_prior=h0_prior,
            f0_prior=f0_prior,
        )
        shift = after["p50"] - before["p50"]
        half_width = 0.5 * (after["p84"] - after["p16"])
        alpha = sed_alpha_before_after(source, cfg, result)
        row: dict[str, float | str | bool] = {
            "label": spec["label"],
            "directory": spec["directory"],
            "dataset_role": spec["dataset_role"],
            "independent_sample": bool(spec["independent"]),
            "redshift": redshift,
            "ebv_raw_p16": float(ebv["ebv_raw_p16"]),
            "ebv_raw_p50": float(ebv["ebv_raw_p50"]),
            "ebv_raw_p84": float(ebv["ebv_raw_p84"]),
            "ebv_p16": float(ebv["ebv_p16"]),
            "ebv_p50": float(ebv["ebv_p50"]),
            "ebv_p84": float(ebv["ebv_p84"]),
            "ebv_truncated_fraction": float(ebv["truncated_fraction"]),
            **alpha,
            "w2_flux_factor_p50": float(correction.loc["W2", "corr_nuc_p50"]),
            "w2_nominal_h0_scale": float(correction.loc["W2", "h0_scale_p50"]),
            "v_flux_factor_p50": float(correction.loc["V", "corr_nuc_p50"]),
            "v_nominal_h0_scale": float(correction.loc["V", "h0_scale_p50"]),
            "h0_before_p16": before["p16"],
            "h0_before_p50": before["p50"],
            "h0_before_p84": before["p84"],
            "h0_after_p16": after["p16"],
            "h0_after_p50": after["p50"],
            "h0_after_p84": after["p84"],
            "h0_shift": shift,
            "h0_shift_percent": 100.0 * (after["p50"] / before["p50"] - 1.0),
            "h0_shift_over_halfwidth": shift / half_width if half_width > 0 else np.nan,
            "h0_before_p_lt25": before["p_lt25"],
            "h0_after_p_lt25": after["p_lt25"],
            "h0_before_p_gt115": before["p_gt115"],
            "h0_after_p_gt115": after["p_gt115"],
            "saved_mcmc_h0_before": float(standard["H0_before"]),
            "saved_mcmc_h0_after": float(standard["H0_after"]),
            "postprocess_result_root": str(result),
            "ebv_source": str(ebv_path),
            "config_source": str(config_path),
            "support_status": "pre-driver-support-fix sensitivity",
        }
        row["quality_note"] = quality_label(row)
        rows.append(row)
    return pd.DataFrame(rows)


def validate_overlap(root: Path) -> dict[str, int | bool]:
    a = pd.read_csv(root / "NGC2617_a" / "raw" / "NGC2617_a.csv")
    combined = pd.read_csv(root / "NGC2617" / "raw" / "NGC2617.csv")
    columns = ["Object", "Filter", "MJD", "Flux", "Error"]
    joined = a.merge(combined[columns].drop_duplicates(), on=columns, how="left", indicator=True)
    matched = int((joined["_merge"] == "both").sum())
    return {
        "ngc2617_a_rows": int(len(a)),
        "ngc2617_combined_rows": int(len(combined)),
        "a_rows_found_in_combined": matched,
        "all_a_rows_found": bool(matched == len(a)),
    }


# 只有显式给 --notebook 才生成 nbformat 文档；生成 notebook 不等于执行其中计算。
def write_notebook(path: Path, root: Path, csv_path: Path) -> None:
    """Create a compact rerunnable notebook companion with ``nbformat``."""
    import nbformat as nbf

    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3 (astro)",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# FVG 核区消光与 standard-$H_0$ 横向比较\n\n"
            "## tl;dr\n\n"
            "现有六波段后验显示，核区消光对 standard-$H_0$ 中位数的确定性网格修正为 "
            "0.0%–4.1%。NGC5548 的消光和 UV 通量修正最大，但其 $H_0$ 后验强贴近下先验边界；"
            "F9A 的活动数据给出全负 raw $E(B-V)$，截断后校正严格为零。所有数值都是既有后验的"
            "廉价后处理，没有运行 MICA 或 FFA。"
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "采用 total-light FVG 的 W2/V 变量色、G03 SMC Bar 消光曲线、"
            "$F_\\lambda\\propto\\lambda^{-7/3}$，并把负 $E(B-V)$ 截到零。"
            "同一 $E(B-V)$ 后验逐波段作用到已有 comp0 亮/暗通量，$\\epsilon$ 不变。"
            "主比较只使用与该谱斜率自洽的 standard 薄盘模型。\n\n"
            "### Key Assumptions\n\n"
            "- 复现旧流程的 observed-frame 波长约定。\n"
            "- total-FVG 与 comp0 后验独立配对，未建模两者共享光变数据的相关性。\n"
            "- 全部 comp0 通量属于 driver-support 修复前版本，因此结果是敏感性分析，不是最终发布值。\n"
            "- NGC2617 的基础目录是 A+B 合并集，不是独立 B-only。"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            f"ROOT = Path({str(root)!r})\n"
            f"CSV_PATH = Path({str(csv_path)!r})\n"
            "sys.path.insert(0, str(ROOT))\n"
            "from scripts.compare_nuclear_extinction_h0 import build_summary, validate_overlap\n"
            "pd.set_option('display.max_colwidth', 80)"
        ),
        nbf.v4.new_markdown_cell("## Data\n\n### 1. Recompute from saved posteriors"),
        nbf.v4.new_code_cell(
            "summary = build_summary(ROOT)\n"
            "overlap = validate_overlap(ROOT)\n"
            "saved = pd.read_csv(CSV_PATH)\n"
            "key = ['label', 'ebv_p50', 'h0_before_p50', 'h0_after_p50', 'h0_shift_percent']\n"
            "merged = summary[key].merge(saved[key], on='label', suffixes=('_new', '_saved'))\n"
            "numeric_diffs = []\n"
            "for column in key[1:]:\n"
            "    numeric_diffs.append(np.max(np.abs(merged[f'{column}_new'] - merged[f'{column}_saved'])))\n"
            "assert max(numeric_diffs) < 1e-9\n"
            "assert overlap['all_a_rows_found']\n"
            "overlap"
        ),
        nbf.v4.new_markdown_cell(
            "## Results\n\n"
            "### 2. Compare extinction and deterministic standard-$H_0$ shifts\n\n"
            "The dense grid integrates the same saved two-parameter $H_0,f_0$ likelihood for every source, "
            "so the reported before/after shift is not contaminated by differences between two finite MCMC chains."
        ),
        nbf.v4.new_code_cell(
            "view = summary[[\n"
            "    'label', 'ebv_p16', 'ebv_p50', 'ebv_p84',\n"
            "    'w2_flux_factor_p50', 'h0_before_p50', 'h0_after_p50',\n"
            "    'h0_shift_percent', 'h0_shift_over_halfwidth', 'quality_note'\n"
            "]].copy()\n"
            "view.round({\n"
            "    'ebv_p16': 4, 'ebv_p50': 4, 'ebv_p84': 4,\n"
            "    'w2_flux_factor_p50': 3, 'h0_before_p50': 2, 'h0_after_p50': 2,\n"
            "    'h0_shift_percent': 2, 'h0_shift_over_halfwidth': 2\n"
            "})"
        ),
        nbf.v4.new_code_cell(
            "plot_data = summary.iloc[::-1]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))\n"
            "color = '#3976B8'\n"
            "axes[0].barh(plot_data['label'], plot_data['ebv_p50'], color=color, edgecolor='#244A73')\n"
            "axes[0].set_title('FVG nuclear E(B-V)')\n"
            "axes[0].set_xlabel('E(B-V), mag')\n"
            "axes[0].grid(axis='x', color='#D8DEE8', linewidth=0.8)\n"
            "axes[1].barh(plot_data['label'], plot_data['h0_shift_percent'], color=color, edgecolor='#244A73')\n"
            "axes[1].axvline(0, color='#333333', linewidth=0.8)\n"
            "axes[1].set_title('Standard H0 median shift')\n"
            "axes[1].set_xlabel('Change, %')\n"
            "axes[1].grid(axis='x', color='#D8DEE8', linewidth=0.8)\n"
            "for axis in axes:\n"
            "    axis.spines[['top', 'right']].set_visible(False)\n"
            "fig.suptitle('Post-hoc nuclear-extinction sensitivity from existing posteriors')\n"
            "fig.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "- **修正不总是小在通量层。** NGC5548 的 W2 通量中位校正约 1.58，"
            "但 standard-$H_0$ 中位数只从 22.79 到 23.73（+4.11%），因为后验强受 $H_0>20$ 边界限制。\n"
            "- **Mrk335 不是异常小。** 其确定性 standard 修正为 35.81 到 37.07（+3.52%），"
            "位于这组样本的常见范围。\n"
            "- **F9A 不能按当前方法得到正消光。** 100% raw $E(B-V)$ 为负，截断后 $H_0$ 严格不变；"
            "旧归档却为正，需先解释数据版本/活动阶段差异。\n"
            "- **NGC2617-B 尚未真正计算。** 当前 `NGC2617` 包含全部 A 数据，不能作为独立 B-only 结果。\n"
            "- 所有 shift 都远小于各自的 $H_0$ 后验半宽，不能据此宣称显著改变宇宙学结论。"
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--notebook", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    summary = build_summary(root)
    overlap = validate_overlap(root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(
            json.dumps(
                {
                    "method": (
                        "saved total-FVG EBV -> comp0 flux scaling and comp0 SED "
                        "alpha refit -> deterministic standard H0/f0 grid"
                    ),
                    "does_not_run": ["MICA", "FFA"],
                    "fvg_pair": ["W2", "V"],
                    "extinction_law": "G03 SMC Bar",
                    "intrinsic_Flambda_exponent": -7.0 / 3.0,
                    "wavelength_frame": "observed_legacy",
                    "h0_grid": 10001,
                    "f0_grid": 1001,
                    "ngc2617_overlap_check": overlap,
                    "chart_map": [
                        {
                            "section": "cross-source correction size",
                            "question": "How much does nuclear extinction move the standard H0 median?",
                            "family": "comparison",
                            "type": "horizontalBar",
                            "fields": ["label", "h0_shift_percent"],
                            "claim": "deterministic shifts span 0.0% to 4.1%",
                            "palette": "single-root blue",
                        },
                        {
                            "section": "extinction amplitude",
                            "question": "Which sources have the largest FVG colour excess?",
                            "family": "comparison",
                            "type": "horizontalBar",
                            "fields": ["label", "ebv_p50"],
                            "claim": "NGC5548 is largest; F9A truncates to zero",
                            "palette": "single-root blue",
                        },
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.notebook:
        if not args.output:
            raise ValueError("--notebook requires --output so the notebook can verify it")
        write_notebook(args.notebook, root, args.output.resolve())
    display = summary[
        [
            "label",
            "ebv_p50",
            "alpha_before_p50",
            "alpha_after_p50",
            "w2_flux_factor_p50",
            "h0_before_p50",
            "h0_after_p50",
            "h0_shift_percent",
            "quality_note",
        ]
    ]
    print(display.to_string(index=False))
    print(json.dumps(overlap, ensure_ascii=False))


if __name__ == "__main__":
    main()
