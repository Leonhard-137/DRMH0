#!/usr/bin/env python3
# 独立的旧多源入口：读各源逐波段 lnD 均值/协方差，边缘化源内波段散度，再联合 H0。
# 不相乘已保存的单源 H0 链，也不把同一 AGN 不同历元或不同模型当独立源。
# SciPy 的 Gauss-Legendre 节点、logsumexp、正态 CDF/求根和积分实现稳定边缘化。
# 本模块采用 lnD 高斯摘要；shared_sed_hubble/h0.py 则采用每源一个线性 D_L 高斯摘要，二者模型不同。
"""Joint H0 from saved, correlated log-distance summaries; no MICA/FFA reruns.

Chart contract: one point per independent AGN, luminosity distance versus z,
with a common cosmological curve and its 68% parameter interval. Separate
fits are compared, never multiplied together. Static, labelled, grayscale
PNG/PDF figures and machine-readable inputs/posteriors are the deliverables.

For source i: m_i ~ N(mu_i(H0) 1, C_i + f_i^2 I + s_i^2 11^T), where
s_i^2 = sigma_int^2 + sigma_velocity,i^2. We integrate a separate band
scatter f_i for each source, then jointly integrate sigma_int. The Gaussian
approximation in lnD is inherited from disk_distance.py, not a new raw-data
likelihood. In particular, we do NOT multiply the old truncated H0 chains.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import brentq
from scipy.special import logsumexp, ndtr
import yaml

C_KMS = 299792.458
PROBS = np.array([ndtr(-2.0), ndtr(-1.0), 0.5, ndtr(1.0), ndtr(2.0)])
Q_NAMES = ("q2p3", "q16", "q50", "q84", "q97p7")


# 限定配置路径在项目根的子目录内，避免错误定位外部输入/输出。
def under(root: Path, relative: str) -> Path:
    """Resolve a configured path without allowing writes/reads outside root."""
    path = (root / relative).resolve()
    if path == root.resolve() or root.resolve() not in path.parents:
        raise ValueError(f"path is not a child of {root}: {relative}")
    return path


# 剥离 _a/_b 等历元后缀检查独立 AGN，防止同源重复计数。
def validate_sources(sources: list[str]) -> None:
    ids = [re.sub(r"_[a-z]$", "", s, flags=re.I).casefold() for s in sources]
    if not sources or len(set(ids)) != len(ids):
        raise ValueError("sources must contain distinct AGN, not multiple epochs of one AGN")
    if any(not re.fullmatch(r"[A-Za-z0-9_+.-]+", s) for s in sources):
        raise ValueError("source names must be directory names, not paths")


# 返回 H0 D；comoving 为 c∫dz/E，luminosity 再乘 (1+z)，对应 Mpc 距离。
def distance_scale(z: float, om0: float, kind: str) -> float:
    """Return H0 * D(z) in flat LambdaCDM; D is in Mpc for H0 in km/s/Mpc."""
    if not np.isfinite(z) or z <= 0 or not 0 <= om0 <= 1:
        raise ValueError("need z > 0 and 0 <= Omega_m <= 1")
    if kind not in ("comoving", "luminosity"):
        raise ValueError("distance_kind must be comoving or luminosity")
    integral = quad(lambda x: (om0 * (1 + x)**3 + 1 - om0)**-0.5, 0, z)[0]
    return C_KMS * integral * ((1 + z) if kind == "luminosity" else 1)


# 一阶传播 σ_lnD=(d lnD/dz)(1+z)σ_v/c；同一源各波段共用此误差，不独立加到对角。
def velocity_ln_sigma(z: float, om0: float, kind: str, velocity: float) -> float:
    """First-order peculiar-velocity uncertainty, common to ALL source bands."""
    if not np.isfinite(velocity) or velocity < 0:
        raise ValueError("peculiar_velocity_kms must be nonnegative")
    integral = distance_scale(z, om0, "comoving") / C_KMS
    derivative = 1 / (np.sqrt(om0 * (1 + z)**3 + 1 - om0) * integral)
    if kind == "luminosity":
        derivative += 1 / (1 + z)
    return derivative * (1 + z) * velocity / C_KMS


# 按真实网格坐标构造梯形权重；即使 H0 用对数间隔，也仍积分 dH0。
def integration_weights(x: np.ndarray) -> np.ndarray:
    if len(x) == 1:
        return np.ones(1)
    if np.any(np.diff(x) <= 0):
        raise ValueError("integration grid must be strictly increasing")
    return np.r_[np.diff(x)[0], x[2:] - x[:-2], np.diff(x)[-1]] / 2


# 积分密度得到 CDF 后插值分位数，单点模型直接重复该点。
def pdf_quantiles(x: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    if len(x) == 1:
        return np.repeat(x[0], len(PROBS))
    cdf = cumulative_trapezoid(pdf, x, initial=0)
    if not np.isfinite(cdf[-1]) or cdf[-1] <= 0:
        raise ValueError("posterior cannot be normalized")
    return np.interp(PROBS, cdf / cdf[-1], x)


# 每源多波段 lnD 似然压成一维正态混合，混合来自对源内波段散度 f_i 的积分。
@dataclass
class BandMixture:
    """A normalized mixture proportional to the source-only lnD likelihood."""
    mean: np.ndarray
    variance: np.ndarray
    log_weight: np.ndarray

    # 先验证完整协方差正定并特征分解，再给每个 f_i 节点加 f_i²I。
    # 对公共 lnD 完成平方，得到各节点的一维均值、方差与相对证据权重。
    # 减去中心再做二次型以避免大数相减；不能先把各波段当独立点平均。
    @classmethod
    def from_summary(cls, mean, covariance, scatter_range=(0, 1), nodes=128):
        mean, covariance = np.asarray(mean, float), np.asarray(covariance, float)
        n = len(mean)
        if n == 0 or covariance.shape != (n, n):
            raise ValueError("mean/covariance dimensions do not match")
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise ValueError("non-finite distance summary")
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-8):
            raise ValueError("asymmetric covariance")
        eigenvalues, vectors = np.linalg.eigh(covariance)
        if eigenvalues[0] <= 0:
            raise ValueError("distance covariance must be positive definite")
        lo, hi = map(float, scatter_range)
        if not 0 <= lo <= hi or nodes < 8:
            raise ValueError("invalid band scatter range or quadrature node count")
        if hi == lo:
            f, weights = np.array([lo]), np.ones(1)
        else:
            x, weights = np.polynomial.legendre.leggauss(int(nodes))
            f, weights = lo + (x + 1) * (hi - lo) / 2, weights / 2
        # Centre before computing quadratic forms to avoid catastrophic cancellation.
        centre = float(np.mean(mean))
        u = vectors.T @ np.ones(n)
        v = vectors.T @ (mean - centre)
        eigen = eigenvalues[None, :] + f[:, None]**2
        a = np.sum(u**2 / eigen, axis=1)
        b = np.sum(u * v / eigen, axis=1)
        fitted = b / a
        residual = np.sum((v[None, :] - fitted[:, None] * u)**2 / eigen, axis=1)
        log_weight = np.log(weights) - 0.5 * (
            np.log(eigen).sum(axis=1) + np.log(a) + residual
        )
        log_weight -= logsumexp(log_weight)
        return cls(centre + fitted, 1 / a, log_weight)

    # 给混合分量增加共同方差 σ_int²+σ_velocity²，再用 logsumexp 求源级似然。
    def log_likelihood(self, mu, extra_variance=0.0):
        mu, extra = np.broadcast_arrays(mu, extra_variance)
        variance = self.variance + extra[..., None]
        terms = self.log_weight - 0.5 * (
            np.log(2 * np.pi * variance) + (mu[..., None] - self.mean)**2 / variance
        )
        return logsumexp(terms, axis=-1)

    # 对正态混合 CDF 数值求根，得到只依赖该源数据的距离区间。
    def quantiles(self):
        weights, sd = np.exp(self.log_weight), np.sqrt(self.variance)
        lower, upper = np.min(self.mean - 12 * sd), np.max(self.mean + 12 * sd)
        return np.array([brentq(
            lambda x: np.sum(weights * ndtr((x - self.mean) / sd)) - p,
            lower, upper,
        ) for p in PROBS])


# 保存输入字节摘要、大小和时间，用于来源追溯。
def file_record(path: Path, content: bytes | None = None) -> dict:
    if content is None:
        content = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content), "mtime_ns": path.stat().st_mtime_ns}


# 读取模型匹配的均值/协方差/运行摘要，按波段标签对齐，并记录上游样本数与 MICA ESS 警告。
def load_source(root: Path, source: str, dataset: str, relative: str, model: str, cfg: dict):
    source_dir = under(root, source)
    result = under(source_dir, relative)
    folder = result / "distance" / model
    files = [folder / "log_distance_mean.csv", folder / "log_distance_covariance.csv",
             folder / "distance_run_summary.csv", source_dir / "config/source_config.yaml"]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(str(path))
    contents = {path: path.read_bytes() for path in files}
    means = pd.read_csv(io.BytesIO(contents[files[0]]), dtype={"band": str}).set_index("band")
    cov = pd.read_csv(io.BytesIO(contents[files[1]]), index_col=0)
    run = pd.read_csv(io.BytesIO(contents[files[2]]))
    if len(run) != 1 or str(run.iloc[0]["distance_model"]) != model:
        raise ValueError(f"inconsistent distance_run_summary: {files[2]}")
    if means.index.has_duplicates or cov.index.has_duplicates or cov.columns.has_duplicates:
        raise ValueError(f"duplicated bands: {folder}")
    if set(cov.index) != set(means.index) or set(cov.columns) != set(means.index):
        raise ValueError(f"covariance band labels do not match the means: {folder}")
    bands = list(means.index) if cfg.get("bands", "all") == "all" else cfg["bands"]
    if not bands or len(set(bands)) != len(bands) or not set(bands) <= set(means.index):
        raise ValueError(f"invalid selected bands for {source}: {bands}")
    inf = cfg["inference"]
    mixture = BandMixture.from_summary(
        means.loc[bands, "mean_log_D"].to_numpy(), cov.loc[bands, bands].to_numpy(),
        inf["band_scatter_range"], int(inf["band_scatter_nodes"]),
    )
    source_cfg = yaml.safe_load(contents[files[3]].decode("utf-8"))
    cosmo, quality = cfg["cosmology"], cfg.get("quality", {})
    z = float(cosmo.get("redshift_overrides", {}).get(source, source_cfg["redshift"]))
    kind, om0 = cosmo["distance_kind"], float(cosmo["om0"])
    scale = distance_scale(z, om0, kind)
    vsig = velocity_ln_sigma(z, om0, kind, float(cosmo["peculiar_velocity_kms"]))
    n_sample = int(run.iloc[0]["n_sample"])
    warnings = []
    if n_sample < quality.get("warn_distance_samples_below", 100):
        warnings.append(f"only {n_sample} saved joint distance samples (not an ESS)")
    ess_path = result.parent / "mica_posterior_selection.csv"
    min_ess = None
    if ess_path.is_file():
        files.append(ess_path)
        contents[ess_path] = ess_path.read_bytes()
        ess = pd.read_csv(io.BytesIO(contents[ess_path]))
        if "conditional_ess" in ess:
            # All lag-fit bands can affect the derived absolute lag, even if the
            # distance combination selects fewer bands. Report the minimum of all.
            values = pd.to_numeric(ess["conditional_ess"], errors="coerce")
            values = values[np.isfinite(values)]
            if len(values):
                min_ess = float(values.min())
                if min_ess < quality.get("warn_mica_ess_below", 100):
                    warnings.append(f"minimum reported MICA ESS = {min_ess:.3g}")
    if vsig > 0.1:
        warnings.append("large peculiar-velocity term; linear redshift propagation is approximate")
    meta = {"source": source, "dataset": dataset, "disk_model": model, "z": z,
            "bands": bands, "n_distance_samples": n_sample, "min_mica_ess": min_ess,
            "inclination_deg": float(run.iloc[0]["inclination_deg"]),
            "result_dir": str(result), "velocity_sigma_lnD": vsig,
            "distance_kind": kind, "warnings": warnings,
            "files": [file_record(p, contents[p]) for p in files]}
    return {"mixture": mixture, "meta": meta, "scale": scale, "velocity": vsig}


# 将一维源级混合评价在 H0 与共同额外散度网格上。
def source_log_grid(data, h0, scatter):
    mu = np.log(data["scale"] / h0)
    result = np.empty((len(scatter), len(h0)))
    # Bounded chunks keep memory modest while MICA runs in other terminals.
    for start in range(0, len(scatter), 8):
        extra = scatter[start:start + 8, None]**2 + data["velocity"]**2
        result[start:start + 8] = data["mixture"].log_likelihood(mu[None, :], extra)
    return result


# 相乘独立源似然并按网格权重归一化，报告 H0/散度边缘分位与边界质量。
def fit_joint(log_grid, h0, scatter):
    wh, ws = integration_weights(h0), integration_weights(scatter)
    log_h = logsumexp(log_grid + np.log(ws)[:, None], axis=0)
    log_norm = logsumexp(log_h + np.log(wh))
    h_pdf = np.exp(log_h - log_norm)
    s_pdf = np.exp(logsumexp(log_grid + np.log(wh)[None, :], axis=1) - log_norm)
    fixed_pdf = np.exp(log_grid[0] - logsumexp(log_grid[0] + np.log(wh)))
    h_q, s_q = pdf_quantiles(h0, h_pdf), pdf_quantiles(scatter, s_pdf)
    edge = (h0 < h0[0] + 0.01 * np.ptp(h0)) | (h0 > h0[-1] - 0.01 * np.ptp(h0))
    edge_mass = float(np.sum(h_pdf[edge] * wh[edge]))
    s_edge = scatter >= 0.9 * scatter[-1]
    scatter_tail = float(np.sum(s_pdf[s_edge] * ws[s_edge])) if len(scatter) > 1 else 0.0
    return {"h0_pdf": h_pdf, "scatter_pdf": s_pdf, "fixed_pdf": fixed_pdf,
            "h0_q": h_q, "scatter_q": s_q, "fixed_q": pdf_quantiles(h0, fixed_pdf),
            "h0_edge_mass": edge_mass, "scatter_upper_tail_mass": scatter_tail}


# 生成未受联合 H0 收缩的源级距离展示点；D_M 输入只在展示 D_L 时乘 (1+z)。
def source_row(data):
    meta = data["meta"]
    q = np.exp(data["mixture"].quantiles())
    # Source-only intervals: flat lnD, no shrinkage to H0, no population scatter.
    factor = (1 + meta["z"]) if meta["distance_kind"] == "comoving" else 1
    row = {k: v for k, v in meta.items() if k not in ("files", "warnings", "bands")}
    row.update({"bands": ",".join(meta["bands"]), "warnings": "; ".join(meta["warnings"])})
    row.update({"D_saved_" + name: float(x) for name, x in zip(Q_NAMES, q)})
    row.update({"DL_" + name: float(x * factor) for name, x in zip(Q_NAMES, q)})
    return row


# 统一本模块静态图风格，不改变统计结果。
def plot_style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": 11,
                         "axes.labelsize": 12, "axes.titlesize": 12,
                         "axes.linewidth": 0.8, "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.top": True,
                         "ytick.right": True, "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})


# 同步保存 PNG/PDF 并关闭图对象。
def save_figure(fig, stem: Path, cfg: dict):
    for suffix in cfg["plot"].get("formats", ["png", "pdf"]):
        if suffix not in ("png", "pdf", "svg"):
            raise ValueError(f"unsupported output format: {suffix}")
        fig.savefig(stem.with_suffix("." + suffix), dpi=cfg["plot"].get("dpi", 220))
    plt.close(fig)


# 用保存的分位数生成 H0 标签。
def h0_label(q):
    return rf"$H_0={q[2]:.1f}^{{+{q[3]-q[2]:.1f}}}_{{-{q[2]-q[1]:.1f}}}$"


# 放置源名，标记仅是绘图元数据。
def annotate_sources(ax, rows):
    """Place the small set of source names without colliding with each other."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    bounds = ax.get_window_extent(renderer)
    occupied = [t.get_window_extent(renderer) for t in ax.texts]
    points = ax.transData.transform([(r["z"], r["DL_q50"]) for r in rows])
    for row in sorted(rows, key=lambda r: r["z"], reverse=True):
        preferred_dx = -5 if row["z"] > 0.8 * ax.get_xlim()[1] else 5
        label = None
        for dy in (6, -12, 18, -24, 30, -36):
            for dx in (preferred_dx, -preferred_dx):
                if label is not None:
                    label.remove()
                label = ax.annotate(row["source"], (row["z"], row["DL_q50"]),
                                    xytext=(dx, dy), textcoords="offset points",
                                    ha="right" if dx < 0 else "left", fontsize=8)
                box = label.get_window_extent(renderer).expanded(1.06, 1.18)
                fits = bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)
                clear = not any(box.overlaps(old) for old in occupied) and not any(box.contains(*p) for p in points)
                if fits and clear:
                    break
            if fits and clear:
                break
        occupied.append(box)


# 每个独立 AGN 画一个光度距离点，模型带表示 H0 参数区间，不是预测散度。
def draw_hubble(ax, result, cfg, compact=False, ymax=None):
    rows, hq = result["rows"], result["fit"]["h0_q"]
    zmax = max(r["z"] for r in rows) * 1.15
    zz = np.linspace(max(1e-5, zmax / 500), zmax, 250)
    scales = np.array([distance_scale(z, cfg["cosmology"]["om0"], "luminosity") for z in zz])
    ax.fill_between(zz, scales / hq[3], scales / hq[1], color="0.86", zorder=0,
                    label=r"68% $H_0$ interval")
    ax.plot(zz, scales / hq[2], color="black", lw=1.2, label="Joint fit")
    reference = float(cfg["plot"].get("reference_h0", 70))
    ax.plot(zz, scales / reference, color="0.45", ls="--", lw=0.9,
            label=rf"$H_0={reference:g}$ reference")
    for r in rows:
        sampling_warning = (r["n_distance_samples"] < cfg["quality"]["warn_distance_samples_below"]
                            or (r["min_mica_ess"] is not None and
                                r["min_mica_ess"] < cfg["quality"]["warn_mica_ess_below"]))
        ax.errorbar(r["z"], r["DL_q50"],
                    yerr=[[r["DL_q50"] - r["DL_q16"]], [r["DL_q84"] - r["DL_q50"]]],
                    fmt="o", color="black", mfc="white" if sampling_warning else "black",
                    ms=3.6 if compact else 4.5, elinewidth=0.8, capsize=2, zorder=3)
    local_max = max(max(r["DL_q84"] for r in rows) * 1.15, float(scales[-1] / hq[1]) * 1.03)
    ax.set(xlim=(0, zmax), ylim=(0, ymax or local_max), xlabel=r"Redshift $z$",
           ylabel=r"Luminosity distance $D_L$ (Mpc)")
    ax.set_title(result["label"], pad=8, fontsize=10 if compact else 12)
    ax.text(0.04, 0.96, f"N = {len(rows)}; " + h0_label(hq), transform=ax.transAxes,
            va="top", fontsize=9 if compact else 11)
    if not compact:
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        annotate_sources(ax, rows)


# 为单个模型结果生成 Hubble 与后验图。
def make_figures(result, cfg, folder, h0):
    fig, ax = plt.subplots(figsize=(7.4, 5.7))
    fig.subplots_adjust(left=0.13, bottom=0.16, top=0.89, right=0.96)
    draw_hubble(ax, result, cfg)
    fig.text(0.13, 0.94, result["model"].capitalize() + " disk; correlated-band distance likelihood", fontsize=10)
    fig.text(0.13, 0.065, "Source bars: 68%; open points: sampling warning. Curve band: H0 uncertainty only.", fontsize=8)
    fig.text(0.13, 0.032, "Model-conditioned; shared calibration and prior-construction uncertainties are not included.", fontsize=8)
    save_figure(fig, folder / "hubble", cfg)
    fit = result["fit"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    fig.subplots_adjust(left=0.12, bottom=0.2, top=0.83, right=0.96)
    ax.plot(h0, fit["h0_pdf"], color="black", lw=1.4, label=r"Marginalize $\sigma_{\rm int}$")
    ax.plot(h0, fit["fixed_pdf"], color="0.45", ls="--", label=r"Fix $\sigma_{\rm int}=0$")
    mask = (h0 >= fit["h0_q"][1]) & (h0 <= fit["h0_q"][3])
    ax.fill_between(h0, 0, fit["h0_pdf"], where=mask, color="0.86", zorder=0)
    xlo = max(h0[0], 0.7 * min(fit["h0_q"][0], fit["fixed_q"][0]))
    xhi = min(h0[-1], 1.25 * max(fit["h0_q"][-1], fit["fixed_q"][-1]))
    ax.set(xlabel=r"$H_0$ (km s$^{-1}$ Mpc$^{-1}$)", ylabel="Posterior density",
           xlim=(xlo, xhi), ylim=(0, None))
    ax.set_title(result["label"] + "; " + result["model"])
    ax.legend(frameon=False, fontsize=9)
    fig.text(0.12, 0.9, h0_label(fit["h0_q"]) + f"; N = {len(result['rows'])}", fontsize=12)
    fig.text(0.12, 0.07, f"Uniform H0 prior: [{h0[0]:g}, {h0[-1]:g}]; plot focuses on the posterior support.", fontsize=8)
    fig.text(0.12, 0.035, "Each curve uses the prior once. Alternative analyses are not independent measurements.", fontsize=8)
    save_figure(fig, folder / "h0_posterior", cfg)


# 并排比较独立模型/数据分支，不相乘不同分析的后验。
def comparison_figures(results, cfg, out):
    for model in cfg["disk_models"]:
        subset = [r for r in results if r["model"] == model]
        if not subset:
            continue
        fig, ax = plt.subplots(figsize=(9.4, 2.2 + 0.55 * len(subset)))
        fig.subplots_adjust(left=0.40, right=0.96, top=0.87, bottom=0.19)
        lo, hi = cfg["plot"].get("reference_interval", [65, 75])
        ax.axvspan(lo, hi, color="0.91", zorder=0)
        ax.axvline(cfg["plot"].get("reference_h0", 70), color="0.55", lw=0.8, ls="--")
        for index, r in enumerate(subset):
            q = r["fit"]["h0_q"]
            ax.plot([q[0], q[4]], [index, index], color="0.5", lw=1)
            ax.plot([q[1], q[3]], [index, index], color="black", lw=2.4)
            ax.plot(q[2], index, "o", color="black", mfc="white" if r["exploratory"] else "black", ms=5)
        ax.set_yticks(range(len(subset)), [r["label"] + f"  (N={len(r['rows'])})" for r in subset], fontsize=10)
        ax.invert_yaxis()
        ax.set(xlabel=r"$H_0$ (km s$^{-1}$ Mpc$^{-1}$)", title=model.capitalize() + " disk: joint H0 constraints")
        ax.tick_params(axis="y", length=0)
        fig.text(0.04, 0.07, "Thick: 68.27%; thin: 95.45%. Source scatter marginalized. Grey strip: reference only.", fontsize=9)
        fig.text(0.04, 0.035, "N and source lists may differ; open marker: outcome-selected, exploratory combination.", fontsize=9)
        save_figure(fig, out / ("h0_comparison_" + model), cfg)
        ncol = min(2, len(subset))
        nrow = (len(subset) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.1 * ncol, 3.7 * nrow), squeeze=False, sharex=True, sharey=True)
        ymax = max(max(x["DL_q84"] for x in r["rows"]) for r in subset) * 1.15
        for r in subset:
            zmax = max(x["z"] for x in r["rows"]) * 1.15
            ymax = max(ymax, distance_scale(zmax, cfg["cosmology"]["om0"], "luminosity") / r["fit"]["h0_q"][1] * 1.03)
        for ax, r in zip(axes.flat, subset):
            draw_hubble(ax, r, cfg, compact=True, ymax=ymax)
        for ax in list(axes.flat)[len(subset):]:
            ax.set_visible(False)
        fig.suptitle(model.capitalize() + " disk: Hubble diagrams", y=0.985, fontsize=15)
        fig.subplots_adjust(left=0.085, right=0.97, top=0.93, bottom=0.09, hspace=0.33, wspace=0.21)
        reference = cfg["plot"].get("reference_h0", 70)
        fig.text(0.085, 0.035, f"Black: joint fit; grey band: 68% H0 interval; dashed: H0={reference:g} reference. Source bars: 68%.", fontsize=10)
        fig.text(0.085, 0.015, "Each panel is a separate fit. Open points flag sampling warnings; shared systematic errors are not included.", fontsize=10)
        save_figure(fig, out / ("hubble_grid_" + model), cfg)


# 检查源/波段/样本策略后读取摘要，在网格上联合推断并写输入清单、后验与图；dry-run 只检查输入。
def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "joint_hubble_config.yaml")
    parser.add_argument("--analysis", nargs="+", help="only these configured analysis names")
    parser.add_argument("--model", nargs="+", choices=["standard", "slim", "free"])
    parser.add_argument("--dry-run", action="store_true", help="validate and list inputs without writing outputs")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_sources(cfg["sources"])
    if cfg.get("missing") not in ("skip", "error"):
        parser.error("missing must be skip or error")
    if cfg.get("sample_policy") not in ("available", "common"):
        parser.error("sample_policy must be available or common")
    analyses = cfg["analyses"]
    if args.analysis:
        if not set(args.analysis) <= set(analyses):
            parser.error("unknown --analysis; use a name in config.analyses")
        analyses = {name: value for name, value in analyses.items() if name in args.analysis}
    cfg["analyses"] = analyses
    if args.model:
        cfg["disk_models"] = args.model
    if (not analyses or not cfg["disk_models"] or len(set(cfg["disk_models"])) != len(cfg["disk_models"])
            or not set(cfg["disk_models"]) <= {"standard", "slim", "free"}):
        parser.error("no analyses selected or invalid disk_models")
    if any(not re.fullmatch(r"[A-Za-z0-9_+-]+", name) for name in analyses):
        parser.error("analysis names must be safe directory names")
    inf = cfg["inference"]
    hmin, hmax = map(float, inf["h0_range"])
    ns, nh, smax = int(inf["intrinsic_scatter_points"]), int(inf["h0_points"]), float(inf["intrinsic_scatter_max"])
    if (not np.isfinite([hmin, hmax, smax]).all() or not 0 < hmin < hmax
            or nh < 101 or smax < 0 or (smax > 0 and ns < 5)):
        parser.error("invalid H0/intrinsic-scatter grid")
    # Nonuniform grid, but integration weights are dH0: the prior stays flat H0.
    h0 = np.geomspace(hmin, hmax, nh)
    scatter = np.linspace(0, smax, ns) if smax > 0 else np.zeros(1)
    loaded, cases, omissions = {}, [], []
    for name, analysis in analyses.items():
        unknown = (set(analysis.get("source_datasets", {})) | set(analysis.get("exclude", []))) - set(cfg["sources"])
        if unknown:
            parser.error(f"{name}: unrecognized source override/exclusion: {sorted(unknown)}")
        for model in cfg["disk_models"]:
            keys = []
            for source in cfg["sources"]:
                if source in analysis.get("exclude", []):
                    omissions.append(dict(analysis=name, disk_model=model, source=source, reason="explicitly excluded in config"))
                    continue
                dataset = analysis.get("source_datasets", {}).get(source, analysis["dataset"])
                if dataset not in cfg["datasets"]:
                    parser.error(f"unknown dataset {dataset}")
                key = (source, dataset, model)
                if key not in loaded:
                    try:
                        loaded[key] = load_source(root, source, dataset, cfg["datasets"][dataset], model, cfg)
                    except FileNotFoundError as exc:
                        if cfg["missing"] == "error":
                            raise
                        loaded[key] = {"missing": str(exc)}
                if "missing" in loaded[key]:
                    omissions.append(dict(analysis=name, disk_model=model, source=source,
                                          dataset=dataset, reason="missing: " + loaded[key]["missing"]))
                else:
                    keys.append(key)
            cases.append((name, model, keys))
    if cfg["sample_policy"] == "common":
        common = set.intersection(*(set(k[0] for k in keys) for _, _, keys in cases))
        filtered = []
        for name, model, keys in cases:
            for key in keys:
                if key[0] not in common:
                    omissions.append(dict(analysis=name, disk_model=model, source=key[0],
                                          dataset=key[1], reason="not in common sample"))
            filtered.append((name, model, [k for k in keys if k[0] in common]))
        cases = filtered
    for name, model, keys in cases:
        print(f"{name}/{model}: N={len(keys)}; " + ", ".join(k[0] for k in keys), flush=True)
    if args.dry_run:
        print(f"{len(omissions)} explicitly listed unavailable/excluded source-model entries; no output written.")
        return
    if any(len(keys) < 2 for _, _, keys in cases):
        parser.error("every selected analysis/model needs at least two available independent sources")

    out = under(root, cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    plot_style()
    summaries, results, grid_cache = [], [], {}
    for name, model, keys in cases:
        analysis = analyses[name]
        total = np.zeros((len(scatter), len(h0)))
        data = [loaded[key] for key in keys]
        for key in keys:
            if key not in grid_cache:
                grid_cache[key] = source_log_grid(loaded[key], h0, scatter)
            total += grid_cache[key]
        fit = fit_joint(total, h0, scatter)
        rows = [source_row(item) for item in data]
        fit_warnings = []
        if fit["h0_edge_mass"] > 0.01:
            fit_warnings.append("H0 posterior reaches prior edges; widen h0_range before interpreting")
        if fit["scatter_upper_tail_mass"] > 0.05:
            fit_warnings.append("intrinsic-scatter upper prior may matter")
        if len(set(r["inclination_deg"] for r in rows)) > 1:
            fit_warnings.append("source distances use different fixed inclinations")
        result = dict(name=name, label=analysis.get("label", name), model=model,
                      exploratory=bool(analysis.get("exploratory", False)), rows=rows, fit=fit)
        folder = out / name / model
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(folder / "source_distances.csv", index=False, float_format="%.10g")
        pd.DataFrame({"h0": h0, "pdf": fit["h0_pdf"], "pdf_sigma_int_zero": fit["fixed_pdf"]}).to_csv(
            folder / "h0_posterior.csv", index=False, float_format="%.10g")
        pd.DataFrame({"sigma_int_lnD": scatter, "pdf": fit["scatter_pdf"]}).to_csv(
            folder / "scatter_posterior.csv", index=False, float_format="%.10g")
        log_norm = logsumexp(total + np.log(integration_weights(scatter))[:, None]
                            + np.log(integration_weights(h0))[None, :])
        np.savez_compressed(folder / "joint_posterior.npz", h0=h0, sigma_int=scatter,
                            pdf=np.exp(total - log_norm))
        summary = dict(analysis=name, label=result["label"], disk_model=model, n_sources=len(keys),
                       sources=",".join(k[0] for k in keys), exploratory=result["exploratory"],
                       h0_edge_mass=fit["h0_edge_mass"], scatter_upper_tail_mass=fit["scatter_upper_tail_mass"],
                       warnings="; ".join(fit_warnings))
        for prefix, q in (("h0", fit["h0_q"]), ("h0_sigma0", fit["fixed_q"]), ("sigma_int", fit["scatter_q"])):
            summary.update({prefix + "_" + qn: float(v) for qn, v in zip(Q_NAMES, q)})
        summaries.append(summary)
        results.append(result)
        (folder / "inputs.json").write_text(json.dumps([d["meta"] for d in data], indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        make_figures(result, cfg, folder, h0)
        q = fit["h0_q"]
        print(f"Saved {name}/{model}: H0={q[2]:.2f} -{q[2]-q[1]:.2f} +{q[3]-q[2]:.2f}; N={len(keys)}"
              + ("; WARNING: " + "; ".join(fit_warnings) if fit_warnings else ""), flush=True)

    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False, float_format="%.10g")
    pd.DataFrame(omissions, columns=["analysis", "disk_model", "source", "dataset", "reason"]).to_csv(out / "missing_or_excluded.csv", index=False)
    comparison_figures(results, cfg, out)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
                "config_file": file_record(args.config.resolve()), "code_file": file_record(Path(__file__).resolve()),
                "summaries": summaries, "omissions": omissions,
                "input_sets": [d["meta"] for d in loaded.values() if "meta" in d]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    notes = """# 联合 H0 与哈勃图

## 阅读顺序

先看 `h0_comparison_standard.png` 和 `hubble_grid_standard.png`，然后查看各方案/盘模型目录。
`summary.csv` 给出联合分位数；`source_distances.csv` 是作图数据；`inputs.json` 与 `manifest.json`
记录实际输入、文件哈希、采样警告及配置。缺失数据见 `missing_or_excluded.csv`，绝不自动换版本。
standard / slim / free 是距离计算采用的盘模型，不是转移函数的形状。
只重写本次选中的输出，不删除以往未选中的图；本次有效清单以 `manifest.json` 为准。

## 统计定义

沿用原流程保存的自然对数距离均值和完整波段协方差，不将波段当独立源，不除以样本数或波段数。
每个源独立边缘化额外波段散度 f_i；f_i 不是 MICA 的转移函数面积占比。
另设源间 lnD 散度 sigma_int，对该源所有波段增加共同偏移，而非独立波段噪声。
本动速度以一阶红移误差传播加入同一共同偏移。H0 先验只在联合计算中使用一次。
H0 网格按对数间隔布点，但积分权重是 dH0，先验仍然是均匀 H0，而非均匀 logH0。
主结果对 sigma_int 边缘化；sigma_int=0 的条件结果另列，不应优先挑更窄的一条。
所有区间为等尾 68.27% / 95.45%，不是最高密度区间。图中点的区间是单源 lnD 似然乘平坦 lnD
测度所得，不借联合 H0 收缩；曲线灰带只表示 H0 参数误差，并非新源预测带。
不直接相乘旧单源 H0 后验，因为它们已经带有先验且常被 [20,120] 截断。

## 距离与校准边界

现有代码使用观测系 tau、lambda 和 Fnu：tau_obs lambda_obs^(-3/2) Fnu_obs^(-1/2)
对应 D_L/(1+z)=D_M。因此默认将保存的 D 当共动距离，画图时乘 (1+z) 得 D_L。
配置 distance_kind=luminosity 可按旧代码的距离约定比较，不会覆盖原有距离结果。
距离关系出处：[Hogg 1999](https://arxiv.org/abs/astro-ph/9905116)，
[光度距离定义](https://ned.ipac.caltech.edu/level5/Hogg/Hogg7.html)。
红移沿用源配置，尚未统一核验 CMB/流场修正；低红移源的速度处理只是近似。

这仍是给定盘模型、固定倾角、输入先验和现有距离后验的近似联合约束，不是完整的端到端宇宙学似然。
共同倾角/盘归一化、消光、重处理几何等共享系统误差没有计入；增加独立源间散度不等于处理共享校准误差。
物理先验若由采用预设 H0 的光度/尺寸关系构造，其依赖也未重新传播，不能视为完全独立的 H0 测量。
原距离流程对 lag 与 FFA 后验独立抽取的近似仍保留，额外误差模型无法补救采样未收敛。
样本数是保存的距离样本数，不是有效样本数；MICA ESS 另列。警告只展示，不自动择优删源。
逐源 selected_2_3comp 来自此前按结果筛选的选择，存在选择偏差，只作探索性对照。
不同方案若缺失名单不同，比较同时混入样本变化；需要固定样本时将 sample_policy 改成 common。

## 重跑

在项目根目录执行 `python main.py hubble`；它会调用本脚本和 `joint_hubble_config.yaml`。
也可直接执行 `python src/joint_hubble.py --config joint_hubble_config.yaml`。
仅标准盘：追加 `--model standard`；仅某方案：追加 `--analysis gaussian2_free`。
先检查输入而不出图：追加 `--dry-run`。不需要 MPI，也不会运行 MICA、FFA 或 ICCF。
"""
    (out / "README.md").write_text(notes, encoding="utf-8")
    print(f"Complete: {len(results)} separate fits; outputs in {out}", flush=True)


if __name__ == "__main__":
    main()
