#!/usr/bin/env python3
"""Compare total-light and short-component continuum lags with PyAT ICCF."""
from __future__ import annotations

# 对比链：去宿主 total 光变与 MICA 原生分量 -> 同观测时刻的 total/short -> PyAT ICCF。
# short 是使用同一批观测反演出的模型重建，因此本脚本提供内部诊断，不构成独立恢复验证。
import argparse
import hashlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
# PyAT 提供插值互相关和 Monte Carlo 延迟分布；这里没有再次调用 MICA 或 FFA 拟合。
from pyat.ccf import iccf, iccf_mc

from collect_mica_results import (
    configured_run_name,
    configured_stages,
    mica_analysis_dir,
    mica_run_root,
    model_label as canonical_model_label,
    ncomp_from_config as configured_ncomp,
    stage_config,
)


# tau 的单位沿用输入时间（本项目为天）；没有在本脚本统一除以 1+z 转为静止系。
ROOT = Path(__file__).resolve().parents[1]
DEFAULTS: dict[str, Any] = {
    "output_dir": "results/iccf",
    "branch": "comp0",
    "component_index": 0,
    "tau_range": [-10.0, 10.0],
    "ntau": 241,
    "threshold": 0.8,
    "nsim": 10000,
    "mode": "single",
    "max_retries": 5,
    "seed": 12345,
    "u_band": "U",
    "u_residual_nsim": 10000,
    "u_residual_beta": 4.0 / 3.0,
}

TF_LABELS = {
    "gaussian": "gaussian",
    "tophat": "tophat",
    "gamma": "gamma",
    "exponential": "exp",
    "exp": "exp",
}


def source_dir(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def rel(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


# stage 决定上游 MICA 配置；ICCF 小节覆盖数值默认值，输出目录随后按原生 comp0 路径重设。
# 若需明确改输出位置，使用 run 的 output_override/命令行 --output-dir。
def read_config(src: Path, stage: str) -> tuple[dict, dict, dict]:
    path = src / "config" / "source_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"source configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    fit = stage_config(cfg, stage)
    fit.setdefault("output_root", "runs/mica")
    fit.setdefault("driver", cfg.get("reference_band", cfg.get("driver_band")))
    fit.setdefault("responses", cfg.get("response_bands", []))

    opt = dict(DEFAULTS)
    opt.update(cfg.get("iccf", {}) or {})
    opt["output_dir"] = str(mica_analysis_dir(src, fit, "comp0") / "iccf")
    return cfg, fit, opt


def ncomp_from_config(fit: dict) -> int:
    return configured_ncomp(fit)


def model_label(fit: dict) -> str:
    return canonical_model_label(fit.get("type_tf", "gaussian"))


# 先检查采样窗口、网格点数、阈值与重复数；这些是估计器设置，不是拟合结果质量判据。
def validate_options(opt: dict) -> None:
    try:
        tau_beg, tau_end = map(float, opt["tau_range"])
    except (TypeError, ValueError) as exc:
        raise ValueError("iccf.tau_range must contain two finite numbers") from exc
    if not np.isfinite([tau_beg, tau_end]).all() or tau_beg >= tau_end:
        raise ValueError("iccf.tau_range must satisfy tau_beg < tau_end")
    if int(opt["ntau"]) < 3:
        raise ValueError("iccf.ntau must be at least 3")
    if int(opt["nsim"]) < 1:
        raise ValueError("iccf.nsim must be positive")
    if int(opt["u_residual_nsim"]) < 1:
        raise ValueError("iccf.u_residual_nsim must be positive")
    if not 0.0 < float(opt["threshold"]) <= 1.0:
        raise ValueError("iccf.threshold must be in (0, 1]")
    if str(opt["mode"]) not in {"single", "multiple"}:
        raise ValueError("iccf.mode must be 'single' or 'multiple'")
    if int(opt["max_retries"]) < 1:
        raise ValueError("iccf.max_retries must be positive")
    if not np.isfinite(float(opt["u_residual_beta"])):
        raise ValueError("iccf.u_residual_beta must be finite")


def wavelengths_from_config(cfg: dict) -> dict[str, float]:
    values: dict[str, Any] = {}
    top = cfg.get("bands", {})
    if isinstance(top, dict):
        values.update(top)
    for section in ("fflux", "fflux_comp0"):
        values.update((cfg.get(section, {}) or {}).get("wavelengths", {}) or {})
    waves = {str(key): float(value) for key, value in values.items()}
    if not waves:
        raise ValueError("no band wavelengths found in the source configuration")
    return waves


def mica_ncomp_root(src: Path, cfg: dict, fit: dict) -> Path:
    root = mica_run_root(src, fit)
    if not (root / "result").is_dir():
        raise FileNotFoundError(f"cannot locate the MICA result root: {root}")
    return root


def run_name(fit: dict, response: str) -> str:
    return configured_run_name(fit, response)


# 当前实现按原生 MICA run 取 pline；branch 参数保留在接口中，但不据此读取条件分支目录。
def branch_run_dir(
    src: Path,
    cfg: dict,
    fit: dict,
    response: str,
    branch: str,
    component_index: int,
) -> Path:
    root = mica_ncomp_root(src, cfg, fit)
    name = run_name(fit, response)
    candidates = [root / name]
    ncomp = ncomp_from_config(fit)
    expected = f"pline.txt_{ncomp}_comp{component_index}"
    for path in candidates:
        if (path / "data" / expected).is_file():
            return path
    checked = "\n  ".join(str(path / "data" / expected) for path in candidates)
    raise FileNotFoundError(
        f"native full-posterior reconstruction for {response} not found; checked:\n  {checked}"
    )


def host_sub_dir(src: Path, cfg: dict) -> Path:
    ff = dict(cfg.get("fflux", {}) or {})
    host = dict(ff.get("host_sub", {}) or {})
    if host.get("output_dir"):
        return rel(src, host["output_dir"])
    return rel(src, ff.get("out", "fflux")) / "host_sub"


def counts(path: Path) -> list[int]:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("#"):
            break
        values = [
            int(token)
            for token in line.replace("#", " ").replace(":", " ").split()
            if token.isdigit()
        ]
        if len(values) >= 2:
            return values[:2]
    return []


def validate_series(series: dict, minimum: int = 5) -> dict:
    label = f"{series['band']} ({series['path']})"
    t = np.asarray(series["t"], float)
    f = np.asarray(series["f"], float)
    e = np.asarray(series["e"], float)
    if not (len(t) == len(f) == len(e)):
        raise ValueError(f"time, flux, and error lengths differ for {label}")
    if len(t) < minimum:
        raise ValueError(f"fewer than {minimum} valid points for {label}")
    if not (np.isfinite(t).all() and np.isfinite(f).all() and np.isfinite(e).all()):
        raise ValueError(f"non-finite values remain in {label}")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError(f"times are not strictly increasing for {label}")
    if np.any(e <= 0.0):
        raise ValueError(f"non-positive flux errors found for {label}")
    if not np.isfinite(np.std(f)) or np.std(f) <= 0.0:
        raise ValueError(f"flux has zero variance for {label}")
    series["t"], series["f"], series["e"] = t, f, e
    return series


# 读三列光变，可按 MICA 计数头提取块；先去掉非有限/非正误差行，再按时间排序。
# 重复时刻并未合并，仍会在 validate_series 的严格单调检查中失败。
def load_series(
    path: Path,
    band: str,
    segment: int | None = None,
    count_file: Path | None = None,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"light curve not found: {path}")
    try:
        data = np.loadtxt(path, comments="#")
    except ValueError as exc:
        raise ValueError(f"cannot parse light curve: {path}") from exc
    if data.size == 0:
        raise ValueError(f"empty light curve: {path}")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 3:
        raise ValueError(f"light curve must have at least three columns: {path}")
    data = np.asarray(data[:, :3], float)

    if segment is not None:
        ns = counts(count_file or path)
        if not ns or segment < 0 or segment >= len(ns):
            raise ValueError(f"cannot identify segment {segment} in {count_file or path}")
        start = sum(ns[:segment])
        stop = start + ns[segment]
        if stop > len(data):
            raise ValueError(f"segment counts exceed the data length in {path}")
        data = data[start:stop]

    good = np.isfinite(data).all(axis=1) & (data[:, 2] > 0.0)
    data = data[good]
    data = data[np.argsort(data[:, 0])]
    return validate_series(
        {"band": band, "t": data[:, 0], "f": data[:, 1], "e": data[:, 2], "path": path}
    )


def load_observed(src: Path, cfg: dict, source: str, band: str) -> dict:
    return load_series(host_sub_dir(src, cfg) / f"{source}_{band}.txt", band)


# pline 的第 0 块是参考波段，第 1 块是响应；pall 只用于核对两者块长度约定一致。
def load_short_grid(
    src: Path,
    cfg: dict,
    fit: dict,
    band: str,
    branch: str,
    component_index: int,
) -> dict:
    rd = branch_run_dir(src, cfg, fit, band, branch, component_index)
    ncomp = ncomp_from_config(fit)
    pall = rd / "data" / f"pall.txt_{ncomp}"
    pline = rd / "data" / f"pline.txt_{ncomp}_comp{component_index}"
    if counts(pall) != counts(pline):
        raise ValueError(f"MICA segment headers differ between {pall} and {pline}")
    return load_series(pline, band, segment=1, count_file=pall)


# total/short 只比较同一组实际观测时刻：不把密集重建网格伪装成额外观测点。
# short 的流量和边际误差插值到这些时刻，但 MICA 重建的时间相关误差未传入 PyAT。
def match_short_to_observed(short_grid: dict, observed: dict) -> tuple[dict, dict]:
    """Return total and short responses on the same observed epochs."""
    lo, hi = short_grid["t"][[0, -1]]
    mask = (observed["t"] >= lo) & (observed["t"] <= hi)
    times = observed["t"][mask]
    if len(times) < 5:
        raise ValueError(
            f"fewer than five {observed['band']} observations fall inside the short-component "
            f"coverage [{lo:.6f}, {hi:.6f}]"
        )
    matched_observed = validate_series(
        {
            "band": observed["band"],
            "t": times,
            "f": observed["f"][mask],
            "e": observed["e"][mask],
            "path": observed["path"],
        }
    )
    short = {
        "band": observed["band"],
        "t": times,
        "f": np.interp(times, short_grid["t"], short_grid["f"]),
        "e": np.interp(times, short_grid["t"], short_grid["e"]),
        "path": short_grid["path"],
    }
    return matched_observed, validate_series(short)


# 恢复校验绑定实际 t/f/e 数组与估计设置，防止只因文件同名就复用另一批 MC 产物。
def pair_signature(driver: dict, response: dict, opt: dict) -> str:
    """Fingerprint the numerical inputs and ICCF settings used by one pair."""
    digest = hashlib.sha256()
    for series in (driver, response):
        for key in ("t", "f", "e"):
            values = np.ascontiguousarray(np.asarray(series[key], dtype="<f8"))
            digest.update(key.encode("ascii"))
            digest.update(values.tobytes())
    settings = (
        tuple(map(float, opt["tau_range"])),
        int(opt["ntau"]),
        float(opt["threshold"]),
        int(opt["nsim"]),
        str(opt["mode"]),
        int(opt["max_retries"]),
    )
    digest.update(repr(settings).encode("utf-8"))
    return digest.hexdigest()


# 估计对象是参考波段与响应波段的插值相关函数，非响应核本身。
# peak 对应相关峰位置，cent 由 PyAT 的 threshold/mode 定义；不要与 MICA 核质心混为同一估计量。
def run_pair(driver: dict, response: dict, opt: dict) -> dict:
    tau_beg, tau_end = map(float, opt["tau_range"])
    ntau = int(opt["ntau"])
    threshold = float(opt["threshold"])
    mode = str(opt["mode"])
    retries = int(opt["max_retries"])

    if driver["t"][0] > response["t"][-1] or response["t"][0] > driver["t"][-1]:
        raise ValueError(
            f"driver and {response['band']} response have no temporal overlap"
        )

    tau, curve, rmax, peak, cent = iccf(
        driver["t"],
        driver["f"],
        response["t"],
        response["f"],
        ntau,
        tau_beg,
        tau_end,
        threshold=threshold,
        mode=mode,
        ignore_warning=True,
        max_retries=retries,
    )

    # 单次 MC 调用返回有效的相关峰/峰位置/质心样本；无效结果随后被共同掩码剔除。
    # 请求次数与保留次数需一起检查，窄分布并不自动说明覆盖率可靠。
    # PyAT resets its internal random seed in iccf_mc.  One call per light-curve
    # pair is therefore essential: splitting a run into chunks repeats draws.
    r_mc, peak_mc, cent_mc = iccf_mc(
        driver["t"],
        driver["f"],
        driver["e"],
        response["t"],
        response["f"],
        response["e"],
        ntau,
        tau_beg,
        tau_end,
        nsim=int(opt["nsim"]),
        threshold=threshold,
        mode=mode,
        ignore_warning=True,
        max_retries=retries,
    )
    r_mc = np.asarray(r_mc, float)
    peak_mc = np.asarray(peak_mc, float)
    cent_mc = np.asarray(cent_mc, float)
    if not (len(r_mc) == len(peak_mc) == len(cent_mc)):
        raise RuntimeError("PyAT returned ICCF Monte Carlo arrays with different lengths")
    finite = np.isfinite(r_mc) & np.isfinite(peak_mc) & np.isfinite(cent_mc)
    r_mc, peak_mc, cent_mc = r_mc[finite], peak_mc[finite], cent_mc[finite]
    if len(cent_mc) == 0:
        raise RuntimeError(f"PyAT returned no positive-peak simulations for {response['band']}")
    return {
        "tau": np.asarray(tau, float),
        "curve": np.asarray(curve, float),
        "rmax": float(rmax),
        "peak": float(peak),
        "cent": float(cent),
        "r_mc": r_mc,
        "peak_mc": peak_mc,
        "cent_mc": cent_mc,
        "n_driver": len(driver["t"]),
        "n_response": len(response["t"]),
        "n_mc_requested": int(opt["nsim"]),
        "pair_signature": pair_signature(driver, response, opt),
        "driver_path": str(driver["path"]),
        "response_path": str(response["path"]),
    }


def quantiles(values: np.ndarray | list[float], fallback: float = np.nan) -> tuple[float, float, float]:
    array = np.asarray(values, float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return fallback, fallback, fallback
    return tuple(float(value) for value in np.percentile(array, [16.0, 50.0, 84.0]))


def save_pair(out: Path, source: str, band: str, stage: str, result: dict) -> None:
    pd.DataFrame({"tau_days": result["tau"], "r": result["curve"]}).to_csv(
        out / f"{source}_{band}_{stage}_curve.csv", index=False
    )
    pd.DataFrame(
        {
            "r_peak": result["r_mc"],
            "tau_peak_days": result["peak_mc"],
            "tau_cent_days": result["cent_mc"],
        }
    ).to_csv(out / f"{source}_{band}_{stage}_mc.csv", index=False)
    with (out / f"{source}_{band}_{stage}_status.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(
            {
                "n_mc_requested": int(result["n_mc_requested"]),
                "n_mc_returned": int(len(result["cent_mc"])),
                "pair_signature": str(result["pair_signature"]),
                "complete": True,
            },
            handle,
            sort_keys=False,
        )


# --resume 只有数值指纹、请求数、落盘行数与完成标记同时匹配才接受旧 MC。
# 它复用 MC 分布，但仍会重算确定性的 CCF 曲线；不是任意半成品的续接。
def load_complete_pair(
    out: Path,
    source: str,
    band: str,
    stage: str,
    driver: dict,
    response: dict,
    opt: dict,
) -> dict | None:
    """Load a finished MC product; never accept a partial simulation file."""
    mc_path = out / f"{source}_{band}_{stage}_mc.csv"
    if not mc_path.is_file():
        return None
    mc = pd.read_csv(mc_path)
    required = {"r_peak", "tau_peak_days", "tau_cent_days"}
    if not required.issubset(mc.columns):
        return None
    requested = int(opt["nsim"])
    expected_signature = pair_signature(driver, response, opt)
    status_path = out / f"{source}_{band}_{stage}_status.yaml"
    if not status_path.is_file():
        return None
    with status_path.open("r", encoding="utf-8") as handle:
        status = yaml.safe_load(handle) or {}
    complete = (
        bool(status.get("complete", False))
        and int(status.get("n_mc_requested", -1)) == requested
        and int(status.get("n_mc_returned", -1)) == len(mc)
        and str(status.get("pair_signature", "")) == expected_signature
    )
    if not complete:
        return None

    r_mc = pd.to_numeric(mc["r_peak"], errors="coerce").to_numpy(float)
    peak_mc = pd.to_numeric(mc["tau_peak_days"], errors="coerce").to_numpy(float)
    cent_mc = pd.to_numeric(mc["tau_cent_days"], errors="coerce").to_numpy(float)
    if not (
        np.isfinite(r_mc).all()
        and np.isfinite(peak_mc).all()
        and np.isfinite(cent_mc).all()
    ):
        return None

    tau_beg, tau_end = map(float, opt["tau_range"])
    tau, curve, rmax, peak, cent = iccf(
        driver["t"],
        driver["f"],
        response["t"],
        response["f"],
        int(opt["ntau"]),
        tau_beg,
        tau_end,
        threshold=float(opt["threshold"]),
        mode=str(opt["mode"]),
        ignore_warning=True,
        max_retries=int(opt["max_retries"]),
    )
    return {
        "tau": np.asarray(tau, float),
        "curve": np.asarray(curve, float),
        "rmax": float(rmax),
        "peak": float(peak),
        "cent": float(cent),
        "r_mc": r_mc,
        "peak_mc": peak_mc,
        "cent_mc": cent_mc,
        "n_driver": len(driver["t"]),
        "n_response": len(response["t"]),
        "n_mc_requested": requested,
        "pair_signature": expected_signature,
        "driver_path": str(driver["path"]),
        "response_path": str(response["path"]),
    }


def summary_row(band: str, wavelength: float, stage: str, result: dict) -> dict:
    c16, c50, c84 = quantiles(result["cent_mc"], result["cent"])
    p16, p50, p84 = quantiles(result["peak_mc"], result["peak"])
    return {
        "band": band,
        "wavelength": wavelength,
        "stage": stage,
        "kind": "ccf",
        "tau_cent": c50,
        "tau_cent_err_low": c50 - c16,
        "tau_cent_err_high": c84 - c50,
        "tau_peak": p50,
        "tau_peak_err_low": p50 - p16,
        "tau_peak_err_high": p84 - p50,
        "rmax": result["rmax"],
        "n_mc": len(result["cent_mc"]),
        "n_mc_requested": result["n_mc_requested"],
        "mc_valid_fraction": len(result["cent_mc"]) / result["n_mc_requested"],
        "n_driver": result["n_driver"],
        "n_response": result["n_response"],
        "driver_file": result["driver_path"],
        "response_file": result["response_path"],
    }


def mica_summary(
    src: Path,
    cfg: dict,
    fit: dict,
    branch: str,
    component_index: int,
) -> pd.DataFrame:
    path = (
        mica_ncomp_root(src, cfg, fit)
        / "result"
        / "mica_component_summary.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(f"MICA component summary not found: {path}")
    table = pd.read_csv(path)
    required = {"band", "component", "tau", "tau_err_low", "tau_err_high"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    component = f"tf{component_index}"
    table = table[table["component"].astype(str).eq(component)].copy()
    if table.empty:
        raise ValueError(f"no {component} rows in {path}")
    if table["band"].astype(str).duplicated().any():
        raise ValueError(f"duplicate {component} bands in {path}")
    return table[["band", "tau", "tau_err_low", "tau_err_high"]].rename(
        columns={
            "tau": "mica_tau",
            "tau_err_low": "mica_tau_err_low",
            "tau_err_high": "mica_tau_err_high",
        }
    )


# 对照表展示三种估计量的数值差；差列没有联合误差传播，不能直接解释成显著性。
def comparison_table(summary: pd.DataFrame, mica: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "band",
        "wavelength",
        "tau_cent",
        "tau_cent_err_low",
        "tau_cent_err_high",
        "rmax",
    ]
    total = summary[summary["stage"].eq("total")][columns].rename(
        columns={column: f"total_{column}" for column in columns if column != "band"}
    )
    short = summary[summary["stage"].eq("short")][columns].rename(
        columns={column: f"short_{column}" for column in columns if column != "band"}
    )
    result = total.merge(short, on="band", validate="one_to_one")
    result = result.merge(mica, on="band", how="left", validate="one_to_one")
    result["total_minus_short"] = result["total_tau_cent"] - result["short_tau_cent"]
    result["short_minus_mica"] = result["short_tau_cent"] - result["mica_tau"]
    return result.sort_values("total_wavelength")


# 带权分位数先按值排序，再反演归一化累积权重；权重不等时不能用普通 percentile 替代。
def weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: tuple[float, ...] = (0.16, 0.5, 0.84),
) -> tuple[float, ...]:
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values, weights = values[valid], weights[valid]
    if len(values) == 0:
        raise ValueError("weighted posterior contains no finite positive-weight samples")
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return tuple(float(np.interp(probability, cdf, values)) for probability in probabilities)


# 按波段提取 MICA 的边际 tau 后验及归一化权重；此接口没有保留跨波段联合抽样身份。
def mica_lag_posteriors(
    src: Path,
    cfg: dict,
    fit: dict,
    branch: str,
    component_index: int,
    responses: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    path = (
        mica_ncomp_root(src, cfg, fit)
        / "result"
        / "mica_component_posterior.txt"
    )
    if not path.is_file():
        raise FileNotFoundError(f"MICA component posterior not found: {path}")
    table = pd.read_csv(path, sep=r"\s+")
    required = {"band", "component", "tau", "weight"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    component = f"tf{component_index}"
    table = table[table["component"].astype(str).eq(component)].copy()
    posteriors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for band in responses:
        selected = table[table["band"].astype(str).eq(band)]
        tau = pd.to_numeric(selected["tau"], errors="coerce").to_numpy(float)
        weight = pd.to_numeric(selected["weight"], errors="coerce").to_numpy(float)
        valid = np.isfinite(tau) & np.isfinite(weight) & (weight > 0.0)
        tau, weight = tau[valid], weight[valid]
        if len(tau) == 0 or float(np.sum(weight)) <= 0.0:
            raise ValueError(f"no valid {component} lag posterior for {band} in {path}")
        posteriors[band] = (tau, weight / np.sum(weight))
    return posteriors


def draw_from_mc(result: dict, size: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(result["cent_mc"], float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("cannot sample an empty ICCF centroid distribution")
    return rng.choice(values, size=size, replace=True)


# 留 U 诊断固定 beta，使用非 U 波段估计 tau0，再比较 U 实际延迟与预测值。
# 关系式 tau(lambda)=tau0*((lambda/lambda_ref)^beta-1) 在参考波段处强制为零。
def u_lag_residuals(
    results: dict[tuple[str, str], dict],
    responses: list[str],
    waves: dict[str, float],
    driver: str,
    opt: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    u_band = str(opt["u_band"])
    if u_band not in responses:
        raise ValueError(f"configured U diagnostic band {u_band!r} is not a response band")
    fit_bands = [band for band in responses if band != u_band]
    if len(fit_bands) < 2:
        raise ValueError("leave-U-out lag fitting requires at least two non-U response bands")
    missing = [band for band in [driver, *responses] if band not in waves]
    if missing:
        raise KeyError("missing wavelengths for: " + ", ".join(missing))

    nsamp = int(opt["u_residual_nsim"])
    beta = float(opt["u_residual_beta"])
    lambda_ref = waves[driver]
    x_fit = np.asarray([(waves[band] / lambda_ref) ** beta - 1.0 for band in fit_bands])
    x_u = (waves[u_band] / lambda_ref) ** beta - 1.0

    frames, rows = [], []
    for offset, stage in enumerate(("total", "short")):
        rng = np.random.default_rng(int(opt["seed"]) + 104729 * offset)
        sigma = np.asarray(
            [
                0.5
                * (
                    quantiles(results[(stage, band)]["cent_mc"])[2]
                    - quantiles(results[(stage, band)]["cent_mc"])[0]
                )
                for band in fit_bands
            ],
            float,
        )
        if not np.isfinite(sigma).all() or np.any(sigma <= 0.0):
            raise ValueError(f"invalid ICCF centroid uncertainties for {stage} lag fit")
        # 以边际 68% 区间半宽近似 sigma，闭式加权最小二乘：
        # tau0 = sum(w_i*x_i*tau_i) / sum(w_i*x_i^2)，每个抽样重复同一映射。
        weight = 1.0 / np.square(sigma)
        denominator = float(np.sum(weight * np.square(x_fit)))
        if denominator <= 0.0:
            raise ValueError("non-U wavelengths do not constrain the fixed-slope lag relation")
        # 各波段边际分别重抽样；共用参考光变引起的跨波段相关没有进入该联合分布。
        draws = np.column_stack(
            [draw_from_mc(results[(stage, band)], nsamp, rng) for band in fit_bands]
        )
        tau_u = draw_from_mc(results[(stage, u_band)], nsamp, rng)
        tau0 = draws @ (weight * x_fit) / denominator
        predicted = tau0 * x_u
        residual = tau_u - predicted
        frames.append(
            pd.DataFrame(
                {
                    "sample": np.arange(nsamp, dtype=int),
                    "stage": stage,
                    "beta": beta,
                    "tau0_days": tau0,
                    "tau_u_days": tau_u,
                    "tau_u_predicted_days": predicted,
                    "u_lag_residual_days": residual,
                }
            )
        )
        t016, t050, t084 = quantiles(tau0)
        u16, u50, u84 = quantiles(tau_u)
        p16, p50, p84 = quantiles(predicted)
        r16, r50, r84 = quantiles(residual)
        rows.append(
            {
                "stage": stage,
                "u_band": u_band,
                "beta": beta,
                "fit_bands": ",".join(fit_bands),
                "fit_weighting": "inverse_variance_from_marginal_iccf_centroid_68pct",
                "posterior_construction": "independent_resampling_of_band_marginals",
                "n_samples": nsamp,
                "tau0": t050,
                "tau0_err_low": t050 - t016,
                "tau0_err_high": t084 - t050,
                "tau_u": u50,
                "tau_u_err_low": u50 - u16,
                "tau_u_err_high": u84 - u50,
                "tau_u_predicted": p50,
                "tau_u_predicted_err_low": p50 - p16,
                "tau_u_predicted_err_high": p84 - p50,
                "u_lag_residual": r50,
                "u_lag_residual_err_low": r50 - r16,
                "u_lag_residual_err_high": r84 - r50,
                "probability_residual_gt_zero": float(np.mean(residual > 0.0)),
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(rows)


# MICA 对照沿用同一个固定 beta 的留 U 关系，但改用带权 MICA 边际后验抽样。
# 得到的 residual>0 比例是该条件模型和独立边际近似下的量，不是自动校准的检验 p 值。
def mica_u_lag_residuals(
    posteriors: dict[str, tuple[np.ndarray, np.ndarray]],
    responses: list[str],
    waves: dict[str, float],
    driver: str,
    opt: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the requested total-ICCF versus short-MICA U-lag comparison."""
    u_band = str(opt["u_band"])
    fit_bands = [band for band in responses if band != u_band]
    nsamp = int(opt["u_residual_nsim"])
    beta = float(opt["u_residual_beta"])
    lambda_ref = waves[driver]
    x_fit = np.asarray([(waves[band] / lambda_ref) ** beta - 1.0 for band in fit_bands])
    x_u = (waves[u_band] / lambda_ref) ** beta - 1.0

    sigma = np.asarray(
        [
            0.5
            * (
                weighted_quantiles(*posteriors[band])[2]
                - weighted_quantiles(*posteriors[band])[0]
            )
            for band in fit_bands
        ],
        float,
    )
    if not np.isfinite(sigma).all() or np.any(sigma <= 0.0):
        raise ValueError("invalid MICA marginal lag uncertainties for leave-U-out fit")
    weight_fit = 1.0 / np.square(sigma)
    denominator = float(np.sum(weight_fit * np.square(x_fit)))
    if denominator <= 0.0:
        raise ValueError("non-U wavelengths do not constrain the MICA lag relation")

    # 独立按每个波段的权重采样，不复原 MICA 跨波段协方差。
    rng = np.random.default_rng(int(opt["seed"]) + 314159)
    draws = np.column_stack(
        [
            rng.choice(posteriors[band][0], size=nsamp, replace=True, p=posteriors[band][1])
            for band in fit_bands
        ]
    )
    tau_u = rng.choice(
        posteriors[u_band][0],
        size=nsamp,
        replace=True,
        p=posteriors[u_band][1],
    )
    tau0 = draws @ (weight_fit * x_fit) / denominator
    predicted = tau0 * x_u
    residual = tau_u - predicted
    posterior = pd.DataFrame(
        {
            "sample": np.arange(nsamp, dtype=int),
            "stage": "short_mica",
            "beta": beta,
            "tau0_days": tau0,
            "tau_u_days": tau_u,
            "tau_u_predicted_days": predicted,
            "u_lag_residual_days": residual,
        }
    )
    t016, t050, t084 = quantiles(tau0)
    u16, u50, u84 = quantiles(tau_u)
    p16, p50, p84 = quantiles(predicted)
    r16, r50, r84 = quantiles(residual)
    summary = pd.DataFrame(
        [
            {
                "stage": "short_mica",
                "u_band": u_band,
                "beta": beta,
                "fit_bands": ",".join(fit_bands),
                "fit_weighting": "inverse_variance_from_marginal_mica_lag_68pct",
                "posterior_construction": "independent_resampling_of_mica_band_marginals",
                "n_samples": nsamp,
                "tau0": t050,
                "tau0_err_low": t050 - t016,
                "tau0_err_high": t084 - t050,
                "tau_u": u50,
                "tau_u_err_low": u50 - u16,
                "tau_u_err_high": u84 - u50,
                "tau_u_predicted": p50,
                "tau_u_predicted_err_low": p50 - p16,
                "tau_u_predicted_err_high": p84 - p50,
                "u_lag_residual": r50,
                "u_lag_residual_err_low": r50 - r16,
                "u_lag_residual_err_high": r84 - r50,
                "probability_residual_gt_zero": float(np.mean(residual > 0.0)),
            }
        ]
    )
    return posterior, summary


# 图中误差条是延迟分布的边际 16/50/84 分位数；画出的 tau0 曲线仅为固定 beta 诊断。
def plot_results(
    out: Path,
    source: str,
    summary: pd.DataFrame,
    mica: pd.DataFrame,
    results: dict[tuple[str, str], dict],
    residual_summary: pd.DataFrame,
    driver: str,
    u_band: str,
    waves: dict[str, float],
) -> None:
    colors = {"total": "0.35", "short": "#2b6f9f"}
    labels = {"total": "Total ICCF", "short": "Short ICCF"}
    markers = {"total": "o", "short": "s"}

    with plt.rc_context(
        {
            "font.size": 8.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), constrained_layout=True)
        ax_lag, ax_ccf = axes

        wave_grid = np.linspace(
            min(waves[driver], *(waves[band] for band in summary["band"].unique())),
            max(waves[driver], *(waves[band] for band in summary["band"].unique())),
            300,
        )
        for stage, offset in (("total", 0.992), ("short", 1.008)):
            stage_rows = summary[summary["stage"].eq(stage)].sort_values("wavelength")
            x = stage_rows["wavelength"].to_numpy(float) * offset / 1000.0
            y = stage_rows["tau_cent"].to_numpy(float)
            yerr = np.vstack(
                [
                    stage_rows["tau_cent_err_low"].to_numpy(float),
                    stage_rows["tau_cent_err_high"].to_numpy(float),
                ]
            )
            ax_lag.errorbar(
                x,
                y,
                yerr=yerr,
                fmt=markers[stage],
                ms=4.0,
                mew=0.8,
                mfc="white" if stage == "total" else colors[stage],
                mec=colors[stage],
                ecolor=colors[stage],
                elinewidth=0.8,
                capsize=1.8,
                linestyle="none",
                label=labels[stage],
                zorder=3,
            )
            residual_row = residual_summary[residual_summary["stage"].eq(stage)].iloc[0]
            tau0 = float(residual_row["tau0"])
            beta = float(residual_row["beta"])
            relation = tau0 * ((wave_grid / waves[driver]) ** beta - 1.0)
            ax_lag.plot(wave_grid / 1000.0, relation, color=colors[stage], lw=0.9, alpha=0.8)

        mica_plot = mica.copy()
        mica_plot["wavelength"] = mica_plot["band"].map(waves)
        mica_plot = mica_plot.dropna(subset=["wavelength"]).sort_values("wavelength")
        ax_lag.errorbar(
            mica_plot["wavelength"].to_numpy(float) / 1000.0,
            mica_plot["mica_tau"].to_numpy(float),
            yerr=np.vstack(
                [
                    mica_plot["mica_tau_err_low"].to_numpy(float),
                    mica_plot["mica_tau_err_high"].to_numpy(float),
                ]
            ),
            fmt="D",
            ms=3.2,
            mew=0.7,
            mfc="white",
            mec="#8b4f79",
            ecolor="#8b4f79",
            elinewidth=0.7,
            capsize=1.5,
            linestyle="none",
            label="Short MICA",
            zorder=2,
        )
        ax_lag.plot(waves[driver] / 1000.0, 0.0, marker="*", ms=6, color="black", zorder=4)
        ax_lag.axhline(0.0, color="0.8", lw=0.6, zorder=0)
        ax_lag.set_xlabel(r"Wavelength ($10^3\,\AA$)")
        ax_lag.set_ylabel("Lag relative to " + driver + " (days)")
        ax_lag.legend(loc="best", handlelength=1.5)
        ax_lag.text(
            0.97,
            0.96,
            "(a)",
            transform=ax_lag.transAxes,
            ha="right",
            va="top",
        )

        for stage in ("total", "short"):
            result = results[(stage, u_band)]
            ax_ccf.plot(
                result["tau"],
                result["curve"],
                color=colors[stage],
                lw=1.1,
                label=labels[stage],
            )
            _, median, _ = quantiles(result["cent_mc"], result["cent"])
            ax_ccf.axvline(median, color=colors[stage], lw=0.8, ls="--")
        ax_ccf.axhline(0.0, color="0.8", lw=0.6, zorder=0)
        ax_ccf.set_xlabel("Lag (days)")
        ax_ccf.set_ylabel(r"ICCF $r$")
        ax_ccf.legend(loc="best", handlelength=1.8)
        ax_ccf.text(0.03, 0.96, f"(b) {u_band}", transform=ax_ccf.transAxes, va="top")

        figure_base = out / f"{source}_iccf_diagnostic"
        fig.savefig(figure_base.with_suffix(".png"), dpi=300)
        fig.savefig(figure_base.with_suffix(".pdf"))
        plt.close(fig)


# 主流程先完成所有 total/short 波段对，再读取现成 MICA 汇总构造对照和 U 残差。
# nsim 默认 10000，因此这也是实际计算入口；阅读流程不需要启动它。
def run(
    value: str | Path,
    *,
    stage: str = "mica_round1",
    branch_override: str | None = None,
    nsim_override: int | None = None,
    mode_override: str | None = None,
    output_override: str | Path | None = None,
    resume: bool = False,
) -> None:
    src = source_dir(value)
    cfg, fit, opt = read_config(src, stage)
    if branch_override is not None:
        opt["branch"] = branch_override
    if nsim_override is not None:
        opt["nsim"] = nsim_override
        opt["u_residual_nsim"] = nsim_override
    if mode_override is not None:
        opt["mode"] = mode_override
    if output_override is not None:
        opt["output_dir"] = output_override
    validate_options(opt)

    source = str(cfg.get("source", src.name))
    driver_band = fit.get("driver")
    if not driver_band:
        raise ValueError("MICA driver band is not configured")
    driver_band = str(driver_band)
    responses = [str(value) for value in fit.get("responses", []) if str(value) != driver_band]
    if not responses:
        raise ValueError("no MICA response bands are configured")
    if len(set(responses)) != len(responses):
        raise ValueError("duplicate MICA response bands are configured")
    waves = wavelengths_from_config(cfg)
    missing_waves = [band for band in [driver_band, *responses] if band not in waves]
    if missing_waves:
        raise KeyError("missing wavelengths for: " + ", ".join(missing_waves))

    branch = str(opt["branch"])
    component_index = int(opt["component_index"])
    if component_index < 0:
        raise ValueError("iccf.component_index must be non-negative")
    out = rel(src, opt["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    driver_lc = load_observed(src, cfg, source, driver_band)
    results: dict[tuple[str, str], dict] = {}
    rows: list[dict] = []
    reused_pairs = 0
    for band in responses:
        observed = load_observed(src, cfg, source, band)
        short_grid = load_short_grid(
            src, cfg, fit, band, branch=branch, component_index=component_index
        )
        matched_total, short = match_short_to_observed(short_grid, observed)
        # 这里的局部变量 stage 会覆盖 run 参数中的 MICA 小节名称；
        # 下方 settings["mica_stage"] 因而会得到最后的 "short"，排查来源时需对照路径/配置。
        for stage, response in (("total", matched_total), ("short", short)):
            result = (
                load_complete_pair(
                    out, source, band, stage, driver_lc, response, opt
                )
                if resume
                else None
            )
            if result is None:
                print(f"ICCF {source}: {driver_band} -> {band} [{stage}]", flush=True)
                result = run_pair(driver_lc, response, opt)
            else:
                reused_pairs += 1
                print(
                    f"ICCF {source}: reused complete {band} [{stage}] MC product",
                    flush=True,
                )
            results[(stage, band)] = result
            save_pair(out, source, band, stage, result)
            rows.append(summary_row(band, waves[band], stage, result))

    summary = pd.DataFrame(rows).sort_values(["wavelength", "stage"])
    summary.insert(0, "source", source)
    summary.to_csv(out / "iccf_summary.csv", index=False)

    mica = mica_summary(src, cfg, fit, branch, component_index)
    compare = comparison_table(summary, mica)
    compare.insert(0, "source", source)
    compare.to_csv(out / "iccf_vs_mica.csv", index=False)

    residual_posterior, residual_summary = u_lag_residuals(
        results, responses, waves, driver_band, opt
    )
    mica_posteriors = mica_lag_posteriors(
        src, cfg, fit, branch, component_index, responses
    )
    mica_residual_posterior, mica_residual_summary = mica_u_lag_residuals(
        mica_posteriors, responses, waves, driver_band, opt
    )
    residual_posterior = pd.concat(
        [residual_posterior, mica_residual_posterior], ignore_index=True
    )
    residual_summary = pd.concat(
        [residual_summary, mica_residual_summary], ignore_index=True
    )
    residual_posterior.insert(0, "source", source)
    residual_summary.insert(0, "source", source)
    residual_posterior.to_csv(out / "u_lag_residual_posterior.csv", index=False)
    residual_summary.to_csv(out / "u_lag_residual_summary.csv", index=False)

    plot_results(
        out,
        source,
        summary,
        mica,
        results,
        residual_summary,
        driver_band,
        str(opt["u_band"]),
        waves,
    )

    # 设置文件还记录有效样本、重建误差相关性和独立边际近似，读结果时应与摘要一起核对。
    settings = {
        "source": source,
        "mica_stage": stage,
        "driver": driver_band,
        "responses": responses,
        "branch": branch,
        "component_index": component_index,
        "host_sub_dir": str(host_sub_dir(src, cfg)),
        "tau_range": [float(value) for value in opt["tau_range"]],
        "ntau": int(opt["ntau"]),
        "threshold": float(opt["threshold"]),
        "nsim_requested_per_pair": int(opt["nsim"]),
        "mode": str(opt["mode"]),
        "max_retries": int(opt["max_retries"]),
        "one_pyat_mc_call_per_pair": True,
        "resume_enabled": bool(resume),
        "n_pairs_reused": int(reused_pairs),
        "pyat_internal_seed": 100,
        "response_epoch_matching": "same_observed_epochs_for_total_and_short",
        "short_uncertainty_note": (
            "MICA reconstruction errors are temporally correlated; short ICCF is an "
            "internal diagnostic, not an independent validation"
        ),
        "u_band": str(opt["u_band"]),
        "u_residual_beta": float(opt["u_residual_beta"]),
        "u_residual_nsim": int(opt["u_residual_nsim"]),
        "u_residual_seed": int(opt["seed"]),
        "u_residual_fit_weighting": (
            "inverse_variance_from_marginal_iccf_centroid_68pct"
        ),
        "u_residual_posterior_construction": (
            "independent_resampling_of_band_marginals"
        ),
    }
    with (out / "iccf_settings.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(settings, handle, sort_keys=False)

    for name in (
        "iccf_summary.csv",
        "iccf_vs_mica.csv",
        "u_lag_residual_summary.csv",
        f"{source}_iccf_diagnostic.pdf",
    ):
        print(out / name)


# 未指定 --stage 时遍历 configured_stages；指定同一个输出目录给多个 stage 会被拒绝。
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--stage", help="one MICA config section; default: all mica_stages")
    parser.add_argument("--branch", help="official MICA posterior branch (default: config or comp0)")
    parser.add_argument("--nsim", type=int, help="override ICCF and U-residual simulation counts")
    parser.add_argument("--mode", choices=("single", "multiple"), help="override ICCF centroid mode")
    parser.add_argument("--output-dir", type=Path, help="override the source-relative output directory")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse only MC products proven complete for the requested nsim",
    )
    args = parser.parse_args()
    src = source_dir(args.source_dir)
    with (src / "config" / "source_config.yaml").open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    stages = [args.stage] if args.stage else configured_stages(cfg)
    if args.output_dir is not None and len(stages) > 1:
        parser.error("--output-dir with multiple stages is ambiguous; select --stage")
    for stage in stages:
        print(f"\n== MICA stage: {stage}; native comp0 ICCF ==", flush=True)
        run(
            src,
            stage=stage,
            branch_override=args.branch,
            nsim_override=args.nsim,
            mode_override=args.mode,
            output_override=args.output_dir,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
