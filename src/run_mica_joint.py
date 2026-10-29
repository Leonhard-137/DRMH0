#!/usr/bin/env python3
"""Fit all configured response bands jointly, without downstream processing.

Standalone: python src/run_mica_joint.py SOURCE --stage mica_gamma2_broad
Integrated: python src/run_mica.py SOURCE --stage mica_gamma2_broad --joint
The existing launcher supplies MPI, checkpoints, and fit-only completion checks.
"""
from __future__ import annotations

from pathlib import Path
import math
import sys

import numpy as np


PRIOR_PATH = Path("param/joint_parameter_priors.txt")


# 联合原型要求至少两个不同响应波段，且不含驱动；文件名要能安全映射到 input 目录。
# 返回顺序也是原生响应参数块的顺序，恢复时不可任意调换。
def input_files(src: Path, source: str, fit: dict) -> dict[str, Path]:
    bands = list(fit["responses"])
    if len(bands) < 2 or len(set(bands)) != len(bands) or fit["driver"] in bands:
        raise ValueError("joint fitting needs at least two distinct response bands, excluding the driver")
    for band in [fit["driver"], *bands]:
        if not isinstance(band, str) or Path(band).name != band or "/" in band or "\\" in band:
            raise ValueError("band names must be simple file-name components")
    folder = Path(fit["prepared_dir"])
    if not folder.is_absolute():
        folder = src / folder
    return {"cont.txt": folder / f"{source}_{fit['driver']}.txt", **{
        f"line_{band}.txt": folder / f"{source}_{band}.txt" for band in bands
    }}


# 检查 Nx3 的 time/flux/error、正误差及严格递增时间。
# 特别按原生写文件的 .6e 精度再检查时间，防止两个不同浮点时刻序列化后重合。
def _native_times(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve)
    if (curve.ndim != 2 or curve.shape[1] != 3 or len(curve) < 2
            or not np.isfinite(curve).all() or np.any(curve[:, 2] <= 0)):
        raise ValueError("MICA inputs must have at least two finite time/flux/positive-error rows")
    # Match pymica's existing data serialization, keeping the single-band inputs unchanged.
    times = np.array([float(f"{t:.6e}") for t in curve[:, 0]])
    if np.any(np.diff(times) <= 0):
        raise ValueError("MICA input times must remain strictly increasing at native precision")
    return times


# 当前联合入口只支持两 Gamma 分量、TypeLagPrior=4；不是任意核族的通用多波段入口。
# 构造一个 set1=[driver,response1,...] 交给原生 MICA，让原生程序处理共享驱动协方差。
# 每个响应占 1+3×2=7 个参数：系统误差，再接两个 (ln A, centroid, ln width)。
# 逐波段质心/宽度约束写成原生先验文本，宽度边界在此转为 ln(width)。
def setup(fit: dict, data: dict) -> tuple[dict, str]:
    """Keep each band's pairwise effective Gamma priors in the joint parameter table."""
    import run_mica

    if fit["_native_type_tf"] != "gamma" or fit["ncomp"] != 2 or fit.get("type_lag_prior") != 4:
        raise ValueError("the joint test supports two Gamma components with explicit type-4 centroid priors")
    input_files(Path("."), "source", fit)  # validate band identity/order
    bands = list(fit["responses"])
    if len(data) != 1 or len(next(iter(data.values()))) != len(bands) + 1:
        raise ValueError("joint data must contain one driver followed by all configured responses in one dataset")
    curves = next(iter(data.values()))
    times = [_native_times(curve) for curve in curves]
    kwargs = run_mica.setup_kwargs(fit, data, bands[0])
    rows = []
    for j, band in enumerate(bands):
        lags = run_mica.component_prior_for_band(fit, band, "lag_prior")
        if lags is None:
            raise ValueError("joint type-4 Gamma fitting requires explicit centroid bounds for every band")
        # pymica serializes explicit prior endpoints with %f for pairwise fits.
        lags = [[float(f"{value:f}") for value in bounds] for bounds in lags]
        widths = run_mica.component_prior_for_band(fit, band, "width_prior", positive=True)
        if widths is not None:
            widths = [[float(f"{value:f}") for value in bounds] for bounds in widths]
        else:
            pair = [times[0], times[j + 1]]
            span = max(float(np.ptp(t)) for t in pair)
            # 这里的 cadence 是区间长度/(点数-1) 的平均间隔，并非最小相邻采样间隔。
            cadence = min(float(np.ptp(t)) / (len(t) - 1) for t in pair)
            lo, hi = cadence / 10.0, span / 3.0
            if fit.get("width_limit") is not None:
                limits = fit["width_limit"]
                if limits[0] is not None:
                    lo = float(limits[0])
                if limits[1] is not None:
                    hi = min(float(limits[1]), hi)
            if not 0 < lo < hi:
                raise ValueError(f"invalid effective width bounds for {band}: {lo}, {hi}")
            widths = [[lo, hi], [lo, hi]]
        if any(lo >= hi for lo, hi in lags) or any(not 0 < lo < hi for lo, hi in widths):
            raise ValueError(f"invalid prior bounds at native precision for {band}")
        # 共享的连续光变三参数（含系统误差）之后按响应顺序排 7 列；分量索引依赖这个约定。
        start = 3 + j * 7
        for k in range(2):
            lo, hi = lags[k]
            rows.append(f"{start + 2 + 3*k} {band}_{k}_centroid UNI {lo:.17g} {hi:.17g} 0 0\n")
            lo, hi = map(math.log, widths[k])
            rows.append(f"{start + 3 + 3*k} {band}_{k}_log_width UNI {lo:.17g} {hi:.17g} 0 0\n")
    return kwargs, "".join(rows)


# 先完成文件写入并广播错误，再让所有 rank 设置同一原生先验，避免 rank 间状态不一致。
def register_priors(model, text: str, comm, rank: int) -> None:
    error = None
    if rank == 0:
        try:
            PRIOR_PATH.parent.mkdir(parents=True, exist_ok=True)
            PRIOR_PATH.write_text(text, encoding="utf-8")
        except Exception as exc:
            error = str(exc)
    error = comm.bcast(error, root=0)
    if error:
        raise RuntimeError(error)
    comm.Barrier()
    model.set_priors(str(PRIOR_PATH))


# 薄包装：添加 --joint 后交给 run_mica 的锁、恢复与 MPI 调度状态机。
def main() -> None:
    import run_mica

    if "--joint" not in sys.argv:
        sys.argv.append("--joint")
    run_mica.main()


if __name__ == "__main__":
    main()
