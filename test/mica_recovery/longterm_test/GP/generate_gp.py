"""生成平滑 GP 的观测数据；直接运行 python generate_gp.py，输出本目录 st.txt。

C(h) = sigma**2 * (1 + |h|/tau) * exp(-|h|/tau)。
在所选观测时刻联合抽样，再按各文件去均值、加上指定均值和逐点测量噪声。
"""

import os
from pathlib import Path

# 固定数值计算线程，使同一环境下的固定种子可重复。
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.linalg import eigh


# 只需修改这五项。SIGMA 是原始 GP 的总体标准差，MEAN 是输出光变的基线。
TAU_DAYS = 55.3735295468
SIGMA = 1.0
MEAN = 1.0
SEED = 2026091901
TIME_FILES = ("Mrk817_UVW2.txt", "Mrk817_U.txt")

HERE = Path(__file__).resolve().parent


def sample_gp(times):
    """在全部时刻抽取同一个 S(t)；重复时刻返回同一值。"""
    if not np.isfinite(TAU_DAYS) or TAU_DAYS <= 0:
        raise ValueError("TAU_DAYS 必须为有限正数")
    if not np.isfinite(SIGMA) or SIGMA < 0:
        raise ValueError("SIGMA 必须为有限非负数")
    if not np.isfinite(MEAN):
        raise ValueError("MEAN 必须为有限数")
    unique_time, index = np.unique(times, return_inverse=True)
    lag = np.abs(unique_time[:, None] - unique_time[None, :]) / TAU_DAYS
    covariance = SIGMA**2 * (1 + lag) * np.exp(-lag)
    values, vectors = eigh(covariance, check_finite=False)
    tolerance = 256 * np.finfo(float).eps * len(values) * max(
        float(values[-1]), np.finfo(float).tiny
    )
    if values[0] < -tolerance:
        raise ValueError("协方差矩阵出现超出舍入误差的负特征值")
    # 沿用旧流程给 GP 分配的子随机流；测量噪声使用另外的随机流。
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(0, 1)))
    sample = vectors @ (np.sqrt(np.maximum(values, 0)) * rng.standard_normal(len(values)))
    return sample[index]


def main():
    observed = [np.loadtxt(HERE / name, ndmin=2) for name in TIME_FILES]
    for data in observed:
        if (data.shape[1] != 3 or not np.isfinite(data).all()
                or data[:, 1].mean() <= 0 or np.any(data[:, 2] <= 0)
                or np.any(np.diff(data[:, 0]) <= 0)):
            raise ValueError("数据须为时间递增的 time/flux/error 三列，平均通量和误差为正")
    t0 = min(data[0, 0] for data in observed)
    time = np.concatenate([data[:, 0] - t0 for data in observed])
    shared = sample_gp(time)
    rows, start = [], 0
    for i, data in enumerate(observed):
        stop = start + len(data)
        signal = shared[start:stop]
        signal = signal - signal.mean()
        error = data[:, 2] / data[:, 1].mean()
        rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(0, 3 + i)))
        measured = MEAN + signal + rng.normal(0, error)
        rows.append(np.column_stack((time[start:stop], measured, error)))
        start = stop
    result = np.vstack(rows)
    result = result[np.argsort(result[:, 0], kind="stable")]
    output = HERE / "st.txt"
    np.savetxt(output, result, fmt="%.12g", header=(
        "time_days flux_observed error\n"
        f"tau_days={TAU_DAYS}; sigma={SIGMA}; mean={MEAN}; seed={SEED}; time_zero_mjd={t0:.9f}\n"
        f"time_files={','.join(TIME_FILES)}; counts={','.join(str(len(d)) for d in observed)}\n"
        "flux_observed=mean+sampled_GP-mean_per_file+noise; error=original_error/mean_flux_per_file"
    ))
    print(f"已生成 {output}，共 {len(result)} 个点")


if __name__ == "__main__":
    main()
