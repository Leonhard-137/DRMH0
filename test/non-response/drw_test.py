#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drw_test.py -- 检验光变曲线 (t, flux, e_flux) 是否与 damped random walk (DRW) 一致
================================================================================

DRW = Ornstein-Uhlenbeck 过程。协方差（celerite 约定）:

    k(dt) = sigma^2 * exp(-|dt| / tau)
    C_ij  = k(|t_i - t_j|) + (e_i^2 + s_jit^2) * delta_ij

    sigma  : 过程长期标准差 (stationary std)
    tau    : 特征时标 (damping timescale)
    s_jit  : 额外白噪声 (jitter, 可选)
    SF_inf = sqrt(2) * sigma     (structure function 的平台值)

------------------------------------------------------------
程序做的事情（4 个层次，由弱到强）
------------------------------------------------------------
[1] 极大似然拟合 + 模型比较
      white noise  : 只有白噪声 (jitter)，无相关变光
      DRW          : 纯 DRW，只用给出的测量误差
      DRW + jitter : DRW + 额外白噪声（推荐默认）
      2OU + jitter : 两个 OU 叠加（更陡 / 双时标 PSD 的近似）
    比较 AIC / BIC / AIC 权重。
    * 用 WN vs DRW 判断"到底有没有相关变光"
    * 用 DRW vs DRW+jitter 判断"是否还有多余白噪声（误差被低估？）"
    * 用 DRW vs 2OU 判断"PSD 是否比 DRW 更陡/更复杂"

[2] 拟合优度检验（本程序的核心判据）—— 参数自助法 (parametric bootstrap)
    零假设 H0: 数据来自以最佳拟合参数构成的 DRW(+jitter) 过程。
    用 OU 过程的 Markov 递推精确生成大量"同采样、同误差"的模拟曲线，
    每条都重新拟合，得到统计量 T = r^T C^-1 r 的零分布，
    再和真实数据的 T_obs 比较得到 p 值。
    （因为每条模拟都重新拟合，所以 p 值自动计入了"参数是估出来的"这一事实。）

[3] 结构函数检验（与模型无关的交叉检查）
    去噪后的 SF(dt) 应该符合 sqrt(2)*sigma*sqrt(1-exp(-dt/tau))；
    短时标 (dt << tau) 的 log SF-log dt 斜率趋于 0.5（SF^2 的斜率为 1）；
    长时标出现平台。
    单独拟合 SF 得到的 (tau, sigma) 应与 GP 拟合结果一致。

[4] 残差白化检验
    白化残差 z = L^-1 (y - mean) 在 H0 下应为 iid N(0,1)：
    检查 std(z)、lag-1 自相关、Ljung-Box 检验、chi2/dof。

------------------------------------------------------------
用法
------------------------------------------------------------
方式 A（你现在的场景，直接传数组）:

    import numpy as np
    from drw_test import test_drw
    res = test_drw(t, flux, e_flux)          # 打印完整报告
    print(res["verdict"]); print(res["p_gof"])

方式 B（命令行 / CSV 文件）:

    python drw_test.py --input lc.csv --n-mc 200 --plot drw_check.png
    python drw_test.py --demo --demo-kind drw        # 自检
    python drw_test.py --demo --demo-kind whitenoise # 负对照

依赖: numpy, scipy, celerite2（强烈建议，N 大时快几个量级）。
      没有 celerite2 也能跑，会退化为 numpy Cholesky，N 大时很慢。

------------------------------------------------------------
重要提醒（写论文时务必声明）
------------------------------------------------------------
* 这只能"拒绝/不能拒绝" DRW，**永远无法证明**数据就是 DRW。p 值不显著
  只说明"在当前的采样、基线、信噪比下，DRW 够用"。
* DRW 与更陡的单幂律 PSD 常常无法区分（AGN 文献里这是著名问题），
  所以本程序特意加入了 2OU 备择模型。
* 时标 tau 只有在基线 >> tau 时才真正被"测到"，否则只是下限。
  程序会自动检查 baseline/tau 并给出警告。
* 若输入是星等，参数单位就是 mag / mag^2；DRW 在星等空间只是近似。
* 均值用的是加权平均并预先减掉（不是 GP 意义下的 ML 均值），
  这对 sigma、tau 的影响可忽略，但与文献比较时注意这一约定。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import minimize

# ---------------------------------------------------------------- celerite2
try:
    import celerite2
    from celerite2 import terms as cterms

    _HAVE_CELERITE = True
except Exception:  # pragma: no cover
    celerite2 = None
    cterms = None
    _HAVE_CELERITE = False

# ---------------------------------------------------------------- tqdm（可选）
try:
    from tqdm.auto import tqdm as _tqdm_cls

    _HAVE_TQDM = True
except Exception:  # pragma: no cover
    _tqdm_cls = None
    _HAVE_TQDM = False


__all__ = [
    "test_drw",
    "prepare_data",
    "OULikelihood",
    "WhiteNoiseLikelihood",
    "structure_function",
    "sf_model",
    "sample_ou",
    "make_demo_light_curve",
]


# ============================================================================
# 0. 小工具
# ============================================================================
def weighted_mean(y: np.ndarray, e: np.ndarray) -> float:
    w = 1.0 / np.maximum(e, 1e-300) ** 2
    return float(np.sum(w * y) / np.sum(w))


def _cho_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """解 C x = b, 其中 C = L L^T。"""
    return np.linalg.solve(L.T, np.linalg.solve(L, b))


def _fmt_time(sec: float) -> str:
    """秒 -> mm:ss / h:mm:ss。"""
    if sec is None or not np.isfinite(sec) or sec < 0:
        return "--:--"
    sec = int(round(float(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class _ProgressBar:
    """轻量进度条。

    * 装了 tqdm 就用 tqdm（自动适配终端 / notebook）；
    * 否则自己画，纯 ASCII，避免 Windows 控制台中文乱码残留；
    * 输出被重定向（非 tty）时自动退化为每 5% 打一行日志，不刷 \\r。
    """

    def __init__(self, total: int, desc: str = "progress", unit: str = "it",
                 enabled: bool = True, width: int = 30, min_dt: float = 0.1):
        self.total = max(int(total), 1)
        self.desc = desc
        self.unit = unit
        self.width = int(width)
        self.min_dt = float(min_dt)
        self.n = 0
        self.t0 = time.time()
        self._last = 0.0
        self._lastlen = 0
        self._post = ""
        self._tqdm = None
        self.disabled = not enabled
        if self.disabled:
            return
        try:
            self.tty = bool(sys.stdout.isatty())
        except Exception:
            self.tty = False
        if _HAVE_TQDM:
            try:
                self._tqdm = _tqdm_cls(total=self.total, desc=desc, unit=unit,
                                       file=sys.stdout, dynamic_ncols=True,
                                       leave=True, mininterval=max(self.min_dt, 0.05))
                return
            except Exception:
                self._tqdm = None
        if self.tty:
            self._render()
        else:
            print(f"  {self.desc} 开始（共 {self.total} {self.unit}）"
                  "，每 5% 报告一次", flush=True)

    # ------------------------------------------------------------------ API
    def update(self, n: int = 1, extra: str = "") -> None:
        if self.disabled:
            return
        self.n = min(self.n + int(n), self.total)
        if extra:
            self._post = extra
        if self._tqdm is not None:
            self._tqdm.update(int(n))
            if extra:
                self._tqdm.set_postfix_str(extra, refresh=False)
            return
        now = time.time()
        if not self.tty:
            step = max(self.total // 20, 1)
            if self.n % step == 0 or self.n >= self.total:
                el = now - self.t0
                print(f"  {self.desc} {self.n}/{self.total}"
                      f" ({100.0 * self.n / self.total:5.1f}%)"
                      f"  elapsed {_fmt_time(el)}"
                      + (f"  {self._post}" if self._post else ""), flush=True)
            return
        if now - self._last >= self.min_dt or self.n >= self.total:
            self._render()

    def set_extra(self, extra: str) -> None:
        """只更新右侧附加信息，不推进计数（用于显示当前 tau/sigma 等）。"""
        if self.disabled:
            return
        self._post = extra
        if self._tqdm is not None:
            try:
                self._tqdm.set_postfix_str(extra, refresh=False)
            except Exception:
                pass
        elif self.tty and time.time() - self._last >= self.min_dt:
            self._render()

    def close(self) -> None:
        if self.disabled:
            return
        if self._tqdm is not None:
            try:
                self._tqdm.close()
            except Exception:
                pass
            return
        if self.tty:
            self._render()
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._lastlen = 0

    # --------------------------------------------------------------- 渲染
    def _render(self) -> None:
        frac = self.n / self.total
        filled = int(round(self.width * frac))
        bar = "#" * filled + "-" * (self.width - filled)
        el = time.time() - self.t0
        eta = (el / self.n) * (self.total - self.n) if self.n > 0 else float("nan")
        msg = (f"  {self.desc} |{bar}| {100.0 * frac:5.1f}%  {self.n}/{self.total}"
               f"  elapsed {_fmt_time(el)}  ETA {_fmt_time(eta)}")
        if self._post:
            msg += f"  {self._post}"
        pad = max(self._lastlen - len(msg), 0)
        try:
            sys.stdout.write("\r" + msg + " " * pad)
            sys.stdout.flush()
        except Exception:
            pass
        self._lastlen = len(msg)
        self._last = time.time()


@dataclass
class LCData:
    """预处理后的光变曲线（已按 t 排序、已减均值）。"""

    t: np.ndarray
    y: np.ndarray          # flux - mean
    yerr: np.ndarray
    mean: float
    n: int
    baseline: float
    dt_min: float
    dt_median: float
    n_dropped: int = 0
    n_dup: int = 0
    snr: float = np.nan


def prepare_data(t, flux, e_flux=None, verbose: bool = True) -> LCData:
    """排序、去 NaN、修零误差、减加权均值。"""
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(flux, dtype=float).ravel()
    if e_flux is None:
        e = np.full_like(y, 1e-6 * (np.std(y) or 1.0))
        if verbose:
            print("[warn] 未提供误差 e_flux，已设为 1e-6*std(flux)；"
                  "请务必补上真实误差，否则 jitter 检验会失真。")
    else:
        e = np.asarray(e_flux, dtype=float).ravel()

    if not (t.size == y.size == e.size):
        raise ValueError(f"t/flux/e_flux 长度不一致: {t.size}, {y.size}, {e.size}")
    if t.size < 10:
        raise ValueError(f"数据点太少 (N={t.size})，无法做 DRW 检验")

    good = np.isfinite(t) & np.isfinite(y) & np.isfinite(e)
    n_bad = int((~good).sum())
    t, y, e = t[good], y[good], e[good]

    order = np.argsort(t, kind="mergesort")
    t, y, e = t[order], y[order], e[order]

    n_dup = int(t.size - np.unique(t).size)

    scale = float(np.std(y))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.median(np.abs(e))) or 1.0
    bad_e = ~np.isfinite(e) | (e <= 0)
    if np.any(bad_e):
        if verbose:
            print(f"[warn] {int(bad_e.sum())} 个非正/无效误差已被替换为 1e-6*std(flux)")
        e = np.where(bad_e, 1e-6 * scale, e)

    mu = weighted_mean(y, e)
    yy = y - mu

    n = t.size
    baseline = float(t[-1] - t[0])
    dts = np.diff(t)
    dt_min = float(np.min(dts)) if dts.size else baseline
    dt_med = float(np.median(dts)) if dts.size else baseline
    if dt_min <= 0:
        dt_min = max(baseline * 1e-6, 1e-12)

    snr = float(np.std(yy) / np.median(e)) if np.median(e) > 0 else np.nan

    dat = LCData(t=t, y=yy, yerr=e, mean=mu, n=n, baseline=baseline,
                 dt_min=dt_min, dt_median=dt_med, n_dropped=n_bad,
                 n_dup=n_dup, snr=snr)
    if verbose:
        if n_bad:
            print(f"[info] 剔除 {n_bad} 个含 NaN/Inf 的点")
        if n_dup:
            print(f"[info] 有 {n_dup} 个重复时刻（协方差仍可逆，不影响拟合）")
        print(f"[info] N={n}, baseline={baseline:.4g}, dt_min={dt_min:.4g}, "
              f"dt_median={dt_med:.4g}, median(err)={np.median(e):.4g}, "
              f"std(flux)={np.std(yy):.4g}")
        if n / max(baseline / max(dt_med, 1e-30), 1) < 5:
            print("[warn] 采样点数相对基线偏少，tau 的约束可能很弱")
    return dat


# ============================================================================
# 1. 似然函数：多成分 OU (+ jitter)
# ============================================================================
class OULikelihood:
    """-lnL for  C_ij = sum_k sigma_k^2 exp(-|dt|/tau_k) + (e_i^2 + jit^2) d_ij

    参数向量 theta = [log sigma_1, log tau_1, (log sigma_2, log tau_2), ..., (log jit)]
    （用 log 参数化 + 边界约束，保证正定性；L-BFGS-B 在 log 空间收敛很好）
    """

    def __init__(self, dat: LCData, n_terms: int = 1, fit_jitter: bool = False,
                 use_celerite: Optional[bool] = None, label: str = "model"):
        self.t = np.ascontiguousarray(dat.t, dtype=float)
        self.y = np.ascontiguousarray(dat.y, dtype=float)
        self.yerr = np.ascontiguousarray(dat.yerr, dtype=float)
        self.n = self.t.size
        self.n_terms = int(n_terms)
        self.fit_jitter = bool(fit_jitter)
        self.label = label

        self.n_params = 2 * self.n_terms + (1 if self.fit_jitter else 0)

        self.sd = float(np.std(self.y, ddof=1)) if self.n > 1 else 1.0
        if not np.isfinite(self.sd) or self.sd <= 0:
            self.sd = float(np.median(self.yerr)) or 1.0
        self.err_med = float(np.median(self.yerr)) or 1e-12
        self.baseline = dat.baseline
        self.dt_min = dat.dt_min

        if use_celerite is None:
            use_celerite = _HAVE_CELERITE
        self._use_celerite = bool(use_celerite and _HAVE_CELERITE)
        self._celerite_failed = False
        self.n_eval = 0

        if not self._use_celerite and self.n > 6000:
            print(f"[warn] 无 celerite2 且 N={self.n}，numpy Cholesky 会非常慢"
                  "（建议 pip install celerite2）")

    # ---------------------------------------------------------------- bounds
    def bounds(self):
        lo_tau = max(self.dt_min, self.baseline * 1e-5) / 100.0
        hi_tau = self.baseline * 100.0
        b = []
        for _ in range(self.n_terms):
            b.append((np.log(1e-8 * self.sd), np.log(1e3 * self.sd)))
            b.append((np.log(lo_tau), np.log(hi_tau)))
        if self.fit_jitter:
            j0 = max(self.err_med, 1e-8 * self.sd)
            b.append((np.log(1e-6 * j0), np.log(1e2 * self.sd)))
        return b

    # ------------------------------------------------------------- unpacking
    def unpack(self, theta) -> Dict[str, object]:
        theta = np.asarray(theta, dtype=float)
        ls = theta[0:2 * self.n_terms:2]
        lt = theta[1:2 * self.n_terms:2]
        d: Dict[str, object] = {
            "sigma": [float(v) for v in np.exp(ls)],
            "tau": [float(v) for v in np.exp(lt)],
        }
        d["jitter"] = float(np.exp(theta[-1])) if self.fit_jitter else 0.0
        return d

    def pack(self, sigmas, taus, jitter: float = 0.0) -> np.ndarray:
        th = []
        for s, ta in zip(np.atleast_1d(sigmas), np.atleast_1d(taus)):
            th += [np.log(max(float(s), 1e-300)), np.log(max(float(ta), 1e-300))]
        if self.fit_jitter:
            th.append(np.log(max(float(jitter), 1e-300)))
        return np.array(th, dtype=float)

    # -------------------------------------------------------------- negative
    def neg_log_like(self, theta) -> float:
        self.n_eval += 1
        theta = np.asarray(theta, dtype=float)
        if theta.size != self.n_params or not np.all(np.isfinite(theta)):
            return 1e12

        sigma = np.exp(theta[0:2 * self.n_terms:2])
        tau = np.exp(theta[1:2 * self.n_terms:2])
        jitter = float(np.exp(theta[-1])) if self.fit_jitter else 0.0

        # 先验式保护：tau 太极端直接给惩罚，避免数值垃圾
        if np.any(~np.isfinite(sigma)) or np.any(~np.isfinite(tau)):
            return 1e12

        var = self.yerr ** 2 + jitter ** 2

        if self._use_celerite and not self._celerite_failed:
            try:
                kernel = None
                for s, ta in zip(sigma, tau):
                    term = cterms.RealTerm(a=float(s * s), c=float(1.0 / ta))
                    kernel = term if kernel is None else kernel + term
                try:
                    gp = celerite2.GaussianProcess(kernel, mean=0.0, quiet=True)
                except TypeError:
                    gp = celerite2.GaussianProcess(kernel)
                gp.compute(self.t, yerr=np.sqrt(var))
                return float(-gp.log_likelihood(self.y))
            except Exception as exc:  # 退化为 numpy
                self._celerite_failed = True
                print(f"[warn] celerite2 调用失败 ({exc!r})，改用 numpy Cholesky（会慢）")

        return self._numpy_neg_log_like(sigma, tau, var)

    def _numpy_neg_log_like(self, sigma, tau, var) -> float:
        dt = np.abs(self.t[:, None] - self.t[None, :])
        C = np.zeros((self.n, self.n), dtype=float)
        for s, ta in zip(sigma, tau):
            C += (s * s) * np.exp(-dt / ta)
        di = np.diag_indices(self.n)
        C[di] += var
        C[di] += 1e-12 * max(float(np.max(np.diag(C))), 1e-300)
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            return 1e12
        z = np.linalg.solve(L, self.y)
        logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
        return 0.5 * (float(z @ z) + logdet + self.n * np.log(2.0 * np.pi))

    # ------------------------------------------------ 供模拟使用：真实参数
    def theta_to_model(self, theta):
        d = self.unpack(theta)
        return d["sigma"], d["tau"], float(d["jitter"])


class WhiteNoiseLikelihood:
    """只有白噪声:  y_i ~ N(0, e_i^2 + jit^2)。DRW 的 sigma -> 0 极限。"""

    def __init__(self, dat: LCData, fit_jitter: bool = True, label: str = "white noise"):
        self.t = dat.t
        self.y = dat.y
        self.yerr = dat.yerr
        self.n = dat.n
        self.fit_jitter = bool(fit_jitter)
        self.n_params = 1 if fit_jitter else 0
        self.n_terms = 0
        self.label = label
        self.sd = float(np.std(self.y, ddof=1)) or 1.0
        self.err_med = float(np.median(self.yerr)) or 1e-12
        self.baseline = dat.baseline
        self.dt_min = dat.dt_min

    def bounds(self):
        if not self.fit_jitter:
            return []
        return [(np.log(1e-6 * self.err_med), np.log(1e2 * self.sd))]

    def unpack(self, theta) -> Dict[str, object]:
        theta = np.asarray(theta, dtype=float)
        jit = float(np.exp(theta[0])) if self.fit_jitter else 0.0
        return {"sigma": [0.0], "tau": [np.nan], "jitter": jit}

    def neg_log_like(self, theta) -> float:
        jit2 = float(np.exp(theta[0])) ** 2 if self.fit_jitter else 0.0
        s2 = self.yerr ** 2 + jit2
        r = self.y
        return 0.5 * float(np.sum(np.log(2.0 * np.pi * s2) + r * r / s2))


# ============================================================================
# 2. 拟合驱动
# ============================================================================
@dataclass
class FitResult:
    label: str
    n_params: int          # 自由参数个数（不含均值）
    k_total: int           # AIC/BIC 里用的参数个数（含均值）
    ln_like: float         # log L（不是优化器最小化的 -log L）
    theta: np.ndarray
    params: Dict[str, object]
    aic: float = np.nan
    bic: float = np.nan
    success: bool = True

    @property
    def sigma(self):
        return float(self.params["sigma"][0])

    @property
    def tau(self):
        return float(self.params["tau"][0])

    @property
    def jitter(self):
        return float(self.params.get("jitter", 0.0))


def _clamp(theta0, bounds):
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    t0 = np.asarray(theta0, dtype=float)
    t0 = np.where(np.isfinite(t0), t0, 0.5 * (lo + hi))
    return np.clip(t0, lo + 1e-12, hi - 1e-12)


def fit_likelihood(lik, theta0_list, maxiter: int = 2000) -> Tuple[np.ndarray, float]:
    """返回最优参数和对应的负对数似然 NLL = -log L。"""
    bounds = lik.bounds()
    best_x, best_f = None, np.inf
    if lik.n_params == 0:  # 无自由参数
        f = float(lik.neg_log_like(np.array([])))
        return np.array([]), f
    for th0 in theta0_list:
        th0 = _clamp(th0, bounds)
        try:
            res = minimize(lik.neg_log_like, th0, method="L-BFGS-B", bounds=bounds,
                           options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-8))
        except Exception:
            continue
        f = float(res.fun)
        if np.isfinite(f) and f < best_f:
            best_f, best_x = f, np.asarray(res.x, dtype=float)
    if best_x is None:
        raise RuntimeError(f"拟合失败: {lik.label}")
    # 从最优点再做一次精修（log 空间里 L-BFGS 偶尔早停）
    for _ in range(2):
        try:
            res = minimize(lik.neg_log_like, best_x, method="L-BFGS-B", bounds=bounds,
                           options=dict(maxiter=maxiter, ftol=1e-14, gtol=1e-10))
            f = float(res.fun)
            if np.isfinite(f) and f < best_f - 1e-10:
                best_f, best_x = f, np.asarray(res.x, dtype=float)
        except Exception:
            break
    return best_x, best_f


def sf_initial_guess(dat: LCData, n_bins: int = 12,
                     pair_max: int = 2_000_000) -> Optional[Tuple[float, float]]:
    """用结构函数粗略估 (sigma, tau)，用于给优化器当初值。"""
    n_all = dat.n * (dat.n - 1) // 2
    pair_idx = None
    if n_all > pair_max:
        pair_idx = make_pairs(dat.n, pair_max, np.random.default_rng(0))
    tab = structure_function(dat.t, dat.y, dat.yerr, n_bins=n_bins, min_pairs=5,
                             pair_idx=pair_idx)
    ok = tab["ok"]
    if int(np.sum(ok)) < 2:
        return None
    dts = np.asarray(tab["dt"])[ok]
    sf = np.asarray(tab["sf"])[ok]
    if not np.any(sf > 0):
        return None
    sf_inf = float(np.max(sf))
    sigma = sf_inf / np.sqrt(2.0)
    thr = 0.795 * sf_inf          # SF(tau)/SF_inf = sqrt(1-1/e) = 0.795
    if np.any(sf >= thr):
        tau = float(dts[int(np.argmax(sf >= thr))])
    else:
        tau = float(dts[-1]) * 3.0
    return max(sigma, 1e-12), max(tau, dat.dt_min)


def initial_guesses(lik, dat: LCData, n_starts: int = 6) -> List[np.ndarray]:
    # --- 纯白噪声模型（只有一个 jitter 参数）---
    if getattr(lik, "n_terms", 0) == 0:
        if lik.n_params == 0:
            return [np.array([])]
        var_y = float(np.var(dat.y, ddof=1)) if dat.n > 1 else 0.0
        exc2 = var_y - float(np.mean(dat.yerr ** 2))
        j0 = float(np.sqrt(max(exc2, 1e-4 * max(var_y, 1e-300))))
        j1 = float(np.median(dat.yerr))
        return [np.array([np.log(max(j0, 1e-300))]),
                np.array([np.log(max(j1, 1e-300))])]

    sd = lik.sd
    base = lik.baseline
    dtmin = lik.dt_min
    rough = sf_initial_guess(dat)

    taus = [max(3 * dtmin, base / 50.0), max(3 * dtmin, base / 10.0),
            max(3 * dtmin, base / 3.0), base]
    if rough is not None:
        taus = [rough[1]] + taus
    taus = sorted({float(np.round(x, 12)) for x in taus if x > 0})

    sig_fracs = [0.5, 0.9]
    if rough is not None and sd > 0:
        sig_fracs = sorted({float(np.clip(rough[0] / sd, 0.05, 0.99)), 0.5, 0.9})

    jit0 = float(np.median(dat.yerr)) * 0.5

    starts: List[np.ndarray] = []
    if lik.n_terms == 1:
        for sf_ in sig_fracs:
            for ta in taus:
                starts.append(lik.pack(sf_ * sd, ta, jit0))
    else:
        # 双 OU：一个短时标 + 一个长时标
        pairs = [(max(3 * dtmin, base / 50.0), base),
                 (max(3 * dtmin, base / 10.0), max(3 * dtmin, base / 2.0)),
                 (max(3 * dtmin, base / 100.0), max(3 * dtmin, base / 5.0))]
        for t1, t2 in pairs:
            for f1, f2 in [(0.6, 0.5), (0.3, 0.9), (0.8, 0.2)]:
                starts.append(lik.pack([f1 * sd, f2 * sd], [t1, t2], jit0))

    # 去重 + 截断
    uniq = []
    seen = set()
    for th in starts:
        key = tuple(np.round(th, 6))
        if key not in seen:
            seen.add(key)
            uniq.append(th)
    if len(uniq) > n_starts:
        idx = np.linspace(0, len(uniq) - 1, n_starts).astype(int)
        uniq = [uniq[i] for i in idx]
    return uniq


def fit_model(lik, dat: LCData, n_starts: int = 6, maxiter: int = 2000) -> FitResult:
    theta, nll = fit_likelihood(lik, initial_guesses(lik, dat, n_starts=n_starts),
                                maxiter=maxiter)
    k_total = lik.n_params + 1  # +1: 均值（虽然预先减掉了，但口径对所有模型一致）
    return FitResult(label=lik.label, n_params=lik.n_params, k_total=k_total,
                     ln_like=-nll, theta=theta, params=lik.unpack(theta))


def fit_model_warmstart(lik, theta0_list, n_starts: int = 2, maxiter: int = 1500) -> FitResult:
    """MC 里用：从观测最佳拟合出发的少量起点，省时间。"""
    theta, nll = fit_likelihood(lik, theta0_list[:n_starts], maxiter=maxiter)
    k_total = lik.n_params + 1
    return FitResult(label=lik.label, n_params=lik.n_params, k_total=k_total,
                     ln_like=-nll, theta=theta, params=lik.unpack(theta))


def information_criteria(fits: Sequence[FitResult], n: int) -> List[FitResult]:
    for f in fits:
        f.aic = 2.0 * f.k_total - 2.0 * f.ln_like
        f.bic = f.k_total * np.log(n) - 2.0 * f.ln_like
    return list(fits)


def aic_weights(fits: Sequence[FitResult]) -> np.ndarray:
    a = np.array([f.aic for f in fits], dtype=float)
    d = a - np.min(a)
    w = np.exp(-0.5 * d)
    return w / np.sum(w)


# ============================================================================
# 3. GP 诊断：协方差、白化残差、chi2、Ljung-Box、条件预测
# ============================================================================
def model_covariance(t: np.ndarray, sigma, tau, diag: np.ndarray) -> np.ndarray:
    dt = np.abs(t[:, None] - t[None, :])
    C = np.zeros((t.size, t.size), dtype=float)
    for s, ta in zip(np.atleast_1d(sigma), np.atleast_1d(tau)):
        C += (s * s) * np.exp(-dt / ta)
    di = np.diag_indices(t.size)
    C[di] += diag
    C[di] += 1e-12 * max(float(np.max(np.diag(C))), 1e-300)
    return C


def gp_chi2(t, y, yerr, sigma, tau, jitter: float = 0.0) -> Dict[str, float]:
    """返回 chi2 = r^T C^-1 r、lnL、白化残差 z、lag-1 自相关、Ljung-Box。"""
    var = yerr ** 2 + jitter ** 2
    C = model_covariance(t, sigma, tau, var)
    L = np.linalg.cholesky(C)
    z = np.linalg.solve(L, y)
    chi2 = float(z @ z)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    lnl = -0.5 * (chi2 + logdet + t.size * np.log(2 * np.pi))

    zc = z - z.mean()
    denom = float(np.sum(zc * zc))
    r1 = float(np.sum(zc[1:] * zc[:-1]) / denom) if denom > 0 and z.size > 2 else np.nan
    Q, pQ = ljung_box(z, n_lags=10)
    return dict(chi2=chi2, lnlike=float(lnl), z=z, logdet=logdet,
                z_std=float(np.std(z, ddof=1)) if z.size > 1 else np.nan,
                acf1=r1, lb_Q=Q, lb_p=pQ)


def ljung_box(z: np.ndarray, n_lags: int = 10) -> Tuple[float, float]:
    n = z.size
    zc = z - z.mean()
    denom = float(np.sum(zc * zc))
    if denom <= 0 or n < 5:
        return np.nan, np.nan
    h = int(min(n_lags, n - 2))
    Q = 0.0
    for k in range(1, h + 1):
        rk = float(np.sum(zc[k:] * zc[:-k]) / denom)
        Q += rk * rk / (n - k)
    Q *= n * (n + 2)
    return float(Q), float(stats.chi2.sf(Q, h))


def conditional_prediction(t, y, yerr, sigma, tau, jitter, t_test):
    """GP 条件均值/方差，用于画最佳拟合模型的预测带。"""
    var = yerr ** 2 + jitter ** 2
    C = model_covariance(t, sigma, tau, var)
    L = np.linalg.cholesky(C)
    dt = np.abs(t[:, None] - t_test[None, :])
    Ks = np.zeros((t.size, t_test.size), dtype=float)
    kss = np.zeros(t_test.size, dtype=float)
    for s, ta in zip(np.atleast_1d(sigma), np.atleast_1d(tau)):
        Ks += (s * s) * np.exp(-dt / ta)
        kss += s * s
    alpha = _cho_solve(L, y)
    beta = _cho_solve(L, Ks)
    mu = Ks.T @ alpha
    v = kss - np.sum(Ks * beta, axis=0)
    return mu, np.clip(v, 0.0, None)


# ============================================================================
# 4. 结构函数
# ============================================================================
def _as_scalar(x, name: str = "param") -> float:
    """把 list/array/标量统一成 float 标量（多成分时只取第一个）。"""
    a = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    if a.size == 0 or not np.isfinite(a[0]):
        raise ValueError(f"{name} 不是有限标量: {x!r}")
    return float(a[0])


def sf_model(dt, sigma, tau):
    """DRW 的 structure function: sqrt(2)*sigma*sqrt(1-exp(-dt/tau))。

    sigma/tau 允许传 list（只取第一个成分），方便直接喂 params["sigma"]。
    """
    dt = np.asarray(dt, dtype=float)
    sigma = _as_scalar(sigma, "sigma")
    tau = _as_scalar(tau, "tau")
    return np.sqrt(2.0) * sigma * np.sqrt(-np.expm1(-dt / tau))


def make_pairs(n: int, pair_max: int, rng: np.random.Generator):
    """全配对（N 小时）或固定的随机子样本（N 大时），保证观测/模拟用同一套 pair。"""
    n_all = n * (n - 1) // 2
    if n_all <= pair_max:
        return np.triu_indices(n, 1)
    i = rng.integers(0, n, size=int(pair_max))
    j = rng.integers(0, n, size=int(pair_max))
    m = i != j
    i, j = i[m], j[m]
    a = np.minimum(i, j)
    b = np.maximum(i, j)
    key = a.astype(np.int64) * np.int64(n) + b.astype(np.int64)
    _, u = np.unique(key, return_index=True)
    return a[u], b[u]


def structure_function(t, y, yerr, n_bins: int = 12, min_pairs: int = 8,
                       pair_idx=None) -> Dict[str, np.ndarray]:
    """去噪 SF：SF^2 = <(df)^2 - (e_i^2 + e_j^2)>，用等对数间隔的分位 bin。

    注意：pair 之间不独立，误差棒是近似的（但观测/模拟用同一算法，比较是一致的）。
    """
    if pair_idx is None:
        pair_idx = np.triu_indices(t.size, 1)
    i, j = pair_idx
    dt = t[j] - t[i]
    df2 = (y[j] - y[i]) ** 2
    noise = yerr[j] ** 2 + yerr[i] ** 2
    df2_corr = df2 - noise

    ldt = np.log(np.maximum(dt, 1e-300))
    edges = np.unique(np.quantile(ldt, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        edges = np.array([ldt.min(), ldt.max() + 1e-12])

    dt_m, sf, sf_err, npr, nfrac, ok = [], [], [], [], [], []
    for b in range(edges.size - 1):
        if b == edges.size - 2:
            sel = (ldt >= edges[b]) & (ldt <= edges[b + 1])
        else:
            sel = (ldt >= edges[b]) & (ldt < edges[b + 1])
        nb = int(np.sum(sel))
        if nb < min_pairs:
            continue
        d_b = dt[sel]
        c_b = df2_corr[sel]
        r_b = df2[sel]
        n_b = noise[sel]
        sf2 = float(np.mean(c_b))
        s = float(np.sqrt(max(sf2, 0.0)))
        e2 = float(np.std(c_b, ddof=1) / np.sqrt(nb)) if nb > 1 else np.nan
        e = e2 / (2.0 * s) if s > 0 else np.nan
        nf = float(np.mean(n_b) / np.mean(r_b)) if np.mean(r_b) > 0 else 1.0
        good = bool(np.isfinite(e) and e > 0 and s > 0 and nf < 0.5)
        dt_m.append(float(np.median(d_b)))
        sf.append(s)
        sf_err.append(e)
        npr.append(nb)
        nfrac.append(nf)
        ok.append(good)

    return dict(dt=np.array(dt_m), sf=np.array(sf), err=np.array(sf_err),
                npairs=np.array(npr, dtype=int), noise_frac=np.array(nfrac),
                ok=np.array(ok, dtype=bool))


def sf_statistic(t, y, yerr, sigma, tau, pair_idx, n_bins: int = 12,
                 min_pairs: int = 8) -> Tuple[float, Dict[str, np.ndarray]]:
    """T_SF = 有效 bin 上 (SF_obs - SF_model)^2 / err^2 的平均。用 MC 标定其零分布。"""
    tab = structure_function(t, y, yerr, n_bins=n_bins, min_pairs=min_pairs,
                             pair_idx=pair_idx)
    ok = tab["ok"]
    if int(np.sum(ok)) < 2:
        return np.nan, tab
    sigma = _as_scalar(sigma, "sigma")
    tau = _as_scalar(tau, "tau")
    m = sf_model(tab["dt"][ok], sigma, tau)
    r = (tab["sf"][ok] - m) / tab["err"][ok]
    return float(np.mean(r * r)), tab


def fit_sf(t, y, yerr, n_bins: int = 12, min_pairs: int = 8, pair_idx=None,
           guess: Optional[Tuple[float, float]] = None):
    """独立地用 SF 拟合 (sigma, tau)，作为 GP 结果的交叉检查。"""
    tab = structure_function(t, y, yerr, n_bins=n_bins, min_pairs=min_pairs,
                             pair_idx=pair_idx)
    ok = tab["ok"]
    if int(np.sum(ok)) < 3:
        return None
    dts = tab["dt"][ok]
    sfs = tab["sf"][ok]
    errs = tab["err"][ok]

    def chi2(p):
        m = sf_model(dts, np.exp(p[0]), np.exp(p[1]))
        return float(np.sum(((sfs - m) / errs) ** 2))

    sd_guess = float(np.max(sfs)) / np.sqrt(2.0)
    tau_guess = guess[1] if guess is not None else float(np.median(dts))
    starts = [np.log(np.array([max(sd_guess, 1e-12), max(tau_guess, 1e-12)]))]
    if guess is not None:
        starts.append(np.log(np.array([max(guess[0], 1e-12), max(guess[1], 1e-12)])))
    best = None
    for x0 in starts:
        try:
            res = minimize(chi2, x0, method="Nelder-Mead",
                           options=dict(maxiter=4000, xatol=1e-8, fatol=1e-8))
        except Exception:
            continue
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        return None
    sig, tau = np.exp(best.x)
    return dict(sigma=float(sig), tau=float(tau), chi2=float(best.fun),
                dof=int(np.sum(ok) - 2), chi2_dof=float(best.fun / max(np.sum(ok) - 2, 1)),
                table=tab)


def short_lag_slope(tab: Dict[str, np.ndarray], tau_ref: float) -> float:
    """拟合短时延 bin 的 log SF-log dt 斜率（DRW 在 dt << tau 时趋于 0.5）。"""
    ok = tab["ok"]
    if int(np.sum(ok)) < 3:
        return np.nan
    dts = tab["dt"][ok]
    sfs = tab["sf"][ok]
    sel = dts < max(tau_ref / 3.0, np.min(dts) * 3)
    if int(np.sum(sel)) < 3:
        sel = np.zeros_like(ok[ok])
        sel[:min(4, sel.size)] = True
    if int(np.sum(sel)) < 3:
        return np.nan
    x = np.log(dts[sel])
    yv = np.log(sfs[sel])
    A = np.vstack([x, np.ones_like(x)]).T
    slope = np.linalg.lstsq(A, yv, rcond=None)[0][0]
    return float(slope)


# ============================================================================
# 5. OU 过程的精确模拟（Markov 递推，O(N)，不需要 Cholesky）
# ============================================================================
def sample_ou(t: np.ndarray, sigma: float, tau: float,
              rng: np.random.Generator) -> np.ndarray:
    """OU 过程在任意（已排序）时刻上的精确采样。

    OU 是 Markov 过程，故联合分布可精确分解：
        X_1 ~ N(0, sigma^2)
        X_k = r X_{k-1} + sigma sqrt(1-r^2) xi_k,  r = exp(-dt/tau)
    """
    n = t.size
    xi = rng.standard_normal(n)
    x = np.empty(n, dtype=float)
    x[0] = sigma * xi[0]
    decay = np.exp(-(t[1:] - t[:-1]) / tau)
    amp = sigma * np.sqrt(np.maximum(1.0 - decay * decay, 0.0))
    for k in range(1, n):
        x[k] = decay[k - 1] * x[k - 1] + amp[k - 1] * xi[k]
    return x


def sample_model(t: np.ndarray, sigmas, taus, yerr, jitter: float,
                 rng: np.random.Generator) -> np.ndarray:
    """多成分 OU 之和 + 测量噪声。"""
    y = np.zeros(t.size, dtype=float)
    for s, ta in zip(np.atleast_1d(sigmas), np.atleast_1d(taus)):
        y += sample_ou(t, float(s), float(ta), rng)
    if jitter > 0:
        y = y + jitter * rng.standard_normal(t.size)
    return y + yerr * rng.standard_normal(t.size)


# ============================================================================
# 6. Monte-Carlo 标定
# ============================================================================
def mc_calibration(dat: LCData, liks: Dict[str, object], best: Dict[str, FitResult],
                   n_mc: int, rng: np.random.Generator, diag_idx: np.ndarray,
                   pair_idx, n_bins: int, min_pairs: int, verbose: bool = True):
    """参数自助法：从最佳拟合 DRW(+jitter) 生成 n_mc 条曲线，各自重拟合。"""
    ref = best["drw_jitter"]
    sigmas = ref.params["sigma"]
    taus = ref.params["tau"]
    jitter = ref.jitter

    t_d = dat.t[diag_idx]
    e_d = dat.yerr[diag_idx]

    out = {k: np.full(n_mc, np.nan) for k in liks}
    out["T_fit"] = np.full(n_mc, np.nan)
    out["T_sf"] = np.full(n_mc, np.nan)
    out["sigma"] = np.full(n_mc, np.nan)
    out["tau"] = np.full(n_mc, np.nan)
    out["jitter"] = np.full(n_mc, np.nan)

    t0 = time.time()
    pbar = _ProgressBar(n_mc, desc=f"MC (n={n_mc})", unit="sim", enabled=verbose)
    try:
        for s in range(n_mc):
            ysim = sample_model(dat.t, sigmas, taus, dat.yerr, jitter, rng)

            sim_best: Dict[str, FitResult] = {}
            for key, lik in liks.items():
                lik.t = dat.t          # 复用对象，只换 y
                lik.y = np.ascontiguousarray(ysim, dtype=float)
                starts = []
                bt = best[key].theta
                starts.append(bt)
                if lik.n_params >= 2:
                    th2 = np.array(bt, dtype=float).copy()
                    # 抖动 tau（偶数索引 1,3,...）
                    for idx in range(1, 2 * lik.n_terms, 2):
                        th2[idx] += 0.7
                    starts.append(th2)
                try:
                    fr = fit_model_warmstart(lik, starts, n_starts=2)
                except Exception:
                    continue
                sim_best[key] = fr
                out[key][s] = fr.ln_like

            ref_sim = sim_best.get("drw_jitter")
            if ref_sim is not None:
                try:
                    g = gp_chi2(t_d, ysim[diag_idx], e_d, ref_sim.params["sigma"],
                                ref_sim.params["tau"], ref_sim.jitter)
                    out["T_fit"][s] = g["chi2"]
                except np.linalg.LinAlgError:
                    pass
                out["sigma"][s] = ref_sim.sigma
                out["tau"][s] = ref_sim.tau
                out["jitter"][s] = ref_sim.jitter
                T, _ = sf_statistic(dat.t, ysim, dat.yerr, ref_sim.params["sigma"],
                                    ref_sim.params["tau"], pair_idx,
                                    n_bins=n_bins, min_pairs=min_pairs)
                out["T_sf"][s] = T

            pbar.update(1, extra=(f"tau={ref_sim.tau:.3g} sigma={ref_sim.sigma:.3g}"
                                 if ref_sim is not None else ""))
    finally:
        pbar.close()
    if verbose:
        n_ok = int(np.sum(np.isfinite(out["T_fit"])))
        print(f"[info] MC 完成：{n_ok}/{n_mc} 条模拟成功，用时 {_fmt_time(time.time() - t0)}")
    return out


def pvalue(null: np.ndarray, obs: float) -> float:
    """p = P(null >= obs)，含 +1 平滑（避免 p=0）。"""
    v = np.asarray(null, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0 or not np.isfinite(obs):
        return np.nan
    return float((np.sum(v >= obs) + 1.0) / (v.size + 1.0))


# ============================================================================
# 7. 主接口
# ============================================================================
def test_drw(t, flux, e_flux=None, *, n_mc: int = 200, seed: int = 42,
             n_bins: int = 12, min_pairs: int = 8, diag_max: int = 1500,
             pair_max: int = 2_000_000, fit_two_ou: bool = True,
             use_celerite: Optional[bool] = None, n_starts: int = 6,
             verbose: bool = True, plot: Optional[str] = None) -> dict:
    """检验光变曲线是否是 damped random walk。

    Parameters
    ----------
    t, flux, e_flux : array-like
        时刻、流量（可以是相对流量或星等）、1-sigma 误差。
    n_mc : int
        参数自助法模拟次数（>=100 才有像样的 p 值分辨率；200 比较稳）。
    diag_max : int
        用于白化残差/chi2 诊断（O(N^3)）的最大点数，超过则随机抽取子样本。
        观测和模拟用同一个子样本，统计量口径一致。
    fit_two_ou : bool
        是否拟合双 OU 备择模型（更陡 PSD 的检验，值得开）。

    Returns
    -------
    dict，包含所有数值结果与 "report" 文本。
    """
    rng = np.random.default_rng(seed)
    dat = prepare_data(t, flux, e_flux, verbose=verbose)
    n = dat.n

    # ---- 诊断子样本（O(N^3) 的白化只在这个子集上做） ----
    if n > diag_max:
        diag_idx = np.sort(rng.choice(n, size=diag_max, replace=False))
        if verbose:
            print(f"[info] N={n} > diag_max={diag_max}：白化残差/chi2 诊断"
                  f"在随机抽取的 {diag_max} 点上做（观测与模拟口径一致）")
    else:
        diag_idx = np.arange(n)

    pair_idx = make_pairs(n, pair_max, rng)
    if verbose and pair_idx[0].size < n * (n - 1) // 2:
        print(f"[info] pair 数过多，已随机子采样到 {pair_idx[0].size} 对"
              "（观测与模拟使用同一套 pair）")

    # ---- 模型集合 ----
    liks: Dict[str, object] = {}
    liks["wn"] = WhiteNoiseLikelihood(dat, fit_jitter=True, label="white noise")
    liks["drw"] = OULikelihood(dat, n_terms=1, fit_jitter=False,
                               use_celerite=use_celerite, label="DRW")
    liks["drw_jitter"] = OULikelihood(dat, n_terms=1, fit_jitter=True,
                                      use_celerite=use_celerite, label="DRW + jitter")
    if fit_two_ou:
        liks["two_ou"] = OULikelihood(dat, n_terms=2, fit_jitter=True,
                                      use_celerite=use_celerite, label="2OU + jitter")

    if verbose:
        print(f"[info] 拟合模型: {', '.join(liks)}"
              f"{' (celerite2)' if (use_celerite is not False and _HAVE_CELERITE) else ' (numpy)'}")

    # ---- 观测数据的拟合 ----
    best: Dict[str, FitResult] = {}
    for key, lik in liks.items():
        ns = n_starts if key != "two_ou" else max(n_starts, 9)
        if verbose:
            print(f"[info] 拟合 {lik.label:<14}({lik.n_params} 自由参数, {ns} 起点) ...",
                  end="", flush=True)
        t_fit = time.time()
        try:
            best[key] = fit_model(lik, dat, n_starts=ns)
        except Exception as exc:
            if verbose:
                print(f" 失败: {exc}")
            continue
        if verbose:
            fr = best[key]
            extra = ""
            if lik.n_terms >= 1:
                extra = (f", tau={float(fr.params['tau'][0]):.5g}"
                         f", sigma={float(fr.params['sigma'][0]):.5g}")
            if float(fr.params.get("jitter", 0.0)) > 0:
                extra += f", jitter={float(fr.params['jitter']):.3g}"
            print(f" lnL={fr.ln_like:.3f}{extra}  ({time.time() - t_fit:.1f}s)")
    for must in ("wn", "drw", "drw_jitter"):
        if must not in best:
            raise RuntimeError(f"模型 {must} 拟合失败，无法继续")

    order = [k for k in ["wn", "drw", "drw_jitter", "two_ou"] if k in best]
    fits = information_criteria([best[k] for k in order], n)
    best = dict(zip(order, fits))       # 保持顺序，且对象已带上 aic/bic
    w_aic = aic_weights(fits)

    ref = best["drw_jitter"]
    sigma_fit = float(ref.params["sigma"][0])
    tau_fit = float(ref.params["tau"][0])
    jit_fit = ref.jitter
    # tau 的拟合上界（与 OULikelihood.bounds() 保持一致）
    tau_ub = float(np.exp(liks["drw_jitter"].bounds()[1][1]))

    # ---- 观测统计量 ----
    t_d = dat.t[diag_idx]
    y_d = dat.y[diag_idx]
    e_d = dat.yerr[diag_idx]
    try:
        diag = gp_chi2(t_d, y_d, e_d, ref.params["sigma"], ref.params["tau"], jit_fit)
    except np.linalg.LinAlgError:
        diag = dict(chi2=np.nan, lnlike=ref.ln_like, z=np.array([]), logdet=np.nan,
                    z_std=np.nan, acf1=np.nan, lb_Q=np.nan, lb_p=np.nan)
    dof = max(diag_idx.size - ref.k_total, 1)
    chi2_dof = diag["chi2"] / dof

    T_sf_obs, sf_tab = sf_statistic(dat.t, dat.y, dat.yerr, ref.params["sigma"],
                                    ref.params["tau"], pair_idx,
                                    n_bins=n_bins, min_pairs=min_pairs)

    # ---- MC 标定 ----
    if verbose:
        print(f"[info] 开始 MC 标定：{n_mc} 条模拟曲线 × {len(liks)} 个模型重拟合 "
              f"(共约 {n_mc * len(liks)} 次拟合)")
    mc = mc_calibration(dat, liks, best, n_mc, rng, diag_idx, pair_idx,
                        n_bins, min_pairs, verbose=verbose)

    p_gof = pvalue(mc["T_fit"], diag["chi2"])
    p_sf = pvalue(mc["T_sf"], T_sf_obs)

    # 用 MC 零分布标定模型比较（真正的嵌套检验，自动处理边界问题）
    dlnl_2ou_obs = best["two_ou"].ln_like - ref.ln_like if "two_ou" in best else np.nan
    dlnl_2ou_null = mc.get("two_ou", np.full(n_mc, np.nan)) - mc["drw_jitter"]
    p_2ou = pvalue(dlnl_2ou_null, dlnl_2ou_obs)

    dlnl_jit_obs = ref.ln_like - best["drw"].ln_like
    # jitter = 0 位于参数空间边界 -> 零分布为 0.5*chi2_0 + 0.5*chi2_1 (Self & Liang)
    p_jitter = 0.5 * float(stats.chi2.sf(max(2.0 * dlnl_jit_obs, 0.0), 1))

    # WN vs DRW: sigma -> 0 同样在边界上；tau 在零假设下不可识别，故用同样的
    # 边界修正作为【保守近似】（p 偏大）。主判据仍以 BIC 为准。
    dlnl_wn_obs = ref.ln_like - best["wn"].ln_like
    p_wn_lrt = 0.5 * float(stats.chi2.sf(max(2.0 * dlnl_wn_obs, 0.0), 1))

    # ---- MC bootstrap 参数误差 ----
    def _pct(v):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 5:
            return (np.nan, np.nan)
        return (float(np.percentile(v, 15.87)), float(np.percentile(v, 84.13)))

    sig_lo, sig_hi = _pct(mc["sigma"])
    tau_lo, tau_hi = _pct(mc["tau"])
    jit_lo, jit_hi = _pct(mc["jitter"])

    # ---- SF 独立拟合 + 短时标斜率 ----
    sf_fit = fit_sf(dat.t, dat.y, dat.yerr, n_bins=n_bins, min_pairs=min_pairs,
                    pair_idx=pair_idx, guess=(sigma_fit, tau_fit))
    slope_short = short_lag_slope(sf_tab, tau_fit)

    # ---- 汇总 ----
    res: Dict[str, object] = dict(
        data=dat, fits={k: best[k] for k in order}, aic_weights=dict(zip(order, w_aic)),
        sigma=sigma_fit, tau=tau_fit, jitter=jit_fit, tau_upper_bound=tau_ub,
        sigma_err=(sig_lo, sig_hi), tau_err=(tau_lo, tau_hi), jitter_err=(jit_lo, jit_hi),
        sf_inf=float(np.sqrt(2.0) * sigma_fit),
        chi2=float(diag["chi2"]), chi2_dof=float(chi2_dof), dof=int(dof),
        z_std=float(diag["z_std"]), acf1=float(diag["acf1"]),
        lb_Q=float(diag["lb_Q"]), lb_p=float(diag["lb_p"]),
        p_gof=p_gof, p_sf=p_sf, p_jitter=p_jitter, p_2ou=p_2ou, p_wn_lrt=p_wn_lrt,
        delta_lnl_2ou=float(dlnl_2ou_obs), delta_lnl_jitter=float(dlnl_jit_obs),
        delta_lnl_wn=float(dlnl_wn_obs),
        sf_table=sf_tab, sf_fit=sf_fit, sf_slope_short=slope_short,
        mc=mc, n_mc=int(n_mc), baseline=dat.baseline, dt_min=dat.dt_min,
        n=int(n), diag_idx=diag_idx, pair_idx=pair_idx,
        whitened_residuals=diag["z"],
    )

    res["verdict"] = _make_verdict(res)
    res["report"] = format_report(res)

    if plot:
        try:
            make_plot(res, dat, plot)
            if verbose:
                print(f"[info] 图已保存: {plot}")
        except Exception as exc:
            print(f"[warn] 画图失败: {exc}")

    if verbose:
        print()
        print(res["report"])
    return res


# ------------------------------------------------------------------ 结论逻辑
def _make_verdict(res: dict) -> List[str]:
    out: List[str] = []
    fits = res["fits"]
    bic = {k: f.bic for k, f in fits.items()}
    k_best = min(bic, key=bic.get)

    # 0) 有没有相关变光？
    d_bic_wn = bic["wn"] - min(bic.get("drw_jitter", np.inf), bic.get("drw", np.inf))
    if k_best == "wn" or d_bic_wn < 2:
        out.append("✗ 数据与【纯白噪声】一致：没有证据表明存在 DRW 型相关变光。"
                   " 此时 DRW 参数没有物理意义。")
    else:
        out.append(f"✓ DRW 型相关变光显著：相对白噪声 ΔBIC ≈ {d_bic_wn:.1f}"
                   f"（LRT 近似 p = {res['p_wn_lrt']:.2e}）。")

    # 1) 拟合优度
    pg = res["p_gof"]
    if np.isfinite(pg):
        if pg < 0.01:
            out.append(f"✗ DRW 被【拒绝】：拟合优度 p = {pg:.4f} < 0.01"
                       f"（chi2/dof = {res['chi2_dof']:.2f}，"
                       f"dof={res['dof']}）。DRW 不足以描述这条曲线。")
        elif pg < 0.05:
            out.append(f"△ 边缘：拟合优度 p = {pg:.4f}，DRW 有些勉强，注意多看几个诊断。")
        else:
            out.append(f"✓ 不能拒绝 DRW：拟合优度 p = {pg:.3f}，"
                       f"chi2/dof = {res['chi2_dof']:.2f}（dof = {res['dof']}）。")

    ps = res["p_sf"]
    if np.isfinite(ps):
        if ps < 0.05:
            out.append(f"✗ 结构函数检验 p = {ps:.4f}：SF 形状与 DRW 预言不符"
                       f"（短时标对数斜率 = {res['sf_slope_short']:.2f}，DRW 在 dt << tau 时预期 ≈ 0.5）。")
        else:
            out.append(f"✓ 结构函数与 DRW 一致：p = {ps:.3f}"
                       f"（短时标对数斜率 = {res['sf_slope_short']:.2f}，DRW 在 dt << tau 时预期 ≈ 0.5）。")

    # 2) 多余白噪声
    d_bic_jit = bic["drw"] - bic["drw_jitter"]
    pj = res["p_jitter"]
    if d_bic_jit > 6 and np.isfinite(pj) and pj < 0.01:
        jit = res["jitter"]
        errmed = float(np.median(res["data"].yerr))
        out.append(f"✗ 需要额外白噪声：jitter = {jit:.4g} vs 中位误差 {errmed:.4g}"
                   f"（p = {pj:.2e}, ΔBIC = {d_bic_jit:.1f}）。"
                   " 可能是误差被低估、或存在 DRW 之外的快变过程。")
    elif d_bic_jit > 2:
        out.append(f"△ jitter 有弱证据（ΔBIC = {d_bic_jit:.1f}, p = {pj:.3g}）："
                   "建议核对一下误差，或直接用 DRW+jitter 模型。")
    else:
        out.append(f"✓ 不需要额外白噪声：DRW + 给出的误差就够了"
                   f"（jitter = {res['jitter']:.3g}，不显著）。")

    # 3) PSD 是否比 DRW 更陡 / 更复杂
    if "two_ou" in fits:
        d_bic_2ou = bic["drw_jitter"] - bic["two_ou"]
        if d_bic_2ou > 10 and np.isfinite(res["p_2ou"]) and res["p_2ou"] < 0.01:
            out.append(f"✗ 数据更偏好双 OU（ΔBIC = {d_bic_2ou:.1f}, "
                       f"MC p = {res['p_2ou']:.4f}）：PSD 可能比单成分 DRW 更陡/有双时标。")
        elif d_bic_2ou > 10:
            out.append(f"△ 双 OU 的 BIC 更优 (ΔBIC = {d_bic_2ou:.1f})，"
                       f"但 MC 标定不显著 (p = {res['p_2ou']:.3f})，"
                       " 很可能是过拟合/时标退化，暂不推翻 DRW。")
        else:
            out.append(f"✓ 无需更复杂的 PSD：双 OU 相对 DRW 无明显改进"
                       f"（ΔBIC = {d_bic_2ou:.1f}, MC p = {res['p_2ou']:.3f}）。")

    # 4) 时标是否真的被测量到
    ratio = res["baseline"] / res["tau"] if res["tau"] > 0 else np.inf
    tau_ub = float(res.get("tau_upper_bound", np.inf))
    if res["tau"] > 0.5 * tau_ub:
        out.append(f"⚠ tau = {res['tau']:.4g} 已贴到拟合上界 ({tau_ub:.4g} = 100×baseline)："
                   " 窗口内看不到 PSD 拐点，DRW 实际上被当成随机游走在用，"
                   " tau 与 sigma 都不可信，只能报下限。此时 p_gof 大也不代表'是 DRW'。")
    elif res["tau"] >= res["baseline"]:
        out.append(f"⚠ tau = {res['tau']:.4g} ≥ baseline = {res['baseline']:.4g}："
                   " tau 完全没有被约束，只能当作【下限】，不要引用具体数值。")
    elif ratio < 3:
        out.append(f"⚠ baseline/tau = {ratio:.2f} < 3：tau 的约束很弱（尤其是上限），"
                   " 报告时应说明这一点。")
    else:
        out.append(f"✓ baseline/tau = {ratio:.1f}，tau 被数据较好地约束。")

    # 5) 残差
    if np.isfinite(res["lb_p"]) and res["lb_p"] < 0.01:
        out.append(f"△ 白化残差仍有结构（Ljung-Box p = {res['lb_p']:.4f}），"
                   "说明模型还没吃干净，值得再看看。")
    if np.isfinite(res["z_std"]) and (res["z_std"] > 1.15 or res["z_std"] < 0.85):
        out.append(f"△ 白化残差 std = {res['z_std']:.2f} 明显偏离 1 "
                   "（>1 说明误差被低估/模型太窄，<1 说明误差被高估）。")

    return out


# ------------------------------------------------------------------ 报告文本
def format_report(res: dict) -> str:
    dat: LCData = res["data"]
    fits: Dict[str, FitResult] = res["fits"]
    w = res["aic_weights"]
    line = "=" * 74
    s: List[str] = []
    s.append(line)
    s.append(" DRW (damped random walk) 检验报告")
    s.append(line)
    s.append(f" N                = {res['n']}" +
             (f"  (剔除 {dat.n_dropped} 个无效点)" if dat.n_dropped else ""))
    s.append(f" baseline         = {res['baseline']:.6g}")
    s.append(f" dt_min           = {res['dt_min']:.6g}")
    s.append(f" median(dt)       = {dat.dt_median:.6g}")
    s.append(f" median(err)      = {np.median(dat.yerr):.6g}")
    s.append(f" std(flux)        = {np.std(dat.y):.6g}"
             + (f"   (≈ {dat.snr:.1f}× median err)" if np.isfinite(dat.snr) else ""))
    s.append(f" 用到的误差模型    = {'celerite2' if _HAVE_CELERITE else 'numpy Cholesky'}")
    s.append(f" MC 模拟次数       = {res['n_mc']}")
    s.append("")

    s.append("-" * 74)
    s.append(" [1] 模型比较     lnL 越大越好; AIC/BIC 越小越好")
    s.append("-" * 74)
    s.append(f"   {'model':<16}{'k':>4}{'lnL':>14}{'AIC':>12}{'BIC':>12}"
             f"{'ΔBIC':>9}{'AICw':>8}")
    bic_min = min(f.bic for f in fits.values())
    for key, f in fits.items():
        star = "  <-- best BIC" if abs(f.bic - bic_min) < 1e-9 else ""
        s.append(f"   {f.label:<16}{f.k_total:>4}{f.ln_like:>14.3f}"
                 f"{f.aic:>12.2f}{f.bic:>12.2f}{f.bic - bic_min:>9.2f}"
                 f"{w[key]:>8.3f}{star}")
    s.append("   ΔBIC 经验判据: <2 无差别, 2-6 弱证据, 6-10 强证据, >10 非常强")
    s.append("")

    s.append("-" * 74)
    s.append(" [2] 最佳拟合 DRW(+jitter) 参数 (区间来自 MC bootstrap, 16-84 分位)")
    s.append("-" * 74)
    tau_lo, tau_hi = res["tau_err"]
    sig_lo, sig_hi = res["sigma_err"]
    jit_lo, jit_hi = res["jitter_err"]
    s.append(f"   tau      = {res['tau']:.6g}"
             + (f"  (-{res['tau'] - tau_lo:.3g} / +{tau_hi - res['tau']:.3g})"
                if np.isfinite(tau_lo) else ""))
    s.append(f"   sigma    = {res['sigma']:.6g}"
             + (f"  (-{res['sigma'] - sig_lo:.3g} / +{sig_hi - res['sigma']:.3g})"
                if np.isfinite(sig_lo) else ""))
    s.append(f"   SF_inf   = sqrt(2)*sigma = {res['sf_inf']:.6g}")
    if res["jitter"] > 0:
        s.append(f"   jitter   = {res['jitter']:.6g}"
                 + (f"  (-{res['jitter'] - jit_lo:.3g} / +{jit_hi - res['jitter']:.3g})"
                    if np.isfinite(jit_lo) else ""))
    s.append(f"   baseline/tau = {res['baseline'] / res['tau']:.3f}"
             f"   (tau 拟合上界 = {res.get('tau_upper_bound', float('nan')):.4g})")
    if "two_ou" in res["fits"]:
        p = res["fits"]["two_ou"].params
        s.append("   双 OU 备择的最佳参数: "
                 + ", ".join(f"sigma={v:.4g}" for v in p["sigma"])
                 + " | " + ", ".join(f"tau={v:.4g}" for v in p["tau"])
                 + (f" | jitter={p['jitter']:.4g}" if p["jitter"] > 0 else ""))
    s.append("")

    s.append("-" * 74)
    s.append(" [3] 拟合优度（参数自助法标定，零假设 = 数据来自最佳拟合 DRW）")
    s.append("-" * 74)
    s.append(f"   chi2 = {res['chi2']:.2f}   dof = {res['dof']}   "
             f"chi2/dof = {res['chi2_dof']:.3f}   (chi2/dof ≈ 1 才算好)")
    s.append(f"   p_gof (chi2 检验)          = {res['p_gof']:.4f}"
             "   < 0.01 拒绝 DRW")
    s.append(f"   p_SF  (结构函数形状检验)    = {res['p_sf']:.4f}")
    s.append(f"   p_2OU (DRW vs 双OU, 嵌套)  = {res['p_2ou']:.4f}"
             "   < 0.01 说明需要更复杂的 PSD")
    s.append(f"   p_jitter (是否需要额外白噪声)= {res['p_jitter']:.4f}"
             "   (边界修正 LRT, 近似)")
    s.append("   p_gof/p_SF 由 MC 标定; p_2OU 亦由 MC 标定; "
             "p_jitter/p_wn 为边界修正 LRT 近似值")
    s.append(f"   白化残差: std = {res['z_std']:.3f} (期望 1), "
             f"lag-1 自相关 = {res['acf1']:.3f} (期望 0)")
    s.append(f"   Ljung-Box Q = {res['lb_Q']:.2f}, p = {res['lb_p']:.4f}"
             "   (p 小 = 残差还有结构)")
    s.append("")

    s.append("-" * 74)
    s.append(" [4] 结构函数 (去噪后)")
    s.append("-" * 74)
    tab = res["sf_table"]
    s.append(f"   {'dt':>12}{'SF':>12}{'err':>12}{'N_pair':>8}{'noise_frac':>12}  ok")
    for i in range(tab["dt"].size):
        s.append(f"   {tab['dt'][i]:>12.5g}{tab['sf'][i]:>12.5g}"
                 f"{tab['err'][i]:>12.4g}{tab['npairs'][i]:>8d}"
                 f"{tab['noise_frac'][i]:>12.3f}  "
                 + ("Y" if tab["ok"][i] else "n(排除)"))
    s.append("   noise_frac = 测量误差贡献的比例; >0.5 的 bin 不可靠（已排除）")
    if res["sf_fit"] is not None:
        f2 = res["sf_fit"]
        s.append(f"   单用 SF 独立拟合: sigma = {f2['sigma']:.6g}, tau = {f2['tau']:.6g}, "
                 f"chi2/dof = {f2['chi2_dof']:.2f}")
        s.append("   （应与 [2] 的 GP 结果一致；若差很多，说明 DRW 形状本身就不对）")
    s.append(f"   短时标双对数斜率 = {res['sf_slope_short']:.3f}  (DRW 在 dt << tau 时预期 ≈ 0.5)")
    s.append("")

    s.append("-" * 74)
    s.append(" [5] 结论")
    s.append("-" * 74)
    for v in res["verdict"]:
        s.append("   " + v)
    s.append("")

    s.append("-" * 74)
    s.append(" [6] 注意事项")
    s.append("-" * 74)
    for c in [
        "不能拒绝 ≠ 证明。DRW 与更陡的单幂律 PSD 常无法区分，所以本程序比较了双 OU 备择。",
        "p 值只在当前采样/基线/信噪比下有效；加长基线常常会把'一致'变成'不一致'。",
        "tau 只有 baseline >> tau 时才是测量值，否则只是下限（见结论中的警告）。",
        "均值用加权平均预先减掉（不是 GP 意义上的 ML 均值），与部分文献口径略有差别。",
        "若 flux 是星等，参数单位是 mag；DRW 在高斯化意义上才严格成立。",
        "被拒绝时，先怀疑误差是否被低估、是否有未被剔除的坏点/系统误差，再怀疑模型。",
    ]:
        s.append("   * " + c)
    s.append(line)
    return "\n".join(s)


# ------------------------------------------------------------------ 画图
def make_plot(res: dict, dat: LCData, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref: FitResult = res["fits"]["drw_jitter"]
    sig, tau, jit = ref.params["sigma"], ref.params["tau"], ref.jitter

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    ax = axes[0, 0]
    ax.errorbar(dat.t, dat.y + dat.mean, yerr=dat.yerr, fmt="o", ms=3, lw=1,
                color="k", alpha=0.3, label="data")
    if dat.n <= 4000:
        tg = np.linspace(dat.t[0], dat.t[-1], 400)
        mu, v = conditional_prediction(dat.t, dat.y, dat.yerr, sig, tau, jit, tg)
        ax.fill_between(tg, dat.mean + mu - np.sqrt(v), dat.mean + mu + np.sqrt(v),
                        color="orange", alpha=0.25, label="DRW prediction (1σ)")
        ax.plot(tg, dat.mean + mu, color="C0", lw=1.2)
    ax.set_xlabel("t"); ax.set_ylabel("flux"); ax.legend(fontsize=8)
    ax.set_title(f"DRW: tau={res['tau']:.4g}, sigma={res['sigma']:.4g}")

    ax = axes[0, 1]
    z = res["whitened_residuals"]
    if z.size:
        ax.plot(dat.t[res["diag_idx"]], z, "o", ms=3, color="C3", alpha=0.7)
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(1, color="gray", ls="--", lw=0.7)
        ax.axhline(-1, color="gray", ls="--", lw=0.7)
    ax.set_xlabel("t"); ax.set_ylabel("whitened residual z")
    ax.set_title(f"z: std={res['z_std']:.2f}, LB p={res['lb_p']:.3f}")

    ax = axes[1, 0]
    tab = res["sf_table"]
    m = tab["ok"]
    if np.any(m):
        ax.errorbar(tab["dt"][m], tab["sf"][m], yerr=tab["err"][m], fmt="o",
                    color="k", ms=4, label="SF (de-biased)")
        dtd = np.logspace(np.log10(tab["dt"][m].min()), np.log10(tab["dt"][m].max()), 200)
        ax.plot(dtd, sf_model(dtd, res["sigma"], res["tau"]), "C0-", lw=1.5,
                label="best-fit DRW")
        if res["sf_fit"] is not None:
            f2 = res["sf_fit"]
            ax.plot(dtd, sf_model(dtd, f2["sigma"], f2["tau"]), "C1--", lw=1.2,
                    label="SF-only fit")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("dt"); ax.set_ylabel("SF"); ax.legend(fontsize=8)
    ax.set_title("structure function")

    ax = axes[1, 1]
    mc = res["mc"]
    v = mc["T_fit"][np.isfinite(mc["T_fit"])]
    if v.size:
        ax.hist(v, bins=25, color="C0", alpha=0.6,
                label=f"null: {v.size} simulated DRW")
    ax.axvline(res["chi2"], color="C3", lw=2,
               label=f"observed = {res['chi2']:.1f}\np_gof = {res['p_gof']:.3f}")
    ax.set_xlabel("T = r^T C^-1 r"); ax.set_ylabel("count")
    ax.legend(fontsize=8); ax.set_title("MC null distribution and observation value")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ============================================================================
# 8. 命令行 / demo
# ============================================================================
def make_demo_light_curve(kind: str = "drw", n: int = 200, seed: int = 7,
                          baseline: float = 1000.0, tau: float = 100.0,
                          sigma: float = 0.1, err: float = 0.01):
    """生成测试用光变曲线。kind: drw / whitenoise / sine / steep(双OU)。"""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, baseline, size=n))
    t = np.unique(np.round(t, 6))
    e = np.full(t.size, err)
    if kind == "drw":
        y = sample_ou(t, sigma, tau, rng)
    elif kind == "whitenoise":
        y = sigma * rng.standard_normal(t.size)
    elif kind == "sine":
        y = sigma * np.sin(2 * np.pi * t / tau) + 0.2 * sigma * rng.standard_normal(t.size)
    elif kind == "steep":
        y = sample_ou(t, 0.7 * sigma, tau / 10.0, rng) + \
            sample_ou(t, 0.7 * sigma, tau, rng)
    else:
        raise ValueError(f"未知 demo kind: {kind}")
    y = y + 1.0 + err * rng.standard_normal(t.size)
    return t, y, e


def load_light_curve(path: str, cols=("t", "flux", "e_flux")):
    """读两列/三列文本或 CSV；有表头则按名字取列，否则按顺序取前三列。"""
    head = open(path, "r", encoding="utf-8", errors="ignore").readline()
    delim = "," if "," in head else None
    has_header = any(ch.isalpha() for ch in head)
    if has_header:
        rec = np.genfromtxt(path, names=True, delimiter=delim, dtype=float)
        names = list(rec.dtype.names)
        arr = np.column_stack([rec[nm] for nm in names])
        want = [c for c in cols if c in names]
        if len(want) >= 2:
            idx = [names.index(c) for c in want]
            arr = arr[:, idx]
        elif arr.shape[1] < 3:
            raise ValueError(f"文件里有表头, 但找不到列 {cols}; 现有列: {names}")
    else:
        arr = np.loadtxt(path, delimiter=delim, dtype=float, ndmin=2)
    if arr.shape[1] == 2:
        print("[warn] 文件只有两列，误差按 0 处理（不推荐）")
        return arr[:, 0], arr[:, 1], None
    return arr[:, 0], arr[:, 1], arr[:, 2]


def main(argv=None):
    p = argparse.ArgumentParser(
        description="检验光变曲线是否与 damped random walk (DRW) 一致",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", help="光变曲线文件 (t, flux, e_flux 三列)")
    p.add_argument("--demo", action="store_true", help="用合成数据自检")
    p.add_argument("--demo-kind", default="drw",
                   choices=["drw", "whitenoise", "sine", "steep"],
                   help="合成数据种类 (whitenoise/sine 应被判为非 DRW)")
    p.add_argument("--n-mc", type=int, default=200, help="MC 模拟次数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bins", type=int, default=12, help="结构函数 bin 数")
    p.add_argument("--diag-max", type=int, default=1500,
                   help="白化残差诊断的最大点数")
    p.add_argument("--no-2ou", action="store_true", help="不拟合双 OU 备择模型")
    p.add_argument("--no-celerite", action="store_true", help="强制用 numpy 后端")
    p.add_argument("--plot", help="输出诊断图 png 路径")
    p.add_argument("--json", help="把主要结果写入 json 文件")
    args = p.parse_args(argv)

    if args.input:
        t, flux, e = load_light_curve(args.input)
    elif args.demo or not sys.stdin.isatty():
        t, flux, e = make_demo_light_curve(args.demo_kind, seed=args.seed)
        print(f"[info] 使用 demo 数据: kind={args.demo_kind}, N={t.size}")
    else:
        p.print_help()
        return 2

    res = test_drw(t, flux, e, n_mc=args.n_mc, seed=args.seed, n_bins=args.n_bins,
                   diag_max=args.diag_max, fit_two_ou=not args.no_2ou,
                   use_celerite=False if args.no_celerite else None,
                   plot=args.plot)

    if args.json:
        def _clean(v):
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, dict):
                return {k: _clean(x) for k, x in v.items()
                        if k not in ("data", "fits", "mc", "sf_table", "pair_idx")}
            if isinstance(v, (list, tuple)):
                return [_clean(x) for x in v]
            return v

        out = {k: _clean(v) for k, v in res.items()
               if k not in ("data", "fits", "sf_table", "pair_idx",
                            "whitened_residuals", "diag_idx", "mc")}
        out["model_lnL"] = {k: f.ln_like for k, f in res["fits"].items()}
        out["model_bic"] = {k: f.bic for k, f in res["fits"].items()}
        out["model_aic"] = {k: f.aic for k, f in res["fits"].items()}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2, default=str)
        print(f"[info] 结果已写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
