#!/usr/bin/env python3
"""Run MICA on nuclear-corrected total light curves.

Usage
-----
python src/run_mica.py SOURCE                  # start, resume, or skip completed bands
python src/run_mica.py SOURCE B                # one response band
python src/run_mica.py SOURCE B resume         # require an existing recoverable run
python src/run_mica.py SOURCE B fresh          # archive old output, then start again
python src/run_mica.py SOURCE post             # redo post-processing
python src/run_mica.py SOURCE B plot           # redraw decomposition plot
python src/run_mica.py SOURCE --stage mica_round2
python src/run_mica.py SOURCE --stage mica_gamma2_broad --joint
python src/run_mica.py SOURCE --stage mica_gamma2_broad --fit-only

By default all sections named in ``mica_stages`` are run in order.  A stage
can be selected explicitly with ``--stage``. ``max_num_saves`` is the cumulative
number of saved samples; keep the same inputs, model settings and MPI size when
resuming. The native checkpoint restores sampler state, but does not save RNG state.
"""
from __future__ import annotations

import os
import importlib
import inspect
import json
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import yaml

# The top-level process is only a scheduler: it launches one real MPI job per
# response band.  Importing pymica/mpi4py in that scheduler initializes MPI in
# singleton mode, and MPICH 5 then rejects the child mpiexec jobs as a nested,
# incompatible PMI runtime.  Only processes created by mpiexec need MPI.
# 外层仅调度，不导入会初始化 MPI 的 pymica/mpi4py；实际 worker 由 PMI/PMIX 环境标识。
_MPI_WORKER = "PMI_RANK" in os.environ or "PMIX_RANK" in os.environ
if _MPI_WORKER:
    import pymica
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
else:
    pymica = None
    MPI = None
    COMM = None
    RANK = 0
ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "output_root": "runs/mica",
    "prepared_dir": "fflux/nuclear_corrected",
    "run_template": "run_{driver}_to_{band}_{ncomp}comp_{type_tf}",
    "ncomp": 2,
    "type_tf": "gaussian",
    "lag_limit": [-10.0, 80.0],
    "type_lag_prior": 0,
    "max_num_saves": 10000,
    "mpi_per_band": 2,
    "max_parallel": 5,
    "plot_decomp": True,
    # Plot/export grid only; never forwarded to gmodel.setup or mmap.setup.
    "plot_tf_lag_range": None,
    "plot_tf_num_points": None,
}

# map type_tf name to subfolder label used in run directory names
_TF_LABEL = {
    "gaussian": "gaussian",
    "tophat": "tophat",
    "gamma": "gamma",
    "exponential": "exp",
    "exp": "exp",
}

SETUP_KEYS = [
    "type_lag_prior",
    "lag_prior",
    "width_limit",
    "width_prior",
    "flag_lag_posivity",
    "flag_negative_resp",
    "flag_con_sys_err",
    "flag_line_sys_err",
    "flag_trend",
    "flag_gap",
    "gap_prior",
    "ptol",
    "num_particles",
    "thread_steps_factor",
    "new_level_interval_factor",
    "save_interval_factor",
    "lam",
    "beta",
    "max_num_levels",
    "nd_rec",
    "trec_ext",
]

ACTIONS = {"run", "resume", "fresh", "post", "plot", "plotdecomp"}
MIX_CODES = {"0", "1", "2", "3"}


def log(msg=""):
    if RANK == 0:
        print(msg, flush=True)


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def source_dir(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def rel(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


# 优先使用当前 Python 环境旁的 mpiexec，减少 MPI 实现与 mpi4py 链接不匹配。
def compatible_mpiexec() -> Path | str:
    """Use the MPI launcher installed beside the active Python interpreter."""
    candidate = Path(sys.executable).with_name("mpiexec")
    return candidate if candidate.exists() else "mpiexec"


# 配置中的 mica_stages 决定运行顺序；未给出时兼容旧的 mica_round1。
# 这里检查 stage 是否存在且不重复，避免同一次调度误跑同一输出目录。
def configured_stages(cfg: dict) -> list[str]:
    """Return the enabled MICA config sections in execution order."""
    raw = cfg.get("mica_stages")
    if raw is None:
        return ["mica_round1"]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("mica_stages must be a non-empty list of section names")
    stages = [str(value).strip() for value in raw]
    if any(not value for value in stages) or len(stages) != len(set(stages)):
        raise ValueError("mica_stages must contain unique, non-empty section names")
    missing = [stage for stage in stages if not (cfg.get(stage) or {})]
    if missing:
        raise ValueError("mica_stages references missing sections: " + ", ".join(missing))
    return stages


def _fixed_component_count(value, key: str) -> int:
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or int(value[0]) != int(value[1]):
            raise ValueError(f"{key} must specify one fixed component count [n, n]")
        return int(value[0])
    return int(value)


# 配置归一化入口：先合并默认值和指定 stage，再解析驱动/响应波段、核族、固定分量数及先验。
# 混合核代码的每一位对应一个原生分量；这里形成的内部字段供 setup 与目录命名共用。
# lag_limit/lag_prior 影响推断；plot_tf_lag_range/plot_tf_num_points 只影响已有结果的显示网格。
def fit_config(cfg: dict, stage: str = "mica_round1") -> dict:
    raw = dict(cfg.get(stage, {}) or {})
    if not raw:
        raise ValueError(f"missing MICA config section: {stage}")
    fit = dict(DEFAULTS)
    fit.update(raw)
    fit.setdefault("driver", cfg.get("reference_band", cfg.get("driver_band")))
    fit.setdefault("responses", cfg.get("response_bands", []))

    type_tf = str(fit["type_tf"]).strip().lower()
    # 数字串不是连续的数值参数：例如 220 表示三分量的核类型代码。
    is_mixed = bool(type_tf) and type_tf.isdigit()
    if is_mixed:
        if any(code not in MIX_CODES for code in type_tf):
            raise ValueError(f"{stage}.type_tf has an invalid mmap code: {type_tf!r}")
        inferred = len(type_tf)
        if "number_component" in raw:
            ncomp = _fixed_component_count(raw["number_component"], f"{stage}.number_component")
        elif "ncomp" in raw:
            ncomp = int(raw["ncomp"])
        else:
            ncomp = inferred
        if ncomp != inferred:
            raise ValueError(
                f"{stage}.type_tf={type_tf!r} encodes {inferred} components, "
                f"but the configured count is {ncomp}"
            )
        tf_label = f"mix{type_tf}"
        native_type_tf = type_tf
    else:
        if "number_component" in raw:
            ncomp = _fixed_component_count(raw["number_component"], f"{stage}.number_component")
        else:
            ncomp = int(fit["ncomp"])
        if type_tf not in _TF_LABEL:
            raise ValueError(
                f"{stage}.type_tf '{type_tf}' not recognised; "
                f"valid values: {list(_TF_LABEL)} or an mmap code such as '220'"
            )
        tf_label = _TF_LABEL[type_tf]
        native_type_tf = tf_label

    if ncomp < 1:
        raise ValueError(f"{stage} component count must be >= 1, got {ncomp}")
    # 不同核族不能交换同一套参数语义，因此混合模型保留原生标签。
    if is_mixed and str(fit.get("component_order", "native")).lower() not in {
        "", "native", "none"
    }:
        raise ValueError(f"{stage} mixed transfer functions require component_order: native")
    if is_mixed and bool(fit.get("flag_lag_posivity", False)):
        raise ValueError(f"{stage} mmap runs require flag_lag_posivity: false")

    bounds = fit.get("plot_tf_lag_range")
    if bounds is not None:
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"{stage}.plot_tf_lag_range must be [low, high]")
        low, high = map(float, bounds)
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError(f"{stage}.plot_tf_lag_range must have finite increasing bounds")
        fit["plot_tf_lag_range"] = [low, high]
    points = fit.get("plot_tf_num_points")
    if points is not None and (
        isinstance(points, bool) or not isinstance(points, int) or points < 2
    ):
        raise ValueError(f"{stage}.plot_tf_num_points must be an integer >= 2")

    # 逐波段目标覆盖全局累计目标，不是把同一批迭代数复制到所有响应。
    band_targets = fit.get("max_num_saves_by_band", {}) or {}
    if not isinstance(band_targets, dict):
        raise ValueError(f"{stage}.max_num_saves_by_band must be a mapping")
    unknown = set(band_targets) - {str(x) for x in fit["responses"]}
    if unknown:
        raise ValueError(f"{stage}.max_num_saves_by_band has unknown response(s): {sorted(unknown)}")
    if any(int(value) < 1 for value in band_targets.values()):
        raise ValueError(f"{stage}.max_num_saves_by_band targets must be positive")
    fit["max_num_saves_by_band"] = {str(key): int(value) for key, value in band_targets.items()}

    fit["ncomp"] = ncomp
    fit["type_tf"] = type_tf
    fit["_native_type_tf"] = native_type_tf
    fit["_tf_label"] = tf_label
    fit["_mixed"] = is_mixed
    fit["_stage"] = stage
    return fit


# 运行目录名是后处理定位输入的契约，包含 driver、band、ncomp 与核族。
# 本函数允许自定义模板并保证是单层目录名；部分下游 reader 仍按 run_*_to_*_*comp_* 解析。
def run_name(fit: dict, response: str) -> str:
    ncomp = fit["ncomp"]
    tf_label = fit["_tf_label"]
    template = str(fit["run_template"])
    try:
        name = template.format(
            driver=fit["driver"],
            band=response,
            response=response,
            ncomp=ncomp,
            type_tf=tf_label,
            model=tf_label,
            tf_label=tf_label,
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid {fit['_stage']}.run_template: {template!r}") from exc

    path = Path(name)
    if (
        not name.strip()
        or name != name.strip()
        or path.is_absolute()
        or len(path.parts) != 1
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
    ):
        raise ValueError(
            f"{fit['_stage']}.run_template must render to one safe directory name"
        )
    return name


# 普通 gmodel 在 output_root 下按分量数分目录；混合 mmap 的 output_root 已单独隔离模型。
def mica_run_root(src: Path, fit: dict) -> Path:
    root = src / fit["output_root"]
    return root if fit["_mixed"] else root / f"{fit['ncomp']}comp"


def run_dir(src: Path, fit: dict, response: str) -> Path:
    return mica_run_root(src, fit) / run_name(fit, response)


# 冻结本次使用的光变曲线：prepared_dir 中的源文件复制为 input/cont.txt 与 input/line.txt。
# 每行沿用 time、flux、error 三列；后续恢复比较的是这些输入副本及其哈希。
def copy_inputs(src: Path, source: str, fit: dict, response: str, rd: Path) -> None:
    pdir = rel(src, fit["prepared_dir"])
    inp = rd / "input"
    inp.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdir / f"{source}_{fit['driver']}.txt", inp / "cont.txt")
    shutil.copy2(pdir / f"{source}_{response}.txt", inp / "line.txt")


# 一个 set1 表示同一驱动对应的一组响应；pairwise 情况只有 driver 与一个 response。
# 只由 rank 0 读磁盘，再广播数组，保证各 MPI rank 用同一输入。
def load_data() -> dict:
    if COMM is None:
        raise RuntimeError("load_data() must run inside an MPI worker")
    data = None
    if RANK == 0:
        data = {"set1": [np.loadtxt("input/cont.txt"), np.loadtxt("input/line.txt")]}
    return COMM.bcast(data, root=0)


# 按波段覆盖全局分量先验；每个分量必须有一对有限且递增的边界。
# width_prior 是正的物理宽度区间；原生采样中的宽度坐标另行取自然对数。
def component_prior_for_band(
    fit: dict,
    response: str | None,
    key: str = "lag_prior",
    *,
    positive: bool = False,
):
    """Resolve and validate a component prior, with an optional per-band map."""
    by_band = fit.get(f"{key}_by_band")
    value = fit.get(key)
    if by_band is not None:
        if response is None:
            raise ValueError(f"{key}_by_band requires a response band")
        if response not in by_band:
            raise ValueError(f"{key}_by_band is missing response band {response!r}")
        value = by_band[response]
    if value is None:
        return None

    ncomp = int(fit["ncomp"])
    if not isinstance(value, (list, tuple)) or len(value) != ncomp:
        raise ValueError(f"{key} must contain exactly {ncomp} [low, high] pairs")
    resolved = []
    for index, bounds in enumerate(value):
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"{key}[{index}] must be a [low, high] pair")
        low, high = (float(bounds[0]), float(bounds[1]))
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError(f"invalid {key}[{index}] bounds: {bounds!r}")
        if positive and low <= 0.0:
            raise ValueError(f"{key}[{index}] must have a positive lower bound")
        resolved.append([low, high])
    return resolved


# 有序 Gaussian 的兼容桥接：gmodel.setup(lag_prior=...) 会切换先验类型，
# 因此在 TypeLagPrior=0 情况下用原生先验文件约束质心，同时保留有序参数化。
# 这里只支持两个同范围 Gaussian；质心索引 5、8 位于连续光变三参数块及响应误差项之后。
# 连续光变三参数实际是 sys_err_con、ln(sigmad)、ln(taud)，其中只有后两项是 DRW 过程参数。
def ordered_gaussian_prior_text(
    fit: dict, data: dict, response: str | None = None
) -> str | None:
    """Keep native short/long labels while overriding MICA's Tspan/2 cap.

    Passing lag_prior to gmodel.setup forces unordered prior type 4.  For a
    two-Gaussian control with identical bounds, retain ordered type 0 and load
    just the two centroid ranges through MICA's native parameter-prior file.
    """
    if fit["_native_type_tf"] != "gaussian" or fit.get("type_lag_prior") != 0:
        return None
    prior = component_prior_for_band(fit, response, "lag_prior")
    if prior is None:
        return None
    if fit["ncomp"] != 2 or prior[0] != prior[1]:
        raise ValueError(
            "ordered Gaussian lag_prior requires two components with identical bounds"
        )
    if len(data) != 1 or len(next(iter(data.values()))) != 2:
        raise ValueError(
            "ordered Gaussian lag_prior requires one dataset with one driver and one response"
        )
    # A single gmodel data pair has 3 DRW parameters, 1 response systematic,
    # then (amplitude, centroid, log-width) per component: centroids are 5, 8.
    # Native load_par_names cannot skip comments/blank lines; keep pure data
    # and a trailing newline so both rows are actually read.
    return "".join(
        f"{5 + 3 * index} {index}-th_component_centroid UNI {low:.17g} {high:.17g} 0 0\n"
        for index, (low, high) in enumerate(prior)
    )


# 先验文件由 rank 0 写好后再同步；set_priors 必须在每个 rank 调用，
# 因为它修改当前进程中的 C 全局状态，不能仅靠广播 Python 对象代替。
def set_ordered_gaussian_priors(
    model, fit: dict, data: dict, response: str | None = None
) -> None:
    text = ordered_gaussian_prior_text(fit, data, response)
    if text is None:
        return
    if COMM is None:
        raise RuntimeError("set_ordered_gaussian_priors() must run inside an MPI worker")
    path = Path("param/ordered_gaussian_lag_priors.txt")
    if RANK == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        log(f"ordered centroid lag_prior[{response}] -> {path}:\n{text.rstrip()}")
    COMM.Barrier()
    # set_priors registers a C-global flag; every MPI rank must call it, also
    # for standalone post/plot jobs that perform native decomposition.
    model.set_priors(str(path))


# 把项目配置翻译成 pymica.setup 的参数；不在这里做拟合。
# number_component 固定为 [n,n]，表示本次不比较不同分量数模型。
def setup_kwargs(fit: dict, data: dict, response: str | None = None) -> dict:
    ncomp = fit["ncomp"]
    out = {
        "data": data,
        "type_tf": fit["_native_type_tf"],
        "number_component": [ncomp, ncomp],
        "lag_limit": fit["lag_limit"],
        "max_num_saves": int(fit["max_num_saves"]),
    }
    for key in SETUP_KEYS:
        if key in {"lag_prior", "width_prior"}:
            continue
        if fit.get(key) is not None:
            out[key] = fit[key]
    lag_prior = component_prior_for_band(fit, response, "lag_prior")
    if lag_prior is not None and ordered_gaussian_prior_text(fit, data, response) is None:
        out["lag_prior"] = lag_prior
    width_prior = component_prior_for_band(
        fit, response, "width_prior", positive=True
    )
    if width_prior is not None:
        out["width_prior"] = width_prior
    return out


def have_decomp(ncomp: int) -> bool:
    return all(
        (Path("data") / f"pline.txt_{ncomp}_comp{index}").is_file()
        for index in range(ncomp)
    )


# 检查当前安装的 pymica 绘图接口是否支持所请求网格参数；缺少补丁时提前报错。
# 这里只检查签名，不自动修改第三方库。
def check_tf_plot_support(fit: dict) -> None:
    """Catch a missing Python-only grid patch before starting a long fit."""
    if fit.get("plot_tf_num_points") is None:
        return
    functions = [pymica.utility.plot_results]
    if fit.get("plot_decomp", True):
        functions.append(pymica.utility.plot_line_decomp)
    if any("tf_num_points" not in inspect.signature(fn).parameters for fn in functions):
        raise RuntimeError(
            "This pymica Python plotter lacks tf_num_points. Apply "
            "patches/mica_tf_grid.patch to its plotting utilities; no C/Cython rebuild is needed."
        )


# 从保存的 param_input 读取实际生效的原生设置，减少绘图配置与历史运行不一致。
def native_plot_arguments(fit: dict) -> tuple[dict, dict]:
    """Read saved model metadata when calling the Python utility directly.

    The compiled model.plot_results wrapper does not expose tf_num_points.
    Use its saved arguments so only the plotting grid changes.
    """
    settings = {}
    for line in Path("param/param_input").read_text().splitlines():
        fields = line.split("#", 1)[0].split(maxsplit=1)
        if len(fields) == 2:
            settings[fields[0]] = fields[1]
    ncomp = fit["ncomp"]
    model = int(settings["TypeModel"])
    common = {
        "fdir": settings["FileDir"].rstrip("/") + "/",
        "fname": settings["DataFile"],
        "ngau": ncomp,
        "tau_low": float(settings["LagLimitLow"]),
        "tau_upp": float(settings["LagLimitUpp"]),
        "typetf": settings["StrTypeTFMix"] if model == 3 else settings["TypeTF"] * ncomp,
        "typemodel": model,
        "flagnegresp": int(settings.get("FlagNegativeResp", 0)),
        "resp_input": None,
        "doshow": False,
        "tf_lag_range": fit.get("plot_tf_lag_range") or fit["lag_limit"],
    }
    if fit.get("plot_tf_num_points") is not None:
        common["tf_num_points"] = fit["plot_tf_num_points"]
    return common, settings


# 只调用原生结果绘图；网格细化改变显示采样，不会增加后验信息。
def plot_results(model, fit: dict) -> None:
    if fit.get("plot_tf_num_points") is None:
        model.plot_results(
            doshow=False,
            tf_lag_range=fit.get("plot_tf_lag_range") or fit["lag_limit"],
            hist_lag_range=fit["lag_limit"],
        )
        return
    common, settings = native_plot_arguments(fit)
    log(f"TF plot/export grid: {common['tf_lag_range']}, {common['tf_num_points']} points")
    pymica.utility.plot_results(
        **common,
        flagvar=int(settings.get("FlagUniformVarParams", 0)),
        flagtran=int(settings.get("FlagUniformTranFuns", 0)),
        flagtrend=int(settings.get("FlagLongtermTrend", 0)),
        hist_lag_range=fit["lag_limit"],
    )


# 按核族和分量数选择原生分解绘图入口；单分量与混合核存在专门兼容路径。
# 分解曲线是模型条件下的响应分量，comp0 的编号本身不证明它就是吸积盘。
def plot_decomposition(model, fit: dict) -> None:
    """Plot exactly the fixed component count, including mmap ncomp>2.

    MICA's generic ``plot_decomp`` currently loops from two through ncomp and
    therefore asks a fixed three-component mmap run for nonexistent 2comp
    files.  Calling the public plotting utility once avoids that upstream bug.
    """
    if fit["ncomp"] == 1 or fit.get("plot_tf_num_points") is not None or fit.get("plot_tf_lag_range") is not None:
        common, _ = native_plot_arguments(fit)
        pymica.utility.plot_line_decomp(**common)
    elif fit["_mixed"] and fit["ncomp"] > 2:
        low, high = (float(value) for value in fit["lag_limit"])
        pymica.utility.plot_line_decomp(
            "./",
            "data/data_input.txt",
            fit["ncomp"],
            low,
            high,
            fit["_native_type_tf"],
            3,
            bool(fit.get("flag_negative_resp", False)),
            None,
            doshow=False,
        )
    else:
        model.plot_decomp(doshow=False)


# 拟合后的流水线：可选后处理，再重建总响应/各分量，最后绘图与后验权重整理。
# 重建仍调用原生 MICA，可能耗时；阅读、注释或语法检查无需执行本函数。
def finish(model, fit: dict, post_run: bool) -> None:
    if COMM is None:
        raise RuntimeError("finish() must run inside an MPI worker")
    if post_run:
        log("MICA post_run()")
        model.post_run()
        COMM.Barrier()

    log("MICA decompose()")
    if fit["ncomp"] == 1:
        # The native public decomposition loop starts at two components.
        from decompose_mica_branches import decompose_conditional_posterior, posterior_column_count
        posterior = Path("data/posterior_sample1d.txt_1")
        decompose_conditional_posterior(
            pymica, 1, int(fit.get("type_lag_prior", 0)),
            str(posterior), posterior_column_count(posterior),
        )
    else:
        model.decompose()
    COMM.Barrier()

    if RANK == 0:
        log("MICA plot_results()")
        plot_results(model, fit)
        if fit.get("plot_decomp", True):
            log("MICA plot_decomp()")
            plot_decomposition(model, fit)
        log("MICA post_process()")
        model.post_process()
    COMM.Barrier()


# 把仅 rank 0 执行的文件检查异常转成字符串并广播，避免其他 rank 永久等在 collective。
def root_call(function, *args, **kwargs):
    """Broadcast root-side validation errors before peers enter native collectives."""
    if COMM is None:
        raise RuntimeError("root_call() must run inside an MPI worker")
    result, error = None, None
    if RANK == 0:
        try:
            result = function(*args, **kwargs)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    result, error = COMM.bcast((result, error), root=0)
    if error is not None:
        raise RuntimeError(error)
    return result


# 单个 MPI 作业的状态机：加锁 → 只读规划/临时 setup 比对 → fresh/resume/post/plot。
# 恢复前先检查输入、参数、原生扩展及 checkpoint，不直接覆盖旧结果；fresh 才归档重开。
# fit_only 只做采样与完成性登记，跳过重建/绘图，并不表示计算便宜。
def run_band(src: Path, cfg: dict, fit: dict, response: str, action: str) -> None:
    if COMM is None or pymica is None:
        raise RuntimeError("run_band() must run inside an MPI worker")
    from mica_resume import (
        acquire_run_lock, apply_plan, compare_saved_setup,
        plan_run, record_complete, record_started,
    )
    from mica_resume_native import run_native_resume

    fit_only = bool(fit.get("fit_only", False))
    joint = bool(fit.get("_joint", False))
    if fit_only and action not in {"run", "fresh", "resume"}:
        raise ValueError("fit-only runs support run/fresh/resume, not downstream post/plot actions")
    if not fit_only:
        check_tf_plot_support(fit)
    source = str(cfg.get("source", src.name))
    rd = run_dir(src, fit, response)
    pdir = rel(src, fit["prepared_dir"])
    input_files = {
        "cont.txt": pdir / f"{source}_{fit['driver']}.txt",
        "line.txt": pdir / f"{source}_{response}.txt",
    }
    if joint:
        import run_mica_joint
        input_files = run_mica_joint.input_files(src, source, fit)
    ncomp = fit["ncomp"]
    log("\n" + "=" * 70)
    log(f"{source}  {fit['driver']} -> {response}  {ncomp} {fit['_tf_label']} component(s)  {action}")
    log(str(rd))
    log("=" * 70)

    previous_cwd = Path.cwd()
    lock_handle = None
    preview = None

    def lock_run():
        nonlocal lock_handle
        lock_handle = acquire_run_lock(rd)

    try:
        root_call(lock_run)
        data = root_call(lambda: {"set1": [np.loadtxt(path) for path in input_files.values()]})
        prior_text = None
        if joint:
            kwargs, prior_text = run_mica_joint.setup(fit, data)
        else:
            kwargs = setup_kwargs(fit, data, response)
        # 用于恢复的身份要包含联合响应顺序和明确先验文本，但数据数组通过输入哈希另行验证。
        settings = {key: value for key, value in kwargs.items() if key not in {"data", "data_file"}}
        if fit_only:
            settings["_fit_only"] = True
        if joint:
            settings["_response_bands"] = list(fit["responses"])
            settings["_parameter_prior_text"] = prior_text
        native_extension = importlib.import_module("pymica.pymica").__file__
        plan = root_call(
            plan_run, rd, action=action, mpi_size=COMM.Get_size(),
            mixed=fit["_mixed"], ncomp=ncomp,
            target_saves=int(fit["max_num_saves"]), input_files=input_files,
            settings=settings, native_extension=native_extension,
        )
        mode = plan["mode"]
        log(f"run decision: {mode}" + (f" ({plan['phase']})" if "phase" in plan else ""))
        if plan.get("notice"):
            log(plan["notice"])
        checkpoint = plan.get("checkpoint")
        if checkpoint:
            log(f"checkpoint: {checkpoint['path']} ({checkpoint['saves']} saved samples)")
            for rejected in checkpoint.get("rejected_newer", []):
                log(f"skipped checkpoint: {rejected['path']}: {rejected['reason']}")

        if kwargs.get("lag_prior") is not None:
            log(f"centroid lag_prior[{response}] = {kwargs['lag_prior']}")

        if mode == "fresh":
            def prepare_fresh():
                apply_plan(rd, plan)
                if joint:
                    (rd / "input").mkdir(parents=True, exist_ok=True)
                    for name, path in input_files.items():
                        shutil.copy2(path, rd / "input" / name)
                else:
                    copy_inputs(src, source, fit, response, rd)
                return plan
            plan = root_call(prepare_fresh)
            if plan.get("archive"):
                log(f"previous run archived: {plan['archive']}")
            os.chdir(rd)
            model = pymica.mmap() if fit["_mixed"] else pymica.gmodel()
            model.setup(**kwargs)
            if joint:
                run_mica_joint.register_priors(model, prior_text, COMM, RANK)
            else:
                set_ordered_gaussian_priors(model, fit, data, response)
        else:
            # setup() serializes inputs and parameters. Validate its output in a
            # shared preview directory before allowing any write to the old run.
            # setup 本身会写文件，先在共享临时目录运行，核对成功后才允许触及旧运行目录。
            preview = Path(root_call(
                tempfile.mkdtemp, prefix=f".{rd.name}.preview-", dir=str(rd.parent)
            ))
            os.chdir(preview)
            model = pymica.mmap() if fit["_mixed"] else pymica.gmodel()
            model.setup(**kwargs)
            if joint:
                run_mica_joint.register_priors(model, prior_text, COMM, RANK)
            else:
                set_ordered_gaussian_priors(model, fit, data, response)
            COMM.Barrier()
            os.chdir(rd)
            root_call(compare_saved_setup, rd, preview)
            if mode == "skip":
                log(f"already complete: {fit['max_num_saves']} saved samples; skipped")
                return

            def prepare_existing():
                apply_plan(rd, plan)
                if mode in {"resume", "post"}:
                    destination = rd / "param/param_input"
                    temporary = destination.with_name(destination.name + ".resume-tmp")
                    shutil.copy2(preview / "param/param_input", temporary)
                    os.replace(temporary, destination)
                return plan
            plan = root_call(prepare_existing)
            if plan.get("backup"):
                log(f"previous files and any rows after the checkpoint saved in: {plan['backup']}")
                if checkpoint:
                    log(f"sample and sample_info aligned to {checkpoint['saves']} saved rows before continuing")

        if mode in {"fresh", "resume"}:
            root_call(record_started, rd, plan)
            phase = "fresh" if mode == "fresh" else plan["phase"]
            log(f"MICA native run: {phase}")
            native_kwargs = {"mixed": fit["_mixed"], "phase": phase}
            if fit_only:
                native_kwargs["fit_only"] = True
            # 真正进入耗时原生采样的位置；阶段检查、锁和备份必须在此之前完成。
            result = run_native_resume(**native_kwargs)
            plan["continuum_prior"] = result["continuum_prior"]
            COMM.Barrier()
            if not fit_only:
                finish(model, fit, post_run=False)
            root_call(record_complete, rd, plan)
        elif mode == "post":
            if action in {"run", "resume"}:
                root_call(record_started, rd, plan)
            finish(model, fit, post_run=True)

            def complete_post():
                sample = rd / "data" / f"sample1d.txt_{ncomp}"
                with sample.open() as handle:
                    rows = sum(1 for line in handle if line.strip() and not line.lstrip().startswith("#"))
                if rows == int(fit["max_num_saves"]):
                    record_complete(rd, plan)
                else:
                    log(f"post-processing finished at {rows} saved samples; cumulative target is {fit['max_num_saves']}")
            root_call(complete_post)
        else:
            if not have_decomp(ncomp):
                model.decompose()
                COMM.Barrier()
            if RANK == 0:
                log("MICA plot_decomp()")
                plot_decomposition(model, fit)
            COMM.Barrier()
    finally:
        os.chdir(previous_cwd)
        if RANK == 0:
            if preview is not None:
                shutil.rmtree(preview, ignore_errors=True)
            if lock_handle is not None:
                lock_handle.close()


def batches(values: list[str], size: int):
    for i in range(0, len(values), size):
        yield values[i : i + size]


# 外层调度器按波段分批启动 mpiexec；每个作业内部再有 mpi_per_band 个 rank。
# 将 BLAS/OMP 线程限制为 1，避免进程并行叠加线程并行导致过量占用。
# 进程退出码 0 仍须配合 run_state 的 complete 与目标保存量检查，不能单独证明成功。
def launch(
    src: Path,
    fit: dict,
    responses: list[str],
    action: str,
    stage: str,
) -> None:
    script = Path(__file__).resolve()
    log_dir = src / fit["output_root"] / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # Codex/WSL sessions may inherit a Windows TEMP path, where MPI cannot
    # create Unix FIFOs.  Worker jobs must use a native Linux temp directory.
    for key in ("TEMP", "TMP"):
        if str(env.get(key, "")).startswith("/mnt/"):
            env.pop(key, None)
    env["TMPDIR"] = "/tmp"
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )

    for group in batches(responses, int(fit["max_parallel"])):
        jobs = []
        try:
            for response in group:
                cmd = [
                    str(compatible_mpiexec()), "-n", str(int(fit["mpi_per_band"])),
                    sys.executable, "-u", str(script), str(src), response,
                    action, "--stage", stage,
                ]
                if fit.get("_joint"):
                    cmd.append("--joint")
                if fit.get("fit_only"):
                    cmd.append("--fit-only")
                base = log_dir / f"{run_name(fit, response)}_{action}.log"
                attempt = 1
                while True:
                    log_path = base if attempt == 1 else base.with_name(f"{base.stem}_attempt{attempt:04d}.log")
                    try:
                        handle = log_path.open("x", encoding="utf-8")
                        break
                    except FileExistsError:
                        attempt += 1
                log("start: " + " ".join(cmd) + f" -> {log_path}")
                try:
                    # 每个 launcher 建独立进程组，异常清理只终止本次启动的子进程。
                    process = subprocess.Popen(
                        cmd, cwd=str(ROOT), stdout=handle,
                        stderr=subprocess.STDOUT, env=env, start_new_session=True,
                    )
                except BaseException:
                    handle.close()
                    raise
                jobs.append((response, process, handle, log_path))

            pending = list(jobs)
            while pending:
                for job in list(pending):
                    response, process, handle, log_path = job
                    code = process.poll()
                    if code is None:
                        continue
                    pending.remove(job)
                    handle.close()
                    if code == 0 and action in {"run", "resume", "fresh"}:
                        # Some native error paths use exit(0), bypassing Python's
                        # completion checks. Require the worker's committed state.
                        state_path = run_dir(src, fit, response) / "run_mica_state.json"
                        try:
                            state = json.loads(state_path.read_text())
                            target_saves = fit["max_num_saves_by_band"].get(
                                response, int(fit["max_num_saves"])
                            )
                            completed = (state.get("status") == "complete" and
                                         state.get("target_saves") == target_saves)
                        except (OSError, ValueError):
                            completed = False
                        if not completed:
                            log(f"FAIL: {response} exited without a completed run state -> {log_path}")
                            raise RuntimeError(f"MICA did not confirm completion for {response}")
                    log(("done: " if code == 0 else "FAIL: ") + f"{response} -> {log_path}")
                    if code:
                        raise SystemExit(code)
                if pending:
                    time.sleep(0.2)
        finally:
            # Each launched mpiexec has its own session. Clean up only these
            # groups, including workers whose launcher has already failed.
            signaled = []
            for _, process, _, _ in jobs:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    signaled.append(process.pid)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + 5.0
            while any(process.poll() is None for _, process, _, _ in jobs) and time.monotonic() < deadline:
                time.sleep(0.1)
            for group_id in signaled:
                try:
                    os.killpg(group_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for _, process, handle, _ in jobs:
                process.wait()
                handle.close()


# 把命令行映射为 source、band、action、stage；plotdecomp 是旧名称兼容。
def parse_args(words: list[str]) -> tuple[str, str | None, str, str | None]:
    source = words[0]
    rest = words[1:]
    stage = None
    if "--stage" in rest:
        index = rest.index("--stage")
        if index + 1 >= len(rest):
            raise ValueError("--stage requires a config section name")
        stage = rest[index + 1]
        del rest[index : index + 2]
    if any(word.startswith("--") for word in rest):
        raise ValueError(f"unknown option(s): {' '.join(rest)}")
    action = "run"
    if rest and rest[-1] in ACTIONS:
        action = rest.pop()
    if action == "plotdecomp":
        action = "plot"
    if len(rest) > 1:
        raise ValueError("at most one response band may be specified")
    band = rest[0] if rest else None
    return source, band, action, stage


# 统一 CLI：调度进程顺序遍历 stage，MPI worker 只接收一个 stage/波段。
# --joint 走多响应原型且隐含 --fit-only；普通入口则是逐波段独立响应拟合。
def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)

    words = list(sys.argv[1:])
    joint = "--joint" in words
    fit_only = joint or "--fit-only" in words
    words = [word for word in words if word not in {"--joint", "--fit-only"}]
    source, band, action, selected_stage = parse_args(words)
    if joint and (selected_stage is None or (band is not None and not (_MPI_WORKER and band == "joint"))):
        raise ValueError("--joint requires an explicit --stage and uses all configured response bands")
    src = source_dir(source)
    cfg = read_yaml(src / "config" / "source_config.yaml")
    stages = [selected_stage] if selected_stage else configured_stages(cfg)

    def execution_fit(stage):
        fit = fit_config(cfg, stage)
        if band is not None:
            fit["max_num_saves"] = fit["max_num_saves_by_band"].get(
                band, int(fit["max_num_saves"])
            )
        if fit_only:
            fit["fit_only"] = True
        if fit.get("fit_only") and action not in {"run", "fresh", "resume"}:
            raise ValueError("fit-only mode does not run postprocessing or plotting")
        if joint:
            from run_mica_joint import input_files as joint_input_files
            joint_input_files(src, str(cfg.get("source", src.name)), fit)
            fit["_joint"] = True
            fit["run_template"] = "joint_{driver}_to_" + "_".join(fit["responses"]) + "_{ncomp}comp_{type_tf}"
        return fit

    if COMM is None:
        def interrupted(signum, frame):
            raise SystemExit(128 + signum)
        previous_handlers = {signum: signal.signal(signum, interrupted)
                             for signum in (signal.SIGTERM, signal.SIGHUP)}
        try:
            for stage in stages:
                fit = execution_fit(stage)
                responses = ["joint"] if joint else ([band] if band else [str(x) for x in fit["responses"]])
                launch(src, fit, responses, action, stage)
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
    else:
        if len(stages) != 1:
            raise ValueError("an MPI worker must receive exactly one --stage")
        fit = execution_fit(stages[0])
        responses = ["joint"] if joint else ([band] if band else [str(x) for x in fit["responses"]])
        if len(responses) != 1:
            raise ValueError("an MPI worker must receive exactly one response band")
        run_band(src, cfg, fit, responses[0], action)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if COMM is not None:
            COMM.Abort(130)
        raise SystemExit(130)
    except BaseException:
        if COMM is not None:
            import traceback
            traceback.print_exc()
            COMM.Abort(1)
        raise
