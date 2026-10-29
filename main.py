#!/usr/bin/env python3
# 本文件只负责流程调度：把源目录、配置分支和动作翻译成子进程命令。
# 科学模型分别实现在 src/；阅读时先看 main → execute，再进入实际计算模块。
# 注意 --stage 指 source_config.yaml 中的 MICA 方案名，不是 prep/ffa 等动作名。
"""Single-source FFA -> MICA -> native-comp0 products -> lag -> distance pipeline.

The optional ``diagnostics`` action builds publication-facing post-processing
products from completed sampler outputs; it does not rerun FFA or MICA. ICCF is
run only through the explicit ``iccf`` action. When ``all`` is combined with
``--stage``, the shared prep/FFA inputs are reused and that MICA stage plus all
of its downstream products (including diagnostics) are completed in one call.

The ``corner`` action reads completed posterior products and writes one corner
figure per response band directly into each selected stage's ``result`` folder.
It also runs automatically after collection in both default pipeline flows.

The ``transfer`` action writes both transfer-function representations for each
selected MICA stage: pointwise posterior quantiles and analytic kernels built
from the posterior-median parameters. Project-wide Hubble diagrams remain a
separate cross-source calculation and are dispatched with ``python main.py
hubble`` using ``joint_hubble_config.yaml`` (or an explicit ``--config``).
The frozen shared-SED source analysis is python main.py shared-hubble.
Its downstream four-case f_D / peculiar-velocity comparison is
python main.py hubble-compare; it reads saved luminosity distances and writes
Hubble diagrams and posterior distributions without rerunning FFA or MICA.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
# 默认完整单源链；ICCF 需单独请求。decompose 目前是提示性占位动作。
FLOW = (
    "prep", "ffa", "mica", "collect", "corner", "decompose", "transfer",
    "ffa_comp0", "lagfit", "distance",
)
# 只重跑所选 MICA 方案及后处理。这里的 diagnostics 最后会调用 transfer，
# 因此元组内虽未单列 transfer，仍会在 diagnostics 中生成两种传递函数图。
STAGE_FLOW = (
    "mica", "collect", "corner", "decompose", "ffa_comp0",
    "lagfit", "distance", "diagnostics",
)
OPTIONAL_ACTIONS = ("iccf", "diagnostics")
ACTIONS = {"all", "fresh", "clean", "ffa_branch", *FLOW, *OPTIONAL_ACTIONS}
STAGE_ACTIONS = {
    "all", "mica", "collect", "corner", "transfer", "ffa_comp0", "ffa_branch",
    "iccf", "diagnostics", "lagfit", "distance",
}


# 把相对源名（例如 Mrk142）解释为项目根目录下的源目录；绝对路径原样解析。
def source_dir(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


# 单源配置入口。返回整个 YAML 字典；各子程序还会按自己的 section 读取它。
def read_config(src: Path) -> dict:
    with (src / "config" / "source_config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


# 配置中的相对路径以源目录为基准；绝对路径不会被强制限制在源目录内。
def rel(src: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else src / path


# 所有外部程序从项目根目录启动，check=True 使子进程失败立即中断主流程。
# 这里只串联命令，不代表命令内没有 MPI 进程或多波段并行。
def run(command: list[str | Path]) -> None:
    command = [str(item) for item in command]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


# 复用当前 Python 解释器调用 src 脚本，避免主程序与子程序落入不同环境。
def python(script: str, src: Path, *args: str) -> None:
    run([sys.executable, SRC / script, src, *args])


# 优先选当前 Python 环境同目录下的 mpiexec；不存在时才从 PATH 查找。
def compatible_mpiexec() -> Path | str:
    candidate = Path(sys.executable).with_name("mpiexec")
    return candidate if candidate.exists() else "mpiexec"


# 读取默认要处理的 MICA 方案列表，并检查对应配置段确实存在。
# 缺省 mica_round1 是兼容旧配置；不同 stage 的先验和输出目录可以不同。
def mica_stage_names(cfg: dict) -> list[str]:
    stages = cfg.get("mica_stages")
    if stages is None:
        return ["mica_round1"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("mica_stages must be a non-empty list")
    names = [str(stage) for stage in stages]
    missing = [stage for stage in names if not (cfg.get(stage) or {})]
    if missing:
        raise ValueError("mica_stages references missing sections: " + ", ".join(missing))
    return names


# 决定后处理遍历哪些方案；显式 --stage 会把范围缩到一个配置段。
def analysis_stages(cfg: dict, selected_stage: str | None = None) -> list[str]:
    """MICA stages whose native comp0 products are analysed downstream."""
    if selected_stage is None:
        return mica_stage_names(cfg)
    selected_stage = str(selected_stage).strip()
    if not selected_stage or not (cfg.get(selected_stage) or {}):
        raise ValueError(f"unknown MICA config section: {selected_stage!r}")
    return [selected_stage]


# 兼容 ncomp 与旧版 number_component 两种写法，得到传递函数分量数。
def ncomp_from_config(fit: dict) -> int:
    if "ncomp" in fit:
        return int(fit["ncomp"])
    nc = fit.get("number_component", [2, 2])
    return int(nc[0] if isinstance(nc, list) else nc)


# 按配置计算运行根目录；数字 type_tf 与名称型 type_tf 的历史布局不同。
# 定位结果时应使用此规则，不能凭目录名自行拼接另一套路径。
def mica_run_root(src: Path, fit: dict) -> Path:
    root = rel(src, fit.get("output_root", "runs/mica"))
    model = str(fit.get("type_tf", "gaussian")).strip().lower()
    return root if model.isdigit() else root / f"{ncomp_from_config(fit)}comp"


# 实际删除输出，不是预览。默认删除总光 FFA 和 MICA 派生 result 目录；
# 只有 purge_mica_runs=True 才把相应 MICA 运行根目录也列入删除目标。
# 本函数接受配置给出的绝对路径，学习时先检查 targets，勿直接执行 clean/fresh。
def clean(src: Path, purge_mica_runs: bool = False) -> None:
    cfg = read_config(src)
    ff = cfg.get("fflux", {}) or {}
    iccf = cfg.get("iccf", {}) or {}
    lag = cfg.get("lambda_tau", {}) or {}
    hubble = cfg.get("hubble_distance", {}) or {}

    targets = [rel(src, ff.get("out", "fflux"))]
    for stage in mica_stage_names(cfg):
        mica = cfg.get(stage, {}) or {}
        run_root = mica_run_root(src, mica)
        output_root = rel(src, mica.get("output_root", "runs/mica"))
        if purge_mica_runs:
            targets.append(run_root)
        else:
            # Preserve expensive completed runs; remove only derived tables.
            targets.extend([run_root / "result", output_root / "result"])

    # Optional or legacy standalone outputs. The current lag/distance outputs
    # live inside the MICA run root and are removed with output_root above.
    for section in (iccf, lag, hubble):
        if section.get("output_dir"):
            targets.append(rel(src, section["output_dir"]))

    for path in dict.fromkeys(targets):
        if path.exists():
            print("remove", path)
            shutil.rmtree(path)

    invalid = src / "results" / "invalid_flux_points.txt"
    if invalid.exists():
        invalid.unlink()


# 同一后验做两种图：逐时间点的响应分位数，以及后验中位参数代入的解析核。
# 两者运算次序不同，曲线不必重合；都属于后处理而非再次采样。
def plot_transfer_figures(src: Path, stage: str) -> None:
    """Render the posterior-band and median-kernel transfer-function products."""
    python("plot_mica_transfer.py", src, "--stage", stage)
    python(
        "plot_mica_transfer.py", src,
        "--stage", stage,
        "--median-parameters",
    )


# 动作分派表。每个分支负责选择配置、组装参数、启动对应脚本。
# 总光 FFA 与原生 comp0 的第二次 FFA 是两套不同输入，不能互换其产物。
def execute(src: Path, action: str, selected_stage: str | None = None) -> None:
    if action == "prep":
        python("prep_data.py", src)

    elif action == "ffa":
        cfg = read_config(src)
        mpi_n = int((cfg.get("pipeline", {}) or {}).get("ffa_mpi", 2))
        ff = dict(cfg.get("fflux", {}) or {})
        configured_steps = ff.get("steps", "all")
        step_arg = (
            configured_steps
            if isinstance(configured_steps, str)
            else ",".join(str(step) for step in configured_steps)
        )
        run([compatible_mpiexec(), "-n", mpi_n, sys.executable, SRC / "run_ffa.py", src, "--steps", step_arg])

    elif action == "mica":
        args = () if selected_stage is None else ("--stage", selected_stage)
        python("run_mica.py", src, *args)

    elif action == "collect":
        cfg = read_config(src)
        for stage in analysis_stages(cfg, selected_stage):
            python("collect_mica_results.py", src, stage)

    elif action == "corner":
        cfg = read_config(src)
        for stage in analysis_stages(cfg, selected_stage):
            python("plot_mica_corner.py", src, "--stage", stage)

    # 原生 MICA 已输出各分量光变；这里不会调用 decompose_mica_branches.py。
    elif action == "decompose":
        print(
            "skip conditional decomposition: MICA already wrote the full-posterior "
            "native component curves used downstream"
        )

    elif action == "transfer":
        cfg = read_config(src)
        for stage in analysis_stages(cfg, selected_stage):
            plot_transfer_figures(src, stage)

    # ffa_branch 在这个入口是 ffa_comp0 的别名，传入的仍固定为 comp0。
    # 第二次 FFA 的输入来自原生分量重建，不是原始观测总光变。
    elif action in {"ffa_comp0", "ffa_branch"}:
        cfg = read_config(src)
        mpi_n = int((cfg.get("pipeline", {}) or {}).get("ffa_mpi", 2))
        ff_comp0 = dict(cfg.get("fflux_comp0", {}) or {})
        component_steps = ff_comp0.get(
            "steps",
            ff_comp0.get(
                "posterior_branch_steps",
                ["prep", "dered", "fit", "summ", "fvg", "distance_flux", "sedfit"],
            ),
        )
        step_arg = (
            component_steps
            if isinstance(component_steps, str)
            else ",".join(str(step) for step in component_steps)
        )
        for stage in analysis_stages(cfg, selected_stage):
            run([
                compatible_mpiexec(), "-n", mpi_n, sys.executable,
                SRC / "run_ffa.py", src,
                "--config-section", "fflux_comp0",
                "--mica-stage", stage,
                "--mica-component", "comp0",
                "--steps", step_arg,
            ])

    elif action == "iccf":
        cfg = read_config(src)
        for stage in analysis_stages(cfg, selected_stage):
            python("run_iccf.py", src, "--stage", stage)

    elif action == "diagnostics":
        cfg = read_config(src)
        for stage in analysis_stages(cfg, selected_stage):
            plot_transfer_figures(src, stage)
            python("results_diagnostics.py", src, "--stage", stage)
            python("sed_u_diagnostics.py", src, "--stage", stage)

    elif action == "lagfit":
        cfg = read_config(src)
        for stage in analysis_stages(cfg, selected_stage):
            python("fit_tau.py", src, "--stage", stage)

    elif action == "distance":
        cfg = read_config(src)
        for stage in analysis_stages(cfg, selected_stage):
            python("disk_distance.py", src, "--stage", stage)

    else:
        raise ValueError(action)


# 先处理跨源专用入口，再解析单源命令。
# 不带 --stage 的 all 从 prep 开始；带 --stage 的 all 复用已准备的输入。
# fresh 必须显式允许删除 MICA 运行，因其随后会从头重跑完整 FLOW。
def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # 读取已保存距离的下游对比入口，与会重新做受约束 FFA 的 shared-hubble 分开。
    if argv and argv[0] == "hubble-compare":
        # Four common-input f_D / peculiar-velocity cases; no upstream refitting.
        run([sys.executable, "-m", "src.shared_sed_hubble.comparison", *argv[1:]])
        return
    if argv and argv[0] == "shared-hubble":
        # Exact shared-SED FFA -> one distance per source -> fixed-f0 H0.
        run([sys.executable, "-m", "src.shared_sed_hubble", *argv[1:]])
        return
    if argv and argv[0] == "hubble":
        # This is intentionally project-level: the dedicated script reads the
        # configured set of independent AGN only after their distance products
        # exist, instead of rerunning the joint fit after every single source.
        run([sys.executable, SRC / "joint_hubble.py", *argv[1:]])
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("action", nargs="?", default="all", choices=sorted(ACTIONS))
    parser.add_argument(
        "--purge-mica-runs",
        action="store_true",
        help="allow clean/fresh to delete completed MICA run directories",
    )
    parser.add_argument(
        "--stage",
        help=(
            "run only one MICA config section for stage-aware actions; "
            "for example mica_mix220"
        ),
    )
    args = parser.parse_args(argv)

    src = source_dir(args.source)

    if args.stage and args.action not in STAGE_ACTIONS:
        parser.error(
            "--stage is valid only with: " + ", ".join(sorted(STAGE_ACTIONS))
        )

    if args.action == "clean":
        clean(src, purge_mica_runs=args.purge_mica_runs)
        return

    if args.action == "fresh":
        if not args.purge_mica_runs:
            parser.error("fresh reruns MICA and requires --purge-mica-runs")
        clean(src, purge_mica_runs=True)
        actions = FLOW
    elif args.action == "all":
        actions = STAGE_FLOW if args.stage else FLOW
    else:
        actions = (args.action,)

    for action in actions:
        print(f"\n== {action} ==", flush=True)
        execute(src, action, args.stage)


if __name__ == "__main__":
    main()
