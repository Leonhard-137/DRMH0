#!/usr/bin/env python3
"""Serial full-posterior reconstruction and SED fitting for Mrk817 free100."""
# 耗时生产入口：顺序处理固定 Mrk817 free100 的 host/raw 两组重建与 FFA/SED。
# 会启动 decompose_mica_branches 和两 MPI rank 的 run_ffa；学习注释时不要执行此文件。
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Mrk817"
BANDS = ("UVM2", "UVW1", "U", "B", "V")
STEPS = "prep,dered,fit,summ,fvg,sedfit"
sys.path.insert(0, str(ROOT / "src"))
from decompose_mica_branches import validate_run_outputs


def group_root(group: str) -> Path:
    return SOURCE / "runs/mica/gamma" / f"free100_{group}" / "2comp/result"


def sed_outputs(group: str) -> list[Path]:
    analysis = group_root(group) / "comp0/fflux/analysis"
    names = (
        "Mrk817_comp0_6band_components.csv",
        "Mrk817_comp0_6band_flux_posterior.txt",
        "Mrk817_comp0_sed_fit_summary.csv",
        "sed_model_posterior.txt",
        "Mrk817_comp0_sed_points.csv",
        "Mrk817_comp0_sed_models.png",
        "Mrk817_comp0_sed_models.pdf",
    )
    return [analysis / name for name in names]


def sed_complete(group: str) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in sed_outputs(group))


# 构造真实子进程命令：full 分支重建后进入 prep,dered,fit,summ,fvg,sedfit，包含耗时 fit。
def commands(group: str) -> tuple[list[str], list[str]]:
    config = str(SOURCE / "config" / f"free100_{group}_sed.yaml")
    decompose = [sys.executable, "-B", str(ROOT / "src/decompose_mica_branches.py"),
                 str(SOURCE), "--config", config, "--branch", "full", "--skip-complete"]
    ffa = [str(Path(sys.executable).with_name("mpiexec")), "-n", "2", sys.executable,
           "-B", str(ROOT / "src/run_ffa.py"), str(SOURCE), "--config", config,
           "--config-section", "fflux_comp0", "--posterior-branch", "full", "--steps", STEPS]
    return decompose, ffa


def controller_cmdline(pid: int) -> list[str] | None:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except FileNotFoundError:
        return None
    return [os.fsdecode(value) for value in raw.split(b"\0") if value] or None


def validate_host_controller(args: list[str], pid: int) -> None:
    if ("--worker" in args
            or not any(Path(value).name == "decompose_mica_branches.py" for value in args)
            or not any(Path(value).name == "free100_host_sed.yaml" for value in args)):
        raise RuntimeError(f"PID {pid} is not the expected free100 host decomposition controller")


# 等待指定 PID 前核对命令行身份，避免把无关进程误当预期重建控制器。
def wait_host_controller(pid: int) -> None:
    while True:
        args = controller_cmdline(pid)
        if args is None:
            print(f"Host decomposition controller PID {pid} has exited; checking outputs.", flush=True)
            return
        validate_host_controller(args, pid)
        print(f"Waiting for existing host decomposition controller PID {pid} (30 s).", flush=True)
        time.sleep(30)


def validate_decomposition(group: str) -> None:
    directory = group_root(group) / "branches/full/decomposition"
    for band in BANDS:
        run = directory / f"run_UVW2_to_{band}_2comp_gamma"
        validate_run_outputs(run, 2, require_plot=True)
    print(f"Validated {group} reconstruction: all five bands.", flush=True)


# 已有完整 SED 则跳过；存在非空但未完成的 FFA sample 则拒绝自动覆盖/恢复，需先检查其来源。
def run_group(group: str, wait_pid: int | None = None) -> None:
    if sed_complete(group):
        print(f"Skip {group}: all core SED outputs already exist.", flush=True)
        return
    sample = group_root(group) / "comp0/fflux/fit/sample.txt"
    if sample.is_file() and sample.stat().st_size > 0:
        raise RuntimeError(f"Incomplete {group} SED has existing FFA samples: {sample}; "
                           "inspect the existing fit before continuing; no overwrite or automatic resume")
    decompose, ffa = commands(group)
    if not Path(ffa[0]).is_file():
        raise FileNotFoundError(f"Missing mpiexec beside the active Python: {ffa[0]}")
    if wait_pid is not None:
        wait_host_controller(wait_pid)
    else:
        print(f"Reconstruct {group} full posterior, reusing valid outputs.", flush=True)
        subprocess.run(decompose, cwd=ROOT, check=True)
    validate_decomposition(group)
    print(f"Start {group} FFA/SED with two MPI ranks.", flush=True)
    subprocess.run(ffa, cwd=ROOT, check=True)
    if not sed_complete(group):
        missing = [str(path) for path in sed_outputs(group)
                   if not path.is_file() or path.stat().st_size == 0]
        raise RuntimeError(f"Missing {group} SED outputs after FFA: {missing}")
    print(f"Completed {group} FFA/SED.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-host-decomposition-pid", type=int)
    args = parser.parse_args()
    if args.wait_host_decomposition_pid is not None and args.wait_host_decomposition_pid < 1:
        parser.error("--wait-host-decomposition-pid must be positive")
    run_group("host", wait_pid=args.wait_host_decomposition_pid)
    run_group("raw")
    print("Both free100 host/raw SED pipelines completed.", flush=True)


if __name__ == "__main__":
    main()
