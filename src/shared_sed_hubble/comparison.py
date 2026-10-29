# 文件调度入口：冻结距离摘要 + 外部红移快照 → h0.py 四情景 → 保存密度 → 绘图。
# 不重算原距离中的 (1+z_saved)，只改变宇宙学预测端，范围写入 provenance。
"""Compare f_D and peculiar velocities using frozen source distances.

Canonical entry: python main.py hubble-compare [--step all|inputs|fit|plot|validate].
The legacy shared-hubble source fitter remains a separate, compatible workflow.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from .config import load_comparison_config


# 保存配置、来源与运行状态，帮助复查使用的是哪批冻结输入。
def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")


# 按 source 一对一连接两表，保留 z_saved；拒绝重复源和歧义重名列。
def assemble_inputs(config):
    """Join one saved luminosity distance and one external redshift per AGN."""
    import numpy as np
    import pandas as pd
    from .redshift import load_redshift_inputs

    distances = pd.read_csv(config.distance_summary)
    required = {"source", "z", "D_mean", "D_std"}
    if not required.issubset(distances.columns):
        raise ValueError(f"Distance table missing: {sorted(required - set(distances.columns))}")
    if distances["source"].duplicated().any():
        raise ValueError("Distance table contains duplicate source identifiers")
    missing = set(config.sources) - set(distances["source"])
    if missing:
        raise ValueError(f"Distance table missing sources: {sorted(missing)}")
    distances = distances.set_index("source").loc[list(config.sources)].reset_index()
    distances = distances.rename(columns={"z": "z_saved"})
    numbers = distances[["z_saved", "D_mean", "D_std"]].to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or (numbers <= 0).any():
        raise ValueError("Saved redshifts, distances and distance errors must be finite and positive")
    redshifts = load_redshift_inputs(config.redshift_inputs, sources=config.sources)
    overlap = (set(distances) & set(redshifts)) - {"source"}
    if overlap:
        raise ValueError(f"Ambiguous shared columns in input tables: {sorted(overlap)}")
    inputs = distances.merge(redshifts, on="source", how="left", validate="one_to_one")
    if len(inputs) != len(config.sources):
        raise ValueError("The merged table must contain exactly one row per selected source")
    inputs["distance_definition"] = "saved_D_L_at_z_saved; frozen"
    return inputs


# 保存输入路径、内容摘要、顺序及未包含的系统误差，不修改科学输入。
def save_input_provenance(config, inputs):
    paths = (config.distance_summary, config.redshift_inputs)
    write_json(config.output_dir / "input_provenance.json", {
        "inputs": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in paths],
        "source_order": list(config.sources),
        "distance_policy": "Frozen saved luminosity-distance means and sample standard deviations",
        "redshift_policy": "z_saved records distance construction; z_cmb is used in all four cases",
        "pv_policy": "C and D marginalize flow mean plus independent Gaussian residuals",
        "residual_sigma_kms": dict(zip(inputs.source, inputs.sigma_v_kms.astype(float))),
        "limitations": [
            "Conditional on saved distance products and Gaussian distance summaries",
            "No re-estimation of lags, SEDs, fluxes, or luminosity-distance conversion",
            "No Doppler beaming corrections or coherent cross-source velocity covariance",
        ],
    })


# 加输出锁后串接 inputs/fit/plot/validate；fit 重做 H0 网格，不启动源级 FFA/MICA。
def execute(config, step):
    from .__main__ import output_lock

    out = config.output_dir
    if step in {"plot", "validate"} and not (out / "run_summary.json").is_file():
        raise FileNotFoundError(f"No fitted comparison exists in {out}")
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out / ".controller.lock"):
        state = {"state": "running", "step": step, "pid": os.getpid(),
                 "started_unix": time.time()}
        write_json(out / "comparison_status.json", state)
        try:
            if step in {"all", "inputs", "fit"}:
                inputs = assemble_inputs(config)
                write_json(out / "comparison_config.json", config.to_dict())
                save_input_provenance(config, inputs)
                if step == "inputs":
                    inputs.to_csv(out / "common_inputs.csv", index=False)
                    print(f"Validated {len(inputs)} source inputs: {out / 'common_inputs.csv'}",
                          flush=True)
                else:
                    from .h0 import run_comparison
                    result = run_comparison(
                        inputs, out, settings=config.settings,
                        baseline_saved_z=config.baseline_saved_z, validate=True,
                    )
                    write_json(out / "validation.json", result["validation"])
                    for row in result["comparison"]:
                        print(f"{row['case']}: H0 = {row['h0_q50']:.4f} "
                              f"[{row['h0_q16']:.4f}, {row['h0_q84']:.4f}], "
                              f"f_D = {row['scatter_q50']:.4f}", flush=True)
                    print(f"Validation: {result['validation']['state']}", flush=True)
            if step == "validate":
                from .h0 import validate_saved_comparison
                validation = validate_saved_comparison(out, check_quadrature=True)
                write_json(out / "validation.json", validation)
                print(json.dumps(validation, indent=2, default=str), flush=True)
                if validation.get("passed") is False or validation.get("ok") is False:
                    raise RuntimeError("Saved comparison failed validation")
            if step in {"all", "plot"}:
                from .comparison_plotting import make_comparison_plots
                manifest = make_comparison_plots(out, include_joint=config.include_joint)
                print(json.dumps(manifest, indent=2, default=str), flush=True)
        except Exception as error:
            state.update(state="failed", finished_unix=time.time(), error=str(error))
            write_json(out / "comparison_status.json", state)
            raise
        state.update(state="complete", finished_unix=time.time())
        write_json(out / "comparison_status.json", state)
    print(f"Comparison {step} complete: {out}", flush=True)


# 读取比较专用 YAML 并设置线程/绘图缓存；它与源级 shared-hubble 配置不同。
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Comparison YAML (not the source-fit YAML)")
    parser.add_argument("--root", type=Path, help="Base directory for configured paths")
    parser.add_argument("--output", type=Path, help="Separate comparison result directory")
    parser.add_argument("--step", choices=("all", "inputs", "fit", "plot", "validate"),
                        default="all")
    parser.add_argument("--include-joint", action="store_true",
                        help="Also plot the B/D joint H0 and f_D posteriors")
    args = parser.parse_args(argv)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    config = load_comparison_config(args.config, root=args.root, output=args.output)
    if args.include_joint:
        from dataclasses import replace
        config = replace(config, include_joint=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config.output_dir / ".mplconfig"))
    execute(config, args.step)


if __name__ == "__main__":
    main()
