#!/usr/bin/env python3
"""Convert raw/<source>.csv to per-band prepared/<source>_<band>.txt for MICA."""
import argparse
from pathlib import Path

import pandas as pd


# 学习入口：这里只做表格拆分和格式转换；科学拟合从后续 FFA/MICA 入口开始。
# 输入契约是 raw/<源名>.csv，列名必须包含 Filter、MJD、Flux、Error。
def prepare(source_dir: Path) -> None:
    source = source_dir.name
    raw_csv = source_dir / "raw" / f"{source}.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(f"missing {raw_csv}")

    df = pd.read_csv(raw_csv)
    prepared = source_dir / "prepared"
    prepared.mkdir(exist_ok=True)

    rows = []
    # 按滤光片分文件，保留原始时间、流量与误差的单位及参考系；此处没有消光或红移修正。
    for band, grp in df.groupby("Filter"):
        fname = f"{source}_{band}.txt"
        path = prepared / fname
        # 输出按 MJD/Flux/Error 排列，供下游的三列光变读取器使用。
        # groupby 不会替每组按时间排序；重复时刻、非有限值和误差符号也未在这里检查。
        grp[["MJD", "Flux", "Error"]].to_csv(
            path, sep=" ", index=False, header=False, float_format="%.6f"
        )
        # summary 是拆分清单，方便逐波段核对点数与覆盖时间，不代表数据质量已经合格。
        rows.append({
            "Object": source,
            "Filter": band,
            "N": len(grp),
            "MJD_min": grp["MJD"].min(),
            "MJD_max": grp["MJD"].max(),
            "File": str(path),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(prepared / "summary.csv", index=False)
    print(f"prepared {len(rows)} bands → {prepared}")
    for _, r in summary.iterrows():
        print(f"  {r['Filter']:4s}  {r['N']:4d} pts  {r['MJD_min']:.3f} – {r['MJD_max']:.3f}")


# 命令行相对路径以项目根目录为基准，而不是用户启动命令时所在目录。
def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw CSV to MICA prepared txt files.")
    parser.add_argument("source", type=Path, help="source directory, e.g. Mrk509")
    args = parser.parse_args()
    path = args.source.expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[1] / path).resolve()
    prepare(path)


if __name__ == "__main__":
    main()
