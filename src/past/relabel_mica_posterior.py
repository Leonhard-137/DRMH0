#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml


def resolve_source_dir(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()

    root_dir = Path(__file__).resolve().parents[1]
    return (root_dir / path).resolve()


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_dir(source_dir: Path, cfg: dict, response: str, ncomp: int) -> Path:
    fit = cfg["mica_fit"]
    driver = fit["driver"]
    type_tf = fit["type_tf"]
    name = f"run_{driver}_to_{response}_{ncomp}comp_{type_tf}"
    return source_dir / fit["output_root"] / f"{ncomp}comp" / name


def load_sample(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, comments="#")
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def write_sample(path: Path, sample: np.ndarray) -> None:
    np.savetxt(
        path,
        sample,
        fmt="%.10e",
        header=str(sample.shape[0]),
        comments="# ",
    )


def read_param_names(path: Path) -> list[str]:
    names: list[str] = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        parts = s.split()
        idx = int(parts[0])
        name = parts[1]

        while len(names) <= idx:
            names.append("")
        names[idx] = name

    if not names:
        raise ValueError(f"no parameter names found in {path}")

    return names


def find_line_blocks(names: list[str], ncomp: int) -> list[list[tuple[int, int, int]]]:
    blocks: list[list[tuple[int, int, int]]] = []

    for i, name in enumerate(names):
        if name != "sys_err_line":
            continue

        triples = []
        ok = True

        for k in range(ncomp):
            amp = i + 1 + 3 * k
            cen = i + 2 + 3 * k
            sig = i + 3 + 3 * k

            if sig >= len(names):
                ok = False
                break

            expected = (
                f"{k}-th_component_amplitude",
                f"{k}-th_component_center",
                f"{k}-th_component_sigma",
            )

            if (names[amp], names[cen], names[sig]) != expected:
                ok = False
                break

            triples.append((amp, cen, sig))

        if ok:
            blocks.append(triples)

    if not blocks:
        raise ValueError("cannot find valid component blocks in para_names_line file")

    return blocks


def relabel_two_components(sample: np.ndarray, blocks: list[list[tuple[int, int, int]]]):
    out = sample.copy()
    nswap = 0

    for block in blocks:
        if len(block) != 2:
            raise ValueError("this script only supports two-component MICA posterior")

        cols0 = list(block[0])
        cols1 = list(block[1])
        cen0 = block[0][1]
        cen1 = block[1][1]

        rows = np.where(out[:, cen0] > out[:, cen1])[0]
        nswap += len(rows)

        if len(rows):
            tmp = out[rows[:, None], cols0].copy()
            out[rows[:, None], cols0] = out[rows[:, None], cols1]
            out[rows[:, None], cols1] = tmp

    return out, nswap


def process_one(source_dir: Path, cfg: dict, response: str, ncomp: int) -> str:
    if ncomp != 2:
        raise ValueError("label relabeling here is defined only for ncomp=2")

    rd = run_dir(source_dir, cfg, response, ncomp)
    data_dir = rd / "data"

    post = data_dir / f"posterior_sample1d.txt_{ncomp}"
    names_path = data_dir / f"para_names_line.txt_{ncomp}"

    if not post.exists():
        raise FileNotFoundError(post)
    if not names_path.exists():
        raise FileNotFoundError(names_path)

    raw_dir = data_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    raw_post = raw_dir / post.name
    if not raw_post.exists():
        shutil.copy2(post, raw_post)

    sample = load_sample(raw_post)
    names = read_param_names(names_path)

    if sample.shape[1] != len(names):
        raise ValueError(
            f"column mismatch in {data_dir}: "
            f"posterior has {sample.shape[1]} columns, "
            f"para_names has {len(names)}"
        )

    blocks = find_line_blocks(names, ncomp)
    relabeled, nswap = relabel_two_components(sample, blocks)
    write_sample(post, relabeled)

    b0 = blocks[0][0][1]
    b1 = blocks[0][1][1]

    report = data_dir / "label_reorder_report.txt"
    report.write_text(
        "\n".join(
            [
                "method = center_ascending",
                f"source_dir = {source_dir}",
                f"run_dir = {rd}",
                f"posterior = {post}",
                f"raw_backup = {raw_post}",
                f"n_samples = {sample.shape[0]}",
                f"n_line_blocks = {len(blocks)}",
                f"n_swapped_block_samples = {nswap}",
                f"swap_fraction_per_block = {nswap / (sample.shape[0] * len(blocks)):.8f}",
                f"tf0 = short_lag_component",
                f"tf1 = long_lag_component",
                f"tf0_center_median = {np.nanmedian(relabeled[:, b0]):.10g}",
                f"tf1_center_median = {np.nanmedian(relabeled[:, b1]):.10g}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return f"{response}: swapped {nswap}/{sample.shape[0] * len(blocks)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    args = ap.parse_args()

    source_dir = resolve_source_dir(args.source)
    cfg = read_yaml(source_dir / "config" / "source_config.yaml")
    fit = cfg["mica_fit"]

    n_low, n_high = fit["number_component"]

    rows = []
    for response in fit["responses"]:
        for ncomp in range(int(n_low), int(n_high) + 1):
            rows.append(process_one(source_dir, cfg, response, ncomp))

    print("\n".join(rows))


if __name__ == "__main__":
    main()