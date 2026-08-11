#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
MPI_N = "2"

ACTIONS = {
    "fresh",   # clean + all
    "all",     # prep + raw + hostsub + config + mica + collect + sed
    "clean",
    "prep",
    "raw",
    "hostsub",
    "config",
    "mica",
    "collect",
    "sed",
}


def source_dir(x: str | Path) -> Path:
    p = Path(x).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, cfg: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def py(script: str, src: Path, *args: str, mpi: bool = False) -> None:
    cmd = [sys.executable, str(ROOT / "src" / script), str(src), *args]
    if mpi:
        cmd = ["mpiexec", "-n", MPI_N, *cmd]
    run(cmd)


def clean(src: Path) -> None:
    for p in [
        src / "fflux_raw",
        src / "fflux",
        src / "runs" / "fprior_r1",
        src / "results" / "fprior_r1",
    ]:
        if p.exists():
            print("remove", p)
            shutil.rmtree(p)
    prior = src / "runs" / "mica_priors.csv"
    if prior.exists():
        print("remove", prior)
        prior.unlink()


def find_ffa_raw_sed(src: Path) -> Path:
    hits = sorted((src / "fflux_raw" / "analysis").glob("*_int_sed.csv"))
    if not hits:
        raise FileNotFoundError("missing fflux_raw/analysis/*_int_sed.csv; run raw first")
    return hits[-1]


def make_hostsub(src: Path) -> None:
    """Build the light curves that MICA must use: MW-dereddened and host-subtracted."""
    cfg = read_yaml(src / "config" / "source_config.yaml")
    source = str(cfg.get("source", src.name))
    bands = [str(b) for b in cfg.get("band_list", list((cfg.get("bands") or {}).keys()))]

    sed = pd.read_csv(find_ffa_raw_sed(src))
    band_col = "Filter" if "Filter" in sed.columns else "band"
    if "F_gal_mid" not in sed.columns:
        raise KeyError("raw FFA SED table must contain F_gal_mid")
    fgal = dict(zip(sed[band_col].astype(str), sed["F_gal_mid"].astype(float)))

    inp = src / "fflux_raw" / "dered"
    out = src / "fflux_raw" / "host_sub"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for b in bands:
        ip = inp / f"{source}_{b}.txt"
        if not ip.exists():
            raise FileNotFoundError(ip)
        a = np.loadtxt(ip, usecols=(0, 1, 2))
        if a.ndim == 1:
            a = a[None, :]
        if b not in fgal:
            raise KeyError(f"missing F_gal_mid for band {b} in {find_ffa_raw_sed(src)}")
        fg = float(fgal[b])
        y = a.copy()
        y[:, 1] = y[:, 1] - fg
        op = out / f"{source}_{b}.txt"
        np.savetxt(
            op,
            y,
            fmt="%.10e %.10e %.10e",
            header=f"time flux_host_sub flux_error; source={ip}; F_gal_mid={fg:.10e}",
            comments="# ",
        )
        rows.append({"band": b, "input": str(ip), "output": str(op), "F_gal_mid": fg, "n": len(y)})

    pd.DataFrame(rows).to_csv(out / "host_sub_manifest.csv", index=False)
    print("host-subtracted MICA inputs ->", out)


def patch_config(src: Path) -> None:
    """Make the config point to the corrected I/O chain."""
    path = src / "config" / "source_config.yaml"
    cfg = read_yaml(path)

    source = str(cfg.get("source", src.name))
    driver = str(cfg.get("driver_band", cfg.get("reference_band", "UVW2")))
    bands = [str(b) for b in cfg.get("band_list", list((cfg.get("bands") or {}).keys()))]
    responses = [b for b in bands if b != driver]
    waves = cfg.get("bands", {}) or {}

    cfg["source"] = source
    cfg["reference_band"] = driver
    cfg["driver_band"] = driver
    cfg["band_list"] = bands
    cfg["response_bands"] = responses

    old_fit = dict(cfg.get("mica_fit", {}) or {})
    m1 = dict(cfg.get("mica_round1", {}) or {})

    for key in ["lag_limit", "max_num_saves", "type_lag_prior", "lag_prior", "width_prior", "width_limit"]:
        if key not in m1 and key in old_fit:
            m1[key] = old_fit[key]
    m1.setdefault("driver", driver)
    m1.setdefault("responses", responses)
    m1["prepared_dir"] = "fflux_raw/host_sub"

    m1.setdefault("output_root", "runs/fprior_r1")
    m1.setdefault("type_tf", "gaussian")
    m1.setdefault("ncomp", 2)
    m1.setdefault("lag_limit", [-10, 80])
    m1.setdefault("type_lag_prior", 0)
    m1.setdefault("max_num_saves", 10000)

    cfg["mica_round1"] = m1

    ff = dict(cfg.get("fflux", {}) or {})
    mica = dict(ff.get("mica", {}) or {})
    mica.setdefault("root", m1.get("output_root", "runs/fprior_r1"))
    mica.setdefault("driver", driver)
    mica.setdefault("ncomp", int(m1.get("ncomp", 2)))
    mica.setdefault("type_tf", m1.get("type_tf", "gaussian"))
    mica["driver_input_dir"] = "fflux_raw/host_sub"
    ff.update({
        "out": "fflux",
        "steps": ["prep", "fit", "summ", "fvg", "nuc"],
        "component": {"name": "tf0", "index": 0},
        "bands": bands,
        "wavelengths": waves,
        "mica": mica,
    })
    ff.pop("components", None)
    cfg["fflux"] = ff

    write_yaml(path, cfg)
    print("patched", path)
    print("MICA input: fflux_raw/host_sub")
    print("FFA output: fflux")


def do_prep(src: Path) -> None:
    py("prep_data.py", src)


def do_raw(src: Path) -> None:
    py("ffa_raw.py", src, "fresh", mpi=True)


def do_mica(src: Path) -> None:
    py("run_mica.py", src)


def do_collect(src: Path) -> None:
    py("collect_mica_results.py", src, "r1")


def do_sed(src: Path) -> None:
    # fflux.steps already says prep,fit,summ,fvg,nuc; fit needs MPI.
    py("run_ffa.py", src, mpi=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Mrk817 f-prior MICA pipeline driver.")
    ap.add_argument("source", help="source directory, e.g. Mrk817")
    ap.add_argument("action", nargs="?", default="all", choices=sorted(ACTIONS))
    args = ap.parse_args()

    src = source_dir(args.source)

    if args.action == "clean":
        clean(src)
        return

    if args.action == "fresh":
        clean(src)
        for step in [do_prep, do_raw, make_hostsub, patch_config, do_mica, do_collect, do_sed]:
            step(src)
        return

    if args.action == "all":
        for step in [do_prep, do_raw, make_hostsub, patch_config, do_mica, do_collect, do_sed]:
            step(src)
        return

    if args.action == "prep":
        do_prep(src)
    elif args.action == "raw":
        do_raw(src)
    elif args.action == "hostsub":
        make_hostsub(src)
    elif args.action == "config":
        patch_config(src)
    elif args.action == "mica":
        do_mica(src)
    elif args.action == "collect":
        do_collect(src)
    elif args.action == "sed":
        do_sed(src)


if __name__ == "__main__":
    main()
