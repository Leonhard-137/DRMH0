#!/usr/bin/env python3
"""ICCF cross-check for the MICA tf0 reconstructed light curves."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pyat.ccf import iccf, iccf_mc

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, *args, **kwargs): pass
        def update(self, n=1): pass
        def close(self): pass


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "output_dir": "results/iccf",
    "tau_range": [-10.0, 10.0],
    "ntau": 241,
    "threshold": 0.8,
    "nsim": 10000,
    "chunk": 500,
    "mode": "multiple",
    "max_retries": 5,
    "seed": 12345,
}

TF_LABELS = {
    "gaussian": "gaussian",
    "tophat": "tophat",
    "gamma": "gamma",
    "exponential": "exp",
    "exp": "exp",
}


def source_dir(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def rel(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def read_config(src: Path) -> tuple[dict, dict, dict]:
    with (src / "config" / "source_config.yaml").open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    fit = dict(cfg.get("mica_round1", {}) or {})
    fit.setdefault("output_root", "runs/mica")
    fit.setdefault("driver", cfg.get("reference_band", cfg.get("driver_band")))
    fit.setdefault("responses", cfg.get("response_bands", []))

    opt = dict(DEFAULTS)
    opt.update(cfg.get("iccf", {}) or {})
    return cfg, fit, opt


def ncomp_from_config(fit: dict) -> int:
    if "ncomp" in fit:
        return int(fit["ncomp"])
    nc = fit.get("number_component", [2, 2])
    return int(nc[0] if isinstance(nc, list) else nc)


def model_label(fit: dict) -> str:
    model = str(fit.get("type_tf", "gaussian")).lower()
    if model not in TF_LABELS:
        raise ValueError(f"unsupported MICA transfer-function type: {model}")
    return TF_LABELS[model]


def run_dir(src: Path, fit: dict, response: str) -> Path:
    ncomp = ncomp_from_config(fit)
    model = model_label(fit)
    return (
        rel(src, fit["output_root"])
        / f"{ncomp}comp"
        / f"run_{fit['driver']}_to_{response}_{ncomp}comp_{model}"
    )


def counts(path: Path) -> list[int]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("#"):
            break
        values = [int(x) for x in line.replace("#", " ").replace(":", " ").split() if x.isdigit()]
        if len(values) >= 2:
            rows = values[:2]
            break
    return rows


def load_series(path: Path, band: str, segment: int, count_file: Path | None = None) -> dict:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    data = data[:, :3]

    ns = counts(count_file or path)
    if ns:
        start = sum(ns[:segment])
        data = data[start : start + ns[segment]]

    good = np.isfinite(data).all(axis=1) & (data[:, 2] > 0)
    data = data[good][np.argsort(data[good, 0])]
    return {"band": band, "t": data[:, 0], "f": data[:, 1], "e": data[:, 2], "path": path}


def load_driver(rd: Path, driver: str, ncomp: int) -> dict:
    return load_series(rd / "data" / f"pall.txt_{ncomp}", driver, 0)


def load_tf0(rd: Path, band: str, ncomp: int) -> dict:
    pall = rd / "data" / f"pall.txt_{ncomp}"
    return load_series(rd / "data" / f"pline.txt_{ncomp}_comp0", band, 1, pall)


def run_pair(a: dict, b: dict, opt: dict, bar=None) -> dict:
    tau_beg, tau_end = map(float, opt["tau_range"])
    ntau = int(opt["ntau"])
    threshold = float(opt["threshold"])
    mode = str(opt["mode"])
    retries = int(opt["max_retries"])

    tau, curve, rmax, peak, cent = iccf(
        a["t"], a["f"], b["t"], b["f"], ntau, tau_beg, tau_end,
        threshold=threshold, mode=mode, ignore_warning=True, max_retries=retries,
    )

    r_mc, peak_mc, cent_mc = [], [], []
    nsim, chunk = int(opt["nsim"]), int(opt["chunk"])
    for done in range(0, nsim, chunk):
        n = min(chunk, nsim - done)
        r, p, c = iccf_mc(
            a["t"], a["f"], a["e"], b["t"], b["f"], b["e"],
            ntau, tau_beg, tau_end, nsim=n, threshold=threshold,
            mode=mode, ignore_warning=True, max_retries=retries,
        )
        r_mc.append(r)
        peak_mc.append(p)
        cent_mc.append(c)
        if bar is not None:
            bar.update(n)

    cat = lambda values: np.concatenate(values) if values else np.array([])
    return {
        "tau": tau,
        "curve": curve,
        "rmax": float(rmax),
        "peak": float(peak),
        "cent": float(cent),
        "r_mc": cat(r_mc),
        "peak_mc": cat(peak_mc),
        "cent_mc": cat(cent_mc),
    }


def quantiles(values, fallback: float) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return fallback, fallback, fallback
    return tuple(np.percentile(values, [16.0, 50.0, 84.0]))


def save_pair(out: Path, source: str, band: str, kind: str, result: dict) -> None:
    pd.DataFrame({"tau_days": result["tau"], "r": result["curve"]}).to_csv(
        out / f"{source}_{band}_{kind}_curve.csv", index=False
    )
    pd.DataFrame({
        "r_peak": result["r_mc"],
        "tau_peak_days": result["peak_mc"],
        "tau_cent_days": result["cent_mc"],
    }).to_csv(out / f"{source}_{band}_{kind}_mc.csv", index=False)


def summary_row(band: str, kind: str, result: dict) -> dict:
    c16, c50, c84 = quantiles(result["cent_mc"], result["cent"])
    p16, p50, p84 = quantiles(result["peak_mc"], result["peak"])
    return {
        "band": band,
        "kind": kind,
        "tau_cent": c50,
        "tau_cent_err_low": c50 - c16,
        "tau_cent_err_high": c84 - c50,
        "tau_peak": p50,
        "tau_peak_err_low": p50 - p16,
        "tau_peak_err_high": p84 - p50,
        "rmax": result["rmax"],
        "n_mc": len(result["cent_mc"]),
    }


def mica_summary(src: Path, fit: dict) -> pd.DataFrame:
    path = (
        rel(src, fit["output_root"])
        / f"{ncomp_from_config(fit)}comp"
        / "result"
        / "mica_component_summary.csv"
    )
    table = pd.read_csv(path)
    table = table[table["component"] == "tf0"].copy()
    return table[["band", "tau", "tau_err_low", "tau_err_high"]].rename(columns={
        "tau": "mica_tau",
        "tau_err_low": "mica_tau_err_low",
        "tau_err_high": "mica_tau_err_high",
    })


def plot_results(out: Path, source: str, rows: list[dict], curves: list[tuple[str, dict]]) -> None:
    fig, axes = plt.subplots(len(curves), 1, figsize=(7, 2.4 * len(curves)), squeeze=False)
    for ax, (band, result) in zip(axes[:, 0], curves):
        ax.plot(result["tau"], result["curve"], color="black")
        q16, q50, q84 = quantiles(result["cent_mc"], result["cent"])
        ax.axvline(q50, color="red", ls="--")
        ax.axvspan(q16, q84, color="0.85")
        ax.set_ylabel(f"{band} r")
    axes[-1, 0].set_xlabel("Lag (days)")
    fig.tight_layout()
    fig.savefig(out / f"{source}_tf0_iccf.png", dpi=250)
    fig.savefig(out / f"{source}_tf0_iccf.pdf")
    plt.close(fig)


def run(value: str | Path) -> None:
    src = source_dir(value)
    cfg, fit, opt = read_config(src)
    source = str(cfg.get("source", src.name))
    driver = str(fit["driver"])
    ncomp = ncomp_from_config(fit)
    responses = [str(x) for x in fit.get("responses", []) if str(x) != driver]
    out = rel(src, opt["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    np.random.seed(int(opt["seed"]))
    bar = tqdm(total=(len(responses) + 1) * int(opt["nsim"]), desc="ICCF MC", unit="sim")
    rows, curves = [], []

    try:
        first = run_dir(src, fit, responses[0])
        driver_lc = load_driver(first, driver, ncomp)
        acf = run_pair(driver_lc, driver_lc, opt, bar)
        save_pair(out, source, driver, "acf", acf)
        rows.append(summary_row(driver, "acf", acf))
        curves.append((driver, acf))

        for band in responses:
            rd = run_dir(src, fit, band)
            result = run_pair(
                load_driver(rd, driver, ncomp),
                load_tf0(rd, band, ncomp),
                opt,
                bar,
            )
            save_pair(out, source, band, "ccf", result)
            rows.append(summary_row(band, "ccf", result))
            curves.append((band, result))
    finally:
        bar.close()

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "iccf_summary.csv", index=False)

    compare = summary[summary["kind"] == "ccf"].merge(mica_summary(src, fit), on="band", how="left")
    compare["iccf_minus_mica"] = compare["tau_cent"] - compare["mica_tau"]
    compare.to_csv(out / "iccf_vs_mica.csv", index=False)
    plot_results(out, source, rows, curves)

    print(out / "iccf_summary.csv")
    print(out / "iccf_vs_mica.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    run(parser.parse_args().source_dir)


if __name__ == "__main__":
    main()
