#!/usr/bin/env python3
from __future__ import annotations
import argparse, shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from mpi4py import MPI
from scipy.optimize import curve_fit

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

# Experiment defaults. Change here when a run design changes.
STAGE = "r1"
NCOMP = 2
DISK_COMPONENT = 0
DELAY_COMPONENT = 1
Q_LOW = 1.0e-4
Q_UPP = 0.2
F_MARGIN = 0.10
NSAMP = 3000
SEED = 12345
OUT = "fflux_raw"


def log(s=""):
    if RANK == 0:
        print(s, flush=True)


def read_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def rel(base: Path, p) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else base / p


def q68(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    p16, p50, p84 = np.percentile(x, [15.865, 50, 84.135])
    return float(p50), float(p50 - p16), float(p84 - p50)


def safe_div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.divide(a, b, out=np.full_like(a, np.nan), where=np.isfinite(a) & np.isfinite(b) & (b != 0))


def section(cfg: dict, name: str) -> dict:
    s = dict(cfg.get(name, {}) or {})
    s.setdefault("out", OUT)
    s.setdefault("prepared_dir", "prepared")
    s.setdefault("driver_band", cfg.get("driver_band", cfg.get("reference_band")))
    s.setdefault("bands", list((cfg.get("bands", {}) or {}).keys()))
    s.setdefault("wavelengths", cfg.get("bands", {}) or {})
    if "extinction" not in s and "galactic_extinction" in cfg:
        s["extinction"] = cfg["galactic_extinction"]
    return s


def deredden_inputs(src_dir: Path, source: str, sec: dict, bands: list[str], waves: dict) -> list[Path]:
    from ffa.ext import dered_one
    root = rel(src_dir, sec.get("out", OUT))
    out = root / "dered"
    out.mkdir(parents=True, exist_ok=True)
    prep = src_dir / sec.get("prepared_dir", "prepared")
    ext = sec.get("extinction", {}) or {}
    law = ext.get("law", "F99")
    ebv = float(ext.get("ebv", 0.0))
    rv = float(ext.get("rv", ext.get("Rv", 3.1)))
    paths = []
    for b in bands:
        ip = prep / f"{source}_{b}.txt"
        op = out / f"{source}_{b}.txt"
        paths.append(op)
        if RANK == 0:
            dered_one(ip, op, float(waves[b]), ebv, law, rv) if ebv > 0 else shutil.copy2(ip, op)
    COMM.Barrier()
    return paths


def read_fit(sample_dir: Path, reordered: list[str]):
    p = pd.read_csv(sample_dir / "flux_flux_params.txt", sep=r"\s+", comment="#", names=["idx", "label", "A", "A_err", "R"])
    A = {b: float(p.iloc[i].A) for i, b in enumerate(reordered)}
    Ae = {b: float(p.iloc[i].A_err) for i, b in enumerate(reordered)}
    R = {b: float(p.iloc[i].R) for i, b in enumerate(reordered)}
    x = np.loadtxt(sample_dir / "reconstructed_X.txt")
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return A, Ae, R, x[:, 1]


def get_fit(src_dir: Path, source: str, sec: dict, bands: list[str], waves: dict, reordered: list[str], fresh: bool):
    root = rel(src_dir, sec.get("out", OUT))
    sd = root / "fit"
    sd.mkdir(parents=True, exist_ok=True)
    have_fit = (sd / "posterior_sample.txt").exists() and (sd / "flux_flux_params.txt").exists()
    if have_fit and not fresh:
        if RANK != 0:
            return None
        log("Reusing existing FFA fit.")
        return read_fit(sd, reordered)

    paths = deredden_inputs(src_dir, source, sec, bands, waves)
    files = [p.name for p in paths]
    driver = reordered[0]
    files.insert(0, files.pop(bands.index(driver)))
    fit = sec.get("fit", {}) or {}
    from ffa.ffa import run_ffa
    log(f"Running FFA on {len(bands)} bands (driver={driver})")
    _, _, res = run_ffa(
        filenames=files, data_dir=str(root / "dered"), sample_dir=str(sd),
        time_shifts=fit.get("time_shifts"), tau_min=float(fit.get("tau_min", 100.0)),
        R_min=float(fit.get("r_min", 1.0e-6)), R_max=float(fit.get("r_max", 30.0)),
        thread_steps_factor=int(fit.get("thread_steps_factor", 2)), max_num_saves=int(fit.get("max_saves", 10000)),
    )
    if RANK != 0:
        return None
    A = {b: float(res["A"][i]) for i, b in enumerate(reordered)}
    Ae = {b: float(res["A_err"][i]) for i, b in enumerate(reordered)}
    R = {b: float(res["R"][i]) for i, b in enumerate(reordered)}
    return A, Ae, R, res["X_grid"]


def powerlaw(df: pd.DataFrame, redshift=0.0, beta0=-4 / 3) -> pd.DataFrame:
    x = df.wavelength.to_numpy(float) / (1 + redshift)
    y = x * df.Delta_F_int_mid.to_numpy(float)
    e = x * 0.5 * (df.Delta_F_int_err_low.to_numpy(float) + df.Delta_F_int_err_high.to_numpy(float))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(e) & (x > 0) & (y > 0) & (e > 0)
    x, y, e = x[ok], y[ok], e[ok]
    row = {"quantity": "lambda_rest * Delta_F_host_sub_no_nuc", "n_fit": int(len(x))}
    if len(x) < 3:
        row.update({"ok": False, "norm": np.nan, "norm_err": np.nan, "pivot": np.nan, "beta": np.nan, "beta_err": np.nan, "chi2": np.nan, "dof": 0, "chi2_dof": np.nan})
        return pd.DataFrame([row])
    pivot = float(np.median(x))
    def f(lam, log_norm, beta):
        return np.exp(log_norm) * (lam / pivot) ** beta
    popt, pcov = curve_fit(f, x, y, sigma=e, p0=[np.log(np.median(y)), beta0], absolute_sigma=True)
    chi2 = float(np.sum(((y - f(x, *popt)) / e) ** 2))
    dof = int(len(x) - 2)
    err = np.sqrt(np.diag(pcov))
    row.update({"ok": True, "norm": float(np.exp(popt[0])), "norm_err": float(np.exp(popt[0]) * err[0]), "pivot": pivot,
                "beta": float(popt[1]), "beta_err": float(err[1]), "chi2": chi2, "dof": dof, "chi2_dof": chi2 / dof if dof > 0 else np.nan})
    return pd.DataFrame([row])


def plot_sed(df, out: Path, source: str):
    x = df.wavelength.to_numpy(float)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(x, x * df.F_Faint_int_mid, x * df.F_Bright_int_mid, color="grey", alpha=0.2, label="AGN variability range")
    ax.errorbar(x, x * df.Delta_F_int_mid, yerr=[x * df.Delta_F_int_err_low, x * df.Delta_F_int_err_high], fmt="o-", color="black", capsize=3, label=r"Difference spectrum ($\Delta F_\lambda$)")
    mask = df.index != int(df.wavelength.idxmin())
    ax.errorbar(x[mask], x[mask] * df.loc[mask, "F_gal_mid"], yerr=[x[mask] * df.loc[mask, "F_gal_err_low"], x[mask] * df.loc[mask, "F_gal_err_high"]], fmt="s--", color="red", capsize=3, label="Host galaxy")
    ax.plot(x, x * df.R_int_mid, "o:", color="grey", label="AGN RMS")
    ax.set(xscale="log", yscale="log", xlabel=r"Observed wavelength ($\rm\AA$)", ylabel=r"$\lambda F_\lambda$ ($10^{-15}$ erg s$^{-1}$ cm$^{-2}$ $\rm\AA^{-1}$)", title=f"{source} SED: MW dereddened + host subtracted, no nuclear correction")
    ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(out, dpi=250); plt.close(fig)


def plot_pl(df, fit, out: Path, redshift=0.0):
    r = fit.iloc[0]
    if not bool(r.ok):
        return
    x = df.wavelength.to_numpy(float) / (1 + redshift)
    y = x * df.Delta_F_int_mid.to_numpy(float)
    elo = x * df.Delta_F_int_err_low.to_numpy(float)
    ehi = x * df.Delta_F_int_err_high.to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y, elo, ehi = x[ok], y[ok], elo[ok], ehi[ok]
    xx = np.logspace(np.log10(x.min() * 0.9), np.log10(x.max() * 1.1), 300)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(x, y, yerr=[elo, ehi], fmt="o", capsize=3, label="Data")
    ax.plot(xx, r.norm * (xx / r.pivot) ** r.beta, "-", label=f"fit beta={r.beta:.2f}+/-{r.beta_err:.2f}")
    i = np.argmin(np.abs(x - 5500.0)); beta_th = -4 / 3
    ax.plot(xx, y[i] * (xx / x[i]) ** beta_th, "--", label=f"theory beta={beta_th:.2f}")
    ax.set(xscale="log", yscale="log", xlabel="Rest wavelength (Angstrom)", ylabel="lambda * Delta F_lambda")
    ax.text(0.05, 0.05, f"chi2/dof = {r.chi2_dof:.2f}", transform=ax.transAxes, bbox={"facecolor": "white", "alpha": 0.75})
    ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(out, dpi=250); plt.close(fig)


def write_priors(src_dir: Path, stage: str, reordered: list[str], driver: str, ratios: dict):
    path = src_dir / "runs" / "mica_priors.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for b in reordered:
        if b == driver:
            continue
        F = float(ratios[b]); Flo, Fhi = F * (1 - F_MARGIN), F * (1 + F_MARGIN)
        fd_lo, fd_hi = Flo / (1 + Q_UPP), Fhi / (1 + Q_LOW)
        fl_lo, fl_hi = Q_LOW * Flo / (1 + Q_LOW), Q_UPP * Fhi / (1 + Q_UPP)
        rows += [[stage, b, NCOMP, f"{DISK_COMPONENT}-th_component_amplitude", "LOG", np.log(fd_lo), np.log(fd_hi), fd_lo, fd_hi],
                 [stage, b, NCOMP, f"{DELAY_COMPONENT}-th_component_amplitude", "LOG", np.log(fl_lo), np.log(fl_hi), fl_lo, fl_hi]]
    new = pd.DataFrame(rows, columns=["stage", "response", "ncomp", "param_name", "prior_space", "prior_min", "prior_max", "physical_min", "physical_max"])
    if path.exists():
        old = pd.read_csv(path)
        key = ["stage", "response", "ncomp", "param_name"]
        old_key = old[key].astype(str).agg("|".join, axis=1)
        new_key = new[key].astype(str).agg("|".join, axis=1)
        new = pd.concat([old.loc[~old_key.isin(set(new_key))], new], ignore_index=True)
    new.to_csv(path, index=False, float_format="%.10g")
    log(f"Saved {path}")


def run(source_dir: Path, cfg: dict, section_name: str, fresh: bool):
    source, z = str(cfg.get("source", source_dir.name)), float(cfg.get("redshift", 0.0))
    sec = section(cfg, section_name)
    bands = [str(b) for b in sec["bands"]]
    waves = {str(k): float(v) for k, v in sec["wavelengths"].items()}
    driver = str(sec.get("driver_band") or bands[0])
    reordered = bands[:]; reordered.insert(0, reordered.pop(reordered.index(driver)))
    root = rel(source_dir, sec.get("out", OUT)); sd, ad = root / "fit", root / "analysis"
    ad.mkdir(parents=True, exist_ok=True)

    fit_result = get_fit(source_dir, source, sec, bands, waves, reordered, fresh)
    if RANK != 0:
        return
    A_pt, A_err, _, X = fit_result
    post = np.loadtxt(sd / "posterior_sample.txt")
    if post.ndim == 1:
        post = post.reshape(1, -1)
    Rpost = {b: np.exp(post[:, 1 + i]) for i, b in enumerate(reordered)}
    rng = np.random.default_rng(int(sec.get("seed", SEED)))
    ns = min(int(sec.get("nsamp", NSAMP)), len(post))
    idx = rng.choice(len(post), ns, replace=ns > len(post))
    xf, xb = float(np.min(X)), float(np.max(X))
    As = {b: rng.normal(A_pt[b], max(A_err[b], 1e-30), ns) for b in reordered}
    X0 = {b: -As[b] / Rpost[b][idx] for b in reordered}
    Xgal = np.max(np.vstack([X0[b] for b in reordered]), axis=0)
    sed_rows, par_rows, ratios = [], [], {}
    ratio_post = pd.DataFrame({"sample": np.arange(len(post))})

    for b in reordered:
        A, R = As[b], Rpost[b][idx]
        Fgal, Ff, Fb = A + R * Xgal, A + R * xf, A + R * xb
        dF, Ffi, Fbi = Fb - Ff, Ff - Fgal, Fb - Fgal
        row = {"source": source, "Filter": b, "band": b, "driver": b == driver, "Wavelength": waves[b], "wavelength": waves[b],
               "Wave_Rest": waves[b] / (1 + z), "A_mid": A_pt[b], "A_err": A_err[b], "note": "MW dereddened + host subtracted; no nuclear extinction"}
        for name, arr in [("F_gal", Fgal), ("F_faint", Ff), ("F_bright", Fb), ("Delta_F", dF), ("F_Faint_int", Ffi), ("F_Bright_int", Fbi), ("Delta_F_int", dF), ("R_int", R), ("Epsilon_int", safe_div(Ffi, Fbi))]:
            m, lo, hi = q68(arr); row[f"{name}_mid"], row[f"{name}_err_low"], row[f"{name}_err_high"] = m, lo, hi
        sed_rows.append(row)
        rr = safe_div(Rpost[b], Rpost[driver]); ar = safe_div(As[b], As[driver])
        ratio_post[f"R_ratio_{b}"] = rr
        rrm, rrlo, rrhi = q68(rr); arm, arlo, arhi = q68(ar); Rm, Rlo, Rhi = q68(Rpost[b])
        ratios[b] = rrm
        par_rows.append({"source": source, "band": b, "driver": b == driver, "F_mid": rrm, "F_err_low": rrlo, "F_err_high": rrhi,
                         "F_definition": "R_i/R_driver_no_nuclear_correction", "R_mid": Rm, "R_err_low": Rlo, "R_err_high": Rhi,
                         "A_ratio_mid": arm, "A_ratio_err_low": arlo, "A_ratio_err_high": arhi})

    tag = f"{source}_ffa_raw_nonuc"
    sed, params = pd.DataFrame(sed_rows), pd.DataFrame(par_rows)
    fit = powerlaw(sed, z)
    sed.to_csv(ad / f"{tag}_int_sed.csv", index=False, float_format="%.6g")
    params.to_csv(ad / f"{tag}_params.csv", index=False, float_format="%.6g")
    ratio_post.to_csv(ad / f"{tag}_R_ratio_posterior.csv", index=False, float_format="%.10g")
    fit.to_csv(ad / f"{tag}_pl_fit.csv", index=False, float_format="%.6g")
    plot_sed(sed, ad / f"{tag}_sed.png", source); plot_pl(sed, fit, ad / f"{tag}_pl.png", z)
    write_priors(source_dir, STAGE, reordered, driver, ratios)
    log(f"Saved {ad / f'{tag}_int_sed.csv'}")
    if bool(fit.iloc[0].ok):
        r = fit.iloc[0]; log(f"Power-law beta = {r.beta:.4f} +/- {r.beta_err:.4f}; chi2/dof = {r.chi2_dof:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source_dir", type=Path)
    p.add_argument("mode", nargs="?", default="auto", choices=["auto", "fresh"])
    p.add_argument("--config", type=Path)
    p.add_argument("--config-section", default="ffa_raw")
    a = p.parse_args()
    src = a.source_dir.expanduser().resolve()
    cfg_path = a.config or src / "config/source_config.yaml"
    if not cfg_path.is_absolute():
        cfg_path = src / cfg_path
    run(src, read_yaml(cfg_path), a.config_section, fresh=a.mode == "fresh")


if __name__ == "__main__":
    main()
