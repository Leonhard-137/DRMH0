#!/usr/bin/env python3
"""Prepare the eight F9 M2/B slow-signal fits; never import or run MICA.

Chart contract: compare source and timescale at fixed F9 observations. Static
line panels show the common disk signal and paired slow additions, with all
observations retained. Blue/solid=response, orange/dashed=independent; grey
marks the common disk contribution. Mother-series PSD/ACF establish the slow
timescales independently of a fitted transfer function.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.signal import fftconvolve, lfilter, periodogram, correlate
from scipy.special import gammainc
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_disk_kernels import build_kernels

DT = 0.02
N_MOTHER = 2**21
MOTHER_START = -20000.0
SEED = 2026090401
DRW_TAU = 10.0
DRW_STD = 0.20
SLOW_RMS_RATIO = 0.30
WIDTHS = {"within": 30.0, "beyond": 400.0}
COLORS = {"response": "#2457A6", "random": "#C67C18", "disk": "#666666"}
BANDS = ("M2", "B")


def slow_psd(frequency: np.ndarray, width: float) -> np.ndarray:
    """Declared marginal PSD; no light curve or transfer array is an input."""
    return 1 / ((1 + (2 * np.pi * frequency * DRW_TAU)**2)
                * (1 + (2 * np.pi * frequency * width)**2)**2)


def independent_slow(n: int, dt: float, width: float, rng) -> np.ndarray:
    """Draw independent Gaussian Fourier amplitudes and phases from the PSD."""
    f = np.fft.rfftfreq(n, dt)
    z = rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))
    z *= np.sqrt(slow_psd(f, width) / 2)
    z[0] = 0
    if n % 2 == 0:
        z[-1] = rng.normal() * np.sqrt(slow_psd(f[-1:], width)[0])
    signal = np.fft.irfft(z, n=n)
    return signal / np.std(signal)


def first_efold(acf, dt):
    crossings = np.flatnonzero(acf <= np.exp(-1))
    if not len(crossings):
        return None
    i = int(crossings[0])
    if i == 0:
        return 0.0
    return float(np.interp(np.exp(-1), acf[i - 1:i + 1][::-1],
                           np.array([i, i - 1]) * dt))


def signal_metrics(signal, dt, baseline):
    x = np.asarray(signal) - np.mean(signal)
    frequency, power = periodogram(x, fs=1 / dt, window="hann", detrend="constant")
    positive = frequency > 0
    acf = correlate(x, x, mode="full", method="fft")[len(x)-1:]
    acf /= acf[0]
    total = power[positive].sum()
    return {
        "n_points": len(x), "span_days": (len(x)-1) * dt,
        "std": float(np.std(x)),
        "power_fraction_period_ge_100d": float(power[positive & (frequency <= .01)].sum() / total),
        "power_fraction_period_ge_baseline": float(power[positive & (frequency <= 1 / baseline)].sum() / total),
        "acf_efold_days": first_efold(acf, dt),
    }, frequency, power, acf


def gamma_mass(width):
    lag = np.arange(int(np.ceil(25 * width / DT)) + 1) * DT
    mass = (gammainc(2, (lag + DT / 2) / width)
            - gammainc(2, np.maximum(lag - DT / 2, 0) / width))
    captured = float(mass.sum())
    mass /= captured
    mean = float(lag @ mass)
    assert captured > 1 - 1e-8
    assert abs(mean - 2 * width) < DT
    return lag, mass, captured


def make_case_config(case, source_cfg, input_dir):
    return {
        "source": "F9SIM", "reference_band": "W2", "driver_band": "W2",
        "redshift": source_cfg["redshift"], "response_bands": list(BANDS),
        "bands": {b: source_cfg["bands"][b] for b in ("W2", *BANDS)},
        "mica_stages": ["mica_recovery"],
        "mica_recovery": {
            "output_root": "runs/mica", "prepared_dir": str(input_dir),
            "driver": "W2", "responses": list(BANDS),
            "run_template": "run_{driver}_to_{band}_{ncomp}comp_{type_tf}",
            "ncomp": 2, "type_tf": "gamma", "component_order": "moment1",
            "type_lag_prior": 4, "lag_limit": [-10.0, 2000.0],
            "lag_prior": [[-10.0, 2000.0], [-10.0, 2000.0]],
            # Explicit component priors override native Tspan/3 width cap.
            "width_limit": None, "width_prior": [[0.01, 1200.0], [0.01, 1200.0]],
            "flag_lag_posivity": False, "flag_negative_resp": False,
            "flag_trend": 0, "max_num_saves": 10000, "num_particles": 4,
            "ptol": 0.1, "mpi_per_band": 2, "max_parallel": 1,
            "plot_decomp": True, "plot_tf_lag_range": [-100.0, 500.0],
            "plot_tf_num_points": 12001, "posterior_branches": {"enabled": False},
        },
        "simulation": {"case": case, "seed": SEED,
                       "plot_window_note": "The standard -100..500d plot is truncated for the 800d slow centroid; full-kernel truth is saved separately."},
    }


def prepare():
    out = HERE / "outputs"
    if (HERE / "run_status.json").exists():
        raise RuntimeError("A run status already exists; do not regenerate inputs for a started experiment.")
    out.mkdir(parents=True, exist_ok=True)
    source_cfg = yaml.safe_load((ROOT / "F9_a/config/source_config.yaml").read_text())
    disk = build_kernels(out / "kernels", dt=DT)
    time = MOTHER_START + np.arange(N_MOTHER) * DT
    seeds = np.random.SeedSequence(SEED).spawn(6)
    rng = np.random.default_rng(seeds[0])
    rho = np.exp(-DT / DRW_TAU)
    innovation = rng.normal(0, DRW_STD * np.sqrt(1 - rho*rho), N_MOTHER)
    innovation[0] = rng.normal(0, DRW_STD)
    hidden = lfilter([1], [1, -rho], innovation)
    del innovation
    curves = {b: fftconvolve(hidden, disk["mass"][b], mode="full")[:N_MOTHER]
              for b in ("W2", *BANDS)}
    observed = {b: np.loadtxt(ROOT / f"F9_a/fflux/host_sub/F9_a_{b}.txt")
                for b in ("W2", *BANDS)}
    epoch = min(float(values[0, 0]) for values in observed.values())
    baseline = max(float(values[-1, 0]) for values in observed.values()) - epoch
    obs_time = {b: values[:, 0] - epoch for b, values in observed.items()}
    # Reuse actual timestamps and each observation's fractional uncertainty.
    noise_sd = {b: values[:, 2] / np.mean(values[:, 1]) for b, values in observed.items()}
    noise = {b: np.random.default_rng(seeds[3 + i]).normal(0, noise_sd[b])
             for i, b in enumerate(("W2", *BANDS))}
    obs_disk = {b: np.interp(obs_time[b], time, curves[b]) for b in curves}
    for values in obs_disk.values():
        values -= values.mean()
    driver_flux = 1 + obs_disk["W2"] + noise["W2"]
    diag_select = (time >= -5000) & (time < 15000)
    diag_step = 10
    display_select = (time >= 0) & (time <= baseline)
    display_idx = np.flatnonzero(display_select)[::10]
    display_time = time[display_idx]
    diagnostics, prepared_cases = {}, []
    plot_data = {}
    summary = {
        "status": "inputs_generated_not_fitted", "source": "F9_a", "seed": SEED,
        "bands": list(BANDS), "driver": "W2", "n_fits": 8,
        "time_frame": "observer_days", "time_zero_mjd": epoch, "baseline_days": baseline,
        "dt_days": DT, "mother_n": N_MOTHER, "mother_start_day": MOTHER_START,
        "mother_end_day": float(time[-1]), "hidden_drw_tau_days": DRW_TAU,
        "hidden_drw_std": DRW_STD, "slow_to_disk_rms_ratio_at_observed_times": SLOW_RMS_RATIO,
        "noise_model": "independent Gaussian noise with F9 per-observation error divided by F9 band mean; the same noise draw is reused across all four cases",
        "disk_truth": disk["metadata"],
        "short_response_note": "Each band is D convolved with its absolute finite-difference disk kernel. W2 is also a response; no kernel subtraction or unstable relative deconvolution is used.",
        "relative_moment_note": "For response cases, injected_total_relative_* are formal low-frequency moment benchmarks of (psi_band + slow_scale*psi_W2*h_slow)/psi_W2 in Fourier space. They do not assert the existence of a positive, causal, or exactly double-Gamma relative kernel. Centroid and second cumulant are derived from full absolute-kernel moments, not from the truncated display window.",
        "nonresponse_note": "Independent Fourier draws receive only a declared marginal PSD and independent seed, not W2, D, or any response-kernel array. One independent slow realization per timescale is shared between M2 and B before separate RMS scaling.",
        "observation_inputs": {}, "slow_cases": {}, "cases": [],
        "inference_note": "One realization demonstrates recovery or misattribution but does not estimate a false-positive rate. There is no injected total transfer-function truth for the nonresponse cases.",
    }
    for b, original in observed.items():
        np.savetxt(out / f"F9_{b}_observing_window.csv",
                   np.column_stack((original[:, 0], obs_time[b], original[:, 2], noise_sd[b])),
                   delimiter=",", header="mjd,observer_day,original_error,fractional_error", comments="")
        summary["observation_inputs"][b] = {
            "path": str(ROOT / f"F9_a/fflux/host_sub/F9_a_{b}.txt"),
            "n_points": len(original), "median_cadence_day": float(np.median(np.diff(original[:, 0]))),
            "mean_cadence_day": float(np.mean(np.diff(original[:, 0]))),
            "median_fractional_error": float(np.median(noise_sd[b])),
            "disk_rms": float(np.std(obs_disk[b])),
        }
    for index, (timescale, width) in enumerate(WIDTHS.items()):
        lag, mass, captured = gamma_mass(width)
        slow_response = fftconvolve(curves["W2"], mass, mode="full")[:N_MOTHER]
        slow_random = independent_slow(N_MOTHER, DT, width, np.random.default_rng(seeds[index + 1]))
        slow_signals = {"response": slow_response, "random": slow_random}
        np.savetxt(out / "kernels" / f"slow_{timescale}_kernel.csv",
                   np.column_stack((lag, mass / DT)), delimiter=",", comments="",
                   header="relative_lag_observer_day,density_per_day", fmt="%.10g")
        # Ensemble correlation scale from the declared marginal PSD, not w.
        f_theory = np.fft.rfftfreq(2**18, .2)
        theory_acf = np.fft.irfft(slow_psd(f_theory, width), n=2**18)
        theory_acf /= theory_acf[0]
        ensemble_efold = first_efold(theory_acf, .2)
        summary["slow_cases"][timescale] = {
            "gamma_shape": 2, "onset_days": 0., "width_days": width,
            "centroid_days": 2 * width, "kernel_std_days": float(np.sqrt(2)*width),
            "kernel_captured_mass": captured, "declared_psd_timescales_days": [DRW_TAU, width],
            "ensemble_acf_efold_days": ensemble_efold,
            "baseline_over_ensemble_acf_time": baseline / ensemble_efold,
        }
        diagnostics[timescale] = {}
        for origin, signal in slow_signals.items():
            met, freq, power, acf = signal_metrics(signal[diag_select][::diag_step], DT*diag_step, baseline)
            diagnostics[timescale][origin] = met
            if met["power_fraction_period_ge_100d"] < .90:
                raise RuntimeError(f"Declared seed failed slow-power check: {timescale}/{origin}; do not silently resample")
            np.savetxt(out / f"{timescale}_{origin}_mother_psd.csv",
                       np.column_stack((freq, power)), delimiter=",", comments="",
                       header="frequency_per_observer_day,power_density", fmt="%.10g")
            np.savetxt(out / f"{timescale}_{origin}_mother_acf.csv",
                       np.column_stack((np.arange(10001)*DT*diag_step, acf[:10001])),
                       delimiter=",", comments="", header="lag_observer_day,acf", fmt="%.10g")
            case = f"{origin}_{timescale}"
            case_root = HERE / "cases" / case
            input_dir = case_root / "input"
            (case_root / "config").mkdir(parents=True, exist_ok=True)
            input_dir.mkdir(exist_ok=True)
            np.savetxt(input_dir / "F9SIM_W2.txt", np.column_stack((obs_time["W2"], driver_flux, noise_sd["W2"])), fmt="%.12g")
            case_record = {"name": case, "origin": origin, "timescale": timescale,
                           "path": str(case_root), "bands": {}}
            for band in BANDS:
                slow_obs = np.interp(obs_time[band], time, signal)
                slow_mean = float(slow_obs.mean())
                slow_obs -= slow_mean
                gain = float(SLOW_RMS_RATIO * np.std(obs_disk[band]) / np.std(slow_obs))
                target = 1 + obs_disk[band] + gain * slow_obs + noise[band]
                assert np.isfinite(target).all() and np.all(target > 0)
                if origin == "response" and gain >= 9.5:
                    raise RuntimeError("Injected slow gain too near native amplitude upper bound 10; expand amplitude prior explicitly before fitting.")
                np.savetxt(input_dir / f"F9SIM_{band}.txt", np.column_stack((obs_time[band], target, noise_sd[band])), fmt="%.12g")
                np.savetxt(case_root / f"truth_at_{band}_observations.csv",
                           np.column_stack((obs_time[band], obs_disk[band], gain*slow_obs, noise[band], target)),
                           delimiter=",", comments="", header="observer_day,disk_variation,slow_variation,noise,observed_flux", fmt="%.12g")
                disk_mass = disk["mass"][band]
                w2_mass = disk["mass"]["W2"]
                disk_mu = float(disk["lag"] @ disk_mass)
                w2_mu = float(disk["lag"] @ w2_mass)
                disk_var = float((disk["lag"]-disk_mu)**2 @ disk_mass)
                w2_var = float((disk["lag"]-w2_mu)**2 @ w2_mass)
                short_mu = disk_mu - w2_mu
                short_var = disk_var - w2_var
                total_mu = (short_mu + gain*2*width)/(1+gain) if origin == "response" else None
                total_var = ((short_var + (short_mu-total_mu)**2 + gain*(2*width**2 + (2*width-total_mu)**2))/(1+gain)
                             if origin == "response" else None)
                case_record["bands"][band] = {
                    "slow_scale": gain, "slow_rms": float(np.std(gain*slow_obs)),
                    "disk_rms": float(np.std(obs_disk[band])),
                    "slow_disk_pearson_r": float(np.corrcoef(slow_obs,obs_disk[band])[0,1]),
                    "short_relative_centroid_days": short_mu,
                    "short_relative_second_cumulant_days2": short_var,
                    "injected_slow_centroid_days": 2*width if origin=="response" else None,
                    "injected_slow_width_days": width if origin=="response" else None,
                    "injected_total_relative_centroid_days": total_mu,
                    "injected_total_relative_second_cumulant_days2": total_var,
                }
                display_disk = curves[band][display_idx] - np.mean(np.interp(obs_time[band], time, curves[band]))
                display_slow = gain*(signal[display_idx]-slow_mean)
                plot_data[(case,band)] = (display_disk,display_slow,target)
                np.savetxt(case_root / f"noiseless_{band}.csv",
                           np.column_stack((display_time, display_disk, display_slow, 1+display_disk+display_slow)),
                           delimiter=",", comments="", header="observer_day,disk_variation,slow_variation,total_noiseless_flux", fmt="%.10g")
            cfg = make_case_config(case, source_cfg, input_dir)
            (case_root / "config/source_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            summary["cases"].append(case_record)
            prepared_cases.append(str(case_root))
        del slow_response, slow_random, slow_signals
    summary["mother_signal_diagnostics"] = diagnostics
    summary["software"] = {"numpy": np.__version__, "matplotlib": matplotlib.__version__}
    (out / "truth.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    (HERE / "manifest.json").write_text(json.dumps({"cases":prepared_cases,"n_fits":8,"truth_file":str(out/'truth.json')}, indent=2)+"\n")
    plot_inputs(plot_data, obs_time, noise_sd, display_time, baseline, out)
    print(json.dumps({"cases":len(prepared_cases),"fits":8,"baseline_days":baseline,"diagnostics":diagnostics},indent=2))


def plot_inputs(data, obs_time, noise_sd, display_time, baseline, out):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
                         "axes.spines.top":False,"axes.spines.right":False})
    for band in BANDS:
        fig, axes = plt.subplots(2,2,figsize=(13,7),sharex=True,layout="constrained")
        for row,timescale in enumerate(WIDTHS):
            for origin,style in (("response","-"),("random","--")):
                disk,slow,target = data[(f"{origin}_{timescale}",band)]
                axes[row,0].plot(display_time,slow,color=COLORS[origin],ls=style,lw=1.5,label=origin)
                axes[row,1].plot(display_time,1+disk+slow,color=COLORS[origin],ls=style,lw=1,label=origin)
                axes[row,1].errorbar(obs_time[band],target,yerr=noise_sd[band],fmt=".",ms=2,
                                    color=COLORS[origin],alpha=.40,elinewidth=.45)
            axes[row,0].plot(display_time,disk,color=COLORS["disk"],lw=.65,alpha=.6,label="common disk")
            axes[row,0].set_title(f"{timescale}: injected components")
            axes[row,1].set_title(f"{timescale}: synthetic observations")
            axes[row,0].set_ylabel("Flux variation (mean total = 1)")
            axes[row,1].set_ylabel("Normalized flux")
            for ax in axes[row]:
                ax.set_xlim(0,baseline)
                ax.legend(loc="upper right",ncol=3,fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("Observer time from F9 window start (day)")
        fig.suptitle(f"F9 {band}: response and independent slow-signal injections\n"
                     f"Actual {len(obs_time[band])}-point sampling, {baseline:.1f} d baseline; slow/disk RMS = {SLOW_RMS_RATIO}")
        for suffix in ("png","pdf"):
            fig.savefig(out / f"F9_{band}_injections.{suffix}",dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
