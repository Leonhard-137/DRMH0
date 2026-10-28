#!/usr/bin/env python3
"""Generate a W2-driven broad-filter experiment; does not import or run MICA.

Chart contract: compare two unit-area Gamma filters with identical centroid
and unequal width. Static Matplotlib PNG/PDF, two blue/orange roots plus grey,
line styles distinguish cases. Show transfer density, analytic power transfer,
and common-realization light curves; preserve their amplitude differences.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import zipfile
import numpy as np
from scipy.signal import fftconvolve, lfilter
from scipy.special import gammainc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(ROOT / "test" / "disk_transfer"))
from disk_transfer import DiskTransferModel, mdot_from_eddington_ratio

DT = 0.02
SPAN = 1000.0
CADENCE = 2.0
NOISE = 0.01
DRW_TAU = 10.0
DRW_STD = 0.20
SEEDS = [2026090401, 2026090402, 2026090403]
CASES = {"NARROW": (58.0, 1.0), "BROAD": (0.0, 30.0)}
Z = 0.0163
LAG = np.arange(45001) * DT  # 0--900 days; broad Gamma omitted mass < 3e-12
TDISK = np.arange(2501) * DT  # 0--50 observer-frame days
TFULL = -1000.0 + np.arange(100001) * DT
TOBS = np.arange(501) * CADENCE
OBS_INDEX = np.rint((TOBS - TFULL[0]) / DT).astype(int)
PLOT_INDEX = np.arange(50000, 100001, 10)  # 0.2-day plotting grid

model = DiskTransferModel(
    wavelengths_angstrom=[1928.0 / (1.0 + Z)],
    time_days=TDISK / (1.0 + Z),
    nr=900, nphi=720, pulse_width_days=DT / (1.0 + Z),
)
disk = model.predict(
    m_bh_msun=5e7,
    mdot_g_s=mdot_from_eddington_ratio(5e7, 0.015, 0.1),
    lx_erg_s=9.68e43, h_rg=48.29, inclination_deg=40.0,
    rin_rg=94.87, reference_index=0,
)
w2_mass = disk["instant_response"][0] * DT / (1.0 + Z)
w2_mass /= w2_mass.sum()
assert disk["diagnostics"]["captured_fraction_min"] > 0.9999
mu_w2 = float(w2_mass @ TDISK)
var_w2 = float(w2_mass @ (TDISK - mu_w2) ** 2)
kernel = {}
metadata = {
    "status": "simulation_generated_MICA_not_run",
    "time_frame": "observer_days",
    "driver": "D convolved with current numerical W2 disk response",
    "disk_weighting": "existing B_lambda-weighted unit-area instant_response",
    "redshift": Z, "w2_wavelength_observer_angstrom": 1928.0,
    "dt_day": DT, "baseline_day": SPAN, "cadence_day": CADENCE,
    "hidden_drw_tau_day": DRW_TAU, "hidden_drw_stationary_std": DRW_STD,
    "mean_flux": 1.0, "error_each_band": NOISE,
    "seeds": SEEDS, "short_window_day": [600.0, 728.0],
    "w2_mean_day": mu_w2, "w2_variance_day2": var_w2,
    "cases": {}, "realizations": [],
}
for name, (c, w) in CASES.items():
    lower = np.maximum((LAG - DT / 2 - c) / w, 0.0)
    upper = np.maximum((LAG + DT / 2 - c) / w, 0.0)
    mass = gammainc(2, upper) - gammainc(2, lower)
    captured = float(mass.sum())
    mass /= captured
    mu = float(mass @ LAG)
    var = float(mass @ (LAG - mu) ** 2)
    assert abs(mu - 60.0) < 1e-4
    assert abs(var - 2 * w*w) < 1e-3
    assert captured > 1 - 1e-9
    kernel[name] = mass
    metadata["cases"][name] = {
        "gamma_shape": 2, "onset_day": c, "width_parameter_day": w,
        "centroid_day": 60.0, "std_day": float(np.sqrt(2) * w),
        "variance_day2": 2*w*w, "area": 1.0,
        "power_half_transmission_period_day": float(2*np.pi*w / np.sqrt(np.sqrt(2)-1)),
        "discrete_centroid_day": mu, "discrete_variance_day2": var,
        "captured_mass": captured,
    }
np.savetxt(OUT / "relative_kernels.csv",
           np.column_stack([LAG, kernel["NARROW"]/DT, kernel["BROAD"]/DT]),
           delimiter=",", header="lag_observer_day,narrow_density_per_day,broad_density_per_day",
           comments="", fmt="%.10g")
np.savetxt(OUT / "w2_absolute_kernel.csv", np.column_stack([TDISK, w2_mass/DT]),
           delimiter=",", header="lag_observer_day,w2_density_per_day", comments="", fmt="%.10g")
first = None
for seed in SEEDS:
    children = np.random.SeedSequence(seed).spawn(4)
    rng = np.random.default_rng(children[0])
    a = np.exp(-DT/DRW_TAU)
    innovations = rng.normal(0, DRW_STD*np.sqrt(1-a*a), TFULL.size)
    innovations[0] = rng.normal(0, DRW_STD)
    hidden = lfilter([1.0], [1.0, -a], innovations)
    w2 = fftconvolve(hidden, w2_mass, mode="full")[:TFULL.size]
    curves = {"W2": w2}
    for name, mass in kernel.items():
        curves[name] = fftconvolve(w2, mass, mode="full")[:TFULL.size]
    # Independent convolution path checks time origin, padding, and associativity.
    absolute_broad = fftconvolve(w2_mass, kernel["BROAD"], mode="full")
    direct_broad = fftconvolve(hidden, absolute_broad, mode="full")[:TFULL.size]
    error = float(np.max(np.abs(direct_broad[OBS_INDEX] - curves["BROAD"][OBS_INDEX])))
    assert error < 1e-10
    folder = OUT / f"seed_{seed}"
    folder.mkdir(exist_ok=True)
    observed = {}
    for j, name in enumerate(["W2", "NARROW", "BROAD"], start=1):
        noise_rng = np.random.default_rng(children[j])
        flux = 1 + curves[name][OBS_INDEX] + noise_rng.normal(0, NOISE, TOBS.size)
        assert np.all(np.isfinite(flux)) and flux.mean() > 0
        observed[name] = flux
        for window, mask in [
            ("full", np.ones(TOBS.size, dtype=bool)),
            ("short128", (TOBS >= 600) & (TOBS <= 728)),
        ]:
            dest = folder / window
            dest.mkdir(exist_ok=True)
            np.savetxt(dest / f"LONG_{name}.txt",
                       np.column_stack([TOBS[mask], flux[mask], np.full(mask.sum(), NOISE)]),
                       fmt="%.10g", header="observer_day flux error")
    np.savetxt(folder / "noiseless_lightcurves.csv",
               np.column_stack([TFULL[PLOT_INDEX], hidden[PLOT_INDEX],
                                w2[PLOT_INDEX], curves["NARROW"][PLOT_INDEX],
                                curves["BROAD"][PLOT_INDEX]]),
               delimiter=",", comments="", fmt="%.10g",
               header="observer_day,hidden_fractional,w2_fractional,narrow_fractional,broad_fractional")
    rmss = {name: float(np.std(curves[name][OBS_INDEX], ddof=1)) for name in curves}
    metadata["realizations"].append({
        "seed": seed, "full_points": len(TOBS), "short_points": 65,
        "noiseless_fractional_rms": rmss,
        "broad_to_narrow_rms": rmss["BROAD"]/rmss["NARROW"],
        "convolution_associativity_max_error": error,
    })
    if first is None:
        first = {name: values[PLOT_INDEX].copy() for name, values in curves.items()}
        first["W2_shifted"] = w2[PLOT_INDEX - int(round(60/DT))]
        first["observed_BROAD"] = observed["BROAD"]
metadata["software"] = {"numpy": np.__version__, "matplotlib": matplotlib.__version__}
(OUT / "truth.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#252a31", "text.color": "#252a31",
    "axes.grid": True, "grid.alpha": .18, "grid.color": "#a5a9ae",
})
fig = plt.figure(figsize=(12, 8.2), layout="constrained")
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.4])
axk, axp, axlc = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,:])
colors = {"NARROW": "#3b6fb6", "BROAD": "#d18a2e"}
styles = {"NARROW": "--", "BROAD": "-"}
for name, (_, w) in CASES.items():
    label = f"{name.title()}: width w = {w:g} d"
    mask = (LAG <= 240) & (kernel[name]/DT > 1e-7)
    axk.plot(LAG[mask], kernel[name][mask]/DT, color=colors[name],
             ls=styles[name], lw=2, label=label)
    period = np.geomspace(2, 2000, 500)
    transmission = (1+(2*np.pi*w/period)**2)**-2
    axp.loglog(period, transmission, color=colors[name], ls=styles[name], lw=2)
axk.axvline(60, color="#4d535b", lw=1, ls=":")
axk.set(xlim=(0,240), ylim=(1e-6,.6), yscale="log",
        xlabel="Relative lag (observer-frame days)", ylabel="Unit-area density (1/day)",
        title="Injected relative transfer functions")
axk.legend(frameon=False, loc="upper right", fontsize=9)
axk.text(.03,.95,"Both centroids = 60 d", transform=axk.transAxes, va="top", fontsize=9)
axp.axhline(.5, color="#4d535b", lw=1, ls=":")
axp.set(xlabel="Variability period (days)", ylabel="Power transmission",
        xlim=(2,2000), ylim=(1e-8,1.2), title="Analytic filter response")
axp.text(.04,.08,"Broad filter: half-power period = 293 d",
         transform=axp.transAxes, fontsize=9)
tp = TFULL[PLOT_INDEX]
axlc.plot(tp, 100*first["W2_shifted"], color="#9ca0a6", lw=.8, alpha=.7,
          label="W2, shifted by 60 d for display")
for name in CASES:
    axlc.plot(tp, 100*first[name], color=colors[name], ls=styles[name],
              lw=1.25 if name=="NARROW" else 2.5,
              label=f"{name.title()} response (noiseless)")
axlc.axvspan(600,728,color="#8e9399",alpha=.12, label="128 d crop")
axlc.axhline(0,color="#4d535b",lw=.7)
axlc.set(xlim=(0,1000), xlabel="Observer-frame time (days)",
         ylabel="Fractional variability (%)", title=f"Common realization: seed {SEEDS[0]}")
axlc.legend(loc="upper center", bbox_to_anchor=(.5,1.02), ncol=2,
            frameon=False, fontsize=8.5)
fig.suptitle("W2-driven broad-filter simulation", fontsize=16, fontweight="bold")
fig.savefig(OUT / "long_timescale_demo.png", dpi=180)
fig.savefig(OUT / "long_timescale_demo.pdf")
plt.close(fig)
archive = HERE / "MICA_long_timescale_simulation.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for path in [HERE/"PLAN_zh.md", Path(__file__).resolve(), *sorted(OUT.rglob("*"))]:
        if path.is_file():
            z.write(path, arcname=str(path.relative_to(HERE)))
print(json.dumps({"output": str(OUT), "archive": str(archive),
                  "realizations": metadata["realizations"]}, ensure_ascii=False, indent=2))
