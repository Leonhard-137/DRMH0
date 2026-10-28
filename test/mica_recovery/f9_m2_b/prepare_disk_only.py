#!/usr/bin/env python3
"""Prepare a paired, disk-only control from the saved F9 simulation truth.

This script never regenerates the random signals or starts a MICA fit.
The original four-case manifest and the running batch remain separate.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE / "cases" / "response_within"
CASE = HERE / "cases" / "disk_only"


def main():
    if CASE.exists():
        raise RuntimeError(f"Control already exists; refusing to overwrite: {CASE}")
    parent_truth = json.loads((HERE / "outputs/truth.json").read_text())
    reference = next(c for c in parent_truth["cases"] if c["name"] == "response_within")
    cfg = yaml.safe_load((BASE / "config/source_config.yaml").read_text())
    cfg["mica_recovery"]["prepared_dir"] = str(CASE / "input")
    cfg["simulation"] = {
        "case": "disk_only", "seed": parent_truth["seed"],
        "parent_case": str(BASE),
        "construction": "Same W2, disk variation, timestamps, errors and noise; added slow variation is zero.",
    }
    truth = {
        "status": "inputs_prepared_not_fitted", "name": "disk_only",
        "path": str(CASE), "driver": "W2", "bands": {}, "n_fits": 2,
        "seed": parent_truth["seed"], "time_frame": parent_truth["time_frame"],
        "time_zero_mjd": parent_truth["time_zero_mjd"],
        "baseline_days": parent_truth["baseline_days"],
        "parent_truth_file": str(HERE / "outputs/truth.json"),
        "disk_parameters_file": str(HERE / "outputs/kernels/disk_parameters.json"),
        "relative_moment_note": parent_truth["relative_moment_note"],
        "component_note": "The total relative disk moments are forward-model benchmarks. Neither fitted Gamma component has a separately injected truth; two components may share the disk signal.",
    }
    inputs, observations, dense_curves = {}, {}, {}
    for band in ("M2", "B"):
        parent_input = np.loadtxt(BASE / "input" / f"F9SIM_{band}.txt")
        saved = np.loadtxt(BASE / f"truth_at_{band}_observations.csv", delimiter=",", skiprows=1)
        if not np.array_equal(parent_input[:, 0], saved[:, 0]):
            raise ValueError(f"Input and truth times do not align for {band}")
        disk, noise = saved[:, 1], saved[:, 3]
        flux = 1.0 + disk + noise
        if not np.isfinite(flux).all() or np.any(flux <= 0):
            raise ValueError(f"Invalid pure-disk flux for {band}")
        inputs[band] = np.column_stack((parent_input[:, 0], flux, parent_input[:, 2]))
        observations[band] = np.column_stack((saved[:, 0], disk, np.zeros(len(saved)), noise, flux))
        dense = np.loadtxt(BASE / f"noiseless_{band}.csv", delimiter=",", skiprows=1)
        dense_curves[band] = np.column_stack((dense[:, 0], dense[:, 1], np.zeros(len(dense)), 1.0+dense[:, 1]))
        original = reference["bands"][band]
        truth["bands"][band] = {
            "n_points": len(saved), "slow_scale": 0.0, "slow_rms": 0.0,
            "disk_rms": float(np.std(disk)),
            "short_relative_centroid_days": original["short_relative_centroid_days"],
            "short_relative_second_cumulant_days2": original["short_relative_second_cumulant_days2"],
            "total_relative_centroid_benchmark_days": original["short_relative_centroid_days"],
            "total_relative_second_cumulant_benchmark_days2": original["short_relative_second_cumulant_days2"],
            "source_observation_truth": str(BASE / f"truth_at_{band}_observations.csv"),
        }
    (CASE / "input").mkdir(parents=True)
    (CASE / "config").mkdir()
    shutil.copyfile(BASE / "input/F9SIM_W2.txt", CASE / "input/F9SIM_W2.txt")
    for band in inputs:
        np.savetxt(CASE / "input" / f"F9SIM_{band}.txt", inputs[band], fmt="%.12g")
        np.savetxt(CASE / f"truth_at_{band}_observations.csv", observations[band], delimiter=",", comments="",
                   header="observer_day,disk_variation,slow_variation,noise,observed_flux", fmt="%.12g")
        np.savetxt(CASE / f"noiseless_{band}.csv", dense_curves[band], delimiter=",", comments="",
                   header="observer_day,disk_variation,slow_variation,total_noiseless_flux", fmt="%.10g")
    (CASE / "config/source_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (CASE / "truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False)+"\n")
    # Only the input-directory pointer differs in the inference configuration.
    original_fit = yaml.safe_load((BASE / "config/source_config.yaml").read_text())["mica_recovery"]
    compared = copy.deepcopy(cfg["mica_recovery"])
    compared["prepared_dir"] = original_fit["prepared_dir"]
    assert compared == original_fit
    print(f"Prepared disk-only M2/B control: {CASE}; no fits started.")


if __name__ == "__main__":
    main()
