#!/usr/bin/env python3
"""Run the eight prepared F9 experiments through the normal MICA scheduler.

This is not an MPI entry point. At most two case schedulers run concurrently;
each scheduler fits M2 then B with two ranks and four particles per rank.
Use a persistent terminal/tmux session for this long-running command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_status(status):
    temporary = HERE / "run_status.json.tmp"
    temporary.write_text(json.dumps(status, indent=2)+"\n")
    temporary.replace(HERE / "run_status.json")


def validate_inputs():
    sys.path.insert(0,str(ROOT / "src"))
    import run_mica
    manifest = json.loads((HERE / "manifest.json").read_text())
    if manifest["n_fits"] != 8 or len(manifest["cases"]) != 4:
        raise ValueError("expected four cases, each with M2 and B")
    for folder in manifest["cases"]:
        path = Path(folder)
        if not path.resolve().is_relative_to(HERE):
            raise ValueError("case must remain under the experiment directory")
        cfg = run_mica.read_yaml(path / "config/source_config.yaml")
        fit = run_mica.fit_config(cfg,"mica_recovery")
        assert fit["driver"] == "W2" and fit["responses"] == ["M2","B"]
        assert fit["mpi_per_band"] == 2 and fit["num_particles"] == 4
        assert fit["max_parallel"] == 1
        for band in ("W2","M2","B"):
            if not (Path(fit["prepared_dir"]) / f"F9SIM_{band}.txt").is_file():
                raise FileNotFoundError(band)
        for band in fit["responses"]:
            kwargs = run_mica.setup_kwargs(fit,{"set1":[None,None]},band)
            assert kwargs["lag_prior"] == [[-10.,2000.],[-10.,2000.]]
            assert kwargs["width_prior"] == [[.01,1200.],[.01,1200.]]
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only",action="store_true")
    args = parser.parse_args()
    manifest = validate_inputs()
    if args.validate_only:
        print("PASS: 4 cases / 8 fits, all inputs and sampler settings verified; no fits started.")
        return
    if (HERE / "run_status.json").exists():
        raise RuntimeError("Experiment already has a run status. Inspect it before restarting any fit.")
    for folder in manifest["cases"]:
        if (Path(folder) / "runs").exists():
            raise RuntimeError(f"Existing run directory: {folder}; will not overwrite it.")
    logdir = HERE / "logs"
    logdir.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
                "MKL_NUM_THREADS":"1","PYTHONUNBUFFERED":"1",
                "MPLBACKEND":"Agg","MPLCONFIGDIR":str(HERE / ".mplconfig"),
                "TMPDIR":"/tmp"})
    status = {"status":"running","started_utc":timestamp(),"n_fits":8,
              "max_concurrent_cases":2,"ranks_per_fit":2,"particles_per_rank":4,
              "cases":{Path(p).name:{"status":"pending","path":p} for p in manifest["cases"]}}
    write_status(status)
    running = {}
    pending = list(manifest["cases"])
    while pending or running:
        while pending and len(running)<2:
            folder = pending.pop(0)
            name = Path(folder).name
            logfile = logdir / f"{name}.log"
            handle = logfile.open("x")
            command = [sys.executable,"-u",str(ROOT / "src/run_mica.py"),folder,
                       "--stage","mica_recovery"]
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            running[name]=(process,handle)
            status["cases"][name].update(status="running",pid=process.pid,
                                           started_utc=timestamp(),log=str(logfile),command=command)
            print(f"Started {name}, scheduler PID {process.pid}",flush=True)
            write_status(status)
        for name,(process,handle) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            casepath = Path(status["cases"][name]["path"])
            required=[casepath / f"runs/mica/2comp/run_W2_to_{band}_2comp_gamma/data/{file}"
                      for band in ("M2","B")
                      for file in ("posterior_sample1d.txt_2","sample1d.txt_2","weights.txt",
                                   "tranfunc_0_1.txt_2","pline.txt_2_comp0","pline.txt_2_comp1")]
            missing=[str(p) for p in required if not p.is_file() or p.stat().st_size==0]
            state = "completed" if code==0 and not missing else "failed"
            status["cases"][name].update(status=state,exit_code=code,missing_outputs=missing,
                                           finished_utc=timestamp())
            print(f"{state}: {name}, exit {code}",flush=True)
            del running[name]
            write_status(status)
        if pending or running:
            time.sleep(5)
    status["status"] = ("fits_completed" if all(c["status"]=="completed" for c in status["cases"].values()) else "failed")
    status["finished_utc"]=timestamp()
    write_status(status)
    if status["status"] == "fits_completed":
        collector = HERE / "summarize_experiment.py"
        if collector.is_file():
            code=subprocess.call([sys.executable,str(collector)],cwd=ROOT,env=env)
            status["analysis_exit_code"]=code
            status["status"]="completed" if code==0 else "analysis_failed"
            write_status(status)
    print(json.dumps(status,indent=2),flush=True)
    if status["status"] in ("failed","analysis_failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
