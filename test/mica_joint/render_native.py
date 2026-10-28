#!/usr/bin/env python3
"""Reconstruct saved MICA posteriors with native C and render native figures.

This never calls model.run(), model.post_run(), dnest(), or a sampler. Native
output_reconstruction_parallel reads the original equal-weight posterior rows
from independent copies. Formal fit directories are read-only inputs.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "NGC2617_a/runs/mica/gamma/broad_np4/2comp"
OUTPUT = ROOT / "test/mica_joint/figures_20260909/native"
PYTHON = Path("/home/leonhard/miniforge3/envs/astro/bin/python")
BANDS = ["M2", "W1", "U", "B", "V"]
COPY_DATA = ["data_input.txt", "posterior_sample1d.txt_2",
             "posterior_sample_info1d.txt_2", "para_names_line.txt_2",
             "normalization.txt", "continuum_prior.json"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_pdf(destination: Path):
    import fitz
    with fitz.open(destination / "native_full.pdf") as pdf:
        if len(pdf) != 1:
            raise ValueError("Expected one native page for one dataset")
        page = pdf[0]
        page.get_pixmap(dpi=150).save(destination / "native_full.png")
        rect = page.rect
        clip = fitz.Rect(rect.width * 0.483, 0, rect.width, rect.height)
        left_margin = rect.width * (0.483 - 0.455)
        # Preserve the native LC panel and add whitespace at its left. The TF
        # panel's final '100' label protrudes beyond its axes into this crop;
        # remove only that identified neighbouring-panel text, not LC ticks.
        with fitz.open() as lc:
            target = lc.new_page(width=clip.width + left_margin, height=clip.height)
            content_rect = fitz.Rect(left_margin, 0, target.rect.width, target.rect.height)
            target.show_pdf_page(content_rect, pdf, 0, clip=clip)
            for word in page.get_text("words"):
                if word[4] == "100" and word[0] < rect.width * 0.49 and word[2] > clip.x0:
                    label_rect = fitz.Rect(max(0, word[0] - clip.x0 + left_margin - 0.2),
                                          word[1] - 0.2, word[2] - clip.x0 + left_margin + 0.2,
                                          word[3] + 0.2)
                    target.add_redact_annot(label_rect, fill=(1, 1, 1))
            target.apply_redactions(images=0, graphics=0, text=0)
            lc.save(destination / "native_light_curves.pdf")
            target.get_pixmap(dpi=170).save(destination / "native_light_curves.png")


def sources():
    return [("joint", RUN_ROOT / "joint_W2_to_M2_W1_U_B_V_2comp_gamma", ["W2", *BANDS])]
    

def all_sources():
    return sources() + [(f"pair_{band}", RUN_ROOT / f"run_W2_to_{band}_2comp_gamma", ["W2", band])
                        for band in BANDS]


def prepare(label, source, labels):
    state = json.loads((source / "run_mica_state.json").read_text())
    if state["status"] != "complete" or state["target_saves"] != 10000:
        raise ValueError(f"Formal fit is not complete: {source}")
    destination = OUTPUT / label
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "data").mkdir(exist_ok=True)
    (destination / "param").mkdir(exist_ok=True)
    originals = [source / "run_mica_state.json", source / "param/param_input"]
    for name in COPY_DATA:
        src = source / "data" / name
        if src.exists():
            shutil.copy2(src, destination / "data" / name)
            originals.append(src)
    shutil.copy2(source / "param/param_input", destination / "param/param_input")
    prior = source / "param/joint_parameter_priors.txt"
    if prior.exists():
        shutil.copy2(prior, destination / "param/joint_parameter_priors.txt")
        originals.append(prior)
    record = {"label": label, "source": str(source), "copy": str(destination),
              "labels": labels, "source_file_sha256": {str(p): digest(p) for p in originals}}
    (destination / "source_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def native_worker(destination: Path):
    os.chdir(destination)
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    import pymica
    from mpi4py import MPI
    from mica_resume_native import _load_native

    comm = MPI.COMM_WORLD
    record = json.loads(Path("source_manifest.json").read_text())
    # Constructor reads the copied saved parameter file; it does not run MICA.
    model = pymica.gmodel("param/param_input")
    library, options, extension = _load_native()
    for name, args, result in [
        ("mc_line_init", [], ct.c_int), ("mc_line_end", [], ct.c_int),
        ("set_idx_line_pm", [], None), ("output_reconstruction_parallel", [], None),
    ]:
        fn = getattr(library, name)
        fn.argtypes, fn.restype = args, result
    sample_path = Path("data/posterior_sample1d.txt_2")
    before = digest(sample_path)
    samples = np.atleast_2d(np.loadtxt(sample_path))
    expected_dimension = 3 + 7 * (len(record["labels"]) - 1)
    if samples.shape[1] != expected_dimension or not np.isfinite(samples).all():
        raise ValueError("Saved posterior dimensions or values are invalid")
    if samples.shape[0] < comm.Get_size():
        raise ValueError("Use at most one MPI rank per saved posterior row")
    library.set_argv(0, 0, 0, 0, 0)
    library.read_data()
    library.init()
    line_initialized = False
    try:
        ct.c_int.in_dll(library, "num_gaussian").value = 2
        ct.c_int.in_dll(library, "num_params_line").value = expected_dimension - 3
        ct.c_int.in_dll(library, "num_params").value = expected_dimension
        (ct.c_char * 256).in_dll(library, "postfix").value = b"_2"
        library.mc_line_init()
        line_initialized = True
        library.set_idx_line_pm()
        # This getter-backed option points directly to the copied saved rows.
        options.posterior_sample_file = b"data/posterior_sample1d.txt_2"
        library.output_reconstruction_parallel()
        comm.Barrier()
        if digest(sample_path) != before:
            raise RuntimeError("Reconstruction changed the saved posterior copy")
        if comm.Get_rank() == 0:
            reconstruction = np.atleast_2d(np.loadtxt("data/pall.txt_2"))
            if reconstruction.shape[1] != 3 or not np.isfinite(reconstruction).all() or np.any(reconstruction[:, 2] < 0):
                raise ValueError("Native reconstruction is invalid")
            # Installed astro plotter already understands Gamma centroids.
            pymica.utility.plot_results(
                fdir="./", fname="data/data_input.txt", ngau=2,
                tau_low=-10.0, tau_upp=100.0, flagvar=0, flagtran=0,
                flagtrend=0, flagnegresp=False, typetf="22", typemodel=0,
                resp_input=None, doshow=False, tf_lag_range=[-10.0, 100.0],
                hist_lag_range=[-10.0, 100.0], labels=[record["labels"]],
                tf_num_points=1000,
            )
            shutil.copy2("data/fig_2.pdf", "native_full.pdf")
            render_pdf(destination)
            result = dict(record, native_extension=extension, mpi_ranks=comm.Get_size(),
                          posterior_rows=int(samples.shape[0]), posterior_dimension=expected_dimension,
                          reconstruction_rows=int(reconstruction.shape[0]),
                          posterior_copy_unchanged=True, sampler_called=False,
                          reconstruction_function="output_reconstruction_parallel",
                          plot_function="pymica.utility.plot_results",
                          products=[str(destination / p) for p in ["native_full.pdf", "native_full.png",
                                    "native_light_curves.pdf", "native_light_curves.png"]])
            Path("render_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        comm.Barrier()
    finally:
        if line_initialized:
            library.mc_line_end()
        library.end_run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--only", choices=[x[0] for x in all_sources()])
    args = parser.parse_args()
    if args.worker:
        native_worker(args.worker.resolve())
        return
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": "/tmp"})
    for key in ("TEMP", "TMP"):
        if env.get(key, "").startswith("/mnt/"):
            env.pop(key, None)
    records = []
    for label, source, labels in all_sources():
        if args.only and label != args.only:
            continue
        existing_manifest = OUTPUT / label / "render_manifest.json"
        if existing_manifest.exists():
            existing = json.loads(existing_manifest.read_text())
            if (existing.get("formal_inputs_unchanged")
                    and all(Path(p).is_file() for p in existing["products"])
                    and all(digest(Path(p)) == value for p, value in existing["source_file_sha256"].items())):
                records.append(existing)
                render_pdf(Path(existing["copy"]))
                print(f"Verified existing native figure: {label}", flush=True)
                continue
        record = prepare(label, source, labels)
        destination = Path(record["copy"])
        cmd = [str(PYTHON.with_name("mpiexec")), "-n", "2", str(PYTHON), "-B",
               str(Path(__file__).resolve()), "--worker", str(destination)]
        print(f"Rendering native saved-posterior figure: {label}", flush=True)
        with (destination / "render.log").open("w") as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((destination / "render_manifest.json").read_text())
        for path, expected in record["source_file_sha256"].items():
            if digest(Path(path)) != expected:
                raise RuntimeError(f"Formal input changed during rendering: {path}")
        result["formal_inputs_unchanged"] = True
        (destination / "render_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        records.append(result)
        print(f"Complete: {destination / 'native_light_curves.png'}", flush=True)
    (OUTPUT / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    if len(records) == 6:
        import fitz
        with fitz.open() as combined:
            for record in records:
                with fitz.open(Path(record["copy"]) / "native_light_curves.pdf") as one:
                    combined.insert_pdf(one)
            combined.save(OUTPUT / "native_light_curves_all.pdf")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if "--worker" in sys.argv:
            from mpi4py import MPI
            import traceback
            traceback.print_exc()
            MPI.COMM_WORLD.Abort(1)
        raise
