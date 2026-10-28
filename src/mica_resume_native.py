"""Run or resume MICA stages through the loaded extension, without changing MICA.

Call after model.setup() / set_priors(), from every MPI rank in the run directory.
The caller owns checkpoint validation, sample-tail alignment, and restart aliases.

The DNestOptions prefix below is shared by upstream pymica 2.2.0 and the parallel
build whose total_particles member was appended to the structure.  Its posterior
path is checked through the native getter before continuum statistics are read.
The scientific mean/std calculation remains src/mc_con.c:postprocess_con(); this
adapter never resamples the continuum posterior in response/response_start mode.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import importlib
import json
import os
import math
import tempfile
from pathlib import Path


_NATIVE_PATH_BYTES = 156  # cdnest/dnest.h: STR_MAX_LENGTH (not MICA_MAX_STR_LENGTH)


class _DNestOptionsPrefix(ct.Structure):
    _fields_ = [
        ("num_particles", ct.c_uint),
        ("new_level_interval", ct.c_uint),
        ("save_interval", ct.c_uint),
        ("thread_steps", ct.c_uint),
        ("max_num_levels", ct.c_uint),
        ("lam", ct.c_double), ("beta", ct.c_double), ("max_ptol", ct.c_double),
        ("max_num_saves", ct.c_uint),
        ("thread_steps_factor", ct.c_double),
        ("new_level_interval_factor", ct.c_double),
        ("save_interval_factor", ct.c_double),
        ("sample_file", ct.c_char * _NATIVE_PATH_BYTES),
        ("sample_info_file", ct.c_char * _NATIVE_PATH_BYTES),
        ("levels_file", ct.c_char * _NATIVE_PATH_BYTES),
        ("sampler_state_file", ct.c_char * _NATIVE_PATH_BYTES),
        ("posterior_sample_file", ct.c_char * _NATIVE_PATH_BYTES),
    ]


def _load_native():
    extension = importlib.import_module("pymica.pymica")
    library = ct.CDLL(extension.__file__)
    signatures = {
        "set_argv": ([ct.c_int] * 5, None),
        "read_data": ([], ct.c_int),
        "init": ([], None),
        "end_run": ([], None),
        "mc_con_init": ([], ct.c_int),
        "mc_con_end": ([], ct.c_int),
        "postprocess_con": ([], None),
        "mc_con": ([ct.POINTER(ct.c_double)], None),
        "mc_line": ([ct.POINTER(ct.c_double)], None),
        "mc_mmap": ([ct.POINTER(ct.c_double)], None),
        "dnest_get_posterior_sample_file": ([ct.POINTER(ct.c_char)], None),
    }
    for name, (args, result) in signatures.items():
        function = getattr(library, name)
        function.argtypes, function.restype = args, result
    if ct.sizeof(ct.c_uint) != 4 or ct.sizeof(ct.c_double) != 8:
        raise RuntimeError("Unsupported native DNestOptions ABI")
    options = _DNestOptionsPrefix.in_dll(library, "options")
    old_path = options.posterior_sample_file
    probe = b"data/posterior_sample.txt"
    try:
        options.posterior_sample_file = probe
        actual = ct.create_string_buffer(_NATIVE_PATH_BYTES)
        library.dnest_get_posterior_sample_file(actual)
        if actual.value != probe:
            raise RuntimeError("Loaded MICA DNestOptions ABI does not match the adapter")
    finally:
        options.posterior_sample_file = old_path
    return library, options, str(Path(extension.__file__).resolve())


def _collective_check(comm, error):
    errors = comm.allgather(error)
    if any(errors):
        raise RuntimeError("; ".join(f"rank {rank}: {item}" for rank, item in enumerate(errors) if item))


def _posterior_metadata(comm, dimensions):
    """Validate the immutable input to C's fscanf loop, without computing stats."""
    result, error = None, None
    if comm.Get_rank() == 0:
        try:
            path = Path("data/posterior_sample.txt")
            raw = path.read_bytes()
            lines = raw.decode("ascii").splitlines()
            header = lines[0].split()
            if len(header) != 2 or header[0] != "#":
                raise ValueError("expected '# N' continuum posterior header")
            count = int(header[1])
            rows = [line.split() for line in lines[1:] if line.strip()]
            if count < 1 or len(rows) != count:
                raise ValueError("continuum posterior row count does not match its header")
            for row in rows:
                if len(row) != dimensions or not all(math.isfinite(float(value)) for value in row):
                    raise ValueError("continuum posterior has a wrong column count or non-finite values")
            result = {"posterior_file": str(path.resolve()), "sample_count": count,
                      "parameter_count": dimensions, "posterior_sha256": hashlib.sha256(raw).hexdigest()}
        except Exception as exc:
            error = f"Cannot reuse continuum posterior: {exc}"
    _collective_check(comm, error)
    return comm.bcast(result, root=0)


def _prior_state(comm, library, dimensions, metadata):
    error, result = None, None
    try:
        mean = ct.POINTER(ct.c_double).in_dll(library, "var_param")
        std = ct.POINTER(ct.c_double).in_dll(library, "var_param_std")
        if not mean or not std:
            raise RuntimeError("native continuum prior arrays were not initialized")
        means, deviations = list(mean[:dimensions]), list(std[:dimensions])
        if not all(math.isfinite(value) for value in means + deviations) or any(value < 0 for value in deviations):
            raise RuntimeError("native continuum prior contains invalid statistics")
        raw = ct.string_at(mean, dimensions * 8) + ct.string_at(std, dimensions * 8)
        result = dict(metadata, mean=means, std=deviations, state_sha256=hashlib.sha256(raw).hexdigest())
    except Exception as exc:
        error = str(exc)
    _collective_check(comm, error)
    hashes = comm.allgather(result["state_sha256"])
    if len(set(hashes)) != 1:
        raise RuntimeError("MICA continuum prior differs between MPI ranks")
    return result


def _persist_or_verify_prior(comm, prior, phase):
    """Commit the exact C prior before RM, or verify an existing commitment."""
    error, result = None, None
    if comm.Get_rank() == 0:
        path = Path("data/continuum_prior.json")
        temporary = None
        try:
            record = dict(prior, schema_version=1, source="MICA postprocess_con",
                          posterior_file="data/posterior_sample.txt", capture_phase=phase)
            checking = phase in {"response", "response_start"} and path.exists()
            if checking:
                saved = json.loads(path.read_text())
                if saved.get("schema_version") != 1:
                    raise ValueError("unsupported continuum prior manifest version")
                keys = ("posterior_sha256", "sample_count", "parameter_count",
                        "mean", "std", "state_sha256")
                different = [key for key in keys if saved.get(key) != record[key]]
                if different:
                    raise ValueError("continuum prior no longer matches the saved run: " + ", ".join(different))
            else:
                with tempfile.NamedTemporaryFile("w", dir=path.parent,
                        prefix=".continuum_prior.", suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    json.dump(record, stream, indent=2, allow_nan=False)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                temporary = None
                directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            result = {"path": str(path.resolve()), "verified_existing": checking,
                      "legacy_capture": phase in {"response", "response_start"} and not checking}
        except Exception as exc:
            error = f"Continuum prior provenance check failed: {exc}"
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as exc:
                    error = f"{error}; temporary prior cleanup failed: {exc}"
    _collective_check(comm, error)
    return comm.bcast(result, root=0)


def run_native_resume(*, mixed: bool, phase: str, fit_only: bool = False) -> dict:
    """Run fresh, response, response_start, or continuum after model.setup().

    fresh calls the same native continuum/RM stages as model.run(), recording the
    actual prior between them. response resumes RM from its prepared checkpoint;
    response_start starts RM using the original completed continuum posterior.
    continuum resumes its own checkpoint, then starts a new RM stage. Native
    checkpoint/RNG limitations remain; the caller must validate compatibility.
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    error = None
    library = options = extension_path = None
    try:
        if phase not in {"fresh", "response", "response_start", "continuum"}:
            raise ValueError(f"Unknown native resume phase: {phase}")
        library, options, extension_path = _load_native()
    except Exception as exc:
        error = str(exc)
    _collective_check(comm, error)
    if len(set(comm.allgather((mixed, phase)))) != 1:
        raise RuntimeError("All MPI ranks must request the same native resume phase")

    library.set_argv(0, 0, 1 if phase == "continuum" else 0, 0, 0)
    library.read_data()
    library.init()
    try:
        dimensions = ct.c_int.in_dll(library, "num_params_var").value
        _collective_check(comm, None if dimensions > 0 else "Invalid native continuum parameter count")
        continuum_logz = None
        if phase in {"fresh", "continuum"}:
            value = ct.c_double()
            library.mc_con(ct.byref(value))
            continuum_logz = value.value
            metadata = _posterior_metadata(comm, dimensions)
        else:
            metadata = _posterior_metadata(comm, dimensions)
            library.mc_con_init()
            ct.c_int.in_dll(library, "num_params").value = dimensions
            previous_path = options.posterior_sample_file
            try:
                options.posterior_sample_file = b"data/posterior_sample.txt"
                # Same C arithmetic and MPI broadcast as the original mc_con.
                library.postprocess_con()
            finally:
                options.posterior_sample_file = previous_path
                library.mc_con_end()

        prior = _prior_state(comm, library, dimensions, metadata)
        prior_manifest = _persist_or_verify_prior(comm, prior, phase)
        # CDNest still writes posterior samples/evidence; skip RM reconstruction
        # and decomposition when the user requested only a fit.
        library.set_argv(0, 0, 1 if phase == "response" else 0, 0, int(fit_only))
        logz = ct.POINTER(ct.c_double).in_dll(library, "logz_arr")
        _collective_check(comm, None if logz else "Native response evidence buffer is missing")
        response = library.mc_mmap if mixed else library.mc_line
        response(logz)
        return {"phase": phase, "mixed": bool(mixed), "extension": extension_path,
                "continuum_log_evidence": continuum_logz, "continuum_prior": prior,
                "continuum_prior_manifest": prior_manifest,
                "native_restart_is_bitwise": False}
    finally:
        library.end_run()
