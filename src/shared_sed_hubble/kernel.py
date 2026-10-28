"""ctypes acceleration of the unchanged marginalized FFA GP likelihood.

An instance owns reusable scratch arrays and is not thread-safe. Separate
processes/instances may evaluate independently. Priors are the caller's job.
Library compilation is explicit and loading is deferred until first evaluation.
"""
from pathlib import Path
import ctypes
import os
import shutil
import subprocess
import numpy as np

_DOUBLE = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
_INT = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
_DEFAULT_LIBRARY_PATH: Path | None = None


def build_kernel(build_dir: Path) -> Path:
    """Explicitly compile the C kernel inside an output build directory.

    Call with ``output_dir / 'build'``. Compilation is never performed by an
    import, ``FastFFA`` construction, or a plotting operation. The returned
    path can be passed to every ``FastFFA`` instance; it also becomes this
    process's default for instances constructed without ``library_path``.
    """
    global _DEFAULT_LIBRARY_PATH
    compiler = shutil.which('cc') or shutil.which('gcc')
    if compiler is None:
        raise RuntimeError('Building the FFA kernel requires an existing cc or gcc compiler.')
    source = Path(__file__).with_name('ffa_kernel.c').resolve()
    build_dir = Path(build_dir).expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    library_path = build_dir / 'libffa_kernel.so'
    temporary_path = build_dir / f'libffa_kernel.{os.getpid()}.so'
    command = [compiler, '-O3', '-std=c99', '-Wall', '-Wextra', '-shared',
               '-fPIC', str(source), '-o', str(temporary_path), '-lm']
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f'Could not run the FFA kernel compiler: {exc}') from exc
    if result.returncode:
        temporary_path.unlink(missing_ok=True)
        message = (result.stderr or result.stdout).strip()
        raise RuntimeError(f'FFA kernel compilation failed: {message}')
    # Keep any already mapped library inode intact during an explicit rebuild.
    os.replace(temporary_path, library_path)
    _DEFAULT_LIBRARY_PATH = library_path
    return library_path


def _load_library(library_path: Path):
    try:
        library = ctypes.CDLL(str(library_path))
    except OSError as exc:
        raise RuntimeError(f'Could not load FFA kernel {library_path}: {exc}') from exc
    library.ffa_evaluate.argtypes = [ctypes.c_int, ctypes.c_int, _DOUBLE,
        _DOUBLE, _DOUBLE, _INT, _DOUBLE, _DOUBLE, _DOUBLE, _DOUBLE, _DOUBLE,
        _DOUBLE, ctypes.c_int]
    library.ffa_evaluate.restype = ctypes.c_int
    return library

class FastFFA:
    def __init__(self, model, library_path: Path | None = None):
        self.library_path = (Path(library_path).expanduser().resolve()
                             if library_path is not None else None)
        self._library = None
        self.model = model
        self.n = int(model.ndall)
        self.p = int(model.nset)
        self.t = np.ascontiguousarray(model.t, dtype=np.float64)
        self.y = np.ascontiguousarray(model.y, dtype=np.float64)
        self.err = np.ascontiguousarray(model.yerr, dtype=np.float64)
        larr = np.asarray(model.Larr, dtype=float)
        if (larr.shape != (self.n,self.p) or not np.all(np.sum(larr,axis=1)==1)
                or not np.all((larr==0)|(larr==1))):
            raise ValueError('FastFFA requires one band per observation')
        if self.n<1 or self.p<1 or np.any(np.diff(self.t)<0):
            raise ValueError('FastFFA requires nonempty time-sorted data')
        self.band = np.ascontiguousarray(np.argmax(larr,axis=1),dtype=np.int32)
        self.work = np.empty(self.n*(self.p+5)+2*self.p*self.p+3*self.p)
        self.stats = np.empty(2)
        self.mean = np.empty(self.p)
        self.cov = np.empty(self.p*self.p)
        self.invres = np.empty(self.n)
        self.last_return_code = None

    def _run(self, logcoords, details):
        if self._library is None:
            library_path = self.library_path or _DEFAULT_LIBRARY_PATH
            if library_path is None:
                raise RuntimeError('Call build_kernel(output_dir / "build") first, '
                                   'or pass an existing library_path to FastFFA.')
            self._library = _load_library(library_path)
        coords = np.ascontiguousarray(logcoords,dtype=np.float64)
        if coords.shape != (self.p+1,):
            raise ValueError(f'expected {self.p+1} log(tau),log(R_b) coordinates')
        self.last_return_code = self._library.ffa_evaluate(self.n,self.p,self.t,self.y,
            self.err,self.band,coords,self.work,self.stats,self.mean,self.cov,
            self.invres,int(details))
        return self.last_return_code

    def loglike(self, logcoords):
        self._run(logcoords,False)
        return float(self.stats[0])

    def evaluate(self, logcoords):
        rc = self._run(logcoords,True)
        if rc:
            return (-np.inf,np.full(self.p,np.nan),
                    np.full((self.p,self.p),np.nan),np.inf)
        return (float(self.stats[0]),self.mean.copy(),
                self.cov.reshape(self.p,self.p).copy(),float(self.stats[1]))

    def reconstruct(self, logcoords, ngrid=500):
        rc = self._run(logcoords,False)
        if rc:
            raise ValueError(f'FFA evaluation failed with code {rc}')
        coords = np.asarray(logcoords,dtype=float)
        tau = np.exp(coords[0])
        U = np.exp(coords[1:])[self.band]
        tgrid = np.linspace(self.t[0],self.t[-1],int(ngrid))
        kernel = np.exp(-np.abs(tgrid[:,None]-self.t[None,:])/tau)
        return tgrid, kernel @ (U*self.invres)
