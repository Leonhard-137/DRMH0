"""Paths and source selection for the frozen shared-distance model."""
from dataclasses import dataclass, asdict
from pathlib import Path
import re
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("defaults.yaml")

@dataclass(frozen=True)
class RunConfig:
    project_root: Path
    output_dir: Path
    sources: tuple[str, ...]
    result_subdir: Path
    parallel: int = 3

    def to_dict(self):
        data = asdict(self)
        for key in ("project_root", "output_dir", "result_subdir"):
            data[key] = str(data[key])
        data["sources"] = list(self.sources)
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(Path(data["project_root"]), Path(data["output_dir"]),
                   tuple(data["sources"]), Path(data["result_subdir"]),
                   int(data.get("parallel", 3)))


def load_config(path=None, *, root=None, output=None, parallel=None):
    path = Path(path) if path is not None else DEFAULT_CONFIG
    data = yaml.safe_load(path.read_text()) or {}
    root = Path(root or data.get("project_root") or PROJECT_ROOT).expanduser().resolve()
    out = Path(output or data.get("output_dir", "output/shared_sed_hubble")).expanduser()
    out = (out if out.is_absolute() else root / out).resolve()
    sources = tuple(data["sources"])
    ids = [re.sub(r"_[a-z]$", "", name, flags=re.I).casefold() for name in sources]
    if not sources or len(set(ids)) != len(ids):
        raise ValueError("Select distinct AGN, not repeated epochs of one source")
    if any(not re.fullmatch(r"[A-Za-z0-9_+.-]+", name) for name in sources):
        raise ValueError("Source identifiers must be directory names")
    result = Path(data["result_subdir"])
    if result.is_absolute() or ".." in result.parts:
        raise ValueError("result_subdir must be relative to each source")
    workers = int(parallel if parallel is not None else data.get("parallel", 3))
    if workers < 1:
        raise ValueError("parallel must be positive")
    return RunConfig(root, out, sources, result, workers)
