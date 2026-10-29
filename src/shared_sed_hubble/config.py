# 路径与配置接口：RunConfig 对应共享 SED 源拟合；ComparisonConfig 对应冻结距离的四情景比较。
# PyYAML 只解析配置，dataclass 冻结属性便于跨进程传递；源名/路径检查用于防止混用输出。
"""Paths and source selection for the frozen shared-distance model."""
from dataclasses import dataclass, asdict
from pathlib import Path
import re
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("defaults.yaml")

# 携带项目根、结果目录、源顺序、上游结果相对路径和最大并行数；不包含拟合状态。
@dataclass(frozen=True)
class RunConfig:
    project_root: Path
    output_dir: Path
    sources: tuple[str, ...]
    result_subdir: Path
    parallel: int = 3

    # 把 Path/tuple 转成 JSON 可序列化字段，保存精确路径和源顺序以便复查。
    def to_dict(self):
        data = asdict(self)
        for key in ("project_root", "output_dir", "result_subdir"):
            data[key] = str(data[key])
        data["sources"] = list(self.sources)
        return data

    # 从控制进程写出的 JSON 字段恢复 Path/tuple；工作进程使用同一份配置。
    @classmethod
    def from_dict(cls, data):
        return cls(Path(data["project_root"]), Path(data["output_dir"]),
                   tuple(data["sources"]), Path(data["result_subdir"]),
                   int(data.get("parallel", 3)))


# 按命令行覆盖 YAML，并拒绝同一 AGN 多历元重复计数及越界的结果相对路径。
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


DEFAULT_COMPARISON_CONFIG = Path(__file__).with_name("comparison.yaml")


# 携带冻结距离摘要、外部红移快照、四情景网格设置；与源级拟合配置分开。
@dataclass(frozen=True)
class ComparisonConfig:
    """Frozen distance inputs and settings for the four-case comparison."""
    project_root: Path
    output_dir: Path
    distance_summary: Path
    redshift_inputs: Path
    sources: tuple[str, ...]
    settings: dict
    baseline_saved_z: bool = True
    include_joint: bool = False

    # 把 Path/tuple 转成 JSON 可序列化字段，保存精确路径和源顺序以便复查。
    def to_dict(self):
        data = asdict(self)
        for key in ("project_root", "output_dir", "distance_summary", "redshift_inputs"):
            data[key] = str(data[key])
        data["sources"] = list(self.sources)
        return data


# 独立解析比较配置；输出目录不能包含输入表，防止比较阶段覆盖冻结距离或红移证据。
def load_comparison_config(path=None, *, root=None, output=None):
    path = Path(path) if path is not None else DEFAULT_COMPARISON_CONFIG
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root = Path(root or data.get("project_root") or PROJECT_ROOT).expanduser().resolve()

    def resolve(value):
        value = Path(value).expanduser()
        return (value if value.is_absolute() else root / value).resolve()

    sources = tuple(data["sources"])
    ids = [re.sub(r"_[a-z]$", "", name, flags=re.I).casefold() for name in sources]
    if not sources or len(set(ids)) != len(ids):
        raise ValueError("Select distinct AGN, not repeated epochs of one source")
    if any(not re.fullmatch(r"[A-Za-z0-9_+.-]+", name) for name in sources):
        raise ValueError("Source identifiers must be directory names")
    distances = resolve(data["distance_summary"])
    redshifts = resolve(data["redshift_inputs"])
    out = resolve(output or data["output_dir"])
    for input_path in (distances, redshifts):
        if out == input_path.parent or out in input_path.parents:
            raise ValueError("Comparison output must not contain its input files")
    return ComparisonConfig(
        root, out, distances, redshifts, sources, dict(data.get("settings") or {}),
        bool(data.get("baseline_saved_z", True)), bool(data.get("include_joint", False)),
    )
