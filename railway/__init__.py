"""线路检测数据：按里程切块的并行可装配复测段查询数据集。"""

from .dataset import AssemblyError, Dataset, assemble, build_block
from .model import (
    BlockArtifact,
    Core,
    EvidenceRun,
    Interval,
    Rule,
    Segment,
    SegmentHit,
    State,
)
from .monoid import SoftComponent, Summary, merge, merge_all

__all__ = [
    "AssemblyError",
    "Dataset",
    "assemble",
    "build_block",
    "BlockArtifact",
    "Core",
    "EvidenceRun",
    "Interval",
    "Rule",
    "Segment",
    "SegmentHit",
    "State",
    "SoftComponent",
    "Summary",
    "merge",
    "merge_all",
]
