"""线路检测数据：按里程切块的并行可装配复测段查询数据集 + 落盘工单对账。"""

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
from .orders import (
    ACCEPT,
    REVOKE,
    AccessStats,
    BatchResult,
    CrewEvidence,
    OrderError,
    OrderState,
    ViewRun,
    WindowView,
    WorkOrder,
    WorkOrderStore,
)

__all__ = [
    "AssemblyError",
    "Dataset",
    "assemble",
    "build_block",
    "BlockArtifact",
    "Core",
    "CrewEvidence",
    "EvidenceRun",
    "Interval",
    "OrderError",
    "OrderState",
    "Rule",
    "Segment",
    "SegmentHit",
    "State",
    "SoftComponent",
    "Summary",
    "ViewRun",
    "WindowView",
    "WorkOrder",
    "WorkOrderStore",
    "merge",
    "merge_all",
    "ACCEPT",
    "REVOKE",
    "AccessStats",
    "BatchResult",
]
