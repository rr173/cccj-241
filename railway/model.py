"""领域模型：里程格、状态区间、规则版次、块产物、段与查询命中。

里程格采用整数、半开区间 ``[start, end)``，全线按里程升序排列。
三种状态：

* ``OK``      —— 合格测量，永远是硬切口，不允许被复测段跨过；
* ``MISSING`` —— 确实缺测，复测段长度门槛只统计这种格；
* ``GAP``     —— 缺报，只有被两侧确实缺测夹住、且整条连续缺报带长度不超过
  所在规则版次的 ``max_gap`` 时，才允许作为“短缺报带”跨过。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class State(str, Enum):
    OK = "OK"
    MISSING = "MISSING"
    GAP = "GAP"


@dataclass(frozen=True)
class Rule:
    """一个规则版次（按危险等级编号）的业务参数。

    ``max_gap``    : 该版允许跨过的最大连续缺报长度（格）；
    ``min_missing``: 一个复测段要求达到的“确实缺测格数”门槛。
    段长度门槛只看 MISSING 格数，跨过的 GAP 不计入。
    """

    version: int
    max_gap: int
    min_missing: int

    def __post_init__(self) -> None:
        if self.max_gap < 0 or self.min_missing < 0:
            raise ValueError("规则阈值不能为负")


@dataclass(frozen=True)
class Interval:
    """块内的一条状态区间，半开 ``[start, end)``。"""

    start: int
    end: int
    state: State

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Core:
    """软组件中的“缺测核”：首尾都是 MISSING。

    核内可以包含已经在当前上下文验证为可跨过的短缺报带（GAP），这些 GAP
    以区间形式保留为证据，但不计入 ``missing`` 门槛长度。核外的边缘 GAP
    留在 :class:`SoftComponent` 上，等待块/区域拼接后再按整条带判定。
    """

    start: int
    end: int
    missing: int                  # 核内确实缺测格数（不含 GAP）
    gaps: tuple = ()              # tuple[Interval, ...] 核内可跨过的短缺报


@dataclass(frozen=True)
class Segment:
    """全线路上的候选复测段（未判门槛的原始连续缺测段）。

    段边界以两侧硬切口（合格测量/超长缺报/换版切口/线路端点）为界，
    内部只含 MISSING 与被跨过的短缺报，首尾格必为 MISSING。
    """

    start: int
    end: int
    missing: int
    version: int  # 段内所有格同属一个规则版次

    @property
    def length(self) -> int:
        return self.end - self.start


class BlockArtifact:
    """可独立构建、可交付的块产物。

    块产物自描述、不可变，装配端仅凭它就能核对全线接缝，无需原始里程明细。
    内部区间表的读取通过 :meth:`read_intervals` 进行，并带计数器，
    以便测试证明“更正不扫描未更正块的内部区间”。
    """

    __slots__ = (
        "block_id",
        "start",
        "end",
        "rule",
        "_intervals",
        "summary",
        "_interval_reads",
    )

    def __init__(self, block_id, start, end, rule, intervals, summary):
        self.block_id = block_id
        self.start = start
        self.end = end
        self.rule = rule
        object.__setattr__(self, "_intervals", tuple(intervals))
        self.summary = summary
        object.__setattr__(self, "_interval_reads", 0)

    @property
    def intervals(self):
        """读取块的完整内部区间表（计数，更正流程不允许对未更正块调用）。"""
        object.__setattr__(self, "_interval_reads", self._interval_reads + 1)
        return self._intervals

    @property
    def interval_reads(self) -> int:
        return self._interval_reads

    @property
    def length(self) -> int:
        return self.end - self.start

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"BlockArtifact({self.block_id!r}, [{self.start},{self.end}), "
            f"v{self.rule.version}, {len(self._intervals)} 区间)"
        )


@dataclass(frozen=True)
class EvidenceRun:
    """窗口内一条分段证据（状态区间在窗口/段内的裁切形态）。"""

    start: int
    end: int
    state: State
    crossed: bool  # 仅对 GAP：是否为被跨过的短缺报带
    gap_band_len: Optional[int]  # 仅对 GAP：整条连续缺报带的全长（可能跨块）


@dataclass(frozen=True)
class SegmentHit:
    """读取窗口与一个全线复测段的相交结果。"""

    segment: Segment          # 全线路上的完整段（门槛按它判定，绝不被窗口截断）
    qualified: bool           # 按全线段的 MISSING 数是否达到门槛
    clip_start: int           # 与窗口相交后的裁切范围
    clip_end: int
    clip_missing: int         # 裁切范围内的确实缺测格数
    evidence: tuple           # tuple[EvidenceRun, ...]，按里程升序
