"""出数回放：按递增截止时刻重建检测快照，并保留每段采用的观测/通告来源。

模块在既有的 :class:`~railway.model.Rule` 与
:class:`~railway.dataset.Dataset` 之上增加一层“出数台账”：

* :class:`Observation`       —— 一条设备采集记录（唯一编号、设备、采集/到达
  时刻、覆盖的非空半开里程区间、「读到」/「明确未读到」）；
* :class:`CalibrationNotice` —— 一条校准通告（唯一编号、设备、公布时刻、
  适用采集时刻的非空半开区间、「可信」/「不可信」）；
* :class:`ReplayLedger`      —— 一次载入固定分块、规则册与两张可乱序提供的
  表，之后可反复按截止时刻出数；
* :class:`Snapshot`          —— 某次出数得到的不可变快照。它本身就是
  :class:`~railway.dataset.Dataset`，可直接用于现有复测段/窗口查询；另带
  :meth:`Snapshot.source_window` 读取同一快照的来源。

出数语义
--------
截止时刻为 ``t`` 时只考虑到达时刻或公布时刻不迟于 ``t`` 的记录。对一条
观测，只看同设备、适用期包含其采集时刻的已公布通告，并以公布时刻最新者
为准（公布时刻相同取通告编号字典序较大者）；无适用通告或该通告不可信，
观测不参加选取。每个位置在所有可信候选中取采集时刻最新者（不按到达
顺序），采集时刻相同则取观测编号字典序较大者。「读到」记为 ``OK``，
「明确未读到」记为 ``MISSING``，无可信候选记为 ``GAP``。

复测段按快照的整条线路判定，与本次改动范围无关。旧快照保留当时的状态、
复测结果和来源；截止时刻只能保持或递增，相同截止时刻返回同一快照，倒退
被拒绝且不改变任何已有快照。

局部性
------
首次构造允许读一遍两张表并建立索引。之后：

* 新通告只回查同设备、采集时刻落在适用期内、且已经到达的观测；
* 新到达的观测只影响它覆盖的里程；
* 只为受影响的里程重选胜出观测；块内未受影响的来源区间与整块产物原样
  复用，复测段仍由全线摘要重新根化，可以跨块变化；
* :meth:`Snapshot.source_window` 只读取与窗口相交的块和来源区间，不扫描
  全线来源或历史台账，也不逐格展开任何里程区间。

调用示例
--------
::

    from railway import Rule
    from railway.replay import (
        CalibrationNotice, Observation, ReplayLedger,
    )

    rule = Rule(1, max_gap=2, min_missing=3)
    ledger = ReplayLedger(
        blocks=[("B0", 0, 50, 1), ("B1", 50, 100, 1)],
        rules={1: rule},
        observations=[
            Observation("O1", "DEV-A", 10, 20, 0, 40, True),
            Observation("O2", "DEV-B", 15, 20, 35, 60, False),
            Observation("O3", "DEV-A", 30, 40, 60, 90, True),
        ],
        notices=[
            CalibrationNotice("C1", "DEV-A", 5, 0, 100, True),
            CalibrationNotice("C2", "DEV-B", 5, 0, 100, True),
        ],
    )

    snap = ledger.advance(20)       # 截止 20：O3 尚未到达
    snap.segments                    # 现有 Dataset 的全线复测段
    snap.query_window(0, 100)        # 现有窗口查询
    snap.source_window(30, 70)       # 同一快照的来源区间
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from .dataset import Dataset, build_block
from .model import Interval, Rule, State
from .monoid import merge_all


class ReplayError(ValueError):
    """台账输入不合法、截止时刻倒退或查询窗口非法，整体拒绝。"""


# ---------------------------------------------------------------------- #
# 严格类型检查（bool 是 int 子类，业务上必须拒绝）
# ---------------------------------------------------------------------- #
def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x)


# ---------------------------------------------------------------------- #
# 输入记录
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class BlockSpec:
    """固定分块中的一块：唯一标识、非空半开里程区间与规则版次。"""

    block_id: str
    start: int
    end: int
    rule_version: int


@dataclass(frozen=True)
class Observation:
    """一条观测：设备在某段里程上的一次采集。

    ``observed=True`` 表示读到数据（状态 ``OK``），``False`` 表示设备
    明确未读到（状态 ``MISSING``）。区间 ``[start,end)`` 非空且落在线路
    内；``arrived_at`` 不得早于 ``acquired_at``。
    """

    obs_id: str
    device: str
    acquired_at: int
    arrived_at: int
    start: int
    end: int
    observed: bool


@dataclass(frozen=True)
class CalibrationNotice:
    """一条校准通告：设备在适用采集期内的结果被判为可信/不可信。

    适用期 ``[applies_start,applies_end)`` 是采集时刻的非空区间，可以
    重叠、也可以追溯到公布时刻之前。``trusted=False`` 表示这段时间的
    采集结果不可信，相关观测不参加选取。
    """

    notice_id: str
    device: str
    published_at: int
    applies_start: int
    applies_end: int
    trusted: bool


@dataclass(frozen=True)
class SourceSpan:
    """快照来源中的一段：状态与胜出观测/通告编号都相同的连续里程。

    ``GAP`` 段没有胜出观测，``obs_id`` 与 ``notice_id`` 均为 ``None``。
    """

    start: int
    end: int
    state: State
    obs_id: Optional[str]
    notice_id: Optional[str]

    @property
    def length(self) -> int:
        return self.end - self.start


def _as_block_spec(raw: Any) -> BlockSpec:
    if isinstance(raw, BlockSpec):
        return raw
    if isinstance(raw, Mapping):
        try:
            return BlockSpec(
                raw["block_id"], raw["start"], raw["end"],
                raw["rule_version"],
            )
        except KeyError as exc:
            raise ReplayError(f"分块说明缺少字段: {exc}") from None
    if isinstance(raw, (tuple, list)) and len(raw) == 4:
        return BlockSpec(raw[0], raw[1], raw[2], raw[3])
    raise ReplayError(f"分块说明类型非法: {raw!r}")


def _as_observation(raw: Any) -> Observation:
    if isinstance(raw, Observation):
        return raw
    if isinstance(raw, Mapping):
        try:
            return Observation(
                raw["obs_id"], raw["device"], raw["acquired_at"],
                raw["arrived_at"], raw["start"], raw["end"],
                raw["observed"],
            )
        except KeyError as exc:
            raise ReplayError(f"观测记录缺少字段: {exc}") from None
    raise ReplayError(f"观测记录类型非法: {raw!r}")


def _as_notice(raw: Any) -> CalibrationNotice:
    if isinstance(raw, CalibrationNotice):
        return raw
    if isinstance(raw, Mapping):
        try:
            return CalibrationNotice(
                raw["notice_id"], raw["device"], raw["published_at"],
                raw["applies_start"], raw["applies_end"], raw["trusted"],
            )
        except KeyError as exc:
            raise ReplayError(f"校准通告缺少字段: {exc}") from None
    raise ReplayError(f"校准通告类型非法: {raw!r}")


# ---------------------------------------------------------------------- #
# 访问计数
# ---------------------------------------------------------------------- #
@dataclass
class ReplayAccessStats:
    """回放过程的局部访问计数（累计值，可用 :meth:`reset` 清零）。

    * ``loaded_observations`` / ``loaded_notices``：首次载入读入的记录数；
    * ``notice_rechecks``：新通告回查的观测数（只含同设备、落在适用期内、
      且已经到达的观测）；
    * ``candidate_lookups``：为确定选取结果检查的候选观测数；
    * ``blocks_rebuilt`` / ``blocks_reused``：推进出数时重建 / 直接复用的块数；
    * ``source_blocks_read`` / ``source_runs_read``：来源小窗读取触碰的块数
      与来源区间数；
    * ``expanded_cells``：里程区间被逐格展开的次数（实现恒为 0）。
    """

    loaded_observations: int = 0
    loaded_notices: int = 0
    notice_rechecks: int = 0
    candidate_lookups: int = 0
    blocks_rebuilt: int = 0
    blocks_reused: int = 0
    source_blocks_read: int = 0
    source_runs_read: int = 0
    expanded_cells: int = 0

    def reset(self) -> None:
        self.loaded_observations = 0
        self.loaded_notices = 0
        self.notice_rechecks = 0
        self.candidate_lookups = 0
        self.blocks_rebuilt = 0
        self.blocks_reused = 0
        self.source_blocks_read = 0
        self.source_runs_read = 0
        self.expanded_cells = 0


# ---------------------------------------------------------------------- #
# 分段常数来源表：值 = 胜出观测（None 表示 GAP）
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Chosen:
    obs_id: str
    notice_id: str
    acquired_at: int
    state: State


SourceKey = Tuple[State, Optional[str], Optional[str]]
_NULL_KEY: SourceKey = (State.GAP, None, None)


def _to_key(val: Optional[_Chosen]) -> SourceKey:
    if val is None:
        return _NULL_KEY
    return (val.state, val.obs_id, val.notice_id)


def _merge_runs(
    starts: Sequence[int],
    ends: Sequence[int],
    vals: Sequence[Optional[_Chosen]],
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[Optional[_Chosen], ...]]:
    """合并相邻同值段并剔除空段。"""
    ms: List[int] = []
    me: List[int] = []
    mv: List[Optional[_Chosen]] = []
    for s, e, v in zip(starts, ends, vals):
        if e <= s:
            continue
        if ms and me[-1] == s and mv[-1] == v:
            me[-1] = e
        else:
            ms.append(s)
            me.append(e)
            mv.append(v)
    return tuple(ms), tuple(me), tuple(mv)


def _merge_ranges(
    ranges: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """合并重叠/相邻的里程段（只做区间算术，不展开里程）。"""
    out: List[Tuple[int, int]] = []
    for a, b in sorted(ranges):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


class _Piece:
    """一个块的来源表：升序、互不重叠、完整覆盖块的分段常数表。

    对象不可变；:meth:`clip` / :meth:`source_runs` / :meth:`intervals`
    都用二分定位后切片，绝不逐格展开。
    """

    __slots__ = ("starts", "ends", "vals")

    def __init__(
        self,
        starts: Sequence[int] = (),
        ends: Sequence[int] = (),
        vals: Sequence[Optional[_Chosen]] = (),
    ) -> None:
        self.starts = tuple(starts)
        self.ends = tuple(ends)
        self.vals = tuple(vals)

    @classmethod
    def null(cls, start: int, end: int) -> "_Piece":
        return cls((start,), (end,), (None,))

    def clip(self, start: int, end: int
             ) -> List[Tuple[int, int, Optional[_Chosen]]]:
        """裁到 ``[start,end)`` 的段列表（二分定位，只触碰相交段）。"""
        out: List[Tuple[int, int, Optional[_Chosen]]] = []
        i = bisect.bisect_right(self.ends, start)
        n = len(self.starts)
        while i < n:
            s = self.starts[i]
            if s >= end:
                break
            e = self.ends[i]
            lo, hi = (s if s > start else start), (e if e < end else end)
            if hi > lo:
                out.append((lo, hi, self.vals[i]))
            i += 1
        return out

    def intervals(self, start: int, end: int) -> List[Interval]:
        """把 ``[start,end)`` 上的来源压成建块用 :class:`Interval` 表。"""
        return [
            Interval(lo, hi, State.GAP if v is None else v.state)
            for lo, hi, v in self.clip(start, end)
        ]

    def source_runs(self, start: int, end: int
                    ) -> List[Tuple[int, int, SourceKey]]:
        """裁到 ``[start,end)`` 的来源区间（相邻同键合并）。"""
        out: List[Tuple[int, int, SourceKey]] = []
        for lo, hi, v in self.clip(start, end):
            key = _to_key(v)
            if out and out[-1][1] == lo and out[-1][2] == key:
                out[-1] = (out[-1][0], hi, key)
            else:
                out.append((lo, hi, key))
        return out

    def spliced(self, ranges: Sequence[Tuple[int, int]],
                replacements: Sequence["_Piece"]) -> "_Piece":
        """把 ``ranges``（升序、互不相交）换成对应 replacement 的新表。

        范围之外的段原样保留，因此局部推进不会重建整块来源；替换后相邻
        同值段照常合并。
        """
        if not ranges:
            return self
        lo, hi = self.starts[0], self.ends[-1]
        starts: List[int] = []
        ends: List[int] = []
        vals: List[Optional[_Chosen]] = []
        cursor = lo
        for (a, b), rep in zip(ranges, replacements):
            for s, e, v in self.clip(cursor, a):
                starts.append(s)
                ends.append(e)
                vals.append(v)
            for s, e, v in rep.clip(a, b):
                starts.append(s)
                ends.append(e)
                vals.append(v)
            cursor = b
        for s, e, v in self.clip(cursor, hi):
            starts.append(s)
            ends.append(e)
            vals.append(v)
        ms, me, mv = _merge_runs(starts, ends, vals)
        return _Piece(ms, me, mv)


# ---------------------------------------------------------------------- #
# 出数快照
# ---------------------------------------------------------------------- #
class Snapshot(Dataset):
    """某一次出数的不可变快照（本身是 :class:`Dataset`）。

    * 现有复测查询（``segments`` / ``qualified_segments`` /
      ``query_window`` 等）全部继承自 :class:`Dataset`，按整条线路判定；
    * :meth:`source_window` 读取与窗口相交块上的来源，按里程升序返回，
      只合并“状态与两个编号都相同”的相邻区间；
    * 块产物与来源表都不可变，后续推进不会改写旧快照。
    """

    __slots__ = ("cutoff", "_pieces", "_access")

    def __init__(self, line_start, line_end, rules, blocks, root, segments,
                 cutoff, pieces, access=None):
        super().__init__(line_start, line_end, rules, blocks, root, segments)
        self.cutoff = cutoff
        self._pieces: Tuple[_Piece, ...] = tuple(pieces)
        self._access = access

    # ---------------- 来源读取 ---------------- #
    def source_window(self, start: int, end: int) -> Tuple[SourceSpan, ...]:
        """读取 ``[start,end)`` 上的来源区间。

        只访问与窗口相交的块；不扫描全线来源、不读取历史台账，也不逐格
        展开任何里程区间。
        """
        if not (_is_int(start) and _is_int(end)):
            raise ReplayError("来源窗口端点必须是整数")
        if end <= start:
            raise ReplayError("来源窗口必须为正长度半开区间")
        if start < self.line_start or end > self.line_end:
            raise ReplayError(
                f"来源窗口 [{start},{end}) 越出线路 "
                f"[{self.line_start},{self.line_end})"
            )
        lo = max(bisect.bisect_right(self._block_ends, start) - 1, 0)
        hi = bisect.bisect_left(self._block_starts, end)
        runs: List[Tuple[int, int, SourceKey]] = []
        blocks_read = 0
        for idx in range(lo, min(hi, len(self._blocks))):
            bs, be = self._block_starts[idx], self._block_ends[idx]
            if be <= start or bs >= end:
                continue
            blocks_read += 1
            for s, e, key in self._pieces[idx].source_runs(
                    max(start, bs), min(end, be)):
                if runs and runs[-1][1] == s and runs[-1][2] == key:
                    runs[-1] = (runs[-1][0], e, key)
                else:
                    runs.append((s, e, key))
        if self._access is not None:
            self._access.source_blocks_read += blocks_read
            self._access.source_runs_read += len(runs)
        return tuple(
            SourceSpan(s, e, key[0], key[1], key[2])
            for s, e, key in runs
        )

    def source_at(self, mile: int) -> SourceSpan:
        """读取单点来源（等价于 ``source_window(mile, mile+1)[0]``）。"""
        if not _is_int(mile) or not (self.line_start <= mile < self.line_end):
            raise ReplayError(f"里程 {mile!r} 不在线路内")
        return self.source_window(mile, mile + 1)[0]

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"Snapshot(cutoff={self.cutoff}, "
            f"[{self.line_start},{self.line_end}), "
            f"{len(self._blocks)} 块, {len(self._segments)} 段)"
        )


# ---------------------------------------------------------------------- #
# 出数台账
# ---------------------------------------------------------------------- #
class ReplayLedger:
    """固定分块 + 规则册 + 观测/通告两张表构成的出数台账。

    构造时整体校验并读一遍两张表，任意一条记录非法即抛
    :class:`ReplayError`，不产生部分台账。之后 :meth:`advance` 按递增
    截止时刻出数，只处理新到达的观测与新公布的通告。
    """

    def __init__(
        self,
        blocks: Iterable[Any],
        rules: Mapping[int, Rule],
        observations: Iterable[Any],
        notices: Iterable[Any],
        *,
        line_start: Optional[int] = None,
        line_end: Optional[int] = None,
    ) -> None:
        # ---------- 规则册 ---------- #
        if not isinstance(rules, Mapping):
            raise ReplayError("规则册必须是 {版次: Rule} 映射")
        rule_book: Dict[int, Rule] = {}
        for version, rule in rules.items():
            if not _is_int(version) or not isinstance(rule, Rule):
                raise ReplayError(f"规则册条目非法: {version!r} -> {rule!r}")
            if rule.version != version:
                raise ReplayError(
                    f"规则册键 {version} 与 Rule.version={rule.version} 不一致"
                )
            rule_book[version] = rule
        if not rule_book:
            raise ReplayError("规则册不能为空")

        # ---------- 固定分块：有序、互斥、完整覆盖，一块一个版次 ---------- #
        specs: List[BlockSpec] = []
        for raw in blocks:
            spec = _as_block_spec(raw)
            if not _is_str(spec.block_id):
                raise ReplayError("块标识必须是非空字符串")
            if not (_is_int(spec.start) and _is_int(spec.end)):
                raise ReplayError(f"块 {spec.block_id} 里程端点必须是整数")
            if spec.end <= spec.start:
                raise ReplayError(f"块 {spec.block_id} 里程范围非法")
            if not _is_int(spec.rule_version) \
                    or spec.rule_version not in rule_book:
                raise ReplayError(
                    f"块 {spec.block_id} 的规则版次 "
                    f"{spec.rule_version!r} 不在规则册中"
                )
            specs.append(spec)
        if not specs:
            raise ReplayError("没有任何分块")
        specs.sort(key=lambda b: (b.start, b.end))
        if len({b.block_id for b in specs}) != len(specs):
            raise ReplayError("块标识重复")
        inferred_start, inferred_end = specs[0].start, specs[-1].end
        if line_start is None:
            line_start = inferred_start
        if line_end is None:
            line_end = inferred_end
        if not (_is_int(line_start) and _is_int(line_end)):
            raise ReplayError("线路端点必须是整数")
        if line_end <= line_start:
            raise ReplayError("线路范围必须是正长度半开区间")
        if specs[0].start != line_start or specs[-1].end != line_end:
            raise ReplayError(
                f"分块未完整覆盖线路 [{line_start},{line_end})："
                f"实际 [{specs[0].start},{specs[-1].end})"
            )
        for prev, cur in zip(specs, specs[1:]):
            if prev.end != cur.start:
                raise ReplayError(
                    f"分块接缝错误：{prev.block_id}[{prev.start},{prev.end}) 与 "
                    f"{cur.block_id}[{cur.start},{cur.end}) 之间"
                    f"{'重叠' if cur.start < prev.end else '缺块'}"
                )
        spec_starts = [b.start for b in specs]
        spec_ends = [b.end for b in specs]

        # ---------- 观测表：全量校验 + 设备/时间/区间索引 ---------- #
        obs_list: List[Observation] = []
        obs_ids: set = set()
        for raw in observations:
            obs = _as_observation(raw)
            if not _is_str(obs.obs_id):
                raise ReplayError("观测编号必须是非空字符串")
            if obs.obs_id in obs_ids:
                raise ReplayError(f"观测编号重复: {obs.obs_id}")
            obs_ids.add(obs.obs_id)
            if not _is_str(obs.device):
                raise ReplayError(f"观测 {obs.obs_id} 的设备号必须是非空字符串")
            if not (_is_int(obs.acquired_at) and _is_int(obs.arrived_at)):
                raise ReplayError(f"观测 {obs.obs_id} 的时刻必须是整数")
            if obs.arrived_at < obs.acquired_at:
                raise ReplayError(
                    f"观测 {obs.obs_id} 到达时刻 {obs.arrived_at} "
                    f"早于采集时刻 {obs.acquired_at}"
                )
            if not (_is_int(obs.start) and _is_int(obs.end)):
                raise ReplayError(f"观测 {obs.obs_id} 里程端点必须是整数")
            if obs.end <= obs.start:
                raise ReplayError(f"观测 {obs.obs_id} 里程区间必须非空")
            if obs.start < line_start or obs.end > line_end:
                raise ReplayError(
                    f"观测 {obs.obs_id} 区间 [{obs.start},{obs.end}) "
                    f"越出线路 [{line_start},{line_end})"
                )
            if not isinstance(obs.observed, bool):
                raise ReplayError(
                    f"观测 {obs.obs_id} 的「读到」标记必须是布尔值"
                )
            obs_list.append(obs)
        obs_by_device: Dict[str, List[Observation]] = {}
        for obs in obs_list:
            obs_by_device.setdefault(obs.device, []).append(obs)
        for group in obs_by_device.values():
            group.sort(key=lambda o: (o.acquired_at, o.obs_id))

        # 每个块可能胜出的候选观测（与块区间相交的观测；区间不逐格展开）。
        obs_by_block: List[List[Observation]] = [[] for _ in specs]
        for obs in obs_list:
            lo = bisect.bisect_right(spec_ends, obs.start)
            hi = bisect.bisect_left(spec_starts, obs.end)
            for idx in range(lo, min(hi, len(specs))):
                obs_by_block[idx].append(obs)

        # ---------- 通告表：全量校验 + 设备索引 ---------- #
        notice_list: List[CalibrationNotice] = []
        notice_ids: set = set()
        for raw in notices:
            notice = _as_notice(raw)
            if not _is_str(notice.notice_id):
                raise ReplayError("通告编号必须是非空字符串")
            if notice.notice_id in notice_ids:
                raise ReplayError(f"通告编号重复: {notice.notice_id}")
            notice_ids.add(notice.notice_id)
            if not _is_str(notice.device):
                raise ReplayError(
                    f"通告 {notice.notice_id} 的设备号必须是非空字符串"
                )
            if not _is_int(notice.published_at):
                raise ReplayError(
                    f"通告 {notice.notice_id} 公布时刻必须是整数"
                )
            if not (_is_int(notice.applies_start)
                    and _is_int(notice.applies_end)):
                raise ReplayError(
                    f"通告 {notice.notice_id} 适用时刻端点必须是整数"
                )
            if notice.applies_end <= notice.applies_start:
                raise ReplayError(
                    f"通告 {notice.notice_id} 适用期必须是非空半开区间"
                )
            if not isinstance(notice.trusted, bool):
                raise ReplayError(
                    f"通告 {notice.notice_id} 的可信标记必须是布尔值"
                )
            notice_list.append(notice)
        notice_by_device: Dict[str, List[CalibrationNotice]] = {}
        for notice in notice_list:
            notice_by_device.setdefault(notice.device, []).append(notice)
        for group in notice_by_device.values():
            group.sort(key=lambda n: (n.published_at, n.notice_id))

        # ---------- 校验全部通过，落定台账状态 ---------- #
        self._rules = rule_book
        self._specs = tuple(specs)
        self._spec_starts = spec_starts
        self._spec_ends = spec_ends
        self.line_start = line_start
        self.line_end = line_end

        self._obs_by_device = obs_by_device
        self._obs_by_block = obs_by_block
        self._notices_by_device = notice_by_device
        # 每设备已公布通告的公布时刻表，供按截止时刻二分定位后逆序回查。
        self._notice_pub_by_device: Dict[str, List[int]] = {
            device: [n.published_at for n in group]
            for device, group in notice_by_device.items()
        }

        self.access = ReplayAccessStats()
        self.access.loaded_observations = len(obs_list)
        self.access.loaded_notices = len(notice_list)

        self._pending_obs = sorted(
            obs_list, key=lambda o: (o.arrived_at, o.obs_id)
        )
        self._pending_notice = sorted(
            notice_list, key=lambda n: (n.published_at, n.notice_id)
        )
        self._obs_cursor = 0
        self._notice_cursor = 0

        self._pieces: List[_Piece] = [
            _Piece.null(spec.start, spec.end) for spec in specs
        ]
        self._last_blocks: List[Any] = []
        self._last_cutoff: Optional[int] = None
        self._snapshot: Optional[Snapshot] = None
        self._history: List[Snapshot] = []

    # ---------------- 只读属性 ---------------- #
    @property
    def rules(self) -> Mapping[int, Rule]:
        return dict(self._rules)

    @property
    def blocks(self) -> Tuple[BlockSpec, ...]:
        return self._specs

    @property
    def cutoff(self) -> Optional[int]:
        return self._last_cutoff

    @property
    def snapshot(self) -> Optional[Snapshot]:
        return self._snapshot

    @property
    def history(self) -> Tuple[Snapshot, ...]:
        """按截止时刻升序排列的全部已出快照（旧快照不会被改写）。"""
        return tuple(self._history)

    def snapshot_at(self, cutoff: int) -> Snapshot:
        """取某个已出截止时刻的快照（用于回看旧出数的来源）。"""
        if not _is_int(cutoff):
            raise ReplayError("截止时刻必须是整数")
        for snap in reversed(self._history):
            if snap.cutoff == cutoff:
                return snap
        raise ReplayError(f"没有截止时刻 {cutoff} 的出数快照")

    # ---------------- 选取规则 ---------------- #
    def _effective_notice(self, obs: Observation,
                          cutoff: int) -> Optional[CalibrationNotice]:
        """按截止时刻取该观测生效的通告；观测不参加选取时返回 None。

        只看同设备、已公布（``published_at <= cutoff``）、适用期包含采集
        时刻的通告，按 ``(公布时刻, 编号)`` 取最大者；该通告不可信则观测
        不参加。通告按 ``(公布时刻, 编号)`` 升序存放，先用公布时刻二分
        定位，再向前扫描到首个适用通告即可停止。
        """
        group = self._notices_by_device.get(obs.device)
        if not group:
            return None
        pub = self._notice_pub_by_device[obs.device]
        # 只回查已公布的通告（published_at <= cutoff），逆序即可提前停止。
        k = bisect.bisect_right(pub, cutoff) - 1
        while k >= 0:
            notice = group[k]
            if notice.applies_start <= obs.acquired_at < notice.applies_end:
                return notice if notice.trusted else None
            k -= 1
        return None

    # ---------------- 推进出数 ---------------- #
    def advance(self, cutoff: int) -> Snapshot:
        """推进到截止时刻 ``cutoff``，返回该时刻的检测快照。

        截止时刻只能保持或递增：相同截止时刻返回同一快照对象；倒退抛
        :class:`ReplayError`，且不改变任何已有快照或台账状态。允许跳过
        中间时刻，结果与逐时刻推进一致。
        """
        if not _is_int(cutoff):
            raise ReplayError("截止时刻必须是整数")
        if self._last_cutoff is not None and cutoff < self._last_cutoff:
            raise ReplayError(
                f"截止时刻不能倒退：当前 {self._last_cutoff}，收到 {cutoff}"
            )
        if self._last_cutoff == cutoff:
            assert self._snapshot is not None
            return self._snapshot

        # 1) 新公布的通告：只回查同设备、采集时刻落在适用期内、已经到达
        #    的观测；只有这些观测的里程需要重选。
        affected_spans: List[Tuple[int, int]] = []
        nc = self._notice_cursor
        pending_notices = self._pending_notice
        while nc < len(pending_notices) and \
                pending_notices[nc].published_at <= cutoff:
            notice = pending_notices[nc]
            nc += 1
            group = self._obs_by_device.get(notice.device, ())
            lo = bisect.bisect_left(
                group, notice.applies_start, key=lambda o: o.acquired_at)
            hi = bisect.bisect_left(
                group, notice.applies_end, key=lambda o: o.acquired_at)
            for obs in group[lo:hi]:
                if obs.arrived_at > cutoff:
                    continue
                self.access.notice_rechecks += 1
                affected_spans.append((obs.start, obs.end))

        # 2) 新到达的观测：只影响其区间覆盖的里程。
        oc = self._obs_cursor
        pending_obs = self._pending_obs
        while oc < len(pending_obs) and pending_obs[oc].arrived_at <= cutoff:
            obs = pending_obs[oc]
            oc += 1
            affected_spans.append((obs.start, obs.end))

        # 3) 只为受影响观测的里程重选胜出观测。按（块, 里程段）聚合，块内
        #    未受影响的里程原样保留，不重建整块来源。
        ranges_by_block: Dict[int, List[Tuple[int, int]]] = {}
        for a, b in affected_spans:
            for idx in self._blocks_for(a, b):
                spec = self._specs[idx]
                lo = max(a, spec.start)
                hi = min(b, spec.end)
                if hi > lo:
                    ranges_by_block.setdefault(idx, []).append((lo, hi))

        rebuild: set = set()
        for idx in sorted(ranges_by_block):
            spec = self._specs[idx]
            ranges = _merge_ranges(ranges_by_block[idx])
            replacements = []
            for a, b in ranges:
                candidates = [
                    obs for obs in self._obs_by_block[idx]
                    if obs.arrived_at <= cutoff
                    and obs.start < b and obs.end > a
                ]
                self.access.candidate_lookups += len(candidates)
                replacements.append(self._reselect(a, b, candidates, cutoff))
            old = self._pieces[idx]
            piece = old.spliced(ranges, replacements)
            if piece.starts != old.starts or piece.ends != old.ends \
                    or piece.vals != old.vals:
                self._pieces[idx] = piece
                rebuild.add(idx)

        # 4) 拼装快照：重建受影响块，复用其余块产物；复测段按整条线路重判。
        blocks: List[Any] = []
        first = not self._last_blocks
        for idx, spec in enumerate(self._specs):
            if first or idx in rebuild:
                blocks.append(build_block(
                    spec.block_id, spec.start, spec.end,
                    self._pieces[idx].intervals(spec.start, spec.end),
                    self._rules[spec.rule_version],
                ))
                self.access.blocks_rebuilt += 1
            else:
                blocks.append(self._last_blocks[idx])
                self.access.blocks_reused += 1
        root = merge_all([b.summary for b in blocks])
        snap = Snapshot(
            line_start=self.line_start,
            line_end=self.line_end,
            rules=self._rules,
            blocks=blocks,
            root=root,
            segments=root.finalize(),
            cutoff=cutoff,
            pieces=tuple(self._pieces),
            access=self.access,
        )
        self._last_blocks = blocks
        self._last_cutoff = cutoff
        self._snapshot = snap
        self._history.append(snap)
        self._obs_cursor = oc
        self._notice_cursor = nc
        return snap

    # ---------------- 局部工具 ---------------- #
    def _blocks_for(self, start: int, end: int) -> List[int]:
        """与区间 ``[start,end)`` 相交的块下标（只按块边界定位）。"""
        lo = bisect.bisect_right(self._spec_ends, start)
        hi = bisect.bisect_left(self._spec_starts, end)
        return list(range(lo, min(hi, len(self._specs))))

    def _reselect(self, start: int, end: int,
                  candidates: Sequence[Observation],
                  cutoff: int) -> _Piece:
        """在 ``[start,end)`` 内按“候选端点”切分，为每段取胜出观测。

        切点只来自区间端点与候选观测区间端点（至多 ``2*候选数+1`` 段），
        绝不逐格展开。同刻比较用 ``(采集时刻, 观测编号)`` 取较大者。
        """
        cuts = {start, end}
        for obs in candidates:
            if start < obs.start < end:
                cuts.add(obs.start)
            if start < obs.end < end:
                cuts.add(obs.end)
        pts = sorted(cuts)
        starts: List[int] = []
        ends: List[int] = []
        vals: List[Optional[_Chosen]] = []
        for a, b in zip(pts, pts[1:]):
            if b <= a:
                continue
            best: Optional[_Chosen] = None
            best_key: Optional[Tuple[int, str]] = None
            for obs in candidates:
                if obs.start > a or obs.end < b:
                    continue
                notice = self._effective_notice(obs, cutoff)
                if notice is None:
                    continue
                key = (obs.acquired_at, obs.obs_id)
                if best_key is None or key > best_key:
                    best_key = key
                    best = _Chosen(
                        obs.obs_id,
                        notice.notice_id,
                        obs.acquired_at,
                        State.OK if obs.observed else State.MISSING,
                    )
            starts.append(a)
            ends.append(b)
            vals.append(best)
        ms, me, mv = _merge_runs(starts, ends, vals)
        return _Piece(ms, me, mv)
