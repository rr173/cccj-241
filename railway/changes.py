"""跨版复测变化清单：按物理位置对照两版检测，而不是按相等里程号。

适用场景：线路改线或重编里程后，养护人员需要核对早晚两次检测之间，
同一物理位置上“应复测”要求的新增 / 解除 / 失联。

输入（见 :meth:`VersionChain.build`）

* 从早到晚的线路版本顺序，每版有唯一标识与整数半开线路范围；
* 每对相邻版本的交接表。每条记录把旧版一个非空整数半开区间对应到
  新版一个**等长**区间，方向可同向或反向。例如反向交接
  ``[10,13) -> [50,53)`` 时，10 对应 52、11 对应 51、12 对应 50。
  每条记录两侧都须落在相应版本范围内，同一张表在任一侧不得重叠；
  允许任意输入顺序和未覆盖位置；空表表示该次交接没有延续位置。
* 早晚两份已装配的 :class:`~railway.dataset.Dataset`，其范围必须与
  版本链首尾版本完全一致。

查询（见 :meth:`VersionChain.change_list`）：使用者选早版或晚版作为
查看坐标，并给出版本内一个非空半开窗口。

语义要点（均由测试守护）

* 只有经过**每次**相邻交接都找得到对应的位置才能比较早晚状态；
  沿途遇到断点就没有对端，后续版本即使复用相同号码也不能接回来。
* 每份检测中，“应复测”只指达标段内实际 ``MISSING`` 的位置；段是否
  达标按该版本的**全段**判定，不按窗口或交接边界重新判。
* 清单只列三类：晚版才应复测（新增 :attr:`ChangeCategory.NEW`）、
  只在早版应复测（解除 :attr:`ChangeCategory.REMOVED`）、查看版应
  复测但无对端（未配对 :attr:`ChangeCategory.UNPAIRED`）。
* 未配对行注明从查看版出发第一次找不到对应的相邻版本对（按历史
  顺序的两版标识），对端范围留空；应复测一侧附完整复测段范围、
  规则版次和全段缺测格数。
* 类别、段证据和断点相同的相邻行合并；配对行合起来必须仍是同向或
  反向的一个连续对应。合法拆分交接记录不改变结果。

局部性（不逐格展开线路）

* 建链时一次性校验全部交接数据，并把每张表压成两侧有序的区间索引；
* 小窗查询只在与窗口相交的交接记录上做区间算术（二分 + 顺序扫描
  相交片段），窗口被改线分送到彼此很远的区间时，也只对这些远端
  小片段调用 :meth:`Dataset.query_window`，不读入中间无关里程，
  访问量可由 :attr:`VersionChain.access_stats` 读取。
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Mapping, Optional, Sequence, Tuple


class ChainError(ValueError):
    """版本链 / 交接表 / 端点数据集 / 查询窗口不合法，整体拒绝。"""


class Direction(str, Enum):
    """查看版 -> 对端版的净方向（反向次数的奇偶性）。"""

    FORWARD = "FORWARD"    # 同向：里程增减方向一致
    REVERSED = "REVERSED"  # 反向：里程增减方向相反


class ChangeCategory(str, Enum):
    """变化清单的三类条目。"""

    NEW = "NEW"                  # 已配对，晚版才应复测（新增）
    REMOVED = "REMOVED"          # 已配对，只在早版应复测（解除）
    UNPAIRED = "UNPAIRED"        # 查看版应复测但无对端（未配对）


@dataclass(frozen=True)
class LineVersion:
    """版本链中的一版：唯一标识与整数半开线路范围 ``[start, end)``。"""

    version_id: str
    start: int
    end: int


@dataclass(frozen=True)
class HandoverRecord:
    """一条相邻版本交接记录：旧版非空区间 -> 新版等长区间。

    ``reversed=False``（同向）：``old_start+k`` 对应 ``new_start+k``；
    ``reversed=True``（反向）：``old_start+k`` 对应
    ``new_end-1-k``，即 ``old [10,13) -> new [50,53)`` 时
    10↔52、11↔51、12↔50。
    """

    old_start: int
    old_end: int
    new_start: int
    new_end: int
    reversed: bool = False


@dataclass(frozen=True)
class DueEvidence:
    """应复测一侧的回查证据：完整复测段范围、规则版次与全段缺测格数。"""

    segment_start: int
    segment_end: int
    rule_version: int
    segment_missing: int


@dataclass(frozen=True)
class ChangeRow:
    """变化清单中的一行（范围均为查看版坐标，半开、升序、互不相交）。"""

    start: int
    end: int
    category: ChangeCategory
    # 配对行：对端范围与净方向；未配对行对端范围为 None。
    peer_start: Optional[int]
    peer_end: Optional[int]
    direction: Optional[Direction]
    # 未配对行：第一次找不到对应的相邻版本对（按历史顺序的两版标识）。
    break_from: Optional[str]
    break_to: Optional[str]
    # 应复测一侧的完整段证据（NEW 在晚版、REMOVED/UNPAIRED 在早/查看版）。
    evidence: Optional[DueEvidence]


@dataclass(frozen=True)
class ChangeList:
    """一次小窗查询的变化清单。"""

    early_id: str
    late_id: str
    view_id: str
    window: Tuple[int, int]
    rows: Tuple[ChangeRow, ...]


@dataclass
class ChangeAccessStats:
    """跨版查询的局部访问计数（测试用于证明访问量与远处数据无关）。

    * ``handover_probes`` : 查询过程中访问的交接记录条数；
    * ``dataset_queries`` : 对早晚 Dataset 发起的小窗查询次数（每次只
      触碰与该小窗相交的块）；
    * ``touched_ranges``  : 实际读取过的全部（版本标识, 起, 止）小片段，
      用于证明改线分送到远端时也不会读入中间无关里程。
    """

    handover_probes: int = 0
    dataset_queries: int = 0
    touched_ranges: Tuple[Tuple[str, int, int], ...] = ()

    def reset(self) -> None:
        self.handover_probes = 0
        self.dataset_queries = 0
        self.touched_ranges = ()

    def _note_probe(self) -> None:
        self.handover_probes += 1

    def _note_query(self, version_id: str, lo: int, hi: int) -> None:
        self.dataset_queries += 1
        self.touched_ranges = self.touched_ranges + ((version_id, lo, hi),)


# ---------------------------------------------------------------------- #
# 严格整数（bool 是 int 子类，业务上必须拒绝）
# ---------------------------------------------------------------------- #
def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


# ---------------------------------------------------------------------- #
# 单张交接表的压缩索引：两侧各一份有序、互不重叠的区间表
# ---------------------------------------------------------------------- #
class _Table:
    """一张交接表的不可压缩索引（构造时完成全部校验）。

    保存旧侧/新侧两套平行数组，按各自坐标升序排列；任一记录在两侧
    的平行下标相同。映射通过区间算术完成，绝不逐格展开。
    """

    __slots__ = (
        "from_id", "to_id",
        "_old_starts", "_old_ends", "_new_starts", "_new_ends", "_reversed",
        "_new_order", "_ord_new_starts", "_ord_new_ends",
    )

    def __init__(self, records: Sequence[HandoverRecord],
                 from_version: LineVersion, to_version: LineVersion):
        self.from_id = from_version.version_id
        self.to_id = to_version.version_id

        old_pts: List[Tuple[int, int, int, int, bool]] = []
        for rec in records:
            if not isinstance(rec, HandoverRecord):
                raise ChainError(
                    f"交接 {from_version.version_id}->{to_version.version_id} "
                    f"存在非 HandoverRecord 记录: {rec!r}"
                )
            os_, oe, ns, ne, rev = (
                rec.old_start, rec.old_end, rec.new_start, rec.new_end,
                rec.reversed,
            )
            if not all(_is_int(v) for v in (os_, oe, ns, ne)):
                raise ChainError(f"交接记录端点必须是整数: {rec}")
            if oe <= os_ or ne <= ns:
                raise ChainError(f"交接记录必须是非空半开区间: {rec}")
            if oe - os_ != ne - ns:
                raise ChainError(
                    f"交接记录两侧区间必须等长: [{os_},{oe}) 长 {oe-os_} "
                    f"vs [{ns},{ne}) 长 {ne-ns}"
                )
            if not (from_version.start <= os_ and oe <= from_version.end):
                raise ChainError(
                    f"交接记录旧侧 [{os_},{oe}) 越出版本 "
                    f"{from_version.version_id} 范围 "
                    f"[{from_version.start},{from_version.end})"
                )
            if not (to_version.start <= ns and ne <= to_version.end):
                raise ChainError(
                    f"交接记录新侧 [{ns},{ne}) 越出版本 "
                    f"{to_version.version_id} 范围 "
                    f"[{to_version.start},{to_version.end})"
                )
            if not isinstance(rev, bool):
                raise ChainError(f"交接记录方向标记必须是布尔值: {rec}")
            old_pts.append((os_, oe, ns, ne, rev))

        old_pts.sort(key=lambda t: (t[0], t[1]))
        # 旧侧重叠检查（含端点相接是允许的：半开不重叠）。
        for a, b in zip(old_pts, old_pts[1:]):
            if b[0] < a[1]:
                raise ChainError(
                    f"交接 {from_version.version_id}->"
                    f"{to_version.version_id} 旧侧区间重叠: "
                    f"[{a[0]},{a[1]}) 与 [{b[0]},{b[1]})"
                )
        # 新侧重叠检查（按新侧起点排序后比对）。
        new_order = sorted(range(len(old_pts)),
                           key=lambda i: (old_pts[i][2], old_pts[i][3]))
        for ia, ib in zip(new_order, new_order[1:]):
            a, b = old_pts[ia], old_pts[ib]
            if b[2] < a[3]:
                raise ChainError(
                    f"交接 {from_version.version_id}->"
                    f"{to_version.version_id} 新侧区间重叠: "
                    f"[{a[2]},{a[3]}) 与 [{b[2]},{b[3]})"
                )

        self._old_starts = [t[0] for t in old_pts]
        self._old_ends = [t[1] for t in old_pts]
        self._new_starts = [t[2] for t in old_pts]
        self._new_ends = [t[3] for t in old_pts]
        self._reversed = [t[4] for t in old_pts]
        # 新侧升序排列（建链时一次性预算，查询不再重排远处记录）。
        self._new_order = sorted(
            range(len(old_pts)), key=lambda i: (old_pts[i][2], old_pts[i][3])
        )
        self._ord_new_starts = [old_pts[i][2] for i in self._new_order]
        self._ord_new_ends = [old_pts[i][3] for i in self._new_order]

    def __len__(self) -> int:
        return len(self._old_starts)

    # ---- 旧侧（早->晚方向的源侧） ---- #
    def split_on_old(self, lo: int, hi: int
                     ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int]]]:
        """把旧侧窗口 ``[lo,hi)`` 切成覆盖片段与空隙。

        覆盖片段返回 ``(old_lo, old_hi, record_index)``（旧侧升序），
        空隙返回未被任何记录覆盖的旧侧区间（升序、互不相交）。
        """
        return self._split(lo, hi, self._old_starts, self._old_ends)

    def split_on_new(self, lo: int, hi: int
                     ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int]]]:
        """新侧切分（反向追溯用）；记录下标仍指向旧侧平行数组。"""
        covered, gaps = self._split(
            lo, hi, self._ord_new_starts, self._ord_new_ends
        )
        order = self._new_order
        return [(a, b, order[k]) for a, b, k in covered], gaps

    @staticmethod
    def _split(lo, hi, starts, ends
               ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int]]]:
        covered: List[Tuple[int, int, int]] = []
        gaps: List[Tuple[int, int]] = []
        i = bisect.bisect_right(ends, lo)
        n = len(starts)
        cursor = lo
        while cursor < hi:
            if i < n and starts[i] < hi and ends[i] > cursor:
                s, e = starts[i], ends[i]
                if s > cursor:
                    gaps.append((cursor, s))
                covered.append((max(s, cursor), min(e, hi), i))
                cursor = min(e, hi)
                i += 1
            else:
                gaps.append((cursor, hi))
                break
        return covered, gaps

    def map_old_to_new(self, lo: int, hi: int, idx: int
                       ) -> Tuple[int, int]:
        """旧侧片段 ``[lo,hi)``（已知在记录 idx 内）映射到新侧。"""
        if self._reversed[idx]:
            return (self._new_ends[idx] - (hi - self._old_starts[idx]),
                    self._new_ends[idx] - (lo - self._old_starts[idx]))
        return (self._new_starts[idx] + (lo - self._old_starts[idx]),
                self._new_starts[idx] + (hi - self._old_starts[idx]))

    def map_new_to_old(self, lo: int, hi: int, idx: int
                       ) -> Tuple[int, int]:
        """新侧片段映射回旧侧（``map_old_to_new`` 的逆，自逆于反向记录）。"""
        if self._reversed[idx]:
            return (self._old_ends[idx] - (hi - self._new_starts[idx]),
                    self._old_ends[idx] - (lo - self._new_starts[idx]))
        return (self._old_starts[idx] + (lo - self._new_starts[idx]),
                self._old_starts[idx] + (hi - self._new_starts[idx]))


# ---------------------------------------------------------------------- #
# 版本链：一次构建、多次小窗查询
# ---------------------------------------------------------------------- #
class VersionChain:
    """已校验的线路版本链与相邻交接表索引（不可变，可反复查询）。"""

    def __init__(self, versions: Tuple[LineVersion, ...],
                 tables: Tuple[_Table, ...],
                 access: ChangeAccessStats):
        self._versions = versions
        self._by_id = {v.version_id: v for v in versions}
        self._tables = tables
        self.access_stats = access

    # ---------------- 构建与整份校验 ---------------- #
    @classmethod
    def build(cls, versions: Sequence[LineVersion],
              handovers: Mapping[Tuple[str, str], Sequence[HandoverRecord]]
              ) -> "VersionChain":
        """校验整份版本链与交接数据，失败抛 :class:`ChainError`。

        成功后交接表被压成两侧有序索引，之后任何小窗查询都不再扫描
        与窗口无关的交接记录。不产生部分对象。
        """
        if not isinstance(versions, (list, tuple)) or not versions:
            raise ChainError("版本链必须是非空的 LineVersion 序列")
        ids = set()
        for v in versions:
            if not isinstance(v, LineVersion):
                raise ChainError(f"版本条目必须是 LineVersion: {v!r}")
            if not isinstance(v.version_id, str) or not v.version_id:
                raise ChainError("版本标识必须是非空字符串")
            if v.version_id in ids:
                raise ChainError(f"版本标识重复: {v.version_id}")
            ids.add(v.version_id)
            if not (_is_int(v.start) and _is_int(v.end)) or v.end <= v.start:
                raise ChainError(
                    f"版本 {v.version_id} 的线路范围必须是非空整数半开区间"
                )

        n = len(versions)
        if not isinstance(handovers, Mapping):
            raise ChainError("交接表必须是以 (旧版,新版) 为键的映射")
        expected = {(versions[i].version_id, versions[i + 1].version_id)
                    for i in range(n - 1)}
        if set(handovers.keys()) != expected:
            raise ChainError(
                "交接表键必须恰好是每对相邻版本 (旧版标识, 新版标识)，"
                f"期望 {sorted(expected)}，实际 {sorted(handovers.keys())}"
            )
        tables = tuple(
            _Table(handovers[(versions[i].version_id,
                              versions[i + 1].version_id)],
                   versions[i], versions[i + 1])
            for i in range(n - 1)
        )
        return cls(tuple(versions), tables, ChangeAccessStats())

    @property
    def versions(self) -> Tuple[LineVersion, ...]:
        return self._versions

    # ---------------- 跨版变化清单 ---------------- #
    def change_list(self, early_dataset, late_dataset,
                    view_id: str, window_start: int, window_end: int
                    ) -> ChangeList:
        """对照早晚两份检测，返回查看版窗口 ``[window_start,window_end)``
        内的跨版复测变化清单。

        版本链、端点数据集范围或窗口不合法时抛 :class:`ChainError`，
        不返回部分清单。输入数据集不会被修改。
        """
        from .dataset import Dataset  # 局部导入，保持模块间单向依赖

        early_v, late_v = self._versions[0], self._versions[-1]
        if not isinstance(early_dataset, Dataset) or \
                not isinstance(late_dataset, Dataset):
            raise ChainError("早晚检测必须是已装配的 Dataset 实例")
        if (early_dataset.line_start, early_dataset.line_end) != \
                (early_v.start, early_v.end):
            raise ChainError(
                f"早版数据集范围 "
                f"[{early_dataset.line_start},{early_dataset.line_end}) 与版本 "
                f"{early_v.version_id} 范围 [{early_v.start},{early_v.end}) 不一致"
            )
        if (late_dataset.line_start, late_dataset.line_end) != \
                (late_v.start, late_v.end):
            raise ChainError(
                f"晚版数据集范围 "
                f"[{late_dataset.line_start},{late_dataset.line_end}) 与版本 "
                f"{late_v.version_id} 范围 [{late_v.start},{late_v.end}) 不一致"
            )
        if view_id not in (early_v.version_id, late_v.version_id):
            raise ChainError(
                f"查看版必须是早版 {early_v.version_id!r} 或晚版 "
                f"{late_v.version_id!r}，实际 {view_id!r}"
            )
        if not (_is_int(window_start) and _is_int(window_end)) \
                or window_end <= window_start:
            raise ChainError("查看窗口必须是非空整数半开区间")
        view_v = self._by_id[view_id]
        if not (view_v.start <= window_start and window_end <= view_v.end):
            raise ChainError(
                f"窗口 [{window_start},{window_end}) 越出查看版 {view_id} "
                f"范围 [{view_v.start},{view_v.end})"
            )

        # 访问计数默认累计，便于重复小窗查询的局部性核对；
        # 调用方可通过 access_stats.reset() 主动清零。
        forward = view_id == early_v.version_id
        if forward:
            paired, broken = self._trace_forward(window_start, window_end)
            view_ds, peer_ds = early_dataset, late_dataset
            peer_id = late_v.version_id
        else:
            paired, broken = self._trace_backward(window_start, window_end)
            view_ds, peer_ds = late_dataset, early_dataset
            peer_id = early_v.version_id

        # 查看版应复测位置（一次小窗查询，段门槛按全线段判定）。
        view_due = self._due_runs(view_ds, view_id, window_start, window_end)

        rows: List[ChangeRow] = []

        # 1) 未配对：只列查看版应复测且无对端的部分。
        for vlo, vhi, from_id, to_id in broken:
            for clo, chi, ev in _clip_runs(view_due, vlo, vhi):
                rows.append(ChangeRow(
                    start=clo, end=chi,
                    category=ChangeCategory.UNPAIRED,
                    peer_start=None, peer_end=None, direction=None,
                    break_from=from_id, break_to=to_id, evidence=ev,
                ))

        # 2) 已配对：逐片段做区间算术映射，并只在该远端小片段上查对端数据。
        # 约定：plo/phi 始终满足 plo < phi（片段端点按数值大小保存）。
        # 等长双射下，查看格在查看片段 [vlo,vhi) 内的偏移 d = 格-vlo，
        # 对应对端片段内偏移同为 d；反向时对端格 = phi-1-d。
        for vlo, vhi, plo, phi, parity in paired:
            direction = Direction.FORWARD if parity == 0 else Direction.REVERSED
            peer_due = self._due_runs(peer_ds, peer_id, plo, phi)
            # 对端应测片段换算回查看坐标。
            peer_in_view: List[Tuple[int, int, DueEvidence]] = []
            for qlo, qhi, ev in peer_due:
                if direction is Direction.FORWARD:
                    vlo2 = vlo + (qlo - plo)
                    vhi2 = vlo + (qhi - plo)
                else:
                    # 对端片段内偏移 d 镜像：查看格 = vhi-1-d，
                    # 区间 [qlo,qhi) 对应查看 [vhi-(qhi-plo), vhi-(qlo-plo))。
                    vlo2 = vhi - (qhi - plo)
                    vhi2 = vhi - (qlo - plo)
                peer_in_view.append((vlo2, vhi2, ev))
            # 反向时对端查询顺序在查看坐标上是降序，统一排序回升序，
            # 使 _overlay 的两路有序区间前提成立。
            peer_in_view.sort()
            view_clip = _clip_runs(view_due, vlo, vhi)
            for clo, chi, side, ev in self._overlay(view_clip, peer_in_view,
                                                    forward):
                # 由查看行在查看片段内的偏移反查对端范围。
                d_lo, d_hi = clo - vlo, chi - vlo
                if direction is Direction.FORWARD:
                    peer_row_lo = plo + d_lo
                    peer_row_hi = plo + d_hi
                else:
                    peer_row_lo = phi - d_hi
                    peer_row_hi = phi - d_lo
                rows.append(ChangeRow(
                    start=clo, end=chi, category=side,
                    peer_start=peer_row_lo, peer_end=peer_row_hi,
                    direction=direction,
                    break_from=None, break_to=None, evidence=ev,
                ))

        rows.sort(key=lambda r: (r.start, r.end))
        rows = self._coalesce(rows)
        return ChangeList(
            early_id=early_v.version_id,
            late_id=late_v.version_id,
            view_id=view_id,
            window=(window_start, window_end),
            rows=tuple(rows),
        )

    # ---------------- 沿版本链追溯（区间算术，不逐格） ---------------- #
    def _trace_forward(self, wlo: int, whe: int
                       ) -> Tuple[List[list], List[Tuple[int, int, str, str]]]:
        """从早版窗口出发逐次交接向前追溯。

        每个存活片段维护五个量：
        ``[查看区间 lo/hi, 当前版本区间 lo/hi, 反向次数奇偶 parity]``。
        查看片段与当前片段等长，但经过反向后方向相反；因此把当前坐标
        换算回查看坐标时，必须按 parity 决定直移还是镜像。

        返回：
          paired: ``[查看lo, 查看hi, 晚版lo, 晚版hi, parity]``；
          broken: ``(查看lo, 查看hi, 断口旧版id, 断口新版id)``。
        """
        alive: List[list] = [[wlo, whe, wlo, whe, 0]]
        broken: List[Tuple[int, int, str, str]] = []
        for ti, table in enumerate(self._tables):
            from_id = self._versions[ti].version_id
            to_id = self._versions[ti + 1].version_id
            next_alive: List[list] = []
            for frag in alive:
                vlo, vhi, clo, chi, parity = frag
                covered, gaps = table.split_on_old(clo, chi)
                for glo, ghi in gaps:
                    broken.append(self._to_view(vlo, vhi, clo, chi,
                                                parity, glo, ghi)
                                  + (from_id, to_id))
                for slo, shi, idx in covered:
                    nlo, nhi = table.map_old_to_new(slo, shi, idx)
                    self.access_stats._note_probe()
                    sub = self._to_view(vlo, vhi, clo, chi, parity, slo, shi)
                    next_alive.append([
                        sub[0], sub[1], nlo, nhi,
                        parity ^ (1 if table._reversed[idx] else 0),
                    ])
            alive = next_alive
        paired = sorted(alive, key=lambda f: f[0])
        broken.sort()
        return paired, broken

    def _trace_backward(self, wlo: int, whe: int
                        ) -> Tuple[List[list], List[Tuple[int, int, str, str]]]:
        """从晚版窗口出发，沿交接表反向追溯（断口仍按历史顺序注明）。"""
        alive: List[list] = [[wlo, whe, wlo, whe, 0]]
        broken: List[Tuple[int, int, str, str]] = []
        for ti in range(len(self._tables) - 1, -1, -1):
            table = self._tables[ti]
            from_id = self._versions[ti].version_id
            to_id = self._versions[ti + 1].version_id
            next_alive: List[list] = []
            for frag in alive:
                vlo, vhi, clo, chi, parity = frag
                covered, gaps = table.split_on_new(clo, chi)
                for glo, ghi in gaps:
                    broken.append(self._to_view(vlo, vhi, clo, chi,
                                                parity, glo, ghi)
                                  + (from_id, to_id))
                for slo, shi, idx in covered:
                    olo, ohi = table.map_new_to_old(slo, shi, idx)
                    self.access_stats._note_probe()
                    sub = self._to_view(vlo, vhi, clo, chi, parity, slo, shi)
                    next_alive.append([
                        sub[0], sub[1], olo, ohi,
                        parity ^ (1 if table._reversed[idx] else 0),
                    ])
            alive = next_alive
        paired = sorted(alive, key=lambda f: f[0])
        broken.sort()
        return paired, broken

    @staticmethod
    def _to_view(vlo: int, vhi: int, clo: int, chi: int, parity: int,
                 slo: int, shi: int) -> Tuple[int, int]:
        """把当前版本片段 ``[slo,shi)``（含于 ``[clo,chi)``）换算回查看坐标。

        parity=0（同向）：当前偏移直接加到查看左端；
        parity=1（反向）：当前偏移镜像到查看右端。
        """
        if parity == 0:
            return vlo + (slo - clo), vlo + (shi - clo)
        return vhi - (shi - clo), vhi - (slo - clo)

    # ---------------- Dataset 小窗读取：只取达标段上的 MISSING ---------------- #
    def _due_runs(self, dataset, version_id: str, lo: int, hi: int
                  ) -> List[Tuple[int, int, DueEvidence]]:
        """返回 ``[lo,hi)`` 内“应复测”游程：达标段内的实际 MISSING 区间。

        段是否达标由 Dataset 按全线段判定，这里不重新判门槛；每次调用
        只触发一次小窗 :meth:`Dataset.query_window`，不读无关块。
        """
        if hi <= lo:
            return []
        from .model import State
        self.access_stats._note_query(version_id, lo, hi)
        out: List[Tuple[int, int, DueEvidence]] = []
        for hit in dataset.query_window(lo, hi):
            if not hit.qualified:
                continue
            seg = hit.segment
            ev = DueEvidence(
                segment_start=seg.start,
                segment_end=seg.end,
                rule_version=seg.version,
                segment_missing=seg.missing,
            )
            for run in hit.evidence:
                if run.state is State.MISSING:
                    out.append((run.start, run.end, ev))
        out.sort()
        return out

    # ---------------- 两版应测游程叠加分类（有序区间做差，不逐格） ---------------- #
    @staticmethod
    def _overlay(view_runs: Sequence[Tuple[int, int, DueEvidence]],
                 peer_runs: Sequence[Tuple[int, int, DueEvidence]],
                 forward: bool
                 ) -> List[Tuple[int, int, ChangeCategory, DueEvidence]]:
        """在同一查看坐标上叠加两侧应测游程，输出有差异的分类片段。

        类别按早/晚绝对语义命名，与查看视角无关：只在早版应测 ->
        REMOVED，只在晚版应测 -> NEW；两侧都应测或都不应测 -> 不列。
        ``peer_runs`` 已换算到查看坐标；``forward=False`` 时查看版是
        晚版、对端是早版，两类命名对调。
        """
        # 视角为早版：查看侧独有=解除，对端（晚版）独有=新增；
        # 视角为晚版：查看侧独有=新增，对端（早版）独有=解除。
        view_only = ChangeCategory.REMOVED if forward else ChangeCategory.NEW
        peer_only = ChangeCategory.NEW if forward else ChangeCategory.REMOVED

        def pieces_only_in(left, right, category):
            """返回落在 left 游程内、但不被任何 right 游程覆盖的子区间。

            两路有序区间做差（纯区间算术，不逐格展开）；每个输出片段在
            right 游程边界处切断，保证部分重叠时共有部分被准确剔除，
            输出按坐标升序，相邻同证据片段交给上层 _coalesce 合并。
            """
            out = []
            r_ends = [r[1] for r in right]
            for ls, le, lev in left:
                cursor = ls
                # 第一个 end > cursor 的 right 游程下标；它可能把 cursor
                # 包在内部（start <= cursor），也可能起点在 cursor 之后。
                i = bisect.bisect_right(r_ends, cursor)
                while cursor < le:
                    if i < len(right) and right[i][0] < le \
                            and right[i][1] > cursor:
                        rs, re_ = right[i][0], right[i][1]
                        if rs > cursor:
                            # cursor..rs 落在 right 游程之前，是 left 独有。
                            out.append((cursor, min(rs, le), category, lev))
                        # 共有部分跳过，并在 right 游程端界处切断；
                        # 右游程完全落在 left 内部时 cursor 仍 < le，
                        # 必须继续处理 left 的剩余部分，不能终止。
                        cursor = max(cursor, min(re_, le))
                        i += 1
                    else:
                        out.append((cursor, le, category, lev))
                        break
            return out

        result = pieces_only_in(view_runs, peer_runs, view_only)
        result += pieces_only_in(peer_runs, view_runs, peer_only)
        result.sort(key=lambda t: (t[0], t[1]))
        return result

    # ---------------- 相邻同类同行合并 ---------------- #
    @staticmethod
    def _coalesce(rows: Sequence[ChangeRow]) -> List[ChangeRow]:
        """合并相邻且类别、段证据、断点相同的行。

        配对行还要求净方向相同，且对端范围相邻后仍是同向（对端升序
        衔接）或反向（对端降序衔接）的一个连续对应，否则不合并。
        """
        out: List[ChangeRow] = []
        for r in rows:
            if out:
                p = out[-1]
                if p.end == r.start and p.category is r.category \
                        and p.evidence == r.evidence \
                        and p.break_from == r.break_from \
                        and p.break_to == r.break_to \
                        and p.direction is r.direction:
                    ok = False
                    if r.category is ChangeCategory.UNPAIRED:
                        ok = True
                    elif r.direction is Direction.FORWARD \
                            and p.peer_end == r.peer_start:
                        ok = True
                    elif r.direction is Direction.REVERSED \
                            and p.peer_start == r.peer_end:
                        # 反向时后一格的对端在左侧：当前行对端右端
                        # 接上一行对端左端，合并后对端区间连续降序衔接。
                        ok = True
                    if ok:
                        if r.category is ChangeCategory.UNPAIRED:
                            new_peer_lo, new_peer_hi = None, None
                        elif r.direction is Direction.FORWARD:
                            new_peer_lo, new_peer_hi = p.peer_start, r.peer_end
                        else:
                            # 反向：对端向左侧扩展，左端取当前行的更小值。
                            new_peer_lo, new_peer_hi = r.peer_start, p.peer_end
                        out[-1] = ChangeRow(
                            start=p.start, end=r.end, category=r.category,
                            peer_start=new_peer_lo, peer_end=new_peer_hi,
                            direction=r.direction,
                            break_from=p.break_from, break_to=p.break_to,
                            evidence=r.evidence,
                        )
                        continue
            out.append(r)
        return out


def _clip_runs(runs: Sequence[Tuple[int, int, DueEvidence]], lo: int, hi: int
               ) -> List[Tuple[int, int, DueEvidence]]:
    """把升序不交的游程表裁到 ``[lo,hi)``，丢弃不相交者。"""
    out: List[Tuple[int, int, DueEvidence]] = []
    for s, e, ev in runs:
        if e <= lo or s >= hi:
            continue
        out.append((max(s, lo), min(e, hi), ev))
    return out
