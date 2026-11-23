"""可落盘的复测工单对账：签发、批次记账、小窗读取。

三类可调用接口（保留既有建块/装配/查询/更正接口不变）：

* :meth:`WorkOrderStore.issue`        —— 用未用过的工单号、已装配 Dataset
  与里程窗签发工单；
* :meth:`WorkOrderStore.record_batch` —— 回执按工单成批登记（验收通过/撤销）；
* :meth:`WorkOrderStore.read_window`  —— 查工单内任意小窗，返回裁切后的
  应复测区间、状态、凭证与全单总数。

语义要点（均由测试守护）：

签发
  只把窗内**达标复测段**上的 MISSING 区间列入工单；GAP、不达标段、窗外格
  一律不计；允许工单范围为空。签发时即把范围快照落盘，此后不持有 Dataset
  或块产物引用：线路更正不改变旧工单，记账/读取也绝不反写线路。

记账
  每条回执带非空回执号、正整数修订号与 ``accept``/``revoke``：
  验收通过带非空班组号及工单窗内的非空半开区间；撤销不带这两项。
  每个回执只以最高修订生效，可换班组、换区间或撤销；允许乱序送达与原样
  重送。同号同版本内容冲突（包括过期版本）或任一记录无效时，**整批拒绝**，
  不留下局部进度，也不占用修订号。

确认与凭证
  应复测位置至少由两个**不同班组**覆盖才算确认。凭证取字典序最前的两个
  班组，各取该位置最小回执号，给出班组、回执号与当前修订号。相邻且状态、
  凭证完全相同的段合并。全单应复测/已确认/待确认格数增量维护，不随窗口
  变化，也不靠扫描其余里程。

局部性
  记账只重算有效回执旧、新范围相交的覆盖段，不碰两侧都不相交的回执；
  小窗读取只访问相交覆盖段与少量回执记录，访问计数可由
  :meth:`WorkOrderStore.access_stats` 读取，供测试证明访问量与远处数据无关。

落盘
  每个工单一个目录内 JSONL 追加日志（首行头、其后每行一个批次），重开时
  按同样的应用路径回放，结果确定一致；被拒批次不产生任何写入。
"""

from __future__ import annotations

import bisect
import json
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .model import State

ACCEPT = "accept"
REVOKE = "revoke"


class OrderState(str, Enum):
    """工单小窗内一个应复测段的对账状态。"""

    CONFIRMED = "CONFIRMED"  # 至少两个不同班组覆盖
    PENDING = "PENDING"      # 不足两个不同班组


class OrderError(ValueError):
    """工单对账被拒绝：未知工单、记录非法、同版冲突、窗口非法等。"""


@dataclass(frozen=True)
class CrewEvidence:
    """单条班组凭证：班组、该班组在该位置的最小回执号及其当前修订号。"""

    crew: str
    receipt_id: str
    revision: int


@dataclass(frozen=True)
class ViewRun:
    """小窗读取结果中的一段（相邻同状态同凭证段已合并）。"""

    start: int
    end: int
    state: OrderState
    evidence: Tuple[CrewEvidence, ...]


@dataclass(frozen=True)
class WindowView:
    """小窗读取结果：窗内明细 + 不随窗口变化的全单总数。"""

    order_id: str
    window: Tuple[int, int]
    runs: Tuple[ViewRun, ...]
    due_total: int
    confirmed_total: int
    pending_total: int

    @property
    def totals(self) -> Tuple[int, int, int]:
        return self.due_total, self.confirmed_total, self.pending_total


@dataclass(frozen=True)
class WorkOrder:
    """工单的不可变快照（全部是可 JSON 化的普通数据，不依赖 Dataset）。"""

    order_id: str
    window: Tuple[int, int]
    line_window: Tuple[int, int]
    scope: Tuple[Tuple[int, int], ...]  # 应复测 MISSING 区间（半开、升序、不交）
    due_total: int


@dataclass(frozen=True)
class BatchResult:
    """批次登记结果：整批成功后实际应用的去重后记录条数。"""

    order_id: str
    applied: int


@dataclass
class AccessStats:
    """覆盖引擎的局部访问计数（测试用于证明访问量与远处里程无关）。

    * ``edit_visits``    : 一次记账变更中访问的覆盖段/空隙数；
    * ``read_run_visits``: 一次小窗读取触碰的覆盖段数；
    * ``scope_probes``   : 一次小窗读取触碰的应复测区间数；
    * ``receipt_lookups``: 读取凭证时查阅的有效回执记录数；
    * ``edit_min``/``edit_max``: 记账触达的里程范围（含端点），用于证明
      远处回执不会把近处数据带进访问面。
    """

    edit_visits: int = 0
    read_run_visits: int = 0
    scope_probes: int = 0
    receipt_lookups: int = 0
    edit_min: Optional[int] = None
    edit_max: Optional[int] = None

    def reset(self) -> None:
        self.edit_visits = 0
        self.read_run_visits = 0
        self.scope_probes = 0
        self.receipt_lookups = 0
        self.edit_min = None
        self.edit_max = None

    def note_edit_span(self, lo: int, hi: int) -> None:
        self.edit_min = lo if self.edit_min is None else min(self.edit_min, lo)
        self.edit_max = hi if self.edit_max is None else max(self.edit_max, hi)


# ---------------------------------------------------------------------- #
# 区间工具
# ---------------------------------------------------------------------- #
def _is_int(x: Any) -> bool:
    """严格整数：布尔值不算（bool 是 int 的子类，业务上必须拒绝）。"""
    return isinstance(x, int) and not isinstance(x, bool)


def canonical_intervals(
    intervals: Any, wlo: int, whi: int
) -> Tuple[Tuple[int, int], ...]:
    """校验并规范化区间表：升序、半开、正长度、不越窗，重叠/相邻则合并。"""
    if not isinstance(intervals, (list, tuple)) or not intervals:
        raise OrderError("区间表必须是非空的半开区间列表")
    pts: List[Tuple[int, int]] = []
    for iv in intervals:
        if not isinstance(iv, (list, tuple)) or len(iv) != 2:
            raise OrderError(f"区间形态非法: {iv!r}")
        lo, hi = iv
        if not (_is_int(lo) and _is_int(hi)):
            raise OrderError(f"区间端点必须是整数: {iv!r}")
        if hi <= lo:
            raise OrderError(f"区间必须是正长度半开区间: {iv!r}")
        if lo < wlo or hi > whi:
            raise OrderError(f"区间 [{lo},{hi}) 越出工单窗 [{wlo},{whi})")
        pts.append((lo, hi))
    pts.sort()
    out: List[Tuple[int, int]] = []
    for lo, hi in pts:
        if out and lo <= out[-1][1]:
            if hi > out[-1][1]:
                out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return tuple(out)


def merge_intervals(pts: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    """把若干（已裁切的）区间合并为升序、不交的半开区间元组。"""
    pts = sorted(pts)
    out: List[Tuple[int, int]] = []
    for lo, hi in pts:
        if hi <= lo:
            continue
        if out and lo <= out[-1][1]:
            if hi > out[-1][1]:
                out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return tuple(out)


# ---------------------------------------------------------------------- #
# 回执记录校验
# ---------------------------------------------------------------------- #
def _validate_entry(entry: Any, window: Tuple[int, int]) -> Dict[str, Any]:
    """把一条外部回执校验并规范化为内部记录（失败抛 OrderError）。"""
    if not isinstance(entry, Mapping):
        raise OrderError(f"回执记录必须是映射对象: {entry!r}")
    rid = entry.get("rid")
    if not isinstance(rid, str) or not rid:
        raise OrderError("回执号必须是非空字符串")
    rev = entry.get("rev")
    if not _is_int(rev) or rev < 1:
        raise OrderError(f"回执 {rid} 的修订号必须是正整数")
    kind = entry.get("kind")
    if kind not in (ACCEPT, REVOKE):
        raise OrderError(f"回执 {rid} 的 kind 必须是 {ACCEPT!r} 或 {REVOKE!r}")
    crew = entry.get("crew")
    intervals = entry.get("intervals")

    if kind == ACCEPT:
        if not isinstance(crew, str) or not crew:
            raise OrderError(f"验收通过回执 {rid} 必须带非空班组号")
        canon = canonical_intervals(intervals, window[0], window[1])
        return {
            "rid": rid, "rev": rev, "kind": ACCEPT,
            "crew": crew, "intervals": canon,
        }
    # 撤销不带班组号与区间。
    if crew is not None or intervals is not None:
        raise OrderError(f"撤销回执 {rid} 不得携带班组号或区间")
    return {
        "rid": rid, "rev": rev, "kind": REVOKE,
        "crew": None, "intervals": None,
    }


def _rec_equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return (
        a["rid"] == b["rid"] and a["rev"] == b["rev"]
        and a["kind"] == b["kind"] and a["crew"] == b["crew"]
        and a["intervals"] == b["intervals"]
    )


# ---------------------------------------------------------------------- #
# 覆盖引擎：区间游程上的班组覆盖，局部增删，增量总数
# ---------------------------------------------------------------------- #
class _Coverage:
    """工单范围内的班组覆盖状态。

    内部只保存三类压缩数据，绝不逐格展开：

    * ``scope``   : 签发时固定的应复测 MISSING 区间表；
    * 覆盖游程表  : 升序、互不相交、仅落在 scope 上，每段记录覆盖它的
      回执集合与班组集合（frozenset）；
    * ``history`` : 每个回执号的全部已受理修订（用于同版冲突检测），
      ``active`` 为各回执当前最高修订记录。

     ``confirmed_total`` 随每次局部增删增量维护；待确认 = 应复测 - 已确认。
    """

    __slots__ = (
        "scope", "_scope_starts", "_scope_ends",
        "_starts", "_ends", "_maps",
        "history", "active", "confirmed_total", "access",
    )

    def __init__(self, scope: Tuple[Tuple[int, int], ...]):
        self.scope = tuple(scope)
        self._scope_starts = [s for s, _ in self.scope]
        self._scope_ends = [e for _, e in self.scope]
        self._starts: List[int] = []
        self._ends: List[int] = []
        # 每个游程自带 rid->crew 字典；班组集合由字典的 crew 视图派生，
        # 增删完全自洽，不依赖外部 active 快照（迁移/撤销期间 active 正在
        # 变更，引用它会得到错误的班组判定甚至 KeyError）。
        self._maps: List[Dict[str, str]] = []
        self.history: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self.active: Dict[str, Dict[str, Any]] = {}
        self.confirmed_total = 0
        self.access = AccessStats()

    # ---------------- 基本量 ---------------- #
    @property
    def due_total(self) -> int:
        return sum(e - s for s, e in self.scope)

    @property
    def pending_total(self) -> int:
        return self.due_total - self.confirmed_total

    # ---------------- scope 裁切 ---------------- #
    def clip_to_scope(
        self, intervals: Tuple[Tuple[int, int], ...]
    ) -> Tuple[Tuple[int, int], ...]:
        """把回执区间裁到应复测范围；落在 GAP/OK/空隙上的部分自然丢弃。

        scope 是升序、互不相交的压缩区间表（条数与 GAP 数同阶，不逐格），
        用二分定位首个可能相交的区间后顺序扫描，避免起点落在 scope 间隙
        （如 GAP 带）时漏掉进位后的相交区间。
        """
        out: List[Tuple[int, int]] = []
        for lo, hi in intervals:
            k = bisect.bisect_right(self._scope_starts, lo)
            k = max(k - 1, 0)
            while k < len(self.scope):
                s, e = self.scope[k]
                if s >= hi:
                    break
                if e > lo:
                    out.append((max(s, lo), min(e, hi)))
                k += 1
        return merge_intervals(out)

    # ---------------- 局部增删 ---------------- #
    def add_receipt(self, rid: str, crew: str,
                    intervals: Tuple[Tuple[int, int], ...]) -> None:
        def transform(rc: Dict[str, str], r=rid, c=crew) -> Dict[str, str]:
            nr = dict(rc)
            nr[r] = c
            return nr

        for lo, hi in self.clip_to_scope(intervals):
            self._edit(lo, hi, transform)

    def remove_receipt(self, rid: str, crew: str,
                       intervals: Tuple[Tuple[int, int], ...]) -> None:
        def transform(rc: Dict[str, str], r=rid,
                      c=crew) -> Dict[str, str]:
            nr = dict(rc)
            nr.pop(r, None)
            # 若该班组没有其余回执覆盖此段，一并从班组集合中消失。
            if c not in nr.values():
                nr = {k: v for k, v in nr.items() if v != c}
            return nr

        for lo, hi in self.clip_to_scope(intervals):
            self._edit(lo, hi, transform)

    @staticmethod
    def _crews_of(rc: Dict[str, str]) -> frozenset:
        return frozenset(rc.values())

    def _edit(self, lo: int, hi: int, transform) -> None:
        """在 ``[lo,hi)``（保证落在 scope 内）上对覆盖映射做局部变换。"""
        if hi <= lo:
            return
        self.access.note_edit_span(lo, hi)
        left = bisect.bisect_right(self._ends, lo)
        right = bisect.bisect_left(self._starts, hi)

        new_s: List[int] = []
        new_e: List[int] = []
        new_maps: List[Dict[str, str]] = []
        cursor = lo
        i = left
        while cursor < hi:
            self.access.edit_visits += 1
            if i < right and self._starts[i] <= cursor:
                s, e = self._starts[i], self._ends[i]
                rc = self._maps[i]
                ov_lo, ov_hi = max(s, cursor), min(e, hi)
                n_rc = transform(rc)
                old_crews = self._crews_of(rc)
                new_crews = self._crews_of(n_rc)
                self._bump(old_crews, new_crews, ov_hi - ov_lo)
                if s < ov_lo:
                    new_s.append(s); new_e.append(ov_lo); new_maps.append(rc)
                new_s.append(ov_lo); new_e.append(ov_hi); new_maps.append(n_rc)
                if ov_hi < e:
                    new_s.append(ov_hi); new_e.append(e); new_maps.append(rc)
                cursor = ov_hi
                i += 1
            else:
                # 两个覆盖游程之间（或两端）的空隙：新建游程。
                gap_hi = self._starts[i] if i < len(self._starts) else hi
                gap_hi = min(gap_hi, hi)
                n_rc = transform({})
                self._bump(frozenset(), self._crews_of(n_rc),
                           gap_hi - cursor)
                new_s.append(cursor); new_e.append(gap_hi); new_maps.append(n_rc)
                cursor = gap_hi

        pieces = [
            (s, e, rc)
            for s, e, rc in zip(new_s, new_e, new_maps)
            if rc
        ]
        self._starts[left:right] = [p[0] for p in pieces]
        self._ends[left:right] = [p[1] for p in pieces]
        self._maps[left:right] = [p[2] for p in pieces]
        if not self._starts:
            return  # 整段删空，无相邻游程可合并
        # 删空后切口两侧可能出现相邻同映射游程，仅在拼接点附近局部合并。
        self._coalesce(min(left, len(self._starts) - 1))
        edge = left + len(pieces)
        if 0 < edge <= len(self._starts):
            self._coalesce(min(edge, len(self._starts) - 1))

    def _bump(self, old_crews: frozenset, new_crews: frozenset,
              length: int) -> None:
        delta = (len(new_crews) >= 2) - (len(old_crews) >= 2)
        if delta:
            self.confirmed_total += delta * length

    def _coalesce(self, idx: int) -> None:
        """合并 idx 处与左/右相邻且覆盖映射相同的游程（仅局部常数步）。"""
        if not self._starts:
            return
        while (idx > 0 and self._ends[idx - 1] == self._starts[idx]
               and self._maps[idx - 1] == self._maps[idx]):
            self._ends[idx - 1] = self._ends[idx]
            del self._starts[idx]; del self._ends[idx]; del self._maps[idx]
            idx -= 1
        while (idx + 1 < len(self._starts)
               and self._ends[idx] == self._starts[idx + 1]
               and self._maps[idx] == self._maps[idx + 1]):
            self._ends[idx] = self._ends[idx + 1]
            del self._starts[idx + 1]; del self._ends[idx + 1]
            del self._maps[idx + 1]

    # ---------------- 回执应用 ---------------- #
    def apply(self, rec: Dict[str, Any]) -> None:
        """应用一条已校验记录；按新旧有效形态做局部覆盖增删。"""
        rid, rev = rec["rid"], rec["rev"]
        hist = self.history.setdefault(rid, {})
        hist[rev] = rec
        old = self.active.get(rid)
        new = hist[max(hist)]
        self.active[rid] = new
        if old is not None and old["kind"] == ACCEPT:
            self.remove_receipt(rid, old["crew"], old["intervals"])
        if new["kind"] == ACCEPT:
            self.add_receipt(rid, new["crew"], new["intervals"])

    # ---------------- 小窗读取 ---------------- #
    def read_window(self, order_id: str, qlo: int, qhi: int,
                    window: Tuple[int, int]) -> WindowView:
        if not (window[0] <= qlo < qhi <= window[1]):
            raise OrderError(
                f"查询窗 [{qlo},{qhi}) 必须是工单窗 [{window[0]},{window[1]}) "
                f"内的正长度半开区间"
            )
        # 1) scope 裁到查询窗（二分，只触碰相交的应复测区间）。
        si = max(bisect.bisect_right(self._scope_ends, qlo) - 1, 0)
        sj = bisect.bisect_left(self._scope_starts, qhi)
        scope_pieces: List[Tuple[int, int]] = []
        for k in range(si, min(sj + 1, len(self.scope))):
            s, e = self.scope[k]
            if e > qlo and s < qhi:
                scope_pieces.append((max(s, qlo), min(e, qhi)))
        self.access.scope_probes += len(scope_pieces)

        # 2) 与覆盖游程表做两路有序合并，逐段生成（状态, 凭证）。
        ri = bisect.bisect_right(self._ends, qlo)
        raw: List[Tuple[int, int, Dict[str, str]]] = []
        for s0, s1 in scope_pieces:
            pos = s0
            while pos < s1:
                if (ri < len(self._starts) and self._starts[ri] <= pos
                        and self._ends[ri] > pos):
                    r_s, r_e = self._starts[ri], self._ends[ri]
                    ov_lo, ov_hi = max(r_s, pos), min(r_e, s1)
                    self.access.read_run_visits += 1
                    raw.append((ov_lo, ov_hi, self._maps[ri]))
                    pos = ov_hi
                    if r_e <= pos:
                        ri += 1
                else:
                    nxt = self._starts[ri] if ri < len(self._starts) else s1
                    gap_hi = min(nxt, s1)
                    if gap_hi > pos:  # 游程正好衔接时跳过零长度空隙
                        raw.append((pos, gap_hi, {}))
                    pos = gap_hi

        # 3) 解析班组/凭证，并合并相邻且状态与凭证相同的段。
        runs: List[ViewRun] = []
        for s, e, rc_map in raw:
            state, evidence = self._resolve(rc_map)
            if runs and runs[-1].end == s and runs[-1].state == state \
                    and runs[-1].evidence == evidence:
                runs[-1] = ViewRun(runs[-1].start, e, state, evidence)
            else:
                runs.append(ViewRun(s, e, state, evidence))

        return WindowView(
            order_id=order_id,
            window=(qlo, qhi),
            runs=tuple(runs),
            due_total=self.due_total,
            confirmed_total=self.confirmed_total,
            pending_total=self.pending_total,
        )

    def _resolve(self, rc_map: Mapping[str, str]
                 ) -> Tuple[OrderState, Tuple[CrewEvidence, ...]]:
        by_crew: Dict[str, List[str]] = {}
        for rid, _crew in rc_map.items():
            self.access.receipt_lookups += 1
            # 班组以游程自带映射为准（迁移/撤销期间仍自洽）；修订号取当前
            # 生效记录，active 在读取时已稳定。
            crew = rc_map[rid]
            by_crew.setdefault(crew, []).append(rid)
        if len(by_crew) < 2:
            return OrderState.PENDING, ()
        evidence: List[CrewEvidence] = []
        for crew in sorted(by_crew)[:2]:  # 字典序最前的两个班组
            rid = min(by_crew[crew])     # 该位置该班组的最小回执号
            evidence.append(
                CrewEvidence(crew, rid, self.active[rid]["rev"])
            )
        return OrderState.CONFIRMED, tuple(evidence)


# ---------------------------------------------------------------------- #
# 落盘工单库
# ---------------------------------------------------------------------- #
_INDEX = "index.json"
_PREFIX = "order_"
_SUFFIX = ".jsonl"


class WorkOrderStore:
    """目录级落盘工单库。

    目录布局::

        <dir>/index.json                 # 工单号 -> 日志文件名
        <dir>/order_00001.jsonl          # 首行工单头，其后每行一个批次
        <dir>/order_00002.jsonl ...

    用 :meth:`open` 打开（``create=True`` 可新建）；被拒批次不产生任何
    磁盘写入，成功批次以单行 JSON 追加，重开按日志原样回放。
    """

    def __init__(self, directory: str, create: bool = False):
        self.directory = os.path.abspath(directory)
        if not os.path.isdir(self.directory):
            if not create:
                raise FileNotFoundError(f"工单库目录不存在: {self.directory}")
            os.makedirs(self.directory)
        index_path = os.path.join(self.directory, _INDEX)
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._index: Dict[str, str] = dict(payload["orders"])
            self._seq = int(payload["next_seq"])
        else:
            if not create:
                raise FileNotFoundError(f"工单库索引不存在: {index_path}")
            self._index = {}
            self._seq = 0
            self._flush_index()
        self._coverage: Dict[str, _Coverage] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}
        for order_id, fname in self._index.items():
            self._load_order(order_id, os.path.join(self.directory, fname))

    @classmethod
    def open(cls, directory: str, create: bool = False) -> "WorkOrderStore":
        return cls(directory, create=create)

    # ---------------- 落盘辅助 ---------------- #
    def _flush_index(self) -> None:
        path = os.path.join(self.directory, _INDEX)
        self._atomic_write(path, {
            "orders": dict(self._index),
            "next_seq": self._seq,
        })

    @staticmethod
    def _atomic_write(path: str, payload: Any) -> None:
        d = os.path.dirname(path)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=d)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @staticmethod
    def _rec_to_json(rec: Mapping[str, Any]) -> Dict[str, Any]:
        iv = rec["intervals"]
        return {
            "rid": rec["rid"],
            "rev": rec["rev"],
            "kind": rec["kind"],
            "crew": rec["crew"],
            "intervals": None if iv is None else [list(x) for x in iv],
        }

    @staticmethod
    def _rec_from_json(obj: Mapping[str, Any]) -> Dict[str, Any]:
        iv = obj["intervals"]
        return {
            "rid": obj["rid"],
            "rev": obj["rev"],
            "kind": obj["kind"],
            "crew": obj["crew"],
            "intervals": None if iv is None else tuple((a, b) for a, b in iv),
        }

    def _load_order(self, order_id: str, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        if not lines:
            raise OrderError(f"工单日志为空: {path}")
        header = json.loads(lines[0])
        if header.get("type") != "header" or header.get("order_id") != order_id:
            raise OrderError(f"工单日志头非法: {path}")
        scope = tuple((a, b) for a, b in header["scope"])
        cov = _Coverage(scope)
        for n, line in enumerate(lines[1:], start=2):
            if not line.strip():
                continue
            try:
                batch = json.loads(line)
            except json.JSONDecodeError:
                # 只容忍最后一行的半截写入（进程崩溃），其余位置损坏直接报错。
                if n == len(lines):
                    break
                raise OrderError(f"工单日志第 {n} 行损坏: {path}")
            for obj in batch["entries"]:
                cov.apply(self._rec_from_json(obj))
        self._coverage[order_id] = cov
        self._meta[order_id] = header

    # ---------------- 签发 ---------------- #
    def issue(self, order_id: str, dataset: Any,
              start: int, end: int) -> WorkOrder:
        """用工单号、已装配 Dataset 与里程窗签发工单。

        只纳入窗内达标复测段上的 MISSING 区间；GAP、不达标段、窗外格不计，
        范围允许为空。签发后只保留区间快照，不依赖 ``dataset`` 或块明细。
        """
        from .dataset import Dataset  # 局部导入避免环依赖

        if not isinstance(order_id, str) or not order_id:
            raise OrderError("工单号必须是非空字符串")
        if order_id in self._index:
            raise OrderError(f"工单号已被使用: {order_id}")
        if not isinstance(dataset, Dataset):
            raise OrderError("签发必须提供已装配的 Dataset")
        if not (_is_int(start) and _is_int(end)):
            raise OrderError("里程窗端点必须是整数")
        if not (dataset.line_start <= start < end <= dataset.line_end):
            raise OrderError(
                f"里程窗 [{start},{end}) 非法或越出线路 "
                f"[{dataset.line_start},{dataset.line_end})"
            )

        # 复用既有窗口读取：只触碰与窗相交的块，段门槛按全线段判定。
        scope_pts: List[Tuple[int, int]] = []
        for hit in dataset.query_window(start, end):
            if not hit.qualified:
                continue  # 不达标段不计
            for ev in hit.evidence:
                if ev.state is State.MISSING:
                    scope_pts.append((ev.start, ev.end))
                # GAP（无论是否被跨过）与 OK 一律不计入工单。
        scope = merge_intervals(scope_pts)

        order = WorkOrder(
            order_id=order_id,
            window=(start, end),
            line_window=(dataset.line_start, dataset.line_end),
            scope=scope,
            due_total=sum(e - s for s, e in scope),
        )
        cov = _Coverage(scope)
        self._seq += 1
        fname = f"{_PREFIX}{self._seq:05d}{_SUFFIX}"
        header = {
            "type": "header",
            "order_id": order_id,
            "window": [start, end],
            "line_window": [dataset.line_start, dataset.line_end],
            "scope": [list(x) for x in scope],
            "due_total": order.due_total,
        }
        path = os.path.join(self.directory, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._index[order_id] = fname
        self._coverage[order_id] = cov
        self._meta[order_id] = header
        self._flush_index()
        return order

    # ---------------- 记账 ---------------- #
    def record_batch(self, order_id: str, entries: List[Mapping[str, Any]]
                     ) -> BatchResult:
        """成批登记回执；任一记录无效或同版冲突则整批拒绝（零写入）。

        严格两阶段：先在**不改动任何状态**的前提下完成全部记录的规范化、
        批内去重/冲突与历史同版冲突检测；全部通过后才逐条应用并追加一行
        批次日志。因此拒绝路径不可能留下局部进度或占用修订号。
        """
        cov, header = self._require(order_id)
        window = (header["window"][0], header["window"][1])
        if not isinstance(entries, (list, tuple)) or not entries:
            raise OrderError("批次必须包含至少一条回执记录")

        # --- 阶段一：全量校验，禁止任何 apply/落盘 ---
        canonical: List[Dict[str, Any]] = []
        seen: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for entry in entries:
            rec = _validate_entry(entry, window)
            key = (rec["rid"], rec["rev"])
            if key in seen:
                # 批内原样重送允许；内容冲突整批拒绝。
                if not _rec_equal(seen[key], rec):
                    raise OrderError(
                        f"批内同号同版本内容冲突: {rec['rid']} r{rec['rev']}"
                    )
                continue
            hist = cov.history.get(rec["rid"])
            if hist is not None and rec["rev"] in hist \
                    and not _rec_equal(hist[rec["rev"]], rec):
                raise OrderError(
                    f"回执 {rec['rid']} 修订号 {rec['rev']} 已存在且内容冲突"
                )
            seen[key] = rec
            canonical.append(rec)

        # --- 阶段二：全部合法，逐条应用并以单行原子追加批次日志 ---
        for rec in canonical:
            cov.apply(rec)
        path = os.path.join(self.directory, self._index[order_id])
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"type": "batch",
                 "entries": [self._rec_to_json(r) for r in canonical]},
                ensure_ascii=False, sort_keys=True,
            ) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return BatchResult(order_id=order_id, applied=len(canonical))

    # ---------------- 读取 ---------------- #
    def read_window(self, order_id: str, start: int, end: int) -> WindowView:
        cov, header = self._require(order_id)
        window = (header["window"][0], header["window"][1])
        return cov.read_window(order_id, start, end, window)

    def get_order(self, order_id: str) -> WorkOrder:
        _, header = self._require(order_id)
        return WorkOrder(
            order_id=order_id,
            window=tuple(header["window"]),
            line_window=tuple(header["line_window"]),
            scope=tuple((a, b) for a, b in header["scope"]),
            due_total=header["due_total"],
        )

    def list_orders(self) -> Tuple[str, ...]:
        return tuple(sorted(self._index))

    def access_stats(self, order_id: str) -> AccessStats:
        cov, _ = self._require(order_id)
        return cov.access

    def _require(self, order_id: str) -> Tuple[_Coverage, Mapping[str, Any]]:
        if not isinstance(order_id, str) or order_id not in self._index:
            raise OrderError(f"未知工单号: {order_id!r}")
        return self._coverage[order_id], self._meta[order_id]
