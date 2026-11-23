"""落盘复测工单对账的端到端测试。

复跑方法（隔离模式，忽略环境变量与已安装包，保证不加载外部 json.py 等）::

    python -I run_tests.py                 # 跑全部测试（管线 + 工单）

本文件覆盖题目点名的小数据独立核对：

* ``test_same_crew_overlap_does_not_confirm`` —— 同班组多回执重叠不算确认；
* ``test_revoke_allows_backfill_crew``        —— 撤销后班组替补；
* ``test_revision_migration_crew_and_range_change`` —— 修订迁移（换班组/区间/撤销）；
* ``test_gap_never_in_scope`` / ``test_accept_on_gap_only_is_valid_but_covers_nothing``
                                              —— GAP 格不进工单、不产生覆盖；
* ``test_conflicting_batch_atomic_no_partial_progress`` 与
  ``test_rejected_batch_leaves_no_disk_trace`` / ``test_reopen_reads_identical_results``
                                              —— 冲突批次整批拒绝后的重开；
* ``LocalityTests`` —— 动态生成“远处回执”，用访问计数证明局部访问量，
  全程不提交大夹具（亿级里程在测试内即时构造）；
* ``test_random_batches_differential`` —— 逐格暴力参考对随机回执批次做差分。
"""

import os
import random
import shutil
import tempfile
import unittest

from railway.dataset import assemble, build_block
from railway.model import Interval, Rule, State
from railway.orders import (
    ACCEPT,
    REVOKE,
    OrderError,
    OrderState,
    WorkOrderStore,
)


# ---------------------------------------------------------------------- #
# 测试夹具
# ---------------------------------------------------------------------- #
def make_ds(rules=None, intervals=None, end=30):
    """全线 v1 单块夹具（默认含达标段、不达标段、GAP、OK）。

    规则 v1：max_gap=2, min_missing=4。段结构：
      * [0,2)   MISSING×2，不达标（2<4）；
      * [4,16)  MISSING×5 + 可跨过的短 GAP（[9,10) 长1、[13,15) 长2）
                + MISSING×1，共 6 个 MISSING，唯一达标段；
      * [19,21) MISSING×2，不达标；其余为 OK。
    """
    rules = rules or {1: Rule(1, max_gap=2, min_missing=4)}
    if intervals is None:
        intervals = [
            Interval(0, 2, State.MISSING),    # 不达标段 [0,2)
            Interval(2, 4, State.OK),
            Interval(4, 9, State.MISSING),    # 达标段 [4,16) 的前 5 个 MISSING
            Interval(9, 10, State.GAP),       # 可跨过的短缺报（长 1<=2）
            Interval(10, 13, State.MISSING),
            Interval(13, 15, State.GAP),      # 核内短缺报（长 2）
            Interval(15, 16, State.MISSING),
            Interval(16, 18, State.OK),
            Interval(18, 19, State.GAP),
            Interval(19, 21, State.MISSING),  # 段 [19,21)，不达标
            Interval(21, 30, State.OK),
        ]
    return build_block("B0", 0, end, intervals, rules[1]), rules


def split_blocks(intervals, rules, cuts, end=30):
    """把同一套区间按切点切成多块（单版次），用于核对与分块无关。"""
    bounds = [0] + sorted(cuts) + [end]
    blocks = []
    for bi, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        ivs = []
        for iv in intervals:
            if iv.end <= lo or iv.start >= hi:
                continue
            ivs.append(Interval(max(iv.start, lo), min(iv.end, hi), iv.state))
        blocks.append(build_block(f"B{bi}", lo, hi, ivs, rules[1]))
    return blocks


# ---------------------------------------------------------------------- #
# 逐格暴力参考（仅测试用）：直接维护每个应测格的班组->回执集合
# ---------------------------------------------------------------------- #
class BruteForceLedger:
    def __init__(self, scope):
        self.cells = {}  # mile -> {crew: set(rid)}
        for lo, hi in scope:
            for x in range(lo, hi):
                self.cells[x] = {}
        self.history = {}   # rid -> {rev: rec}
        self.active = {}    # rid -> 当前最高修订 rec

    def apply(self, rec):
        rid, rev = rec["rid"], rec["rev"]
        self.history.setdefault(rid, {})[rev] = rec
        old = self.active.get(rid)
        new = self.history[rid][max(self.history[rid])]
        self.active[rid] = new
        # 先完整摘除该回执号的旧覆盖（遍历所有班组，避免班组改名后遗留）。
        if old is not None and old["kind"] == ACCEPT:
            for lo, hi in old["intervals"]:
                for x in range(lo, hi):
                    cov = self.cells.get(x)
                    if cov is None:
                        continue
                    for crew in list(cov):
                        cov[crew].discard(rid)
                        if not cov[crew]:
                            del cov[crew]
        # 再按新形态加入。
        if new["kind"] == ACCEPT:
            for lo, hi in new["intervals"]:
                for x in range(lo, hi):
                    if x in self.cells:
                        self.cells[x].setdefault(new["crew"], set()).add(rid)

    def view(self, qlo, qhi):
        """返回 (runs, (due, confirmed, pending))，runs 元素为 (s,e,state,ev)。"""
        due = len(self.cells)
        confirmed_cells = set(
            x for x, cov in self.cells.items() if len(cov) >= 2
        )
        confirmed, pending = len(confirmed_cells), due - len(confirmed_cells)
        raw = []
        for x in range(qlo, qhi):
            if x not in self.cells:
                continue
            cov = self.cells[x]
            if len(cov) < 2:
                raw.append((x, x + 1, OrderState.PENDING, ()))
            else:
                ev = []
                for crew in sorted(cov)[:2]:
                    rid = min(cov[crew])
                    ev.append((crew, rid, self.active[rid]["rev"]))
                raw.append((x, x + 1, OrderState.CONFIRMED, tuple(ev)))
        merged = []
        for s, e, st, ev in raw:
            if merged and merged[-1][1] == s and merged[-1][2] == st \
                    and merged[-1][3] == ev:
                merged[-1] = (merged[-1][0], e, st, ev)
            else:
                merged.append((s, e, st, ev))
        return merged, (due, confirmed, pending)


def ev_tuples(view):
    return [
        (r.start, r.end, r.state,
         tuple((e.crew, e.receipt_id, e.revision) for e in r.evidence))
        for r in view.runs
    ]


# ---------------------------------------------------------------------- #
# 签发
# ---------------------------------------------------------------------- #
class IssueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wo-")
        self.block, self.rules = make_ds()
        self.ds = assemble([self.block], 0, 30, self.rules)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_scope_only_qualified_missing_in_window(self):
        store = WorkOrderStore.open(self.tmp, create=True)
        order = store.issue("WO1", self.ds, 5, 25)
        # 唯一达标段 [4,16) 在窗 [5,25) 内的 MISSING：[5,9)+[10,13)+[15,16)；
        # 短 GAP [9,10)、[13,15) 与窗外/OK/不达标段都不得进入工单。
        self.assertEqual(order.scope, ((5, 9), (10, 13), (15, 16)))
        self.assertEqual(order.due_total, 8)

        # 换一种分块，同一套数据，工单范围必须一致（不依赖分块方式）。
        blocks2 = split_blocks(
            self.block._intervals, self.rules, [4, 12, 17, 22]
        )
        ds2 = assemble(blocks2, 0, 30, self.rules)
        store2 = WorkOrderStore.open(
            os.path.join(self.tmp, "s2"), create=True
        )
        order2 = store2.issue("WO1", ds2, 5, 25)
        self.assertEqual(order2.scope, order.scope)
        self.assertEqual(order2.due_total, order.due_total)

    def test_window_clip_qualified_segment(self):
        store = WorkOrderStore.open(self.tmp, create=True)
        # 窗只切到达标段的一部分：[11,14) 内 MISSING 为 [11,13)，
        # [13,14) 是 GAP 不计；门槛仍按全线段判定（qualified=True）。
        order = store.issue("WO2", self.ds, 11, 14)
        self.assertEqual(order.scope, ((11, 13),))
        self.assertEqual(order.due_total, 2)

    def test_gap_never_in_scope(self):
        store = WorkOrderStore.open(self.tmp, create=True)
        # 窗只落在 GAP 带 [13,15)：没有任何应复测格。
        order = store.issue("WO3", self.ds, 13, 15)
        self.assertEqual(order.scope, ())
        self.assertEqual(order.due_total, 0)
        view = store.read_window("WO3", 13, 15)
        self.assertEqual(view.runs, ())
        self.assertEqual(view.totals, (0, 0, 0))

    def test_unqualified_segment_excluded_even_all_missing(self):
        # 达标段 [4,16) 实有 9 个 MISSING；min_missing 提到 10 后整段不达标，
        # 其余段也都不足 10，工单范围为空。
        rules = {1: Rule(1, max_gap=2, min_missing=10)}
        block, _ = make_ds(rules=rules)
        ds = assemble([block], 0, 30, rules)
        store = WorkOrderStore.open(
            os.path.join(self.tmp, "uq"), create=True
        )
        order = store.issue("WO4", ds, 5, 25)
        self.assertEqual(order.scope, ())
        self.assertEqual(order.due_total, 0)

    def test_issue_validation_and_unused_id(self):
        store = WorkOrderStore.open(self.tmp, create=True)
        store.issue("WO1", self.ds, 10, 16)
        with self.assertRaises(OrderError):
            store.issue("WO1", self.ds, 10, 16)  # 工单号必须未用过
        with self.assertRaises(OrderError):
            store.issue("", self.ds, 10, 16)
        with self.assertRaises(OrderError):
            store.issue("WO9", self.ds, 16, 16)  # 非正长度窗
        with self.assertRaises(OrderError):
            store.issue("WO9", self.ds, -1, 16)
        with self.assertRaises(OrderError):
            store.issue("WO9", self.ds, 10, 31)
        with self.assertRaises(OrderError):
            store.issue("WO9", object(), 10, 16)  # 必须是已装配 Dataset

    def test_issue_snapshot_independent_of_correction(self):
        store = WorkOrderStore.open(self.tmp, create=True)
        order_before = store.issue("WO1", self.ds, 5, 25)
        reads_before = self.block.interval_reads
        # 线路更正（达标段被重测为 OK），旧工单范围与对账不反写线路。
        fixed = build_block("B0", 0, 30, [
            Interval(0, 30, State.OK),
        ], self.rules[1])
        ds2 = self.ds.correct(fixed)
        order_after = store.get_order("WO1")
        self.assertEqual(order_after.scope, order_before.scope)
        self.assertEqual(order_after.due_total, order_before.due_total)
        view = store.read_window("WO1", 10, 16)
        self.assertEqual(view.due_total, order_before.due_total)
        # 对账过程不读取任何块明细（工单不持有 Dataset/块引用）。
        self.assertEqual(self.block.interval_reads, reads_before)
        # 新数据集是另一条线路，旧工单不随它变化；新签发基于新线路。
        self.assertEqual(ds2.qualified_segments(), ())
        order2 = store.issue("WO2", ds2, 0, 30)
        self.assertEqual(order2.scope, ())


# ---------------------------------------------------------------------- #
# 记账校验与批次原子性
# ---------------------------------------------------------------------- #
class BatchValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wo-")
        self.block, self.rules = make_ds()
        self.ds = assemble([self.block], 0, 30, self.rules)
        self.store = WorkOrderStore.open(self.tmp, create=True)
        self.store.issue("WO1", self.ds, 5, 25)
        # scope：[5,9)+[10,13)+[15,16)，共 8 个应复测格

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_invalid_records_rejected(self):
        with self.assertRaises(OrderError):
            self.store.record_batch("WO1", [])  # 空批次
        cases = [
            {"rid": "", "rev": 1, "kind": ACCEPT,
             "crew": "C1", "intervals": [[10, 13]]},                 # 空回执号
            {"rid": "R1", "rev": 0, "kind": ACCEPT,
             "crew": "C1", "intervals": [[10, 13]]},                 # 修订号非正
            {"rid": "R1", "rev": True, "kind": ACCEPT,
             "crew": "C1", "intervals": [[10, 13]]},                 # 布尔修订号
            {"rid": "R1", "rev": 1, "kind": "pass",
             "crew": "C1", "intervals": [[10, 13]]},                 # 非法 kind
            {"rid": "R1", "rev": 1, "kind": ACCEPT,
             "crew": "", "intervals": [[10, 13]]},                   # 空班组
            {"rid": "R1", "rev": 1, "kind": ACCEPT,
             "crew": "C1", "intervals": []},                         # 空区间
            {"rid": "R1", "rev": 1, "kind": ACCEPT,
             "crew": "C1", "intervals": [[13, 13]]},                 # 零长度
            {"rid": "R1", "rev": 1, "kind": ACCEPT,
             "crew": "C1", "intervals": [[4, 13]]},                  # 越工单窗左界
            {"rid": "R1", "rev": 1, "kind": REVOKE,
             "crew": "C1", "intervals": [[10, 13]]},                 # 撤销带班组区间
            {"rid": "R1", "rev": 1, "kind": REVOKE,
             "crew": None, "intervals": [[10, 13]]},                 # 撤销带区间
            {"rid": "R1", "rev": 1, "kind": REVOKE,
             "crew": "C1", "intervals": None},                       # 撤销带班组
        ]
        for bad in cases:
            with self.assertRaises(OrderError, msg=f"应拒绝 {bad}"):
                self.store.record_batch("WO1", [bad])

    def test_accept_on_gap_only_is_valid_but_covers_nothing(self):
        # [13,15) 全是 GAP：区间在工单窗内、记录合法，但与 scope 不相交，
        # 不产生任何覆盖（GAP 格本就不在工单里）。
        self.store.record_batch("WO1", [
            {"rid": "R1", "rev": 1, "kind": ACCEPT,
             "crew": "C1", "intervals": [[13, 15]]},
        ])
        view = self.store.read_window("WO1", 5, 25)
        self.assertEqual(view.totals, (8, 0, 8))
        cov = self.store._coverage["WO1"]
        self.assertEqual(len(cov._starts), 0)

    def test_unknown_order_rejected(self):
        with self.assertRaises(OrderError):
            self.store.record_batch("NOPE", [
                {"rid": "R1", "rev": 1, "kind": ACCEPT,
                 "crew": "C1", "intervals": [[10, 13]]}
            ])

    def test_conflicting_batch_atomic_no_partial_progress(self):
        # 先成功登记一批。
        self.store.record_batch("WO1", [
            {"rid": "R1", "rev": 1, "kind": ACCEPT,
             "crew": "C1", "intervals": [[10, 13]]},
        ])
        before = self.store.read_window("WO1", 10, 16)
        before_files = self._log_lines()
        # 冲突批次：同号同版本内容不同（且夹带另一条本可成功的记录）。
        with self.assertRaises(OrderError):
            self.store.record_batch("WO1", [
                {"rid": "R1", "rev": 1, "kind": ACCEPT,
                 "crew": "C9", "intervals": [[10, 13]]},  # 同版换班组=冲突
                {"rid": "R2", "rev": 1, "kind": ACCEPT,
                 "crew": "C2", "intervals": [[10, 13]]},  # 不得局部生效
            ])
        # 内存状态无局部进度。
        after = self.store.read_window("WO1", 10, 16)
        self.assertEqual(ev_tuples(after), ev_tuples(before))
        self.assertEqual(after.totals, before.totals)
        # 磁盘无新增批次行，修订号未被占用。
        self.assertEqual(self._log_lines(), before_files)
        # 重开后结果一致。
        reopened = WorkOrderStore.open(self.tmp)
        self.assertEqual(
            ev_tuples(reopened.read_window("WO1", 10, 16)),
            ev_tuples(before),
        )
        # 被拒后同修订号仍可用于新内容（未“占用”修订号）。
        self.store.record_batch("WO1", [
            {"rid": "R2", "rev": 1, "kind": ACCEPT,
             "crew": "C2", "intervals": [[10, 13]]},
        ])
        view = self.store.read_window("WO1", 10, 13)
        self.assertEqual(view.runs[0].state, OrderState.CONFIRMED)

    def test_intra_batch_duplicate_conflict_and_resend(self):
        with self.assertRaises(OrderError):
            self.store.record_batch("WO1", [
                {"rid": "R1", "rev": 1, "kind": ACCEPT,
                 "crew": "C1", "intervals": [[10, 13]]},
                {"rid": "R1", "rev": 1, "kind": ACCEPT,
                 "crew": "C2", "intervals": [[10, 13]]},  # 批内同版冲突
            ])
        rec = {"rid": "R1", "rev": 1, "kind": ACCEPT,
               "crew": "C1", "intervals": [[10, 13]]}
        r1 = self.store.record_batch("WO1", [rec, dict(rec)])  # 原样重送
        self.assertEqual(r1.applied, 1)
        # 跨批次原样重送同样幂等，不新增覆盖、不报错。
        r2 = self.store.record_batch("WO1", [dict(rec)])
        self.assertEqual(r2.applied, 1)
        view = self.store.read_window("WO1", 10, 13)
        self.assertEqual(view.runs[0].state, OrderState.PENDING)

    def test_stale_revision_conflict_but_old_revision_ok(self):
        # 先到 rev=3，再补送 rev=1（乱序）：不冲突则接受，生效仍是 rev=3。
        self.store.record_batch("WO1", [
            {"rid": "R1", "rev": 3, "kind": ACCEPT,
             "crew": "C1", "intervals": [[10, 13]]},
        ])
        self.store.record_batch("WO1", [
            {"rid": "R1", "rev": 1, "kind": ACCEPT,
             "crew": "C1", "intervals": [[10, 12]]},
        ])
        view = self.store.read_window("WO1", 10, 13)
        # 生效的是 rev=3（覆盖到 13）；单班组不确认，故无凭证，只核对范围。
        self.assertEqual(view.runs[-1].end, 13)
        self.assertTrue(all(r.state is OrderState.PENDING for r in view.runs))
        # 再补一个班组确认后，证据修订号显示当前生效的 rev=3。
        self.store.record_batch("WO1", [
            {"rid": "R2", "rev": 1, "kind": ACCEPT,
             "crew": "C2", "intervals": [[10, 13]]},
        ])
        view = self.store.read_window("WO1", 10, 13)
        self.assertEqual(view.runs[0].state, OrderState.CONFIRMED)
        self.assertEqual(
            {e.revision for r in view.runs for e in r.evidence}, {3, 1}
        )
        # 过期版本若内容与历史不同，仍按同版冲突整批拒绝。
        with self.assertRaises(OrderError):
            self.store.record_batch("WO1", [
                {"rid": "R1", "rev": 1, "kind": ACCEPT,
                 "crew": "C1", "intervals": [[11, 12]]},
            ])

    def _log_lines(self):
        fname = self.store._index["WO1"]
        with open(os.path.join(self.tmp, fname), encoding="utf-8") as f:
            return f.read().splitlines()


# ---------------------------------------------------------------------- #
# 对账语义：覆盖、确认、凭证、修订、撤销
# ---------------------------------------------------------------------- #
class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wo-")
        self.block, self.rules = make_ds()
        self.ds = assemble([self.block], 0, 30, self.rules)
        self.store = WorkOrderStore.open(self.tmp, create=True)
        # scope：[5,9)+[10,13)+[15,16)，共 8 个应复测格
        self.order = self.store.issue("WO1", self.ds, 5, 25)
        self.brute = BruteForceLedger(self.order.scope)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _check(self, qlo, qhi):
        got = self.store.read_window("WO1", qlo, qhi)
        want_runs, want_totals = self.brute.view(qlo, qhi)
        self.assertEqual(ev_tuples(got), want_runs)
        self.assertEqual(got.totals, want_totals)

    def _batch(self, entries):
        recs = []
        for e in entries:
            rec = dict(e)
            recs.append(rec)
        self.store.record_batch("WO1", recs)
        for rec in recs:
            self.brute.apply(rec)

    def test_same_crew_overlap_does_not_confirm(self):
        # 同一班组两个不同回执重叠覆盖，班组数仍为 1，不确认。
        self._batch([
            {"rid": "R1", "rev": 1, "kind": ACCEPT,
             "crew": "C1", "intervals": [[10, 13]]},
            {"rid": "R2", "rev": 1, "kind": ACCEPT,
             "crew": "C1", "intervals": [[11, 13]]},
        ])
        self._check(10, 16)
        view = self.store.read_window("WO1", 10, 16)
        self.assertTrue(all(r.state is OrderState.PENDING for r in view.runs))
        self.assertEqual(view.totals, (8, 0, 8))

    def test_two_crews_confirm_and_evidence_lexicographic(self):
        self._batch([
            {"rid": "R9", "rev": 1, "kind": ACCEPT,
             "crew": "C9", "intervals": [[10, 13]]},
            {"rid": "R2", "rev": 1, "kind": ACCEPT,
             "crew": "C2", "intervals": [[10, 13]]},
            {"rid": "R1", "rev": 1, "kind": ACCEPT,
             "crew": "C1", "intervals": [[10, 12]]},
        ])
        self._check(10, 16)
        view = self.store.read_window("WO1", 10, 16)
        # [10,12) 有三个班组：凭证取字典序最前 C1、C2，各取最小回执号。
        first = view.runs[0]
        self.assertEqual(first.start, 10)
        self.assertEqual(first.state, OrderState.CONFIRMED)
        self.assertEqual(
            [(e.crew, e.receipt_id, e.revision) for e in first.evidence],
            [("C1", "R1", 1), ("C2", "R2", 1)],
        )
        # [12,13) 只有 C2、C9：最前两班组即 C2、C9。
        second = next(r for r in view.runs if r.start == 12)
        self.assertEqual(
            [e.crew for e in second.evidence], ["C2", "C9"]
        )
        # [15,16) 无人覆盖 -> 待确认；[5,10) 也无人覆盖（5 格）。
        self.assertEqual(view.totals, (8, 3, 5))

    def test_evidence_uses_min_receipt_and_current_revision(self):
        # 同一班组 C1 用两个回执号覆盖；凭证取该位置最小回执号，
        # 且修订号显示该最小回执号的当前（最高）修订，而非首次修订。
        self._batch([
            {"rid": "R8", "rev": 1, "kind": ACCEPT,
             "crew": "C1", "intervals": [[10, 13]]},
            {"rid": "R2", "rev": 1, "kind": ACCEPT,
             "crew": "C1", "intervals": [[10, 13]]},  # 同班组，不确认
            {"rid": "R5", "rev": 1, "kind": ACCEPT,
             "crew": "C2", "intervals": [[10, 13]]},
        ])
        # 最小回执号 R2 升到 rev=4（换区间仍覆盖该位置），凭证应显示 rev=4。
        self._batch([
            {"rid": "R2", "rev": 4, "kind": ACCEPT,
             "crew": "C1", "intervals": [[10, 13]]},
        ])
        view = self.store.read_window("WO1", 10, 13)
        self.assertEqual(view.runs[0].state, OrderState.CONFIRMED)
        self.assertEqual(
            [(e.crew, e.receipt_id, e.revision)
             for e in view.runs[0].evidence],
            [("C1", "R2", 4), ("C2", "R5", 1)],
        )

    def test_revoke_allows_backfill_crew(self):
        self._batch([
            {"rid": "R1", "rev": 1, "kind": ACCEPT,
             "crew": "C1", "intervals": [[10, 13]]},
            {"rid": "R2", "rev": 1, "kind": ACCEPT,
             "crew": "C2", "intervals": [[10, 13]]},
        ])
        self._check(10, 16)
        self.assertEqual(
            self.store.read_window("WO1", 10, 13).confirmed_total, 3
        )
        # 撤销 C2 的 R2：位置回落到待确认；随后 C3 替补重新确认。
        self._batch([
            {"rid": "R2", "rev": 2, "kind": REVOKE},
        ])
        self._check(10, 16)
        view = self.store.read_window("WO1", 10, 13)
        self.assertTrue(all(r.state is OrderState.PENDING for r in view.runs))
        self.assertEqual(view.totals, (8, 0, 8))
        self._batch([
            {"rid": "R3", "rev": 1, "kind": ACCEPT,
             "crew": "C3", "intervals": [[10, 13]]},
        ])
        self._check(10, 16)
        view = self.store.read_window("WO1", 10, 13)
        self.assertTrue(all(r.state is OrderState.CONFIRMED for r in view.runs))
        self.assertEqual(
            [e.crew for e in view.runs[0].evidence], ["C1", "C3"]
        )

    def test_revision_migration_crew_and_range_change(self):
        # 换班组、换区间、再撤销，全部只以最高修订生效。
        self._batch([
            {"rid": "R1", "rev": 1, "kind": ACCEPT,
             "crew": "C1", "intervals": [[10, 13]]},
            {"rid": "R2", "rev": 1, "kind": ACCEPT,
             "crew": "C2", "intervals": [[10, 13]]},
        ])
        # R1 升 rev=2：班组 C1->C3，区间缩到 [10,12)。
        self._batch([
            {"rid": "R1", "rev": 2, "kind": ACCEPT,
             "crew": "C3", "intervals": [[10, 12]]},
        ])
        self._check(10, 16)
        view = self.store.read_window("WO1", 10, 16)
        # [10,12) 班组为 C2、C3 -> 确认；[12,13) 只剩 C2 -> 待确认。
        states = {(r.start, r.end): r.state for r in view.runs}
        self.assertEqual(states[(10, 12)], OrderState.CONFIRMED)
        self.assertEqual(states[(12, 13)], OrderState.PENDING)
        # R1 再升 rev=3 撤销：全部回到仅 C2。
        self._batch([{"rid": "R1", "rev": 3, "kind": REVOKE}])
        self._check(10, 16)
        self.assertEqual(
            self.store.read_window("WO1", 10, 16).totals, (8, 0, 8)
        )
        # 撤销后再用更高修订恢复（同一回执号可复活）。
        self._batch([
            {"rid": "R1", "rev": 4, "kind": ACCEPT,
             "crew": "C1", "intervals": [[15, 16]]},
        ])
        self._check(10, 16)
        view = self.store.read_window("WO1", 15, 16)
        self.assertEqual(view.runs[0].state, OrderState.PENDING)  # 仅 C1

    def test_adjacent_runs_merge_on_same_state_and_evidence(self):
        # 两张回执都覆盖整个 scope，状态/凭证相同，相邻段必须合并。
        self._batch([
            {"rid": "R1", "rev": 1, "kind": ACCEPT,
             "crew": "C1",
             "intervals": [[5, 9], [10, 13], [15, 16]]},
            {"rid": "R2", "rev": 1, "kind": ACCEPT,
             "crew": "C2",
             "intervals": [[5, 9], [10, 13], [15, 16]]},
        ])
        view = self.store.read_window("WO1", 5, 25)
        # 三个 scope 区间隔着 GAP/OK（非 scope），相邻性只在 scope 内部判定，
        # 因此是三个 ViewRun，但各自内部不碎、状态与凭证完全一致。
        self.assertEqual(
            [(r.start, r.end) for r in view.runs],
            [(5, 9), (10, 13), (15, 16)],
        )
        self.assertTrue(
            all(r.state is OrderState.CONFIRMED for r in view.runs)
        )
        self.assertEqual(view.totals, (8, 8, 0))
        # 同一 scope 区间内部再切窗，相邻同态同凭证段必须合并为一段。
        tiny = self.store.read_window("WO1", 6, 8)
        self.assertEqual(
            [(r.start, r.end) for r in tiny.runs], [(6, 8)]
        )

    def test_window_clip_and_invariant_totals(self):
        self._batch([
            {"rid": "R1", "rev": 1, "kind": ACCEPT,
             "crew": "C1", "intervals": [[10, 13]]},
            {"rid": "R2", "rev": 1, "kind": ACCEPT,
             "crew": "C2", "intervals": [[10, 13]]},
        ])
        full = self.store.read_window("WO1", 5, 25)
        tiny = self.store.read_window("WO1", 11, 12)   # 单格窗
        edge = self.store.read_window("WO1", 12, 16)
        self.assertEqual(full.totals, tiny.totals)
        self.assertEqual(tiny.totals, edge.totals)
        self.assertEqual(full.totals, (8, 3, 5))
        self.assertEqual([(r.start, r.end) for r in tiny.runs], [(11, 12)])
        self.assertEqual(
            [(r.start, r.end) for r in edge.runs],
            [(12, 13), (15, 16)],
        )
        with self.assertRaises(OrderError):
            self.store.read_window("WO1", 13, 13)
        with self.assertRaises(OrderError):
            self.store.read_window("WO1", 4, 13)   # 越工单窗左界
        with self.assertRaises(OrderError):
            self.store.read_window("WO1", 10, 26)

    def test_random_batches_differential(self):
        rng = random.Random(20261123)
        crews = ["C1", "C2", "C3", "C4"]
        rids = [f"R{i}" for i in range(8)]
        for step in range(120):
            entries = []
            for _ in range(rng.randint(1, 4)):
                rid = rng.choice(rids)
                rev = rng.randint(1, 4)
                if rng.random() < 0.25:
                    entries.append(
                        {"rid": rid, "rev": rev, "kind": REVOKE}
                    )
                    continue
                a = rng.randint(10, 15)
                b = rng.randint(a + 1, 16)
                entries.append({
                    "rid": rid, "rev": rev, "kind": ACCEPT,
                    "crew": rng.choice(crews),
                    "intervals": [[a, b]],
                })
            # 引擎可能因同版冲突拒绝；参考侧只在引擎接受时同步应用。
            # 必须复刻引擎的批内去重（原样重送只生效一次）与区间规范化，
            # 否则参考侧会比引擎多应用记录，造成假性分歧。
            try:
                result = self.store.record_batch("WO1", entries)
            except OrderError:
                continue
            seen_keys = set()
            applied = []
            for rec in entries:
                key = (rec["rid"], rec["rev"])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                if rec["kind"] == ACCEPT:
                    norm = dict(
                        rec,
                        intervals=tuple((lo, hi)
                                        for lo, hi in rec["intervals"]),
                    )
                else:
                    norm = dict(rec, crew=None, intervals=None)
                applied.append(norm)
            self.assertEqual(result.applied, len(applied))
            for norm in applied:
                self.brute.apply(norm)
            for _ in range(4):
                a = rng.randint(5, 24)
                b = rng.randint(a + 1, 25)
                self._check(a, b)


# ---------------------------------------------------------------------- #
# 落盘重开
# ---------------------------------------------------------------------- #
class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wo-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(self, sub):
        block, rules = make_ds()
        ds = assemble([block], 0, 30, rules)
        store = WorkOrderStore.open(os.path.join(self.tmp, sub), create=True)
        store.issue("WO1", ds, 5, 25)
        store.issue("WO2", ds, 0, 30)
        return store

    def test_reopen_reads_identical_results(self):
        store = self._build("a")
        batches = [
            [{"rid": "R1", "rev": 1, "kind": ACCEPT,
              "crew": "C1", "intervals": [[10, 13]]}],
            [{"rid": "R2", "rev": 1, "kind": ACCEPT,
              "crew": "C2", "intervals": [[10, 13], [15, 16]]}],
            [{"rid": "R1", "rev": 2, "kind": REVOKE}],
            [{"rid": "R3", "rev": 1, "kind": ACCEPT,
              "crew": "C3", "intervals": [[10, 12]]}],
        ]
        for b in batches:
            store.record_batch("WO1", b)
        windows = [(5, 25), (10, 16), (11, 13), (15, 16), (12, 16)]
        before = {
            w: (ev_tuples(store.read_window("WO1", *w)),
                store.read_window("WO1", *w).totals)
            for w in windows
        }
        order_before = store.get_order("WO1")

        reopened = WorkOrderStore.open(os.path.join(self.tmp, "a"))
        self.assertEqual(set(reopened.list_orders()), {"WO1", "WO2"})
        self.assertEqual(reopened.get_order("WO1").scope, order_before.scope)
        for w in windows:
            view = reopened.read_window("WO1", *w)
            self.assertEqual((ev_tuples(view), view.totals), before[w])

        # 重开后继续记账，历史仍然可做同版冲突检测。
        with self.assertRaises(OrderError):
            reopened.record_batch("WO1", [
                {"rid": "R1", "rev": 1, "kind": ACCEPT,
                 "crew": "CX", "intervals": [[10, 12]]},
            ])
        reopened.record_batch("WO1", [
            {"rid": "R4", "rev": 1, "kind": ACCEPT,
             "crew": "C4", "intervals": [[15, 16]]},
        ])
        again = WorkOrderStore.open(os.path.join(self.tmp, "a"))
        self.assertEqual(
            ev_tuples(again.read_window("WO1", 5, 25)),
            ev_tuples(reopened.read_window("WO1", 5, 25)),
        )

    def test_rejected_batch_leaves_no_disk_trace(self):
        store = self._build("b")
        store.record_batch("WO1", [
            {"rid": "R1", "rev": 1, "kind": ACCEPT,
             "crew": "C1", "intervals": [[10, 13]]},
        ])
        fname = store._index["WO1"]
        path = os.path.join(self.tmp, "b", fname)
        with open(path, encoding="utf-8") as f:
            lines_after_ok = f.read().splitlines()
        with self.assertRaises(OrderError):
            store.record_batch("WO1", [
                {"rid": "R1", "rev": 1, "kind": ACCEPT,
                 "crew": "C9", "intervals": [[10, 13]]},
            ])
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read().splitlines(), lines_after_ok)

    def test_torn_final_log_line_tolerated(self):
        store = self._build("c")
        store.record_batch("WO1", [
            {"rid": "R1", "rev": 1, "kind": ACCEPT,
             "crew": "C1", "intervals": [[10, 13]]},
        ])
        fname = store._index["WO1"]
        path = os.path.join(self.tmp, "c", fname)
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"type": "batch", "entr')  # 模拟崩溃留下的半截行
        reopened = WorkOrderStore.open(os.path.join(self.tmp, "c"))
        view = reopened.read_window("WO1", 10, 13)
        self.assertEqual(len(view.runs), 1)
        self.assertEqual(view.runs[0].state, OrderState.PENDING)


# ---------------------------------------------------------------------- #
# 局部访问量：动态生成远处回执，验证访问量与远处里程/回执无关
# ---------------------------------------------------------------------- #
class LocalityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wo-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_far_receipts_do_not_inflate_local_access(self):
        BIG = 1_000_000
        # 稀疏夹具：近处两个应测格 + 远处（动态生成的）N 个独立回执，
        # 全部压在大里程上，避免提交任何大夹具文件。
        rules = {1: Rule(1, max_gap=0, min_missing=1)}
        intervals = [
            Interval(0, 1, State.MISSING),       # 近处应测格 0
            Interval(1, BIG, State.OK),
            Interval(BIG, BIG + 1, State.MISSING),  # 远处应测格
        ]
        block = build_block("B0", 0, BIG + 1, intervals, rules[1])
        ds = assemble([block], 0, BIG + 1, rules)
        store = WorkOrderStore.open(self.tmp, create=True)
        order = store.issue("WO1", ds, 0, BIG + 1)
        self.assertEqual(order.scope, ((0, 1), (BIG, BIG + 1)))
        stats = store.access_stats("WO1")

        # 动态在远处堆 N 个班组的回执（每个只覆盖远处一格）。
        n_far = 200
        entries = [
            {"rid": f"FAR-{i}", "rev": 1, "kind": ACCEPT,
             "crew": f"FC{i % 50:02d}", "intervals": [[BIG, BIG + 1]]}
            for i in range(n_far)
        ]
        stats.reset()
        store.record_batch("WO1", entries)
        # 整批 200 条回执的触达里程始终只在远处那一格，从不靠近处数据；
        # 覆盖只维护一个游程（压缩存储，不随回执数扩张）。
        self.assertEqual((stats.edit_min, stats.edit_max),
                         (BIG, BIG + 1))
        self.assertEqual(len(store._coverage["WO1"]._starts), 1)

        # 近处再登记两个班组：触达范围严格限定在近处一格，远处 200 条不碰。
        stats.reset()
        store.record_batch("WO1", [
            {"rid": "L1", "rev": 1, "kind": ACCEPT,
             "crew": "LC1", "intervals": [[0, 1]]},
            {"rid": "L2", "rev": 1, "kind": ACCEPT,
             "crew": "LC2", "intervals": [[0, 1]]},
        ])
        self.assertEqual((stats.edit_min, stats.edit_max), (0, 1))
        self.assertLessEqual(stats.edit_visits, 2)

        # 近处小窗读取：不读远处覆盖段、不查远处回执。
        stats.reset()
        view = store.read_window("WO1", 0, 1)
        self.assertEqual(len(view.runs), 1)
        self.assertEqual(view.runs[0].state, OrderState.CONFIRMED)
        self.assertEqual(stats.read_run_visits, 1)
        self.assertEqual(stats.scope_probes, 1)
        self.assertEqual(stats.receipt_lookups, 2)

        # 总数不随窗口变化，也不靠扫描远处里程。
        self.assertEqual(view.totals, (2, 2, 0))

        # 再翻倍远处回执，近处访问量保持不变（局部性与规模无关）。
        more = [
            {"rid": f"FAR2-{i}", "rev": 1, "kind": ACCEPT,
             "crew": f"GD{i % 50:02d}", "intervals": [[BIG, BIG + 1]]}
            for i in range(n_far)
        ]
        store.record_batch("WO1", more)
        stats.reset()
        view2 = store.read_window("WO1", 0, 1)
        self.assertEqual(stats.read_run_visits, 1)
        self.assertEqual(stats.receipt_lookups, 2)
        self.assertEqual(view2.totals, (2, 2, 0))
        # 覆盖游程表始终压缩：只有两个非空游程（近、远各一），不逐格展开。
        cov = store._coverage["WO1"]
        self.assertEqual(len(cov._starts), 2)
        self.assertEqual(cov.due_total, 2)

    def test_no_per_cell_expansion_in_storage(self):
        BIG = 10_000_000
        rules = {1: Rule(1, max_gap=0, min_missing=1)}
        block = build_block("B0", 0, BIG, [
            Interval(0, BIG - 1, State.MISSING),
            Interval(BIG - 1, BIG, State.OK),
        ], rules[1])
        ds = assemble([block], 0, BIG, rules)
        store = WorkOrderStore.open(
            os.path.join(self.tmp, "big"), create=True
        )
        order = store.issue("WO1", ds, 0, BIG)
        # 工单头只保存一个区间，而不是一千万个格。
        self.assertEqual(order.scope, ((0, BIG - 1),))
        self.assertEqual(order.due_total, BIG - 1)
        # 两条覆盖整个 scope 的回执，内部仍是常数个游程。
        store.record_batch("WO1", [
            {"rid": "R1", "rev": 1, "kind": ACCEPT,
             "crew": "C1", "intervals": [[0, BIG - 1]]},
            {"rid": "R2", "rev": 1, "kind": ACCEPT,
             "crew": "C2", "intervals": [[0, BIG - 1]]},
        ])
        cov = store._coverage["WO1"]
        self.assertEqual(len(cov._starts), 1)
        view = store.read_window("WO1", 100, 200)
        self.assertEqual(view.runs[0].state, OrderState.CONFIRMED)
        self.assertEqual(view.totals, (BIG - 1, BIG - 1, 0))
        # 落盘文件规模与里程长度无关（头一行 + 批次一行）。
        fname = store._index["WO1"]
        size = os.path.getsize(os.path.join(self.tmp, "big", fname))
        self.assertLess(size, 500)


if __name__ == "__main__":
    unittest.main(verbosity=2)
