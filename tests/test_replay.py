"""出数回放（railway.replay）的端到端测试。

复跑方法（隔离模式）::

    python -I run_tests.py
    python -I -m unittest tests.test_replay -v

覆盖：

* 观测选取（采集时刻最新、同刻编号字典序较大、按到达顺序不决胜）；
* 通告生效（公布时刻最新、同刻编号较大、适用期可重叠/追溯、不可信剔除）；
* 截止时刻过滤、快照不可变、递增/相同/跳过中间时刻/倒退；
* 来源小窗（升序裁切、只合并状态与两编号都相同的相邻区间、GAP 空编号）；
* 无效输入整体拒绝、不产生部分台账；
* 局部性（推进不重扫旧台账、未受影响块不重建、来源小窗不扫全线）；
* 随机差分：独立逐格暴力参考核对状态、来源与复测段。
"""

import random
import unittest

from railway import Interval, Rule, State
from railway.replay import (
    BlockSpec,
    CalibrationNotice,
    Observation,
    ReplayError,
    ReplayLedger,
    Snapshot,
)


def build_ledger(blocks, rules, observations, notices):
    return ReplayLedger(blocks, rules, observations, notices)


def span_tuples(spans):
    return [(s.start, s.end, s.state, s.obs_id, s.notice_id) for s in spans]


# ---------------------------------------------------------------------- #
# 逐格暴力参考：直接展开整条线路
# ---------------------------------------------------------------------- #
class Brute:
    def __init__(self, line_start, line_end, observations, notices):
        self.lo = line_start
        self.hi = line_end
        self.obs = list(observations)
        self.notices = list(notices)

    def notice_for(self, obs, cutoff):
        best = None
        best_key = None
        for n in self.notices:
            if n.device != obs.device:
                continue
            if n.published_at > cutoff:
                continue
            if not (n.applies_start <= obs.acquired_at < n.applies_end):
                continue
            key = (n.published_at, n.notice_id)
            if best_key is None or key > best_key:
                best_key = key
                best = n
        if best is None or not best.trusted:
            return None
        return best

    def cell(self, x, cutoff):
        best = None
        best_key = None
        for o in self.obs:
            if o.arrived_at > cutoff:
                continue
            if not (o.start <= x < o.end):
                continue
            n = self.notice_for(o, cutoff)
            if n is None:
                continue
            key = (o.acquired_at, o.obs_id)
            if best_key is None or key > best_key:
                best_key = key
                best = (o, n)
        if best is None:
            return (State.GAP, None, None)
        o, n = best
        st = State.OK if o.observed else State.MISSING
        return (st, o.obs_id, n.notice_id)

    def spans(self, a, b, cutoff):
        raw = []
        for x in range(a, b):
            key = self.cell(x, cutoff)
            if raw and raw[-1][1] == x and raw[-1][2] == key:
                raw[-1] = (raw[-1][0], x + 1, key)
            else:
                raw.append((x, x + 1, key))
        return [(s, e, k[0], k[1], k[2]) for s, e, k in raw]

    def states(self, cutoff):
        return [self.cell(x, cutoff)[0] for x in range(self.lo, self.hi)]


def random_case(rng, length=None):
    length = length or rng.randint(5, 60)
    lo = rng.randint(-20, 20)
    hi = lo + length
    # 固定分块：随机切点，单版次或两版次
    k = rng.randint(1, min(5, length))
    cuts = sorted(rng.sample(range(1, length), k=k - 1)) if length > 1 else []
    bounds = [lo] + [lo + c for c in cuts] + [hi]
    rules = {}
    blocks = []
    for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
        v = rng.randint(1, 2)
        rules.setdefault(v, Rule(v, max_gap=rng.randint(0, 3),
                               min_missing=rng.randint(0, 4)))
        blocks.append(BlockSpec(f"B{i}", a, b, v))
    rng.shuffle(blocks)

    devices = [f"D{k}" for k in range(rng.randint(1, 3))]
    observations = []
    for i in range(rng.randint(0, 14)):
        a = rng.randint(lo, hi - 1)
        b = rng.randint(a + 1, hi)
        acquired = rng.randint(0, 20)
        arrived = acquired + rng.randint(0, 20)
        observations.append(Observation(
            f"O{i:02d}", rng.choice(devices), acquired, arrived,
            a, b, rng.random() < 0.5,
        ))
    # 编号唯一化（随机可能撞号）
    seen = {}
    uniq = []
    for o in observations:
        if o.obs_id in seen:
            continue
        seen[o.obs_id] = True
        uniq.append(o)
    observations = uniq

    notices = []
    for i in range(rng.randint(0, 8)):
        s = rng.randint(-5, 20)
        e = s + rng.randint(1, 15)
        notices.append(CalibrationNotice(
            f"C{i:02d}", rng.choice(devices), rng.randint(0, 30),
            s, e, rng.random() < 0.65,
        ))
    # 去重编号
    seen = set()
    nn = []
    for n in notices:
        if n.notice_id in seen:
            continue
        seen.add(n.notice_id)
        nn.append(n)
    notices = nn
    return lo, hi, blocks, rules, observations, notices


class DifferentialTests(unittest.TestCase):
    def test_random_against_brute(self):
        rng = random.Random(20261123)
        for trial in range(120):
            lo, hi, blocks, rules, observations, notices = random_case(rng)
            ledger = ReplayLedger(blocks, rules, observations, notices)
            brute = Brute(lo, hi, observations, notices)
            # 截止时刻集合：覆盖所有到达/公布边界附近
            cuts = {0, 40}
            for o in observations:
                cuts.update([o.arrived_at, o.arrived_at + 1])
            for n in notices:
                cuts.update([n.published_at, n.published_at + 1])
            cuts = sorted(cuts)
            prev = None
            for t in cuts:
                snap = ledger.advance(t)
                self.assertIsInstance(snap, Snapshot)
                expect_states = brute.states(t)
                for x in range(lo, hi):
                    sp = snap.source_at(x)
                    self.assertEqual(
                        (sp.state, sp.obs_id, sp.notice_id),
                        brute.cell(x, t),
                        msg=f"trial {trial} t={t} mile={x}",
                    )
                # 来源区间应完整覆盖线路
                spans = span_tuples(snap.source_window(lo, hi))
                self.assertEqual(spans, brute.spans(lo, hi, t),
                                 msg=f"trial {trial} t={t} spans")
                # 复测段与参考的逐格段一致（复用既有 Dataset 语义）
                from railway.reference import reference_segments
                states, versions = self._grid(lo, hi, blocks, snap)
                ref = reference_segments(states, versions, rules, lo)
                self.assertEqual(snap.segments, ref,
                                 msg=f"trial {trial} t={t} segments")
                prev = snap
            self.assertIsNotNone(prev)

    def _grid(self, lo, hi, blocks, snap):
        from railway.reference import cells_from_blocks
        states, versions = cells_from_blocks(snap._blocks, lo, hi)
        return states, versions

    def test_skip_vs_stepwise(self):
        rng = random.Random(99)
        for trial in range(40):
            lo, hi, blocks, rules, observations, notices = random_case(rng)
            a = ReplayLedger(blocks, rules, observations, notices)
            b = ReplayLedger(blocks, rules, observations, notices)
            cuts = {0, 30}
            for o in observations:
                cuts.add(o.arrived_at)
            for n in notices:
                cuts.add(n.published_at)
            cuts = sorted(cuts)
            last = None
            for t in cuts:
                last = a.advance(t)
            step = None
            for t in range(0, cuts[-1] + 1):
                step = b.advance(t)
            self.assertEqual(span_tuples(last.source_window(lo, hi)),
                             span_tuples(step.source_window(lo, hi)))
            self.assertEqual(last.segments, step.segments)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.rule = Rule(1, max_gap=5, min_missing=1)

    def test_latest_acquired_wins_not_arrival(self):
        obs = [
            Observation("OLD", "D", 1, 100, 0, 5, True),
            Observation("NEW", "D", 9, 10, 0, 5, False),
        ]
        ledger = ReplayLedger([("B", 0, 5, 1)], {1: self.rule}, obs,
                              [CalibrationNotice("C", "D", 0, 0, 50, True)])
        snap = ledger.advance(100)
        self.assertEqual(span_tuples(snap.source_window(0, 5)),
                         [(0, 5, State.MISSING, "NEW", "C")])

    def test_tie_on_acquired_uses_lexicographically_larger_id(self):
        obs = [
            Observation("A", "D", 5, 6, 0, 5, True),
            Observation("B", "D", 5, 6, 0, 5, False),
        ]
        ledger = ReplayLedger([("B", 0, 5, 1)], {1: self.rule}, obs,
                              [CalibrationNotice("C", "D", 0, 0, 50, True)])
        snap = ledger.advance(10)
        self.assertEqual(snap.source_at(0).obs_id, "B")
        self.assertEqual(snap.source_at(0).state, State.MISSING)

    def test_gap_has_no_ids(self):
        ledger = ReplayLedger([("B", 0, 5, 1)], {1: self.rule}, [], [])
        snap = ledger.advance(0)
        sp = snap.source_window(0, 5)
        self.assertEqual(len(sp), 1)
        self.assertEqual((sp[0].state, sp[0].obs_id, sp[0].notice_id),
                         (State.GAP, None, None))

    def test_notice_latest_published_wins_overlapping(self):
        obs = [Observation("O", "D", 5, 6, 0, 5, True)]
        notices = [
            CalibrationNotice("TRUST", "D", 10, 0, 20, True),
            CalibrationNotice("DISTRUST", "D", 20, 0, 10, False),
        ]
        ledger = ReplayLedger([("B", 0, 5, 1)], {1: self.rule}, obs, notices)
        self.assertEqual(ledger.advance(15).source_at(0).state, State.OK)
        self.assertEqual(ledger.advance(20).source_at(0).state, State.GAP)

    def test_notice_tie_on_published_lexicographically_larger(self):
        obs = [Observation("O", "D", 5, 6, 0, 5, True)]
        notices = [
            CalibrationNotice("A", "D", 10, 0, 20, True),
            CalibrationNotice("B", "D", 10, 0, 20, False),
        ]
        ledger = ReplayLedger([("B", 0, 5, 1)], {1: self.rule}, obs, notices)
        self.assertEqual(ledger.advance(10).source_at(0).state, State.GAP)

    def test_notice_not_yet_published_does_not_apply(self):
        obs = [Observation("O", "D", 5, 6, 0, 5, True)]
        notices = [CalibrationNotice("C", "D", 50, 0, 20, False)]
        ledger = ReplayLedger([("B", 0, 5, 1)], {1: self.rule}, obs, notices)
        self.assertEqual(ledger.advance(20).source_at(0).state, State.GAP)
        self.assertEqual(ledger.advance(50).source_at(0).state, State.GAP)

    def test_observation_not_considered_before_arrival(self):
        obs = [Observation("O", "D", 1, 9, 0, 5, True)]
        ledger = ReplayLedger([("B", 0, 5, 1)], {1: self.rule}, obs,
                              [CalibrationNotice("C", "D", 0, 0, 50, True)])
        self.assertEqual(ledger.advance(8).source_at(0).state, State.GAP)
        self.assertEqual(ledger.advance(9).source_at(0).state, State.OK)

    def test_obsolete_record_becomes_evidence_again(self):
        # 设备 A 的早期观测与设备 B 的新观测；A 先被判不可信，后来 B 也
        # 被判不可信，A 的旧观测重新成为依据。
        obs = [
            Observation("A_OLD", "A", 1, 2, 0, 5, True),
            Observation("B_NEW", "B", 5, 6, 0, 5, False),
        ]
        notices = [
            CalibrationNotice("A_OK", "A", 0, 0, 50, True),
            CalibrationNotice("B_OK", "B", 4, 0, 50, True),
            CalibrationNotice("B_BAD", "B", 20, 0, 10, False),
        ]
        ledger = ReplayLedger([("B", 0, 5, 1)], {1: self.rule}, obs, notices)
        self.assertEqual(ledger.advance(10).source_at(0).obs_id, "B_NEW")
        snap = ledger.advance(20)
        # B 的新观测被追溯判为不可信，A 的旧观测重新成为依据（状态 OK）
        self.assertEqual(snap.source_at(0).obs_id, "A_OLD")
        self.assertEqual(snap.source_at(0).state, State.OK)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.rule = Rule(1, max_gap=1, min_missing=2)
        self.obs = [
            Observation("O1", "D", 10, 10, 0, 20, True),
            Observation("O2", "D", 20, 25, 20, 40, False),
        ]
        self.notices = [CalibrationNotice("C", "D", 0, 0, 100, True)]

    def test_snapshot_is_dataset_and_qualifies(self):
        ledger = ReplayLedger([("B0", 0, 40, 1)], {1: self.rule},
                              self.obs, self.notices)
        snap = ledger.advance(10)
        self.assertIsInstance(snap, Snapshot)
        self.assertEqual(len(snap.source_window(0, 40)), 2)
        segs = snap.segments
        self.assertTrue(any(s.end == 20 for s in segs) or segs == ())

    def test_old_snapshot_preserved(self):
        ledger = ReplayLedger([("B0", 0, 40, 1)], {1: self.rule},
                              self.obs, self.notices)
        s1 = ledger.advance(10)
        before = span_tuples(s1.source_window(0, 40))
        s2 = ledger.advance(25)
        self.assertIsNot(s1, s2)
        self.assertEqual(span_tuples(s1.source_window(0, 40)), before)
        self.assertEqual(s1.cutoff, 10)
        self.assertEqual(s2.cutoff, 25)
        self.assertEqual(len(ledger.history), 2)

    def test_same_cutoff_same_snapshot(self):
        ledger = ReplayLedger([("B0", 0, 40, 1)], {1: self.rule},
                              self.obs, self.notices)
        s1 = ledger.advance(10)
        self.assertIs(ledger.advance(10), s1)

    def test_backwards_rejected_and_state_unchanged(self):
        ledger = ReplayLedger([("B0", 0, 40, 1)], {1: self.rule},
                              self.obs, self.notices)
        s1 = ledger.advance(20)
        with self.assertRaises(ReplayError):
            ledger.advance(19)
        self.assertIs(ledger.snapshot, s1)
        self.assertEqual(ledger.cutoff, 20)

    def test_skip_intermediate_equals_stepwise(self):
        a = ReplayLedger([("B0", 0, 40, 1)], {1: self.rule}, self.obs,
                         self.notices)
        b = ReplayLedger([("B0", 0, 40, 1)], {1: self.rule}, self.obs,
                         self.notices)
        sa = a.advance(25)
        for t in (10, 11, 25):
            sb = b.advance(t)
        self.assertEqual(span_tuples(sa.source_window(0, 40)),
                         span_tuples(sb.source_window(0, 40)))
        self.assertEqual(sa.segments, sb.segments)

    def test_segments_judged_on_whole_line(self):
        # 两个块各有一段 MISSING，拼起来达到门槛，段跨块
        rule = Rule(1, max_gap=0, min_missing=4)
        obs = [Observation("O", "D", 1, 1, 0, 40, False)]
        ledger = ReplayLedger([("B0", 0, 20, 1), ("B1", 20, 40, 1)],
                              {1: rule}, obs,
                              [CalibrationNotice("C", "D", 0, 0, 100, True)])
        snap = ledger.advance(1)
        self.assertEqual([(s.start, s.end, s.missing) for s in snap.segments],
                         [(0, 40, 40)])
        self.assertEqual(len(snap.qualified_segments()), 1)

    def test_source_window_clips_and_merges_only_identical(self):
        obs = [
            Observation("A", "D", 5, 5, 0, 10, True),
            Observation("B", "D", 5, 5, 10, 20, True),
            Observation("B2", "D", 6, 6, 20, 30, True),
        ]
        ledger = ReplayLedger([("B", 0, 30, 1)], {1: self.rule}, obs,
                              [CalibrationNotice("C", "D", 0, 0, 100, True)])
        snap = ledger.advance(10)
        spans = span_tuples(snap.source_window(0, 30))
        # 三段都是 OK，但胜出观测编号不同：一律不得合并。
        self.assertEqual(spans, [
            (0, 10, State.OK, "A", "C"),
            (10, 20, State.OK, "B", "C"),
            (20, 30, State.OK, "B2", "C"),
        ])
        self.assertEqual(span_tuples(snap.source_window(5, 15)),
                         [(5, 10, State.OK, "A", "C"),
                          (10, 15, State.OK, "B", "C")])

    def test_source_window_validation(self):
        ledger = ReplayLedger([("B", 0, 10, 1)], {1: self.rule}, [], [])
        snap = ledger.advance(0)
        for a, b in [(5, 5), (5, 4), (-1, 5), (5, 11)]:
            with self.assertRaises(ReplayError):
                snap.source_window(a, b)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.rule = Rule(1, 1, 1)
        self.obs = [Observation("O", "D", 1, 1, 0, 5, True)]
        self.notices = [CalibrationNotice("C", "D", 0, 0, 10, True)]

    def _bad(self, blocks=None, rules=None, observations=None, notices=None):
        with self.assertRaises(ReplayError):
            ReplayLedger(
                blocks if blocks is not None else [("B", 0, 5, 1)],
                rules if rules is not None else {1: self.rule},
                observations if observations is not None else self.obs,
                notices if notices is not None else self.notices,
            )

    def test_duplicate_observation_id(self):
        self._bad(observations=[Observation("O", "D", 1, 1, 0, 5, True),
                                Observation("O", "D", 2, 2, 0, 5, True)])

    def test_duplicate_notice_id(self):
        self._bad(notices=[CalibrationNotice("C", "D", 0, 0, 10, True),
                           CalibrationNotice("C", "D", 0, 0, 10, True)])

    def test_empty_ids(self):
        self._bad(observations=[Observation("", "D", 1, 1, 0, 5, True)])
        self._bad(observations=[Observation("O", "", 1, 1, 0, 5, True)])
        self._bad(notices=[CalibrationNotice("", "D", 0, 0, 10, True)])

    def test_arrival_before_acquired(self):
        self._bad(observations=[Observation("O", "D", 5, 4, 0, 5, True)])

    def test_empty_or_out_of_line_span(self):
        self._bad(observations=[Observation("O", "D", 1, 1, 5, 5, True)])
        self._bad(observations=[Observation("O", "D", 1, 1, -1, 5, True)])

    def test_notice_bad_period(self):
        self._bad(notices=[CalibrationNotice("C", "D", 0, 5, 5, True)])

    def test_non_integer_time(self):
        self._bad(observations=[Observation("O", "D", 1.0, 1, 0, 5, True)])

    def test_bool_not_integer(self):
        self._bad(observations=[Observation("O", "D", True, 1, 0, 5, True)])

    def test_blocks_must_cover_line_without_overlap(self):
        self._bad(blocks=[("A", 0, 3, 1), ("B", 3, 4, 1)])  # 缺块
        self._bad(blocks=[("A", 0, 4, 1), ("B", 3, 5, 1)])  # 重叠
        self._bad(blocks=[])

    def test_block_unknown_version(self):
        self._bad(blocks=[("B", 0, 5, 2)])

    def test_invalid_input_creates_no_partial_ledger(self):
        with self.assertRaises(ReplayError):
            ReplayLedger([("B", 0, 5, 1)], {1: self.rule},
                         [Observation("O", "D", 1, 1, 0, 5, True),
                          Observation("X", "D", 9, 1, 0, 5, True)],
                         self.notices)


class LocalityTests(unittest.TestCase):
    def test_advance_does_not_rescan_old_ledger(self):
        # 20 块，每块一条观测；首次推进读全部，之后只回查受影响者。
        rule = Rule(1, 0, 1)
        blocks = [(f"B{i}", i * 10, (i + 1) * 10, 1) for i in range(20)]
        obs = [Observation(f"O{i:02d}", "D", i, i, i * 10, (i + 1) * 10, True)
               for i in range(20)]
        notices = [CalibrationNotice("C", "D", 0, 0, 100, True)]
        ledger = ReplayLedger(blocks, {1: rule}, obs, notices)
        ledger.advance(0)
        # 推进到 19：新到达 19 条观测，每条只影响各自的 1 块
        ledger.access.reset()
        ledger.advance(19)
        self.assertEqual(ledger.access.blocks_rebuilt, 19)
        self.assertEqual(ledger.access.blocks_reused, 1)
        self.assertLessEqual(ledger.access.candidate_lookups, 19 * 2)

    def test_far_notice_does_not_touch_near_blocks(self):
        rule = Rule(1, 0, 1)
        blocks = [(f"B{i}", i * 10, (i + 1) * 10, 1) for i in range(10)]
        obs = [Observation(f"O{i:02d}", "D", i * 10 + 5, i * 10 + 5,
                           i * 10, (i + 1) * 10, True)
               for i in range(10)]
        notices = [
            CalibrationNotice("EARLY", "D", 0, 0, 100, True),
            CalibrationNotice("FAR", "D", 50, 25, 45, False),
        ]
        ledger = ReplayLedger(blocks, {1: rule}, obs, notices)
        ledger.advance(45)   # O00..O04 已到达
        ledger.access.reset()
        snap = ledger.advance(50)
        # FAR 只回查设备 D 中采集时刻 25..45 的观测 -> 仅 O02, O03
        self.assertEqual(ledger.access.notice_rechecks, 2)
        self.assertEqual(ledger.access.blocks_rebuilt, 2)
        self.assertEqual(ledger.access.blocks_reused, 8)
        self.assertEqual(snap.source_at(0).state, State.OK)
        self.assertEqual(snap.source_at(25).state, State.GAP)
        self.assertEqual(snap.source_at(45).state, State.OK)

    def test_notice_recheck_scoped_by_device_and_acquired(self):
        # 通告只回查同设备且采集时刻落在适用期内的观测：异设备与期外观测
        # 均不计入 notice_rechecks，也不触发其里程重选。
        rule = Rule(1, 0, 1)
        blocks = [(f"B{i}", i * 10, (i + 1) * 10, 1) for i in range(10)]
        obs = [
            Observation("A0", "A", 15, 15, 0, 10, True),
            Observation("B0", "B", 15, 15, 20, 30, True),
            Observation("A1", "A", 50, 50, 40, 50, True),
        ]
        notices = [
            CalibrationNotice("A-OK", "A", 0, 0, 100, True),
            CalibrationNotice("B-OK", "B", 0, 0, 100, True),
            CalibrationNotice("A-BAD", "A", 60, 10, 20, False),
        ]
        ledger = ReplayLedger(blocks, {1: rule}, obs, notices)
        ledger.advance(50)
        ledger.access.reset()
        snap = ledger.advance(60)
        self.assertEqual(ledger.access.notice_rechecks, 1)   # 只有 A0
        self.assertEqual(ledger.access.blocks_rebuilt, 1)
        self.assertEqual(ledger.access.blocks_reused, 9)
        self.assertEqual(snap.source_at(0).state, State.GAP)
        self.assertEqual(snap.source_at(20).state, State.OK)  # 异设备不受影响
        self.assertEqual(snap.source_at(45).state, State.OK)

    def test_source_window_reads_only_touched_blocks(self):
        rule = Rule(1, 0, 1)
        blocks = [(f"B{i}", i * 10, (i + 1) * 10, 1) for i in range(30)]
        obs = [Observation(f"O{i:02d}", "D", 1, 1, i * 10, (i + 1) * 10, True)
               for i in range(30)]
        ledger = ReplayLedger(
            blocks, {1: rule}, obs,
            [CalibrationNotice("C", "D", 0, 0, 300, True)])
        snap = ledger.advance(1)
        ledger.access.reset()
        spans = snap.source_window(100, 125)
        self.assertEqual(ledger.access.source_blocks_read, 3)
        self.assertEqual(len(spans), 3)
        self.assertEqual(ledger.access.expanded_cells, 0)

    def test_partial_block_source_retained(self):
        # 一块 100 格；通告只影响设备 D 的采集时刻区间，重选只改局部里程，
        # 其余来源段原样保留。
        rule = Rule(1, 0, 1)
        obs = [
            Observation("NEAR", "D", 5, 5, 0, 100, True),
            Observation("FAR", "D", 50, 50, 70, 80, True),
        ]
        notices = [
            CalibrationNotice("OK", "D", 0, 0, 100, True),
            CalibrationNotice("BAD", "D", 60, 45, 55, False),
        ]
        ledger = ReplayLedger([("B", 0, 100, 1)], {1: rule}, obs, notices)
        ledger.advance(50)
        ledger.access.reset()
        snap = ledger.advance(60)
        self.assertEqual(ledger.access.blocks_rebuilt, 1)
        spans = span_tuples(snap.source_window(0, 100))
        # NEAR 仍是 0..100 的胜者，FAR 的 70..80 来源未变
        self.assertEqual(spans, [
            (0, 100, State.OK, "NEAR", "OK"),
        ])
        far_span = [s for s in spans if s[3] == "NEAR"]
        self.assertEqual(far_span[0][:2], (0, 100))

    def test_partial_override_keeps_untouched_runs(self):
        rule = Rule(1, 0, 1)
        obs = [
            Observation("BASE", "D", 1, 1, 0, 100, True),
            Observation("PATCH", "D", 9, 9, 40, 60, False),
        ]
        notices = [CalibrationNotice("OK", "D", 0, 0, 100, True)]
        ledger = ReplayLedger([("B", 0, 100, 1)], {1: rule}, obs, notices)
        ledger.advance(1)
        ledger.access.reset()
        snap = ledger.advance(9)
        # 只有 PATCH 的 40..60 被重选，BASE 的两侧来源段原样保留
        self.assertEqual(ledger.access.candidate_lookups, 2)
        self.assertEqual(span_tuples(snap.source_window(0, 100)), [
            (0, 40, State.OK, "BASE", "OK"),
            (40, 60, State.MISSING, "PATCH", "OK"),
            (60, 100, State.OK, "BASE", "OK"),
        ])

    def test_snapshot_usable_as_dataset(self):
        import tempfile
        from railway.orders import WorkOrderStore
        rule = Rule(1, 0, 1)
        ledger = ReplayLedger(
            [("B", 0, 20, 1)], {1: rule},
            [Observation("O", "D", 1, 1, 0, 20, False)],
            [CalibrationNotice("C", "D", 0, 0, 50, True)])
        snap = ledger.advance(1)
        self.assertEqual(len(snap.qualified_segments()), 1)
        with tempfile.TemporaryDirectory() as d:
            store = WorkOrderStore.open(d, create=True)
            order = store.issue("W1", snap, 0, 20)
            self.assertEqual(order.due_total, 20)

    def test_no_per_cell_expansion_for_huge_spans(self):
        BIG = 10**8
        rule = Rule(1, 0, 1)
        ledger = ReplayLedger([("B", 0, BIG, 1)], {1: rule},
                              [Observation("O", "D", 1, 1, 0, BIG, True)],
                              [CalibrationNotice("C", "D", 0, 0, 10, True)])
        snap = ledger.advance(1)
        spans = snap.source_window(BIG // 2, BIG // 2 + 10)
        self.assertEqual(len(spans), 1)
        self.assertEqual(ledger.access.expanded_cells, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
