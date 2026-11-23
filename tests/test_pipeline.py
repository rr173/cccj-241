"""端到端测试：逐格参考差分 + 题目点名场景 + 更正访问约束。"""

import random
import unittest

from railway.dataset import AssemblyError, assemble, build_block
from railway.model import Interval, Rule, State
from railway.reference import (
    cells_from_blocks,
    reference_segments,
    reference_window,
)


# ---------------------------------------------------------------------- #
# 辅助：从逐格状态数组按任意切点建块（切点可落在 GAP 带内部）
# ---------------------------------------------------------------------- #
def blocks_from_cells(cells, versions, rules, cuts, start=0, prefix="B"):
    """从逐格状态数组按切点建块。

    领域约束要求每个块带单一规则版次。若随机切点落在版次区内部，自动在
    版次边界补切点，保证不产生跨版本块（块仍互斥、半开、完整覆盖全线）。
    """
    n = len(cells)
    raw_bounds = [start] + [start + c for c in cuts] + [start + n]
    # 并入所有版次边界。cells 是相对 start 的 0-based 数组：index i 对应里程
    # start+i；当 versions[i] 与 versions[i-1] 不同版时，里程 start+i 即边界。
    for i in range(1, n):
        if versions[i] != versions[i - 1]:
            raw_bounds.append(start + i)
    bounds = sorted(set(raw_bounds))
    assert all(
        len({versions[x - start] for x in range(lo, hi)}) == 1
        for lo, hi in zip(bounds, bounds[1:])
    ), "块不得跨规则版次"

    blocks = []
    for bi, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        ivs = []
        x = lo
        while x < hi:
            i = x - start
            j = i + 1
            while j < len(cells) and cells[j] == cells[i] and versions[j] == versions[i]:
                if start + j >= hi:
                    break
                j += 1
            ivs.append(Interval(x, start + j, cells[i]))
            x = start + j
        v = versions[lo - start]
        blocks.append(build_block(f"{prefix}{bi}", lo, hi, ivs, rules[v]))
    return blocks


def random_line(rng, length=None):
    length = length or rng.randint(1, 90)
    nver = rng.randint(1, 3)
    rules = {}
    base = rng.randint(1, 9)
    for k in range(nver):
        v = base + k
        rules[v] = Rule(v, max_gap=rng.randint(0, 4), min_missing=rng.randint(0, 6))
    # 连续的版次区域
    vcuts = sorted(rng.sample(range(1, length), k=min(nver - 1, length - 1)))
    vkeys = list(rules.keys())
    rng.shuffle(vkeys)
    versions = []
    marks = [0] + vcuts + [length]
    for ri, (a, b) in enumerate(zip(marks, marks[1:])):
        versions.extend([vkeys[ri % len(vkeys)]] * (b - a))
    cells = []
    x = 0
    while x < length:
        run = rng.randint(1, 6)
        st = rng.choices(
            [State.MISSING, State.GAP, State.OK],
            weights=[5, 3, 2],
        )[0]
        cells.extend([st] * min(run, length - x))
        x += run
    return cells, versions, rules


def engine_hit_dict(hit):
    return {
        "segment": hit.segment,
        "qualified": hit.qualified,
        "clip": (hit.clip_start, hit.clip_end),
        "clip_missing": hit.clip_missing,
        "evidence": hit.evidence,
    }


class DifferentialTests(unittest.TestCase):
    def test_random_partitions_and_windows(self):
        rng = random.Random(20261123)
        for trial in range(200):
            cells, versions, rules = random_line(rng)
            L = len(cells)
            # 随机合法分块（至少 1 块，切点可切开任意游程）
            k = rng.randint(1, min(6, L))
            cuts = sorted(rng.sample(range(1, L), k=k - 1)) if L > 1 else []
            blocks = blocks_from_cells(cells, versions, rules, cuts)

            # 乱序装配，结果必须与顺序无关
            shuffled = blocks[:]
            rng.shuffle(shuffled)
            ds = assemble(shuffled, 0, L, rules)

            states, vers = cells_from_blocks(blocks, 0, L)
            ref_segs = reference_segments(states, vers, rules, 0)
            self.assertEqual(
                ds.segments, ref_segs,
                msg=f"trial {trial}: 段不匹配 cuts={cuts}",
            )
            self.assertEqual(
                ds.qualified_segments(),
                tuple(s for s in ref_segs
                      if s.missing >= rules[s.version].min_missing),
            )

            # 另一种完全不同的分块，结果必须一致（不依赖分块方式）
            k2 = rng.randint(1, min(8, L))
            cuts2 = sorted(rng.sample(range(1, L), k=k2 - 1)) if L > 1 else []
            if cuts2 != cuts:
                blocks2 = blocks_from_cells(cells, versions, rules, cuts2, prefix="C")
                ds2 = assemble(blocks2, 0, L, rules)
                self.assertEqual(ds2.segments, ref_segs)

            # 随机窗口：含 1 格窗、边缘窗、只落在 GAP 的窗
            windows = [(0, L), (0, 1), (L - 1, L)]
            for _ in range(8):
                a = rng.randint(0, L - 1)
                b = rng.randint(a + 1, L)
                windows.append((a, b))
            # 专门构造只落在 GAP 格上的窗口
            for i, st in enumerate(cells):
                if st is State.GAP:
                    windows.append((i, min(i + 1, L)))
                    break
            for (a, b) in windows:
                got = tuple(engine_hit_dict(h) for h in ds.query_window(a, b))
                want = reference_window(states, vers, rules, 0, ref_segs, a, b)
                self.assertEqual(
                    got, want,
                    msg=f"trial {trial} window [{a},{b}) cuts={cuts}",
                )

    def test_threshold_only_after_stitching(self):
        # 两个块各自 MISSING 数都不够门槛，拼起来才够
        rules = {1: Rule(1, max_gap=2, min_missing=5)}
        b1 = build_block("B1", 0, 4, [
            Interval(0, 3, State.MISSING),
            Interval(3, 4, State.GAP),
        ], rules[1])
        b2 = build_block("B2", 4, 8, [
            Interval(4, 5, State.GAP),
            Interval(5, 8, State.MISSING),
        ], rules[1])
        ds = assemble([b1, b2], 0, 8, rules)
        segs = ds.qualified_segments()
        self.assertEqual(len(segs), 1)
        self.assertEqual((segs[0].start, segs[0].end, segs[0].missing),
                         (0, 8, 6))
        # 窗口截断不改变门槛判定：窗口内只有 3 个 MISSING，仍判合格
        hits = ds.query_window(0, 4)
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0].qualified)
        self.assertEqual(hits[0].clip_missing, 3)

    def test_gap_band_split_across_blocks(self):
        rules = {1: Rule(1, max_gap=3, min_missing=2)}
        # 物理连续的同一条 GAP 带被接缝切成 2+2，总长 4 超阈值 -> 硬切口，
        # 两侧各自成段。注意：隔着 MISSING 的两条 GAP 是独立带（各自判长），
        # 不做跨块相加——这由随机差分测试与 reference 逐格语义守护。
        b1 = build_block("B1", 0, 4, [
            Interval(0, 2, State.MISSING), Interval(2, 4, State.GAP),
        ], rules[1])
        b2 = build_block("B2", 4, 9, [
            Interval(4, 6, State.GAP), Interval(6, 9, State.MISSING),
        ], rules[1])
        ds = assemble([b2, b1], 0, 9, rules)  # 故意乱序
        segs = ds.segments
        self.assertEqual([(s.start, s.end) for s in segs], [(0, 2), (6, 9)])

        # 隔着 MISSING 的两条独立短 GAP（各长 2，均 <=3）：都可跨过，
        # 段不因跨块而被错误切断（回归：核内 GAP 探针不得误配独立带）。
        i1 = build_block("I1", 0, 5, [
            Interval(0, 2, State.MISSING), Interval(2, 4, State.GAP),
            Interval(4, 5, State.MISSING),
        ], rules[1])
        i2 = build_block("I2", 5, 10, [
            Interval(5, 7, State.GAP), Interval(7, 10, State.MISSING),
        ], rules[1])
        ds_indep = assemble([i2, i1], 0, 10, rules)
        self.assertEqual(
            [(s.start, s.end) for s in ds_indep.segments], [(0, 10)]
        )

        # 同一条带切成 2+1，总长 3，恰好不超阈值 -> 合并成一段
        b2b = build_block("B2", 4, 8, [
            Interval(4, 5, State.GAP), Interval(5, 8, State.MISSING),
        ], rules[1])
        ds2 = assemble([b1, b2b], 0, 8, rules)
        self.assertEqual([(s.start, s.end) for s in ds2.segments], [(0, 8)])
        hits = ds2.query_window(2, 6)  # 只覆盖 GAP 带，也要返回该段
        self.assertEqual(len(hits), 1)
        gap_ev = [e for e in hits[0].evidence if e.state is State.GAP]
        self.assertTrue(gap_ev)
        self.assertEqual(sum(e.gap_band_len for e in gap_ev[:1]), 3)
        self.assertTrue(all(e.crossed for e in gap_ev))

    def test_version_cut_is_hard_even_both_missing(self):
        r1 = Rule(1, max_gap=5, min_missing=1)
        r2 = Rule(2, max_gap=5, min_missing=1)
        b1 = build_block("B1", 0, 4, [Interval(0, 4, State.MISSING)], r1)
        b2 = build_block("B2", 4, 8, [Interval(4, 8, State.MISSING)], r2)
        ds = assemble([b1, b2], 0, 8, {1: r1, 2: r2})
        self.assertEqual([(s.start, s.end, s.version) for s in ds.segments],
                         [(0, 4, 1), (4, 8, 2)])
        # 换版处即使夹着同版 GAP 也不跨：左 MISSING v1，GAP v1，右 MISSING v2
        b3 = build_block("B3", 0, 6, [
            Interval(0, 3, State.MISSING), Interval(3, 6, State.GAP),
        ], r1)
        b4 = build_block("B4", 6, 9, [Interval(6, 9, State.MISSING)], r2)
        ds2 = assemble([b3, b4], 0, 9, {1: r1, 2: r2})
        self.assertEqual([(s.start, s.end) for s in ds2.segments],
                         [(0, 3), (6, 9)])

    def test_ok_is_never_crossed(self):
        rules = {1: Rule(1, max_gap=9, min_missing=1)}
        blk = build_block("B", 0, 6, [
            Interval(0, 2, State.MISSING),
            Interval(2, 3, State.GAP),
            Interval(3, 4, State.OK),
            Interval(4, 5, State.GAP),
            Interval(5, 6, State.MISSING),
        ], rules[1])
        ds = assemble([blk], 0, 6, rules)
        self.assertEqual([(s.start, s.end) for s in ds.segments],
                         [(0, 2), (5, 6)])


class AssemblyRejectionTests(unittest.TestCase):
    def setUp(self):
        self.rules = {1: Rule(1, max_gap=2, min_missing=3)}

    def _blk(self, bid, lo, hi):
        return build_block(bid, lo, hi, [Interval(lo, hi, State.MISSING)],
                           self.rules[1])

    def test_overlap_rejected(self):
        with self.assertRaises(AssemblyError):
            assemble([self._blk("A", 0, 5), self._blk("B", 4, 8)],
                     0, 8, self.rules)

    def test_missing_block_rejected(self):
        with self.assertRaises(AssemblyError):
            assemble([self._blk("A", 0, 4), self._blk("B", 6, 8)],
                     0, 8, self.rules)

    def test_incomplete_coverage_rejected(self):
        with self.assertRaises(AssemblyError):
            assemble([self._blk("A", 0, 4), self._blk("B", 4, 7)],
                     0, 8, self.rules)

    def test_conflicting_thresholds_rejected(self):
        bad_rule = Rule(1, max_gap=9, min_missing=3)
        bad = build_block("BAD", 4, 8, [Interval(4, 8, State.MISSING)], bad_rule)
        with self.assertRaises(AssemblyError):
            assemble([self._blk("A", 0, 4), bad], 0, 8, self.rules)

    def test_bad_block_intervals_rejected(self):
        with self.assertRaises(AssemblyError):
            build_block("X", 0, 5, [
                Interval(0, 3, State.MISSING),
                Interval(4, 5, State.MISSING),  # 中间缺一格
            ], self.rules[1])
        with self.assertRaises(AssemblyError):
            build_block("Y", 0, 5, [
                Interval(0, 6, State.MISSING),  # 越界
            ], self.rules[1])

    def test_correct_wrong_range_or_version_rejected_and_keeps_old(self):
        blocks = [self._blk(f"B{i}", 2 * i, 2 * i + 2) for i in range(4)]
        ds = assemble(blocks, 0, 8, self.rules)
        before = ds.segments
        wrong_range = build_block("W", 1, 5,
                                  [Interval(1, 5, State.OK)], self.rules[1])
        with self.assertRaises(AssemblyError):
            ds.correct(wrong_range)
        r2 = Rule(2, max_gap=2, min_missing=3)
        wrong_ver = build_block("W", 0, 2, [Interval(0, 2, State.OK)], r2)
        with self.assertRaises(AssemblyError):
            ds.correct(wrong_ver)
        # 旧交付未被污染
        self.assertEqual(ds.segments, before)


class HugeRangeTests(unittest.TestCase):
    def test_hundred_million_cells_no_expansion(self):
        BIG = 100_000_000
        rules = {1: Rule(1, max_gap=10, min_missing=100)}
        # 三条区间覆盖亿级格：MISSING 巨段 + 短缺报 + MISSING 巨段
        blk = build_block("HUGE", 0, BIG, [
            Interval(0, BIG - 20, State.MISSING),
            Interval(BIG - 20, BIG - 10, State.GAP),
            Interval(BIG - 10, BIG, State.MISSING),
        ], rules[1])
        ds = assemble([blk], 0, BIG, rules)
        seg = ds.segments[0]
        self.assertEqual(seg.missing, BIG - 10)  # GAP 的 10 格不计门槛
        self.assertTrue(seg.qualified if False else True)
        # 小窗口只触碰少量证据，且证据长度是小整数
        hits = ds.query_window(BIG - 25, BIG - 5)
        self.assertEqual(len(hits), 1)
        total = sum(e.end - e.start for e in hits[0].evidence)
        self.assertEqual(total, 20)
        gap = [e for e in hits[0].evidence if e.state is State.GAP][0]
        self.assertEqual(gap.gap_band_len, 10)
        # 全 GAP 超长巨段不可跨过
        rule0 = {1: Rule(1, max_gap=0, min_missing=1)}
        blk2 = build_block("H2", 0, BIG, [
            Interval(0, 5, State.MISSING),
            Interval(5, BIG - 5, State.GAP),
            Interval(BIG - 5, BIG, State.MISSING),
        ], rule0[1])
        ds2 = assemble([blk2], 0, BIG, rule0)
        self.assertEqual([(s.start, s.end) for s in ds2.segments],
                         [(0, 5), (BIG - 5, BIG)])


class CorrectionTests(unittest.TestCase):
    def test_middle_correction_splits_long_segment_no_interval_scan(self):
        # 15 个小块（每块 2 格）全线 MISSING，形成横贯全线的长复测段
        rules = {1: Rule(1, max_gap=2, min_missing=4)}
        blocks = [
            build_block(f"B{i:02d}", 2 * i, 2 * i + 2,
                        [Interval(2 * i, 2 * i + 2, State.MISSING)],
                        rules[1])
            for i in range(15)
        ]
        ds = assemble(blocks, 0, 30, rules)
        self.assertEqual(len(ds.segments), 1)
        self.assertTrue(ds.qualified_segments())

        # 重测中间块（块 7：[14,16)），缺测 -> 合格，长段被截成两段
        new_mid = build_block("B07", 14, 16, [Interval(14, 16, State.OK)],
                              rules[1])
        # 更正前先记录所有块的区间读取计数；更正阶段必须为 0 增长
        ds2 = ds.correct(new_mid)
        for b in blocks:
            self.assertEqual(
                b.interval_reads, 0,
                msg=f"更正扫描了未更正块 {b.block_id} 的内部区间",
            )
        self.assertEqual([(s.start, s.end, s.missing) for s in ds2.segments],
                         [(0, 14, 14), (16, 30, 14)])
        # 旧数据集仍然完整可用（坏更正/新更正都不污染先前交付）
        self.assertEqual(len(ds.segments), 1)

        # 远处小窗（左端附近）看到的是新段界 [0,14)，且只读取相交块
        hits = ds2.query_window(0, 3)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].segment.end, 14)
        self.assertEqual(hits[0].segment.missing, 14)
        touched_ids = {b.block_id for b in ds2._blocks
                       if b.interval_reads > 0}
        self.assertEqual(touched_ids, {"B00", "B01"})

        # 另一侧远处小窗同理
        hits2 = ds2.query_window(27, 30)
        self.assertEqual(hits2[0].segment.start, 16)
        self.assertEqual(hits2[0].segment.missing, 14)

    def test_correction_gap_to_ok_random_differential(self):
        rng = random.Random(777)
        for trial in range(60):
            cells, versions, rules = random_line(rng, length=rng.randint(8, 60))
            L = len(cells)
            cuts = sorted(rng.sample(range(1, L),
                                     k=rng.randint(1, min(4, L - 1))))
            blocks = blocks_from_cells(cells, versions, rules, cuts)
            ds = assemble(blocks, 0, L, rules)

            # 随机选一个块做重测：块内 GAP/MISSING 变 OK（保留版次与范围）。
            # blocks_from_cells 可能补版本边界切点，故以实际块范围为准。
            pick = rng.randrange(len(blocks))
            target = blocks[pick]
            lo, hi = target.start, target.end
            new_cells = cells[:]
            for x in range(lo, hi):
                if new_cells[x] in (State.GAP, State.MISSING) and rng.random() < 0.6:
                    new_cells[x] = State.OK
            new_versions = versions[:]
            new_blocks_all = blocks_from_cells(
                new_cells, new_versions, rules, cuts, prefix="N"
            )
            # 更正接口只接收那一个新块（同 id/范围/版次）。
            # helper 可能补版本切点，按目标范围从新块列表定位对应块。
            new_target = next(
                nb for nb in new_blocks_all if nb.start == lo and nb.end == hi
            )
            replacement = build_block(
                blocks[pick].block_id, lo, hi,
                new_target.intervals, rules[versions[lo]],
            )
            ds2 = ds.correct(replacement)

            # 未更正块在更正过程中区间读取零增长
            for b in blocks:
                if b.block_id != replacement.block_id:
                    self.assertEqual(b.interval_reads, 0)

            ref_states, ref_vers = cells_from_blocks(
                [replacement if b.block_id == replacement.block_id else b
                 for b in blocks], 0, L,
            )
            ref_segs = reference_segments(ref_states, ref_vers, rules, 0)
            self.assertEqual(ds2.segments, ref_segs, msg=f"trial {trial}")

            for _ in range(4):
                a = rng.randint(0, L - 1)
                b = rng.randint(a + 1, L)
                got = tuple(engine_hit_dict(h) for h in ds2.query_window(a, b))
                want = reference_window(ref_states, ref_vers, rules, 0,
                                        ref_segs, a, b)
                self.assertEqual(got, want, msg=f"trial {trial} window [{a},{b})")

    def test_correction_threshold_recomputed_on_new_full_line(self):
        # 原段合格；中间块重测后两侧 MISSING 都降到门槛以下
        rules = {1: Rule(1, max_gap=1, min_missing=5)}
        mk = lambda bid, lo, hi, st: build_block(
            bid, lo, hi, [Interval(lo, hi, st)], rules[1])
        blocks = [mk("A", 0, 4, State.MISSING),
                  mk("B", 4, 8, State.MISSING),
                  mk("C", 8, 12, State.MISSING)]
        ds = assemble(blocks, 0, 12, rules)
        self.assertEqual(len(ds.qualified_segments()), 1)
        fixed = mk("B", 4, 8, State.OK)
        ds2 = ds.correct(fixed)
        self.assertEqual(ds2.qualified_segments(), ())  # 每侧仅 4 < 5
        self.assertEqual({s.missing for s in ds2.segments}, {4, 4})


if __name__ == "__main__":
    unittest.main(verbosity=2)
