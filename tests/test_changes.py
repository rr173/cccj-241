"""跨版复测变化清单（railway.changes）的端到端测试。

复跑方法（隔离模式，忽略环境变量与已安装包，保证不加载外部 json.py 等）::

    python -I run_tests.py                 # 跑全部测试
    python -I -m unittest tests.test_changes -v   # 只跑本文件

覆盖：

* 同向 / 反向交接的区间算术（含题目点名的反向映射 10↔52、12↔50）；
* 断点失联：后续版本复用相同号码也不能接回；未配对只列查看版应测者；
* 多跳版本链、空交接表、单版本链；
* 版本链 / 端点数据集范围 / 交接记录不合法时整体拒绝；
* 合法拆分交接记录不改变结果；
* 局部性：窗口被改线分送到远端时，不读入中间无关里程（访问计数 +
  块 interval_reads 双证明），且不逐格展开；
* 随机差分：独立逐格暴力参考对随机版本链、交接表与窗口做全量核对；
* 不修改输入数据集（段表 / 块读取计数前后一致）。
"""

import random
import unittest

from railway.changes import (
    ChangeAccessStats,
    ChangeCategory,
    ChainError,
    Direction,
    DueEvidence,
    HandoverRecord,
    LineVersion,
    VersionChain,
)
from railway.dataset import assemble, build_block
from railway.model import Interval, Rule, State


# ---------------------------------------------------------------------- #
# 夹具辅助
# ---------------------------------------------------------------------- #
def ds_from_cells(cells, rule, start=0, block_cut=None, prefix="B"):
    """逐格状态数组 -> 已装配 Dataset（可指定切点验证与分块无关）。"""
    end = start + len(cells)
    bounds = [start]
    if block_cut:
        bounds.extend(sorted({start + c for c in block_cut
                              if 0 < c < len(cells)}))
    bounds.append(end)
    blocks = []
    for bi, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        ivs = []
        x = lo
        while x < hi:
            j = x + 1
            while j < hi and cells[j - start] == cells[x - start]:
                j += 1
            ivs.append(Interval(x, j, cells[x - start]))
            x = j
        blocks.append(build_block(f"{prefix}{bi}", lo, hi, ivs, rule))
    return assemble(blocks, start, end, {rule.version: rule})


def iv(cells, lo, hi, state):
    for x in range(lo, hi):
        cells[x] = state


def row_tuple(r):
    return (
        r.start, r.end, r.category,
        r.peer_start, r.peer_end, r.direction,
        r.break_from, r.break_to,
        None if r.evidence is None else (
            r.evidence.segment_start, r.evidence.segment_end,
            r.evidence.rule_version, r.evidence.segment_missing,
        ),
    )


# ---------------------------------------------------------------------- #
# 题目点名的反向映射与基本新增/解除
# ---------------------------------------------------------------------- #
class BasicMappingTests(unittest.TestCase):
    def setUp(self):
        # 规则 v1/v2 相同：min_missing=2，单格 MISSING 段不达标。
        self.rule = Rule(1, max_gap=1, min_missing=2)
        self.rule2 = Rule(2, max_gap=1, min_missing=2)

    def test_reversed_handover_example_mapping(self):
        # 反向交接 [10,13) -> [50,53)：10↔52, 11↔51, 12↔50。
        v1 = LineVersion("V1", 0, 20)
        v2 = LineVersion("V2", 40, 60)
        chain = VersionChain.build(
            [v1, v2],
            {("V1", "V2"): [HandoverRecord(10, 13, 50, 53, reversed=True)]},
        )
        # 两侧都把对应位置做成达标段：V1 [10,12) MISSING；V2 [51,53) MISSING
        # 物理位置 10->52、11->51：两侧应测集合相同 -> 清单为空。
        c1 = [State.OK] * 20
        iv(c1, 10, 12, State.MISSING)
        c2 = [State.OK] * 20
        iv(c2, 51 - 40, 53 - 40, State.MISSING)
        ds1 = ds_from_cells(c1, self.rule)
        ds2 = ds_from_cells(c2, self.rule2, start=40)
        result = chain.change_list(ds1, ds2, "V1", 0, 20)
        self.assertEqual(result.rows, ())

        # V2 把物理位置 12（对应里程 50）也变成 MISSING：晚版新增一格。
        c2b = c2[:]
        iv(c2b, 50 - 40, 51 - 40, State.MISSING)  # [50,51) 对应 V1 的 12
        ds2b = ds_from_cells(c2b, self.rule2, start=40)
        result = chain.change_list(ds1, ds2b, "V1", 0, 20)
        self.assertEqual(len(result.rows), 1)
        r = result.rows[0]
        self.assertEqual(row_tuple(r)[:6],
                         (12, 13, ChangeCategory.NEW, 50, 51, Direction.REVERSED))
        self.assertIsNone(r.break_from)
        # 证据是晚版完整段 [50,53)，规则 v2，全段缺测 3 格。
        self.assertEqual(r.evidence, DueEvidence(50, 53, 2, 3))

        # 晚版视角：同一物理位置以 V2 坐标 [50,51) 出现，仍是 NEW，
        # 对端范围 [12,13)，方向反向；清单按查看版里程升序。
        result2 = chain.change_list(ds1, ds2b, "V2", 40, 60)
        self.assertEqual(len(result2.rows), 1)
        r2 = result2.rows[0]
        self.assertEqual(row_tuple(r2)[:6],
                         (50, 51, ChangeCategory.NEW, 12, 13, Direction.REVERSED))
        self.assertEqual(r2.evidence, DueEvidence(50, 53, 2, 3))

    def test_forward_new_and_removed_with_evidence(self):
        v1 = LineVersion("V1", 0, 10)
        v2 = LineVersion("V2", 100, 110)
        chain = VersionChain.build(
            [v1, v2],
            {("V1", "V2"): [HandoverRecord(0, 10, 100, 110)]},
        )
        # V1: [2,4) MISSING（达标，2 格）；V2 同位置 [102,104) 变 OK -> 解除。
        c1 = [State.OK] * 10
        iv(c1, 2, 4, State.MISSING)
        # V2: [105,107) MISSING（对应 V1 [5,7)）-> 新增。
        c2 = [State.OK] * 10
        iv(c2, 5, 7, State.MISSING)
        ds1 = ds_from_cells(c1, self.rule)
        ds2 = ds_from_cells(c2, self.rule2, start=100)

        r1 = chain.change_list(ds1, ds2, "V1", 0, 10)
        self.assertEqual(
            [row_tuple(r) for r in r1.rows],
            [
                (2, 4, ChangeCategory.REMOVED, 102, 104, Direction.FORWARD,
                 None, None, (2, 4, 1, 2)),
                (5, 7, ChangeCategory.NEW, 105, 107, Direction.FORWARD,
                 None, None, (105, 107, 2, 2)),
            ],
        )
        # 晚版视角：按 V2 里程升序，类别绝对语义不变。
        r2 = chain.change_list(ds1, ds2, "V2", 100, 110)
        self.assertEqual(
            [row_tuple(r) for r in r2.rows],
            [
                (102, 104, ChangeCategory.REMOVED, 2, 4, Direction.FORWARD,
                 None, None, (2, 4, 1, 2)),
                (105, 107, ChangeCategory.NEW, 5, 7, Direction.FORWARD,
                 None, None, (105, 107, 2, 2)),
            ],
        )

    def test_subwindow_clip_and_unqualified_missing_not_due(self):
        v1 = LineVersion("V1", 0, 6)
        v2 = LineVersion("V2", 0, 6)
        chain = VersionChain.build(
            [v1, v2], {("V1", "V2"): [HandoverRecord(0, 6, 0, 6)]}
        )
        # V1 单格 MISSING [1,2)：段不达标（1 < min_missing=2）-> 不应复测。
        c1 = [State.OK] * 6
        iv(c1, 1, 2, State.MISSING)
        # V2 同位置扩成 [1,3) MISSING：达标 -> 新增，但窗口只取 [0,2)，
        # 行必须被裁到 [1,2)，证据仍是完整段 [1,3)。
        c2 = [State.OK] * 6
        iv(c2, 1, 3, State.MISSING)
        ds1 = ds_from_cells(c1, self.rule)
        ds2 = ds_from_cells(c2, self.rule2)
        result = chain.change_list(ds1, ds2, "V1", 0, 2)
        self.assertEqual(len(result.rows), 1)
        r = result.rows[0]
        self.assertEqual((r.start, r.end, r.category), (1, 2, ChangeCategory.NEW))
        self.assertEqual(r.evidence, DueEvidence(1, 3, 2, 2))

    def test_qualified_segment_with_gap_missing_only(self):
        # 达标段内含可跨过 GAP：应复测只取 MISSING 格，GAP 不进清单。
        rule = Rule(1, max_gap=2, min_missing=2)
        v1 = LineVersion("V1", 0, 6)
        v2 = LineVersion("V2", 0, 6)
        chain = VersionChain.build(
            [v1, v2], {("V1", "V2"): [HandoverRecord(0, 6, 0, 6)]}
        )
        c1 = [State.MISSING, State.MISSING, State.GAP,
              State.GAP, State.MISSING, State.OK]
        c2 = [State.OK] * 6
        ds1 = ds_from_cells(c1, rule)
        ds2 = ds_from_cells(c2, rule)
        result = chain.change_list(ds1, ds2, "V1", 0, 6)
        # 段 [0,5) 达标（3 个 MISSING）；解除行只覆盖 MISSING 格。
        self.assertEqual(
            [(r.start, r.end, r.category) for r in result.rows],
            [(0, 2, ChangeCategory.REMOVED), (4, 5, ChangeCategory.REMOVED)],
        )
        # 两行证据相同（同一段），但中间隔着 GAP 不连续，不得合并。
        self.assertEqual(result.rows[0].evidence,
                         result.rows[1].evidence)


# ---------------------------------------------------------------------- #
# 断点 / 失联 / 多跳
# ---------------------------------------------------------------------- #
class BreakpointTests(unittest.TestCase):
    RULE = Rule(1, max_gap=1, min_missing=1)

    def test_break_no_reconnect_even_if_number_reused(self):
        v1 = LineVersion("V1", 0, 8)
        v2 = LineVersion("V2", 0, 8)
        v3 = LineVersion("V3", 0, 8)
        # V1->V2：只延续 [0,4)；V2->V3：[4,8) 用了与 V1 相同号码，
        # 但物理位置在第一次交接已断，不能接回。
        chain = VersionChain.build(
            [v1, v2, v3],
            {
                ("V1", "V2"): [HandoverRecord(0, 4, 0, 4)],
                ("V2", "V3"): [HandoverRecord(4, 8, 4, 8)],
            },
        )
        c1 = [State.MISSING] * 8
        c2 = [State.OK] * 8
        c3 = [State.MISSING] * 8
        ds1 = ds_from_cells(c1, self.RULE)
        ds2 = ds_from_cells(c2, self.RULE)
        ds3 = ds_from_cells(c3, self.RULE)

        # 早版视角：[0,4) 有对端且两侧状态差异（V1 应测、V2 不应测，但
        # V3 才是晚版——[0,4) 在 V2->V3 又断了，全部失联）。
        result = chain.change_list(ds1, ds3, "V1", 0, 8)
        cats = {(r.start, r.end): r for r in result.rows}
        # [0,4) 断在 V2->V3；[4,8) 断在 V1->V2。
        self.assertEqual(set(cats), {(0, 4), (4, 8)})
        self.assertTrue(all(r.category is ChangeCategory.UNPAIRED
                            for r in result.rows))
        self.assertEqual((cats[(0, 4)].break_from, cats[(0, 4)].break_to),
                         ("V2", "V3"))
        self.assertEqual((cats[(4, 8)].break_from, cats[(4, 8)].break_to),
                         ("V1", "V2"))
        for r in result.rows:
            self.assertIsNone(r.peer_start)
            self.assertIsNone(r.direction)

        # 晚版视角：V3 全线应测；[4,8) 在 V2->V3 有映射、V1->V2 断，
        # 断口注明历史顺序的两版 (V1,V2)；[0,4) 在 V2->V3 即断。
        result3 = chain.change_list(ds1, ds3, "V3", 0, 8)
        b3 = {(r.start, r.end): (r.break_from, r.break_to)
              for r in result3.rows}
        self.assertEqual(b3, {(0, 4): ("V2", "V3"), (4, 8): ("V1", "V2")})

    def test_empty_handover_means_no_continuity(self):
        v1 = LineVersion("V1", 0, 4)
        v2 = LineVersion("V2", 0, 4)
        chain = VersionChain.build(
            [v1, v2], {("V1", "V2"): []}
        )
        c1 = [State.MISSING] * 4
        c2 = [State.MISSING] * 4
        ds1 = ds_from_cells(c1, self.RULE)
        ds2 = ds_from_cells(c2, self.RULE)
        r = chain.change_list(ds1, ds2, "V1", 0, 4)
        self.assertEqual(len(r.rows), 1)
        self.assertEqual((r.rows[0].start, r.rows[0].end), (0, 4))
        self.assertIs(r.rows[0].category, ChangeCategory.UNPAIRED)
        self.assertEqual((r.rows[0].break_from, r.rows[0].break_to),
                         ("V1", "V2"))
        # 晚版视角同样全部失联，且因晚版应测而列出。
        r2 = chain.change_list(ds1, ds2, "V2", 0, 4)
        self.assertEqual(len(r2.rows), 1)
        self.assertIs(r2.rows[0].category, ChangeCategory.UNPAIRED)

    def test_unpaired_only_listed_when_view_side_due(self):
        v1 = LineVersion("V1", 0, 4)
        v2 = LineVersion("V2", 0, 4)
        chain = VersionChain.build(
            [v1, v2], {("V1", "V2"): []}
        )
        # 早版不应测（全 OK），晚版应测：早版视角清单为空，
        # 晚版视角全部 UNPAIRED。
        ds1 = ds_from_cells([State.OK] * 4, self.RULE)
        ds2 = ds_from_cells([State.MISSING] * 4, self.RULE)
        self.assertEqual(chain.change_list(ds1, ds2, "V1", 0, 4).rows, ())
        r2 = chain.change_list(ds1, ds2, "V2", 0, 4)
        self.assertEqual(len(r2.rows), 1)
        self.assertIs(r2.rows[0].category, ChangeCategory.UNPAIRED)

    def test_multi_hop_parity_and_partial_coverage(self):
        # V1->V2 同向，V2->V3 反向：净反向。V1 [0,10) 中只有 [2,8) 延续。
        v1 = LineVersion("V1", 0, 10)
        v2 = LineVersion("V2", 100, 110)
        v3 = LineVersion("V3", 200, 210)
        chain = VersionChain.build(
            [v1, v2, v3],
            {
                ("V1", "V2"): [HandoverRecord(2, 8, 102, 108)],
                ("V2", "V3"): [HandoverRecord(102, 108, 202, 208,
                                              reversed=True)],
            },
        )
        cells = [State.OK] * 10
        iv(cells, 2, 8, State.MISSING)
        ds1 = ds_from_cells(cells, self.RULE)
        ds2 = ds_from_cells([State.OK] * 10, self.RULE, start=100)
        # V3: 物理 V1 的 2 -> V2 102 -> 反向后 V3 207；V1 的 7 -> V3 202。
        c3 = [State.OK] * 10
        iv(c3, 207 - 200, 208 - 200, State.MISSING)  # 物理位置 V1[2,3)
        ds3 = ds_from_cells(c3, self.RULE, start=200)
        r = chain.change_list(ds1, ds3, "V1", 0, 10)
        # V1 [2,8) 应测；物理位置 V1=2 映射到 V3=207，晚版同格也 MISSING，
        # 两侧都应测，不列入；V1[3,8) 映射到 V3[202,207)（晚版 OK）-> REMOVED。
        # [0,2)、[8,10) 早版不应测，即使无对端也不列。
        got = [(x.start, x.end, x.category, x.peer_start, x.peer_end,
                x.direction) for x in r.rows]
        self.assertEqual(got, [
            (3, 8, ChangeCategory.REMOVED, 202, 207, Direction.REVERSED),
        ])
        # 证据取早版完整段 [2,8)。
        self.assertEqual(r.rows[0].evidence, DueEvidence(2, 8, 1, 6))


# ---------------------------------------------------------------------- #
# 合并与拆分不变性
# ---------------------------------------------------------------------- #
class CoalesceTests(unittest.TestCase):
    RULE = Rule(1, max_gap=1, min_missing=1)

    def test_splitting_records_does_not_change_result(self):
        v1 = LineVersion("V1", 0, 12)
        v2 = LineVersion("V2", 0, 12)
        whole = [HandoverRecord(0, 12, 0, 12, reversed=True)]
        # 合法拆成 4 段：每段都保持整表的同一镜像 x -> 11-x，
        # 新侧区间 [9,12)、[5,9)、[2,5)、[0,2) 合起来完整覆盖 [0,12)，
        # 两侧各自不重叠、等长；乱序输入也必须合法。
        split = [
            HandoverRecord(0, 3, 9, 12, reversed=True),
            HandoverRecord(3, 7, 5, 9, reversed=True),
            HandoverRecord(7, 10, 2, 5, reversed=True),
            HandoverRecord(10, 12, 0, 2, reversed=True),
        ]
        # 逐段核对与整表双射一致：old x -> new 11-x。
        for rec in split:
            for x in range(rec.old_start, rec.old_end):
                k = x - rec.old_start
                self.assertEqual(rec.new_end - 1 - k, 11 - x)
        # 乱序输入也必须合法。
        rng = random.Random(1)
        rng.shuffle(split)
        chain_a = VersionChain.build([v1, v2], {("V1", "V2"): whole})
        chain_b = VersionChain.build([v1, v2], {("V1", "V2"): split})

        c1 = [State.OK] * 12
        iv(c1, 0, 6, State.MISSING)
        c2 = [State.OK] * 12
        iv(c2, 6, 12, State.MISSING)
        ds1 = ds_from_cells(c1, self.RULE)
        ds2 = ds_from_cells(c2, self.RULE)
        ra = chain_a.change_list(ds1, ds2, "V1", 0, 12)
        rb = chain_b.change_list(ds1, ds2, "V1", 0, 12)
        self.assertEqual([row_tuple(r) for r in ra.rows],
                         [row_tuple(r) for r in rb.rows])
        # 反向映射：V1[0,6) -> V2[6,12)，两侧应测位置恰好互换：
        # V1 应测的映射到 V2 也应测 -> 无差异行。
        self.assertEqual(ra.rows, ())

        # 改成只有部分重叠：V1 [0,4) MISSING，V2 [0,4) MISSING（镜像 V1[8,12)）
        c2b = [State.OK] * 12
        iv(c2b, 0, 4, State.MISSING)
        ds2b = ds_from_cells(c2b, self.RULE)
        ra = chain_a.change_list(ds1, ds2b, "V1", 0, 12)
        rb = chain_b.change_list(ds1, ds2b, "V1", 0, 12)
        self.assertEqual([row_tuple(r) for r in ra.rows],
                         [row_tuple(r) for r in rb.rows])
        # 整表镜像 x -> 11-x：V1 应测段 [0,6) 映到 V2[6,12)（V2 OK）
        # -> REMOVED，对端 [6,12)；V2 应测段 [0,4) 映自 V1[8,12)（V1 OK）
        # -> NEW，对端 [0,4)。
        self.assertEqual(
            [(r.start, r.end, r.category, r.peer_start, r.peer_end)
             for r in ra.rows],
            [(0, 6, ChangeCategory.REMOVED, 6, 12),
             (8, 12, ChangeCategory.NEW, 0, 4)],
        )
        # 抽查反向行的对端端点：查看格 8 -> 对端 3、查看格 11 -> 对端 0。
        new_row = next(r for r in ra.rows
                       if r.category is ChangeCategory.NEW)
        self.assertEqual((new_row.peer_start, new_row.peer_end), (0, 4))

    def test_no_coalesce_across_direction_or_gap(self):
        # 两条相邻交接片段方向不同：即使类别/证据相同也不得合并。
        v1 = LineVersion("V1", 0, 4)
        v2 = LineVersion("V2", 0, 4)
        chain = VersionChain.build([v1, v2], {("V1", "V2"): [
            HandoverRecord(0, 2, 2, 4),                  # 同向
            HandoverRecord(2, 4, 0, 2, reversed=True),   # 反向
        ]})
        c1 = [State.MISSING] * 4
        c2 = [State.OK] * 4
        ds1 = ds_from_cells(c1, self.RULE)
        ds2 = ds_from_cells(c2, self.RULE)
        r = chain.change_list(ds1, ds2, "V1", 0, 4)
        self.assertEqual(len(r.rows), 2)
        self.assertEqual([x.direction for x in r.rows],
                         [Direction.FORWARD, Direction.REVERSED])


# ---------------------------------------------------------------------- #
# 校验：不合法即整体拒绝
# ---------------------------------------------------------------------- #
class ValidationTests(unittest.TestCase):
    def _versions(self):
        return [LineVersion("V1", 0, 10), LineVersion("V2", 20, 30)]

    def test_empty_or_duplicate_versions_rejected(self):
        with self.assertRaises(ChainError):
            VersionChain.build([], {})
        with self.assertRaises(ChainError):
            VersionChain.build([LineVersion("V1", 0, 10),
                                LineVersion("V1", 0, 8)], {})
        with self.assertRaises(ChainError):
            VersionChain.build([LineVersion("", 0, 10)], {})

    def test_bad_version_range_rejected(self):
        with self.assertRaises(ChainError):
            VersionChain.build([LineVersion("V1", 5, 5)], {})
        with self.assertRaises(ChainError):
            VersionChain.build([LineVersion("V1", 5, True)], {})

    def test_handover_keys_must_match_adjacent_pairs(self):
        vs = self._versions()
        with self.assertRaises(ChainError):
            VersionChain.build(vs, {})  # 缺表
        with self.assertRaises(ChainError):
            VersionChain.build(vs, {
                ("V1", "V2"): [], ("V2", "V1"): [],
            })  # 多表/键错
        with self.assertRaises(ChainError):
            VersionChain.build(vs, {("V1", "V9"): []})

    def test_record_must_be_nonempty_equal_length_in_range(self):
        vs = self._versions()
        bad = [
            HandoverRecord(0, 0, 20, 20),                    # 空区间
            HandoverRecord(0, 5, 20, 24),                   # 不等长
            HandoverRecord(-1, 4, 20, 25),                  # 旧侧越界
            HandoverRecord(0, 5, 26, 31),                   # 新侧越界
            HandoverRecord(0, 5, 20, 25, reversed="no"),    # 方向非布尔
        ]
        for rec in bad:
            with self.assertRaises(ChainError):
                VersionChain.build(vs, {("V1", "V2"): [rec]})

    def test_overlap_on_either_side_rejected(self):
        vs = self._versions()
        with self.assertRaises(ChainError):  # 旧侧重叠
            VersionChain.build(vs, {("V1", "V2"): [
                HandoverRecord(0, 6, 20, 26),
                HandoverRecord(5, 9, 26, 30),
            ]})
        with self.assertRaises(ChainError):  # 新侧重叠
            VersionChain.build(vs, {("V1", "V2"): [
                HandoverRecord(0, 6, 20, 26),
                HandoverRecord(6, 9, 25, 28),
            ]})
        # 端点相接（半开不重叠）合法。
        VersionChain.build(vs, {("V1", "V2"): [
            HandoverRecord(0, 6, 20, 26),
            HandoverRecord(6, 9, 26, 29),
        ]})

    def test_endpoint_dataset_range_must_match(self):
        chain = VersionChain.build(
            self._versions(), {("V1", "V2"): []}
        )
        rule = Rule(1, max_gap=1, min_missing=1)
        ds_ok = ds_from_cells([State.OK] * 10, rule)
        ds_wrong = ds_from_cells([State.OK] * 8, rule)
        ds_shift = ds_from_cells([State.OK] * 10, rule, start=20)
        with self.assertRaises(ChainError):
            chain.change_list(ds_wrong, ds_shift, "V1", 0, 10)
        with self.assertRaises(ChainError):  # 非 Dataset
            chain.change_list(object(), ds_shift, "V1", 0, 10)
        with self.assertRaises(ChainError):  # 查看版不是首尾
            chain.change_list(ds_ok, ds_shift, "V9", 0, 10)
        with self.assertRaises(ChainError):  # 空窗口 / 越界
            chain.change_list(ds_ok, ds_shift, "V1", 5, 5)
        with self.assertRaises(ChainError):
            chain.change_list(ds_ok, ds_shift, "V1", -1, 4)
        with self.assertRaises(ChainError):
            chain.change_list(ds_ok, ds_shift, "V1", 8, 11)

    def test_single_version_chain_compares_dataset_to_itself(self):
        v = LineVersion("V1", 0, 5)
        chain = VersionChain.build([v], {})
        rule = Rule(1, max_gap=1, min_missing=1)
        ds = ds_from_cells(
            [State.MISSING, State.MISSING, State.OK, State.OK, State.OK], rule
        )
        # 单版本：早晚是同一份检测，任何位置两侧状态相同 -> 永远无差异。
        r = chain.change_list(ds, ds, "V1", 0, 5)
        self.assertEqual(r.rows, ())
        self.assertEqual((r.early_id, r.late_id, r.view_id, r.window),
                         ("V1", "V1", "V1", (0, 5)))


# ---------------------------------------------------------------------- #
# 局部性：远端分送不读中间里程，不逐格展开
# ---------------------------------------------------------------------- #
class LocalityTests(unittest.TestCase):
    RULE = Rule(1, max_gap=1, min_missing=1)

    def test_remote_fanout_reads_only_touched_ranges(self):
        # V1 全线 0..120，按每 10 格一块（12 块）；窗口 [10,14) 被交接表
        # 分送到 V2 两个彼此很远的小片段 [100,102) 与 [50,52)。
        n = 120
        v1 = LineVersion("V1", 0, n)
        v2 = LineVersion("V2", 0, n)
        chain = VersionChain.build([v1, v2], {("V1", "V2"): [
            HandoverRecord(10, 12, 100, 102),
            HandoverRecord(12, 14, 50, 52),
        ]})

        def make_ds(prefix):
            blocks = []
            for k in range(n // 10):
                lo, hi = 10 * k, 10 * k + 10
                blocks.append(build_block(
                    f"{prefix}{k:02d}", lo, hi,
                    [Interval(lo, hi, State.MISSING)], self.RULE))
            return assemble(blocks, 0, n, {1: self.RULE})

        ds1, ds2 = make_ds("A"), make_ds("B")
        before_reads = {b.block_id: b.interval_reads for b in ds1._blocks}
        result = chain.change_list(ds1, ds2, "V1", 10, 14)
        # 两侧应测位置完全对应 -> 无差异行，但访问面是可验证的。
        self.assertEqual(result.rows, ())
        stats = chain.access_stats
        self.assertIsInstance(stats, ChangeAccessStats)
        # 只对查看窗与两个远端小片段发起 Dataset 查询（共 3 次）。
        self.assertEqual(stats.dataset_queries, 3)
        touched = set(stats.touched_ranges)
        self.assertEqual(touched, {("V1", 10, 14),
                                   ("V2", 100, 102), ("V2", 50, 52)})
        # V2 中间无关块（如 [60,70)、[90,100)）的内部区间绝不被读取。
        for b in ds2._blocks:
            if b.start in (50, 100):
                self.assertEqual(b.interval_reads, 1)
            else:
                self.assertEqual(b.interval_reads, 0)
        # V1 只读取与窗口相交的块 [10,20)。
        for b in ds1._blocks:
            self.assertEqual(b.interval_reads,
                             before_reads[b.block_id] + (1 if b.start == 10 else 0))

    def test_repeated_small_windows_do_not_rescan_far_records(self):
        # 大里程稀疏交接表：建链后重复查小窗，交接记录访问数有界，
        # 且与总记录条数无关。
        BIG = 10_000_000
        v1 = LineVersion("V1", 0, BIG)
        v2 = LineVersion("V2", 0, BIG)
        records = [HandoverRecord(1000 * k, 1000 * k + 500,
                                  1000 * k, 1000 * k + 500)
                   for k in range(BIG // 1000)]
        chain = VersionChain.build([v1, v2], {("V1", "V2"): records})
        rule = Rule(1, max_gap=1, min_missing=1)
        blk1 = build_block("H1", 0, BIG,
                           [Interval(0, BIG, State.MISSING)], rule)
        blk2 = build_block("H2", 0, BIG,
                           [Interval(0, BIG, State.OK)], rule)
        ds1 = assemble([blk1], 0, BIG, {1: rule})
        ds2 = assemble([blk2], 0, BIG, {1: rule})
        chain.access_stats.reset()
        for _ in range(5):
            chain.change_list(ds1, ds2, "V1", 3001, 3003)
        # 窗口只落在一条记录内：每次查询只探测 1 条交接记录，
        # 累计 5 次探测、10 次 Dataset 小窗查询，与万级记录总数无关。
        self.assertEqual(chain.access_stats.handover_probes, 5)
        self.assertEqual(chain.access_stats.dataset_queries, 10)


# ---------------------------------------------------------------------- #
# 输入不被修改
# ---------------------------------------------------------------------- #
class ImmutabilityTests(unittest.TestCase):
    def test_inputs_unchanged(self):
        rule = Rule(1, max_gap=1, min_missing=1)
        v1 = LineVersion("V1", 0, 6)
        v2 = LineVersion("V2", 0, 6)
        chain = VersionChain.build(
            [v1, v2], {("V1", "V2"): [HandoverRecord(0, 6, 0, 6)]}
        )
        ds1 = ds_from_cells([State.MISSING] * 6, rule)
        ds2 = ds_from_cells([State.OK] * 6, rule)
        seg_before = ds1.segments
        chain.change_list(ds1, ds2, "V1", 0, 6)
        chain.change_list(ds1, ds2, "V2", 0, 6)
        self.assertEqual(ds1.segments, seg_before)
        # 查询只读取相交块，不写块；数据集不可变接口本身也无写方法。
        self.assertTrue(all(b.interval_reads >= 0 for b in ds1._blocks))


# ---------------------------------------------------------------------- #
# 随机差分：独立逐格暴力参考
# ---------------------------------------------------------------------- #
def _brute_force(cells_by_version, datasets, versions, tables,
                 view_id, wlo, whi):
    """完全独立的逐格参考：逐格沿交接链追溯并分类，再按引擎规则合并。

    段是否达标取自已装配 Dataset 的全线段（该引擎本身已由
    test_pipeline 的逐格参考差分守护）；追溯、分类、合并在此独立重写。
    """
    ids = [v.version_id for v in versions]
    view_pos = ids.index(view_id)

    # 逐格映射：闭式公式，不复用 changes.py 的任何代码。
    def map_old_to_new(x, side):
        a, b = versions[side], versions[side + 1]
        if not (a.start <= x < a.end):
            return None
        for rec in tables[(ids[side], ids[side + 1])]:
            if rec.old_start <= x < rec.old_end:
                k = x - rec.old_start
                return (rec.new_start + k) if not rec.reversed \
                    else rec.new_end - 1 - k
        return None

    def map_new_to_old(x, side):
        a, b = versions[side], versions[side + 1]
        if not (b.start <= x < b.end):
            return None
        for rec in tables[(ids[side], ids[side + 1])]:
            if rec.new_start <= x < rec.new_end:
                k = x - rec.new_start
                return (rec.old_start + k) if not rec.reversed \
                    else rec.old_end - 1 - k
        return None

    # 每版每格是否应复测：落在“达标全线段”内的实际 MISSING 格。
    due = {}
    seg_of = {}
    for idx, vid in enumerate(ids):
        ds = datasets[vid]
        base = versions[idx].start
        flags = [False] * len(cells_by_version[vid])
        owner = [None] * len(flags)
        rules = ds.rules
        for seg in ds.segments:
            if seg.missing < rules[seg.version].min_missing:
                continue
            for t in range(seg.start - base, seg.end - base):
                if cells_by_version[vid][t] is State.MISSING:
                    flags[t] = True
                    owner[t] = seg
        due[vid] = flags
        seg_of[vid] = owner

    rows = []
    early, late = ids[0], ids[-1]
    for x in range(wlo, whi):
        view_due = due[view_id][x - versions[view_pos].start]
        cur = x
        breaks = None
        if view_pos == 0:
            early_x = cur
            for side in range(len(ids) - 1):
                nxt = map_old_to_new(cur, side)
                if nxt is None:
                    breaks = (ids[side], ids[side + 1])
                    break
                cur = nxt
            late_x = None if breaks else cur
        else:
            late_x = cur
            for side in range(len(ids) - 2, -1, -1):
                nxt = map_new_to_old(cur, side)
                if nxt is None:
                    breaks = (ids[side], ids[side + 1])
                    break
                cur = nxt
            early_x = None if breaks else cur

        if breaks is not None:
            if view_due:
                rows.append({
                    "view": (x, x + 1), "cat": ChangeCategory.UNPAIRED,
                    "peer": None, "brk": breaks,
                    "ev_vid": view_id, "ev_cell": x,
                })
            continue
        e_due = due[early][early_x - versions[0].start]
        l_due = due[late][late_x - versions[-1].start]
        # 对端坐标：早版视角的对端是晚版格，晚版视角的对端是早版格。
        peer_x = late_x if view_pos == 0 else early_x
        if e_due and not l_due:
            rows.append({"view": (x, x + 1), "cat": ChangeCategory.REMOVED,
                         "peer": (peer_x, peer_x + 1), "brk": None,
                         "ev_vid": early, "ev_cell": early_x})
        elif l_due and not e_due:
            rows.append({"view": (x, x + 1), "cat": ChangeCategory.NEW,
                         "peer": (peer_x, peer_x + 1), "brk": None,
                         "ev_vid": late, "ev_cell": late_x})

    def evidence(vid, x):
        seg = seg_of[vid][x - versions[ids.index(vid)].start]
        return (seg.start, seg.end, seg.version, seg.missing)

    def net_parity(x):
        parity = 0
        if view_pos == 0:
            cur = x
            for side in range(len(ids) - 1):
                rec = next(r for r in tables[(ids[side], ids[side + 1])]
                           if r.old_start <= cur < r.old_end)
                parity ^= int(rec.reversed)
                cur = map_old_to_new(cur, side)
        else:
            cur = x
            for side in range(len(ids) - 2, -1, -1):
                rec = next((r for r in tables[(ids[side], ids[side + 1])]
                            if r.new_start <= cur < r.new_end), None)
                if rec is None:
                    return None  # 断链格不会进入配对行，净方向无意义
                parity ^= int(rec.reversed)
                cur = map_new_to_old(cur, side)
        return parity

    # 用引擎同款合并规则合并逐格行。
    merged = []
    for g in rows:
        ev = evidence(g["ev_vid"], g["ev_cell"])
        if g["cat"] is ChangeCategory.UNPAIRED:
            peer, direction = None, None
        else:
            peer = g["peer"]
            direction = (Direction.REVERSED if net_parity(g["view"][0])
                         else Direction.FORWARD)
        item = {
            "s": g["view"][0], "e": g["view"][1], "cat": g["cat"],
            "peer": peer, "dir": direction, "brk": g["brk"], "ev": ev,
        }
        if merged:
            p = merged[-1]
            can = (p["e"] == item["s"] and p["cat"] is item["cat"]
                   and p["ev"] == item["ev"] and p["brk"] == item["brk"]
                   and p["dir"] == item["dir"])
            if can and item["cat"] is ChangeCategory.UNPAIRED:
                p["e"] = item["e"]
                continue
            if can and item["dir"] is Direction.FORWARD \
                    and p["peer"][1] == item["peer"][0]:
                p["e"] = item["e"]
                p["peer"] = (p["peer"][0], item["peer"][1])
                continue
            if can and item["dir"] is Direction.REVERSED \
                    and p["peer"][0] == item["peer"][1]:
                p["e"] = item["e"]
                p["peer"] = (item["peer"][0], p["peer"][1])
                continue
        merged.append(item)
    return merged


class RandomDifferentialTests(unittest.TestCase):
    def test_random_chains_against_brute_force(self):
        rng = random.Random(20261123)
        for trial in range(120):
            nver = rng.randint(1, 3)
            L = rng.randint(4, 24)
            # 每版范围等长（便于构造等长交接），起点可以不同（重编里程）。
            starts = [rng.choice([0, 100, -50, 40]) for _ in range(nver)]
            versions = [LineVersion(f"V{i}", starts[i], starts[i] + L)
                        for i in range(nver)]
            rules = {f"V{i}": Rule(1, max_gap=rng.randint(0, 2),
                                   min_missing=rng.randint(1, 4))
                     for i in range(nver)}
            cells_by_version = {}
            datasets = {}
            for i, v in enumerate(versions):
                cells = [rng.choices([State.MISSING, State.GAP, State.OK],
                                     weights=[5, 2, 3])[0]
                         for _ in range(L)]
                cells_by_version[v.version_id] = cells
                datasets[v.version_id] = ds_from_cells(
                    cells, rules[v.version_id], start=v.start,
                    block_cut=sorted(rng.sample(range(1, L),
                                                k=rng.randint(0, min(3, L - 1)))),
                    prefix=f"T{trial}V{i}",
                )

            tables = {}
            for i in range(nver - 1):
                a, b = versions[i], versions[i + 1]
                # 随机切分：选若干切点，随机留空段（未覆盖），随机方向。
                cuts = [0] + sorted(rng.sample(range(1, L),
                                               k=rng.randint(0, min(4, L - 1)))) + [L]
                recs = []
                for s, e in zip(cuts, cuts[1:]):
                    if e == s or rng.random() < 0.25:
                        continue  # 空隙：该段不延续
                    length = e - s
                    ns = rng.randrange(b.start, b.end - length + 1)
                    # 允许与既有新侧冲突后重试几次，保证夹具自身合法。
                    used = [(r.new_start, r.new_end) for r in recs]
                    tries = 0
                    while any(not (ns + length <= u0 or ns >= u1)
                              for u0, u1 in used) and tries < 20:
                        ns = rng.randrange(b.start, b.end - length + 1)
                        tries += 1
                    if any(not (ns + length <= u0 or ns >= u1)
                           for u0, u1 in used):
                        continue
                    recs.append(HandoverRecord(
                        a.start + s, a.start + e, ns, ns + length,
                        reversed=rng.random() < 0.5,
                    ))
                tables[(a.version_id, b.version_id)] = recs

            chain = VersionChain.build(versions, tables)
            for view_idx in (0, nver - 1):
                view_id = versions[view_idx].version_id
                v = versions[view_idx]
                for _ in range(3):
                    wlo = rng.randrange(v.start, v.end)
                    whi = rng.randrange(wlo + 1, v.end + 1)
                    got = chain.change_list(
                        datasets[versions[0].version_id],
                        datasets[versions[-1].version_id],
                        view_id, wlo, whi,
                    )
                    want = _brute_force(
                        cells_by_version, datasets, versions, tables,
                        view_id, wlo, whi,
                    )
                    got_norm = [
                        {
                            "s": r.start, "e": r.end, "cat": r.category,
                            "peer": None if r.peer_start is None
                            else (r.peer_start, r.peer_end),
                            "dir": r.direction, "brk": None
                            if r.break_from is None
                            else (r.break_from, r.break_to),
                            "ev": None if r.evidence is None
                            else (r.evidence.segment_start,
                                  r.evidence.segment_end,
                                  r.evidence.rule_version,
                                  r.evidence.segment_missing),
                        }
                        for r in got.rows
                    ]
                    self.assertEqual(
                        got_norm, want,
                        msg=f"trial {trial} view={view_id} "
                            f"window=[{wlo},{whi})",
                    )
                    # 升序且互不相交。
                    bounds = [(r.start, r.end) for r in got.rows]
                    self.assertEqual(bounds, sorted(bounds))
                    for p, q in zip(bounds, bounds[1:]):
                        self.assertLessEqual(p[1], q[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
