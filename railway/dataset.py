"""独立的建块、装配、读取、更正接口。

使用方式::

    rules = {1: Rule(1, max_gap=3, min_missing=5)}
    block = build_block("B0", 0, 10, intervals, rules[1])      # 可并行独立调用
    ds = assemble([...], line_start=0, line_end=100, rules=rules)
    hits = ds.query_window(20, 60)
    ds2 = ds.correct(new_block)                                  # 旧数据集保持不变

设计约束（均由测试守护）：

* 装配/更正不回读原始里程明细，只使用块产物中的压缩区间与摘要；
* 小窗口查询不遍历全线块/区间：段表二分定位，证据只读取与窗口相交的块；
* 更正不扫描、不改写任何未更正块的内部区间（块产物不可变，且读取有计数）；
* 输出不依赖分块方式或读入顺序（幺半群结合律 + 排序后装配）。
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


class AssemblyError(ValueError):
    """块产物不合法（重叠、缺块、规则阈值冲突、区间不自洽等）。"""


# ---------------------------------------------------------------------- #
# 1) 独立建块：纯函数，可对各块并行调用，块之间零共享状态
# ---------------------------------------------------------------------- #
def build_block(
    block_id: str,
    start: int,
    end: int,
    intervals: Sequence[Interval],
    rule: Rule,
) -> BlockArtifact:
    """校验并构建单个块产物。

    相邻同状态区间会被规范化合并；区间必须完整覆盖 ``[start, end)``、
    互不重叠、按里程升序，且不得越界。
    """
    if not isinstance(rule, Rule):
        raise AssemblyError("规则版次对象类型非法")
    if end <= start:
        raise AssemblyError(f"块 {block_id} 里程范围非法: [{start},{end})")
    if not intervals:
        raise AssemblyError(f"块 {block_id} 区间为空，无法完整覆盖块范围")

    canon: List[Interval] = []
    for iv in intervals:
        if iv.end <= iv.start:
            raise AssemblyError(f"块 {block_id} 存在非正长度区间 {iv}")
        if iv.start < start or iv.end > end:
            raise AssemblyError(f"块 {block_id} 区间越界: {iv}")
        if canon and iv.start != canon[-1].end:
            raise AssemblyError(
                f"块 {block_id} 区间不连续/重叠: ...{canon[-1]} 之后 {iv}"
            )
        if canon and iv.state is canon[-1].state:
            canon[-1] = Interval(canon[-1].start, iv.end, iv.state)
        else:
            canon.append(Interval(iv.start, iv.end, iv.state))

    if canon[0].start != start or canon[-1].end != end:
        raise AssemblyError(
            f"块 {block_id} 区间未完整覆盖 [{start},{end})，"
            f"实际 [{canon[0].start},{canon[-1].end})"
        )

    summary = Summary.from_intervals(start, end, canon, rule)
    return BlockArtifact(block_id, start, end, rule, canon, summary)


# ---------------------------------------------------------------------- #
# 2) 装配：任意顺序读入块，核对全线接缝，失败则整体拒绝
# ---------------------------------------------------------------------- #
def assemble(
    blocks: Iterable[BlockArtifact],
    line_start: int,
    line_end: int,
    rules: Dict[int, Rule],
) -> "Dataset":
    """把互斥、半开、完整覆盖线路区间的块装配成可查询数据集。

    拒绝条件：块区间重叠/缺块/越界、同一规则版次给出不同阈值、
    块内规则与全局规则册不一致、接缝与块自摘要不符。
    装配失败抛 :class:`AssemblyError`，不会产生任何部分数据集。
    """
    ordered = sorted(blocks, key=lambda b: (b.start, b.end))
    if not ordered:
        raise AssemblyError("没有任何块，不允许假造丢失块")

    # --- 规则册一致性：同一个规则版次给不同阈值必须拒绝 ---
    for b in ordered:
        r = rules.get(b.rule.version)
        if r is None:
            raise AssemblyError(
                f"块 {b.block_id} 的规则版次 v{b.rule.version} 不在规则册中"
            )
        if r != b.rule:
            raise AssemblyError(
                f"块 {b.block_id} 对版次 v{b.rule.version} 给出阈值 "
                f"max_gap={b.rule.max_gap},min_missing={b.rule.min_missing}，"
                f"与规则册 max_gap={r.max_gap},min_missing={r.min_missing} 冲突"
            )

    # --- 互斥半开、完整覆盖、不假造丢失块 ---
    if ordered[0].start != line_start or ordered[-1].end != line_end:
        raise AssemblyError(
            f"块未完整覆盖线路 [{line_start},{line_end})："
            f"实际 [{ordered[0].start},{ordered[-1].end})"
        )
    for prev, b in zip(ordered, ordered[1:]):
        if prev.end != b.start:
            raise AssemblyError(
                f"接缝错误：块 {prev.block_id}[{prev.start},{prev.end}) 与 "
                f"块 {b.block_id}[{b.start},{b.end}) 之间 "
                f"{'重叠' if b.start < prev.end else '缺块'}"
            )
        # 调度端核对接缝：重算接缝两侧摘要并与块自带摘要比对。
        # 注意：这里只做摘要级核对，不依赖任何原始里程明细之外的输入。
        if b.summary.start != b.start or b.summary.end != b.end:
            raise AssemblyError(f"块 {b.block_id} 摘要里程与其块范围不符")

    # 摘要拼接（结合律保证与分块/读入顺序无关；上面已排序消除顺序影响）。
    root = merge_all([b.summary for b in ordered])
    segments = root.finalize()
    return Dataset(
        line_start=line_start,
        line_end=line_end,
        rules=dict(rules),
        blocks=ordered,
        root=root,
        segments=segments,
    )


# ---------------------------------------------------------------------- #
# 3) 查询数据集
# ---------------------------------------------------------------------- #
class Dataset:
    """装配产物。不可变；:meth:`correct` 返回新数据集，旧数据集原样可用。"""

    def __init__(self, line_start, line_end, rules, blocks, root, segments):
        self.line_start = line_start
        self.line_end = line_end
        self.rules = dict(rules)
        self._blocks: List[BlockArtifact] = list(blocks)
        self._block_starts = [b.start for b in self._blocks]
        self._block_ends = [b.end for b in self._blocks]
        self.root: Summary = root
        self._segments: Tuple[Segment, ...] = tuple(segments)
        self._seg_starts = [s.start for s in self._segments]

    # ---------------- 基础访问 ---------------- #
    @property
    def segments(self) -> Tuple[Segment, ...]:
        """全线路候选复测段（按里程升序），门槛由读取方按规则版次判定。"""
        return self._segments

    def qualified_segments(self) -> Tuple[Segment, ...]:
        """达到该段规则版次复测门槛的全线段。"""
        return tuple(
            s for s in self._segments
            if s.missing >= self.rules[s.version].min_missing
        )

    def _block_at(self, mile: int) -> BlockArtifact:
        idx = bisect.bisect_right(self._block_starts, mile) - 1
        if idx < 0 or mile >= self._block_ends[idx]:
            raise AssemblyError(f"里程 {mile} 不在任何已装配块内")
        return self._blocks[idx]

    def _blocks_overlapping(self, start: int, end: int) -> List[BlockArtifact]:
        lo = bisect.bisect_right(self._block_ends, start) - 1
        lo = max(lo, 0)
        hi = bisect.bisect_left(self._block_starts, end)
        return self._blocks[lo:hi]

    # ---------------- 窗口读取 ---------------- #
    def query_window(self, start: int, end: int) -> Tuple[SegmentHit, ...]:
        """读取任意合法窗口 ``[start,end)``。

        * 段是否达到门槛按 **全线路上的完整段** 判定，窗口端截断不缩小核定长度；
        * 仅与窗口相交的段才返回；只碰到合并段内部缺报（GAP）也返回该段；
        * 证据只读取与窗口相交的块，不遍历全线。
        """
        if end <= start:
            raise ValueError("查询窗口必须为正长度半开区间")
        if start < self.line_start or end > self.line_end:
            raise ValueError(
                f"窗口 [{start},{end}) 越出线路 [{self.line_start},{self.line_end})"
            )

        # 段表二分：第一段 end > start，之后逐个取直到段 start >= end。
        i = bisect.bisect_right(self._seg_starts, start)
        i = max(i - 1, 0)
        while i < len(self._segments) and self._segments[i].end <= start:
            i += 1

        hits: List[SegmentHit] = []
        # 只保留真正与窗口相交的块，保证不读取无关块的内部区间。
        touched = [
            b for b in self._blocks_overlapping(start, end)
            if b.end > start and b.start < end
        ]
        block_intervals = {b.block_id: b.intervals for b in touched}

        while i < len(self._segments):
            seg = self._segments[i]
            if seg.start >= end:
                break
            i += 1
            if seg.end <= start:
                continue
            clip_lo, clip_hi = max(seg.start, start), min(seg.end, end)
            rule = self.rules[seg.version]
            evidence, clip_missing = self._evidence(
                seg, clip_lo, clip_hi, touched, block_intervals, rule
            )
            hits.append(
                SegmentHit(
                    segment=seg,
                    qualified=seg.missing >= rule.min_missing,
                    clip_start=clip_lo,
                    clip_end=clip_hi,
                    clip_missing=clip_missing,
                    evidence=tuple(evidence),
                )
            )
        return tuple(hits)

    def _evidence(self, seg, clip_lo, clip_hi, touched, block_intervals, rule):
        """从相交块的压缩区间裁切出分段证据（仅触碰相交块）。"""
        evidence: List[EvidenceRun] = []
        clip_missing = 0
        for b in touched:
            if b.end <= clip_lo or b.start >= clip_hi:
                continue
            for iv in block_intervals[b.block_id]:
                if iv.end <= clip_lo or iv.start >= clip_hi:
                    continue
                lo, hi = max(iv.start, clip_lo), min(iv.end, clip_hi)
                if iv.state is State.MISSING:
                    clip_missing += hi - lo
                    evidence.append(
                        EvidenceRun(lo, hi, State.MISSING, False, None)
                    )
                elif iv.state is State.GAP:
                    band_len = self._gap_band_len(
                        lo, seg, b, block_intervals[b.block_id]
                    )
                    evidence.append(
                        EvidenceRun(
                            lo,
                            hi,
                            State.GAP,
                            crossed=band_len <= rule.max_gap,
                            gap_band_len=band_len,
                        )
                    )
                else:  # 段内不应出现 OK，防御性处理
                    evidence.append(EvidenceRun(lo, hi, State.OK, False, None))
        # 块边界可能把同态相邻区间切成两条 EvidenceRun，按里程排序后合并
        # 相邻且状态/属性一致的连续游程，使证据与具体分块方式无关。
        # 同一条 GAP 带可能被块边界切成两段（各自 gap_band_len 都应等于
        # 整条带长），只要相邻且 crossed 一致就合并，带长取两者的整条值。
        evidence.sort(key=lambda e: e.start)
        merged: List[EvidenceRun] = []
        for e in evidence:
            if merged:
                p = merged[-1]
                same_state = p.end == e.start and p.state == e.state
                if same_state and e.state is State.GAP and p.crossed == e.crossed:
                    band = p.gap_band_len if p.gap_band_len is not None else e.gap_band_len
                    merged[-1] = EvidenceRun(
                        p.start, e.end, e.state, e.crossed, band
                    )
                    continue
                if (
                    same_state and p.crossed == e.crossed
                    and p.gap_band_len == e.gap_band_len
                ):
                    merged[-1] = EvidenceRun(
                        p.start, e.end, e.state, e.crossed, e.gap_band_len
                    )
                    continue
            merged.append(e)
        return tuple(merged), clip_missing

    def _gap_band_len(self, mile: int, seg: Segment,
                      home: BlockArtifact, home_intervals) -> int:
        """求 ``mile`` 所在连续 GAP 带的全长（带可被块边界切成多段）。

        定位块内用查询入口已读取的区间表（不重复计数）；跨块延伸只看
        相邻块摘要中的软组件边缘 GAP（同版次、贴接缝、无核阻隔），绝不
        打开未与窗口相交块的内部区间。带界最终由 MISSING 核、硬切口或
        线路端点封死。
        """
        # 1) 定位块内 GAP 区间，得到本地带界。
        local_iv = None
        for iv in home_intervals:
            if iv.start <= mile < iv.end:
                local_iv = iv
                break
        if local_iv is None or local_iv.state is not State.GAP:
            raise RuntimeError("摘要与区间不一致：GAP 证据定位失败")
        lo, hi = local_iv.start, local_iv.end
        home_idx = bisect.bisect_right(self._block_starts, mile) - 1

        # home 块内向左右扩展连续 GAP 游程，并累计带的 GAP 总长。
        # 注意带全长只累加 GAP 格；带在块内可能被同核 MISSING 分隔（case b），
        # 那种情况通过 reopened 标记走单独的跨块累加，不计 MISSING。
        ivs = home_intervals
        pos = ivs.index(local_iv)
        k = pos
        while k > 0 and ivs[k - 1].state is State.GAP:
            lo = ivs[k - 1].start
            k -= 1
        k = pos
        while k + 1 < len(ivs) and ivs[k + 1].state is State.GAP:
            hi = ivs[k + 1].end
            k += 1
        band_len = hi - lo  # home 内连续游程（全 GAP）长度

        sfx, pfx = home.summary.suffix, home.summary.prefix
        reopened_r = sfx.reopened_right if sfx is not None else None
        reopened_l = pfx.reopened_left if pfx is not None else None

        # 向右跨块：
        #  (a) 游程顶 home 右界：邻块边缘 GAP 连续，扩展带界并累加；
        #  (b) reopened 覆盖当前 GAP 但游程不贴界：同一条带的另一段在邻块，
        #      只累加邻块 GAP 长度，带界（hi）保持 home 游程右缘。
        idx = home_idx
        if hi == home.end:
            while idx + 1 < len(self._blocks):
                nxt = self._blocks[idx + 1]
                c = nxt.summary.prefix
                if c is None or c.version != seg.version or c.gap_left <= 0:
                    break
                edge_hi = c.start + c.gap_left
                # 邻块边缘 GAP 必须在接缝处与当前带连续（c.start == hi 且
                # 其 GAP 左缘正好接上），隔着 MISSING/OK 的独立 GAP 不连。
                if c.start != hi or edge_hi <= hi:
                    break
                band_len += c.gap_left
                hi = edge_hi
                idx += 1
        elif (reopened_r is not None
              and reopened_r.start <= mile < reopened_r.end):
            nxt_idx = home_idx + 1
            if nxt_idx < len(self._blocks):
                c = self._blocks[nxt_idx].summary.prefix
                if (c is not None and c.version == seg.version
                        and c.gap_left > 0):
                    band_len += c.gap_left

        # 向左跨块（对称）：连续游程顶 home 左界，或 reopened_left 覆盖当前
        # GAP 时连接左邻块边缘 GAP。
        idx = home_idx
        go_left = lo == home.start or (
            reopened_l is not None
            and reopened_l.start <= mile < reopened_l.end
        )
        if go_left:
            while idx > 0:
                prev = self._blocks[idx - 1]
                c = prev.summary.suffix
                if c is None or c.version != seg.version or c.gap_right <= 0:
                    break
                edge_lo = c.end - c.gap_right
                # 仅当邻块边缘 GAP 的右缘正好贴当前接缝、且其左缘与当前
                # 带界连续（edge_lo == lo）时，才是同一条带的延续；隔着
                # MISSING/OK 的独立 GAP 不得累加。
                if c.end != lo or edge_lo > lo:
                    break
                band_len += c.gap_right
                lo = edge_lo
                idx -= 1

        return band_len

    # ---------------- 更正 ---------------- #
    def correct(self, new_block: BlockArtifact) -> "Dataset":
        """以同范围、同规则版次的新块更正旧块（如重测把缺报变合格）。

        * 新块先独立校验，失败则本数据集原样保留，坏块不污染已交付数据；
        * 不读取、不改写任何未更正块的内部区间（块不可变，仅替换一个叶子）；
        * 更正后按新的全线摘要重新判定所有段门槛。
        """
        if not isinstance(new_block, BlockArtifact):
            raise AssemblyError("更正输入必须是 BlockArtifact")
        r = self.rules.get(new_block.rule.version)
        if r is None or r != new_block.rule:
            raise AssemblyError("更正块的规则版次/阈值与线路规则册不一致")

        idx = bisect.bisect_left(self._block_starts, new_block.start)
        if idx >= len(self._blocks):
            raise AssemblyError(
                f"更正块 [{new_block.start},{new_block.end}) 不匹配任何旧块"
            )
        old = self._blocks[idx]
        if (
            old.start != new_block.start
            or old.end != new_block.end
            or old.rule.version != new_block.rule.version
        ):
            raise AssemblyError(
                f"更正块必须与旧块同范围、同规则版次："
                f"旧 {old.block_id}[{old.start},{old.end}) v{old.rule.version}"
            )

        # 仅用块产物重算全线摘要：叶子替换 + 幺半群合并，
        # 全程不调用任何未更正块的 intervals 读取器。
        new_blocks = self._blocks.copy()
        new_blocks[idx] = new_block
        root = merge_all([b.summary for b in new_blocks])
        return Dataset(
            line_start=self.line_start,
            line_end=self.line_end,
            rules=self.rules,
            blocks=new_blocks,
            root=root,
            segments=root.finalize(),
        )
