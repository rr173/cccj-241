"""区间流的可结合摘要（幺半群）。

硬切口（hard cut）有三种来源：

1. 合格测量 ``OK`` —— 永远不可跨过；
2. 超长缺报 —— 整条连续 GAP 带长度超过所在规则版次的 ``max_gap``；
3. 规则换版 —— 硬切口，即使两侧都缺测也不跨。

块边界 **不是** 切口：同一连续 GAP 带被切成多个块时，先在拼接处把两侧
GAP 长度相加，再与阈值比较。

一个区域（块或块集合）的摘要由三部分组成：

* ``prefix`` : 区域开头尚未被左侧硬切口封死的“软组件”；
* ``suffix`` : 区域结尾尚未被右侧硬切口封死的软组件；
* ``closed`` : 两端都已被硬切口（或区域内部切口）封死的复测段候选。

软组件 :class:`SoftComponent` 只含同一规则版次的 MISSING/GAP，无 OK，
内部所有已并入缺测核的 GAP 都已确认是短缺报；组件首尾未并入核的边缘
GAP（``gap_left``/``gap_right``）等待拼接后再判定。组件至多含一个缺测核
:class:`Core`，纯 GAP 组件没有核。

合并两个相邻区域摘要的代价是常数级，因此：

* 任意合法分块的装配结果与分块方式、读入顺序无关（结合律）；
* 更正一个块时只需重算该块叶子并沿摘要树向上合并，
  不需要打开任何未更正块的内部区间。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional

from .model import Core, Interval, Rule, Segment, State


@dataclass(frozen=True)
class SoftComponent:
    """软组件：同一版次、无 OK 的连续 MISSING/GAP 流。

    * ``gap_left`` / ``gap_right``：组件两端尚未被 MISSING 核封死的边缘 GAP；
    * ``core``：缺测核，首尾必为 MISSING，核内可含已验证的短缺报（``Core.gaps``）；
    * 纯 GAP 组件没有核。
    跨块接缝若切在核内某条 GAP 中间，:func:`merge` 会把它当作重新打开的
    边缘 GAP 重新判定，必要时沿接缝把核切成两个段。
    """

    start: int
    end: int
    version: int
    max_gap: int
    gap_left: int
    gap_right: int
    core: Optional[Core]
    cut_left: bool = False
    cut_right: bool = False
    # 当核内末/首条 GAP 可能被块边界切开（核一直延伸到块界）时，记录该
    # GAP 区间。它同时已被计入 gap_right/gap_left 参与接缝配对；若到根化
    # 都没有接缝消费（单块或带在块内闭合），finalize 按完整 core 出段。
    reopened_right: Optional[Interval] = None
    reopened_left: Optional[Interval] = None

    @property
    def missing(self) -> int:
        return self.core.missing if self.core is not None else 0

    @staticmethod
    def missing_run(start: int, end: int, rule: Rule) -> "SoftComponent":
        return SoftComponent(
            start=start, end=end, version=rule.version, max_gap=rule.max_gap,
            gap_left=0, gap_right=0,
            core=Core(start, end, end - start, ()),
        )

    @staticmethod
    def gap_run(start: int, end: int, rule: Rule) -> "SoftComponent":
        g = end - start
        return SoftComponent(
            start=start, end=end, version=rule.version, max_gap=rule.max_gap,
            gap_left=g, gap_right=g, core=None,
        )


@dataclass(frozen=True)
class Summary:
    """一个区域的压缩摘要。``start/end`` 为区域里程范围。"""

    start: int
    end: int
    prefix: Optional[SoftComponent] = None
    suffix: Optional[SoftComponent] = None
    closed: tuple = field(default_factory=tuple)  # tuple[Segment, ...]

    # ------------------------------------------------------------------ #
    # 叶子构造：把块内区间折叠成自洽摘要（构造块时使用）
    # ------------------------------------------------------------------ #
    @staticmethod
    def from_intervals(start: int, end: int, intervals, rule: Rule) -> "Summary":
        """直接扫描块区间构造自洽摘要。

        扫描产出一串“软组件”，相邻软组件之间隔着硬切口（OK 或块内即可
        判定的超长 GAP）。块首/块尾若为硬切口，则对应 prefix/suffix 为
        None（接缝端没有可向外连接的软组件）；中间组件的核进入 closed。
        最后对首尾软组件做块界 GAP 重开复核。
        """
        comps: List[SoftComponent] = []
        cur: Optional[SoftComponent] = None
        leading_hard = False
        trailing_hard = False

        def feed_gap(iv: Interval, hard: bool):
            nonlocal cur, trailing_hard
            g = SoftComponent.gap_run(iv.start, iv.end, rule)
            if hard:
                if cur is not None:
                    # 超长 GAP 是硬切口：封死左侧软组件（打 cut_right）。
                    comps.append(replace(cur, cut_right=True))
                    cur = None
                comps.append(g)  # 无核纯 GAP 组件，不产生段
                trailing_hard = True
            elif cur is None:
                cur = g
                trailing_hard = False
            else:
                cur = _merge_soft(cur, g, iv.length, iv.start)
                trailing_hard = False

        def feed_missing(iv: Interval):
            nonlocal cur, trailing_hard
            m = SoftComponent.missing_run(iv.start, iv.end, rule)
            if cur is None:
                cur = m
            else:
                cur = _merge_soft(cur, m, 0, iv.start)
            trailing_hard = False

        for iv in intervals:
            if iv.state is State.OK:
                if cur is not None:
                    # OK 是硬切口：封死当前软组件，打 cut_right 标记。
                    comps.append(replace(cur, cut_right=True))
                    cur = None
                if not comps:
                    leading_hard = True
                trailing_hard = True
            elif iv.state is State.GAP:
                feed_gap(iv, hard=iv.length > rule.max_gap)
            else:
                feed_missing(iv)
        if cur is not None:
            comps.append(cur)

        if not comps:
            # 整块都是 OK（或无软组件）：纯硬切口摘要。
            return Summary(start, end)

        comps = [_reopen_edge_gaps(c, start, end) for c in comps]
        # 最左/最右软组件作为 prefix/suffix。块首/块尾硬切口给边缘组件打
        # cut 标记（仅表示朝块界一侧封死，不立即固化，保留跨块接缝连接）。
        # 单组件块若块首/块尾是硬切口，prefix/suffix 是同一逻辑组件的两个
        # 视图（cut 标记不同），用 _same_comp 的范围判等识别；多组件块两个
        # 边缘范围互斥，不会误判。
        prefix = comps[0]
        suffix = comps[-1]
        if leading_hard:
            prefix = replace(prefix, cut_left=True)
            if suffix is comps[0]:
                suffix = prefix
        if trailing_hard:
            suffix = replace(suffix, cut_right=True)
            if prefix is comps[-1]:
                prefix = suffix
        # 只有“确定内闭”的段在叶子直接入 closed：
        #  * 中间组件（comps[1:-1]）：两侧都是块内硬切口；
        #  * 双 cut 边缘组件（cut_left 且 cut_right）：块首块尾均硬切口。
        # 单侧 cut 的边缘组件仍可能跨块接缝连接，不在叶子固化，交给 merge
        # 依据实际接缝状态决定，避免在信息不足时误判内闭。
        closed_list = [
            Segment(c.core.start, c.core.end, c.core.missing, c.version)
            for c in comps[1:-1] if c.core is not None
        ]
        for edge in (prefix, suffix):
            if (
                edge is not None and edge.core is not None
                and edge.cut_left and edge.cut_right
            ):
                seg = Segment(edge.core.start, edge.core.end,
                              edge.core.missing, edge.version)
                if seg not in closed_list:
                    closed_list.append(seg)
        closed_list.sort(key=lambda s: s.start)
        return Summary(
            start, end,
            prefix=prefix,
            suffix=suffix,
            closed=tuple(closed_list),
        )

    @staticmethod
    def hard(start: int, end: int) -> "Summary":
        """纯硬切口区域（OK、超长 GAP 或零长度占位）。"""
        return Summary(start=start, end=end)

    # ------------------------------------------------------------------ #
    # 根化：全线（或完整查询上下文）摘要转成最终段列表
    # ------------------------------------------------------------------ #
    def finalize(self) -> tuple:
        """返回按里程升序、互不重叠的 :class:`Segment` 元组。"""
        out = list(self.closed)
        seen = {(s.start, s.end, s.version) for s in self.closed}

        def emit(c: Optional[SoftComponent]):
            if c is None or c.core is None:
                return
            g_r = c.reopened_right
            g_l = c.reopened_left
            # 仅当重开 GAP 已被移出核（贴界情形）且没有接缝消费时，才恢复
            # 为完整段；核内 GAP（不贴界）本就保留在 core.gaps，直接出段。
            if g_r is not None and g_r not in c.core.gaps:
                full = Core(c.core.start, c.end, c.core.missing,
                            c.core.gaps + (g_r,))
                seg = Segment(full.start, full.end, full.missing, c.version)
            elif g_l is not None and g_l not in c.core.gaps:
                full = Core(c.start, c.core.end, c.core.missing,
                            (g_l,) + c.core.gaps)
                seg = Segment(full.start, full.end, full.missing, c.version)
            else:
                seg = Segment(c.core.start, c.core.end,
                              c.core.missing, c.version)
            key = (seg.start, seg.end, seg.version)
            if key not in seen:  # 已被硬切口固化到 closed 的不重复发射
                seen.add(key)
                out.append(seg)

        # 双侧都被硬切口封死的边缘组件（块首/块尾均为硬切口，无任何跨块
        # 接缝可连接）是内闭段，正常发射；单侧 cut 的边缘组件也由 emit
        # 处理——其 cut 标记只在 merge 中阻断对应方向，根化时核仍是合法段。
        emit(self.prefix)
        if not _same_comp(self.prefix, self.suffix):
            emit(self.suffix)
        out.sort(key=lambda s: s.start)
        return tuple(out)


def _same_comp(x: Optional[SoftComponent], y: Optional[SoftComponent]) -> bool:
    """是否为同一个逻辑软组件（区域内组件范围互斥，范围相等即同一组件）。

    单组件块在块首/块尾硬切口下，prefix/suffix 是同一逻辑组件的两个视图
    （cut 标记不同、对象不同但范围相同），因此这里按范围判等；多组件块的
    边缘组件范围互斥，不会误判。
    """
    return (
        x is not None
        and y is not None
        and x.start == y.start
        and x.end == y.end
    )


def _reopen_edge_gaps(c: Optional[SoftComponent],
                      block_start: int, block_end: int) -> Optional[SoftComponent]:
    """块界复核：标记可能被块边界切开的核内 GAP。

    两类候选都记录到 reopened 标记，并把带长放到边缘 GAP 参与接缝配对：

    1. GAP 区间本身贴块界：核在 GAP 缘截断（GAP 移出核）；
    2. 核内最后/第一条 GAP 之后/之前的 MISSING 一直延伸到块界：带可能在
       邻块延续，核保留完整信息，仅额外标记，根化时按完整核出段。
    """
    if c is None or c.core is None:
        return c
    core = c.core
    gap_left, gap_right = c.gap_left, c.gap_right
    reopened_right = c.reopened_right
    reopened_left = c.reopened_left

    # 右界重开有两种情形（都要求核内最后一条 GAP 被 MISSING 夹在核中）：
    #  (a) GAP 本身贴块右界（g.end == c.end）：接缝直接切带，核截断、GAP
    #      移出核并暴露为边缘 GAP，参与接缝处整条带长的配对判定；
    #  (b) GAP 后还有同核 MISSING 延伸到块右界（g.end < c.end）：接缝切在
    #      MISSING 中。该 GAP 与右邻块边缘 GAP 隔着 MISSING，是两条独立带，
    #      不参与接缝配对；这里仅保留重开探针，供根化 finalize 恢复完整核。
    if (
        c.end == block_end
        and gap_right == 0
        and core.gaps
        and core.gaps[-1].state is State.GAP
        and core.end == c.end
    ):
        g = core.gaps[-1]
        if g.end == block_end:
            # case a：GAP 本身贴块右界，接缝直接切带——移出核并暴露为
            # 边缘 GAP，交由接缝配对判定整条带长。
            reopened_right = g
            gap_right = g.end - g.start
            kept = core.gaps[:-1]
            kept_missing = (g.start - core.start) - sum(
                iv.end - iv.start for iv in kept)
            core = Core(core.start, g.start, kept_missing, kept)
        else:
            # case b：GAP 后还有 MISSING 延伸到块右界。仅设置重开探针
            # （核保持完整），根化时按完整核出段；不参与接缝 GAP 配对。
            reopened_right = g

    # 左界重开（对称）。
    if (
        c.start == block_start
        and gap_left == 0
        and core.gaps
        and core.gaps[0].state is State.GAP
        and core.start == c.start
    ):
        g = core.gaps[0]
        if g.start == block_start:
            # case a：GAP 本身贴块左界，移出核暴露为边缘 GAP。
            reopened_left = g
            gap_left = g.end - g.start
            kept = core.gaps[1:]
            kept_missing = (core.end - g.end) - sum(
                iv.end - iv.start for iv in kept)
            core = Core(g.end, core.end, kept_missing, kept)
        else:
            # case b：GAP 前的 MISSING 贴块左界，只设探针，核保持完整。
            reopened_left = g

    return replace(c, core=core, gap_left=gap_left, gap_right=gap_right,
                   reopened_right=reopened_right,
                   reopened_left=reopened_left)


# ---------------------------------------------------------------------- #
# 合并：相邻区域 A（左）与 B（右），常数时间
# ---------------------------------------------------------------------- #
def _junction_gap(ca: SoftComponent, cb: SoftComponent, seam: int) -> int:
    """接缝处被切开的连续 GAP 带全长；不是 GAP 接缝时返回 0。

    只有真正贴接缝的边缘 GAP（``gap_right`` / ``gap_left``）才可能属于
    同一条被块界切开的带：两侧都探测到 GAP 才配对，长度相加。核内、与接缝
    隔着 MISSING 的 GAP 是独立缺报带，不参与跨块拼接。
    """
    rlen = _right_edge_gap(ca, seam)
    llen = _left_edge_gap(cb, seam)
    if rlen == 0 or llen == 0:
        return 0
    return rlen + llen


def _right_edge_gap(c: SoftComponent, seam: int) -> int:
    """组件右端贴接缝的边缘 GAP 长度；无边缘 GAP 返回 0。"""
    return c.gap_right


def _left_edge_gap(c: SoftComponent, seam: int) -> int:
    """组件左端贴接缝的边缘 GAP 长度；无边缘 GAP 返回 0。"""
    return c.gap_left


def merge(a: Summary, b: Summary) -> Summary:
    if a.end != b.start:
        raise ValueError(f"摘要接缝不连续: [{a.start},{a.end}) 与 [{b.start},{b.end})")
    if a.start == a.end:
        return Summary(b.start, b.end, b.prefix, b.suffix, closed=b.closed)
    if b.start == b.end:
        return Summary(a.start, a.end, a.prefix, a.suffix, closed=a.closed)

    closed = a.closed + b.closed
    ca, cb = a.suffix, b.prefix
    seam = a.end

    # 边缘组件的 cut 标记只表示朝某一侧被块内硬切口封死。处理原则：
    #   cut_right 只阻断组件“向右”的软连接，cut_left 只阻断“向左”；
    #   组件朝另一方向仍可能跨块接缝连接，因此这里只调整接缝槽位、绝不
    #   立即固化核。真正的固化只发生在组件两侧都被封死（内闭）时，由叶子
    #   的双侧 cut 规则或下方 None 分支的对侧硬切口判定统一处理。
    ca_cut_right = ca is not None and ca.cut_right
    cb_cut_left = cb is not None and cb.cut_left

    # 非活动的内闭边缘：左区域 prefix 若与 suffix 不同组件且带 cut_right，
    # 说明它右侧已被块内硬切口封死、无法经任何接缝向右连接，是确定内闭段，
    # 在每次 merge 幂等补固化（不依赖当前接缝是否硬切口）。右区域 suffix
    # 带 cut_left 的对称情形同理。
    if (
        a.prefix is not None and a.prefix.core is not None
        and not _same_comp(a.prefix, a.suffix) and a.prefix.cut_right
    ):
        closed = _emit_once(closed, a.prefix)
    if (
        b.suffix is not None and b.suffix.core is not None
        and not _same_comp(b.prefix, b.suffix) and b.suffix.cut_left
    ):
        closed = _emit_once(closed, b.suffix)

    if ca is not None and ca.cut_right:
        # ca 右端是块内硬切口，与右侧被切口隔开：内闭固化（幂等）。
        # 若 ca 同时是 prefix（单组件块），其左端仍可能已向左连接，但向右
        # 已封死，核在此收尾入 closed；prefix 槽置空避免 finalize 重复。
        closed = _emit_once(closed, ca)
        if _same_comp(a.prefix, a.suffix):
            a = Summary(a.start, a.end, None, None, closed=a.closed)
        ca = None
    if cb is not None and cb.cut_left:
        # cb 左端是块内硬切口，与左侧被切口隔开。
        #  - 带核且非唯一组件：cb 确为内闭段，核固化；
        #  - 带核的单组件块：还要向右连接，仅清 prefix 槽、保留 suffix；
        #  - 无核纯 GAP：不产生段，直接清 prefix 槽阻断向左连接。
        if cb.core is not None and not _same_comp(b.prefix, b.suffix):
            closed = _emit_once(closed, cb)
        else:
            b = Summary(b.start, b.end, None, b.suffix, closed=b.closed)
        cb = None
    # cb 朝右（内侧）被封死、且左侧 ca 也是硬切口（无法向左连接）时，
    # cb 两侧均无软连接，才是内闭段，固化。ca 活动则 cb 仍可向左软合并。
    if (
        ca is None
        and cb is not None and cb.cut_right and cb.core is not None
        and not _same_comp(b.prefix, b.suffix)
    ):
        closed = _emit_once(closed, cb)
        cb = None
    # ca 朝左（内侧）被封死、右侧 cb 为硬切口的对称情形。
    if (
        cb is None
        and ca is not None and ca.cut_left and ca.core is not None
        and not _same_comp(a.prefix, a.suffix)
    ):
        closed = _emit_once(closed, ca)
        ca = None

    if ca is None or cb is None:
        # 接缝处至少一侧是硬切口。
        extra = ()
        new_prefix = a.prefix
        new_suffix = b.suffix
        if cb is None and ca is not None:
            # 右块块首硬切口：左活动后缀 ca 被封死（幂等，避免重复入 closed）。
            if not _already_emitted(closed, ca):
                extra = _emit(ca)
            if _same_comp(a.prefix, ca):
                new_prefix = None
        elif ca is None and cb is not None:
            # 左块块尾硬切口：右块的接缝前缀 cb 不与左侧连接，但右块自身的
            # 最右软组件 b.suffix 才是合并后区域的后缀（cb 可能只是其前缀）。
            new_suffix = b.suffix
        # ca 因 cut_right 被置空、且右块块首也是硬切口（cb 为 None）：
        # 该后缀核两侧均无软连接，是真内闭段，固化到 closed。
        if ca_cut_right and cb is None and a.suffix is not None \
                and a.suffix.core is not None:
            extra = extra + tuple(
                s for s in _emit(a.suffix) if not _already_emitted(closed, a.suffix)
            )
        # cb 因 cut_left 被置空、且左块块尾也是硬切口（ca 为 None）：对称。
        if cb_cut_left and ca is None and b.prefix is not None \
                and b.prefix.core is not None:
            extra = extra + tuple(
                s for s in _emit(b.prefix) if not _already_emitted(closed, b.prefix)
            )
        # 左区域的 prefix 若是独立于 suffix 的带核组件，且其右端已被块内
        # 硬切口封死（cut_right），它在更早接缝尚未固化，这里补一次（幂等）。
        if (
            a.prefix is not None
            and a.prefix.core is not None
            and not _same_comp(a.prefix, a.suffix)
            and a.prefix.cut_right
            and not _already_emitted(closed, a.prefix)
        ):
            extra = extra + _emit(a.prefix)
        return Summary(
            a.start, b.end, new_prefix, new_suffix,
            closed=closed + extra,
        )

    # 接缝处被切开的连续 GAP 带全长；MISSING 零距离贴合等情形返回 0。
    # 注意：任一侧的纯 GAP 组件若自身已超长（叶子已判为硬切口），它不是
    # 等待拼接判定的边缘短缺报，接缝按硬切口处理，不能把两段长度相加。
    ca_hard_gap = ca.core is None and ca.gap_right > ca.max_gap
    cb_hard_gap = cb.core is None and cb.gap_left > cb.max_gap
    if not ca_hard_gap and not cb_hard_gap:
        junction = _junction_gap(ca, cb, seam)
    else:
        junction = 0
    version_cut = ca.version != cb.version
    long_gap_cut = junction > ca.max_gap or ca_hard_gap or cb_hard_gap

    if version_cut:
        # 换版硬切口：即使两侧都缺测也不跨，组件原样封死。
        new_prefix, extra_l = _seal_right(a, ca, mark=True, closed=closed)
        new_suffix, extra_r = _seal_left(b, cb, mark=True, closed=closed)
        return Summary(
            a.start, b.end, new_prefix, new_suffix,
            closed=closed + extra_l + extra_r,
        )

    if long_gap_cut:
        if ca_hard_gap or cb_hard_gap:
            # 一侧纯 GAP 组件自身已超长：它就是硬切口，按普通硬切口封死，
            # 保留左右区域各自的 prefix/suffix，不产生跨切口的 GAP 残留。
            new_prefix, extra_l = _seal_right(a, ca, mark=True, closed=closed)
            new_suffix, extra_r = _seal_left(b, cb, mark=True, closed=closed)
            return Summary(
                a.start, b.end, new_prefix, new_suffix,
                closed=closed + extra_l + extra_r,
            )
        # 接缝处整条（短）GAP 带拼接后超长：沿接缝切断。若接缝切在某个核
        # 的内部 GAP 中间，需要把该核拆成两段，切口两侧的纯 GAP 残留作为
        # 新区域的 suffix/prefix 保留。
        new_prefix, extra_l, left_gap_residue = _split_right(a, ca, seam)
        new_suffix, extra_r, right_gap_residue = _split_left(b, cb, seam)
        # 残留是无核纯 GAP，不能顶替真正的带核 suffix/prefix。
        #  - 右区域单逻辑组件时，右残留可作为新 prefix；否则保留其原 prefix。
        #  - 左区域单逻辑组件（无独立带核 prefix）时，左残留作新 suffix；
        #    否则 suffix 取右区域的活动组件（单组件取 new_suffix，多组件取
        #    b.suffix），避免用左 GAP 残留覆盖右区域内容。
        if right_gap_residue is not None and _same_comp(b.prefix, b.suffix):
            prefix = right_gap_residue
        else:
            prefix = new_prefix
        a_single = _same_comp(a.prefix, a.suffix)
        if right_gap_residue is not None:
            # 右残留贴新接缝右侧，是合并后区域最靠右的软组件，suffix 取它
            # （new_suffix 即右残留）；左残留只在没有右残留时才兜底。
            suffix = new_suffix
        elif left_gap_residue is not None and a_single:
            suffix = left_gap_residue
        else:
            suffix = new_suffix if _same_comp(b.prefix, b.suffix) else b.suffix
        return Summary(
            a.start, b.end, prefix, suffix,
            closed=closed + extra_l + extra_r,
        )

    # 防御：若 ca/cb 因上游接线异常在接缝处不连续，不能软合并，按硬切口
    # 封死处理（各自保留，活动核经 finalize/closed 幂等发射），保证不崩溃。
    if ca.end != seam or cb.start != seam:
        new_prefix, extra_l = _seal_right(a, ca, mark=True, closed=closed)
        new_suffix, extra_r = _seal_left(b, cb, mark=True, closed=closed)
        return Summary(
            a.start, b.end, new_prefix, new_suffix,
            closed=closed + extra_l + extra_r,
        )

    merged = _merge_soft(ca, cb, junction, seam)
    # 软合并把左后缀 ca 与右前缀 cb 接成 merged。
    new_prefix = _wire_prefix(a, override=merged)
    if merged.core is None:
        if _same_comp(b.prefix, b.suffix):
            # 右区域单组件：merged（即便无核）已承接 cb 并贴新右边界，
            # 必须作为后缀；返回 b.suffix 会退回未合并的旧范围。
            new_suffix = merged
        else:
            # 右区域还有更右的独立带核组件：无核 merged 不承载段，后缀
            # 让给右区域自己的活动 suffix，避免丢失其边缘/内闭段。
            new_suffix = b.suffix
    elif _same_comp(b.prefix, b.suffix):
        # 右区域单组件且 merged 带核：直接成为后缀。
        new_suffix = merged
    else:
        # 右区域还有更右的独立组件。merged 朝右被硬切口封死则内闭固化，
        # 后缀让给右区域自己的活动 suffix。
        if merged.cut_right and merged.core is not None:
            closed = _emit_once(closed, merged)
        new_suffix = b.suffix
    return Summary(
        a.start, b.end, new_prefix, new_suffix,
        closed=closed,
    )


def _split_right(a: Summary, ca: SoftComponent, seam: int):
    """超长 GAP 切口在左区域后缀 ``ca`` 右端的处理。

    返回 ``(新前缀, 新增段, 切口处纯GAP残留或None)``。接缝切在核内 GAP
    中间时，核在该 GAP 左缘截断固化为段，GAP 作为无核纯 GAP 组件留在
    区域右端（不产生段，作为合并后区域的后缀候选）。
    """
    if ca.gap_right > 0:
        g_start = seam - ca.gap_right
        if ca.reopened_right is not None and ca.reopened_right.end < seam:
            g_start = ca.reopened_right.start  # 带后还隔着尾部 MISSING
        residue = SoftComponent(
            g_start, seam, ca.version, ca.max_gap,
            gap_left=seam - g_start, gap_right=seam - g_start,
            core=None, cut_left=True,
        )
        if ca.core is None:
            # 纯 GAP 组件：没有核可固化，只留残留。
            if _same_comp(a.prefix, ca):
                return residue, (), residue
            return a.prefix, (), residue
        # 有核：段尾截到重开 GAP 的左缘，GAP 及其后尾部 MISSING 都归切口侧。
        if ca.reopened_right is not None:
            seg_end = ca.reopened_right.start
            kept_gaps = tuple(
                g for g in ca.core.gaps if g.end <= seg_end
            )
            kept_missing = (seg_end - ca.core.start) - sum(
                g.end - g.start for g in kept_gaps)
            seg_core = Core(ca.core.start, seg_end, kept_missing, kept_gaps)
        else:
            seg_core = ca.core
        seg = Segment(seg_core.start, seg_core.end,
                      seg_core.missing, ca.version)
        if _same_comp(a.prefix, ca):
            head = replace(ca, end=seg_core.end, core=seg_core,
                           gap_right=0, reopened_right=None, cut_right=True)
            return head, (seg,), residue
        return a.prefix, (seg,), residue

    inside_gap = (
        ca.core is not None
        and ca.core.gaps
        and ca.core.gaps[-1].end == seam
    )
    if not inside_gap:
        # 防御性兜底：正常流程中长带切口只发生在贴接缝 GAP（inside_gap 成立）。
        new_prefix, extra = _seal_right(a, ca, mark=True)
        return new_prefix, extra, None

    gap_iv = ca.core.gaps[-1]
    left_core = Core(ca.core.start, gap_iv.start,
                     ca.core.missing, ca.core.gaps[:-1])
    seg = Segment(left_core.start, left_core.end,
                  left_core.missing, ca.version)
    residue = SoftComponent(
        gap_iv.start, seam, ca.version, ca.max_gap,
        gap_left=seam - gap_iv.start,
        gap_right=seam - gap_iv.start,
        core=None, cut_left=True,
    )
    if _same_comp(a.prefix, ca):
        # 单组件块：右侧超长 GAP 被切走（residue），GAP 左侧的核保留为活动
        # prefix 继续向左连接，不固化 seg，避免与后续合并段重复。
        head = replace(ca, end=gap_iv.start, core=left_core,
                       gap_right=0, cut_right=True)
        return head, (), residue
    return a.prefix, (seg,), residue


def _split_left(b: Summary, cb: SoftComponent, seam: int):
    """超长 GAP 切口在右区域前缀 ``cb`` 左端的对称处理。

    返回 ``(新后缀, 新增段, 切口处纯GAP残留或None)``。
    """
    if cb.gap_left > 0:
        g_end = seam + cb.gap_left
        if cb.reopened_left is not None and cb.reopened_left.start > seam:
            g_end = cb.reopened_left.end
        residue = SoftComponent(
            seam, g_end, cb.version, cb.max_gap,
            gap_left=g_end - seam, gap_right=g_end - seam,
            core=None, cut_right=True,
        )
        if cb.core is None:
            if _same_comp(b.suffix, cb):
                return residue, (), residue
            return b.suffix, (), residue
        # 段头从重开 GAP 的右缘开始，GAP 及其前导 MISSING 归切口侧。
        if cb.reopened_left is not None:
            seg_start = cb.reopened_left.end
            kept_gaps = tuple(
                g for g in cb.core.gaps if g.start >= seg_start
            )
            kept_missing = (cb.core.end - seg_start) - sum(
                g.end - g.start for g in kept_gaps)
            seg_core = Core(seg_start, cb.core.end, kept_missing, kept_gaps)
        else:
            seg_core = cb.core
        seg = Segment(seg_core.start, seg_core.end,
                      seg_core.missing, cb.version)
        if _same_comp(b.suffix, cb):
            # 单组件块：GAP 左侧的超长带被切走（residue），但 GAP 右侧的核
            # 仍是活动 suffix，要继续向右连接后续 MISSING——不能把 seg 固化
            # 进 closed（否则与后续合并段重复）。只返回截断后的 tail。
            tail = replace(cb, start=seg_core.start, core=seg_core,
                           gap_left=0, reopened_left=None, cut_left=True)
            return tail, (), residue
        return b.suffix, (seg,), residue

    inside_gap = (
        cb.core is not None
        and cb.core.gaps
        and cb.core.gaps[0].start == seam
    )
    if not inside_gap:
        # 防御性兜底：正常流程中长带切口只发生在贴接缝 GAP（inside_gap 成立）。
        new_suffix, extra = _seal_left(b, cb, mark=True)
        return new_suffix, extra, None

    gap_iv = cb.core.gaps[0]
    right_core = Core(gap_iv.end, cb.core.end,
                      cb.core.missing, cb.core.gaps[1:])
    seg = Segment(right_core.start, right_core.end,
                  right_core.missing, cb.version)
    residue = SoftComponent(
        seam, gap_iv.end, cb.version, cb.max_gap,
        gap_left=gap_iv.end - seam,
        gap_right=gap_iv.end - seam,
        core=None, cut_right=True,
    )
    if _same_comp(b.suffix, cb):
        tail = replace(cb, start=gap_iv.end, core=right_core,
                       gap_left=0, cut_left=True)
        return tail, (seg,), residue
    return b.suffix, (seg,), residue


def _merge_soft(x: SoftComponent, y: SoftComponent,
                junction: int, seam: int) -> SoftComponent:
    """合并两个同版次、接缝处非超长 GAP（或零距离）的软组件。

    ``junction`` 为接缝处整条连续 GAP 带长度（调用方已保证不超阈值）。
    两侧都有缺测核时，这条带作为可跨过的内部短缺报并入核；只有一侧有核
    时，它仍是新核外缘上等待下一次拼接判定的边缘 GAP。
    """
    assert x.version == y.version and x.end == y.start
    if x.core is None and y.core is None:
        # 纯 GAP + 纯 GAP：整条带长度相加，仍无核。
        total = x.gap_left + y.gap_left
        return SoftComponent(
            x.start, y.end, x.version, x.max_gap,
            gap_left=total, gap_right=total, core=None,
            cut_left=x.cut_left, cut_right=y.cut_right,
        )
    if x.core is None:
        # 左侧纯 GAP：保留为新核的前导边缘 GAP（可能贴块左界、等待拼接判定）。
        return SoftComponent(
            x.start, y.end, x.version, x.max_gap,
            gap_left=x.gap_left + y.gap_left,
            gap_right=y.gap_right, core=y.core,
            cut_left=x.cut_left, cut_right=y.cut_right,
        )
    if y.core is None:
        # 右侧纯 GAP：成为后随边缘 GAP，等待后续拼接判定，核暂不延伸。
        return SoftComponent(
            x.start, y.end, x.version, x.max_gap,
            gap_left=x.gap_left, gap_right=x.gap_right + y.gap_right,
            core=x.core,
            cut_left=x.cut_left, cut_right=y.cut_right,
        )
    # 两侧都有核：接缝短缺报验证通过，并入核内证据，MISSING 数相加。
    gaps = tuple(x.core.gaps)
    if junction > 0:
        gaps += (Interval(seam - _right_edge_gap(x, seam),
                          seam + _left_edge_gap(y, seam), State.GAP),)
    gaps += tuple(y.core.gaps)
    # 若 x 还带着后随边缘 GAP（叶子折叠中“核-GAP-核”的中间形态），
    # 此刻 y 的核已经接上，这条 GAP 验证通过，收入核内证据。
    if x.gap_right > 0 and junction == 0:
        gaps = tuple(x.core.gaps) + (
            Interval(x.end - x.gap_right, x.end, State.GAP),
        ) + tuple(y.core.gaps)
    core = Core(x.core.start, y.core.end,
                x.core.missing + y.core.missing, gaps)
    # 边缘重开标记沿合并方向传播：左标记来自 x，右标记来自 y。
    # 接缝处被消费的重开 GAP（junction>0）清除对应侧标记。
    reopened_left = x.reopened_left
    reopened_right = y.reopened_right
    if junction > 0 and x.reopened_right is not None:
        reopened_right = y.reopened_right  # x 的右标记已在接缝消费
    # 两核软合并后接缝被填满：y 朝接缝的 cut_left、x 朝接缝的 cut_right 都
    # 失效。合并体只保留背向接缝、贴外边缘的 x.cut_left 与 y.cut_right。
    return SoftComponent(
        x.start, y.end, x.version, x.max_gap,
        gap_left=x.gap_left, gap_right=y.gap_right, core=core,
        cut_left=x.cut_left, cut_right=y.cut_right,
        reopened_right=reopened_right,
        reopened_left=reopened_left,
    )


def _emit(c: Optional[SoftComponent]) -> tuple:
    """把一个在接缝端被硬切口封死的软组件中的核固化为段。"""
    if c is None or c.core is None:
        return ()
    return (Segment(c.core.start, c.core.end, c.core.missing, c.version),)


def _emit_once(closed: tuple, c: Optional[SoftComponent]) -> tuple:
    """幂等发射：核已在 closed 中则返回原 closed，否则追加。"""
    if c is None or c.core is None or _already_emitted(closed, c):
        return closed
    return closed + _emit(c)


def _already_emitted(closed: tuple, c: SoftComponent) -> bool:
    """该组件的核是否已经存在于 closed（幂等保护，避免重复发射）。"""
    if c.core is None:
        return True
    return any(
        s.start == c.core.start and s.end == c.core.end and s.version == c.version
        for s in closed
    )


def _seal_right(a: Summary, ca: Optional[SoftComponent], mark: bool = False,
                closed: tuple = ()):
    """左区域后缀在其右端被硬切口封死（发射经 closed 幂等去重）。"""
    if ca is None:
        return a.prefix, ()
    if _same_comp(a.prefix, ca):
        return (replace(ca, cut_right=True) if mark else ca), ()
    return a.prefix, (_emit(ca) if not _already_emitted(closed, ca) else ())


def _seal_left(b: Summary, cb: Optional[SoftComponent], mark: bool = False,
               closed: tuple = ()):
    """右区域前缀在其左端被硬切口封死（对称、幂等）。"""
    if cb is None:
        return b.suffix, ()
    if _same_comp(b.suffix, cb):
        return (replace(cb, cut_left=True) if mark else cb), ()
    return b.suffix, (_emit(cb) if not _already_emitted(closed, cb) else ())


def _wire_prefix(a: Summary, override: Optional[SoftComponent] = None) -> Optional[SoftComponent]:
    """合并后新区域的前缀：左区域若只有一个软组件，则它被替换为合并体。"""
    if a.prefix is None:
        return None
    if override is not None and _same_comp(a.prefix, a.suffix):
        return override
    return a.prefix


def merge_all(summaries) -> Summary:
    """按顺序折叠任意多个相邻摘要（空序列非法，全线至少有一个块）。"""
    it = iter(summaries)
    try:
        acc = next(it)
    except StopIteration:
        raise ValueError("至少需要一个块摘要")
    for s in it:
        acc = merge(acc, s)
    return acc
