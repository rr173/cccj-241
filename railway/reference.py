"""小范围逐格参考实现，用于对任意分块做差分核对。

仅适用于测试规模（线路长度有限）：把整条线显式展开成逐格数组，
用最直白的规则重算硬切口、复测段与窗口证据，与幺半群引擎的结果逐格比对。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .model import BlockArtifact, EvidenceRun, Rule, Segment, State


def cells_from_blocks(
    blocks: Sequence[BlockArtifact], line_start: int, line_end: int
) -> Tuple[List[State], List[int]]:
    """把块产物展开成逐格状态/版次数组（仅测试用，规模受限）。"""
    n = line_end - line_start
    states: List[State] = [None] * n  # type: ignore[list-item]
    versions: List[int] = [-1] * n
    for b in sorted(blocks, key=lambda x: x.start):
        assert b.start >= line_start and b.end <= line_end
        for iv in b.intervals:  # 参考实现允许逐格读
            for x in range(iv.start, iv.end):
                i = x - line_start
                assert states[i] is None, "参考侧发现块重叠"
                states[i] = iv.state
                versions[i] = b.rule.version
    assert all(s is not None for s in states), "参考侧发现缺块"
    return states, versions


def reference_segments(
    states: List[State],
    versions: List[int],
    rules: Dict[int, Rule],
    line_start: int,
) -> Tuple[Segment, ...]:
    """逐格判定硬切口并产出复测段候选。"""
    n = len(states)
    crossable = [False] * n
    i = 0
    while i < n:
        if states[i] is not State.GAP:
            i += 1
            continue
        j = i
        v = versions[i]
        while j < n and states[j] is State.GAP and versions[j] == v:
            j += 1
        band_len = j - i
        ok = band_len <= rules[v].max_gap
        for k in range(i, j):
            crossable[k] = ok
        i = j

    def hard_after(k: int) -> bool:
        """单元格 k 与 k+1 之间是否为硬切口。"""
        if states[k] is State.OK or states[k + 1] is State.OK:
            return True
        if versions[k] != versions[k + 1]:
            return True  # 换版硬切口，即使两侧都缺测也不跨
        if states[k] is State.GAP and not crossable[k]:
            return True  # 超长缺报硬切口
        return False

    segments: List[Segment] = []
    region_lo = 0
    for k in range(n - 1):
        if hard_after(k):
            _emit_region(states, versions, region_lo, k + 1, line_start, segments)
            region_lo = k + 1
    _emit_region(states, versions, region_lo, n, line_start, segments)
    return tuple(segments)


def _emit_region(states, versions, lo, hi, line_start, out):
    first = next((x for x in range(lo, hi) if states[x] is State.MISSING), None)
    if first is None:
        return  # 纯 GAP/OK 边缘带不形成复测段
    last = next(
        x for x in range(hi - 1, lo - 1, -1) if states[x] is State.MISSING
    )
    missing = sum(1 for x in range(first, last + 1) if states[x] is State.MISSING)
    v = versions[first]
    out.append(
        Segment(
            line_start + first,
            line_start + last + 1,
            missing,
            v,
        )
    )


def reference_window(
    states: List[State],
    versions: List[int],
    rules: Dict[int, Rule],
    line_start: int,
    segments: Tuple[Segment, ...],
    start: int,
    end: int,
) -> Tuple[dict, ...]:
    """逐格裁切窗口证据，并按全线段判门槛。"""
    hits = []
    for seg in segments:
        if seg.end <= start or seg.start >= end:
            continue
        clip_lo, clip_hi = max(seg.start, start), min(seg.end, end)
        evidence: List[EvidenceRun] = []
        clip_missing = 0
        x = clip_lo
        while x < clip_hi:
            i = x - line_start
            j = i + 1
            while (
                line_start + j < clip_hi
                and states[j] == states[i]
                and versions[j] == versions[i]
            ):
                j += 1
            run_lo, run_hi = x, line_start + j
            st = states[i]
            if st is State.MISSING:
                clip_missing += run_hi - run_lo
                evidence.append(
                    EvidenceRun(run_lo, run_hi, State.MISSING, False, None)
                )
            elif st is State.GAP:
                # 求整条 GAP 带长度（参考侧逐格扫描）。
                a = i
                while a > 0 and states[a - 1] is State.GAP and versions[a - 1] == versions[i]:
                    a -= 1
                b = i
                while b + 1 < len(states) and states[b + 1] is State.GAP and versions[b + 1] == versions[i]:
                    b += 1
                band = b - a + 1
                evidence.append(
                    EvidenceRun(
                        run_lo,
                        run_hi,
                        State.GAP,
                        band <= rules[versions[i]].max_gap,
                        band,
                    )
                )
            else:
                evidence.append(EvidenceRun(run_lo, run_hi, State.OK, False, None))
            x = run_hi
        hits.append(
            {
                "segment": seg,
                "qualified": seg.missing >= rules[seg.version].min_missing,
                "clip": (clip_lo, clip_hi),
                "clip_missing": clip_missing,
                "evidence": tuple(evidence),
            }
        )
    return tuple(hits)
