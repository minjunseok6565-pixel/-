from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

Combo = Tuple[str, str]  # (off, def)

@dataclass(frozen=True)
class Match:
    a: Combo
    b: Combo
    leg: int  # 0..legs-1 (for home/away swapping)

def make_schedule(
    rng: random.Random,
    *,
    combos: List[Combo],
    mode: str,
    legs: int,
    k_opponents: int = 12,
    baseline: Optional[Combo] = None,
) -> List[Match]:
    mode = str(mode)
    legs = max(1, int(legs))

    if mode == "full_matrix":
        out: List[Match] = []
        for i in range(len(combos)):
            for j in range(i + 1, len(combos)):
                for leg in range(legs):
                    out.append(Match(combos[i], combos[j], leg))
        return out

    if mode == "vs_baseline":
        if baseline is None:
            baseline = combos[0]
        out: List[Match] = []
        for c in combos:
            if c == baseline:
                continue
            for leg in range(legs):
                out.append(Match(c, baseline, leg))
        return out

    # swiss (default): each combo plays k random opponents (undirected de-dup)
    k = max(1, int(k_opponents))
    edges = set()
    for c in combos:
        others = [x for x in combos if x != c]
        rng.shuffle(others)
        for opp in others[:k]:
            a, b = (c, opp) if c <= opp else (opp, c)
            edges.add((a, b))
    out: List[Match] = []
    for a, b in sorted(edges):
        for leg in range(legs):
            out.append(Match(a, b, leg))
    return out
