from __future__ import annotations

from typing import Any, Dict, List, Tuple
from ..calibration.aggregate import safe_div

def combo_key(off: str, de: str) -> str:
    return f"{off}__{de}"

def summarize_alerts(combos: Dict[str, Any], *, min_games: int = 20) -> Dict[str, Any]:
    rows: List[Tuple[str, float, float, int]] = []
    for k, v in combos.items():
        g = int(v.get("games", 0))
        if g < min_games:
            continue
        wp = float(v.get("win_pct", 0.0))
        nr = float(v.get("net_rating", 0.0))
        rows.append((k, wp, nr, g))

    rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
    top = [{"combo": k, "win_pct": wp, "net_rating": nr, "games": g} for k, wp, nr, g in rows[:10]]

    rows.sort(key=lambda x: (x[1], x[2]))
    bot = [{"combo": k, "win_pct": wp, "net_rating": nr, "games": g} for k, wp, nr, g in rows[:10]]

    return {"top_overperformers": top, "bottom_underperformers": bot}

def build_matchup_alerts(matchups: Dict[str, Any], *, min_games: int = 10, top_n: int = 10) -> Dict[str, Any]:
    # Find extreme matchup edges: A vs B where A win% is very high with enough games.
    edges: List[Tuple[float, str, str, int, float]] = []
    for a, vs in matchups.items():
        if not isinstance(vs, dict):
            continue
        for b, rec in vs.items():
            if not isinstance(rec, dict):
                continue
            g = int(rec.get("games", 0))
            if g < min_games:
                continue
            wp = float(rec.get("win_pct", 0.0))
            nr = float(rec.get("net_rating", 0.0))
            edges.append((wp, a, b, g, nr))

    edges.sort(reverse=True, key=lambda x: (x[0], x[4]))
    top = [{"A": a, "B": b, "A_win_pct": wp, "A_net_rating": nr, "games": g} for wp, a, b, g, nr in edges[:top_n]]
    return {"top_matchup_edges": top}
