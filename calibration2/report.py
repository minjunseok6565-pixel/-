from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple, Optional

from ..calibration.aggregate import safe_div

def combo_key(off: str, de: str) -> str:
    return f"{off}__{de}"


def wilson_ci95(wins: int, games: int) -> Dict[str, float]:
    """95% Wilson score interval for a binomial proportion.

    Returns dict with keys: low, high.
    """
    n = int(games)
    if n <= 0:
        return {"low": 0.0, "high": 1.0}
    z = 1.96
    phat = float(wins) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = phat + (z * z) / (2.0 * float(n))
    rad = z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * float(n))) / float(n))
    low = (center - rad) / denom
    high = (center + rad) / denom
    return {"low": float(max(0.0, low)), "high": float(min(1.0, high))}


def compute_scheme_rankings(
    combos: Dict[str, Any],
    *,
    min_games: int = 20,
) -> Dict[str, Any]:
    """Aggregate combo results into offense-only and defense-only rankings.

    - win% uses summed wins/games
    - net_rating uses games-weighted mean
    """
    off_acc: Dict[str, Dict[str, float]] = {}
    def_acc: Dict[str, Dict[str, float]] = {}

    for cid, v in combos.items():
        g = int(v.get("games", 0))
        if g < min_games:
            continue
        wins = int(v.get("wins", 0))
        nr = float(v.get("net_rating", 0.0))
        off, de = cid.split("__", 1)

        if off not in off_acc:
            off_acc[off] = {"games": 0.0, "wins": 0.0, "nr_wsum": 0.0}
        if de not in def_acc:
            def_acc[de] = {"games": 0.0, "wins": 0.0, "nr_wsum": 0.0}

        off_acc[off]["games"] += g
        off_acc[off]["wins"] += wins
        off_acc[off]["nr_wsum"] += nr * g

        def_acc[de]["games"] += g
        def_acc[de]["wins"] += wins
        def_acc[de]["nr_wsum"] += nr * g

    offense_rows: List[Dict[str, Any]] = []
    for off, a in off_acc.items():
        g = int(a["games"])
        w = int(a["wins"])
        win_pct = safe_div(w, g)
        net_rating = safe_div(a["nr_wsum"], a["games"])
        offense_rows.append({
            "offense_scheme": off,
            "games": g,
            "wins": w,
            "losses": g - w,
            "win_pct": win_pct,
            "win_ci95": wilson_ci95(w, g),
            "net_rating": float(net_rating),
        })
    defense_rows: List[Dict[str, Any]] = []
    for de, a in def_acc.items():
        g = int(a["games"])
        w = int(a["wins"])
        win_pct = safe_div(w, g)
        net_rating = safe_div(a["nr_wsum"], a["games"])
        defense_rows.append({
            "defense_scheme": de,
            "games": g,
            "wins": w,
            "losses": g - w,
            "win_pct": win_pct,
            "win_ci95": wilson_ci95(w, g),
            "net_rating": float(net_rating),
        })

    offense_rows.sort(key=lambda r: (r["net_rating"], r["win_pct"]), reverse=True)
    for i, r in enumerate(offense_rows, start=1):
        r["rank_net_rating"] = i
    offense_rows.sort(key=lambda r: (r["win_pct"], r["net_rating"]), reverse=True)
    for i, r in enumerate(offense_rows, start=1):
        r["rank_win_pct"] = i

    defense_rows.sort(key=lambda r: (r["net_rating"], r["win_pct"]), reverse=True)
    for i, r in enumerate(defense_rows, start=1):
        r["rank_net_rating"] = i
    defense_rows.sort(key=lambda r: (r["win_pct"], r["net_rating"]), reverse=True)
    for i, r in enumerate(defense_rows, start=1):
        r["rank_win_pct"] = i

    return {
        "offense": offense_rows,
        "defense": defense_rows,
    }


def compute_baseline_deltas(combos: Dict[str, Any], baseline_combo_id: str) -> Dict[str, Any]:
    base = combos.get(baseline_combo_id) or {}
    base_wp = float(base.get("win_pct", 0.0))
    base_nr = float(base.get("net_rating", 0.0))
    out: Dict[str, Any] = {}
    for cid, v in combos.items():
        out[cid] = {
            "delta_win_pct": float(v.get("win_pct", 0.0)) - base_wp,
            "delta_net_rating": float(v.get("net_rating", 0.0)) - base_nr,
        }
    return {
        "baseline_combo": baseline_combo_id,
        "baseline_win_pct": base_wp,
        "baseline_net_rating": base_nr,
        "deltas": out,
    }


def compute_matchup_extremes(
    matchups: Dict[str, Any],
    *,
    min_games: int = 4,
    strong_edge_nr: float = 5.0,
) -> Dict[str, Any]:
    """For each combo A, compute best/worst matchup netRtg and spread.

    Also count free_wins (nr >= strong_edge_nr) and hard_counters (nr <= -strong_edge_nr).
    """
    by_combo: Dict[str, Any] = {}
    for a, vs in matchups.items():
        if not isinstance(vs, dict):
            continue
        best: Optional[Tuple[float, str]] = None
        worst: Optional[Tuple[float, str]] = None
        free_wins = 0
        hard_counters = 0
        for b, rec in vs.items():
            if not isinstance(rec, dict):
                continue
            g = int(rec.get("games", 0))
            if g < min_games:
                continue
            nr = float(rec.get("net_rating", 0.0))
            if (best is None) or (nr > best[0]):
                best = (nr, b)
            if (worst is None) or (nr < worst[0]):
                worst = (nr, b)
            if nr >= strong_edge_nr:
                free_wins += 1
            if nr <= -strong_edge_nr:
                hard_counters += 1

        if best is None or worst is None:
            continue
        by_combo[a] = {
            "best_matchup": {"opponent": best[1], "net_rating": float(best[0])},
            "worst_matchup": {"opponent": worst[1], "net_rating": float(worst[0])},
            "matchup_spread": float(best[0] - worst[0]),
            "free_wins_count": int(free_wins),
            "hard_counters_count": int(hard_counters),
        }

    # Global top lists
    spread_rows = [(v["matchup_spread"], a) for a, v in by_combo.items()]
    spread_rows.sort(reverse=True)
    top_spread = [
        {"combo": a, **by_combo[a]} for _, a in spread_rows[:10]
    ]
    return {
        "by_combo": by_combo,
        "top_matchup_spread": top_spread,
        "params": {"min_games": int(min_games), "strong_edge_nr": float(strong_edge_nr)},
    }


def compute_effect_decomposition(
    combos: Dict[str, Any],
    *,
    min_games: int = 20,
    residual_flag_z: float = 2.5,
) -> Dict[str, Any]:
    """Decompose combo net_rating into grand + off_effect + def_effect + interaction(residual).

    Uses games-weighted means.
    """
    rows: List[Tuple[str, str, float, int]] = []
    for cid, v in combos.items():
        g = int(v.get("games", 0))
        if g < min_games:
            continue
        nr = float(v.get("net_rating", 0.0))
        off, de = cid.split("__", 1)
        rows.append((off, de, nr, g))

    if not rows:
        return {"grand_mean": 0.0, "off_effect": {}, "def_effect": {}, "interactions": []}

    total_w = sum(g for *_rest, g in rows)
    grand = safe_div(sum(nr * g for _o, _d, nr, g in rows), total_w)

    off_sum: Dict[str, float] = {}
    off_w: Dict[str, int] = {}
    def_sum: Dict[str, float] = {}
    def_w: Dict[str, int] = {}

    for o, d, nr, g in rows:
        off_sum[o] = off_sum.get(o, 0.0) + nr * g
        off_w[o] = off_w.get(o, 0) + g
        def_sum[d] = def_sum.get(d, 0.0) + nr * g
        def_w[d] = def_w.get(d, 0) + g

    off_eff = {o: float(safe_div(off_sum[o], off_w[o]) - grand) for o in off_sum}
    def_eff = {d: float(safe_div(def_sum[d], def_w[d]) - grand) for d in def_sum}

    # Residuals
    residuals: List[Tuple[float, str, str, int]] = []
    for o, d, nr, g in rows:
        pred = grand + off_eff.get(o, 0.0) + def_eff.get(d, 0.0)
        res = float(nr - pred)
        residuals.append((res, o, d, g))

    # Weighted residual std
    res_mean = safe_div(sum(res * g for res, _o, _d, g in residuals), sum(g for res, _o, _d, g in residuals))
    res_var = safe_div(
        sum(((res - res_mean) ** 2) * g for res, _o, _d, g in residuals),
        sum(g for res, _o, _d, g in residuals),
    )
    res_std = math.sqrt(max(0.0, float(res_var)))

    flagged: List[Dict[str, Any]] = []
    if res_std > 1e-9:
        for res, o, d, g in residuals:
            z = float((res - res_mean) / res_std)
            if abs(z) >= float(residual_flag_z):
                flagged.append({
                    "combo": combo_key(o, d),
                    "offense": o,
                    "defense": d,
                    "residual": float(res),
                    "z": float(z),
                    "games": int(g),
                })
    flagged.sort(key=lambda x: abs(float(x.get("z", 0.0))), reverse=True)

    return {
        "grand_mean": float(grand),
        "off_effect": off_eff,
        "def_effect": def_eff,
        "residual_mean": float(res_mean),
        "residual_std": float(res_std),
        "flagged_interactions": flagged[:20],
        "params": {"min_games": int(min_games), "residual_flag_z": float(residual_flag_z)},
    }

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
