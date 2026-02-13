from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

# ---- schema shim (runner-only) ----
def _ensure_schema_module() -> None:
    try:
        import schema  # type: ignore
        _ = schema.GameContext  # type: ignore[attr-defined]
        _ = schema.normalize_team_id  # type: ignore[attr-defined]
        return
    except Exception:
        pass

    import types
    m = types.ModuleType("schema")

    def normalize_team_id(x: str) -> str:
        return str(x or "").strip()

    @dataclass
    class GameContext:
        game_id: str
        home_team_id: str
        away_team_id: str

    m.normalize_team_id = normalize_team_id  # type: ignore[attr-defined]
    m.GameContext = GameContext  # type: ignore[attr-defined]
    sys.modules["schema"] = m

_ensure_schema_module()
import schema  # type: ignore

from ..sim_game import simulate_game
from ..era import load_era_config
from ..game_config import build_game_config

from ..calibration.generate import OFFENSE_SCHEMES, DEFENSE_SCHEMES  # type: ignore
from ..calibration.aggregate import StatsAccumulator, safe_div

from .generate import generate_balanced_roster, build_team_from_roster_and_schemes
from .schedule import make_schedule, Match
from .report import (
    summarize_alerts,
    build_matchup_alerts,
    compute_scheme_rankings,
    compute_baseline_deltas,
    compute_matchup_extremes,
    compute_effect_decomposition,
    wilson_ci95,
)


Combo = Tuple[str, str]  # (off, def)


def _team_to_metrics(team_summary: Dict[str, Any]) -> Dict[str, Any]:
    keep = dict(team_summary)
    keep.pop("Players", None)
    keep.pop("PlayerBox", None)
    # Drop verbose per-game breakdowns from the calibration2 summary JSON
    for k in ("ShotZoneDetail", "OffActionCounts", "OutcomeCounts", "AvgFatigue"):
        keep.pop(k, None)
    return keep


def _combo_id(c: Combo) -> str:
    return f"{c[0]}__{c[1]}"

def _round_json_numbers(x: Any, ndigits: int = 2) -> Any:
    """Recursively round floats in JSON-able structures."""
    if isinstance(x, float):
        return round(x, ndigits)
    if isinstance(x, dict):
        return {k: _round_json_numbers(v, ndigits) for k, v in x.items()}
    if isinstance(x, list):
        return [_round_json_numbers(v, ndigits) for v in x]
    if isinstance(x, tuple):
        return [_round_json_numbers(v, ndigits) for v in x]
    return x

def run_calibration2(
    *,
    seed: int,
    era: str,
    mode: str,
    n_rosters: int,
    legs: int,
    k_opponents: int,
    knobs: str,
    knobs_sd: float,
    baseline_off: str,
    baseline_def: str,
    off_schemes: Optional[List[str]] = None,
    def_schemes: Optional[List[str]] = None,
    strict_validation: bool = False,
    replay_disabled: bool = True,
) -> Dict[str, Any]:
    rng_master = random.Random(int(seed))

    era_cfg, _, _ = load_era_config(era)
    game_cfg = build_game_config(era_cfg)
    allowed_off = set(getattr(game_cfg, "off_scheme_action_weights", {}).keys()) or set(OFFENSE_SCHEMES)
    allowed_def = set(getattr(game_cfg, "defense_scheme_mult", {}).keys()) or set(DEFENSE_SCHEMES)

    offs = [s for s in (off_schemes or list(OFFENSE_SCHEMES)) if s in allowed_off]
    defs = [s for s in (def_schemes or list(DEFENSE_SCHEMES)) if s in allowed_def]
    combos: List[Combo] = [(o, d) for o in offs for d in defs]
    combos.sort()

    baseline: Combo = (baseline_off if baseline_off in offs else offs[0], baseline_def if baseline_def in defs else defs[0])

    # Output accumulators
    combo_acc: Dict[str, StatsAccumulator] = { _combo_id(c): StatsAccumulator() for c in combos }
    combo_wl: Dict[str, Dict[str, float]] = {
        _combo_id(c): {
            "games": 0,
            "wins": 0,
            "pts_for": 0.0,
            "pts_against": 0.0,
            "poss": 0.0,
            # for per-team-game net_rating std
            "nr_sum": 0.0,
            "nr2_sum": 0.0,
        }
        for c in combos
    }

    matchup_wl: Dict[str, Dict[str, Dict[str, float]]] = {}  # a->b->rec (only filled for full_matrix)

    def add_matchup(a_id: str, b_id: str, win: int, pts_for: float, pts_against: float, poss: float) -> None:
        if a_id not in matchup_wl:
            matchup_wl[a_id] = {}
        if b_id not in matchup_wl[a_id]:
            matchup_wl[a_id][b_id] = {"games": 0, "wins": 0, "pts_for": 0.0, "pts_against": 0.0, "poss": 0.0}
        rec = matchup_wl[a_id][b_id]
        rec["games"] += 1
        rec["wins"] += int(win)
        rec["pts_for"] += float(pts_for)
        rec["pts_against"] += float(pts_against)
        rec["poss"] += float(poss)

    # For each roster seed, run a schedule
    total_games = 0
    for r in range(int(n_rosters)):
        roster_rng = random.Random(int(seed) + 10000 + r)
        base_players = generate_balanced_roster(roster_rng, roster_id=f"R{r:02d}", name_prefix=f"R{r:02d}")

        sched_rng = random.Random(int(seed) + 20000 + r)
        matches: List[Match] = make_schedule(
            sched_rng,
            combos=combos,
            mode=mode,
            legs=int(legs),
            k_opponents=int(k_opponents),
            baseline=baseline,
        )

        for mi, m in enumerate(matches):
            # leg parity decides home/away swap
            a, b = m.a, m.b
            if (m.leg % 2) == 0:
                home_c, away_c = a, b
            else:
                home_c, away_c = b, a

            home_id = f"H_R{r:02d}_M{mi:05d}"
            away_id = f"A_R{r:02d}_M{mi:05d}"

            # team build rng: stable per roster+combo+match+leg
            team_rng_h = random.Random(int(seed) + 30000 + r * 100000 + mi * 2 + 0)
            team_rng_a = random.Random(int(seed) + 30000 + r * 100000 + mi * 2 + 1)

            home_team, meta_h = build_team_from_roster_and_schemes(
                team_rng_h,
                base_players=base_players,
                team_id=home_id,
                name=f"{home_c[0]}_{home_c[1]}",
                offense_scheme=home_c[0],
                defense_scheme=home_c[1],
                knobs_mode=knobs,
                knobs_sd=knobs_sd,
            )
            away_team, meta_a = build_team_from_roster_and_schemes(
                team_rng_a,
                base_players=base_players,
                team_id=away_id,
                name=f"{away_c[0]}_{away_c[1]}",
                offense_scheme=away_c[0],
                defense_scheme=away_c[1],
                knobs_mode=knobs,
                knobs_sd=knobs_sd,
            )

            ctx = schema.GameContext(
                game_id=f"CAL2_{seed}_R{r}_M{mi}_L{m.leg}",
                home_team_id=home_id,
                away_team_id=away_id,
            )

            game_rng = random.Random(int(seed) + 40000 + r * 100000 + mi * 10 + m.leg)
            result = simulate_game(
                game_rng,
                home_team,
                away_team,
                context=ctx,
                era=era,
                strict_validation=bool(strict_validation),
                replay_disabled=bool(replay_disabled),
            )

            scores = (result.get("game_state", {}) or {}).get("scores", {}) or {}
            sh = int(scores.get(home_id, 0))
            sa = int(scores.get(away_id, 0))

            # team summaries
            teams = result.get("teams", {}) or {}
            summ_h = _team_to_metrics(teams.get(home_id, {}) or {})
            summ_a = _team_to_metrics(teams.get(away_id, {}) or {})

            poss_h = float(summ_h.get("Possessions", result.get("possessions_per_team", 0) or 0))
            poss_a = float(summ_a.get("Possessions", result.get("possessions_per_team", 0) or 0))
            poss = float(max(poss_h, poss_a, 1e-6))

            # Update combo stats for both teams
            id_home = _combo_id(home_c)
            id_away = _combo_id(away_c)

            # WL
            home_win = 1 if sh > sa else 0
            away_win = 1 if sa > sh else 0

            for cid, win, pf, pa, summ in [
                (id_home, home_win, sh, sa, summ_h),
                (id_away, away_win, sa, sh, summ_a),
            ]:
                combo_wl[cid]["games"] += 1
                combo_wl[cid]["wins"] += int(win)
                combo_wl[cid]["pts_for"] += float(pf)
                combo_wl[cid]["pts_against"] += float(pa)
                combo_wl[cid]["poss"] += float(poss)
                nr_game = safe_div((float(pf) - float(pa)), poss) * 100.0
                combo_wl[cid]["nr_sum"] += float(nr_game)
                combo_wl[cid]["nr2_sum"] += float(nr_game) * float(nr_game)
                combo_acc[cid].add(summ)

            if mode == "full_matrix":
                add_matchup(id_home, id_away, home_win, sh, sa, poss)
                add_matchup(id_away, id_home, away_win, sa, sh, poss)

            total_games += 1

    # Build output
    combos_out: Dict[str, Any] = {}
    for cid, wl in combo_wl.items():
        g = int(wl["games"])
        w = int(wl["wins"])
        pf = float(wl["pts_for"])
        pa = float(wl["pts_against"])
        poss = float(wl["poss"])
        win_pct = safe_div(w, g)
        net_rating = safe_div((pf - pa), poss) * 100.0
        nr_mean = safe_div(float(wl.get("nr_sum", 0.0)), g)
        nr2_mean = safe_div(float(wl.get("nr2_sum", 0.0)), g)
        nr_var = max(0.0, float(nr2_mean) - float(nr_mean) * float(nr_mean))
        nr_std = nr_var ** 0.5

        combos_out[cid] = {
            "games": g,
            "wins": w,
            "losses": g - w,
            "win_pct": win_pct,
            "win_ci95": wilson_ci95(w, g),
            "pts_for": pf,
            "pts_against": pa,
            "poss": poss,
            "net_rating": net_rating,
            "net_rating_std": float(nr_std),
            "avg_team_game": combo_acc[cid].mean(),
        }

    matchups_out: Dict[str, Any] = {}
    if mode == "full_matrix":
        for a, vs in matchup_wl.items():
            matchups_out[a] = {}
            for b, rec in vs.items():
                g = int(rec["games"])
                w = int(rec["wins"])
                pf = float(rec["pts_for"])
                pa = float(rec["pts_against"])
                poss = float(rec["poss"])
                matchups_out[a][b] = {
                    "games": g,
                    "wins": w,
                    "losses": g - w,
                    "win_pct": safe_div(w, g),
                    "net_rating": safe_div((pf - pa), poss) * 100.0,
                }

    out: Dict[str, Any] = {
        "meta": {
            "seed": int(seed),
            "era": str(era),
            "mode": str(mode),
            "n_rosters": int(n_rosters),
            "legs": int(legs),
            "k_opponents": int(k_opponents),
            "knobs": str(knobs),
            "knobs_sd": float(knobs_sd),
            "baseline": {"offense": baseline[0], "defense": baseline[1]},
            "total_games": int(total_games),
            "replay_disabled": bool(replay_disabled),
            "strict_validation": bool(strict_validation),
        },
        "schemes": {
            "offense": list(offs),
            "defense": list(defs),
            "n_combos": len(combos),
        },
        "combos": combos_out,
    }

    if matchups_out:
        out["matchups"] = matchups_out

    # ---- derived aggregates (scheme-level + meta diagnostics) ----
    baseline_id = _combo_id(baseline)
    derived: Dict[str, Any] = {}
    derived["scheme_rankings"] = compute_scheme_rankings(
        combos_out,
        min_games=max(10, int(n_rosters) * int(legs)),
    )
    derived["baseline_deltas"] = compute_baseline_deltas(combos_out, baseline_id)
    if matchups_out:
        derived["matchup_extremes"] = compute_matchup_extremes(
            matchups_out,
            min_games=max(4, int(n_rosters)),
            strong_edge_nr=5.0,
        )
    derived["effect_decomposition"] = compute_effect_decomposition(
        combos_out,
        min_games=max(10, int(n_rosters) * int(legs)),
        residual_flag_z=2.5,
    )
    out["derived"] = derived

    # alerts (human-friendly)
    out["alerts"] = {
        "combo": summarize_alerts(combos_out, min_games=max(10, int(n_rosters) * int(legs))),
    }
    if matchups_out:
        out["alerts"]["matchup"] = build_matchup_alerts(matchups_out, min_games=max(4, int(n_rosters)), top_n=10)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibration2: scheme combo balance harness")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--era", type=str, default="default")

    ap.add_argument("--mode", type=str, default="swiss", choices=["swiss", "vs_baseline", "full_matrix"])
    ap.add_argument("--n_rosters", type=int, default=8)
    ap.add_argument("--legs", type=int, default=2, help="Number of legs per pairing (2 = home/away swap).")
    ap.add_argument("--k_opponents", type=int, default=12, help="Swiss: opponents per combo.")
    ap.add_argument("--knobs", type=str, default="pure", choices=["pure", "variation"])
    ap.add_argument("--knobs_sd", type=float, default=0.03)

    ap.add_argument("--baseline_off", type=str, default="Spread_HeavyPnR")
    ap.add_argument("--baseline_def", type=str, default="Drop")

    ap.add_argument("--off_schemes", type=str, default="", help="Comma-separated offense scheme allowlist.")
    ap.add_argument("--def_schemes", type=str, default="", help="Comma-separated defense scheme allowlist.")

    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--out", type=str, default="calibration2_output.json")

    args = ap.parse_args()

    off_list = [x.strip() for x in str(args.off_schemes).split(",") if x.strip()] or None
    def_list = [x.strip() for x in str(args.def_schemes).split(",") if x.strip()] or None

    res = run_calibration2(
        seed=args.seed,
        era=args.era,
        mode=args.mode,
        n_rosters=args.n_rosters,
        legs=args.legs,
        k_opponents=args.k_opponents,
        knobs=args.knobs,
        knobs_sd=args.knobs_sd,
        baseline_off=args.baseline_off,
        baseline_def=args.baseline_def,
        off_schemes=off_list,
        def_schemes=def_list,
        strict_validation=args.strict,
        replay_disabled=(not args.replay),
    )
 
    res = _round_json_numbers(res, 2)
    with open(str(args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(str(args.out))


if __name__ == "__main__":
    main()
