from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

from match_engine.era import (
    apply_tunable_updates,
    get_era_targets,
    restore_tunables,
    snapshot_tunables,
)
from match_engine.tools.calibrate_runner import aggregate_metrics, compute_game_metrics, make_random_team
from match_engine.sim import simulate_game


def run_iter(games: int, seed: int, era: str) -> Dict[str, float]:
    metrics: List[Dict[str, float]] = []
    for gi in range(games):
        grng = random.Random(seed + gi)
        teamA = make_random_team(grng, f"TuneA_{gi}")
        teamB = make_random_team(grng, f"TuneB_{gi}")
        res = simulate_game(grng, teamA, teamB, era=era, strict_validation=False)
        metrics.extend([m.metrics for m in compute_game_metrics(res)])
    return {k: v.get("mean", 0.0) for k, v in aggregate_metrics(metrics).items()}


def capped_scale(val: float, mult: float, cap: float = 0.03) -> float:
    mult = max(1.0 - cap, min(1.0 + cap, mult))
    return val * mult


def propose_updates(current: Dict[str, float], targets: Dict[str, Any]) -> Dict[str, float]:
    updates: Dict[str, float] = {}
    tgt = targets.get("targets", {})
    tol = targets.get("tolerances", {})
    # ORtg -> shot base + TO_BASE
    if "ortg" in current and "ortg" in tgt:
        if current["ortg"] > tgt["ortg"] + tol.get("ortg", 0):
            updates["SHOT_BASE_RIM"] = capped_scale(1.0, 0.985)
            updates["SHOT_BASE_MID"] = capped_scale(1.0, 0.985)
            updates["SHOT_BASE_3"] = capped_scale(1.0, 0.985)
            updates["TO_BASE"] = capped_scale(1.0, 1.015)
        elif current["ortg"] < tgt["ortg"] - tol.get("ortg", 0):
            updates["SHOT_BASE_RIM"] = capped_scale(1.0, 1.015)
            updates["SHOT_BASE_MID"] = capped_scale(1.0, 1.015)
            updates["SHOT_BASE_3"] = capped_scale(1.0, 1.015)
            updates["TO_BASE"] = capped_scale(1.0, 0.985)
    # TOV%
    if "tov_pct" in current and "tov_pct" in tgt:
        if current["tov_pct"] > tgt["tov_pct"] + tol.get("tov_pct", 0):
            updates["TO_BASE"] = capped_scale(1.0, 0.98)
            updates["PASS_BASE_SUCCESS_MULT"] = capped_scale(1.0, 1.01)
        elif current["tov_pct"] < tgt["tov_pct"] - tol.get("tov_pct", 0):
            updates["TO_BASE"] = capped_scale(1.0, 1.02)
    # 3PA rate
    if "three_rate" in current and "three_rate" in tgt:
        if current["three_rate"] > tgt["three_rate"] + tol.get("three_rate", 0):
            updates["SHOT_BASE_3"] = capped_scale(1.0, 0.985)
        elif current["three_rate"] < tgt["three_rate"] - tol.get("three_rate", 0):
            updates["SHOT_BASE_3"] = capped_scale(1.0, 1.015)
    # FTr
    if "ftr" in current and "ftr" in tgt:
        if current["ftr"] > tgt["ftr"] + tol.get("ftr", 0):
            updates["FOUL_BASE"] = capped_scale(1.0, 0.985)
        elif current["ftr"] < tgt["ftr"] - tol.get("ftr", 0):
            updates["FOUL_BASE"] = capped_scale(1.0, 1.015)
    # ORB%
    if "orb_pct" in current and "orb_pct" in tgt:
        if current["orb_pct"] > tgt["orb_pct"] + tol.get("orb_pct", 0):
            updates["ORB_BASE"] = capped_scale(1.0, 0.985)
        elif current["orb_pct"] < tgt["orb_pct"] - tol.get("orb_pct", 0):
            updates["ORB_BASE"] = capped_scale(1.0, 1.015)
    return updates


def apply_updates_relative(updates: Dict[str, float]) -> None:
    if not updates:
        return
    cur = snapshot_tunables()
    new_vals: Dict[str, float] = {}
    for k, mult in updates.items():
        if k not in cur:
            continue
        base = cur[k] if isinstance(cur[k], (int, float)) else 1.0
        new_vals[k] = base * mult
    apply_tunable_updates(new_vals)


def within_tolerance(current: Dict[str, float], targets: Dict[str, Any]) -> bool:
    tgt = targets.get("targets", {})
    tol = targets.get("tolerances", {})
    for k, v in tgt.items():
        if k not in current:
            continue
        if abs(current[k] - v) > tol.get(k, 0):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Heuristic tuning loop")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--era", type=str, default="era_modern_nbaish_v1")
    parser.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "..", "reports"))
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()

    targets = get_era_targets(args.era)
    snap = snapshot_tunables()
    history: List[Dict[str, Any]] = []
    stagnant = 0
    try:
        for it in range(args.iters):
            metrics = run_iter(args.games, args.seed + it * 1000, args.era)
            history.append({"iter": it, "metrics": metrics})
            if within_tolerance(metrics, targets):
                break
            updates = propose_updates(metrics, targets)
            if not updates:
                stagnant += 1
                if stagnant >= 3:
                    break
            else:
                stagnant = 0
                apply_updates_relative(updates)
        if args.commit:
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, "tune_loop_final.json"), "w", encoding="utf-8") as f:
                json.dump({"history": history}, f, indent=2)
    finally:
        if not args.commit:
            restore_tunables(snap)

    print(json.dumps({"history": history}, indent=2))


if __name__ == "__main__":
    main()
