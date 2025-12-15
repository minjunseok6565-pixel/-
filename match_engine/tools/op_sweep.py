from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

from match_engine.era import get_era_targets
from match_engine.sim import simulate_game
from match_engine.tools.calibrate_runner import aggregate_metrics, compute_game_metrics, make_random_team
from match_engine.tactics import TacticsConfig


EXTREME_RANGE = (0.70, 1.40)


def random_extreme_mult(rng: random.Random) -> float:
    return EXTREME_RANGE[0] if rng.random() < 0.5 else EXTREME_RANGE[1]


def build_extreme_tactics(rng: random.Random) -> TacticsConfig:
    tac = TacticsConfig()
    tac.offense_scheme = rng.choice([
        "Spread_HeavyPnR",
        "Drive_Kick",
        "Transition_Early",
        "Post_InsideOut",
    ])
    tac.defense_scheme = rng.choice([
        "Drop",
        "Switch_Everything",
        "Zone",
        "PackLine_GapHelp",
        "Blitz_TrapPnR",
    ])
    tac.action_weight_mult = {k: random_extreme_mult(rng) for k in ["PnR", "Drive", "TransitionEarly", "PostUp"]}
    tac.outcome_global_mult = {k: random_extreme_mult(rng) for k in ["SHOT_3_CS", "SHOT_RIM_LAYUP", "SHOT_MID_CS"]}
    tac.context = {"PACE_MULT": random_extreme_mult(rng)}
    return tac


def run_sample(seed: int, games: int, era: str) -> Dict[str, float]:
    rng = random.Random(seed)
    metrics: List[Dict[str, float]] = []
    for gi in range(games):
        grng = random.Random(seed + gi)
        ta = make_random_team(grng, f"OP_A_{gi}")
        tb = make_random_team(grng, f"OP_B_{gi}")
        ta.tactics = build_extreme_tactics(grng)
        tb.tactics = build_extreme_tactics(grng)
        res = simulate_game(grng, ta, tb, era=era, strict_validation=False)
        metrics.extend([m.metrics for m in compute_game_metrics(res)])
    return {k: v.get("mean", 0.0) for k, v in aggregate_metrics(metrics).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="OP sweep")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--games_per_sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--era", type=str, default="era_modern_nbaish_v1")
    parser.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "..", "reports"))
    args = parser.parse_args()

    thresholds = get_era_targets(args.era).get("op_thresholds", {})
    flagged: List[Dict[str, Any]] = []
    for idx in range(args.samples):
        metrics = run_sample(args.seed + idx * 100, args.games_per_sample, args.era)
        warn = False
        reasons = []
        if metrics.get("ortg", 0) > thresholds.get("ortg_hi", float("inf")):
            warn = True
            reasons.append("ORtg high")
        if metrics.get("tov_pct", 0) >= thresholds.get("tov_pct_hi", float("inf")):
            warn = True
            reasons.append("TOV% high")
        pace = metrics.get("pace", 0)
        if pace < thresholds.get("pace_lo", -float("inf")) or pace > thresholds.get("pace_hi", float("inf")):
            warn = True
            reasons.append("Pace out of band")
        if warn:
            flagged.append({"sample": idx, "metrics": metrics, "reasons": reasons})
            print(f"WARN sample {idx}: {reasons} -> {metrics}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "op_sweep.json"), "w", encoding="utf-8") as f:
        json.dump({"flagged": flagged}, f, indent=2)


if __name__ == "__main__":
    main()
