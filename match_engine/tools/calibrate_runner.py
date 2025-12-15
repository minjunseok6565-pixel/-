from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from match_engine.era import get_era_targets
from match_engine.models import Player, TeamState
from match_engine.sim import simulate_game
from match_engine.tactics import TacticsConfig

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


@dataclass
class GameMetrics:
    team: str
    opponent: str
    metrics: Dict[str, float]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def random_player(rng: random.Random, pid: str) -> Player:
    keys = [
        "FIN_RIM",
        "FIN_DUNK",
        "FIN_CONTACT",
        "SHOT_MID_CS",
        "SHOT_3_CS",
        "SHOT_FT",
        "SHOT_MID_PU",
        "SHOT_3_OD",
        "SHOT_TOUCH",
        "POST_SCORE",
        "POST_CONTROL",
        "SEAL_POWER",
        "DRIVE_CREATE",
        "HANDLE_SAFE",
        "FIRST_STEP",
        "PASS_SAFE",
        "PASS_CREATE",
        "PNR_READ",
        "SHORTROLL_PLAY",
        "DEF_POA",
        "DEF_HELP",
        "DEF_STEAL",
        "DEF_RIM",
        "DEF_POST",
        "REB_OR",
        "REB_DR",
        "PHYSICAL",
        "ENDURANCE",
    ]
    base = {k: _clamp(rng.gauss(55, 12), 25, 95) for k in keys}
    pos = rng.choice(["G", "F", "C"])
    return Player(pid=pid, name=pid, pos=pos, derived=base)


def make_random_team(rng: random.Random, name: str) -> TeamState:
    players = [random_player(rng, f"{name}_{i}") for i in range(10)]
    roles = {
        "ball_handler": players[0].pid,
        "secondary_handler": players[1].pid,
        "screener": players[4].pid,
        "post": players[4].pid,
        "shooter": players[1].pid,
        "cutter": players[2].pid,
        "rim_runner": players[4].pid,
    }
    tac = TacticsConfig()
    return TeamState(name=name, lineup=players, roles=roles, tactics=tac)


def compute_game_metrics(res: Dict[str, Any]) -> List[GameMetrics]:
    teams = list(res.get("teams", {}).keys())
    out: List[GameMetrics] = []
    for i, t in enumerate(teams):
        opp = teams[1 - i] if len(teams) > 1 else None
        team_dat = res["teams"][t]
        opp_dat = res["teams"].get(opp, {}) if opp else {}
        poss = float(team_dat.get("Possessions") or res.get("possessions_per_team") or 0)
        fga = float(team_dat.get("FGA", 0))
        fta = float(team_dat.get("FTA", 0))
        tpa = float(team_dat.get("3PA", 0))
        tov = float(team_dat.get("TOV", 0))
        pts = float(team_dat.get("PTS", 0))
        orb = float(team_dat.get("ORB", 0))
        opp_drb = float(opp_dat.get("DRB", 0)) if opp_dat else 0.0
        zones = team_dat.get("ShotZones", {}) or {}
        rim = float(zones.get("rim", 0))
        mid = float(zones.get("mid", 0))
        three = float(zones.get("3", 0))

        metrics = {
            "pace": poss,
            "ortg": (pts / poss * 100.0) if poss else 0.0,
            "tov_pct": (tov / poss) if poss else 0.0,
            "three_rate": (tpa / fga) if fga else 0.0,
            "ftr": (fta / fga) if fga else 0.0,
            "orb_pct": (orb / (orb + opp_drb)) if (orb + opp_drb) else 0.0,
            "shot_share_rim": (rim / fga) if fga else 0.0,
            "shot_share_mid": (mid / fga) if fga else 0.0,
            "shot_share_three": (three / fga) if fga else 0.0,
            "corner3_share": None,
        }
        out.append(GameMetrics(team=t, opponent=opp or "", metrics=metrics))
    return out


def aggregate_metrics(samples: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    if not samples:
        return summary
    for key in samples[0].keys():
        vals = [s.get(key, 0.0) for s in samples if s.get(key) is not None]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        summary[key] = {
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "p95": vals_sorted[int(0.95 * (len(vals_sorted) - 1))] if vals_sorted else 0.0,
            "p05": vals_sorted[int(0.05 * (len(vals_sorted) - 1))] if vals_sorted else 0.0,
        }
    return summary


def run_batch(games: int, seed: int, era: str, roster_mode: str, roster_file: str | None = None) -> Tuple[List[GameMetrics], Dict[str, Dict[str, float]]]:
    rng = random.Random(seed)
    metrics: List[GameMetrics] = []
    for gi in range(games):
        grng = random.Random(seed + gi)
        teamA = make_random_team(grng, f"TeamA_{gi}")
        teamB = make_random_team(grng, f"TeamB_{gi}")
        res = simulate_game(grng, teamA, teamB, era=era, strict_validation=False)
        for gm in compute_game_metrics(res):
            metrics.append(gm)
    summary = aggregate_metrics([m.metrics for m in metrics])
    return metrics, summary


def save_report(out_dir: str, summary: Dict[str, Any], per_game: List[GameMetrics], seed: int, games: int, era: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "calibrate_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "games": games, "era": era, "summary": summary}, f, indent=2)

    csv_path = os.path.join(out_dir, "calibrate_games.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header_written = False
        for gm in per_game:
            row = {"team": gm.team, "opponent": gm.opponent, **gm.metrics}
            if not header_written:
                writer.writerow(list(row.keys()))
                header_written = True
            writer.writerow(list(row.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Match engine calibration runner")
    parser.add_argument("--games", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--era", type=str, default="era_modern_nbaish_v1")
    parser.add_argument("--roster_mode", type=str, default="random", choices=["random", "file"])
    parser.add_argument("--roster_file", type=str, default=None)
    parser.add_argument("--out", type=str, default=REPORT_DIR)
    args = parser.parse_args()

    per_game, summary = run_batch(args.games, args.seed, args.era, args.roster_mode, args.roster_file)
    save_report(args.out, summary, per_game, args.seed, args.games, args.era)

    targets = get_era_targets(args.era)
    print(json.dumps({"summary": summary, "targets": targets}, indent=2))


if __name__ == "__main__":
    main()
