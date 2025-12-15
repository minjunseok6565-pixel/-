from __future__ import annotations

import random

from .core import clamp
from .era import get_mvp_rules
from .models import Player, TeamState
from .sim import simulate_game
from .tactics import TacticsConfig

# -------------------------
# Demo (sample derived stats)
# -------------------------

def make_sample_player(rng: random.Random, pid: str, name: str, archetype: str) -> Player:
    keys = [
        "FIN_RIM","FIN_DUNK","FIN_CONTACT","SHOT_MID_CS","SHOT_3_CS","SHOT_FT","SHOT_MID_PU","SHOT_3_OD","SHOT_TOUCH",
        "POST_SCORE","POST_CONTROL","SEAL_POWER",
        "DRIVE_CREATE","HANDLE_SAFE","FIRST_STEP",
        "PASS_SAFE","PASS_CREATE","PNR_READ","SHORTROLL_PLAY",
        "DEF_POA","DEF_HELP","DEF_STEAL","DEF_RIM","DEF_POST",
        "REB_OR","REB_DR","PHYSICAL","ENDURANCE"
    ]
    base = {k: 50.0 for k in keys}

    def bump(ks, lo, hi):
        for k in ks:
            base[k] = clamp(base[k] + rng.uniform(lo, hi), 25, 95)

    if archetype == "PG_SHOOT":
        bump(["SHOT_3_CS","SHOT_3_OD","PASS_CREATE","PASS_SAFE","PNR_READ","HANDLE_SAFE","FIRST_STEP","DRIVE_CREATE"], 12, 25)
        bump(["DEF_POA","ENDURANCE"], 5, 12)
    elif archetype == "WING_3D":
        bump(["SHOT_3_CS","DEF_POA","DEF_HELP","DEF_STEAL","ENDURANCE"], 10, 20)
        bump(["DRIVE_CREATE","HANDLE_SAFE"], 2, 10)
    elif archetype == "BIG_RIM":
        bump(["DEF_RIM","DEF_POST","REB_DR","PHYSICAL","ENDURANCE"], 12, 25)
        bump(["FIN_RIM","FIN_DUNK","FIN_CONTACT","SHORTROLL_PLAY","REB_OR"], 6, 15)
    elif archetype == "BIG_SKILL":
        bump(["SHOT_MID_CS","PASS_SAFE","PASS_CREATE","SHORTROLL_PLAY","POST_SCORE","POST_CONTROL"], 8, 18)
        bump(["DEF_HELP","DEF_POST","ENDURANCE"], 6, 14)
    elif archetype == "SLASH":
        bump(["FIN_RIM","FIN_CONTACT","FIRST_STEP","DRIVE_CREATE","HANDLE_SAFE","ENDURANCE"], 12, 24)
        bump(["SHOT_3_CS"], 0, 10)
    else:
        bump(keys, -5, 10)

    return Player(pid=pid, name=name, derived=base)

def demo(seed: int = 7) -> None:
    rules = get_mvp_rules()

    def run_game(def_scheme: str, label: str) -> None:
        rng = random.Random(seed)

        tA_tac = TacticsConfig(
            offense_scheme="Spread_HeavyPnR",
            defense_scheme="Drop",
            scheme_weight_sharpness=1.10,
            scheme_outcome_strength=1.05,
            def_scheme_weight_sharpness=1.00,
            def_scheme_outcome_strength=1.00,
            action_weight_mult={"PnR":1.15},
            outcome_global_mult={"SHOT_3_CS":1.05},
            outcome_by_action_mult={"PnR":{"PASS_SHORTROLL":1.10}},
            context={"PACE_MULT":1.05}
        )

        tB_tac = TacticsConfig(
            offense_scheme="Drive_Kick",
            defense_scheme=def_scheme,
            scheme_weight_sharpness=1.05,
            scheme_outcome_strength=1.05,
            def_scheme_weight_sharpness=1.05,
            def_scheme_outcome_strength=1.05,
            outcome_global_mult={"PASS_KICKOUT":1.10},
            context={"PACE_MULT":1.02}
        )

        teamA = TeamState(
            name="A_SpreadPnR",
            lineup=[
                make_sample_player(rng,"A1","A1_PG","PG_SHOOT"),
                make_sample_player(rng,"A2","A2_W","WING_3D"),
                make_sample_player(rng,"A3","A3_S","SLASH"),
                make_sample_player(rng,"A4","A4_B","BIG_SKILL"),
                make_sample_player(rng,"A5","A5_C","BIG_RIM"),
            ],
            roles={"ball_handler":"A1","secondary_handler":"A2","screener":"A5","post":"A4","shooter":"A2","cutter":"A3","rim_runner":"A5"},
            tactics=tA_tac
        )

        teamB = TeamState(
            name="B_DriveKick",
            lineup=[
                make_sample_player(rng,"B1","B1_PG","SLASH"),
                make_sample_player(rng,"B2","B2_W","WING_3D"),
                make_sample_player(rng,"B3","B3_W","WING_3D"),
                make_sample_player(rng,"B4","B4_B","BIG_SKILL"),
                make_sample_player(rng,"B5","B5_C","BIG_RIM"),
            ],
            roles={"ball_handler":"B1","secondary_handler":"B2","screener":"B5","post":"B4","shooter":"B2","cutter":"B3","rim_runner":"B5"},
            tactics=tB_tac
        )

        res = simulate_game(rng, teamA, teamB)
        score_a = res["teams"][teamA.name]["PTS"]
        score_b = res["teams"][teamB.name]["PTS"]
        fouls = res.get("game_state", {}).get("team_fouls", {})
        fatigue = res.get("game_state", {}).get("fatigue", {})
        hist = res["teams"][teamA.name]["OffActionCounts"]
        total = sum(hist.values()) or 1
        freq = {k: round(v / total * 100, 2) for k, v in sorted(hist.items(), key=lambda kv: -kv[1])}

        print(f"\n=== Run: {label} (Defense scheme={def_scheme}) ===")
        print(f"Final Score: {teamA.name} {score_a} - {teamB.name} {score_b}")
        print("Team fouls:", fouls)
        sample_fatigue = {pid: round(fatigue.get(pid, 1.0), 3) for pid in list(fatigue.keys())[:4]}
        print("Sample fatigue:", sample_fatigue)
        print("Action frequency (team A offense):", freq)
        print("Possessions per team:", res["possessions_per_team"])

    print(f"Quarter length: {rules['quarter_length']}s | Shot clock: {rules['shot_clock']}s")
    run_game("Drop", "Baseline Drop")
    run_game("Switch_Everything", "Switch Everything")

if __name__ == "__main__":
    demo()
