from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

from .builders import get_action_base
from .core import clamp, dot_profile
from .defense import team_def_snapshot
from .era import DEFAULT_PROB_MODEL, ERA_PROB_MODEL
from .participants import (
    choose_creator_for_pulloff,
    choose_finisher_rim,
    choose_post_target,
    choose_passer,
    choose_shooter_for_mid,
    choose_shooter_for_three,
    choose_weighted_player,
)
from .prob import (
    PASS_BASE_SUCCESS_MULT,
    SHOT_BASE_3,
    SHOT_BASE_MID,
    SHOT_BASE_RIM,
    _shot_kind_from_outcome,
    _team_variance_mult,
    prob_from_scores,
)
from .profiles import OUTCOME_PROFILES, PASS_BASE_SUCCESS, SHOT_BASE
from .models import GameState, Player, ROLE_FALLBACK_RANK, TeamState

ORB_BASE = 1.0
TO_BASE = 1.0
FOUL_BASE = 1.0

# -------------------------
# Rebound / Free throws
# -------------------------

def resolve_free_throws(rng: random.Random, shooter: Player, n: int, team: TeamState) -> None:
    pm = ERA_PROB_MODEL if isinstance(ERA_PROB_MODEL, dict) else DEFAULT_PROB_MODEL
    ft = shooter.get("SHOT_FT")
    p = clamp(
        float(pm.get("ft_base", 0.45)) + (ft / 100.0) * float(pm.get("ft_range", 0.47)),
        float(pm.get("ft_min", 0.40)),
        float(pm.get("ft_max", 0.95)),
    )
    for _ in range(n):
        team.fta += 1
        team.add_player_stat(shooter.pid, "FTA", 1)
        if rng.random() < p:
            team.ftm += 1
            team.pts += 1
            team.add_player_stat(shooter.pid, "FTM", 1)
            team.add_player_stat(shooter.pid, "PTS", 1)

def rebound_orb_probability(offense: TeamState, defense: TeamState, orb_mult: float, drb_mult: float) -> float:
    off_orb = sum(p.get("REB_OR") for p in offense.lineup) / len(offense.lineup)
    def_drb = sum(p.get("REB_DR") for p in defense.lineup) / len(defense.lineup)
    off_orb *= orb_mult
    def_drb *= drb_mult
    base = float(ERA_PROB_MODEL.get('orb_base', 0.26)) * float(ORB_BASE)
    return prob_from_scores(None, base, off_orb, def_drb, kind='rebound', variance_mult=1.0)

def choose_orb_rebounder(rng: random.Random, offense: TeamState) -> Player:
    # top 3 ORB
    cand = sorted(offense.lineup, key=lambda p: p.get("REB_OR") + 0.20*p.get("PHYSICAL"), reverse=True)[:3]
    return choose_weighted_player(rng, cand, "REB_OR", power=1.15)

def choose_drb_rebounder(rng: random.Random, defense: TeamState) -> Player:
    cand = sorted(defense.lineup, key=lambda p: p.get("REB_DR") + 0.20*p.get("PHYSICAL"), reverse=True)[:3]
    return choose_weighted_player(rng, cand, "REB_DR", power=1.10)


# -------------------------
# Outcome helpers
# -------------------------

def is_shot(o: str) -> bool: return o.startswith("SHOT_")
def is_pass(o: str) -> bool: return o.startswith("PASS_")
def is_to(o: str) -> bool: return o.startswith("TO_")
def is_foul(o: str) -> bool: return o.startswith("FOUL_")
def is_reset(o: str) -> bool: return o.startswith("RESET_")


def shot_zone_from_outcome(outcome: str) -> Optional[str]:
    if outcome in ("SHOT_RIM_LAYUP", "SHOT_RIM_DUNK", "SHOT_RIM_CONTACT", "SHOT_TOUCH_FLOATER"):
        return "rim"
    if outcome in ("SHOT_MID_CS", "SHOT_MID_PU"):
        return "mid"
    if outcome in ("SHOT_3_CS", "SHOT_3_OD"):
        return "3"
    return None

def outcome_points(o: str) -> int:
    return 3 if o in ("SHOT_3_CS","SHOT_3_OD") else 2 if o.startswith("SHOT_") else 0


# -------------------------
# Resolve sampled outcome into events
# -------------------------

def resolve_outcome(
    rng: random.Random,
    outcome: str,
    action: str,
    offense: TeamState,
    defense: TeamState,
    tags: Dict[str, Any],
    pass_chain: int,
    ctx: Optional[Dict[str, Any]] = None,
    game_state: Optional[GameState] = None,
) -> Tuple[str, Dict[str, Any]]:
    # count outcome
    offense.outcome_counts[outcome] = offense.outcome_counts.get(outcome, 0) + 1

    if outcome == "TO_SHOTCLOCK":
        actor = offense.get_role_player("ball_handler", ROLE_FALLBACK_RANK["ball_handler"])
        offense.tov += 1
        offense.add_player_stat(actor.pid, "TOV", 1)
        return "TURNOVER", {"outcome": outcome, "pid": actor.pid}

    # role-fit bad outcome logging (internal; only when role-fit was applied on this step)
    try:
        if bool(tags.get("role_fit_applied", False)):
            g = str(tags.get("role_fit_grade", "B"))
            if is_to(outcome):
                offense.role_fit_bad_totals["TO"] = offense.role_fit_bad_totals.get("TO", 0) + 1
                offense.role_fit_bad_by_grade.setdefault(g, {}).setdefault("TO", 0)
                offense.role_fit_bad_by_grade[g]["TO"] += 1
            elif is_reset(outcome):
                offense.role_fit_bad_totals["RESET"] = offense.role_fit_bad_totals.get("RESET", 0) + 1
                offense.role_fit_bad_by_grade.setdefault(g, {}).setdefault("RESET", 0)
                offense.role_fit_bad_by_grade[g]["RESET"] += 1
    except Exception:
        pass


    base_action = get_action_base(action)
    def_snap = team_def_snapshot(defense)
    prof = OUTCOME_PROFILES.get(outcome)
    if not prof:
        return "RESET", {"outcome": outcome}

    # choose participants
    if is_shot(outcome):
        if outcome in ("SHOT_3_CS",):
            actor = choose_shooter_for_three(rng, offense)
        elif outcome in ("SHOT_MID_CS",):
            actor = choose_shooter_for_mid(rng, offense)
        elif outcome in ("SHOT_3_OD","SHOT_MID_PU"):
            actor = choose_creator_for_pulloff(rng, offense, outcome)
        elif outcome == "SHOT_POST":
            actor = choose_post_target(offense)
        elif outcome in ("SHOT_RIM_DUNK",):
            actor = choose_finisher_rim(rng, offense, dunk_bias=True)
        else:
            actor = choose_finisher_rim(rng, offense, dunk_bias=False)
    elif is_pass(outcome):
        actor = choose_passer(rng, offense, base_action, outcome)
    elif is_foul(outcome):
        # foul draw actor: tie to most likely attempt type
        if outcome == "FOUL_DRAW_POST":
            actor = choose_post_target(offense)
        elif outcome == "FOUL_DRAW_JUMPER":
            actor = choose_creator_for_pulloff(rng, offense, "SHOT_3_OD")
        else:
            actor = choose_finisher_rim(rng, offense, dunk_bias=False)
    else:
        actor = offense.get_role_player("ball_handler", ROLE_FALLBACK_RANK["ball_handler"])

    # fatigue cost
    base_cost_off = 0.58 if tags.get("in_transition", False) else 0.42
    base_cost_def = 0.54 if tags.get("in_transition", False) else 0.40
    for p in offense.lineup:
        p.add_fatigue(base_cost_off)
    for p in defense.lineup:
        p.add_fatigue(base_cost_def)

    ctx = ctx or {}
    variance_mult = _team_variance_mult(offense) * float(ctx.get("variance_mult", 1.0))

    # compute scores
    off_vals = {k: actor.get(k) for k in prof["offense"].keys()}
    off_score = dot_profile(off_vals, prof["offense"])
    def_vals = {k: float(def_snap.get(k, 50.0)) for k in prof["defense"].keys()}
    def_score = dot_profile(def_vals, prof["defense"])
    def_score *= float((ctx or {}).get("def_eff_mult", 1.0))

    fatigue_map = (ctx or {}).get("fatigue_map", {}) or {}
    fatigue_logit_max = float((ctx or {}).get("fatigue_logit_max", -0.25))
    fatigue_val = float(fatigue_map.get(actor.pid, 1.0))
    fatigue_logit_delta = (1.0 - fatigue_val) * fatigue_logit_max

    # resolve by type
    if is_shot(outcome):
        base_p = SHOT_BASE.get(outcome, 0.45)
        kind = _shot_kind_from_outcome(outcome)
        if kind == "shot_rim":
            base_p *= float(SHOT_BASE_RIM)
        elif kind == "shot_mid":
            base_p *= float(SHOT_BASE_MID)
        else:
            base_p *= float(SHOT_BASE_3)
        p_make = prob_from_scores(
            rng,
            base_p,
            off_score,
            def_score,
            kind=kind,
            variance_mult=variance_mult,
            logit_delta=float(tags.get('role_logit_delta', 0.0)),
            fatigue_logit_delta=fatigue_logit_delta,
        )
        pts = outcome_points(outcome)

        offense.fga += 1
        zone = shot_zone_from_outcome(outcome)
        if zone:
            offense.shot_zones[zone] = offense.shot_zones.get(zone, 0) + 1
        offense.add_player_stat(actor.pid, "FGA", 1)
        if pts == 3:
            offense.tpa += 1
            offense.add_player_stat(actor.pid, "3PA", 1)

        if rng.random() < p_make:
            offense.fgm += 1
            offense.add_player_stat(actor.pid, "FGM", 1)
            if pts == 3:
                offense.tpm += 1
                offense.add_player_stat(actor.pid, "3PM", 1)
            offense.pts += pts
            offense.add_player_stat(actor.pid, "PTS", pts)
            return "SCORE", {"outcome": outcome, "pid": actor.pid, "points": pts}
        else:
            return "MISS", {"outcome": outcome, "pid": actor.pid, "points": pts}

    if is_pass(outcome):
        base_s = PASS_BASE_SUCCESS.get(outcome, 0.90) * float(PASS_BASE_SUCCESS_MULT)
        p_ok = prob_from_scores(
            rng,
            base_s,
            off_score,
            def_score,
            kind="pass",
            variance_mult=variance_mult,
            logit_delta=float(tags.get('role_logit_delta', 0.0)),
        )
        if rng.random() < p_ok:
            return "CONTINUE", {"outcome": outcome, "pass_chain": pass_chain + 1}
        else:
            # pass failure does NOT automatically become a turnover (TO is controlled by priors only)
            return "RESET", {"outcome": outcome, "type": "PASS_FAIL"}

    if is_to(outcome):
        offense.tov += 1
        offense.add_player_stat(actor.pid, "TOV", 1)
        return "TURNOVER", {"outcome": outcome, "pid": actor.pid}

    if is_foul(outcome):
        fouler_pid = None
        team_fouls = (ctx or {}).get("team_fouls") or {}
        player_fouls = (ctx or {}).get("player_fouls") or {}
        foul_out_limit = int((ctx or {}).get("foul_out", 6))
        def_on_court = (ctx or {}).get("def_on_court") or [p.pid for p in defense.lineup]
        if def_on_court:
            fouler_pid = rng.choice(list(def_on_court))
            player_fouls[fouler_pid] = player_fouls.get(fouler_pid, 0) + 1
            if game_state is not None:
                game_state.player_fouls[fouler_pid] = player_fouls[fouler_pid]
        team_fouls[defense.name] = team_fouls.get(defense.name, 0) + 1
        if game_state is not None:
            game_state.team_fouls[defense.name] = team_fouls[defense.name]

        shot_key = "SHOT_3_OD" if outcome == "FOUL_DRAW_JUMPER" else "SHOT_RIM_DUNK"
        pts = 3 if shot_key == "SHOT_3_OD" else 2
        base_p = SHOT_BASE.get(shot_key, 0.45)
        kind = _shot_kind_from_outcome(shot_key)
        if kind == "shot_rim":
            base_p *= float(SHOT_BASE_RIM)
        elif kind == "shot_mid":
            base_p *= float(SHOT_BASE_MID)
        else:
            base_p *= float(SHOT_BASE_3)
        kind = _shot_kind_from_outcome(shot_key)
        p_make = prob_from_scores(
            rng,
            base_p,
            off_score,
            def_score,
            kind=kind,
            variance_mult=variance_mult,
            logit_delta=float(tags.get('role_logit_delta', 0.0)),
            fatigue_logit_delta=fatigue_logit_delta,
        )

        and_one = False
        if rng.random() < p_make:
            offense.fga += 1
            offense.add_player_stat(actor.pid, "FGA", 1)
            if pts == 3:
                offense.tpa += 1
                offense.add_player_stat(actor.pid, "3PA", 1)
            offense.fgm += 1
            offense.add_player_stat(actor.pid, "FGM", 1)
            if pts == 3:
                offense.tpm += 1
                offense.add_player_stat(actor.pid, "3PM", 1)
            offense.pts += pts
            offense.add_player_stat(actor.pid, "PTS", pts)
            and_one = True

        nfts = (3 if outcome == "FOUL_DRAW_JUMPER" else 2) + (1 if and_one else 0)
        resolve_free_throws(rng, actor, nfts, offense)

        if fouler_pid and player_fouls.get(fouler_pid, 0) >= foul_out_limit:
            if game_state is not None:
                game_state.fatigue[fouler_pid] = 0.0

        return "FOUL", {"outcome": outcome, "pid": actor.pid, "fts": nfts, "and_one": and_one, "fouler": fouler_pid}

    if is_reset(outcome):
        return "RESET", {"outcome": outcome}

    return "RESET", {"outcome": outcome}


