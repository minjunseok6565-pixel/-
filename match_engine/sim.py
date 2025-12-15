from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

from .builders import build_defense_action_probs, build_offense_action_probs, effective_scheme_multiplier, get_action_base
from .core import ENGINE_VERSION, apply_multipliers, normalize_weights, weighted_choice
from .defense import team_def_snapshot
from .models import Player, TeamState
from .participants import (
    choose_creator_for_pulloff,
    choose_drb_rebounder,
    choose_finisher_rim,
    choose_orb_rebounder,
    choose_shooter_for_mid,
    choose_shooter_for_three,
)
from .prob import prob_from_scores
from .profiles import ACTION_ALIASES, ACTION_OUTCOME_PRIORS, DEFENSE_SCHEME_MULT, OFFENSE_SCHEME_MULT, PASS_BASE_SUCCESS, SHOT_BASE
from .resolve import rebound_orb_probability, resolve_free_throws
from .role_fit import apply_role_fit_to_priors_and_tags
from .validation import (
    ValidationConfig,
    ValidationReport,
    sanitize_tactics_config,
    validate_and_sanitize_team,
)
from .era import _ACTIVE_ERA_NAME, _ACTIVE_ERA_VERSION, apply_era_config, load_era_config

# -------------------------
# Possession simulation
# -------------------------

def simulate_possession(rng: random.Random, offense: TeamState, defense: TeamState, max_steps: int = 7) -> None:
    offense.possessions += 1

    off_probs = build_offense_action_probs(offense.tactics, defense.tactics)
    def_probs = build_defense_action_probs(defense.tactics)

    action = weighted_choice(rng, off_probs)
    offense.off_action_counts[action] = offense.off_action_counts.get(action, 0) + 1

    def_action = weighted_choice(rng, def_probs)
    defense.def_action_counts[def_action] = defense.def_action_counts.get(def_action, 0) + 1

    tags = {
        "in_transition": (get_action_base(action) == "TransitionEarly"),
        "is_side_pnr": (action == "SideAnglePnR"),
    }

    steps = 0
    resets = 0
    pass_chain = 0

    while steps < max_steps:
        steps += 1

        pri = build_outcome_priors(action, offense.tactics, defense.tactics, tags)
        pri = apply_role_fit_to_priors_and_tags(pri, get_action_base(action), offense, tags)
        outcome = weighted_choice(rng, pri)

        term, payload = resolve_outcome(rng, outcome, action, offense, defense, tags, pass_chain)

        if term in ("SCORE","TURNOVER","FOUL"):
            return

        if term == "MISS":
            orb_mult = float(offense.tactics.context.get("ORB_MULT", 1.0))
            drb_mult = float(defense.tactics.context.get("DRB_MULT", 1.0))
            p_orb = rebound_orb_probability(offense, defense, orb_mult, drb_mult)
            if rng.random() < p_orb:
                offense.orb += 1
                rbd = choose_orb_rebounder(rng, offense)
                offense.add_player_stat(rbd.pid, "ORB", 1)
                # after ORB: continue, but prefer quick kickout or putback depending on lineup
                action = "Kickout" if rng.random() < 0.55 else "Drive"
                resets += 1
                pass_chain = 0
                continue
            else:
                defense.drb += 1
                rbd = choose_drb_rebounder(rng, defense)
                defense.add_player_stat(rbd.pid, "DRB", 1)
                return

        if term == "RESET":
            resets += 1
            if resets >= 2:
                # shot clock turnover
                offense.tov += 1
                bh = offense.get_role_player("ball_handler", ROLE_FALLBACK_RANK["ball_handler"])
                offense.add_player_stat(bh.pid, "TOV", 1)
                offense.outcome_counts["TO_SHOT_CLOCK"] = offense.outcome_counts.get("TO_SHOT_CLOCK", 0) + 1
                return
            action = weighted_choice(rng, off_probs)
            offense.off_action_counts[action] = offense.off_action_counts.get(action, 0) + 1
            pass_chain = 0
            continue

        if term == "CONTINUE":
            pass_chain = payload.get("pass_chain", pass_chain + 1)
            # after pass, steer toward spotup / extra
            if outcome in ("PASS_KICKOUT","PASS_SKIP","PASS_EXTRA"):
                action = "SpotUp" if rng.random() < 0.72 else "ExtraPass"
            elif outcome == "PASS_SHORTROLL":
                action = "Drive" if rng.random() < 0.55 else "Kickout"
            else:
                action = weighted_choice(rng, off_probs)

            if pass_chain >= 3:
                action = "SpotUp"
            continue

    # max steps -> TO
    offense.tov += 1
    bh = offense.get_role_player("ball_handler", ROLE_FALLBACK_RANK["ball_handler"])
    offense.add_player_stat(bh.pid, "TOV", 1)
    offense.outcome_counts["TO_SHOT_CLOCK"] = offense.outcome_counts.get("TO_SHOT_CLOCK", 0) + 1


# -------------------------
# Game simulation
# -------------------------

def estimate_possessions(teamA: TeamState, teamB: TeamState) -> int:
    base = 96
    a = float(teamA.tactics.context.get("PACE_MULT", 1.0))
    b = float(teamB.tactics.context.get("PACE_MULT", 1.0))
    pace = base * ((a + b) / 2.0)
    return int(clamp(pace, 78, 112))

def init_player_boxes(team: TeamState) -> None:
    for p in team.lineup:
        team.player_stats[p.pid] = {"PTS":0,"FGM":0,"FGA":0,"3PM":0,"3PA":0,"FTM":0,"FTA":0,"TOV":0,"ORB":0,"DRB":0}

def summarize_team(team: TeamState) -> Dict[str, Any]:
    return {
        "PTS": team.pts,
        "FGM": team.fgm, "FGA": team.fga,
        "3PM": team.tpm, "3PA": team.tpa,
        "FTM": team.ftm, "FTA": team.fta,
        "TOV": team.tov,
        "ORB": team.orb, "DRB": team.drb,
        "OffActionCounts": dict(sorted(team.off_action_counts.items(), key=lambda x: -x[1])),
        "DefActionCounts": dict(sorted(team.def_action_counts.items(), key=lambda x: -x[1])),
        "OutcomeCounts": dict(sorted(team.outcome_counts.items(), key=lambda x: -x[1])),
        "Players": team.player_stats,
        "AvgFatigue": sum(p.fatigue for p in team.lineup) / len(team.lineup),
    }


def simulate_game(
    rng: random.Random,
    teamA: TeamState,
    teamB: TeamState,
    era: str = "default",
    strict_validation: bool = True,
    validation: Optional[ValidationConfig] = None,
) -> Dict[str, Any]:
    """Simulate a full game with input validation/sanitization.

    0-2 (commercial safety):
    - clamps all UI multipliers to [0.70, 1.40]
    - ignores unknown tactic keys (but logs warnings)
    - validates required derived keys (error by default; can 'fill' via ValidationConfig)
    """
    report = ValidationReport()
    cfg = validation if validation is not None else ValidationConfig(strict=strict_validation)

    # 0-1: load era tuning parameters (priors/base%/scheme multipliers/prob model)
    era_cfg, era_warnings, era_errors = load_era_config(era)
    for w in era_warnings:
        report.warn(f"era[{era}]: {w}")
    for e in era_errors:
        report.error(f"era[{era}]: {e}")

    # Apply era tuning to engine globals BEFORE sanitizing tactics.
    apply_era_config(era_cfg)

    # If caller did not pass a custom ValidationConfig, adopt knob clamp bounds from era.
    if validation is None and isinstance(era_cfg.get("knobs"), dict):
        k = era_cfg.get("knobs", {})
        if isinstance(k.get("mult_lo"), (int, float)):
            cfg.mult_lo = float(k["mult_lo"])
        if isinstance(k.get("mult_hi"), (int, float)):
            cfg.mult_hi = float(k["mult_hi"])

    validate_and_sanitize_team(teamA, cfg, report, label=f"team[{teamA.name}]")
    validate_and_sanitize_team(teamB, cfg, report, label=f"team[{teamB.name}]")

    if cfg.strict and report.errors:
        # Raise with a compact, actionable message (full list is also in report)
        head = "\n".join(report.errors[:6])
        more = f"\n... (+{len(report.errors)-6} more)" if len(report.errors) > 6 else ""
        raise ValueError(f"MatchEngine input validation failed:\n{head}{more}")

    init_player_boxes(teamA)
    init_player_boxes(teamB)

    n = estimate_possessions(teamA, teamB)
    for i in range(n * 2):
        if i % 2 == 0:
            simulate_possession(rng, teamA, teamB)
        else:
            simulate_possession(rng, teamB, teamA)

    replay_token = make_replay_token(rng, teamA, teamB, era=era)

    return {
        "meta": {
            "engine_version": ENGINE_VERSION,
            "era": era,
            "era_version": str(globals().get("_ACTIVE_ERA_VERSION", "1.0")),
            "replay_token": replay_token,
            "validation": report.to_dict(),
            "internal_debug": {
                "role_fit": {
                    "role_counts": {teamA.name: teamA.role_fit_role_counts, teamB.name: teamB.role_fit_role_counts},
                    "grade_counts": {teamA.name: teamA.role_fit_grade_counts, teamB.name: teamB.role_fit_grade_counts},
                    "pos_log": {teamA.name: teamA.role_fit_pos_log, teamB.name: teamB.role_fit_pos_log},
                    "bad_totals": {teamA.name: teamA.role_fit_bad_totals, teamB.name: teamB.role_fit_bad_totals},
                    "bad_by_grade": {teamA.name: teamA.role_fit_bad_by_grade, teamB.name: teamB.role_fit_bad_by_grade},
                }
            },
        },
        "possessions_per_team": n,
        "teams": {
            teamA.name: summarize_team(teamA),
            teamB.name: summarize_team(teamB),
        }
    }



