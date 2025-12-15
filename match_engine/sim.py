from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .builders import (
    build_defense_action_probs,
    build_offense_action_probs,
    build_outcome_priors,
    get_action_base,
)
from .core import ENGINE_VERSION, make_replay_token, weighted_choice
from .models import GameState, ROLE_FALLBACK_RANK, TeamState
from .resolve import (
    choose_drb_rebounder,
    choose_orb_rebounder,
    rebound_orb_probability,
    resolve_outcome,
)
from .role_fit import apply_role_fit_to_priors_and_tags
from .validation import (
    ValidationConfig,
    ValidationReport,
    sanitize_tactics_config,
    validate_and_sanitize_team,
)
from .era import _ACTIVE_ERA_NAME, _ACTIVE_ERA_VERSION, apply_era_config, get_mvp_rules, load_era_config


def _init_targets(team: TeamState, rules: Dict[str, Any]) -> Dict[str, int]:
    tcfg = rules.get("fatigue_targets", {})
    starter_sec = int(tcfg.get("starter_sec", 32 * 60))
    rotation_sec = int(tcfg.get("rotation_sec", 16 * 60))
    bench_sec = int(tcfg.get("bench_sec", 8 * 60))
    targets: Dict[str, int] = {}
    for idx, p in enumerate(team.lineup):
        if idx < 5:
            targets[p.pid] = starter_sec
        elif idx < 8:
            targets[p.pid] = rotation_sec
        else:
            targets[p.pid] = bench_sec
    return targets


def _get_on_court(game_state: GameState, team: TeamState, home: TeamState) -> List[str]:
    return game_state.on_court_home if team is home else game_state.on_court_away


def _set_on_court(game_state: GameState, team: TeamState, home: TeamState, players: List[str]) -> None:
    if team is home:
        game_state.on_court_home = list(players)
    else:
        game_state.on_court_away = list(players)


def _update_minutes(game_state: GameState, pids: List[str], delta_sec: float) -> None:
    inc = int(max(delta_sec, 0))
    for pid in pids:
        game_state.minutes_played_sec[pid] = game_state.minutes_played_sec.get(pid, 0) + inc


def _fatigue_loss_for_role(role: str, rules: Dict[str, Any]) -> float:
    fl = rules.get("fatigue_loss", {})
    if role == "handler":
        return float(fl.get("handler", 0.012))
    if role == "big":
        return float(fl.get("big", 0.009))
    return float(fl.get("wing", 0.010))


def _apply_fatigue_loss(team: TeamState, on_court: List[str], game_state: GameState, rules: Dict[str, Any], intensity: Dict[str, bool]) -> None:
    for pid in on_court:
        role = "wing"
        if team.roles.get("ball_handler") == pid:
            role = "handler"
        elif team.roles.get("secondary_handler") == pid:
            role = "handler"
        elif team.roles.get("screener") == pid or team.roles.get("post") == pid:
            role = "big"
        p = team.find_player(pid)
        if p and p.pos in ("C", "F"):
            role = "big"
        loss = _fatigue_loss_for_role(role, rules)
        if intensity.get("transition_emphasis"):
            loss += float(rules.get("fatigue_loss", {}).get("transition_emphasis", 0.001))
        if intensity.get("heavy_pnr") and role in ("handler", "big"):
            loss += float(rules.get("fatigue_loss", {}).get("heavy_pnr", 0.001))
        game_state.fatigue[pid] = max(0.0, min(1.0, game_state.fatigue.get(pid, 1.0) - loss))


def _perform_rotation(rng: random.Random, team: TeamState, home: TeamState, game_state: GameState, rules: Dict[str, Any], is_garbage: bool) -> None:
    thresholds = rules.get("fatigue_thresholds", {})
    sub_out_th = float(thresholds.get("sub_out", 0.35))
    sub_in_th = float(thresholds.get("sub_in", 0.70))
    foul_out = int(rules.get("foul_out", 6))
    targets = game_state.targets_sec_home if team is home else game_state.targets_sec_away
    on_court = _get_on_court(game_state, team, home)
    bench = [p.pid for p in team.lineup if p.pid not in on_court and game_state.player_fouls.get(p.pid, 0) < foul_out]

    def fatigue(pid: str) -> float:
        return float(game_state.fatigue.get(pid, 1.0))

    def minutes(pid: str) -> int:
        return int(game_state.minutes_played_sec.get(pid, 0))

    out_candidates = []
    for pid in list(on_court):
        tired = fatigue(pid) < sub_out_th or game_state.player_fouls.get(pid, 0) >= foul_out
        over_target = minutes(pid) > targets.get(pid, 0) + 120
        if tired or over_target:
            out_candidates.append(pid)
        elif is_garbage and on_court and targets.get(pid, 0) >= targets.get(on_court[0], 0):
            out_candidates.append(pid)

    in_candidates = [pid for pid in bench if fatigue(pid) > sub_in_th and minutes(pid) <= targets.get(pid, 0) + 240]
    swaps = 0
    for pid_out in sorted(out_candidates, key=lambda pid: fatigue(pid)):
        if swaps >= 2:
            break
        if not in_candidates:
            break
        # choose the bench player furthest below target, tie-break by fresh fatigue
        pid_in = max(in_candidates, key=lambda pid: (targets.get(pid, 0) - minutes(pid), fatigue(pid)))
        in_candidates.remove(pid_in)
        if pid_out in on_court:
            on_court[on_court.index(pid_out)] = pid_in
            swaps += 1

    _set_on_court(game_state, team, home, on_court[:5])
def apply_time_cost(game_state: GameState, cost: float, tempo_mult: float) -> None:
    adj = float(cost) * float(tempo_mult)
    game_state.shot_clock_sec -= adj
    game_state.clock_sec = max(game_state.clock_sec - adj, 0.0)


def commit_shot_clock_turnover(offense: TeamState) -> None:
    offense.tov += 1
    bh = offense.get_role_player("ball_handler", ROLE_FALLBACK_RANK["ball_handler"])
    offense.add_player_stat(bh.pid, "TOV", 1)
    offense.outcome_counts["TO_SHOTCLOCK"] = offense.outcome_counts.get("TO_SHOTCLOCK", 0) + 1


# -------------------------
# Possession simulation
# -------------------------

def simulate_possession(
    rng: random.Random,
    offense: TeamState,
    defense: TeamState,
    game_state: GameState,
    rules: Dict[str, Any],
    ctx: Dict[str, Any],
    max_steps: int = 7,
) -> None:
    offense.possessions += 1

    tempo_mult = float(ctx.get("tempo_mult", 1.0))
    time_costs = rules.get("time_costs", {})

    off_probs = build_offense_action_probs(offense.tactics, defense.tactics, ctx=ctx)
    def_probs = build_defense_action_probs(defense.tactics)

    action = weighted_choice(rng, off_probs)
    offense.off_action_counts[action] = offense.off_action_counts.get(action, 0) + 1

    def_action = weighted_choice(rng, def_probs)
    defense.def_action_counts[def_action] = defense.def_action_counts.get(def_action, 0) + 1

    tags = {
        "in_transition": (get_action_base(action) == "TransitionEarly"),
        "is_side_pnr": (action == "SideAnglePnR"),
        "avg_fatigue_off": ctx.get("avg_fatigue_off"),
        "fatigue_bad_mult_max": ctx.get("fatigue_bad_mult_max"),
        "fatigue_bad_critical": ctx.get("fatigue_bad_critical"),
        "fatigue_bad_bonus": ctx.get("fatigue_bad_bonus"),
        "fatigue_bad_cap": ctx.get("fatigue_bad_cap"),
    }

    steps = 0
    pass_chain = 0

    while steps < max_steps and game_state.clock_sec > 0:
        steps += 1

        action_cost = float(time_costs.get(get_action_base(action), 0.0))
        if action_cost > 0:
            apply_time_cost(game_state, action_cost, tempo_mult)
            if game_state.shot_clock_sec <= 0:
                commit_shot_clock_turnover(offense)
                return
            if game_state.clock_sec <= 0:
                game_state.clock_sec = 0
                return

        pri = build_outcome_priors(action, offense.tactics, defense.tactics, tags)
        pri = apply_role_fit_to_priors_and_tags(pri, get_action_base(action), offense, tags)
        outcome = weighted_choice(rng, pri)

        term, payload = resolve_outcome(rng, outcome, action, offense, defense, tags, pass_chain, ctx=ctx, game_state=game_state)

        if term in ("SCORE", "TURNOVER", "FOUL"):
            return

        if term == "MISS":
            orb_mult = float(offense.tactics.context.get("ORB_MULT", 1.0))
            drb_mult = float(defense.tactics.context.get("DRB_MULT", 1.0))
            p_orb = rebound_orb_probability(offense, defense, orb_mult, drb_mult)
            if rng.random() < p_orb:
                offense.orb += 1
                rbd = choose_orb_rebounder(rng, offense)
                offense.add_player_stat(rbd.pid, "ORB", 1)
                game_state.shot_clock_sec = float(rules.get("orb_reset", game_state.shot_clock_sec))
                action = "Kickout" if rng.random() < 0.55 else "Drive"
                pass_chain = 0
                continue
            defense.drb += 1
            rbd = choose_drb_rebounder(rng, defense)
            defense.add_player_stat(rbd.pid, "DRB", 1)
            return

        if term == "RESET":
            reset_cost = float(time_costs.get("Reset", 0.0))
            if reset_cost > 0:
                apply_time_cost(game_state, reset_cost, tempo_mult)
                if game_state.shot_clock_sec <= 0:
                    commit_shot_clock_turnover(offense)
                    return
                if game_state.clock_sec <= 0:
                    game_state.clock_sec = 0
                    return
            off_probs = build_offense_action_probs(offense.tactics, defense.tactics, ctx=ctx)
            action = weighted_choice(rng, off_probs)
            offense.off_action_counts[action] = offense.off_action_counts.get(action, 0) + 1
            pass_chain = 0
            continue

        if term == "CONTINUE":
            pass_chain = payload.get("pass_chain", pass_chain + 1)
            pass_cost = 0.0
            if outcome in ("PASS_KICKOUT", "PASS_SKIP"):
                pass_cost = float(time_costs.get("Kickout", 0.0))
            elif outcome == "PASS_EXTRA":
                pass_cost = float(time_costs.get("ExtraPass", 0.0))
            if pass_cost > 0:
                apply_time_cost(game_state, pass_cost, tempo_mult)
                if game_state.shot_clock_sec <= 0:
                    commit_shot_clock_turnover(offense)
                    return
                if game_state.clock_sec <= 0:
                    game_state.clock_sec = 0
                    return
            # after pass, steer toward spotup / extra
            if outcome in ("PASS_KICKOUT", "PASS_SKIP", "PASS_EXTRA"):
                action = "SpotUp" if rng.random() < 0.72 else "ExtraPass"
            elif outcome == "PASS_SHORTROLL":
                action = "Drive" if rng.random() < 0.55 else "Kickout"
            else:
                action = weighted_choice(rng, off_probs)

            if pass_chain >= 3:
                action = "SpotUp"
            continue

    commit_shot_clock_turnover(offense)


# -------------------------
# Game simulation
# -------------------------

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
        "Possessions": team.possessions,
        "OffActionCounts": dict(sorted(team.off_action_counts.items(), key=lambda x: -x[1])),
        "DefActionCounts": dict(sorted(team.def_action_counts.items(), key=lambda x: -x[1])),
        "OutcomeCounts": dict(sorted(team.outcome_counts.items(), key=lambda x: -x[1])),
        "Players": team.player_stats,
        "AvgFatigue": sum(p.fatigue for p in team.lineup) / len(team.lineup),
        "ShotZones": dict(team.shot_zones),
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

    rules = get_mvp_rules()
    targets_home = _init_targets(teamA, rules)
    targets_away = _init_targets(teamB, rules)
    game_state = GameState(
        quarter=1,
        clock_sec=0,
        shot_clock_sec=0,
        score_home=teamA.pts,
        score_away=teamB.pts,
        possession=0,
        team_fouls={teamA.name: 0, teamB.name: 0},
        player_fouls={},
        fatigue={p.pid: 1.0 for p in teamA.lineup + teamB.lineup},
        minutes_played_sec={p.pid: 0 for p in teamA.lineup + teamB.lineup},
        on_court_home=[p.pid for p in teamA.lineup[:5]],
        on_court_away=[p.pid for p in teamB.lineup[:5]],
        targets_sec_home=targets_home,
        targets_sec_away=targets_away,
    )
    orig_lineups = {teamA.name: list(teamA.lineup), teamB.name: list(teamB.lineup)}

    total_possessions = 0
    for q in range(int(rules.get("quarters", 4))):
        game_state.quarter = q + 1
        game_state.clock_sec = float(rules.get("quarter_length", 720))
        game_state.team_fouls[teamA.name] = 0
        game_state.team_fouls[teamB.name] = 0

        while game_state.clock_sec > 0:
            offense, defense = (teamA, teamB) if total_possessions % 2 == 0 else (teamB, teamA)
            game_state.possession = total_possessions
            game_state.shot_clock_sec = float(rules.get("shot_clock", 24))

            start_clock = game_state.clock_sec

            off_on_court = _get_on_court(game_state, offense, teamA)
            def_on_court = _get_on_court(game_state, defense, teamA)
            offense.lineup = [offense.find_player(pid) or offense.lineup[0] for pid in off_on_court if offense.find_player(pid)]
            defense.lineup = [defense.find_player(pid) or defense.lineup[0] for pid in def_on_court if defense.find_player(pid)]

            avg_off_fatigue = sum(game_state.fatigue.get(pid, 1.0) for pid in off_on_court) / max(len(off_on_court), 1)
            avg_def_fatigue = sum(game_state.fatigue.get(pid, 1.0) for pid in def_on_court) / max(len(def_on_court), 1)
            def_eff_mult = float(rules.get("fatigue_effects", {}).get("def_mult_min", 0.90)) + 0.10 * avg_def_fatigue

            score_diff = teamA.pts - teamB.pts
            is_clutch = game_state.quarter == 4 and game_state.clock_sec <= 120 and abs(score_diff) <= 8
            is_garbage = game_state.quarter == 4 and game_state.clock_sec <= 360 and abs(score_diff) >= 20
            variance_mult = 0.80 if is_clutch else 1.25 if is_garbage else 1.0
            tempo_mult = (1.0 / 1.08) if is_garbage else 1.0

            ctx = {
                "score_diff": score_diff,
                "is_clutch": is_clutch,
                "is_garbage": is_garbage,
                "variance_mult": variance_mult,
                "tempo_mult": tempo_mult,
                "avg_fatigue_off": avg_off_fatigue,
                "fatigue_bad_mult_max": float(rules.get("fatigue_effects", {}).get("bad_mult_max", 1.12)),
                "fatigue_bad_critical": float(rules.get("fatigue_effects", {}).get("bad_critical", 0.25)),
                "fatigue_bad_bonus": float(rules.get("fatigue_effects", {}).get("bad_bonus", 0.08)),
                "fatigue_bad_cap": float(rules.get("fatigue_effects", {}).get("bad_cap", 1.20)),
                "fatigue_logit_max": float(rules.get("fatigue_effects", {}).get("logit_delta_max", -0.25)),
                "def_eff_mult": def_eff_mult,
                "fatigue_map": game_state.fatigue,
                "def_on_court": def_on_court,
                "off_on_court": off_on_court,
                "team_fouls": game_state.team_fouls,
                "player_fouls": game_state.player_fouls,
                "foul_out": int(rules.get("foul_out", 6)),
                "bonus_threshold": int(rules.get("bonus_threshold", 5)),
            }

            setup_cost = float(rules.get("time_costs", {}).get("possession_setup", 0.0))
            if setup_cost > 0:
                apply_time_cost(game_state, setup_cost, tempo_mult)
                if game_state.shot_clock_sec <= 0:
                    commit_shot_clock_turnover(offense)
                    offense.lineup = list(orig_lineups[offense.name])
                    defense.lineup = list(orig_lineups[defense.name])
                    total_possessions += 1
                    game_state.score_home = teamA.pts
                    game_state.score_away = teamB.pts
                    if game_state.clock_sec <= 0:
                        game_state.clock_sec = 0
                        break
                    continue
                if game_state.clock_sec <= 0:
                    game_state.clock_sec = 0
                    offense.lineup = list(orig_lineups[offense.name])
                    defense.lineup = list(orig_lineups[defense.name])
                    break

            simulate_possession(rng, offense, defense, game_state, rules, ctx)
            offense.lineup = list(orig_lineups[offense.name])
            defense.lineup = list(orig_lineups[defense.name])
            elapsed = max(start_clock - game_state.clock_sec, 0.0)
            _update_minutes(game_state, off_on_court, elapsed)
            _update_minutes(game_state, def_on_court, elapsed)

            intensity_off = {
                "transition_emphasis": bool(offense.tactics.context.get("TRANSITION_EMPHASIS", False)),
                "heavy_pnr": bool(offense.tactics.context.get("HEAVY_PNR", False)) or "PnR" in offense.tactics.offense_scheme,
            }
            intensity_def = {
                "transition_emphasis": bool(defense.tactics.context.get("TRANSITION_EMPHASIS", False)),
                "heavy_pnr": bool(defense.tactics.context.get("HEAVY_PNR", False)) or "PnR" in defense.tactics.offense_scheme,
            }
            _apply_fatigue_loss(offense, off_on_court, game_state, rules, intensity_off)
            _apply_fatigue_loss(defense, def_on_court, game_state, rules, intensity_def)

            _perform_rotation(rng, offense, teamA, game_state, rules, is_garbage)
            _perform_rotation(rng, defense, teamA, game_state, rules, is_garbage)
            total_possessions += 1

            game_state.score_home = teamA.pts
            game_state.score_away = teamB.pts

            if game_state.clock_sec <= 0:
                game_state.clock_sec = 0
                break

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
        "possessions_per_team": max(teamA.possessions, teamB.possessions),
        "teams": {
            teamA.name: summarize_team(teamA),
            teamB.name: summarize_team(teamB),
        },
        "game_state": {
            "team_fouls": dict(game_state.team_fouls),
            "player_fouls": dict(game_state.player_fouls),
            "fatigue": dict(game_state.fatigue),
            "minutes_played_sec": dict(game_state.minutes_played_sec),
        }
    }



