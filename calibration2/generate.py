from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from ..models import Player, TeamState
from ..tactics import TacticsConfig
from ..role_fit import role_fit_score
from ..sim_rotation import ROLE_TO_GROUPS

# Reuse player generator & required keys from calibration v1 (keeps contracts aligned)
from ..calibration.generate import generate_player, ARCHETYPES, REQUIRED_DERIVED_KEYS  # type: ignore


# -----------------------------
# Balanced roster plan (12 players)
# -----------------------------
BALANCED_ROSTER_PLAN: Tuple[str, ...] = (
    "lead_guard",
    "scoring_guard",
    "movement_shooter",
    "three_d_wing",
    "three_d_wing",
    "slashing_wing",
    "three_d_wing",
    "connector_forward",
    "connector_forward",
    "stretch_big",
    "roll_big",
    "rim_protector",
)

ROLES_12: Tuple[str, ...] = tuple(ROLE_TO_GROUPS.keys())


def _overall_rating(p: Player) -> float:
    # Same simple blend as calibration v1
    keys = [
        ("PASS_CREATE", 0.12),
        ("HANDLE_SAFE", 0.10),
        ("SHOT_3_CS", 0.10),
        ("SHOT_3_OD", 0.08),
        ("FIN_RIM", 0.08),
        ("DEF_POA", 0.10),
        ("DEF_HELP", 0.10),
        ("DEF_RIM", 0.10),
        ("REB_DR", 0.08),
        ("PHYSICAL", 0.07),
        ("ENDURANCE", 0.07),
    ]
    s = 0.0
    for k, w in keys:
        try:
            s += float(p.get(k)) * float(w)
        except Exception:
            pass
    return s


def _role_priority_for_scheme(off_scheme: str) -> List[str]:
    # Copied logic from calibration v1: a lightweight role priority list
    base = [
        "Initiator_Primary",
        "Initiator_Secondary",
        "PnR_BallHandler",
        "Spacer_Movement",
        "Spacer_Spotup",
        "Cutter_Slasher",
        "Connector_Playmaker",
        "Roller_Finisher",
        "Pop_Spacer_Big",
        "Post_Hub",
        "Def_POA_Stopper",
        "Def_Rim_Anchor",
    ]
    if off_scheme == "Transition_Early":
        base = ["Transition_Handler"] + [x for x in base if x != "Transition_Handler"]
    if off_scheme == "FiveOut":
        base = ["Pop_Spacer_Big"] + [x for x in base if x != "Pop_Spacer_Big"]
    if off_scheme == "Post_InsideOut":
        base = ["Post_Hub"] + [x for x in base if x != "Post_Hub"]
    return base


def assign_roles_12(
    rng: random.Random,
    players: List[Player],
    off_scheme: str,
    *,
    unique_first_n: int = 8,
) -> Dict[str, str]:
    # Score table
    scores: Dict[str, List[Tuple[str, float]]] = {}
    for role in ROLES_12:
        lst: List[Tuple[str, float]] = []
        for p in players:
            s = float(role_fit_score(p, role))
            # scheme hints (small, but consistent) — keep identical to v1
            if off_scheme == "FiveOut" and role == "Pop_Spacer_Big":
                s *= 1.08
            if off_scheme == "Post_InsideOut" and role == "Post_Hub":
                s *= 1.10
            if off_scheme == "Transition_Early" and role == "Transition_Handler":
                s *= 1.10
            if off_scheme in ("Motion_SplitCut", "DHO_Chicago") and role in ("Spacer_Movement", "Connector_Playmaker"):
                s *= 1.06
            lst.append((p.pid, s))
        lst.sort(key=lambda x: x[1], reverse=True)
        scores[role] = lst

    priority = _role_priority_for_scheme(off_scheme)
    out: Dict[str, str] = {}
    used: set[str] = set()

    # Phase 1: enforce uniqueness for key roles
    for i, role in enumerate(priority):
        if role not in scores:
            continue
        pick = None
        for pid, _ in scores[role]:
            if (i < unique_first_n) and (pid in used):
                continue
            pick = pid
            break
        if pick is None:
            pick = scores[role][0][0]
        out[role] = pick
        if i < unique_first_n:
            used.add(pick)

    # Phase 2: fill any missing roles (allow overlaps)
    for role in ROLES_12:
        if role in out:
            continue
        out[role] = scores[role][0][0]

    return out


def _choose_starters(players: List[Player], roles: Dict[str, str]) -> List[str]:
    init = roles.get("Initiator_Primary")
    big_candidates = [roles.get("Roller_Finisher"), roles.get("Pop_Spacer_Big"), roles.get("Post_Hub")]
    big_candidates = [pid for pid in big_candidates if pid]
    big = None
    for pid in big_candidates:
        p = next((x for x in players if x.pid == pid), None)
        if p and p.pos in ("C", "F"):
            big = pid
            break

    ranked = sorted(players, key=_overall_rating, reverse=True)
    starters: List[str] = []
    if init:
        starters.append(init)
    if big and big not in starters:
        starters.append(big)
    for p in ranked:
        if p.pid in starters:
            continue
        starters.append(p.pid)
        if len(starters) >= 5:
            break
    return starters[:5]


def _build_rotation_targets(players: List[Player], starters: List[str]) -> Dict[str, int]:
    ranked = sorted(players, key=_overall_rating, reverse=True)
    targets: Dict[str, int] = {}
    for i, p in enumerate(ranked):
        if p.pid in starters:
            targets[p.pid] = int(32 * 60)
        elif i < 8:
            targets[p.pid] = int(20 * 60)
        elif i < 10:
            targets[p.pid] = int(12 * 60)
        else:
            targets[p.pid] = int(6 * 60)
    return targets


def generate_balanced_roster(
    rng: random.Random,
    *,
    roster_id: str,
    name_prefix: str = "Roster",
) -> List[Player]:
    players: List[Player] = []
    for i, archetype in enumerate(BALANCED_ROSTER_PLAN):
        pid = f"{roster_id}_{i:02d}"
        players.append(
            generate_player(
                rng,
                pid=pid,
                name=f"{name_prefix}_{archetype}_{i:02d}",
                archetype=archetype,
            )
        )
    return players


def _clone_roster_players(base_players: List[Player], *, team_id: str, team_name: str) -> List[Player]:
    # Deep copy to avoid cross-game mutation; re-id to keep team uniqueness contracts clean
    out: List[Player] = []
    for i, p in enumerate(base_players):
        cp: Player = copy.deepcopy(p)
        cp.pid = f"{team_id}_{i:02d}"
        cp.name = f"{team_name}_{i:02d}"
        out.append(cp)
    return out


def sample_knobs(
    rng: random.Random,
    *,
    mode: str,
    sd: float = 0.03,
) -> Dict[str, float]:
    if mode == "pure":
        return {
            "scheme_weight_sharpness": 1.0,
            "scheme_outcome_strength": 1.0,
            "def_scheme_weight_sharpness": 1.0,
            "def_scheme_outcome_strength": 1.0,
        }

    # variation: narrow truncated normal around 1.0
    def tn(mu: float = 1.0) -> float:
        for _ in range(12):
            x = rng.gauss(mu, sd)
            if 0.85 <= x <= 1.15:
                return float(x)
        return float(max(0.85, min(1.15, mu)))

    return {
        "scheme_weight_sharpness": tn(),
        "scheme_outcome_strength": tn(),
        "def_scheme_weight_sharpness": tn(),
        "def_scheme_outcome_strength": tn(),
    }


def build_team_from_roster_and_schemes(
    rng: random.Random,
    *,
    base_players: List[Player],
    team_id: str,
    name: str,
    offense_scheme: str,
    defense_scheme: str,
    knobs_mode: str = "pure",
    knobs_sd: float = 0.03,
) -> Tuple[TeamState, Dict[str, Any]]:
    players = _clone_roster_players(base_players, team_id=team_id, team_name=name)

    knobs = sample_knobs(rng, mode=str(knobs_mode), sd=float(knobs_sd))

    tac = TacticsConfig(
        offense_scheme=str(offense_scheme),
        defense_scheme=str(defense_scheme),
        scheme_weight_sharpness=float(knobs["scheme_weight_sharpness"]),
        scheme_outcome_strength=float(knobs["scheme_outcome_strength"]),
        def_scheme_weight_sharpness=float(knobs["def_scheme_weight_sharpness"]),
        def_scheme_outcome_strength=float(knobs["def_scheme_outcome_strength"]),
    )

    roles = assign_roles_12(rng, players, str(offense_scheme), unique_first_n=8)
    starters = _choose_starters(players, roles)

    pid_to_player = {p.pid: p for p in players}
    lineup = [pid_to_player[pid] for pid in starters if pid in pid_to_player]
    lineup += [p for p in players if p.pid not in starters]

    team = TeamState(
        team_id=str(team_id),
        name=str(name),
        lineup=lineup,
        roles=roles,
        tactics=tac,
    )

    # rotation hints
    pid_role: Dict[str, str] = {}
    for role, pid in roles.items():
        if not pid:
            continue
        prev = pid_role.get(pid)
        if prev is None:
            pid_role[pid] = role
        else:
            def pri(rn: str) -> int:
                groups = ROLE_TO_GROUPS.get(rn, tuple())
                g0 = groups[0] if groups else ""
                return 3 if g0 == "Handler" else 2 if g0 == "Wing" else 1 if g0 == "Big" else 0
            if pri(role) > pri(prev):
                pid_role[pid] = role

    team.rotation_offense_role_by_pid = dict(pid_role)
    team.rotation_target_sec_by_pid = _build_rotation_targets(players, starters)

    meta = {
        "team_id": team.team_id,
        "name": team.name,
        "offense_scheme": tac.offense_scheme,
        "defense_scheme": tac.defense_scheme,
        "knobs": dict(knobs),
        "roles": dict(roles),
        "starters": list(starters),
    }
    return team, meta
