from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .core import clamp
from .era import ERA_ROLE_FIT
from .models import Player, TeamState

# -------------------------
# Role Fit (3) - Role realism via priors distortion (60%) + logit delta (40%)
# -------------------------


ROLE_FIT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "PnR_PrimaryHandler": {"PNR_READ":0.25, "DRIVE_CREATE":0.20, "HANDLE_SAFE":0.20, "SHOT_3_OD":0.15, "SHOT_MID_PU":0.10, "PASS_CREATE":0.10},
    "PnR_SecondaryHandler": {"SHOT_3_CS":0.25, "PNR_READ":0.20, "PASS_SAFE":0.15, "DRIVE_CREATE":0.15, "HANDLE_SAFE":0.15, "SHOT_3_OD":0.10},
    "DHO_PrimaryHandler": {"DRIVE_CREATE":0.20, "HANDLE_SAFE":0.20, "PASS_SAFE":0.15, "SHOT_MID_PU":0.15, "PASS_CREATE":0.10, "SHOT_3_OD":0.10, "SHOT_3_CS":0.10},
    "Elbow_Hub": {"PASS_SAFE":0.25, "PASS_CREATE":0.20, "PNR_READ":0.15, "SHORTROLL_PLAY":0.15, "SHOT_MID_CS":0.10, "SHOT_TOUCH":0.10, "HANDLE_SAFE":0.05},
    "Point_Forward": {"DRIVE_CREATE":0.18, "PASS_CREATE":0.18, "PASS_SAFE":0.18, "HANDLE_SAFE":0.14, "PNR_READ":0.12, "FIRST_STEP":0.10, "SHOT_3_CS":0.10},
    "Transition_Pusher": {"FIRST_STEP":0.20, "DRIVE_CREATE":0.20, "PASS_SAFE":0.15, "PASS_CREATE":0.15, "HANDLE_SAFE":0.15, "FIN_RIM":0.10, "ENDURANCE":0.05},

    "3pt_OffDribble_Scorer": {"SHOT_3_OD":0.35, "HANDLE_SAFE":0.15, "DRIVE_CREATE":0.15, "PNR_READ":0.10, "SHOT_MID_PU":0.10, "ENDURANCE":0.10, "SHOT_FT":0.05},
    "Mid_PullUp_Scorer": {"SHOT_MID_PU":0.35, "DRIVE_CREATE":0.15, "HANDLE_SAFE":0.15, "SHOT_TOUCH":0.10, "PNR_READ":0.10, "ENDURANCE":0.10, "SHOT_FT":0.05},
    "SpotUp_Wing": {"SHOT_3_CS":0.40, "SHOT_MID_CS":0.15, "PASS_SAFE":0.10, "HANDLE_SAFE":0.10, "FIRST_STEP":0.10, "ENDURANCE":0.10, "FIN_RIM":0.05},
    "Corner_Specialist": {"SHOT_3_CS":0.50, "PASS_SAFE":0.10, "HANDLE_SAFE":0.10, "ENDURANCE":0.10, "FIRST_STEP":0.10, "SHOT_FT":0.05, "FIN_RIM":0.05},
    "Movement_Shooter": {"SHOT_3_CS":0.35, "ENDURANCE":0.15, "SHOT_MID_CS":0.10, "PASS_SAFE":0.10, "FIRST_STEP":0.10, "HANDLE_SAFE":0.10, "DRIVE_CREATE":0.05, "SHOT_3_OD":0.05},
    "Relocation_Shooter": {"SHOT_3_CS":0.40, "ENDURANCE":0.15, "PASS_SAFE":0.10, "HANDLE_SAFE":0.10, "FIRST_STEP":0.10, "SHOT_MID_CS":0.10, "FIN_RIM":0.05},

    "Roll_Man": {"FIN_RIM":0.25, "FIN_DUNK":0.25, "FIN_CONTACT":0.15, "REB_OR":0.10, "PHYSICAL":0.10, "ENDURANCE":0.10, "SHORTROLL_PLAY":0.05},
    "ShortRoll_Playmaker": {"SHORTROLL_PLAY":0.30, "PASS_SAFE":0.20, "PASS_CREATE":0.15, "HANDLE_SAFE":0.10, "FIN_RIM":0.10, "PHYSICAL":0.10, "PNR_READ":0.05},
    "Pop_Big": {"SHOT_3_CS":0.35, "SHOT_MID_CS":0.15, "PASS_SAFE":0.15, "SHORTROLL_PLAY":0.10, "PHYSICAL":0.10, "ENDURANCE":0.10, "HANDLE_SAFE":0.05},
    "DHO_Hub_Big": {"PASS_SAFE":0.22, "SHORTROLL_PLAY":0.20, "PASS_CREATE":0.15, "SHOT_3_CS":0.15, "HANDLE_SAFE":0.10, "PHYSICAL":0.10, "SHOT_MID_CS":0.08},
    "Horns_Big_A": {"SHORTROLL_PLAY":0.22, "PASS_SAFE":0.20, "PASS_CREATE":0.15, "SHOT_MID_CS":0.15, "PHYSICAL":0.10, "FIN_RIM":0.10, "HANDLE_SAFE":0.08},
    "Horns_Big_B": {"FIN_RIM":0.20, "FIN_DUNK":0.15, "SHOT_MID_CS":0.15, "SHOT_3_CS":0.15, "FIN_CONTACT":0.10, "PHYSICAL":0.10, "SHORTROLL_PLAY":0.10, "PASS_SAFE":0.05},

    "Post_Scorer": {"POST_SCORE":0.45, "POST_CONTROL":0.25, "FIN_CONTACT":0.10, "SHOT_TOUCH":0.10, "PHYSICAL":0.10},
    "Post_Facilitator": {"POST_CONTROL":0.30, "PASS_SAFE":0.20, "PASS_CREATE":0.15, "POST_SCORE":0.15, "HANDLE_SAFE":0.10, "SHOT_TOUCH":0.10},
    "Seal_Finisher": {"SEAL_POWER":0.35, "FIN_RIM":0.20, "FIN_DUNK":0.15, "PHYSICAL":0.15, "REB_OR":0.10, "FIN_CONTACT":0.05},

    "Primary_Cutter": {"FIRST_STEP":0.25, "FIN_RIM":0.20, "HANDLE_SAFE":0.15, "ENDURANCE":0.15, "SHOT_3_CS":0.10, "PASS_SAFE":0.10, "FIN_CONTACT":0.05},
    "Dunker_Spot": {"FIN_DUNK":0.30, "FIN_RIM":0.20, "REB_OR":0.15, "PHYSICAL":0.15, "ENDURANCE":0.10, "FIN_CONTACT":0.10},
    "Backdoor_Threat": {"FIRST_STEP":0.20, "FIN_RIM":0.20, "HANDLE_SAFE":0.15, "PASS_SAFE":0.15, "ENDURANCE":0.10, "SHOT_3_CS":0.10, "SHOT_TOUCH":0.10},
    "Rim_Runner": {"ENDURANCE":0.20, "FIN_RIM":0.20, "FIN_DUNK":0.20, "FIRST_STEP":0.10, "REB_OR":0.10, "PHYSICAL":0.10, "FIN_CONTACT":0.10},

    "ExtraPass_Connector": {"PASS_SAFE":0.35, "PASS_CREATE":0.20, "HANDLE_SAFE":0.15, "SHOT_3_CS":0.15, "ENDURANCE":0.10, "PNR_READ":0.05},
    "Kickout_Trigger": {"DRIVE_CREATE":0.25, "PASS_SAFE":0.20, "PASS_CREATE":0.15, "HANDLE_SAFE":0.15, "PNR_READ":0.10, "SHOT_3_OD":0.10, "FIN_CONTACT":0.05},
}

ROLE_FIT_CUTS: Dict[str, Tuple[int,int,int,int]] = {
    "PnR_PrimaryHandler": (80,72,64,56),
    "PnR_SecondaryHandler": (78,70,62,54),
    "DHO_PrimaryHandler": (78,70,62,54),
    "Elbow_Hub": (80,72,64,56),
    "Point_Forward": (78,70,62,54),
    "Transition_Pusher": (75,67,59,51),

    "3pt_OffDribble_Scorer": (79,71,63,55),
    "Mid_PullUp_Scorer": (78,70,62,54),
    "SpotUp_Wing": (80,72,64,56),
    "Corner_Specialist": (82,74,66,58),
    "Movement_Shooter": (80,72,64,56),
    "Relocation_Shooter": (80,72,64,56),

    "Roll_Man": (76,68,60,52),
    "ShortRoll_Playmaker": (78,70,62,54),
    "Pop_Big": (80,72,64,56),
    "DHO_Hub_Big": (78,70,62,54),
    "Horns_Big_A": (78,70,62,54),
    "Horns_Big_B": (76,68,60,52),

    "Post_Scorer": (77,69,61,53),
    "Post_Facilitator": (78,70,62,54),
    "Seal_Finisher": (75,67,59,51),

    "Primary_Cutter": (74,66,58,50),
    "Dunker_Spot": (72,64,56,48),
    "Backdoor_Threat": (74,66,58,50),
    "Rim_Runner": (74,66,58,50),

    "ExtraPass_Connector": (78,70,62,54),
    "Kickout_Trigger": (76,68,60,52),
}

ROLE_PRIOR_MULT_RAW = {
    "S": {"GOOD": 1.06, "BAD": 0.94},
    "A": {"GOOD": 1.03, "BAD": 0.97},
    "B": {"GOOD": 1.00, "BAD": 1.00},
    "C": {"GOOD": 0.93, "BAD": 1.10},
    "D": {"GOOD": 0.85, "BAD": 1.25},
}
ROLE_LOGIT_DELTA_RAW = {"S": 0.18, "A": 0.10, "B": 0.00, "C": -0.18, "D": -0.35}

def role_fit_score(player: Player, role: str) -> float:
    w = ROLE_FIT_WEIGHTS.get(role)
    if not w:
        return 50.0
    s = 0.0
    for k, a in w.items():
        s += float(player.get(k)) * float(a)
    return clamp(s, 0.0, 100.0)

def role_fit_grade(role: str, fit: float) -> str:
    cuts = ROLE_FIT_CUTS.get(role)
    if not cuts:
        return "B" if fit >= 60 else "C" if fit >= 52 else "D"
    s_min, a_min, b_min, c_min = cuts
    if fit >= s_min: return "S"
    if fit >= a_min: return "A"
    if fit >= b_min: return "B"
    if fit >= c_min: return "C"
    return "D"

def _get_role_fit_strength(offense: TeamState) -> float:
    try:
        v = (offense.tactics.context or {}).get("ROLE_FIT_STRENGTH", None)
    except Exception:
        v = None
    if v is None:
        try:
            v = float((ERA_ROLE_FIT or {}).get("default_strength", 0.65))
        except Exception:
            v = 0.65
    try:
        return clamp(float(v), 0.0, 1.0)
    except Exception:
        return 0.65

def _choose_best_role(offense: TeamState, roles: List[str]) -> Optional[Tuple[str, Player, float]]:
    best = None
    for r in roles:
        pid = offense.roles.get(r)
        if not pid:
            continue
        p = offense.find_player(pid)
        if not p:
            continue
        fit = role_fit_score(p, r)
        if best is None or fit > best[2]:
            best = (r, p, fit)
    return best

def _collect_roles_for_action_family(action_family: str, offense: TeamState) -> List[Tuple[str, Player, float]]:
    parts: List[Tuple[str, Player, float]] = []
    fam = action_family

    if fam == "PnR":
        pick = _choose_best_role(offense, ["PnR_PrimaryHandler"])
        if pick: parts.append(pick)
        pick = _choose_best_role(offense, ["PnR_SecondaryHandler"])
        if pick: parts.append(pick)

        # Roll/Shortroll: 둘 다 평가
        for r in ["Roll_Man", "ShortRoll_Playmaker"]:
            pid = offense.roles.get(r)
            if pid:
                p = offense.find_player(pid)
                if p:
                    parts.append((r, p, role_fit_score(p, r)))

        # optional Pop_Big
        pid = offense.roles.get("Pop_Big")
        if pid:
            p = offense.find_player(pid)
            if p:
                parts.append(("Pop_Big", p, role_fit_score(p, "Pop_Big")))

    elif fam == "DHO":
        for group in [["DHO_PrimaryHandler"], ["Movement_Shooter", "Relocation_Shooter"], ["DHO_Hub_Big"]]:
            pick = _choose_best_role(offense, group)
            if pick: parts.append(pick)

    elif fam == "Drive":
        pick = _choose_best_role(offense, ["Kickout_Trigger", "PnR_PrimaryHandler"])
        if pick: parts.append(pick)

    elif fam == "Kickout":
        for group in [["Kickout_Trigger"], ["SpotUp_Wing", "Corner_Specialist"]]:
            pick = _choose_best_role(offense, group)
            if pick: parts.append(pick)

    elif fam == "ExtraPass":
        for group in [["ExtraPass_Connector"], ["Elbow_Hub", "Point_Forward"]]:
            pick = _choose_best_role(offense, group)
            if pick: parts.append(pick)

    elif fam == "PostUp":
        pick = _choose_best_role(offense, ["Post_Scorer", "Post_Facilitator"])
        if pick: parts.append(pick)
        pick2 = _choose_best_role(offense, ["SpotUp_Wing", "Corner_Specialist"])
        if pick2: parts.append(pick2)

    elif fam == "HornsSet":
        for group in [["Elbow_Hub"], ["Horns_Big_A"], ["Horns_Big_B"]]:
            pick = _choose_best_role(offense, group)
            if pick: parts.append(pick)

    elif fam == "SpotUp":
        pick = _choose_best_role(offense, ["SpotUp_Wing", "Corner_Specialist", "Relocation_Shooter"])
        if pick: parts.append(pick)

    elif fam == "Cut":
        pick = _choose_best_role(offense, ["Primary_Cutter", "Backdoor_Threat"])
        if pick: parts.append(pick)
        pick2 = _choose_best_role(offense, ["Elbow_Hub", "ExtraPass_Connector"])
        if pick2: parts.append(pick2)

    elif fam == "TransitionEarly":
        for group in [["Transition_Pusher"], ["Rim_Runner"], ["Corner_Specialist"]]:
            pick = _choose_best_role(offense, group)
            if pick: parts.append(pick)

    return parts

def _role_fit_effective_score(fits: List[float]) -> float:
    if not fits:
        return 50.0
    if len(fits) == 1:
        return float(fits[0])
    mn = min(fits)
    av = sum(fits) / len(fits)
    return clamp(0.70 * mn + 0.30 * av, 0.0, 100.0)

def _effective_grade_from_fit_eff(participants: List[Tuple[str, Player, float]], fit_eff: float) -> str:
    """Spec: grade is computed from Fit_eff. For multi-role, take the worst grade among participants."""
    if not participants:
        return "B"
    sev = {"S":0, "A":1, "B":2, "C":3, "D":4}
    grades = [role_fit_grade(r, fit_eff) for (r,_,_) in participants]
    return max(grades, key=lambda g: sev.get(g, 2))

def apply_role_fit_to_priors_and_tags(priors: Dict[str, float], action_family: str, offense: TeamState, tags: Dict[str, Any]) -> Dict[str, float]:
    strength = _get_role_fit_strength(offense)
    participants = _collect_roles_for_action_family(action_family, offense)
    applied = bool(participants)

    fits = [f for (_, _, f) in participants]
    fit_eff = _role_fit_effective_score(fits) if applied else 50.0
    grade = _effective_grade_from_fit_eff(participants, fit_eff) if applied else "B"

    mults_applied: List[float] = []
    if applied and strength > 1e-9:
        for o in list(priors.keys()):
            if o.startswith("FOUL_"):
                continue
            cat = "GOOD" if (o.startswith("SHOT_") or o.startswith("PASS_")) else "BAD" if (o.startswith("TO_") or o.startswith("RESET_")) else None
            if not cat:
                continue
            mult_raw = ROLE_PRIOR_MULT_RAW.get(grade, ROLE_PRIOR_MULT_RAW["B"])[cat]
            mult_final = 1.0 + (0.60 * strength) * (float(mult_raw) - 1.0)
            priors[o] *= mult_final
            mults_applied.append(mult_final)
        priors = normalize_weights(priors)

    avg_mult_final = (sum(mults_applied) / len(mults_applied)) if mults_applied else 1.0
    delta_raw = float(ROLE_LOGIT_DELTA_RAW.get(grade, 0.0))
    delta_final = (0.40 * strength) * delta_raw if applied else 0.0

    tags["role_fit_applied"] = bool(applied)
    tags["role_logit_delta"] = float(delta_final)
    tags["role_fit_eff"] = float(fit_eff)
    tags["role_fit_grade"] = str(grade)

    # internal debug (possession-step)
    if hasattr(offense, "role_fit_pos_log"):
        offense.role_fit_pos_log.append({
            "action_family": str(action_family),
            "applied": bool(applied),
            "n_roles": int(len(participants)),
            "fit_eff": float(fit_eff),
            "grade": str(grade),
            "role_fit_strength": float(strength),
            "avg_mult_final": float(avg_mult_final),
            "delta_final": float(delta_final),
        })

    # game-level aggregates (only when applied)
    if applied and hasattr(offense, "role_fit_grade_counts"):
        offense.role_fit_grade_counts[grade] = offense.role_fit_grade_counts.get(grade, 0) + 1
    if applied and hasattr(offense, "role_fit_role_counts"):
        for r, _, _ in participants:
            offense.role_fit_role_counts[r] = offense.role_fit_role_counts.get(r, 0) + 1

    return priors


