from __future__ import annotations

from typing import Dict, Optional

from .core import clamp, normalize_weights
from .profiles import ACTION_ALIASES, DEF_SCHEME_ACTION_WEIGHTS, OFF_SCHEME_ACTION_WEIGHTS
from .tactics import TacticsConfig


# -------------------------
# Builders
# -------------------------

def get_action_base(action: str) -> str:
    return ACTION_ALIASES.get(action, action)

def build_offense_action_probs(off_tac: TacticsConfig, def_tac: Optional[TacticsConfig] = None) -> Dict[str, float]:
    """Build offense action distribution.

    UI rule (fixed): normalize((W_scheme[action] ^ sharpness) * off_action_mult[action] * def_opp_action_mult[action]).
    """
    base = dict(OFF_SCHEME_ACTION_WEIGHTS.get(off_tac.offense_scheme, OFF_SCHEME_ACTION_WEIGHTS["Spread_HeavyPnR"]))
    sharp = clamp(off_tac.scheme_weight_sharpness, 0.70, 1.40)
    # 1) scheme sharpening first
    base = {a: (max(w, 0.0) ** sharp) for a, w in base.items()}
    # 2) offense UI multipliers
    for a, m in off_tac.action_weight_mult.items():
        base[a] = base.get(a, 0.5) * float(m)
    # 3) defense can distort opponent action choice (e.g., transition defense priority)
    if def_tac is not None:
        for a, m in getattr(def_tac, 'opp_action_weight_mult', {}).items():
            base[a] = base.get(a, 0.5) * float(m)
    return normalize_weights(base)

def build_defense_action_probs(tac: TacticsConfig) -> Dict[str, float]:
    """Build defense 'action' distribution (mostly for logging/feel).

    UI rule (fixed): normalize((Wdef_scheme[action] ^ sharpness) * def_action_mult[action]).
    """
    base = dict(DEF_SCHEME_ACTION_WEIGHTS.get(tac.defense_scheme, DEF_SCHEME_ACTION_WEIGHTS["Drop"]))
    sharp = clamp(tac.def_scheme_weight_sharpness, 0.70, 1.40)
    base = {a: (max(w, 0.0) ** sharp) for a, w in base.items()}
    for a, m in tac.def_action_weight_mult.items():
        base[a] = base.get(a, 0.5) * float(m)
    return normalize_weights(base)

def effective_scheme_multiplier(base_mult: float, strength: float) -> float:
    s = clamp(strength, 0.70, 1.40)
    return 1.0 + (float(base_mult) - 1.0) * s

def build_outcome_priors(action: str, off_tac: TacticsConfig, def_tac: TacticsConfig, tags: Dict[str, Any]) -> Dict[str, float]:
    base_action = get_action_base(action)
    pri = dict(ACTION_OUTCOME_PRIORS.get(base_action, ACTION_OUTCOME_PRIORS["SpotUp"]))

    # offense global
    pri = apply_multipliers(pri, off_tac.outcome_global_mult)

    # offense per-action
    pri = apply_multipliers_typesafe(pri, off_tac.outcome_by_action_mult.get(action, {}))
    pri = apply_multipliers_typesafe(pri, off_tac.outcome_by_action_mult.get(base_action, {}))

    # offense scheme
    sm = OFFENSE_SCHEME_MULT.get(off_tac.offense_scheme, {}).get(action) or OFFENSE_SCHEME_MULT.get(off_tac.offense_scheme, {}).get(base_action) or {}
    for o, m in sm.items():
        if o in pri:
            pri[o] *= effective_scheme_multiplier(m, off_tac.scheme_outcome_strength)

    # defense knobs on opponent priors
    pri = apply_multipliers(pri, def_tac.opp_outcome_global_mult)
    pri = apply_multipliers_typesafe(pri, def_tac.opp_outcome_by_action_mult.get(action, {}))
    pri = apply_multipliers_typesafe(pri, def_tac.opp_outcome_by_action_mult.get(base_action, {}))

    # defense scheme
    dm = DEFENSE_SCHEME_MULT.get(def_tac.defense_scheme, {}).get(action) or DEFENSE_SCHEME_MULT.get(def_tac.defense_scheme, {}).get(base_action) or {}
    for o, m in dm.items():
        if o in pri:
            pri[o] *= effective_scheme_multiplier(m, def_tac.def_scheme_outcome_strength)

    # conditional (MVP subset)
    if def_tac.defense_scheme == "ICE_SidePnR" and not tags.get("is_side_pnr", False):
        for o in ("RESET_RESREEN","PASS_KICKOUT"):
            if o in pri:
                pri[o] *= 1.03

    if tags.get("in_transition", False):
        for o in ("TO_BAD_PASS","TO_HANDLE_LOSS","TO_CHARGE","RESET_HUB","RESET_RESREEN"):
            if o in pri:
                pri[o] *= 0.92

    if def_tac.defense_scheme == "Blitz_TrapPnR" and base_action == "PnR":
        pri["PASS_SHORTROLL"] = max(pri.get("PASS_SHORTROLL", 0.0), 0.10)
        # reach foul can show up vs blitz
        pri["FOUL_REACH_TRAP"] = pri.get("FOUL_REACH_TRAP", 0.0) + 0.02

    return normalize_weights(pri)

def apply_multipliers_typesafe(pri: Dict[str, float], mults: Dict[str, float]) -> Dict[str, float]:
    out = dict(pri)
    for o, m in mults.items():
        if o in out:
            out[o] *= float(m)
    return out


