
"""
match_engine_mvp.py

Basketball tactics-driven match engine MVP
-----------------------------------------
End-to-end possession simulator using:
- offense scheme action weights + UI multipliers
- defense scheme action weights + UI multipliers (for intensity/logging)
- action -> outcome priors, distorted by offense + defense schemes and UI knobs
- outcome resolution using derived abilities (0~100)

MVP traits:
- no full shot-clock model (simple reset cap)
- no bonus/free throw rules (basic shooting foul + side-out trap foul)
- no lineup rotations (single lineup)
- lightweight fatigue (affects abilities slightly)

You can integrate by:
- feeding real Player derived stats + user-selected roles + tactics knobs
- running simulate_game() and consuming the returned dict
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math, random, json, hashlib, pickle, os, copy


ENGINE_VERSION: str = "mvp_plus_0.2"

def make_replay_token(rng: random.Random, teamA: 'TeamState', teamB: 'TeamState', era: str = "default") -> str:
    """Create a short stable token to reproduce/debug a game.

    Token is derived from: engine version, era, RNG state hash, rosters, roles, and tactics.
    """
    try:
        state_bytes = pickle.dumps(rng.getstate())
        rng_hash = hashlib.sha256(state_bytes).hexdigest()
    except Exception:
        rng_hash = "no_state"

    def _player_payload(p: 'Player') -> Dict[str, Any]:
        # Keep it deterministic; derived is already 0~100 numbers
        return {
            'pid': p.pid,
            'pos': p.pos,
            'derived': p.derived,
        }

    def _tactics_payload(t: 'TacticsConfig') -> Dict[str, Any]:
        return {
            'offense_scheme': t.offense_scheme,
            'defense_scheme': t.defense_scheme,
            'scheme_weight_sharpness': t.scheme_weight_sharpness,
            'scheme_outcome_strength': t.scheme_outcome_strength,
            'def_scheme_weight_sharpness': t.def_scheme_weight_sharpness,
            'def_scheme_outcome_strength': t.def_scheme_outcome_strength,
            'action_weight_mult': t.action_weight_mult,
            'outcome_global_mult': t.outcome_global_mult,
            'outcome_by_action_mult': t.outcome_by_action_mult,
            'def_action_weight_mult': t.def_action_weight_mult,
            'opp_action_weight_mult': getattr(t, 'opp_action_weight_mult', {}),
            'opp_outcome_global_mult': t.opp_outcome_global_mult,
            'opp_outcome_by_action_mult': t.opp_outcome_by_action_mult,
            'context': t.context,
        }

    payload = {
        'engine_version': ENGINE_VERSION,
        'era': era,
        'rng_state_hash': rng_hash,
        'teamA': {
            'name': teamA.name,
            'roles': teamA.roles,
            'lineup': [_player_payload(p) for p in teamA.lineup],
            'tactics': _tactics_payload(teamA.tactics),
        },
        'teamB': {
            'name': teamB.name,
            'roles': teamB.roles,
            'lineup': [_player_payload(p) for p in teamB.lineup],
            'tactics': _tactics_payload(teamB.tactics),
        },
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()[:12]

# -------------------------
# Helpers
# -------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def normalize_weights(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(v, 0.0) for v in d.values())
    if s <= 1e-12:
        n = len(d) if d else 1
        return {k: 1.0 / n for k in d} if d else {}
    return {k: max(v, 0.0) / s for k, v in d.items()}

def weighted_choice(rng: random.Random, weights: Dict[str, float]) -> str:
    total = sum(max(w, 0.0) for w in weights.values())
    if total <= 1e-12:
        return next(iter(weights.keys()))
    r = rng.random() * total
    upto = 0.0
    for k, w in weights.items():
        w = max(w, 0.0)
        upto += w
        if upto >= r:
            return k
    return next(iter(weights.keys()))

def dot_profile(vals: Dict[str, float], profile: Dict[str, float], missing_default: float = 50.0) -> float:
    s = 0.0
    for k, w in profile.items():
        s += float(vals.get(k, missing_default)) * float(w)
    return s

def apply_multipliers(base: Dict[str, float], mults: Dict[str, float]) -> Dict[str, float]:
    out = dict(base)
    for k, m in mults.items():
        if k in out:
            out[k] *= float(m)
    return out


# -------------------------
# Core Data Models
# -------------------------

DERIVED_DEFAULT = 50.0

@dataclass
class Player:
    pid: str
    name: str
    pos: str = "G"
    derived: Dict[str, float] = field(default_factory=dict)
    fatigue: float = 0.0  # 0 fresh -> 100 exhausted

    def get(self, key: str, fatigue_sensitive: bool = True) -> float:
        v = float(self.derived.get(key, DERIVED_DEFAULT))
        if not fatigue_sensitive:
            return v
        # Fatigue model: 0..100 -> factor 1.00..0.82
        f = clamp(1.0 - (self.fatigue / 560.0), 0.82, 1.0)
        return v * f

    def add_fatigue(self, cost: float) -> None:
        endu = float(self.derived.get("ENDURANCE", DERIVED_DEFAULT))
        gain = cost * (1.12 - (endu / 220.0))  # ENDURANCE=100 -> ~0.67x cost
        self.fatigue = clamp(self.fatigue + gain, 0.0, 100.0)

@dataclass
class TeamState:
    name: str
    lineup: List[Player]
    roles: Dict[str, str]  # role -> pid (chosen via UI)
    tactics: "TacticsConfig"

    # team totals
    pts: int = 0
    fgm: int = 0
    fga: int = 0
    tpm: int = 0
    tpa: int = 0
    ftm: int = 0
    fta: int = 0
    tov: int = 0
    orb: int = 0
    drb: int = 0
    possessions: int = 0

    # breakdowns
    off_action_counts: Dict[str, int] = field(default_factory=dict)
    def_action_counts: Dict[str, int] = field(default_factory=dict)
    outcome_counts: Dict[str, int] = field(default_factory=dict)

    # player box
    player_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # internal debug (role fit)
    role_fit_pos_log: List[Dict[str, Any]] = field(default_factory=list)
    role_fit_role_counts: Dict[str, int] = field(default_factory=dict)
    role_fit_grade_counts: Dict[str, int] = field(default_factory=dict)
    role_fit_bad_totals: Dict[str, int] = field(default_factory=dict)  # {'TO': n, 'RESET': n}
    role_fit_bad_by_grade: Dict[str, Dict[str, int]] = field(default_factory=dict)  # grade -> {'TO': n, 'RESET': n}

    def find_player(self, pid: str) -> Optional[Player]:
        for p in self.lineup:
            if p.pid == pid:
                return p
        return None

    def add_player_stat(self, pid: str, key: str, inc: int = 1) -> None:
        if pid not in self.player_stats:
            self.player_stats[pid] = {"PTS":0,"FGM":0,"FGA":0,"3PM":0,"3PA":0,"FTM":0,"FTA":0,"TOV":0,"ORB":0,"DRB":0}
        self.player_stats[pid][key] = self.player_stats[pid].get(key, 0) + inc

    def get_role_player(self, role: str, fallback_rank_key: Optional[str] = None) -> Player:
        pid = self.roles.get(role)
        if pid:
            p = self.find_player(pid)
            if p:
                return p
        if fallback_rank_key:
            return max(self.lineup, key=lambda x: x.get(fallback_rank_key))
        return self.lineup[0]


# -------------------------
# Minimal role ranking keys (for fallbacks)
# -------------------------

ROLE_FALLBACK_RANK = {
    "ball_handler": "PNR_READ",
    "secondary_handler": "PASS_CREATE",
    "screener": "SHORTROLL_PLAY",
    "post": "POST_SCORE",
    "shooter": "SHOT_3_CS",
    "cutter": "FIRST_STEP",
    "rim_runner": "FIN_DUNK",
}


# -------------------------
# Outcome resolution profiles (derived ability weights)
# -------------------------

OUTCOME_PROFILES: Dict[str, Dict[str, Dict[str, float]]] = {
    "SHOT_RIM_LAYUP": {
        "offense": {"FIN_RIM":0.55, "FIN_CONTACT":0.15, "SHOT_TOUCH":0.10, "HANDLE_SAFE":0.10, "ENDURANCE":0.10},
        "defense": {"DEF_RIM":0.45, "DEF_HELP":0.25, "PHYSICAL":0.15, "DEF_POA":0.10, "ENDURANCE":0.05},
    },
    "SHOT_RIM_DUNK": {
        "offense": {"FIN_DUNK":0.55, "FIN_CONTACT":0.20, "FIN_RIM":0.10, "HANDLE_SAFE":0.05, "ENDURANCE":0.10},
        "defense": {"DEF_RIM":0.50, "PHYSICAL":0.20, "DEF_HELP":0.20, "ENDURANCE":0.10},
    },
    "SHOT_RIM_CONTACT": {
        "offense": {"FIN_CONTACT":0.55, "FIN_RIM":0.20, "SHOT_TOUCH":0.10, "PHYSICAL":0.10, "ENDURANCE":0.05},
        "defense": {"DEF_RIM":0.40, "PHYSICAL":0.30, "DEF_HELP":0.20, "DEF_POST":0.10},
    },
    "SHOT_TOUCH_FLOATER": {
        "offense": {"SHOT_TOUCH":0.55, "FIN_RIM":0.15, "FIN_CONTACT":0.10, "DRIVE_CREATE":0.10, "ENDURANCE":0.10},
        "defense": {"DEF_RIM":0.30, "DEF_HELP":0.35, "DEF_POA":0.15, "PHYSICAL":0.10, "ENDURANCE":0.10},
    },
    "SHOT_MID_CS": {
        "offense": {"SHOT_MID_CS":0.85, "ENDURANCE":0.15},
        "defense": {"DEF_POA":0.35, "DEF_HELP":0.35, "ENDURANCE":0.20, "PHYSICAL":0.10},
    },
    "SHOT_3_CS": {
        "offense": {"SHOT_3_CS":0.85, "ENDURANCE":0.15},
        "defense": {"DEF_POA":0.35, "DEF_HELP":0.35, "ENDURANCE":0.25, "PHYSICAL":0.05},
    },
    "SHOT_MID_PU": {
        "offense": {"SHOT_MID_PU":0.65, "HANDLE_SAFE":0.15, "FIRST_STEP":0.10, "ENDURANCE":0.10},
        "defense": {"DEF_POA":0.50, "DEF_HELP":0.25, "ENDURANCE":0.15, "PHYSICAL":0.10},
    },
    "SHOT_3_OD": {
        "offense": {"SHOT_3_OD":0.60, "HANDLE_SAFE":0.20, "FIRST_STEP":0.10, "ENDURANCE":0.10},
        "defense": {"DEF_POA":0.55, "DEF_HELP":0.20, "ENDURANCE":0.20, "PHYSICAL":0.05},
    },
    "SHOT_POST": {
        "offense": {"POST_SCORE":0.40, "POST_CONTROL":0.20, "FIN_CONTACT":0.20, "SHOT_TOUCH":0.10, "PHYSICAL":0.10},
        "defense": {"DEF_POST":0.55, "DEF_HELP":0.20, "PHYSICAL":0.20, "DEF_RIM":0.05},
    },

    "PASS_KICKOUT": {
        "offense": {"PASS_CREATE":0.45, "PASS_SAFE":0.35, "PNR_READ":0.20},
        "defense": {"DEF_STEAL":0.55, "DEF_HELP":0.30, "DEF_POA":0.15},
    },
    "PASS_EXTRA": {
        "offense": {"PASS_SAFE":0.55, "PASS_CREATE":0.30, "PNR_READ":0.15},
        "defense": {"DEF_STEAL":0.50, "DEF_HELP":0.35, "ENDURANCE":0.15},
    },
    "PASS_SKIP": {
        "offense": {"PASS_CREATE":0.60, "PASS_SAFE":0.25, "PNR_READ":0.15},
        "defense": {"DEF_STEAL":0.55, "DEF_HELP":0.35, "DEF_POA":0.10},
    },
    "PASS_SHORTROLL": {
        "offense": {"SHORTROLL_PLAY":0.55, "PASS_SAFE":0.25, "PASS_CREATE":0.20},
        "defense": {"DEF_HELP":0.45, "DEF_STEAL":0.30, "ENDURANCE":0.25},
    },

    "TO_HANDLE_LOSS": {
        "offense": {"HANDLE_SAFE":0.60, "DRIVE_CREATE":0.20, "ENDURANCE":0.20},
        "defense": {"DEF_STEAL":0.50, "DEF_POA":0.30, "DEF_HELP":0.20}
    },
    "TO_BAD_PASS": {
        "offense": {"PASS_SAFE":0.55, "PASS_CREATE":0.25, "PNR_READ":0.20},
        "defense": {"DEF_STEAL":0.55, "DEF_HELP":0.30, "DEF_POA":0.15}
    },
    "TO_CHARGE": {
        "offense": {"DRIVE_CREATE":0.35, "PHYSICAL":0.35, "PNR_READ":0.15, "ENDURANCE":0.15},
        "defense": {"DEF_POA":0.40, "DEF_HELP":0.35, "PHYSICAL":0.25}
    },
    "TO_SHOT_CLOCK": {
        "offense": {"PNR_READ":0.35, "PASS_CREATE":0.25, "DRIVE_CREATE":0.20, "HANDLE_SAFE":0.10, "ENDURANCE":0.10},
        "defense": {"DEF_POA":0.35, "DEF_HELP":0.35, "ENDURANCE":0.20, "PHYSICAL":0.10}
    },

    "FOUL_DRAW_RIM": {
        "offense": {"FIN_CONTACT":0.60, "FIN_RIM":0.15, "PHYSICAL":0.15, "ENDURANCE":0.10},
        "defense": {"DEF_RIM":0.40, "PHYSICAL":0.25, "DEF_HELP":0.25, "ENDURANCE":0.10}
    },
    "FOUL_DRAW_POST": {
        "offense": {"FIN_CONTACT":0.40, "POST_SCORE":0.25, "PHYSICAL":0.20, "POST_CONTROL":0.15},
        "defense": {"DEF_POST":0.45, "PHYSICAL":0.35, "DEF_HELP":0.20}
    },
    "FOUL_DRAW_JUMPER": {
        "offense": {"SHOT_3_OD":0.30, "SHOT_MID_PU":0.30, "HANDLE_SAFE":0.20, "ENDURANCE":0.20},
        "defense": {"DEF_POA":0.45, "ENDURANCE":0.35, "PHYSICAL":0.20}
    },
    "FOUL_REACH_TRAP": {
        "offense": {"HANDLE_SAFE":0.35, "PASS_SAFE":0.35, "PNR_READ":0.20, "ENDURANCE":0.10},
        "defense": {"DEF_STEAL":0.45, "PHYSICAL":0.25, "ENDURANCE":0.30}
    },

    "RESET_HUB": {
        "offense": {"PASS_SAFE":0.55, "PNR_READ":0.25, "ENDURANCE":0.20},
        "defense": {"DEF_HELP":0.45, "DEF_STEAL":0.25, "ENDURANCE":0.30}
    },
    "RESET_RESREEN": {
        "offense": {"PNR_READ":0.35, "HANDLE_SAFE":0.20, "ENDURANCE":0.25, "PASS_SAFE":0.20},
        "defense": {"DEF_POA":0.35, "DEF_HELP":0.35, "ENDURANCE":0.30}
    },
    "RESET_REDO_DHO": {
        "offense": {"HANDLE_SAFE":0.30, "PASS_SAFE":0.30, "ENDURANCE":0.25, "PNR_READ":0.15},
        "defense": {"DEF_POA":0.40, "DEF_STEAL":0.20, "ENDURANCE":0.40}
    },
    "RESET_POST_OUT": {
        "offense": {"POST_CONTROL":0.35, "PASS_SAFE":0.40, "PASS_CREATE":0.15, "PHYSICAL":0.10},
        "defense": {"DEF_POST":0.40, "DEF_STEAL":0.30, "DEF_HELP":0.30}
    },
}

SHOT_BASE = {
    "SHOT_RIM_LAYUP": 0.56,
    "SHOT_RIM_DUNK": 0.70,
    "SHOT_RIM_CONTACT": 0.47,
    "SHOT_TOUCH_FLOATER": 0.41,
    "SHOT_MID_CS": 0.43,
    "SHOT_MID_PU": 0.41,
    "SHOT_3_CS": 0.36,
    "SHOT_3_OD": 0.33,
    "SHOT_POST": 0.50,
}
PASS_BASE_SUCCESS = {
    "PASS_KICKOUT": 0.92,
    "PASS_EXTRA": 0.93,
    "PASS_SKIP": 0.90,
    "PASS_SHORTROLL": 0.88,
}


# -------------------------
# Scheme action weights
# -------------------------

OFF_SCHEME_ACTION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Spread_HeavyPnR": {"PnR":28, "SideAnglePnR":10, "DoubleDrag":8, "Rescreen":5, "SlipScreen":4, "SpainPnR":4, "ShortRollPlay":6,
                       "Drive":8, "Kickout":8, "ExtraPass":6, "SpotUp":8, "Cut":5},
    "Drive_Kick": {"Drive":30, "Kickout":18, "ExtraPass":12, "Relocation":8, "SpotUp":12, "Cut":6, "SkipPass":5, "Hammer":4, "PnR":3, "DHO":2},
    "FiveOut": {"Drive":18, "SpotUp":16, "Kickout":14, "ExtraPass":10, "Relocation":10, "Cut":10, "DHO":8, "ZoomDHO":6, "PnR":5, "SlipScreen":3},
    "Motion_SplitCut": {"ElbowHub":12, "OffBallScreen":14, "ScreenTheScreener_STS":6, "Cut":18, "PostSplit":10, "DHO":8,
                        "Drive":10, "Kickout":6, "ExtraPass":6, "SpotUp":6, "PnR":4},
    "DHO_Chicago": {"Chicago":18, "DHO":16, "ZoomDHO":8, "ReDHO_Handback":6, "Drive":12, "Kickout":10, "ExtraPass":6,
                    "SpotUp":10, "PnR":6, "SlipScreen":4, "OffBallScreen":4},
    "Post_InsideOut": {"PostEntry":12, "PostUp":22, "Kickout":14, "ExtraPass":8, "SpotUp":12, "Cut":8, "PostSplit":10, "HighLow":6, "Drive":4, "DHO":4},
    "Horns_Elbow": {"HornsSet":18, "ElbowHub":12, "PnR":12, "DHO":8, "HighLow":10, "Drive":10, "Kickout":8, "ExtraPass":6, "SpotUp":8, "Cut":6, "SpainPnR":2},
    "Transition_Early": {"TransitionEarly":40, "DragScreen":14, "DoubleDrag":8, "SecondaryBreak":10, "Drive":8, "Kickout":8, "SpotUp":8, "QuickPost":4},
}

DEF_SCHEME_ACTION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Drop": {"DropCoverage":34, "GoOver":18, "GoUnder":6, "ContainOnBall":10, "LowManTagRoll":10, "StuntAndRecover":8, "CloseoutControl":6, "RimProtectVertical":6, "BoxOutRebound":2},
    "Switch_Everything": {"Switch":38, "ContainOnBall":16, "CloseoutControl":10, "StuntAndRecover":8, "XOutRecover":6, "FrontPost":8, "PostDouble":4, "RimProtectVertical":4, "BoxOutRebound":6},
    "Hedge_ShowRecover": {"HedgeShow":26, "XOutRecover":16, "GoOver":18, "ContainOnBall":10, "LowManTagRoll":10, "StuntAndRecover":8, "CloseoutControl":6, "RimProtectVertical":4, "BoxOutRebound":2},
    "Blitz_TrapPnR": {"BlitzTrap":28, "RotateXOut":14, "StuntAndRecover":12, "CloseoutControl":10, "ContainOnBall":6, "RimProtectVertical":6, "LowManTagRoll":6, "BoxOutRebound":4, "XOutRecover":14},
    "ICE_SidePnR": {"ICEForceBaseline":26, "GoOver":18, "ContainOnBall":12, "DropCoverage":10, "NailHelp":10, "LowManTagRoll":10, "StuntAndRecover":6, "CloseoutControl":6, "RimProtectVertical":2},
    "Zone": {"ZoneShift":28, "ZoneCloseout":18, "ZoneBumpCutter":12, "ProtectPaintFirst":12, "StuntAndRecover":8, "RotateXOut":8, "RimProtectVertical":6, "BoxOutRebound":8},
    "PackLine_GapHelp": {"GapHelp":24, "ContainOnBall":16, "StuntAndRecover":14, "CloseoutControl":10, "ProtectPaintFirst":10, "LowManTagRoll":10, "RimProtectVertical":6, "FrontPost":4, "BoxOutRebound":6},
}


# -------------------------
# Action outcome priors (include fouls)
# -------------------------

ACTION_OUTCOME_PRIORS: Dict[str, Dict[str, float]] = {
    "PnR": {
        "PASS_SHORTROLL":0.13, "PASS_KICKOUT":0.17, "SHOT_3_OD":0.11, "SHOT_MID_PU":0.09,
        "SHOT_RIM_LAYUP":0.11, "SHOT_RIM_DUNK":0.04, "SHOT_3_CS":0.10,
        "FOUL_DRAW_RIM":0.03, "FOUL_DRAW_JUMPER":0.01,
        "TO_HANDLE_LOSS":0.07, "TO_BAD_PASS":0.05, "RESET_RESREEN":0.09
    },
    "DHO": {
        "SHOT_3_OD":0.13, "SHOT_MID_PU":0.09, "SHOT_RIM_LAYUP":0.09,
        "PASS_KICKOUT":0.16, "PASS_EXTRA":0.12, "SHOT_3_CS":0.14,
        "FOUL_DRAW_JUMPER":0.01, "FOUL_DRAW_RIM":0.02,
        "TO_HANDLE_LOSS":0.08, "TO_BAD_PASS":0.06, "RESET_REDO_DHO":0.10
    },
    "Drive": {
        "SHOT_RIM_LAYUP":0.20, "SHOT_RIM_DUNK":0.05, "SHOT_RIM_CONTACT":0.07, "SHOT_TOUCH_FLOATER":0.08,
        "PASS_KICKOUT":0.20, "PASS_EXTRA":0.09,
        "FOUL_DRAW_RIM":0.08,
        "TO_CHARGE":0.06, "TO_HANDLE_LOSS":0.08, "RESET_HUB":0.09
    },
    "Kickout": {
        "SHOT_3_CS":0.40, "SHOT_MID_CS":0.10, "PASS_EXTRA":0.24, "PASS_SKIP":0.08,
        "FOUL_DRAW_JUMPER":0.02,
        "TO_BAD_PASS":0.06, "RESET_HUB":0.10
    },
    "ExtraPass": {
        "SHOT_3_CS":0.43, "SHOT_MID_CS":0.08, "PASS_EXTRA":0.18, "PASS_SKIP":0.12,
        "FOUL_DRAW_JUMPER":0.02,
        "TO_BAD_PASS":0.07, "RESET_HUB":0.10
    },
    "PostUp": {
        "SHOT_POST":0.24, "SHOT_RIM_CONTACT":0.08,
        "PASS_KICKOUT":0.25, "PASS_EXTRA":0.12, "PASS_SKIP":0.08,
        "FOUL_DRAW_POST":0.07,
        "TO_BAD_PASS":0.07, "TO_HANDLE_LOSS":0.03, "RESET_POST_OUT":0.06
    },
    "HornsSet": {
        "PASS_KICKOUT":0.16, "SHOT_MID_CS":0.10, "SHOT_3_CS":0.14, "PASS_EXTRA":0.18,
        "FOUL_DRAW_JUMPER":0.01,
        "TO_BAD_PASS":0.06, "RESET_HUB":0.35
    },
    "SpotUp": {
        "SHOT_3_CS":0.68, "SHOT_MID_CS":0.20, "FOUL_DRAW_JUMPER":0.02, "TO_BAD_PASS":0.02, "RESET_HUB":0.08
    },
    "Cut": {
        "SHOT_RIM_LAYUP":0.34, "SHOT_RIM_DUNK":0.07, "SHOT_RIM_CONTACT":0.09,
        "FOUL_DRAW_RIM":0.05,
        "PASS_KICKOUT":0.14, "TO_BAD_PASS":0.06, "TO_HANDLE_LOSS":0.04, "RESET_HUB":0.21
    },
    "TransitionEarly": {
        "SHOT_RIM_LAYUP":0.18, "SHOT_RIM_DUNK":0.13, "SHOT_3_CS":0.18,
        "FOUL_DRAW_RIM":0.06,
        "PASS_KICKOUT":0.18, "TO_HANDLE_LOSS":0.07, "TO_BAD_PASS":0.05, "RESET_HUB":0.15
    }
}

ACTION_ALIASES = {
    "DragScreen": "PnR",
    "DoubleDrag": "PnR",
    "Rescreen": "PnR",
    "SideAnglePnR": "PnR",
    "SlipScreen": "PnR",
    "SpainPnR": "PnR",
    "ShortRollPlay": "PnR",
    "ZoomDHO": "DHO",
    "ReDHO_Handback": "DHO",
    "Chicago": "DHO",
    "Relocation": "SpotUp",
    "SkipPass": "ExtraPass",
    "Hammer": "Kickout",
    "PostEntry": "PostUp",
    "PostSplit": "Cut",
    "HighLow": "PostUp",
    "ElbowHub": "HornsSet",
    "OffBallScreen": "Cut",
    "ScreenTheScreener_STS": "Cut",
    "SecondaryBreak": "TransitionEarly",
    "QuickPost": "PostUp",
}


# -------------------------
# Distortion multipliers (schemes) - same as MVP v0
# -------------------------

OFFENSE_SCHEME_MULT: Dict[str, Dict[str, Dict[str, float]]] = {
    "Spread_HeavyPnR": {"PnR": {"PASS_SHORTROLL":1.10, "PASS_KICKOUT":1.05, "SHOT_3_OD":1.10, "SHOT_MID_PU":1.05, "RESET_RESREEN":1.05}},
    "Drive_Kick": {"Drive": {"PASS_KICKOUT":1.25, "PASS_EXTRA":1.15, "SHOT_RIM_LAYUP":0.90},
                   "Kickout": {"SHOT_3_CS":1.12, "PASS_EXTRA":1.08, "PASS_SKIP":1.05},
                   "ExtraPass": {"SHOT_3_CS":1.10, "PASS_SKIP":1.08}},
    "FiveOut": {"Drive": {"PASS_KICKOUT":1.10, "PASS_EXTRA":1.10, "SHOT_RIM_LAYUP":0.95},
                "Kickout": {"SHOT_3_CS":1.15, "PASS_SKIP":1.10},
                "ExtraPass": {"SHOT_3_CS":1.15, "PASS_SKIP":1.12},
                "Cut": {"SHOT_RIM_LAYUP":1.08, "RESET_HUB":0.95},
                "PostUp": {"SHOT_POST":0.80}},
    "Motion_SplitCut": {"Cut": {"SHOT_RIM_LAYUP":1.18, "PASS_KICKOUT":1.05, "RESET_HUB":0.95},
                        "ExtraPass": {"PASS_EXTRA":1.10, "SHOT_3_CS":1.05},
                        "DHO": {"RESET_REDO_DHO":0.95, "PASS_KICKOUT":1.05},
                        "PnR": {"SHOT_3_OD":0.90, "SHOT_MID_PU":0.95}},
    "DHO_Chicago": {"DHO": {"SHOT_3_OD":1.10, "SHOT_MID_PU":1.05, "RESET_REDO_DHO":0.95},
                    "Chicago": {"SHOT_3_CS":1.10, "SHOT_3_OD":1.05, "PASS_KICKOUT":1.05},
                    "Drive": {"SHOT_RIM_LAYUP":1.05}},
    "Post_InsideOut": {"PostUp": {"SHOT_POST":1.20, "PASS_KICKOUT":1.05, "FOUL_DRAW_POST":1.10, "RESET_POST_OUT":0.95},
                       "ExtraPass": {"SHOT_3_CS":1.05}},
    "Horns_Elbow": {"HornsSet": {"RESET_HUB":0.95, "PASS_EXTRA":1.05, "SHOT_MID_CS":1.10, "PASS_KICKOUT":1.05},
                    "PnR": {"PASS_SHORTROLL":1.05},
                    "HighLow": {"SHOT_POST":1.05, "SHOT_RIM_CONTACT":1.05}},
    "Transition_Early": {"TransitionEarly": {"SHOT_RIM_DUNK":1.15, "SHOT_3_CS":1.10, "RESET_HUB":0.85}},
}

DEFENSE_SCHEME_MULT: Dict[str, Dict[str, Dict[str, float]]] = {
    "Drop": {"PnR": {"SHOT_MID_PU":1.35, "SHOT_3_OD":1.15, "PASS_SHORTROLL":0.75, "SHOT_RIM_LAYUP":0.85, "SHOT_RIM_DUNK":0.85, "RESET_RESREEN":1.05},
             "Drive": {"SHOT_RIM_LAYUP":0.90}},
    "Switch_Everything": {"PnR": {"RESET_RESREEN":1.25, "TO_SHOT_CLOCK":1.15, "PASS_SHORTROLL":0.85, "SHOT_3_OD":1.10},
                          "DHO": {"RESET_REDO_DHO":1.15, "TO_HANDLE_LOSS":1.10},
                          "PostUp": {"SHOT_POST":1.35, "FOUL_DRAW_POST":1.20},
                          "Drive": {"TO_CHARGE":1.10}},
    "Hedge_ShowRecover": {"PnR": {"PASS_SHORTROLL":1.25, "PASS_KICKOUT":1.10, "RESET_RESREEN":1.10},
                          "Drive": {"SHOT_TOUCH_FLOATER":1.10}},
    "Blitz_TrapPnR": {"PnR": {"PASS_SHORTROLL":1.55, "PASS_KICKOUT":1.20, "SHOT_3_OD":0.75, "SHOT_MID_PU":0.75, "TO_BAD_PASS":1.35, "TO_HANDLE_LOSS":1.20, "FOUL_REACH_TRAP":1.20, "RESET_HUB":1.15},
                      "DHO": {"TO_BAD_PASS":1.20, "RESET_REDO_DHO":1.10},
                      "Drive": {"TO_HANDLE_LOSS":1.10}},
    "ICE_SidePnR": {"PnR": {"RESET_RESREEN":1.10, "PASS_KICKOUT":1.10, "SHOT_MID_PU":0.85, "SHOT_TOUCH_FLOATER":1.15}},
    "Zone": {"Drive": {"SHOT_RIM_LAYUP":0.75, "PASS_EXTRA":1.25, "PASS_SKIP":1.30, "SHOT_3_CS":1.15, "TO_BAD_PASS":1.10},
             "Kickout": {"PASS_EXTRA":1.15, "TO_BAD_PASS":1.08},
             "PostUp": {"SHOT_POST":0.85, "PASS_SKIP":1.15},
             "HornsSet": {"SHOT_MID_CS":1.15}},
    "PackLine_GapHelp": {"Drive": {"SHOT_RIM_LAYUP":0.65, "SHOT_RIM_DUNK":0.70, "PASS_KICKOUT":1.25, "PASS_EXTRA":1.20, "SHOT_3_CS":1.20, "TO_CHARGE":1.15},
                         "PnR": {"PASS_KICKOUT":1.15, "SHOT_MID_PU":1.05},
                         "ExtraPass": {"TO_BAD_PASS":1.05}},
}


# -------------------------
# Era / Parameter externalization (0-1)
# -------------------------
# Commercial goal: make tuning possible WITHOUT touching code.
# We externalize priors, scheme weights/multipliers, shot/pass bases, and prob model parameters into a JSON "era" file.

from pathlib import Path

DEFAULT_PROB_MODEL: Dict[str, float] = {
    # Generic success-prob model clamps
    "base_p_min": 0.02,
    "base_p_max": 0.98,
    "prob_min": 0.03,
    "prob_max": 0.97,

    # OffScore-DefScore scaling (bigger = less sensitive)
    "shot_scale": 18.0,
    "pass_scale": 20.0,
    "rebound_scale": 22.0,

    # ORB baseline used in rebound_orb_probability()
    "orb_base": 0.26,

    # FT model used in resolve_free_throws()
    "ft_base": 0.45,
    "ft_range": 0.47,
    "ft_min": 0.40,
    "ft_max": 0.95,
}

ERA_PROB_MODEL: Dict[str, float] = dict(DEFAULT_PROB_MODEL)

# Logistic parameters by outcome kind (2-1, 2-2)
# NOTE: 'scale' and 'sensitivity' are redundant (sensitivity ~= 1/scale). We keep both for readability.
DEFAULT_LOGISTIC_PARAMS: Dict[str, Dict[str, float]] = {
    "default": {"scale": 18.0, "sensitivity": 1.0 / 18.0},

    # 2-2 table (user-provided)
    "shot_3":   {"scale": 30.0, "sensitivity": 1.0 / 30.0},   # 3PT make
    "shot_mid": {"scale": 24.0, "sensitivity": 1.0 / 24.0},   # midrange make
    "shot_rim": {"scale": 18.0, "sensitivity": 1.0 / 18.0},   # rim finishes
    "shot_post":{"scale": 20.0, "sensitivity": 1.0 / 20.0},   # post shots
    "pass":     {"scale": 28.0, "sensitivity": 1.0 / 28.0},   # pass success
    "rebound":  {"scale": 22.0, "sensitivity": 1.0 / 22.0},   # ORB% model (legacy)
    "turnover": {"scale": 24.0, "sensitivity": 1.0 / 24.0},   # reserved (TO is prior-only)
}

# Variance knob (2-3): logit-space Gaussian noise, so mean stays roughly stable.
DEFAULT_VARIANCE_PARAMS: Dict[str, Any] = {
    "logit_noise_std": 0.18,  # global volatility
    "kind_mult": {
        "shot_3": 1.15,
        "shot_mid": 1.05,
        "shot_rim": 0.95,
        "shot_post": 1.00,
        "pass": 0.85,
        "rebound": 0.60,
    },
    # optional per-team multiplier range (clamped)
    "team_mult_lo": 0.70,
    "team_mult_hi": 1.40,
}

ERA_LOGISTIC_PARAMS: Dict[str, Dict[str, float]] = dict(DEFAULT_LOGISTIC_PARAMS)
ERA_VARIANCE_PARAMS: Dict[str, Any] = copy.deepcopy(DEFAULT_VARIANCE_PARAMS)

DEFAULT_ROLE_FIT = {"default_strength": 0.65}
ERA_ROLE_FIT: Dict[str, Any] = copy.deepcopy(DEFAULT_ROLE_FIT)


# Snapshot built-in defaults (used as fallback if era json is missing keys)
DEFAULT_ERA: Dict[str, Any] = {
    "name": "builtin_default",
    "version": "1.0",
    "knobs": {"mult_lo": 0.70, "mult_hi": 1.40},
    "prob_model": dict(DEFAULT_PROB_MODEL),

    "logistic_params": copy.deepcopy(DEFAULT_LOGISTIC_PARAMS),
    "variance_params": copy.deepcopy(DEFAULT_VARIANCE_PARAMS),

    "role_fit": {"default_strength": 0.65},

    "shot_base": dict(SHOT_BASE),
    "pass_base_success": dict(PASS_BASE_SUCCESS),

    "action_outcome_priors": copy.deepcopy(ACTION_OUTCOME_PRIORS),
    "action_aliases": dict(ACTION_ALIASES),

    "off_scheme_action_weights": copy.deepcopy(OFF_SCHEME_ACTION_WEIGHTS),
    "def_scheme_action_weights": copy.deepcopy(DEF_SCHEME_ACTION_WEIGHTS),

    "offense_scheme_mult": copy.deepcopy(OFFENSE_SCHEME_MULT),
    "defense_scheme_mult": copy.deepcopy(DEFENSE_SCHEME_MULT),
}

_ERA_CACHE: Dict[str, Dict[str, Any]] = {}
_ACTIVE_ERA_NAME: str = "builtin_default"
_ACTIVE_ERA_VERSION: str = "1.0"


def _resolve_era_path(era_name: str) -> Optional[str]:
    """Resolve an era name into an on-disk JSON file path, if it exists."""
    if not isinstance(era_name, str) or not era_name:
        return None
    # direct path
    if era_name.endswith(".json") or "/" in era_name or "\\" in era_name:
        return era_name if os.path.exists(era_name) else None

    here = Path(__file__).resolve().parent
    candidates = [
        here / f"era_{era_name}.json",
        here / f"era_{era_name.lower()}.json",
        here / "eras" / f"era_{era_name}.json",
        here / "eras" / f"era_{era_name.lower()}.json",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _merge_dict(dst: Dict[str, Any], src2: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(dst)
    for k, v in (src2 or {}).items():
        out[k] = v
    return out


def load_era_config(era: Any) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Load an era config (dict) + return (config, warnings, errors)."""
    warnings: List[str] = []
    errors: List[str] = []

    if isinstance(era, dict):
        raw = era
        era_name = str(raw.get("name") or "custom")
    else:
        era_name = str(era or "default")
        if era_name in _ERA_CACHE:
            return _ERA_CACHE[era_name], [], []

        path = _resolve_era_path("default" if era_name == "default" else era_name)
        if path is None:
            warnings.append(f"era file not found for '{era_name}', using built-in defaults")
            cfg = copy.deepcopy(DEFAULT_ERA)
            cfg["name"] = era_name
            _ERA_CACHE[era_name] = cfg
            return cfg, warnings, errors

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            errors.append(f"failed to read era json ({path}): {e}")
            cfg = copy.deepcopy(DEFAULT_ERA)
            cfg["name"] = era_name
            _ERA_CACHE[era_name] = cfg
            return cfg, warnings, errors

        if not isinstance(raw, dict):
            errors.append(f"era json root must be an object/dict (got {type(raw).__name__})")
            cfg = copy.deepcopy(DEFAULT_ERA)
            cfg["name"] = era_name
            _ERA_CACHE[era_name] = cfg
            return cfg, warnings, errors

    cfg, w2, e2 = validate_and_fill_era_dict(raw)
    warnings.extend(w2)
    errors.extend(e2)

    cfg["name"] = str(raw.get("name") or era_name)
    cfg["version"] = str(raw.get("version") or cfg.get("version") or "1.0")

    _ERA_CACHE[cfg["name"]] = cfg
    return cfg, warnings, errors


def validate_and_fill_era_dict(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Validate an era dict and fill missing keys from DEFAULT_ERA."""
    warnings: List[str] = []
    errors: List[str] = []

    cfg = copy.deepcopy(DEFAULT_ERA)
    for k, v in raw.items():
        cfg[k] = v

    required_blocks = [
        "shot_base", "pass_base_success",
        "action_outcome_priors", "action_aliases",
        "off_scheme_action_weights", "def_scheme_action_weights",
        "offense_scheme_mult", "defense_scheme_mult",
        "prob_model", "knobs",
        "logistic_params", "variance_params",
    ]
    for k in required_blocks:
        if k not in cfg or cfg[k] is None:
            warnings.append(f"missing key '{k}' (filled from defaults)")
            cfg[k] = copy.deepcopy(DEFAULT_ERA.get(k))

    dict_blocks = list(required_blocks)
    for k in dict_blocks:
        if not isinstance(cfg.get(k), dict):
            errors.append(f"'{k}' must be an object/dict (got {type(cfg.get(k)).__name__}); using defaults")
            cfg[k] = copy.deepcopy(DEFAULT_ERA.get(k))

    # Light sanity warnings
    for kk, vv in (cfg.get("prob_model") or {}).items():
        if not isinstance(vv, (int, float)) and vv is not None:
            warnings.append(f"prob_model.{kk}: expected number, got {type(vv).__name__}")
    for kk, vv in (cfg.get("knobs") or {}).items():
        if not isinstance(vv, (int, float)) and vv is not None:
            warnings.append(f"knobs.{kk}: expected number, got {type(vv).__name__}")

    return cfg, warnings, errors


def apply_era_config(era_cfg: Dict[str, Any]) -> None:
    """Apply an era config to global tuning parameters."""
    global SHOT_BASE, PASS_BASE_SUCCESS
    global ACTION_OUTCOME_PRIORS, ACTION_ALIASES
    global OFF_SCHEME_ACTION_WEIGHTS, DEF_SCHEME_ACTION_WEIGHTS
    global OFFENSE_SCHEME_MULT, DEFENSE_SCHEME_MULT
    global ERA_PROB_MODEL, ERA_LOGISTIC_PARAMS, ERA_VARIANCE_PARAMS, ERA_ROLE_FIT, _ACTIVE_ERA_NAME, _ACTIVE_ERA_VERSION

    if not isinstance(era_cfg, dict):
        return

    if isinstance(era_cfg.get("shot_base"), dict):
        SHOT_BASE = dict(era_cfg["shot_base"])
    if isinstance(era_cfg.get("pass_base_success"), dict):
        PASS_BASE_SUCCESS = dict(era_cfg["pass_base_success"])

    if isinstance(era_cfg.get("action_outcome_priors"), dict):
        ACTION_OUTCOME_PRIORS = copy.deepcopy(era_cfg["action_outcome_priors"])
    if isinstance(era_cfg.get("action_aliases"), dict):
        ACTION_ALIASES = dict(era_cfg["action_aliases"])

    if isinstance(era_cfg.get("off_scheme_action_weights"), dict):
        OFF_SCHEME_ACTION_WEIGHTS = copy.deepcopy(era_cfg["off_scheme_action_weights"])
    if isinstance(era_cfg.get("def_scheme_action_weights"), dict):
        DEF_SCHEME_ACTION_WEIGHTS = copy.deepcopy(era_cfg["def_scheme_action_weights"])

    if isinstance(era_cfg.get("offense_scheme_mult"), dict):
        OFFENSE_SCHEME_MULT = copy.deepcopy(era_cfg["offense_scheme_mult"])
    if isinstance(era_cfg.get("defense_scheme_mult"), dict):
        DEFENSE_SCHEME_MULT = copy.deepcopy(era_cfg["defense_scheme_mult"])

    pm = era_cfg.get("prob_model")
    if isinstance(pm, dict):
        ERA_PROB_MODEL = _merge_dict(DEFAULT_PROB_MODEL, pm)

    lp = era_cfg.get("logistic_params")
    if isinstance(lp, dict):
        # keep only dict->dict entries
        ERA_LOGISTIC_PARAMS = copy.deepcopy(lp)

    vp = era_cfg.get("variance_params")
    if isinstance(vp, dict):
        ERA_VARIANCE_PARAMS = copy.deepcopy(vp)


    rf = era_cfg.get("role_fit")
    if isinstance(rf, dict):
        ERA_ROLE_FIT = _merge_dict(DEFAULT_ROLE_FIT, rf)

    _ACTIVE_ERA_NAME = str(era_cfg.get("name") or "unknown")
    _ACTIVE_ERA_VERSION = str(era_cfg.get("version") or "1.0")

    # Refresh allowed sets for validation (defined later in the file)
    if "refresh_allowed_sets" in globals():
        try:
            globals()["refresh_allowed_sets"]()
        except Exception:
            pass

# -------------------------
# Tactics config
# -------------------------

@dataclass
class TacticsConfig:
    offense_scheme: str = "Spread_HeavyPnR"
    defense_scheme: str = "Drop"
    scheme_weight_sharpness: float = 1.00
    scheme_outcome_strength: float = 1.00
    def_scheme_weight_sharpness: float = 1.00
    def_scheme_outcome_strength: float = 1.00

    action_weight_mult: Dict[str, float] = field(default_factory=dict)
    outcome_global_mult: Dict[str, float] = field(default_factory=dict)
    outcome_by_action_mult: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def_action_weight_mult: Dict[str, float] = field(default_factory=dict)
    opp_action_weight_mult: Dict[str, float] = field(default_factory=dict)

    opp_outcome_global_mult: Dict[str, float] = field(default_factory=dict)
    opp_outcome_by_action_mult: Dict[str, Dict[str, float]] = field(default_factory=dict)

    context: Dict[str, Any] = field(default_factory=dict)


# -------------------------
# Validation / Sanitization (Commercial-ready input safety)
# -------------------------

@dataclass
class ValidationConfig:
    """Controls how strictly we validate and sanitize user inputs."""
    strict: bool = True  # True: raise on critical issues (missing derived keys, invalid schemes, invalid lineup)
    mult_lo: float = 0.70
    mult_hi: float = 1.40
    derived_lo: float = 0.0
    derived_hi: float = 100.0
    missing_derived_policy: str = "error"  # "error" or "fill"
    default_derived_value: float = DERIVED_DEFAULT
    # If True, we will clamp out-of-range numbers instead of erroring (still logs warnings).
    clamp_out_of_range: bool = True


@dataclass
class ValidationReport:
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        return {"warnings": list(self.warnings), "errors": list(self.errors), "ok": (len(self.errors) == 0)}


def _is_finite_number(x: Any) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return math.isfinite(v)


def _collect_required_derived_keys() -> List[str]:
    keys: set[str] = set()

    # Anything that affects outcome resolution
    for _, sides in OUTCOME_PROFILES.items():
        keys.update(sides.get("offense", {}).keys())
        keys.update(sides.get("defense", {}).keys())

    # Role fallbacks and other selectors
    keys.update(ROLE_FALLBACK_RANK.values())
    keys.update([
        "SHOT_FT",
        "REB_OR", "REB_DR",
        "PHYSICAL", "ENDURANCE",
        "DEF_POA", "DEF_HELP", "DEF_STEAL", "DEF_RIM", "DEF_POST",
        "FIN_RIM", "FIN_DUNK", "FIN_CONTACT",
        "SHOT_3_CS", "SHOT_3_OD", "SHOT_MID_CS", "SHOT_MID_PU", "SHOT_TOUCH",
        "POST_SCORE", "POST_CONTROL",
        "DRIVE_CREATE", "HANDLE_SAFE", "FIRST_STEP",
        "PASS_SAFE", "PASS_CREATE", "PNR_READ", "SHORTROLL_PLAY",
    ])
    return sorted(keys)


REQUIRED_DERIVED_KEYS: List[str] = _collect_required_derived_keys()


def _collect_allowed_off_actions() -> set[str]:
    s: set[str] = set()
    for d in OFF_SCHEME_ACTION_WEIGHTS.values():
        s.update(d.keys())
    # base actions and aliases should also be allowed
    s.update(ACTION_OUTCOME_PRIORS.keys())
    s.update(ACTION_ALIASES.keys())
    s.update(ACTION_ALIASES.values())
    return s


def _collect_allowed_def_actions() -> set[str]:
    s: set[str] = set()
    for d in DEF_SCHEME_ACTION_WEIGHTS.values():
        s.update(d.keys())
    return s


def _collect_allowed_outcomes() -> set[str]:
    s: set[str] = set()
    s.update(OUTCOME_PROFILES.keys())
    s.update(SHOT_BASE.keys())
    s.update(PASS_BASE_SUCCESS.keys())
    for pri in ACTION_OUTCOME_PRIORS.values():
        s.update(pri.keys())
    return s


ALLOWED_OFF_ACTIONS: set[str] = set()
ALLOWED_DEF_ACTIONS: set[str] = set()
ALLOWED_OUTCOMES: set[str] = set()

def refresh_allowed_sets() -> None:
    """Recompute allowed keys after era parameters change."""
    global ALLOWED_OFF_ACTIONS, ALLOWED_DEF_ACTIONS, ALLOWED_OUTCOMES
    ALLOWED_OFF_ACTIONS = _collect_allowed_off_actions()
    ALLOWED_DEF_ACTIONS = _collect_allowed_def_actions()
    ALLOWED_OUTCOMES = _collect_allowed_outcomes()

refresh_allowed_sets()


def _clamp_mult(v: float, cfg: ValidationConfig) -> float:
    return clamp(v, cfg.mult_lo, cfg.mult_hi)


def _sanitize_mult_dict(
    mults: Dict[str, Any],
    allowed_keys: set[str],
    cfg: ValidationConfig,
    report: ValidationReport,
    path: str,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, raw in (mults or {}).items():
        if k not in allowed_keys:
            report.warn(f"{path}: unknown key '{k}' ignored")
            continue
        if not _is_finite_number(raw):
            msg = f"{path}.{k}: non-numeric multiplier '{raw}'"
            if cfg.strict:
                report.error(msg)
            else:
                report.warn(msg + " (ignored)")
            continue
        v = float(raw)
        vv = _clamp_mult(v, cfg)
        if abs(vv - v) > 1e-9:
            report.warn(f"{path}.{k}: clamped {v:.3f} -> {vv:.3f}")
        out[k] = vv
    return out


def _sanitize_outcome_mult_dict(
    mults: Dict[str, Any],
    cfg: ValidationConfig,
    report: ValidationReport,
    path: str,
) -> Dict[str, float]:
    return _sanitize_mult_dict(mults, ALLOWED_OUTCOMES, cfg, report, path)


def _sanitize_nested_outcome_by_action(
    nested: Dict[str, Any],
    cfg: ValidationConfig,
    report: ValidationReport,
    path: str,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for act, sub in (nested or {}).items():
        if act not in ALLOWED_OFF_ACTIONS:
            report.warn(f"{path}: unknown action '{act}' ignored")
            continue
        if not isinstance(sub, dict):
            msg = f"{path}.{act}: expected dict, got {type(sub).__name__}"
            if cfg.strict:
                report.error(msg)
            else:
                report.warn(msg + " (ignored)")
            continue
        clean = _sanitize_outcome_mult_dict(sub, cfg, report, f"{path}.{act}")
        if clean:
            out[act] = clean
    return out


def sanitize_tactics_config(tac: TacticsConfig, cfg: ValidationConfig, report: ValidationReport, label: str) -> None:
    """Mutates tactics in-place: clamps all UI knobs and ignores unknown keys."""

    if tac.offense_scheme not in OFF_SCHEME_ACTION_WEIGHTS:
        msg = f"{label}.offense_scheme: unknown scheme '{tac.offense_scheme}'"
        if cfg.strict:
            report.error(msg)
        else:
            report.warn(msg + " (fallback to Spread_HeavyPnR)")
            tac.offense_scheme = "Spread_HeavyPnR"

    if tac.defense_scheme not in DEF_SCHEME_ACTION_WEIGHTS:
        msg = f"{label}.defense_scheme: unknown scheme '{tac.defense_scheme}'"
        if cfg.strict:
            report.error(msg)
        else:
            report.warn(msg + " (fallback to Drop)")
            tac.defense_scheme = "Drop"

    # Scalar knobs
    for attr in ("scheme_weight_sharpness", "scheme_outcome_strength", "def_scheme_weight_sharpness", "def_scheme_outcome_strength"):
        raw = getattr(tac, attr, 1.0)
        if not _is_finite_number(raw):
            msg = f"{label}.{attr}: non-numeric '{raw}'"
            if cfg.strict:
                report.error(msg)
            else:
                report.warn(msg + " (set to 1.0)")
                setattr(tac, attr, 1.0)
            continue
        v = float(raw)
        vv = _clamp_mult(v, cfg)
        if abs(vv - v) > 1e-9:
            report.warn(f"{label}.{attr}: clamped {v:.3f} -> {vv:.3f}")
        setattr(tac, attr, vv)

    # Offense multipliers
    tac.action_weight_mult = _sanitize_mult_dict(tac.action_weight_mult, ALLOWED_OFF_ACTIONS, cfg, report, f"{label}.action_weight_mult")
    tac.outcome_global_mult = _sanitize_outcome_mult_dict(tac.outcome_global_mult, cfg, report, f"{label}.outcome_global_mult")
    tac.outcome_by_action_mult = _sanitize_nested_outcome_by_action(tac.outcome_by_action_mult, cfg, report, f"{label}.outcome_by_action_mult")

    # Defense multipliers
    tac.def_action_weight_mult = _sanitize_mult_dict(tac.def_action_weight_mult, ALLOWED_DEF_ACTIONS, cfg, report, f"{label}.def_action_weight_mult")
    tac.opp_action_weight_mult = _sanitize_mult_dict(getattr(tac, "opp_action_weight_mult", {}), ALLOWED_OFF_ACTIONS, cfg, report, f"{label}.opp_action_weight_mult")
    tac.opp_outcome_global_mult = _sanitize_outcome_mult_dict(tac.opp_outcome_global_mult, cfg, report, f"{label}.opp_outcome_global_mult")
    tac.opp_outcome_by_action_mult = _sanitize_nested_outcome_by_action(tac.opp_outcome_by_action_mult, cfg, report, f"{label}.opp_outcome_by_action_mult")

    # Context values (some are multipliers, some are special knobs)
    if tac.context is None:
        tac.context = {}
    clean_ctx: Dict[str, Any] = {}
    for k, v in tac.context.items():
        if k.endswith("_MULT"):
            if not _is_finite_number(v):
                msg = f"{label}.context.{k}: non-numeric '{v}'"
                if cfg.strict:
                    report.error(msg)
                    continue
                report.warn(msg + " (set to 1.0)")
                clean_ctx[k] = 1.0
                continue
            fv = float(v)
            fvv = _clamp_mult(fv, cfg)
            if abs(fvv - fv) > 1e-9:
                report.warn(f"{label}.context.{k}: clamped {fv:.3f} -> {fvv:.3f}")
            clean_ctx[k] = fvv

        elif k == "ROLE_FIT_STRENGTH":
            # 0..1 scalar (separate from multiplier clamp range)
            if not _is_finite_number(v):
                msg = f"{label}.context.{k}: non-numeric '{v}'"
                if cfg.strict:
                    report.error(msg)
                    continue
                report.warn(msg + " (set to 0.65)")
                clean_ctx[k] = 0.65
                continue
            fv = float(v)
            fvv = clamp(fv, 0.0, 1.0)
            if abs(fvv - fv) > 1e-9:
                report.warn(f"{label}.context.{k}: clamped {fv:.3f} -> {fvv:.3f}")
            clean_ctx[k] = fvv

        else:
            clean_ctx[k] = v
    tac.context = clean_ctx


def sanitize_player_derived(p: Player, cfg: ValidationConfig, report: ValidationReport, label: str) -> None:
    """Ensures derived stats exist, are numeric, and contain required keys."""
    if p.derived is None:
        report.warn(f"{label}.{p.pid}: derived is None (set to empty)")
        p.derived = {}

    # Coerce numeric & clamp
    clean: Dict[str, float] = {}
    for k, raw in p.derived.items():
        if not _is_finite_number(raw):
            msg = f"{label}.{p.pid}.derived.{k}: non-numeric '{raw}'"
            if cfg.strict:
                report.error(msg)
                continue
            report.warn(msg + " (dropped)")
            continue
        v = float(raw)
        if cfg.clamp_out_of_range:
            vv = clamp(v, cfg.derived_lo, cfg.derived_hi)
            if abs(vv - v) > 1e-9:
                report.warn(f"{label}.{p.pid}.derived.{k}: clamped {v:.2f} -> {vv:.2f}")
            v = vv
        clean[k] = v
    p.derived = clean

    # Required keys
    missing = [k for k in REQUIRED_DERIVED_KEYS if k not in p.derived]
    if missing:
        msg = f"{label}.{p.pid}: missing derived keys ({len(missing)}): {', '.join(missing[:8])}{'...' if len(missing)>8 else ''}"
        if cfg.missing_derived_policy == "fill":
            report.warn(msg + f" (filled with {cfg.default_derived_value})")
            for k in missing:
                p.derived[k] = float(cfg.default_derived_value)
        else:
            report.error(msg)


def validate_and_sanitize_team(team: TeamState, cfg: ValidationConfig, report: ValidationReport, label: str) -> None:
    # Lineup sanity
    if not isinstance(team.lineup, list) or len(team.lineup) == 0:
        report.error(f"{label}: lineup missing")
        return

    if len(team.lineup) != 5:
        msg = f"{label}: lineup size is {len(team.lineup)} (expected 5)"
        if cfg.strict:
            report.error(msg)
        else:
            report.warn(msg + " (engine will use first 5)")
            team.lineup = team.lineup[:5] if len(team.lineup) > 5 else team.lineup

    # Unique PIDs
    pids = [p.pid for p in team.lineup]
    if len(set(pids)) != len(pids):
        report.error(f"{label}: duplicate player pid in lineup")
    if any((not isinstance(pid, str)) or (pid.strip() == "") for pid in pids):
        report.error(f"{label}: invalid empty pid in lineup")

    # Player derived validation
    for p in team.lineup:
        sanitize_player_derived(p, cfg, report, label)

    # Roles sanity (warn-only; engine already has fallbacks)
    if team.roles is None:
        team.roles = {}
        report.warn(f"{label}: roles missing (empty roles)")
    lineup_pid_set = set(pids)
    for role, pid in list(team.roles.items()):
        if pid not in lineup_pid_set:
            report.warn(f"{label}.roles.{role}: pid '{pid}' not in lineup (fallback will be used)")
            team.roles.pop(role, None)

    # Tactics sanity + clamp
    if team.tactics is None:
        report.error(f"{label}: tactics missing")
        return
    sanitize_tactics_config(team.tactics, cfg, report, f"{label}.tactics")



# -------------------------
# Defense snapshot (no matchups MVP)
# -------------------------

def team_def_snapshot(team: TeamState) -> Dict[str, float]:
    onball = max(team.lineup, key=lambda p: p.get("DEF_POA"))
    rim = max(team.lineup, key=lambda p: p.get("DEF_RIM"))
    steal = max(team.lineup, key=lambda p: p.get("DEF_STEAL"))
    avg_keys = ["DEF_HELP","PHYSICAL","ENDURANCE","DEF_POST"]
    avg = {k: sum(p.get(k) for p in team.lineup) / len(team.lineup) for k in avg_keys}
    return {
        "DEF_POA": onball.get("DEF_POA"),
        "DEF_RIM": rim.get("DEF_RIM"),
        "DEF_STEAL": steal.get("DEF_STEAL"),
        "DEF_HELP": avg["DEF_HELP"],
        "DEF_POST": avg["DEF_POST"],
        "PHYSICAL": avg["PHYSICAL"],
        "ENDURANCE": avg["ENDURANCE"],
    }


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


# -------------------------
# Probability model
# -------------------------

def prob_from_scores(
    rng: Optional[random.Random],
    base_p: float,
    off_score: float,
    def_score: float,
    *,
    kind: str = "default",
    variance_mult: float = 1.0,
    logit_delta: float = 0.0,
) -> float:
    """Convert an OffScore/DefScore matchup into a probability using a logistic model.

    Model:
      p = sigmoid( logit(base_p) + (off_score - def_score) * sensitivity + noise )

    - base_p: the baseline probability for this outcome type (e.g., 3PT base%, pass base%).
    - sensitivity: per-kind slope, externalized in the era file (logistic_params).
    - noise: optional variance knob (2-3). Uses logit-space Gaussian noise so the mean stays stable.
    """
    pm = ERA_PROB_MODEL if isinstance(ERA_PROB_MODEL, dict) else DEFAULT_PROB_MODEL
    base_p = clamp(float(base_p), float(pm.get("base_p_min", 0.02)), float(pm.get("base_p_max", 0.98)))
    base_logit = math.log(base_p / (1.0 - base_p))

    # ---- sensitivity (2-1, 2-2) ----
    lp = ERA_LOGISTIC_PARAMS if isinstance(ERA_LOGISTIC_PARAMS, dict) else DEFAULT_LOGISTIC_PARAMS
    spec = lp.get(kind) or lp.get("default") or {}
    sens = spec.get("sensitivity")
    scale = spec.get("scale")

    # Back-compat fallback (older era json without logistic_params)
    if sens is None:
        if scale is not None and float(scale) > 1e-9:
            sens = 1.0 / float(scale)
        else:
            # old single-scale knobs
            if kind.startswith("pass"):
                sens = 1.0 / float(pm.get("pass_scale", 20.0))
            elif kind.startswith("rebound"):
                sens = 1.0 / float(pm.get("rebound_scale", 22.0))
            else:
                sens = 1.0 / float(pm.get("shot_scale", 18.0))

    gap = (float(off_score) - float(def_score)) * float(sens)

    # ---- variance knob (2-3) ----
    noise = 0.0
    if rng is not None:
        vp = ERA_VARIANCE_PARAMS if isinstance(ERA_VARIANCE_PARAMS, dict) else DEFAULT_VARIANCE_PARAMS
        std = float(vp.get("logit_noise_std", 0.0))
        kind_mult = float((vp.get("kind_mult") or {}).get(kind, 1.0)) if isinstance(vp.get("kind_mult"), dict) else 1.0
        # team volatility multiplier (clamped)
        tlo, thi = 0.70, 1.40
        if isinstance(vp.get("team_mult_lo"), (int, float)):
            tlo = float(vp["team_mult_lo"])
        if isinstance(vp.get("team_mult_hi"), (int, float)):
            thi = float(vp["team_mult_hi"])
        vm = clamp(float(variance_mult), tlo, thi)
        std = std * kind_mult * vm
        if std > 1e-9:
            noise = rng.gauss(0.0, std)

    p = sigmoid(base_logit + gap + noise + float(logit_delta))
    return clamp(p, float(pm.get("prob_min", 0.03)), float(pm.get("prob_max", 0.97)))


def _shot_kind_from_outcome(outcome: str) -> str:
    # 2-2 categories
    if outcome in ("SHOT_3_CS", "SHOT_3_OD"):
        return "shot_3"
    if outcome in ("SHOT_MID_CS", "SHOT_MID_PU"):
        return "shot_mid"
    if outcome == "SHOT_POST":
        return "shot_post"
    return "shot_rim"

def _team_variance_mult(team: "TeamState") -> float:
    vp = ERA_VARIANCE_PARAMS if isinstance(ERA_VARIANCE_PARAMS, dict) else DEFAULT_VARIANCE_PARAMS
    try:
        vm = float((team.tactics.context or {}).get("VARIANCE_MULT", 1.0))
    except Exception:
        vm = 1.0
    lo = float(vp.get("team_mult_lo", 0.70))
    hi = float(vp.get("team_mult_hi", 1.40))
    return clamp(vm, lo, hi)



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


# -------------------------
# Participant selection
# -------------------------

def choose_weighted_player(rng: random.Random, players: List[Player], key: str, power: float = 1.2) -> Player:
    weights = {p.pid: (max(p.get(key), 1.0) ** power) for p in players}
    pid = weighted_choice(rng, weights)
    for p in players:
        if p.pid == pid:
            return p
    return players[0]

def choose_shooter_for_three(rng: random.Random, offense: TeamState) -> Player:
    # Use up to 3 best 3pt shooters, weighted
    sorted_p = sorted(offense.lineup, key=lambda p: p.get("SHOT_3_CS"), reverse=True)
    cand = sorted_p[:3]
    return choose_weighted_player(rng, cand, "SHOT_3_CS", power=1.35)

def choose_shooter_for_mid(rng: random.Random, offense: TeamState) -> Player:
    sorted_p = sorted(offense.lineup, key=lambda p: p.get("SHOT_MID_CS"), reverse=True)
    cand = sorted_p[:3]
    return choose_weighted_player(rng, cand, "SHOT_MID_CS", power=1.25)

def choose_creator_for_pulloff(rng: random.Random, offense: TeamState, outcome: str) -> Player:
    bh = offense.get_role_player("ball_handler", ROLE_FALLBACK_RANK["ball_handler"])
    sh = offense.get_role_player("secondary_handler", ROLE_FALLBACK_RANK["secondary_handler"])
    cand = [bh, sh] if bh.pid != sh.pid else [bh]
    key = "SHOT_3_OD" if outcome == "SHOT_3_OD" else "SHOT_MID_PU"
    return choose_weighted_player(rng, cand, key, power=1.20)

def choose_finisher_rim(rng: random.Random, offense: TeamState, dunk_bias: bool = False) -> Player:
    bh = offense.get_role_player("ball_handler", ROLE_FALLBACK_RANK["ball_handler"])
    rr = offense.get_role_player("rim_runner", ROLE_FALLBACK_RANK["rim_runner"])
    sc = offense.get_role_player("screener", ROLE_FALLBACK_RANK["screener"])
    cu = offense.get_role_player("cutter", ROLE_FALLBACK_RANK["cutter"])
    cand = [bh, rr, sc, cu]
    # remove duplicates while preserving order
    seen = set()
    uniq = []
    for p in cand:
        if p.pid not in seen:
            uniq.append(p); seen.add(p.pid)
    key = "FIN_DUNK" if dunk_bias else "FIN_RIM"
    return choose_weighted_player(rng, uniq, key, power=1.15)

def choose_post_target(offense: TeamState) -> Player:
    return offense.get_role_player("post", ROLE_FALLBACK_RANK["post"])

def choose_passer(rng: random.Random, offense: TeamState, base_action: str, outcome: str) -> Player:
    # heuristic: Drive -> BH/Slasher; PostUp -> post; Shortroll -> screener; else BH
    bh = offense.get_role_player("ball_handler", ROLE_FALLBACK_RANK["ball_handler"])
    post = offense.get_role_player("post", ROLE_FALLBACK_RANK["post"])
    sc = offense.get_role_player("screener", ROLE_FALLBACK_RANK["screener"])
    if outcome == "PASS_SHORTROLL":
        return sc
    if base_action == "PostUp":
        return post
    if base_action == "Drive":
        # pick between BH and best driver
        cand = [bh, max(offense.lineup, key=lambda p: p.get("DRIVE_CREATE"))]
        # unique
        seen=set(); uniq=[]
        for p in cand:
            if p.pid not in seen:
                uniq.append(p); seen.add(p.pid)
        return choose_weighted_player(rng, uniq, "PASS_CREATE", power=1.10)
    return bh


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
    return prob_from_scores(None, float(ERA_PROB_MODEL.get('orb_base', 0.26)), off_orb, def_drb, kind='rebound', variance_mult=1.0)

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
    pass_chain: int
) -> Tuple[str, Dict[str, Any]]:
    # count outcome
    offense.outcome_counts[outcome] = offense.outcome_counts.get(outcome, 0) + 1

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

    # compute scores
    off_vals = {k: actor.get(k) for k in prof["offense"].keys()}
    off_score = dot_profile(off_vals, prof["offense"])
    def_vals = {k: float(def_snap.get(k, 50.0)) for k in prof["defense"].keys()}
    def_score = dot_profile(def_vals, prof["defense"])

    # resolve by type
    if is_shot(outcome):
        base_p = SHOT_BASE.get(outcome, 0.45)
        kind = _shot_kind_from_outcome(outcome)
        p_make = prob_from_scores(rng, base_p, off_score, def_score, kind=kind, variance_mult=_team_variance_mult(offense), logit_delta=float(tags.get('role_logit_delta', 0.0)))
        pts = outcome_points(outcome)

        offense.fga += 1
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
        base_s = PASS_BASE_SUCCESS.get(outcome, 0.90)
        p_ok = prob_from_scores(rng, base_s, off_score, def_score, kind="pass", variance_mult=_team_variance_mult(offense), logit_delta=float(tags.get('role_logit_delta', 0.0)))
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
        if outcome == "FOUL_REACH_TRAP":
            # non-shooting foul -> reset
            return "RESET", {"outcome": outcome, "type": "SIDE_OUT"}
        nfts = 3 if outcome == "FOUL_DRAW_JUMPER" else 2
        resolve_free_throws(rng, actor, nfts, offense)
        return "FOUL", {"outcome": outcome, "pid": actor.pid, "fts": nfts}

    if is_reset(outcome):
        return "RESET", {"outcome": outcome}

    return "RESET", {"outcome": outcome}


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
        defense_scheme="PackLine_GapHelp",
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

    def print_team(label: str, r: Dict[str, Any]) -> None:
        print(f"\n=== {label} ===")
        print(f"PTS {r['PTS']} | FG {r['FGM']}/{r['FGA']} | 3PT {r['3PM']}/{r['3PA']} | FT {r['FTM']}/{r['FTA']} | TOV {r['TOV']} | ORB {r['ORB']} DRB {r['DRB']}")
        top_off = list(r["OffActionCounts"].items())[:6]
        top_out = list(r["OutcomeCounts"].items())[:8]
        print("Top Off Actions:", ", ".join([f"{a}:{c}" for a,c in top_off]))
        print("Top Outcomes:", ", ".join([f"{o}:{c}" for o,c in top_out]))
        players = r["Players"]
        top = sorted(players.items(), key=lambda kv: -kv[1]["PTS"])[:5]
        print("Top scorers:", ", ".join([f"{pid}:{st['PTS']}pts" for pid,st in top]))

    print("Possessions per team:", res["possessions_per_team"])
    print_team("A_SpreadPnR", res["teams"]["A_SpreadPnR"])
    print_team("B_DriveKick", res["teams"]["B_DriveKick"])

if __name__ == "__main__":
    demo()
