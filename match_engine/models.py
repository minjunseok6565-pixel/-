from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .core import clamp

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


