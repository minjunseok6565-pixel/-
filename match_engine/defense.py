from __future__ import annotations

from typing import Dict

from .models import TeamState

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


