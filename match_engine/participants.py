# -------------------------
from __future__ import annotations

import random
from typing import List

from .core import weighted_choice
from .models import Player, ROLE_FALLBACK_RANK, TeamState

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


