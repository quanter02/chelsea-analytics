"""Team-match metrics built from StatsBomb event data.

Coordinate system (StatsBomb): pitch is 120 x 80, every event is recorded from the
acting team's perspective attacking left -> right. y = 0 is the attacker's LEFT touchline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import data

SET_PIECE_PATTERNS = {"From Corner", "From Free Kick"}
SET_PIECE_SHOT_TYPES = {"Penalty", "Free Kick", "Corner"}
DEF_ACTIONS = {"Interception", "Foul Committed"}  # + tackles (Duel/Tackle), see _is_def_action
FINAL_THIRD_X = 80
HIGH_TURNOVER_X = 80  # regained within 40m of goal
PPDA_OPP_ZONE_X = 72  # opponent passes in their own 60%

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "derived"


def channel(y: float) -> str:
    """Channel from the ATTACKING team's point of view."""
    if y < 80 / 3:
        return "left"
    if y > 160 / 3:
        return "right"
    return "center"


def _is_def_action(e: dict) -> bool:
    t = e["type"]["name"]
    if t in DEF_ACTIONS:
        return True
    return t == "Duel" and e.get("duel", {}).get("type", {}).get("name") == "Tackle"


def _in_box(xy) -> bool:
    return xy is not None and xy[0] >= 102 and 18 <= xy[1] <= 62


def shots_table(match_id: int) -> pd.DataFrame:
    ev = data.events(match_id)
    by_id = {e["id"]: e for e in ev}
    teams = sorted({e["team"]["name"] for e in ev if "team" in e})
    score = {t: 0 for t in teams}
    rows = []
    for e in ev:
        t = e["type"]["name"]
        if t == "Own Goal For":
            score[e["team"]["name"]] += 1
        if t != "Shot":
            continue
        s = e["shot"]
        team = e["team"]["name"]
        opp = next(x for x in teams if x != team)
        diff = score[team] - score[opp]
        state = "leading" if diff > 0 else "trailing" if diff < 0 else "level"
        if s["outcome"]["name"] == "Goal":
            score[team] += 1
        kp = by_id.get(s.get("key_pass_id")) if s.get("key_pass_id") else None
        pattern = e["play_pattern"]["name"]
        stype = s["type"]["name"]
        origin = ("set_piece" if (pattern in SET_PIECE_PATTERNS or stype in SET_PIECE_SHOT_TYPES)
                  else "counter" if pattern == "From Counter" else "open_play")
        rows.append({
            "match_id": match_id,
            "team": team,
            "player": e["player"]["name"],
            "period": e["period"],
            "possession": e["possession"],
            "state": state,  # shooting team's game state before the shot
            "minute": e["minute"],
            "x": e["location"][0], "y": e["location"][1],
            "xg": s.get("statsbomb_xg", 0.0),
            "goal": s["outcome"]["name"] == "Goal",
            "on_target": s["outcome"]["name"] in {"Goal", "Saved", "Saved To Post"},
            "shot_type": stype,
            "body_part": s["body_part"]["name"],
            "pattern": pattern,
            "origin": origin,
            "assist_player": kp["player"]["name"] if kp else None,
            "assist_x": kp["location"][0] if kp else np.nan,
            "assist_y": kp["location"][1] if kp else np.nan,
            "assist_cross": bool(kp and kp.get("pass", {}).get("cross")),
        })
    return pd.DataFrame(rows)


def _team_match_rows(match_id: int) -> list[dict]:
    ev = data.events(match_id)
    teams = sorted({e["team"]["name"] for e in ev if "team" in e})
    shots = shots_table(match_id)
    out = []
    for team in teams:
        opp = next(t for t in teams if t != team)
        mine = [e for e in ev if e.get("team", {}).get("name") == team]
        theirs = [e for e in ev if e.get("team", {}).get("name") == opp]
        passes = [e for e in mine if e["type"]["name"] == "Pass"]
        opp_passes = [e for e in theirs if e["type"]["name"] == "Pass"]
        ft = sum(1 for p in passes if p["location"][0] >= FINAL_THIRD_X)
        opp_ft = sum(1 for p in opp_passes if p["location"][0] >= FINAL_THIRD_X)
        def_actions = sum(1 for e in mine if _is_def_action(e) and e["location"][0] > 120 - PPDA_OPP_ZONE_X)
        opp_build = sum(1 for p in opp_passes if p["location"][0] < PPDA_OPP_ZONE_X)
        pressures = [e["location"][0] for e in mine if e["type"]["name"] == "Pressure"]
        regains = sum(
            1 for e in mine
            if e["type"]["name"] in {"Ball Recovery", "Interception"}
            and not e.get("ball_recovery", {}).get("recovery_failure")
            and e["location"][0] >= HIGH_TURNOVER_X
        )
        completed = [p for p in passes if "outcome" not in p["pass"]]
        box_entries = sum(1 for p in completed if _in_box(p["pass"]["end_location"]) and not _in_box(p["location"]))
        carries_in = sum(
            1 for e in mine if e["type"]["name"] == "Carry"
            and _in_box(e["carry"]["end_location"]) and not _in_box(e["location"])
        )
        long_balls = sum(1 for p in passes if p["pass"].get("length", 0) >= 32 and p["location"][0] < 60)
        own_half_passes = sum(1 for p in passes if p["location"][0] < 60)
        crosses = sum(1 for p in passes if p["pass"].get("cross"))
        ents = [channel(p["pass"]["end_location"][1]) for p in completed
                if p["location"][0] < FINAL_THIRD_X <= p["pass"]["end_location"][0]]
        s_for = shots[shots.team == team] if len(shots) else shots
        s_ag = shots[shots.team == opp] if len(shots) else shots

        def xg(df, origin=None):
            if not len(df):
                return 0.0
            return float(df.xg.sum() if origin is None else df.loc[df.origin == origin, "xg"].sum())

        out.append({
            "match_id": match_id, "team": team, "opponent": opp,
            "goals": int(s_for.goal.sum()) if len(s_for) else 0,
            "shots": len(s_for), "shots_against": len(s_ag),
            "xg": xg(s_for), "xga": xg(s_ag),
            "npxg": xg(s_for[s_for.shot_type != "Penalty"]) if len(s_for) else 0.0,
            "npxga": xg(s_ag[s_ag.shot_type != "Penalty"]) if len(s_ag) else 0.0,
            "xg_open": xg(s_for, "open_play"), "xg_counter": xg(s_for, "counter"),
            "xg_setpiece": xg(s_for, "set_piece"),
            "xga_open": xg(s_ag, "open_play"), "xga_counter": xg(s_ag, "counter"),
            "xga_setpiece": xg(s_ag, "set_piece"),
            "passes": len(passes),
            "pass_share": len(passes) / max(1, len(passes) + len(opp_passes)),
            "pass_accuracy": len(completed) / max(1, len(passes)),
            "field_tilt": ft / max(1, ft + opp_ft),
            "ppda": opp_build / max(1, def_actions),
            "press_height": float(np.mean(pressures)) if pressures else np.nan,
            "pressures": len(pressures),
            "high_turnovers": regains,
            "box_entries": box_entries + carries_in,
            "crosses": crosses,
            "long_ball_share": long_balls / max(1, own_half_passes),
            "ft_entries_left": ents.count("left"),
            "ft_entries_center": ents.count("center"),
            "ft_entries_right": ents.count("right"),
        })
    return out


def season_team_matches(refresh: bool = False) -> pd.DataFrame:
    """All team-match rows for PL 2015/16 (760 rows). Cached to CSV."""
    path = CACHE_DIR / "team_matches.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["date"])
    m = data.matches()
    rows = [r for mid in m.match_id for r in _team_match_rows(mid)]
    df = pd.DataFrame(rows).merge(
        m[["match_id", "date", "home", "home_score", "away_score", "home_manager", "away_manager"]], on="match_id")
    is_home = df.team == df.home
    df["venue"] = np.where(is_home, "H", "A")
    df["manager"] = df.home_manager.where(is_home, df.away_manager)
    # official score (includes own goals, which are not shots)
    df["goals"] = df.home_score.where(is_home, df.away_score)
    df["goals_against"] = df.away_score.where(is_home, df.home_score)
    df = df.drop(columns=["home", "home_score", "away_score", "home_manager", "away_manager"])
    df["points"] = np.select([df.goals > df.goals_against, df.goals == df.goals_against], [3, 1], 0)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.sort_values(["team", "date"]).to_csv(path, index=False)
    return df


def season_shots(refresh: bool = False) -> pd.DataFrame:
    path = CACHE_DIR / "shots.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    df = pd.concat([shots_table(mid) for mid in data.matches().match_id], ignore_index=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def team_season(tm: pd.DataFrame) -> pd.DataFrame:
    """Per-team season averages (per match) — used as the league baseline."""
    cols = [c for c in tm.columns if tm[c].dtype.kind in "fi" and c not in {"match_id"}]
    agg = tm.groupby("team")[cols].mean()
    agg["matches"] = tm.groupby("team").size()
    agg["total_points"] = tm.groupby("team").points.sum()
    return agg.sort_values("total_points", ascending=False)


def expected_points(shots: pd.DataFrame, tm: pd.DataFrame, n_sims: int = 10000, seed: int = 7) -> pd.DataFrame:
    """Expected points per team-match.

    Shots are first collapsed per possession (a rebound after a save is not an independent
    chance): p(goal in possession) = 1 - prod(1 - xG_i). Each possession is then simulated as a
    Bernoulli trial. Own goals are not shots, so they are ignored (as in most public xPts models).
    """
    shots = possession_xg(shots)
    rng = np.random.default_rng(seed)
    out = []
    for mid, grp in tm.groupby("match_id"):
        teams = grp.team.tolist()
        sims = {}
        for t in teams:
            p = shots.loc[(shots.match_id == mid) & (shots.team == t), "pxg"].to_numpy()
            sims[t] = (rng.random((n_sims, len(p))) < p).sum(axis=1) if len(p) else np.zeros(n_sims)
        a, b = teams
        win_a, draw = (sims[a] > sims[b]).mean(), (sims[a] == sims[b]).mean()
        win_b = 1 - win_a - draw
        out += [{"match_id": mid, "team": a, "xpts": 3 * win_a + draw},
                {"match_id": mid, "team": b, "xpts": 3 * win_b + draw}]
    return pd.DataFrame(out)


def possession_xg(shots: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team, possession) with the combined chance of scoring in it."""
    g = shots.groupby(["match_id", "team", "possession"])
    out = g.agg(minute=("minute", "first"), period=("period", "first"), goal=("goal", "max"),
                n_shots=("xg", "size")).reset_index()
    out["pxg"] = g.xg.apply(lambda x: 1 - np.prod(1 - x.to_numpy())).to_numpy()
    return out
