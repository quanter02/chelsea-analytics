"""Current-season data from Understat (shot-level xG + match-level PPDA / deep completions).

Two ways to get the data:
  1. `fetch_season(2026)` — run on your own machine; downloads via Understat's JSON endpoints.
  2. A compact snapshot in data/understat_2026/ (what the published report was built from).

Both produce the same tables, in the same schema metrics.py / scouting.py use for StatsBomb, so
the scouting rules run unchanged. Columns Understat cannot provide (pass-based metrics, counter
attacks, assist locations) are left as NaN and the rules that need them simply don't fire.

Understat is free for personal / research use. Do not resell its data.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "data" / "understat_2026"

RESULT = {"G": "Goal", "W": "OwnGoal", "S": "SavedShot", "M": "MissedShots", "B": "BlockedShot", "P": "ShotOnPost"}
SITUATION = {"O": "OpenPlay", "C": "FromCorner", "S": "SetPiece", "F": "DirectFreekick", "P": "Penalty"}
BODY = {"R": "Right Foot", "L": "Left Foot", "H": "Head", "X": "Other"}


# ------------------------------------------------------------------ loading
def fetch_season(season: int = 2026, league: str = "EPL", pause: float = 0.5) -> tuple[list, dict]:
    """Download a season from Understat (run locally; needs `requests`)."""
    import time

    import requests

    s = requests.Session()
    s.headers.update({"X-Requested-With": "XMLHttpRequest", "User-Agent": "Mozilla/5.0"})
    league_data = s.get(f"https://understat.com/getLeagueData/{league}/{season}", timeout=30).json()
    shots = []
    for m in league_data["dates"]:
        if not m["isResult"]:
            continue
        d = s.get(f"https://understat.com/getMatchData/{m['id']}", timeout=30).json()
        for side in ("h", "a"):
            shots += d["shots"][side]
        time.sleep(pause)
    return shots, league_data


def _raw_from_api(shots: list, league_data: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sh = pd.DataFrame(shots)
    sh = pd.DataFrame({
        "match_id": sh.match_id.astype(int), "side": sh.h_a, "minute": sh.minute.astype(int),
        "X": sh.X.astype(float), "Y": sh.Y.astype(float), "xg": sh.xG.astype(float),
        "result": sh.result, "situation": sh.situation, "body_part": sh.shotType.map(
            {"RightFoot": "Right Foot", "LeftFoot": "Left Foot", "Head": "Head"}).fillna("Other"),
        "last_action": sh.lastAction.fillna("None"), "player": sh.player.map(html.unescape),
        "assist_player": sh.player_assisted.map(lambda p: html.unescape(p) if p else None),
    })
    ms = pd.DataFrame([{
        "match_id": int(m["id"]), "date": m["datetime"][:10], "home": m["h"]["title"], "away": m["a"]["title"],
        "home_score": int(m["goals"]["h"]), "away_score": int(m["goals"]["a"]),
    } for m in league_data["dates"] if m["isResult"]])
    th = pd.DataFrame([{
        "team": t["title"], "date": h["date"][:10], "ppda_att": h["ppda"]["att"], "ppda_def": h["ppda"]["def"],
        "pa_att": h["ppda_allowed"]["att"], "pa_def": h["ppda_allowed"]["def"],
        "deep": h["deep"], "deep_allowed": h["deep_allowed"], "xpts": h["xpts"],
    } for t in league_data["teams"].values() for h in t["history"]])
    return sh, ms, th


def _raw_from_snapshot(path: Path = SNAP) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (path / "matches.txt").exists():
        raise FileNotFoundError(f"No Understat snapshot in {path}. Run `python -m cfc.live --fetch` "
                                "on your own machine, or see data/README.md.")
    names = [html.unescape(n) for n in (path / "names.txt").read_text().strip().split("|")]
    la = {v: k for k, v in json.loads((path / "lastaction.json").read_text()).items()}
    ms = pd.DataFrame([r.split(",") for r in (path / "matches.txt").read_text().strip().split(";")],
                      columns=["match_id", "date", "home", "away", "home_score", "away_score"])
    ms = ms.astype({"match_id": int, "home_score": int, "away_score": int})
    rows = []
    for f in ("shots_1.txt", "shots_2.txt"):
        rows += [r.split(",") for r in (path / f).read_text().strip().split(";")]
    sh = pd.DataFrame(rows, columns=["m", "side", "minute", "X", "Y", "xg", "res", "sit", "body", "la", "p", "a"])
    sh = pd.DataFrame({
        "match_id": ms.match_id.to_numpy()[sh.m.astype(int)], "side": sh.side.map({"0": "h", "1": "a"}),
        "minute": sh.minute.astype(int), "X": sh.X.astype(int) / 1000, "Y": sh.Y.astype(int) / 1000,
        "xg": sh.xg.astype(int) / 10000, "result": sh.res.map(RESULT), "situation": sh.sit.map(SITUATION),
        "body_part": sh.body.map(BODY), "last_action": sh.la.map(la),
        "player": sh.p.map(lambda i: names[int(i)] if i else None),
        "assist_player": sh.a.map(lambda i: names[int(i)] if i else None),
    })
    th = pd.DataFrame([r.split(",") for r in (path / "teams.txt").read_text().strip().split(";")],
                      columns=["team", "date", "h_a", "ppda_att", "ppda_def", "pa_att", "pa_def",
                               "deep", "deep_allowed", "xpts"])
    th = th.astype({"ppda_att": int, "ppda_def": int, "deep": int, "deep_allowed": int, "xpts": float})
    th = th.astype({"pa_att": int, "pa_def": int})
    return sh, ms, th[["team", "date", "ppda_att", "ppda_def", "pa_att", "pa_def", "deep", "deep_allowed", "xpts"]]


# ------------------------------------------------------------------ tables in the project schema
def build(sh: pd.DataFrame, ms: pd.DataFrame, th: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (team_matches, shots) in the same schema as metrics.season_team_matches / season_shots."""
    sh = sh.merge(ms, on="match_id").sort_values(["match_id", "minute"], kind="stable").reset_index(drop=True)
    own = sh.result == "OwnGoal"
    shooter = np.where(sh.side == "h", sh.home, sh.away)
    other = np.where(sh.side == "h", sh.away, sh.home)
    sh["team"] = shooter
    sh["benefit"] = np.where(own, other, shooter)  # own goals count for the other side

    # game state before each shot (from the shooting team's view), own goals included
    states = []
    for mid, g in sh.groupby("match_id", sort=False):
        score = {}
        for _, r in g.iterrows():
            opp = r.away if r.team == r.home else r.home
            d = score.get(r.team, 0) - score.get(opp, 0)
            states.append("leading" if d > 0 else "trailing" if d < 0 else "level")
            if r.result in ("Goal", "OwnGoal"):
                score[r.benefit] = score.get(r.benefit, 0) + 1
    sh["state"] = states

    shots = sh[~own].copy()
    shots = pd.DataFrame({
        "match_id": shots.match_id, "team": shots.team, "player": shots.player, "minute": shots.minute,
        # StatsBomb-like 120x80 pitch, attacking left -> right (Understat Y=0 is the attacker's right)
        "x": shots.X * 120, "y": (1 - shots.Y) * 80, "xg": shots.xg,
        "goal": shots.result == "Goal", "on_target": shots.result.isin(["Goal", "SavedShot"]),
        "shot_type": np.where(shots.situation == "Penalty", "Penalty", "Open Play"),
        "body_part": shots.body_part, "pattern": shots.situation,
        "origin": np.where(shots.situation == "OpenPlay", "open_play", "set_piece"),
        "assist_player": shots.assist_player, "assist_x": np.nan, "assist_y": np.nan,
        "assist_cross": shots.last_action == "Cross", "assist_through": shots.last_action == "Throughball",
        "state": shots.state, "period": np.where(shots.minute < 46, 1, 2), "possession": np.nan,
    }).reset_index(drop=True)

    rows = []
    for _, m in ms.iterrows():
        for team, opp, gf, ga in ((m.home, m.away, m.home_score, m.away_score),
                                  (m.away, m.home, m.away_score, m.home_score)):
            f = shots[(shots.match_id == m.match_id) & (shots.team == team)]
            a = shots[(shots.match_id == m.match_id) & (shots.team == opp)]
            npf, npa = f[f.shot_type != "Penalty"], a[a.shot_type != "Penalty"]
            rows.append({
                "match_id": m.match_id, "date": pd.Timestamp(m.date), "team": team, "opponent": opp,
                "venue": "H" if team == m.home else "A", "goals": gf, "goals_against": ga,
                "points": 3 if gf > ga else 1 if gf == ga else 0,
                "shots": len(f), "shots_against": len(a), "xg": f.xg.sum(), "xga": a.xg.sum(),
                "npxg": npf.xg.sum(), "npxga": npa.xg.sum(),
                "xg_open": f.loc[f.origin == "open_play", "xg"].sum(),
                "xg_setpiece": f.loc[f.origin == "set_piece", "xg"].sum(),
                "xga_open": a.loc[a.origin == "open_play", "xg"].sum(),
                "xga_setpiece": a.loc[a.origin == "set_piece", "xg"].sum(),
                "xg_through": f.loc[f.assist_through, "xg"].sum(),
                "xg_cross": f.loc[f.assist_cross, "xg"].sum(),
                # not available from Understat
                "xg_counter": np.nan, "xga_counter": np.nan, "pass_share": np.nan, "field_tilt": np.nan,
                "long_ball_share": np.nan, "high_turnovers": np.nan, "box_entries": np.nan,
            })
    tm = pd.DataFrame(rows)
    th = th.assign(date=pd.to_datetime(th.date))
    tm = tm.merge(th, on=["team", "date"], how="left")
    tm["ppda"] = tm.ppda_att / tm.ppda_def.clip(lower=1)
    return tm, shots


def load_snapshot() -> tuple[pd.DataFrame, pd.DataFrame]:
    return build(*_raw_from_snapshot())


def load_live(season: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build(*_raw_from_api(*fetch_season(season)))
