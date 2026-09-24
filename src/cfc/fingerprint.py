"""Manager fingerprint: compare a team's style this season with the manager's previous team.

Raw numbers from different leagues are not comparable (the Bundesliga is more open than the EPL),
so every metric is turned into a percentile *within its own league-season*. A fingerprint is the
vector of those percentiles; similarity = 1 - mean absolute difference.

All metrics come from Understat team-match data, so both sides use the same definitions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "data" / "fingerprint"

# label, higher-is-"more" orientation already applied in _metrics
METRICS = {
    "attack": "공격 생산 (npxG/경기)",
    "defence": "실점 억제 (npxGA/경기, 역순)",
    "control": "경기 지배 (xG 점유율)",
    "press": "전방 압박 강도 (1/PPDA)",
    "press_resist": "압박 회피·빌드업 (상대 PPDA)",
    "penetration": "침투 (딥 컴플리션/경기)",
    "box_guard": "박스 앞 봉쇄 (허용 딥 컴플리션, 역순)",
}


def _metrics(t: pd.DataFrame) -> pd.DataFrame:
    """t: one row per team with season totals: matches, npxg, npxga, xg, xga, ppda_att, ppda_def,
    pa_att, pa_def, deep, deep_allowed."""
    m = pd.DataFrame(index=t.index)
    g = t.matches
    m["attack"] = t.npxg / g
    m["defence"] = -t.npxga / g
    m["control"] = t.xg / (t.xg + t.xga)
    m["press"] = t.ppda_def / t.ppda_att            # 1 / PPDA
    m["press_resist"] = t.pa_att / t.pa_def         # PPDA the opponents manage against us
    m["penetration"] = t.deep / g
    m["box_guard"] = -t.deep_allowed / g
    return m


def percentiles(t: pd.DataFrame) -> pd.DataFrame:
    return _metrics(t).rank(pct=True)


def load_bundesliga_2023() -> pd.DataFrame:
    rows = [r.split(",") for r in (SNAP / "bundesliga_2023.txt").read_text().strip().split(";")]
    t = pd.DataFrame(rows, columns=["team", "matches", "xg", "xga", "npxg", "npxga", "ppda_att", "ppda_def",
                                    "pa_att", "pa_def", "deep", "deep_allowed", "xpts", "pts", "gf", "ga"])
    return t.set_index("team").astype(float)


def from_team_matches(tm: pd.DataFrame) -> pd.DataFrame:
    """Season totals from understat.build() team-match rows."""
    agg = tm.groupby("team").agg(matches=("match_id", "size"), xg=("xg", "sum"), xga=("xga", "sum"),
                                 npxg=("npxg", "sum"), npxga=("npxga", "sum"), ppda_att=("ppda_att", "sum"),
                                 ppda_def=("ppda_def", "sum"), pa_att=("pa_att", "sum"), pa_def=("pa_def", "sum"),
                                 deep=("deep", "sum"), deep_allowed=("deep_allowed", "sum"))
    return agg.astype(float)


def similarity(a: pd.Series, b: pd.Series) -> float:
    return float(1 - (a - b).abs().mean())


def compare(now: pd.DataFrame, team: str, before: pd.DataFrame, old_team: str) -> dict:
    pn, pb = percentiles(now), percentiles(before)
    a, b = pn.loc[team], pb.loc[old_team]
    diff = (a - b)
    closest = {t: similarity(pn.loc[t], b) for t in pn.index}
    return {
        "now": a, "before": b, "diff": diff, "similarity": similarity(a, b),
        "matched": [METRICS[k] for k in diff.index if abs(diff[k]) <= 0.2],
        "gaps": [(METRICS[k], float(diff[k])) for k in diff.abs().sort_values(ascending=False).index if abs(diff[k]) > 0.2],
        "closest": pd.Series(closest).sort_values(ascending=False),
    }

METRICS_EN = {
    "attack": "Attack (npxG per game)",
    "defence": "Defence (npxGA per game, inverted)",
    "control": "Control (xG share)",
    "press": "Pressing intensity (1/PPDA)",
    "press_resist": "Press resistance (opponents' PPDA)",
    "penetration": "Penetration (deep completions)",
    "box_guard": "Box protection (deep allowed, inv.)",
}
