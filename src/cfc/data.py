"""StatsBomb open-data loader with a local JSON cache.

Data: https://github.com/statsbomb/open-data (free, non-commercial use, credit StatsBomb).
"""
from __future__ import annotations

import gzip
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
CACHE = Path(__file__).resolve().parents[2] / "data" / "cache"

PREMIER_LEAGUE = 2
SEASON_2015_16 = 27


def _get_json(rel: str):
    path = CACHE / (rel + ".gz")
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{BASE}/{rel}", timeout=60) as r:
        raw = r.read()
    path.write_bytes(gzip.compress(raw, compresslevel=6))
    return json.loads(raw)


def _manager(side: dict) -> str:
    mgrs = side.get("managers") or []
    return mgrs[0].get("name", "Unknown") if mgrs else "Unknown"


def matches(competition_id: int = PREMIER_LEAGUE, season_id: int = SEASON_2015_16) -> pd.DataFrame:
    """One row per match, with home/away team, score and managers."""
    rows = []
    for m in _get_json(f"matches/{competition_id}/{season_id}.json"):
        h, a = m["home_team"], m["away_team"]
        rows.append({
            "match_id": m["match_id"],
            "date": pd.Timestamp(m["match_date"]),
            "home": h["home_team_name"],
            "away": a["away_team_name"],
            "home_score": m["home_score"],
            "away_score": m["away_score"],
            "home_manager": _manager(h),
            "away_manager": _manager(a),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def team_matches(team: str, **kw) -> pd.DataFrame:
    """Matches of one team from its own perspective (goals for/against, opponent, manager)."""
    df = matches(**kw)
    df = df[(df.home == team) | (df.away == team)].copy()
    is_home = df.home == team
    df["venue"] = is_home.map({True: "H", False: "A"})
    df["opponent"] = df.away.where(is_home, df.home)
    df["gf"] = df.home_score.where(is_home, df.away_score)
    df["ga"] = df.away_score.where(is_home, df.home_score)
    df["manager"] = df.home_manager.where(is_home, df.away_manager)
    df["points"] = (df.gf > df.ga) * 3 + (df.gf == df.ga) * 1
    return df.reset_index(drop=True)


def events(match_id: int) -> list[dict]:
    return _get_json(f"events/{match_id}.json")


def prefetch(match_ids, workers: int = 8) -> None:
    """Download many matches in parallel (cached after the first run)."""
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(events, match_ids))
