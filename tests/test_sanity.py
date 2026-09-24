"""Sanity checks against the official 2015/16 Premier League table."""
from pathlib import Path

import pytest

from cfc import metrics as M, scouting as S

DATA = Path(__file__).resolve().parents[1] / "data"
needs = lambda d: pytest.mark.skipif(not (DATA / d).exists(), reason=f"data/{d} not present (see data/README.md)")

OFFICIAL = {"Leicester City": 81, "Arsenal": 71, "Tottenham Hotspur": 70, "Chelsea": 50, "Aston Villa": 17}


def test_points_match_official_table():
    tm = M.season_team_matches()
    pts = tm.groupby("team").points.sum()
    for team, p in OFFICIAL.items():
        assert pts[team] == p


def test_every_match_has_two_teams():
    tm = M.season_team_matches()
    assert len(tm) == 760 and tm.groupby("match_id").size().eq(2).all()


def test_scouting_plan_is_generated():
    tm, sh = M.season_team_matches(), M.season_shots()
    plan = S.game_plan(S.scout("Leicester City", tm, sh))
    assert 3 <= len(plan) <= 5


@needs("understat_2026")
def test_understat_snapshot_matches_scores():
    from cfc import understat as U
    tm, shots = U.load_snapshot()
    assert tm.match_id.nunique() == 50 and len(tm) == 100
    chelsea = tm[tm.team == "Chelsea"]
    assert chelsea.points.sum() == 7 and chelsea.goals.sum() == 10 and chelsea.goals_against.sum() == 12


@needs("translation")
def test_translation_model_is_sane():
    from cfc import translation as T
    models = T.fit_all()
    for m in ("npxG", "xA"):
        assert 0.6 < models[m]["b"] < 1.0                     # regression to the mean
        for lg in T.FREE:                                      # other leagues are easier than the EPL
            assert 0.7 < float(__import__("numpy").exp(-models[m]["beta"][lg])) < 1.05
