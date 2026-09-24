"""League translation model: how does a player's per-90 output change when he moves league?

Model (per metric m, per-90, fitted on consecutive seasons with >= 900 minutes in both):

    log(m_next + eps) = a + b * log(m_now + eps) + beta[league_next] - beta[league_now]

* a, b     : year-to-year persistence. b < 1 is regression to the mean (a hot season cools down).
             Estimated mostly from ~11,000 players who STAYED in the same league.
* beta[L]  : league effect, EPL fixed at 0. beta[L] > 0 means the same player produces MORE of
             that metric in league L than in the EPL (i.e. L is "easier" for that metric).
             Identified by players who MOVED between leagues.

Projection of a player from league L into the EPL next season:
    m_EPL = exp(a + b * log(m_now + eps) - beta[L]) - eps

Stayers enter only through sufficient statistics (sum w, sum wx, ...), so the snapshot stays small.
Weights: harmonic mean of the two seasons' minutes / 900.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "data" / "translation"
LEAGUES = {"E": "Premier League", "S": "La Liga", "D": "Bundesliga", "I": "Serie A", "F": "Ligue 1", "R": "Russian PL"}
FREE = ["S", "D", "I", "F", "R"]  # EPL is the reference (beta = 0)
METRICS = ["npxG", "xA", "xGChain"]
EPS = 0.05


# ------------------------------------------------------------------ data
def load_movers(path: Path = SNAP) -> pd.DataFrame:
    if not list(path.glob("movers_*.txt")):
        raise FileNotFoundError(f"No translation snapshot in {path}; see data/README.md for the format.")
    rows = []
    for f in sorted(path.glob("movers_*.txt")):
        rows += [r.split(",") for r in f.read_text().strip().split(";")]
    df = pd.DataFrame(rows, columns=["player", "pos", "move", "season", "min1", "min2",
                                     "npxG_1", "xA_1", "xGChain_1", "npxG_2", "xA_2", "xGChain_2"])
    df["from_lg"], df["to_lg"] = df.move.str[0], df.move.str[1]
    df["season"] = 2000 + df.season.astype(int)
    for c in ["min1", "min2"]:
        df[c] = df[c].astype(int)
    for m in METRICS:
        for k in ("1", "2"):
            df[f"{m}_{k}"] = df[f"{m}_{k}"].astype(int) / 1000
    df["w"] = 2 / (900 / df.min1 + 900 / df.min2)
    return df.drop(columns="move")


def load_stayers(path: Path = SNAP) -> dict:
    return json.loads((path / "stayers.json").read_text())


def load_candidates(path: Path = SNAP) -> pd.DataFrame:
    rows = [r.split(",") for r in (path / "candidates_2025.txt").read_text().strip().split(";")]
    df = pd.DataFrame(rows, columns=["player", "pos", "league", "team", "minutes",
                                     "npxG", "xA", "xGChain", "goals", "assists"])
    for c in ["minutes", "goals", "assists"]:
        df[c] = df[c].astype(int)
    for m in METRICS:
        df[m] = df[m].astype(int) / 1000
    return df


def load_epl(path: Path = SNAP) -> pd.DataFrame:
    rows = [r.split(",") for r in (path / "epl_2025.txt").read_text().strip().split(";")]
    df = pd.DataFrame(rows, columns=["player", "pos", "npxG", "xA", "xGChain"])
    for m in METRICS:
        df[m] = df[m].astype(int) / 1000
    return df


# ------------------------------------------------------------------ model
def _design(movers: pd.DataFrame, metric: str):
    x = np.log(movers[f"{metric}_1"].to_numpy() + EPS)
    y = np.log(movers[f"{metric}_2"].to_numpy() + EPS)
    L = np.zeros((len(movers), len(FREE)))
    for j, lg in enumerate(FREE):
        L[:, j] = (movers.to_lg == lg).astype(float) - (movers.from_lg == lg).astype(float)
    X = np.column_stack([np.ones(len(movers)), x, L])
    return X, y, movers.w.to_numpy()


def fit(movers: pd.DataFrame, stay: dict, metric: str, ridge: float = 1e-6) -> dict:
    X, y, w = _design(movers, metric)
    XtWX = (X * w[:, None]).T @ X
    XtWy = (X * w[:, None]).T @ y
    s = stay[metric]
    XtWX[:2, :2] += np.array([[s["w"], s["x"]], [s["x"], s["xx"]]])
    XtWy[:2] += np.array([s["y"], s["xy"]])
    theta = np.linalg.solve(XtWX + ridge * np.eye(len(XtWX)), XtWy)
    return {"metric": metric, "a": theta[0], "b": theta[1],
            "beta": {"E": 0.0, **{lg: theta[2 + j] for j, lg in enumerate(FREE)}}}


def project(model: dict, value, league_from: str, league_to: str = "E"):
    """Expected per-90 value next season after moving from league_from to league_to."""
    z = model["a"] + model["b"] * np.log(np.asarray(value) + EPS) + model["beta"][league_to] - model["beta"][league_from]
    return np.maximum(np.exp(z) - EPS, 0)


def bootstrap(movers: pd.DataFrame, stay: dict, metric: str, n: int = 500, seed: int = 11) -> pd.DataFrame:
    """Bootstrap the league effects by resampling movers (stayers are held fixed)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(movers), len(movers))
        out.append(fit(movers.iloc[idx], stay, metric)["beta"])
    return pd.DataFrame(out)


def cross_validate(movers: pd.DataFrame, stay: dict, metric: str, k: int = 10, seed: int = 3) -> dict:
    """k-fold CV on movers: weighted MAE of predicted next-season per-90 value.

    Compared with two naive baselines:
      * 'same'   : next season = this season (what a raw stat sheet implies)
      * 'no_lg'  : persistence model without league effects (regression to the mean only)
    """
    rng = np.random.default_rng(seed)
    folds = rng.integers(0, k, len(movers))
    err = {"model": [], "same": [], "no_lg": []}
    wts = []
    for f in range(k):
        tr, te = movers[folds != f], movers[folds == f]
        mdl = fit(tr, stay, metric)
        truth = te[f"{metric}_2"].to_numpy()
        pred = project(mdl, te[f"{metric}_1"].to_numpy(), "E", "E")  # placeholder, replaced below
        pred = np.array([project(mdl, v, a, b) for v, a, b in zip(te[f"{metric}_1"], te.from_lg, te.to_lg)])
        no_lg = {**mdl, "beta": {lg: 0.0 for lg in mdl["beta"]}}
        base = np.array([project(no_lg, v, "E", "E") for v in te[f"{metric}_1"]])
        err["model"] += list(np.abs(pred - truth))
        err["same"] += list(np.abs(te[f"{metric}_1"].to_numpy() - truth))
        err["no_lg"] += list(np.abs(base - truth))
        wts += list(te.w)
    wts = np.array(wts)
    return {name: float(np.average(e, weights=wts)) for name, e in err.items()}


def fit_all(movers=None, stay=None):
    movers = load_movers() if movers is None else movers
    stay = load_stayers() if stay is None else stay
    return {m: fit(movers, stay, m) for m in METRICS}


def shortlist(models: dict, cand: pd.DataFrame, epl: pd.DataFrame) -> pd.DataFrame:
    """Project every non-EPL candidate into the EPL and rank against EPL regulars' own projections."""
    c = cand.copy()
    for m in METRICS:
        c[f"{m}_epl"] = [float(project(models[m], v, lg, "E")) for v, lg in zip(c[m], c.league)]
    e = epl.copy()
    for m in METRICS:
        e[f"{m}_epl"] = project(models[m], e[m].to_numpy(), "E", "E")
    c["output_epl"] = c.npxG_epl + c.xA_epl
    e["output_epl"] = e.npxG_epl + e.xA_epl
    ref = np.sort(e.output_epl.to_numpy())
    c["epl_percentile"] = np.searchsorted(ref, c.output_epl.to_numpy(), side="right") / len(ref)
    c["league_name"] = c.league.map(LEAGUES)
    return c.sort_values("output_epl", ascending=False).reset_index(drop=True)
