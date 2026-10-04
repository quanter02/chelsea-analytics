"""프리미어리그 경기 예측 (openfootball 결과 자료, 2010/11~).

모델: 팀마다 공격력 a, 수비력 d 를 경기마다 갱신하는 '온라인 포아송' 모델
  홈 기대득점  λ_h = exp(μ + h + a_홈 − d_원정)
  원정 기대득점 λ_a = exp(μ + a_원정 − d_홈)
  결과 확률 = 두 포아송의 곱 (무승부 칸은 (1+ρ)배 후 재정규화)
  경기 후: 실제 골 − 기대골 만큼 k 비율로 a, d 갱신 (경사 상승)
  시즌 시작: 기존 팀은 c 배로 평균 쪽으로 당김, 승격 팀은 공격·수비 p 로 시작

예측은 경기 전에 알 수 있는 정보만으로 만든다 (경기 순서대로 한 번 지나가며 예측 → 갱신).
"""
from __future__ import annotations

import json
from math import exp, factorial
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).parent.parent / "data_epl"
SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2010, 2027)]
MAXG = 10


def norm(team: str) -> str:
    """시즌마다 다른 표기 통일: 'Aston Villa FC' = 'Aston Villa', 'AFC Bournemouth' = 'Bournemouth'."""
    t = team.strip()
    for suf in (" AFC", " FC"):
        if t.endswith(suf):
            t = t[: -len(suf)]
    return t[4:] if t.startswith("AFC ") else t


def load() -> pd.DataFrame:
    rows = []
    for s in SEASONS:
        f = D / f"en1_{s}.json"
        if not f.exists():
            continue
        for x in json.load(open(f, encoding="utf-8"))["matches"]:
            sc = x.get("score")
            ft = sc.get("ft") if isinstance(sc, dict) else (sc if isinstance(sc, list) and len(sc) == 2 else None)
            rows.append(dict(season=s, date=x["date"], home=norm(x["team1"]), away=norm(x["team2"]),
                             hg=ft[0] if ft else np.nan, ag=ft[1] if ft else np.nan, played=ft is not None))
    df = pd.DataFrame(rows).sort_values(["date"], kind="stable").reset_index(drop=True)
    df["result"] = np.select([df.hg > df.ag, df.hg == df.ag], ["H", "D"], "A")
    df.loc[~df.played, "result"] = None
    return df


_FACT = np.array([factorial(i) for i in range(MAXG + 1)], float)


def probs(lh: float, la: float, rho: float) -> np.ndarray:
    g = np.arange(MAXG + 1)
    ph = np.exp(-lh) * lh ** g / _FACT
    pa = np.exp(-la) * la ** g / _FACT
    m = np.outer(ph, pa)
    m[np.diag_indices_from(m)] *= (1 + rho)
    m /= m.sum()
    return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])   # 홈승, 무, 원정승


def run(df: pd.DataFrame, k: float = 0.05, h: float = 0.25, c: float = 0.8, p: float = -0.2, rho: float = 0.0,
        mu0: float = 0.3) -> pd.DataFrame:
    """경기 순서대로 예측 → 갱신. 반환: 경기별 [pH, pD, pA] 와 기대골."""
    a, d = {}, {}
    mu = mu0
    season, teams_prev = None, set()
    out = np.zeros((len(df), 5))
    for i, r in enumerate(df.itertuples()):
        if r.season != season:                                   # 시즌 시작
            teams = set(df.loc[df.season == r.season, "home"])
            for t in list(a):
                a[t] *= c; d[t] *= c
            for t in teams - set(a):
                a[t], d[t] = p, p
            season = r.season
        lh = exp(mu + h + a[r.home] - d[r.away]); la = exp(mu + a[r.away] - d[r.home])
        out[i, :3] = probs(lh, la, rho); out[i, 3:] = (lh, la)
        if r.played:
            eh, ea = r.hg - lh, r.ag - la
            a[r.home] += k * eh; d[r.away] -= k * eh
            a[r.away] += k * ea; d[r.home] -= k * ea
            mu += k * 0.05 * (eh + ea)
    res = df.copy()
    res[["pH", "pD", "pA", "xh", "xa"]] = out
    return res


def losses(res: pd.DataFrame, kind: str = "rps") -> np.ndarray:
    """경기별 손실. rps = 순위 확률 점수 (축구 예측 표준), logloss = 로그 손실."""
    r = res[res.played]
    P = r[["pH", "pD", "pA"]].to_numpy()
    y = np.zeros_like(P); y[np.arange(len(r)), r.result.map({"H": 0, "D": 1, "A": 2}).to_numpy()] = 1
    if kind == "logloss":
        return -np.log(np.clip((P * y).sum(axis=1), 1e-12, 1))
    cp, cy = np.cumsum(P, axis=1)[:, :2], np.cumsum(y, axis=1)[:, :2]
    return ((cp - cy) ** 2).sum(axis=1) / 2


SPLITS = {"train": SEASONS[:11], "val": ["2021-22", "2022-23", "2023-24"], "test": ["2024-25", "2025-26"], "live": ["2026-27"]}
GRID = {"k": [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1], "h": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        "c": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "p": [0.0, -0.1, -0.2, -0.3, -0.4], "rho": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]}
START = dict(k=0.05, h=0.25, c=0.8, p=-0.2, rho=0.0)


def make_spec(df: pd.DataFrame, kind: str = "rps"):
    from .tuning import Spec
    cache = {}

    def evaluate(params, split):
        key = tuple(sorted(params.items()))
        if key not in cache:
            cache[key] = run(df, **params)
        res = cache[key]
        return losses(res[res.season.isin(SPLITS[split])], kind)
    return Spec(name="EPL 경기 예측", grid=GRID, start=START, evaluate=evaluate, metric=kind.upper())


def baseline(df: pd.DataFrame, split: str, kind: str = "rps") -> np.ndarray:
    """학습 구간의 홈승·무·원정승 비율을 모든 경기에 똑같이 쓰는 기준선."""
    tr = df[df.season.isin(SPLITS["train"]) & df.played]
    f = tr.result.value_counts(normalize=True).reindex(["H", "D", "A"]).to_numpy()
    r = df[df.season.isin(SPLITS[split])].copy()
    r[["pH", "pD", "pA"]] = f
    return losses(r, kind)
