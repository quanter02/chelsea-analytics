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


FBREF = {"Brighton": "Brighton & Hove Albion", "Huddersfield": "Huddersfield Town", "Manchester Utd": "Manchester United",
         "Newcastle Utd": "Newcastle United", "Nott'ham Forest": "Nottingham Forest", "Sheffield Utd": "Sheffield United",
         "Tottenham": "Tottenham Hotspur", "West Brom": "West Bromwich Albion", "West Ham": "West Ham United", "Wolves": "Wolverhampton Wanderers"}


def load_xg() -> pd.DataFrame:
    """fbref 경기별 xG (worldfootballR_data 공개 사본, 2017/18~2025/26 초반). 이름을 openfootball 표기로 맞춘다."""
    f = D / "fbref_epl_xg.csv"
    if not f.exists():
        return pd.DataFrame(columns=["date", "home", "away", "hxg", "axg"])
    x = pd.read_csv(f)
    out = pd.DataFrame(dict(date=x.Date, home=x.Home.replace(FBREF), away=x.Away.replace(FBREF), hxg=x.Home_xG, axg=x.Away_xG))
    for m in sorted(D.glob("xg_understat_*.csv")) + [D / "xg_manual.csv"]:   # understat 자동 수집 → 손으로 넣은 값이 마지막에 우선
        if not m.exists():
            continue
        add = pd.read_csv(m)
        if add.empty:
            continue
        add["home"], add["away"] = add.home.map(norm), add.away.map(norm)
        out = pd.concat([out, add[["date", "home", "away", "hxg", "axg"]]]).drop_duplicates(["date", "home", "away"], keep="last")
    return out


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
    xg = load_xg()
    df = df.merge(xg, on=["date", "home", "away"], how="left")
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


BASE = np.array([0.46, 0.25, 0.29])      # 리그 평균 홈승·무·원정승 (2010/11~2020/21 학습 구간)


def run(df: pd.DataFrame, k: float = 0.05, h: float = 0.25, c: float = 0.8, p: float = -0.2, rho: float = 0.0,
        w: float = 0.0, g: float = 0.0, mu0: float = 0.3) -> pd.DataFrame:
    """w: 확률을 리그 평균 쪽으로 당기는 비율 (과신 보정). 0이면 그대로.
    g: 팀 실력 갱신에 쓰는 '득점'을 실제 골과 xG 중 얼마나 xG로 볼지 (xG가 있는 경기만). 0이면 골만."""
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
        out[i, :3] = (1 - w) * probs(lh, la, rho) + w * BASE; out[i, 3:] = (lh, la)
        if r.played:
            th, ta = r.hg, r.ag
            if g and not np.isnan(r.hxg):                           # 골은 운이 섞이므로 xG를 섞어 실력을 더 안정적으로 갱신
                th, ta = (1 - g) * r.hg + g * r.hxg, (1 - g) * r.ag + g * r.axg
            eh, ea = th - lh, ta - la
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
        "c": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "p": [0.0, -0.1, -0.2, -0.3, -0.4], "rho": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25], "w": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3], "g": [0.0, 0.25, 0.5, 0.75, 1.0]}
START = dict(k=0.05, h=0.25, c=0.8, p=-0.2, rho=0.0, w=0.0, g=0.0)
# 실험대가 고른 설정 (2026-10-04). 골 모드: 이번 시즌 xG 없음 → 지금 사용. xG 모드: 이번 시즌 xG가 쌓이면 사용.
GOALS_MODE = dict(k=0.04, h=0.25, c=0.8, p=-0.2, rho=0.0, w=0.0, g=0.0)
XG_MODE = dict(k=0.05, h=0.25, c=0.9, p=-0.2, rho=0.0, w=0.0, g=0.5)


def current_mode(df: pd.DataFrame, season: str = "2026-27", need: int = 30) -> tuple[str, dict]:
    """이번 시즌 xG가 붙은 경기가 need개 이상이면 xG 모드."""
    n = int(df[(df.season == season) & df.played].hxg.notna().sum())
    return ("xG 모드", XG_MODE) if n >= need else ("골 모드", GOALS_MODE)


def hide_xg(df: pd.DataFrame, seasons) -> pd.DataFrame:
    """지정 시즌의 xG를 가림. 지금(이번 시즌 xG 없음)과 같은 조건으로 검증·시험하기 위해."""
    out = df.copy()
    out.loc[out.season.isin(seasons), ["hxg", "axg"]] = np.nan
    return out


def make_spec(df: pd.DataFrame, kind: str = "rps", deploy_like: bool = False):
    """deploy_like=True: 검증·시험 시즌의 xG를 가리고 튜닝 (실제로 쓸 때와 같은 조건)."""
    if deploy_like:
        df = hide_xg(df, SPLITS["val"] + SPLITS["test"] + SPLITS["live"])
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


def score_benchmark(path: str | Path, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """비교 기록(우리 모델 vs Opta 등)을 실제 결과로 채점. 결과 없는 경기는 pending."""
    df = load() if df is None else df
    b = pd.read_csv(path)
    m = b.merge(df[["date", "home", "away", "result", "played"]], on=["date", "home", "away"], how="left")
    y = m.result.map({"H": 0, "D": 1, "A": 2})
    P = m[["pH", "pD", "pA"]].to_numpy()
    rps = []
    for i in range(len(m)):
        if pd.isna(y.iloc[i]):
            rps.append(np.nan); continue
        o = np.zeros(3); o[int(y.iloc[i])] = 1
        rps.append(((np.cumsum(P[i])[:2] - np.cumsum(o)[:2]) ** 2).sum() / 2)
    m["rps"] = rps
    m["status"] = np.where(m.rps.isna(), "pending", "scored")
    return m
