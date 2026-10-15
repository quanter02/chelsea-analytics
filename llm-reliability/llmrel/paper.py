"""가상 베팅 장부: 돈을 걸지 않고, 경기 전에 '이 배당이면 건다'를 장부에 고정한 뒤 시즌 후 채점한다.

블룸(Starlizard) 방식 1단계 = 우위 증명을 합법적으로 하는 틀. 실제 돈은 걸지 않는다 (한국 국적자의 사설·해외 베팅은 불법).

규칙 v1 (결과 보기 전에 장부에 고정, kind=paper_rules, key=v1)
  - 배당: 경기 전 Bet365 (실제로 받을 수 있었던 가격). 마감 배당은 채점에만 쓴다
  - 기대 이득 EV = 우리 확률 × 배당 − 1. 한 경기에서 EV가 가장 큰 쪽 하나, EV ≥ 5% 이고 배당 ≤ 6.0 일 때만
  - 판돈 1단위 고정
  - 주 지표 CLV = 받은 배당 × 마감 공정 확률(평균 마감 배당, 마진 제거) − 1.
    운보다 덜 흔들려서 프로들이 '우위가 있나'를 보는 기준. 보조 지표는 실제 수익률(ROI)
  - 판정: CLV 평균의 95% 범위 전체 > 0 이면 우위 있음(맞음), 전체 < 0 이면 없음(틀림), 걸치면 불확실
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import market

RULES = dict(version="v1", odds="B365 경기 전", ev_min=0.05, odds_max=6.0, stake=1.0, one_per_match=True,
             clv="받은 배당 × 평균 마감 공정 확률 − 1", verdict="CLV 평균 95% 범위로 3단계")
SIDES = ["H", "D", "A"]


def pre_and_close() -> pd.DataFrame:
    """경기별 경기 전 Bet365 배당(oH,oD,oA)과 마감 공정 확률(cH,cD,cA)."""
    parts = []
    for f in sorted(market.ODDS.glob("E0_*.csv")):
        x = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip").dropna(subset=["HomeTeam"])
        d = pd.to_datetime(x.Date, format="%d/%m/%Y", errors="coerce").fillna(pd.to_datetime(x.Date, format="%d/%m/%y", errors="coerce"))
        out = pd.DataFrame(dict(date=d.dt.strftime("%Y-%m-%d"), home=x.HomeTeam.replace(market.NAMES), away=x.AwayTeam.replace(market.NAMES)))
        for s in SIDES:
            out["o" + s] = pd.to_numeric(x.get("B365" + s), errors="coerce")
        close = None
        for cols in (("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA")):
            if all(c in x for c in cols):
                close = x[list(cols)].apply(pd.to_numeric, errors="coerce")
                break
        if close is None:
            continue
        inv = 1 / close.where(close > 1)
        out[["cH", "cD", "cA"]] = inv.div(inv.sum(axis=1), axis=0).to_numpy()
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def pick(df: pd.DataFrame, rules: dict = RULES) -> pd.DataFrame:
    """df: pH,pD,pA (우리 확률) + oH,oD,oA (배당). 규칙에 맞는 베팅만 돌려준다."""
    P = df[["pH", "pD", "pA"]].to_numpy()
    O = df[["oH", "oD", "oA"]].to_numpy()
    ev = np.where(O <= rules["odds_max"], P * O - 1, -np.inf)
    ev = np.nan_to_num(ev, nan=-np.inf)
    j = ev.argmax(1)
    best = ev[np.arange(len(df)), j]
    b = df.assign(side=np.array(SIDES)[j], ev=best, odds=O[np.arange(len(df)), j], p=P[np.arange(len(df)), j])
    return b[b.ev >= rules["ev_min"]].copy()


def settle(b: pd.DataFrame, stake: float = 1.0) -> pd.DataFrame:
    """b: side, odds + result(H/D/A) + cH,cD,cA → 수익(pnl), CLV."""
    b = b.copy()
    cp = b.apply(lambda r: r["c" + r.side], axis=1) if len(b) else pd.Series(dtype=float)
    b["clv"] = b.odds * cp - 1
    b["pnl"] = np.where(b.result == b.side, (b.odds - 1) * stake, -stake)
    return b


def summary(b: pd.DataFrame, n_boot: int = 3000, seed: int = 0) -> dict:
    if b.empty:
        return dict(n=0)
    rng = np.random.default_rng(seed)
    c, p = b.clv.dropna().to_numpy(), b.pnl.to_numpy()
    bc = [c[rng.integers(0, len(c), len(c))].mean() for _ in range(n_boot)]
    bp = [p[rng.integers(0, len(p), len(p))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(bc, [2.5, 97.5])
    v = "맞음" if lo > 0 else ("틀림" if hi < 0 else "불확실")
    return dict(n=len(b), win=float((b.result == b.side).mean()), avg_odds=float(b.odds.mean()),
                clv=float(c.mean()), clv_lo=float(lo), clv_hi=float(hi),
                roi=float(p.mean()), roi_lo=float(np.percentile(bp, 2.5)), roi_hi=float(np.percentile(bp, 97.5)), units=float(p.sum()), verdict=v)
