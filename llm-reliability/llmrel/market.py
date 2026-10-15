"""시장(베팅 배당) 확률과 우리 EPL 모델 비교.

자료: football-data.co.uk 시즌별 E0.csv (data_epl/odds/E0_<yy><yy>.csv).
시장 확률 = 마감 직전 배당(피나클 PSC → 없으면 평균 AvgC → 없으면 경기 전 평균 Avg)을 1/배당 으로 바꾸고,
북메이커 마진(세 확률 합이 1을 넘는 만큼)을 비율대로 빼서 합이 1이 되게 맞춘다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import epl

ODDS = Path(__file__).parent.parent / "data_epl" / "odds"
NAMES = {"Brighton": "Brighton & Hove Albion", "Coventry": "Coventry City", "Hull": "Hull City", "Ipswich": "Ipswich Town",
         "Leeds": "Leeds United", "Leicester": "Leicester City", "Luton": "Luton Town", "Man City": "Manchester City",
         "Man United": "Manchester United", "Newcastle": "Newcastle United", "Norwich": "Norwich City",
         "Nott'm Forest": "Nottingham Forest", "Tottenham": "Tottenham Hotspur", "West Ham": "West Ham United",
         "Wolves": "Wolverhampton Wanderers", "Sheffield United": "Sheffield United",
         "Birmingham": "Birmingham City", "Blackburn": "Blackburn Rovers", "Bolton": "Bolton Wanderers", "Cardiff": "Cardiff City",
         "Huddersfield": "Huddersfield Town", "QPR": "Queens Park Rangers", "Stoke": "Stoke City", "Swansea": "Swansea City",
         "West Brom": "West Bromwich Albion", "Wigan": "Wigan Athletic"}
COLS = [("PSCH", "PSCD", "PSCA", "피나클 마감"), ("AvgCH", "AvgCD", "AvgCA", "평균 마감"), ("AvgH", "AvgD", "AvgA", "평균 경기 전"),
        ("BbAvH", "BbAvD", "BbAvA", "BetBrain 평균"), ("B365H", "B365D", "B365A", "Bet365")]   # 예전 시즌은 아래쪽 열만 있음


def load_odds() -> pd.DataFrame:
    parts = []
    for f in sorted(ODDS.glob("E0_*.csv")):
        x = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip").dropna(subset=["HomeTeam"])
        d = pd.to_datetime(x.Date, format="%d/%m/%Y", errors="coerce")                # 예전 시즌은 두 자리 연도
        d = d.fillna(pd.to_datetime(x.Date, format="%d/%m/%y", errors="coerce"))
        out = pd.DataFrame(dict(date=d.dt.strftime("%Y-%m-%d"), home=x.HomeTeam.replace(NAMES), away=x.AwayTeam.replace(NAMES)))
        out[["mH", "mD", "mA"]] = np.nan
        out["msrc"] = None
        for h, dr, a, name in COLS:
            if h not in x:
                continue
            o = x[[h, dr, a]].apply(pd.to_numeric, errors="coerce")
            ok = out.mH.isna() & o.notna().all(axis=1) & (o > 1).all(axis=1)
            inv = 1 / o[ok]
            out.loc[ok, ["mH", "mD", "mA"]] = (inv.div(inv.sum(axis=1), axis=0)).to_numpy()
            out.loc[ok, "msrc"] = name
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def rps_rows(P: np.ndarray, result: pd.Series) -> np.ndarray:
    y = np.zeros_like(P)
    y[np.arange(len(P)), result.map({"H": 0, "D": 1, "A": 2}).to_numpy()] = 1
    return ((np.cumsum(P, 1)[:, :2] - np.cumsum(y, 1)[:, :2]) ** 2).sum(1) / 2


def compare(params: dict | None = None, n_boot: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """구간별: 같은 경기에서 우리 모델 vs 시장 RPS, 차이의 95% 범위, 그리고 둘을 섞었을 때 (섞는 비율은 검증 구간에서만 고름)."""
    df = epl.load()
    res = epl.run(df, **(params or epl.XG_MODE))
    m = res[res.played].merge(load_odds(), on=["date", "home", "away"], how="inner").dropna(subset=["mH"])
    rng = np.random.default_rng(seed)
    P_us, P_mk = m[["pH", "pD", "pA"]].to_numpy(), m[["mH", "mD", "mA"]].to_numpy()
    m["rps_us"], m["rps_mk"] = rps_rows(P_us, m.result), rps_rows(P_mk, m.result)
    val = m.season.isin(epl.SPLITS["val"])
    grid = np.round(np.arange(0, 1.01, 0.1), 1)                 # 우리 모델 비중
    vloss = [rps_rows(b * P_us[val] + (1 - b) * P_mk[val], m.result[val]).mean() for b in grid]
    best = float(grid[int(np.argmin(vloss))])
    m["rps_mix"] = rps_rows(best * P_us + (1 - best) * P_mk, m.result)
    rows = []
    for split in ["val", "test", "live"]:
        s = m[m.season.isin(epl.SPLITS[split])]
        if s.empty:
            continue
        d = (s.rps_mk - s.rps_us).to_numpy()                      # 양수면 우리가 나음
        dm = (s.rps_mk - s.rps_mix).to_numpy()
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
        bm = [dm[rng.integers(0, len(dm), len(dm))].mean() for _ in range(n_boot)]
        rows.append(dict(구간=split, 경기=len(s), 우리=s.rps_us.mean(), 시장=s.rps_mk.mean(), 차이=d.mean(),
                         lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), 섞기=s.rps_mix.mean(),
                         섞기차이=dm.mean(), mlo=np.percentile(bm, 2.5), mhi=np.percentile(bm, 97.5)))
    out = pd.DataFrame(rows)
    out.attrs.update(best_weight=best, val_curve=dict(zip(grid.tolist(), np.round(vloss, 5).tolist())),
                     sources=m.msrc.value_counts().to_dict(), matched=len(m))
    return out, m
