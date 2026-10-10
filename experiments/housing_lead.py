"""인구 이동 경보가 이후 주택 매매 거래량 변화를 앞서는가 (사전 등록: preregistration_housing_lead.md).

경보: 뉴스레터 주력 주제(E 전체 순이동, 통일 규칙 6차) — 이미 정해진 기준 그대로.
결과: 한국부동산원 행정구역별 주택매매거래현황(KOSIS orgId 408, DT_408_2006_S0057, 동(호)수).
이 파일과 문서는 거래 자료를 받기 전에 커밋합니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import kosis_monitor as K
import niche_candidates as N

TRADES = K.Topic(
    name="시군구 주택 매매 거래량", table="DT_408_2006_S0057", org="408", kind="flow",
    series={"y": K.Series("13103114441T1", {"objL1": "ALL"})}, unit_cols=("C1",),
    start="200601", end="202607", cache="data_kosis/pipeline_housing_trades.csv")
YEARS = range(2018, 2025)          # 경보 연도 (결과 창이 2025년 8월까지 공표된 해까지)
ISSUE_MONTH = 6                    # 1~6월 자료로 낸 경보 (6월 자료는 7월 말경 공표)
SECONDARY_MONTH = 3
PASS_GAP = 0.05                    # 5%p
PASS_YEARS = 5                     # 7개 연도 중


def trade_names(tidy: pd.DataFrame) -> dict:
    """부동산원 코드(…A.0002 = 서울, …A.00020001 = 종로구) → '서울 종로구'."""
    names = tidy.drop_duplicates("unit").set_index("unit").name.astype(str).str.strip()
    out = {}
    for u, n in names.items():
        head, _, tail = u.partition("A.")
        if len(tail) == 8:
            sd = names.get(f"{head}A.{tail[:4]}")
            if sd is not None:
                out[u] = f"{K.SIDO_SHORT.get(sd, sd)} {n.replace(' ', '')}"
    return out


def growth(tidy: pd.DataFrame, names: dict, year: int) -> pd.Series:
    """결과: 거래량(Y년 9월~Y+1년 8월) / 거래량(Y-1년 9월~Y년 8월) − 1. 경보가 공표된 뒤의 12개월."""
    t = tidy[tidy.unit.isin(names)].copy()
    t["name"] = t.unit.map(names)
    p = t.pivot_table(index="name", columns="ym", values="value", aggfunc="sum")
    nxt = [f"{year}{m:02d}" for m in range(9, 13)] + [f"{year + 1}{m:02d}" for m in range(1, 9)]
    prv = [f"{year - 1}{m:02d}" for m in range(9, 13)] + [f"{year}{m:02d}" for m in range(1, 9)]
    if not set(nxt + prv) <= set(p.columns): return pd.Series(dtype=float)
    a, b = p[nxt].sum(1, min_count=12), p[prv].sum(1, min_count=12)
    return (a / b - 1).where(b >= 50)            # 거래가 너무 적은 곳(연 50건 미만)은 제외


def alerts(out, year: int, k: int) -> pd.DataFrame:
    """k월까지 자료로 '유지' 중인 경보 방향(+1/−1, 없으면 0). 평가 대상 전 지역."""
    tp, cb, U = out["topic"], out["calibration"], out["units"]
    rows = []
    for (unit, y), u in U.items():
        if y != year or u["size"] < tp.min_size or u["n"] < k: continue
        uu = {**u, "n": k}
        mo, s = K.alarm(uu, K._phi(cb, unit, y), cb.rule, cb.alpha, cb.m)
        S, Sc = u["e"][:k].sum(), u["scale_inc"][:k].sum()
        keep = s != 0 and np.sign(S) == s and abs(S / Sc) >= cb.m
        rows.append(dict(name=u["name"], dir=int(s) if keep else 0))
    return pd.DataFrame(rows)


def run(k: int = ISSUE_MONTH):
    out = K.run(N.CANDIDATES["E 전체 순이동"], rule="v6", verbose=False)
    tidy = K.fetch(TRADES, verbose=False)
    names = trade_names(tidy)
    per_year, pooled = [], []
    for y in YEARS:
        A = alerts(out, y, k)
        g = growth(tidy, names, y)
        d = A.assign(g=A.name.map(g)).dropna(subset=["g"])
        pooled.append(d.assign(year=y))
        med = d.groupby("dir").g.median()
        per_year.append(dict(연도=y, 증가경보=int((d.dir == 1).sum()), 경보없음=int((d.dir == 0).sum()), 감소경보=int((d.dir == -1).sum()),
                             증가_중앙값=med.get(1, np.nan), 없음_중앙값=med.get(0, np.nan), 감소_중앙값=med.get(-1, np.nan)))
    Y = pd.DataFrame(per_year)
    Y["D+"] = Y.증가_중앙값 - Y.없음_중앙값
    Y["D-"] = Y.없음_중앙값 - Y.감소_중앙값
    P = pd.concat(pooled, ignore_index=True)
    m = P.groupby("dir").g.median()
    dp, dm = m.get(1, np.nan) - m.get(0, np.nan), m.get(0, np.nan) - m.get(-1, np.nan)
    ok_years = int(((Y["D+"] > 0) & (Y["D-"] > 0)).sum())
    verdict = "선행 신호 확인" if dp >= PASS_GAP and dm >= PASS_GAP and ok_years >= PASS_YEARS else \
              "약한 신호 (방향만 맞음)" if dp > 0 and dm > 0 else "선행 신호 확인 안 됨"
    matched = len(set(names.values()) & {u["name"] for u in out["units"].values()})
    return dict(years=Y, pooled=P, D_plus=dp, D_minus=dm, ok_years=ok_years, verdict=verdict, matched_units=matched, k=k)
