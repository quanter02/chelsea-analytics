"""AI 예측 검증 (사전 등록: preregistration_ai_forecast.md).

문항: 시군구 × 2026년 9~12월 — "그달 순이동(전입−전출)이 플랜 A 예측보다 높은가?" (예측값은 지금 고정해 저장)
경쟁자: B0 동전(0.5), B1 통계 기준선(올해 1~8월 중 예측보다 높았던 달 비율, (k+1)/(n+2)), AI = B1 + 근거를 적은 지역별 조정.
예측은 9월 자료 공표(2026년 10월 말) 전에 커밋하고 SHA-256 지문을 남깁니다. 채점은 공표된 달부터.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

import kosis_monitor as K
import niche_candidates as N

HERE = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(HERE, "ai_forecast", "predictions_2026-09_12.csv")
MONTHS = ["202609", "202610", "202611", "202612"]
YEAR = 2026


def questions():
    """문항과 고정 예측값(플랜 A 월별 예측 순이동) + B1."""
    tp = N.CANDIDATES["E 전체 순이동"]
    tidy = K.fetch(tp, verbose=False)
    U = K.units(tp, tidy)
    t = tidy.copy(); t["y"] = t.ym.str[:4].astype(int); t["m"] = t.ym.str[4:].astype(int)
    rows = []
    years = sorted(t.y.unique())
    for (unit, y), u in U.items():
        if y != YEAR or u["size"] < tp.min_size or u["n"] < 6: continue
        g = t[t.unit == unit]
        r = {s: {(a.y, a.m): a.value for a in g[g.series == s].itertuples()} for s in ("in", "out")}
        mi, _ = K._plan_a_flow(r["in"], YEAR, years); mo, _ = K._plan_a_flow(r["out"], YEAR, years)
        if mi is None or mo is None: continue
        e = u["e"][:u["n"]]
        k, n = int((e > 0).sum()), len(e)
        b1 = (k + 1) / (n + 2)
        for ym in MONTHS:
            m = int(ym[4:])
            rows.append(dict(unit=unit, 지역=u["name"], ym=ym, 예측_순이동=round(float(mi[m - 1] - mo[m - 1]), 1),
                             올해_누적차이=round(float(e.sum()), 1), 올해_누적이탈률=round(float(e.sum() / u["scale_inc"][:n].sum()), 4),
                             B0=0.5, B1=round(b1, 4)))
    return pd.DataFrame(rows)


def apply_ai(Q: pd.DataFrame, adjust: dict) -> pd.DataFrame:
    """adjust: {지역: (델타, 근거)} 또는 {지역: {ym: 델타, '근거': ...}}. AI = clip(B1 + 델타, 0.02, 0.98)."""
    Q = Q.copy(); Q["AI"] = Q.B1; Q["AI_근거"] = ""
    for name, spec in adjust.items():
        idx = Q.지역 == name
        if not idx.any(): raise KeyError(name)
        d, why = spec
        Q.loc[idx, "AI"] = (Q.loc[idx, "B1"] + d).clip(0.02, 0.98)
        Q.loc[idx, "AI_근거"] = why
    return Q


def save(Q: pd.DataFrame) -> str:
    os.makedirs(os.path.dirname(PRED), exist_ok=True)
    Q.to_csv(PRED, index=False)
    h = hashlib.sha256(open(PRED, "rb").read()).hexdigest()
    open(PRED + ".sha256", "w").write(h + "\n")
    return h


def resolve(Q: pd.DataFrame) -> pd.DataFrame:
    """공표된 달의 실제 순이동으로 답(1/0)을 채움. 고정된 예측값과 비교.
    2026년 7월 행정구역 개편으로 옛 코드 자료가 끊긴 지역은, 새 코드 중 시도가 '전남광주'이고 시군구 이름이 같은 곳으로 매칭(미리 정한 규칙).
    매칭이 안 되면(예: 인천 중구·동구 → 제물포구·영종구) 그 문항은 채점에서 뺌."""
    tidy = K.fetch(N.CANDIDATES["E 전체 순이동"], verbose=False)
    w = tidy.pivot_table(index=["unit", "ym"], columns="series", values="value")
    names = tidy.drop_duplicates("unit").set_index("unit").name.astype(str)
    new_by_name = {}
    for u, n in names.items():
        if len(u) == 5 and u.startswith("12"):
            new_by_name[n.split()[-1].replace(" ", "")] = u
    def actual(u, name, ym):
        if (u, ym) in w.index:
            return w.loc[(u, ym), "in"] - w.loc[(u, ym), "out"]
        if name.split()[0] in ("전남", "광주"):
            nu = new_by_name.get(name.split()[-1])
            if nu and (nu, ym) in w.index:
                return w.loc[(nu, ym), "in"] - w.loc[(nu, ym), "out"]
        return np.nan
    out = Q.copy()
    out["실제_순이동"] = [actual(u, n, ym) for u, n, ym in zip(Q.unit, Q.지역, Q.ym)]
    out["답"] = np.where(np.isnan(out.실제_순이동), np.nan, (out.실제_순이동 > out.예측_순이동).astype(float))
    return out


def score(R: pd.DataFrame, n_boot=4000, seed=0) -> dict:
    R = R.dropna(subset=["답"])
    br = {c: float(((R[c] - R.답) ** 2).mean()) for c in ("B0", "B1", "AI")}
    S = R[R.AI != R.B1]                              # AI가 손댄 문항
    out = dict(문항=len(R), 브라이어=br, 조정문항=len(S))
    if len(S):
        S = S.assign(d=(S.B1 - S.답) ** 2 - (S.AI - S.답) ** 2)          # +면 AI가 나음
        d = S.d.values
        g = S.groupby("지역").d.agg(["sum", "count"])                    # 같은 지역 4개월은 묶어서 다시 뽑음
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            b = g.iloc[rng.integers(0, len(g), len(g))]
            boots.append(b["sum"].sum() / b["count"].sum())
        lo, hi = np.quantile(boots, [0.05, 0.95])
        out.update(조정_브라이어_B1=float(((S.B1 - S.답) ** 2).mean()), 조정_브라이어_AI=float(((S.AI - S.답) ** 2).mean()),
                   개선=float(d.mean()), 개선_90구간=(float(lo), float(hi)),
                   판정="AI 판단 가치 확인" if lo > 0 else "방향만 맞음 (불확실)" if d.mean() > 0 else "AI 판단 가치 확인 안 됨")
    # llm-reliability 프로파일: 정답 / 확신 오답 / 기권 (AI)
    p = R.AI; hit = ((p > 0.5) == (R.답 == 1))
    out["AI_프로파일"] = dict(기권=float(((p >= 0.4) & (p <= 0.6)).mean()),
                           정답=float((hit & ((p < 0.4) | (p > 0.6))).mean()),
                           확신_오답=float((~hit & ((p <= 0.2) | (p >= 0.8))).mean()))
    return out
