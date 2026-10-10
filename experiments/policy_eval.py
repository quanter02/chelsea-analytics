"""정책 효과 사전 등록 평가: 농어촌 기본소득 시범사업(2026년 2월 말 첫 지급) 뒤 시범 군의 순이동이 예측보다 확실히 늘었나.

사전 등록 문서: preregistration_basic_income_migration.md. 이 파일과 문서는 평가 자료(전체 순이동)를 받기 전에 커밋합니다.
판정은 이미 정해진 범용 파이프라인 + 통일 규칙 6차 그대로 (기준값은 보정 기간 2010~2017에서 결정).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

import kosis_monitor as K

TREAT = ["경기 연천군", "강원 정선군", "충북 옥천군", "충남 청양군", "전북 장수군", "전북 순창군",
         "전남 곡성군", "전남 신안군", "경북 영양군", "경남 남해군"]           # 2026년 2월 말 첫 지급 10개 군
LATER = ["강원 화천군", "충북 보은군", "전북 진안군", "전북 무주군", "전남 구례군", "전남 보성군", "경북 청송군"]  # 2026.06 추가 선정, 8월 첫 지급 → 대조군에서 제외
METRO = ("부산", "대구", "인천", "울산")                                   # 광역시 소속 군은 대조군에서 제외


def groups(U, year):
    names = {u["name"] for (unit, y), u in U.items() if y == year}
    t = [n for n in TREAT if n in names]
    c = sorted(n for n in names if n.endswith("군") and n not in TREAT and n not in LATER and not n.startswith(METRO))
    return t, c


def status(out, year, k):
    """year년 1~k월 자료 기준: 지역별 경보 방향(유지 중인 것만)과 누적 이탈률."""
    tp, cb, U = out["topic"], out["calibration"], out["units"]
    rows = []
    for (unit, y), u in U.items():
        if y != year or u["size"] < tp.min_size: continue
        n = min(k, u["n"]); uu = {**u, "n": n}
        mo, s = K.alarm(uu, K._phi(cb, unit, y), cb.rule, cb.alpha, cb.m)
        S, Sc = u["e"][:n].sum(), u["scale_inc"][:n].sum()
        keep = s != 0 and np.sign(S) == s and abs(S / Sc) >= cb.m
        rows.append(dict(지역=u["name"], 월수=n, 경보월=mo, 방향={1: "증가", -1: "감소", 0: ""}[int(s) if keep else 0],
                         누적차이=round(float(S), 1), 이탈률=float(S / Sc)))
    return pd.DataFrame(rows)


def judge(st: pd.DataFrame, t: list, c: list) -> dict:
    T, C = st[st.지역.isin(t)], st[st.지역.isin(c)]
    pt, pc = (T.방향 == "증가").mean(), (C.방향 == "증가").mean()
    verdict = ("효과 확인" if pt >= 0.7 and pt - pc >= 0.3 else
               "부분 확인 (일부 군만)" if pt - pc >= 0.3 else "효과 확인 안 됨")
    return dict(시범군_수=len(T), 대조군_수=len(C), 시범군_증가경보=round(pt, 3), 대조군_증가경보=round(pc, 3), 차이=round(pt - pc, 3),
                시범군_이탈률_중앙값=round(float(T.이탈률.median()), 3), 대조군_이탈률_중앙값=round(float(C.이탈률.median()), 3), 판정=verdict)


def run(topic: K.Topic, year=2026, placebo_year=2025, placebo_k=9):
    out = K.run(topic, rule="v6", verbose=False)
    t, c = groups(out["units"], year)
    k = max(u["n"] for (unit, y), u in out["units"].items() if y == year)
    main = status(out, year, k)
    plac = status(out, placebo_year, placebo_k)
    return dict(out=out, treat=t, control=c, k=k, main=main, judge=judge(main, t, c),
                placebo=judge(plac, *groups(out["units"], placebo_year)), test=out["test"])
