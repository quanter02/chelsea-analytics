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


# ───────────────────────── 2차 (2026-10-10 사전 등록): 휘슬(발표) 기준 예측 ─────────────────────────
EVAL2 = ("202609", "202612")      # 평가 기간: 사전 등록 시점에 아직 공표되지 않은 달만
WHISTLE = {"추가 선정 7개 군": ("202606", LATER), "첫 10개 군": ("202510", TREAT)}


def _wide(tidy, names):
    d = tidy[tidy.unit.isin(names)].copy()
    w = d.pivot_table(index=["unit", "ym"], columns="series", values="value").reset_index()
    return {u: g.set_index("ym") for u, g in w.groupby("unit")}


def whistle_unit(g, whistle: str, months: list[str]):
    """발표 전 12개월 합계 × 발표 전 5개 달력 연도 평균 월별 비중 → 평가 달의 전입·전출 예측. 자료가 모자라면 None."""
    yms = list(g.index)
    w = yms.index(whistle) if whistle in yms else None
    if w is None or w < 12: return None
    base = g.iloc[w - 12:w][["in", "out"]].sum()
    wy = int(whistle[:4])
    sh = {}
    for s in ("in", "out"):
        M = []
        for y in range(wy - 5, wy):
            row = [g[s].get(f"{y}{m:02d}", np.nan) for m in range(1, 13)]
            if np.isnan(row).any() or sum(row) <= 0: return None
            M.append(np.array(row) / sum(row))
        sh[s] = np.mean(M, axis=0)
    have = [m for m in months if m in g.index]
    if not have: return None
    fi = np.array([base["in"] * sh["in"][int(m[4:]) - 1] for m in have])
    fo = np.array([base["out"] * sh["out"][int(m[4:]) - 1] for m in have])
    act = np.array([g.loc[m, "in"] - g.loc[m, "out"] for m in have])
    return dict(e=act - (fi - fo), var_raw=fi + fo, scale_inc=fi, n=len(have), size=float(base["in"]), have=have)


def run2(topic: K.Topic, months=EVAL2, year=2026):
    """2차 판정. months 안에서 공표된 달까지만 씀(중간 판정 가능). 판정 규칙·α·m·지역별 φ는 후보 E 보정 그대로."""
    out = K.run(topic, rule="v6", verbose=False)
    cb, U = out["calibration"], out["units"]
    tidy = K.fetch(topic, verbose=False)
    names = {u: d["name"] for (u, y), d in U.items() if y == year}
    unit_of = {v: k for k, v in names.items()}
    W = _wide(tidy, list(names))
    span = [f"{y}{m:02d}" for y in range(int(months[0][:4]), int(months[1][:4]) + 1) for m in range(1, 13)
            if months[0] <= f"{y}{m:02d}" <= months[1]]
    t_all, c = groups(U, year)
    res = {}
    for label, (whistle, treat) in WHISTLE.items():
        rows = []
        for n in [*treat, *c]:
            uid = unit_of.get(n)
            if uid is None or uid not in W: continue
            u = whistle_unit(W[uid], whistle, span)
            if u is None: continue
            mo, s = K.alarm(u, K._phi(cb, uid, year), cb.rule, cb.alpha, cb.m)
            S, Sc = u["e"].sum(), u["scale_inc"].sum()
            keep = s != 0 and np.sign(S) == s and abs(S / Sc) >= cb.m
            rows.append(dict(지역=n, 그룹="시범" if n in treat else "대조", 월=f"{u['have'][0]}~{u['have'][-1]}",
                             방향={1: "증가", -1: "감소", 0: ""}[int(s) if keep else 0], 누적차이=round(float(S), 1), 이탈률=float(S / Sc)))
        st = pd.DataFrame(rows)
        j = judge(st, [n for n in treat if n in set(st.지역)], [n for n in c if n in set(st.지역)]) if len(st) else {}
        if len(st):
            j["시범군_감소경보"] = round(float((st[st.그룹 == "시범"].방향 == "감소").mean()), 3)
            j["판정"] = j["판정"].replace("효과 확인", "효과 지속 확인").replace("효과 지속 확인 안 됨", "효과 지속 확인 안 됨")
        res[label] = dict(status=st, judge=j, whistle=whistle)
    return res
