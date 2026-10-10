"""인구 이동 경보가 다음 해 출점(신규 개인사업자)을 앞서는가 (사전 등록: preregistration_shop_lead.md).

경보: 뉴스레터 주력 주제(E 전체 순이동, 통일 규칙 6차) — 이미 정해진 기준 그대로. 경보 계산은 housing_lead.alerts와 같음.
결과: 국세청 신규사업자 현황(시·군·구) DT_133001N_9822, 개인사업자(B02). 보조: 가동사업자 현황(시·군·구) DT_133N_A9811 총계.
이 파일과 문서는 국세청 자료를 받기 전에 커밋합니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import housing_lead as H
import kosis_monitor as K
import niche_candidates as N

NEW = K.Topic(name="시군구 신규 개인사업자", table="DT_133001N_9822", org="133", kind="flow",
              series={"y": K.Series("T001", {"objL1": "ALL", "objL2": "B02"})}, unit_cols=("C1",),
              start="2016", end="2025", cache="data_kosis/pipeline_new_business.csv")
ACTIVE = K.Topic(name="시군구 가동사업자", table="DT_133N_A9811", org="133", kind="flow",
                 series={"y": K.Series("T01", {"objL1": "ALL", "objL2": "16133T2008_0245"})}, unit_cols=("C1",),
                 start="2008", end="2025", cache="data_kosis/pipeline_active_business.csv")
YEARS = range(2018, 2025)          # 경보 연도 Y → 결과 Y+1 (2019~2025)
PASS_GAP = 0.03                    # 3%p (출점 증가율은 주택 거래보다 변동이 작아 5%p 대신 3%p, 자료를 보기 전에 정함)
PASS_YEARS = 5                     # 7개 연도 중
MIN_BASE = 100                     # Y년 신규 개인사업자 100명 미만 지역 제외


def fetch_annual(tp: K.Topic) -> pd.DataFrame:
    """연간 표 받기 (kosis_monitor.fetch는 월 단위라 연 단위 요청을 따로 함). 저장본이 있으면 그것을 씀."""
    import os
    if tp.cache and os.path.exists(tp.cache):
        return pd.read_csv(tp.cache, dtype={"unit": str, "ym": str})
    key = K._key()
    s = tp.series["y"]
    d = K._get(K.KOSIS_DATA, dict(method="getList", apiKey=key, orgId=tp.org, tblId=tp.table, itmId=s.itm, format="json",
                                  jsonVD="Y", prdSe="Y", startPrdDe=tp.start, endPrdDe=tp.end, **s.objs))
    if not isinstance(d, list):
        raise RuntimeError(str(d)[:200])
    df = pd.DataFrame(d)
    out = pd.DataFrame(dict(unit=df.C1, name=df.C1_NM.str.strip(), ym=df.PRD_DE.astype(str),
                            value=pd.to_numeric(df.DT, errors="coerce")))
    out.to_csv(tp.cache, index=False)
    return out


def region_names(df: pd.DataFrame) -> dict:
    """부모 코드(끝 2자리를 뗀 코드)가 시도인 지역 → '시도 시군구'."""
    names = df.drop_duplicates("unit").set_index("unit").name.astype(str).str.strip()
    out = {}
    for u, n in names.items():
        p = names.get(u[:-2])
        if p is not None and u[:-2] in names.index and len(u) > 2:
            sd = K.SIDO_SHORT.get(p, p)
            if sd in K.SIDO_SHORT.values() or sd in ("서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"):
                out[u] = f"{sd} {n.replace(' ', '')}"
    return out


def growth(df: pd.DataFrame, names: dict, y: int, min_base=MIN_BASE) -> pd.Series:
    p = df[df.unit.isin(names)].assign(name=lambda x: x.unit.map(names)).pivot_table(index="name", columns="ym", values="value", aggfunc="sum")
    a, b = str(y + 1), str(y)
    if a not in p.columns or b not in p.columns: return pd.Series(dtype=float)
    return (p[a] / p[b] - 1).where(p[b] >= min_base)


def run(tp: K.Topic = NEW, k: int = H.ISSUE_MONTH, min_base=MIN_BASE):
    out = K.run(N.CANDIDATES["E 전체 순이동"], rule="v6", verbose=False)
    df = fetch_annual(tp)
    names = region_names(df)
    per_year, pooled = [], []
    for y in YEARS:
        A = H.alerts(out, y, k)
        g = growth(df, names, y, min_base)
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
    ok = int(((Y["D+"] > 0) & (Y["D-"] > 0)).sum())
    verdict = "선행 신호 확인" if dp >= PASS_GAP and dm >= PASS_GAP and ok >= PASS_YEARS else \
              "약한 신호 (방향만 맞음)" if dp > 0 and dm > 0 else "선행 신호 확인 안 됨"
    matched = len(set(names.values()) & {u["name"] for u in out["units"].values()})
    return dict(years=Y, pooled=P, D_plus=dp, D_minus=dm, ok_years=ok, verdict=verdict, matched_units=matched)
