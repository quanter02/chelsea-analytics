"""한국(KOSIS)·일본(e-Stat) 공식 통계에서 연령대별 미혼율·혼인율·평균 초혼연령을 내려받아 한 표로 정리.

결과 열: country, indicator, sex, age_band, year, value, unit, source, table_id, basis
  indicator = unmarried_pct    연령대 인구 중 미혼 비율(%)
              marriage_rate    해당 연령 인구 천 명당 혼인 건수
              first_marriage_age 평균 초혼연령(세)
basis 열에 집계 기준(전수/표본/등록, 국적 범위 등)을 적어 두어, 기준이 다른 값을 한 선으로 잇지 않게 한다.
"""
from __future__ import annotations

import json
import time

import pandas as pd

from .collect import _curl_json, clean_key

AGE_BANDS = ["~19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54"]
SEX = {"남자": "M", "여자": "F", "남편": "M", "아내": "F", "男": "M", "女": "F", "夫": "M", "妻": "F"}


def _band(label: str) -> str | None:
    """'25~29세', '25 - 29세', '25～29歳', '19歳以下', '15-19세' → '25-29' / '~19'. 범위 밖이면 None."""
    s = label.replace(" ", "").replace("～", "~").replace("세", "").replace("歳", "").replace("-", "~")
    if s in ("19以下", "20미만", "15~19"):
        return "~19"
    parts = s.split("~")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        b = f"{parts[0]}-{parts[1]}"
        return b if b in AGE_BANDS else None
    return None


# ── KOSIS ──────────────────────────────────────────────────────────────
def kosis(tbl: str, start: str, end: str, itm: str = "ALL", prd: str = "Y", **objs) -> pd.DataFrame:
    key = clean_key("KOSIS_API_KEY")
    if not key:
        raise RuntimeError("환경 변수 KOSIS_API_KEY 필요")
    p = dict(method="getList", apiKey=key, orgId="101", tblId=tbl, itmId=itm, format="json", jsonVD="Y",
             prdSe=prd, startPrdDe=start, endPrdDe=end, **objs)
    for attempt in range(4):                                   # 이 환경에서 간헐적으로 연결이 끊김
        code, body = _curl_json("https://kosis.kr/openapi/Param/statisticsParameterData.do", p, timeout=90)
        if code and body.strip():
            break
        time.sleep(2 ** attempt)
    d = json.loads(body)
    if isinstance(d, dict):
        raise RuntimeError(f"KOSIS {tbl}: {d.get('errMsg', d)}")
    df = pd.DataFrame(d)
    df["value"] = pd.to_numeric(df.DT, errors="coerce")
    df["year"] = df.PRD_DE.astype(int)
    return df


def kr_unmarried() -> pd.DataFrame:
    """미혼율. 2005·2010 인구총조사(전수), 2015·2020 인구총조사(표본), 2022~ 등록 기반 인구(18세 이상)."""
    out = []
    # 2005, 2010: 항목 = 성별×혼인상태, 분류 = 지역, 연령
    for tbl, yr, basis in (("DT_1IN0508", "2005", "인구총조사 전수, 내국인 15세+"),
                           ("DT_1IN1005", "2010", "인구총조사 전수, 내국인 15세+")):
        df = kosis(tbl, yr, yr, itm="T20 T21 T30 T31", prd="F", objL1="00", objL2="ALL")   # 인구총조사 = 5년 주기(F)
        df["sex"] = df.ITM_NM.str[:2].map(SEX)
        df["kind"] = df.ITM_NM.str.contains("미혼").map({True: "u", False: "t"})
        out.append(_ratio(df, "C2_NM", tbl, basis))
    # 2015, 2020: 분류 = 지역, 성별, 연령, 혼인상태 / 항목 T10 = 내국인 15세+ 계
    df = kosis("DT_1PM1504", "2015", "2015", itm="T10", prd="F", objL1="00", objL2="1 2", objL3="ALL", objL4="0 1")
    df["sex"] = df.C2_NM.map(SEX)
    df["kind"] = df.C4_NM.map({"합계": "t", "미혼": "u"})
    out.append(_ratio(df, "C3_NM", "DT_1PM1504", "인구총조사 표본(20%), 내국인 15세+"))
    # 2020: 같은 구성의 표(DT_1PM2003)는 API가 요청을 거부해, 출생지 유형 표의 '계'를 쓴다. 15~24세가 한 구간이라 25세 이상만 사용.
    df = kosis("DT_1PB2004", "2020", "2020", itm="T00", prd="F", objL1="00", objL2="1 2", objL3="0 1", objL4="ALL")
    df["sex"] = df.C2_NM.map(SEX)
    df["kind"] = df.C3_NM.map({"합계": "t", "미혼": "u"})
    out.append(_ratio(df, "C4_NM", "DT_1PB2004", "인구총조사 표본(20%), 15세+ (출생지 유형 표)"))
    # 2022~: 등록 기반. 18세 이상만 집계되므로 20세 미만 구간은 쓰지 않는다.
    df = kosis("DT_1MR2060", "2022", "2030", itm="T5 T6 T9 T10", objL1="00", objL2="ALL")
    df["sex"] = df.ITM_NM.str[:2].map(SEX)
    df["kind"] = df.ITM_NM.str.contains("미혼").map({True: "u", False: "t"})
    r = _ratio(df, "C2_NM", "DT_1MR2060", "등록 기반 인구, 내국인 18세+")
    out.append(r[r.age_band != "~19"])
    return pd.concat(out, ignore_index=True).assign(country="KR", source="KOSIS")


def _ratio(df: pd.DataFrame, age_col: str, tbl: str, basis: str) -> pd.DataFrame:
    df = df.assign(age_band=df[age_col].map(_band)).dropna(subset=["age_band", "sex", "kind"])
    w = df.pivot_table(index=["year", "sex", "age_band"], columns="kind", values="value", aggfunc="sum").reset_index()
    w["value"] = (100 * w.u / w.t).round(2)
    return w[["year", "sex", "age_band", "value"]].assign(indicator="unmarried_pct", unit="%", table_id=tbl, basis=basis)


def kr_marriage_rate() -> pd.DataFrame:
    df = kosis("DT_1B83A15", "1990", "2030", objL1="00", objL2="ALL")
    df["sex"] = df.ITM_NM.str[:2].map(SEX)
    df["age_band"] = df.C2_NM.map(_band)
    df = df.dropna(subset=["age_band", "sex", "value"])
    return df[["year", "sex", "age_band", "value"]].assign(
        indicator="marriage_rate", unit="‰", table_id="DT_1B83A15", basis="인구동향조사, 신고 연도·혼인 당시 연령, 해당 연령 인구 천 명당",
        country="KR", source="KOSIS")


def kr_first_marriage_age() -> pd.DataFrame:
    df = kosis("DT_1B83A05", "1990", "2030", objL1="00")
    df["sex"] = df.ITM_NM.str[:2].map(SEX)
    return df[["year", "sex", "value"]].assign(
        age_band="all", indicator="first_marriage_age", unit="세", table_id="DT_1B83A05", basis="인구동향조사, 신고 연도 기준",
        country="KR", source="KOSIS")


# ── e-Stat ─────────────────────────────────────────────────────────────
def estat(sid: str, **filters) -> pd.DataFrame:
    app = clean_key("ESTAT_APP_ID")
    if not app:
        raise RuntimeError("환경 변수 ESTAT_APP_ID 필요")
    code, body = _curl_json("https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData",
                            dict(appId=app, statsDataId=sid, metaGetFlg="Y", cntGetFlg="N", **filters), timeout=90)
    d = json.loads(body)["GET_STATS_DATA"]
    if d["RESULT"]["STATUS"] != 0:
        raise RuntimeError(f"e-Stat {sid}: {d['RESULT']['ERROR_MSG']}")
    sd = d["STATISTICAL_DATA"]
    names = {}
    for o in sd["CLASS_INF"]["CLASS_OBJ"]:
        cl = o["CLASS"] if isinstance(o["CLASS"], list) else [o["CLASS"]]
        names[o["@id"]] = {c["@code"]: c["@name"] for c in cl}
    vals = sd["DATA_INF"]["VALUE"]
    rows = []
    for v in vals if isinstance(vals, list) else [vals]:
        r = {k[1:]: names.get(k[1:], {}).get(c, c) for k, c in v.items() if k.startswith("@")}
        r["value"] = pd.to_numeric(v["$"], errors="coerce")
        rows.append(r)
    df = pd.DataFrame(rows)
    df["year"] = df.time.str[:4].astype(int)
    return df


def jp_unmarried() -> pd.DataFrame:
    """국세조사 1920~2020. 배우관계 '불상'을 뺀 인구를 분모로 쓴다 (국립사회보장·인구문제연구소 방식)."""
    df = estat("0003410382", cdTab="1060")
    df = df[~df.time.str.contains("不詳補完")]
    df["sex"] = df.cat01.str[0].map(SEX)
    df["age_band"] = df.cat02.map(_band)
    df = df.dropna(subset=["age_band", "sex", "value"])
    w = df.pivot_table(index=["year", "sex", "age_band"], columns="cat03", values="value", aggfunc="sum").reset_index()
    known = w[["未婚", "有配偶", "死別", "離別"]].sum(axis=1)
    w["value"] = (100 * w["未婚"] / known).round(2)
    return w[["year", "sex", "age_band", "value"]].assign(
        indicator="unmarried_pct", unit="%", table_id="0003410382", basis="국세조사 전수, 일본 거주 전체(외국인 포함), 배우관계 불상 제외",
        country="JP", source="e-Stat")


def jp_marriage_rate() -> pd.DataFrame:
    """인구동태통계 확정수. 초혼율+재혼율 = 전체 혼인율 (분모가 같은 연령 인구이므로 더할 수 있다)."""
    df = estat("0003413965")
    df["sex"] = df.cat02.map(SEX)
    df["age_band"] = df.cat01.map(_band)
    df = df.dropna(subset=["age_band", "sex", "value"])
    df = df.groupby(["year", "sex", "age_band"], as_index=False).value.sum()
    return df.assign(indicator="marriage_rate", unit="‰", table_id="0003413965",
                     basis="인구동태 확정수, 그 해 동거 시작·신고분, 동거 시작 당시 연령, 해당 연령 인구 천 명당",
                     country="JP", source="e-Stat")


def jp_first_marriage_age() -> pd.DataFrame:
    df = estat("0003411845", cdArea="00000")
    df["sex"] = df.cat01.map(SEX)
    return df[["year", "sex", "value"]].dropna().assign(
        age_band="all", indicator="first_marriage_age", unit="세", table_id="0003411845", basis="인구동태 확정수, 신고 연도 기준",
        country="JP", source="e-Stat")


COLLECTORS = [kr_unmarried, kr_marriage_rate, kr_first_marriage_age, jp_unmarried, jp_marriage_rate, jp_first_marriage_age]
COLS = ["country", "indicator", "sex", "age_band", "year", "value", "unit", "source", "table_id", "basis"]


def collect_all() -> tuple[pd.DataFrame, list[dict]]:
    """모든 표를 받아 하나로 합친다. 실패한 표는 건너뛰고 log에 남긴다."""
    parts, log = [], []
    for f in COLLECTORS:
        try:
            df = f()
            parts.append(df[COLS])
            log.append(dict(collector=f.__name__, ok=True, rows=len(df), years=f"{df.year.min()}–{df.year.max()}"))
        except Exception as e:                                  # 한 표가 실패해도 나머지는 계속
            log.append(dict(collector=f.__name__, ok=False, rows=0, years="", error=str(e)[:120]))
    data = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)
    return data.sort_values(["indicator", "country", "sex", "age_band", "year"]).reset_index(drop=True), log
