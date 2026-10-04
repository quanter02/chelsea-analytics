"""시군구별 '결혼 시장 성비': 미혼 남성이 미혼 여성의 몇 배인가, 그리고 왜 그런가.

미혼 성비 = 미혼 남성 / 미혼 여성 (같은 연령대)
          = 인구 성비 (남성 / 여성)  ×  미혼율 비 (남성 미혼율 / 여성 미혼율)
첫째 항은 '그 지역에 젊은 여성이 적어서', 둘째 항은 '남성이 결혼을 덜 해서'다.
로그로 바꾸면 두 항이 더해지므로, 지역 간 차이가 어느 쪽에서 오는지 나눠 볼 수 있다.

자료: KOSIS DT_1MR2060 (등록 기반 인구, 내국인, 2022~), DT_1B26006 (시군구/성/연령별 이동률).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .official import kosis

AGES = {"024": "20-24", "029": "25-29", "034": "30-34", "039": "35-39", "044": "40-44"}
CORE = ["25-29", "30-34", "35-39"]
MIG_AGES = {"120": "20-24", "130": "25-29", "150": "30-34"}
SIDO = {"11": "서울", "21": "부산", "22": "대구", "23": "인천", "24": "광주", "25": "대전", "26": "울산", "29": "세종", "31": "경기",
        "32": "강원", "33": "충북", "34": "충남", "35": "전북", "36": "전남", "37": "경북", "38": "경남", "39": "제주"}
METRO = {"11", "21", "22", "23", "24", "25", "26"}
# 이동률 표(행정표준코드) 시도 앞자리 → 인구 표(통계청 코드) 시도 앞자리
MIG_SIDO = {"11": "11", "26": "21", "27": "22", "28": "23", "29": "24", "30": "25", "31": "26", "36": "29", "41": "31",
            "51": "32", "42": "32", "43": "33", "44": "34", "52": "35", "45": "35", "46": "36", "47": "37", "48": "38", "50": "39"}


def analysis_regions(codes: set[str]) -> list[str]:
    """시군구 단위: 동부·읍부·면부 합계와, 시 아래 일반구(수원시 장안구 등)는 빼고 시 단위로 본다."""
    return sorted(c for c in codes if len(c) == 5 and c[2:] not in ("003", "004", "005")
                  and not (c[-1] != "0" and c[:4] + "0" in codes))


def fetch_unmarried(years=range(2022, 2026)) -> pd.DataFrame:
    """연도마다 한 번씩 (요청당 셀 수 제한 때문)."""
    parts = []
    for y in years:
        d = kosis("DT_1MR2060", str(y), str(y), itm="T5 T6 T9 T10", objL1="ALL", objL2=" ".join(AGES))
        parts.append(d)
    d = pd.concat(parts)
    d["sex"] = d.ITM_NM.str[:2].map({"남자": "M", "여자": "F"})
    d["kind"] = np.where(d.ITM_NM.str.contains("미혼"), "unmarried", "total")
    d["age_band"] = d.C2.map(AGES)
    return d.rename(columns={"C1": "code", "C1_NM": "name"})[["year", "code", "name", "sex", "kind", "age_band", "value"]]


def fetch_migration(years=range(2015, 2026)) -> pd.DataFrame:
    """순이동률 (인구 100명당, 그 해 전입 − 전출). 이 표는 행정표준코드를 써서 이름으로 맞춘다."""
    d = kosis("DT_1B26006", str(min(years)), str(max(years)), itm="T30", objL1="ALL", objL2="1 2", objL3=" ".join(MIG_AGES))
    d["sex"] = d.C2_NM.map({"남자": "M", "여자": "F"})
    d["age_band"] = d.C3.map(MIG_AGES)
    return d.rename(columns={"C1": "mcode", "C1_NM": "name"})[["year", "mcode", "name", "sex", "age_band", "value"]]


def region_table(u: pd.DataFrame, year: int, ages=CORE) -> pd.DataFrame:
    """지역별 미혼 성비와 분해."""
    codes = set(u.code)
    keep = analysis_regions(codes)
    names = u.drop_duplicates("code").set_index("code").name
    x = u[(u.year == year) & u.age_band.isin(ages) & u.code.isin(keep + ["00"])]
    w = x.pivot_table(index="code", columns=["sex", "kind"], values="value", aggfunc="sum")
    t = pd.DataFrame(index=w.index)
    t["미혼 남"] = w[("M", "unmarried")]; t["미혼 여"] = w[("F", "unmarried")]
    t["남"] = w[("M", "total")]; t["여"] = w[("F", "total")]
    t["unmarried_ratio"] = t["미혼 남"] / t["미혼 여"]
    t["pop_ratio"] = t["남"] / t["여"]
    t["mrate_ratio"] = (t["미혼 남"] / t["남"]) / (t["미혼 여"] / t["여"])
    t["f_unmarried_pct"] = 100 * t["미혼 여"] / t["여"]
    t["m_unmarried_pct"] = 100 * t["미혼 남"] / t["남"]
    t["excess_men"] = t["미혼 남"] - t["미혼 여"]
    t["sido"] = [c[:2] for c in t.index]
    t["sido_name"] = t.sido.map(SIDO)
    t["name"] = [names.get(c, c) for c in t.index]
    t["label"] = [f"{SIDO.get(c[:2], '')} {names.get(c, c)}" if c != "00" else "전국" for c in t.index]
    t["metro"] = t.sido.isin(METRO)
    t["kind"] = np.where(t.sido.isin(METRO), "광역시 구·군", np.where(t.name.str.endswith("군"), "도 지역 군", "도 지역 시"))
    return t.reset_index()


def decompose(t: pd.DataFrame) -> dict:
    """지역 간 log(미혼 성비) 분산 중 인구 성비 몫과 미혼율 비 몫 (공분산 반씩 배분)."""
    r = t[t.code != "00"]
    a, b = np.log(r.pop_ratio), np.log(r.mrate_ratio)
    v = np.var(a + b)
    return {"인구 성비 몫": (np.var(a) + np.cov(a, b, bias=True)[0, 1]) / v,
            "미혼율 비 몫": (np.var(b) + np.cov(a, b, bias=True)[0, 1]) / v}


def attach_migration(t: pd.DataFrame, mig: pd.DataFrame, years=range(2015, 2025)) -> pd.DataFrame:
    """지역별 20~34세 여성·남성 평균 순이동률 (연평균, 100명당)을 이름으로 붙인다."""
    m = mig[mig.year.isin(years)]
    # 이동률 표의 시군구 이름은 시도 안에서 유일하므로 (시도 앞 2자리 + 이름)으로 맞춘다
    m = m.assign(sido=m.mcode.str[:2].map(MIG_SIDO), nm=m.name.str.strip())
    g = m.groupby(["sido", "nm", "sex"]).value.mean().unstack()
    g.columns = [f"mig_{c}" for c in g.columns]
    out = t.merge(g, left_on=["sido", "name"], right_index=True, how="left")
    # 시 단위 지역(수원시 등)은 이동률 표에 일반구로만 있을 수 있음 → 이름이 '수원시'로 시작하는 구의 평균
    for i, r in out[out.mig_F.isna()].iterrows():
        sub = g[(g.index.get_level_values(0) == r.sido) & g.index.get_level_values(1).str.startswith(r["name"])]
        if not len(sub) and (out.sido == r.sido).sum() == 1:            # 시도 = 시군구 하나 (세종)
            sub = g[g.index.get_level_values(0) == r.sido]
        if len(sub):
            out.loc[i, ["mig_F", "mig_M"]] = sub.mean().values
    return out


# ── 지도 ──────────────────────────────────────────────────────────────────
def build_map(geo_path: str | Path, regions: pd.DataFrame, tol: float = 0.006) -> tuple[dict, list]:
    """2018년 시군구 경계(통계청)를 분석 단위(2025 코드)로 합친다. 이름이 바뀐 곳·옮겨진 곳은 이름으로 맞춘다."""
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union

    g = json.load(open(geo_path, encoding="utf-8"))
    rk = regions[regions.code != "00"][["code", "name", "sido"]]
    rename = {"남구": "미추홀구"}                                          # 인천 남구 → 미추홀구 (2018.7)
    groups, unmatched = {}, []
    for f in g["features"]:
        c, n = f["properties"]["code"], f["properties"]["name"]
        sido = c[:2]
        cand = rk[rk.sido == sido]
        hit = cand[cand.name == n]
        if hit.empty and sido == "23":
            hit = cand[cand.name == rename.get(n, n)]
        if hit.empty:
            hit = cand[[n.startswith(x) and x.endswith("시") for x in cand.name]]
        if hit.empty:                                                     # 시도가 바뀐 곳 (군위군: 경북 → 대구, 2023)
            hit = rk[rk.name == n]
        if len(hit) != 1:
            unmatched.append((c, n)); continue
        groups.setdefault(hit.code.iloc[0], []).append(shape(f["geometry"]))

    def rnd(co):
        return [rnd(x) for x in co] if isinstance(co[0], (list, tuple)) else [round(co[0], 3), round(co[1], 3)]

    feats = []
    for code, shapes in groups.items():
        geom = unary_union(shapes).simplify(tol, preserve_topology=True)
        m = mapping(geom)
        feats.append({"type": "Feature", "properties": {"code": code}, "geometry": {"type": m["type"], "coordinates": rnd(m["coordinates"])}})
    return {"type": "FeatureCollection", "features": feats}, unmatched
