"""숫자 검증기의 정답 저장소: 한·일 핵심 인구 통계의 공식 값 (기준별).

항목: births 출생아 수, deaths 사망자 수, marriages 혼인 건수, divorces 이혼 건수,
      tfr 합계출산율, cmr 조혼인율(인구 천 명당), fm_age_F / fm_age_M 평균 초혼연령
기준(basis): '확정' 이 기본. 잠정치·속보치처럼 기준이 다른 값은 따로 넣어 '기준 다름'을 잡는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .official import estat, kosis

KR_ITEMS = {"11": "births", "12": "deaths", "30": "tfr", "41": "marriages", "42": "cmr", "51": "divorces"}
JP_ITEMS = {"出生数": "births", "死亡数": "deaths", "合計特殊出生率": "tfr", "婚姻件数": "marriages", "婚姻率": "cmr", "離婚件数": "divorces"}

# 공식 표에 아직 없지만 공식 발표로 알려진 값 (주장 DB에서 '보도 일치'로 채택한 것) → 기준을 명시해 따로 보관
EXTRA = [
    dict(country="JP", indicator="births", year=2025, value=671236, basis="개수(잠정)·일본인만", source="후생노동성 2025 개수 (보도 인용, 원문 접근 차단)"),
    dict(country="JP", indicator="marriages", year=2025, value=489119, basis="개수(잠정)", source="후생노동성 2025 개수 (보도 인용)"),
    dict(country="JP", indicator="tfr", year=2025, value=1.14, basis="개수(잠정)", source="후생노동성 2025 개수 (보도 인용)"),
    dict(country="KR", indicator="births", year=2025, value=254500, basis="잠정", source="통계청 2025 출생 잠정 (2026.2)"),
]


def build(start: int = 2000) -> list[dict]:
    rows = []
    k = kosis("DT_1B8000F", str(start), "2030", objL1=" ".join(KR_ITEMS))
    for r in k.itertuples():
        rows.append(dict(country="KR", indicator=KR_ITEMS[r.C1], year=int(r.PRD_DE), value=float(r.value), basis="확정",
                         source="KOSIS DT_1B8000F 인구동태건수 및 동태율 추이"))
    j = estat("0003411561")
    j = j[j.cat01.isin(JP_ITEMS) & (j.year >= start)]
    for r in j.itertuples():
        rows.append(dict(country="JP", indicator=JP_ITEMS[r.cat01], year=int(r.year), value=float(r.value), basis="확정·일본인만",
                         source="e-Stat 0003411561 人口動態総覧"))
    off = pd.read_csv(Path(__file__).parent.parent / "data_official" / "kr_jp_marriage_official.csv")
    fm = off[(off.indicator == "first_marriage_age") & (off.year >= start)]
    for r in fm.itertuples():
        rows.append(dict(country=r.country, indicator=f"fm_age_{r.sex}", year=int(r.year), value=float(r.value),
                         basis="확정", source=f"{'KOSIS' if r.country == 'KR' else 'e-Stat'} {r.table_id}"))
    rows += EXTRA
    return rows


def save(path: str | Path) -> int:
    rows = build()
    Path(path).write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    return len(rows)
