"""틈새시장(지자체 인구정책 담당자용 무료 뉴스레터) 주제 후보와 사전 등록 순위 규칙.

사전 등록 문서: preregistration_niche_topic.md. 이 파일과 문서는 후보 자료(값)를 받기 전에 커밋합니다.
모든 후보는 같은 범용 파이프라인(kosis_monitor)과 같은 통일 규칙 6차로 채점합니다.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

import kosis_monitor as K

AGES_YOUTH = "120 130 150 160"          # 20~39세
AGES_CHILD = "020 050"                  # 0~9세 (아이가 있는 가구의 이동을 대신 보는 값)
AGES_OLD = "280 310 330 340"            # 65세 이상
MIG = dict(table="DT_1B26001", kind="net", unit_cols=("C1",), unit_filter=lambda u, n: len(u) == 5,
           start="200501", end="202608", cal=range(2010, 2018), test=range(2018, 2026), live_year=2026,
           thresholds=("quantile", 0.8, 0.5), min_size=300,
           size_cuts=(0, 2000, 10000, np.inf), size_labels=("300~2천", "2천~1만", "1만 이상"))


def _mig(name, sex, ages, cache):
    s = {"objL1": "ALL", "objL2": sex, "objL3": ages}
    return K.Topic(name=name, series={"in": K.Series("T10", s), "out": K.Series("T20", s)}, cache=cache, **MIG)


CANDIDATES = {
    "A 청년 순이동(전입 300+)": dataclasses.replace(K.TOPICS["청년 순이동"], name="시군구 청년(20~39세) 순이동, 전입 300명 이상",
                                              min_size=300, size_cuts=MIG["size_cuts"], size_labels=MIG["size_labels"]),
    "B 영유아(0~9세) 순이동": _mig("시군구 영유아(0~9세) 순이동", "0", AGES_CHILD, "data_kosis/pipeline_child_migration.csv"),
    "C 청년 여성 순이동": _mig("시군구 청년 여성(20~39세) 순이동", "2", AGES_YOUTH, "data_kosis/pipeline_youngwomen_migration.csv"),
    "D 고령(65+) 순이동": _mig("시군구 고령(65세 이상) 순이동", "0", AGES_OLD, "data_kosis/pipeline_old_migration.csv"),
    "E 전체 순이동": _mig("시군구 전체 순이동", "0", "000", "data_kosis/pipeline_total_migration.csv"),
    "F 주민등록인구": K.Topic(
        name="시군구 주민등록인구", table="DT_1B040A3", kind="stock",
        series={"y": K.Series("T20", {"objL1": "ALL"})}, unit_cols=("C1",), unit_filter=lambda u, n: len(u) == 5,
        start="201101", end="202609", cal=range(2012, 2018), test=range(2018, 2026), live_year=2026,
        min_size=0, thresholds=("quantile", 0.8, 0.5), size_cuts=(0, 30000, 100000, np.inf),
        size_labels=("3만 미만", "3만~10만", "10만 이상"), stock_floor=50, cache="data_kosis/pipeline_population.csv"),
}
BASELINE = "청년 순이동"   # 현재 월간 경보 주제 (전입 2000+), 참고 기준


def coverage(out, year=2025) -> int:
    """채점 마지막 해에 평가 대상(최소 규모 이상)인 시군구 수."""
    tp = out["topic"]
    return sum(1 for (u, y), d in out["units"].items() if y == year and d["size"] >= tp.min_size)


def score(out) -> dict:
    t, g = out["test"], out["groups"]
    passed = t["잘못된경보율"] <= 0.10 and bool((g["잘못된경보율"] <= 0.10).all())
    cov = coverage(out)
    return dict(통과=passed, 감지율=t["감지율"], 잘못된경보=t["잘못된경보율"], 구간최대_잘못된경보=float(g["잘못된경보율"].max()),
                상반기내=t["상반기내_감지율"], 커버_시군구=cov, 점수=(t["감지율"] * cov) if passed else 0.0)


def rank(rows: pd.DataFrame, baseline_score: float) -> tuple[str, str]:
    """사전 등록 결정 규칙. rows: 후보별 score() 결과 (index = 후보 이름)."""
    ok = rows[rows.통과]
    if ok.empty:
        return BASELINE, "통과한 후보 없음 → 현재 주제 유지"
    best = ok.점수.idxmax()
    if ok.loc[best, "점수"] >= 1.10 * baseline_score:
        return best, f"{best} 채택 (점수 {ok.loc[best, '점수']:.1f} ≥ 기준 {baseline_score:.1f} × 1.1)"
    return BASELINE, f"최고 후보 {best} 점수 {ok.loc[best, '점수']:.1f} < 기준 {baseline_score:.1f} × 1.1 → 현재 주제 유지"
