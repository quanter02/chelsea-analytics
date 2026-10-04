"""시군구별 연간 혼인 건수 예측 (1~2년 뒤) + 점진 개선 실험대용 규칙.

예측(로그) = 수준 + s × h × 추세
  수준  = (1 − β) × 마지막 해 + β × 최근 3년 평균            (작은 지역의 들쭉날쭉 완화)
  추세  = (1 − α) × 자기 최근 k년 평균 변화 + α × 소속 시도의 같은 추세   (작은 지역은 시도에 기대기)

cap     한 해 추세 상한 (로그, 99 = 끔)
robust  1이면 추세(자기·시도)를 최근 k년 변화의 중앙값으로

검증: 원점 2013~2016 / 시험: 원점 2017~2023 (목표 ≤ 2025). 손실 = |로그 오차|, 시군구마다 같은 무게.
최신: 원점 2024 → 2025 (1년 뒤, 튜닝에 안 쓴 원점).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

GRID = {"k": [2, 3, 5, 8], "alpha": [0.0, 0.25, 0.5, 0.75, 1.0], "s": [0.0, 0.25, 0.5, 0.75, 1.0], "beta": [0.0, 0.25, 0.5, 0.75],
        "cap": [99.0, 0.15, 0.10, 0.07, 0.05], "robust": [0, 1]}
START = {"k": 5, "alpha": 0.0, "s": 1.0, "beta": 0.0, "cap": 99.0, "robust": 0}
FINAL = {**START, "k": 2, "alpha": 1.0}                        # 2026-10-04 최종 규칙으로 채택된 것
ORIGINS = {"val": range(2013, 2017), "test": range(2017, 2024), "live": range(2024, 2025)}   # 추세 기간 8년까지 쓰려면 원점 ≥ 2013


def load(path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = path or Path(__file__).parent.parent / "data_regional" / "kr_marriages_by_region.csv"
    d = pd.read_csv(path, dtype={"code": str})
    w = d.pivot_table(index="year", columns="code", values="value")
    sido = w[[c for c in w.columns if len(c) == 2 and c != "00"]]
    sgg = w[[c for c in w.columns if len(c) == 5 and (w[c] >= 30).all() and w[c].notna().all()]]   # 21년 모두 있고 연 30건 이상
    return np.log(sgg), np.log(sido)


def make_spec(sgg: pd.DataFrame, sido: pd.DataFrame):
    from .tuning import Spec
    parent = {c: c[:2] for c in sgg.columns}

    @lru_cache(maxsize=None)
    def trends(origin, k, robust=0):
        own = sgg.loc[origin - k:origin].diff().iloc[1:].median() if robust else (sgg.loc[origin] - sgg.loc[origin - k]) / k
        par = sido.loc[origin - k:origin].diff().iloc[1:].median() if robust else (sido.loc[origin] - sido.loc[origin - k]) / k
        return own, own.index.map(lambda c: par.get(parent[c], np.nan)).to_numpy()

    def evaluate(p, split):
        errs = []
        for o in ORIGINS[split]:
            own, par = trends(o, p["k"], p.get("robust", 0))
            par = np.where(np.isnan(par), own.to_numpy(), par)
            tr = np.clip((1 - p["alpha"]) * own.to_numpy() + p["alpha"] * par, -p.get("cap", 99.0), p.get("cap", 99.0))
            lvl = (1 - p["beta"]) * sgg.loc[o].to_numpy() + p["beta"] * sgg.loc[o - 2:o].mean().to_numpy()
            for h in ((1,) if split == "live" else (1, 2)):
                if o + h in sgg.index:
                    errs.append(np.abs(sgg.loc[o + h].to_numpy() - (lvl + p["s"] * h * tr)))
        return np.concatenate(errs)

    return Spec(name="시군구 혼인 건수 예측", grid=GRID, start=START, evaluate=evaluate, metric="|로그 오차|")
