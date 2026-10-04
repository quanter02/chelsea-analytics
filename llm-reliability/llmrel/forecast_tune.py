"""연령별 혼인율 예측을 점진 개선 실험대에 올리기 위한 규칙화 버전.

규칙(파라미터)
  drift_k  최근 몇 년의 평균 변화를 추세로 볼지
  lam      공동 학습 회귀의 규제 강도
  wp       앙상블에서 공동 학습 회귀의 비중 (나머지는 추세·감쇠 추세 반씩)
  s        추세를 얼마나 믿을지: 예측 = 마지막 값 + s × (앙상블 − 마지막 값)

검증: 원점 2008~2014 (목표 연도 ≤ 2017), 시험: 원점 2015~2021 (목표 ≤ 2024). 손실 = |로그 오차| (1~3년 뒤).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import forecast as F

GRID = {"drift_k": [2, 3, 5, 8], "lam": [0.1, 1.0, 10.0, 100.0], "wp": [0.0, 0.2, 1 / 3, 0.5, 0.7], "s": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]}
START = {"drift_k": 5, "lam": 1.0, "wp": 1 / 3, "s": 1.0}
ORIGINS = {"val": range(2008, 2015), "test": range(2015, 2022)}


def make_spec(d: pd.DataFrame):
    from .tuning import Spec
    L = np.log(F.series_table(d))
    L = L[[c for c in L.columns if "|marriage_rate|" in c]]

    @lru_cache(maxsize=None)
    def pooled(origin, h, lam):
        return F._pooled_fit(L, origin, h, lam)

    @lru_cache(maxsize=None)
    def base(origin, h, drift_k):
        """원점·거리별 계열마다 (마지막 값, 추세, 감쇠 추세, 특징, 정답)."""
        out = []
        for key in L.columns:
            col = L[key].loc[:origin].dropna()
            if len(col) < 8 or col.index[-1] != origin or origin + h not in L.index or np.isnan(L.at[origin + h, key]):
                continue
            y = col.to_numpy()
            out.append((key, y[-1], F._drift(y, h, drift_k), F._damped(y, h), F._features(L, key, origin), L.at[origin + h, key]))
        return tuple(out)

    def evaluate(params, split):
        errs = []
        for o in ORIGINS[split]:
            for h in (1, 2, 3):
                w = pooled(o, h, params["lam"])
                for key, last, dr, dm, f, truth in base(o, h, params["drift_k"]):
                    pp = last + float(f @ w) if (w is not None and f is not None and not np.isnan(f).any()) else dr
                    ens = (1 - params["wp"]) * (dr + dm) / 2 + params["wp"] * pp
                    pred = last + params["s"] * (ens - last)
                    errs.append(abs(truth - pred))
        return np.array(errs)

    return Spec(name="연령별 혼인율 예측", grid=GRID, start=START, evaluate=evaluate, metric="|로그 오차|")
