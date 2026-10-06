"""시군구마다 '어느 예측이 더 나은가'를 과거로만 고르고, 그 뒤 해로 채점한다.

후보 (모두 그해 8월 말에 알 수 있는 정보만 씀, 로그 건수 예측)
  naive   작년 값 그대로
  monthly 작년 값 + 소속 시도 1~6월 증가율 (nowcast)
  half    작년 값 + 0.5 × 소속 시도 1~6월 증가율 (신호를 절반만 믿기)
  trend   작년 값 + 소속 시도 최근 2년 평균 변화 (연초 모델의 현재 규칙)

선택: 고르는 구간(2008~2018) 평균 |로그 오차|가 가장 작은 후보. 시험: 2019~2025.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import nowcast as NC
from . import regional_tune as RT

CANDS = ["naive", "monthly", "half", "trend"]
LABEL = {"naive": "작년 값 그대로", "monthly": "1~6월 신호", "half": "1~6월 신호 절반", "trend": "시도 2년 추세"}
PICK, TEST = range(2008, 2019), range(2019, 2026)


def predictions(years) -> pd.DataFrame:
    sgg, sido = RT.load()
    g = NC.ytd_growth(6)
    rows = []
    for y in years:
        last = sgg.loc[y - 1]
        par = lambda s: sgg.columns.map(lambda c: s.get(c[:2], np.nan)).to_numpy()
        gy = par(g.loc[y])
        tr = par((sido.loc[y - 1] - sido.loc[y - 3]) / 2)
        truth = sgg.loc[y] if y in sgg.index else pd.Series(np.nan, index=sgg.columns)
        preds = {"naive": last.to_numpy(), "monthly": last.to_numpy() + gy, "half": last.to_numpy() + 0.5 * gy, "trend": last.to_numpy() + tr}
        for k, v in preds.items():
            rows.append(pd.DataFrame({"year": y, "code": sgg.columns, "cand": k, "pred": v, "err": np.abs(truth.to_numpy() - v)}))
    return pd.concat(rows, ignore_index=True)


def select(p: pd.DataFrame) -> pd.DataFrame:
    """시군구별 선택 결과와 시험 성적."""
    e = p.pivot_table(index=["code", "year"], columns="cand", values="err").reset_index()
    pick = e[e.year.isin(PICK)].groupby("code")[CANDS].mean().idxmin(axis=1).rename("chosen")
    t = e[e.year.isin(TEST)].merge(pick, on="code")
    t["chosen_err"] = t.apply(lambda r: r[r.chosen], axis=1)
    out = t.groupby("code").agg(chosen=("chosen", "first"), test_err=("chosen_err", "mean"), naive_err=("naive", "mean"),
                                monthly_err=("monthly", "mean"),
                                wins=("chosen_err", lambda s: int((s < t.loc[s.index, "naive"] - 1e-12).sum())), n=("year", "size"))
    out["gain_pct"] = (1 - np.expm1(out.test_err) / np.expm1(out.naive_err)) * 100
    return out.reset_index()


def final_choice(p: pd.DataFrame) -> pd.Series:
    """2026년 예측에 쓸 후보: 2008~2025 전체로 다시 고름."""
    e = p.pivot_table(index=["code", "year"], columns="cand", values="err")
    return e.groupby("code")[CANDS].mean().idxmin(axis=1)
