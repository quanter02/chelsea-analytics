"""시즌 중반 예측(나우캐스트): 올해 1~k월 시도별 혼인 건수가 작년 같은 달보다 얼마나 늘었는지를
소속 시군구의 작년 연간 건수에 곱해 올해 연간 건수를 예측한다. 파라미터 없음.

자료: KOSIS DT_1B8000G (월별 인구동향, 시도), 약 2개월 뒤 공표 → 1~6월 신호는 8월 말에 쓸 수 있음.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
MONTHLY = ROOT / "data_monthly" / "kr_marriages_monthly_sido.csv"


def fetch(start: str = "200501", end: str = "203012") -> pd.DataFrame:
    from .official import kosis
    out = []
    for y0 in range(int(start[:4]), int(end[:4]) + 1, 5):
        d = kosis("DT_1B8000G", f"{y0}01", f"{y0 + 4}12", prd="M", objL1="ALL", objL2="20")   # 20 = 혼인건수
        out.append(d[["C1", "C1_NM", "PRD_DE", "value"]])
    d = pd.concat(out).rename(columns={"C1": "code", "C1_NM": "name", "PRD_DE": "month"})
    d.to_csv(MONTHLY, index=False)
    return d


def ytd_growth(k: int = 6, path=MONTHLY) -> pd.DataFrame:
    """연도 × 시도: log(올해 1~k월 합 / 작년 1~k월 합). 그해 k월까지 자료가 없으면 NaN."""
    m = pd.read_csv(path, dtype={"code": str})
    m["y"], m["mo"] = m.month // 100, m.month % 100
    full = m.groupby(["y", "code"]).mo.max().unstack() >= k
    t = np.log(m[m.mo <= k].pivot_table(index="y", columns="code", values="value", aggfunc="sum")).where(full)
    return t.diff()


def predict(sgg_log: pd.DataFrame, year: int, k: int = 6) -> pd.Series:
    """올해 연간 로그 건수 예측 = 작년 값 + 소속 시도의 1~k월 증가율."""
    g = ytd_growth(k).loc[year]
    return sgg_log.loc[year - 1] + sgg_log.columns.map(lambda c: g.get(c[:2], np.nan)).to_numpy()


def backtest(sgg_log: pd.DataFrame, years, k: int = 6) -> pd.DataFrame:
    rows = []
    for y in years:
        pred, truth, last = predict(sgg_log, y, k).to_numpy(), sgg_log.loc[y].to_numpy(), sgg_log.loc[y - 1].to_numpy()
        rows.append(pd.DataFrame({"year": y, "code": sgg_log.columns, "model": np.abs(truth - pred), "base": np.abs(truth - last)}))
    return pd.concat(rows, ignore_index=True)
