"""은행 ROE 검증·관문 비교·금리 인하기 ROE.

입력: out/fin_fundamentals.csv(fin_compare.py --dart), out/bank_docs.csv(dart_docs.py)
출력: out/bank_roe_check.csv  — 연도별 ROE_지배평균 vs 공시 ROE
      out/bank_gate.csv       — ROE 기준 8% vs 9% 관문 비교
      out/bank_rate_cut.csv   — 금리 인하기(2019~2021) ROE 변화

    python invest/bank_roe.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "out"
BANKS = ["KB금융", "신한지주", "하나금융", "우리금융", "기업은행", "BNK금융", "iM금융", "JB금융", "카카오뱅크"]
YEARS = list(range(2019, 2026))
CUT = (2019, 2021)  # 기준금리 1.75%(2019.7 인하 전) → 0.50%(2020.5), 2021.8부터 인상
TOL = 0.5  # 공시 ROE와 %p 차이 허용폭


def panel() -> pd.DataFrame:
    """은행·연도별 지배주주 순이익·자본(원). API 값 우선, 없으면 원문(백만원) 값."""
    api = pd.read_csv(OUT / "fin_fundamentals.csv")
    api = api[api.종목.isin(BANKS)].set_index(["종목", "연도"])
    doc = pd.read_csv(OUT / "bank_docs.csv")
    # 같은 연도가 여러 보고서에 있으면 가장 최근 보고서(정정·재작성 반영) 값을 쓴다; ROE는 그 해 보고서(당기) 우선
    doc["당기"] = doc.보고서 == doc.연도
    fin = (doc[doc.항목 != "roe"].sort_values("보고서").groupby(["종목", "연도", "항목"]).값.last().unstack("항목") * 1e6)
    roe = doc[doc.항목 == "roe"].sort_values(["당기", "보고서"]).groupby(["종목", "연도"]).값.last().rename("공시ROE")
    idx = pd.MultiIndex.from_product([BANKS, range(2018, 2026)], names=["종목", "연도"])
    p = pd.DataFrame(index=idx)
    for f in ("ni_ctrl", "equity_ctrl"):
        a = api[f] if f in api else pd.Series(dtype=float)
        p[f] = a.reindex(idx)
        p[f"src_{f}"] = np.where(p[f].notna(), "api", None)
        d = fin[f].reindex(idx) if f in fin else pd.Series(np.nan, index=idx)
        fill = p[f].isna() & d.notna()
        p.loc[fill, f] = d[fill]
        p.loc[fill, f"src_{f}"] = "원문"
    for f in ("ni", "equity"):
        p[f] = api[f].reindex(idx) if f in api else np.nan
    p["공시ROE"] = roe.reindex(idx) / 100
    p = p.reset_index()
    g = p.groupby("종목")
    p["ROE_기말"] = p.ni / p.equity  # 이전 방식: 전체 순이익 ÷ 기말 자본총계
    p["ROE_지배평균"] = p.ni_ctrl / ((p.equity_ctrl + g.equity_ctrl.shift(1)) / 2)
    p["차이%p"] = (p.ROE_지배평균 - p.공시ROE) * 100
    p["일치"] = np.where(p.공시ROE.isna() | p.ROE_지배평균.isna(), "", np.where(p["차이%p"].abs() <= TOL, "O", "X"))
    return p[p.연도.isin(YEARS)].reset_index(drop=True)


def gate(p: pd.DataFrame, cuts=(0.08, 0.09)) -> pd.DataFrame:
    """ROE 관문: 7년 평균 ROE_지배평균 ≥ 기준, 그리고 기준 이상인 해가 7년 중 몇 해인지.

    자본 관문: 지배주주 자본이 2018년 말 대비 매년 늘었는지(감소한 해 수). 연체 관문은 DART 재무 API에 없어 따로 표시만 한다.
    """
    rows = []
    for b, d in p.groupby("종목", sort=False):
        r = d.ROE_지배평균
        eq = d.set_index("연도").equity_ctrl
        row = {"종목": b, "산출연수": int(r.notna().sum()), "평균ROE": r.mean(), "최저ROE": r.min(),
               "최저연도": int(d.연도[r.idxmin()]) if r.notna().any() else None,
               "공시ROE평균": d.공시ROE.mean(), "자본감소연수": int((eq.diff() < 0).sum())}
        for c in cuts:
            k = f"{int(c * 100)}%"
            row[f"{k}이상연수"] = int((r >= c).sum())
            row[f"ROE관문_{k}"] = "통과" if r.notna().sum() == len(YEARS) and r.mean() >= c else ("자료부족" if r.notna().sum() < len(YEARS) else "탈락")
        rows.append(row)
    return pd.DataFrame(rows)


def rate_cut(p: pd.DataFrame) -> pd.DataFrame:
    w = p.pivot(index="종목", columns="연도", values="ROE_지배평균").reindex(BANKS)
    o = pd.DataFrame(index=w.index)
    for y in range(CUT[0], CUT[1] + 2):
        o[f"{y}"] = w.get(y)
    o["2019→2020(%p)"] = (w[2020] - w[2019]) * 100
    o["2020→2021(%p)"] = (w[2021] - w[2020]) * 100
    o["인하기평균"] = w[[2019, 2020, 2021]].mean(axis=1)
    o["이후평균(2022~2025)"] = w[[2022, 2023, 2024, 2025]].mean(axis=1)
    return o.reset_index()


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    p = panel()
    p.to_csv(OUT / "bank_roe_check.csv", index=False)
    print("== ROE_지배평균 vs 공시 ROE (%) ==")
    v = p.assign(계산=p.ROE_지배평균 * 100, 공시=p.공시ROE * 100)
    print(v.pivot(index="종목", columns="연도", values="계산").reindex(BANKS).round(2).to_string())
    print(v.pivot(index="종목", columns="연도", values="공시").reindex(BANKS).round(2).to_string())
    print(p.pivot(index="종목", columns="연도", values="차이%p").reindex(BANKS).round(2).to_string())
    gt = gate(p); gt.to_csv(OUT / "bank_gate.csv", index=False)
    print("== 관문 ==\n", gt.round(4).to_string(index=False))
    rc = rate_cut(p); rc.to_csv(OUT / "bank_rate_cut.csv", index=False)
    print("== 금리 인하기 ==\n", rc.round(4).to_string(index=False))
