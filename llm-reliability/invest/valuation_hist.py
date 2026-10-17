"""알파벳·테이크투 장기 밸류에이션 이력 (SEC EDGAR 10-K + 야후 주가). 개인 계좌 정보 없음.

    python invest/valuation_hist.py    # data.sec.gov, query1.finance.yahoo.com 허용 필요 → invest/out/valuation_hist.md

알파벳: '영업이익 기준 PER' = 시가총액 ÷ (영업이익 × (1−17%)).  일회성 투자이익이 섞인 순이익 대신 영업이익을 쓴다.
  희석주식수는 10-K에 2022년부터만 있어서, 이전 해는 순이익 ÷ 희석 EPS로 구하고 2022년 20:1 분할 전 값은 ×20.
테이크투: 적자 해가 많아 PER 대신 P/S(시가총액 ÷ 매출). 회계연도는 3월 말에 끝난다(EDGAR frame CY2025 = FY2026).
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "out"
UA = {"User-Agent": "chelsea-analytics research contact@example.com"}
TAX = 0.17


def edgar(cik: str) -> dict:
    u = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60).read())["facts"]["us-gaap"]


def annual(g: dict, concept: str) -> dict[int, float]:
    if concept not in g:
        return {}
    out = {}
    for x in list(g[concept]["units"].values())[0]:
        if x.get("form") == "10-K" and x.get("frame") and len(x["frame"]) == 6:
            out[int(x["frame"][2:])] = x["val"]
    return out


def close(sym: str) -> pd.Series:
    u = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1=1356998400&period2=1900000000&interval=1d"
    d = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read())["chart"]["result"][0]
    return pd.Series(d["indicators"]["quote"][0]["close"], index=pd.to_datetime(d["timestamp"], unit="s").normalize()).dropna()


def googl() -> pd.DataFrame:
    g = edgar("0001652044")
    op, ni, eps, sh = annual(g, "OperatingIncomeLoss"), annual(g, "NetIncomeLoss"), annual(g, "EarningsPerShareDiluted"), annual(g, "WeightedAverageNumberOfDilutedSharesOutstanding")
    shares = {y: v / 1e9 for y, v in sh.items()}
    for y in eps:
        if y not in shares and y in ni:
            shares[y] = ni[y] / eps[y] / 1e9 * (20 if eps[y] > 15 else 1)
    p = close("GOOGL")
    rows = []
    for y in sorted(op):
        s = shares.get(y) or shares.get(min(shares, key=lambda k: abs(k - y)))   # 2014~2015: 가장 가까운 해 주식수
        price = p[:f"{y}-12-31"].iloc[-1]
        rows.append(dict(year=str(y), price=price, op=op[y] / 1e9, shares=s, pe=price * s / (op[y] / 1e9 * (1 - TAX))))
    return pd.DataFrame(rows)


def ttwo() -> pd.DataFrame:
    g = edgar("0000946581")
    rev, op, sh = annual(g, "RevenueFromContractWithCustomerExcludingAssessedTax"), annual(g, "OperatingIncomeLoss"), annual(g, "WeightedAverageNumberOfDilutedSharesOutstanding")
    p = close("TTWO")
    rows = []
    for y in sorted(rev):
        s = sh.get(y, sh[max(sh)]) / 1e9
        price = p[:f"{y + 1}-03-31"].iloc[-1]
        rows.append(dict(year=f"FY{y + 1}", price=price, rev=rev[y] / 1e9, margin=op.get(y, np.nan) / rev[y], ps=price * s / (rev[y] / 1e9)))
    return pd.DataFrame(rows)


def main():
    G, T = googl(), ttwo()
    pe_now = 349.2 * 12.31 / ((31.23 + 35.93 + 39.70 + 40.77) * (1 - TAX))   # 최근 4분기 영업이익(야후), 2026-10-15 주가
    ps_now = 209.4 * 0.185 / (1.77 + 1.70 + 1.68 + 1.53)
    pct = lambda s, v: float((s < v).mean())
    summary = {"googl_pe_median": float(G.pe.median()), "googl_pe_now": pe_now, "ttwo_ps_median": float(T.ps.median()), "ttwo_ps_now": ps_now,
               "ttwo_margin_peak": float(T.margin.max())}
    OUT.mkdir(exist_ok=True)
    (OUT / "valuation_hist.json").write_text(json.dumps(summary))
    (OUT / "valuation_hist.md").write_text(f"""# 장기 밸류에이션 이력 (EDGAR 10-K)

## 알파벳: 영업이익 기준 PER (연말)
{G[['year', 'price', 'op', 'pe']].round(1).to_markdown(index=False)}

- 지금 {pe_now:.1f}배 · {G.year.iloc[0]}~{G.year.iloc[-1]} 중앙값 {G.pe.median():.1f}배 · 지금보다 낮았던 해 {pct(G.pe, pe_now):.0%}

## 테이크투: P/S (회계연도 말, 3월)
{T[['year', 'price', 'rev', 'margin', 'ps']].round(2).to_markdown(index=False)}

- 지금 {ps_now:.1f}배 · 중앙값 {T.ps.median():.1f}배 · 지금보다 낮았던 해 {pct(T.ps, ps_now):.0%} · 영업이익률 최고 {T.margin.max():.0%}
""", encoding="utf-8")
    print((OUT / "valuation_hist.md").read_text())


if __name__ == "__main__":
    main()
