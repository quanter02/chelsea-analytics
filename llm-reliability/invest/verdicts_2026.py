"""종목 밸류에이션 판정(역산 방식)을 장부에 등록하고, 1년 뒤 채점한다. 개인 계좌 정보 없음.

    python invest/verdicts_2026.py            # 등록 (한 번만, 이미 있으면 건너뜀)
    python invest/verdicts_2026.py --score    # 2027-10-16 이후: 1년 수익률로 채점 → invest/out/verdicts_2026_score.md

판정 방법(2026-10-17): 오늘 시가총액으로 사서 연 9%를 벌려면 5년 뒤 세후 영업이익이 얼마여야 하는지 역산
(5년 뒤 성숙 PER 가정: 알파벳 25~30.7, 테이크투 20~25, 하이닉스 10~12, 삼성 12~14, 팔란티어 30~40, 아이온큐 30~40)
하고 과거 실적과 비교. 은행(신한)은 P/B에서 시장이 깔고 있는 장기 ROE를 역산.

채점 규칙(실제 결과를 보기 전에 등록): 각 종목의 원화 기준 1년 총수익률 − 같은 기간 ACWI(전세계 주식, 원화) 수익률.
  저평가·적정 판정 → 초과수익 ≥ 0 이면 맞음,  고평가·비쌈 판정 → 초과수익 < 0 이면 맞음.
  종합: 판정 순서(점수)와 실제 초과수익 순서의 스피어만 상관. 7종목·1년이라 우연이 크다 → 5년 뒤 한 번 더 채점.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llmrel import ledger  # noqa: E402

OUT = Path(__file__).parent / "out"
DATE, SCORE_FROM = "2026-10-16", "2027-10-16"
# 점수: 높을수록 싸다(초과수익 기대). 판정과 근거 숫자는 대화의 역산 결과 그대로.
VERDICTS = {
    "055550.KS": ("신한지주", "약간 저평가", 2, "시장이 깐 장기 ROE 7.8~8.7% vs 실제 9.1~9.5%"),
    "GOOGL": ("알파벳", "적정", 1, "필요 영업이익 성장 연 11~16% vs 지난 10년 연 21%"),
    "TTWO": ("테이크투", "적정~다소 비쌈", 0, "5년 뒤 매출 113억~148억 달러 + 이익률 19~25% 필요 (지금 67억, 최고 19%)"),
    "000660.KS": ("SK하이닉스", "사이클 기준 비쌈", -1, "5년 뒤 영업이익 140조~168조 필요 vs 10년 평균 12.8조, 2025 47.2조"),
    "005930.KS": ("삼성전자", "사이클 기준 비쌈", -1, "5년 뒤 영업이익 190조~221조 필요 vs 10년 평균 38.9조, 2018 58.9조"),
    "PLTR": ("팔란티어", "고평가", -2, "매출 연 +48~52% 5년 + 이익률 45~50% 필요"),
    "IONQ": ("아이온큐", "고평가(복권형)", -2, "매출 연 +50~63% + 흑자 전환 필요"),
}
BENCH = "ACWI"


def series(sym: str, a: str, b: str, field: str = "adjclose") -> pd.Series:
    p1, p2 = int(pd.Timestamp(a).timestamp()), int(pd.Timestamp(b).timestamp())
    u = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d&events=div"
    d = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read())["chart"]["result"][0]
    v = d["indicators"]["adjclose"][0]["adjclose"] if field == "adjclose" else d["indicators"]["quote"][0]["close"]
    return pd.Series(v, index=pd.to_datetime(d["timestamp"], unit="s").normalize()).dropna()


def krw(sym: str, a: str, b: str) -> pd.Series:
    s = series(sym, a, b)
    if sym.endswith(".KS"):
        return s
    fx = series("KRW=X", a, b, "close")
    return (s * fx.reindex(s.index).ffill()).dropna()


def register():
    if any(x["kind"] == "valuation_verdict_2026" for x in ledger.read()):
        print("이미 등록됨")
        return
    a, b = "2026-10-01", "2026-10-17"
    ent = [dict(kind="scoring_rule", key="valuation_verdict_2026", model="invest/verdicts_2026",
                payload={"metric": "원화 1년 총수익률 − ACWI(원화)", "correct": "저평가·적정 → 초과≥0, 비쌈·고평가 → 초과<0",
                         "summary": "판정 점수와 실제 초과수익의 스피어만 상관", "from": DATE, "to": SCORE_FROM, "recheck": "2031-10-16"},
                evidence="실제 결과를 보기 전에 등록")]
    for sym, (name, verdict, score, why) in VERDICTS.items():
        ent.append(dict(kind="valuation_verdict_2026", key=sym, model="역산 밸류에이션 (요구수익 9%, 5년)",
                        payload={"name": name, "verdict": verdict, "score": score, "why": why,
                                 "price": round(float(series(sym, a, b, "close")[:DATE].iloc[-1]), 2)},
                        evidence=f"{DATE} 종가 기준, 채점 {SCORE_FROM}"))
    n = ledger.append(ent)
    ok, total, _ = ledger.verify()
    print(f"{n}줄 등록 (전체 {total}줄, 체인 {'정상' if ok else '어긋남'})")
    for e in ent[1:]:
        print(" ", e["payload"]["name"], e["payload"]["verdict"], e["payload"]["price"])


def score():
    regs = {x["key"]: x["payload"] for x in ledger.read() if x["kind"] == "valuation_verdict_2026"}
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    assert end >= SCORE_FROM, f"채점은 {SCORE_FROM} 이후에"
    ret = lambda s: float(krw(s, "2026-10-01", end).pipe(lambda x: x.iloc[-1] / x[:DATE].iloc[-1] - 1))
    b = ret(BENCH)
    rows = [dict(종목=p["name"], 판정=p["verdict"], 점수=p["score"], 초과수익=ret(s) - b) for s, p in regs.items()]
    d = pd.DataFrame(rows)
    d["맞음"] = np.where(d.점수 >= 1, d.초과수익 >= 0, np.where(d.점수 <= -1, d.초과수익 < 0, np.nan))
    rho = d[["점수", "초과수익"]].rank().corr().iloc[0, 1]
    OUT.mkdir(exist_ok=True)
    (OUT / "verdicts_2026_score.md").write_text(f"# 밸류에이션 판정 채점 ({DATE} → {end}, 기준 ACWI {b:+.1%})\n\n{d.to_markdown(index=False)}\n\n스피어만 상관 {rho:+.2f}\n", encoding="utf-8")
    print((OUT / "verdicts_2026_score.md").read_text())


if __name__ == "__main__":
    score() if "--score" in sys.argv else register()
