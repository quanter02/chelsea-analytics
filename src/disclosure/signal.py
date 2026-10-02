"""Buyback disclosure → graded signal with its evidence, an action and the risks.

Grades come only from what disclosure_event_study_buyback_v2.ipynb validated on 2016-01 – 2026-08
(trade: buy at the t0+1 close, sell 20 trading days later, vs the equal-weight benchmark, 0.3% round-trip cost,
delisted firms restored). Features the backtest has not tested yet (size vs market cap, 소각, programme length)
are shown as information and never move the grade. Validate them with `disclosure.research.bucket_test` first.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HOLD = 20
# (market, type) → grade and the backtest numbers behind it: mean / median trade return %, edge vs the
# same stocks on random dates (%p) and its t-stat. Source: notebook cell "delist_fix3", table B.
EVIDENCE = {
    ("KOSDAQ", "신탁"): dict(grade="A", n=1789, mean=2.05, median=0.57, edge=2.26, t=5.19, pos_years="11/11"),
    ("KOSPI", "신탁"): dict(grade="B", n=785, mean=1.28, median=0.49, edge=2.05, t=5.00, pos_years=None),
    ("KOSPI", "직접"): dict(grade="C", n=751, mean=0.98, median=-0.34, edge=1.31, t=2.52, pos_years=None),
    ("KOSDAQ", "직접"): dict(grade="C", n=1015, mean=16.18, median=-0.32, edge=16.36, t=1.09, pos_years=None),
}
GRADE_NOTE = {
    "A": "전 기간 검증 통과 (플라시보 대비 우위, 중앙값 > 0, 연도별 일관성)",
    "B": "플라시보 대비 유의하지만 연도별 일관성 미확인",
    "C": "중앙값이 마이너스. 평균은 소수 급등주가 만든 값이라 따라 사기 어려움",
}
# Trust disclosures market-wide in the previous 30 days (notebook 8-6): few 2.69% / mid 1.68% / many 1.41%
# (median −0.27%). Found by looking at the results, so it is a warning, not a filter.
CROWD_CUTS = (16, 30)
CROWD_EVIDENCE = {"적음": 2.69, "보통": 1.68, "많음": 1.41}
POSITION_KRW = 14_000_000   # 2.8억 portfolio / 20 slots (notebook 7단계 capacity check)
MAX_ADV_SHARE = 0.05        # buy at most 5% of a day's traded value


@dataclass
class Signal:
    rcept_no: str
    corp_name: str
    stock_code: str
    market: str
    type: str
    event_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    grade: str
    evidence: str
    action: str
    risks: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={self.rcept_no}"


def trading_day_on_or_after(d) -> pd.Timestamp:
    """t0. Weekends roll forward; KRX holidays are not modelled, so check the entry date around holidays."""
    return pd.bdate_range(pd.Timestamp(d).normalize(), periods=1)[0]


def entry_exit(event_date, hold: int = HOLD) -> tuple[pd.Timestamp, pd.Timestamp]:
    t0 = trading_day_on_or_after(event_date)
    days = pd.bdate_range(t0, periods=hold + 2)
    return days[1], days[1 + hold]


def crowd_level(n: int) -> str:
    lo, hi = CROWD_CUTS
    return "적음" if n <= lo else "보통" if n <= hi else "많음"


def trust_count_30d(all_events: pd.DataFrame, event_date) -> int:
    """Trust disclosures (all firms) in the 30 calendar days before event_date. Uses past data only."""
    d = pd.Timestamp(event_date)
    t = all_events[all_events["type"] == "신탁"]
    return int(((t["event_date"] < d) & (t["event_date"] >= d - pd.Timedelta(days=30))).sum())


def grade(ev: pd.Series, crowd_n: int | None = None, adv: float | None = None, marcap: float | None = None,
          feats: dict | None = None, repeat: bool = False, position: float = POSITION_KRW) -> Signal:
    """ev: one row of dart.clean(). adv = 20-day average traded value (KRW), marcap in KRW.
    repeat: the same stock already filed in the previous 30 days (the backtest kept only the first)."""
    e = EVIDENCE[(ev["market"], ev["type"])]
    entry, exit_ = entry_exit(ev["event_date"])
    evidence = (f"{ev['market']} {ev['type']} 공시 {e['n']:,}건 백테스트: 20일 평균 {e['mean']:+.2f}%, "
                f"중앙값 {e['median']:+.2f}%, 무작위 날짜 대비 {e['edge']:+.2f}%p (t={e['t']:.1f})"
                + (f", 플러스 연도 {e['pos_years']}" if e["pos_years"] else "") + f". {GRADE_NOTE[e['grade']]}")
    action = (f"{entry:%m/%d} 종가 진입 → {exit_:%m/%d} 종가 청산 (20거래일 보유)" if e["grade"] in ("A", "B")
              else "관찰만. 이 유형은 따라 사는 매매로 검증되지 않음")

    risks, info = [], []
    if repeat:
        risks.append("같은 종목이 30일 안에 이미 공시함. 백테스트는 첫 공시만 썼으므로 이 공시는 검증 범위 밖")
    if crowd_n is not None:
        lvl = crowd_level(crowd_n)
        if lvl == "많음":
            risks.append(f"직전 30일 신탁 공시 {crowd_n}건으로 몰림 구간. 과거 이 구간 평균 {CROWD_EVIDENCE['많음']:.2f}%, "
                         "중앙값 마이너스 (보통 급락장)")
        else:
            info.append(f"직전 30일 신탁 공시 {crowd_n}건 ({lvl}, 과거 평균 {CROWD_EVIDENCE[lvl]:.2f}%)")
    if adv is not None and not np.isnan(adv):
        cap = adv * MAX_ADV_SHARE
        if cap < position:
            risks.append(f"거래대금이 작음: 하루 거래대금의 5% = {cap / 1e6:.0f}백만 원 < 목표 비중 {position / 1e6:.0f}백만 원")
        info.append(f"20일 평균 거래대금 {adv / 1e8:.1f}억 원")
    f = feats or {}
    if marcap and not np.isnan(f.get("amount", np.nan)):
        info.append(f"매입 규모 {f['amount'] / 1e8:.0f}억 원 = 시가총액의 {f['amount'] / marcap * 100:.2f}% (등급 미반영, 검증 전)")
    if f.get("cancel") is not None:
        info.append(f"소각 언급 {'있음' if f['cancel'] else '없음'} (등급 미반영, 검증 전)")
    if not np.isnan(f.get("days", np.nan)):
        info.append(f"계약/취득 기간 {int(f['days'])}일")
    risks.append("휴장일은 반영하지 않음. 연휴 전후에는 진입일을 직접 확인")

    return Signal(ev["rcept_no"], ev["corp_name"], ev["stock_code"], ev["market"], ev["type"],
                  pd.Timestamp(ev["event_date"]), entry, exit_, e["grade"], evidence, action, risks, info)
