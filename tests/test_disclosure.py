"""Buyback signal checks (no network)."""
import numpy as np
import pandas as pd

from disclosure import alert, dart, research as R, signal as S


def _raw():
    return pd.DataFrame([
        dict(corp_code="1", corp_name="가", stock_code="000001", corp_cls="K", rcept_no="r1", rcept_dt="20260930",
             report_nm="주요사항보고서(자기주식취득신탁계약체결결정)"),
        dict(corp_code="2", corp_name="나", stock_code="000002", corp_cls="Y", rcept_no="r2", rcept_dt="20260930",
             report_nm="주요사항보고서(자기주식취득결정)"),
        dict(corp_code="3", corp_name="다", stock_code="000003", corp_cls="K", rcept_no="r3", rcept_dt="20260930",
             report_nm="[기재정정]주요사항보고서(자기주식취득결정)"),
        dict(corp_code="4", corp_name="라", stock_code="000004", corp_cls="E", rcept_no="r4", rcept_dt="20260930",
             report_nm="주요사항보고서(자기주식취득결정)"),
        dict(corp_code="5", corp_name="마", stock_code="000005", corp_cls="K", rcept_no="r5", rcept_dt="20260930",
             report_nm="주요사항보고서(자기주식취득신탁계약연장결정)"),
    ])


def test_clean_keeps_original_kospi_kosdaq_decisions():
    ev = dart.clean(_raw())
    assert list(ev.rcept_no) == ["r1", "r2"]
    assert list(ev.type) == ["신탁", "직접"] and list(ev.market) == ["KOSDAQ", "KOSPI"]


def test_entry_is_next_trading_day_close():
    entry, exit_ = S.entry_exit("2026-10-02")           # Friday
    assert entry == pd.Timestamp("2026-10-05")          # Monday
    assert len(pd.bdate_range(entry, exit_)) == S.HOLD + 1
    entry, _ = S.entry_exit("2026-10-03")               # Saturday filing → t0 Monday → entry Tuesday
    assert entry == pd.Timestamp("2026-10-06")


def test_window_covers_weekend_on_monday():
    assert alert.window("2026-10-05") == (pd.Timestamp("2026-10-03"), pd.Timestamp("2026-10-05"))
    assert alert.window("2026-10-07") == (pd.Timestamp("2026-10-07"), pd.Timestamp("2026-10-07"))


def test_grades_follow_backtest():
    ev = dart.clean(_raw())
    assert S.grade(ev.iloc[0]).grade == "A"     # KOSDAQ trust
    assert S.grade(ev.iloc[1]).grade == "C"     # KOSPI direct: negative median
    assert "관찰만" in S.grade(ev.iloc[1]).action


def test_unvalidated_features_never_change_grade():
    ev = dart.clean(_raw()).iloc[0]
    base = S.grade(ev)
    rich = S.grade(ev, feats={"amount": 5e9, "cancel": True, "days": 180}, marcap=1e11)
    assert rich.grade == base.grade and any("검증 전" in x for x in rich.info)


def test_crowding_and_liquidity_warnings():
    ev = dart.clean(_raw()).iloc[0]
    s = S.grade(ev, crowd_n=45, adv=1e8)
    assert any("몰림" in r for r in s.risks) and any("거래대금이 작음" in r for r in s.risks)
    assert any("백테스트는 첫 공시만" in r for r in S.grade(ev, repeat=True).risks)


def test_trust_count_uses_past_only():
    hist = pd.DataFrame({"type": ["신탁"] * 3 + ["직접"],
                         "event_date": pd.to_datetime(["2026-09-01", "2026-09-20", "2026-09-30", "2026-09-25"])})
    assert S.trust_count_30d(hist, "2026-09-30") == 2


def test_detail_features_reads_either_report_type():
    trust = {"rcept_no": "r1", "ctr_prc": "5,000,000,000", "ctr_pd_bgd": "2026년 10월 01일",
             "ctr_pd_edd": "2027년 04월 01일", "ctr_pp": "주주가치 제고"}
    direct = {"rcept_no": "r2", "aqpln_prc_ostk": "1,000,000,000", "aqexpd_bgd": "2026-10-01",
              "aqexpd_edd": "2026-12-31", "aq_pp": "주가안정 및 소각"}
    f1, f2 = dart.detail_features(trust), dart.detail_features(direct)
    assert f1["amount"] == 5e9 and f1["days"] == 182 and f1["cancel"] is False
    assert f2["amount"] == 1e9 and f2["cancel"] is True
    assert dart.detail_features(None)["cancel"] is None


def test_markdown_has_disclaimer():
    ev = dart.clean(_raw())
    md = alert.to_markdown([S.grade(r) for _, r in ev.iterrows()], "2026-09-30")
    assert "[A] 가" in md and "투자 권유가 아닙니다" in md
    assert "없습니다" in alert.to_markdown([], "2026-09-30")


def test_verdict_needs_both_halves():
    rng = np.random.default_rng(0)
    n = 1000
    d = pd.DataFrame({"event_date": pd.date_range("2016-01-01", periods=n, freq="D"),
                      "size": rng.permutation(n)})
    noise = rng.normal(0, 0.01, n)
    d["ret"] = d["size"] / n * 0.05 + noise                       # bigger is better in every period
    assert R.verdict(d, "size", ret="ret")["통과"]
    first = d.index < n // 2
    d["ret"] = np.where(first, d["size"] / n * 0.10, 0) + noise    # only in the first half
    assert not R.verdict(d, "size", ret="ret")["통과"]
