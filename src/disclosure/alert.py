"""Daily buyback alert: new disclosures → graded signals → Markdown (and optionally Telegram).

    export DART_API_KEY=...
    python -m disclosure.alert                      # today's disclosures (KST), print Markdown
    python -m disclosure.alert --date 2026-09-30    # a past day
    python -m disclosure.alert --send               # also send grade A/B to Telegram
    python -m disclosure.alert --inspect            # print raw DART detail keys of the first disclosure

Run it after 19:00 KST on trading days: the backtest buys at the next day's close, so there is time.
Telegram: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.

Not investment advice. Selling these signals for a fee in Korea requires registering as a
유사투자자문업자 with the FSC; personal use and free content do not.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

from . import dart, market, signal as S

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "disclosure"
GRADE_ORDER = {"A": 0, "B": 1, "C": 2}
DISCLAIMER = ("과거 백테스트 기반 정보이며 투자 권유가 아닙니다. 수익을 보장하지 않으며 투자 판단과 책임은 본인에게 있습니다.")


def window(date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Days whose disclosures are new as of `date`: everything after the previous business day.
    On a Monday that is Sat–Mon, so weekend filings are not missed."""
    d = pd.Timestamp(date).normalize()
    prev = d - pd.offsets.BDay(1)
    return prev + pd.Timedelta(days=1), d


def build(date, key: str, with_market: bool = True) -> list[S.Signal]:
    start, end = window(date)
    hist = dart.clean(dart.fetch_list(key, start - pd.Timedelta(days=45), end))  # 45 days covers the 30-day count
    new = hist[(hist.event_date >= start) & (hist.event_date <= end)]
    new = new.drop_duplicates("stock_code", keep="first")
    lst = market.listing() if with_market and len(new) else pd.DataFrame(columns=["Marcap"])
    out = []
    for _, ev in new.iterrows():
        feats = dart.detail_features(dart.fetch_detail(key, ev))
        adv = market.avg_traded_value(ev.stock_code, ev.event_date) if with_market else None
        mc = float(lst.Marcap.get(ev.stock_code, float("nan"))) if len(lst) else None
        prior = hist[(hist.stock_code == ev.stock_code) & (hist.event_date < ev.event_date)
                     & (hist.event_date >= ev.event_date - pd.Timedelta(days=30))]
        out.append(S.grade(ev, crowd_n=S.trust_count_30d(hist, ev.event_date), adv=adv, marcap=mc, feats=feats,
                           repeat=len(prior) > 0))
        time.sleep(0.12)
    return sorted(out, key=lambda s: (GRADE_ORDER[s.grade], s.corp_name))


def to_markdown(sigs: list[S.Signal], date) -> str:
    head = f"# 자사주 매입 공시 시그널 · {pd.Timestamp(date):%Y-%m-%d}\n"
    if not sigs:
        return head + "\n새 자사주 매입 결정 공시가 없습니다.\n"
    counts = pd.Series([s.grade for s in sigs]).value_counts()
    lines = [head, "등급별: " + " · ".join(f"{g} {counts.get(g, 0)}건" for g in "ABC") + "\n"]
    for s in sigs:
        lines += [f"## [{s.grade}] {s.corp_name} ({s.stock_code}) · {s.market} {s.type}",
                  f"- 공시: {s.event_date:%Y-%m-%d} · [원문]({s.url})",
                  f"- 근거: {s.evidence}",
                  f"- 실행: {s.action}"]
        lines += [f"- 참고: {x}" for x in s.info]
        lines += [f"- 위험: {x}" for x in s.risks]
        lines.append("")
    lines.append(f"> {DISCLAIMER}")
    return "\n".join(lines) + "\n"


def telegram_text(s: S.Signal) -> str:
    risks = "\n".join(f"⚠ {r}" for r in s.risks[:-1])  # the holiday note is in every message; keep it short
    return (f"[{s.grade}] {s.corp_name} ({s.stock_code}) {s.market} {s.type}\n{s.action}\n"
            f"{s.evidence}\n{risks}\n{s.url}\n\n{DISCLAIMER}").replace("\n\n\n", "\n\n")


def send_telegram(text: str) -> None:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        raise SystemExit("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 환경 변수가 필요합니다.")
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": text, "disable_web_page_preview": True}, timeout=30)
    r.raise_for_status()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d"))
    ap.add_argument("--send", action="store_true", help="send grade A/B signals to Telegram")
    ap.add_argument("--no-market", action="store_true", help="skip market cap / liquidity lookups")
    ap.add_argument("--inspect", action="store_true", help="print raw detail keys and exit")
    a = ap.parse_args()
    key = dart.api_key()

    if a.inspect:
        start, end = window(a.date)
        ev = dart.clean(dart.fetch_list(key, start - pd.Timedelta(days=10), end))
        for _, r in ev.groupby("type").head(1).iterrows():
            print(r.type, r.corp_name, json.dumps(dart.fetch_detail(key, r), ensure_ascii=False, indent=1))
        return

    sigs = build(a.date, key, with_market=not a.no_market)
    md = to_markdown(sigs, a.date)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"alert_{pd.Timestamp(a.date):%Y%m%d}.md"
    path.write_text(md, encoding="utf-8")
    print(md)
    print(f"→ {path}")
    if a.send:
        for s in sigs:
            if s.grade in ("A", "B"):
                send_telegram(telegram_text(s))


if __name__ == "__main__":
    main()
