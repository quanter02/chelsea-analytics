"""옵션 가격에서 이벤트(실적 발표, 게임 출시)에 시장이 예상하는 주가 변동폭을 뽑는다. 개인 계좌 정보 없음.

    python invest/options_event.py     # fc.yahoo.com, query1.finance.yahoo.com 허용 필요 → invest/out/options_event.md

방법: 만기별 ATM 내재변동성의 총분산(σ²·T)을 구하고, 이벤트를 사이에 둔 두 만기의 총분산 차이에서
'평소 변동성 × 기간'을 빼면 그 이벤트 하루의 분산이 남는다 (이벤트 분리, 표준적인 term-structure 방법).
평소 변동성 = 이벤트가 없는 가장 가까운 만기의 ATM 내재변동성.
"""
from __future__ import annotations

import datetime as dt
import http.cookiejar
import json
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "out"
EVENTS = {"TTWO": [("실적 발표", "2026-11-09"), ("GTA VI 출시", "2026-11-19")], "GOOGL": [("실적 발표", "2026-10-28")]}


def opener():
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    op.addheaders = [("User-Agent", "Mozilla/5.0")]
    try:
        op.open("https://fc.yahoo.com", timeout=20)
    except Exception:
        pass   # 404여도 쿠키는 남는다
    crumb = op.open("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=20).read().decode()
    return op, crumb


def term(op, crumb, sym: str):
    get = lambda d=None: json.loads(op.open(f"https://query1.finance.yahoo.com/v7/finance/options/{sym}?crumb={crumb}" + (f"&date={d}" if d else ""), timeout=30).read())["optionChain"]["result"][0]
    c = get()
    S, today = c["quote"]["regularMarketPrice"], dt.date.today()
    rows = []
    for e in c["expirationDates"][:9]:
        o = get(e)["options"][0]
        calls, puts = {x["strike"]: x for x in o["calls"]}, {x["strike"]: x for x in o["puts"]}
        k = min(set(calls) & set(puts), key=lambda k: abs(k - S))
        iv = (calls[k].get("impliedVolatility", 0) + puts[k].get("impliedVolatility", 0)) / 2
        mid = lambda x: (x["bid"] + x["ask"]) / 2 if x.get("bid") and x.get("ask") else x.get("lastPrice", 0)
        T = max((dt.datetime.utcfromtimestamp(e).date() - today).days, 1) / 365
        rows.append(dict(exp=dt.datetime.utcfromtimestamp(e).date(), T=T, iv=iv, var=iv * iv * T, straddle=(mid(calls[k]) + mid(puts[k])) / S))
    return S, rows


def event_sigma(rows, day: str, base_iv: float) -> float:
    d = dt.date.fromisoformat(day)
    before = max((r for r in rows if r["exp"] < d), key=lambda r: r["exp"])
    after = min((r for r in rows if r["exp"] >= d), key=lambda r: r["exp"])
    ev = after["var"] - before["var"] - base_iv ** 2 * (after["T"] - before["T"])
    return max(ev, 0) ** 0.5


def main():
    op, crumb = opener()
    lines, res = [f"# 옵션이 보는 이벤트 변동폭 ({dt.date.today()})", ""], {}
    for sym, evs in EVENTS.items():
        S, rows = term(op, crumb, sym)
        first = min(dt.date.fromisoformat(d) for _, d in evs)
        base = min((r for r in rows if r["exp"] < first), key=lambda r: r["exp"])["iv"]
        lines += [f"## {sym} (주가 {S})", "", "| 만기 | ATM 내재변동성 | 스트래들 (예상 절대 변동) |", "|---|---|---|"]
        lines += [f"| {r['exp']} | {r['iv']:.0%} | ±{r['straddle']:.1%} |" for r in rows]
        sig = {nm: event_sigma(rows, d, base) for nm, d in evs}
        res[sym] = {nm: round(s, 4) for nm, s in sig.items()}
        lines += ["", f"평소 변동성(이벤트 전 가장 가까운 만기) {base:.0%}"] + \
                 [f"- {nm}: 하루 표준편차 ±{s:.1%} (평균 절대 변동 약 ±{s * 0.8:.1%})" for nm, s in sig.items()] + [""]
    OUT.mkdir(exist_ok=True)
    (OUT / "options_event.json").write_text(json.dumps(res, ensure_ascii=False))
    (OUT / "options_event.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
