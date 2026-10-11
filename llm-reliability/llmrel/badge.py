"""검증 배지: 예측 장부 한 줄을 남의 글·영상 설명에 붙일 수 있는 카드로 만든다.

세 가지 모양 (같은 내용)
  - HTML 카드: 블로그·웹페이지에 붙여넣기 (인라인 CSS만, 외부 파일 없음)
  - SVG 이미지: GitHub README·노션·이미지를 받는 곳
  - 한 줄 텍스트: 유튜브 설명·X처럼 이미지가 안 되는 곳

들어가는 것: 예측(확률), 장부에 기록된 시각, 경기 시작까지 남았던 시간, 해시 앞 16자리, 공개 커밋,
누구나 해 볼 수 있는 확인 방법, 결과가 나오면 채점(RPS, 낮을수록 좋음).
숫자와 기호 위주로 만들어 언어와 상관없이 읽힌다 (라벨은 한·영 병기).

    python -m llmrel.badge "2026-10-10|Manchester United|Tottenham Hotspur" --kickoff 2026-10-10T16:30:00Z [--result 2-1]
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd

from . import ledger

ROOT = Path(__file__).parent.parent
OUT = ROOT / "content" / "badges"
REPO = "https://github.com/quanter02/chelsea-analytics"
LEDGER_URL = f"{REPO}/blob/claude/adoring-dijkstra-jshoun/llm-reliability/ledger/predictions.jsonl"


def rps(p: tuple[float, float, float], outcome: int) -> float:
    """3결과 RPS. p = (홈, 무, 원정), outcome 0/1/2."""
    o = [1.0 if i == outcome else 0.0 for i in range(3)]
    c1, c2 = p[0] - o[0], p[0] + p[1] - o[0] - o[1]
    return (c1 ** 2 + c2 ** 2) / 2


def find(key: str, kind: str = "epl") -> dict:
    rows = [r for r in ledger.read() if r["kind"] == kind and r["key"] == key]
    if not rows:
        raise KeyError(key)
    return rows[0]                                            # 처음 기록된 줄 (경기 전 증거)


def card_data(key: str, kickoff: str, result: str | None = None, commit: str = "25644f2") -> dict:
    r = find(key)
    date, home, away = key.split("|")
    p = (r["payload"]["pH"], r["payload"]["pD"], r["payload"]["pA"])
    rec, ko = pd.Timestamp(r["recorded_at"]), pd.Timestamp(kickoff)
    lead = ko - rec
    d = {"home": home, "away": away, "date": date, "p": p, "model": r["model"], "hash": r["hash"], "seq": r["seq"],
         "recorded": rec.strftime("%Y-%m-%d %H:%M UTC"), "lead": f"{lead.days}d {lead.seconds // 3600}h", "before": lead.total_seconds() > 0,
         "commit": commit, "result": None, "rps": None}
    if result:
        hg, ag = map(int, result.split("-"))
        d["result"] = result
        d["rps"] = rps(p, 0 if hg > ag else 1 if hg == ag else 2)
    return d


def to_text(d: dict) -> str:
    pct = " / ".join(f"{x * 100:.0f}%" for x in d["p"])
    tail = f" · result {d['result']} · RPS {d['rps']:.3f}" if d["result"] else " · scoring after the match"
    return (f"✅ Pre-registered forecast | {d['home']} v {d['away']}: {pct} (H/D/A) · recorded {d['recorded']} "
            f"({d['lead']} before kickoff) · hash {d['hash'][:16]}{tail} · verify: {LEDGER_URL}")


def to_html(d: dict) -> str:
    e = html.escape
    bars = "".join(f'<div style="flex:{max(x, 0.02):.3f};background:{c};height:8px;border-radius:2px" title="{lab} {x * 100:.0f}%"></div>'
                   for x, c, lab in zip(d["p"], ("#2a63d4", "#9aa3ac", "#df6330"), ("Home", "Draw", "Away")))
    score = (f'<div style="margin-top:6px">결과 Result <b>{e(d["result"])}</b> · RPS <b>{d["rps"]:.3f}</b> <span style="color:#5a6672">(낮을수록 좋음 · lower is better)</span></div>'
             if d["result"] else '<div style="margin-top:6px;color:#5a6672">경기 후 채점 예정 · scoring after the match</div>')
    return f"""<div style="font-family:system-ui,-apple-system,'Segoe UI',sans-serif;max-width:420px;border:1px solid #dce2e7;border-radius:10px;padding:12px 14px;background:#fff;color:#15202b;font-size:13px;line-height:1.5">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:8px">
    <b style="font-size:13px">✅ 경기 전 기록 · Pre-registered</b>
    <a href="{REPO}" style="color:#2a63d4;text-decoration:none;font-weight:700">셈법</a>
  </div>
  <div style="margin:6px 0 2px;font-size:15px;font-weight:700">{e(d["home"])} v {e(d["away"])}</div>
  <div style="font-family:ui-monospace,Menlo,monospace">{d["p"][0] * 100:.0f}% · {d["p"][1] * 100:.0f}% · {d["p"][2] * 100:.0f}% <span style="color:#5a6672">(H · D · A)</span></div>
  <div style="display:flex;gap:2px;margin:6px 0">{bars}</div>
  <div style="color:#5a6672;font-size:12px">기록 Recorded {e(d["recorded"])} · 킥오프 {e(d["lead"])} 전 before kickoff</div>
  <div style="font-family:ui-monospace,Menlo,monospace;font-size:12px">hash {d["hash"][:16]} · commit {e(d["commit"])}</div>
  {score}
  <div style="margin-top:6px;font-size:12px"><a href="{LEDGER_URL}" style="color:#2a63d4">장부에서 확인 · Verify</a> <span style="color:#5a6672">— python -m llmrel.ledger verify</span></div>
</div>"""


def to_svg(d: dict) -> str:
    e = html.escape
    W, H = 420, 150
    x0, bw = 14, W - 28
    segs, x = [], x0
    for v, c in zip(d["p"], ("#2a63d4", "#9aa3ac", "#df6330")):
        w = bw * v
        segs.append(f'<rect x="{x:.1f}" y="70" width="{max(w - 2, 1):.1f}" height="8" rx="2" fill="{c}"/>')
        x += w
    res = (f'<text x="14" y="138" font-size="12" fill="#15202b">결과 Result {e(d["result"])} · RPS {d["rps"]:.3f} (lower is better)</text>'
           if d["result"] else '<text x="14" y="138" font-size="12" fill="#5a6672">경기 후 채점 예정 · scoring after the match</text>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Pre-registered forecast {e(d['home'])} v {e(d['away'])}">
  <rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="10" fill="#ffffff" stroke="#dce2e7"/>
  <g font-family="system-ui,-apple-system,Segoe UI,sans-serif">
    <text x="14" y="22" font-size="12.5" font-weight="700" fill="#148f68">✓ 경기 전 기록 · Pre-registered</text>
    <text x="{W - 14}" y="22" font-size="12.5" font-weight="700" fill="#2a63d4" text-anchor="end">셈법</text>
    <text x="14" y="42" font-size="15" font-weight="700" fill="#15202b">{e(d['home'])} v {e(d['away'])}</text>
    <text x="14" y="61" font-size="12.5" fill="#15202b" font-family="ui-monospace,Menlo,monospace">{d['p'][0] * 100:.0f}% · {d['p'][1] * 100:.0f}% · {d['p'][2] * 100:.0f}%  (H · D · A)</text>
    {''.join(segs)}
    <text x="14" y="96" font-size="11.5" fill="#5a6672">Recorded {e(d['recorded'])} · {e(d['lead'])} before kickoff</text>
    <text x="14" y="114" font-size="11.5" fill="#15202b" font-family="ui-monospace,Menlo,monospace">hash {d['hash'][:16]} · commit {e(d['commit'])}</text>
    {res}
  </g>
</svg>
"""


def build(key: str, kickoff: str, result: str | None = None, slug: str | None = None) -> dict:
    d = card_data(key, kickoff, result)
    slug = slug or f"{d['date']}_{d['home'].split()[0].lower()}_{d['away'].split()[0].lower()}"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{slug}.html").write_text(to_html(d), encoding="utf-8")
    (OUT / f"{slug}.svg").write_text(to_svg(d), encoding="utf-8")
    (OUT / f"{slug}.txt").write_text(to_text(d) + "\n", encoding="utf-8")
    return {**d, "slug": slug}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("key"); ap.add_argument("--kickoff", required=True); ap.add_argument("--result")
    a = ap.parse_args()
    d = build(a.key, a.kickoff, a.result)
    print(to_text(d))
