"""월간 경보 결과로 무료 뉴스레터(지자체 인구정책 담당자·지역 기자용) 한 호를 만듦.

    python newsletter.py --month 2026-10                      # 주력 주제(MAIN) + 보조 꼭지(SIDE)
    python newsletter.py --month 2026-10 --card "강원 정선군"    # 한 지역 한 장 (현실 점검 영업용)

결과: experiments/newsletter/<YYYY-MM>/issue.md, issue.html (메일 본문에 그대로 붙여 넣는 용도)
     experiments/newsletter/ledger.csv  (호마다 낸 경보를 쌓아 두는 적중 기록. 해가 끝나면 확정치로 채점)
     experiments/newsletter/<YYYY-MM>/card_<지역>.md
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import html
import os

import numpy as np
import pandas as pd

import kosis_monitor as K

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "newsletter")
LEDGER = os.path.join(OUT_ROOT, "ledger.csv")
REPO = "https://github.com/quanter02/chelsea-analytics/blob/claude/great-dirac-3wwgzy/experiments/"
MAIN = "E 전체 순이동"               # 주력 주제: 2026-10-10 사전 등록 선정 결과 (preregistration_niche_topic.md, niche_topic_selection.ipynb)
SIDE = ["B 영유아(0~9세) 순이동"]     # 보조 꼭지: 같은 선정에서 통과한 후보
TOP_N = 10
RATE_BASE = {"net": "예측 전입 대비", "flow": "예측 대비", "stock": "작년 말 대비"}


def topic_by_name(name: str) -> K.Topic:
    import monthly_update as MU
    return MU.topic_by_name(name)


def compute(name: str, today: dt.date) -> dict:
    """저장된 자료로 파이프라인을 돌려 경보·채점 성적·지역별 월 자료를 얻음 (월간 실행과 같은 규칙)."""
    os.chdir(HERE)
    import monthly_update as MU
    tp = topic_by_name(name)
    tidy = K.fetch(tp, verbose=False)
    latest = tidy.ym.max()
    tp = dataclasses.replace(tp, live_year=int(latest[:4]))
    out = K.run(tp, tidy=tidy, rule=MU.RULE.get(name, "v6"), verbose=False)
    out["latest"] = latest
    return out


def _fmt_num(x):
    return f"{x:+,.0f}"


def _section(name: str, out: dict) -> dict:
    tp, cb, t, L = out["topic"], out["calibration"], out["test"], out["live"]
    keep = L[L.상태 == "유지"] if len(L) else L
    up = int((keep.방향 == "증가").sum()) if len(keep) else 0
    down = int((keep.방향 == "감소").sum()) if len(keep) else 0
    rows = keep.head(TOP_N)
    return dict(name=name, kind=tp.kind, title=tp.name, test_years=f"{tp.test.start}~{tp.test.stop - 1}", table=tp.table, latest=out["latest"], cb=cb, t=t, keep=keep, rows=rows, up=up, down=down,
                n_eval=sum(1 for (u, y), d in out["units"].items() if y == tp.live_year and d["size"] >= tp.min_size))


def _ym(s):
    return f"{s[:4]}년 {int(s[4:])}월"


def build_issue(month: str, names: list[str], issue_no: int | None = None, extra_md: str | None = None) -> tuple[str, str, list[dict]]:
    today = dt.date.fromisoformat(month + "-01")
    outs = {n: compute(n, today) for n in names}
    secs = [_section(n, o) for n, o in outs.items()]
    no = issue_no or (len(_ledger().호.unique()) + 1 if os.path.exists(LEDGER) else 1)
    main = secs[0]
    title = f"지역 인구 이동 월간 경보 제{no}호 ({month[:4]}년 {int(month[5:])}월)"
    md = [f"# {title}", "",
          f"> {_ym(main['latest'])}까지 공표된 통계청 자료로, **예측보다 확실히 벗어난 시군구**만 골랐습니다. "
          f"기준은 미리 공개해 두었고 매달 바꾸지 않습니다. 원인은 말하지 않습니다.", ""]
    for s in secs:
        md += [f"## {s['title']}", "",
               f"평가 대상 {s['n_eval']}개 시군구 중 **{len(s['keep'])}곳**이 예측에서 확실히 벗어났습니다 "
               f"(예측보다 많음 {s['up']} · 적음 {s['down']}). {_ym(s['latest'])}까지 누적.", ""]
        if len(s["rows"]):
            md += [f"| 지역 | 예측 대비 누적 차이 | 이탈률({RATE_BASE[s['kind']]}) | 처음 경보가 뜬 달 |", "|---|---|---|---|"]
            md += [f"| {r.지역} | {_fmt_num(r.예측대비_누적차이)}명 | {r.이탈률} | {r.경보월} |" for r in s["rows"].itertuples()]
            gun = s["keep"][s["keep"].지역.str.endswith("군")]
            if len(gun):
                md += ["", f"**군 지역 경보 {len(gun)}곳**", "", "| 지역 | 예측 대비 누적 차이 | 이탈률 | 처음 경보가 뜬 달 |", "|---|---|---|---|"]
                md += [f"| {r.지역} | {_fmt_num(r.예측대비_누적차이)}명 | {r.이탈률} | {r.경보월} |" for r in gun.itertuples()]
            csv = f"{REPO}monthly/{month}/live_{s['name']}.csv".replace(" ", "%20")
            md += ["", f"전체 {len(s['keep'])}곳 목록: [CSV]({csv})"]
        md += [""]
    if extra_md:
        md += [extra_md.strip(), ""]
    cb, t = main["cb"], main["t"]
    import track_record as TR
    k = int(main["latest"][4:])
    pr = TR.precision(outs[names[0]], months=(k,)).iloc[0] if k < 12 else None
    alerts = sorted(f"{s['name']}|{r.지역}|{r.방향}|{r.경보월}" for s in secs for r in s["keep"].itertuples())
    digest = hashlib.sha256("\n".join(alerts).encode()).hexdigest()[:16]
    md += ["## 이 경보를 얼마나 믿을 수 있나", "",
           f"- 과거 채점({main['test_years']}): 실제로 크게 벗어난 해의 **{t['감지율']:.0%}**를 잡았고, "
           f"평범한 해에 잘못 울린 비율은 **{t['잘못된경보율']:.1%}**입니다. 상반기 안에 잡은 비율은 {t['상반기내_감지율']:.0%}입니다.",
           *([f"- 같은 기준으로 그 기간에 매달 이 뉴스레터를 냈다면, {k}월 자료로 낸 경보의 **{pr.같은방향:.0%}**가 연말 확정치에서도 같은 방향이었고, "
              f"{pr.진짜이탈:.0%}는 크게 벗어난 해였으며, {pr.평범한해:.0%}는 평범한 해(잘못 울린 경보)였습니다."] if pr is not None else []),
           "- 예측: 작년 연간 값 × 최근 5년 평균 월별 비중 (모든 지역에 같은 방식 하나).",
           "- 판정: 매달 확인해도 유효한 신뢰 구간이 0(예측과 같음)을 벗어날 때" + (f", 그리고 누적 차이가 규모의 {cb.m:.1%} 이상일 때" if cb.m > 0 else "") + f" (통일 규칙 6차, α = {cb.alpha}).",
           "- 기준은 보정 기간에서 한 번 정하고, 채점 기간 결과를 본 뒤 고치지 않았습니다. "
           f"[주제 선정 사전 등록]({REPO}preregistration_niche_topic.md) · [규칙 사전 등록]({REPO}preregistration_births_v5_v6.md) · [코드]({REPO}kosis_monitor.py)",
           "- 이번 호의 경보는 모두 [적중 기록]({0}newsletter/ledger.csv)에 남기고, 해가 끝나 확정치가 나오면 맞았는지 공개합니다.".format(REPO),
           f"- 이번 호 경보 목록의 지문(SHA-256 앞 16자리): `{digest}`. 발행 뒤 목록을 바꾸지 않았다는 증거로, 저장소 기록과 대조할 수 있습니다.",
           "", "## 출처", "",
           f"통계청 KOSIS 국내인구이동통계 `{main['table']}`" + "".join(f", `{s['table']}`" for s in secs[1:] if s['table'] != main['table'])
           + " (공공데이터, 가공함). 경보는 예측 대비 차이만 알려 주며, 원인은 확인 전까지 추정입니다.",
           "", "---", "우리 지역만 따로 받아 보기, 기준에 대한 질문은 이 메일에 답장해 주세요."]
    ledger_rows = [dict(호=no, 발행월=month, 주제=s["name"], 공표기준=s["latest"], 지역=r.지역, 방향=r.방향, 경보월=r.경보월,
                        누적차이=r.예측대비_누적차이, 이탈률=r.이탈률, 지문=digest) for s in secs for r in s["keep"].itertuples()]
    return "\n".join(md), _html(title, secs, md), ledger_rows


def _html(title, secs, md_lines) -> str:
    """메일 본문용 단순 HTML (외부 CSS·스크립트 없음)."""
    body, in_table = [], False
    cell = "border-bottom:1px solid #ddd;padding:4px 8px;text-align:left"
    for line in md_lines:
        if line.startswith("|"):
            if line.startswith("|---"): continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            tag = "td" if in_table else "th"
            if not in_table: body.append("<table style='border-collapse:collapse;font-size:13px'>"); in_table = True
            body.append("<tr>" + "".join(f"<{tag} style='{cell}'>{_inline(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table: body.append("</table>"); in_table = False
        if line.startswith("# "): body.append(f"<h1 style='font-size:20px'>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "): body.append(f"<h2 style='font-size:16px;margin-top:24px'>{html.escape(line[3:])}</h2>")
        elif line.startswith("> "): body.append(f"<p style='color:#555'>{_inline(line[2:])}</p>")
        elif line.startswith("- "): body.append(f"<p style='margin:4px 0'>• {_inline(line[2:])}</p>")
        elif line == "---": body.append("<hr>")
        elif line: body.append(f"<p>{_inline(line)}</p>")
    if in_table: body.append("</table>")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title></head>"
            "<body style='font-family:sans-serif;max-width:640px;margin:auto;padding:0 16px;line-height:1.6;color:#222'>" + "\n".join(body) + "</body></html>")


def _inline(s: str) -> str:
    import re
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return re.sub(r"\[(.+?)\]\((.+?)\)", r"<a href='\2'>\1</a>", s)


def _ledger() -> pd.DataFrame:
    return pd.read_csv(LEDGER) if os.path.exists(LEDGER) else pd.DataFrame(columns=["호"])


def region_card(month: str, name: str, region: str) -> str:
    """한 지역 한 장: 월별 예측 대비 차이와 누적, 판정 기준. 담당 부서에 보내 반응을 보는 용도."""
    out = compute(name, dt.date.fromisoformat(month + "-01"))
    tp = out["topic"]
    hits = [(k, u) for k, u in out["units"].items() if u["name"] == region and k[1] == tp.live_year]
    if not hits:
        raise KeyError(f"{region}: {tp.live_year}년 자료 없음")
    (unit, y), u = hits[0]
    L = out["live"]; row = L[L.지역 == region]
    e = u["e"][:u["n"]]; cum = np.cumsum(e); sc = np.cumsum(u["scale_inc"][:u["n"]])
    lines = [f"# {region}: {tp.name} ({y}년 1~{u['n']}월)", "",
             (f"**경보: 예측보다 {'많음' if row.iloc[0].방향 == '증가' else '적음'}** ({row.iloc[0].경보월}부터, 현재 {row.iloc[0].상태})" if len(row)
              else "**경보 없음**: 예측 범위 안입니다."), "",
             "| 월 | 예측 대비 차이 | 누적 차이 | 누적 이탈률 |", "|---|---|---|---|"]
    lines += [f"| {m + 1}월 | {_fmt_num(e[m])} | {_fmt_num(cum[m])} | {cum[m] / sc[m]:+.1%} |" for m in range(len(e))]
    t = out["test"]
    lines += ["", f"- 예측: 작년 연간 × 최근 5년 평균 월별 비중. 판정 기준은 미리 정해 공개 (과거 채점 감지율 {t['감지율']:.0%}, 잘못된 경보 {t['잘못된경보율']:.1%}).",
              f"- 출처: 통계청 KOSIS `{tp.table}` (가공). 원인은 말하지 않습니다."]
    return "\n".join(lines)


def main(month: str, names: list[str], write_ledger=True, extra_md: str | None = None):
    md, page, rows = build_issue(month, names, extra_md=extra_md)
    d = os.path.join(OUT_ROOT, month); os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "issue.md"), "w", encoding="utf-8").write(md)
    open(os.path.join(d, "issue.html"), "w", encoding="utf-8").write(page)
    if write_ledger and rows:
        old = _ledger()
        old = old[old.get("발행월", pd.Series(dtype=str)) != month] if len(old) else old
        pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(LEDGER, index=False)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default=dt.date.today().strftime("%Y-%m"))
    ap.add_argument("--topics", nargs="*")
    ap.add_argument("--card")
    ap.add_argument("--extra", help="특집 꼭지 마크다운 파일")
    a = ap.parse_args()
    if a.card:
        d = os.path.join(OUT_ROOT, a.month); os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"card_{a.card.replace(' ', '_')}.md")
        open(p, "w", encoding="utf-8").write(region_card(a.month, (a.topics or [MAIN])[0], a.card)); print(p)
    else:
        print(main(a.month, a.topics or [MAIN] + SIDE, extra_md=open(a.extra, encoding='utf-8').read() if a.extra else None))
