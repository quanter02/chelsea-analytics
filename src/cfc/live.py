"""Current-season dossier (Premier League 2026/27, Understat data).

    python -m cfc.live                         # from the snapshot in data/understat_2026/
    python -m cfc.live --fetch                 # re-download the latest data (run locally)
    python -m cfc.live --opponent "Everton"    # scout a different opponent
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd

from . import charts, fingerprint as FP, scouting as S, understat as U
from .report import CSS, _img

ROOT = Path(__file__).resolve().parents[2]
TEAM = "Chelsea"
STYLE = {"Pressing intensity (1/PPDA)"}
NEXT_FIXTURE = {"opponent": "Bournemouth", "date": "2026-10-10", "venue": "H"}  # Understat fixture list
# (team this season, manager, manager's reference team, reference season label) — sources in README
MANAGERS = {"Chelsea": ("사비 알론소", "Bayer Leverkusen", "레버쿠젠 2023/24 (분데스리가 무패 우승)"),
            "Bournemouth": ("마르코 로제", "RasenBallsport Leipzig", "라이프치히 2023/24")}


FP_NOTES = {
    "Chelsea": "공격 생산과 침투는 이미 레버쿠젠 수준에 가깝다. 벌어진 건 실점 억제와 볼 통제(압박 회피·xG 점유율). "
               "알론소 축구의 핵심인 '공을 오래 지키며 상대를 묶는 구조'는 아직 이식되지 않았다.",
    "Bournemouth": "전방 압박 강도는 로제의 라이프치히와 거의 같다. 반면 압박을 풀어내는 빌드업과 박스 앞 봉쇄가 크게 떨어진다. "
                   "→ 첼시는 본머스의 첫 압박만 넘기면 뒷공간이 열리고, 반대로 본머스 빌드업을 압박하면 볼을 뺏을 기회가 많다.",
}


def _findings(items, cls, own_view: bool = False):
    return "".join(
        f'<li class="{cls}"><h4>{html.escape(f.title)}</h4><p class="ev">{html.escape(f.evidence)}</p>'
        f'<p class="act">→ {"<b>우리 대응:</b> " if own_view else ""}{html.escape(f.counter if own_view and f.counter else f.action)}</p></li>'
        for f in items)


def build(tm: pd.DataFrame, shots: pd.DataFrame, opponent: str, as_of: str, out: Path) -> Path:
    c = tm[tm.team == TEAM].sort_values("date").reset_index(drop=True)
    n = len(c)
    first, last = c.iloc[:2], c.iloc[2:]
    k = {
        "pts": int(c.points.sum()), "xpts": c.xpts.sum(), "gf": int(c.goals.sum()), "ga": int(c.goals_against.sum()),
        "xg": c.xg.sum(), "xga": c.xga.sum(),
        "f_xg": first.xg.mean(), "f_xga": first.xga.mean(), "l_xg": last.xg.mean(), "l_xga": last.xga.mean(),
        "sp_pg": c.xga_setpiece.mean(), "lg_sp_pg": tm.xga_setpiece.mean(),
        "sp_worst": c.sort_values("xga_setpiece", ascending=False).head(2)[["opponent", "xga_setpiece"]].values.tolist(),
        "deep_allowed_last": last.deep_allowed.mean(), "deep_allowed_first": first.deep_allowed.mean(),
        "deep_worst": c.loc[c.deep_allowed.idxmax(), ["opponent", "deep_allowed"]].tolist(),
    }
    fp_now, fp_ref = FP.from_team_matches(tm), FP.load_bundesliga_2023()
    fps = {t: FP.compare(fp_now, t, fp_ref, MANAGERS[t][1]) for t in (TEAM, opponent) if t in MANAGERS}
    own = S.scout(TEAM, tm, shots)
    mu = S.matchup(TEAM, opponent, tm, shots)
    if opponent in fps:
        o_fp = fps[opponent]["now"]
        if o_fp["press_resist"] <= 0.3:
            mu.append(S.Finding("edge", "공략 포인트: 상대 빌드업",
                                f"{opponent}가 상대 압박을 풀어내는 능력은 리그 백분위 {o_fp['press_resist']:.0%} (감독 지문 분석)",
                                "골킥·후방 빌드업 때 전방 3명이 센터백과 수비형 미드필더의 패스 길을 막는 압박 트리거를 설정.", 0.75))
        if o_fp["press"] >= 0.6 and o_fp["box_guard"] <= 0.3:
            mu.append(S.Finding("edge", "공략 포인트: 첫 압박 뒤의 공간",
                                f"압박 강도 백분위 {o_fp['press']:.0%}인데 박스 앞 봉쇄는 {o_fp['box_guard']:.0%}. 압박이 풀리면 수비 라인 앞이 비는 구조",
                                "골키퍼·센터백에서 풀백·10번으로 이어지는 3자 패스로 첫 압박을 넘긴 뒤 곧바로 전진 패스.", 0.7))
    buckets, labels = [0, 15, 30, 45, 60, 75, 200], ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]
    sa = S._shots_against(shots, tm, TEAM)
    lvl = sa[sa.state == "level"].assign(b=lambda d: pd.cut(d.minute, buckets, right=False, labels=labels))
    lg = shots[shots.state == "level"].assign(b=lambda d: pd.cut(d.minute, buckets, right=False, labels=labels))
    late_curve = lvl.groupby("b", observed=False).xg.sum() / lvl.xg.sum()
    lg_curve = lg.groupby("b", observed=False).xg.sum() / lg.xg.sum()
    rep = S.scout(opponent, tm, shots)
    plan = S.game_plan(rep, max_items=6)
    fp_edges = [f for f in mu if f.title.startswith("공략 포인트: 상대 빌드업")]
    if fp_edges:
        plan.insert(2, f"[압박] {fp_edges[0].evidence} → {fp_edges[0].action}")
    o = tm[tm.team == opponent]
    o_pts, o_xpts = int(o.points.sum()), o.xpts.sum()

    season = tm.groupby("team").mean(numeric_only=True)

    def pct(s, invert=False):
        r = s.rank(pct=True)[opponent]
        return 1 - r + 1 / len(s) if invert else r

    prof = pd.Series({
        "Non-penalty xG": pct(season.npxg),
        "Non-penalty xG against (inv.)": pct(season.npxga, invert=True),
        "Set-piece xG": pct(season.xg_setpiece),
        "Set-piece xG against (inv.)": pct(season.xga_setpiece, invert=True),
        "Through-ball xG": pct(season.xg_through),
        "Cross xG": pct(season.xg_cross),
        "Deep completions": pct(season.deep),
        "Deep completions allowed (inv.)": pct(season.deep_allowed, invert=True),
        "Pressing intensity (1/PPDA)": pct(season.ppda, invert=True),
    })
    table = tm.groupby("team")[["points", "xpts"]].sum()

    cs = shots[shots.match_id.isin(c.match_id)]
    figs = {
        "bars": _img(charts.match_bars(c, TEAM), "경기별 xG"),
        "map": _img(charts.shotmap_pair(cs[cs.team == TEAM], cs[cs.team != TEAM],
                                        (f"Chelsea shots ({n} games)", f"Shots conceded ({n} games)"),
                                        "Chelsea 2026/27 shot maps (red = goal, size = xG, own goals excluded)"), "슈팅 지도"),
        "prof": _img(charts.percentile_profile(prof, f"{opponent} — league percentile after {n} games", STYLE),
                     "상대 프로필"),
        "table": _img(charts.points_vs_xpts(table, (TEAM, opponent)), "승점과 기대 승점"),
        "late": _img(charts.late_fade(late_curve, lg_curve, TEAM), "시간대별 허용 xG"),
        **{f"fp_{t}": _img(charts.fingerprint_dumbbell(
            r["now"], r["before"], FP.METRICS_EN, f"{t} 2026/27 ({n} games)", MANAGERS[t][1].replace("RasenBallsport ", "RB ") + " 2023/24",
            f"{t}: this season vs {MANAGERS[t][0] and MANAGERS[t][1].replace('RasenBallsport ', 'RB ')} 2023/24"),
            f"{t} 감독 지문") for t, r in fps.items()},
    }
    sh_rows = "".join(f"<tr><td>{html.escape(p)}</td><td>{int(r.shots)}</td><td>{r.xg:.1f}</td><td>{int(r.goals)}</td></tr>"
                      for p, r in rep.key_players["shooters"].iterrows())
    cr_rows = "".join(f"<tr><td>{html.escape(p)}</td><td>{int(r.chances)}</td><td>{r.xa:.1f}</td></tr>"
                      for p, r in rep.key_players["creators"].iterrows())
    plan_html = "".join(f"<li>{html.escape(p)}</li>" for p in plan)
    luck = o_xpts - o_pts
    fx = NEXT_FIXTURE if NEXT_FIXTURE["opponent"] == opponent else None
    fixture_txt = (f"{fx['date'][5:7].lstrip('0')}월 {fx['date'][8:].lstrip('0')}일 {'홈' if fx['venue'] == 'H' else '원정'} 경기"
                   if fx else "가상 매치업")

    def _fp_block(t):
        r = fps[t]
        mgr, _, ref = MANAGERS[t]
        gaps = "".join(f"<li>{html.escape(k)}: 백분위 {r['now'][kk]:.0%} ← 기준 {r['before'][kk]:.0%}</li>"
                       for k, d in r["gaps"][:3] for kk in [next(x for x, lab in FP.METRICS.items() if lab == k)])
        matched = ", ".join(r["matched"]) or "아직 없음"
        return (f"<h3 class='col-h'>{html.escape(t)} · {mgr} (기준: {ref})</h3>"
                f"<figure>{figs['fp_' + t]}</figure>"
                f"<div class='finding-grid'><article class='finding'><h3>이미 닮은 부분</h3><p>{html.escape(matched)}</p></article>"
                f"<article class='finding'><h3>가장 큰 격차</h3><ul style='margin:0;padding-left:1.1em'>{gaps}</ul></article>"
                f"<article class='finding'><h3>읽는 법</h3><p>{FP_NOTES.get(t, '')}</p></article></div>")

    fp_html = "".join(_fp_block(t) for t in fps)

    page = f"""<meta charset="utf-8">
<title>Chelsea 2026/27 Live</title>
<meta name="description" content="Understat 슈팅 xG로 만든 첼시 2026/27 시즌 진단과 다음 상대 공략 플랜">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+KR:wght@400;500;700&display=swap">
<style>{CSS}
.badge{{display:inline-block;font-family:var(--mono);font-size:12px;letter-spacing:.06em;border:1px solid var(--line);
background:var(--surface);padding:3px 10px;border-radius:999px;color:var(--muted);margin-top:14px}}</style>
<div class="wrap">
<header class="mast">
  <p class="kicker">Match-prep dossier · Premier League 2026/27 · Understat data</p>
  <h1>Chelsea <span>2026/27</span></h1>
  <p class="sub">{n}경기 만에 승점 {k['pts']}점, 득실 {k['gf']}-{k['ga']}. 출발은 좋았는데 왜 흔들리는지 진단하고,
  다음 상대 {html.escape(opponent)}({fixture_txt})를 이길 플랜을 만든다.</p>
  <span class="badge">{as_of} 기준 · 팀당 {n}경기 표본</span>
  <div class="summary">
    <div><b>{k['pts']} / {k['xpts']:.1f}</b><span>승점 / 기대 승점</span></div>
    <div><b>{k['gf']} / {k['xg']:.1f}</b><span>득점 / xG</span></div>
    <div><b>{k['ga']} / {k['xga']:.1f}</b><span>실점 / 허용 xG</span></div>
  </div>
  <nav class="toc" aria-label="목차"><a href="#part1">Part 1 · 첼시 진단</a>
  <a href="#part2">Part 2 · {html.escape(opponent)} 공략 플랜</a><a href="#fingerprint">감독 지문</a><a href="#league">리그 전체</a><a href="#method">방법과 한계</a></nav>
</header>

<section id="part1" class="part">
  <p class="eyebrow">Part 1 · 진단</p>
  <h2>좋은 출발, 흔들리는 3경기</h2>
  <p class="lede">첫 2경기 평균 xG {k['f_xg']:.2f} 대 허용 {k['f_xga']:.2f}로 경기를 지배했지만,
  이후 3경기는 {k['l_xg']:.2f} 대 {k['l_xga']:.2f}로 뒤집혔다. 결과보다 경기 내용이 먼저 나빠졌다는 신호다.</p>
  <figure>{figs['bars']}</figure>
  <div class="finding-grid">
    <article class="finding"><h3>세트피스 실점 위험</h3>
      <p>경기당 세트피스 허용 xG <b>{k['sp_pg']:.2f}</b>로 리그 평균({k['lg_sp_pg']:.2f})보다 높다.
      특히 {k['sp_worst'][0][0]}전 {k['sp_worst'][0][1]:.2f}, {k['sp_worst'][1][0]}전 {k['sp_worst'][1][1]:.2f}로
      두 경기에서 집중됐다.</p></article>
    <article class="finding"><h3>수비 라인 앞 공간</h3>
      <p>상대의 딥 컴플리션(골문 근처로 들어온 패스)이 첫 2경기 평균 {k['deep_allowed_first']:.1f}회에서
      최근 3경기 {k['deep_allowed_last']:.1f}회로 늘었다. 가장 많았던 경기는 {k['deep_worst'][0]}전 {k['deep_worst'][1]}회.</p></article>
    <article class="finding"><h3>득점은 두 명에게</h3>
      <p>팀 xG 1위 {html.escape(own.key_players['shooters'].index[0])}(xG {own.key_players['shooters'].xg.iloc[0]:.1f}),
      찬스 창출 1위 {html.escape(own.key_players['creators'].index[0])}(xA {own.key_players['creators'].xa.iloc[0]:.1f}).
      두 선수가 막히면 공격이 급격히 줄어든다.</p></article>
  </div>
  <figure>{figs['map']}</figure>
  <figure>{figs['late']}
  <figcaption>동점 상황에서 첼시가 허용한 xG의 시간대별 비중. 60분 이후 막대가 리그 평균보다 크게 솟는다
  (표본이 5경기라 한두 장면의 영향이 크다는 점은 감안).</figcaption></figure>
  <h3 class="col-h threat">상대가 노릴 첼시의 약점과 우리 대응</h3>
  <ul class="findings">{_findings(own.weaknesses, "threat", own_view=True)}</ul>
</section>

<section id="part2" class="part">
  <p class="eyebrow">Part 2 · 경기 준비</p>
  <h2>{html.escape(opponent)} 공략 플랜</h2>
  <p class="lede">{n}경기 승점 {o_pts}점으로 순위표에서는 하위권이지만, 기대 승점은 <b>{o_xpts:.1f}</b>점.
  {"운이 따르지 않았을 뿐 경기력은 중위권 이상이다. 순위만 보고 방심하면 안 되는 상대." if luck >= 2
   else "기대 승점과 실제 승점이 비슷해, 순위가 실력을 잘 반영하고 있다."}</p>
  <h3 class="col-h">매치업 교차 분석 · 이 경기의 승부처</h3>
  <ul class="findings">{"".join(
      f'<li class="{"threat" if f.kind == "risk" else "weak"}"><h4>{html.escape(f.title)}</h4>'
      f'<p class="ev">{html.escape(f.evidence)}</p><p class="act">→ {html.escape(f.action)}</p></li>' for f in mu)}</ul>
  <div class="plan">
    <p class="plan-label">Game plan · 자동 생성</p>
    <ol>{plan_html}</ol>
  </div>
  <figure>{figs['prof']}</figure>
  <div class="cols">
    <div><h3 class="col-h weak">약점 · 공략할 곳</h3><ul class="findings">{_findings(rep.weaknesses, "weak")}</ul></div>
    <div><h3 class="col-h threat">위협 · 막아야 할 것</h3><ul class="findings">{_findings(rep.threats, "threat") or
      '<li class="style"><p class="ev">리그 상위 20%에 드는 뚜렷한 공격 무기는 아직 없음 (5경기 기준).</p></li>'}</ul></div>
  </div>
  <h3 class="col-h">운영 스타일</h3><ul class="findings">{_findings(rep.style, "style")}</ul>
  <div class="tables">
    <div class="tbl"><table><caption>득점원 (슈팅 xG)</caption>
      <thead><tr><th>선수</th><th>슈팅</th><th>xG</th><th>골</th></tr></thead><tbody>{sh_rows}</tbody></table></div>
    <div class="tbl"><table><caption>창조자 (키패스 xA)</caption>
      <thead><tr><th>선수</th><th>찬스</th><th>xA</th></tr></thead><tbody>{cr_rows}</tbody></table></div>
  </div>
</section>


<section id="fingerprint" class="part">
  <p class="eyebrow">Manager fingerprint</p>
  <h2>감독의 축구가 얼마나 자리 잡았나</h2>
  <p class="lede">두 팀 모두 이번 시즌 감독이 바뀌어 지난 시즌 기록은 기준이 되지 못한다. 대신 감독이 이전 팀에서 보여 준 축구를 기준으로 삼았다.
  리그마다 수준이 달라 숫자를 직접 비교하지 않고, 각자 리그 안에서의 백분위로 바꿔 비교했다(같은 Understat 정의).</p>
  {fp_html}
  <p class="method">주의: 기준 팀(레버쿠젠, 라이프치히)은 당시 리그 최상위권 전력이었다. 지금 팀이 모든 지표에서 뒤처지는 건
  전술보다 전력 차이일 수 있으므로, 격차의 <b>크기 순서</b>에 주목할 것. 이번 시즌은 {n}경기 표본.</p>
</section>

<section id="league" class="part">
  <p class="eyebrow">League</p>
  <h2>순위표가 말해 주지 않는 것</h2>
  <p class="lede">파란 점이 실제 승점, 금색 점이 기대 승점. 선이 길수록 운의 영향이 컸다는 뜻이다.</p>
  <figure>{figs['table']}</figure>
</section>

<section id="method" class="part method">
  <p class="eyebrow">Method</p>
  <h2>방법과 한계</h2>
  <ul>
    <li><b>데이터</b>: Understat의 2026/27 프리미어리그 {tm.match_id.nunique()}경기 슈팅 {len(shots)}개(xG, 위치, 상황, 직전 동작)와
    경기별 PPDA·딥 컴플리션·기대 승점.</li>
    <li><b>분석 규칙</b>: 2015/16 StatsBomb 버전과 같은 스카우팅 규칙을 그대로 사용. Understat에 없는 지표(패스 점유율, 역습,
    어시스트 위치)가 필요한 규칙은 빠지고, 대신 크로스·스루패스·딥 컴플리션 규칙을 추가.</li>
    <li><b>표본</b>: 팀당 {n}경기라 백분위가 크게 흔들린다. 플랜은 방향 제시용이며, 경기가 쌓일수록 신뢰도가 올라간다.</li>
    <li><b>이용 조건</b>: Understat 데이터는 개인·연구 목적으로만 사용. 유료 서비스에는 라이선스 데이터가 필요하다.</li>
  </ul>
</section>
<footer>데이터: <a href="https://understat.com/league/EPL/2026" target="_blank" rel="noopener">Understat</a>.
분석·코드: chelsea-analytics.</footer>
</div>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default=NEXT_FIXTURE["opponent"])
    ap.add_argument("--fetch", action="store_true", help="download fresh data from Understat")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "chelsea_2026_27_live.html")
    a = ap.parse_args()
    tm, shots = U.load_live() if a.fetch else U.load_snapshot()
    as_of = tm.date.max().strftime("%Y-%m-%d")
    print(build(tm, shots, a.opponent, as_of, a.out))


if __name__ == "__main__":
    main()
