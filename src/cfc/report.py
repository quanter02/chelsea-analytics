"""Build the analyst dossier (self-contained HTML with embedded charts).

    python -m cfc.report                       # Chelsea 2015/16 + Leicester game plan
    python -m cfc.report --opponent "Arsenal"  # same dossier, different opponent
"""
from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path

import numpy as np
import pandas as pd

from . import charts, metrics as M, scouting as S

ROOT = Path(__file__).resolve().parents[2]
TEAM = "Chelsea"
SHORT = {"Eden Hazard": "Hazard", "Pedro Eliezer Rodríguez Ledesma": "Pedro", "Willian Borges da Silva": "Willian",
         "Oscar dos Santos Emboaba Júnior": "Oscar", "Nemanja Matić": "Matić", "Branislav Ivanović": "Ivanović",
         "Diego da Silva Costa": "Diego Costa", "Francesc Fàbregas i Soler": "Fàbregas"}
STYLE = {"Possession (pass share)", "Pressing intensity (1/PPDA)", "Long-ball share"}
BUCKETS = [0, 15, 30, 45, 60, 75, 200]
BUCKET_LABELS = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]


def _img(png: bytes, alt: str) -> str:
    b64 = base64.b64encode(png).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}">'


def _era(manager: str) -> str:
    return "Mourinho" if "Mourinho" in manager else "Hiddink"


def load():
    tm = M.season_team_matches()
    shots = M.season_shots()
    xp_path = M.CACHE_DIR / "xpts.csv"
    if xp_path.exists():
        xp = pd.read_csv(xp_path)
    else:
        xp = M.expected_points(shots, tm)
        xp.to_csv(xp_path, index=False)
    return tm.merge(xp, on=["match_id", "team"]), shots


# ---------------------------------------------------------------- Part 1
def chelsea_section(tm: pd.DataFrame, shots: pd.DataFrame) -> tuple[str, dict]:
    c = tm[tm.team == TEAM].sort_values("date").reset_index(drop=True)
    c["era"] = c.manager.map(_era)
    n_mou = int((c.era == "Mourinho").sum())
    mou, hid = c[c.era == "Mourinho"], c[c.era == "Hiddink"]

    k = {
        "n_mou": n_mou,
        "mou_pts": int(mou.points.sum()), "mou_xpts": mou.xpts.sum(),
        "mou_ga": int(mou.goals_against.sum()), "mou_xga": mou.xga.sum(),
        "mou_gf": int(mou.goals.sum()), "mou_xg": mou.xg.sum(),
        "hid_ppg": hid.points.mean(), "mou_ppg": mou.points.mean(),
        "hid_pts": int(hid.points.sum()), "hid_xpts": hid.xpts.sum(),
        "mou_open": mou.xg_open.mean(), "hid_open": hid.xg_open.mean(), "lg_open": tm.xg_open.mean(),
        "mou_tilt": mou.field_tilt.mean(), "hid_tilt": hid.field_tilt.mean(),
        "mou_xppg": mou.xpts.mean(),
    }

    sa = shots.merge(c[["match_id", "era"]], on="match_id")
    sa = sa[sa.team != TEAM]
    ot = sa.groupby("era").agg(n=("on_target", "size"), ot=("on_target", "sum"), g=("goal", "sum"))
    k["mou_ot_rate"] = ot.loc["Mourinho", "ot"] / ot.loc["Mourinho", "n"]
    k["hid_ot_rate"] = ot.loc["Hiddink", "ot"] / ot.loc["Hiddink", "n"]
    k["mou_save"] = 1 - ot.loc["Mourinho", "g"] / ot.loc["Mourinho", "ot"]
    k["hid_save"] = 1 - ot.loc["Hiddink", "g"] / ot.loc["Hiddink", "ot"]

    era_tbl = pd.DataFrame({
        "Mourinho": mou[["npxg", "npxga", "xg_open", "field_tilt", "box_entries"]].mean(),
        "Hiddink": hid[["npxg", "npxga", "xg_open", "field_tilt", "box_entries"]].mean(),
        "League avg": tm[["npxg", "npxga", "xg_open", "field_tilt", "box_entries"]].mean(),
    })
    era_tbl.index = ["npxG / game", "npxG against", "Open-play xG", "Field tilt", "Box entries"]

    fs = shots.merge(c[["match_id", "era"]], on="match_id")
    fs = fs[(fs.team == TEAM) & (fs.shot_type != "Penalty")]
    fin = fs.groupby(["player", "era"]).agg(n=("xg", "size"), xg=("xg", "sum"), g=("goal", "sum"))
    fin["diff"] = fin.g - fin.xg
    piv = fin["diff"].unstack()
    cnt = fin["n"].unstack().fillna(0)
    keep = cnt[(cnt.Mourinho >= 10) & (cnt.Hiddink >= 10)].index
    piv = piv.loc[keep].fillna(0).sort_values("Hiddink", ascending=False)
    piv.index = [SHORT.get(p, p) for p in piv.index]
    haz = fin.loc["Eden Hazard"]
    k["haz_mou"] = (int(haz.loc["Mourinho", "g"]), haz.loc["Mourinho", "xg"])
    k["haz_hid"] = (int(haz.loc["Hiddink", "g"]), haz.loc["Hiddink", "xg"])

    figs = [
        (charts.cumulative_points(c, n_mou), "누적 승점과 기대 승점"),
        (charts.rolling_xg(c, n_mou), "6경기 이동평균 xG"),
        (charts.era_bars(era_tbl), "감독별 핵심 지표"),
        (charts.shotmap_pair(sa[sa.era == "Mourinho"], sa[sa.era == "Hiddink"],
                             ("Under Mourinho (16 games)", "Under Holland/Hiddink (22 games)"),
                             "Shots Chelsea conceded (red = goal, size = xG)"), "허용 슈팅 지도"),
        (charts.player_finishing(piv), "선수별 결정력"),
    ]
    body = f"""
<section id="part1" class="part">
  <p class="eyebrow">Part 1 · 사후 분석</p>
  <h2>무리뉴의 마지막 {n_mou}경기: 정말 그렇게 못했나?</h2>
  <p class="lede">무리뉴 체제 {n_mou}경기에서 첼시는 승점 <b>{k['mou_pts']}</b>점(경기당 {k['mou_ppg']:.2f})에 그쳤다.
  하지만 슈팅의 질로 계산한 기대 승점은 <b>{k['mou_xpts']:.1f}</b>점. 약 {k['mou_xpts'] - k['mou_pts']:.0f}점을 '잃어버린' 셈이다.</p>
  <figure>{_img(*figs[0])}</figure>
  <div class="finding-grid">
    <article class="finding"><h3>수비는 평균, 실점은 최악</h3>
      <p>허용 xG는 {k['mou_xga']:.1f}로 리그 평균 수준이었지만 실점은 <b>{k['mou_ga']}</b>골. 상대 슈팅의 유효슈팅 비율이
      {k['mou_ot_rate']:.0%}로, 히딩크 체제({k['hid_ot_rate']:.0%})보다 훨씬 높았고 선방률은 {k['mou_save']:.0%} 대 {k['hid_save']:.0%}.
      운과 골키퍼 변수(쿠르투아 부상 공백기와 겹침)가 크게 작용했다.</p></article>
    <article class="finding"><h3>진짜 문제는 공격</h3>
      <p>필드 틸트(최종 3분의 1 지역 패스 점유)는 {k['mou_tilt']:.0%}로 높았는데, 오픈 플레이 xG는 경기당 {k['mou_open']:.2f}로
      리그 평균({k['lg_open']:.2f})보다 낮았다. <b>공은 상대 진영에 있었지만 좋은 슈팅으로 이어지지 않은 '무딘 점유'</b>.
      히딩크 체제에서는 {k['hid_open']:.2f}로 개선.</p></article>
    <article class="finding"><h3>아자르의 증발과 복귀</h3>
      <p>아자르는 무리뉴 체제에서 기대득점 {k['haz_mou'][1]:.1f}에 <b>{k['haz_mou'][0]}골</b>,
      이후 {k['haz_hid'][1]:.1f} 기대득점에 {k['haz_hid'][0]}골. 핵심 선수들의 결정력이 동시에 식은 것이 추락을 증폭시켰다.</p></article>
  </div>
  <figure>{_img(*figs[1])}</figure>
  <figure>{_img(*figs[2])}</figure>
  <figure>{_img(*figs[3])}</figure>
  <figure>{_img(*figs[4])}</figure>
  <p class="verdict"><b>결론.</b> 기대 승점 기준 무리뉴의 첼시는 경기당 {k['mou_xppg']:.2f}점, 즉 강등권이 아니라 <b>중위권 팀</b>이었다.
  순위표가 보여 준 것보다는 나았지만, 우승을 다툴 공격력은 아니었다. 감독 교체의 효과는 전술보다 결정력·선방의 평균 회귀가 더 컸을 가능성이 높다.</p>
</section>"""
    return body, k


# ---------------------------------------------------------------- Part 2
def opponent_section(opp: str, tm: pd.DataFrame, shots: pd.DataFrame) -> str:
    rep = S.scout(opp, tm, shots)
    plan = S.game_plan(rep)
    season = tm.groupby("team").mean(numeric_only=True)

    def pct(s, invert=False):
        r = s.rank(pct=True)[opp]
        return 1 - r + 1 / len(s) if invert else r

    prof = pd.Series({
        "Non-penalty xG": pct(season.npxg),
        "Non-penalty xG against (inv.)": pct(season.npxga, invert=True),
        "Counter-attack xG": pct(season.xg_counter),
        "Set-piece xG": pct(season.xg_setpiece),
        "Set-piece xG against (inv.)": pct(season.xga_setpiece, invert=True),
        "Possession (pass share)": pct(season.pass_share),
        "Pressing intensity (1/PPDA)": pct(season.ppda, invert=True),
        "High turnovers": pct(season.high_turnovers),
        "Long-ball share": pct(season.long_ball_share),
    })

    sa = S._shots_against(shots, tm, opp)
    lvl = sa[sa.state == "level"].copy()
    lvl["b"] = pd.cut(lvl.minute, BUCKETS, right=False, labels=BUCKET_LABELS)
    team_curve = lvl.groupby("b", observed=False).xg.sum() / lvl.xg.sum()
    all_sa = shots[shots.state == "level"].copy()
    all_sa["b"] = pd.cut(all_sa.minute, BUCKETS, right=False, labels=BUCKET_LABELS)
    league_curve = all_sa.groupby("b", observed=False).xg.sum() / all_sa.xg.sum()

    figs = [
        (charts.percentile_profile(prof, f"{opp} — league percentile (higher = stronger / more)", STYLE),
         "상대 프로필"),
        (charts.late_fade(team_curve, league_curve, opp), "시간대별 허용 xG"),
    ]
    h2h = tm[(tm.team == TEAM) & (tm.opponent == opp)].sort_values("date")
    ch = tm[tm.team == TEAM].sort_values("date")
    last_mou = ch[ch.manager.str.contains("Mourinho")].match_id.iloc[-1]
    race = ""
    if len(h2h):
        first = h2h.iloc[0]
        ms = M.possession_xg(shots[shots.match_id == first.match_id])
        venue_home = first.venue == "H"
        home, away = (TEAM, opp) if venue_home else (opp, TEAM)
        score = f"{first.goals}-{first.goals_against}" if venue_home else f"{first.goals_against}-{first.goals}"
        race = f"""<figure>{_img(charts.match_xg_race(ms, home, away,
                     f"{home} {score} {away} · {first.date:%d %b %Y}"), "맞대결 xG 흐름")}
        <figcaption>{first.date:%Y년 %m월 %d일} 맞대결{" (무리뉴의 마지막 경기)" if first.match_id == last_mou else ""}의
        누적 xG 흐름. 같은 공격 안의 리바운드 슈팅은 하나의 찬스로 합산, 점은 득점.</figcaption></figure>"""

    def finding_list(items, cls):
        return "".join(
            f'<li class="{cls}"><h4>{html.escape(f.title)}</h4><p class="ev">{html.escape(f.evidence)}</p>'
            f'<p class="act">→ {html.escape(f.action)}</p></li>' for f in items)

    shooters = rep.key_players["shooters"]
    creators = rep.key_players["creators"]
    kp_rows = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{int(r.shots)}</td><td>{r.xg:.1f}</td><td>{int(r.goals)}</td></tr>"
        for n, r in shooters.iterrows())
    cr_rows = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{int(r.chances)}</td><td>{r.xa:.1f}</td></tr>" for n, r in creators.iterrows())
    r = rep.record
    plan_html = "".join(f"<li>{html.escape(p)}</li>" for p in plan)
    return f"""
<section id="part2" class="part">
  <p class="eyebrow">Part 2 · 경기 준비</p>
  <h2>{html.escape(opp)} 공략 플랜</h2>
  <p class="lede">시즌 {r['matches']}경기 {r['points']}점, 득실 {r['gf']}-{r['ga']} (xG {r['xg']:.1f} / xGA {r['xga']:.1f}).
  리그 20개 팀 대비 백분위로 강점과 약점을 찾고, 규칙 기반으로 경기 플랜을 생성했다.</p>
  {race}
  <div class="plan">
    <p class="plan-label">Game plan · 자동 생성</p>
    <ol>{plan_html}</ol>
  </div>
  <figure>{_img(*figs[0])}</figure>
  <div class="cols">
    <div><h3 class="col-h weak">약점 · 공략할 곳</h3><ul class="findings">{finding_list(rep.weaknesses, "weak")}</ul></div>
    <div><h3 class="col-h threat">위협 · 막아야 할 것</h3><ul class="findings">{finding_list(rep.threats, "threat")}</ul></div>
  </div>
  <h3 class="col-h">운영 스타일</h3><ul class="findings">{finding_list(rep.style, "style")}</ul>
  <figure>{_img(*figs[1])}</figure>
  <div class="tables">
    <div class="tbl"><table><caption>득점원 (슈팅 xG)</caption>
      <thead><tr><th>선수</th><th>슈팅</th><th>xG</th><th>골</th></tr></thead><tbody>{kp_rows}</tbody></table></div>
    <div class="tbl"><table><caption>창조자 (키패스 xA)</caption>
      <thead><tr><th>선수</th><th>찬스</th><th>xA</th></tr></thead><tbody>{cr_rows}</tbody></table></div>
  </div>
</section>"""


CSS = """
:root{--bg:#F3F5F9;--surface:#FFFFFF;--ink:#16213A;--muted:#5E6780;--line:#DCE1EA;--blue:#1C4FB8;
--gold:#A8812A;--red:#B83A34;--green:#2B7A55;--plate:#FFFFFF;
--display:"Barlow Condensed","Arial Narrow",sans-serif;--body:"IBM Plex Sans KR","Apple SD Gothic Neo","Malgun Gothic",sans-serif;
--mono:"IBM Plex Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0D1424;--surface:#152036;--ink:#E7ECF6;
--muted:#9AA6BF;--line:#26324B;--blue:#7FA2F2;--gold:#D9B45E;--red:#E57A73;--green:#6CC59A;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0D1424;--surface:#152036;--ink:#E7ECF6;--muted:#9AA6BF;--line:#26324B;--blue:#7FA2F2;
--gold:#D9B45E;--red:#E57A73;--green:#6CC59A;color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--body);font-size:15.5px;line-height:1.7;margin:0}
.wrap{max-width:880px;margin:0 auto;padding-inline:20px;padding-block:40px 64px}
header.mast{border-bottom:3px solid var(--blue);padding-bottom:22px;margin-bottom:8px}
.kicker{font-family:var(--mono);font-size:12px;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin:0 0 6px}
h1{font-family:var(--display);font-weight:700;font-size:clamp(40px,8vw,64px);line-height:.95;margin:0;letter-spacing:.01em;text-transform:uppercase}
h1 span{color:var(--blue)}
.sub{color:var(--muted);margin:14px 0 0;max-width:62ch}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1px;background:var(--line);
border:1px solid var(--line);margin:28px 0 8px}
.summary div{background:var(--surface);padding:16px 18px}
.summary b{display:block;font-family:var(--display);font-size:34px;line-height:1;color:var(--blue);font-variant-numeric:tabular-nums}
.summary span{font-size:13px;color:var(--muted)}
nav.toc{display:flex;flex-wrap:wrap;gap:8px 20px;font-size:14px;margin:18px 0 0}
nav.toc a{color:var(--blue);text-decoration:none;border-bottom:1px solid transparent}
nav.toc a:hover,nav.toc a:focus-visible{border-bottom-color:var(--blue)}
.part{padding-top:40px;margin-top:36px;border-top:1px solid var(--line)}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--gold);margin:0}
h2{font-family:var(--display);font-size:clamp(28px,5vw,38px);line-height:1.1;margin:6px 0 12px;text-wrap:balance;font-weight:700}
h3{font-size:16px;margin:0 0 6px}
.lede{font-size:17px;max-width:65ch}
figure{margin:26px 0;background:var(--plate);border:1px solid var(--line);border-radius:6px;padding:10px;overflow-x:auto}
figure img{display:block;width:100%;height:auto}
figcaption{font-size:13px;color:#5E6780;padding:6px 4px 0}
.finding-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px;margin:8px 0}
.finding{background:var(--surface);border:1px solid var(--line);padding:16px 18px;border-radius:6px}
.finding p{margin:0;font-size:14.5px}
.verdict{background:var(--surface);border-left:4px solid var(--gold);padding:16px 20px;margin:28px 0 0}
.plan{background:var(--blue);color:#fff;padding:20px 24px;border-radius:6px;margin:24px 0}
:root[data-theme="dark"] .plan{color:#0D1424}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .plan{color:#0D1424}}
.plan-label{font-family:var(--mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;margin:0 0 8px;opacity:.85}
.plan ol{margin:0;padding-left:1.3em;display:grid;gap:8px}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:24px;margin-top:8px}
.col-h{font-family:var(--display);font-size:20px;text-transform:uppercase;letter-spacing:.03em;margin:18px 0 8px}
.col-h.weak{color:var(--green)}.col-h.threat{color:var(--red)}
ul.findings{list-style:none;padding:0;margin:0;display:grid;gap:10px}
ul.findings li{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px 16px}
ul.findings h4{margin:0 0 4px;font-size:15px}
ul.findings .ev{margin:0;font-size:13.5px;color:var(--muted);font-variant-numeric:tabular-nums}
ul.findings .act{margin:6px 0 0;font-size:14px}
li.weak{border-left:3px solid var(--green)}li.threat{border-left:3px solid var(--red)}li.style{border-left:3px solid var(--gold)}
.tables{margin-top:28px;display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px}
.tbl{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums;background:var(--surface)}
caption{text-align:left;font-weight:700;padding:0 0 6px}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
th{font-family:var(--mono);font-size:11.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);font-weight:500}
.method{font-size:14px;color:var(--muted)}
.method h2{color:var(--ink)}
.method li{margin-bottom:6px}
footer{margin-top:40px;font-size:13px;color:var(--muted);border-top:1px solid var(--line);padding-top:16px}
footer a{color:var(--blue)}
"""


def build(opponent: str = "Leicester City", out: Path | None = None) -> Path:
    tm, shots = load()
    p1, k = chelsea_section(tm, shots)
    p2 = opponent_section(opponent, tm, shots)
    out = out or ROOT / "reports" / "chelsea_2015_16_dossier.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    page = f"""<meta charset="utf-8">
<title>Chelsea 2015/16 Dossier</title>
<meta name="description" content="StatsBomb 오픈 데이터로 만든 첼시 2015/16 사후 분석과 상대팀 공략 플랜">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+KR:wght@400;500;700&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header class="mast">
  <p class="kicker">Analyst dossier · Premier League 2015/16 · StatsBomb open data</p>
  <h1>Chelsea <span>2015/16</span></h1>
  <p class="sub">디펜딩 챔피언이 10위로 추락한 시즌. 무리뉴 경질이 정답이었는지 데이터로 되짚고,
  같은 도구로 우승팀 {html.escape(opponent)}를 이길 경기 플랜을 만든다.</p>
  <div class="summary">
    <div><b>{k['mou_pts']} / {k['mou_xpts']:.1f}</b><span>무리뉴 {k['n_mou']}경기 실제 승점 / 기대 승점</span></div>
    <div><b>{k['mou_ga']} / {k['mou_xga']:.1f}</b><span>같은 기간 실점 / 허용 xG</span></div>
    <div><b>{k['hid_pts']} / {k['hid_xpts']:.1f}</b><span>교체 후 22경기 실제 승점 / 기대 승점</span></div>
  </div>
  <nav class="toc" aria-label="목차"><a href="#part1">Part 1 · 무리뉴 사후 분석</a>
  <a href="#part2">Part 2 · {html.escape(opponent)} 공략 플랜</a><a href="#method">방법과 한계</a></nav>
</header>
{p1}
{p2}
<section id="method" class="part method">
  <p class="eyebrow">Method</p>
  <h2>방법과 한계</h2>
  <ul>
    <li><b>데이터</b>: StatsBomb 오픈 데이터, 프리미어리그 2015/16 전 380경기 이벤트. xG는 StatsBomb 모델 값.</li>
    <li><b>기대 승점(xPts)</b>: 같은 공격(포제션) 안의 슈팅은 1-∏(1-xG)로 합쳐 리바운드 중복을 없앤 뒤,
    1만 번 시뮬레이션해 승·무·패 확률을 계산. 자책골은 제외.</li>
    <li><b>PPDA</b>: 상대가 자기 진영 60% 구역에서 한 패스 수 ÷ 우리 팀의 같은 구역 태클·인터셉트·파울. 낮을수록 강한 압박.</li>
    <li><b>필드 틸트</b>: 양 팀의 최종 3분의 1 지역 패스 중 우리 팀 비중.</li>
    <li><b>후반 흔들림</b>: 점수 상황(리드/동점/열세)에 따른 착시를 줄이려고 동점 상황의 슈팅만 사용.</li>
    <li><b>한계</b>: 한 시즌, 한 리그의 표본이며 경기 영상·트래킹 데이터 없이 이벤트 데이터만 사용. 플랜 문장은 규칙 기반이라
    코칭스태프의 영상 검증이 전제되어야 한다.</li>
  </ul>
</section>
<footer>데이터: <a href="https://github.com/statsbomb/open-data" target="_blank" rel="noopener">StatsBomb Open Data</a>
(비상업적 사용). 분석·코드: chelsea-analytics.</footer>
</div>"""
    out.write_text(page, encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="Leicester City")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    print(build(a.opponent, a.out))


if __name__ == "__main__":
    main()
