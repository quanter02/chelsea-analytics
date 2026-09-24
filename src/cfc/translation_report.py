"""Recruitment dossier: league translation factors + EPL-ready shortlist.

    python -m cfc.translation_report
"""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from . import charts, translation as T
from .report import CSS, _img

ROOT = Path(__file__).resolve().parents[2]
ELITE = {"Real Madrid", "Barcelona", "Atletico Madrid", "Bayern Munich", "Borussia Dortmund", "Bayer Leverkusen",
         "Paris Saint Germain", "Inter", "Juventus", "AC Milan", "Napoli"}
ASIA = {"Lee Kang-In": "이강인", "Jae-Sung Lee": "이재성", "Takefusa Kubo": "구보 다케후사", "Yuito Suzuki": "스즈키 유이토"}


def build(out: Path) -> Path:
    mv, st = T.load_movers(), T.load_stayers()
    models = T.fit_all(mv, st)
    rows = {}
    for m in ("npxG", "xA"):
        bs = T.bootstrap(mv, st, m, n=400)
        for lg in T.FREE:
            f = float(np.exp(-models[m]["beta"][lg]))
            lo, hi = np.exp(-np.percentile(bs[lg], [95, 5]))
            rows.setdefault(T.LEAGUES[lg], {}).update({m: f, f"{m}_lo": lo, f"{m}_hi": hi})
    tbl = pd.DataFrame(rows).T.sort_values("npxG", ascending=False)
    cv = {m: T.cross_validate(mv, st, m) for m in ("npxG", "xA")}
    sl = T.shortlist(models, T.load_candidates(), T.load_epl())
    value = sl[~sl.team.isin(ELITE)].head(12)
    asia = sl[sl.player.isin(ASIA)]
    n_to_epl = int((mv.to_lg == "E").sum())

    def trs(df):
        return "".join(
            f"<tr><td>{html.escape(r.player)}</td><td>{html.escape(r.team)}</td><td>{r.league_name}</td>"
            f"<td>{r.minutes:,}</td><td>{r.npxG:.2f} → <b>{r.npxG_epl:.2f}</b></td><td>{r.xA:.2f} → <b>{r.xA_epl:.2f}</b></td>"
            f"<td>{r.epl_percentile:.0%}</td></tr>" for r in df.itertuples())

    asia_cards = "".join(
        f'<article class="finding"><h3>{ASIA[r.player]} <span class="muted">{html.escape(r.player)}</span></h3>'
        f"<p>{html.escape(r.team)} · {r.league_name}, {r.minutes:,}분<br>"
        f"EPL 환산 npxG {r.npxG_epl:.2f} + xA {r.xA_epl:.2f} /90 → EPL 공격 자원 중 <b>상위 {1 - r.epl_percentile:.0%}</b> 수준"
        f"</p></article>" for r in asia.itertuples())
    cvrow = lambda m: (f"<tr><td>{m} /90</td><td>{cv[m]['same']:.3f}</td><td>{cv[m]['no_lg']:.3f}</td>"
                       f"<td><b>{cv[m]['model']:.3f}</b></td><td>{1 - cv[m]['model'] / cv[m]['same']:.0%}</td></tr>")

    page = f"""<meta charset="utf-8">
<title>League Translation Lab</title>
<meta name="description" content="다른 리그 선수가 EPL에서 통할지 xG로 환산하는 모델과 영입 후보 리스트">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+KR:wght@400;500;700&display=swap">
<style>{CSS}
.muted{{color:var(--muted);font-weight:400;font-size:13px}}
.tbl table td:nth-child(n+4),.tbl table th:nth-child(n+4){{white-space:nowrap}}
.src li{{margin-bottom:8px}}</style>
<div class="wrap">
<header class="mast">
  <p class="kicker">Recruitment lab · Understat 2014/15–2025/26 · 6 leagues</p>
  <h1>League <span>Translation</span></h1>
  <p class="sub">다른 리그에서 잘하는 선수가 프리미어리그에 오면 얼마나 남을까? 리그를 옮긴 선수들의 기록으로
  리그별 환산 계수를 만들고, 그 계수로 지금 EPL에서 통할 선수를 찾는다.</p>
  <div class="summary">
    <div><b>×{tbl.npxG.min():.2f}–{tbl.npxG.max():.2f}</b><span>EPL로 올 때 비페널티 xG/90 환산 계수</span></div>
    <div><b>{len(mv)}명</b><span>리그를 옮긴 선수 표본 (그중 EPL행 {n_to_epl}명)</span></div>
    <div><b>−{1 - cv['xA']['model'] / cv['xA']['same']:.0%}</b><span>기록을 그대로 믿을 때보다 줄어든 예측 오차 (xA)</span></div>
  </div>
  <nav class="toc" aria-label="목차"><a href="#factors">환산 계수</a><a href="#check">검증</a>
  <a href="#shortlist">영입 후보</a><a href="#asia">아시아 선수</a><a href="#data">K리그·J리그 데이터</a><a href="#method">방법과 한계</a></nav>
</header>

<section id="factors" class="part">
  <p class="eyebrow">Result</p>
  <h2>EPL에 오면 기대득점은 약 {(1 - tbl.npxG.max()) * 100:.0f}~{(1 - tbl.npxG.min()) * 100:.0f}% 줄어든다</h2>
  <p class="lede">같은 선수가 다른 리그에서 EPL로 오면 경기당 기대득점(npxG)은 대략 {tbl.npxG.min():.2f}~{tbl.npxG.max():.2f}배,
  기대 어시스트(xA)는 {tbl.xA.min():.2f}~{tbl.xA.max():.2f}배가 된다. 막대 끝의 선은 90% 신뢰구간.</p>
  <figure>{_img(charts.translation_factors(tbl, "League → Premier League translation factors"), "리그별 환산 계수")}</figure>
  <div class="finding-grid">
    <article class="finding"><h3>평균 회귀가 먼저</h3><p>리그를 옮기지 않아도 좋은 시즌 다음 해엔 기록이 내려온다
    (지속 계수 b = {models['npxG']['b']:.2f}). 환산 계수는 이 효과를 뺀 뒤의 순수한 리그 차이다.</p></article>
    <article class="finding"><h3>리그마다 깎이는 지표가 다름</h3><p>{tbl.xA.idxmax()} 출신은 xA가 가장 잘 유지되고(×{tbl.xA.max():.2f}),
    {tbl.xA.idxmin()} 출신은 가장 많이 줄어든다(×{tbl.xA.min():.2f}). 기대득점도 {tbl.npxG.idxmax()}(×{tbl.npxG.max():.2f})와
    {tbl.npxG.idxmin()}(×{tbl.npxG.min():.2f})의 차이가 있다. 어느 리그 출신이냐에 따라 할인 폭이 다르다.</p></article>
    <article class="finding"><h3>표본이 적은 리그는 구간을 볼 것</h3><p>러시아 리그처럼 EPL과 직접 오간 선수가 적은 리그는
    신뢰구간이 넓다(npxG ×{tbl.loc['Russian PL','npxG_lo']:.2f}~{tbl.loc['Russian PL','npxG_hi']:.2f}). 계수 하나보다 구간을 기준으로 판단해야 한다.</p></article>
  </div>
</section>

<section id="check" class="part">
  <p class="eyebrow">Validation</p>
  <h2>정말 예측이 좋아졌나</h2>
  <p class="lede">리그를 옮긴 선수들을 10개 묶음으로 나눠, 한 묶음을 빼고 학습한 모델로 그 묶음의 다음 시즌 기록을 예측했다(10겹 교차검증).
  숫자는 경기당 평균 절대 오차(낮을수록 좋음).</p>
  <div class="tbl"><table><caption>다음 시즌 per-90 예측 오차</caption>
    <thead><tr><th>지표</th><th>기록 그대로</th><th>평균 회귀만</th><th>환산 모델</th><th>개선</th></tr></thead>
    <tbody>{cvrow('npxG')}{cvrow('xA')}</tbody></table></div>
  <p class="method" style="margin-top:14px">"기록 그대로"는 스카우팅 리포트에 흔히 적히는 방식(작년 숫자를 그대로 기대).
  개선 폭이 크지 않은 건 정상이다. 선수 개인의 다음 시즌은 원래 잡음이 크고, 모델이 할 일은 그 잡음 속에서 체계적인 편향을 걷어 내는 것이다.</p>
</section>

<section id="shortlist" class="part">
  <p class="eyebrow">Shortlist</p>
  <h2>빅클럽 밖에서 찾은 EPL급 공격 자원</h2>
  <p class="lede">2025/26 시즌 5개 리그에서 1,500분 이상 뛴 공격수·미드필더를 EPL 기준으로 환산하고,
  같은 방식으로 예측한 EPL 공격 자원 {len(T.load_epl())}명과 비교했다. 레알·바르사·바이에른 같은 빅클럽 소속은 제외.</p>
  <div class="tbl"><table><caption>EPL 환산 공격 생산량 상위 (npxG + xA per 90)</caption>
    <thead><tr><th>선수</th><th>팀</th><th>리그</th><th>출전</th><th>npxG/90 → EPL</th><th>xA/90 → EPL</th><th>EPL 백분위</th></tr></thead>
    <tbody>{trs(value)}</tbody></table></div>
  <p class="method" style="margin-top:14px">나이, 이적료, 계약 기간, 수비 기여는 반영하지 않았다. 실제 영입 판단에는 이 리스트를 영상 스카우팅과 시장가치 데이터로 좁혀야 한다.</p>
</section>

<section id="asia" class="part">
  <p class="eyebrow">Asia</p>
  <h2>유럽에서 뛰는 한국·일본 선수를 EPL 기준으로 보면</h2>
  <div class="finding-grid">{asia_cards}</div>
  <p class="method" style="margin-top:14px">이강인은 적은 출전 시간(PSG)에도 xA가 높아 EPL 기준으로도 찬스메이커 상위권.
  단, 공격 지표만 본 결과라 수비 가담·압박 기여가 큰 선수(예: 이재성)는 과소평가된다.</p>
</section>

<section id="data" class="part src">
  <p class="eyebrow">Next</p>
  <h2>K리그·J리그로 넓히려면: 데이터 조사 결과</h2>
  <ul>
    <li><b>J1리그 2024 무료 데이터 (가장 유망)</b>: Hudl StatsBomb이 J1리그 2024 시즌 이벤트 데이터와 피지컬 데이터를 무료로 공개했다.
    이 데이터로 "J1에서 유럽으로 간 선수"의 환산 계수를 만들 수 있다.
    <a href="https://www.hudl.com/blog/j1-league-free-data-statsbomb" target="_blank" rel="noopener">Hudl 공지</a></li>
    <li><b>J리그 공식 사이트</b>: 구단별 xG를 공식 제공. 선수별 xG는 확인되지 않음.
    <a href="https://www.jleague.jp/en/j1/stats/" target="_blank" rel="noopener">jleague.jp</a></li>
    <li><b>K리그 데이터 포털</b>: 일반 공개용(data.kleague.com)은 로그인 없이 기본 기록을 제공. 상세 포털(portal.kleague.com)은 연맹·구단 관계자 전용이고,
    데이터 재가공·게시는 연맹 허락이 필요하다고 명시. <a href="https://portal.kleague.com/" target="_blank" rel="noopener">K LEAGUE PORTAL</a></li>
    <li><b>K리그 xG</b>: FootyStats가 팀·선수 xG 상위 목록을 무료로 보여 주고, 전체 데이터는 유료.
    FotMob에도 K리그 1 통계 페이지가 있다(xG 제공 범위는 확인 필요).
    <a href="https://footystats.org/south-korea/k-league-1/xg" target="_blank" rel="noopener">FootyStats</a> ·
    <a href="https://www.fotmob.com/leagues/9080/stats/k-league-1" target="_blank" rel="noopener">FotMob</a></li>
  </ul>
  <p class="method">정리: <b>J1리그는 무료 이벤트 데이터가 있어 바로 착수 가능</b>, K리그는 연맹 협조나 유료 데이터가 필요하다.
  K리그 연맹의 데이터 교육·협업 프로그램에 참여하는 것이 가장 현실적인 경로다.</p>
</section>

<section id="method" class="part method">
  <p class="eyebrow">Method</p>
  <h2>방법과 한계</h2>
  <ul>
    <li><b>모델</b>: log(다음 시즌 지표+0.05) = a + b·log(이번 시즌 지표+0.05) + β(옮긴 리그) − β(원래 리그). EPL의 β는 0.
    두 시즌 모두 900분 이상 뛴 선수만, 출전 시간 가중.</li>
    <li><b>표본</b>: 같은 리그 잔류 선수 11,079건(평균 회귀 추정)과 리그를 옮긴 선수 {len(mv)}건(리그 차이 추정,
    {mv.season.min()}/{str(mv.season.min() + 1)[2:]}~{mv.season.max()}/{str(mv.season.max() + 1)[2:]} 시즌 이적 전체).</li>
    <li><b>빠진 것</b>: 나이, 포지션·역할 변화, 팀 전력 차이(강팀으로 가면 찬스가 늘어남), 수비 지표. 다음 버전에서 팀 전력 보정을 추가할 것.</li>
    <li><b>데이터 이용 조건</b>: Understat 데이터는 개인·연구용. 상업적 스카우팅 서비스에는 라이선스 데이터가 필요하다.</li>
  </ul>
</section>
<footer>데이터: <a href="https://understat.com" target="_blank" rel="noopener">Understat</a>. 분석·코드: chelsea-analytics / cfc.translation.</footer>
</div>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build(ROOT / "reports" / "league_translation.html"))
