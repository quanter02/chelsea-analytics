"""Opponent scouting: find a team's weaknesses/threats vs the league and turn them into a game plan.

Every finding is a rule over league percentiles, so the output is explainable:
each recommendation carries the number and the league rank that triggered it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import channel

LATE_MINUTE = 60
# attacker's left channel is the DEFENDER's right side, and vice versa
DEF_SIDE_KO = {"left": "오른쪽", "right": "왼쪽", "center": "중앙"}
ATT_SIDE_KO = {"left": "왼쪽", "right": "오른쪽", "center": "중앙"}


@dataclass
class Finding:
    kind: str          # "weakness" | "threat" | "style"
    title: str
    evidence: str
    action: str
    score: float = 0.0  # how far from league norm (0..1), used for ordering
    counter: str = ""   # what the scouted team itself should do to fix it (own-team view)


@dataclass
class ScoutingReport:
    team: str
    record: dict
    weaknesses: list[Finding] = field(default_factory=list)
    threats: list[Finding] = field(default_factory=list)
    style: list[Finding] = field(default_factory=list)
    key_players: dict = field(default_factory=dict)


def _shots_against(shots: pd.DataFrame, tm: pd.DataFrame, team: str) -> pd.DataFrame:
    games = tm.loc[tm.team == team, ["match_id", "opponent"]]
    sa = shots.merge(games, on="match_id")
    return sa[sa.team == sa.opponent]


def defensive_profile(shots: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    """Per team: how the xG they concede is distributed (share-based, so size-independent)."""
    rows = []
    for team in sorted(tm.team.unique()):
        sa = _shots_against(shots, tm, team)
        tot = sa.xg.sum()
        ach = sa.assist_y.dropna().map(channel)
        by_ch = sa.loc[ach.index].groupby(ach).xg.sum()
        level = sa[sa.state == "level"]  # control for game state: only shots taken at level score
        rows.append({
            "team": team,
            "late_share": level.loc[level.minute >= LATE_MINUTE, "xg"].sum() / level.xg.sum(),
            # shooter trailing == this team leading
            "leading_share": sa.loc[sa.state == "trailing", "xg"].sum() / tot,
            "header_share": sa.loc[sa.body_part == "Head", "xg"].sum() / tot,
            "cross_share": sa.loc[sa.assist_cross, "xg"].sum() / tot,
            "setpiece_share": sa.loc[sa.origin == "set_piece", "xg"].sum() / tot,
            **{f"ch_{c}_share": by_ch.get(c, 0) / tot for c in ("left", "center", "right")},
        })
    return pd.DataFrame(rows).set_index("team")


def _trimmed(tm: pd.DataFrame, col: str) -> pd.Series:
    """Per-team mean after dropping that team's single most extreme (highest) match.

    With 5-10 matches one freak game can create a 'weakness'; a finding only counts if it survives this.
    """
    return tm.groupby("team")[col].apply(lambda v: v.drop(v.idxmax()).mean() if len(v) > 3 else v.mean())


def _pct(series: pd.Series, team: str) -> float:
    return float(series.rank(pct=True)[team])


def _rank(series: pd.Series, team: str, ascending: bool = False) -> int:
    return int(series.rank(ascending=ascending, method="min")[team])


def scout(team: str, tm: pd.DataFrame, shots: pd.DataFrame) -> ScoutingReport:
    season = tm.groupby("team").mean(numeric_only=True)
    prof = defensive_profile(shots, tm)
    t, p = season.loc[team], prof.loc[team]
    n = len(season)
    g = tm[tm.team == team]
    rep = ScoutingReport(team, {
        "matches": len(g), "points": int(g.points.sum()),
        "gf": int(g.goals.sum()), "ga": int(g.goals_against.sum()),
        "xg": float(g.xg.sum()), "xga": float(g.xga.sum()),
    })

    # ---------- weaknesses (where they concede) ----------
    pl = _pct(prof.late_share, team)
    if pl >= 0.7:
        rep.weaknesses.append(Finding(
            "weakness", "후반 60분 이후 급격히 흔들림",
            f"동점 상황에서 허용한 xG의 {p.late_share:.0%}가 60분 이후 발생 (리그 평균 {prof.late_share.mean():.0%}, "
            f"{_rank(prof.late_share, team)}/{n}위). 스코어 상황을 통제해도 후반에 수비가 느슨해짐",
            "체력이 좋은 공격 자원을 60분 전후 교체 투입하고, 승부처를 후반 30분에 설정. "
            "전반은 무리하지 말고 템포를 조절.", pl,
            counter="55~65분 사이 수비형 미드필더·풀백 교체 카드를 먼저 쓰고, 동점이면 라인을 한 칸 내려 "
                    "수비 형태를 유지할 것."))
    ps = _pct(season.xga_setpiece, team)
    sp_trim = _trimmed(tm, "xga_setpiece")
    if ps >= 0.6:
        robust = _pct(sp_trim, team) >= 0.6
        rep.weaknesses.append(Finding(
            "weakness", "세트피스 수비 취약" if robust else "세트피스 수비 취약 (한 경기 영향 큼)",
            f"경기당 세트피스 허용 xG {t.xga_setpiece:.2f} (리그 평균 {season.xga_setpiece.mean():.2f}, "
            f"{_rank(season.xga_setpiece, team)}/{n}위). 최악의 한 경기를 빼면 {sp_trim[team]:.2f}"
            + ("로 여전히 상위권" if robust else "로 리그 평균 수준"),
            "코너킥·프리킥 약속된 플레이를 2~3개 준비하고, 파이널 서드에서 파울을 유도하는 드리블 돌파를 늘릴 것.",
            ps if robust else ps * 0.5,
            counter="세트피스 수비 전담 마크 재정비, 상대 공중볼 타깃에게 가장 강한 수비수를 대인으로 붙이고 "
                    "박스 근처 불필요한 파울을 줄일 것."))
    ph = _pct(prof.header_share, team)
    if ph >= 0.7:
        rep.weaknesses.append(Finding(
            "weakness", "공중볼·헤더 실점 위험",
            f"허용 xG 중 헤더 비중 {p.header_share:.0%} (리그 평균 {prof.header_share.mean():.0%}, "
            f"{_rank(prof.header_share, team)}/{n}위)",
            "측면 크로스와 세트피스에서 타깃형 공격수·센터백을 적극 활용.", ph,
            counter="크로스 출발점 차단(풀백의 1대1 대응)과 박스 안 공중볼 경합 배치를 우선 점검할 것."))
    pc = _pct(season.xga_counter, team)
    if pc >= 0.7:
        rep.weaknesses.append(Finding(
            "weakness", "역습에 노출",
            f"경기당 역습 허용 xG {t.xga_counter:.2f} (리그 평균 {season.xga_counter.mean():.2f})",
            "상대 빌드업을 유도한 뒤 볼 탈취 즉시 3~4초 안에 전진하는 역습 트리거 설정.", pc))
    ch = {c: p[f"ch_{c}_share"] for c in ("left", "center", "right")}
    lg_ch = {c: prof[f"ch_{c}_share"].mean() for c in ch}
    worst = max(ch, key=lambda c: ch[c] - lg_ch[c])
    has_channels = shots.assist_y.notna().any()
    if has_channels and ch[worst] - lg_ch[worst] >= 0.03:
        rep.weaknesses.append(Finding(
            "weakness", f"수비 {DEF_SIDE_KO[worst]} 측면에서 찬스 허용이 많음",
            f"어시스트 기준 허용 xG의 {ch[worst]:.0%}가 공격팀의 {ATT_SIDE_KO[worst]} 채널에서 발생 "
            f"(리그 평균 {lg_ch[worst]:.0%})",
            f"우리 팀의 {ATT_SIDE_KO[worst]} 측면에 최고의 1대1 자원을 배치하고 오버래핑으로 수적 우위를 만들 것.",
            ch[worst] - lg_ch[worst]))

    pcr = _pct(prof.cross_share, team)
    if not has_channels and pcr >= 0.75:
        rep.weaknesses.append(Finding(
            "weakness", "크로스에 약함",
            f"허용 xG의 {p.cross_share:.0%}가 크로스에서 시작 (리그 평균 {prof.cross_share.mean():.0%}, "
            f"{_rank(prof.cross_share, team)}/{n}위)",
            "측면에서 얼리 크로스와 컷백을 늘리고, 반대편 윙어가 파포스트로 침투할 것.", pcr))
    if ("deep_allowed" in season and _pct(season.deep_allowed, team) >= 0.7
            and _pct(_trimmed(tm, "deep_allowed"), team) >= 0.7):   # must survive dropping the worst match
        pdp = _pct(season.deep_allowed, team)
        rep.weaknesses.append(Finding(
            "weakness", "골문 근처까지 패스를 쉽게 허용",
            f"경기당 딥 컴플리션(골문 약 18m 이내로 들어온 패스, 크로스 제외) 허용 {t.deep_allowed:.1f}회 "
            f"(리그 평균 {season.deep_allowed.mean():.1f}, {_rank(season.deep_allowed, team)}/{n}위). "
            f"팀마다 최악의 한 경기를 빼고 비교해도 {_trimmed(tm, 'deep_allowed')[team]:.1f}회 "
            f"(리그 {_trimmed(tm, 'deep_allowed').mean():.1f})",
            "하프스페이스로 빠르게 전진 패스를 넣어 수비 라인 앞 공간을 공략.", pdp))

    # ---------- threats (how they hurt you) ----------
    own = shots[shots.team == team]
    sp_air = own[(own.origin == "set_piece") & (own.body_part == "Head")].groupby("player").xg.sum()
    if len(sp_air) and sp_air.max() >= 0.6:
        tgt = sp_air.idxmax()
        share = sp_air.max() / max(own.loc[own.player == tgt, "xg"].sum(), 1e-9)
        rep.threats.append(Finding(
            "threat", f"세트피스 공중볼 타깃: {tgt}",
            f"{tgt}의 xG {own.loc[own.player == tgt, 'xg'].sum():.1f} 중 {share:.0%}가 세트피스 헤더 "
            f"(세트피스 헤더 xG {sp_air.max():.2f}, 팀 내 1위)",
            f"코너킥·프리킥 수비 시 {tgt}에게 가장 강한 공중볼 수비수를 대인 배치하고, 니어포스트 첫 터치를 차단.",
            min(1.0, sp_air.max())))
    if "xg_through" in season and _pct(season.xg_through, team) >= 0.8:
        rep.threats.append(Finding(
            "threat", "스루패스 침투",
            f"경기당 스루패스에서 나온 xG {t.xg_through:.2f}, 리그 {_rank(season.xg_through, team)}/{n}위",
            "수비 라인 뒤 공간 관리: 오프사이드 라인을 일정하게 유지하고, 패스 공급원에 즉시 압박.",
            _pct(season.xg_through, team)))
    if "xg_cross" in season and _pct(season.xg_cross, team) >= 0.8:
        rep.threats.append(Finding(
            "threat", "크로스 공격",
            f"경기당 크로스에서 나온 xG {t.xg_cross:.2f}, 리그 {_rank(season.xg_cross, team)}/{n}위",
            "크로스 출발점(풀백·윙어)에 대한 압박을 먼저 하고, 박스 안에서는 타깃맨에 공중볼 전담 마크.",
            _pct(season.xg_cross, team)))
    if _pct(season.xg_counter, team) >= 0.8:
        rep.threats.append(Finding(
            "threat", "리그 최상위 역습",
            f"경기당 역습 xG {t.xg_counter:.2f}, 리그 {_rank(season.xg_counter, team)}/{n}위 "
            f"(평균 {season.xg_counter.mean():.2f})",
            "공격 시에도 최소 3명(센터백 2 + 수비형 미드필더 1)을 후방에 남기는 '레스트 디펜스' 유지. "
            "볼을 잃은 직후 전술적 파울로 역습 첫 패스를 끊을 것.", _pct(season.xg_counter, team)))
    if _pct(season.xg_setpiece, team) >= 0.8:
        rep.threats.append(Finding(
            "threat", "세트피스 득점력",
            f"경기당 세트피스 xG {t.xg_setpiece:.2f}, 리그 {_rank(season.xg_setpiece, team)}/{n}위",
            "위험 지역(페널티 박스 근처) 파울 금지. 코너 수비 시 상대 핵심 타깃에 전담 마크.",
            _pct(season.xg_setpiece, team)))
    if _pct(season.long_ball_share, team) >= 0.8:
        rep.threats.append(Finding(
            "threat", "뒷공간을 노리는 롱볼",
            f"자기 진영 패스 중 롱볼 비율 {t.long_ball_share:.0%}, 리그 {_rank(season.long_ball_share, team)}/{n}위",
            "수비 라인을 너무 높이지 말고, 센터백 뒤 커버를 위해 풀백 한 명은 항상 내려와 있을 것.",
            _pct(season.long_ball_share, team)))

    # ---------- style (how to take the ball off them) ----------
    pld = _pct(prof.leading_share, team)
    if pld >= 0.75:
        rep.style.append(Finding(
            "style", "리드하면 깊게 내려앉음",
            f"허용 xG의 {p.leading_share:.0%}가 자신들이 앞선 상황에서 발생 (리그 평균 {prof.leading_share.mean():.0%}, "
            f"{_rank(prof.leading_share, team)}/{n}위)",
            "선제골을 내줘도 당황하지 말 것. 상대가 내려앉으면 박스 앞 세컨드볼과 크로스·세트피스로 압박을 누적.",
            pld))
    if _pct(season.pass_share, team) <= 0.3:
        rep.style.append(Finding(
            "style", "점유를 내주는 팀",
            f"패스 점유율 {t.pass_share:.0%} ({_rank(season.pass_share, team)}/{n}위)",
            "우리가 공을 오래 가질 것을 전제로, 무의미한 점유 대신 '볼을 잃는 위치'를 관리. "
            "중앙에서의 횡패스 실수를 최소화.", 1 - _pct(season.pass_share, team)))
    if _pct(season.ppda, team) >= 0.6:
        rep.style.append(Finding(
            "style", "전방 압박이 강하지 않음",
            f"PPDA {t.ppda:.1f} (높을수록 압박이 느슨, 리그 평균 {season.ppda.mean():.1f})",
            "후방 빌드업을 서두를 필요 없음. 센터백이 전진 드리블로 첫 수비 라인을 끌어낼 것.",
            _pct(season.ppda, team)))
    elif _pct(season.ppda, team) <= 0.3:
        rep.style.append(Finding(
            "style", "강한 전방 압박",
            f"PPDA {t.ppda:.1f} (리그 평균 {season.ppda.mean():.1f})",
            "골키퍼 포함 3인 빌드업 + 압박을 넘기는 대각선 롱패스 루트를 준비.",
            1 - _pct(season.ppda, team)))

    # ---------- key players ----------
    sf = shots[shots.team == team]
    shooters = sf.groupby("player").agg(shots=("xg", "size"), xg=("xg", "sum"), goals=("goal", "sum"))
    creators = sf.dropna(subset=["assist_player"]).groupby("assist_player").agg(
        chances=("xg", "size"), xa=("xg", "sum"))
    ach = sf.assist_y.dropna().map(channel)
    att = sf.loc[ach.index].groupby(ach).xg.sum()
    rep.key_players = {
        "shooters": shooters.sort_values("xg", ascending=False).head(3),
        "creators": creators.sort_values("xa", ascending=False).head(3),
        "attack_channel": att / att.sum(),
    }
    for lst in (rep.weaknesses, rep.threats, rep.style):
        lst.sort(key=lambda f: f.score, reverse=True)
    return rep


def game_plan(rep: ScoutingReport, max_items: int = 5) -> list[str]:
    """Condense findings into an ordered plan: first stop their best weapon, then attack their weak spot."""
    plan = []
    if rep.threats:
        plan.append(f"[막을 것] {rep.threats[0].title} → {rep.threats[0].action}")
    sh = rep.key_players["shooters"]
    cr = rep.key_players["creators"]
    if len(sh) and len(cr) and len(rep.key_players["attack_channel"]) == 0:
        if sh.index[0] == cr.index[0]:
            plan.append(f"[봉쇄 대상] {sh.index[0]}가 팀 최다 xG({sh.xg.iloc[0]:.1f})와 최다 xA({cr.xa.iloc[0]:.1f})를 "
                        "모두 가짐. 공격이 한 명에 몰려 있으니 전담 마크와 첫 터치 압박으로 연결 고리를 끊을 것.")
        else:
            plan.append(f"[봉쇄 대상] 득점원 {sh.index[0]} (xG {sh.xg.iloc[0]:.1f}) · "
                        f"창조자 {cr.index[0]} (xA {cr.xa.iloc[0]:.1f}). 두 선수 사이의 패스 연결을 끊는 것이 1순위.")
    elif len(sh) and len(cr):
        top_ch = rep.key_players["attack_channel"].idxmax()
        plan.append(
            f"[봉쇄 대상] 득점원 {sh.index[0]} (xG {sh.xg.iloc[0]:.1f}) · 창조자 {cr.index[0]} (xA {cr.xa.iloc[0]:.1f}). "
            f"상대 찬스의 {rep.key_players['attack_channel'][top_ch]:.0%}가 공격 {ATT_SIDE_KO[top_ch]} 채널에서 나오므로 "
            + ("수비형 미드필더를 센터백 앞에 고정해 중앙 하프스페이스 패스 길을 차단."
               if top_ch == "center" else f"우리 {DEF_SIDE_KO[top_ch]} 풀백·윙어의 수비 가담을 강화."))
    for f in rep.weaknesses[:2]:
        plan.append(f"[공략] {f.title} → {f.action}")
    for f in rep.style[:1]:
        plan.append(f"[운영] {f.title} → {f.action}")
    return plan[:max_items]


def matchup(us: str, them: str, tm: pd.DataFrame, shots: pd.DataFrame) -> list[Finding]:
    """Cross our profile with theirs: where the game is most likely to be decided."""
    ours, theirs = scout(us, tm, shots), scout(them, tm, shots)
    season = tm.groupby("team").mean(numeric_only=True)
    prof = defensive_profile(shots, tm)
    out = []
    their_air = [f for f in theirs.threats if f.title.startswith("세트피스 공중볼")]
    our_air_weak = [f for f in ours.weaknesses if f.title.startswith(("세트피스 수비", "공중볼"))]
    if their_air and our_air_weak:
        out.append(Finding("risk", "가장 큰 위험: 상대 세트피스 공중볼",
                           f"{their_air[0].evidence} ↔ 우리 약점: {', '.join(f.title for f in our_air_weak)}",
                           their_air[0].action, 1.0))
    their_air_weak = [f for f in theirs.weaknesses if f.title.startswith(("공중볼", "크로스", "세트피스 수비"))]
    our_sp = _pct(season.xg_setpiece, us)
    our_cross = _pct(season.xg_cross, us) if "xg_cross" in season else float("nan")
    if their_air_weak:
        good = [n for n, v in (("세트피스", our_sp), ("크로스", our_cross)) if v == v and v >= 0.5]
        tone = (f"우리 {'·'.join(good)} 공격과 맞물림" if good
                else "우리도 이 무기가 강하지 않음 → 훈련으로 보완 필요")
        out.append(Finding("edge", "공략 포인트: 상대의 공중볼·크로스 수비",
                           f"상대 약점: {', '.join(f.title for f in their_air_weak)} · 우리 세트피스 xG 리그 백분위 {our_sp:.0%}"
                           + (f", 크로스 xG 백분위 {our_cross:.0%}" if our_cross == our_cross else "") + f" ({tone})",
                           "세트피스 약속 플레이 2개와 얼리 크로스·컷백 루트를 이 경기의 주 득점 경로로 설계.", 0.8))
    if any(f.title.startswith("후반 60분") for f in ours.weaknesses):
        out.append(Finding("risk", "시간 관리: 우리 후반 약점",
                           f"우리는 동점 상황 60분 이후 허용 xG 비중이 리그 {_rank(prof.late_share, us)}위",
                           "리드 시 60분 전후 수비 보강 교체를 먼저, 동점이면 무리한 라인 상승 금지.", 0.7))
    return out
