"""셈법 스카우트 v1: understat 2025/26 전체 시즌으로 2026/27 저평가 베테랑을 사전등록한다.

    python scout_v1.py     # understat 5대 리그 2017~2025 받기(캐시) → 과거 검증 → 2026/27 선발 명단 장부 등록
                           # → output/football/scout_v1.md

v0(fbref)와 다른 점: fbref 공개 사본은 2025/26이 시즌 초반까지만 있어서 understat 선수 시즌 기록을 쓴다.
  - 나이·포지션은 understat에 없음 → fbref 기록과 이름으로 연결 (가장 최근 fbref 시즌 나이 + 지난 햇수)
  - 역할 지표: FW npxG+xA, MF·DF xGBuildup (90분당). v0의 태클+가로채기·전진 패스는 understat에 없음
  - 팀 강도: understat 팀 npxG 차이의 리그 내 백분위
규칙은 실행 전에 장부에 고정 (kind=football_hypothesis, key=scout_v1_rules). 원자료는 레포에 올리지 않음.
2026/27은 이미 시작했지만, 선발은 2025/26 자료만으로 기계적으로 하고 2026/27 출전 시간은 보지 않는다.
"""
from __future__ import annotations

import json
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

from llmrel import ledger
from llmrel.understat import _curl
from scout_v0 import diff_ci

ROOT = Path(__file__).parent
CACHE = ROOT / "data_cache" / "understat"
OUT = ROOT / "output" / "football"
LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"]
YEARS = range(2017, 2026)          # understat 시작 연도 (2025 = 2025/26)
MIN_T = 1500
RULES = dict(version="v1", source="understat 선수 시즌 + fbref 나이·포지션", min_minutes=MIN_T,
             veteran_age="29~31 (선발 시즌 기준)", role_pct_min=0.75, team="리그 팀 npxG 차이 하위 절반",
             role="FW npxG+xA / MF·DF xGBuildup, 90분당, 시즌·포지션 내 백분위 (1500분+ 선수끼리)",
             outcomes="O1 다음 시즌 1500분+, O3 다음 시즌 역할 상위 절반", baseline="같은 나이·하위 절반 팀·1500분+, 역할 백분위 75 미만",
             verdict="선발군 − 기준선 차이 95% 범위로 3단계")


def key(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace("-", " ").replace(".", " ").split())


def league(lg: str, y: int) -> dict:
    f = CACHE / f"{lg}_{y}.json"
    if f.exists():
        return json.loads(f.read_text())
    for i in range(4):
        code, body = _curl(f"https://understat.com/getLeagueData/{lg}/{y}", {"X-Requested-With": "XMLHttpRequest"})
        if code == 200 and body.strip().startswith("{"):
            CACHE.mkdir(parents=True, exist_ok=True)
            f.write_text(body)
            return json.loads(body)
        time.sleep(2 ** i)
    raise RuntimeError(f"understat {lg} {y} 받기 실패 ({code})")


def seasons() -> pd.DataFrame:
    rows, teams = [], []
    for lg in LEAGUES:
        for y in YEARS:
            d = league(lg, y)
            for t in d["teams"].values():
                h = pd.DataFrame(t["history"])
                teams.append(dict(lg=lg, y=y, team=t["title"], diff=(h.npxG - h.npxGA).sum()))
            for p in d["players"]:
                rows.append(dict(id=p["id"], name=p["player_name"], lg=lg, y=y, team=p["team_title"].split(",")[0], minutes=float(p["time"]),
                                 npxg=float(p["npxG"]), xa=float(p["xA"]), build=float(p["xGBuildup"])))
    t = pd.DataFrame(teams)
    t["str_pct"] = t.groupby(["lg", "y"])["diff"].rank(pct=True)
    s = pd.DataFrame(rows).merge(t[["lg", "y", "team", "str_pct"]], on=["lg", "y", "team"], how="left")
    return s.groupby(["id", "y"]).agg(name=("name", "first"), lg=("lg", "first"), team=("team", "first"), minutes=("minutes", "sum"),
                                      npxg=("npxg", "sum"), xa=("xa", "sum"), build=("build", "sum"), str_pct=("str_pct", "first")).reset_index()


def fbref_bio() -> pd.DataFrame:
    """이름 → (기준 시즌의 나이, 포지션). 같은 이름이 여럿이면 버림 (잘못 연결 방지)."""
    d = pyreadr.read_r(ROOT / "data_cache" / "fbref" / "big5_player_standard.rds")[None]
    d = d[d.Url.notna() & d.Age.notna()].copy()
    d["age"] = pd.to_numeric(d.Age.astype(str).str.split("-").str[0], errors="coerce")
    d["pos"] = d.Pos.fillna("").str.split(",").str[0]
    d = d.sort_values("Min_Playing", ascending=False).drop_duplicates(["Url", "Season_End_Year"])
    d["k"] = d.Player.map(key)
    amb = d.groupby("k").Url.nunique()
    d = d[d.k.isin(amb[amb == 1].index)]
    return d[["k", "Season_End_Year", "age", "pos"]]


def attach_bio(s: pd.DataFrame) -> pd.DataFrame:
    b = fbref_bio()
    s = s.copy()
    s["k"] = s.name.map(key)
    s["end"] = s.y + 1                                                # understat 2025 = fbref Season_End_Year 2026
    bmap = {k: g for k, g in b.groupby("k")}
    out = []
    for k, g in s.groupby("k"):
        bb = bmap.get(k)
        if bb is None:
            continue
        for r in g.itertuples():
            prior = bb[bb.Season_End_Year <= r.end]                   # 그 시즌까지 알려진 기록만 (미래 기록으로 나이·포지션을 정하지 않음)
            ref = prior.iloc[prior.Season_End_Year.argmax()] if len(prior) else bb.iloc[bb.Season_End_Year.argmin()]
            out.append(dict(id=r.id, y=r.y, age=ref.age + (r.end - ref.Season_End_Year), pos=ref.pos))
    return s.merge(pd.DataFrame(out), on=["id", "y"], how="left")


def build() -> tuple[pd.DataFrame, float]:
    s = seasons()
    s = attach_bio(s)
    reg_all = s[s.minutes >= MIN_T]
    match = float(reg_all.age.notna().mean())
    s = s[s.pos.isin(["DF", "MF", "FW"])].copy()
    per90 = s.minutes / 90
    s["role"] = np.where(s.pos == "FW", (s.npxg + s.xa) / per90, s.build / per90)
    reg = s[s.minutes >= MIN_T].copy()
    reg["role_pct"] = reg.groupby(["y", "pos"]).role.rank(pct=True)
    s = s.merge(reg[["id", "y", "role_pct"]], on=["id", "y"], how="left")
    nxt = s[["id", "y", "minutes", "role_pct"]].rename(columns={"minutes": "minutes_1", "role_pct": "role_pct_1"})
    nxt["y"] -= 1
    p = s[s.minutes >= MIN_T].merge(nxt, on=["id", "y"], how="left")
    p["has_next"] = p.y < max(YEARS)
    p["minutes_1"] = p.minutes_1.fillna(0)
    p["O1"] = p.minutes_1 >= MIN_T
    p["O3"] = p.role_pct_1 >= 0.5
    bottom = p.str_pct <= 0.5
    vet = p.age.between(29, 31)
    p["V"] = vet & bottom & (p.role_pct >= 0.75)
    p["Vb"] = vet & bottom & (p.role_pct < 0.75)
    return p, match


def main():
    ledger.append([dict(kind="football_hypothesis", key="scout_v1_rules", payload=RULES, model="scout_v1", evidence="과거 검증·선발 전 고정")])
    p, match = build()
    hist = p[p.has_next]
    rows, res = [], {}
    for k, out, label in [("S1", "O1", "베테랑 다음 시즌 1500분+"), ("S2", "O3", "베테랑 다음 시즌 역할 상위 절반")]:
        a = hist[hist.V][out].astype(float).to_numpy()
        b = hist[hist.Vb][out].astype(float).to_numpy()
        ma, mb, lo, hi = diff_ci(a, b)
        v = "맞음" if lo > 0 else ("틀림" if hi <= 0 else "불확실")
        res[k] = dict(selected=round(ma, 4), baseline=round(mb, 4), n=[len(a), len(b)], ci=[round(lo, 4), round(hi, 4)], verdict=v)
        rows.append(f"| {k} | {label} | {ma:.0%} (n={len(a)}) | {mb:.0%} (n={len(b)}) | {ma - mb:+.1%}p ({lo:+.1%} ~ {hi:+.1%}) | {v} |")
    ledger.append([dict(kind="football_result", key="scout_v1_backtest", payload=res, model="scout_v1",
                        evidence=f"understat 5대 리그 선발 2017/18~2024/25, 채점 ~2025/26, 나이 연결률 {match:.0%}")])

    # 2026/27 사전등록: 2025/26 자료만으로 선발
    cur = p[p.y == max(YEARS)]
    picks, base = cur[cur.V].sort_values("role_pct", ascending=False), cur[cur.Vb]
    s1, s2 = res["S1"], res["S2"]
    payload = dict(season="2026-27", picks=[dict(player=r.name, team=r.team, league=r.lg, age=int(r.age), pos=r.pos, role_pct=round(float(r.role_pct), 3))
                                             for r in picks.itertuples()],
                   baseline_ids=sorted(base.id.astype(str).tolist()),
                   forecast=dict(claim="선발군의 2026/27 1500분+ 비율이 기준선보다 높다", backtest_gap=round(s1["selected"] - s1["baseline"], 4),
                                 confidence=0.7 if s1["verdict"] == "맞음" else 0.55),
                   scoring="2027년 여름, understat 2026/27 시즌 기록으로 O1·O3 채점")
    n_new = ledger.append([dict(kind="scout_pick", key="2026-27:veteran_v1", payload=payload, model="scout_v1",
                                evidence="2025/26 시즌 자료만 사용, 2026/27 출전 시간 보지 않고 등록")])
    ok, n_rows, last = ledger.verify()
    tbl = "\n".join(f"| {i} | {r.name} | {int(r.age)} | {r.pos} | {r.team} ({r.lg.replace('_', ' ')}) | {r.role_pct:.0%} |"
                    for i, r in enumerate(picks.itertuples(), 1))
    pos_n = picks.pos.value_counts().to_dict()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scout_v1.md").write_text(f"""# 셈법 스카우트 v1: 2026/27 저평가 베테랑 사전등록

자료: understat 5대 리그 선수 시즌 기록 2017/18~2025/26 + fbref 나이·포지션 (이름 연결, 1500분+ 선수 연결률 {match:.0%}).
규칙은 결과·선발 전에 장부 고정. 선발은 2025/26 자료만 쓰고, 이미 시작한 2026/27 출전 시간은 보지 않았다.
- 베테랑 29~31세, 소속팀이 리그 npxG 차이 하위 절반, 역할 지표 백분위 75 이상, 1500분+
- 역할 지표 (90분당): FW npxG+xA, MF·DF xGBuildup(공격 전개 관여)
- 기준선: 같은 나이·하위 절반 팀·1500분+ 이면서 역할 백분위 75 미만

## 과거 검증 (선발 2017/18~2024/25)
| 가설 | 결과 지표 | 선발군 | 기준선 | 차이 (95%) | 판정 |
|---|---|---|---|---|---|
{chr(10).join(rows)}

## 2026/27 선발 명단 ({len(picks)}명, 기준선 {len(base)}명) — 장부 등록 {'완료' if n_new else '이미 있음(같은 내용)'}
포지션: {', '.join(f'{k} {v}명' for k, v in pos_n.items())}

| # | 선수 | 나이 | 포지션 | 소속 (2025/26) | 역할 백분위 |
|---|---|---|---|---|---|
{tbl}

- 예측: "선발군의 2026/27 1500분+ 비율이 기준선보다 높다" (자신감 {payload['forecast']['confidence']:.0%}). 채점: 2027년 여름
- 장부 전체 {n_rows}줄, 체인 {'정상' if ok else '어긋남'}

## 한계
- 나이는 이름으로 fbref와 연결 → 동명이인은 버렸고, 연결 안 된 선수(주로 2025/26 새 얼굴)는 빠짐. 나이는 ±1세 오차 가능
- DF 역할을 수비 지표가 아니라 빌드업 관여로 잼 → '수비 잘하는 베테랑'이 아니라 '공을 잘 다루는 베테랑'을 고름
- 가격(이적료·연봉) 없음: 하위 절반 팀은 '쌀 가능성'의 대리일 뿐
- 2026/27 여름 이적으로 리그를 떠나면(사우디 등) 0분으로 셈
""", encoding="utf-8")
    print((OUT / "scout_v1.md").read_text())


if __name__ == "__main__":
    main()
