"""셈법 스카우트 v0: 원칙 3 선발 규칙을 과거 시즌으로 검증 (블룸 방식 1단계).

    python scout_v0.py    # data_cache/fbref/{big5_player_standard,big5_player_defense,big5_team_standard}.rds 필요
                          # → output/football/scout_v0.md

규칙은 실행 전에 장부에 고정 (kind=football_hypothesis, key=scout_v0_rules). 원자료는 레포에 올리지 않음 (Opta/Stats Perform 권리).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

from llmrel import ledger

ROOT = Path(__file__).parent
C = ROOT / "data_cache" / "fbref"
OUT = ROOT / "output" / "football"
FIRST, LAST = 2018, 2024          # 선발 시즌 (끝나는 해). 다음 시즌 채점은 2025까지
MIN_T = 1500


def rd(name: str) -> pd.DataFrame:
    return pyreadr.read_r(C / f"{name}.rds")[None]


def team_strength() -> pd.DataFrame:
    t = rd("big5_team_standard")
    t = t[t.Season_End_Year >= FIRST]
    w = t.pivot_table(index=["Season_End_Year", "Comp", "Squad"], columns="Team_or_Opponent", values="npxG_Expected").reset_index()
    w["diff"] = w["team"] - w["opponent"]
    w["str_pct"] = w.groupby(["Season_End_Year", "Comp"])["diff"].rank(pct=True)
    return w[["Season_End_Year", "Squad", "str_pct"]]


def player_seasons() -> pd.DataFrame:
    s = rd("big5_player_standard")
    d = rd("big5_player_defense")[["Url", "Season_End_Year", "Squad", "Tkl+Int"]]
    s = s[(s.Season_End_Year >= FIRST) & s.Url.notna()].merge(d, on=["Url", "Season_End_Year", "Squad"], how="left")
    s["contrib"] = s.npxG_Expected.fillna(0) + s.xAG_Expected.fillna(0)
    s = s.merge(team_strength(), on=["Season_End_Year", "Squad"], how="left")
    s = s.sort_values("Min_Playing", ascending=False)
    g = s.groupby(["Url", "Season_End_Year"]).agg(player=("Player", "first"), age=("Age", "min"), pos=("Pos", "first"),
                                                 squad=("Squad", "first"), comp=("Comp", "first"), minutes=("Min_Playing", "sum"),
                                                 prgp=("PrgP_Progression", "sum"), tklint=("Tkl+Int", "sum"), contrib=("contrib", "sum"),
                                                 str_pct=("str_pct", "first")).reset_index()   # 이적 시즌: 가장 많이 뛴 팀 기준
    g["age"] = pd.to_numeric(g.age.astype(str).str.split("-").str[0], errors="coerce")   # 2023/24부터 "29-123"(세-일) 표기
    g["pg"] = g.pos.fillna("").str.split(",").str[0]
    g = g[g.pg.isin(["DF", "MF", "FW"])]
    per90 = g.minutes / 90
    g["role"] = np.select([g.pg == "DF", g.pg == "MF"], [g.tklint / per90, g.prgp / per90], g.contrib / per90)
    reg = g[g.minutes >= MIN_T].copy()
    reg["role_pct"] = reg.groupby(["Season_End_Year", "pg"]).role.rank(pct=True)
    return g.merge(reg[["Url", "Season_End_Year", "role_pct"]], on=["Url", "Season_End_Year"], how="left")


def build() -> pd.DataFrame:
    g = player_seasons()
    nxt = g[["Url", "Season_End_Year", "minutes", "role_pct", "str_pct"]].rename(columns=lambda c: c if c in ("Url", "Season_End_Year") else c + "_1")
    nxt["Season_End_Year"] -= 1
    nx2 = g[["Url", "Season_End_Year", "minutes"]].rename(columns={"minutes": "minutes_2"})
    nx2["Season_End_Year"] -= 2
    p = g[(g.Season_End_Year <= LAST) & (g.minutes >= MIN_T)].merge(nxt, on=["Url", "Season_End_Year"], how="left").merge(nx2, on=["Url", "Season_End_Year"], how="left")
    p["minutes_1"] = p.minutes_1.fillna(0)
    p["O1"] = p.minutes_1 >= MIN_T
    p["O3"] = p.role_pct_1 >= 0.5                                   # 다음 시즌 1500분 미만이면 백분위 없음 → 실패로 셈
    p["O4"] = np.where(p.Season_End_Year + 2 <= 2025, p.minutes_2.fillna(0) >= MIN_T, np.nan)
    p["O2"] = p.str_pct_1 > p.str_pct
    p["bottom"] = p.str_pct <= 0.5
    p["V"] = p.age.between(29, 31) & p.bottom & (p.role_pct >= 0.75)
    p["Y"] = (p.age <= 22) & p.bottom & (p.role_pct >= 0.75)
    p["Vb"] = p.age.between(29, 31) & p.bottom & (p.role_pct < 0.75)
    p["Yb"] = (p.age <= 22) & p.bottom & (p.role_pct < 0.75)
    return p


def diff_ci(a: np.ndarray, b: np.ndarray, n: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)
    d = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(n)]
    return a.mean(), b.mean(), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    p = build()
    tests = [("S1", "V", "Vb", "O1", "베테랑 다음 시즌 1500분+"), ("S2", "V", "Vb", "O3", "베테랑 다음 시즌 역할 상위 절반"),
             ("S3", "V", "Vb", "O4", "베테랑 2시즌 뒤에도 1500분+"), ("S4", "Y", "Yb", "O1", "유망주 다음 시즌 1500분+"),
             ("S5", "Y", "Yb", "O2", "유망주 더 강한 팀으로 (팀 순위 백분위 상승)")]
    rows, res = [], {}
    for key, sel, base, out, label in tests:
        a = p[p[sel]][out].dropna().astype(float).to_numpy()
        b = p[p[base]][out].dropna().astype(float).to_numpy()
        ma, mb, lo, hi = diff_ci(a, b)
        v = "맞음" if lo > 0 else ("틀림" if hi <= 0 else "불확실")
        res[key] = dict(selected=round(ma, 4), baseline=round(mb, 4), n=[len(a), len(b)], ci=[round(lo, 4), round(hi, 4)], verdict=v)
        rows.append(f"| {key} | {label} | {ma:.0%} (n={len(a)}) | {mb:.0%} (n={len(b)}) | {ma - mb:+.1%}p ({lo:+.1%} ~ {hi:+.1%}) | {v} |")
    ledger.append([dict(kind="football_result", key="scout_v0", payload=res, model="scout_v0", evidence="fbref 5대 리그 2017/18~2024/25 과거 검증")])
    # 포지션별 베테랑 (원칙: 조율·수비 역할)
    pos = []
    for pg in ["DF", "MF", "FW"]:
        a = p[p.V & (p.pg == pg)]; b = p[p.Vb & (p.pg == pg)]
        pos.append(f"| {pg} | {a.O1.mean():.0%} / {a.O3.mean():.0%} (n={len(a)}) | {b.O1.mean():.0%} / {b.O3.mean():.0%} (n={len(b)}) |")
    ex = p[p.V & (p.Season_End_Year == LAST)].sort_values("role_pct", ascending=False).head(8)
    exl = "\n".join(f"| {r.player} | {int(r.age)} | {r.pg} | {r.squad} | {r.role_pct:.0%} | {'✔' if r.O1 else '✘'} | {'✔' if r.O3 else '✘'} |" for r in ex.itertuples())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scout_v0.md").write_text(f"""# 셈법 스카우트 v0: 원칙 3 선발 규칙 과거 검증

선발 시즌 2017/18~2023/24 (5대 리그, 1500분+), 다음 시즌 2024/25까지 채점. 규칙은 실행 전 장부 고정.
- 역할 지표 (90분당, 시즌·포지션 내 백분위): DF 태클+가로채기, MF 전진 패스, FW npxG+xAG
- '쌀 가능성' 대리: 소속팀이 리그 기대득점 차이 하위 절반
- 선발: 베테랑 29~31세 / 유망주 22세 이하, 역할 백분위 75 이상 + 하위 절반 팀
- 기준선: 같은 나이·하위 절반 팀·1500분+ 이면서 역할 백분위 75 미만

## 판정
| 가설 | 결과 지표 | 선발군 | 기준선 | 차이 (95%) | 판정 |
|---|---|---|---|---|---|
{chr(10).join(rows)}

## 베테랑 포지션별 (다음 시즌 1500분+ / 역할 상위 절반)
| 포지션 | 선발군 | 기준선 |
|---|---|---|
{chr(10).join(pos)}

## 예: 2023/24 시즌 끝 규칙이 고른 베테랑 (2024/25 채점)
| 선수 | 나이 | 포지션 | 소속 | 역할 백분위 | 다음 시즌 1500분+ | 역할 상위 절반 유지 |
|---|---|---|---|---|---|---|
{exl}

## 한계
- 가격(이적료·연봉) 없음: '하위 절반 팀'은 쌀 가능성의 대리일 뿐
- 기준선보다 낫다는 건 '이 규칙이 정보가 있다'는 뜻이지 '이적 시장에서 이익'은 아님
- 역할 지표가 높은 선수는 다음 시즌 평균 쪽으로 돌아오는 경향(평균 회귀)이 있음 → O3는 보수적으로 해석
""", encoding="utf-8")
    print((OUT / "scout_v0.md").read_text())


if __name__ == "__main__":
    main()
