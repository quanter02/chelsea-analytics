"""셈법 스카우트 v2: 진짜 가격(트랜스퍼마크트 시장가치)을 붙인다.

    python scout_v2.py    # data_cache/understat/*.json (scout_v1.py가 받음) + data_cache/tm/*.csv.gz 필요
                          # → output/football/scout_v2.md

자료: understat 선수 시즌 (출전·역할) + transfermarkt-datasets (생년월일·포지션·시장가치·계약 만료, 2026-07-06 스냅샷).
  - 이름으로 연결 (같은 이름이 여럿인 트랜스퍼마크트 선수는 버림)
  - 나이 = 선발 시즌 9월 1일 기준 만 나이 (생년월일로 정확히)
  - 가격 = 선발 시즌이 끝난 7월 1일 직전(400일 안) 시장가치. 시장가치는 실제 이적료가 아니라 추정치
가설·규칙은 실행 전에 장부 고정 (kind=football_hypothesis, key=scout_v2_rules). 원자료는 레포에 올리지 않음.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from llmrel import ledger
from scout_v0 import diff_ci
from scout_v1 import key, seasons, YEARS, MIN_T, OUT, ROOT

TM = ROOT / "data_cache" / "tm"
BIG5 = {"GB1", "ES1", "L1", "IT1", "FR1"}
POS = {"Attack": "FW", "Midfield": "MF", "Defender": "DF"}
RULES = dict(version="v2", price="트랜스퍼마크트 시장가치 (선발 시즌 끝 7월 1일 직전 400일 안 최신값)",
             age="선발 시즌 9월 1일 만 나이", cheap="같은 시즌·같은 포지션 1500분+ 선수 중 시장가치 하위 절반",
             role="FW npxG+xA / MF·DF xGBuildup, 90분당 백분위 (v1과 같음)",
             H1=dict(claim="같은 가격대(시즌 내 시장가치 10분위)에서 29~31세가 25~28세보다 다음 시즌 '좋은 시즌'(1500분+ & 역할 상위 절반) 비율이 높다 → 시장이 나이를 과하게 깎는다",
                     confidence=0.5, verdict="차이 95% 범위 전체 > 0 맞음, 전체 < 0 틀림, 걸치면 불확실"),
             H2=dict(claim="29~31세·시장가치 하위 절반 중 역할 백분위 75+ 선발군이 기준선(75 미만)보다 다음 시즌 1500분+ 비율이 높다",
                     confidence=0.6, verdict="차이 95% 범위 3단계"),
             picks="2025/26 자료로 2026/27 선발: 29~31세, 1500분+, 역할 백분위 75+, 시장가치 하위 절반")


def tm_bio() -> pd.DataFrame:
    p = pd.read_csv(TM / "players.csv.gz", usecols=["player_id", "name", "date_of_birth", "position", "contract_expiration_date"])
    v = pd.read_csv(TM / "player_valuations.csv.gz", usecols=["player_id", "date", "market_value_in_eur", "player_club_domestic_competition_id"])
    big5 = set(v[v.player_club_domestic_competition_id.isin(BIG5) & (v.date >= "2016-01-01")].player_id)
    p = p[p.player_id.isin(big5) & p.position.isin(POS)].copy()
    p["k"] = p.name.map(key)
    p = p[~p.k.duplicated(keep=False)]
    p["pos"] = p.position.map(POS)
    p["dob"] = pd.to_datetime(p.date_of_birth, errors="coerce")
    return p[["player_id", "k", "pos", "dob", "contract_expiration_date"]], v[v.player_id.isin(p.player_id)]


def value_at(v: pd.DataFrame, pairs: pd.DataFrame) -> pd.Series:
    """pairs: player_id, y → y+1년 7월 1일 직전 400일 안의 최신 시장가치."""
    v = v.assign(date=pd.to_datetime(v.date)).sort_values("date")
    out = []
    for y, g in pairs.groupby("y"):
        cut = pd.Timestamp(f"{y + 1}-07-01")
        w = v[(v.date <= cut) & (v.date > cut - pd.Timedelta(days=400))].drop_duplicates("player_id", keep="last")
        out.append(g[["player_id", "y"]].merge(w[["player_id", "market_value_in_eur"]], on="player_id", how="left"))
    return pd.concat(out)


def build() -> tuple[pd.DataFrame, float]:
    s = seasons()
    bio, v = tm_bio()
    s["k"] = s.name.map(key)
    s = s.merge(bio, on="k", how="left")
    match = float(s[s.minutes >= MIN_T].player_id.notna().mean())
    s = s[s.player_id.notna() & s.pos.notna()].copy()
    s["age"] = np.floor((pd.to_datetime(s.y.astype(str) + "-09-01") - s.dob).dt.days / 365.25)
    per90 = s.minutes / 90
    s["role"] = np.where(s.pos == "FW", (s.npxg + s.xa) / per90, s.build / per90)
    reg = s[s.minutes >= MIN_T].copy()
    reg["role_pct"] = reg.groupby(["y", "pos"]).role.rank(pct=True)
    reg = reg.merge(value_at(v, reg[["player_id", "y"]]), on=["player_id", "y"], how="left")
    reg["val_pct"] = reg.groupby(["y", "pos"]).market_value_in_eur.rank(pct=True)
    nxt = s[["player_id", "y", "minutes"]].merge(reg[["player_id", "y", "role_pct"]], on=["player_id", "y"], how="left")
    nxt = nxt.rename(columns={"minutes": "minutes_1", "role_pct": "role_pct_1"}).assign(y=lambda d: d.y - 1)
    p = reg.merge(nxt, on=["player_id", "y"], how="left")
    p["minutes_1"] = p.minutes_1.fillna(0)
    p["O1"] = p.minutes_1 >= MIN_T
    p["G"] = p.O1 & (p.role_pct_1 >= 0.5)
    p["has_next"] = p.y < max(YEARS)
    vet, cheap = p.age.between(29, 31), p.val_pct <= 0.5
    p["V"] = vet & cheap & (p.role_pct >= 0.75)
    p["Vb"] = vet & cheap & (p.role_pct < 0.75)
    return p, match


def h1(h: pd.DataFrame, n: int = 3000, seed: int = 0):
    d = h[h.market_value_in_eur.notna() & (h.age.between(25, 28) | h.age.between(29, 31))].copy()
    d["vet"] = d.age >= 29
    d["bin"] = d.groupby("y").market_value_in_eur.transform(lambda x: pd.qcut(np.log(x + 1).rank(method="first"), 10, labels=False))

    def stat(x):
        g = x.groupby(["y", "bin", "vet"]).G.mean().unstack()
        w = x[x.vet].groupby(["y", "bin"]).size()
        g = g.dropna()
        w = w.reindex(g.index).fillna(0)
        return float(((g[True] - g[False]) * w).sum() / w.sum())

    rng = np.random.default_rng(seed)
    est = stat(d)
    bs = [stat(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(n)]
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(d.vet.sum()), int((~d.vet).sum())


def tri(lo, hi):
    return "맞음" if lo > 0 else ("틀림" if hi < 0 else "불확실")


def main():
    ledger.append([dict(kind="football_hypothesis", key="scout_v2_rules", payload=RULES, model="scout_v2", evidence="과거 검증·선발 전 고정")])
    p, match = build()
    hist = p[p.has_next]
    e1, lo1, hi1, nv, npr = h1(hist)
    a = hist[hist.V].O1.astype(float).to_numpy(); b = hist[hist.Vb].O1.astype(float).to_numpy()
    ma, mb, lo2, hi2 = diff_ci(a, b)
    res = dict(H1=dict(diff=round(e1, 4), ci=[round(lo1, 4), round(hi1, 4)], n=[nv, npr], verdict=tri(lo1, hi1)),
               H2=dict(selected=round(ma, 4), baseline=round(mb, 4), n=[len(a), len(b)], ci=[round(lo2, 4), round(hi2, 4)], verdict=tri(lo2, hi2)))
    ledger.append([dict(kind="football_result", key="scout_v2_backtest", payload=res, model="scout_v2",
                        evidence=f"understat+트랜스퍼마크트, 선발 2017/18~2024/25, 연결률 {match:.0%}")])
    # 나이별 가격 대비 좋은 시즌 (설명용)
    by_age = hist[hist.market_value_in_eur.notna()].assign(ag=lambda d: pd.cut(d.age, [17, 22, 25, 28, 31, 40], labels=["~22", "23~25", "26~28", "29~31", "32~"]))
    tab = by_age.groupby("ag", observed=True).agg(n=("G", "size"), good=("G", "mean"), med=("market_value_in_eur", "median"))
    tab_md = "\n".join(f"| {i} | {int(r.n)} | {r.good:.0%} | €{r.med / 1e6:.1f}m | {r.good / (r.med / 1e7):.2f} |" for i, r in tab.iterrows())

    cur = p[p.y == max(YEARS)]
    picks = cur[cur.V].sort_values("role_pct", ascending=False)
    base = cur[cur.Vb]
    payload = dict(season="2026-27", picks=[dict(player=r.name, team=r.team, league=r.lg, age=int(r.age), pos=r.pos, role_pct=round(float(r.role_pct), 3),
                                                 value_eur=int(r.market_value_in_eur), contract_until=str(r.contract_expiration_date)[:10])
                                            for r in picks.itertuples()],
                   baseline_ids=sorted(base.player_id.astype(int).astype(str).tolist()),
                   forecast=dict(claim="선발군의 2026/27 1500분+ 비율이 기준선보다 높다", confidence=0.6 if res["H2"]["verdict"] == "맞음" else 0.55),
                   scoring="2027년 여름, understat 2026/27 기록으로 1500분+ · 좋은 시즌(역할 상위 절반) 채점")
    n_new = ledger.append([dict(kind="scout_pick", key="2026-27:veteran_v2", payload=payload, model="scout_v2",
                                evidence="2025/26 자료 + 2026년 7월 이전 시장가치만 사용, 2026/27 출전 시간 보지 않음")])
    ok, n_rows, _ = ledger.verify()
    tbl = "\n".join(f"| {i} | {r.name} | {int(r.age)} | {r.pos} | {r.team} | {r.role_pct:.0%} | €{r.market_value_in_eur / 1e6:.1f}m | {str(r.contract_expiration_date)[:7] if isinstance(r.contract_expiration_date, str) else '–'} |"
                    for i, r in enumerate(picks.itertuples(), 1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scout_v2.md").write_text(f"""# 셈법 스카우트 v2: 가격(시장가치)을 붙인 베테랑 검증

자료: understat 선수 시즌 (5대 리그 2017/18~2025/26) + transfermarkt-datasets (생년월일·시장가치·계약, 2026-07-06 스냅샷).
1500분+ 선수의 트랜스퍼마크트 연결률 {match:.0%}. 가설·규칙은 실행 전에 장부 고정.
'싸다' = 같은 시즌·포지션 1500분+ 선수 중 시장가치 하위 절반 (v0·v1의 '하위 절반 팀' 대리 대신 가격 자체).
'좋은 시즌' = 다음 시즌 1500분+ 이고 역할 지표 상위 절반.

## 가설 판정 (선발 2017/18~2024/25)
| 가설 | 결과 | 판정 |
|---|---|---|
| H1 같은 가격대에서 29~31세가 25~28세보다 좋은 시즌 비율이 높다 (시장이 나이를 과하게 깎음, 자신감 50%) | 차이 {e1:+.1%}p (95% {lo1:+.1%} ~ {hi1:+.1%}), 베테랑 {nv} / 25~28세 {npr} | {res['H1']['verdict']} |
| H2 싸고 역할 좋은 베테랑이 싸고 역할 평범한 베테랑보다 다음 시즌 1500분+ 비율이 높다 (자신감 60%) | {ma:.0%} (n={len(a)}) vs {mb:.0%} (n={len(b)}), 차이 {ma - mb:+.1%}p (95% {lo2:+.1%} ~ {hi2:+.1%}) | {res['H2']['verdict']} |

## 나이대별: 가격 대비 좋은 시즌 (설명용, 판정 아님)
| 나이 | 선수-시즌 | 다음 시즌 좋은 시즌 비율 | 시장가치 중앙값 | 좋은 시즌 비율 ÷ (중앙값/€10m) |
|---|---|---|---|---|
{tab_md}

## 2026/27 선발 명단 ({len(picks)}명, 기준선 {len(base)}명) — 장부 등록 {'완료' if n_new else '이미 있음(같은 내용)'}
| # | 선수 | 나이 | 포지션 | 소속 (2025/26) | 역할 백분위 | 시장가치 (2026년 여름 전) | 계약 만료 |
|---|---|---|---|---|---|---|---|
{tbl}

- 계약 만료가 2027년이면 다가오는 여름에 자유계약(이적료 0)으로 데려올 수 있는 후보. 계약 정보는 2026년 7월 스냅샷 기준이라 그 뒤 연장됐을 수 있음
- 예측: "선발군의 2026/27 1500분+ 비율이 기준선보다 높다" (자신감 {payload['forecast']['confidence']:.0%}). 채점 2027년 여름
- 장부 전체 {n_rows}줄, 체인 {'정상' if ok else '어긋남'}

## 한계
- 시장가치는 트랜스퍼마크트 사용자 추정치이지 실제 이적료가 아님. 연봉 없음
- 이름 연결: 같은 이름이 여럿이면 버림. 연결 안 된 선수는 빠짐
- DF 역할은 수비가 아니라 빌드업 관여로 잼
- 2026/27에 5대 리그를 떠나면(사우디 등) 0분으로 셈
""", encoding="utf-8")
    print((OUT / "scout_v2.md").read_text())


if __name__ == "__main__":
    main()
