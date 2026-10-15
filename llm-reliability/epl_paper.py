"""가상 베팅 장부 실행 (돈은 걸지 않음).

    python epl_paper.py            # 규칙 고정 확인 → 과거 백테스트 → 지난 가상 베팅 채점 → 다가오는 경기 가상 베팅 등록
    python epl_paper.py --no-fetch # 받아 둔 자료로만

경기 라운드 전(보통 금요일)에 한 번 돌린다. 다가오는 경기 배당은 football-data.co.uk fixtures.csv.
한 경기는 한 번만 등록하고, 등록 뒤에는 배당이 바뀌어도 고치지 않는다.
결과 → output/epl_paper/report.md
"""
from __future__ import annotations

import datetime as dt
import subprocess
import sys
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from llmrel import epl, ledger, market, paper

ROOT = Path(__file__).parent
FIX = ROOT / "data_epl" / "odds" / "fixtures.csv"
OUT = ROOT / "output" / "epl_paper"
SEASON = "2026-27"


def fetch() -> None:
    subprocess.run(["curl", "-sSL", "-f", "-m", "60", "-o", str(FIX), "https://www.football-data.co.uk/fixtures.csv"], check=True)
    subprocess.run(["curl", "-sSL", "-f", "-m", "60", "-o", str(market.ODDS / "E0_2627.csv"),
                    "https://www.football-data.co.uk/mmz4281/2627/E0.csv"], check=True)


def upcoming(now: dt.datetime) -> pd.DataFrame:
    if not FIX.exists():
        return pd.DataFrame()
    x = pd.read_csv(FIX, encoding="utf-8-sig", on_bad_lines="skip")
    x = x[x.Div == "E0"]
    if x.empty:
        return pd.DataFrame()
    ko = pd.to_datetime(x.Date + " " + x.Time.fillna("12:00"), format="%d/%m/%Y %H:%M", errors="coerce").dt.tz_localize(ZoneInfo("Europe/London"))
    out = pd.DataFrame(dict(date=ko.dt.strftime("%Y-%m-%d"), kickoff=ko, home=x.HomeTeam.replace(market.NAMES), away=x.AwayTeam.replace(market.NAMES)))
    for s in paper.SIDES:
        out["o" + s] = pd.to_numeric(x["B365" + s], errors="coerce")
    return out[out.kickoff > now]


def main() -> None:
    warnings.filterwarnings("ignore")
    note = "받기 생략 (--no-fetch)"
    if "--no-fetch" not in sys.argv:
        try:
            fetch(); note = "fixtures.csv · E0_2627.csv 받음"
        except Exception as e:
            note = f"받기 실패: {e}"
    # 1. 규칙은 결과보다 먼저 (같은 내용이면 장부가 건너뜀, 바뀌면 새 줄이 남아 이력이 공개됨)
    ledger.append([dict(kind="paper_rules", key=paper.RULES["version"], payload=paper.RULES, model="paper",
                        evidence="백테스트·실전 결과 보기 전 고정")])
    rows = ledger.read()
    rules_at = next(r["recorded_at"] for r in rows if r["kind"] == "paper_rules" and r["key"] == paper.RULES["version"])

    # 2. 과거 백테스트 (규칙 고정 후, 조정 없음). 우리 확률은 경기 전까지 자료만 쓴 온라인 모델
    df = epl.load()
    res = epl.run(df, **epl.XG_MODE)
    hist = res[res.played].merge(paper.pre_and_close(), on=["date", "home", "away"], how="inner").dropna(subset=["oH", "cH"])
    back = {}
    for split in ["val", "test"]:
        s = hist[hist.season.isin(epl.SPLITS[split])]
        back[split] = (len(s), paper.summary(paper.settle(paper.pick(s))))

    # 3. 지난 가상 베팅 채점
    bets = {r["key"]: r for r in rows if r["kind"] == "paper_bet"}
    settled = {r["key"] for r in rows if r["kind"] == "paper_bet_result"}
    live = res[(res.season == SEASON) & res.played][["date", "home", "away", "result"]]
    close = paper.pre_and_close()[["date", "home", "away", "cH", "cD", "cA"]]
    done = live.merge(close, on=["date", "home", "away"], how="inner").dropna(subset=["cH"])
    new_res = []
    for r in done.itertuples():
        k = f"{r.date}|{r.home}|{r.away}"
        if k in bets and k not in settled:
            b = bets[k]["payload"]
            one = paper.settle(pd.DataFrame([dict(side=b["side"], odds=b["odds"], result=r.result, cH=r.cH, cD=r.cD, cA=r.cA)])).iloc[0]
            new_res.append(dict(kind="paper_bet_result", key=k, model="paper",
                                payload=dict(side=b["side"], odds=b["odds"], result=r.result, pnl=round(float(one.pnl), 4), clv=round(float(one.clv), 4)),
                                evidence="실제 결과 + 평균 마감 배당"))
    n_settled = ledger.append(new_res)

    # 4. 다가오는 경기 등록 (킥오프 전, 한 경기 한 번)
    now = dt.datetime.now(dt.timezone.utc)
    up = upcoming(now)
    n_new, placed = 0, pd.DataFrame()
    if len(up):
        mode, params = epl.current_mode(df)
        probs = epl.run(df, **params)
        probs = probs[(probs.season == SEASON) & ~probs.played][["date", "home", "away", "pH", "pD", "pA"]]
        cand = up.merge(probs, on=["date", "home", "away"], how="inner")
        placed = paper.pick(cand)
        placed = placed[[f"{r.date}|{r.home}|{r.away}" not in bets for r in placed.itertuples()]]
        n_new = ledger.append([dict(kind="paper_bet", key=f"{r.date}|{r.home}|{r.away}", model=f"paper v1 · {mode}",
                                    payload=dict(side=r.side, odds=float(r.odds), p=round(float(r.p), 4), ev=round(float(r.ev), 4),
                                                 kickoff=r.kickoff.isoformat(), stake=1.0),
                                    evidence="킥오프 전 등록, 실제 돈 없음") for r in placed.itertuples()])
    ok, n_rows, last = ledger.verify()

    # 5. 이번 시즌 누적
    rows = ledger.read()
    rs = pd.DataFrame([r["payload"] for r in rows if r["kind"] == "paper_bet_result"])
    live_sum = paper.summary(rs)
    pending = [r for r in rows if r["kind"] == "paper_bet" and r["key"] not in {x["key"] for x in rows if x["kind"] == "paper_bet_result"}]

    def line(name, n_match, s):
        if not s.get("n"):
            return f"| {name} | {n_match} | 0 | – | – | – | – |"
        return (f"| {name} | {n_match} | {s['n']} | {s['avg_odds']:.2f} | {s['clv']:+.1%} ({s['clv_lo']:+.1%} ~ {s['clv_hi']:+.1%}) | "
                f"{s['roi']:+.1%} ({s['roi_lo']:+.1%} ~ {s['roi_hi']:+.1%}) | {s['verdict']} |")

    side = {"H": "홈승", "D": "무", "A": "원정승"}
    pend = "\n".join(f"| {r['key'].split('|')[0]} | {r['key'].split('|')[1]} vs {r['key'].split('|')[2]} | "
                     f"{side[r['payload']['side']]} | {r['payload']['odds']:.2f} | {r['payload']['p']:.0%} | {r['payload']['ev']:+.0%} |" for r in pending) or "| – | 등록된 경기 없음 | | | | |"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text(f"""# 가상 베팅 장부 (돈 없음, {dt.date.today()})

블룸 방식 1단계: 경기 전에 '이 배당이면 건다'를 장부에 고정하고, 시즌 뒤에 채점한다. **실제 돈은 걸지 않는다.**
규칙 v1 (장부 고정 {rules_at[:10]}): 경기 전 Bet365 배당, 기대 이득 ≥ 5%, 배당 ≤ 6.0, 경기당 한 쪽, 1단위 고정.
주 지표 **CLV** = 받은 배당 × 마감 공정 확률 − 1 (양수면 시장이 나중에 우리 쪽으로 움직였다는 뜻 = 우위의 증거). ROI는 운이 크게 섞임.

- 자료: {note}
- 장부: 이번에 채점 {n_settled}줄, 등록 {n_new}줄 추가, 전체 {n_rows}줄, 체인 {'정상' if ok else '어긋남'}

## 과거 백테스트 (규칙 고정 후 그대로 적용)
| 구간 | 경기 | 베팅 | 평균 배당 | CLV (95%) | ROI (95%) | 우위 판정 |
|---|---|---|---|---|---|---|
{line('검증 2021/22~23/24', *back['val'])}
{line('시험 2024/25~25/26', *back['test'])}

## 이번 시즌 실전 (2026/27, 채점 끝난 것)
| 구간 | 경기 | 베팅 | 평균 배당 | CLV (95%) | ROI (95%) | 우위 판정 |
|---|---|---|---|---|---|---|
{line('2026/27 누적', live_sum.get('n', 0), live_sum)}

## 채점 기다리는 가상 베팅
| 날짜 | 경기 | 쪽 | 배당 | 우리 확률 | 기대 이득 |
|---|---|---|---|---|---|
{pend}

## 읽는 법
- 이 장부의 목적은 돈이 아니라 **우위가 있는지 증명**하는 것. 우위가 없으면 '틀림'이 나오는 게 정상이고, 그 기록도 그대로 공개한다
- 새 아이디어는 `ideas/match_notes.md`에 적고, 숫자로 바꿀 수 있으면 아이디어 실험대(`epl_idea.py`)를 먼저 통과해야 이 장부의 규칙(v2)으로 들어온다
""", encoding="utf-8")
    print((OUT / "report.md").read_text())


if __name__ == "__main__":
    main()
