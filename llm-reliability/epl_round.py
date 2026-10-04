"""매 라운드 실행: 최신 결과 받기 → 다음 경기 예측 기록 → 지난 예측·Opta 비교 채점 → 요약.

    python epl_round.py              # 결과 갱신 + 다음 7일 경기 예측 추가 + 채점
    python epl_round.py --no-fetch   # 받은 자료로만

기록 파일
  data_epl/benchmark_opta.csv  같은 경기의 우리 예측과 Opta 확률을 나란히 (Opta는 사용자가 제공)
  data_epl/xg_manual.csv       이번 시즌 경기 xG (있으면 자동으로 xG 모드)
"""
from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from llmrel import epl

ROOT = Path(__file__).parent
BENCH = ROOT / "data_epl" / "benchmark_opta.csv"
OUT = ROOT / "output" / "epl_round"
SEASON = "2026-27"


def fetch() -> None:
    url = f"https://raw.githubusercontent.com/openfootball/football.json/master/{SEASON}/en.1.json"
    subprocess.run(["curl", "-s", "-f", "-o", str(ROOT / "data_epl" / f"en1_{SEASON}.json"), url], check=True)


def main() -> None:
    if "--no-fetch" not in sys.argv:
        fetch()
    df = epl.load()
    mode, params = epl.current_mode(df)
    res = epl.run(df, **params)
    today = dt.date.today()
    last_played = df[(df.season == SEASON) & df.played].date.max()

    # 다음 경기 예측을 비교 기록에 추가 (같은 경기·같은 출처가 있으면 최신으로 교체)
    nxt = res[(res.season == SEASON) & ~res.played & (pd.to_datetime(res.date) <= pd.Timestamp(today) + pd.Timedelta(days=10))]
    b = pd.read_csv(BENCH) if BENCH.exists() else pd.DataFrame()
    src = f"우리 모델 ({mode})"
    new = pd.DataFrame(dict(date=nxt.date, home=nxt.home, away=nxt.away, source=src, pH=nxt.pH.round(3), pD=nxt.pD.round(3), pA=nxt.pA.round(3),
                            recorded_on=today.isoformat(), note=f"결과 반영 {last_played}까지"))
    if len(b):
        b = b[~(b.source.str.startswith("우리 모델") & b.set_index(["date", "home", "away"]).index.isin(new.set_index(["date", "home", "away"]).index))]
    b = pd.concat([b, new], ignore_index=True).sort_values(["date", "home", "source"])
    b.to_csv(BENCH, index=False)

    # 채점
    sc = epl.score_benchmark(BENCH, df)
    sc["who"] = np.where(sc.source.str.startswith("우리 모델"), "우리 모델", sc.source.str.split(" ").str[0])
    done = sc[sc.status == "scored"]
    both = done.groupby(["date", "home", "away"]).who.nunique()
    pairs = both[both >= 2].index
    paired = done.set_index(["date", "home", "away"]).loc[pairs].reset_index() if len(pairs) else done.iloc[0:0]
    summ = paired.groupby("who").rps.agg(["mean", "count"]) if len(paired) else pd.DataFrame()

    OUT.mkdir(parents=True, exist_ok=True)
    L = [f"# EPL 라운드 기록 ({today})", "", f"- 모드: **{mode}** (이번 시즌 xG 붙은 경기 {int(df[(df.season == SEASON) & df.played].hxg.notna().sum())}개)",
         f"- 결과 반영: {last_played}까지, 이번 시즌 {int(df[(df.season == SEASON) & df.played].shape[0])}경기", "",
         "## Opta와 같은 경기 비교 (채점 끝난 것만, RPS 낮을수록 좋음)", "",
         (summ.round(4).to_markdown() if len(summ) else "아직 채점된 비교 경기가 없습니다."), "",
         "## 기록된 예측", "", sc[["date", "home", "away", "source", "pH", "pD", "pA", "status", "rps"]].round(3).to_markdown(index=False)]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:6])); print(sc[sc.status == "pending"][["date", "home", "away", "source", "pH", "pD", "pA"]].to_string(index=False))


if __name__ == "__main__":
    main()
