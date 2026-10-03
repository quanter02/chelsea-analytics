"""온라인 시나리오: 공지된 버전 교체 → 조용한 성능 저하 → 신규 모델 등장. 세 전략 비교.

    python online_demo.py              # 시드 8개, output/online/ 에 report.md 와 차트 생성
    python online_demo.py --seeds 2    # 빠르게 확인
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from llmrel import db, online, profiles as P, simulate as S
from llmrel.online_report import write


def one(args):
    seed, name, n = args
    con = db.connect()
    preds, outcomes = S.simulate(n, days=S.DRIFT_DAYS, seed=seed, models=S.DRIFT_MODELS, label_all=True)
    db.insert(con, "predictions", preds)
    db.insert(con, "outcomes", outcomes)
    db.insert(con, "loss_matrix", S.loss_matrix())
    _, scored = P.load(con)
    r, al = online.run(scored, S.HUMAN_REVIEW_USD, name, end_day=S.DRIFT_DAYS)
    return r.assign(seed=seed), al.assign(strategy=name, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "output" / "online"))
    ap.add_argument("--n", type=int, default=4000, help="시드당 판단 건수")
    ap.add_argument("--seeds", type=int, default=8)
    a = ap.parse_args()

    jobs = [(s, name, a.n) for s in range(1, a.seeds + 1) for name in online.STRATEGIES]
    with Pool() as pool:
        res = pool.map(one, jobs)
    runs = pd.concat([r for r, _ in res]); alarms = pd.concat([al for _, al in res if len(al)])
    for name, g in runs.groupby("strategy", sort=False):
        print(f"{online.STRATEGIES[name]['label']:<24} 평균 비용/건 ${g.cost.mean():.2f}")
    print(f"리포트: {write(Path(a.out), runs, alarms, S.DRIFT_EVENTS)}")


if __name__ == "__main__":
    main()
