"""가상 로그 생성 → DB 적재 → 프로파일 → 라우터 탐색 → 리포트.

    python run_demo.py                    # output/ 에 report.md 와 차트 생성
    python run_demo.py --db rel.sqlite    # DB 파일로 저장해서 직접 SQL 조회
"""
from __future__ import annotations

import argparse
from pathlib import Path

from llmrel import db, profiles as P, report, router as R, simulate as S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=":memory:")
    ap.add_argument("--out", default=str(Path(__file__).parent / "output"))
    ap.add_argument("--n", type=int, default=3000, help="가상 판단 건수")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    con = db.connect(a.db)
    preds, outcomes = S.simulate(a.n, seed=a.seed)
    db.insert(con, "predictions", preds)
    db.insert(con, "outcomes", outcomes)
    db.insert(con, "loss_matrix", S.loss_matrix())

    allp, scored = P.load(con)
    prof = P.outcome_profile(allp, scored)
    cal_tab, ece = P.calibration(scored)
    top, test, info = R.search(scored, S.loss_matrix(), S.HUMAN_REVIEW_USD)
    path = report.write(Path(a.out), prof=prof, cal_tab=cal_tab, ece=ece, abst=P.abstention_quality(scored),
                        corr=P.error_correlation(scored), eloss=P.expected_loss(scored), search_top=top, test=test, info=info)

    n_ce = con.execute("SELECT COUNT(*) FROM confident_errors").fetchone()[0]
    n_ab = con.execute("SELECT COUNT(*) FROM abstentions").fetchone()[0]
    n_pd = con.execute("SELECT COUNT(*) FROM pending").fetchone()[0]
    print(f"예측 {len(preds):,}건 | 확신 오답 {n_ce:,} | 기권 {n_ab:,} | 정답 대기 {n_pd:,}")
    print(f"검증 구간 최적 정책: {test.iloc[0].policy}  (건당 ${test.iloc[0].total:.2f})")
    print(f"리포트: {path}")


if __name__ == "__main__":
    main()
