"""실제 데이터 검증: StatsBomb 2015/16 4개 리그 실제 경기 1,517개.

예측 모델 5개가 경기 전에 승/무/패와 확신도를 기록하고, 실제 결과가 정답으로 붙는다.
이 구조에서 신뢰도 프레임워크가 '얼마나 정확히' 모델을 평가할 수 있는지 잰다.

    python real_data_demo.py      # 첫 실행 시 StatsBomb에서 경기 목록 다운로드 → output/realdata/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from llmrel import db, football as F, online, precision as Q, profiles as P, router as R
from llmrel.report import AQUA, BLUE, GRAY, INK, MUTED, ORANGE, PAPER, VIOLET, YELLOW, md_table, plt

COLORS = {"elo@v1": BLUE, "elo-cautious@v1": AQUA, "elo-sharp@v1": ORANGE, "home-always@v1": GRAY, "poisson-form@v1": VIOLET}
PAIRS = [("elo-cautious@v1", "elo@v1"), ("elo@v1", "home-always@v1"), ("elo@v1", "poisson-form@v1")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "output" / "realdata"))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    m = F.fetch_matches(); preds = F.predict(m)
    con = db.connect()
    db.insert(con, "predictions", preds.drop(columns="league")); db.insert(con, "outcomes", F.outcomes(m)); db.insert(con, "loss_matrix", F.loss_matrix())
    allp, sc = P.load(con)
    prof = P.outcome_profile(allp, sc); _, ece = P.calibration(sc); el = P.expected_loss(sc)

    # 1) 표본 수별 정밀도
    boot, pairs = Q.bootstrap(sc, pairs=PAIRS)

    # 2) 라우터: 앞 70% 경기로 정책 선택 → 뒤 30% 경기로 검증 (부트스트랩 신뢰구간)
    W = R.wide(sc); when = pd.to_datetime(W["when"]).astype("int64").to_numpy(); train = when <= np.quantile(when, 0.7)
    best = min(R.candidates(W["models"]), key=lambda p: R.evaluate(W, p, {}, F.ABSTAIN_LOSS, train)["total"])
    single = min(W["models"], key=lambda mm: R.evaluate(W, R.Policy(((mm, 0.0),)), {}, F.ABSTAIN_LOSS, train)["total"])
    picks = {f"최적 정책 (학습 구간에서 선택)": best, f"{single} 단독 (학습 구간 최적 단일)": R.Policy(((single, 0.0),)),
             "elo 단독": R.Policy((("elo", 0.0),)), "home-always 단독": R.Policy((("home-always", 0.0),)), "항상 판단 보류": R.Policy(())}
    rng = np.random.default_rng(0); rrows = []
    for k, p in picks.items():
        r = R.evaluate(W, p, {}, F.ABSTAIN_LOSS, ~train, per_input=True); v = r["per_input"].to_numpy()
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
        rrows.append(dict(name=k, policy=str(p), cost=v.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), abstain=r["human_share"]))
    rt = pd.DataFrame(rrows)
    d = (R.evaluate(W, best, {}, F.ABSTAIN_LOSS, ~train, per_input=True)["per_input"]
         - R.evaluate(W, R.Policy(((single, 0.0),)), {}, F.ABSTAIN_LOSS, ~train, per_input=True)["per_input"]).to_numpy()
    bd = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    rdiff = dict(mean=d.mean(), lo=np.percentile(bd, 2.5), hi=np.percentile(bd, 97.5), p_better=float((bd < 0).mean()))

    # 3) 온라인: 시즌을 주 단위로 진행
    origin = m.ts.min().normalize(); end = int((m.ts.max() - origin).days) + 1
    onl, n_alarm = {}, 0
    for s in online.STRATEGIES:
        r, al = online.run(sc, F.ABSTAIN_LOSS, s, start_day=42, end_day=end, origin=origin)
        onl[online.STRATEGIES[s]["label"]] = r.cost.mean(); n_alarm += len(al)
    n_weeks = len(r)

    # 4) 월별 정답률 (시즌이 지나며 모델이 배우는 모습)
    s2 = sc.assign(month=pd.to_datetime(sc.created_at).dt.to_period("M"))
    monthly = s2[s2.answer != "ABSTAIN"].groupby(["month", "model"]).outcome.apply(lambda o: (o == "correct").mean()).unstack()

    # ── 차트 ──
    fig, ax = plt.subplots(2, 2, figsize=(15, 10.5), dpi=150)
    a1 = ax[0, 0]
    for (pair, g), c in zip(pairs.groupby("pair", sort=False), (AQUA, BLUE, VIOLET)):
        a1.plot(g.N, g.same_order, color=c, lw=2, marker="o", ms=5, label=f"{pair.replace('@v1', '')} (차이 {abs(g.full_diff.iloc[0]):.3f})")
    a1.axhline(0.95, color=INK, ls="--", lw=1); a1.text(560, 0.935, "95% 확실히 구분", fontsize=9, color=INK, va="bottom")
    a1.set_xscale("log"); a1.set_xticks([100, 250, 500, 1000, 1517]); a1.set_xticklabels(["100", "250", "500", "1000", "1517"])
    a1.set_ylim(0.5, 1.02); a1.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    a1.set_title("① 두 모델 중 누가 나은지 맞게 가려낼 확률"); a1.set_xlabel("정답이 확정된 판단 수 (로그축)"); a1.legend(fontsize=9, loc="lower right")

    a2 = ax[0, 1]; e = boot[boot.metric == "ECE"]
    for mm, g in e.groupby("model"):
        a2.plot(g.N, g["mean"], color=COLORS[mm], lw=2, marker="o", ms=5, label=mm.replace("@v1", ""))
    a2.set_xscale("log"); a2.set_xticks([100, 250, 500, 1000, 1517]); a2.set_xticklabels(["100", "250", "500", "1000", "1517"])
    a2.set_title("② 측정된 보정 오차(ECE): 표본이 적으면 부풀려짐"); a2.set_xlabel("판단 수 (로그축)"); a2.set_ylabel("ECE 평균 (부트스트랩)")
    a2.legend(fontsize=9, ncol=2)

    a3 = ax[1, 0]
    for mm in monthly.columns:
        a3.plot(monthly.index.astype(str), monthly[mm], color=COLORS[mm], lw=2, marker="o", ms=4, label=mm.replace("@v1", ""))
    a3.text(0.01, 0.02, "elo-sharp는 elo와 답이 같아 선이 겹칩니다", transform=a3.transAxes, fontsize=9, color=MUTED)
    a3.set_title("③ 월별 정답률 (답한 경기 기준, 실제 시즌)"); a3.set_ylabel("정답률"); a3.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    a3.legend(fontsize=9, ncol=2); plt.setp(a3.get_xticklabels(), rotation=30)

    a4 = ax[1, 1]; rr = rt.iloc[::-1].reset_index(drop=True)
    a4.barh(rr.name, rr.cost, xerr=[rr.cost - rr.lo, rr.hi - rr.cost], color=[AQUA if "최적 정책" in n else BLUE for n in rr.name],
            height=0.55, error_kw=dict(ecolor=INK, lw=1.2, capsize=3))
    for i, r in rr.iterrows(): a4.text(r.hi + 0.004, i, f"{r.cost:.3f}", va="center", fontsize=9, color=INK)
    a4.set_xlim(0.38, max(rr.hi) + 0.05); a4.set_title(f"④ 라우터: 뒤 30% 경기에서 판단 1건당 손실 (95% 구간)")
    a4.set_xlabel("평균 손실 (틀림 1, 보류 0.5)")
    fig.tight_layout(); fig.savefig(out / "realdata.png", facecolor=PAPER); plt.close(fig)

    # ── 리포트 ──
    hw = boot.pivot_table(index=["metric", "model"], columns="N", values="half_width").reset_index()
    hw.columns = [str(c) for c in hw.columns]
    p = prof.merge(ece, left_on="model", right_index=True).merge(el[["model", "expected_loss"]], on="model")
    md = [
        "# 실제 데이터 검증: 실제 경기 결과로 신뢰도 프레임워크 시험",
        "",
        f"StatsBomb 공개 데이터, 2015/16 EPL·라리가·세리에 A·리그 1 **실제 경기 {len(m):,}개**. 예측 모델 5개가 매 경기 전에 그때까지의 결과만 보고 "
        "승/무/패와 확신도를 기록했고, 실제 결과를 정답으로 붙였습니다. 틀리면 손실 1, 판단 보류는 0.5입니다.",
        "",
        "| 모델 | 방식 |", "|---|---|",
        "| home-always | 항상 홈승. 확신도 = 리그 홈승률 |", "| elo | 엘로 레이팅 (시즌 첫 경기부터 학습) |",
        "| elo-cautious | elo와 같지만 최고 확률 < 0.5 이면 기권 |", "| elo-sharp | elo와 같은 답, 확신도만 일부러 부풀림 (과신 탐지 시험용) |",
        "| poisson-form | 최근 6경기 득실로 포아송 예측 |",
        "",
        "![실제 데이터 검증](realdata.png)",
        "",
        "## 1. 전체 표본 프로필",
        "",
        md_table(p[["model", "n_labeled", "correct_rate", "confident_error_rate", "abstain_rate", "accuracy_when_answering", "ECE", "expected_loss"]]
                 .rename(columns={"model": "모델", "n_labeled": "경기", "correct_rate": "정답", "confident_error_rate": "확신 오답", "abstain_rate": "기권",
                                  "accuracy_when_answering": "답했을 때 정답률", "expected_loss": "기대 손실"}),
                 {"정답": "{:.1%}", "확신 오답": "{:.1%}", "기권": "{:.1%}", "답했을 때 정답률": "{:.1%}", "ECE": "{:.3f}", "기대 손실": "{:.3f}"}),
        "",
        "## 2. 표본 수별 측정 정밀도 (95% 신뢰구간 반폭, 부트스트랩 400회)",
        "",
        md_table(hw, {c: "±{:.3f}" for c in hw.columns if c not in ("metric", "model")}),
        "",
        "### 두 모델 중 누가 나은지 가려낼 수 있나",
        "",
        md_table(pairs.rename(columns={"pair": "비교", "full_diff": "전체 표본 차이", "half_width": "차이의 ±", "same_order": "같은 결론 비율"}),
                 {"전체 표본 차이": "{:+.3f}", "차이의 ±": "±{:.3f}", "같은 결론 비율": "{:.0%}"}),
        "",
        "## 3. 라우터 (앞 70% 경기로 선택 → 뒤 30% 경기로 검증)",
        "",
        md_table(rt.rename(columns={"name": "정책 구분", "policy": "정책", "cost": "평균 손실", "lo": "하한", "hi": "상한", "abstain": "보류 비율"}),
                 {"평균 손실": "{:.3f}", "하한": "{:.3f}", "상한": "{:.3f}", "보류 비율": "{:.0%}"}),
        "",
        f"- 최적 정책 − 최적 단일 모델: {rdiff['mean']:+.4f} (95% 구간 {rdiff['lo']:+.4f} ~ {rdiff['hi']:+.4f}), 최적 정책이 더 나을 확률 {rdiff['p_better']:.0%}",
        "",
        "## 4. 온라인 갱신 (시즌을 주 단위로 진행)",
        "",
        md_table(pd.DataFrame([dict(전략=k, 평균손실=v) for k, v in onl.items()]), {"평균손실": "{:.3f}"}),
        "",
        f"- {n_weeks}주 × 3전략 동안 변화 감지 경보: {n_alarm}건",
        "",
    ]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(p[["model", "accuracy_when_answering", "ECE", "expected_loss"]].round(3).to_string())
    print(pairs.round(3).to_string()); print(rt.round(3).to_string()); print(rdiff); print(onl, "alarms", n_alarm, "weeks", n_weeks)


if __name__ == "__main__":
    main()
