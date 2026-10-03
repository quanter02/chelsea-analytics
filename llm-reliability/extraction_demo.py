"""한계효용 기반 추출 데모: 같은 사건에 대한 k번째 영상의 가치, 정책 비교, 채널 수에 따른 순가치.

    python extraction_demo.py     # output/extraction/ 에 report.md 와 차트 생성
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from llmrel import extraction as X
from llmrel.report import AQUA, BLUE, GRAY, INK, MUTED, ORANGE, PAPER, VIOLET, YELLOW, md_table, plt


def policies_for(seed):
    ev, _, vids = X.simulate_feed(seed)
    return [X.run(ev, vids, p)[0] | {"seed": seed} for p in X.POLICIES]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "output" / "extraction"))
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    with Pool() as pool:
        res = pd.DataFrame([r for rs in pool.map(policies_for, range(a.seeds)) for r in rs])
    summ = res.groupby("policy", sort=False)[["extracted", "extract_cost", "verify_cost", "decision_loss", "total", "missed"]].mean().reset_index()
    wins = (res.pivot(index="seed", columns="policy", values="total").idxmin(axis=1) == "한계효용 기준").mean()

    ev, ch, vids = X.simulate_feed(0)
    _, log = X.run(ev, vids, X.Policy("기록", extract="all", copy_aware=True), log_mu=True)
    rel = set(ev[ev.relevant].event)
    lg = log[log.event.isin(rel) & ~log.dup]
    hi = lg.value > lg.value.quantile(0.8)
    curve = pd.DataFrame({"중요도 상위 20% 사건": lg[hi].groupby("k").mu.mean(),
                          "나머지 사건": lg[~hi].groupby("k").mu.mean()}).loc[:6]
    dup_share = log[log.event.isin(rel)].dup.mean()
    greedy = X.greedy_channels(ev, ch, vids, X.POLICIES[-1])

    fig, ax = plt.subplots(1, 3, figsize=(19, 6), dpi=150)
    a1 = ax[0]
    for col, c in zip(curve.columns, (BLUE, ORANGE)):
        a1.plot(curve.index, curve[col], color=c, lw=2, marker="o", ms=6, label=col)
    a1.axhline(X.EXTRACT_USD, color=INK, ls="--", lw=1.2)
    a1.text(1, X.EXTRACT_USD * 0.8, f"추출 비용 ${X.EXTRACT_USD}  (이 선 아래면 추출 안 함)", ha="left", va="top", fontsize=9, color=INK)
    a1.set_yscale("log"); a1.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("$%g"))
    a1.set_xlabel("같은 사건에 대한 k번째 영상 (복제 채널 제외)"); a1.set_ylabel("영상 1개를 더 읽는 한계효용 (로그축)")
    a1.set_title("① 같은 사건, 영상이 쌓일수록 한계효용 체감"); a1.legend(loc="upper right")

    a2 = ax[1]; d = summ.iloc[::-1]; left = 0
    for col, lab, c in (("decision_loss", "판단 손실", ORANGE), ("extract_cost", "추출 비용", BLUE), ("verify_cost", "검증 비용", GRAY)):
        a2.barh(d.policy, d[col], left=left, color=c, label=lab, height=0.6, edgecolor=PAPER, linewidth=2); left = left + d[col]
    for i, (_, r) in enumerate(d.iterrows()):
        a2.text(r.total + 1, i, f"${r.total:.0f}  ({r.extracted:.0f}개 추출)", va="center", fontsize=9, color=INK)
    a2.set_xlim(0, d.total.max() * 1.35); a2.set_xlabel(f"60일 총비용 (USD, 시드 {a.seeds}개 평균)")
    a2.set_title("② 추출 정책 비교"); a2.legend(ncol=3, loc="lower right", bbox_to_anchor=(1, 1.02))

    a3 = ax[2]
    a3.bar(greedy.n, greedy.marginal, color=[BLUE if m > 0 else ORANGE for m in greedy.marginal], width=0.6)
    for _, r in greedy.iterrows():
        a3.text(r.n, max(r.marginal, 0) + 0.5, r.channel, ha="center", fontsize=8, color=INK, rotation=90, va="bottom")
    a3.axhline(0, color=INK, lw=1); a3.set_ylim(top=greedy.marginal.max() * 1.18)
    a3.set_xlabel("추가한 채널 수 (가치가 큰 순서)"); a3.set_ylabel("채널 1개 추가로 늘어난 순가치 (USD)")
    a3.set_title("③ 채널을 늘릴수록 한계효용 체감 (시드 0)")
    fig.tight_layout(); fig.savefig(out / "extraction.png", facecolor=PAPER); plt.close(fig)

    copies = ", ".join(f"{c.channel}→{c.copies}" for c in ch.itertuples() if c.copies)
    usd = "${:.2f}"
    md = [
        "# 한계효용 기반 추출 (가상 데이터)",
        "",
        "규칙은 하나입니다: **다음 한 단위의 한계효용 > 그 단위의 비용**일 때만 처리합니다.",
        "",
        "| 단계 | 한 단위 | 한계효용 | 비용 |",
        "|---|---|---|---|",
        "| 영상 추출 | 새 영상 1개 | 관련 확률 × 그 영상이 줄여줄 불확실성 (중요도 × p(1-p)의 기대 감소) | $0.05 |",
        "| 원문 검증 | 사건 1개 | 검증 성공률 × 남은 불확실성 전부 | $0.40 |",
        "| 채널 구독 | 채널 1개 | 그 채널을 넣었을 때 늘어나는 순가치 | (감시 비용) |",
        "",
        f"설정: 60일, 업데이트 사건 300개(절반이 관심 주제, 20%는 루머), 채널 12개 중 3개는 다른 채널을 베낌({copies}).",
        "",
        "![한계효용 기반 추출](extraction.png)",
        "",
        f"## 정책 비교 (시드 {a.seeds}개 평균)",
        "",
        md_table(summ.rename(columns={"policy": "정책", "extracted": "추출한 영상", "extract_cost": "추출 비용", "verify_cost": "검증 비용",
                                      "decision_loss": "판단 손실", "total": "총비용", "missed": "놓친 사실 사건"}),
                 {"추출한 영상": "{:.0f}", "추출 비용": usd, "검증 비용": usd, "판단 손실": usd, "총비용": usd, "놓친 사실 사건": "{:.1f}"}),
        "",
        f"- 한계효용 기준이 총비용 최저였던 시드: {wins:.0%}",
        f"- 관심 사건에 대한 영상 중 복제 채널 영상 비율: {dup_share:.0%} (한계효용 0으로 처리)",
        "",
        "## k번째 영상의 평균 한계효용 (시드 0, 복제 제외)",
        "",
        md_table(curve.reset_index().rename(columns={"k": "k번째"}), {c: "${:.3f}" for c in curve.columns}),
        "",
        "## 채널 추가 순서와 한계 순가치 (시드 0)",
        "",
        md_table(greedy.merge(ch[["channel", "reliability", "coverage", "copies"]], on="channel")
                 .rename(columns={"n": "순서", "channel": "채널", "net_value": "누적 순가치", "marginal": "한계 순가치",
                                  "reliability": "신뢰도", "coverage": "관심 사건 커버율", "copies": "베끼는 대상"}),
                 {"누적 순가치": usd, "한계 순가치": usd, "신뢰도": "{:.2f}", "관심 사건 커버율": "{:.2f}"}),
        "",
    ]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(summ.round(2).to_string()); print("wins", wins, "dup share", round(dup_share, 3))
    print(curve.round(3)); print(greedy.round(2).to_string())


if __name__ == "__main__":
    main()
