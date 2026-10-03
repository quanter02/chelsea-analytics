"""온라인 시나리오 리포트."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import online
from .report import AQUA, BLUE, GRAY, INK, MUTED, ORANGE, PAPER, VIOLET, YELLOW, md_table, plt

COLOR = {"static": ORANGE, "cumulative": BLUE, "adaptive": AQUA}
MODEL_COLOR = {"large": BLUE, "mid-bold": ORANGE, "mid-cautious": AQUA, "small-fast": YELLOW, "newcomer": VIOLET, "사람": GRAY}
PERIODS = [(90, 110, "정상 (90~110일)"), (110, 150, "조용한 성능 저하 후 (110~150일)"), (150, 240, "신규 모델 등장 후 (150~240일)")]


def write(out: Path, runs: pd.DataFrame, alarms: pd.DataFrame, events: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    label = {k: v["label"] for k, v in online.STRATEGIES.items()}
    seeds = runs.seed.nunique()

    fig, ax = plt.subplots(2, 1, figsize=(12, 8.5), dpi=150, gridspec_kw={"height_ratios": [2.2, 1]})
    a = ax[0]
    for name, g in runs.groupby("strategy", sort=False):
        m = g.groupby("day").cost.mean()
        a.plot(m.index + 3.5, m.values, color=COLOR[name], lw=2, marker="o", ms=4, label=label[name])
    for i, (d, txt) in enumerate((d, t) for d, t in events.items() if d >= runs.day.min()):
        a.axvline(d, color=MUTED, ls=":", lw=1)
        a.text(d + 1, 0.97 - 0.06 * i, txt, transform=a.get_xaxis_transform(), fontsize=9, color=INK, va="top")
    if len(alarms):
        ad = alarms[alarms.strategy == "adaptive"].groupby("day").size()
        a.scatter(ad.index, np.full(len(ad), 0.3), marker="v", s=30 + 20 * ad.values, color=AQUA, zorder=4,
                  transform=a.get_xaxis_transform(), label="변화 감지 경보 (적응형, 크기=시드 수)")
    a.set_title(f"주별 판단 1건당 실제 비용 (USD, 시드 {seeds}개 평균)")
    a.set_ylabel("비용/건 (USD)"); a.legend(loc="lower left", fontsize=9)

    a = ax[1]                                     # 시드 1의 정책 타임라인 (첫 단계 모델)
    s1 = runs[runs.seed == runs.seed.min()]
    for i, (name, g) in enumerate(s1.groupby("strategy", sort=False)):
        for _, r in g.iterrows():
            a.barh(i, 7, left=r.day, color=MODEL_COLOR.get(r.first_model, GRAY), edgecolor=PAPER, linewidth=1, height=0.6)
    a.set_yticks(range(3)); a.set_yticklabels([label[k] for k in s1.strategy.unique()])
    a.invert_yaxis(); a.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in MODEL_COLOR.values()]
    a.legend(handles, list(MODEL_COLOR), ncol=6, loc="lower right", bbox_to_anchor=(1, 1.02), fontsize=9)
    a.set_title("정책의 첫 단계 모델 (시드 1)", pad=22)
    a.set_xlabel("일차")
    for x in (a, ax[0]):
        x.set_xlim(runs.day.min(), runs.day.max() + 7)
    fig.tight_layout(); fig.savefig(out / "online.png", facecolor=PAPER); plt.close(fig)

    per = []
    for lo, hi, name in PERIODS:
        g = runs[(runs.day >= lo) & (runs.day < hi)].groupby("strategy", sort=False)
        per.append(pd.Series({label[k]: v.cost.mean() for k, v in g}, name=name))
    per = pd.DataFrame(per)
    per = per.T.assign(전체=runs.groupby("strategy", sort=False).cost.mean().rename(index=label)).reset_index(names="전략")
    seed_tot = runs.groupby(["strategy", "seed"]).cost.mean().unstack(0)
    wins = (seed_tot["adaptive"] < seed_tot["cumulative"]).sum()

    det = pd.DataFrame()
    if len(alarms):
        det = alarms[alarms.strategy == "adaptive"].groupby(["model", "direction"]).agg(
            경보_시드수=("seed", "nunique"), 첫경보_중앙값=("day", "median"),
            기준_확신오답률=("baseline", "mean"), 경보주_확신오답률=("this_week", "mean")).reset_index()

    usd = "${:.2f}"
    md = [
        "# 온라인 업데이트 시나리오 (가상 데이터)",
        "",
        "매주 그 시점까지 **정답이 확정된 데이터만** 보고 프로파일과 라우팅 정책을 갱신합니다. 정답은 평균 2주 늦게 확정됩니다.",
        "",
        "| 일차 | 사건 |", "|---|---|", *[f"| {d} | {t} |" for d, t in events.items()],
        "",
        "![온라인 비교](online.png)",
        "",
        f"## 기간별 판단 1건당 평균 비용 (시드 {seeds}개)",
        "",
        md_table(per, {c: usd for c in per.columns if c != "전략"}),
        "",
        f"- 적응형이 전체 누적보다 비용이 낮았던 시드: {wins}/{seeds}",
        "",
        "## 변화 감지 경보 (적응형)",
        "",
        md_table(det.rename(columns={"model": "모델", "direction": "방향"}),
                 {"첫경보_중앙값": "{:.0f}일", "기준_확신오답률": "{:.1%}", "경보주_확신오답률": "{:.1%}"}) if len(det) else "경보 없음",
        "",
    ]
    path = out / "report.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path
