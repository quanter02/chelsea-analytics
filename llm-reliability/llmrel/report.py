"""마크다운 리포트와 차트(PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

_KO = ("NanumGothic", "Noto Sans CJK KR", "AppleGothic", "Malgun Gothic")
_installed = {f.name for f in font_manager.fontManager.ttflist}

INK, MUTED, GRID, PAPER = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#a8a7a2"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, VIOLET]

plt.rcParams.update({
    "font.family": [f for f in _KO if f in _installed] + ["DejaVu Sans"],   # 한글 폰트가 없으면 글자가 깨질 수 있음
    "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "axes.titlelocation": "left", "axes.titlecolor": INK,
    "legend.frameon": False, "legend.fontsize": 9,
})


def md_table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) if pd.notna(r[c]) else "-" for c in cols) + " |")
    return "\n".join(out)


def chart_outcomes(prof: pd.DataFrame, path: Path) -> None:
    d = prof.sort_values("confident_error_rate")
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(d) + 1.6), dpi=150)
    left = 0
    for col, lab, c in (("correct_rate", "정답", BLUE), ("confident_error_rate", "확신 오답", ORANGE), ("abstain_rate", "기권", GRAY)):
        ax.barh(d.model, d[col], left=left, color=c, label=lab, height=0.6, edgecolor=PAPER, linewidth=2)
        left = left + d[col]
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(r.correct_rate / 2, i, f"{r.correct_rate:.0%}", va="center", ha="center", color="white", fontsize=9)
        ax.text(r.correct_rate + r.confident_error_rate / 2, i, f"{r.confident_error_rate:.0%}", va="center", ha="center", color="white", fontsize=9)
    ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("모델별 오답 구성 (정답이 확정된 예측 기준)")
    ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1, 1.02))
    fig.tight_layout(); fig.savefig(path, facecolor=PAPER); plt.close(fig)


def chart_calibration(tab: pd.DataFrame, path: Path, min_n: int = 20) -> None:
    tab = tab[tab.n >= min_n]                       # 표본이 적은 구간은 선을 왜곡하므로 제외
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    ax.plot([0.3, 1], [0.3, 1], color=MUTED, ls="--", lw=1, label="완벽한 보정")
    for (m, d), c in zip(tab.groupby("model"), SERIES):
        ax.plot(d.said, d.actual, color=c, lw=2, marker="o", ms=5, label=m)
    ax.set_xlabel("모델이 말한 확신도"); ax.set_ylabel("실제 정답률")
    ax.set_title(f"확신도 보정: 대각선 아래면 과신 (구간당 {min_n}건 이상)")
    ax.set_xlim(0.3, 1.02); ax.set_ylim(0.3, 1.02); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(path, facecolor=PAPER); plt.close(fig)


def chart_router(test: pd.DataFrame, path: Path) -> None:
    d = test.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(d) + 1.6), dpi=150)
    left = 0
    for col, lab, c in (("error_loss", "오답 피해", ORANGE), ("human_cost", "사람 검토", GRAY), ("call_cost", "모델 호출", BLUE)):
        ax.barh(d["정책 구분"], d[col], left=left, color=c, label=lab, height=0.6, edgecolor=PAPER, linewidth=2)
        left = left + d[col]
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(r.total + 0.1, i, f"${r.total:.2f}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("판단 1건당 총비용 (USD, 검증 구간)"); ax.set_title("라우팅 정책 비교")
    ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1, 1.02))
    fig.tight_layout(); fig.savefig(path, facecolor=PAPER); plt.close(fig)


def write(out: Path, *, prof, cal_tab, ece, abst, corr, eloss, search_top, test, info) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    chart_outcomes(prof, out / "outcomes.png")
    chart_calibration(cal_tab, out / "calibration.png")
    chart_router(test, out / "router.png")
    pct, usd, f2 = "{:.1%}", "${:.2f}", "{:.2f}"
    p = prof.merge(ece, left_on="model", right_index=True)
    md = [
        "# LLM 판단 신뢰도 리포트 (가상 데이터)",
        "",
        "> 시뮬레이션 데이터로 만든 예시 리포트입니다. 모델 이름과 수치는 실제 제품과 무관합니다.",
        "",
        "## 1. 오답 구성",
        "![오답 구성](outcomes.png)",
        "",
        md_table(p[["model", "n_labeled", "pending_share", "correct_rate", "confident_error_rate", "confident_error_rate_eb", "abstain_rate", "accuracy_when_answering", "ECE"]]
                 .rename(columns={"model": "모델", "n_labeled": "정답 확정", "pending_share": "정답 대기", "correct_rate": "정답",
                                  "confident_error_rate": "확신 오답", "confident_error_rate_eb": "확신 오답(축소 추정)",
                                  "abstain_rate": "기권", "accuracy_when_answering": "답했을 때 정답률", "ECE": "보정 오차(ECE)"}),
                 {"정답 대기": pct, "정답": pct, "확신 오답": pct, "확신 오답(축소 추정)": pct, "기권": pct, "답했을 때 정답률": pct, "보정 오차(ECE)": "{:.3f}"}),
        "",
        "- **확신 오답(축소 추정)**: 표본이 적은 모델의 비율을 전체 평균 쪽으로 당긴 값 (경험적 베이즈).",
        "- **ECE**: 모델이 말한 확신도와 실제 정답률의 평균 차이. 클수록 확신도를 라우팅 기준으로 쓰기 어렵습니다.",
        "",
        "## 2. 확신도 보정",
        "![확신도 보정](calibration.png)",
        "",
        "## 3. 기권이 타당했는가",
        "기권한 문항에서 다른 모델들의 정답률이 평소보다 낮을수록, 실제로 어려운 문제에서 기권했다는 뜻입니다.",
        "",
        md_table(abst.rename(columns={"model": "모델", "n_abstain": "기권 수", "others_acc_on_abstained": "기권 문항에서 타 모델 정답률",
                                      "others_acc_overall": "타 모델 평소 정답률", "difficulty_ratio": "비율 (낮을수록 타당)"}),
                 {"기권 문항에서 타 모델 정답률": pct, "타 모델 평소 정답률": pct, "비율 (낮을수록 타당)": f2}),
        "",
        "## 4. 모델 간 오류 상관",
        "같이 틀리는 모델끼리는 투표해도 이득이 작습니다.",
        "",
        md_table(corr.sort_values("error_corr", ascending=False).rename(columns={"model_a": "모델 A", "model_b": "모델 B", "n": "공통 문항",
                 "error_corr": "오류 상관", "both_wrong_rate": "둘 다 틀림", "same_wrong_answer": "같은 오답 비율"}),
                 {"오류 상관": f2, "둘 다 틀림": pct, "같은 오답 비율": pct}),
        "",
        "## 5. 호출 1회당 기대 손실 (USD)",
        "",
        md_table(eloss.rename(columns={"model": "모델", "n": "표본", "error_loss": "오답 피해", "abstain_loss": "기권→사람 검토",
                                       "call_cost": "호출 비용", "expected_loss": "기대 손실"}),
                 {"오답 피해": usd, "기권→사람 검토": usd, "호출 비용": "${:.3f}", "기대 손실": usd}),
        "",
        "## 6. 라우팅 정책",
        f"앞 70% 기간({info['n_train']}건)에서 정책을 고르고, 뒤 30% 기간({info['n_test']}건)에서 검증했습니다. "
        "모델마다 최신 버전 데이터만 사용했습니다.",
        "",
        "![라우팅 정책 비교](router.png)",
        "",
        md_table(test[["정책 구분", "policy", "total", "error_loss", "human_cost", "call_cost", "human_share", "wrong_share"]]
                 .rename(columns={"policy": "정책", "total": "총비용/건", "error_loss": "오답 피해", "human_cost": "사람 검토",
                                  "call_cost": "모델 호출", "human_share": "사람에게 넘긴 비율", "wrong_share": "틀린 채택 비율"}),
                 {"총비용/건": usd, "오답 피해": usd, "사람 검토": usd, "모델 호출": "${:.3f}", "사람에게 넘긴 비율": pct, "틀린 채택 비율": pct}),
        "",
        "### 학습 구간 상위 정책 10개",
        "",
        md_table(search_top.head(10)[["policy", "total", "human_share", "wrong_share"]]
                 .rename(columns={"policy": "정책", "total": "총비용/건", "human_share": "사람 비율", "wrong_share": "틀린 채택"}),
                 {"총비용/건": usd, "사람 비율": pct, "틀린 채택": pct}),
        "",
    ]
    path = out / "report.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path
