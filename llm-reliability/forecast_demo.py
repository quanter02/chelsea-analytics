"""연령별 혼인율·평균 초혼연령 예측: 백테스트 → 모델 선택 검증 → 구간 보정 → 예측 저장 → (정답이 나오면) 채점.

    python forecast_demo.py     # data_official CSV 사용 → output/forecast/report.md, data_forecast/forecasts_<날짜>.csv
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llmrel import forecast as F
from llmrel.report import md_table

ROOT = Path(__file__).parent
DATA = ROOT / "data_official" / "kr_jp_marriage_official.csv"
OUT = ROOT / "output" / "forecast"
SAVE = ROOT / "data_forecast"
MODEL = "ensemble"
COLOR = {"KR": "#2a63d4", "JP": "#df6330"}
NAME = {"KR": "한국", "JP": "일본"}
MODEL_KO = {"naive": "마지막 값 유지", "drift": "최근 5년 추세", "damped": "감쇠 추세", "loglin": "10년 선형 추세",
            "pooled": "공동 학습 회귀", "ensemble": "앙상블 (추세·감쇠·공동)"}


def chart_fan(d: pd.DataFrame, fc: pd.DataFrame, path: Path) -> None:
    bands = ["25-29", "30-34", "35-39"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, band in zip(axes, bands):
        for c in ("JP", "KR"):
            h = d[(d.country == c) & (d.indicator == "marriage_rate") & (d.sex == "F") & (d.age_band.astype(str) == band) & (d.year >= 2005)]
            f = fc[(fc.country == c) & (fc.indicator == "marriage_rate") & (fc.sex == "F") & (fc.age_band == band)].sort_values("year")
            ax.plot(h.year, h.value, "-", color=COLOR[c], lw=2)
            x = [h.year.max()] + list(f.year); last = h.value.iloc[-1]
            ax.fill_between(x, [last] + list(f.lo), [last] + list(f.hi), color=COLOR[c], alpha=.15, lw=0)
            ax.plot(x, [last] + list(f.forecast), "--", color=COLOR[c], lw=2)
            ax.annotate(f"{NAME[c]} {f.forecast.iloc[-1]:.0f}", (f.year.iloc[-1], f.forecast.iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=9, color="#333")
        ax.axvspan(2020, 2021, color="grey", alpha=.1)
        ax.set_title(f"여성 {band}세 혼인율 (천 명당)"); ax.grid(alpha=.3); ax.set_xlim(2005, 2030.5)
    axes[0].plot([], [], "k-", label="실제"); axes[0].plot([], [], "k--", label="예측 (음영 = 80% 구간)")
    axes[0].legend(loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def chart_models(score: pd.DataFrame, path: Path) -> None:
    cols = ["1년 뒤", "2년 뒤", "3년 뒤"]
    s = score[cols]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(cols)); w = 0.13
    shades = ["#5a6672", "#c3cad1", "#aab3bc", "#dce2e7", "#8d97a1", "#2a63d4"]   # 비교 기준(마지막 값)과 쓴 모델만 눈에 띄게
    for i, (m, row) in enumerate(s.iterrows()):
        ax.bar(x + (i - 2.5) * w, row.values * 100, w - 0.02, color=shades[i], label=MODEL_KO[m])
        if m in ("naive", MODEL):
            for xi, v in zip(x + (i - 2.5) * w, row.values * 100):
                ax.text(xi, v + 0.2, f"{v:.1f}", ha="center", fontsize=7.5, color="#333")
    ax.set_xticks(x, cols); ax.set_ylabel("평균 오차율 (%)"); ax.grid(axis="y", alpha=.3)
    ax.set_title("백테스트 오차 (연령별 혼인율, 원점 2005~2023)"); ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); SAVE.mkdir(exist_ok=True)
    d = pd.read_csv(DATA); d["age_band"] = d.age_band.astype(str)
    L = np.log(F.series_table(d))

    bt = F.backtest(L, range(2005, 2024))
    mr = bt[bt.indicator == "marriage_rate"]
    score = F.score_table(mr)
    score_age = F.score_table(bt[bt.indicator == "first_marriage_age"])
    sel = F.selection_check(bt)
    cover = F.interval_check(mr, MODEL)
    fc = F.forecast(L, bt, MODEL)
    today = dt.date.today().isoformat()
    fc.insert(0, "made_on", today)
    fc.to_csv(SAVE / f"forecasts_{today}.csv", index=False)

    chart_fan(d, fc, OUT / "fan.png")
    chart_models(score, OUT / "models.png")

    gain = 1 - score.loc[MODEL] / score.loc["naive"]
    show = fc[(fc.sex == "F")].copy()
    show["예측 (80% 구간)"] = show.apply(lambda r: f"{r.forecast:.1f} ({r.lo:.1f}~{r.hi:.1f})", axis=1)
    show["최근 실제"] = show.apply(lambda r: f"{r.last_value:.1f} ({r.origin})", axis=1)
    show["지표"] = show.indicator.map({"marriage_rate": "혼인율", "first_marriage_age": "초혼연령"}) + " " + show.age_band.replace("all", "")
    tab = show[show.year == show.origin + 1][["country", "지표", "최근 실제", "year", "예측 (80% 구간)", "resolves"]]
    tab = tab.rename(columns={"country": "나라", "year": "예측 연도", "resolves": "정답 공개"})
    tab3 = show[(show.year == show.origin + 3) & (show.indicator == "marriage_rate")][["country", "지표", "year", "예측 (80% 구간)"]]
    tab3 = tab3.rename(columns={"country": "나라", "year": "예측 연도"})

    # 선행 신호 대조: 수집한 보도 수치(혼인 건수 증감)와 모델이 암시하는 방향 비교
    def implied(c, yr):
        f = fc[(fc.country == c) & (fc.year == yr) & (fc.indicator == "marriage_rate") & (fc.sex == "F") & fc.age_band.isin(["25-29", "30-34", "35-39"])]
        return float(np.mean(f.forecast / f.last_value - 1))
    jp25, kr26 = implied("JP", 2025), implied("KR", 2026)

    lines = [
        "# 연령별 혼인율 예측 모델", "",
        f"공식 통계(KOSIS·e-Stat)로 학습한 예측입니다. 계열 {L.shape[1]}개 (한·일 × 남녀 × 연령대 5개 혼인율 + 평균 초혼연령), 2000년부터.", "",
        "## 1. 모델 비교 (롤링 원점 백테스트)", "",
        "원점 연도 T까지의 자료만 보고 T+1~T+3년을 맞히게 한 뒤 실제 값과 비교했습니다 (원점 2005~2023).", "",
        "![모델 오차](models.png)", "",
        md_table(score.rename(index=MODEL_KO).reset_index().rename(columns={"model": "모델"}), {c: "{:.1%}" for c in score.columns}), "",
        f"- 쓴 모델(앙상블)은 '마지막 값 유지'보다 오차가 1년 뒤 {gain['1년 뒤']:.0%}, 3년 뒤 {gain['3년 뒤']:.0%} 작습니다. **개선 폭이 크지 않습니다.** 혼인율은 정책·경기·코로나 같은 바깥 충격이 크고, 과거 숫자만으로는 그 충격을 알 수 없기 때문입니다.",
        "- 평균 초혼연령은 훨씬 잘 맞습니다 (3년 뒤도 오차 0.6% 안팎, 약 0.2세). 천천히 움직이는 지표라서입니다.", "",
        md_table(score_age.rename(index=MODEL_KO).reset_index().rename(columns={"model": "모델 (초혼연령)"}), {c: "{:.2%}" for c in score_age.columns}), "",
        "## 2. 모델 고르기도 검증", "",
        md_table(sel, {"고른 모델의 이후 오차": "{:.1%}", "그 모델의 이후 오차": "{:.1%}", "naive 이후 오차": "{:.1%}"}), "",
        "- 2014년까지 성적으로 고른 모델이 2015년 이후에도 상위권이었지만, 1등은 아니었습니다 (차이 0.1~0.9%p). 실제 경기 데이터의 라우터 실험과 같은 결론입니다: **앞 구간 1등이 뒤 구간 1등이라는 보장은 없고, 차이가 작으면 단순하고 안정적인 쪽을 쓰는 게 낫다.** 그래서 한 모델을 고르지 않고 세 모델 평균(앙상블)을 씁니다.", "",
        "## 3. 예측 구간은 믿을 만한가", "",
        md_table(cover, {"목표": "{:.0%}", "보정 전 포함률": "{:.0%}", "보정 후 포함률 (이후 원점)": "{:.0%}"}), "",
        "- 과거 오차를 그대로 써서 만든 80% 구간은 실제로 54~78%만 담았습니다. **구간이 너무 좁았습니다.** 2년·3년 뒤일수록 심합니다.",
        "- 2014년 이전(조용했던 시기)에서 배수를 고르면 k = 1로 충분해 보였지만, 이후(코로나, 한국 2024~25 급반등)에는 50~73%로 떨어졌습니다. 조용한 시기의 오차로는 다음 충격의 크기를 알 수 없다는 뜻입니다.",
        "- 그래서 최종 구간은 전체 기간으로 배수를 다시 맞췄습니다 (1년 1.05배, 2년 1.55배, 3년 1.75배). 이 배수도 지난 20년 기준이라 **다음 충격이 더 크면 여전히 좁을 수 있습니다.** 정답이 나올 때마다 포함률을 다시 잽니다.", "",
        "## 4. 예측", "", "![예측](fan.png)", "",
        "### 다음 해 (여성)", "", md_table(tab), "",
        "### 3년 뒤 (여성 혼인율)", "", md_table(tab3), "",
        "- 한국: 30~34세 혼인율이 계속 오르는 쪽(2028년 66.5, 구간 48~92)입니다. 구간이 넓습니다. 최근 2년의 급반등이 추세에 반영된 결과라, 반등이 정책 효과로 일시적이면 크게 빗나갈 수 있습니다.",
        "- 일본: 모든 연령대에서 완만한 감소가 이어집니다 (25~29세 41.5 → 2027년 36.6).",
        "- 초혼연령: 한국 여성 2028년 32.1세, 일본 2027년 29.9세. 격차는 계속 벌어집니다.", "",
        "## 5. 선행 신호 대조 (예측 vs 이미 수집한 보도 수치)", "",
        f"- **일본 2025년**: 모델은 여성 25~39세 혼인율이 평균 {jp25:+.1%} 변한다고 봅니다. 그런데 수집해 둔 공식 발표(주장 DB)에 따르면 2025년 혼인 건수는 **+0.8%** (48만 9,119건)였습니다. 젊은 여성 인구가 해마다 1~2%씩 줄어드는 것을 감안하면 율은 오히려 소폭 올랐을 가능성이 큽니다. **모델이 일본 2025년을 낮게 볼 가능성이 있습니다.** 2026년 9월 확정치로 채점됩니다.",
        f"- **한국 2026년**: 모델은 {kr26:+.1%}를 예상합니다. 2026년 2분기 혼인 +4.9%, 7월 +9.3%(주장 DB)와 방향이 같습니다.",
        "- 이 대조가 한계효용 엔진의 다음 단계입니다. 과거 숫자만 보는 모델은 바깥 충격을 모르고, 그 해 월별·분기 발표 같은 **새 정보가 예측을 가장 크게 바꾸는 자료**입니다 (한계효용이 큼).", "",
        "## 6. 채점 루프", "",
        f"- 예측은 `data_forecast/forecasts_{today}.csv`에 저장됩니다 (계열·연도·예측·구간·정답 공개 예정일).",
        "- 새 공식 값이 나오면 `python official_demo.py`로 받고 `llmrel.forecast.score_forecasts()`로 채점합니다: 오차율과 구간 포함 여부가 기록되고, 아직 안 나온 값은 pending으로 남습니다. 앞서 만든 예측 로그(predictions → outcomes) 구조와 같습니다.", "",
        "## 한계", "",
        "- 인구 구조(연령대 인구), 경기, 주거비, 정책 같은 설명 변수는 넣지 않았습니다. 과거 값의 패턴만 씁니다.",
        "- 연령별 '혼인율'이라 미혼율과는 다릅니다. 미혼율 예측은 5년 주기 자료라 표본이 너무 적어 이번에는 하지 않았습니다.",
        "- 한국·일본은 혼인율 정의가 조금 달라 수준 비교보다 각 나라 안의 변화로 읽어야 합니다.",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUT / "report.md")
    print(score.round(3)); print(sel); print(cover)


if __name__ == "__main__":
    main()
