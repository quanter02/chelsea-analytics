"""과열 온도계 v0: 백테스트 채점 + 이번 달 확률 사전등록.

    python crisis_report.py      # → output/crisis/report.md, 그림 2장, walk_forward.csv, 장부 기록(kind=crisis_risk)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llmrel import crisis, ledger

ROOT = Path(__file__).parent
OUT = ROOT / "output" / "crisis"
plt.rcParams["font.family"] = "NanumGothic"
COL = {"S&P500": "#2a63d4", "닛케이225": "#df6330", "코스피": "#148f68"}


def tags_table() -> pd.DataFrame:
    k = pd.read_csv(ROOT / "data_crisis" / "kindleberger_appendix.csv")
    t = k.tags.str.get_dummies(sep=";")
    return pd.DataFrame({"건수": t.sum(), "비율": t.mean()}).sort_values("건수", ascending=False)


def charts(w: pd.DataFrame, sc: dict):
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for ax, (m, g) in zip(axes, w.groupby("market", sort=False)):
        g = g.set_index("date")
        ax.fill_between(g.index, 0, 1, where=(g.y == 1).to_numpy(), color="#df6330", alpha=.15, step="mid", lw=0, label="실제로 12개월 안 -20% (정답)")
        ax.plot(g.index, g.p, color=COL[m], lw=1.4, label="모델 확률")
        ax.plot(g.index, g.base, color="#9aa3ac", lw=1, ls="--", label="기준선 (그때까지 빈도)")
        ax.set_ylim(0, 1); ax.set_ylabel(m); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    axes[0].set_title("walk-forward 확률 vs 실제 급락 — 가격·금리만으로는 기준선을 못 이긴다", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "walk_forward.png", dpi=150); plt.close(fig)
    cal = sc["calibration"]
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.plot([0, 1], [0, 1], color="#9aa3ac", ls=":")
    ax.plot(cal["예측"], cal["실제"], "o-", color="#15202b")
    for _, r in cal.iterrows():
        ax.annotate(f"n={int(r.n)}", (r["예측"], r["실제"]), textcoords="offset points", xytext=(5, -10), fontsize=7)
    ax.set_xlabel("예측 확률"); ax.set_ylabel("실제 빈도"); ax.set_xlim(0, .6); ax.set_ylim(0, .6); ax.grid(alpha=.2)
    ax.set_title("보정: 높게 부를수록 과신", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "calibration.png", dpi=150); plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    p = crisis.panel()
    w = crisis.walk_forward(p)
    sc = crisis.score(w)
    now = crisis.current(p)
    w.to_csv(OUT / "walk_forward.csv", index=False)
    charts(w, sc)
    asof = now.index.max().strftime("%Y-%m")
    ref = {m: float(crisis.load(c).iloc[-1]) for m, c in crisis.MARKETS.items()}
    entries = [{"kind": "crisis_risk", "key": f"{r.market}:{asof}", "model": "crisis_v0_logit6",
                "payload": {"p": round(float(r.p), 3), "lo80": round(float(r.lo80), 3), "hi80": round(float(r.hi80), 3),
                            "base": round(float(r.base), 3), "ref_close": round(ref[r.market], 1),
                            "threshold": round(ref[r.market] * (1 + crisis.DROP), 1),
                            "rule": "2026-11~2027-10 월말 종가 중 최저가 ≤ threshold 이면 1",
                            "out_of_range": r.out_of_range}} for _, r in now.iterrows()]
    # v1 (신용 추가): 같은 달끼리 v0와 비교, 사전에 정한 규칙으로 채택 여부 판정. 채택 안 돼도 도전자로 장부에 남겨 함께 채점
    p1 = crisis.panel(crisis.FEATS_V1)
    v0s, v1s = crisis.score(crisis.walk_forward(p1)), crisis.score(crisis.walk_forward(p1, feats=crisis.FEATS_V1))
    adopt = v1s["skill"] > max(v0s["skill"], 0)
    now1 = crisis.current(p1, feats=crisis.FEATS_V1)
    entries += [{"kind": "crisis_risk", "key": f"{r.market}:{asof}:v1", "model": "crisis_v1_credit_challenger",
                 "payload": {"p": round(float(r.p), 3), "lo80": round(float(r.lo80), 3), "hi80": round(float(r.hi80), 3),
                             "cg3": round(float(r.cg3), 2), "dd12": round(float(r.dd12), 3), "adopted": bool(adopt),
                             "threshold": round(ref[r.market] * (1 + crisis.DROP), 1), "out_of_range": r.out_of_range}} for _, r in now1.iterrows()]
    hh = crisis.household_credit()
    ledger.append(entries)                                          # 같은 값이면 다시 쓰지 않는다
    n = sum(r["kind"] == "crisis_risk" and asof in r["key"] for r in ledger.read())
    v1rows = "\n".join(f"| {r.market} | {r.p:.0%} ({r.lo80:.0%}–{r.hi80:.0%}) | {r.cg3:+.1f}%p | {r.dd12:+.0%} | {r.out_of_range or '—'} |" for _, r in now1.iterrows())
    v1bm = "\n".join(f"| {m} | {v0s['by_market'].loc[m, 'skill']:+.3f} | {r.skill:+.3f} |" for m, r in v1s["by_market"].iterrows())
    v1cal = "\n".join(f"| {iv} | {int(r.n)} | {r['예측']:.1%} | {r['실제']:.1%} |" for iv, r in v1s["calibration"].iterrows())
    hhrows = "\n".join(f"| {q} | {r['조원']:,.0f} | {r['전년비']:+.1%} |" for q, r in hh.tail(6).iterrows())
    coef1 = ", ".join(f"{k} {v:+.2f}" for k, v in now1.attrs["coef"].items())
    verdict = "**v1 채택**" if adopt else "**v1 탈락 → v0 유지.** 신용을 붙여도 기준선을 못 이겼다"
    rows = "\n".join(f"| {r.market} | {r.p:.0%} ({r.lo80:.0%}–{r.hi80:.0%}) | {r.base:.0%} | {ref[r.market]:,.0f} → {ref[r.market]*0.8:,.0f} | "
                     f"{r.ret36:+.0%} | {r.vol12:.0%} | {r.out_of_range or '—'} |" for _, r in now.iterrows())
    bm = sc["by_market"]
    bmr = "\n".join(f"| {m} | {int(r.n)} | {r['실제 빈도']:.1%} | {r.skill:+.3f} |" for m, r in bm.iterrows())
    cal = "\n".join(f"| {iv} | {int(r.n)} | {r['예측']:.1%} | {r['실제']:.1%} |" for iv, r in sc["calibration"].iterrows())
    tg = "\n".join(f"| {i} | {int(r['건수'])} | {r['비율']:.0%} |" for i, r in tags_table().iterrows())
    coef = ", ".join(f"{k} {v:+.2f}" for k, v in now.attrs["coef"].items())
    (OUT / "report.md").write_text(f"""# 과열 온도계 v0 — 12개월 안에 20% 넘게 빠질 확률

기준 {asof} (10월은 13일까지 종가). 장부 기록 {n}줄 (kind=crisis_risk, v0 3 + v1 도전자 3, 해시 체인). 투자 권유가 아니라 채점받을 예측입니다.

## 한 줄 결론
**가격·금리만으로 만든 v0는 '그냥 평소 빈도'보다 못 맞힌다** (Brier {sc['brier']:.4f} vs 기준선 {sc['brier_base']:.4f}, skill {sc['skill']:+.3f}, n={sc['n']}).
킨들버거 표가 말하는 핵심 연료는 **신용(새 은행·대출·자본 유입)** 인데 v0엔 그게 없다. 신용 자료를 붙이는 v1이 진짜 시험.

## 이번 달 확률 (사전등록)
| 시장 | 확률 (80% 범위) | 기준선 | 기준가 → −20% 선 | 3년 수익률 | 12개월 변동성 | 학습 범위 밖 |
|---|---|---|---|---|---|---|
{rows}

- 채점: 2026-11 ~ 2027-10 월말 종가 중 최저가가 −20% 선 이하이면 1, 아니면 0. 2027-11에 Brier로 채점.
- **코스피 수치는 외삽.** 12개월 변동성·3년 수익률이 1960년 이후 세 시장 어디에도 없던 값이라, 모델이 본 적 없는 영역이다.
  80% 범위는 계수 흔들림만 반영하고 이 외삽 위험은 반영하지 못한다. 이미 고점 대비 크게 빠진 뒤라는 점도 모델은 모른다(현재 낙폭 특성 없음).
- 계수(표준화): {coef}

## 백테스트 (walk-forward, 1975~, 매달 그때 알 수 있던 자료로만 다시 학습)
| 시장 | n | 실제 급락 빈도 | skill |
|---|---|---|---|
{bmr}

보정표 — 높게 부를수록 과신한다:

| 예측 구간 | n | 평균 예측 | 실제 |
|---|---|---|---|
{cal}

![walk-forward](walk_forward.png)
![calibration](calibration.png)

### 정직하게: 시험한 변형
설계(특성 6개)를 정한 뒤 결과를 보고 단일 특성·작은 조합 9개를 더 돌려봤다. 가장 좋은 것(10개월선 이격 + 변동성)이 skill +0.016.
9개 중 최고라 운일 가능성이 커서 채택하지 않았다. 사전등록은 처음 설계 그대로.

## 킨들버거 부록 46건 — 투기 대상·연료 태그
| 태그 | 건수 | 비율 |
|---|---|---|
{tg}

new_banks(새 은행·신용)와 new_tech(신기술)가 절반 가까이. rate_cut은 3건뿐이라 금리만 보는 v0가 약한 것과 맞아떨어진다.

## v1: KOSIS 신용 자료를 붙인 결과 (규칙은 결과 보기 전에 정함)
- 추가 특성: cg3 = 민간신용/GDP 3년 변화(%p, KOSIS DT_2AQ551, Y년 값은 Y+1년 6월부터 사용), dd12 = 12개월 고점 대비 낙폭
- 같은 {v1s['n']}개월에서: v0 skill {v0s['skill']:+.3f} vs v1 skill {v1s['skill']:+.3f} → {verdict}
- 시장별로는 갈린다: 미국·일본은 v1이 나아졌고(기준선엔 여전히 못 미침) 코스피에서 크게 나빠져 전체가 떨어졌다. 한국 표본은 2000년 이후 26년뿐
- 계수 방향은 이론대로(cg3 양수: 신용이 빨리 늘수록 위험). 그런데 밖에서 맞혀 보면 과신이 더 심해졌다 — 50% 넘게 부른 달 실제 빈도가 낮다

| 시장 | v0 skill | v1 skill |
|---|---|---|
{v1bm}

| 예측 구간 | n | 평균 예측 | 실제 |
|---|---|---|---|
{v1cal}

### 도전자로 함께 등록 (v1, 채점은 v0와 같은 날)
| 시장 | v1 확률 (80% 범위) | 신용/GDP 3년 변화 | 12개월 고점 대비 | 학습 범위 밖 |
|---|---|---|---|---|
{v1rows}

계수(표준화): {coef1}

### 한국 가계신용 (KOSIS DT_151Y001, 참고 지표 — 한국만 있어 학습엔 안 씀)
| 분기 | 잔액(조원) | 전년비 |
|---|---|---|
{hhrows}

**해석 (판단):** 코스피의 높은 확률은 가격(3년 +97%, 변동성)에서 나온다. 신용 쪽은 과열 신호가 약하다 — 민간신용/GDP는 3년 사이 줄었고
가계신용 증가율도 3%대. 킨들버거 기준으로는 '신용 연료 없는 가격 급등'에 가까워, 금융 위기보다는 가격 조정 쪽 그림이다.

## 다음 단계
- 과신 보정: 모델 확률을 기준선 쪽으로 줄이는 축소 — v0·v1 모두 높게 부를 때 틀린다.
- 연간 신용/GDP는 너무 거칠다(발표 시차 6~18개월). 분기 신용 갭(BIS)은 도메인이 막혀 있음.
- 표본: 세 시장 50년에 급락 구간이 몇십 개뿐. 나라를 늘리려면 지수 자료(유럽·대만 등)가 더 필요.
""")
    print((OUT / "report.md").read_text()[:1500])


if __name__ == "__main__":
    main()
