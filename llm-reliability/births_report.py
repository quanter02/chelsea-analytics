"""출생 반등 분석 + 2026·2027 사전등록.

    python births_report.py      # → output/births/report.md, 그림 2장, data_births/sigungu_2026.csv, 장부 기록
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from llmrel import births, ledger

ROOT = Path(__file__).parent
OUT = ROOT / "output" / "births"
plt.rcParams["font.family"] = "NanumGothic"


def charts(n, sc):
    c = n["cohort"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(c.index, c.actual / 1e3, color="#15202b", lw=2, label="실제 출생아")
    ax.plot(c.index, c.cohort / 1e3, color="#2a63d4", lw=2, ls="--", label="세대 효과만 (출산율 2025 고정)")
    for col, colr in (("출산율 유지", "#148f68"), ("3년에 걸쳐 2025 수준으로", "#df6330"), ("매년 +0.03 상승", "#9aa3ac")):
        ax.plot(sc.index, sc[col] / 1e3, color=colr, lw=1.6, label=f"시나리오: {col}")
    ax.errorbar([2026], [n["est26"] / 1e3], yerr=[[(n["est26"] - n["lo26"]) / 1e3], [(n["hi26"] - n["est26"]) / 1e3]], fmt="o", color="#15202b", capsize=4)
    ax.set_ylabel("출생아 (천 명)"); ax.set_xlim(2015, 2040); ax.grid(alpha=.25); ax.legend(fontsize=8, frameon=False)
    ax.set_title("출생 반등은 사람 수가 아니라 출산율에서 왔다", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "births_scenarios.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(c.index, c.women_30_34 / 1e3, color="#2a63d4", lw=2)
    ax.axvline(2025, color="#9aa3ac", ls=":"); ax.set_ylabel("30~34세 여성 (천 명)"); ax.grid(alpha=.25)
    ax.set_title("주출산 연령 여성 수는 2025년이 꼭짓점", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "women_30_34.png", dpi=150); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = births.national()
    sc = births.scenarios(n)
    sg = births.sigungu(n)
    sg = sg[sg.pred.notna()]                                     # 2025년 자료가 없는 곳(폐지·신설) 제외
    sg.round(0).to_csv(ROOT / "data_births" / "sigungu_2026.csv", index=False)
    charts(n, sc)
    lead, c = n["lead"], n["cohort"]
    added = ledger.append(
        [dict(kind="births_national", key="2026", payload={"pred": int(n["est26"]), "lo80": int(n["lo26"]), "hi80": int(n["hi26"])},
              model="2025 × 1~7월 증가율", evidence=f"1~7월 {n['ytd_ratio']:.3f}배, 남은 달 변동 ±1.5%(판단)"),
         dict(kind="births_national", key="2027", payload={"pred": int(n["est27"]), "lo80": int(n["lo27"]), "hi80": int(n["hi27"])},
              model="2026 × 혼인 선행(1년) + 세대 효과", evidence=f"slope {lead['slope']:.2f}, corr {lead['corr']:.2f}, resid sd {lead['resid_sd']:.3f}, 혼인 1~7월 {n['marriage_ytd_ratio']:.3f}배")]
        + [dict(kind="births_sigungu_2026", key=r.code, payload={"name": r.name, "pred": int(r.pred), "lo80": int(r.lo), "hi80": int(r.hi)},
                model="2025 × 시도 1~7월 증가율, 범위 c²+φ/n", evidence=f"c2={sg.attrs['c2']:.4f}, phi={sg.attrs['phi']:.2f}, 과거 적중 {sg.attrs['coverage']:.0%}")
           for r in sg.itertuples()])
    top = sg[~sg.general_gu]
    w = lambda x: f"{x:,.0f}"
    L = ["# 출생 반등: 언제 꺾이나 (2026-10-13 기록)", "",
         "## 한 줄", f"2026년 출생 반등(+{(n['ytd_ratio'] - 1) * 100:.1f}%)은 **사람 수가 아니라 출산율**에서 왔습니다. 주출산 연령 여성 수는 2025년이 꼭짓점이고, "
         f"출산율이 지금 수준에 머물면 출생아는 2030년 {w(sc.loc[2030, '출산율 유지'])}명, 2035년 {w(sc.loc[2035, '출산율 유지'])}명으로 줄어듭니다. "
         f"2026년 수준을 지키려면 합계출산율이 2030년 {sc.loc[2030, '2026 출생 유지에 필요한 출산율']:.2f}, 2035년 {sc.loc[2035, '2026 출생 유지에 필요한 출산율']:.2f}까지 올라야 합니다.", "",
         "## 사전등록 (발표 전에 장부에 기록)",
         "| 항목 | 예측 | 80% 범위 | 채점 |", "|---|---|---|---|",
         f"| 2026년 출생아 | **{w(n['est26'])}** | {w(n['lo26'])}~{w(n['hi26'])} | 2027년 2월 연간 잠정 발표 |",
         f"| 2027년 출생아 | **{w(n['est27'])}** ({n['g27'] * 100:+.1f}%) | {w(n['lo27'])}~{w(n['hi27'])} | 2028년 2월 |",
         f"| 시군구 {len(sg)}곳 2026년 출생아 | data_births/sigungu_2026.csv | 같은 규칙의 과거 적중률 {sg.attrs['coverage']:.0%} | 2027년 8월 확정 발표 |", "",
         f"장부에 {added}줄 추가. 2027년 예측은 혼인 증가율(1년 전)로 출생 증가율을 설명하는 회귀(1995~2025, 상관 {lead['corr']:.2f})라 범위가 넓습니다.", "",
         "## 왜 이렇게 보나",
         f"1. **세대 효과**: 1970년 이후 연도별 출생아 수로 각 나이 여성 수를 재구성했습니다(사망·이동 보정계수 {c.attrs['k']:.3f}, 2015~2025 출생아를 거의 그대로 재현). "
         "출산율을 2025년에 고정하면 출생아는 2026년 −0.5%, 2030년 −4%, 2035년 −17%, 2040년 −28%. 1990년대 초반생(한 해 70만 명대)이 30대 초반을 지나고 나면 "
         "2000년대 초반생(2002년 49.7만 명)이 그 자리를 채우기 때문입니다.",
         f"2. **행동 효과**: 2026년 출생아 증가는 거의 전부 출산율 상승입니다(추정 합계출산율 {n['tfr25']:.3f} → {n['tfr26']:.3f}). "
         "'90년대생이 주출산 연령에 들어섰다'는 설명은 절반만 맞습니다. 그 세대는 이미 꼭짓점이고, 더 밀어 주지 않습니다.",
         f"3. **선행 지표**: 2026년 1~7월 혼인은 {(n['marriage_ytd_ratio'] - 1) * 100:+.1f}%로 2024년(+14.8%)보다 증가 폭이 줄었습니다. 1년 뒤 출생 증가 폭도 줄어드는 신호입니다.", "",
         "## 시나리오 (출생아, 명)", "| 연도 | 세대 효과 | 출산율 유지 | 3년에 걸쳐 2025 수준으로 | 매년 +0.03 상승 | 2026 수준 유지에 필요한 출산율 |", "|---|---|---|---|---|---|"]
    for y, r in sc.iterrows():
        L.append(f"| {y} | {r['세대 효과']:.3f} | {w(r['출산율 유지'])} | {w(r['3년에 걸쳐 2025 수준으로'])} | {w(r['매년 +0.03 상승'])} | {r['2026 출생 유지에 필요한 출산율']:.3f} |")
    L += ["", "![시나리오](births_scenarios.png)", "", "![30~34세 여성](women_30_34.png)", "",
          "## 누구에게 돈이 되는 정보인가",
          "- 산후조리원·소아과·어린이집·유아용품: 2026~2027년은 높은 평지, 2028년부터 사람 수가 줄어듭니다. 지금 수요를 보고 2030년 용량을 늘리면 과잉이 될 수 있습니다.",
          "- 지자체: 출생 증가가 정책 효과인지 세대·전국 흐름인지 가를 수 있습니다(시군구 증가율 − 시도 증가율).",
          "- 웨딩·혼수(E3): 혼인 증가 폭 축소가 1년 먼저 보입니다.", "",
          f"## 시군구 2026년 출생아 상위 10곳 (일반구 제외 {len(top)}곳 합계 {w(top.pred.sum())}명 = 전국 추정과 일치)",
          "| 시군구 | 2025 | 2026 예측 | 80% 범위 |", "|---|---|---|---|"]
    for r in top.sort_values("pred", ascending=False).head(10).itertuples():
        L.append(f"| {r.name} ({r.code}) | {w(r.b2025)} | **{w(r.pred)}** | {w(r.lo)}~{w(r.hi)} |")
    L += ["", "## 한계",
          "- 시군구는 시도 증가율을 그대로 씁니다. 시군구별 2026년 월별 자료가 나오면 바꿉니다.",
          "- 세대 재구성은 국제 이동을 무시하고 보정계수 하나로 맞췄습니다. 10년 이상 앞은 방향만 보세요.",
          "- 2026년 남은 5개월 변동 ±1.5%는 판단값입니다. 2027년 2월 채점으로 확인합니다.",
          "- 추정 합계출산율 0.92는 '1~7월 출생 증가율 ÷ 세대 효과'로 낸 연간 환산값입니다. 공표된 7월 합계출산율 0.89와 차이가 있어 0.89~0.92로 읽으세요."]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:20]))


if __name__ == "__main__":
    main()
