"""예측 범위(80%) 백테스트 리포트 + 2026년 범위를 장부에 기록.

    python intervals_report.py      # → output/intervals/report.md, data_monthly/nowcast_2026_range.csv, 장부 추가
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from llmrel import intervals as I, ledger

from pathlib import Path
ROOT = Path(__file__).parent
OUT = ROOT / "output" / "intervals"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = I.residuals()
    e = I.evaluate(r)
    assert I.choose(e) == I.CHOSEN, "고르는 구간 1위가 바뀜: CHOSEN을 다시 확인"
    s = I.summary(e)
    ch = e[e.rule == I.CHOSEN].copy()
    ch["seg"] = np.where(ch.year.isin(I.PICK), "2010~2018", "2019~2025")
    ch["크기"] = pd.cut(ch.n, [0, 300, 1500, 1e9], labels=["작은 곳 (<300건)", "중간", "큰 곳 (1,500건+)"])
    by_size = ch.groupby(["seg", "크기"], observed=True).inside.mean().unstack()
    by_year = ch.groupby("year").inside.mean()
    pc, pr = I.params(I.CHOSEN, r, 2026), I.params(I.CHALLENGER, r, 2026)
    f, g = I.forecast(I.CHOSEN, 2026, r), I.forecast(I.CHALLENGER, 2026, r)
    names = pd.read_csv(ROOT / "data_regional" / "kr_marriages_by_region.csv", dtype={"code": str}).drop_duplicates("code").set_index("code").name
    t = pd.DataFrame({"code": f.index, "pred_2026": f.pred.round().astype(int), "lo80": f.lo.round().astype(int), "hi80": f.hi.round().astype(int),
                      "lo80_challenger": g.lo.round().astype(int), "hi80_challenger": g.hi.round().astype(int)})
    t.to_csv(ROOT / "data_monthly" / "nowcast_2026_range.csv", index=False)
    pct = lambda x: f"{x * 100:.0f}%"
    L = ["# 예측 범위 (80%) 백테스트", "",
         "각 해의 범위는 **그 전 해까지 채점된 오차만으로** 만들었습니다. 목표는 '실제 값이 범위 안에 들어오는 비율 80%'.", "",
         "| 규칙 | 2010~2018 적중 | 2019~2025 적중 | 폭(중앙값) 2019~2025 | 점수 2010~2018 | 점수 2019~2025 |", "|---|---|---|---|---|---|"]
    for k in I.RULES:
        mark = " **(채택)**" if k == I.CHOSEN else (" (도전자)" if k == I.CHALLENGER else "")
        L.append(f"| {I.LABEL[k]}{mark} | {pct(s.loc[k, ('cover', 'pick')])} | {pct(s.loc[k, ('cover', 'test')])} | ±{s.loc[k, ('width', 'test')] * 50:.0f}% | {s.loc[k, ('score', 'pick')]:.3f} | {s.loc[k, ('score', 'test')]:.3f} |")
    L += ["", "점수 = interval score (범위가 넓을수록, 벗어날수록 손해. 작을수록 좋음). 채택은 2010~2018 점수 1위로만 정했습니다.", "",
          "## 채택 규칙", "",
          f"범위 = 예측 × exp(z × √(c² + φ / 작년 혼인 건수)),  c = {pc['c']:.3f}, φ = {pc['phi']:.2f}, z = {pc['z_lo']:.2f} ~ {pc['z_hi']:.2f} (2006~2025 오차로 추정)", "",
          "- c: 지역 크기와 무관한 흔들림 (전국 경기·제도 변화)",
          f"- φ/n: 작은 지역일수록 큰 우연한 흔들림. φ = {pc['phi']:.1f}로, 순수한 우연(φ=1)보다 약 {pc['phi']:.1f}배 더 흔들립니다.", "",
          "## 지역 크기별 적중률", "", "| 구간 | " + " | ".join(by_size.columns.astype(str)) + " |", "|---|" + "---|" * len(by_size.columns)]
    for seg, row in by_size.iterrows():
        L.append(f"| {seg} | " + " | ".join(pct(v) for v in row) + " |")
    L += ["", "## 해마다 적중률 (채택 규칙)", "", "| " + " | ".join(str(y) for y in by_year.index) + " |", "|" + "---|" * len(by_year), "| " + " | ".join(pct(v) for v in by_year) + " |", "",
          "## 솔직한 결론", "",
          f"- 2010~2018에는 80% 범위가 {pct(s.loc[I.CHOSEN, ('cover', 'pick')])} 맞았습니다. 약속대로입니다.",
          f"- 2019~2025에는 {pct(s.loc[I.CHOSEN, ('cover', 'test')])}로 떨어졌습니다. 특히 2023년({pct(by_year[2023])})·2024년({pct(by_year[2024])}) 결혼 반등기에 범위가 좁았습니다.",
          "- 그래서 리포트에는 '80% 범위 (최근 7년 실제 적중 71%)'처럼 실제 성적을 같이 적습니다.",
          f"- 도전자(최근 3년 오차로만 폭을 잼)는 2019~2025 적중 {pct(s.loc[I.CHALLENGER, ('cover', 'test')])}로 더 정직했지만, 고르는 구간 점수에서는 근소하게 졌습니다({s.loc[I.CHALLENGER, ('score', 'pick')]:.3f} vs {s.loc[I.CHOSEN, ('score', 'pick')]:.3f}). 두 범위를 모두 장부에 남기고 2026~2028년 실전으로 판정합니다.", "",
          "## 2026년 범위 예 (채택 / 도전자)", "", "| 지역 | 예측 | 80% 범위 | 도전자 범위 |", "|---|---|---|---|"]
    for c in ["31010", "31190", "31020", "31070", "37630"]:
        if c in f.index:
            L.append(f"| {names.get(c, c)} | {f.pred[c]:,.0f} | {f.lo[c]:,.0f} ~ {f.hi[c]:,.0f} | {g.lo[c]:,.0f} ~ {g.hi[c]:,.0f} |")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    n = ledger.append([dict(kind="nowcast_2026_range", key=c, payload={"region": names.get(c, ""), "pred_2026": int(row.pred_2026), "lo80": int(row.lo80), "hi80": int(row.hi80),
                                                                      "lo80_challenger": int(row.lo80_challenger), "hi80_challenger": int(row.hi80_challenger)},
                            model=f"1~6월 신호 + 80% 범위 ({I.LABEL[I.CHOSEN]}; 도전자 {I.LABEL[I.CHALLENGER]})",
                            evidence=f"c={pc['c']:.4f}, phi={pc['phi']:.3f}, z=[{pc['z_lo']:.3f},{pc['z_hi']:.3f}] / 도전자 c={pr['c']:.4f}, phi={pr['phi']:.3f}, z=[{pr['z_lo']:.3f},{pr['z_hi']:.3f}]")
                       for c, row in t.set_index("code").iterrows()])
    print("\n".join(L)); print(f"\nledger +{n}")
    return pc


if __name__ == "__main__":
    main()
