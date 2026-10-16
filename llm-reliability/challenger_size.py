"""지역 크기별 되돌림 보정 세기 (v3). 사후 설계 → 도전자로만 등록.

    python challenger_size.py   # → output/model_upgrade/challenger_size.md, 장부 kind=nowcast_2026_v3

정직한 출처: v2(ρ=0.25 일괄)의 시험 구간 결과에서 큰 지역은 보정이 손해였다는 것을 본 뒤 설계했다.
크기별 ρ는 2010~2018 평균 절대 오차가 가장 작아지는 값으로만 고른다 (후보 0, 0.05, …, 0.6).
예측 = 작년 + 시도 1~7월 증가율 − ρ(작년 건수 구간) × 작년 고유 변화.  판정: 2027년 3월 발표 (v1·v2와 나란히).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from llmrel import ledger, nowcast as NC, regional_tune as RT
from model_upgrade import OUT

PICK, TEST = range(2010, 2019), range(2019, 2026)
BINS, LABELS = [0, 300, 1000, 3000, 1e9], ["300 미만", "300~1천", "1천~3천", "3천 이상"]
GRID = np.round(np.arange(0, 0.61, 0.05), 2)


def frame(sgg: pd.DataFrame, years) -> pd.DataFrame:
    g7, g12 = NC.ytd_growth(7), NC.ytd_growth(12)
    rows = []
    for y in years:
        for c in sgg.columns:
            s = c[:2]
            last, prev = sgg.loc[y - 1, c], sgg.loc[y - 2, c]
            rows.append(dict(year=y, code=c, truth=sgg.loc[y, c], last=last, base=last + g7.loc[y].get(s, np.nan),
                             idio=(last - prev) - g12.loc[y - 1].get(s, np.nan)))
    f = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
    f["size"] = pd.cut(np.exp(f["last"]), BINS, labels=LABELS)
    return f


def mae(f, rho):
    return float(np.expm1((f.truth - (f.base - rho * f.idio)).abs()).mean())


def main():
    sgg, _ = RT.load()
    pick, test = frame(sgg, PICK), frame(sgg, TEST)
    rhos = {}
    for lab in LABELS:
        p = pick[pick["size"] == lab]
        rhos[lab] = float(GRID[int(np.argmin([mae(p, g) for g in GRID]))])
    ledger.append([dict(kind="model_candidate", key="marriage_upgrade_v3_size", model="challenger_size",
                        payload={"rho_by_size": rhos, "chosen_on": "2010~2018 평균 절대 오차", "note": "사후 설계 (v2 시험 결과를 본 뒤)"},
                        evidence="크기별 ρ 등록, 판정은 2027년 3월 발표")])
    rows = []
    for lab in LABELS:
        p, t = pick[pick["size"] == lab], test[test["size"] == lab]
        rows.append(f"| {lab} | {rhos[lab]:.2f} | {mae(p, 0):.2%} / {mae(p, 0.25):.2%} / {mae(p, rhos[lab]):.2%} | {mae(t, 0):.2%} / {mae(t, 0.25):.2%} / {mae(t, rhos[lab]):.2%} |")
    tt = test.copy()
    tt["rho"] = tt["size"].map(rhos).astype(float)
    v3_test = float(np.expm1((tt.truth - (tt.base - tt.rho * tt.idio)).abs()).mean())
    # 2026 v3 예측
    g7, g12 = NC.ytd_growth(7), NC.ytd_growth(12)
    names = pd.read_csv("content/b2b/districts_all.csv", dtype={"code": str}).set_index("code")["지역"]
    v2 = {x["key"]: x["payload"]["pred_2026"] for x in ledger.read() if x["kind"] == "nowcast_2026_v2"}
    ent, diffs = [], []
    for c in sgg.columns:
        s = c[:2]
        last, prev = sgg.loc[2025, c], sgg.loc[2024, c]
        idio = (last - prev) - g12.loc[2025].get(s, np.nan)
        lab = pd.cut([np.exp(last)], BINS, labels=LABELS)[0]
        p = last + g7.loc[2026].get(s, np.nan) - rhos[lab] * idio
        if np.isfinite(p):
            v = int(round(np.exp(p)))
            ent.append(dict(kind="nowcast_2026_v3", key=c, payload={"pred_2026": v, "region": names.get(c, c), "rho": rhos[lab]},
                            model="v3: 크기별 되돌림 보정 + 7개월 신호", evidence="사후 설계 도전자, 2027년 3월 판정"))
            if c in v2:
                diffs.append(v / v2[c] - 1)
    n_add = ledger.append(ent)
    ok, total, _ = ledger.verify()
    (OUT / "challenger_size.md").write_text(f"""# 지역 크기별 되돌림 보정 세기 (v3, 사후 설계 도전자)

**주의:** 'v2의 일괄 보정이 큰 지역에서 손해였다'는 시험 구간 결과를 본 뒤 설계했습니다. ρ는 2010~2018로만 골랐지만, 2019~2025 숫자는 공정한 증거가 아닙니다.

| 작년 혼인 건수 | 고른 ρ | 2010~2018 오차 (보정 없음 / 0.25 / 고른 ρ) | 2019~2025 오차 (보정 없음 / 0.25 / 고른 ρ) |
|---|---|---|---|
{chr(10).join(rows)}

- 2019~2025 전체 평균 절대 오차: v1(현재) 9.37% · v2 8.73% · v3 {v3_test:.2%}
- 2026 v3 예측 {n_add}줄 등록 (kind=nowcast_2026_v3, 전체 {total}줄, 체인 {'정상' if ok else '어긋남'}). v2와 다른 정도: 중앙값 {np.median(np.abs(diffs)):.1%}, 최대 {np.max(np.abs(diffs)):.1%}
""", encoding="utf-8")
    print((OUT / "challenger_size.md").read_text())


if __name__ == "__main__":
    main()
