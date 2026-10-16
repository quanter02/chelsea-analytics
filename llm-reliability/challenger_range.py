"""v2(되돌림 보정 + 7개월 신호) 예측에 80% 범위를 다시 맞춘다.

    python challenger_range.py   # 규칙은 2010~2018 interval score로만 고름 → 2019~2025 적중 확인 → 2026 범위 장부 등록
                                 # → output/model_upgrade/challenger_range.md

규칙 후보는 llmrel/intervals.py와 같음 (pooled, size, disp, local, recent). 각 해의 범위는 그 전 해까지의 오차로만 만듦.
주의: 2019~2025는 점 예측 후보 판정에 이미 한 번 썼음. 여기서는 규칙을 고르는 데 쓰지 않고 적중만 보고하며, 진짜 판정은 2027년 3월 발표.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from llmrel import intervals as I, ledger, nowcast as NC, regional_tune as RT
from model_upgrade import OUT, RHO

PICK, TEST = range(2010, 2019), range(2019, 2026)


def residuals_v2(sgg: pd.DataFrame) -> pd.DataFrame:
    g7, g12 = NC.ytd_growth(7), NC.ytd_growth(12)
    rows = []
    for y in range(2007, int(sgg.index.max()) + 1):
        for c in sgg.columns:
            s = c[:2]
            last, prev = sgg.loc[y - 1, c], sgg.loc[y - 2, c]
            idio = (last - prev) - g12.loc[y - 1].get(s, np.nan)
            pred = last + g7.loc[y].get(s, np.nan) - RHO * idio
            rows.append(dict(year=y, code=c, pred=pred, truth=sgg.loc[y, c], n=np.exp(last)))
    r = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
    r["res"] = r.truth - r.pred
    return r


def main():
    sgg, _ = RT.load()
    r = residuals_v2(sgg)
    e = I.evaluate(r, years=range(2010, 2026))
    pick = e[e.year.isin(PICK)].groupby("rule").agg(cover=("inside", "mean"), score=("score", "mean"), width=("width", "median"))
    test = e[e.year.isin(TEST)].groupby("rule").agg(cover=("inside", "mean"), score=("score", "mean"), width=("width", "median"))
    rule = pick.score.idxmin()
    t = e[(e.rule == rule) & e.year.isin(TEST)].copy()
    t["size"] = pd.cut(t.n, [0, 300, 1000, 3000, 1e9], labels=["300 미만", "300~1천", "1천~3천", "3천 이상"])
    by = t.groupby("size", observed=True).inside.mean()
    # v1(현재 공개 범위)과 비교: 같은 규칙 disp, 원래 잔차
    e1 = I.evaluate(I.residuals(), years=TEST)
    v1 = e1[e1.rule == I.CHOSEN]
    v1["size"] = pd.cut(v1.n, [0, 300, 1000, 3000, 1e9], labels=["300 미만", "300~1천", "1천~3천", "3천 이상"])
    by1 = v1.groupby("size", observed=True).inside.mean()
    # 2026 범위 (2025년까지의 오차로)
    now = r[r.year == 2025][["code"]].copy()
    v2p = {x["key"]: x["payload"]["pred_2026"] for x in ledger.read() if x["kind"] == "nowcast_2026_v2"}
    now = pd.DataFrame({"code": list(v2p), "n": [np.exp(sgg.loc[2025, c]) for c in v2p]})
    lo, hi = I.bounds(rule, r[r.year <= 2025], now)
    entries = []
    for c, a, b in zip(now.code, lo, hi):
        p = v2p[c]
        entries.append(dict(kind="nowcast_2026_v2_range", key=c, payload={"lo80": int(round(p * np.exp(a))), "hi80": int(round(p * np.exp(b))), "rule": rule},
                            model=f"v2 범위: {I.LABEL[rule]}, 2010~2018 interval score로 고름", evidence="v2 예측에 붙이는 80% 범위, 2025년까지의 오차만 사용"))
    n_add = ledger.append(entries)
    ok, total, _ = ledger.verify()
    ledger.append([dict(kind="model_result", key="marriage_range_v2", model="challenger_range",
                        payload={"rule": rule, "pick_cover": round(float(pick.loc[rule, "cover"]), 4), "test_cover": round(float(test.loc[rule, "cover"]), 4),
                                 "test_cover_v1": round(float(v1.inside.mean()), 4), "test_by_size": {str(k): round(float(v), 4) for k, v in by.items()}},
                        evidence="규칙은 2010~2018로 고름, 2019~2025 적중은 보고만")])
    rows = "\n".join(f"| {I.LABEL[k]}{' ← 선택' if k == rule else ''} | {pick.loc[k, 'cover']:.1%} | {pick.loc[k, 'score']:.3f} | {test.loc[k, 'cover']:.1%} | ±{test.loc[k, 'width'] / 2:.1%} |" for k in I.RULES)
    sz = "\n".join(f"| {k} | {by1.get(k, np.nan):.1%} | {by.get(k, np.nan):.1%} |" for k in by.index)
    (OUT / "challenger_range.md").write_text(f"""# v2 예측의 80% 범위 다시 맞추기

규칙은 2010~2018 interval score(좁고 덜 벗어날수록 좋음)로만 골랐습니다. 2019~2025는 점 예측 판정에 이미 한 번 썼으므로 여기서는 적중만 보고합니다.

| 규칙 | 고르는 구간 적중 | 고르는 구간 점수 | 2019~2025 적중 | 2019~2025 폭(중앙값) |
|---|---|---|---|---|
{rows}

## 지역 크기별 적중 (2019~2025, 목표 80%)
| 작년 혼인 건수 | 현재 공개 범위 (v1) | 새 범위 (v2) |
|---|---|---|
{sz}

- 전체 적중: v1 {v1.inside.mean():.1%} → v2 {t.inside.mean():.1%}
- 2026 범위 등록: 장부 kind=nowcast_2026_v2_range {n_add}줄 (전체 {total}줄, 체인 {'정상' if ok else '어긋남'})
""", encoding="utf-8")
    print((OUT / "challenger_range.md").read_text())


if __name__ == "__main__":
    main()
