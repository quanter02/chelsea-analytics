"""시기에 따라 폭을 넓히는 80% 범위 규칙 (v2 예측용). 사후 설계 → 도전자로만 등록.

    python challenger_adaptive.py   # → output/model_upgrade/challenger_adaptive.md, 장부 kind=nowcast_2026_v2_range_adaptive

정직한 출처: 2019~2025 적중이 72%로 낮다는 것을 본 뒤 설계했다. 그래서 2019~2025 숫자는 증거가 아니고, 판정은 2027년 3월 발표.
규칙 (예측 시점에 이미 알 수 있는 정보만 사용)
  adapt  기본(작은 지역 보정) 폭 × f,  f = max(1, sqrt(최근 3년 표준화 오차² 평균 / 그 전 전체 평균))
         → 최근 몇 년 예측이 평소보다 많이 빗나갔으면 폭을 넓힘. 조정할 숫자 없음
  shock  기본 폭 × (1 + κ·|그해 전국 1~7월 증가율|), κ는 2010~2018 interval score로만 고름
         → 그해 상반기에 혼인이 크게 흔들리면 폭을 넓힘
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from challenger_range import residuals_v2
from llmrel import intervals as I, ledger, nowcast as NC, regional_tune as RT
from model_upgrade import OUT

PICK, TEST = range(2010, 2019), range(2019, 2026)
KAPPAS = [0, 1, 2, 3, 4, 6, 8]


def base_bounds(past: pd.DataFrame, now: pd.DataFrame):
    return I.bounds("disp", past, now)


def inflate_adapt(past: pd.DataFrame) -> float:
    c, phi = I._cphi(past)
    z2 = (past.res / np.sqrt(c ** 2 + phi / past.n)) ** 2
    last3 = past.year >= past.year.max() - 2
    if last3.all() or (~last3).sum() == 0:
        return 1.0
    return float(max(1.0, np.sqrt(z2[last3].mean() / z2[~last3].mean())))


def nat_growth():
    s = pd.read_csv(NC.MONTHLY, dtype={"code": str})
    n = s[s.code == "00"].copy()
    n["y"], n["mo"] = n.month // 100, n.month % 100
    t = n[n.mo <= 7].groupby("y").value.sum()
    return np.log(t).diff()


def score(lo, hi, x):
    a = 1 - I.LEVEL
    return (hi - lo) + 2 / a * np.maximum(lo - x, 0) + 2 / a * np.maximum(x - hi, 0)


def evaluate(r: pd.DataFrame, rule: str, kappa: float = 0.0, years=range(2010, 2026)) -> pd.DataFrame:
    g = nat_growth()
    out = []
    for y in years:
        past, now = r[r.year < y], r[r.year == y]
        lo, hi = base_bounds(past, now)
        f = inflate_adapt(past) if rule == "adapt" else (1 + kappa * abs(g.get(y, 0.0)) if rule == "shock" else 1.0)
        lo, hi, x = lo * f, hi * f, now.res.to_numpy()
        out.append(pd.DataFrame({"year": y, "n": now.n.to_numpy(), "inside": (x >= lo) & (x <= hi), "score": score(lo, hi, x),
                                 "width": np.expm1(hi) - np.expm1(lo), "f": f}))
    return pd.concat(out, ignore_index=True)


def main():
    sgg, _ = RT.load()
    r = residuals_v2(sgg)
    pick_scores = {k: evaluate(r, "shock", k, PICK).score.mean() for k in KAPPAS}
    kappa = min(pick_scores, key=pick_scores.get)
    runs = {"기본 (작은 지역 보정)": evaluate(r, "disp"), "adapt (최근 오차로 넓힘)": evaluate(r, "adapt"), f"shock (상반기 충격으로 넓힘, κ={kappa})": evaluate(r, "shock", kappa)}
    rows, fl = [], []
    for name, e in runs.items():
        p, t = e[e.year.isin(PICK)], e[e.year.isin(TEST)]
        rows.append(f"| {name} | {p.inside.mean():.1%} | {p.score.mean():.3f} | {t.inside.mean():.1%} | {t.score.mean():.3f} | ±{t.width.median() / 2:.1%} |")
    ad = runs["adapt (최근 오차로 넓힘)"].groupby("year").f.first()
    yr = "\n".join(f"| {y} | {f:.2f} | {runs['기본 (작은 지역 보정)'].query('year==@y').inside.mean():.0%} | {runs['adapt (최근 오차로 넓힘)'].query('year==@y').inside.mean():.0%} |" for y, f in ad.items() if y >= 2016)
    # 2026 범위 (adapt): 2025년까지 오차로
    v2p = {x["key"]: x["payload"]["pred_2026"] for x in ledger.read() if x["kind"] == "nowcast_2026_v2"}
    now = pd.DataFrame({"code": list(v2p), "n": [np.exp(sgg.loc[2025, c]) for c in v2p]})
    past = r[r.year <= 2025]
    lo, hi = base_bounds(past, now)
    f26 = inflate_adapt(past)
    n_add = ledger.append([dict(kind="nowcast_2026_v2_range_adaptive", key=c,
                                payload={"lo80": int(round(v2p[c] * np.exp(a * f26))), "hi80": int(round(v2p[c] * np.exp(b * f26))), "rule": "adapt", "factor": round(f26, 3)},
                                model="v2 범위 도전자: 최근 3년 오차가 크면 넓힘 (조정 숫자 없음)",
                                evidence="사후 설계: 2019~2025 적중 72%를 본 뒤 만든 규칙. 판정은 2027년 3월 발표")
                           for c, a, b in zip(now.code, lo, hi)])
    ok, total, _ = ledger.verify()
    (OUT / "challenger_adaptive.md").write_text(f"""# 시기에 따라 폭을 넓히는 범위 규칙 (사후 설계, 도전자)

**주의:** 이 규칙들은 2019~2025 적중이 72%로 낮다는 걸 본 뒤 설계했습니다. 아래 2019~2025 숫자는 공정한 증거가 아닙니다. 진짜 판정은 2027년 3월 발표입니다.
shock의 κ는 2010~2018 interval score로만 골랐습니다 (후보 {KAPPAS}). adapt는 조정할 숫자가 없습니다.

| 규칙 | 2010~2018 적중 | 2010~2018 점수 | 2019~2025 적중 | 2019~2025 점수 | 2019~2025 폭 |
|---|---|---|---|---|---|
{chr(10).join(rows)}

점수는 interval score (좁을수록, 덜 벗어날수록 작음). 적중 목표 80%.

## adapt가 해마다 폭을 얼마나 넓혔나
| 연도 | 넓힌 배수 | 기본 적중 | adapt 적중 |
|---|---|---|---|
{yr}

- 2026년 배수: {f26:.2f} → 2026 범위 {n_add}줄 등록 (kind=nowcast_2026_v2_range_adaptive, 전체 {total}줄, 체인 {'정상' if ok else '어긋남'})
""", encoding="utf-8")
    print((OUT / "challenger_adaptive.md").read_text())


if __name__ == "__main__":
    main()
