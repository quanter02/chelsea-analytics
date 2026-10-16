"""월별 청년 인구 신호 (v5). 사후 설계 → 도전자로만 등록.

    python challenger_youthpop.py   # → output/model_upgrade/challenger_youthpop.md, 장부 kind=nowcast_2026_v5

신호: 올해 1~7월 25~34세 주민등록 인구 증가율(작년 12월 → 올해 7월, 로그) − 같은 시도 평균.
혼인 1~7월 신호(k=7)와 같은 시점까지만 쓴다. 가설: 연중에 청년이 늘어나는 지역은 혼인도 시도보다 더 늘어난다.
예측 = v3 + γ × 신호.  γ는 2013~2018(월별 인구가 2012년 7월부터라 이 6년만 설계에 쓸 수 있음)에서 평균 절대 오차가 가장 작은 값.
2019~2025는 앞 단계에서 이미 여러 번 봤으므로 증거가 아님. 판정은 2027년 3월 발표.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from challenger_size import LABELS, frame
from llmrel import ledger, regional as R, regional_tune as RT
from model_upgrade import OUT

DESIGN, TEST = range(2013, 2019), range(2019, 2026)
GRID = np.round(np.arange(-1.0, 3.01, 0.25), 2)


def pop_signal(codes, names: dict) -> pd.DataFrame:
    """연도 × 시군구 코드: log(7월/작년 12월) 25~34세 인구 − 시도 평균."""
    p = pd.read_csv("data_monthly/kr_pop_2534_monthly.csv", dtype={"code": str, "month": str})
    p = p[p.code.str.len() == 5].assign(sido=lambda x: x.code.str[:2].map(R.MIG_SIDO), nm=lambda x: x.name.str.strip())
    # 2026년 7월 광주·전남 통합(코드 12): 이름은 그대로라 옛 시도로 되돌려 잇는다. 인천 중·동·서구는 개편으로 끊김 → v4 유지
    gj = p.code.str.startswith("12")
    p.loc[gj, "sido"] = np.where(p.loc[gj, "nm"].isin(["동구", "서구", "남구", "북구", "광산구"]), "24", "36")
    g = p.groupby(["sido", "nm", "month"]).value.sum().unstack()
    out = {}
    for c in codes:
        key = (c[:2], names.get(c, ""))
        if key not in g.index:
            continue
        row = g.loc[key]
        out[c] = {y: np.log(row.get(f"{y}07", np.nan) / row.get(f"{y - 1}12", np.nan)) for y in range(2013, 2027)}
    sig = pd.DataFrame(out)
    return sig - sig.T.groupby(sig.columns.str[:2]).transform("mean").T


def main():
    sgg, _ = RT.load()
    mr = pd.read_csv("data_regional/kr_marriages_by_region.csv", dtype={"code": str})
    names = mr.drop_duplicates("code").set_index("code").name.str.strip().to_dict()
    sig = pop_signal(sgg.columns, names)
    rhos = [x["payload"]["rho_by_size"] for x in ledger.read() if x["kind"] == "model_candidate" and x["key"] == "marriage_upgrade_v3_size"][-1]

    def prep(years):
        f = frame(sgg, years)
        f["v3"] = f.base - f["size"].map(rhos).astype(float) * f.idio
        f["sig"] = [sig[c].get(y, np.nan) if c in sig else np.nan for c, y in zip(f.code, f.year)]
        return f.dropna(subset=["sig"])

    d, t = prep(DESIGN), prep(TEST)
    mae = lambda f, g: float(np.expm1((f.truth - (f.v3 + g * f.sig)).abs()).mean())
    gamma = float(GRID[int(np.argmin([mae(d, g) for g in GRID]))])
    ledger.append([dict(kind="model_candidate", key="marriage_upgrade_v5_youthpop", model="challenger_youthpop",
                        payload={"gamma": gamma, "signal": "올해 1~7월 25~34세 인구 로그 증가율 − 시도 평균", "design": "2013~2018",
                                 "base": "v3", "note": "사후 설계, 설계 자료 6년"},
                        evidence="γ 등록, 판정은 2027년 3월 발표")])
    corr = lambda f: float(np.corrcoef(f.sig, f.truth - f.v3)[0, 1])
    per_year = "\n".join(f"| {y} | {mae(d[d.year == y], 0):.2%} | {mae(d[d.year == y], gamma):.2%} |" for y in DESIGN)
    t["size"] = t["size"].astype(str)
    sz = "\n".join(f"| {lab} | {mae(t[t['size'] == lab], 0):.2%} | {mae(t[t['size'] == lab], gamma):.2%} |" for lab in LABELS if (t["size"] == lab).any())
    # 2026 v5 = v4 × exp(γ × 신호)  (v4에 청년 이동 신호가 이미 들어 있음)
    v4 = {x["key"]: x["payload"]["pred_2026"] for x in ledger.read() if x["kind"] == "nowcast_2026_v4"}
    ent, moved = [], []
    for c, p in v4.items():
        s = sig[c].get(2026, np.nan) if c in sig else np.nan
        v = int(round(p * np.exp(gamma * s))) if np.isfinite(s) else p
        moved.append(v / p - 1)
        ent.append(dict(kind="nowcast_2026_v5", key=c, payload={"pred_2026": v, "gamma": gamma, "signal_2026": None if not np.isfinite(s) else round(float(s), 4)},
                        model="v5: v4 + 월별 청년 인구 신호", evidence="사후 설계 도전자, 2027년 3월 판정"))
    n_add = ledger.append(ent)
    ok, total, _ = ledger.verify()
    (OUT / "challenger_youthpop.md").write_text(f"""# 월별 청년 인구 신호 (v5, 사후 설계 도전자)

신호 = 올해 1~7월 25~34세 주민등록 인구 로그 증가율 − 같은 시도 평균 (혼인 1~7월 신호와 같은 시점까지만 사용).
γ는 2013~2018에서만 골랐고(후보 {GRID[0]}~{GRID[-1]}), **γ = {gamma}**. 연결된 시군구 {sig.shape[1]}곳 / {sgg.shape[1]}곳.
2019~2025는 이미 여러 번 봤으므로 아래 숫자는 참고용입니다. 판정은 2027년 3월 발표입니다.

| 구간 | 시군구-연도 | v3 오차 | v3+신호 오차 | 신호와 v3 잔차의 상관 |
|---|---|---|---|---|
| 2013~2018 (설계) | {len(d)} | {mae(d, 0):.2%} | {mae(d, gamma):.2%} | {corr(d):+.2f} |
| 2019~2025 (증거 아님) | {len(t)} | {mae(t, 0):.2%} | {mae(t, gamma):.2%} | {corr(t):+.2f} |

## 설계 구간 연도별
| 연도 | v3 | v3+신호 |
|---|---|---|
{per_year}

## 크기별 (2019~2025, 참고)
| 작년 혼인 건수 | v3 | v3+신호 |
|---|---|---|
{sz}

- 2026 v5 예측 {n_add}줄 등록 (kind=nowcast_2026_v5 = v4 × exp(γ×신호), 전체 {total}줄, 체인 {'정상' if ok else '어긋남'}). v4 대비 변화: 중앙값 {np.median(np.abs(moved)):.1%}, 최대 {np.max(np.abs(moved)):.1%}
""", encoding="utf-8")
    print((OUT / "challenger_youthpop.md").read_text())


if __name__ == "__main__":
    main()
