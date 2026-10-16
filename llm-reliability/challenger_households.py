"""주민등록 세대 수 신호 점검. 청년 인구(v5) 위에 더할 가치가 있는지만 본다.

    python challenger_households.py   # → output/model_upgrade/challenger_households.md, 장부 kind=model_result

신호: 올해 1~7월 주민등록 세대 수 로그 증가율(작년 12월 → 7월) − 같은 시도 평균. 자료 KOSIS DT_1B040B3 (2011-01~).
설계 2013~2018. 점검(dry run)에서 v5 위 효과가 없어서, 2026 예측은 등록하지 않고 '효과 없음'만 기록한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import challenger_youthpop as Y
from challenger_size import frame
from llmrel import ledger, regional as R, regional_tune as RT
from model_upgrade import OUT

GRID = np.round(np.arange(-3.0, 4.01, 0.25), 2)


def hh_signal(codes, names: dict) -> pd.DataFrame:
    p = pd.read_csv("data_monthly/kr_households_monthly.csv", dtype={"code": str, "month": str})
    p = p[p.code.str.len() == 5].assign(sido=lambda x: x.code.str[:2].map(R.MIG_SIDO), nm=lambda x: x.name.str.strip())
    gj = p.code.str.startswith("12")   # 2026년 7월 광주·전남 통합 (challenger_youthpop와 같은 처리)
    p.loc[gj, "sido"] = np.where(p.loc[gj, "nm"].isin(["동구", "서구", "남구", "북구", "광산구"]), "24", "36")
    g = p.groupby(["sido", "nm", "month"]).value.sum().unstack()
    out = {c: {y: np.log(g.loc[k].get(f"{y}07", np.nan) / g.loc[k].get(f"{y - 1}12", np.nan)) for y in range(2012, 2027)}
           for c in codes if (k := (c[:2], names.get(c, ""))) in g.index}
    s = pd.DataFrame(out)
    return s - s.T.groupby(s.columns.str[:2]).transform("mean").T


def main():
    sgg, _ = RT.load()
    mr = pd.read_csv("data_regional/kr_marriages_by_region.csv", dtype={"code": str})
    names = mr.drop_duplicates("code").set_index("code").name.str.strip().to_dict()
    ps, hs = Y.pop_signal(sgg.columns, names), hh_signal(sgg.columns, names)
    rhos = [x["payload"]["rho_by_size"] for x in ledger.read() if x["kind"] == "model_candidate" and x["key"] == "marriage_upgrade_v3_size"][-1]
    gamma = [x["payload"]["gamma"] for x in ledger.read() if x["kind"] == "model_candidate" and x["key"] == "marriage_upgrade_v5_youthpop"][-1]

    def prep(years):
        f = frame(sgg, years)
        f["v3"] = f.base - f["size"].map(rhos).astype(float) * f.idio
        f["ps"] = [ps[c].get(y, np.nan) for c, y in zip(f.code, f.year)]
        f["v5"] = f.v3 + gamma * f.ps
        f["hh"] = [hs[c].get(y, np.nan) if c in hs else np.nan for c, y in zip(f.code, f.year)]
        return f.dropna(subset=["ps", "hh"])

    d, t = prep(Y.DESIGN), prep(Y.TEST)
    mae = lambda f, b, g: float(np.expm1((f.truth - (f[b] + g * f.hh)).abs()).mean())
    rows, pick = [], {}
    for b in ("v3", "v5"):
        g = pick[b] = float(GRID[int(np.argmin([mae(d, b, x) for x in GRID]))])
        rows.append(f"| {b} + 세대 | {g} | {mae(d, b, 0):.2%} → {mae(d, b, g):.2%} | {mae(t, b, 0):.2%} → {mae(t, b, g):.2%} |")
    gain = mae(d, "v5", 0) - mae(d, "v5", pick["v5"])
    corr = float(np.corrcoef(d.hh, d.ps)[0, 1])
    ledger.append([dict(kind="model_result", key="marriage_upgrade_households", model="challenger_households",
                        payload={"delta_on_v3": pick["v3"], "delta_on_v5": pick["v5"], "design_gain_on_v5_pp": round(gain * 100, 3),
                                 "corr_with_youthpop": round(corr, 2), "decision": "v5 위 효과 없음 → 2026 예측 등록 안 함"},
                        evidence="설계 2013~2018, KOSIS DT_1B040B3")])
    (OUT / "challenger_households.md").write_text(f"""# 주민등록 세대 수 신호 (효과 없음으로 기록)

신호 = 올해 1~7월 세대 수 증가율 − 시도 평균 (KOSIS DT_1B040B3). 설계 2013~2018, 2019~2025는 참고.

| 모델 | 고른 δ | 2013~2018 오차 | 2019~2025 오차 (참고) |
|---|---|---|---|
{chr(10).join(rows)}

- 혼자 넣으면(v3 기준) 청년 인구 신호와 비슷하게 좋아집니다.
- 그런데 청년 인구 신호와의 상관이 {corr:+.2f}로 거의 같은 정보라, v5 위에서는 {gain * 100:.2f}%p만 줄어듭니다.
- 점검 결과를 보고 2026 예측은 등록하지 않고 '효과 없음'만 장부에 기록했습니다 (kind=model_result).
""", encoding="utf-8")
    print((OUT / "challenger_households.md").read_text())


if __name__ == "__main__":
    main()
