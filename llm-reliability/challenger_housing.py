"""아파트 입주 물량 신호 (v6). 사후 설계 → 도전자로만 등록.

    python challenger_housing.py   # → output/model_upgrade/challenger_housing.md, 장부 kind=nowcast_2026_v6

자료: 2025 주택총조사 '주택의 종류·연면적·건축연도별 주택 - 시군구'(KOSIS DT_1JU1520). 2025년에 남아 있는 아파트를
건축연도별로 세면 해마다 지어진(= 입주한) 아파트 수가 된다. 2010~2025 단일 연도. 철거로 조금 빠질 수 있음(최근 연도일수록 작음).
신호: 그해 지어진 아파트 수 ÷ 2025 전체 주택 수 − 같은 시도 평균.
  same = 올해 입주 (입주 예정 물량은 1~2년 전에 공개되지만, 시군구별 2026 예정 자료는 이 환경에서 받을 수 없음)
  lag  = 작년 입주 (2026 예측에 지금 바로 쓸 수 있음)
기준: v5(v3 + 1.75 × 청년 인구 신호). δ는 2013~2018에서만 고른다. 2019~2025는 이미 여러 번 봤으므로 증거가 아님.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import challenger_youthpop as Y
from challenger_size import frame
from llmrel import ledger, regional_tune as RT
from model_upgrade import OUT

GRID = np.round(np.arange(-1.0, 4.01, 0.25), 2)


def housing_signal(codes) -> pd.DataFrame:
    """연도 × 시군구 코드: 그해 지어진 아파트 ÷ 2025 전체 주택 − 시도 평균."""
    h = pd.read_csv("data_housing/kr_housing_by_built_year_2025.csv", dtype={"code": str})
    apt = h[h.kind == "아파트"].pivot(index="code", columns="built", values="value").fillna(0)
    stock = h[(h.kind == "주택_계") & (h.built == "T000")].set_index("code").value
    rate = pd.DataFrame({y: apt[f"T{y}"] / stock for y in range(2010, 2026)}).T[[c for c in codes if c in stock.index]]
    return rate - rate.T.groupby(rate.columns.str[:2]).transform("mean").T


def main():
    sgg, _ = RT.load()
    mr = pd.read_csv("data_regional/kr_marriages_by_region.csv", dtype={"code": str})
    names = mr.drop_duplicates("code").set_index("code").name.str.strip().to_dict()
    ps, hs = Y.pop_signal(sgg.columns, names), housing_signal(sgg.columns)
    rhos = [x["payload"]["rho_by_size"] for x in ledger.read() if x["kind"] == "model_candidate" and x["key"] == "marriage_upgrade_v3_size"][-1]
    gamma = [x["payload"]["gamma"] for x in ledger.read() if x["kind"] == "model_candidate" and x["key"] == "marriage_upgrade_v5_youthpop"][-1]

    def prep(years):
        f = frame(sgg, years)
        f["v3"] = f.base - f["size"].map(rhos).astype(float) * f.idio
        f["v5"] = f.v3 + gamma * pd.Series([ps[c].get(y, np.nan) for c, y in zip(f.code, f.year)], index=f.index)
        f["same"] = [hs[c].get(y, np.nan) for c, y in zip(f.code, f.year)]
        f["lag"] = [hs[c].get(y - 1, np.nan) for c, y in zip(f.code, f.year)]
        return f.dropna(subset=["v5", "same", "lag"])

    d, t = prep(Y.DESIGN), prep(Y.TEST)
    mae = lambda f, base, col, g: float(np.expm1((f.truth - (f[base] + g * f[col])).abs()).mean())
    pick = {(base, col): float(GRID[int(np.argmin([mae(d, base, col, g) for g in GRID]))]) for base in ("v3", "v5") for col in ("same", "lag")}
    ledger.append([dict(kind="model_candidate", key=f"marriage_upgrade_v6_housing_{col}", model="challenger_housing",
                        payload={"delta": pick[("v5", col)], "base": "v5", "signal": f"{'올해' if col == 'same' else '작년'} 입주 아파트 ÷ 전체 주택 − 시도 평균",
                                 "design": "2013~2018", "source": "2025 주택총조사 건축연도별(DT_1JU1520)", "note": "사후 설계"},
                        evidence="δ 등록, 판정은 2027년 3월 발표") for col in ("same", "lag")])
    rows = []
    for base in ("v3", "v5"):
        for col in ("same", "lag"):
            g = pick[(base, col)]
            rows.append(f"| {base} + {'올해 입주' if col == 'same' else '작년 입주'} | {g} | {mae(d, base, col, 0):.2%} → {mae(d, base, col, g):.2%} | "
                        f"{mae(t, base, col, 0):.2%} → {mae(t, base, col, g):.2%} | {np.corrcoef(d[col], d.truth - d[base])[0, 1]:+.2f} |")
    # 2026 v6: 지금 쓸 수 있는 건 작년(2025) 입주뿐
    delta = pick[("v5", "lag")]
    v5 = {x["key"]: x["payload"]["pred_2026"] for x in ledger.read() if x["kind"] == "nowcast_2026_v5"}
    ent, moved = [], []
    for c, p in v5.items():
        s = hs[c].get(2025, np.nan) if c in hs else np.nan
        v = int(round(p * np.exp(delta * s))) if np.isfinite(s) else p
        moved.append(v / p - 1)
        ent.append(dict(kind="nowcast_2026_v6", key=c, payload={"pred_2026": v, "delta": delta, "signal_2025": None if not np.isfinite(s) else round(float(s), 4)},
                        model="v6: v5 + 작년 아파트 입주 신호", evidence="사후 설계 도전자, 2027년 3월 판정"))
    n_add = ledger.append(ent)
    ok, total, _ = ledger.verify()
    nm = lambda c: f"{names.get(c[:2], '')} {names.get(c, c)}"
    top26 = ", ".join(f"{nm(c)} {v * 100:+.1f}%p" for c, v in hs.loc[2025].sort_values(ascending=False).head(6).items())
    (OUT / "challenger_housing.md").write_text(f"""# 아파트 입주 물량 신호 (v6, 사후 설계 도전자)

자료: 2025 주택총조사 건축연도별 아파트 수(KOSIS DT_1JU1520) → 시군구별 해마다 지어진 아파트. 신호 = 그해 입주 아파트 ÷ 전체 주택 − 시도 평균.
δ는 2013~2018에서만 골랐습니다. 2019~2025는 이미 여러 번 봤으므로 참고용입니다.

| 모델 | 고른 δ | 2013~2018 오차 | 2019~2025 오차 (참고) | 신호와 잔차의 상관 (설계) |
|---|---|---|---|---|
{chr(10).join(rows)}

## 읽는 법
- 입주 물량만 따로 넣으면(v3 기준) 오차가 꽤 줄어듭니다. 신호 자체는 진짜입니다.
- 그런데 월별 청년 인구(v5)가 이미 같은 효과를 대부분 잡고 있어서, 그 위에 더하면 개선이 작습니다. 입주하면 청년 인구가 바로 늘기 때문입니다.
- 입주 물량의 진짜 장점은 **시점**입니다. 입주 예정 물량은 1~2년 전에 공개되므로, 월별 인구가 쌓이기 전(연초)에 예측을 낼 수 있습니다.
  시군구별 2026 입주 예정 자료(국토부 통계누리·부동산원)는 이 환경의 네트워크에서 막혀 있어 아직 못 받았습니다.

## 2026 예측 (v6)
- 지금 쓸 수 있는 '작년(2025) 입주' 신호로 v5를 보정해 {n_add}줄 등록 (kind=nowcast_2026_v6, δ={delta}, 전체 {total}줄, 체인 {'정상' if ok else '어긋남'}).
  v5 대비 변화: 중앙값 {np.median(np.abs(moved)):.2%}, 최대 {np.max(np.abs(moved)):.1%}
- 2025년 입주가 시도보다 많았던 곳: {top26}
""", encoding="utf-8")
    print((OUT / "challenger_housing.md").read_text())


if __name__ == "__main__":
    main()
