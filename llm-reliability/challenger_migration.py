"""청년 이동 신호 (v4). 사후 설계 → 도전자로만 등록.

    python challenger_migration.py   # → output/model_upgrade/challenger_migration.md, 장부 kind=nowcast_2026_v4

신호: 작년 25~34세 순이동률(남녀 평균, 100명당) − 같은 시도 평균. 젊은 사람이 들어오는 지역은 혼인이 시도보다 더 늘 것이라는 가설.
예측 = v3 + β × 신호/100.  β는 2016~2018(이동 자료가 2015년부터라 설계에 쓸 수 있는 해가 3년뿐)에서 평균 절대 오차가 가장 작은 값.
2019~2025는 앞 단계들에서 이미 여러 번 봤으므로 증거가 아님. 판정은 2027년 3월 발표.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from challenger_size import BINS, LABELS, frame
from llmrel import ledger, regional as R, regional_tune as RT
from model_upgrade import OUT

DESIGN, TEST = range(2016, 2019), range(2019, 2026)
GRID = np.round(np.arange(-1.0, 3.01, 0.25), 2)


def migration_signal(codes, names: dict) -> pd.DataFrame:
    """연도 × 시군구 코드: 25~34세 순이동률 평균 − 시도 평균."""
    m = pd.read_csv(R.ROOT / "data_regional" / "kr_migration_by_region.csv", dtype={"mcode": str}) if hasattr(R, "ROOT") else \
        pd.read_csv("data_regional/kr_migration_by_region.csv", dtype={"mcode": str})
    m = m[m.age_band.isin(["25-29", "30-34"]) & (m.mcode.str.len() == 5)]
    m = m.assign(sido=m.mcode.str[:2].map(R.MIG_SIDO), nm=m.name.str.strip())
    g = m.groupby(["year", "sido", "nm"]).value.mean()
    out = {}
    for c in codes:
        s, nm = c[:2], names.get(c, "")
        vals = {}
        for y in g.index.get_level_values(0).unique():
            gy = g.loc[y]
            if (s, nm) in gy.index:
                vals[y] = gy.loc[(s, nm)]
            else:
                sub = gy[(gy.index.get_level_values(0) == s) & gy.index.get_level_values(1).str.startswith(nm)] if nm else gy.iloc[0:0]
                if len(sub):
                    vals[y] = sub.mean()
        out[c] = vals
    sig = pd.DataFrame(out)
    sido_mean = sig.T.groupby(sig.columns.str[:2]).transform("mean").T
    return sig - sido_mean


def main():
    sgg, _ = RT.load()
    mr = pd.read_csv("data_regional/kr_marriages_by_region.csv", dtype={"code": str})
    names = mr.drop_duplicates("code").set_index("code").name.str.strip().to_dict()
    sig = migration_signal(sgg.columns, names)
    rhos = [x["payload"]["rho_by_size"] for x in ledger.read() if x["kind"] == "model_candidate" and x["key"] == "marriage_upgrade_v3_size"][-1]

    def prep(years):
        f = frame(sgg, years)
        f["rho"] = f["size"].map(rhos).astype(float)
        f["v3"] = f.base - f.rho * f.idio
        f["sig"] = [sig.get(c, pd.Series(dtype=float)).get(y - 1, np.nan) for c, y in zip(f.code, f.year)]
        return f.dropna(subset=["sig"])

    d, t = prep(DESIGN), prep(TEST)
    mae = lambda f, b: float(np.expm1((f.truth - (f.v3 + b * f.sig / 100)).abs()).mean())
    beta = float(GRID[int(np.argmin([mae(d, b) for b in GRID]))])
    cover = d.code.nunique()
    ledger.append([dict(kind="model_candidate", key="marriage_upgrade_v4_migration", model="challenger_migration",
                        payload={"beta": beta, "signal": "작년 25~34세 순이동률 − 시도 평균", "design": "2016~2018", "note": "사후 설계, 설계 자료 3년"},
                        evidence="β 등록, 판정은 2027년 3월 발표")])
    corr_d = float(np.corrcoef(d.sig, d.truth - d.v3)[0, 1])
    rows = [f"| 2016~2018 (설계) | {len(d)} | {mae(d, 0):.2%} | {mae(d, beta):.2%} | {corr_d:+.2f} |",
            f"| 2019~2025 (증거 아님) | {len(t)} | {mae(t, 0):.2%} | {mae(t, beta):.2%} | {np.corrcoef(t.sig, t.truth - t.v3)[0, 1]:+.2f} |"]
    t["size"] = t["size"].astype(str)
    sz = "\n".join(f"| {lab} | {mae(t[t['size'] == lab], 0):.2%} | {mae(t[t['size'] == lab], beta):.2%} |" for lab in LABELS if (t["size"] == lab).any())
    # 2026 v4
    v3 = {x["key"]: x["payload"]["pred_2026"] for x in ledger.read() if x["kind"] == "nowcast_2026_v3"}
    ent, moved = [], []
    for c, p in v3.items():
        s = sig.get(c, pd.Series(dtype=float)).get(2025, np.nan)
        v = int(round(p * np.exp(beta * s / 100))) if np.isfinite(s) else p
        moved.append(v / p - 1)
        ent.append(dict(kind="nowcast_2026_v4", key=c, payload={"pred_2026": v, "beta": beta, "signal_2025": None if not np.isfinite(s) else round(float(s), 2)},
                        model="v4: v3 + 청년 순이동 신호", evidence="사후 설계 도전자, 2027년 3월 판정"))
    n_add = ledger.append(ent)
    ok, total, _ = ledger.verify()
    (OUT / "challenger_migration.md").write_text(f"""# 청년 이동 신호 (v4, 사후 설계 도전자)

신호 = 작년 25~34세 순이동률(남녀 평균) − 같은 시도 평균. β는 2016~2018에서만 골랐고(후보 {GRID[0]}~{GRID[-1]}), **β = {beta}**.
설계 자료가 3년({cover}곳)뿐이고 2019~2025는 이미 여러 번 봤으므로, 아래 숫자는 참고용입니다. 판정은 2027년 3월 발표입니다.

| 구간 | 시군구-연도 | v3 오차 | v4 오차 | 신호와 v3 잔차의 상관 |
|---|---|---|---|---|
{chr(10).join(rows)}

## 크기별 (2019~2025, 참고)
| 작년 혼인 건수 | v3 | v4 |
|---|---|---|
{sz}

- 2026 v4 예측 {n_add}줄 등록 (kind=nowcast_2026_v4, 전체 {total}줄, 체인 {'정상' if ok else '어긋남'}). v3 대비 변화: 중앙값 {np.median(np.abs(moved)):.1%}, 최대 {np.max(np.abs(moved)):.1%}
""", encoding="utf-8")
    print((OUT / "challenger_migration.md").read_text())


if __name__ == "__main__":
    main()
