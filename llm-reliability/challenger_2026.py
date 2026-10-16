"""2026년 시군구 혼인 예측 도전자(v2) 등록: 개선 후보 B (되돌림 보정 + 7개월 신호).

    python challenger_2026.py    # 장부 kind=nowcast_2026_v2 로 243곳 등록 (원래 kind=nowcast_2026 은 그대로) → output/model_upgrade/challenger_2026.md

예측 = 2025년 로그 건수 + 시도 2026년 1~7월 증가율 − 0.25 × 2025년 고유 변화.
채점: 2027년 3월 통계청 2026년 혼인 확정 발표, 원래 예측(v1)과 같은 지표로 나란히.
범위(80%)는 아직 새로 맞추지 않아 등록하지 않음 (다음 단계).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from llmrel import ledger, nowcast as NC, regional_tune as RT
from model_upgrade import OUT, RHO


def main():
    sgg, _ = RT.load()
    g7, g12 = NC.ytd_growth(7), NC.ytd_growth(12)
    v1 = {r["key"]: r["payload"]["pred_2026"] for r in ledger.read() if r["kind"] == "nowcast_2026"}
    names = pd.read_csv("content/b2b/districts_all.csv", dtype={"code": str}).set_index("code")["지역"]
    rows = []
    for c in sgg.columns:
        s = c[:2]
        last, prev = sgg.loc[2025, c], sgg.loc[2024, c]
        idio = (last - prev) - g12.loc[2025].get(s, np.nan)
        p = last + g7.loc[2026].get(s, np.nan) - RHO * idio
        if np.isfinite(p):
            rows.append(dict(code=c, name=names.get(c, c), last=int(round(np.exp(last))), v1=v1.get(c), v2=int(round(np.exp(p)))))
    d = pd.DataFrame(rows)
    n = ledger.append([dict(kind="nowcast_2026_v2", key=r.code, payload={"pred_2026": int(r.v2), "region": r.name},
                            model="v2: 2025 + 시도 1~7월 증가율 − 0.25×2025 고유 변화",
                            evidence="개선 후보 B (시험 구간 2019~2025 채택), 2026년 7월 월별 자료까지") for r in d.itertuples()])
    ok, total, _ = ledger.verify()
    d["diff"] = d.v2 / d.v1 - 1
    big = d.reindex(d["diff"].abs().sort_values(ascending=False).index).head(10)
    tbl = "\n".join(f"| {r.name} | {r.last:,} | {r.v1:,} | {r.v2:,} | {r.diff:+.1%} |" for r in big.itertuples())
    (OUT / "challenger_2026.md").write_text(f"""# 2026년 시군구 혼인 예측: 원래(v1) vs 도전자(v2)

- 등록: 장부 kind=nowcast_2026_v2, {n}줄 추가 (전체 {total}줄, 체인 {'정상' if ok else '어긋남'}). 원래 예측 v1은 그대로 둠
- v2 = 2025년 건수 × 시도 1~7월 증가율 × 되돌림 보정(작년에 튄 만큼의 25%)
- 전국 합계: v1 {d.v1.sum():,} · v2 {d.v2.sum():,} ({d.v2.sum() / d.v1.sum() - 1:+.1%})
- v1과 5% 넘게 다른 지역: {int((d['diff'].abs() > 0.05).sum())}곳
- 채점: 2027년 3월 통계청 2026년 혼인 발표, 두 예측을 같은 지표로 나란히 공개

## v1과 가장 많이 다른 10곳
| 지역 | 2025 실제 | v1 | v2 | 차이 |
|---|---|---|---|---|
{tbl}
""", encoding="utf-8")
    print((OUT / "challenger_2026.md").read_text())


if __name__ == "__main__":
    main()
