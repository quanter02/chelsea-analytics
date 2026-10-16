"""셈법 수익 모델 시제품 페이지 만들기 (공개 통계만 사용 → 결과도 레포에 올려도 됨).

    python products_build.py   # → content/products/demand.html (E3 지역 혼인 수요 조회)
                               #   content/products/factcheck.html (숫자 검증 서비스 견본)

자료: content/b2b/districts_all.csv (2026 예측·범위·신뢰도), data_regional/kr_marriages_by_region.csv (KOSIS 연간 혼인),
장부 kind=nowcast_2026 (등록 시각·해시).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llmrel import ledger

ROOT = Path(__file__).parent
P = ROOT / "content" / "products"


def main():
    d = pd.read_csv(ROOT / "content" / "b2b" / "districts_all.csv", dtype={"code": str})
    m = pd.read_csv(ROOT / "data_regional" / "kr_marriages_by_region.csv", dtype={"code": str})
    led = {r["key"]: r for r in ledger.read() if r["kind"] == "nowcast_2026"}
    rows = []
    for r in d.itertuples(index=False):
        s = m[(m.code == r.code) & (m.year >= 2015)].sort_values("year")
        l = led[r.code]
        rows.append(dict(c=r.code, n=r[1], v=int(r[3]), p=int(r[4]), lo=int(r[5]), hi=int(r[6]), r=float(r[2]),
                         strong=bool(r[9]), weak=bool(r[8] < 0), s=[[int(a), int(b)] for a, b in zip(s.year, s.value)],
                         at=l["recorded_at"][:10], h=l["hash"][:16]))
    rows.sort(key=lambda x: x["n"])
    (P / "demand.html").write_text((P / "demand.tpl.html").read_text(encoding="utf-8").replace("/*DATA*/[]", json.dumps(rows, ensure_ascii=False)), encoding="utf-8")

    names = dict(zip(d.code, d["지역"]))
    fc = {}
    for code, g in m.groupby("code"):
        if code.endswith("국외") or g.name.iloc[0] == "국외":
            continue
        nm = names.get(code, g.name.iloc[0]) if len(code) == 5 else g.name.iloc[0]
        fc[nm] = {int(y): int(v) for y, v in zip(g.year, g.value)}
    (P / "factcheck.html").write_text((P / "factcheck.tpl.html").read_text(encoding="utf-8").replace("/*DATA*/{}", json.dumps(fc, ensure_ascii=False)), encoding="utf-8")
    print(f"demand {len(rows)}곳, factcheck {len(fc)}곳")


if __name__ == "__main__":
    main()
