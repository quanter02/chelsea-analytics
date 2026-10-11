"""우리 동네 결혼 시장 카드: 243곳 데이터를 웹 조회 페이지용 JSON으로 만든다.

카드는 웹페이지(조회 → 카드 → 이미지 저장·공유)에서 그린다. 그림 코드는 content/web/town_section.html 하나뿐이고,
이 스크립트는 숫자만 만든다(숫자 = 공식 통계 + 장부).

카드에 넣는 것 (편 가르기 없이 '이 동네는 왜 이런가')
  - 미혼 성비 (25~39세 미혼 남성 ÷ 미혼 여성)와 전국 값
  - 두 요인: 인구 구성(젊은 남성 ÷ 여성) × 결혼 비율 차이(남성 미혼율 ÷ 여성 미혼율) = 미혼 성비
  - 2026년 혼인 예측과 80% 범위, 장부 기록 번호 (발표 전에 적어 둔 숫자)
  - 20~34세 순이동 (여성·남성)
  - 한계: 등록 주소 기준, 통근권과 다를 수 있음
넣지 않는 것: 전국 꼴찌 순위, 성별 비하 표현, 개인 정보

    python town_cards.py      # → content/town_cards/towns.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llmrel import ledger

ROOT = Path(__file__).parent
OUT = ROOT / "content" / "town_cards"


def build() -> dict:
    web = json.load(open(ROOT / "data_regional" / "regions_web.json", encoding="utf-8"))
    d = pd.read_csv(ROOT / "content" / "b2b" / "districts_all.csv", dtype={"code": str})
    h = {r["key"]: r["hash"][:16] for r in ledger.read() if r["kind"] == "nowcast_2026_range"}
    rows = []
    for _, r in d.iterrows():
        c = r.code
        w, parent = web["regions"].get(c), None
        if w is None:
            parent = c[:4] + "0"
            w = web["regions"][parent]
        name, ratio, popr, mratio, men, women, fpct, mpct, mig_f, mig_m, chg, kind = w
        rows.append({"c": c, "n": r["지역"], "r": round(float(r["미혼 성비"]), 2), "pop": round(popr, 2), "mr": round(mratio, 2),
                     "mf": round(mig_f, 1), "mm": round(mig_m, 1), "p25": int(r["2025 혼인"]), "pred": int(r["2026 예측"]),
                     "lo": int(r["80% 하한"]), "hi": int(r["80% 상한"]), "h": h.get(c, ""), "par": web["regions"][parent][0] if parent else ""})
    rows.sort(key=lambda x: x["n"])
    out = {"nat": {"r": round(web["nat"]["unmarried_ratio"], 2), "pop": round(web["nat"]["pop_ratio"], 2), "mr": round(web["nat"]["mrate_ratio"], 2)},
           "towns": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "towns.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return out




def section_html() -> str:
    """웹 섹션 = 템플릿 + 243곳 데이터."""
    tpl = (ROOT / "content" / "web" / "town_section.tpl.html").read_text(encoding="utf-8")
    data = (OUT / "towns.json").read_text(encoding="utf-8")
    return tpl.replace("__DATA__", data)


if __name__ == "__main__":
    o = build()
    (ROOT / "content" / "web" / "town_section.html").write_text(section_html(), encoding="utf-8")
    print(len(o["towns"]), "towns", (OUT / "towns.json").stat().st_size, "bytes")
