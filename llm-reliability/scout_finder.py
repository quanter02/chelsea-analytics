"""셈법 스카우트 찾기: 조건을 고르면 2025/26 후보 선수가 바로 나오는 페이지를 만든다.

    python scout_finder.py    # scout_v2.py와 같은 캐시 필요 → data_cache/finder/index.html

선수 자료(트랜스퍼마크트 시장가치 등)가 들어가므로 결과 HTML은 레포에 올리지 않는다 (data_cache는 gitignore).
가성비 = 역할 백분위 − 시장가치 백분위 (같은 시즌·포지션 1500분+ 선수끼리). 양수면 가격에 비해 역할이 크다.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

from scout_v2 import build

ROOT = Path(__file__).parent
TPL = ROOT / "content" / "scout" / "finder.tpl.html"
OUT = ROOT / "data_cache" / "finder" / "index.html"
LG = {"EPL": "잉글랜드", "La_liga": "스페인", "Bundesliga": "독일", "Serie_A": "이탈리아", "Ligue_1": "프랑스"}


def main():
    warnings.filterwarnings("ignore")
    p, _ = build()
    c = p[(p.y == p.y.max()) & p.market_value_in_eur.notna()].copy()
    rows = []
    for r in c.itertuples():
        ce = str(r.contract_expiration_date)[:7] if isinstance(r.contract_expiration_date, str) else ""
        rows.append([r.name, int(r.age), r.pos, r.team, LG[r.lg], int(r.minutes), round(r.market_value_in_eur / 1e6, 2), ce,
                     round(r.role_pct * 100), round((r.role_pct - r.val_pct) * 100), int(bool(r.V))])
    html = TPL.read_text(encoding="utf-8").replace("/*DATA*/[]", json.dumps(rows, ensure_ascii=False))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"{OUT}  선수 {len(rows)}명, 셈법 명단 {sum(x[-1] for x in rows)}명")


if __name__ == "__main__":
    main()
