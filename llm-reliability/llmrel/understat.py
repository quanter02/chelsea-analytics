"""understat.com 에서 프리미어리그 경기별 xG 받기.

understat 시즌 표기는 시작 연도 (2026 = 2026/27).
두 가지 방식을 순서대로 시도한다:
  1) JSON 엔드포인트  https://understat.com/getLeagueData/EPL/<연도>   (최근 사이트)
  2) 리그 페이지 HTML 안의  var datesData = JSON.parse('...')        (예전 사이트)
네트워크 허용 목록에 understat.com 이 있어야 한다.
"""
from __future__ import annotations

import codecs
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

from .epl import norm

# understat 팀 이름 → openfootball 표기 (norm 이후)
NAMES = {"Brighton": "Brighton & Hove Albion", "Tottenham": "Tottenham Hotspur", "West Ham": "West Ham United",
         "Wolverhampton Wanderers": "Wolverhampton Wanderers", "Newcastle United": "Newcastle United", "Leeds": "Leeds United",
         "Ipswich": "Ipswich Town", "Coventry": "Coventry City", "Hull": "Hull City", "Luton": "Luton Town",
         "Sheffield United": "Sheffield United", "Nottingham Forest": "Nottingham Forest", "Leicester": "Leicester City",
         "Norwich": "Norwich City", "Cardiff": "Cardiff City", "Swansea": "Swansea City", "Stoke": "Stoke City",
         "Huddersfield": "Huddersfield Town", "West Bromwich Albion": "West Bromwich Albion", "Middlesbrough": "Middlesbrough"}


def _curl(url: str, headers: dict | None = None, timeout: int = 40) -> tuple[int, str]:
    args = ["curl", "-s", "-L", "-m", str(timeout), "-w", "\n%{http_code}", "-A", "Mozilla/5.0"]
    for k, v in (headers or {}).items():
        args += ["-H", f"{k}: {v}"]
    out = subprocess.run(args + [url], capture_output=True, text=True).stdout
    body, _, code = out.rpartition("\n")
    return int(code or 0), body


def parse_dates(items: list[dict]) -> pd.DataFrame:
    """datesData 항목들 → 끝난 경기의 date, home, away, hxg, axg."""
    rows = []
    for m in items:
        if not m.get("isResult"):
            continue
        h, a = m["h"]["title"], m["a"]["title"]
        rows.append(dict(date=str(m["datetime"])[:10], home=NAMES.get(h, norm(h)), away=NAMES.get(a, norm(a)),
                         hxg=float(m["xG"]["h"]), axg=float(m["xG"]["a"]), hg=int(m["goals"]["h"]), ag=int(m["goals"]["a"]), source="understat"))
    return pd.DataFrame(rows)


def parse_html(html: str) -> list[dict]:
    m = re.search(r"var\s+datesData\s*=\s*JSON\.parse\('(.*?)'\)", html, re.S)
    if not m:
        raise ValueError("datesData 를 찾지 못함 (사이트 구조가 바뀌었을 수 있음)")
    return json.loads(codecs.decode(m.group(1), "unicode_escape"))


def fetch_season(year: int) -> pd.DataFrame:
    code, body = _curl(f"https://understat.com/getLeagueData/EPL/{year}", {"X-Requested-With": "XMLHttpRequest"})
    if code == 200 and body.strip().startswith("{"):
        return parse_dates(json.loads(body).get("dates", []))
    code, body = _curl(f"https://understat.com/league/EPL/{year}")
    if code != 200:
        raise RuntimeError(f"understat 연결 실패 (HTTP {code}). 네트워크 허용 목록에 understat.com 이 있는지 확인")
    return parse_dates(parse_html(body))


def update(year: int, out: str | Path) -> int:
    df = fetch_season(year)
    df.to_csv(out, index=False)
    return len(df)
