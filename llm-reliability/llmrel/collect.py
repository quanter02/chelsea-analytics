"""수집 파이프라인: 출처 점검 → 정규화 → 중복 제거 → 출처 분류 → 수치 추출 → 기존 주장과 대조.

출처
  KOSIS OpenAPI      KOSIS_API_KEY        한국 공식 통계
  e-Stat API         ESTAT_APP_ID         일본 공식 통계
  YouTube Data API   YOUTUBE_API_KEY      정리 채널의 새 영상 (제목·설명란)
  검색 기록 JSON     (키 없음)            웹 검색 결과를 저장한 파일

수치 추출은 규칙 기반 기준선입니다. LLM 추출은 API 키가 생기면 같은 정답 기준으로 비교합니다.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

# ── 출처 점검 ────────────────────────────────────────────────────────────
def _curl_json(url: str, params: dict, timeout: int = 30) -> tuple[int, str]:
    """프록시 CA를 쓰는 curl로 호출 (파이썬 urllib는 이 환경에서 연결이 끊기는 경우가 있음)."""
    args = ["curl", "-s", "-G", "--max-time", str(timeout), "-w", "\n%{http_code}", url]
    for k, v in params.items():
        args += ["--data-urlencode", f"{k}={v}"]
    out = subprocess.run(args, capture_output=True, text=True).stdout
    body, _, code = out.rpartition("\n")
    return int(code or 0), body


def clean_key(name: str) -> str | None:
    """환경 변수의 키에서 앞뒤에 잘못 붙은 글자를 걸러낸다 (예: 붙여넣을 때 섞인 접두어).
    KOSIS 키는 base64 44자, e-Stat 앱 ID는 16진수 40자 형식을 찾아 그 부분만 쓴다. 형식을 못 찾으면 원래 값."""
    import base64
    v = (os.environ.get(name) or "").strip().strip("\"'")
    if not v:
        return None
    if name == "ESTAT_APP_ID":
        m = re.search(r"[0-9a-f]{40}", v)
        return m[0] if m else v
    if name == "KOSIS_API_KEY" and len(v) % 4:
        for i in range(len(v) - 43):
            try:
                if re.fullmatch(rb"[0-9a-fA-F]{32}", base64.b64decode(v[i:i + 44], validate=True)):
                    return v[i:i + 44]
            except Exception:
                pass
    return v


def key_note(name: str) -> str:
    raw = (os.environ.get(name) or "").strip()
    return "" if clean_key(name) == raw else f" · 환경 변수 값에 불필요한 글자 {len(raw) - len(clean_key(name) or '')}자가 있어 걸러서 사용"


@dataclass
class Health:
    source: str
    ok: bool
    status: str
    detail: str


def check_kosis() -> Health:
    key = clean_key("KOSIS_API_KEY")
    if not key:
        return Health("KOSIS", False, "키 없음", "환경 변수 KOSIS_API_KEY 필요")
    for _ in range(2):                                         # 이 환경에서 간헐적으로 연결이 끊김
        code, body = _curl_json("https://kosis.kr/openapi/statisticsSearch.do",
                                dict(method="getList", apiKey=key, searchNm="혼인", resultCount=1, format="json", jsonVD="Y"))
        if code:
            break
    if not code:
        return Health("KOSIS", False, "연결 실패", "응답 없음")
    if '"err"' in body:
        return Health("KOSIS", False, "인증 실패", json.loads(body).get("errMsg", body)[:80])
    return Health("KOSIS", True, "정상", f"HTTP {code}" + key_note("KOSIS_API_KEY"))


def check_estat() -> Health:
    app = clean_key("ESTAT_APP_ID")
    if not app:
        return Health("e-Stat", False, "키 없음", "환경 변수 ESTAT_APP_ID 필요")
    code, body = _curl_json("https://api.e-stat.go.jp/rest/3.0/app/json/getStatsList",
                            dict(appId=app, searchWord="婚姻", limit=1))
    if not code:
        return Health("e-Stat", False, "연결 실패", "응답 없음")
    res = json.loads(body).get("GET_STATS_LIST", {}).get("RESULT", {})
    if res.get("STATUS") != 0:
        return Health("e-Stat", False, "인증 실패", res.get("ERROR_MSG", "")[:80])
    return Health("e-Stat", True, "정상", f"HTTP {code}" + key_note("ESTAT_APP_ID"))


def check_youtube() -> Health:
    key = os.environ.get("YOUTUBE_API_KEY")
    code, body = _curl_json("https://www.googleapis.com/youtube/v3/search",
                            dict(part="snippet", q="test", maxResults=1, **({"key": key} if key else {})))
    if not code:
        return Health("YouTube Data API", False, "연결 실패", "응답 없음")
    if not key:
        return Health("YouTube Data API", False, "키 없음", "서버 연결은 됨 · 환경 변수 YOUTUBE_API_KEY 필요")
    if code != 200:
        return Health("YouTube Data API", False, "인증 실패", body[:80])
    return Health("YouTube Data API", True, "정상", f"HTTP {code}")


# ── 검색 기록 → 항목 ──────────────────────────────────────────────────────
def load_search_run(path: str | Path) -> tuple[dict, pd.DataFrame]:
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for qi, q in enumerate(run["queries"]):
        for li, l in enumerate(q["links"]):
            rows.append(dict(query_id=qi, query=q["query"], lang=q["lang"], rank=li + 1, title=l["title"], url=l["url"]))
    return run, pd.DataFrame(rows)


def canonical_url(url: str) -> str:
    """모바일·AMP·추적 변형을 하나로 합친다."""
    s = urlsplit(url)
    host = re.sub(r"^(www\d*|m|mobile)\.", "", s.netloc.lower())
    path = re.sub(r"/amp/?$", "/", s.path).rstrip("/")
    keep = "&".join(p for p in s.query.split("&") if p and not p.startswith(("utm_", "fbclid", "pvs=")))
    return urlunsplit(("https", host, path, keep, ""))


OFFICIAL = ("go.kr", "go.jp", "kosis.kr", "index.go.kr", "e-stat", "mhlw", "stat.go.jp", "lg.jp", "pref.")
RESEARCH = ("kdi.re.kr", "ipss", "psrn.jp", "isvd.or.jp")
SECONDARY = ("tistory", "blog", "pe.kr", "datanow", "yugacrew", "koreabridge", "soranews24", "wikitree", "insight.co.kr")
FOREIGN_REPRINT = ("capitalfm", "nationthailand", "thesun.my", "bangkokpost", "thestar.com.my", "upi.com", "adnkronos")


def classify(url: str) -> str:
    h = urlsplit(url).netloc.lower()
    if any(k in h for k in OFFICIAL) or ".go.jp" in h:
        return "공식"
    if any(k in h for k in RESEARCH):
        return "연구·전문기관"
    if any(k in h for k in SECONDARY):
        return "2차 정리·블로그"
    if any(k in h for k in FOREIGN_REPRINT):
        return "해외 재인용"
    return "언론"


def url_date(url: str) -> str | None:
    """주소에 들어 있는 날짜 (YYYYMMDD 또는 /YYYY/MM/)."""
    m = re.search(r"(20\d{2})[/_-]?(0[1-9]|1[0-2])[/_-]?([0-3]\d)", url)
    if m:
        return f"{m[1]}-{m[2]}-{m[3]}"
    m = re.search(r"/(20\d{2})/(0[1-9]|1[0-2])/", url)
    return f"{m[1]}-{m[2]}" if m else None


# ── 수치 추출 (규칙 기반 기준선) ────────────────────────────────────────────
INDICATORS = [   # (지표, 키워드 정규식) — 앞에 있을수록 우선
    ("tfr", r"합계출산율|合計特殊出生率|total fertility rate|fertility rate"),
    ("mothers35_share", r"35세 이상 산모|mothers aged 35"),
    ("fert_30_34", r"30대 초반 여성의 출산율|early 30s"),
    ("natural_decrease", r"자연(?:감소|減)|natural decrease"),
    ("divorces", r"이혼|離婚|divorces?"),
    ("deaths", r"사망|死亡|deaths?"),
    ("marriages", r"혼인|婚姻|marriages?"),
    ("births", r"출생|出生|births?|babies"),
]
PERIOD_RX = [
    (r"1\s*[∼~～\-]\s*7월|January through July", "2026-01~07"),
    (r"1\s*[∼~～\-]\s*5월|January through May", "2026-01~05"),
    (r"1\s*[∼~～\-]\s*6月|上半期|January-June|first (?:six months|half)", "2026-H1"),
    (r"2분기|second quarter|Q2", "2026-Q2"),
    (r"7월|July", "2026-07"), (r"5월|May", "2026-05"), (r"4월|April", "2026-04"),
    (r"2025년 출생|2025", "2025"),
]
NUM = r"(\d{1,3}(?:,\d{3})+|\d+[만万]\s?\d{1,4}|\d+(?:\.\d+)?)"
SKIP_BEFORE = re.compile(r"(보다|by|up|증가한|exceed(?:ed)?|over)\s*$", re.I)        # 증감분·기준선
SKIP_AFTER = re.compile(r"^\s*(?:명|人|건|組)?\s*(?:을|를)?\s*(?:넘|이상)|^\s*(?:명|人|건|組)?\s*[増減]|^\s*명당|^\s*%p|^\s*per 1,000", re.I)
OLD_YEAR = re.compile(r"(?<!\d)(19\d{2}|20(?:0\d|1\d|2[0-4]))(?!\d)")            # 2025·2026 이외 연도 = 과거 비교값
NEGATIVE = re.compile(r"감소|줄|fell|declin|減少|減")


def _to_number(s: str) -> float:
    s = s.replace(",", "").replace(" ", "")
    for man in ("만", "万"):
        if man in s:
            a, b = s.split(man)
            return int(a) * 10000 + (int(b) if b else 0)
    return float(s)


def extract_claims(text: str, lang: str, version: int = 1) -> list[dict]:
    """version 0: 처음 규칙 / version 1: 증감분·과거 비교값·기간 이어받기 보완."""
    out, last_period = [], None
    for sent in re.split(r"(?<=[.。!?])\s+|(?<=다\.)|(?<=。)", text):
        if not sent.strip():
            continue
        period = next((p for rx, p in PERIOD_RX if re.search(rx, sent)), None)
        if version >= 1:
            period = period or last_period
            last_period = period
        hits = [(name, m.start()) for name, rx in INDICATORS for m in re.finditer(rx, sent, re.I)]
        if not hits:
            continue
        num_rx = NUM if version >= 1 else NUM.replace("[만万]", "만")
        for m in re.finditer(num_rx + r"\s*(%p|%|퍼센트|per ?cent|percent|명|건|人|組|件|babies|marriages)?", sent):
            raw, unit = m[1], (m[2] or "")
            val = _to_number(raw)
            if re.fullmatch(r"20\d{2}", raw) or (val < 1000 and not unit and "." not in raw):
                continue
            if version >= 1:
                before, after = sent[max(0, m.start() - 14):m.start()], sent[m.end() - len(unit):m.end() + 8]
                pct = unit in ("%", "퍼센트", "percent", "per cent")
                if unit == "%p" or (not pct and (SKIP_BEFORE.search(before) or SKIP_AFTER.search(after))) \
                        or OLD_YEAR.search(sent[max(0, m.start() - 10):m.end() + (6 if pct else 22)]):
                    continue                                      # 증감분(건수)·기준선·과거 비교값
                prev = [h for h in hits if h[1] <= m.start()]
                name = (max(prev, key=lambda h: h[1]) if prev else min(hits, key=lambda h: abs(h[1] - m.start())))[0]
            else:
                name = min(hits, key=lambda h: abs(h[1] - m.start()))[0]
            is_pct = unit in ("%", "퍼센트", "percent", "per cent")
            if is_pct and version >= 1 and NEGATIVE.search(sent[m.end():m.end() + 25] + sent[max(0, m.start() - 25):m.start()]):
                val = -val
            kind = name if name in ("tfr", "fert_30_34", "mothers35_share") else name + ("_yoy" if is_pct else "")
            cum = period and "~" in period
            if cum and name == "births":
                kind = "births_cum_yoy" if is_pct else "births_cum"
            out.append(dict(indicator=kind, period=period, value=val, unit=unit, sentence=sent.strip()[:120]))
    return out


def dedupe_claims(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(["country", "indicator", "period", "value"]).reset_index(drop=True)


def score_extraction(pred: pd.DataFrame, gold: pd.DataFrame) -> dict:
    key = lambda d: set(zip(d.country, d.indicator, d.period.fillna("?"), d.value.round(2)))
    p, g = key(pred), key(gold)
    tp = p & g
    loose = lambda d: set(zip(d.country, d.value.round(2)))
    return dict(predicted=len(p), gold=len(g), correct=len(tp), precision=len(tp) / max(len(p), 1), recall=len(tp) / max(len(g), 1),
                value_recall=len(loose(pred) & loose(gold)) / max(len(loose(gold)), 1),
                missed=sorted(g - p), wrong=sorted(p - g))
