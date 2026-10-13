"""사업보고서 원문(document.xml)에서 지배주주 순이익·자본과 회사가 공시한 ROE를 꺼낸다.

DART 재무제표 API(fnlttSinglAcnt/All)는 금융회사의 2022 사업연도 이전 값을 주지 않는다(status 013).
그 공백(2019~2020)을 원문 표로 채우고, 원문 경영지표 표의 ROE를 '은행 공시 ROE'로 쓴다.

    DART_API_KEY=... python invest/dart_docs.py      # out/bank_docs.csv
"""
from __future__ import annotations

import html, io, json, os, re, time, zipfile
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "out"
CACHE = Path(__file__).parent / "data_cache" / "dart_docs"
BANKS = {"055550": "신한지주", "105560": "KB금융", "086790": "하나금융", "316140": "우리금융", "024110": "기업은행",
         "138930": "BNK금융", "139130": "iM금융", "175330": "JB금융", "323410": "카카오뱅크"}
DOC_YEARS = (2020, 2022, 2024, 2025)  # 각 보고서가 3년치 순이익·자본, 2년치 공시 ROE를 담는다
NUM = re.compile(r"^\(?-?[\d,]+(\.\d+)?\)?$")


def _get(url: str, timeout: int = 600) -> bytes:
    for i in range(5):
        try:
            return urllib.request.urlopen(url, timeout=timeout).read()
        except Exception:
            time.sleep(2 ** i + 2)
    raise RuntimeError(url.split("crtfc_key")[0])


def report_index(key: str, codes: dict) -> dict:
    """{(종목코드, 사업연도): [rcept_no, ...]} 최근 접수(정정 반영)부터. [첨부정정]은 본문이 없을 수 있어 뺀다."""
    idx: dict = {}
    for sc in BANKS:
        d = json.loads(_get(f"https://opendart.fss.or.kr/api/list.json?crtfc_key={key}&corp_code={codes[sc]}"
                            f"&bgn_de=20200101&end_de=20261231&pblntf_detail_ty=A001&page_count=100"))
        time.sleep(1)
        for x in sorted(d.get("list", []), key=lambda x: x["rcept_no"], reverse=True):
            if "첨부정정" not in x["report_nm"]:
                idx.setdefault((sc, int(x["report_nm"].split("(")[-1][:4])), []).append(x["rcept_no"])
    return idx


def fetch_doc(key: str, rnos: list[str]) -> tuple[str, bytes] | None:
    """원문 zip. 정정본 파일이 없으면(status 014 등 XML 응답) 이전 접수분으로 내려간다."""
    for rno in rnos:
        p = CACHE / f"{rno}.zip"
        if not p.exists():
            b = _get(f"https://opendart.fss.or.kr/api/document.xml?crtfc_key={key}&rcept_no={rno}")
            time.sleep(1)
            if not zipfile.is_zipfile(io.BytesIO(b)):
                print("원문 없음", rno, b[:200].decode("utf-8", "replace"))
                continue
            p.write_bytes(b)
        return rno, p.read_bytes()
    return None


def doc_text(blob: bytes) -> str:
    """원문 zip의 본문 XML. 인코딩 선언(utf-8)과 달리 cp949인 파일이 있다."""
    z = zipfile.ZipFile(io.BytesIO(blob))
    raw = z.read(min(z.namelist(), key=len))  # 본문 = 접미사(_00760 등) 없는 파일
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp949", errors="replace")


def mdna(t: str) -> str:
    """'IV. 이사의 경영진단 및 분석의견' 섹션. 그룹 ROE는 여기 경영지표 표에 있다(앞쪽 ROE 표는 자회사 것이 많다)."""
    m = re.search(r"<TITLE[^>]*>\s*IV\.\s*이사의\s*경영진단[^<]*</TITLE>", t)
    if not m:
        return ""
    n = re.search(r"<TITLE[^>]*>\s*V\.\s", t[m.end():])
    return t[m.end(): m.end() + n.start()] if n else t[m.end():]


def rows_of(t: str) -> list[list[str]]:
    rows = []
    for tr in re.finditer(r"<TR[^>]*>(.*?)</TR>", t, re.S | re.I):
        cells = re.findall(r"<(?:TD|TH|TE|TU)[^>]*>(.*?)</(?:TD|TH|TE|TU)>", tr.group(1), re.S | re.I)
        rows.append([re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", c.replace("&cr;", " ")))).strip()
                     for c in cells])
    return rows


def _num(s: str) -> float:
    s = s.replace(",", "").strip()
    neg = s.startswith("(") or s.startswith("-")
    v = float(s.strip("()-"))
    return -v if neg else v


def _nums(cells: list[str]) -> list[float]:
    return [_num(c) for c in cells if NUM.match(c.replace(" ", ""))]


ROE_LABEL = re.compile(r"(ROE|자기자본(순)?이익(률|율))(\(.*\))?(\(주\d\)|\*\d?)?")
PCT = re.compile(r"-?\d+(\.\d+)?%?")


def _head(c: str) -> str:
    return re.sub(r"[\sⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVX.\d()\[\]]", "", c)


def _year_cols(rows: list[list[str]], i: int) -> list[int | None] | None:
    """i행 위 12행 안의 연도 헤더(2024년, 2023년(B) …)를 오른쪽 정렬로 i행 셀 위치에 맞춘다."""
    for j in range(i - 1, max(-1, i - 13), -1):
        h = rows[j]
        ys = [int(m.group(1)) if (m := re.match(r"(20\d\d)\s*년?", c.replace(" ", ""))) else None for c in h]
        if sum(y is not None for y in ys) >= 2:
            n = len(rows[i])
            return [None] * max(0, n - len(ys)) + ys[-n:] if len(ys) <= n else ys[-n:]
    return None


def parse_roe(rows: list[list[str]], fy: int) -> tuple[dict, str] | None:
    """경영지표 표의 그룹 ROE {연도: %}와 근거 행.

    1) 'ROE(지배기업지분)' 행의 첫 셀이 '12.98 (13.00)' 꼴이면(JB) 괄호 안이 지배기업지분 ROE, 표가 당기→전기 순으로 나온다.
    2) 그 밖에는 'ROE'/'자기자본(순)이익률' 행. 지배기준(비지배 제외) 행이 있으면 그것을 쓴다.
       위쪽 연도 헤더가 있으면 열을 연도에 맞춘다. 없으면 '당기·증감·증감률(-)·전기' 꼴(셋째 칸 '-')은 첫째·넷째,
       그 밖에는 증감(괄호·%p·△·+ 표기)을 뺀 앞의 두 값을 당기·전기로 본다.
    """
    jb = [r for r in rows if r and "지배기업지분" in r[0].replace(" ", "") and r[0].replace(" ", "").startswith("ROE")]
    if jb:
        vals = [float(m.group(1)) for r in jb if len(r) > 1 and (m := re.search(r"\((\d+\.\d+)\)", r[1]))]
        if len(vals) >= 2:
            return {fy: vals[0], fy - 1: vals[1]}, " / ".join(" ".join(r[:3]) for r in jb[:2])
    cand = [(i, r, k) for i, r in enumerate(rows) for k, c in enumerate(r) if ROE_LABEL.fullmatch(c.replace(" ", ""))]
    if not cand:
        return None
    ctrl = [x for x in cand if "제외" in " ".join(x[1])]
    i, r, k = (ctrl or cand)[0]
    yc = _year_cols(rows, i)
    out = {}
    if yc:
        for c, y in zip(r, yc):
            if y in (fy, fy - 1) and PCT.fullmatch(c.replace(" ", "")):
                out[y] = float(c.replace(" ", "").rstrip("%"))
    if len(out) < 2:
        raw = [x.replace(" ", "") for x in r[k + 1:] if not re.fullmatch(r"[^\d]*[가-힣][^\d]*", x)]  # 설명 글 셀 제외
        if len(raw) >= 4 and raw[2] == "-" and PCT.fullmatch(raw[0]) and PCT.fullmatch(raw[3]):
            out = {fy: float(raw[0].rstrip("%")), fy - 1: float(raw[3].rstrip("%"))}  # 당기·증감·증감률(-)·전기
        else:
            v = [float(x.rstrip("%")) for x in raw if PCT.fullmatch(x)]
            if len(v) >= 2:
                out = {fy: v[0], fy - 1: v[1]}
    return (out, " ".join(r[:7])) if len(out) == 2 else None


def parse_fin(rows: list[list[str]]) -> dict:
    """요약연결재무정보(III. 재무에 관한 사항 첫 표들)에서 지배주주 자본·순이익 3년치(백만원)."""
    def vals(r):
        v = _nums(r[1:])[:3]
        return [x / 1e6 for x in v] if v and max(abs(x) for x in v) > 1e10 else v  # 원 단위 공시

    eq = nci = tot = ni = ni_ctrl = None
    for r in rows[:300]:
        if not r or len(_nums(r[1:])) < 3:
            continue
        h = _head(r[0])
        if ni is None:  # 재무상태표 구간
            if eq is None and re.fullmatch(r"지배기업(의)?(소유주|주주|소유)?(에게귀속되는)?(지분|자본)|지배주주지분", h):
                eq = vals(r)
            elif nci is None and re.fullmatch(r"비지배(주주)?지분", h):
                nci = vals(r)
            elif tot is None and h == "자본총계":
                tot = vals(r)
            elif "당기순이익" in h and "포괄" not in h:
                ni = vals(r)
        elif ni_ctrl is None and "지배" in h and "비지배" not in h and "포괄" not in h:
            ni_ctrl = vals(r)
            break
    out = {}
    if eq is None and tot is not None:
        eq = [t - n for t, n in zip(tot, nci)] if nci else tot  # 지배 행이 없으면 자본총계 − 비지배지분
    if eq:
        out["equity_ctrl"] = eq
    if ni_ctrl is None and ni is not None and nci is None:
        ni_ctrl = ni  # 종속회사가 없으면 순이익 전체가 지배주주 몫
    if ni_ctrl:
        out["ni_ctrl"] = ni_ctrl
    return out


def parse(blob: bytes, fy: int) -> dict:
    """{'ni_ctrl': {연도: 백만원}, 'equity_ctrl': {...}, 'roe': {연도: %}, 'roe_src': 근거 행}."""
    t = doc_text(blob)
    out: dict = {}
    m = re.search(r"<TITLE[^>]*>\s*III\.\s*재무에[^<]*</TITLE>", t)
    fin = parse_fin(rows_of(t[m.end(): m.end() + 600000])) if m else {}
    if "ni_ctrl" not in fin:  # 주요배당지표 표의 (연결)당기순이익 — 서식상 지배주주 기준(일부 회사는 전체 순이익)
        for r in rows_of(t):
            if r and r[0].replace(" ", "").startswith("(연결)당기순이익") and len(_nums(r[1:])) >= 3:
                fin["ni_ctrl"] = _nums(r[1:])[:3]
                break
    for f, v in fin.items():
        out[f] = {fy - k: x for k, x in enumerate(v)}
    roe = parse_roe(rows_of(mdna(t)), fy)
    if roe:
        out["roe"], out["roe_src"] = roe
    return out


def bank_docs(key: str, codes: dict) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    idx = report_index(key, codes)
    rows = []
    for (sc, fy), rnos in sorted(idx.items()):
        if fy not in DOC_YEARS or not (doc := fetch_doc(key, rnos)):
            continue
        rno, blob = doc
        got = parse(blob, fy)
        src = got.pop("roe_src", "")
        for f, byyear in got.items():
            for y, v in byyear.items():
                rows.append({"종목": BANKS[sc], "연도": y, "항목": f, "값": v, "보고서": fy, "rcept_no": rno,
                             "근거": src if f == "roe" else ""})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from fin_compare import dart_corp_codes
    key = os.environ["DART_API_KEY"].strip()
    OUT.mkdir(exist_ok=True)
    d = bank_docs(key, dart_corp_codes(key))
    d.to_csv(OUT / "bank_docs.csv", index=False)
    print(d.pivot_table(index=["종목", "항목"], columns="연도", values="값", aggfunc="first").to_string())
