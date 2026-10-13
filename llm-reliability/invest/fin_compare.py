"""금융주 장기 보유 비교 (가격 + DART 재무지표).

    python invest/fin_compare.py            # 가격 지표 (야후, query1.finance.yahoo.com 허용 필요)
    DART_API_KEY=... python invest/fin_compare.py --dart   # + 재무지표 (opendart.fss.or.kr 허용 필요)

가격 지표: 10년 연수익(배당 반영), 변동성, 최대 낙폭, 하락/상승 포착(주봉, 2018-01-29~2020-06-30 제외),
금리 인상 하락장(2021-06-25~2022-09-30) 성과, 코스피 3년 초과 비율, 알파벳(원화)과의 상관.
재무지표(DART, 연결 사업보고서): ROE, 순이익 성장, 자본, 배당 — 은행용 관문(자본·ROE·연체)에 쓴다.
건전성(FISIS, 은행 단위 분기): BIS·보통주자본비율, 연체율, 고정이하여신비율 — 지주의 핵심 은행으로 본다(JB=전북+광주).
ROE(FISIS, 지주 연결, 자기자본 평잔): 관문은 최근 3개 사업연도 평균. DART ROE는 교차 확인용.

    FISIS_API_KEY=... python invest/fin_compare.py --gate   # 건전성 + ROE 관문 비교 (fisis.fss.or.kr 허용, 브라우저 UA 필요)

개인 계좌 정보는 이 파일에 넣지 않는다.
"""
from __future__ import annotations

import io, json, os, sys, time, zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "out"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
STOCKS = {"055550.KS": "신한지주", "105560.KS": "KB금융", "086790.KS": "하나금융", "316140.KS": "우리금융", "024110.KS": "기업은행",
          "138930.KS": "BNK금융", "139130.KS": "iM금융", "175330.KS": "JB금융", "000810.KS": "삼성화재", "032830.KS": "삼성생명",
          "005830.KS": "DB손해보험", "138040.KS": "메리츠금융", "323410.KS": "카카오뱅크", "039490.KS": "키움증권",
          "006800.KS": "미래에셋증권", "071050.KS": "한국금융지주", "016360.KS": "삼성증권", "JPM": "JP모건", "BRK-B": "버크셔B", "V": "비자"}
EX_A, EX_B = "2018-01-29", "2020-06-30"
RH_A, RH_B = "2021-06-25", "2022-09-30"


def get(url: str) -> bytes:
    for i in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
        except Exception:
            time.sleep(2 ** i)
    raise RuntimeError(url)


def yahoo(sym: str) -> pd.Series:
    d = json.loads(get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=10y&interval=1d&events=div"))["chart"]["result"][0]
    ts = pd.to_datetime(d["timestamp"], unit="s").normalize()
    return pd.Series(d["indicators"]["adjclose"][0]["adjclose"], index=ts).dropna().groupby(level=0).last()


def price_metrics() -> pd.DataFrame:
    fx, kospi, goog = yahoo("KRW=X"), yahoo("^KS11"), yahoo("GOOGL")
    goog_krw = (goog * fx.reindex(goog.index).ffill()).dropna()
    wk = lambda s: np.log(s.resample("W-FRI").last()).diff().dropna()
    rows = []
    for sym, name in STOCKS.items():
        s = yahoo(sym)
        if not sym.endswith(".KS"):
            s = (s * fx.reindex(s.index).ffill()).dropna()
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        j = pd.concat([wk(s).rename("x"), wk(kospi).rename("k"), wk(goog_krw).rename("g")], axis=1, sort=True).dropna()
        jj = j[~((j.index >= EX_A) & (j.index <= EX_B))]
        dn, up = jj[jj.k < 0], jj[jj.k > 0]
        seg = s[(s.index >= RH_A) & (s.index <= RH_B)]
        m = s.resample("ME").last(); km = kospi.resample("ME").last().reindex(m.index)
        r3 = ((m.shift(-36) / m) / (km.shift(-36) / km)).dropna()
        rows.append({"종목": name, "코드": sym, "시작": s.index[0].date(), "연수익": (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1,
                     "변동성": np.log(s).diff().std() * np.sqrt(252), "최대낙폭": (s / s.cummax() - 1).min(),
                     "하락포착": dn.x.mean() / dn.k.mean(), "상승포착": up.x.mean() / up.k.mean(),
                     "금리하락장": seg.iloc[-1] / seg.iloc[0] - 1 if len(seg) > 10 else np.nan,
                     "코스피3년초과": (r3 > 1).mean() if len(r3) else np.nan, "알파벳상관": j.x.corr(j.g)})
    return pd.DataFrame(rows).set_index("종목")


def dart_corp_codes(key: str) -> dict:
    z = zipfile.ZipFile(io.BytesIO(get(f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={key}")))
    import xml.etree.ElementTree as ET
    root = ET.fromstring(z.read(z.namelist()[0]))
    return {e.findtext("stock_code").strip(): e.findtext("corp_code") for e in root.iter("list") if (e.findtext("stock_code") or "").strip()}


def dart_fin(key: str, corp: str, year: int) -> dict:
    """연결 사업보고서(11011) 주요 계정: 자본총계, 당기순이익(지배)."""
    url = f"https://opendart.fss.or.kr/api/fnlttSinglAcnt.json?crtfc_key={key}&corp_code={corp}&bsns_year={year}&reprt_code=11011"
    d = json.loads(get(url))
    out = {}
    for r in d.get("list", []):
        if r.get("fs_div") != "CFS":
            continue
        nm, v = r["account_nm"], r.get("thstrm_amount", "").replace(",", "")
        if not v.lstrip("-").isdigit():
            continue
        if nm.startswith("자본총계"):
            out["equity"] = int(v)
        elif nm.startswith("당기순이익"):
            out.setdefault("ni", int(v))
    return out


def fundamentals(key: str, years=range(2019, 2026), names=None) -> pd.DataFrame:
    codes = dart_corp_codes(key)
    rows = []
    for sym, name in STOCKS.items():
        if not sym.endswith(".KS") or (names and name not in names):
            continue
        corp = codes.get(sym[:6])
        if not corp:
            continue
        for y in years:
            f = dart_fin(key, corp, y)
            if f:
                rows.append({"종목": name, "연도": y, **f})
            time.sleep(0.2)
    d = pd.DataFrame(rows)
    d["ROE"] = d.ni / d.equity
    return d


FISIS_BANKS = {"신한은행": "0011625", "국민은행": "0010927", "하나은행": "0013909", "우리은행": "0010001",
               "전북은행": "0010022", "광주은행": "0010019"}
HOLDCO_BANKS = {"신한지주": ["신한은행"], "KB금융": ["국민은행"], "하나금융": ["하나은행"], "우리금융": ["우리은행"],
                "JB금융": ["전북은행", "광주은행"]}
HOLDCO_FISIS = {"신한지주": "0010926", "KB금융": "0012028", "하나금융": "0011573", "우리금융": "0016619", "JB금융": "0013406"}
# 통계표: 계정코드 -> 이름. 비율은 금액에서 다시 계산해 JB(전북+광주)를 합산할 수 있게 한다.
# B(자기자본)는 공제 후 금액이지만 B0(보통주자본)는 공제 전이라 B30을 빼야 공시 CET1과 맞는다.
FISIS_ITEMS = {"SA014": {"B": "자기자본", "B0": "보통주자본", "B30": "보통주공제", "C": "위험가중자산", "A": "BIS", "A2": "CET1"},
               "SA040": {"A1": "대출잔액", "A2": "연체잔액", "A3": "연체율"},
               "SA015": {"A": "총여신", "B4": "고정이하", "C": "NPL"}}


def fisis(key: str, list_no: str, finance_cd: str, start: str, end: str) -> list:
    url = (f"https://fisis.fss.or.kr/openapi/statisticsInfoSearch.json?lang=kr&auth={key}&financeCd={finance_cd}"
           f"&listNo={list_no}&term=Q&startBaseMm={start}&endBaseMm={end}")
    r = json.loads(get(url))["result"]
    if r.get("err_cd") != "000":
        raise RuntimeError(f"FISIS {list_no} {finance_cd}: {r.get('err_msg')}")
    return r.get("list", [])


def fisis_raw(key: str, start="201903", end=None) -> pd.DataFrame:
    """은행별 분기 원자료 (금액은 원, 비율은 %)."""
    end = end or pd.Timestamp.today().strftime("%Y%m")
    rows = []
    for bank, cd in FISIS_BANKS.items():
        for list_no, items in FISIS_ITEMS.items():
            for r in fisis(key, list_no, cd, start, end):
                if r["account_cd"] in items and r["a"].strip():
                    rows.append({"은행": bank, "분기": r["base_month"], "항목": items[r["account_cd"]], "값": float(r["a"])})
            time.sleep(0.2)
    return pd.DataFrame(rows).pivot_table(index=["은행", "분기"], columns="항목", values="값").reset_index()


def holdco_roe(key: str, start="201912", end=None) -> pd.DataFrame:
    """지주 연결 ROE(%, 자기자본 평잔 기준, FISIS SL010). 12월은 연간, 그 외 분기는 연환산 누적."""
    end = end or pd.Timestamp.today().strftime("%Y%m")
    rows = []
    for hold, cd in HOLDCO_FISIS.items():
        for r in fisis(key, "SL010", cd, start, end):
            if r["account_cd"] == "G" and r["a"].strip():
                rows.append({"지주": hold, "분기": r["base_month"], "ROE": float(r["a"])})
        time.sleep(0.2)
    return pd.DataFrame(rows)


def soundness(raw: pd.DataFrame) -> pd.DataFrame:
    """지주별 분기 건전성(%). 은행 하나면 FISIS 공시 비율을 그대로 쓰고,
    합산 은행(JB=전북+광주)은 금액을 더한 뒤 비율을 낸다. (국민은행은 25.2Q부터 금액으로 CET1이 재현되지 않는다)"""
    rows = []
    for hold, banks in HOLDCO_BANKS.items():
        g = raw[raw.은행.isin(banks)].groupby("분기")
        if len(banks) == 1:
            s = g[["BIS", "CET1", "연체율", "NPL"]].first()
        else:
            a = g[["자기자본", "보통주자본", "보통주공제", "위험가중자산", "대출잔액", "연체잔액", "총여신", "고정이하"]].sum(min_count=1)
            a = a[g.size() == len(banks)]  # 한 은행만 있는 분기는 버린다
            s = pd.DataFrame({"BIS": 100 * a.자기자본 / a.위험가중자산,
                              "CET1": 100 * (a.보통주자본 - a.보통주공제) / a.위험가중자산,
                              "연체율": 100 * a.연체잔액 / a.대출잔액, "NPL": 100 * a.고정이하 / a.총여신})
        rows.append(s.assign(지주=hold).reset_index())
    return pd.concat(rows, ignore_index=True)[["지주", "분기", "BIS", "CET1", "연체율", "NPL"]]


# 관문 기준 (2019년 이후 분기 자료, 최근 분기 기준)
GATE = {"BIS": 14.0, "CET1": 12.0, "연체율": 0.6, "NPL": 0.6, "연체율_4Q상승": 0.2, "ROE_3년": 8.0}  # 모두 %


def gate_table(snd: pd.DataFrame, roe: pd.DataFrame) -> pd.DataFrame:
    """건전성 관문(최근 분기, 핵심 은행) + ROE 관문(지주 연결, 최근 3개 사업연도 평균)."""
    last = snd.분기.max()
    cur = snd[snd.분기 == last].set_index("지주")
    prev = snd[snd.분기 == sorted(snd.분기.unique())[-5]].set_index("지주")
    worst = snd.groupby("지주")[["연체율", "NPL"]].max()
    yr = roe[roe.분기.str.endswith("12")].assign(연도=lambda d: d.분기.str[:4]).pivot(index="지주", columns="연도", values="ROE")
    yrs = sorted(yr.columns)[-3:]
    ytd = roe[roe.분기 == roe.분기.max()].set_index("지주").ROE
    t = pd.DataFrame({"기준분기": last, "BIS": cur.BIS, "CET1": cur.CET1, "연체율": cur.연체율,
                      "연체율_4Q변화": cur.연체율 - prev.연체율, "연체율_19년후최고": worst.연체율,
                      "NPL": cur.NPL, "NPL_4Q변화": cur.NPL - prev.NPL,
                      f"ROE_{yrs[0]}-{yrs[-1]}": yr[yrs].mean(axis=1), f"ROE_{roe.분기.max()}": ytd}).reindex(list(HOLDCO_BANKS))
    t["자본관문"] = (t.BIS >= GATE["BIS"]) & (t.CET1 >= GATE["CET1"])
    t["자산관문"] = (t.연체율 <= GATE["연체율"]) & (t.NPL <= GATE["NPL"]) & (t.연체율_4Q변화 <= GATE["연체율_4Q상승"])
    t["ROE관문"] = t[f"ROE_{yrs[0]}-{yrs[-1]}"] >= GATE["ROE_3년"]
    t["통과"] = t.자본관문 & t.자산관문 & t.ROE관문
    return t


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    p = price_metrics(); p.to_csv(OUT / "fin_price_metrics.csv")
    print(p.round(3).to_string())
    if "--dart" in sys.argv:
        key = os.environ.get("DART_API_KEY", "").strip()
        if not key:
            sys.exit("DART_API_KEY 환경 변수가 없습니다 (새 세션에서 적용됨)")
        f = fundamentals(key); f.to_csv(OUT / "fin_fundamentals.csv", index=False)
        print(f.pivot_table(index="종목", columns="연도", values="ROE").round(3).to_string())
    if "--gate" in sys.argv:
        fk, dk = os.environ.get("FISIS_API_KEY", "").strip(), os.environ.get("DART_API_KEY", "").strip()
        if not fk:
            sys.exit("FISIS_API_KEY 환경 변수가 필요합니다 (DART_API_KEY는 교차 확인용, 선택)")
        raw = fisis_raw(fk); raw.to_csv(OUT / "fisis_raw.csv", index=False)
        snd = soundness(raw); snd.to_csv(OUT / "fin_soundness.csv", index=False)
        roe = holdco_roe(fk); roe.to_csv(OUT / "fin_holdco_roe.csv", index=False)
        g = gate_table(snd, roe); g.to_csv(OUT / "fin_gate.csv")
        if dk:  # 교차 확인: DART 연결 사업보고서 ROE(기말 자본총계 기준이라 FISIS보다 낮게 나온다)
            fund = fundamentals(dk, names=list(HOLDCO_BANKS)); fund.to_csv(OUT / "fin_fundamentals_holdco.csv", index=False)
            print(fund.pivot_table(index="종목", columns="연도", values="ROE").mul(100).round(1).to_string())
        print(g.round(3).to_string())
