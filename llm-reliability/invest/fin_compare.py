"""금융주 장기 보유 비교 (가격 + DART 재무지표).

    python invest/fin_compare.py            # 가격 지표 (야후, query1.finance.yahoo.com 허용 필요)
    DART_API_KEY=... python invest/fin_compare.py --dart   # + 재무지표 (opendart.fss.or.kr 허용 필요)
    DART_API_KEY=... python invest/fin_compare.py --dart --no-price   # 재무지표만

가격 지표: 10년 연수익(배당 반영), 변동성, 최대 낙폭, 하락/상승 포착(주봉, 2018-01-29~2020-06-30 제외),
금리 인상 하락장(2021-06-25~2022-09-30) 성과, 코스피 3년 초과 비율, 알파벳(원화)과의 상관.
재무지표(DART, 연결 사업보고서): ROE, 순이익 성장, 자본, 배당 — 은행용 관문(자본·ROE·연체)에 쓴다.
개인 계좌 정보는 이 파일에 넣지 않는다.
"""
from __future__ import annotations

import io, json, os, sys, time, zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "out"
UA = {"User-Agent": "Mozilla/5.0"}
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
    cache = Path(__file__).parent / "data_cache" / "corpCode.zip"  # 3MB대, 느리므로 한 번만 받는다
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(get(f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={key}"))
    z = zipfile.ZipFile(io.BytesIO(cache.read_bytes()))
    import xml.etree.ElementTree as ET
    root = ET.fromstring(z.read(z.namelist()[0]))
    return {e.findtext("stock_code").strip(): e.findtext("corp_code") for e in root.iter("list") if (e.findtext("stock_code") or "").strip()}


DART_IDS = {"ifrs-full_ProfitLoss": "ni", "ifrs-full_ProfitLossAttributableToOwnersOfParent": "ni_ctrl",
            "ifrs-full_Equity": "equity", "ifrs-full_EquityAttributableToOwnersOfParent": "equity_ctrl"}
COLS = {"thstrm_amount": 0, "frmtrm_amount": 1, "bfefrmtrm_amount": 2}  # 당기·전기·전전기 → 연도 차이
STATUS_LOG: list[dict] = []


def dart_json(url: str, wait: float) -> dict:
    """status 020(요청 제한)이면 대기 시간을 늘려 다시 요청한다."""
    for i in range(5):
        d = json.loads(get(url))
        if d.get("status") != "020":
            return d
        time.sleep(wait * 2 ** (i + 2))
    return d


def dart_fin(key: str, corp: str, year: int, wait: float = 1.0) -> dict:
    """사업보고서(11011) 전체 재무제표에서 연도별 순이익·자본(전체, 지배주주)을 꺼낸다.

    한 보고서에는 당기·전기·전전기 값이 함께 있으므로 {연도: {계정: 값}}으로 돌려준다.
    연결(CFS)이 없으면(013) 별도(OFS)를 쓴다. 응답 status/message는 STATUS_LOG에 남긴다.
    """
    for fs in ("CFS", "OFS"):
        url = (f"https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json?crtfc_key={key}&corp_code={corp}"
               f"&bsns_year={year}&reprt_code=11011&fs_div={fs}")
        d = dart_json(url, wait)
        time.sleep(wait)
        STATUS_LOG.append({"corp_code": corp, "연도": year, "fs_div": fs, "status": d.get("status"),
                           "message": d.get("message"), "rows": len(d.get("list", []))})
        if d.get("status") == "000":
            break
    out: dict[int, dict] = {}
    for r in d.get("list", []):
        f = DART_IDS.get(r.get("account_id"))
        if not f or r.get("sj_div") not in ("BS", "IS", "CIS"):
            continue
        for col, lag in COLS.items():
            v = (r.get(col) or "").replace(",", "")
            if v.lstrip("-").isdigit():
                out.setdefault(year - lag, {}).setdefault(f, int(v))
    if fs == "OFS":  # 종속회사가 없어 별도만 있으면 전체가 지배주주 몫
        for f in out.values():
            f.setdefault("ni_ctrl", f.get("ni")); f.setdefault("equity_ctrl", f.get("equity"))
    if out:
        out.setdefault(year, {})["fs_div"] = fs
    return out


def fundamentals(key: str, years=range(2019, 2026), wait: float = 1.0, names=None) -> pd.DataFrame:
    """종목·연도별 순이익/자본. 그 연도 보고서(당기) 값을 우선하고, 없으면 가장 최근 보고서(재작성 반영)의 전기·전전기 값."""
    codes = dart_corp_codes(key)
    rows = []
    for sym, name in STOCKS.items():
        if not sym.endswith(".KS") or (names and name not in names):
            continue
        corp = codes.get(sym[:6])
        if not corp:
            continue
        got: dict[int, dict] = {}
        for y in sorted(years, reverse=True):
            for yy, f in dart_fin(key, corp, y, wait).items():
                src = "api" if yy == y else f"api:{y}보고서"
                cur = got.setdefault(yy, {})
                for kk, v in f.items():
                    if kk not in cur or yy == y:
                        cur[kk] = v
                        if kk != "fs_div":
                            cur[f"src_{kk}"] = src
        for yy, f in got.items():
            rows.append({"종목": name, "코드": sym[:6], "연도": yy, **f})
        for s in STATUS_LOG:
            s.setdefault("종목", name)
    return pd.DataFrame(rows).sort_values(["종목", "연도"]).reset_index(drop=True)


def add_roe(d: pd.DataFrame) -> pd.DataFrame:
    """ROE_기말 = 순이익 ÷ 기말 자본총계(이전 방식), ROE_지배평균 = 지배주주 순이익 ÷ 평균 지배주주 자본."""
    d = d.sort_values(["종목", "연도"]).copy()
    d["ROE_기말"] = d.ni / d.equity
    prev = d.groupby("종목").equity_ctrl.shift(1).where(d.groupby("종목").연도.diff() == 1)
    d["ROE_지배평균"] = d.ni_ctrl / ((d.equity_ctrl + prev) / 2)
    return d


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    if "--no-price" not in sys.argv:
        p = price_metrics(); p.to_csv(OUT / "fin_price_metrics.csv")
        print(p.round(3).to_string())
    if "--dart" in sys.argv:
        key = os.environ.get("DART_API_KEY", "").strip()
        if not key:
            sys.exit("DART_API_KEY 환경 변수가 없습니다 (새 세션에서 적용됨)")
        f = add_roe(fundamentals(key)); f.to_csv(OUT / "fin_fundamentals.csv", index=False)
        pd.DataFrame(STATUS_LOG).to_csv(OUT / "dart_status.csv", index=False)
        print(pd.DataFrame(STATUS_LOG).pivot_table(index="종목", columns="연도", values="status", aggfunc="first").to_string())
        print(f.pivot_table(index="종목", columns="연도", values="ROE_지배평균").round(3).to_string())
