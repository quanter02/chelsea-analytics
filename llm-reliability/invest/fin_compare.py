"""금융주 장기 보유 비교 (가격 + DART 재무지표).

    python invest/fin_compare.py            # 가격 지표 (야후, query1.finance.yahoo.com 허용 필요)
    DART_API_KEY=... python invest/fin_compare.py --dart   # + 재무지표 (opendart.fss.or.kr 허용 필요)

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


def fundamentals(key: str, years=range(2019, 2026)) -> pd.DataFrame:
    codes = dart_corp_codes(key)
    rows = []
    for sym, name in STOCKS.items():
        if not sym.endswith(".KS"):
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
