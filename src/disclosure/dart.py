"""DART Open API client for buyback disclosures (자기주식 취득 / 신탁계약 체결 결정).

API key: opendart.fss.or.kr → 인증키 신청 (free). Set it as the DART_API_KEY environment variable.
The list/clean logic is the same as disclosure_event_study_buyback_v2.ipynb, so live alerts and the backtest
see events the same way.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import requests

LIST_URL = "https://opendart.fss.or.kr/api/list.json"
DETAIL_URL = {
    "직접": "https://opendart.fss.or.kr/api/tsstkAqDecsn.json",          # 자기주식 취득 결정
    "신탁": "https://opendart.fss.or.kr/api/tsstkAqTrctrCnsDecsn.json",  # 자기주식취득 신탁계약 체결 결정
}
LIST_COLUMNS = ["corp_code", "corp_name", "stock_code", "corp_cls", "report_nm", "rcept_no", "flr_nm", "rcept_dt", "rm"]


def api_key() -> str:
    key = os.environ.get("DART_API_KEY", "").strip()
    if not key:
        raise SystemExit("DART_API_KEY 환경 변수가 없습니다. opendart.fss.or.kr에서 인증키를 받아 설정하세요.")
    return key


def get(url: str, params: dict, key: str, retries: int = 3) -> dict:
    p = dict(params, crtfc_key=key)
    last = None
    for i in range(retries):
        try:
            return requests.get(url, params=p, timeout=30).json()
        except Exception as ex:  # network / JSON errors: retry with backoff
            last = ex
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"DART 요청 실패: {last}")


def date_chunks(start, end, days: int = 85) -> list[tuple[str, str]]:
    """Without corp_code, DART limits a search to 3 months, so split into 85-day chunks."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    while s <= e:
        ce = min(s + pd.Timedelta(days=days), e)
        out.append((s.strftime("%Y%m%d"), ce.strftime("%Y%m%d")))
        s = ce + pd.Timedelta(days=1)
    return out


def fetch_list(key: str, start, end, keyword: str = "자기주식취득", sleep: float = 0.15) -> pd.DataFrame:
    """Major-event reports (pblntf_ty=B) whose title contains keyword."""
    rows = []
    for bgn, en in date_chunks(start, end):
        page = 1
        while True:
            j = get(LIST_URL, dict(bgn_de=bgn, end_de=en, pblntf_ty="B", page_no=page, page_count=100), key)
            st = j.get("status")
            if st == "013":  # no results
                break
            if st != "000":
                raise RuntimeError(f"DART 오류 {st}: {j.get('message')}")
            rows += [it for it in j.get("list", []) if keyword in it.get("report_nm", "").replace(" ", "")]
            if page >= int(j.get("total_page", 1)):
                break
            page += 1
            time.sleep(sleep)
        time.sleep(sleep)
    return pd.DataFrame(rows, columns=LIST_COLUMNS) if not rows else pd.DataFrame(rows)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Original decisions only, KOSPI/KOSDAQ only. Same filter as the notebook's clean_events (no dedupe here:
    the live crowding count needs every disclosure, and dedupe is applied by the caller)."""
    if raw.empty:
        return pd.DataFrame(columns=["rcept_no", "corp_code", "corp_name", "stock_code", "market", "type",
                                     "report_nm", "event_date"])
    df = raw.copy()
    nm = df["report_nm"].str.replace(" ", "")
    df = df[nm.str.contains("결정") & ~nm.str.contains("정정|처분|해지|결과|연장")]
    df = df[df["stock_code"].fillna("").str.strip().str.len() == 6]
    df = df[df["corp_cls"].isin(["Y", "K"])].copy()
    df["event_date"] = pd.to_datetime(df["rcept_dt"])
    df["type"] = np.where(df["report_nm"].str.replace(" ", "").str.contains("신탁"), "신탁", "직접")
    df["market"] = df["corp_cls"].map({"Y": "KOSPI", "K": "KOSDAQ"})
    return df[["rcept_no", "corp_code", "corp_name", "stock_code", "market", "type",
               "report_nm", "event_date"]].sort_values("event_date").reset_index(drop=True)


def fetch_detail(key: str, ev: pd.Series) -> dict | None:
    """Raw detail record for one disclosure, or None if DART has none."""
    d = ev["event_date"].strftime("%Y%m%d")
    j = get(DETAIL_URL[ev["type"]], dict(corp_code=ev["corp_code"], bgn_de=d, end_de=d), key)
    if j.get("status") != "000":
        return None
    for it in j.get("list", []):
        if it.get("rcept_no") == ev["rcept_no"]:
            return it
    return None


# ---------------------------------------------------------------- detail → features
# DART field names differ between the two report types (and have changed over time), so each feature
# tries a list of candidate keys. `python -m disclosure.alert --inspect` prints the raw keys of a real record.
AMOUNT_KEYS = ["ctr_prc", "aqpln_prc_ostk", "aqpln_prc"]
START_KEYS = ["ctr_pd_bgd", "aqexpd_bgd"]
END_KEYS = ["ctr_pd_edd", "aqexpd_edd"]
PURPOSE_KEYS = ["ctr_pp", "aq_pp"]


def _num(v) -> float:
    s = str(v or "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _date(v):
    s = str(v or "").strip()
    for fmt in ("%Y년 %m월 %d일", "%Y-%m-%d", "%Y.%m.%d", "%Y%m%d"):
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT


def _first(rec: dict, keys: list[str]):
    for k in keys:
        v = rec.get(k)
        if v not in (None, "", "-"):
            return v
    return None


def detail_features(rec: dict | None) -> dict:
    """Amount (KRW), program length (days), purpose text and whether cancellation (소각) is mentioned."""
    if not rec:
        return {"amount": np.nan, "days": np.nan, "purpose": "", "cancel": None}
    s, e = _date(_first(rec, START_KEYS)), _date(_first(rec, END_KEYS))
    return {
        "amount": _num(_first(rec, AMOUNT_KEYS)),
        "days": (e - s).days if pd.notna(s) and pd.notna(e) else np.nan,
        "purpose": str(_first(rec, PURPOSE_KEYS) or ""),
        "cancel": "소각" in " ".join(str(v) for v in rec.values()),
    }
