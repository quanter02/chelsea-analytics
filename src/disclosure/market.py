"""Market cap and liquidity for live alerts (FinanceDataReader, pykrx as fallback). Both are optional:
without them the alert still runs, it just skips the size and liquidity lines."""
from __future__ import annotations

import numpy as np
import pandas as pd


def listing() -> pd.DataFrame:
    """Current KRX listing indexed by 6-digit code with a Marcap column (KRW). Empty if unavailable."""
    try:
        import FinanceDataReader as fdr
        d = fdr.StockListing("KRX")
        return d.set_index(d["Code"].astype(str).str.zfill(6))[["Marcap"]]
    except Exception:
        return pd.DataFrame(columns=["Marcap"])


def avg_traded_value(code: str, before, days: int = 20) -> float:
    """Mean of close × volume over the `days` trading days before `before` (past data only)."""
    end = pd.Timestamp(before) - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days * 2 + 10)
    p = None
    try:
        import FinanceDataReader as fdr
        p = fdr.DataReader(code, start, end)
    except Exception:
        try:
            from pykrx import stock
            d = stock.get_market_ohlcv(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), code)
            p = d.rename(columns={"종가": "Close", "거래량": "Volume"})
        except Exception:
            return np.nan
    if p is None or len(p) < days // 2:
        return np.nan
    w = p.tail(days)
    return float((w["Close"] * w["Volume"]).mean())
