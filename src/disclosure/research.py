"""Test a new disclosure feature before it is allowed to move a grade.

Use it in the notebook after step 3, on `ev_res` joined with features from `dart.detail_features`:

    from disclosure import research as R
    ev = ev_res.join(feats)                        # feats: one row per event, e.g. amount, cancel, days
    ev["size_pct"] = ev.amount / ev.marcap * 100
    R.bucket_test(ev[ev.type == "신탁"], "size_pct", ret="trade_ew")
    R.verdict(ev[ev.type == "신탁"], "size_pct", ret="trade_ew")

A feature passes only if the top bucket beats the bottom one in both halves of the sample, on the median as
well as the mean. Otherwise it stays information-only in signal.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def tstat(x) -> float:
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def welch_t(a, b) -> float:
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def buckets(x: pd.Series, q: int = 5) -> pd.Series:
    """Quantile buckets for numeric features, the values themselves for booleans / categories."""
    if x.dtype == bool or x.dropna().nunique() <= 2 or not pd.api.types.is_numeric_dtype(x):
        return x.astype("object")
    return pd.qcut(x, q, labels=[f"Q{i}" for i in range(1, q + 1)], duplicates="drop")


def bucket_test(ev: pd.DataFrame, feature: str, ret: str = "trade_ret", q: int = 5) -> pd.DataFrame:
    d = ev[[feature, ret]].dropna()
    g = d.groupby(buckets(d[feature], q), observed=True)[ret]
    return pd.DataFrame({
        "N": g.size(), "평균%": g.mean() * 100, "중앙값%": g.median() * 100,
        "t": g.apply(tstat), "승률%": g.apply(lambda x: (x > 0).mean() * 100),
    }).round(2)


def verdict(ev: pd.DataFrame, feature: str, ret: str = "trade_ret", q: int = 5, date: str = "event_date") -> dict:
    """Top vs bottom bucket, overall and in each half of the period."""
    d = ev[[feature, ret, date]].dropna().sort_values(date)
    d["b"] = buckets(d[feature], q)
    cats = [c for c in pd.Series(d["b"]).dropna().unique()]
    cats = sorted(cats, key=lambda c: (str(type(c)), c))
    lo, hi = cats[0], cats[-1]
    half = d[date].iloc[len(d) // 2]
    out = {"bottom": lo, "top": hi}
    for name, part in {"전체": d, "앞 절반": d[d[date] < half], "뒤 절반": d[d[date] >= half]}.items():
        a, b = part[part.b == hi][ret], part[part.b == lo][ret]
        out[name] = {"차이 평균%p": round((a.mean() - b.mean()) * 100, 2),
                     "차이 중앙값%p": round((a.median() - b.median()) * 100, 2),
                     "t": round(welch_t(a, b), 2)}
    out["통과"] = all(out[k]["차이 평균%p"] > 0 and out[k]["차이 중앙값%p"] > 0 for k in ("앞 절반", "뒤 절반")) \
        and out["전체"]["t"] > 2
    return out
