"""Labelled posts → topic shares by age band, with confidence intervals.

The unit is a writer, not a post: if one person posts ten times about money, that is one person who talks about
money. Posts without an author hash count as their own writer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import extract as X

MIN_N = 100   # bands with fewer writers are reported but flagged


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def writers(df: pd.DataFrame, gender: str | None = "F") -> pd.DataFrame:
    """One row per writer with a known age band (and gender, unless gender=None): their band and the union
    of topics over all their posts. A writer whose posts disagree on band keeps the most common one."""
    d = df[df["age_band"].notna()].copy()
    if gender:
        d = d[d["gender"] == gender]
    d["writer"] = d["author_hash"].where(d["author_hash"].notna(), "post_" + d.index.astype(str)) \
        if "author_hash" in d else "post_" + d.index.astype(str)
    band = d.groupby("writer")["age_band"].agg(lambda s: s.mode().iloc[0])
    tops = d.groupby("writer")["topics"].agg(lambda s: sorted({t for ts in s for t in ts}))
    neg = d.groupby("writer")["sentiment"].agg(lambda s: (s == "부정").mean())
    return pd.DataFrame({"age_band": band, "topics": tops, "neg_share": neg})


def share_table(w: pd.DataFrame) -> pd.DataFrame:
    """Rows: topic × age band. Share of writers in the band who mention the topic, Wilson 95% CI, and the
    difference from the other bands combined."""
    bands = [b for _, _, b in X.BANDS if b in set(w.age_band)]
    rows = []
    for topic in X.TOPICS:
        has = w["topics"].apply(lambda ts: topic in ts)
        for b in bands:
            inb = w.age_band == b
            n, k = int(inb.sum()), int((has & inb).sum())
            n_o, k_o = int((~inb).sum()), int((has & ~inb).sum())
            lo, hi = wilson(k, n)
            p, p_o = k / n if n else np.nan, k_o / n_o if n_o else np.nan
            se = np.sqrt(p * (1 - p) / n + p_o * (1 - p_o) / n_o) if n and n_o else np.nan
            rows.append({"topic": topic, "age_band": b, "n": n, "k": k, "share": p, "lo": lo, "hi": hi,
                         "vs_rest": p - p_o, "z": (p - p_o) / se if se else np.nan, "small": n < MIN_N})
    return pd.DataFrame(rows)


def standouts(tbl: pd.DataFrame, z: float = 2.5, top: int = 8) -> pd.DataFrame:
    """Band × topic cells that differ most from the other bands. z=2.5 rather than 2 because the table tests
    ~40 cells at once, so a couple of |z|>2 cells are expected by chance."""
    t = tbl[(~tbl.small) & (tbl.z.abs() >= z)]
    return t.reindex(t.vs_rest.abs().sort_values(ascending=False).index).head(top)


def coverage(df: pd.DataFrame) -> dict:
    """How much of the raw data survives each filter (the report shows this so readers see the funnel)."""
    return {
        "posts": len(df),
        "with_age": int(df.age_band.notna().sum()),
        "with_gender": int(df.gender.notna().sum()),
        "female_with_age": int((df.age_band.notna() & (df.gender == "F")).sum()),
        "with_topic": int(df.topics.apply(len).gt(0).sum()),
    }
