"""SQLite 연결과 스키마."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = Path(__file__).resolve().parents[1] / "schema.sql"


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.executescript(SCHEMA.read_text(encoding="utf-8"))
    return con


def insert(con: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    cols = ", ".join(df.columns)
    marks = ", ".join("?" * len(df.columns))
    rows = [tuple(None if pd.isna(v) else v for v in r) for r in df.itertuples(index=False)]
    con.executemany(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({marks})", rows)
    con.commit()


def read(con: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, con)
