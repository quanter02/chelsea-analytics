"""Collect public text into data/trends/raw/*.jsonl (one post per line: text, created_at, author_hash, source).

    export YOUTUBE_API_KEY=...        # console.cloud.google.com → YouTube Data API v3
    python -m trends.collect youtube --query "연애 고민 상담" --videos 20
    python -m trends.collect youtube --video-id dQw4w9WgXcQ
    python -m trends.collect file --path my_export.csv --text-col 본문 --date-col 작성일

Privacy: names and channel IDs are never stored. Authors become a salted SHA-256 hash, used only to count
each person once. Set TRENDS_SALT to your own secret so the hashes can't be matched across datasets.
Respect each site's terms: use official APIs or datasets you have the right to use (e.g. AI Hub), and don't
scrape logged-in or members-only boards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "trends" / "raw"
YT = "https://www.googleapis.com/youtube/v3"


def author_hash(author_id: str | None) -> str | None:
    if not author_id:
        return None
    salt = os.environ.get("TRENDS_SALT", "change-me")
    return hashlib.sha256((salt + author_id).encode()).hexdigest()[:16]


def _yt(endpoint: str, key: str, **params) -> dict:
    r = requests.get(f"{YT}/{endpoint}", params=dict(params, key=key), timeout=30)
    if r.status_code == 403 and "commentsDisabled" in r.text:
        return {"items": []}
    r.raise_for_status()
    return r.json()


def youtube_search(query: str, key: str, n: int = 20) -> list[str]:
    """Video IDs for a query, Korean region/language. Costs 100 quota units per call (10,000 per day free)."""
    j = _yt("search", key, part="id", q=query, type="video", maxResults=min(n, 50),
            regionCode="KR", relevanceLanguage="ko", order="relevance")
    return [it["id"]["videoId"] for it in j.get("items", [])]


def youtube_comments(video_id: str, key: str, max_pages: int = 10) -> list[dict]:
    """Top-level comments of one video (100 per page, 1 quota unit per page)."""
    rows, token = [], None
    for _ in range(max_pages):
        j = _yt("commentThreads", key, part="snippet", videoId=video_id, maxResults=100,
                textFormat="plainText", **({"pageToken": token} if token else {}))
        for it in j.get("items", []):
            s = it["snippet"]["topLevelComment"]["snippet"]
            rows.append({"text": s.get("textOriginal", ""), "created_at": s.get("publishedAt"),
                         "author_hash": author_hash((s.get("authorChannelId") or {}).get("value")),
                         "likes": s.get("likeCount", 0), "source": f"youtube:{video_id}"})
        token = j.get("nextPageToken")
        if not token:
            break
    return rows


def load_file(path: str, text_col: str = "text", date_col: str | None = None, author_col: str | None = None,
              source: str | None = None) -> list[dict]:
    """CSV / JSONL / JSON you already have (AI Hub, your own exports). Author column is hashed on the way in."""
    p = Path(path)
    df = pd.read_json(p, lines=p.suffix == ".jsonl") if p.suffix in (".json", ".jsonl") else pd.read_csv(p)
    return [{"text": str(r[text_col]),
             "created_at": str(r[date_col]) if date_col else None,
             "author_hash": author_hash(str(r[author_col])) if author_col else None,
             "source": source or f"file:{p.name}"} for _, r in df.iterrows()]


def save(rows: list[dict], name: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{name}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def load_raw(paths: list[str] | None = None) -> pd.DataFrame:
    files = [Path(p) for p in paths] if paths else sorted(RAW.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"수집된 데이터가 없습니다: {RAW}")
    df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    return df.drop_duplicates(subset=["text", "author_hash"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    y = sub.add_parser("youtube")
    y.add_argument("--query")
    y.add_argument("--video-id", action="append", default=[])
    y.add_argument("--videos", type=int, default=20)
    y.add_argument("--pages", type=int, default=10)
    f = sub.add_parser("file")
    f.add_argument("--path", required=True)
    f.add_argument("--text-col", default="text")
    f.add_argument("--date-col")
    f.add_argument("--author-col")
    a = ap.parse_args()

    if a.cmd == "youtube":
        key = os.environ.get("YOUTUBE_API_KEY") or ap.error("YOUTUBE_API_KEY 환경 변수가 필요합니다.")
        ids = list(a.video_id) + (youtube_search(a.query, key, a.videos) if a.query else [])
        rows = [r for v in ids for r in youtube_comments(v, key, a.pages)]
        name = "youtube_" + (a.query or "videos").replace(" ", "_")
    else:
        rows = load_file(a.path, a.text_col, a.date_col, a.author_col)
        name = "file_" + Path(a.path).stem
    print(f"{len(rows)}건 → {save(rows, name)}")


if __name__ == "__main__":
    main()
