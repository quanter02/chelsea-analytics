"""Claude-based extraction: same output columns as extract.extract, better at context
("서른 앞두고" → 29, "울 오빠가" is a boyfriend, not a brother).

    pip install anthropic          # credentials: ANTHROPIC_API_KEY or `ant auth login`
    python -m trends.run --extractor llm --limit 200     # try a small sample first, check the cost

Posts go 25 per request; results are cached by text hash in data/trends/llm_cache.jsonl, so re-runs are free.
Rough cost on the default model: about $3 per 1,000 short comments. TRENDS_MODEL overrides the model.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from . import extract as X

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "trends" / "llm_cache.jsonl"
MODEL = os.environ.get("TRENDS_MODEL", "claude-opus-5-5")
BATCH = 25

SYSTEM = f"""You label Korean social-media posts for an aggregate trend report. For each post, describe its WRITER only.
- age: the writer's own age in years if the post states or clearly implies it ("28살인데", "서른 앞두고" → 29, "96년생" with the post year). Ages of partners, friends or family don't count. null if unknown.
- gender: "F" or "M" only if the writer states it or it is unambiguous from how they refer to themselves; null otherwise. Don't infer gender from writing style or from the partner's gender.
- topics: every topic the post is substantially about, from this list only: {", ".join(X.TOPICS)}. Empty if none.
- sentiment: the writer's overall tone: 긍정, 중립 or 부정.
Return one result per post, using the post's id."""

SCHEMA = {
    "type": "object",
    "properties": {"results": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "age": {"type": ["integer", "null"]},
            "gender": {"type": ["string", "null"], "enum": ["F", "M", None]},
            "topics": {"type": "array", "items": {"type": "string", "enum": list(X.TOPICS)}},
            "sentiment": {"type": "string", "enum": ["긍정", "중립", "부정"]},
        },
        "required": ["id", "age", "gender", "topics", "sentiment"],
        "additionalProperties": False,
    }}},
    "required": ["results"],
    "additionalProperties": False,
}


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def _load_cache() -> dict:
    if not CACHE.exists():
        return {}
    out = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r.pop("key")] = r
    return out


def label_batch(client, posts: list[tuple[int, str, int | None]]) -> dict[int, dict]:
    """posts: (id, text, year). Returns id → label. A refused or truncated batch returns {}."""
    body = "\n\n".join(f"<post id=\"{i}\" year=\"{y or ''}\">\n{t[:1500]}\n</post>" for i, t, y in posts)
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM,
        messages=[{"role": "user", "content": body}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if resp.stop_reason in ("refusal", "max_tokens"):
        return {}
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return {r["id"]: r for r in json.loads(text)["results"]}


def extract(df: pd.DataFrame, client=None, verbose: bool = True) -> pd.DataFrame:
    """Same columns as extract.extract. Posts the model could not label fall back to the rule-based result."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    rule = X.extract(df)
    cache = _load_cache()
    years = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True) if "created_at" in df else None
    texts = rule["text"].fillna("").astype(str).tolist()
    todo = [(i, t, int(years.iloc[i].year) if years is not None and pd.notna(years.iloc[i]) else None)
            for i, t in enumerate(texts) if _key(t) not in cache]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(todo), BATCH):
        chunk = todo[k:k + BATCH]
        got = label_batch(client, chunk)
        with CACHE.open("a", encoding="utf-8") as f:
            for i, t, _ in chunk:
                if i in got:
                    r = {kk: got[i][kk] for kk in ("age", "gender", "topics", "sentiment")}
                    cache[_key(t)] = r
                    f.write(json.dumps({"key": _key(t), **r}, ensure_ascii=False) + "\n")
        if verbose:
            print(f"  LLM 라벨링 {min(k + BATCH, len(todo))}/{len(todo)}")

    out = rule.copy()
    for i, t in enumerate(texts):
        r = cache.get(_key(t))
        if r is None:
            continue
        out.at[i, "age"] = r["age"]
        out.at[i, "age_how"] = "llm" if r["age"] is not None else ""
        out.at[i, "age_band"] = X.age_band(r["age"])
        out.at[i, "gender"] = r["gender"]
        out.at[i, "gender_how"] = "llm" if r["gender"] else ""
        out.at[i, "topics"] = r["topics"]
        out.at[i, "sentiment"] = r["sentiment"]
    return out
