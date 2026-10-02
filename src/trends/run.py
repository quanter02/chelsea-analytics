"""Collected posts → labels → report.

    python -m trends.run                                  # rule-based, women 20–39, reports/trends_report.html
    python -m trends.run --extractor llm --limit 500      # Claude labels (costs money, cached)
    python -m trends.run --gender all --title "2030 연애 트렌드"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import collect, extract as X, report

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", help="jsonl files (default: data/trends/raw/*.jsonl)")
    ap.add_argument("--extractor", choices=["rule", "llm"], default="rule")
    ap.add_argument("--gender", choices=["F", "M", "all"], default="F")
    ap.add_argument("--partner-cue", action="store_true", help="rule extractor: also read 남친→F, 여친→M (a guess)")
    ap.add_argument("--limit", type=int, help="only the first N posts (try the LLM on a sample first)")
    ap.add_argument("--title", default="2030 여성 연애·소비 트렌드")
    ap.add_argument("--baseline", type=Path, default=ROOT / "data" / "trends" / "baseline.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "trends_report.html")
    a = ap.parse_args()

    df = collect.load_raw(a.inputs or None)
    if a.limit:
        df = df.head(a.limit)
    if a.extractor == "llm":
        from . import llm
        lab = llm.extract(df)
    else:
        lab = X.extract(df, partner_cue=a.partner_cue)
    labels = ROOT / "data" / "trends" / "labels.jsonl"   # without the text, for your own analysis
    labels.parent.mkdir(parents=True, exist_ok=True)
    lab.drop(columns=["text"]).to_json(labels, orient="records", lines=True, force_ascii=False, date_format="iso")
    sources = ", ".join(sorted({s.split(":")[0] for s in lab.get("source", []) if isinstance(s, str)}))
    out = report.build(lab, a.out, a.title, gender=None if a.gender == "all" else a.gender, sources=sources,
                       baseline=a.baseline, extractor=a.extractor)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
