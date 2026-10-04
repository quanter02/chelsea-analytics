"""engine.js + reference.json + 예시 원고 → 한 파일짜리 웹 페이지 (factcheck/index.html)."""
import datetime as dt
import json
from pathlib import Path

D = Path(__file__).parent
samples = [
    dict(name="예시 대본 (오류 섞음)", text=(D / "sample_script.txt").read_text(encoding="utf-8")),
    dict(name="처음 보는 문장", text=(D / "sample_holdout.txt").read_text(encoding="utf-8")),
    dict(name="2026 실제 기사 요약", text=(D / "sample_real_2026.txt").read_text(encoding="utf-8")),
]
html = (D / "page_template.html").read_text(encoding="utf-8")
html = html.replace("/*ENGINE*/", (D / "engine.js").read_text(encoding="utf-8"))
html = html.replace("/*REF*/null", json.dumps(json.loads((D / "reference.json").read_text(encoding="utf-8")), ensure_ascii=False, separators=(",", ":")))
html = html.replace("/*SAMPLES*/null", json.dumps(samples, ensure_ascii=False))
html = html.replace("/*ASOF*/", dt.date.today().isoformat())
(D / "index.html").write_text(html, encoding="utf-8")
print(len(html))
