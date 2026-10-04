"""수집 점검: 출처가 실제로 열리는지, 수집 결과가 깨끗한지, 수치를 맞게 뽑는지, 기존 주장과 어떻게 맞물리는지.

    python collect_demo.py      # → output/collection/report.md
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from llmrel import collect as C
from llmrel.report import md_table

ROOT = Path(__file__).parent
RUN = ROOT / "data_collect" / "search_run_2026-10-04.json"
GOLD = ROOT / "data_collect" / "search_run_2026-10-04_gold.csv"
CLAIMS = ROOT / "data_claims" / "kr_jp_marriage_2026.csv"
COUNTRY = {0: "KR", 1: "KR", 2: "JP", 3: "KR", 4: "JP", 5: "JP"}
LABEL = {"births": "출생아 수", "births_yoy": "출생아 증감률", "births_cum": "출생아 누적", "births_cum_yoy": "출생아 누적 증감률",
         "marriages": "혼인건수", "marriages_yoy": "혼인 증감률", "tfr": "합계출산율", "fert_30_34": "30~34세 여성 출산율",
         "mothers35_share": "35세 이상 산모 비중", "deaths": "사망자 수", "deaths_yoy": "사망 증감률", "divorces": "이혼건수",
         "natural_decrease": "자연감소"}


def main() -> None:
    out = ROOT / "output" / "collection"; out.mkdir(parents=True, exist_ok=True)

    # 1) 출처 점검
    health = pd.DataFrame([h.__dict__ for h in (C.check_kosis(), C.check_estat(), C.check_youtube())])
    health.loc[len(health)] = dict(source="웹 검색 (세션 도구)", ok=True, status="정상", detail="6개 검색어, 결과를 JSON으로 저장")

    # 2) 수집 항목 정리
    run, items = C.load_search_run(RUN)
    items["canonical"] = items.url.map(C.canonical_url)
    items["type"] = items.url.map(C.classify)
    items["date"] = items.url.map(C.url_date)
    uniq = items.drop_duplicates("canonical")
    dup_groups = items[items.duplicated("canonical", keep=False)].groupby("canonical").url.apply(list)
    fresh = uniq.date.dropna().str[:4].astype(int).ge(2026)
    per_query = items.groupby("query_id").agg(링크=("url", "size"), 날짜확인=("date", lambda d: d.notna().sum()),
                                              최신2026=("date", lambda d: d.dropna().str[:4].astype(int).ge(2026).sum())).reset_index()
    per_query["검색어"] = [q["query"][:38] for q in run["queries"]]

    # 3) 수치 추출 (규칙 v0 → v1) vs 사람이 뽑은 정답
    gold = C.dedupe_claims(pd.read_csv(GOLD))
    scores, preds = {}, {}
    for v in (0, 1):
        p = pd.DataFrame([dict(query_id=qi, country=COUNTRY[qi], **c) for qi, q in enumerate(run["queries"])
                          for c in C.extract_claims(q["summary"], q["lang"], v)])
        preds[v] = C.dedupe_claims(p); scores[v] = C.score_extraction(preds[v], gold)

    # 4) 기존 주장과 대조 (같은 국가·지표·기간)
    old = pd.read_csv(CLAIMS)
    rev = {v: k for k, v in LABEL.items()}
    old["ind"] = old.indicator.map(lambda s: rev.get(s, s)); old["per"] = old.period.astype(str)
    rows = []
    for g in gold.drop_duplicates(["country", "indicator", "period"]).itertuples():
        m = old[(old.country == g.country) & (old.ind == g.indicator) & (old.per == str(g.period))]
        if m.empty:
            status, before = "새 정보", ""
        else:
            o = m.iloc[0]; same = abs(float(o.value) - g.value) < 1e-6
            status = "일치 (재현)" if same else f"값 다름 → {'확정치로 갱신' if o.decision == '보류' else '확인 필요'}"
            before = f"{o.value} ({o.decision})"
        rows.append(dict(국가=g.country, 지표=LABEL.get(g.indicator, g.indicator), 기간=g.period, 이번값=g.value, 기존값=before, 판정=status))
    cmp_ = pd.DataFrame(rows)

    sc = lambda s: f"정밀도 {s['precision']:.0%} · 재현율 {s['recall']:.0%} · 숫자만 기준 재현율 {s['value_recall']:.0%} ({s['correct']}/{s['gold']})"
    md = [
        "# 수집 점검 (2026-10-04)", "",
        "## 1. 출처별 작동 여부", "",
        md_table(health.rename(columns={"source": "출처", "ok": "작동", "status": "상태", "detail": "내용"})), "",
        "## 2. 수집 항목", "",
        f"- 링크 {len(items)}개 → 주소 정규화 후 고유 {len(uniq)}개 (중복 {len(items) - len(uniq)}개, {1 - len(uniq) / len(items):.0%})",
        f"- 주소에서 날짜를 확인한 항목 {uniq.date.notna().sum()}개 중 2026년 자료 {int(fresh.sum())}개",
        "- 출처 종류: " + ", ".join(f"{k} {v}" for k, v in uniq.type.value_counts().items()), "",
        md_table(per_query[["검색어", "링크", "날짜확인", "최신2026"]]), "",
        "중복으로 합쳐진 예: " + "; ".join(" = ".join(u.split("//")[1][:45] for u in g) for g in list(dup_groups)[:3]), "",
        "## 3. 수치 추출 정확도 (정답: 사람이 요약문에서 직접 뽑은 수치)", "",
        f"- 규칙 v0 (처음 규칙): {sc(scores[0])}",
        f"- 규칙 v1 (이번 데이터의 오류를 보고 보완): {sc(scores[1])}",
        "- v1은 같은 데이터로 고친 결과라 낙관적입니다. 다음 수집분에서 다시 재야 실제 성능입니다.", "",
        "v1이 놓친 것: " + ", ".join(f"{c} {LABEL.get(i, i)} {p} {v:g}" for c, i, p, v in scores[1]["missed"]), "",
        "v1이 잘못 뽑은 것: " + ", ".join(f"{c} {LABEL.get(i, i)} {p} {v:g}" for c, i, p, v in scores[1]["wrong"]), "",
        "## 4. 기존 주장과 대조", "",
        md_table(cmp_), "",
    ]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(health.to_string()); print(len(items), len(uniq), uniq.type.value_counts().to_dict())
    for v in (0, 1): print("v", v, {k: round(x, 3) for k, x in scores[v].items() if isinstance(x, float)})
    print(cmp_.to_string())


if __name__ == "__main__":
    main()
