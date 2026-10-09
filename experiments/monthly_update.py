"""매달 KOSIS 발표 뒤 실행: 최근 자료 갱신 → 파이프라인 → 리포트·경보 목록 → 지난달과 비교.

    python monthly_update.py                 # 모든 주제
    python monthly_update.py --topics 청년 순이동 미분양
    python monthly_update.py --no-fetch      # 저장된 자료로만 (시험용)

결과: experiments/monthly/<YYYY-MM>/
    summary.md                  이번 달 요약 (주제별 최신 공표 월, 경보 수, 새 경보·해소)
    report_<주제>.md            주제별 리포트 (채점 성적 + 실시간 경보)
    live_<주제>.csv             실시간 경보 목록
    changes_<주제>.csv          지난달 대비 새로 생긴 경보 / 해소된 경보
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import glob
import os

import pandas as pd

import kosis_monitor as K

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "monthly")
MONTHLY_TOPICS = ["혼인", "청년 순이동", "미분양"]   # 월간 갱신 대상 (검증을 마친 주제만. TOPICS에 새 주제를 넣어도 자동으로 포함되지 않음)
RULE = {"혼인": "v5", "청년 순이동": "v5", "미분양": "v5"}   # 2026-10-09 미분양 사전 등록 검증에서 5차 채택 (preregistration_unsold_v2_v5.md)


def refresh(topic: K.Topic, today: dt.date, years_back: int = 2, verbose=False) -> pd.DataFrame:
    """저장된 자료에 최근 years_back년을 KOSIS에서 다시 받아 덮어씀 (수정·추가 공표 반영)."""
    old = K.fetch(topic, verbose=verbose)                       # 저장본 (없으면 저장소 → KOSIS)
    start = f"{today.year - years_back}01"
    end = f"{today.year}{today.month:02d}"
    tp = dataclasses.replace(topic, start=start, end=end, cache=None)
    try:
        new = K.fetch(tp, verbose=verbose)
    except Exception as e:                                      # KOSIS 장애 시 저장본으로 진행
        print(f"  {topic.name}: 갱신 실패 ({str(e)[:60]}) → 저장본 사용")
        return old
    if new.empty:
        return old
    merged = pd.concat([old[old.ym < start], new], ignore_index=True)
    merged = merged.drop_duplicates(["unit", "ym", "series"], keep="last").sort_values(["unit", "series", "ym"])
    if topic.cache:
        merged.to_csv(os.path.join(HERE, topic.cache), index=False)
    return merged


def previous_live(name: str, month_dir: str):
    """이번 달 폴더보다 앞선 가장 최근 달의 경보 목록."""
    dirs = sorted(d for d in glob.glob(os.path.join(OUT_ROOT, "*")) if os.path.isdir(d) and d < month_dir)
    for d in reversed(dirs):
        f = os.path.join(d, f"live_{name}.csv")
        if os.path.exists(f):
            return pd.read_csv(f), os.path.basename(d)
    return None, None


def changes(prev: pd.DataFrame | None, cur: pd.DataFrame) -> pd.DataFrame:
    """지난달 '유지' 경보와 이번 달 '유지' 경보 비교."""
    active = lambda df: set(zip(df[df.상태 == "유지"].지역, df[df.상태 == "유지"].방향)) if df is not None and len(df) else set()
    p, c = active(prev), active(cur)
    rows = [dict(구분="새 경보", 지역=r, 방향=d) for r, d in sorted(c - p)] + [dict(구분="해소", 지역=r, 방향=d) for r, d in sorted(p - c)]
    return pd.DataFrame(rows, columns=["구분", "지역", "방향"])


def main(topics=None, fetch=True, today=None):
    today = today or dt.date.today()
    month = f"{today.year}-{today.month:02d}"
    out_dir = os.path.join(OUT_ROOT, month)
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(HERE)
    lines = [f"# 월간 경보 요약 ({month})", "", f"- 실행일: {today.isoformat()}",
             "- 판정: 통일 점진 규칙 (보정 기간에서 정한 기준 그대로, 매달 다시 고르지 않음)", "",
             "| 주제 | 최신 공표 | 채점 감지율 | 채점 잘못된 경보 | 실시간 경보 (유지) | 새 경보 | 해소 |", "|---|---|---|---|---|---|---|"]
    for name in topics or MONTHLY_TOPICS:
        topic = dataclasses.replace(K.TOPICS[name], live_year=today.year if name != "혼인" else K.TOPICS[name].live_year)
        tidy = refresh(topic, today) if fetch else K.fetch(topic, verbose=False)
        latest = tidy.ym.max()
        if name != "혼인" and latest[:4] != str(today.year):           # 연초에는 아직 올해 자료가 없을 수 있음
            topic = dataclasses.replace(topic, live_year=int(latest[:4]))
        out = K.run(topic, tidy=tidy, rule=RULE.get(name, "v2"), verbose=False)
        L = out["live"]
        L.to_csv(os.path.join(out_dir, f"live_{name}.csv"), index=False)
        prev, prev_month = previous_live(name, out_dir)
        ch = changes(prev, L)
        ch.to_csv(os.path.join(out_dir, f"changes_{name}.csv"), index=False)
        open(os.path.join(out_dir, f"report_{name}.md"), "w", encoding="utf-8").write(
            K.report(out) + "\n\n## 지난달 대비\n\n" + (f"비교 기준: {prev_month}\n\n" + (K._md(ch, index=False) if len(ch) else "(변화 없음)") if prev_month else "(첫 실행)"))
        t = out["test"]
        n_keep = int((L.상태 == "유지").sum()) if len(L) else 0
        lines.append(f"| {name} | {latest[:4]}.{latest[4:]} | {t['감지율']:.0%} | {t['잘못된경보율']:.1%} | {len(L)} ({n_keep}) | "
                     f"{int((ch.구분 == '새 경보').sum()) if prev_month else '-'} | {int((ch.구분 == '해소').sum()) if prev_month else '-'} |")
        print(f"{name}: 최신 {latest}, 경보 {len(L)} (유지 {n_keep}), 지난달 {prev_month or '-'}")
    lines += ["", "주제별 상세: `report_<주제>.md`, 경보 목록: `live_<주제>.csv`, 변화: `changes_<주제>.csv`"]
    open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", nargs="*")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--date", help="YYYY-MM-DD (시험용)")
    a = ap.parse_args()
    d = dt.date.fromisoformat(a.date) if a.date else None
    print("결과:", main(a.topics, fetch=not a.no_fetch, today=d))
