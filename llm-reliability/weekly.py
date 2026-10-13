"""일요일 20시 주간 루틴: 노션 '주간 기록' 입력 → 기록·장부·재계산 → 다음 주 배분.

입력(data_biz/week_input.json, 루틴이 노션에서 읽어 만든다)
  {"week_of": "2026-10-19",            # 계획할 주의 월요일
   "hours_total": 60 | null,           # 비우면 지난주 값 이어받기
   "exam_week": false,                 # 체크하면 학업 바닥 40시간
   "actual": {"활동 이름": 시간, ...},   # 지난주 실제로 쓴 시간 (대충 30분 단위)
   "notes": "E3 업체A L1 통화 15분"}     # 실험 증거: 'E번호 익명상대 L단계 [금액] 메모' 한 줄에 하나

하는 일
  1) 지난주 계획 대비 실제를 data_biz/weeks.csv에 기록 (계획도 채점)
  2) notes의 실험 증거를 data_biz/events.csv에 추가 (익명 상대만; 실명·연락처는 넣지 않음)
  3) 마감 지난 실험은 결과를 장부에 기록 (실현 정보량 포함)
  4) biz_update(세계 재계산) → allocator(다음 주 배분) 실행, 노션에 쓸 요약 JSON 출력

    python weekly.py                    # → output/strategy/week_summary.json
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd

from llmrel import allocator, biz, ledger

ROOT = Path(__file__).parent
NOTION_DS = "collection://43b2de76-19ac-447d-a823-bd73e6fa6f72"   # 노션 '주간 기록' (행 = 주, 루틴이 다음 주 행을 미리 만듦)
ACT_MAP = {  # 노션 '실제: …' 칸 → 배분기 활동 이름
    "실제: 인턴 준비": "동계 인턴 준비 (SQL·포트폴리오·지원서)", "실제: E3 혼인 수요": "E3 혼인 수요 (웨딩·혼수 3곳)", "실제: E4 배지": "E4 검증 배지 (예측자 10명)", "실제: E5 동네 카드": "E5 동네 카드 (3장 게시·요청 응답)", "실제: E6 말값": "E6 말값 (2030 다짐 적중률)",
    "실제: E2 노트": "E2 노트: 콘텐츠 발행·구독자 모집", "실제: E1 지자체": "E1 지자체 샘플 (첫 고객 만들기)",
    "실제: 사업계획서": "예비창업패키지 사업계획서", "실제: EPL 콘텐츠": "EPL 라운드 채점·콘텐츠", "실제: 학업 추가": "학업 추가 (학점 올리기)"}
INPUT = ROOT / "data_biz" / "week_input.json"
WEEKS = ROOT / "data_biz" / "weeks.csv"
SUMMARY = ROOT / "output" / "strategy" / "week_summary.json"
EXAM_FLOOR = 40.0
NOTE_RE = re.compile(r"(E\d+)\s+(\S+)\s+(L[0-5])(?:\s+(\d[\d,]*)원?)?\s*(.*)")


def parse_notes(text: str, date: str) -> list[list]:
    rows = []
    for line in (text or "").splitlines():
        m = NOTE_RE.match(line.strip())
        if m:
            exp, who, lvl, amt, memo = m.groups()
            rows.append([date, exp, who, lvl, int(amt.replace(",", "")) if amt else 0, memo.strip()])
    return rows


def last_plan() -> dict:
    p = ROOT / "output" / "strategy" / "week_plan.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def run(inp: dict) -> dict:
    cfg = allocator.load()
    prev = last_plan()
    # 1) 지난주 계획 대비 실제
    if inp.get("actual") and prev:
        new = not WEEKS.exists()
        with WEEKS.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["week_of", "activity", "planned_h", "actual_h"])
            for act, planned in prev.get("hours", {}).items():
                w.writerow([prev["week_of"], act, round(planned, 1), float(inp["actual"].get(act, 0))])
    # 2) 실험 증거
    added = parse_notes(inp.get("notes", ""), inp["week_of"])
    if added:
        with biz.EVENTS.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(added)
    # 3) 마감 지난 실험 결과 기록
    done = {r["key"] for r in ledger.read() if r["kind"] == "biz_result"}
    today = pd.Timestamp(inp["week_of"])
    closed = [k for k, x in biz.EXPERIMENTS.items() if k not in done and today > pd.Timestamp(x["deadline"])]
    for k in closed:
        biz.record_result(k)
    # 4) 재계산 + 다음 주 배분
    cfg["week_of"] = inp["week_of"]
    if inp.get("hours_total"):
        cfg["hours_total"] = float(inp["hours_total"])
    if inp.get("exam_week"):                                  # 시험 주: 이번 주만 학업 바닥을 올림 (설정 파일은 그대로)
        cfg["floor"] = [dict(f, hours=EXAM_FLOOR) if f["name"].startswith("학업") else f for f in cfg["floor"]]
    saved = allocator.load()
    saved.update(week_of=inp["week_of"], hours_total=cfg["hours_total"])  # 가용 시간은 다음 주 기본값으로 이어받음
    allocator.CONFIG.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    import biz_update
    biz_update.main()
    res = allocator.main(inp["week_of"], cfg)
    open_ = res[res["시간"] < res["max"] - 1e-6]
    nxt = (open_ if len(open_) else res).sort_values("한계효용", ascending=False).iloc[0]["활동"]
    hours = {r["활동"]: round(float(r["시간"]), 1) for _, r in res.iterrows()}
    soon = [f'{x["name"]} 마감 {x["deadline"]}' for x in biz.EXPERIMENTS.values()
            if 0 <= (pd.Timestamp(x["deadline"]) - today).days <= 14]
    plan = {"week_of": inp["week_of"], "hours": hours}
    (ROOT / "output" / "strategy" / "week_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"week_of": inp["week_of"], "budget": res.attrs["budget"], "hours": hours, "next_hour": nxt,
               "events_added": len(added), "results_recorded": closed, "deadline_alerts": soon,
               "plan_text": " · ".join(f"{k} {v}h" for k, v in sorted(hours.items(), key=lambda kv: -kv[1]) if v > 0)}
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(json.loads(INPUT.read_text(encoding="utf-8"))), ensure_ascii=False, indent=2))
