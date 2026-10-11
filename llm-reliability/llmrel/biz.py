"""수익 실험 장부: 모델 규칙을 검증하던 방식 그대로, 수익원도 '결과 전에 적고 결과 뒤에 채점'한다.

- biz_hypothesis: 실험 전에 가설·목표·마감·정리 기준을 예측 장부(같은 해시 체인)에 기록
- biz_result:     실험 뒤에 도달 단계와 결과를 기록. 실패도 지울 수 없다.
- 증거 기록:       data_biz/events.csv (공개 레포이므로 상대는 '지자체A'처럼 익명 표기, 연락처·실명 금지)

검증 단계 (데이터 분할과 같은 사다리)
  L0 가정  L1 회신·통화  L2 관심 확인서·사전 예약  L3 첫 결제  L4 재결제·갱신  L5 소개로 온 고객
  L4부터 '검증된 현금흐름'(메이저 트로피). L3은 컵.

    python -m llmrel.biz register        # EXPERIMENTS를 장부에 사전등록 (내용이 같으면 건너뜀)
    python -m llmrel.biz status          # 실험별 현재 단계
    python -m llmrel.biz result E1       # events.csv 기준으로 E1 결과를 장부에 기록
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from . import ledger

ROOT = Path(__file__).parent.parent
EVENTS = ROOT / "data_biz" / "events.csv"
LEVELS = {"L0": "가정", "L1": "회신·통화", "L2": "관심 확인서·사전 예약", "L3": "첫 결제", "L4": "재결제·갱신", "L5": "소개로 온 고객"}
COLUMNS = ["date", "exp", "counterpart", "level", "amount_krw", "note"]

EXPERIMENTS = {
    "E1": {"name": "지자체 무료 샘플", "stream": "report",
           "hypothesis": "전화로 동의를 받고 무료 샘플 리포트를 보내면, 3곳 중 1곳 이상이 회신하거나 통화에 응한다",
           "action": "강한 지역 1~3순위(수원·용인·성남) 인구정책 부서에 전화 → 동의 시 샘플 메일",
           "target_level": "L1", "target_count": 1, "attempts": 3, "deadline": "2026-12-31",
           "kill": "0곳이면 메시지(제목·첫 숫자)를 바꿔 다음 3곳(4~6순위)으로 1회 재시도, 그래도 0곳이면 지자체 직접 영업 보류"},
    "E2": {"name": "셈법 노트 유료판 사전 예약", "stream": "note",
           "hypothesis": "콘텐츠 발행 후 모은 무료 구독자 중 10% 이상이 유료판(가설 가격 월 4,900원)을 사전 예약한다",
           "action": "콘텐츠 #1·#2 발행 → 무료 구독 모집 → 구독자 30명 이상 모이면 사전 예약 링크 공개",
           "target_level": "L2", "target_count": 3, "attempts": 30, "deadline": "2027-01-31",
           "kill": "구독자 30명 미만이면 판정 보류(표본 부족)로 기록, 30명 이상인데 예약 3명 미만이면 유료판 보류"},
    "E3": {"name": "혼인 수요 예측 (웨딩·혼수 업계)", "stream": "demand",
           "hypothesis": "웨딩홀·혼수 가전·가구 업체 3곳에 지역별 혼인 예측과 80% 범위를 보여주면 1곳 이상이 회신한다",
           "action": "공개된 대표 문의 창구로 동의 요청 → 동의 시 해당 지역 1장 요약 전달",
           "target_level": "L1", "target_count": 1, "attempts": 3, "deadline": "2027-02-28",
           "kill": "0곳이면 데이터바우처 공급기업 경로(2027 공고)로만 남기고 직접 영업 보류"},
    "E4": {"name": "검증 배지 시제품 (예측 장부 공증)", "stream": "ledger",
           "hypothesis": "예측을 공개하는 사람(유튜버·애널리스트·블로거) 10명에게 맨유–토트넘 배지 시제품을 보여주면 2명 이상이 자기 예측에 배지를 붙여 본다",
           "action": "배지 카드(HTML·SVG·한 줄 텍스트) 공개 → 공개 문의 창구로 동의 요청 → 원하는 사람의 예측 1건을 장부에 기록하고 배지 발급",
           "target_level": "L2", "target_count": 2, "attempts": 10, "deadline": "2027-01-31",
           "kill": "10명 중 0~1명이면 배지 단독 확장 보류, 셈법 자체 콘텐츠의 신뢰 표시로만 사용"},
}


def _rank(level: str) -> int:
    return int(level[1]) if isinstance(level, str) and level[:1] == "L" and level[1:].isdigit() else 0


def events(path: Path = EVENTS) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"exp": str, "counterpart": str, "level": str})


def register(exps: dict = EXPERIMENTS, path: Path = ledger.PATH) -> int:
    return ledger.append([dict(kind="biz_hypothesis", key=k, payload=v, model="수익 실험 사전등록",
                               evidence="결과 전 기록. 수정하면 새 줄로 남음") for k, v in exps.items()], path=path)


def status(ev: pd.DataFrame | None = None, exps: dict = EXPERIMENTS) -> pd.DataFrame:
    """실험별: 목표 단계에 도달한 상대 수, 최고 단계, 매출 합계, 판정."""
    ev = events() if ev is None else ev
    rows = []
    for k, x in exps.items():
        e = ev[ev.exp == k]
        best = e.groupby("counterpart")["level"].apply(lambda s: max(map(_rank, s))) if len(e) else pd.Series(dtype=int)
        hit = int((best >= _rank(x["target_level"])).sum())
        rows.append({"exp": k, "name": x["name"], "target": f'{x["target_level"]} {x["target_count"]}곳/명',
                     "reached": hit, "best_level": f"L{int(best.max())}" if len(best) else "L0",
                     "revenue_krw": int(pd.to_numeric(e.amount_krw, errors="coerce").fillna(0).sum()),
                     "deadline": x["deadline"], "verdict": "성공" if hit >= x["target_count"] else "진행 중"})
    return pd.DataFrame(rows)


def record_result(exp: str, verdict: str | None = None, path: Path = ledger.PATH, ev: pd.DataFrame | None = None) -> int:
    """장부에 결과를 남긴다. verdict를 주지 않으면 status 기준 (마감 뒤 '진행 중'은 '실패'로 적는다)."""
    s = status(ev).set_index("exp").loc[exp].to_dict()
    if verdict is None:
        verdict = s["verdict"]
        if verdict == "진행 중" and pd.Timestamp.today().normalize() > pd.Timestamp(EXPERIMENTS[exp]["deadline"]):
            verdict = "실패"
    payload = {"reached": int(s["reached"]), "best_level": s["best_level"], "revenue_krw": int(s["revenue_krw"]), "verdict": verdict}
    try:                                                     # 실현 정보량: 이 결과로 얼마나 배웠고 결정이 바뀌었나
        from . import voi
        if exp in voi.EXPS and int(s["reached"]) <= voi.EXPS[exp][1]:
            rz = voi.realized(exp, int(s["reached"]))
            payload.update({"info_bits": round(rz["bits"], 3), "decision_before": rz["decision_before"],
                            "decision_after": rz["decision_after"], "decision_changed": rz["changed"]})
    except Exception:                                        # 정보량 계산이 실패해도 결과 기록은 남긴다
        pass
    return ledger.append([dict(kind="biz_result", key=exp, payload=payload, model="수익 실험 결과",
                               evidence=f"data_biz/events.csv 기준, 목표 {s['target']}")], path=path)


if __name__ == "__main__":
    cmd = sys.argv[1:] or ["status"]
    if cmd[0] == "register":
        print("장부 추가:", register())
    elif cmd[0] == "result":
        print("장부 추가:", record_result(cmd[1], cmd[2] if len(cmd) > 2 else None))
    else:
        print(status().to_string(index=False))
