# 수익 실험 기록

- `events.csv`: 실험 증거. **공개 레포이므로 상대는 '지자체A', '업체B'처럼 익명으로만** 적습니다. 실명, 연락처, 메일 주소는 적지 않습니다.
  - 열: `date, exp(E1~), counterpart(익명), level(L1~L5), amount_krw, note`
  - 예: `2026-11-20,E1,지자체A,L1,0,통화 15분, 샘플 수령 동의`
- `profile.json`: 판매 가능 시작일, 리포트 노력, 시작 자금, 확장형 시도 기록 (biz_update.py 입력)

단계: L1 회신·통화 → L2 관심 확인서·사전 예약 → L3 첫 결제 → **L4 재결제(검증된 현금흐름)** → L5 소개

    python -m llmrel.biz register     # 실험 사전등록 (장부)
    python -m llmrel.biz status       # 현재 단계
    python -m llmrel.biz result E1    # 결과를 장부에 기록 (마감 후)
    python biz_update.py              # 기록으로 가정 세계를 지우고 전략 다시 계산
