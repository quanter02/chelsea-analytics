# 통계 숫자 검증기 (시제품)

대본·기사 속 한·일 인구 통계 수치를 공식 값과 대조합니다.

```bash
python -c "from llmrel import factcheck_ref as F; F.save('factcheck/reference.json')"   # 정답 저장소 갱신 (KOSIS·e-Stat 키 필요)
node factcheck/run.js factcheck/sample_script.txt                                      # 판정 (터미널)
python factcheck/build_page.py                                                          # 웹 페이지 factcheck/index.html 생성
```

- `engine.js`: 추출·판정 엔진 (브라우저·node 공용)
- `reference.json`: 정답 저장소 400개 값 (한국 2000~2025, 일본 2000~2024 확정치 + 2025 잠정치·개수)
- `expected.json`: 테스트 기대 판정 (`tests/test_reliability.py::test_factcheck_engine_verdicts`)

판정: 일치 / 근사 일치(허용 오차 이내, 기본 1%·증감률 0.5%p) / 기준 다름 / 연도 혼동 / 나라 혼동 / 틀림 / 확인 불가(공표 전) / 대상 아님(월·분기 값, 나라 불명 → 판정 보류)

2026-10-04 결과
- 오류를 섞은 예시 대본 12개 수치: 전부 기대대로 판정 (연도 혼동 3, 잠정·확정 차이 1, 틀림 1, 공표 전 1)
- 실제 2026 기사 요약 20개 수치: 연간 값 3개 일치, 월·분기 값 17개는 판정 보류, 오판 0
- 처음 보는 문장 13개: 1차에서 3개 오판 → 규칙 보완 뒤 전부 기대대로. 보완 후 다시 처음 보는 문장으로 재야 하는 낙관적 수치
