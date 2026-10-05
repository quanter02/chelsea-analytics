# 데이터 출처와 이용 조건

유료 상품(리포트, 구독, API)에 쓰기 전에 확인하는 표입니다. "상업 이용"은 이 표를 쓴 시점(2026-10-05)의 이해이며 법률 자문이 아닙니다. 유료화 직전에 각 출처의 최신 약관을 다시 확인합니다.

| 출처 | 레포 안 위치 | 쓰는 곳 | 상업 이용 | 꼭 할 일 |
|---|---|---|---|---|
| KOSIS 국가통계포털 (통계청) OpenAPI | `data_official/`, `data_regional/`, `data_monthly/`, `factcheck/reference.json` | 한·일 비교, 혼인 예측, 시군구, 나우캐스트, 팩트체커 | 가능 (공공데이터) | 출처 "통계청 KOSIS"와 표 번호 표기. 가공했음을 밝힘 |
| e-Stat 政府統計の総合窓口 (일본 총무성 통계국) API | `data_official/`, `data_preference/`, `data_jobs/` | 한·일 비교, 결혼 기준, 일자리 | 가능 (출처 표기 조건) | "出典：政府統計の総合窓口(e-Stat)" 표기, 가공했음을 밝힘. API 이용 시 e-Stat 지정 문구 표시 |
| 통계청 2018년 시군구 경계 (공개 사본) | `data_regional/kr_regions_map.json` | 성비 지도 | **확인 필요** | 받은 경로가 공개 패키지 사본이라 원 이용 조건(SGIS) 확인 전까지 유료 상품의 지도는 SGIS 원본으로 다시 만들 것 |
| openfootball football.json | `data_epl/en1_*.json` | EPL 경기 결과 | 가능 (퍼블릭 도메인 CC0) | 없음 (표기 권장) |
| fbref 경기별 xG (worldfootballR_data 공개 사본) | `data_epl/fbref_epl_xg.csv` | EPL 과거 xG (학습·검증) | **불가로 취급** | 원 데이터 권리는 Opta/Stats Perform. 연구·내부 검증용만, 재배포·판매 금지 |
| understat.com 경기별 xG | `data_epl/xg_understat_*.csv` (로컬 전용, git 제외) | EPL 이번 시즌 xG 모드 | **불명확 → 불가로 취급** | 원본 xG는 재배포하지 않고 우리 예측 확률만 공개. 유료화 전 허락 받거나 다른 xG 원천으로 교체 |
| Opta 경기 확률 (사용자가 기사·화면에서 옮겨 적음) | `data_epl/benchmark_opta.csv` | 비교 채점 | **불가** (독점 데이터) | 비교용 짧은 인용과 출처 표기까지만. 대량 수집·재판매 금지 |
| StatsBomb Open Data | `data_cache/matches_*.json` | 초기 신뢰도 프레임워크 검증 (실제 경기 1,517개) | **불가** (비상업 이용) | StatsBomb 출처·로고 표기 조건. 유료 상품에서는 제외 |
| 기사·영상에서 뽑은 주장 | `data_claims/`, `data_collect/` | 팩트체커 시험, 주장 선별 | 인용 범위 | 원문 재게시 금지, 링크와 짧은 인용만 |
| YouTube Data API 검색 결과 | `data_collect/` | 수집 실험 | API 약관 범위 | 결과 재판매 금지, 보관 기간 제한 확인 |

## 유료 상품별 정리

| 상품 | 쓸 수 있는 데이터 | 빼야 하는 데이터 |
|---|---|---|
| 지역별 결혼 시장 리포트 | KOSIS | 지도는 SGIS 원본 경계로 다시 그릴 때까지 보류 |
| 한·일 결혼 리포트 | KOSIS, e-Stat | 없음 |
| EPL 예측 구독 | openfootball(결과), **우리 예측 확률** | understat·fbref 원본 xG, Opta 수치 대량 재게시 |
| 숫자 검증 서비스 | KOSIS, e-Stat | 없음 |
