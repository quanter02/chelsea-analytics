# 셈법 SQL — 공식 통계를 SQL로 다시 셈하기

셈법의 원자료(KOSIS 인구·혼인 통계, 예측 장부)를 Oracle에 넣고, 파이썬 분석과 **같은 숫자가 SQL로도 나오는지** 대조합니다.
SQLD(SQL 개발자) 공부와 데이터 직무 포트폴리오를 겸합니다.

- `load.py`가 적재 후 시군구 230곳(전국 포함)의 미혼 성비를 SQL로 계산해 `llmrel/regional.py` 결과와 비교합니다. 현재 **차이 0**으로 일치합니다.
- `practice/answers.sql` Q21은 예측 장부 500줄의 SHA-256 체인을 SQL만으로 다시 검증합니다. 현재 불일치 0건입니다.

## 시작하기

```bash
cd llm-reliability/sql
docker compose up -d                 # Oracle Free 23 (처음 1~2분 초기화)
docker compose logs -f               # "DATABASE IS READY TO USE!" 확인 후 Ctrl+C
pip install oracledb pandas
python load.py                       # 테이블 생성 + 적재 + 파이썬 결과 대조
python practice/grade.py             # 연습 문제 채점
```

쿼리를 직접 써 볼 도구로는 DBeaver나 VS Code의 Oracle SQL Developer 확장을 쓰면 됩니다.
접속 정보: `localhost:1521`, 서비스 이름 `FREEPDB1`, 계정 `sembeop` / `sembeop` (로컬 연습용)

## 파일

| 파일 | 내용 |
|---|---|
| `00_schema.sql` | 테이블 8개. REGION 분리(정규화), 자기참조 관계, 제약조건 |
| `load.py` | CSV·장부 적재와 파이썬 결과 대조 |
| `practice/problems.md` | 1과목 모델링 질문 3개 + 2과목 SQL 문제 21개 |
| `practice/my_answers.sql` | 내 풀이를 쓰는 곳 |
| `practice/grade.py` | 내 풀이를 정답과 **결과로** 비교 (정답 파일은 열지 않아도 됨) |
| `practice/answers.sql` | 정답과 해설. 다 풀고 나서 열기 |

## 제63회 SQLD까지 5주 (시험 2026-11-14 토)

원서 접수는 **10/12(월)~10/16(금)**, [데이터자격시험](https://www.dataq.or.kr)에서 합니다. 합격 발표는 12/4입니다.

| 주 | 기간 | 셈법 데이터 | 이론·기출 |
|---|---|---|---|
| 1 | 10/8~10/18 | 환경 설정, `load.py`, M1~M3, Q01~Q03 | **접수**, 1과목 전체 (엔터티·속성·관계·식별자·정규화) |
| 2 | 10/19~10/25 | Q04~Q10 (조인·서브쿼리·집합) | 2과목: SELECT·함수·조인·서브쿼리 |
| 3 | 10/26~11/1 | Q11~Q17 (그룹·윈도우·계층) | 2과목: 그룹 함수·윈도우 함수·계층형·PIVOT |
| 4 | 11/2~11/8 | Q18~Q21, 틀린 문제 다시 풀기 | DML·TCL·DDL·DCL, 기출 3~4회차 |
| 5 | 11/9~11/13 | 기출에서 틀린 문제를 이 DB로 재현 | 기출만, 90분 시간 재기. 1과목 과락 방지 복습 |

- 셈법 데이터는 개념을 이해하는 데 쓰고, 점수는 기출로 올립니다. 마지막 2주는 기출이 80% 이상입니다.
- 시험에는 SQL Server 문법(`TOP`, `ISNULL` 등)도 가끔 나옵니다. 기출을 풀다가 나오면 Oracle 문법과 짝지어 외워 두세요.

## 포트폴리오에 쓰는 법

> 통계청 KOSIS 원자료를 Oracle에 정규화해 적재하고, 시군구 230곳의 미혼 성비를 SQL로 다시 계산해 파이썬 분석 결과와 대조했습니다(차이 0). 예측 장부 500줄의 해시 체인도 SQL(`CONNECT BY`, `STANDARD_HASH`)로 검증했습니다.
