-- 셈법 SQLD 연습 정답 — 문제를 다 푼 뒤에 여세요. 채점은 grade.py 가 이 파일을 대신 읽습니다.
-- 문법은 시험 기준인 Oracle. 같은 결과를 내는 다른 풀이도 정답입니다.

-- Q01  WHERE · LIKE
SELECT COUNT(*) AS cnt
FROM region
WHERE region_type = '시군구'
  AND name LIKE '%군';

-- Q02  셀프 조인 · GROUP BY · HAVING
SELECT s.name AS sido, COUNT(*) AS cnt
FROM region r
JOIN region s ON s.code = r.parent_code
WHERE r.is_analysis = 'Y'
GROUP BY s.code, s.name
HAVING COUNT(*) >= 15
ORDER BY cnt DESC, s.code;

-- Q03  CASE 로 조건부 집계
SELECT ROUND(SUM(CASE WHEN sex = 'M' THEN pop END)
           / SUM(CASE WHEN sex = 'F' THEN pop END), 2) AS ratio
FROM population
WHERE yr = 2025 AND code = '00' AND kind = 'unmarried'
  AND age_band IN ('25-29', '30-34', '35-39');

-- Q04  조인 + 집계 + 정렬 (Top-N 은 FETCH FIRST 로)
SELECT s.name AS sido, r.name, ROUND(x.ratio, 2) AS ratio
FROM (SELECT code,
             SUM(CASE WHEN sex = 'M' THEN pop END) / SUM(CASE WHEN sex = 'F' THEN pop END) AS ratio
      FROM population
      WHERE yr = 2025 AND kind = 'unmarried' AND age_band IN ('25-29', '30-34', '35-39')
      GROUP BY code) x
JOIN region r ON r.code = x.code
JOIN region s ON s.code = r.parent_code
WHERE r.is_analysis = 'Y'
ORDER BY x.ratio DESC
FETCH FIRST 5 ROWS ONLY;

-- Q05  PIVOT
SELECT age_band, m, f, ROUND(m / f, 2) AS ratio
FROM (SELECT age_band, sex, pop
      FROM population
      WHERE yr = 2025 AND code = '00' AND kind = 'unmarried')
PIVOT (SUM(pop) FOR sex IN ('M' AS m, 'F' AS f))
ORDER BY age_band;

-- Q06  ROWNUM Top-N — 정렬을 인라인 뷰 안에서 먼저 해야 한다
--   틀린 예: SELECT ROWNUM, ... FROM ... WHERE ROWNUM <= 10 ORDER BY ratio
--           → ROWNUM 이 정렬 전에 붙으므로 '아무 10곳'을 뽑은 뒤 정렬한다
SELECT ROWNUM AS rn, name, ROUND(ratio, 2) AS ratio
FROM (SELECT r.name,
             SUM(CASE WHEN p.sex = 'M' THEN p.pop END) / SUM(CASE WHEN p.sex = 'F' THEN p.pop END) AS ratio
      FROM population p
      JOIN region r ON r.code = p.code
      WHERE p.yr = 2025 AND p.kind = 'unmarried' AND p.age_band IN ('25-29', '30-34', '35-39')
        AND r.is_analysis = 'Y'
      GROUP BY r.code, r.name
      ORDER BY ratio)
WHERE ROWNUM <= 10
ORDER BY rn;

-- Q07  WITH 절 · 스칼라 서브쿼리
WITH ratio AS (
  SELECT code,
         SUM(CASE WHEN sex = 'M' THEN pop END) / SUM(CASE WHEN sex = 'F' THEN pop END) AS ratio
  FROM population
  WHERE yr = 2025 AND kind = 'unmarried' AND age_band IN ('25-29', '30-34', '35-39')
  GROUP BY code
)
SELECT COUNT(*) AS cnt
FROM ratio x
JOIN region r ON r.code = x.code
WHERE r.is_analysis = 'Y'
  AND x.ratio > (SELECT ratio FROM ratio WHERE code = '00');

-- Q08  상관 서브쿼리 — 바깥 행의 시도(parent_code)를 안쪽에서 참조
WITH ratio AS (
  SELECT r.code, r.name, r.parent_code,
         SUM(CASE WHEN p.sex = 'M' THEN p.pop END) / SUM(CASE WHEN p.sex = 'F' THEN p.pop END) AS ratio
  FROM population p
  JOIN region r ON r.code = p.code
  WHERE p.yr = 2025 AND p.kind = 'unmarried' AND p.age_band IN ('25-29', '30-34', '35-39')
    AND r.is_analysis = 'Y'
  GROUP BY r.code, r.name, r.parent_code
)
SELECT s.name AS sido, a.name, ROUND(a.ratio, 2) AS ratio
FROM ratio a
JOIN region s ON s.code = a.parent_code
WHERE a.ratio = (SELECT MAX(b.ratio) FROM ratio b WHERE b.parent_code = a.parent_code)
ORDER BY s.code;

-- Q09  외부 조인 — 짝이 없는 행 찾기 (Oracle 전용 (+) 표기)
SELECT DISTINCT m.code, r.name
FROM marriage_year m, region r,
     (SELECT DISTINCT code FROM population) p
WHERE r.code = m.code
  AND m.code = p.code(+)
  AND p.code IS NULL
ORDER BY m.code;
--   ANSI 표기: FROM marriage_year m LEFT OUTER JOIN (SELECT DISTINCT code FROM population) p ON p.code = m.code
--              ... WHERE p.code IS NULL

-- Q10  집합 연산 MINUS
WITH ratio AS (
  SELECT p.yr, p.code, r.name,
         SUM(CASE WHEN p.sex = 'M' THEN p.pop END) / SUM(CASE WHEN p.sex = 'F' THEN p.pop END) AS ratio
  FROM population p
  JOIN region r ON r.code = p.code
  WHERE p.kind = 'unmarried' AND p.age_band IN ('25-29', '30-34', '35-39') AND r.is_analysis = 'Y'
  GROUP BY p.yr, p.code, r.name
)
SELECT code, name FROM ratio WHERE yr = 2025 AND ratio < 1
MINUS
SELECT code, name FROM ratio WHERE yr = 2022 AND ratio < 1
ORDER BY code;

-- Q11  ROLLUP + GROUPING — 시도별 소계와 합계를 한 번에
SELECT CASE GROUPING(s.code) WHEN 1 THEN '합계' ELSE s.name END AS sido,
       SUM(CASE WHEN p.sex = 'M' THEN p.pop ELSE -p.pop END) AS excess_men
FROM population p
JOIN region r ON r.code = p.code
JOIN region s ON s.code = r.parent_code
WHERE p.yr = 2025 AND p.kind = 'unmarried' AND p.age_band IN ('25-29', '30-34', '35-39')
  AND r.is_analysis = 'Y'
GROUP BY ROLLUP ((s.code, s.name))
ORDER BY GROUPING(s.code), s.code;

-- Q12  GROUPING SETS — 성별 합, 연령대별 합, 전체 합을 따로 묶어 한 결과로
SELECT sex, age_band, SUM(pop) AS total
FROM population
WHERE yr = 2025 AND code = '00' AND kind = 'unmarried'
  AND age_band IN ('25-29', '30-34', '35-39')
GROUP BY GROUPING SETS ((sex), (age_band), ())
ORDER BY sex NULLS LAST, age_band NULLS LAST;

-- Q13  윈도우 함수 LAG — 필터를 LAG 계산 '뒤'에 걸어야 2026-01 의 전년 값이 살아 있다
SELECT ym, cnt, prev_cnt, ROUND(100 * (cnt - prev_cnt) / prev_cnt, 1) AS pct
FROM (SELECT ym, cnt, LAG(cnt, 12) OVER (ORDER BY ym) AS prev_cnt
      FROM marriage_month
      WHERE code = '00')
WHERE ym LIKE '2026%'
ORDER BY ym;

-- Q14  윈도우 함수 SUM OVER (PARTITION BY ... ORDER BY ...) — 해마다 누적
SELECT mm,
       MAX(CASE WHEN yy = '2026' THEN cum END) AS cum_2026,
       MAX(CASE WHEN yy = '2025' THEN cum END) AS cum_2025
FROM (SELECT SUBSTR(ym, 1, 4) AS yy, SUBSTR(ym, 5, 2) AS mm,
             SUM(cnt) OVER (PARTITION BY SUBSTR(ym, 1, 4) ORDER BY ym) AS cum
      FROM marriage_month
      WHERE code = '00' AND SUBSTR(ym, 1, 4) IN ('2025', '2026'))
WHERE mm <= '07'
GROUP BY mm
ORDER BY mm;

-- Q15  순위 함수 RANK + PARTITION BY
SELECT sido, name, rk, ROUND(ratio, 2) AS ratio
FROM (SELECT s.code AS sido_code, s.name AS sido, r.name,
             SUM(CASE WHEN p.sex = 'M' THEN p.pop END) / SUM(CASE WHEN p.sex = 'F' THEN p.pop END) AS ratio,
             RANK() OVER (PARTITION BY s.code
                          ORDER BY SUM(CASE WHEN p.sex = 'M' THEN p.pop END)
                                 / SUM(CASE WHEN p.sex = 'F' THEN p.pop END) DESC) AS rk
      FROM population p
      JOIN region r ON r.code = p.code
      JOIN region s ON s.code = r.parent_code
      WHERE p.yr = 2025 AND p.kind = 'unmarried' AND p.age_band IN ('25-29', '30-34', '35-39')
        AND r.is_analysis = 'Y'
      GROUP BY s.code, s.name, r.code, r.name)
WHERE rk <= 3
ORDER BY sido_code, rk;

-- Q16  계층형 질의 CONNECT BY — 위(경기도)에서 아래로
SELECT LEVEL AS lvl, COUNT(*) AS cnt
FROM region
START WITH code = '31'
CONNECT BY PRIOR code = parent_code
GROUP BY LEVEL
ORDER BY lvl;

-- Q17  계층형 질의로 장부 체인 따라가기 — 첫 줄(prev 가 0 64개)에서 hash → prev_hash
SELECT LEVEL AS depth, SUBSTR(hash, 1, 16) AS head16
FROM ledger
WHERE CONNECT_BY_ISLEAF = 1
START WITH prev_hash = LPAD('0', 64, '0')
CONNECT BY PRIOR hash = prev_hash;

-- Q18  NULL 과 집계 함수 — COUNT(*) 는 NULL 행도 세고 COUNT(열) 은 세지 않는다
SELECT COUNT(*)                         AS total,
       COUNT(actual_2026)               AS scored,
       COUNT(*) - COUNT(actual_2026)    AS unscored,
       ROUND(AVG(hi80 - lo80))          AS avg_width,
       NVL(SUM(actual_2026), 0)         AS actual_sum
FROM nowcast;

-- Q19  NOT IN 과 NULL 함정
--   틀린 예: WHERE code NOT IN (SELECT parent_code FROM region)
--           → 전국(00)의 parent_code 가 NULL 이라 'code <> NULL' 이 UNKNOWN 이 되어 0행
SELECT COUNT(*) AS leaf_cnt
FROM region r
WHERE NOT EXISTS (SELECT 1 FROM region c WHERE c.parent_code = r.code);
--   또는: WHERE code NOT IN (SELECT parent_code FROM region WHERE parent_code IS NOT NULL)

-- Q20  TCL — SAVEPOINT 와 ROLLBACK TO
INSERT INTO practice_log VALUES (1, '셈');
SAVEPOINT a;
INSERT INTO practice_log VALUES (2, '법');
SAVEPOINT b;
UPDATE practice_log SET note = '채점' WHERE id = 1;
ROLLBACK TO a;
INSERT INTO practice_log VALUES (3, '장부');
SELECT id, note FROM practice_log ORDER BY id;
--   ROLLBACK TO a 는 a 이후의 INSERT(2)·UPDATE 를 모두 취소한다. 결과: (1, 셈), (3, 장부)
--   CREATE TABLE 같은 DDL 이 중간에 있었다면 그 순간 자동 COMMIT 되어 그 전 작업은 되돌릴 수 없다.

-- Q21  (보너스, 시험 범위 밖) SQL 만으로 장부 위변조 검사
SELECT COUNT(*) AS total_rows,
       SUM(CASE WHEN LOWER(RAWTOHEX(STANDARD_HASH(prev_hash || body_json, 'SHA256'))) <> hash THEN 1 ELSE 0 END) AS bad_hash,
       SUM(CASE WHEN prev_hash <> NVL(prev_row_hash, LPAD('0', 64, '0')) THEN 1 ELSE 0 END) AS bad_link
FROM (SELECT l.*, LAG(hash) OVER (ORDER BY seq) AS prev_row_hash FROM ledger l);


-- 1과목(데이터 모델링) 질문 해설 ------------------------------------------------
-- M1. CSV 처럼 모든 행에 name 을 반복하면, 지역 이름이 바뀔 때(예: 강원도 → 강원특별자치도) 수만 행을 고쳐야 하고
--     한 곳이라도 빠지면 같은 코드에 이름이 둘이 된다 = 갱신 이상. name 은 code 에만 함수 종속이므로
--     (yr, code, sex, kind, age_band) 를 키로 갖는 표에서는 키의 일부에 종속된 부분 함수 종속 → 2정규형 위반.
--     REGION 으로 분리해 해결했다.
-- M2. REGION.parent_code → REGION.code 는 같은 엔터티 안의 순환(재귀) 관계, 1:N, 선택(전국은 부모 없음).
--     그래서 parent_code 는 NULL 허용이고, 이 NULL 이 Q19 의 함정을 만든다.
-- M3. POPULATION 의 주식별자 (yr, code, sex, kind, age_band) 는 복합·본질 식별자.
--     대리 식별자(일련번호 id)를 쓰면 키가 짧아지지만 같은 행이 두 번 들어가는 것을 막으려면
--     결국 다섯 열에 UNIQUE 제약이 따로 필요하다.
