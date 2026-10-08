-- 셈법 SQL 연습 스키마 (Oracle)
-- 원자료: ../data_regional, ../data_monthly, ../ledger (모두 KOSIS·공개 장부)
-- 적재: python load.py  (이 파일을 먼저 실행한 뒤 데이터를 넣는다)
--
-- 1과목 메모
--   * CSV는 모든 줄에 지역 이름이 반복된다 → 이름을 REGION 하나로 모아 갱신 이상을 없앴다 (3정규형).
--   * REGION.parent_code 는 자기 자신을 가리키는 순환 관계 (전국 → 시도 → 시군구 → 일반구).
--   * MIGRATION 은 다른 코드 체계(행정표준코드)라서 일부러 REGION 과 FK 를 걸지 않았다.

BEGIN
  FOR t IN (SELECT table_name FROM user_tables
            WHERE table_name IN ('PRACTICE_LOG', 'NOWCAST', 'LEDGER', 'MIGRATION',
                                 'MARRIAGE_MONTH', 'MARRIAGE_YEAR', 'POPULATION', 'REGION')) LOOP
    EXECUTE IMMEDIATE 'DROP TABLE ' || t.table_name || ' CASCADE CONSTRAINTS PURGE';
  END LOOP;
END;
/

-- 지역: 전국(00) · 시도(11 등) · 시군구(5자리) · 일반구 · 동부/읍부/면부 합계 · 변동전 코드 · 기타(국외)
CREATE TABLE region (
  code         VARCHAR2(5)  CONSTRAINT pk_region PRIMARY KEY,
  name         VARCHAR2(60) NOT NULL,
  parent_code  VARCHAR2(5)  CONSTRAINT fk_region_parent REFERENCES region (code),
  region_type  VARCHAR2(12) NOT NULL
               CONSTRAINT ck_region_type CHECK (region_type IN ('전국', '시도', '시군구', '일반구', '동읍면부', '변동전', '기타')),
  is_analysis  CHAR(1)      DEFAULT 'N' NOT NULL CONSTRAINT ck_region_analysis CHECK (is_analysis IN ('Y', 'N'))
  -- is_analysis = 'Y': 성비 분석에 쓰는 시군구 (llmrel/regional.py 의 analysis_regions 와 같은 기준)
);

-- 등록 기반 인구 (내국인, 성·연령·혼인상태별) KOSIS DT_1MR2060, 2022~2025
CREATE TABLE population (
  yr        NUMBER(4)   NOT NULL,
  code      VARCHAR2(5) NOT NULL CONSTRAINT fk_pop_region REFERENCES region (code),
  sex       CHAR(1)     NOT NULL CONSTRAINT ck_pop_sex CHECK (sex IN ('M', 'F')),
  kind      VARCHAR2(10) NOT NULL CONSTRAINT ck_pop_kind CHECK (kind IN ('total', 'unmarried')),
  age_band  VARCHAR2(5) NOT NULL,   -- '20-24' ~ '40-44'
  pop       NUMBER      NOT NULL,
  CONSTRAINT pk_population PRIMARY KEY (yr, code, sex, kind, age_band)
);

-- 시군구 연간 혼인 건수 2005~2025
CREATE TABLE marriage_year (
  code  VARCHAR2(5) NOT NULL CONSTRAINT fk_my_region REFERENCES region (code),
  yr    NUMBER(4)   NOT NULL,
  cnt   NUMBER      NOT NULL,
  CONSTRAINT pk_marriage_year PRIMARY KEY (code, yr)
);

-- 시도 월간 혼인 건수 2005-01 ~ 2026-07 (월별 인구동향)
CREATE TABLE marriage_month (
  code  VARCHAR2(5) NOT NULL CONSTRAINT fk_mm_region REFERENCES region (code),
  ym    CHAR(6)     NOT NULL,   -- 'YYYYMM'
  cnt   NUMBER      NOT NULL,
  CONSTRAINT pk_marriage_month PRIMARY KEY (code, ym)
);

-- 시군구 순이동률 (인구 100명당) KOSIS DT_1B26006 — 행정표준코드라 REGION 과 코드가 다르다
CREATE TABLE migration (
  yr        NUMBER(4)    NOT NULL,
  mcode     VARCHAR2(5)  NOT NULL,
  name      VARCHAR2(60) NOT NULL,
  sex       CHAR(1)      NOT NULL,
  age_band  VARCHAR2(5)  NOT NULL,
  rate      NUMBER       NOT NULL,
  CONSTRAINT pk_migration PRIMARY KEY (yr, mcode, sex, age_band)
);

-- 예측 장부 (ledger/predictions.jsonl 한 줄 = 한 행). 추가만 하고 고치지 않는다.
--   hash = sha256(prev_hash || body_json)   body_json = hash 를 뺀 줄 내용 (키 정렬 JSON)
CREATE TABLE ledger (
  seq          NUMBER         CONSTRAINT pk_ledger PRIMARY KEY,
  recorded_at  TIMESTAMP      NOT NULL,
  kind         VARCHAR2(30)   NOT NULL,
  key          VARCHAR2(100)  NOT NULL,
  model        VARCHAR2(400),
  payload      VARCHAR2(1000) CONSTRAINT ck_ledger_payload CHECK (payload IS JSON),
  code_commit  VARCHAR2(40),
  prev_hash    CHAR(64)       NOT NULL,
  hash         CHAR(64)       NOT NULL CONSTRAINT uq_ledger_hash UNIQUE,
  body_json    VARCHAR2(4000) NOT NULL
);

-- 2026년 시군구 혼인 건수 예측 (장부의 nowcast_2026·nowcast_2026_range 마지막 값)
--   actual_2026 은 2027년 통계 공표 전까지 NULL → NULL 처리 연습용
CREATE TABLE nowcast (
  code         VARCHAR2(5) CONSTRAINT pk_nowcast PRIMARY KEY CONSTRAINT fk_nowcast_region REFERENCES region (code),
  pred_2026    NUMBER NOT NULL,
  lo80         NUMBER,
  hi80         NUMBER,
  actual_2026  NUMBER
);

-- DML·TCL 연습용 빈 테이블 (문제 20)
CREATE TABLE practice_log (
  id    NUMBER       CONSTRAINT pk_practice_log PRIMARY KEY,
  note  VARCHAR2(50)
);
