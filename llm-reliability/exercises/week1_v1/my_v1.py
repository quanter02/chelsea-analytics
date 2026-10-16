"""1주차 과제: v1 예측 다시 만들기. TODO를 채우고 check.py로 채점한다.

    python exercises/week1_v1/check.py     # llm-reliability 폴더에서 실행
"""
import pandas as pd

ANNUAL = "data_regional/kr_marriages_by_region.csv"
MONTHLY = "data_monthly/kr_marriages_monthly_sido.csv"
ANSWER = "data_monthly/nowcast_2026.csv"


def last_year_counts() -> pd.Series:
    """1단계. 2025년 시군구(코드 5자리)별 혼인 건수.
    반환: 인덱스 = 시군구 코드(문자열, 예 '11010'), 값 = 건수."""
    # TODO
    raise NotImplementedError


def half_year_sums() -> pd.DataFrame:
    """2단계. 시도별 1~6월 혼인 건수 합계.
    반환: 인덱스 = 시도 코드(문자열, 예 '11'), 열 = 2025, 2026 (각 연도 1~6월 합)."""
    # TODO
    raise NotImplementedError


def sido_growth() -> pd.Series:
    """3단계. 시도별 증가 배율 = 2026년 1~6월 합 ÷ 2025년 1~6월 합.
    반환: 인덱스 = 시도 코드, 값 = 배율(예 1.109)."""
    # TODO  (half_year_sums()를 써도 된다)
    raise NotImplementedError


def predict() -> pd.Series:
    """4단계. 시군구별 2026 예측 = 2025 건수 × 소속 시도 배율, 정수로 반올림.
    정답 파일에 있는 243곳만 남긴다.
    반환: 인덱스 = 시군구 코드, 값 = 예측 건수(int)."""
    # TODO  (last_year_counts(), sido_growth()를 쓰고, 정답 파일의 code 목록으로 거른다)
    raise NotImplementedError


def compare() -> dict:
    """5단계. 내 예측과 정답(pred_2026) 비교.
    반환: {"n": 비교한 곳 수, "same": 정확히 같은 곳 수, "max_diff": 가장 큰 차이(건)}"""
    # TODO
    raise NotImplementedError
