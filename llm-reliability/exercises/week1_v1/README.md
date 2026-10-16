# 1주차 과제: v1 예측을 내 손으로 다시 만들기

목표: 장부에 등록된 **2026 시군구 혼인 예측 v1**(243곳)을 직접 짠 코드로 똑같이 만들어 낸다.
v1의 아이디어는 한 줄이다.

> **올해 예측 = 작년 건수 × (우리 시도의 올해 1~6월 ÷ 작년 1~6월)**

예: 종로구 2025년 615건, 서울의 2026년 1~6월이 2025년 1~6월보다 10.9% 많았다면 → 615 × 1.109 ≈ 682건.

## 하는 법
1. `my_v1.py`를 열고 `TODO`가 있는 함수 5개를 순서대로 채운다. 하루 1~2개면 충분하다.
2. 하나 채울 때마다 채점한다 (`llm-reliability` 폴더에서):
   ```
   python exercises/week1_v1/check.py
   ```
   통과한 단계는 ✅, 아직이면 ❌와 힌트가 나온다. 정답 코드는 출력하지 않는다.
3. 5단계가 모두 ✅면 v1을 완전히 재현한 것이다.

## 쓰는 자료 (이미 폴더에 있음)
| 파일 | 내용 | 열 |
|---|---|---|
| `data_regional/kr_marriages_by_region.csv` | 시군구·시도 연간 혼인 건수 | code, name, year, value |
| `data_monthly/kr_marriages_monthly_sido.csv` | 시도 월별 혼인 건수 | code, name, month(예: 202603), value |
| `data_monthly/nowcast_2026.csv` | 정답(장부의 v1) — 채점에만 쓴다 | made_on, code, last_2025, pred_2026 |

코드 규칙: 시군구 코드는 5자리(예: `11010` 종로구), **앞 2자리가 소속 시도 코드**(`11` 서울).

## 단계
| 단계 | 함수 | 하는 일 | 연습되는 것 |
|---|---|---|---|
| 1 | `last_year_counts` | 2025년 시군구별 건수 (code → 건수) | `read_csv`, 행 고르기, `set_index` |
| 2 | `half_year_sums` | 시도별 1~6월 합계, 2025와 2026 각각 | 열 만들기(`// 100`, `% 100`), `groupby().sum()` |
| 3 | `sido_growth` | 시도별 증가 배율 (2026 합 ÷ 2025 합) | 시리즈끼리 나누기 |
| 4 | `predict` | 시군구마다 작년 건수 × 소속 시도 배율, 반올림 | 문자열 자르기 `.str[:2]`, `map` |
| 5 | `compare` | 정답과 비교: 일치 개수, 가장 큰 차이 | 두 표 합치기, 절댓값 |

## 힌트 (막힐 때만 하나씩 보기)
<details><summary>1단계</summary>

`df[df.year == 2025]`로 2025년만 남기고, 코드 길이가 5인 행만 남긴다(`df.code.str.len() == 5`). code를 문자열로 읽으려면 `dtype={"code": str}`.
</details>
<details><summary>2단계</summary>

`month`가 202603 같은 숫자다. `month // 100`이 연도, `month % 100`이 월. 월이 6 이하인 행만 남긴 뒤 `groupby(["y", "code"]).value.sum()`.
</details>
<details><summary>3단계</summary>

2단계 결과에서 `.loc[2026]`과 `.loc[2025]`를 꺼내 나눈다. 인덱스(시도 코드)가 같으면 자동으로 맞춰서 나눠진다.
</details>
<details><summary>4단계</summary>

시군구 코드 시리즈에 `.str[:2]`를 하면 시도 코드가 된다. 그걸 `.map(배율)` 하면 시군구마다 배율이 붙는다. 곱한 뒤 `.round().astype(int)`.
</details>
<details><summary>5단계</summary>

정답 파일을 code 기준으로 읽고, 내 예측과 `(내 예측 - 정답).abs()`를 구한다. 일치 개수는 `(차이 == 0).sum()`.
</details>

## 다 끝나면 스스로 답해 보기 (면접 연습)
1. v1이 시군구마다 다른 정보는 무엇이고, 같은 정보는 무엇인가?
2. 어떤 시군구에서 v1이 크게 틀릴까? (힌트: 지도의 신도시)
3. v5는 v1에 무엇을 더했고, 왜 오차가 줄었나? → `output/model_upgrade/challenger_youthpop.md`
