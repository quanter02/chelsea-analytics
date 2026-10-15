# EPL 라운드 기록 (2026-10-15)

- xG: understat 수집 안 함 (--no-fetch), 결과 자료와 연결 60/60
- 모드: **xG 모드** (이번 시즌 xG 붙은 경기 60개)
- 결과 반영: 2026-10-12까지, 이번 시즌 60경기
- 예측 장부: 이번에 20줄 추가, 전체 829줄, 체인 정상, 마지막 hash `9d94c72630dc7378`

## Opta와 같은 경기 비교 (채점 끝난 것만, RPS 낮을수록 좋음)

| who          |   mean |   count |
|:-------------|-------:|--------:|
| Dimers       | 0.1763 |       1 |
| Opta         | 0.1753 |       2 |
| StatsInsider | 0.1725 |       1 |
| StatsUltra   | 0.1788 |       1 |
| 우리 모델        | 0.2585 |       1 |

## 기록된 예측

| date       | home                   | away                   | source                   |    pH |    pD |    pA | status   |     rps |
|:-----------|:-----------------------|:-----------------------|:-------------------------|------:|------:|------:|:---------|--------:|
| 2026-10-10 | Arsenal                | Leeds United           | 우리 모델 (xG 모드)            | 0.579 | 0.251 | 0.17  | scored   |   0.103 |
| 2026-10-10 | Aston Villa            | Brentford              | 우리 모델 (xG 모드)            | 0.357 | 0.234 | 0.408 | scored   |   0.147 |
| 2026-10-10 | Chelsea                | Bournemouth            | 우리 모델 (xG 모드)            | 0.336 | 0.235 | 0.429 | scored   |   0.312 |
| 2026-10-10 | Ipswich Town           | Fulham                 | 우리 모델 (xG 모드)            | 0.331 | 0.261 | 0.408 | scored   |   0.307 |
| 2026-10-10 | Manchester United      | Tottenham Hotspur      | Dimers (웹 검색 10/6)       | 0.544 | 0.218 | 0.238 | scored   |   0.176 |
| 2026-10-10 | Manchester United      | Tottenham Hotspur      | Opta (사용자 제공)            | 0.559 | 0.226 | 0.215 | scored   |   0.179 |
| 2026-10-10 | Manchester United      | Tottenham Hotspur      | Opta (프리뷰 기사 10/8)       | 0.541 | 0.236 | 0.223 | scored   |   0.171 |
| 2026-10-10 | Manchester United      | Tottenham Hotspur      | StatsInsider (웹 검색 10/6) | 0.533 | 0.22  | 0.247 | scored   |   0.173 |
| 2026-10-10 | Manchester United      | Tottenham Hotspur      | StatsUltra (웹 검색 10/6)   | 0.56  | 0.23  | 0.21  | scored   |   0.179 |
| 2026-10-10 | Manchester United      | Tottenham Hotspur      | 우리 모델 (xG 모드)            | 0.709 | 0.171 | 0.12  | scored   |   0.259 |
| 2026-10-10 | Sunderland             | Brighton & Hove Albion | 우리 모델 (xG 모드)            | 0.285 | 0.225 | 0.489 | scored   |   0.171 |
| 2026-10-11 | Crystal Palace         | Nottingham Forest      | 우리 모델 (xG 모드)            | 0.297 | 0.258 | 0.446 | scored   |   0.346 |
| 2026-10-11 | Hull City              | Everton                | 우리 모델 (xG 모드)            | 0.364 | 0.27  | 0.366 | scored   |   0.133 |
| 2026-10-11 | Liverpool              | Manchester City        | 우리 모델 (xG 모드)            | 0.301 | 0.235 | 0.464 | scored   |   0.189 |
| 2026-10-12 | Coventry City          | Newcastle United       | 우리 모델 (xG 모드)            | 0.319 | 0.245 | 0.436 | scored   |   0.327 |
| 2026-10-17 | Brentford              | Liverpool              | 우리 모델 (xG 모드)            | 0.504 | 0.23  | 0.266 | pending  | nan     |
| 2026-10-17 | Everton                | Chelsea                | 우리 모델 (xG 모드)            | 0.502 | 0.217 | 0.281 | pending  | nan     |
| 2026-10-17 | Fulham                 | Hull City              | 우리 모델 (xG 모드)            | 0.477 | 0.256 | 0.267 | pending  | nan     |
| 2026-10-17 | Manchester City        | Ipswich Town           | 우리 모델 (xG 모드)            | 0.856 | 0.095 | 0.049 | pending  | nan     |
| 2026-10-17 | Newcastle United       | Aston Villa            | 우리 모델 (xG 모드)            | 0.462 | 0.214 | 0.323 | pending  | nan     |
| 2026-10-18 | Bournemouth            | Sunderland             | 우리 모델 (xG 모드)            | 0.563 | 0.224 | 0.214 | pending  | nan     |
| 2026-10-18 | Brighton & Hove Albion | Crystal Palace         | 우리 모델 (xG 모드)            | 0.747 | 0.145 | 0.108 | pending  | nan     |
| 2026-10-18 | Leeds United           | Manchester United      | 우리 모델 (xG 모드)            | 0.409 | 0.248 | 0.343 | pending  | nan     |
| 2026-10-18 | Nottingham Forest      | Arsenal                | 우리 모델 (xG 모드)            | 0.284 | 0.271 | 0.446 | pending  | nan     |
| 2026-10-19 | Tottenham Hotspur      | Coventry City          | 우리 모델 (xG 모드)            | 0.489 | 0.256 | 0.256 | pending  | nan     |
| 2026-10-23 | Ipswich Town           | Nottingham Forest      | 우리 모델 (xG 모드)            | 0.253 | 0.235 | 0.512 | pending  | nan     |
| 2026-10-24 | Arsenal                | Everton                | 우리 모델 (xG 모드)            | 0.639 | 0.214 | 0.147 | pending  | nan     |
| 2026-10-24 | Aston Villa            | Manchester City        | 우리 모델 (xG 모드)            | 0.208 | 0.201 | 0.591 | pending  | nan     |
| 2026-10-24 | Chelsea                | Tottenham Hotspur      | 우리 모델 (xG 모드)            | 0.553 | 0.216 | 0.231 | pending  | nan     |
| 2026-10-24 | Coventry City          | Fulham                 | 우리 모델 (xG 모드)            | 0.371 | 0.267 | 0.362 | pending  | nan     |
| 2026-10-25 | Crystal Palace         | Newcastle United       | 우리 모델 (xG 모드)            | 0.421 | 0.221 | 0.358 | pending  | nan     |
| 2026-10-25 | Hull City              | Brentford              | 우리 모델 (xG 모드)            | 0.3   | 0.243 | 0.457 | pending  | nan     |
| 2026-10-25 | Liverpool              | Brighton & Hove Albion | 우리 모델 (xG 모드)            | 0.338 | 0.234 | 0.428 | pending  | nan     |
| 2026-10-25 | Manchester United      | Bournemouth            | 우리 모델 (xG 모드)            | 0.508 | 0.227 | 0.265 | pending  | nan     |
| 2026-10-25 | Sunderland             | Leeds United           | 우리 모델 (xG 모드)            | 0.36  | 0.261 | 0.379 | pending  | nan     |