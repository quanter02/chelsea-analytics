# 2030 연애·소비 트렌드 리포트

공개 텍스트(유튜브 댓글, 보유한 데이터셋 등)에서 **작성자 본인의** 나이·성별·주제·감정을 뽑아 연령대별 관심사 분포를 신뢰구간과 함께 보여 줍니다.

## 파이프라인

```
collect.py  수집 → data/trends/raw/*.jsonl   (이름·ID 저장 안 함, 작성자는 해시)
extract.py  규칙 기반 추출 (무료)            ┐
llm.py      Claude 추출 (정확, 유료, 캐시)   ┘→ age, age_band, gender, topics, sentiment
aggregate.py 작성자 단위 집계, Wilson 95% CI, 다른 연령대 대비 차이(z)
report.py   reports/trends_report.html
```

## 실행

```bash
pip install -r requirements.txt -r requirements-extra.txt
export PYTHONPATH=src

# 1) 수집 (YouTube Data API v3 키: console.cloud.google.com, 하루 10,000 유닛 무료)
export YOUTUBE_API_KEY=... TRENDS_SALT=아무도-모르는-문자열
python -m trends.collect youtube --query "20대 연애 고민" --videos 20
python -m trends.collect youtube --query "30대 결혼 현실" --videos 20
python -m trends.collect file --path aihub_export.csv --text-col 본문 --date-col 작성일

# 2) 리포트
python -m trends.run                                 # 규칙 기반, 여성 20~39세
python -m trends.run --extractor llm --limit 300     # Claude로 300건만 먼저 (비용 확인)
python -m trends.run --gender all                    # 성별 무관
python -m pytest -q tests/test_trends.py
```

## 정확도에 대해 알아야 할 것

- **나이 확인율이 낮습니다.** 댓글 중 본인 나이를 밝히는 비율은 보통 몇 %입니다. 1만 건을 모아도 분석 대상은 수백 명일 수 있습니다. 리포트 맨 위의 깔때기 숫자(수집 → 나이 확인 → 성별 확인 → 대상)로 확인하세요.
- **규칙 기반은 보수적입니다.** "28살인데", "29살 여자입니다", "96년생" 같은 본인 소개만 인정하고 "남친이 32살"은 버립니다. "서른 앞두고"는 30으로 잡습니다. 맥락이 필요한 표현은 `--extractor llm`이 더 잘 처리합니다.
- **성별은 명시한 경우만 셉니다.** `--partner-cue`를 켜면 "남친 → 여성"으로 추정하지만 틀릴 수 있어 기본값은 꺼져 있습니다.
- **주제 사전은 단순합니다.** "돈"이 "돈까스"에도 걸리는 식의 오탐이 있습니다. LLM으로 200건 정도 라벨링해 규칙 결과와 비교해 보고, 많이 틀리는 주제는 `extract.TOPICS`의 정규식을 고치세요.

## 리포트를 상품으로 만들려면

- **구매자는 마케터와 브랜드입니다.** "25~29세 여성은 연애 고민에서 경제력 언급이 다른 연령대보다 +18%p"처럼 숫자와 신뢰구간이 붙은 한 줄이 상품입니다. 주제를 그들의 카테고리(뷰티, 웨딩, 금융, 여행)로 바꿔 `extract.TOPICS`를 맞춤 제작하세요.
- **공식 통계 기준선:** 통계청 사회조사(결혼관, 소비) 수치를 `data/trends/baseline.csv`(`age_band,item,share,source`)로 넣으면 리포트에 나란히 나옵니다. 텍스트 결과와 방향이 다르면 그 차이 자체가 이야깃거리지만, 먼저 편향을 의심하세요.
- **반복성:** 분기마다 같은 쿼리로 다시 수집하면 변화 추이 리포트가 되고, 구독 상품이 됩니다.

## 지켜야 할 것

- 공식 API나 이용 권한이 있는 데이터셋만 쓰세요. 로그인이 필요한 게시판이나 크롤링을 금지한 커뮤니티는 수집하지 마세요.
- 원문, 작성자 이름, 채널 ID는 리포트에 넣지 않습니다. 해시는 `TRENDS_SALT`를 바꿔 두면 다른 데이터와 맞춰 볼 수 없습니다.
- 결과는 집단 비율입니다. 특정 개인을 판단하는 데 쓰지 마세요. 같은 연령대 안의 차이가 연령대 사이의 차이보다 큽니다.
