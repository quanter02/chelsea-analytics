# 자사주 매입 공시 시그널

`disclosure_event_study_buyback_v2.ipynb`에서 검증한 결과를 **매일 돌아가는 알림**으로 바꾼 모듈입니다.

## 무엇을 근거로 등급을 매기나

등급은 노트북의 전 기간(2016-01 ~ 2026-08, 상장폐지 기업 복원 후) 백테스트 결과에서만 나옵니다.
매매 규칙: 공시 다음 거래일 종가에 사서 20거래일 뒤 종가에 매도합니다. 동일가중 벤치마크 대비 수익에 왕복 비용 0.3%를 반영했습니다.

| 등급 | 그룹 | N | 20일 평균 | 중앙값 | 무작위 날짜 대비 | t |
|---|---|---|---|---|---|---|
| **A** | 코스닥 · 신탁 | 1,789 | +2.05% | +0.57% | +2.26%p | 5.2 (플러스 연도 11/11) |
| **B** | 코스피 · 신탁 | 785 | +1.28% | +0.49% | +2.05%p | 5.0 |
| C | 코스피 · 직접 | 751 | +0.98% | −0.34% | +1.31%p | 2.5 |
| C | 코스닥 · 직접 | 1,015 | +16.18% | −0.32% | +16.36%p | 1.1 (평균은 급등주 몇 개 때문) |

등급에 더해 세 가지 경고를 붙입니다.
- **몰림 구간:** 직전 30일에 신탁 공시가 30건을 넘으면 경고합니다. 과거 이 구간은 평균 1.41%, 중앙값이 마이너스였습니다. 결과를 보고 찾은 규칙이라 필터가 아니라 경고로만 씁니다.
- **유동성:** 하루 거래대금의 5%가 종목당 목표 금액(1,400만 원 = 2.8억 원 ÷ 20슬롯)보다 작으면 경고합니다.
- **30일 내 재공시:** 백테스트는 종목별 첫 공시만 썼으므로, 재공시는 검증 범위 밖이라고 표시합니다.

매입 규모(시가총액 대비 %), 소각 언급, 계약 기간은 **정보로만** 표시하고 등급은 바꾸지 않습니다.
이 항목들을 등급에 넣으려면 먼저 `disclosure.research.verdict`를 통과해야 합니다. 통과 조건은 상위 구간이 하위 구간을 앞뒤 기간 모두에서 평균과 중앙값 둘 다 이기고, 전체 t > 2인 것입니다.

## 실행

```bash
pip install -r requirements.txt -r requirements-extra.txt
export PYTHONPATH=src DART_API_KEY=...       # opendart.fss.or.kr 무료 인증키

python -m disclosure.alert --inspect          # 처음 한 번: DART 상세 필드 이름 확인
python -m disclosure.alert                    # 오늘 공시 → data/disclosure/alert_YYYYMMDD.md
python -m disclosure.alert --date 2026-09-30  # 과거 날짜
python -m disclosure.alert --send             # A·B 등급을 텔레그램으로 (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
python -m pytest -q tests/test_disclosure.py
```

장 마감 후, 공시가 대부분 올라온 19시 이후에 돌리세요. 진입은 다음 날 종가라 시간이 충분합니다.
월요일에는 토~월 공시를 함께 봅니다. 휴장일은 반영하지 않으므로 연휴 전후에는 진입일을 직접 확인하세요.

**자동 실행:** `.github/workflows/disclosure_alert.yml`이 평일 20:10(KST)에 돌아갑니다.
저장소 Settings → Secrets에 `DART_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`를 넣고 Variables에 `DISCLOSURE_ALERTS=on`을 추가하면 켜집니다.

## 새 특징 검증하기 (노트북에서)

```python
import sys; sys.path.append('src')
from disclosure import dart, research as R
feats = pd.DataFrame({i: dart.detail_features(dart.fetch_detail(API_KEY, r)) for i, r in ev_res.iterrows()}).T
ev = ev_res.join(feats)
ev["size_pct"] = ev.amount.astype(float) / ev.marcap * 100     # marcap: 공시 전일 시가총액 (pykrx get_market_cap)
R.bucket_test(ev[ev.type == "신탁"], "size_pct", ret="trade_ew")
R.verdict(ev[ev.type == "신탁"], "size_pct", ret="trade_ew")   # {'통과': True/False, ...}
```

통과한 특징만 `signal.py`의 등급 규칙에 넣으세요. 여러 특징을 시험할수록 우연히 통과하는 것이 생기므로, 시험한 개수를 기록해 두세요.

## 돈을 받고 팔기 전에

- **유사투자자문업 신고:** 불특정 다수에게 돈을 받고 종목 신호를 제공하려면 금융위원회(금융감독원)에 신고해야 합니다. 본인 투자나 무료 공개는 해당하지 않습니다.
- **수익 보장 표현 금지:** "연 18%" 같은 백테스트 수치는 과거 시뮬레이션이라고 명시해야 합니다. 모든 메시지에 면책 문구가 붙어 있습니다.
- **운용 규모:** 소형 코스닥 종목이 많아 포트폴리오 약 2.8억 원을 넘으면 체결이 어렵습니다. 구독자가 늘면 같은 종목을 동시에 사서 수익이 줄어듭니다. 구독 서비스의 숨은 비용입니다.
- **실전 검증 먼저:** 소액으로 3~6개월 운용해 백테스트와 실제 체결 수익의 차이를 기록한 다음에 공개하는 편이 신뢰를 얻기 쉽습니다.
