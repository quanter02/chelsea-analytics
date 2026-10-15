"""우리 EPL 모델 vs 시장(마감 배당) 비교 보고서.

    python epl_market.py     # 먼저 data_epl/odds/ 에 football-data.co.uk E0.csv 를 받아 둔다 → output/epl_market/report.md
"""
from __future__ import annotations

import warnings
from pathlib import Path

from llmrel import epl, market

OUT = Path(__file__).parent / "output" / "epl_market"


def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    xg, m = market.compare(epl.XG_MODE)
    gl, _ = market.compare(epl.GOALS_MODE)
    def table(o):
        return "\n".join(f"| {r.구간} | {r.경기} | {r.우리:.4f} | {r.시장:.4f} | {r.차이:+.4f} | {r.lo:+.4f} ~ {r.hi:+.4f} |" for r in o.itertuples())
    t = m[m.season.isin(epl.SPLITS["test"])]
    base_test = epl.baseline(epl.load(), "test").mean()
    gap_closed = (base_test - xg.set_index("구간").loc["test", "우리"]) / (base_test - xg.set_index("구간").loc["test", "시장"])
    (OUT / "report.md").write_text(f"""# EPL 모델 vs 시장 배당

시장 확률 = 마감 직전 배당(피나클 {xg.attrs['sources'].get('피나클 마감', 0)}경기, 평균 {xg.attrs['sources'].get('평균 마감', 0)}경기)에서 북메이커 마진을 뺀 값. 같은 경기 {xg.attrs['matched']}개 비교.
RPS 낮을수록 좋음. 차이 = 시장 − 우리 (**음수면 시장이 더 잘 맞힘**). 95% 범위는 경기 단위 부트스트랩.

## xG 모드 (지금 쓰는 설정)
| 구간 | 경기 | 우리 | 시장 | 차이 | 95% 범위 |
|---|---|---|---|---|---|
{table(xg)}

## 골 모드
| 구간 | 경기 | 우리 | 시장 | 차이 | 95% 범위 |
|---|---|---|---|---|---|
{table(gl)}

## 결론
- **시장이 더 잘 맞힌다.** 검증 구간(1,140경기)에서 확실히 지고(범위가 0 아래), 시험 구간(760경기)에서도 차이 −0.004.
- 리그 평균 기준선({base_test:.4f})에서 시장({xg.set_index('구간').loc['test', '시장']:.4f})까지 거리 중 **{gap_closed:.0%}**를 우리 모델이 따라감.
- 시장과 섞어 보면 검증 구간 최적 비중이 우리 모델 {xg.attrs['best_weight']:.0%} → 우리 모델은 시장이 이미 아는 것 외에 더하는 정보가 없다.
- 두 확률의 상관 {t.pH.corr(t.mH):.2f}, 홈승 확률 평균 차이 {(t.pH - t.mH).abs().mean():.3f}: 대체로 비슷한 판단을 하되 시장이 조금 더 정확.
- 이번 시즌 60경기는 범위가 넓어 판정 불가.
""", encoding="utf-8")
    print((OUT / "report.md").read_text())


if __name__ == "__main__":
    main()
