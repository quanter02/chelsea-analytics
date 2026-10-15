"""셈법 스카우트: 유망주 되팔기 차익 (브라이턴 방식) 검증.

    python scout_youth.py   # scout_v1/v2와 같은 캐시 필요 → output/football/scout_youth.md

질문: 시장(트랜스퍼마크트 시장가치)은 유망주의 경기력을 늦게 반영하는가? 그렇다면 '싸게 사서 비싸게 팔기'가 된다.
가설은 실행 전에 장부 고정 (kind=football_hypothesis, key=scout_youth_rules).
  R1 같은 가격대(시즌 내 시장가치 10분위)의 22세 이하 1500분+ 선수 중 역할 백분위 75+ 가 75 미만보다
     2년 뒤 시장가치가 더 오른다 (로그 변화) → 시장이 경기력을 늦게 반영
  R2 22세 이하·시장가치 하위 절반·역할 75+ 선발군이 기준선(역할 75 미만)보다
     3년 안에 '선발 시점 시장가치의 2배 이상 이적료'로 팔리는 비율이 높다 → 실제 매매 차익
선발 시즌 2017/18~2023/24 (2년 뒤 가치 = 2026년 7월까지). 시장가치는 추정치, 이적료는 실제 기록.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from llmrel import ledger
from scout_v0 import diff_ci
from scout_v1 import MIN_T, OUT, YEARS
from scout_v2 import TM, build, tri, value_at, tm_bio

RULES = dict(version="youth_v1", youth="선발 시즌 9월 1일 만 22세 이하, 1500분+", role="v2와 같음", cheap="같은 시즌·포지션 1500분+ 중 시장가치 하위 절반",
             R1=dict(claim="같은 가격대에서 역할 75+ 유망주가 75 미만보다 2년 뒤 시장가치 로그 상승이 크다", confidence=0.6),
             R2=dict(claim="싸고 역할 75+ 유망주가 3년 안에 선발 시점 가치 2배+ 이적료로 팔리는 비율이 기준선보다 높다", confidence=0.55),
             verdict="차이 95% 범위 3단계", seasons="선발 2017/18~2023/24")
LAST_SEL = max(YEARS) - 2           # 2023 = 2023/24


def main():
    ledger.append([dict(kind="football_hypothesis", key="scout_youth_rules", payload=RULES, model="scout_youth", evidence="결과 보기 전 고정")])
    p, match = build()
    _, v = tm_bio()
    y = p[(p.age <= 22) & p.market_value_in_eur.gt(0)].copy()
    later = value_at(v, y[["player_id", "y"]].assign(y=lambda d: d.y + 2)).rename(columns={"market_value_in_eur": "v2"}).assign(y=lambda d: d.y - 2)
    y = y.merge(later, on=["player_id", "y"], how="left")
    y["g"] = np.log(y.v2 / y.market_value_in_eur)
    hist = y[y.y <= LAST_SEL].copy()

    # R1: 가격대 층별 차이 (역할 상위 − 하위), 부트스트랩
    r1d = hist.dropna(subset=["g"]).copy()
    r1d["bin"] = r1d.groupby("y").market_value_in_eur.transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False))
    r1d["hi"] = r1d.role_pct >= 0.75

    def stat(x):
        m = x.groupby(["y", "bin", "hi"]).g.mean().unstack().dropna()
        w = x[x.hi].groupby(["y", "bin"]).size().reindex(m.index).fillna(0)
        return float(((m[True] - m[False]) * w).sum() / w.sum())

    rng = np.random.default_rng(0)
    e1 = stat(r1d)
    bs = [stat(r1d.iloc[rng.integers(0, len(r1d), len(r1d))]) for _ in range(3000)]
    lo1, hi1 = np.percentile(bs, [2.5, 97.5])

    # R2: 실제 이적료 2배+
    t = pd.read_csv(TM / "transfers.csv.gz", usecols=["player_id", "transfer_date", "transfer_fee"])
    t = t[t.transfer_fee > 0].assign(transfer_date=lambda d: pd.to_datetime(d.transfer_date, errors="coerce"))
    sold = []
    for r in hist.itertuples():
        w = t[(t.player_id == r.player_id) & (t.transfer_date > pd.Timestamp(f"{r.y + 1}-07-01")) & (t.transfer_date <= pd.Timestamp(f"{r.y + 4}-07-01"))]
        sold.append(float(w.transfer_fee.max()) if len(w) else 0.0)
    hist["max_fee"] = sold
    hist["flip"] = hist.max_fee >= 2 * hist.market_value_in_eur
    cheap = hist.val_pct <= 0.5
    S, B = hist[cheap & (hist.role_pct >= 0.75)], hist[cheap & (hist.role_pct < 0.75)]
    ma, mb, lo2, hi2 = diff_ci(S.flip.astype(float).to_numpy(), B.flip.astype(float).to_numpy())
    res = dict(R1=dict(diff_log=round(e1, 4), ci=[round(float(lo1), 4), round(float(hi1), 4)], n=[int(r1d.hi.sum()), int((~r1d.hi).sum())], verdict=tri(lo1, hi1)),
               R2=dict(selected=round(ma, 4), baseline=round(mb, 4), n=[len(S), len(B)], ci=[round(lo2, 4), round(hi2, 4)], verdict=tri(lo2, hi2)))
    ledger.append([dict(kind="football_result", key="scout_youth_backtest", payload=res, model="scout_youth",
                        evidence=f"understat+트랜스퍼마크트, 선발 2017/18~2023/24, 연결률 {match:.0%}")])

    # 돈으로 보면: 선발군 한 명당 (최고 이적료 − 선발 시점 가치), 팔린 경우만 / 전체 평균
    def money(d):
        gain = np.where(d.max_fee > 0, d.max_fee - d.market_value_in_eur, np.nan)
        return np.nanmedian(gain) / 1e6 if np.isfinite(gain).any() else np.nan, (d.max_fee > 0).mean(), d.market_value_in_eur.median() / 1e6
    mS, mB = money(S), money(B)
    ex = S.sort_values("max_fee", ascending=False).head(10)
    exl = "\n".join(f"| {r.name} | {r.y}/{str(r.y + 1)[2:]} | {int(r.age)} | {r.pos} | {r.team} | €{r.market_value_in_eur / 1e6:.1f}m | "
                    f"{'€%.1fm' % (r.max_fee / 1e6) if r.max_fee > 0 else '안 팔림'} |" for r in ex.itertuples())

    # 2026/27 등록: 2025/26 자료로 선발, 2028년 여름 가치로 채점
    cur = y[(y.y == max(YEARS)) & (y.val_pct <= 0.5) & (y.role_pct >= 0.75)].sort_values("role_pct", ascending=False)
    payload = dict(season="2026-27", picks=[dict(player=r.name, team=r.team, age=int(r.age), pos=r.pos, value_eur=int(r.market_value_in_eur),
                                                 role_pct=round(float(r.role_pct), 3)) for r in cur.itertuples()],
                   scoring="2028년 7월 시장가치 로그 변화를 같은 가격대 기준선과 비교, 2029년 7월까지 이적료 2배+ 여부")
    n_new = ledger.append([dict(kind="scout_pick", key="2026-27:youth_v1", payload=payload, model="scout_youth",
                                evidence="2025/26 자료 + 2026년 7월 이전 시장가치만 사용")])
    ok, n_rows, _ = ledger.verify()
    cl = "\n".join(f"| {r.name} | {int(r.age)} | {r.pos} | {r.team} | €{r.market_value_in_eur / 1e6:.1f}m | {r.role_pct:.0%} |" for r in cur.itertuples())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scout_youth.md").write_text(f"""# 유망주 되팔기 차익 검증 (브라이턴 방식)

자료: understat 5대 리그 선수 시즌 + transfermarkt-datasets (시장가치·실제 이적료). 22세 이하·1500분+, 선발 2017/18~2023/24.
가설은 실행 전 장부 고정. 시장가치는 추정치, 이적료는 실제 기록 (임대·자유계약 0원은 '판매' 아님).

## 가설 판정
| 가설 | 결과 | 판정 |
|---|---|---|
| R1 같은 가격대에서 역할 좋은 유망주가 2년 뒤 시장가치가 더 오른다 (시장이 경기력을 늦게 반영, 자신감 60%) | 로그 차이 {e1:+.3f} (약 {np.expm1(e1):+.0%}, 95% {np.expm1(lo1):+.0%} ~ {np.expm1(hi1):+.0%}), n={res['R1']['n'][0]}/{res['R1']['n'][1]} | {res['R1']['verdict']} |
| R2 싸고 역할 좋은 유망주가 3년 안에 선발 시점 가치의 2배+ 이적료로 팔린다 (자신감 55%) | {ma:.0%} (n={len(S)}) vs {mb:.0%} (n={len(B)}), 차이 {ma - mb:+.1%}p (95% {lo2:+.1%} ~ {hi2:+.1%}) | {res['R2']['verdict']} |

## 돈으로 보면 (설명용)
| 그룹 | 선발 시점 가치 중앙값 | 3년 안에 이적료 받고 팔린 비율 | 팔린 경우 차익 중앙값 (최고 이적료 − 선발 시점 가치) |
|---|---|---|---|
| 선발군 (싸고 역할 75+) | €{mS[2]:.1f}m | {mS[1]:.0%} | €{mS[0]:.1f}m |
| 기준선 (싸고 역할 75 미만) | €{mB[2]:.1f}m | {mB[1]:.0%} | €{mB[0]:.1f}m |

## 선발군 중 가장 비싸게 팔린 예
| 선수 | 선발 시즌 | 나이 | 포지션 | 소속 | 선발 시점 가치 | 3년 안 최고 이적료 |
|---|---|---|---|---|---|---|
{exl}

## 2026/27 유망주 명단 ({len(cur)}명) — 장부 등록 {'완료' if n_new else '이미 있음(같은 내용)'}
| 선수 | 나이 | 포지션 | 소속 (2025/26) | 시장가치 | 역할 백분위 |
|---|---|---|---|---|---|
{cl}

- 채점: 2028년 7월 시장가치 변화, 2029년 7월까지 이적료 2배+ 여부. 장부 전체 {n_rows}줄, 체인 {'정상' if ok else '어긋남'}

## 한계
- 이적료 차익은 '구단이 실제로 번 돈'이 아님: 연봉, 에이전트 수수료, 셀온 조항, 선수를 쓰는 동안의 가치가 빠짐
- 차익 중앙값은 팔린 선수만 본 값 → 안 팔린 선수(대부분)는 빠져 있음
- 5대 리그 1500분+ 유망주는 이미 검증된 선수라 브라이턴이 노리는 '남미·하부 리그 유망주'와 다름 (그 자료는 없음)
""", encoding="utf-8")
    print((OUT / "scout_youth.md").read_text())


if __name__ == "__main__":
    main()
