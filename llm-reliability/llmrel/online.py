"""온라인 모드: 매주 그 시점까지 '보이는' 데이터만으로 프로파일과 라우팅을 갱신한다.

보이는 데이터 = 이미 만들어졌고(created_at ≤ T) 정답도 확정된(labeled_at ≤ T) 예측.
정답은 평균 2주 늦게 붙으므로, 시스템은 항상 조금 늦은 과거를 보고 판단한다.

전략
  static      시작 시점에 정책을 한 번 고르고 그대로 둔다
  cumulative  매주 다시 고르되, 지금까지의 데이터를 모두 같은 무게로 쓴다
  adaptive    매주 다시 고르고, 오래된 데이터는 덜 믿고(망각 계수),
              모델별 확신 오답률을 CUSUM으로 감시하다가 급변하면 그 모델의 과거 데이터를 버린다
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import router as R
from .simulate import START

STRATEGIES = {
    "static":     dict(label="한 번만 선택",            reselect=False, half_life=None, cusum=False),
    "cumulative": dict(label="매주 재선택 (전체 누적)",  reselect=True,  half_life=None, cusum=False),
    "adaptive":   dict(label="매주 재선택 + 망각 + 변화 감지", reselect=True, half_life=30, cusum=True),
}


def _days(ts: pd.Series, origin=START) -> pd.Series:
    return (pd.to_datetime(ts) - pd.Timestamp(origin)).dt.total_seconds() / 86400


def _input_weights(vis: pd.DataFrame, T: float, half_life: float | None) -> pd.Series | None:
    if half_life is None:
        return None
    age = T - vis.groupby("input_hash").cday.min()
    return 0.5 ** (age / half_life)


def run(scored_all: pd.DataFrame, human_usd: float, strategy: str, start_day: int = 90, end_day: int = 240,
        step: int = 7, min_weight: float = 100.0, cusum_h: float = 5.0, cusum_k: float = 0.5, discard_back: int = 21,
        origin=START):
    cfg = STRATEGIES[strategy]
    S = scored_all.assign(cday=lambda d: _days(d.created_at, origin), lday=lambda d: _days(d.labeled_at, origin))
    discard: dict[str, float] = {}           # 모델 → 이 날 이전에 만들어진 예측은 버림
    cus: dict[str, tuple[float, float]] = {}
    policy, rows, alarms = None, [], []

    for T in np.arange(start_day, end_day, step):
        vis = S[(S.cday <= T) & (S.lday <= T)]
        drop = lambda v: v[~np.logical_or.reduce([(v.model_id == m) & (v.cday < d) for m, d in discard.items()] or [np.zeros(len(v), bool)])]
        vis = drop(vis)

        if cfg["cusum"]:                      # 이번 주에 새로 확정된 정답으로 모델별 확신 오답률 감시
            for key, g in R.latest_versions(vis).groupby("model"):
                g = g[g.answer != "ABSTAIN"]
                old, new = g[g.lday <= T - step], g[g.lday > T - step]
                if len(old) < 50 or len(new) == 0:
                    continue
                w = 0.5 ** ((T - old.cday) / cfg["half_life"])
                p0 = float(np.clip(np.average(old.outcome == "confident_error", weights=w), 0.02, 0.98))
                x, n = (new.outcome == "confident_error").sum(), len(new)
                z = (x - n * p0) / np.sqrt(n * p0 * (1 - p0))
                up, dn = cus.get(key, (0.0, 0.0))
                up, dn = max(0.0, up + z - cusum_k), max(0.0, dn - z - cusum_k)
                if max(up, dn) > cusum_h:
                    m = key.split("@")[0]
                    discard[m] = T - discard_back
                    alarms.append(dict(day=T, model=key, direction="악화" if up > dn else "개선",
                                       baseline=p0, this_week=x / n))
                    up = dn = 0.0
                cus[key] = (up, dn)
            vis = drop(vis)

        if policy is None or cfg["reselect"]:
            W = R.wide(vis, _input_weights(vis, T, cfg["half_life"]))
            eligible = [m for m in W["models"] if W["weight"][W["answer"][m].notna().to_numpy()].sum() >= min_weight]
            best = None
            for p in R.candidates(eligible):
                r = R.evaluate(W, p, {}, human_usd)
                if r["weight"] >= min_weight and (best is None or r["total"] < best[1]):
                    best = (p, r["total"])
            policy = best[0]

        # 다음 한 주 동안 이 정책을 실제로 썼을 때의 비용 (정답은 시뮬레이션이 알고 있음)
        week = S[(S.cday > T) & (S.cday <= T + step)]
        if week.empty:                                     # 판단할 일이 없는 주 (예: 리그 휴식기)
            continue
        real = R.evaluate(R.wide(week), policy, {}, human_usd)
        rows.append(dict(strategy=strategy, day=T, policy=str(policy),
                         first_model=policy.steps[0][0] if policy.steps else "사람",
                         cost=real["total"], error_loss=real["error_loss"], human_share=real["human_share"],
                         wrong_share=real["wrong_share"]))
    return pd.DataFrame(rows), pd.DataFrame(alarms, columns=["day", "model", "direction", "baseline", "this_week"])
