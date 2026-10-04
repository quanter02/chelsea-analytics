"""한계효용 의사결정 규칙 세 가지를 같은 환경에서 비교.

환경 (기간 = 1개월, 10년 = 120기간)
  - 선택지(정보원·채널·사업 기회) k 마다 기본 가치 a_k. 한 기간에 n번째 단위의 한계효용 = a_k × 0.8^(n-1) (체감)
  - 단위마다 비용 c (시간·돈). 순가치 = 한계효용 − c
  - a_k 는 조금씩 변하고(드리프트), 일부는 꾸준히 낡고, 새 선택지가 계속 생긴다 → 시간이 갈수록 기회 총량이 늘어난다
  - 실제 얻은 효용은 잡음이 섞여 관측되고, 규칙은 관측값으로 a_k 를 추정한다

규칙
  R1 최소치 올리기   추정 한계효용 ≥ 기준선. 기준선은 매 기간 '지난 기간 받아들인 것 중 하위 10%' 이상으로 올라가고 내려가지 않는다
  R2 최소치 = 비용   추정 한계효용 ≥ c (점추정, 누적 평균으로 추정)
  R3 R2 + 보수적 판단 + 탐색 + 재추정
                     하한(추정 − 불확실성) ≥ c 일 때만 실행, 단위의 10%는 불확실성이 큰 선택지 시험,
                     추정은 최근 관측에 더 무게 (망각 계수)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DECAY = 0.8          # 같은 선택지 안에서 다음 단위 가치 비율
MAX_UNITS = 12       # 한 선택지에서 한 기간에 할 수 있는 최대 단위
COST = 1.0
NOISE = 0.6          # 관측 잡음 (곱셈, 로그 표준편차)


@dataclass
class Env:
    a: np.ndarray            # [기간, 선택지] 실제 기본 가치 (아직 없는 선택지는 0)
    born: np.ndarray         # 선택지가 생긴 기간


def make_env(seed: int, periods: int = 120, k0: int = 8, new_every: float = 10.0) -> Env:
    rng = np.random.default_rng(seed)
    births = [0] * k0
    t = 0.0
    while True:
        t += rng.exponential(new_every)
        if t >= periods:
            break
        births.append(int(t))
    K = len(births)
    base = np.exp(rng.normal(np.log(2.0), 0.6, K))
    trend = rng.normal(-0.004, 0.006, K)                      # 대체로 조금씩 낡음
    a = np.zeros((periods, K))
    for k in range(K):
        la = np.log(base[k])
        for t in range(births[k], periods):
            la += trend[k] + rng.normal(0, 0.04)
            a[t, k] = np.exp(la)
    return Env(a=a, born=np.array(births))


def _units(est: float, floor: float) -> int:
    """추정 기본 가치 est 일 때 한계효용이 floor 이상인 단위 수."""
    if est <= floor or est <= 0:
        return 0
    return int(min(MAX_UNITS, np.floor(np.log(floor / est) / np.log(DECAY)) + 1))


def run(env: Env, rule: str, seed: int = 0, explore: float = 0.10, forget: float = 0.85, z: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 7)
    T, K = env.a.shape
    s = np.zeros(K); w = np.zeros(K)                           # 가중 합, 가중치 (추정용)
    prior = 2.0                                                # 처음 보는 선택지의 추정 가치 (전체 평균 정도)
    floor = COST
    tried = np.zeros(K, bool)
    rows = []
    for t in range(T):
        alive = env.born <= t
        est = np.where(w > 0, s / np.maximum(w, 1e-9), prior)
        sd = NOISE * est / np.sqrt(np.maximum(w, 0.25))          # 관측이 적을수록 불확실
        plan = np.zeros(K, int)
        for k in np.flatnonzero(alive):
            if rule == "R1":
                plan[k] = _units(est[k], floor)
            elif rule.startswith("F"):                         # 기준선 고정 (비용의 몇 배)
                plan[k] = _units(est[k], COST * float(rule[1:]))
            elif rule.startswith("O"):                         # 진짜 가치를 안다고 가정 (추정 문제 없음)
                plan[k] = _units(env.a[t, k], COST * float(rule[1:]))
            elif rule == "R2":
                plan[k] = _units(est[k], COST)
            else:
                plan[k] = _units(est[k] - z * sd[k], COST)
        if rule == "R3":                                       # 탐색: 전체 단위의 10%를 상한이 가장 큰 미실행 선택지에
            budget = max(1, int(round(explore * max(plan.sum(), 10))))
            ucb = np.where(alive & (plan == 0), est + 2 * sd, -np.inf)
            for k in np.argsort(-ucb)[:budget]:
                if np.isfinite(ucb[k]):
                    plan[k] = 1
        mus = []
        tried[plan > 0] = True
        for k in np.flatnonzero(plan):
            n = plan[k]
            true_mu = env.a[t, k] * DECAY ** np.arange(n)
            obs = true_mu * np.exp(rng.normal(0, NOISE, n) - NOISE ** 2 / 2)
            mus += list(true_mu)
            # 관측으로 기본 가치 추정 갱신
            a_hat = obs / DECAY ** np.arange(n)
            if rule == "R3":
                s[k] *= forget; w[k] *= forget
            s[k] += a_hat.sum(); w[k] += n
        if rule == "R1" and mus:                               # 기준선은 올라가기만 한다
            accepted_est = [est[k] * DECAY ** i for k in np.flatnonzero(plan) for i in range(plan[k])]
            floor = max(floor, float(np.quantile(accepted_est, 0.10)))
        mus = np.array(mus)
        # 그 기간에 '알았다면' 얻을 수 있었던 최대 순가치 (실제 값으로 비용 이상 전부 실행)
        best = sum(max(0.0, m - COST) for k in np.flatnonzero(alive) for m in env.a[t, k] * DECAY ** np.arange(MAX_UNITS))
        rows.append(dict(t=t, rule=rule, units=len(mus), utility=mus.sum(), net=(mus - COST).sum() if len(mus) else 0.0,
                         avg=mus.mean() if len(mus) else np.nan, min_mu=mus.min() if len(mus) else np.nan,
                         best_net=best, floor=floor if rule == "R1" else COST, options=int(alive.sum()),
                         new_tried=float(tried[alive & (env.born > 0)].mean()) if (alive & (env.born > 0)).any() else np.nan))
    return pd.DataFrame(rows)


def compare(seeds=range(40), periods: int = 120, rules=("R1", "R2", "R3")) -> pd.DataFrame:
    out = []
    for sd in seeds:
        env = make_env(sd, periods)
        for r in rules:
            out.append(run(env, r, seed=sd).assign(seed=sd))
    return pd.concat(out, ignore_index=True)
