"""한계효용 기반 추출 스케줄러 (유튜브 정리 채널 → 업데이트 사건).

모든 처리 단계에서 같은 규칙을 쓴다:  다음 한 단위의 한계효용 > 그 단위의 비용  일 때만 처리.

  영상 추출   새 영상 하나를 LLM으로 읽을까?  가치 = 그 영상이 사건에 대한 불확실성을 얼마나 줄일지
  원문 검증   공식 발표와 대조할까?           가치 = 남은 불확실성 전부 (검증은 정답을 알려줌)
  채널 구독   채널 하나를 더 볼까?            가치 = 그 채널만이 주는 추가 정보

불확실성은 브라이어 위험  v · p(1-p)  로 잰다 (v = 사건의 중요도, p = 사건이 사실일 확률).
같은 사건에 대한 영상이 쌓일수록 p 가 0 또는 1 로 가까워져 다음 영상의 가치가 줄어든다 → 한계효용 체감.
다른 채널을 베끼는 채널의 영상은 새 정보가 없으므로 한계효용이 0 이다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PRIOR_TRUE = 0.8          # 정리 채널이 다루는 업데이트가 사실일 사전 확률 (나머지는 루머·오보)
EXTRACT_USD = 0.05        # 영상 1개 LLM 추출 비용 (설명란 + 챕터)
VERIFY_USD = 0.40         # 원문 대조 비용
VERIFY_SUCCESS = 0.8      # 원문 링크가 있어 검증이 가능한 비율


# ── 한계효용 계산 ────────────────────────────────────────────────────────
def risk(p, v):
    """브라이어 위험: 지금 믿음 p 로 판단했을 때의 기대 손실."""
    return v * p * (1 - p)


def mu_report(p: float, r: float, v: float) -> float:
    """신뢰도 r 인 채널의 보고 1건을 더 읽었을 때 줄어드는 기대 손실 (정보의 기대가치)."""
    q = p * r + (1 - p) * (1 - r)                       # '사실이다'라고 보고할 확률
    pt, pf = p * r / q, p * (1 - r) / (1 - q)
    return risk(p, v) - (q * risk(pt, v) + (1 - q) * risk(pf, v))


def mu_discovery(r: float, v: float, prior: float = PRIOR_TRUE) -> float:
    """처음 보는 사건의 첫 보고: 모르면 p=0 (사실이면 손실 v), 읽으면 p=사후확률."""
    q = prior * r + (1 - prior) * (1 - r)
    pt, pf = prior * r / q, prior * (1 - r) / (1 - q)
    return v * prior - (q * risk(pt, v) + (1 - q) * risk(pf, v))


def mu_verify(p: float, v: float) -> float:
    return VERIFY_SUCCESS * risk(p, v)


def update(p: float, r: float, says_true: bool) -> float:
    lr = r / (1 - r) if says_true else (1 - r) / r
    o = p / (1 - p) * lr
    return o / (1 + o)


# ── 가상 피드 ────────────────────────────────────────────────────────────
def simulate_feed(seed: int = 0, n_events: int = 300, days: int = 60, n_channels: int = 12):
    rng = np.random.default_rng(seed)
    ev = pd.DataFrame(dict(
        event=np.arange(n_events), day=rng.uniform(0, days, n_events),
        true=rng.random(n_events) < PRIOR_TRUE,
        relevant=rng.random(n_events) < 0.5,                       # 관심 주제인가
        value=np.exp(rng.normal(np.log(1.0), 1.0, n_events)),     # 중요도 (USD, 중앙값 1)
    ))
    ch = pd.DataFrame(dict(
        channel=[f"ch{c:02d}" for c in range(n_channels)],
        reliability=rng.uniform(0.62, 0.95, n_channels),
        coverage=rng.uniform(0.15, 0.55, n_channels),
        lag=rng.uniform(0.2, 3.0, n_channels),
        copies=None,
    ))
    for c, src in ((9, 0), (10, 1), (11, 0)):                   # 베끼는 채널 3개
        ch.loc[c, "copies"] = ch.channel[src]
    ch["cluster"] = ch.copies.fillna(ch.channel)

    vids = []
    for e in ev.itertuples():
        said = {}
        for c in ch.itertuples():
            if c.copies is not None:
                if c.copies not in said or rng.random() > 0.8:
                    continue
                s = said[c.copies]                                   # 원본의 판단을 그대로 복제
            else:
                if rng.random() > c.coverage * (1.0 if e.relevant else 0.8):
                    continue
                s = (rng.random() < c.reliability) == e.true         # 사실이면 r 확률로 '사실'이라고 말함
                said[c.channel] = s
            # 제목으로 미리 알 수 있는 것: 관련성(부정확), 중요도(부정확)
            vids.append(dict(event=e.event, channel=c.channel, cluster=c.cluster, reliability=c.reliability,
                             day=e.day + c.lag * rng.uniform(0.5, 1.5) + (0.5 if c.copies else 0),
                             says_true=bool(s),
                             p_relevant=float(np.clip((0.75 if e.relevant else 0.3) + rng.normal(0, 0.1), 0.02, 0.98)),
                             value_hint=float(e.value * np.exp(rng.normal(0, 0.3)))))
    vids = pd.DataFrame(vids).sort_values("day").reset_index(drop=True)
    return ev, ch, vids


# ── 정책 실행 ────────────────────────────────────────────────────────────
@dataclass
class Policy:
    name: str
    extract: str = "all"          # all | first_k | marginal
    k: int = 2
    verify: str = "none"          # none | all | marginal
    copy_aware: bool = True       # 같은 클러스터(복제 채널)의 보고를 하나로 볼 것인가


POLICIES = [
    Policy("전부 추출 + 전부 검증", extract="all", verify="all", copy_aware=False),
    Policy("전부 추출, 검증 없음", extract="all", copy_aware=False),
    Policy("사건당 앞 2개만 추출", extract="first_k", k=2, copy_aware=False),
    Policy("한계효용 기준 (복제 미인식)", extract="marginal", verify="marginal", copy_aware=False),
    Policy("한계효용 기준", extract="marginal", verify="marginal", copy_aware=True),
]


def run(ev: pd.DataFrame, vids: pd.DataFrame, pol: Policy, channels: set | None = None, log_mu: bool = False):
    p: dict[int, float] = {}               # 발견한 사건의 믿음
    relevant: dict[int, bool] = {}
    seen_clusters: dict[int, set] = {}
    verified: set = set(); tried: set = set()
    n_seen: dict[int, int] = {}
    cost_x = cost_v = 0.0; log = []
    truth = ev.set_index("event")

    for v in vids.itertuples():
        if channels is not None and v.channel not in channels:
            continue
        e = v.event
        k = n_seen.get(e, 0)
        dup = v.cluster in seen_clusters.get(e, set())
        if e in relevant and not relevant[e]:
            mu = 0.0                                         # 관심 밖으로 확인된 사건
        elif e in verified or (dup and pol.copy_aware):
            mu = 0.0                                         # 이미 답을 알거나, 같은 출처의 복제
        elif e not in p:
            mu = v.p_relevant * mu_discovery(v.reliability, v.value_hint)
        else:
            mu = mu_report(p[e], v.reliability, v.value_hint)

        take = (pol.extract == "all" or (pol.extract == "first_k" and k < pol.k)
                or (pol.extract == "marginal" and mu > EXTRACT_USD))
        if log_mu:
            log.append(dict(event=e, k=k + 1, dup=dup, mu=mu, value=v.value_hint))
        if not take:
            continue
        cost_x += EXTRACT_USD
        n_seen[e] = k + 1
        relevant[e] = bool(truth.relevant[e])               # LLM이 읽으면 관심 주제인지 안다
        if not relevant[e] or e in verified:
            continue
        if not (dup and pol.copy_aware):
            p[e] = update(p.get(e, PRIOR_TRUE), v.reliability, v.says_true)
        seen_clusters.setdefault(e, set()).add(v.cluster)
        if e not in tried and (pol.verify == "all" or (pol.verify == "marginal" and mu_verify(p[e], truth.value[e]) > VERIFY_USD)):
            tried.add(e); cost_v += VERIFY_USD
            if np.random.default_rng(e).random() < VERIFY_SUCCESS:
                verified.add(e); p[e] = float(truth.true[e])

    rel = truth[truth.relevant]
    pf = np.array([p.get(e, 0.0) for e in rel.index])
    loss = float((rel.value * (pf - rel.true.astype(float)) ** 2).sum())
    out = dict(policy=pol.name, extracted=sum(n_seen.values()), extract_cost=cost_x, verify_cost=cost_v,
               decision_loss=loss, total=cost_x + cost_v + loss,
               missed=int(((pf == 0) & rel.true).sum()))
    return out, pd.DataFrame(log)


def greedy_channels(ev, ch, vids, pol: Policy, monitor_usd: float = 0.0):
    """채널을 하나씩 추가할 때 늘어나는 순가치 (탐욕적 선택). 순가치 = 아무것도 안 볼 때 손실 - 총비용."""
    rel = ev[ev.relevant]
    base = float((rel.value * rel.true).sum())               # 아무 채널도 안 보면 모든 사실 사건을 놓침
    chosen, rows, left = [], [], list(ch.channel)
    while left:
        best = None
        for c in left:
            r, _ = run(ev, vids, pol, set(chosen + [c]))
            net = base - r["total"] - monitor_usd * (len(chosen) + 1)
            if best is None or net > best[1]:
                best = (c, net)
        chosen.append(best[0]); left.remove(best[0])
        rows.append(dict(n=len(chosen), channel=best[0], net_value=best[1]))
    out = pd.DataFrame(rows)
    out["marginal"] = out.net_value.diff().fillna(out.net_value)
    return out
