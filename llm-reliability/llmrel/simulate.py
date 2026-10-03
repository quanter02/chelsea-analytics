"""가상 로그 생성기. 실제 로그가 생기기 전 파이프라인을 검증하기 위한 것이며, 모든 수치는 가정이다.

모델의 성격은 네 가지 손잡이로 정한다.
  skill      문제를 맞히는 능력 (난이도와 비교)
  abstain_t  내부 확신이 이 값보다 낮으면 기권 (0이면 절대 기권하지 않음)
  gamma      보고하는 확신도 = 내부 확신 ** gamma  (1보다 작으면 과신)
  cost       호출당 비용 (USD)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

LABELS = ("R", "M", "N")
TASK = "change_cause"
START = datetime(2026, 1, 1)

MODELS = [
    dict(model_id="small-fast",   version="v1", skill=0.2, abstain_t=0.00, gamma=0.55, cost=0.002, latency=300),
    dict(model_id="mid-bold",     version="v1", skill=0.7, abstain_t=0.00, gamma=0.50, cost=0.008, latency=900),
    dict(model_id="mid-cautious", version="v1", skill=0.6, abstain_t=0.62, gamma=1.00, cost=0.008, latency=900),
    dict(model_id="large",        version="v1", skill=1.1, abstain_t=0.50, gamma=0.90, cost=0.030, latency=2500, until_day=60),
    dict(model_id="large",        version="v2", skill=1.5, abstain_t=0.50, gamma=1.00, cost=0.030, latency=2200, from_day=60),
]

# 피해 크기 (USD). 실제 변화를 놓치는 것보다 가짜 변화에 데이터를 버리는 쪽이 더 비싸다고 가정.
LOSS = {
    ("R", "M"): 50, ("R", "N"): 40,
    ("M", "R"): 80, ("M", "N"): 40,
    ("N", "R"): 30, ("N", "M"): 10,
}
HUMAN_REVIEW_USD = 5.0          # 기권 → 사람이 확인 (사람은 맞힌다고 가정)


def loss_matrix() -> pd.DataFrame:
    rows = [(TASK, t, a, LOSS.get((t, a), 0.0)) for t in LABELS for a in LABELS]
    rows += [(TASK, t, "ABSTAIN", HUMAN_REVIEW_USD) for t in LABELS]
    return pd.DataFrame(rows, columns=["task_type", "true_label", "answer", "loss_usd"])


def simulate(n_inputs: int = 3000, days: int = 180, seed: int = 0, label_delay_days: float = 14.0):
    """(predictions, outcomes) 데이터프레임을 반환. 정답은 지연되어 붙고, 마지막 날 기준 일부는 대기 중."""
    rng = np.random.default_rng(seed)
    day = np.sort(rng.uniform(0, days, n_inputs))
    truth = rng.choice(LABELS, n_inputs, p=[0.3, 0.3, 0.4])
    difficulty = rng.normal(0, 1, n_inputs)
    # 모든 모델이 같이 끌려가는 '그럴듯한 오답' → 모델 간 오류 상관의 원천
    decoy = np.array([rng.choice([l for l in LABELS if l != t]) for t in truth])
    hashes = [hashlib.sha1(f"{seed}-{i}".encode()).hexdigest()[:16] for i in range(n_inputs)]

    preds = []
    for i in range(n_inputs):
        for m in MODELS:
            if not (m.get("from_day", 0) <= day[i] < m.get("until_day", days + 1)):
                continue
            q = 1 / 3 + (2 / 3) / (1 + np.exp(-1.5 * (m["skill"] - difficulty[i])))   # 내부 확신 = 맞힐 확률
            if q + rng.normal(0, 0.05) < m["abstain_t"]:
                ans, conf, ev_ok = "ABSTAIN", None, None
            else:
                correct = rng.random() < q
                ans = truth[i] if correct else (decoy[i] if rng.random() < 0.7 else
                                                next(l for l in LABELS if l not in (truth[i], decoy[i])))
                conf = float(np.clip(q ** m["gamma"], 0, 1))
                ev_ok = int(rng.random() < (0.9 if correct else 0.4))       # 환각 근거는 로그 대조에서 자주 걸림
            preds.append(dict(
                model_id=m["model_id"], model_version=m["version"], prompt_version="p1", task_type=TASK,
                input_hash=hashes[i], answer=ans, confidence=conf,
                evidence=json.dumps({"cites": ["deploy_log"] if ans == "M" else ["metrics"]}) if ans != "ABSTAIN" else None,
                evidence_ok=ev_ok, cost_usd=m["cost"], latency_ms=int(m["latency"] * rng.uniform(0.7, 1.4)),
                created_at=(START + timedelta(days=float(day[i]))).isoformat(timespec="seconds"),
            ))
    preds = pd.DataFrame(preds)

    labeled_day = day + rng.exponential(label_delay_days, n_inputs)
    done = labeled_day < days
    outcomes = pd.DataFrame(dict(
        input_hash=np.array(hashes)[done], task_type=TASK, true_label=truth[done],
        label_source=rng.choice(["deploy_log", "human_review", "later_data"], done.sum(), p=[0.4, 0.3, 0.3]),
        labeled_at=[(START + timedelta(days=float(d))).isoformat(timespec="seconds") for d in labeled_day[done]],
    ))
    return preds, outcomes
