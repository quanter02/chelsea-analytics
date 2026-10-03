"""단계적 호출 정책(라우터) 탐색.

정책 = [(모델, 확신 기준), ...] 순서. 각 단계에서
  기권이 아니고, 확신 ≥ 기준이고, (require_evidence면) 근거 검증 통과 → 그 답을 채택
  아니면 다음 단계로. 마지막까지 채택되지 않으면 사람 검토.
비용 = 호출한 모델 비용 합 + (사람 검토 비용) + (채택한 답이 틀렸을 때의 피해).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

TAUS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


@dataclass(frozen=True)
class Policy:
    steps: tuple[tuple[str, float], ...]
    require_evidence: bool = False

    def __str__(self) -> str:
        s = " → ".join(f"{m}(≥{t:.2f})" for m, t in self.steps) or "(모델 없음)"
        return s + (" +근거검증" if self.require_evidence else "") + " → 사람"


def latest_versions(scored: pd.DataFrame) -> pd.DataFrame:
    """모델마다 가장 최근 버전 데이터만 사용. 이전 버전의 성적으로 현재 라우팅을 정하지 않는다."""
    last = scored.sort_values("created_at").groupby("model_id").model_version.last()
    return scored[scored.model_version == scored.model_id.map(last)]


def wide(scored: pd.DataFrame, weights: pd.Series | None = None) -> dict:
    """문항 × 모델 배열로 변환. 어떤 모델이 답하지 않은 문항은 그 모델 칸이 비어 있다.
    weights: 문항별 가중치 (온라인 모드의 망각 계수). 없으면 모두 1."""
    d = latest_versions(scored)
    models = sorted(d.model_id.unique())
    piv = {c: d.pivot_table(index="input_hash", columns="model_id", values=c, aggfunc="first").reindex(columns=models)
           for c in ("answer", "confidence", "evidence_ok", "cost_usd", "loss_usd")}
    keep = piv["answer"].index
    truth = d.drop_duplicates("input_hash").set_index("input_hash").true_label.reindex(keep)
    when = d.groupby("input_hash").created_at.min().reindex(keep)
    w = np.ones(len(keep)) if weights is None else weights.reindex(keep).fillna(0).to_numpy()
    return dict(models=models, truth=truth.to_numpy(), when=when.to_numpy(), index=keep, weight=w,
                loss=piv.pop("loss_usd"), **piv)


def evaluate(W: dict, policy: Policy, loss_lookup: dict, human_usd: float, mask=None, per_input: bool = False) -> dict:
    """정책의 문항당 평균 비용. 정책에 들어간 모델이 모두 답한 문항만 평가한다 (가중 평균)."""
    n = len(W["truth"]); mask = np.ones(n, bool) if mask is None else mask.copy()
    for m, _ in policy.steps:
        if m not in W["answer"].columns:                         # 이 기간에 답이 없는 모델
            mask[:] = False
            break
        mask &= W["answer"][m].notna().to_numpy()
    w = W.get("weight", np.ones(n))[mask]
    truth = W["truth"][mask]
    decided = np.zeros(mask.sum(), bool); call = np.zeros(mask.sum()); err = np.zeros(mask.sum())
    for m, tau in policy.steps:
        ans = W["answer"][m].to_numpy()[mask]; conf = W["confidence"][m].fillna(-1).to_numpy()[mask]
        ok = W["evidence_ok"][m].fillna(0).to_numpy()[mask] == 1
        call += np.where(~decided, W["cost_usd"][m].to_numpy()[mask], 0)
        take = ~decided & (ans != "ABSTAIN") & (conf >= tau) & (ok | (not policy.require_evidence))
        if "loss" in W:
            loss = W["loss"][m].fillna(0).to_numpy()[mask]
        else:
            loss = np.array([loss_lookup.get((t, a), 0.0) for t, a in zip(truth, ans)])
        err += np.where(take, loss, 0)
        decided |= take
    human = np.where(~decided, human_usd, 0.0)
    total = call + err + human
    avg = (lambda x: np.average(x, weights=w)) if w.sum() > 0 else (lambda x: np.nan)
    extra = {"per_input": pd.Series(total, index=np.asarray(W["index"])[mask] if W.get("index") is not None else None)} if per_input else {}
    return extra | dict(policy=str(policy), steps=len(policy.steps), n_eval=int(mask.sum()), weight=float(w.sum()),
                total=avg(total), error_loss=avg(err), human_cost=avg(human), call_cost=avg(call),
                human_share=avg(~decided), wrong_share=avg(err > 0))


def candidates(models: list[str], max_steps: int = 2):
    yield Policy(())                                                              # 전부 사람
    for k in range(1, max_steps + 1):
        for chain in itertools.permutations(models, k):
            for taus in itertools.product(TAUS, repeat=k):
                for ev in (False, True):
                    yield Policy(tuple(zip(chain, taus)), ev)


def search(scored: pd.DataFrame, loss_matrix: pd.DataFrame, human_usd: float, train_share: float = 0.7, max_steps: int = 2):
    """시간 순으로 앞부분에서 정책을 고르고, 뒷부분에서 성능을 확인 (미래 데이터 누설 방지)."""
    W = wide(scored)
    full = W["answer"].notna().all(axis=1).to_numpy()            # 모든 모델이 답한 문항만 (정책 간 공정 비교)
    W = {k: (v[full] if isinstance(v, (np.ndarray, pd.Index)) else v.loc[full] if isinstance(v, pd.DataFrame) else v) for k, v in W.items()}
    lookup = {(r.true_label, r.answer): r.loss_usd for r in loss_matrix.itertuples() if r.answer != "ABSTAIN"}
    cut = np.quantile(pd.to_datetime(W["when"]).astype("int64"), train_share)
    train = pd.to_datetime(W["when"]).astype("int64").to_numpy() <= cut
    res = pd.DataFrame([evaluate(W, p, lookup, human_usd, train) | {"_p": p} for p in candidates(W["models"], max_steps)])
    res = res.sort_values("total").reset_index(drop=True)
    picks = {"최적 정책 (학습 구간에서 선택)": res.iloc[0]._p,
             "모두 사람 검토": Policy(()),
             **{f"{m} 단독, 전부 채택": Policy(((m, 0.0),)) for m in W["models"]}}
    best_single = res[res.steps == 1].iloc[0]._p
    picks["최적 단일 모델 정책"] = best_single
    test = pd.DataFrame([{"정책 구분": k} | evaluate(W, p, lookup, human_usd, ~train) for k, p in picks.items()])
    return res.drop(columns="_p"), test.sort_values("total").reset_index(drop=True), dict(n_train=int(train.sum()), n_test=int((~train).sum()))
