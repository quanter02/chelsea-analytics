"""모델(× 버전 × 업무)별 신뢰도 프로파일: 오답 구성, 확신도 보정, 기권의 타당성, 오류 상관, 기대 손실."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import db

KEY = ["model_id", "model_version", "task_type"]


def load(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(전체 예측, 정답이 붙은 예측) 반환."""
    preds = db.read(con, "SELECT * FROM predictions")
    scored = db.read(con, "SELECT * FROM scored")
    for d in (preds, scored):
        d["model"] = d.model_id + "@" + d.model_version
    return preds, scored


def eb_shrink(k: np.ndarray, n: np.ndarray) -> np.ndarray:
    """경험적 베이즈(베타-이항) 축소: 표본이 적은 모델의 비율을 전체 평균 쪽으로 당긴다."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / np.maximum(n, 1)
    m = k.sum() / n.sum()
    v = max(np.average((p - m) ** 2, weights=n) - m * (1 - m) / n.mean(), 1e-6)   # 모델 간 실제 분산
    strength = max(m * (1 - m) / v - 1, 0.5)                                         # α + β
    return (k + m * strength) / (n + strength)


def outcome_profile(preds: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """정답·확신 오답·기권 비율 (+ 축소 추정치, 정답 대기 비율)."""
    g = scored.groupby(KEY + ["model"])
    t = g.outcome.value_counts().unstack(fill_value=0).reindex(columns=["correct", "confident_error", "abstain"], fill_value=0)
    t["n_labeled"] = t.sum(axis=1)
    t["n_total"] = preds.groupby(KEY + ["model"]).size().reindex(t.index)
    t["pending_share"] = 1 - t.n_labeled / t.n_total
    for c in ("correct", "confident_error", "abstain"):
        t[f"{c}_rate"] = t[c] / t.n_labeled
    t["confident_error_rate_eb"] = eb_shrink(t.confident_error, t.n_labeled)
    committed = t.correct + t.confident_error
    t["accuracy_when_answering"] = t.correct / committed.where(committed > 0)
    return t.reset_index()


def calibration(scored: pd.DataFrame, bins=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)) -> tuple[pd.DataFrame, pd.Series]:
    """확신도 구간별 실제 정답률과 ECE(기대 보정 오차). ECE가 클수록 '말한 확신'을 믿기 어렵다."""
    c = scored[scored.answer != "ABSTAIN"].copy()
    c["hit"] = (c.outcome == "correct").astype(float)
    c["bin"] = pd.cut(c.confidence, bins, right=False)
    tab = c.groupby(["model", "bin"], observed=True).agg(n=("hit", "size"), said=("confidence", "mean"), actual=("hit", "mean")).reset_index()
    ece = tab.assign(w=lambda d: d.n * (d.said - d.actual).abs()).groupby("model").apply(
        lambda d: d.w.sum() / d.n.sum(), include_groups=False)
    return tab, ece.rename("ECE")


def abstention_quality(scored: pd.DataFrame) -> pd.DataFrame:
    """기권한 문항에서 '다른 모델들'의 정답률. 평소보다 훨씬 낮으면 기권이 타당했다는 뜻."""
    committed = scored[scored.answer != "ABSTAIN"].assign(hit=lambda d: (d.outcome == "correct").astype(float))
    rows = []
    for model, ab in scored[scored.answer == "ABSTAIN"].groupby("model"):
        others = committed[committed.model != model]
        on_abst = others[others.input_hash.isin(ab.input_hash)].hit.mean()
        rows.append(dict(model=model, n_abstain=len(ab), others_acc_on_abstained=on_abst, others_acc_overall=others.hit.mean()))
    out = pd.DataFrame(rows)
    if len(out):
        out["difficulty_ratio"] = out.others_acc_on_abstained / out.others_acc_overall
    return out


def error_correlation(scored: pd.DataFrame, min_overlap: int = 30) -> pd.DataFrame:
    """두 모델이 같은 문항에서 같이 틀리는 정도. 상관이 높으면 투표로 얻는 이득이 작다."""
    c = scored[scored.answer != "ABSTAIN"].assign(wrong=lambda d: (d.outcome == "confident_error").astype(float))
    wide = c.pivot_table(index="input_hash", columns="model", values="wrong")
    ans = c.pivot_table(index="input_hash", columns="model", values="answer", aggfunc="first")
    rows = []
    for a, b in itertools.combinations(wide.columns, 2):
        both = wide[[a, b]].dropna()
        if len(both) < min_overlap or both[a].std() == 0 or both[b].std() == 0:
            continue
        bw = (both[a] == 1) & (both[b] == 1)
        same = (ans.loc[bw[bw].index, a] == ans.loc[bw[bw].index, b]).mean() if bw.any() else np.nan
        rows.append(dict(model_a=a, model_b=b, n=len(both), error_corr=both[a].corr(both[b]),
                         both_wrong_rate=bw.mean(), same_wrong_answer=same))
    return pd.DataFrame(rows)


def expected_loss(scored: pd.DataFrame) -> pd.DataFrame:
    """호출 1회당 기대 손실(USD) = 오답 피해 + 기권 시 사람 검토 비용 + 호출 비용."""
    d = scored.assign(err_loss=lambda x: np.where(x.outcome == "confident_error", x.loss_usd, 0.0),
                      abstain_loss=lambda x: np.where(x.outcome == "abstain", x.loss_usd, 0.0))
    t = d.groupby("model").agg(n=("outcome", "size"), error_loss=("err_loss", "mean"),
                               abstain_loss=("abstain_loss", "mean"), call_cost=("cost_usd", "mean"))
    t["expected_loss"] = t.error_loss + t.abstain_loss + t.call_cost
    return t.sort_values("expected_loss").reset_index()
