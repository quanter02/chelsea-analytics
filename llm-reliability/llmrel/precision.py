"""실제 데이터에서 '얼마나 정확히 잴 수 있나': 표본 수별 부트스트랩 신뢰구간."""
from __future__ import annotations

import numpy as np
import pandas as pd

BINS = np.array([0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])


def arrays(scored: pd.DataFrame) -> tuple[list[str], dict]:
    """모델 × 문항 배열 (정답 여부, 기권 여부, 확신도, 손실)."""
    d = scored.copy()
    hashes = sorted(d.input_hash.unique())
    models = sorted(d.model.unique())
    A = {}
    for k, col in (("hit", None), ("abst", None), ("conf", "confidence"), ("loss", "loss_usd")):
        if col:
            A[k] = d.pivot_table(index="input_hash", columns="model", values=col, aggfunc="first").reindex(index=hashes, columns=models).to_numpy(float)
    A["abst"] = (d.assign(v=(d.answer == "ABSTAIN").astype(float)).pivot_table(index="input_hash", columns="model", values="v")
                 .reindex(index=hashes, columns=models).to_numpy())
    A["hit"] = (d.assign(v=(d.outcome == "correct").astype(float)).pivot_table(index="input_hash", columns="model", values="v")
                .reindex(index=hashes, columns=models).to_numpy())
    return models, A


def metrics(A: dict, idx: np.ndarray) -> dict:
    hit, ab, conf, loss = (A[k][idx] for k in ("hit", "abst", "conf", "loss"))
    ans = 1 - ab
    n_ans = ans.sum(0)
    out = {"accuracy_when_answering": hit.sum(0) / np.maximum(n_ans, 1),
           "confident_error_rate": ((1 - hit) * ans).sum(0) / len(idx),
           "expected_loss": loss.mean(0)}
    ece = []
    for j in range(hit.shape[1]):
        m = ans[:, j] == 1
        c, h = conf[m, j], hit[m, j]
        b = np.digitize(c, BINS) - 1
        cnt = np.bincount(b, minlength=len(BINS)); sc = np.bincount(b, c, len(BINS)); sh = np.bincount(b, h, len(BINS))
        nz = cnt > 0
        ece.append(np.abs(sc[nz] - sh[nz]).sum() / max(m.sum(), 1))
    out["ECE"] = np.array(ece)
    return out


def bootstrap(scored: pd.DataFrame, sizes=(100, 250, 500, 1000, None), B: int = 400, seed: int = 0, pairs=()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """표본이 N건뿐이라면 각 지표의 95% 신뢰구간 반폭은? (None = 전체)
    pairs: [(모델A, 모델B)] 기대 손실 차이를 구분할 수 있는지 (부트스트랩에서 A<B 인 비율)."""
    models, A = arrays(scored)
    n_all = A["hit"].shape[0]
    rng = np.random.default_rng(seed)
    full = metrics(A, np.arange(n_all))
    rows, prow = [], []
    for N in sizes:
        N = N or n_all
        draws = [metrics(A, rng.integers(0, n_all, N)) for _ in range(B)]
        for k in full:
            arr = np.array([d[k] for d in draws])                     # B × 모델
            lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
            for j, m in enumerate(models):
                rows.append(dict(N=N, metric=k, model=m, full_value=full[k][j], mean=arr[:, j].mean(), half_width=(hi[j] - lo[j]) / 2))
        for a, b in pairs:
            ia, ib = models.index(a), models.index(b)
            diff = np.array([d["expected_loss"][ia] - d["expected_loss"][ib] for d in draws])
            prow.append(dict(N=N, pair=f"{a} vs {b}", full_diff=full["expected_loss"][ia] - full["expected_loss"][ib],
                             half_width=(np.percentile(diff, 97.5) - np.percentile(diff, 2.5)) / 2,
                             same_order=float(np.mean(np.sign(diff) == np.sign(full["expected_loss"][ia] - full["expected_loss"][ib])))))
    return pd.DataFrame(rows), pd.DataFrame(prow)
