"""점진 개선 실험대: 규칙을 하나씩 바꾸고, 검증 구간에서 우연 이상으로 좋아질 때만 채택한다.

절차
  1. 기본 규칙(파라미터)으로 검증 손실을 잰다
  2. 파라미터 하나를 이웃 값으로 바꿔 본다 (한 번에 하나)
  3. 검증 구간에서 '항목별 손실 차이'의 부트스트랩 95% 구간이 모두 0보다 작으면(= 확실히 개선) 채택
  4. 채택이 없을 때, 또는 개선 폭이 기준(min_gain)보다 작을 때 멈춤 → 한계효용 규칙
  5. 최종 시험 구간은 끝에서 단 한 번만 연다 (기본 규칙 vs 최종 규칙)

모델은 evaluate(params, split) → 항목별 손실 배열(낮을수록 좋음)만 제공하면 된다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class Spec:
    name: str
    grid: dict[str, list]                       # 파라미터 → 후보 값 (순서 있음)
    start: dict                                 # 시작 규칙
    evaluate: Callable[[dict, str], np.ndarray]  # (규칙, "val"|"test") → 항목별 손실
    metric: str = "손실"


def paired_ci(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """b − a 의 평균과 부트스트랩 95% 구간 (항목 짝지음)."""
    d = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def hill_climb(spec: Spec, max_steps: int = 20, min_gain: float = 0.0, log: Callable | None = None, mode: str = "certain") -> dict:
    """mode
    certain  개선이 가장 확실한 후보(개선 95% 구간 하한이 가장 큰 것)를 고르고, 하한 > 0 일 때만 채택 (기본)
    best     평균 개선이 가장 큰 후보를 고르고, 하한 > 0 일 때만 채택 (1차 규칙: 불확실한 1등에 막혀 일찍 멈출 수 있음)
    naive    평균 개선이 가장 큰 후보를 고르고, 조금이라도 좋아지면 채택 (흔한 방식, 비교용)"""
    cur = dict(spec.start)
    cur_loss = spec.evaluate(cur, "val")
    history = [dict(step=0, change="시작", params=dict(cur), val=float(cur_loss.mean()), gain=0.0, lo=np.nan, hi=np.nan, accepted=True)]
    tried = 0
    for step in range(1, max_steps + 1):
        best = None
        for p, vals in spec.grid.items():
            i = vals.index(cur[p])
            for j in (i - 1, i + 1):
                if not 0 <= j < len(vals):
                    continue
                cand = {**cur, p: vals[j]}
                loss = spec.evaluate(cand, "val"); tried += 1
                mean, lo, hi = paired_ci(cur_loss, loss)
                rec = dict(step=step, change=f"{p}: {cur[p]} → {vals[j]}", params=cand, val=float(loss.mean()), gain=-mean, lo=-hi, hi=-lo, accepted=False)
                key = hi if mode == "certain" else mean               # certain: 손실 차이 상한이 가장 작은 것 = 개선 하한이 가장 큰 것
                if best is None or key < best[4]:
                    best = (mean, hi, rec, loss, key)
        if best is None:
            break
        mean, hi, rec, loss, _ = best
        if (mean < 0 if mode == "naive" else hi < 0) and -mean > min_gain:                          # 95% 구간 전체가 개선 쪽 + 최소 개선 폭
            rec["accepted"] = True
            history.append(rec)
            cur, cur_loss = rec["params"], loss
            if log:
                log(f"  채택 {rec['change']}  검증 {rec['val']:.4f} (개선 {rec['gain']:.4f}, 95% {rec['lo']:.4f}~{rec['hi']:.4f})")
        else:
            history.append(rec)
            if log:
                log(f"  멈춤: 최선 후보 {rec['change']} 개선 {rec['gain']:.4f} (95% {rec['lo']:.4f}~{rec['hi']:.4f}) → 우연과 구분 안 됨")
            break
    start_test = spec.evaluate(spec.start, "test")
    final_test = spec.evaluate(cur, "test")
    m, lo, hi = paired_ci(start_test, final_test)
    return dict(final=cur, history=pd.DataFrame(history), tried=tried,
                test_start=float(start_test.mean()), test_final=float(final_test.mean()), test_gain=-m, test_lo=-hi, test_hi=-lo,
                n_test=len(final_test))
