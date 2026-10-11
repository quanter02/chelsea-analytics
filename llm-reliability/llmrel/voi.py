"""빠른 실패의 '의미'를 계산한다 (정보의 가치, value of information).

원칙: 실패 기록은 하나도 지우지 않는다(장부). 대신 실험마다 '얼마나 배웠나'를 점수로 매겨 우선순위를 정한다.
  - 정보량(bit): 실험 전후로 가정 세계에 대한 불확실성(엔트로피)이 얼마나 줄었나
  - 결정 변화 확률: 그 결과 때문에 우리가 고를 행동이 바뀔 확률
  - 점수: 결정 변화 확률 ÷ 걸리는 주 수  (같으면 정보량 ÷ 주 수)

결제(L3)까지 기다리면 세계 하나를 지우는 데 1~2년이 걸린다(biz_update.py). 그래서 앞 단계 신호를 쓴다:
  리포트   세계 = 회신율 r × 회신→결제 전환율 c.  월 고객 확률 lam = 월 접촉 6곳(전력) × r × c
           결정 = 바닥 먼저 독립(C2) / 직장 유지·고객 3곳에서 독립 / 직장 유지·고객 5곳에서 독립
           효용 = P(10년 자산 10억+) − P(파산)   (strategy_sim으로 세계마다 계산)
  노트     세계 = 무료 구독자 → 유료 사전 예약률 s.  결정 = 유료판 출시 / 보류.  효용 = s − 0.10 (출시 시)
  검증배지 세계 = 배지를 본 예측자 중 붙여 보는 비율 a.  결정 = 개발 계속 / 보류.  효용 = a − 0.10 (계속 시)
  혼인수요 세계 = 업체 회신율 r3 × 전환율 c3.     결정 = 직접 영업 / 바우처 경로만.  효용 = r3·c3 − 0.03 (직접 영업 시)
기준값(0.10, 0.03)과 격자는 사전등록 가정이다. 바꾸면 장부에 새 줄로 남긴다.

    python -m llmrel.voi            # → output/strategy/voi.md
"""
from __future__ import annotations

import itertools
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output" / "strategy"
CONTACTS_FULL = 6.0                                     # 리포트에 전력할 때 월 접촉 수 (가정)
FIXED = {"p": 0.08, "M": 300_000, "half": 0.5, "W0": 2000.0}   # 리포트 외 축은 중간값으로 고정
REPORT_ACTIONS = {"바닥 먼저 독립": ("C2", {}), "직장 유지·3곳에서 독립": ("H", {"K": 9, "quit": 1.0, "eve": 0.5}),
                  "직장 유지·5곳에서 독립": ("H", {"K": 9, "quit": 1.5, "eve": 0.5})}


def _H(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


@lru_cache(maxsize=None)
def report_utility(r: float, c: float, n: int = 1500) -> tuple:
    import strategy_sim as S
    w = {**FIXED, "lam": CONTACTS_FULL * r * c}
    out = []
    for a, (s, extra) in REPORT_ACTIONS.items():
        res = S.run(s, {**w, **extra}, n=n, seed=int(r * 100) * 7 + int(c * 100))
        out.append(res["10억+"] - res["ruin"])
    return tuple(out)


def streams() -> dict:
    """수익원별 세계, 사전확률(균등), 행동별 효용표."""
    rep = list(itertools.product([0.1, 0.3, 0.5], [0.1, 0.25, 0.5]))
    note = [0.02, 0.05, 0.10, 0.20]
    dem = list(itertools.product([0.05, 0.2, 0.4], [0.1, 0.3]))
    led = [0.02, 0.05, 0.10, 0.25]                       # 배지를 본 예측자 중 실제로 붙여 보는 비율
    return {
        "report": {"worlds": rep, "actions": list(REPORT_ACTIONS), "U": np.array([report_utility(r, c) for r, c in rep]),
                   "rate": lambda w: w[0], "rate_conv": lambda w: w[0] * w[1]},
        "note": {"worlds": note, "actions": ["유료판 출시", "보류"], "U": np.array([[s - 0.10, 0.0] for s in note]),
                 "rate": lambda w: w},
        "ledger": {"worlds": led, "actions": ["배지 개발 계속", "보류"], "U": np.array([[a - 0.10, 0.0] for a in led]),
                   "rate": lambda w: w},
        "demand": {"worlds": dem, "actions": ["직접 영업", "바우처 경로만"], "U": np.array([[r * c - 0.03, 0.0] for r, c in dem]),
                   "rate": lambda w: w[0], "rate_conv": lambda w: w[0] * w[1]},
    }


EXPS = {  # 실험 = (수익원, 시도 수, 걸리는 주, 관측) — 관측 'rate'는 회신 수, 'rate_conv'는 가격을 보고도 남는 관심(회신×전환)
    "E1": ("report", 3, 2, "rate"), "E1+ (6곳)": ("report", 6, 3, "rate"), "E1++ (10곳)": ("report", 10, 4, "rate"),
    "E1-가격 (6곳)": ("report", 6, 4, "rate_conv"), "E1-가격 (10곳)": ("report", 10, 6, "rate_conv"),
    "E2": ("note", 30, 8, "rate"), "E3": ("demand", 3, 3, "rate"), "E3+ (6곳)": ("demand", 6, 4, "rate"),
    "E3-가격 (6곳)": ("demand", 6, 5, "rate_conv"),
    "E4": ("ledger", 10, 4, "rate"),
}


def posterior(st: dict, n: int, y: int, prior: np.ndarray | None = None, obs: str = "rate") -> np.ndarray:
    k = len(st["worlds"])
    prior = np.full(k, 1 / k) if prior is None else prior
    like = np.array([binom.pmf(y, n, st[obs](w)) for w in st["worlds"]])
    post = prior * like
    return post / post.sum()


def evaluate(exp: str, S: dict | None = None) -> dict:
    """예상 정보량, 결정 변화 확률, 사전 결정, 결과별 사후 결정."""
    S = S or streams()
    stream, n, weeks, obs = EXPS[exp]
    st = S[stream]
    k = len(st["worlds"])
    prior = np.full(k, 1 / k)
    a0 = int(np.argmax(prior @ st["U"]))
    h0, eh, p_change, evsi = _H(prior), 0.0, 0.0, 0.0
    by_y = []
    for y in range(n + 1):
        py = float(sum(prior[i] * binom.pmf(y, n, st[obs](w)) for i, w in enumerate(st["worlds"])))
        if py < 1e-12:
            continue
        post = posterior(st, n, y, prior, obs)
        a = int(np.argmax(post @ st["U"]))
        eh += py * _H(post)
        p_change += py * (a != a0)
        evsi += py * (post @ st["U"])[a]
        by_y.append((y, py, st["actions"][a]))
    evsi -= (prior @ st["U"])[a0]
    return {"exp": exp, "stream": stream, "n": n, "weeks": weeks, "prior_action": st["actions"][a0],
            "bits": h0 - eh, "p_change": p_change, "evsi": evsi, "score": p_change / weeks, "bits_per_week": (h0 - eh) / weeks, "by_y": by_y}


def realized(exp: str, y: int, S: dict | None = None) -> dict:
    """실험이 끝난 뒤: 실제 결과 y로 줄어든 불확실성과 결정 변화."""
    S = S or streams()
    stream, n, _, obs = EXPS[exp]
    st = S[stream]
    k = len(st["worlds"])
    prior = np.full(k, 1 / k)
    post = posterior(st, n, y, prior, obs)
    a0, a1 = int(np.argmax(prior @ st["U"])), int(np.argmax(post @ st["U"]))
    return {"bits": _H(prior) - _H(post), "decision_before": st["actions"][a0], "decision_after": st["actions"][a1], "changed": a0 != a1}


def ranking() -> pd.DataFrame:
    S = streams()
    rows = [evaluate(e, S) for e in EXPS]
    return pd.DataFrame(rows).sort_values(["score", "bits_per_week"], ascending=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = ranking()
    L = ["# 실험 우선순위: 어떤 빠른 실패가 가장 많이 가르쳐 주나", "",
         "실패 기록은 지우지 않습니다. 대신 실험마다 **결정이 바뀔 확률**과 **정보량**을 미리 계산해, 같은 시간에 가장 많이 배우는 실험부터 합니다.", "",
         "| 순위 | 실험 | 시도 | 주 | 지금 결정 | 결정 변화 확률 | 정보량(bit) | 주당 결정 변화 |", "|---|---|---|---|---|---|---|---|"]
    for i, (_, r) in enumerate(t.iterrows(), 1):
        L.append(f"| {i} | {r.exp} | {r.n} | {r.weeks} | {r.prior_action} | {r.p_change * 100:.0f}% | {r.bits:.2f} | {r.score * 100:.1f}%p |")
    L += ["", "## 결과별로 바뀌는 결정", ""]
    for _, r in t.iterrows():
        if r.exp in ("E1", "E1-가격 (6곳)", "E2", "E3", "E4"):
            L.append(f"- **{r.exp}** (지금: {r.prior_action}): " + " · ".join(f"{y}건 → {a} ({py * 100:.0f}%)" for y, py, a in r.by_y if py >= 0.01))
    L += ["", "## 가정", "",
          f"- 리포트: 전력 시 월 {CONTACTS_FULL:.0f}곳 접촉, 회신율 10/30/50% × 회신→결제 10/25/50%. 효용 = 10년 자산 10억+ 확률 − 파산 확률 (strategy_sim, 시작 자금 2,000만원)",
          "- 노트: 구독자→사전 예약률 2/5/10/20%. 10% 이상이면 출시가 이득",
          "- 검증 배지: 배지를 본 예측자 중 실제로 붙여 보는 비율 2/5/10/25%. 10% 이상이면 배지 개발 계속이 이득",
          "- 혼인 수요: 업체 회신율 5/20/40% × 전환율 10/30%. 회신×전환이 3% 이상이면 직접 영업이 이득",
          "- '-가격' 실험: 가격(A/B/C안)을 보여주고 관심 확인서(L2)를 받는 수를 셉니다. 회신이 아니라 '가격을 알고도 남는 관심'을 재므로 결제 전환을 직접 가늠합니다.",
          "- 모든 세계의 사전확률은 같게 둡니다. 결과가 나오면 사후확률이 다음 실험의 사전확률이 됩니다."]
    (OUT / "voi.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return t


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
