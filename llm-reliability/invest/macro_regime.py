"""거시 시나리오 확률을 FRED 자료로 계산 (투자 시뮬레이션의 1년차 시나리오 가중치).

    python invest/macro_regime.py      # FRED_API_KEY, api.stlouisfed.org 허용 필요 → invest/out/macro_regime.md

시나리오: ① 긴축 장기화 ② AI 사이클 둔화 ③ 연착륙 ④ 위기(12개월 안 경기침체)
- ④ = 로지스틱 회귀: 앞으로 12개월 안에 미국 경기침체(USREC)가 시작/진행될 확률 ~ 10년-3개월 금리차 + 회사채(Baa) 스프레드.
       1986-01~(최근-12개월) 월별로 적합. 침체가 4번(1990, 2001, 2008, 2020)뿐이라 과적합 위험 → NY연준 금리차 프로빗과 평균.
- ① vs ③ = 2년 국채금리 − 정책금리: 시장이 인상/동결(+)을 보면 긴축, 인하(−)를 보면 연착륙 쪽. 0.5%p당 한 단계(tanh).
- ② = 거시 자료로 알 수 없는 AI 투자 사이클 → 판단값 25% 유지 (이전 모델과 같음).
개인 계좌 정보는 넣지 않는다.
"""
from __future__ import annotations

import json
import os
import urllib.request
from math import erf, sqrt, tanh
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "out"
AI_SLOWDOWN = 0.25


def fred(sid: str, freq: str | None = "m") -> pd.Series:
    u = (f"https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={os.environ['FRED_API_KEY']}"
         f"&file_type=json" + (f"&frequency={freq}&aggregation_method=avg" if freq else ""))
    obs = json.loads(urllib.request.urlopen(u, timeout=60).read())["observations"]
    s = pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in obs if o["value"] not in (".", "")})
    return s.sort_index()


def logit_fit(X: np.ndarray, y: np.ndarray, iters: int = 50, l2: float = 1e-3) -> np.ndarray:
    X = np.c_[np.ones(len(X)), X]
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(len(b))
        b += np.linalg.solve(H, X.T @ (y - p) - l2 * b)
    return b


def main():
    rec, spread, hy = fred("USREC"), fred("T10Y3M"), fred("BAA10Y")   # 하이일드(ICE)는 FRED에 최근 3년만 있어 Baa−10년 사용
    d = pd.concat({"rec": rec, "spread": spread, "hy": hy}, axis=1).dropna()
    d = d[d.index >= "1986-01-01"]
    d["y"] = d.rec[::-1].rolling(12, min_periods=12).max()[::-1].shift(-1)   # 다음 12개월 중 침체 달이 있나
    fit = d.dropna(subset=["y"])
    b = logit_fit(fit[["spread", "hy"]].to_numpy(), fit.y.to_numpy())
    now_s, now_hy = fred("T10Y3M", None).iloc[-1], fred("BAA10Y", None).iloc[-1]
    p_logit = float(1 / (1 + np.exp(-(b[0] + b[1] * now_s + b[2] * now_hy))))
    p_probit = 0.5 * (1 + erf((-0.5333 - 0.6330 * now_s) / sqrt(2)))           # NY 연준 금리차 모형 계수
    p_crisis = (p_logit + p_probit) / 2
    # 적합도: 과거 각 달의 예측 확률 평균 (침체 직전 vs 평상시)
    p_hist = 1 / (1 + np.exp(-(b[0] + fit[["spread", "hy"]].to_numpy() @ b[1:])))
    sep = (p_hist[fit.y == 1].mean(), p_hist[fit.y == 0].mean())

    dgs2, dff = fred("DGS2", None).iloc[-1], fred("DFF", None).iloc[-1]
    tight_share = 0.5 + 0.4 * tanh((dgs2 - dff) / 0.5)
    rest = 1 - p_crisis - AI_SLOWDOWN
    probs = {"긴축 장기화": rest * tight_share, "AI 사이클 둔화": AI_SLOWDOWN, "연착륙": rest * (1 - tight_share), "위기": p_crisis}

    OUT.mkdir(exist_ok=True)
    (OUT / "macro_regime.json").write_text(json.dumps({k: round(v, 3) for k, v in probs.items()}, ensure_ascii=False))
    (OUT / "macro_regime.md").write_text(f"""# 거시 시나리오 확률 (FRED, {pd.Timestamp.today().date()})

| 시나리오 | 이전 (판단) | 지금 (자료) |
|---|---|---|
| 긴축 장기화 | 40% | {probs['긴축 장기화']:.0%} |
| AI 사이클 둔화 | 25% | {probs['AI 사이클 둔화']:.0%} (판단값 유지) |
| 연착륙 | 25% | {probs['연착륙']:.0%} |
| 위기 (12개월 안 침체) | 10% | {probs['위기']:.0%} |

- 위기: 로지스틱(금리차 {now_s:.2f}%p, Baa 스프레드 {now_hy:.2f}%p) {p_logit:.1%} · NY 연준 프로빗 {p_probit:.1%} → 평균 {p_crisis:.1%}
  - 로지스틱 계수: 상수 {b[0]:+.2f}, 금리차 {b[1]:+.2f}, Baa 스프레드 {b[2]:+.2f} ({fit.index[0]:%Y-%m}~{fit.index[-1]:%Y-%m}, {len(fit)}개월, 침체 {int((fit.rec.diff() == 1).sum())}번)
  - 과거 적합: 침체 직전 12개월 평균 확률 {sep[0]:.0%} vs 평상시 {sep[1]:.0%}
- 긴축 vs 연착륙: 2년 국채 {dgs2:.2f}% − 정책금리 {dff:.2f}% = {dgs2 - dff:+.2f}%p → 긴축 쪽 비중 {tight_share:.0%}
""", encoding="utf-8")
    print((OUT / "macro_regime.md").read_text())


if __name__ == "__main__":
    main()
