"""예측 범위: 1~6월 신호 예측에 '80% 범위'를 붙인다.

범위는 그해 8월 말 이전에 채점이 끝난 과거 오차(작년까지)로만 만든다. 후보 규칙 4개:
  pooled  전국 모든 시군구의 과거 로그 오차 10%·90% 분위를 그대로 씀
  size    오차 크기 = sqrt(c² + 1/n), n = 작년 혼인 건수(작은 지역일수록 넓게). c는 과거 오차로 추정
  disp    size와 같되 1/n 앞의 계수 φ도 과거 오차로 추정 (작은 지역이 이론보다 더 흔들리면 φ>1)
  local   그 지역 자신의 과거 오차 분위(5년 미만이면 pooled)
  recent  disp와 같되 최근 3년 오차로만 폭을 잼(흐름이 바뀌면 빨리 넓어짐) — 도전자

채점: 실제 값이 범위 안에 든 비율(목표 80%)과 interval score(넓을수록·벗어날수록 손해, 작을수록 좋음).
고르는 구간 2010~2018, 시험 구간 2019~2025. 플랜 A: 규칙 하나를 모든 지역에 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import nowcast as NC
from . import regional_tune as RT

RULES = ["pooled", "size", "disp", "local", "recent"]
LABEL = {"pooled": "전국 공통 폭", "size": "지역 크기 반영", "disp": "지역 크기 반영 (작은 지역 보정)", "local": "지역 자기 과거", "recent": "작은 지역 보정 + 최근 3년"}
LEVEL = 0.8
LO, HI = (1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2
PICK, TEST = range(2010, 2019), range(2019, 2026)
CHOSEN, CHALLENGER = "disp", "recent"           # CHOSEN = choose(evaluate()) 결과를 고정 (테스트로 확인)
FIRST = 2006                                                    # 1~6월 신호가 있는 첫 해


def residuals() -> pd.DataFrame:
    """연도 × 시군구: 실제 로그 건수 − 예측 로그 건수, 그리고 작년 건수."""
    sgg, _ = RT.load()
    rows = []
    for y in range(FIRST, int(sgg.index.max()) + 1):
        pred = NC.predict(sgg, y, 6)
        rows.append(pd.DataFrame({"year": y, "code": sgg.columns, "pred": pred.to_numpy(),
                                  "truth": sgg.loc[y].to_numpy(), "n": np.exp(sgg.loc[y - 1].to_numpy())}))
    r = pd.concat(rows, ignore_index=True)
    r["res"] = r.truth - r.pred
    return r.dropna(subset=["pred"])


def _c(past: pd.DataFrame) -> float:
    """오차 중 지역 크기로 설명 안 되는 몫(체계 오차)의 크기."""
    return float(np.sqrt(max(np.mean(past.res ** 2 - 1 / past.n), 1e-6)))


def _cphi(past: pd.DataFrame) -> tuple[float, float]:
    """오차² ≈ c² + φ/n 을 과거 오차로 최소제곱 추정 (둘 다 0 이상)."""
    x, y = 1 / past.n.to_numpy(), past.res.to_numpy() ** 2
    A = np.c_[np.ones_like(x), x]
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    a, b = max(a, 1e-6), max(b, 0.0)
    return float(np.sqrt(a)), float(b)


def _scale(rule: str, src: pd.DataFrame, n) -> np.ndarray:
    if rule in ("disp", "recent"):
        c, phi = _cphi(src)
        return np.sqrt(c ** 2 + phi / np.asarray(n, float))
    c = _c(src)
    return np.sqrt(c ** 2 + 1 / np.asarray(n, float))


def bounds(rule: str, past: pd.DataFrame, now: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """past(작년까지 채점된 오차)만으로 now 각 행의 (하한, 상한) 로그 오프셋."""
    if rule == "pooled":
        q = past.res.quantile([LO, HI]).to_numpy()
        return np.full(len(now), q[0]), np.full(len(now), q[1])
    if rule in ("size", "recent", "disp"):
        src = past[past.year >= past.year.max() - 2] if rule == "recent" else past
        z = (src.res / _scale(rule, src, src.n)).quantile([LO, HI]).to_numpy()
        s = _scale(rule, src, now.n)
        return z[0] * s, z[1] * s
    if rule == "local":
        q0 = past.res.quantile([LO, HI]).to_numpy()
        g = past.groupby("code").res
        cnt, lo, hi = g.size(), g.quantile(LO), g.quantile(HI)
        ok = now.code.map(cnt).fillna(0).to_numpy() >= 5
        return (np.where(ok, now.code.map(lo).to_numpy(), q0[0]), np.where(ok, now.code.map(hi).to_numpy(), q0[1]))
    raise ValueError(rule)


def evaluate(r: pd.DataFrame | None = None, years=range(2010, 2026)) -> pd.DataFrame:
    """각 해를 그 전 해까지의 오차로만 범위를 만들어 채점."""
    r = residuals() if r is None else r
    out = []
    for y in years:
        past, now = r[r.year < y], r[r.year == y]
        for rule in RULES:
            lo, hi = bounds(rule, past, now)
            x = now.res.to_numpy()
            inside = (x >= lo) & (x <= hi)
            a = 1 - LEVEL
            score = (hi - lo) + 2 / a * np.maximum(lo - x, 0) + 2 / a * np.maximum(x - hi, 0)
            out.append(pd.DataFrame({"year": y, "code": now.code.to_numpy(), "rule": rule, "n": now.n.to_numpy(),
                                     "inside": inside, "width": np.expm1(hi) - np.expm1(lo), "score": score}))
    return pd.concat(out, ignore_index=True)


def summary(e: pd.DataFrame) -> pd.DataFrame:
    seg = np.where(e.year.isin(PICK), "pick", "test")
    t = e.assign(seg=seg).groupby(["rule", "seg"]).agg(cover=("inside", "mean"), width=("width", "median"), score=("score", "mean"))
    return t.unstack("seg")


def choose(e: pd.DataFrame) -> str:
    """고르는 구간의 interval score가 가장 작은 규칙."""
    return e[e.year.isin(PICK)].groupby("rule").score.mean().idxmin()


def params(rule: str, r: pd.DataFrame, year: int) -> dict:
    """size·disp·recent 규칙의 상수: 범위 = 예측 × exp(z × sqrt(c² + φ/작년 건수))."""
    past = r[r.year < year]
    src = past[past.year >= past.year.max() - 2] if rule == "recent" else past
    c, phi = _cphi(src) if rule in ("disp", "recent") else (_c(src), 1.0)
    z = (src.res / _scale(rule, src, src.n)).quantile([LO, HI]).to_numpy()
    return {"c": c, "phi": phi, "z_lo": float(z[0]), "z_hi": float(z[1])}


def forecast(rule: str, year: int = 2026, r: pd.DataFrame | None = None) -> pd.DataFrame:
    """올해 예측 + 80% 범위 (건수)."""
    r = residuals() if r is None else r
    sgg, _ = RT.load()
    now = pd.DataFrame({"code": sgg.columns, "pred": NC.predict(sgg, year, 6).to_numpy(), "n": np.exp(sgg.loc[year - 1].to_numpy())})
    lo, hi = bounds(rule, r[r.year < year], now)
    return pd.DataFrame({"code": now.code, "pred": np.exp(now.pred), "lo": np.exp(now.pred + lo), "hi": np.exp(now.pred + hi)}).set_index("code")
