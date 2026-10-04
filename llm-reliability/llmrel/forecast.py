"""연령별 혼인율·평균 초혼연령 예측.

모든 계열은 로그로 바꿔 다룬다 (비율이라 양수이고, 변화가 곱셈적이기 때문).
모델 6개를 같은 조건에서 비교한다:
  naive     마지막 값 유지
  drift     최근 5년 평균 변화가 계속된다
  damped    감쇠 추세 지수평활 (추세가 점점 약해진다)
  loglin    최근 10년 로그-선형 추세
  pooled    모든 계열을 함께 학습한 릿지 회귀 (자기 최근 변화 + 같은 나라 다른 연령대 변화)
  ensemble  drift·damped·pooled 평균

검증 = 롤링 원점: 원점 연도 T까지의 자료만으로 T+h를 맞히게 하고, 실제 값과 비교한다.
예측 구간 = 그 원점 이전에 정답이 나온 오차의 분포에서만 만든다 (미래 오차를 미리 보지 않음).
과거 오차를 그대로 쓰면 구간이 좁게 나오므로 (충격·추세 전환), 배수 k를 앞 구간에서 고르고 뒤 구간에서 확인한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MODELS = ["naive", "drift", "damped", "loglin", "pooled", "ensemble"]
BANDS = ["20-24", "25-29", "30-34", "35-39", "40-44"]
COVID = {2020, 2021}
PUBLISH = {"KR": (3, "다음 해 3월 (통계청 혼인·이혼 통계)"), "JP": (9, "다음 해 9월 (인구동태 확정수)")}


def series_table(d: pd.DataFrame, start: int = 2000) -> pd.DataFrame:
    """연도 × 계열 표. 계열 이름 = 나라|지표|성별|연령대."""
    keep = ((d.indicator == "marriage_rate") & d.age_band.astype(str).isin(BANDS)) | (d.indicator == "first_marriage_age")
    s = d[keep & (d.year >= start)].copy()
    s["key"] = s.country + "|" + s.indicator + "|" + s.sex + "|" + s.age_band.astype(str)
    return s.pivot_table(index="year", columns="key", values="value").sort_index()


# ── 개별 계열 모델 (입력: 원점까지의 로그 값 배열) ───────────────────────
def _naive(y, h):
    return y[-1]


def _drift(y, h, k=5):
    k = min(k, len(y) - 1)
    return y[-1] + h * (y[-1] - y[-1 - k]) / k


def _damped(y, h):
    best = None
    for a in (0.3, 0.5, 0.7, 0.9):
        for b in (0.05, 0.1, 0.2, 0.4):
            for phi in (0.8, 0.9, 0.98):
                lv, tr, sse = y[0], y[1] - y[0], 0.0
                for v in y[1:]:
                    f = lv + phi * tr
                    sse += (v - f) ** 2
                    nlv = a * v + (1 - a) * f
                    tr = b * (nlv - lv) + (1 - b) * phi * tr
                    lv = nlv
                if best is None or sse < best[0]:
                    best = (sse, lv, tr, phi)
    _, lv, tr, phi = best
    return lv + sum(phi ** i for i in range(1, h + 1)) * tr


def _loglin(y, h, k=10):
    z = y[-k:]
    t = np.arange(len(z))
    b, a = np.polyfit(t, z, 1)
    return a + b * (len(z) - 1 + h)


SINGLE = {"naive": _naive, "drift": _drift, "damped": _damped, "loglin": _loglin}


# ── 함께 학습하는 모델 ───────────────────────────────────────────────────
def _features(L: pd.DataFrame, key: str, t: int) -> np.ndarray | None:
    """원점 t에서 알 수 있는 정보만 사용."""
    col = L[key]
    if t - 3 not in col.index or col.loc[t - 3:t].isna().any():
        return None
    c = key.split("|")[0]; ind = key.split("|")[1]
    peers = [k for k in L.columns if k.startswith(f"{c}|{ind}|") and k != key]
    peer1 = np.nanmean([L.at[t, k] - L.at[t - 1, k] for k in peers]) if peers else 0.0
    return np.array([1.0, col[t] - col[t - 1], (col[t] - col[t - 3]) / 3, peer1, float(c == "KR"), float(ind == "marriage_rate")])


def _pooled_fit(L: pd.DataFrame, origin: int, h: int, lam: float = 1.0):
    """원점 이전에 '정답까지 확인된' 사례만으로 학습: t + h <= origin."""
    X, Y = [], []
    for key in L.columns:
        for t in L.index:
            if t + h > origin or t + h not in L.index:
                continue
            f = _features(L, key, t)
            if f is None or np.isnan(L.at[t + h, key]) or np.isnan(f).any():
                continue
            X.append(f); Y.append(L.at[t + h, key] - L.at[t, key])
    if len(X) < 20:
        return None
    X, Y = np.array(X), np.array(Y)
    P = lam * np.eye(X.shape[1]); P[0, 0] = 0
    return np.linalg.solve(X.T @ X + P, X.T @ Y)


def predict_all(L: pd.DataFrame, origin: int, h: int, _cache: dict | None = None) -> dict:
    """원점 origin까지의 자료로 모든 계열의 origin+h 로그 값을 모델별로 예측."""
    w = _pooled_fit(L, origin, h)
    out = {}
    for key in L.columns:
        y = L[key].loc[:origin].dropna().to_numpy()
        if len(y) < 8 or L[key].loc[:origin].dropna().index[-1] != origin:
            continue
        p = {m: f(y, h) for m, f in SINGLE.items()}
        f = _features(L, key, origin)
        p["pooled"] = y[-1] + float(f @ w) if (w is not None and f is not None and not np.isnan(f).any()) else p["drift"]
        p["ensemble"] = np.mean([p["drift"], p["damped"], p["pooled"]])
        out[key] = p
    return out


# ── 검증 ─────────────────────────────────────────────────────────────────
def backtest(L: pd.DataFrame, origins: range, horizons=(1, 2, 3)) -> pd.DataFrame:
    rows = []
    for h in horizons:
        for o in origins:
            preds = predict_all(L, o, h)
            for key, p in preds.items():
                if o + h not in L.index or np.isnan(L.at[o + h, key]):
                    continue
                truth = L.at[o + h, key]
                for m, v in p.items():
                    rows.append(dict(key=key, origin=o, h=h, target=o + h, model=m, log_err=truth - v,
                                     ape=abs(np.exp(v) / np.exp(truth) - 1)))
    bt = pd.DataFrame(rows)
    bt["country"] = bt.key.str.split("|").str[0]
    bt["indicator"] = bt.key.str.split("|").str[1]
    bt["covid"] = bt.target.isin(COVID)
    return bt


def score_table(bt: pd.DataFrame) -> pd.DataFrame:
    """모델 × 예측 거리별 평균 절대 오차율. 코로나 해(2020·2021) 포함/제외 둘 다."""
    a = bt.groupby(["model", "h"]).ape.mean().unstack()
    b = bt[~bt.covid].groupby(["model", "h"]).ape.mean().unstack()
    a.columns = [f"{h}년 뒤" for h in a.columns]
    b.columns = [f"{h}년 뒤 (코로나 제외)" for h in b.columns]
    return a.join(b).reindex(MODELS)


def selection_check(bt: pd.DataFrame, split: int = 2014) -> pd.DataFrame:
    """모델 고르기도 검증: split 이전 성적으로 고른 모델이 이후에도 좋은가."""
    rows = []
    for h, g in bt[bt.indicator == "marriage_rate"].groupby("h"):
        early = g[g.target <= split].groupby("model").ape.mean()
        late = g[g.target > split].groupby("model").ape.mean()
        pick = early.idxmin()
        rows.append({"예측 거리": f"{h}년", f"~{split} 성적으로 고른 모델": pick, "고른 모델의 이후 오차": late[pick],
                     "이후 가장 좋았던 모델": late.idxmin(), "그 모델의 이후 오차": late.min(), "naive 이후 오차": late["naive"]})
    return pd.DataFrame(rows)


def _hits(g: pd.DataFrame, level: float, k: float, min_hist: int, origins) -> list:
    """원점마다 그 시점에 정답이 나온 과거 오차의 |오차| 분위수 × k 를 구간 반폭으로 쓴다."""
    hits = []
    for o in origins:
        hist = g[g.target <= o].log_err.abs()
        if len(hist) < min_hist:
            continue
        half = hist.quantile(level) * k
        hits += list(g[g.origin == o].log_err.abs() <= half)
    return hits


def interval_check(bt: pd.DataFrame, model: str, level: float = 0.8, split: int = 2014, min_hist: int = 30) -> pd.DataFrame:
    """구간 보정 검증.
    1) 과거 오차 분위수를 그대로 쓰면 실제 포함률이 목표보다 낮은지 (k = 1)
    2) split 이전 원점에서 목표를 맞추는 배수 k를 고르고, 이후 원점에서 포함률을 다시 잰다."""
    rows = []
    for h, g in bt[bt.model == model].groupby("h"):
        og = sorted(g.origin.unique())
        early, late = [o for o in og if o + h <= split], [o for o in og if o + h > split]
        raw = _hits(g, level, 1.0, min_hist, og)
        ks = np.arange(1.0, 3.01, 0.05)
        cov = [np.mean(_hits(g, level, k, min_hist, early) or [np.nan]) for k in ks]
        k = next((k for k, c in zip(ks, cov) if c >= level), ks[-1])
        after = _hits(g, level, k, min_hist, late)
        rows.append({"예측 거리": f"{h}년", "목표": level, "보정 전 포함률": np.mean(raw), "고른 배수 k": round(float(k), 2),
                     "보정 후 포함률 (이후 원점)": np.mean(after) if after else np.nan, "판정 수": len(after)})
    return pd.DataFrame(rows)


def interval_scale(bt: pd.DataFrame, model: str, level: float = 0.8, min_hist: int = 30) -> dict:
    """최종 예측에 쓸 배수: 전체 원점에서 목표 포함률을 맞추는 가장 작은 k (예측 거리별)."""
    out = {}
    for h, g in bt[bt.model == model].groupby("h"):
        og = sorted(g.origin.unique())
        out[h] = next((float(k) for k in np.arange(1.0, 3.01, 0.05) if np.mean(_hits(g, level, k, min_hist, og)) >= level), 3.0)
    return out


# ── 예측 ─────────────────────────────────────────────────────────────────
def forecast(L: pd.DataFrame, bt: pd.DataFrame, model: str, horizons=(1, 2, 3), level: float = 0.8) -> pd.DataFrame:
    """계열마다 자기 마지막 연도를 원점으로 예측.
    구간 반폭 = 백테스트 |로그 오차|의 level 분위수 × 배수 k (백테스트에서 실제 포함률이 level이 되도록 고름).
    코로나 해 오차도 넣는다: 충격은 또 올 수 있다."""
    scale = interval_scale(bt[bt.indicator == "marriage_rate"], model, level)
    rows = []
    last_year = {k: int(L[k].dropna().index[-1]) for k in L.columns}
    for origin in sorted(set(last_year.values())):
        keys = [k for k, y in last_year.items() if y == origin]
        for h in horizons:
            preds = predict_all(L[[c for c in L.columns if last_year[c] >= origin]], origin, h)
            for k in keys:
                if k not in preds:
                    continue
                c, ind, sx, band = k.split("|")
                e = bt[(bt.model == model) & (bt.h == h) & (bt.indicator == ind)].log_err.abs()
                half = e.quantile(level) * scale[h]
                v = preds[k][model]
                month, _ = PUBLISH[c]
                rows.append(dict(country=c, indicator=ind, sex=sx, age_band=band, origin=origin, year=origin + h, model=model,
                                 forecast=round(float(np.exp(v)), 2), lo=round(float(np.exp(v - half)), 2),
                                 hi=round(float(np.exp(v + half)), 2), level=level, k=round(scale[h], 2),
                                 last_value=round(float(np.exp(L.at[origin, k])), 2), resolves=f"{origin + h + 1}-{month:02d}"))
    return pd.DataFrame(rows)


def score_forecasts(fc: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """저장해 둔 예측을 새로 받은 공식 값과 맞춰 채점. 아직 안 나온 값은 pending."""
    truth = d.assign(age_band=d.age_band.astype(str)).set_index(["country", "indicator", "sex", "age_band", "year"]).value
    out = fc.copy()
    idx = list(zip(out.country, out.indicator, out.sex, out.age_band.astype(str), out.year))
    out["actual"] = [truth.get(i, np.nan) for i in idx]
    out["status"] = np.where(out.actual.isna(), "pending", "resolved")
    out["ape"] = (out.forecast / out.actual - 1).abs()
    out["in_interval"] = np.where(out.actual.isna(), np.nan, (out.actual >= out.lo) & (out.actual <= out.hi))
    return out
