"""과열 온도계 v0: 12개월 안에 20% 넘게 하락할 확률.

킨들버거 패턴(신기술 투기 → 신용 팽창 → 자본 유입 → 금리 전환 → 붕괴) 중 지금 받을 수 있는 가격·금리로 만든 첫 판.
신용(가계대출·신용 대 GDP 갭)은 v1 — 한국은행·FRED·BIS 접속이 열리면 붙인다.

특성 (월말 기준)
  ret36   3년 로그 수익률                       — 가격 과열
  accel   최근 12개월 − 3년 연평균               — 상승 가속 (도취 단계)
  gap10   10개월 이동평균 대비 이격               — 추세 이탈
  vol12   최근 12개월 월간 변동성                 — 불안
  rate6   미국 3개월물 금리 6개월 변화(%p)        — 금리 전환 (나라 공통 v0)
  turn    직전 12개월 하락 뒤 6개월 상승이면 1     — 인하 → 인상 전환
정답 y = 이후 12개월 중 최저가가 지금보다 20% 이상 낮으면 1.

v1 (2026-10-13, 결과 보기 전에 정함) — KOSIS 자료로 신용을 붙인다
  cg3     민간부문 국내신용/GDP의 3년 변화(%p). KOSIS DT_2AQ551(세계은행 계열, 연간).
          Y년 값은 Y+1년 6월 말부터 쓸 수 있다고 놓는다 (발표 시차 → 미래 정보 차단)
  dd12    최근 12개월 고점 대비 현재 낙폭 — 이미 빠진 시장과 막 꼭짓점인 시장을 구분
  채택 규칙: walk-forward skill이 v0보다 높고 0보다 크면 v1 채택, 아니면 v0 유지하고 '신용도 못 이김'을 공개.
  한국 가계신용(DT_151Y001, 분기)은 한국만 있어 합동 학습에 못 넣고 보고서 참고 지표로만 쓴다.

학습: 시장 셋(S&P500·닛케이·코스피)을 합친 L2 로지스틱 회귀, 한 달씩 앞으로 가며(walk-forward) 그 시점에
정답을 아는 자료(12개월 이전)만으로 다시 맞춘다. 기준선 = 그때까지의 하락 빈도(기후값).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent.parent / "data_crisis"
MARKETS = {"S&P500": "GSPC", "닛케이225": "N225", "코스피": "KS11"}
FEATS = ["ret36", "accel", "gap10", "vol12", "rate6", "turn"]
FEATS_V1 = FEATS + ["cg3", "dd12"]
COUNTRY = {"S&P500": "미국", "닛케이225": "일본", "코스피": "대한민국"}
H, DROP = 12, -0.20


def load(code: str) -> pd.Series:
    s = pd.read_csv(DATA / f"m_{code}.csv", index_col=0, parse_dates=True).iloc[:, 0]
    return s[~s.index.duplicated()].dropna()


def credit_gdp() -> dict[str, pd.Series]:
    """나라별 민간신용/GDP(%) 월말 시계열 — Y년 값은 Y+1년 6월 말부터 보인다."""
    c = pd.read_csv(DATA / "_raw_credit_gdp.csv")
    out = {}
    for name, g in c.groupby("C1_NM"):
        g = g.dropna(subset=["value"])
        s = pd.Series(g.value.to_numpy(), index=pd.to_datetime([f"{y + 1}-06-30" for y in g.year]))
        out[name] = s.sort_index().resample("ME").last().ffill()
    return out


def household_credit() -> pd.DataFrame:
    """한국 가계신용(조원) 분기말 잔액 + 전년 대비 증가율. 분기 끝 2개월 뒤 발표."""
    h = pd.read_csv(DATA / "_raw_household_credit.csv", dtype={"PRD_DE": str})
    q = pd.PeriodIndex([f"{p[:4]}Q{int(p[4:])}" for p in h.PRD_DE], freq="Q")
    d = pd.DataFrame({"조원": h.value.to_numpy() / 1000}, index=q)
    d["전년비"] = d["조원"].pct_change(4)
    return d


def features(px: pd.Series, irx: pd.Series, credit: pd.Series | None = None) -> pd.DataFrame:
    lp = np.log(px)
    r = lp.diff()
    d = pd.DataFrame(index=px.index)
    d["ret36"] = lp - lp.shift(36)
    d["accel"] = (lp - lp.shift(12)) - d.ret36 / 3
    d["gap10"] = px / px.rolling(10).mean() - 1
    d["vol12"] = r.rolling(12).std() * np.sqrt(12)
    i = irx.reindex(px.index, method="ffill")
    d["rate6"] = i - i.shift(6)
    d["turn"] = ((i.shift(6) - i.shift(18) < 0) & (d.rate6 > 0)).astype(float)
    d["dd12"] = px / px.rolling(12).max() - 1
    if credit is not None:
        c = credit.reindex(px.index, method="ffill")
        d["cg3"] = c - c.shift(36)
    fut_min = pd.concat([px.shift(-k) for k in range(1, H + 1)], axis=1).min(axis=1)
    d["y"] = np.where(px.shift(-H).notna(), (fut_min / px - 1 <= DROP).astype(float), np.nan)
    d["fut_min"] = fut_min / px - 1
    return d


def panel(feats: list[str] = FEATS) -> pd.DataFrame:
    irx = load("IRX")
    cr = credit_gdp() if "cg3" in feats else {}
    parts = []
    for name, code in MARKETS.items():
        f = features(load(code), irx, cr.get(COUNTRY[name]))
        f["market"] = name
        parts.append(f)
    p = pd.concat(parts).dropna(subset=feats)
    return p.sort_index()


def fit_logit(X: np.ndarray, y: np.ndarray, lam: float = 1.0, iters: int = 60) -> np.ndarray:
    """L2 로지스틱 회귀 (뉴턴법). X 첫 열은 상수."""
    w = np.zeros(X.shape[1])
    R = lam * np.eye(X.shape[1]); R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + R @ w
        Hs = (X * (p * (1 - p))[:, None]).T @ X + R
        step = np.linalg.solve(Hs, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def design(df: pd.DataFrame, mu: pd.Series, sd: pd.Series) -> np.ndarray:
    Z = ((df[mu.index] - mu) / sd).to_numpy()
    return np.c_[np.ones(len(Z)), Z]


def walk_forward(p: pd.DataFrame, start: str = "1975-01-31", lam: float = 1.0, feats: list[str] = FEATS) -> pd.DataFrame:
    """각 달 t에서 정답이 확정된 자료(t−12개월까지)로만 학습해 t의 확률을 낸다."""
    out = []
    months = sorted(p.index[p.index >= start].unique())
    for t in months:
        train = p[(p.index <= t - pd.DateOffset(months=H)) & p.y.notna()]
        if train.y.sum() < 10:
            continue
        mu, sd = train[feats].mean(), train[feats].std().replace(0, 1)
        w = fit_logit(design(train, mu, sd), train.y.to_numpy(), lam)
        now = p.loc[[t]]
        prob = 1 / (1 + np.exp(-design(now, mu, sd) @ w))
        base = train.y.mean()
        for (_, row), pr in zip(now.iterrows(), prob):
            out.append({"date": t, "market": row.market, "p": pr, "base": base, "y": row.y, "fut_min": row.fut_min})
    return pd.DataFrame(out)


def score(w: pd.DataFrame) -> dict:
    s = w.dropna(subset=["y"])
    b_model = np.mean((s.p - s.y) ** 2)
    b_base = np.mean((s.base - s.y) ** 2)
    bins = pd.cut(s.p, [0, .05, .1, .2, .3, .5, 1])
    cal = s.groupby(bins, observed=True).agg(n=("y", "size"), 예측=("p", "mean"), 실제=("y", "mean"))
    return {"n": len(s), "brier": b_model, "brier_base": b_base, "skill": 1 - b_model / b_base, "calibration": cal,
            "by_market": s.groupby("market").apply(lambda g: pd.Series({"n": len(g), "실제 빈도": g.y.mean(),
                                                                       "skill": 1 - np.mean((g.p - g.y) ** 2) / np.mean((g.base - g.y) ** 2)}))}


def current(p: pd.DataFrame, lam: float = 1.0, n_boot: int = 300, seed: int = 0, feats: list[str] = FEATS) -> pd.DataFrame:
    """가장 최근 달의 확률 + 80% 범위(학습 자료를 다시 뽑아 계수를 흔든 결과)."""
    rng = np.random.default_rng(seed)
    t = p.index.max()
    train = p[p.y.notna()]
    mu, sd = train[feats].mean(), train[feats].std().replace(0, 1)
    X, y = design(train, mu, sd), train.y.to_numpy()
    now = p[p.index == t]
    Xn = design(now, mu, sd)
    w = fit_logit(X, y, lam)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        boots.append(1 / (1 + np.exp(-Xn @ fit_logit(X[idx], y[idx], lam))))
    B = np.array(boots)
    out = now[["market"] + feats].copy()
    out["p"] = 1 / (1 + np.exp(-Xn @ w))
    out["lo80"], out["hi80"] = np.percentile(B, 10, axis=0), np.percentile(B, 90, axis=0)
    out["base"] = y.mean()
    lo, hi = train[feats].min(), train[feats].max()
    out["out_of_range"] = [", ".join(c for c in feats if not lo[c] <= r[c] <= hi[c]) for _, r in now.iterrows()]   # 학습에 없던 값 → 외삽
    out.attrs["coef"] = dict(zip(["const"] + feats, w))
    return out
