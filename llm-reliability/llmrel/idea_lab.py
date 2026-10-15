"""아이디어 실험대: '시장 확률 + 내 아이디어 조정값'이 시장보다 잘 맞히는지 시험.

규칙 (결과 보기 전에 정함)
  1. 아이디어는 경기 전에 알 수 있는 정보만 쓰는 함수 idea(ctx) -> (n, 3) 배열 z (홈승·무·원정승 쪽으로 미는 힘).
     조정 확률 = softmax(log 시장 확률 + β·z). β 하나만 학습 구간(2010/11~2020/21)에서 고른다.
  2. 시험 전에 장부에 등록 (kind=epl_idea: 가설 + 코드 지문). 코드가 바뀌면 새 시도로 센다.
  3. 판정은 검증 구간(2021/22~2023/24): 시장 대비 RPS 이득의 95% 하한 > 0 이고 β ≠ 0 이면 통과.
  4. 시험 구간(2024/25~2025/26)은 검증을 통과한 아이디어만, 아이디어당 한 번만 연다.
  5. 시도한 아이디어 수를 함께 보고한다 (k개를 시험하면 우연히 하나쯤 통과할 확률 ≈ 1 − 0.975^k).
"""
from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pandas as pd

from . import epl, ledger, market

BETAS = np.round(np.arange(-2, 2.001, 0.05), 2)


def context(params: dict | None = None) -> pd.DataFrame:
    """경기마다 경기 전에 알 수 있는 정보 + 시장 확률 + 결과. 경기 후 정보(xG, 골)는 넣지 않는다."""
    df = epl.load()
    res = epl.run(df, **(params or epl.XG_MODE))
    # 휴식일: 리그 경기 기준 직전 경기 후 며칠 (컵 대회 경기는 자료에 없음)
    long = pd.concat([res[["date", "home"]].rename(columns={"home": "team"}), res[["date", "away"]].rename(columns={"away": "team"})])
    long["date"] = pd.to_datetime(long.date)
    long = long.sort_values("date")
    long["rest"] = long.groupby("team").date.diff().dt.days
    rest = long.drop_duplicates(["team", "date"]).set_index(["team", "date"]).rest
    d = pd.to_datetime(res.date)
    res["rest_h"] = [rest.get((t, x), np.nan) for t, x in zip(res.home, d)]
    res["rest_a"] = [rest.get((t, x), np.nan) for t, x in zip(res.away, d)]
    res["n_h"] = res.groupby(["season", "home"]).cumcount()            # 이번 시즌 홈 경기 몇 번째 (대략적 시즌 진행)
    res["month"] = d.dt.month
    ctx = res[res.played].merge(market.load_odds(), on=["date", "home", "away"], how="inner").dropna(subset=["mH"])
    keep = ["season", "date", "home", "away", "result", "mH", "mD", "mA", "pH", "pD", "pA", "xh", "xa", "rest_h", "rest_a", "n_h", "month"]
    return ctx[keep].reset_index(drop=True)


def _adjust(M: np.ndarray, z: np.ndarray, beta: float) -> np.ndarray:
    lg = np.log(np.clip(M, 1e-9, 1)) + beta * z
    e = np.exp(lg - lg.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _code_hash(fn) -> str:
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = repr(fn)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def _split(ctx: pd.DataFrame, name: str) -> np.ndarray:
    seasons = epl.SPLITS["train"] if name == "train" else epl.SPLITS[name]
    return ctx.season.isin(seasons).to_numpy()


def evaluate(name: str, idea, hypothesis: str, ctx: pd.DataFrame | None = None, open_test: bool = False,
             n_boot: int = 2000, seed: int = 0, record: bool = True) -> dict:
    ctx = context() if ctx is None else ctx
    code = _code_hash(idea)
    rows = ledger.read() if record else []
    if record:
        ledger.append([dict(kind="epl_idea", key=name, payload={"hypothesis": hypothesis, "code": code},
                            model="idea_lab", evidence="결과 보기 전 등록")])
    z = np.nan_to_num(np.asarray(idea(ctx), float))
    if z.shape != (len(ctx), 3):
        raise ValueError(f"아이디어는 (경기 수, 3) 배열을 돌려줘야 함: {z.shape}")
    M = ctx[["mH", "mD", "mA"]].to_numpy()
    tr, va, te = _split(ctx, "train"), _split(ctx, "val"), _split(ctx, "test")
    loss = [market.rps_rows(_adjust(M[tr], z[tr], b), ctx.result[tr]).mean() for b in BETAS]
    beta = float(BETAS[int(np.argmin(loss))])
    rng = np.random.default_rng(seed)

    def gain(mask):
        g = market.rps_rows(M[mask], ctx.result[mask]) - market.rps_rows(_adjust(M[mask], z[mask], beta), ctx.result[mask])
        bs = [g[rng.integers(0, len(g), len(g))].mean() for _ in range(n_boot)]
        return float(g.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(mask.sum())

    out = {"name": name, "hypothesis": hypothesis, "code": code, "beta": beta, "train_gain": float(loss[list(BETAS).index(0.0)] - min(loss))}
    out["val"], out["val_lo"], out["val_hi"], out["n_val"] = gain(va)
    out["passed"] = bool(out["val_lo"] > 0 and beta != 0)
    if open_test:
        opened = [r for r in rows if r["kind"] == "epl_idea_test" and r["key"] == name and r["payload"].get("code") == code]
        if not out["passed"]:
            raise PermissionError("검증 구간을 통과하지 못한 아이디어는 시험 구간을 열 수 없음")
        if opened:
            raise PermissionError("이 아이디어(같은 코드)는 이미 시험 구간을 한 번 열었음")
        out["test"], out["test_lo"], out["test_hi"], out["n_test"] = gain(te)
    if record:
        ledger.append([dict(kind="epl_idea_test" if open_test else "epl_idea_result", key=name,
                            payload={k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items() if k != "hypothesis"},
                            model="idea_lab", evidence="검증 구간 판정" + (" + 시험 구간 1회" if open_test else ""))])
    return out


def tried() -> pd.DataFrame:
    """장부에 등록된 아이디어 (같은 이름이라도 코드가 바뀌면 다른 시도)."""
    rows = [r for r in ledger.read() if r["kind"] in ("epl_idea_result", "epl_idea_test")]
    df = pd.DataFrame([{**r["payload"], "kind": r["kind"]} for r in rows])
    return df.drop_duplicates(["name", "code", "kind"], keep="last") if len(df) else df


# ── 예시 아이디어 (형식 보기용) ─────────────────────────────────────────────
def idea_our_model(ctx):
    """우리 포아송 모델이 시장과 다르게 본 만큼 민다."""
    return np.log(ctx[["pH", "pD", "pA"]].to_numpy()) - np.log(ctx[["mH", "mD", "mA"]].to_numpy())


def idea_rest_days(ctx):
    """더 오래 쉰 팀 쪽으로 민다 (휴식일 차이, 7일에서 자름)."""
    d = (ctx.rest_h.fillna(7) - ctx.rest_a.fillna(7)).clip(-7, 7).to_numpy() / 7
    return np.c_[d, np.zeros(len(d)), -d]


def idea_longshot(ctx):
    """시장이 10% 미만으로 본 결과는 더 낮춘다 (이변 과대평가 가설)."""
    M = ctx[["mH", "mD", "mA"]].to_numpy()
    return -(M < 0.10).astype(float)


EXAMPLES = {
    "우리 모델 섞기": (idea_our_model, "우리 포아송 모델이 시장이 모르는 정보를 갖고 있다"),
    "휴식일 차이": (idea_rest_days, "리그 경기 사이 휴식일이 긴 팀이 시장 기대보다 잘한다"),
    "이변 과대평가": (idea_longshot, "시장은 10% 미만 이변 확률을 실제보다 높게 매긴다"),
}
