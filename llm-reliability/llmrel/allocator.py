"""주간 시간 배분기: '다음 1시간의 한계효용'이 모든 활동에서 같아지도록 이번 주 시간을 나눈다.

플랜 A 구조
  1) 바닥(4-4-2 고정): 학업·회복 최소 시간은 최적화하지 않고 먼저 뺀다. 바닥이 무너지면 모든 게 무너진다.
  2) 나머지 시간: 활동마다 체감하는 가치 곡선 V(h) = w·(1 − e^(−h/τ))을 두고,
     도함수 V'(h) = (w/τ)·e^(−h/τ)가 모든 활동에서 같아지는 점(물 채우기, water-filling)을 찾는다.
       h* = τ·ln(w / (τ·μ))   (μ = 모두 같아진 한계효용, 합이 예산이 되도록 이분법으로 찾음)
  3) 확실 모드: 가중치 w는 가정이라, w를 ±(로그정규 σ=0.4)로 흔들어 200번 다시 나눠 10~90% 범위를 같이 낸다.

가중치(w) 출처
  - 수익 실험(E1~E3): llmrel.voi의 결정 변화 확률(%)을 그대로 쓴다. 단 E1은 결정을 못 바꾸므로 '바닥 만들기' 가치로 따로 둔다.
  - 그 외(인턴 준비, 사업계획서, 콘텐츠 등): data_biz/allocator.json에 적은 사전등록 가정. 마감이 멀면 0.3배(긴급도).
  - 장부에 결과(biz_result)가 기록된 실험은 가치 0 → 배분에서 빠진다.

    python -m llmrel.allocator                # → output/strategy/week.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import ledger

ROOT = Path(__file__).parent.parent
CONFIG = ROOT / "data_biz" / "allocator.json"
OUT = ROOT / "output" / "strategy"


def load(path: Path = CONFIG) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def weights(cfg: dict, today: pd.Timestamp, voi_table: pd.DataFrame | None = None, done: set | None = None) -> pd.DataFrame:
    """활동표: name, w, tau, min, max, 출처. 실험 가중치는 VOI에서 가져온다."""
    done = done if done is not None else {r["key"] for r in ledger.read() if r["kind"] == "biz_result"}
    if voi_table is None:
        from . import voi
        voi_table = voi.ranking()
    pv = voi_table.set_index("exp").p_change
    rows = []
    for a in cfg["activities"]:
        w = float(a.get("w", 0))
        src = a.get("source", "가정")
        if a.get("voi"):
            w = float(pv.get(a["voi"], 0.0)) * 100
            src = f"VOI 결정 변화 확률 {w:.0f}%"
        if a.get("exp") in done:
            w, src = 0.0, "결과 기록됨 (완료)"
        due = a.get("due")
        if due and w > 0:
            weeks_left = (pd.Timestamp(due) - today).days / 7
            if weeks_left < 0:
                w, src = 0.0, f"마감 지남 ({due})"
            elif weeks_left > a.get("window_weeks", 8):
                w, src = w * 0.3, src + f" × 0.3 (마감 {due}까지 {weeks_left:.0f}주)"
        rows.append({"활동": a["name"], "w": w, "tau": float(a["tau"]), "min": float(a.get("min", 0)), "max": float(a.get("max", 99)), "근거": src})
    return pd.DataFrame(rows)


def water_fill(w: np.ndarray, tau: np.ndarray, lo: np.ndarray, hi: np.ndarray, budget: float) -> tuple[np.ndarray, float]:
    """V_i'(h_i) = μ 가 되도록 h를 고른다. 반환: (시간, μ)."""
    def hours(mu):
        with np.errstate(divide="ignore"):
            h = np.where(w > 0, tau * np.log(np.maximum(w, 1e-12) / (tau * mu)), 0.0)
        return np.clip(h, lo, hi)
    if hours(1e-9).sum() <= budget:
        return hours(1e-9), 0.0
    a, b = 1e-9, max(1e-6, float((w / tau).max()) * 2)
    for _ in range(200):
        m = (a * b) ** 0.5
        a, b = (m, b) if hours(m).sum() > budget else (a, m)
    mu = (a * b) ** 0.5
    return hours(mu), mu


def allocate(cfg: dict, acts: pd.DataFrame, n_draws: int = 200, seed: int = 0) -> pd.DataFrame:
    budget = cfg["hours_total"] - sum(f["hours"] for f in cfg["floor"])
    w, tau, lo, hi = (acts[c].to_numpy(float) for c in ("w", "tau", "min", "max"))
    h, mu = water_fill(w, tau, lo, hi, budget)
    rng = np.random.default_rng(seed)
    draws = np.array([water_fill(w * np.exp(rng.normal(0, 0.4, len(w))), tau, lo, hi, budget)[0] for _ in range(n_draws)])
    out = acts.copy()
    out["시간"] = h
    out["10%"], out["90%"] = np.quantile(draws, 0.1, axis=0), np.quantile(draws, 0.9, axis=0)
    out["한계효용"] = np.where(h > 0, (w / tau) * np.exp(-h / tau), (w / tau))
    out["가치"] = w * (1 - np.exp(-h / tau))
    out.attrs.update(budget=budget, mu=mu)
    return out


def main(today: str | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load()
    t = pd.Timestamp(today or cfg.get("week_of") or pd.Timestamp.today().normalize())
    acts = weights(cfg, t)
    res = allocate(cfg, acts)
    open_ = res[res["시간"] < res["max"] - 1e-6]                 # 상한에 닿은 활동은 1시간을 더 넣을 수 없음
    nxt = (open_ if len(open_) else res).sort_values("한계효용", ascending=False).iloc[0]
    capped = res[(res["시간"] >= res["max"] - 1e-6) & (res["한계효용"] > res.attrs["mu"] + 1e-6)]
    L = [f"# 주간 시간 배분 ({t.date()} 주)", "",
         f"전체 {cfg['hours_total']:.0f}시간 중 **바닥 {sum(f['hours'] for f in cfg['floor']):.0f}시간**(" + ", ".join(f"{f['name']} {f['hours']:.0f}" for f in cfg["floor"]) +
         f")을 먼저 빼고, 남은 **{res.attrs['budget']:.0f}시간**을 '1시간당 효용 증가'가 같아지도록 나눴습니다.", "",
         "| 활동 | 이번 주 시간 | 흔들어도 (10~90%) | 마지막 1시간의 가치 | 가중치 근거 |", "|---|---|---|---|---|"]
    for _, r in res.sort_values("시간", ascending=False).iterrows():
        L.append(f"| {r['활동']} | **{r['시간']:.1f}h** | {r['10%']:.1f}~{r['90%']:.1f}h | {r['한계효용']:.1f} | {r['근거']} |")
    L += ["", f"- 모든 활동의 '마지막 1시간 가치'가 **{res.attrs['mu']:.1f}**로 같아지는 점이 최적입니다. 0시간인 활동은 첫 1시간 가치도 이보다 낮다는 뜻입니다.",
          f"- 시간이 1시간 더 생기면: **{nxt['활동']}**",
          *([f"- 상한에 닿은 활동(더 하면 이득이지만 주간 상한으로 막음): " + ", ".join(capped["활동"])] if len(capped) else []),
          "- '흔들어도' 범위가 좁은 활동은 가정이 조금 틀려도 같은 결론(확실), 넓은 활동은 가중치 가정에 민감합니다.",
          "", "가중치와 곡선은 `data_biz/allocator.json`의 가정입니다. 실험 결과가 장부에 기록되면 그 실험은 자동으로 빠지고, VOI가 다시 계산됩니다."]
    (OUT / "week.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return res


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
