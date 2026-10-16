"""2026 시군구 혼인 예측 채점표: 장부에 등록된 v1~v6와 범위 규칙을 한 번에 비교.

    python scoreboard_2026.py                    # 지금: 버전별 2026 예측 요약 (실제값 없음)
    python scoreboard_2026.py --actual a.csv     # 2027년 3월: code,value(2026 실제 건수) CSV로 채점
    (data_regional/kr_marriages_by_region.csv에 2026이 들어오면 --actual 없이도 채점)

채점 규칙은 실제값을 보기 전에 장부에 등록한다(kind=scoring_rule, key=marriage_2026, 한 번만):
  점 예측 = 시군구 평균 절대 오차(%). v1(현재) 대비 차이를 시군구 짝 부트스트랩 2000번 → 95% 구간.
  구간이 0보다 아래 = 맞음(개선), 0보다 위 = 틀림(악화), 0을 걸침 = 불확실.
  범위 = 80% 범위에 실제값이 들어간 비율 + interval score(작을수록 좋음).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from llmrel import ledger, regional_tune as RT
from model_upgrade import OUT

POINTS = {"v1 (현재)": "nowcast_2026", "v2": "nowcast_2026_v2", "v3": "nowcast_2026_v3", "v4": "nowcast_2026_v4",
          "v5": "nowcast_2026_v5", "v6": "nowcast_2026_v6"}
RANGES = {"v1 범위": ("nowcast_2026_range", "lo80", "hi80"), "v1 범위 도전자": ("nowcast_2026_range", "lo80_challenger", "hi80_challenger"),
          "v2 범위": ("nowcast_2026_v2_range", "lo80", "hi80"), "v2 최근 3년": ("nowcast_2026_v2_range_recent", "lo80", "hi80"),
          "v2 적응형": ("nowcast_2026_v2_range_adaptive", "lo80", "hi80")}
RULE = {"metric": "시군구 평균 절대 오차(%)", "baseline": "v1", "test": "시군구 짝 부트스트랩 2000번, 95% 구간",
        "verdict": "구간<0 맞음 / 구간>0 틀림 / 0 걸침 불확실", "ranges": "80% 포함률 + interval score(α=0.2)"}


def register_rule():
    if not any(x["kind"] == "scoring_rule" and x["key"] == "marriage_2026" for x in ledger.read()):
        ledger.append([dict(kind="scoring_rule", key="marriage_2026", model="scoreboard_2026", payload=RULE,
                            evidence="실제값(2027년 3월 발표)을 보기 전에 등록")])


def table(kind: str) -> pd.DataFrame:
    return pd.DataFrame({x["key"]: x["payload"] for x in ledger.read() if x["kind"] == kind}).T


def actuals() -> pd.Series | None:
    if "--actual" in sys.argv:
        a = pd.read_csv(sys.argv[sys.argv.index("--actual") + 1], dtype={"code": str})
        return a.set_index("code").value.astype(float)
    sgg, _ = RT.load()
    return np.exp(sgg.loc[2026]) if 2026 in sgg.index else None


def boot_ci(diff: np.ndarray, B: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = diff[rng.integers(0, len(diff), (B, len(diff)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    register_rule()
    sgg, _ = RT.load()
    last = np.exp(sgg.loc[2025])
    preds = pd.DataFrame({name: table(kind).pred_2026.astype(float) for name, kind in POINTS.items()})
    act = actuals()
    lines = ["# 2026 시군구 혼인 예측 채점표", "",
             f"채점 규칙(장부 kind=scoring_rule): {RULE['metric']}, 기준 {RULE['baseline']}, {RULE['test']}, {RULE['verdict']}. 범위는 {RULE['ranges']}.", ""]
    if act is None:
        lines += ["**실제값 없음** — 2027년 3월 KOSIS 발표 후 다시 실행하면 채점됩니다. 지금은 버전별 예측 요약입니다.", "",
                  "| 버전 | 전국 합계 | 2025 대비 | v1과 다른 정도(중앙값) | v1과 다른 정도(최대) |", "|---|---|---|---|---|"]
        for name in POINTS:
            d = (preds[name] / preds["v1 (현재)"] - 1).abs()
            lines.append(f"| {name} | {preds[name].sum():,.0f} | {preds[name].sum() / last[preds.index].sum() - 1:+.1%} | {d.median():.1%} | {d.max():.1%} |")
        widths = {n: (table(k)[hi].astype(float) / table(k)[lo].astype(float)).median() for n, (k, lo, hi) in RANGES.items()}
        lines += ["", "| 범위 규칙 | 80% 범위 폭 (위/아래, 중앙값) |", "|---|---|"] + [f"| {n} | ×{w:.2f} |" for n, w in widths.items()]
    else:
        codes = preds.index.intersection(act.index)
        ape = preds.loc[codes].sub(act[codes], axis=0).abs().div(act[codes], axis=0) * 100
        lines += [f"실제값 {len(codes)}곳으로 채점.", "", "| 버전 | 평균 절대 오차 | v1 대비 차이 (95% 구간) | 판정 |", "|---|---|---|---|"]
        for name in POINTS:
            diff = (ape[name] - ape["v1 (현재)"]).to_numpy()
            lo, hi = boot_ci(diff)
            verdict = "기준" if name.startswith("v1") else ("맞음" if hi < 0 else "틀림" if lo > 0 else "불확실")
            lines.append(f"| {name} | {ape[name].mean():.2f}% | {diff.mean():+.2f}%p ({lo:+.2f} ~ {hi:+.2f}) | {verdict} |")
        lines += ["", "| 범위 규칙 | 80% 포함률 | interval score |", "|---|---|---|"]
        for n, (k, lo_c, hi_c) in RANGES.items():
            t = table(k).loc[codes]
            lo_v, hi_v, y = t[lo_c].astype(float), t[hi_c].astype(float), act[codes]
            score = (hi_v - lo_v) + 10 * (lo_v - y).clip(lower=0) + 10 * (y - hi_v).clip(lower=0)
            lines.append(f"| {n} | {((y >= lo_v) & (y <= hi_v)).mean():.0%} | {(score / y).mean():.3f} |")
    (OUT / "scoreboard_2026.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((OUT / "scoreboard_2026.md").read_text())


if __name__ == "__main__":
    main()
