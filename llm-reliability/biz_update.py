"""실제 판매 기록으로 전략 시뮬레이션의 '가정 세계'를 지우고 다시 계산한다.

데이터가 쌓일수록 가정 108개 세계 중 기록과 맞지 않는 세계가 탈락하고, 남은 세계로 전략 성적을 다시 낸다.
(모델 규칙을 시험 구간 성적으로 채택하던 것과 같은 원리: 의견이 아니라 기록이 판정한다.)

입력
  data_biz/events.csv     실험 증거 (llmrel.biz). 리포트 계열(stream=report) 상대 중 L3 이상 = 유료 고객
  data_biz/profile.json   판매 가능 시작일, 리포트에 쓰는 노력, 시작 자금, 확장형 시도 기록

판정 규칙 (사전등록)
  - 리포트 고객 도착: 포아송. 기대 고객 수 = 세계의 월 확률(lam) × 노력 × 판매 가능 개월 수
  - 확장형 시도: 9개월 판정이 끝난 시도마다, 트랙션 있음 = p×효율, 없음 = 1 − p×효율
  - 시작 자금: 실제 값에 가장 가까운 세계만 남김
  - 가능도가 최고 세계의 1/20 미만이면 탈락 (우도비 20:1)

    python biz_update.py                 # → output/strategy/updated.md
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

import strategy_sim as S
from llmrel import biz

ROOT = Path(__file__).parent
PROFILE = ROOT / "data_biz" / "profile.json"
OUT = ROOT / "output" / "strategy"
RATIO = 20.0
H_BEST = {"K": 9, "quit": 1.5, "eve": 0.5}           # 전략 탐색에서 고른 H 손잡이 (최악 세계 파산 ≤ 2%)


def load_profile(path: Path = PROFILE) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def observed(ev: pd.DataFrame, prof: dict, today: pd.Timestamp | None = None) -> dict:
    today = today or pd.Timestamp.today().normalize()
    report_exps = [k for k, v in biz.EXPERIMENTS.items() if v["stream"] == "report"]
    e = ev[ev.exp.isin(report_exps)]
    paid = int((e.groupby("counterpart")["level"].apply(lambda s: max(map(biz._rank, s))) >= 3).sum()) if len(e) else 0
    start = pd.Timestamp(prof.get("selling_from", "2100-01-01"))
    months = max(0.0, (today - start).days / 30.4)
    return {"paid": paid, "months": months, "effort": float(prof.get("report_effort", 0.3)),
            "bets": prof.get("bets", []), "W0": prof.get("W0")}


def likelihood(w: dict, obs: dict) -> float:
    L = poisson.pmf(obs["paid"], w["lam"] * obs["effort"] * obs["months"]) if obs["months"] > 0 else 1.0
    q = min(1.0, w["p"] * w["half"])                  # 병행 시도의 성공 확률
    for b in obs["bets"]:
        if b.get("judged"):
            L *= q if b.get("traction") else (1 - q)
    return float(L)


def surviving(obs: dict) -> pd.DataFrame:
    keys = list(S.WORLDS)
    ws = pd.DataFrame([dict(zip(keys, c)) for c in itertools.product(*S.WORLDS.values())])
    if obs.get("W0") is not None:
        near = min(S.WORLDS["W0"], key=lambda v: abs(v - obs["W0"]))
        ws = ws[ws.W0 == near]
    ws["L"] = [likelihood(w, obs) for w in ws.to_dict("records")]
    ws["keep"] = ws.L >= ws.L.max() / RATIO
    return ws


def evaluate(ws: pd.DataFrame, n: int = 2000) -> pd.DataFrame:
    keep = ws[ws.keep].drop(columns=["L", "keep"]).to_dict("records")
    rows = []
    for s in ["A", "C2", "D", "H", "E", "F"]:
        extra = H_BEST if s == "H" else {}
        r = pd.DataFrame([S.run(s, {**w, **extra}, n=n, seed=i) for i, w in enumerate(keep)])
        rows.append({"전략": S.NAME[s] + (" (손잡이 조정)" if s == "H" else ""), "10억+": r["10억+"].median(), "30억+": r["30억+"].median(),
                     "파산(중앙)": r.ruin.median(), "파산(최악)": r.ruin.max(), "자산 중앙값(억)": r["median"].median() / 1e4})
    return pd.DataFrame(rows)


def lam_table(ws: pd.DataFrame) -> str:
    t = ws.groupby("lam").keep.mean()
    return " · ".join(f"연 {l * 12:.1f}곳 {'남음' if v > 0 else '탈락'}" for l, v in t.items())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prof, ev = load_profile(), biz.events()
    obs = observed(ev, prof)
    ws = surviving(obs)
    res = evaluate(ws)
    pct = lambda x: f"{x * 100:.1f}%"
    L = ["# 기록으로 다시 계산한 전략 성적", "",
         f"- 리포트 유료 고객: **{obs['paid']}곳**, 판매 가능 기간 {obs['months']:.1f}개월 (리포트 노력 {obs['effort']:.1f})",
         f"- 판정이 끝난 확장형 시도: {sum(1 for b in obs['bets'] if b.get('judged'))}개",
         f"- 남은 가정 세계: **{int(ws.keep.sum())} / {len(ws)}** (우도비 {RATIO:.0f}:1 미만 탈락)",
         f"- 판매 속도: {lam_table(ws)}", "",
         "| 전략 | 10억+ | 30억+ | 파산(중앙) | 파산(최악) | 자산 중앙값 |", "|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append(f"| {r['전략']} | {pct(r['10억+'])} | {pct(r['30억+'])} | {pct(r['파산(중앙)'])} | {pct(r['파산(최악)'])} | {r['자산 중앙값(억)']:.1f}억 |")
    if obs["months"] == 0:
        L += ["", "아직 판매 가능 기간이 시작되지 않아 판매 속도 세계는 하나도 지우지 않았습니다. (사업자 등록 뒤 profile.json의 selling_from을 채움)"]
    L += ["", "## 예시: 판매를 시작한 뒤 기록이 이렇다면 (리포트 노력 0.5)", "", "| 기간 | 유료 고객 | 남는 판매 속도 세계 |", "|---|---|---|"]
    for m, k in [(6, 0), (6, 6), (12, 0), (12, 1), (12, 6), (24, 0), (24, 2), (24, 12)]:
        demo = surviving({**obs, "paid": k, "months": float(m), "effort": 0.5})
        L.append(f"| {m}개월 | {k}곳 | {lam_table(demo)} |")
    L += ["", "고객 수가 작은 사업은 기록이 천천히 쌓입니다. 6개월로는 거의 아무 세계도 지울 수 없고, 1~2년은 쌓여야 판매 속도가 판정됩니다. 그 전까지는 모든 세계에서 버티는 전략(H)이 맞습니다."]
    (OUT / "updated.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
