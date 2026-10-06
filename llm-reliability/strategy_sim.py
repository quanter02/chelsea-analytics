"""전략 시뮬레이터: 우리 알고리즘(플랜 A, 확실 모드 채택, 시즌 리플레이)을 '사업 전략 고르기'에 그대로 적용.

주의: 여기 숫자는 데이터가 아니라 **가정**이다. 그래서 가정을 하나로 정하지 않고 54개 '세계'(가정 조합)를
모두 돌려, 대부분의 세계에서 이기는 전략만 채택한다 (모델 규칙 채택과 같은 '확실 모드').

    python strategy_sim.py          # → output/strategy/report.md

단위: 만원, 1개월 단위, 10년(120개월). 시작 저축 2,000, 생활비 월 250.

전략
  A  리포트만        셈법 리포트 사업에 전력 (바닥만)
  B  바닥+고집       리포트 반 + 확장형 제품 시도 반, 실패해도 36개월 버팀
  C  바닥+FA         리포트 반 + 값싼 시도를 9개월마다 판정, 안 되면 바로 정리하고 다음 시도 (FM식)
  C2 바닥 먼저→FA    처음 12개월은 리포트에 전력, 그 뒤 C
  D  올인 FA         바닥 없이 시도에 전력, 9개월 판정
  E  초기 멤버 취업  빠르게 크는 회사에 월급 + 지분
  F  레버리지 투자   월급 + 저축을 3배 레버리지로 투자
  H  직장 유지→바닥→FA  월급을 유지하며 저녁에 리포트(노력 0.3). 리포트 수입이 생활비+사업비를 넘으면 독립해 C로 전환
                    (공이 없을 때 4-4-2 고정 = 수비 대형을 먼저 지키고 공격)

모든 창업형 전략 공통
  - 저축이 0 아래로 내려가면 '파산' → 월급 일자리로 가고 더 이상 시도하지 않음 (다시 못 함 = 진짜 손해)
  - 확장형 시도가 성장 신호(트랙션)를 보이면 거기에 전력, 36개월 뒤 지분 가치 실현(생존 50%, 지분 50%)
  - 새 시도를 시작할 때 20% 확률로 지원금 3,000
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "output" / "strategy"
T, LIVING, TOOLS, BIZ = 120, 250.0, 100.0, 50.0
JOB_SAVE = 150.0                                         # 월급 일자리의 월 저축 (생활비 제외)
TOPS = {"10억+": 100_000.0, "30억+": 300_000.0, "100억+": 1_000_000.0}
MAIN = "30억+"                                            # 이 시뮬레이션에서 정한 '상위권' 기준
STRATS = ["A", "B", "C", "C2", "D", "H", "E", "F"]
NAME = {"A": "리포트만", "B": "바닥+고집", "C": "바닥+FA", "C2": "바닥 먼저→FA", "D": "올인 FA", "E": "초기 멤버 취업", "F": "레버리지 투자", "H": "직장 유지→바닥→FA"}
WORLDS = {"p": [0.04, 0.08, 0.15],          # 시도 하나가 '될 놈'일 확률 (전력 투구 기준)
          "M": [100_000, 300_000, 1_000_000],  # 될 놈의 회사 가치 중앙값 (10억/30억/100억)
          "lam": [0.15, 0.3, 0.5],          # 리포트 신규 고객 월 확률 (전력 기준) = 연 1.8~6곳
          "half": [0.5, 0.75],              # 반만 쏟을 때 성공 확률 배수
          "W0": [2000.0, 6000.0]}           # 시작 자금 (저축 + 받을 수 있는 지원금)


def run(strat: str, w: dict, n: int = 4000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    W0 = w["W0"]
    W = np.full(n, W0)
    employed = np.full(n, strat == "H")
    ruined = np.zeros(n, bool)
    clients = np.zeros(n)
    short_years = np.zeros(n)                # 수입 < 생활비인 해 수 ('빈 시즌')
    year_inc = np.zeros(n)
    payout = np.zeros(n)
    if strat in ("A", "B", "C", "C2", "D", "H"):
        bets = strat != "A"
        K = 36 if strat == "B" else w.get("K", 9)
        age = np.zeros(n); winner = np.zeros(n, bool); traction = np.full(n, -1.0); grow = np.full(n, -1.0)
        exit_at = np.full(n, -1.0)
        started = np.zeros(n, bool)

        def new_bet(mask, t, eff):
            age[mask] = 0
            winner[mask] = rng.random(mask.sum()) < w["p"] * eff[mask]
            traction[mask] = np.where(winner[mask], rng.integers(3, 10, mask.sum()), -1)
            grant = mask & (rng.random(n) < 0.2)
            W[grant] += 3000
            started[mask] = True
        for t in range(T):
            live = ~ruined
            growing = grow >= 0
            # 노력 배분
            if strat == "A":
                ef, eb = np.ones(n), np.zeros(n)
            elif strat == "D":
                ef, eb = np.zeros(n), np.ones(n)
            elif strat == "C2" and t < 12:
                ef, eb = np.ones(n), np.zeros(n)
            elif strat == "H":
                ef, eb = np.where(employed, w.get("eve", 0.3), 0.5), np.where(employed, 0.0, 0.5)
            else:
                ef, eb = np.full(n, 0.5), np.full(n, 0.5)
            ef = np.where(growing, 0.2, ef); eb = np.where(growing, 0.0, eb)
            effmul = np.where(eb >= 1, 1.0, w["half"])
            # 시도 관리
            if bets:
                need = live & ~growing & (eb > 0) & (~started | (age >= K))
                new_bet(need, t, effmul)
                hit = live & ~growing & started & (traction >= 0) & (age >= traction)
                grow[hit] = t
                exit_at[hit] = t + 36
                age[live & started & ~growing] += 1
            # 리포트 고객
            gain = rng.random(n) < w["lam"] * ef
            clients = np.where(live, np.minimum(clients + gain, 25 * np.maximum(ef, 0.2) / 0.5), clients)
            clients = clients - (rng.random(n) < 0.015 * clients)
            inc = np.where(live, clients * 100.0, 0.0)
            cost = LIVING + np.where(live & (ef > 0), BIZ, 0) + np.where(live & (eb > 0) & ~growing, TOOLS, 0)
            sal = np.where(live & growing, 300.0, 0.0)          # 투자받은 회사에서 대표 급여
            job = np.where(live & employed, LIVING + JOB_SAVE, 0.0)
            cost = cost - np.where(live & employed, BIZ, 0)     # 부업 단계는 사업비 최소
            W += np.where(live, inc + sal + job - cost, JOB_SAVE)
            year_inc += np.where(live, inc + sal + job, LIVING + JOB_SAVE)
            employed &= ~(inc >= w.get("quit", 1.0) * (LIVING + BIZ))                   # 리포트 수입이 생활비+사업비를 넘으면 독립
            # 지분 실현
            ex = live & (exit_at == t)
            surv = rng.random(n) < 0.5
            val = w["M"] * np.exp(rng.normal(0, 1.2, n)) * 0.5
            payout[ex & surv] += val[ex & surv]
            W[ex] += np.where(surv[ex], val[ex], 0)
            grow[ex] = -1; started[ex] = False; exit_at[ex] = -1
            # 파산
            ruined |= W < 0
            W = np.where(ruined & (W < 0), 0.0, W)
            if t % 12 == 11:
                short_years += year_inc < LIVING * 12
                year_inc[:] = 0
        # 10년 시점 성장 중이면 장부가치 절반만 인정
        open_ = (exit_at > 0) & ~ruined
        W += np.where(open_, w["M"] * 0.5 * 0.5 * 0.5, 0)
    elif strat == "E":
        W = W0 + 200.0 * T + np.zeros(n)
        big = rng.random(n) < min(0.5, w["p"] * 2.5)
        W += np.where(big, 30_000 * (w["M"] / 300_000) * np.exp(rng.normal(0, 1.0, n)), 0)
    elif strat == "F":
        for t in range(T):
            r = rng.normal(0.15 / 12, 0.54 / np.sqrt(12), n)
            wiped = r <= -1 / 3                                  # 한 달 -33% 이하면 3배 레버리지 원금 소멸 (반대매매)
            ruined |= wiped
            W = np.where(wiped, 0.0, W * (1 + r)) + 200
    return {**{k: float((W >= v).mean()) for k, v in TOPS.items()}, "ruin": float(ruined.mean()), "median": float(np.median(W)),
            "p10": float(np.quantile(W, 0.1)), "short3": float((short_years >= 3).mean())}


def tune_h(n: int = 2000) -> pd.DataFrame:
    """H의 손잡이 3개를 격자로 돌려 '모든 세계에서 파산 2% 이하'를 지키는 조합 중 30억+ 확률 중앙값 1위를 찾는다."""
    keys = list(WORLDS)
    worlds = [dict(zip(keys, c)) for c in itertools.product(*WORLDS.values())]
    rows = []
    for K, quit, eve in itertools.product([6, 9, 12, 18], [0.5, 0.75, 1.0, 1.5], [0.2, 0.3, 0.5]):
        res = [run("H", {**w, "K": K, "quit": quit, "eve": eve}, n=n, seed=i) for i, w in enumerate(worlds)]
        r = pd.DataFrame(res)
        rows.append({"K": K, "quit": quit, "eve": eve, "30억+": r[MAIN].median(), "10억+": r["10억+"].median(),
                     "ruin_max": r.ruin.max(), "ruin_med": r.ruin.median(), "median_W": r["median"].median() / 10000})
    return pd.DataFrame(rows).sort_values("30억+", ascending=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    keys = list(WORLDS)
    for i, combo in enumerate(itertools.product(*WORLDS.values())):
        w = dict(zip(keys, combo))
        for s in STRATS:
            rows.append({"world": i, **w, "strat": s, **run(s, w, seed=i)})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "worlds.csv", index=False)
    piv = lambda col: d.pivot(index="world", columns="strat", values=col)
    top, ruin, med, p10 = piv(MAIN), piv("ruin"), piv("median"), piv("p10")
    # 확실 모드: X가 Y를 '채택'하려면 세계의 90% 이상에서 상위권 확률이 같거나 높고, 파산 확률이 2%p 넘게 나쁘지 않아야 함
    beat = pd.DataFrame(index=STRATS, columns=STRATS, dtype=float)
    for x, y in itertools.permutations(STRATS, 2):
        ok = (top[x] >= top[y]) & (ruin[x] <= ruin[y] + 0.02)
        beat.loc[x, y] = ok.mean()
    summ = pd.DataFrame({"10억+ 확률": piv("10억+").median(), "30억+ 확률": top.median(), "100억+ 확률": piv("100억+").median(),
                         "파산 확률 중앙값": ruin.median(), "10년 자산 중앙값(억)": med.median() / 10000,
                         "하위 10% 자산(억)": p10.median() / 10000, "빈 시즌 3년+ 확률": piv("short3").median()}).loc[STRATS]
    winners = top.idxmax(axis=1).value_counts()
    L = ["# 전략 시뮬레이션: 우리 알고리즘으로 '어떻게 부자가 될까'를 채점", "",
         "**숫자는 데이터가 아니라 가정입니다.** 그래서 가정을 108개 세계로 바꿔 가며 돌리고, 대부분의 세계에서 이기는 전략만 채택합니다.", "",
         "가정 세계 108개: 시도 성공 확률 4/8/15% × 성공 회사 가치 중앙값 10/30/100억 × 리포트 고객 연 1.8/3.6/6곳(전력 기준) × 반만 쏟을 때 효율 50/75% × 시작 자금 2,000/6,000만원", "",
         "## 전략별 성적 (108개 세계의 중앙값)", "", "| 전략 | " + " | ".join(summ.columns) + " |", "|---|" + "---|" * len(summ.columns)]
    for s, r in summ.iterrows():
        L.append(f"| {s} {NAME[s]} | {r.iloc[0]*100:.1f}% | {r.iloc[1]*100:.1f}% | {r.iloc[2]*100:.2f}% | {r.iloc[3]*100:.0f}% | {r.iloc[4]:.1f} | {r.iloc[5]:.1f} | {r.iloc[6]*100:.0f}% |")
    L += ["", "## 세계별 1위 (30억+ 확률 기준)", "", " · ".join(f"{NAME[s]} {c}곳" for s, c in winners.items()), "",
          "## 확실 모드 승률표", "", "칸 = 행 전략이 열 전략을 이긴 세계 비율 (30억+ 확률 ≥, 파산 확률 +2%p 이내). **90% 이상이면 채택.**", "",
          "| 행 \\ 열 | " + " | ".join(NAME[s] for s in STRATS) + " |", "|---|" + "---|" * len(STRATS)]
    for x in STRATS:
        L.append(f"| {NAME[x]} | " + " | ".join("—" if x == y else f"{beat.loc[x, y]*100:.0f}%" for y in STRATS) + " |")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return d, beat, summ


if __name__ == "__main__":
    main()
